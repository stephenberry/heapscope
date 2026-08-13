//! A native profile, read for the two things this tool does to it.
//!
//! Everything here is a view over the [`json::Value`] the file parsed into,
//! never a copy of it. The profile carries fields this tool has no opinion
//! about — every counter, the shape histograms, the attribution rows — and the
//! format's own rule is that a reader ignores what it does not know. A reader
//! that also *writes* has to preserve what it ignored, and the surest way to
//! preserve something is never to have taken it apart.

use std::collections::BTreeMap;

use crate::json::{self, Value};
use crate::tool::Resolution;

/// The format this tool reads, and the only version of it that exists.
///
/// Refused rather than attempted on anything else, which is the second half of
/// the compatibility rule every profile states about itself: *ignore unknown
/// fields; refuse an unknown `formatVersion`*. A tool that tried anyway would be
/// writing frame indices into a file whose frame table may mean something else.
const FORMAT: &str = "heapscope-profile";
const FORMAT_VERSION: u64 = 1;

/// One image, as the module map recorded it.
#[derive(Clone, Debug)]
pub struct Module {
    pub path: String,
    /// Where the image was mapped. What `atos -l` takes.
    ///
    /// The only number needed from a module here. The bias the map also records
    /// converts a runtime address into a file address, and this tool never has
    /// to: the profile already carries both, per frame, as `addr` and
    /// `fileAddr`. Recomputing one from the other would be a second opinion
    /// about an answer the file already gives.
    pub load: u64,
}

/// Which counter a folded rendering carries.
///
/// Spelled as the native profile's own field names, so that `--metric` names
/// something the reader can find in the file, and so that this tool and
/// [`heapscope::FoldedMetric`] cannot drift into two vocabularies for one idea.
pub const METRICS: &[&str] = &["totalBytes", "totalBlocks", "atGmaxBytes", "atEndBytes"];

#[derive(Debug)]
pub struct Profile {
    root: Value,
    modules: Vec<Module>,
}

impl Profile {
    /// Reads `text` as a native profile.
    pub fn parse(text: &str) -> Result<Profile, String> {
        let root = json::parse(text).map_err(|error| format!("not JSON: {error}"))?;

        match root.get("format").and_then(Value::as_str) {
            Some(FORMAT) => {}
            Some(other) => return Err(format!("this is a `{other}` file, not a {FORMAT}")),
            None => {
                // The likeliest wrong file by far, and worth naming: it is the
                // one this crate writes by default, and it renders its frames as
                // text, so there are no addresses in it left to resolve.
                let hint = if root.get("dhatFileVersion").is_some() {
                    ". This looks like a DHAT v2 file; symbolize the native \
                     profile instead — `Output::native` writes one"
                } else {
                    ""
                };
                return Err(format!(
                    "no `format` field, so this is not a {FORMAT}{hint}"
                ));
            }
        }

        match root.get("formatVersion").and_then(Value::as_u64) {
            Some(FORMAT_VERSION) => {}
            Some(other) => {
                return Err(format!(
                    "formatVersion {other}, and this tool knows version {FORMAT_VERSION}. \
                     A profile says a reader must refuse a version it does not know"
                ))
            }
            None => return Err(String::from("no `formatVersion`")),
        }

        let modules = root
            .get("modules")
            .and_then(Value::as_array)
            .unwrap_or(&[])
            .iter()
            .map(|module| Module {
                path: module
                    .get("path")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                load: module.get("load").and_then(Value::as_address).unwrap_or(0),
            })
            .collect();

        Ok(Profile { root, modules })
    }

    pub fn modules(&self) -> &[Module] {
        &self.modules
    }

    pub fn frame_count(&self) -> usize {
        self.frames().len()
    }

    fn frames(&self) -> &[Value] {
        self.root
            .get("frames")
            .and_then(Value::as_array)
            .unwrap_or(&[])
    }

    /// The addresses to ask about, grouped by the image they are in.
    ///
    /// One entry per module that any frame falls in, holding the frame's index
    /// in the table and the address to send. Which address that is depends on
    /// the tool — `atos` works from where the image was mapped and the other two
    /// from where the code sits in the file — so the choice is made here, once,
    /// against [`Tool::wants_runtime_addresses`](crate::tool::Tool::wants_runtime_addresses).
    ///
    /// Frames already carrying a resolved `function` are skipped, so running
    /// this tool twice over one profile does no work the second time and cannot
    /// overwrite a better answer with a worse one.
    pub fn batches(&self, runtime_addresses: bool) -> BTreeMap<usize, Vec<(usize, u64)>> {
        let mut batches: BTreeMap<usize, Vec<(usize, u64)>> = BTreeMap::new();
        for (at, frame) in self.frames().iter().enumerate() {
            if frame.get("function").is_some() {
                continue;
            }
            let Some(module) = frame.get("module").and_then(Value::as_u64) else {
                // An address in no image at all, which is what a truncated stack
                // walk produces. There is nothing to resolve it against.
                continue;
            };
            let Some(module) = usize::try_from(module)
                .ok()
                .filter(|&at| at < self.modules.len())
            else {
                continue;
            };
            let address = if runtime_addresses {
                frame.get("addr").and_then(Value::as_address)
            } else {
                frame.get("fileAddr").and_then(Value::as_address)
            };
            // A frame with no `fileAddr` is one whose image reported no bias —
            // the Windows module map does not — and asking a file-address tool
            // about a runtime address would name whatever happens to live there.
            if let Some(address) = address {
                batches.entry(module).or_default().push((at, address));
            }
        }
        batches
    }

    /// Records what a symbolizer said about the frame at `at`.
    ///
    /// Added as new members rather than written over `symbol`, and the
    /// distinction is the point: `symbol` is what the *running process* knew the
    /// address by, read from a loaded image's symbol table, and it is often
    /// absent precisely because that table was stripped. What is added here came
    /// from a file on disk, possibly on another machine, possibly from an
    /// archived build. Keeping both means a reader can see when they disagree,
    /// which is the symptom of resolving against the wrong binary.
    ///
    /// New fields need no version bump: the format's rule is that a reader
    /// ignores what it does not know, so a viewer that has never heard of
    /// `function` reads the profile exactly as it did before.
    pub fn resolve_frame(&mut self, at: usize, resolution: &Resolution) {
        let Some(Value::Array(frames)) = self.root_mut("frames") else {
            return;
        };
        let Some(frame) = frames.get_mut(at) else {
            return;
        };
        let Some(innermost) = resolution.frames.first() else {
            return;
        };

        frame.set("function", Value::String(innermost.function.clone()));
        if let Some(file) = &innermost.file {
            frame.set("file", Value::String(file.clone()));
        }
        if let Some(line) = innermost.line {
            frame.set("line", Value::number(u64::from(line)));
        }

        // The callers an optimiser folded into this one. Absent rather than an
        // empty array where there are none, so that a profile resolved by a tool
        // that cannot report inlining is distinguishable from one where nothing
        // was inlined.
        if resolution.frames.len() > 1 {
            let inlined = resolution.frames[1..]
                .iter()
                .map(|frame| {
                    let mut members = vec![(
                        String::from("function"),
                        Value::String(frame.function.clone()),
                    )];
                    if let Some(file) = &frame.file {
                        members.push((String::from("file"), Value::String(file.clone())));
                    }
                    if let Some(line) = frame.line {
                        members.push((String::from("line"), Value::number(u64::from(line))));
                    }
                    Value::Object(members)
                })
                .collect();
            frame.set("inlinedBy", Value::Array(inlined));
        }
    }

    fn root_mut(&mut self, key: &str) -> Option<&mut Value> {
        let Value::Object(members) = &mut self.root else {
            return None;
        };
        members
            .iter_mut()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
    }

    /// How many frames now carry a resolved name.
    pub fn resolved_frames(&self) -> usize {
        self.frames()
            .iter()
            .filter(|frame| frame.get("function").is_some())
            .count()
    }

    /// The profile, rendered back to JSON.
    pub fn to_json(&self) -> String {
        json::render(&self.root)
    }

    /// One rendered frame, in the shape this crate renders frames everywhere.
    ///
    /// ```text
    /// 0x1044c81f0: profile_a_program::churn (/path/to/program+0x2c1f0)
    /// ```
    ///
    /// The shape is not cosmetic. It is what
    /// [`trim::worth_showing`](heapscope::symbol::trim::worth_showing) parses,
    /// so rendering some other way would silently disable trimming; and keeping
    /// the image and file address after the name is the rule
    /// [`Symbolized`](heapscope::symbol::Symbolized) documents — a name is
    /// *added to* the attribution rather than replacing it, so nothing is lost
    /// by having resolved one.
    ///
    /// The best available name wins: what this tool resolved, then what the
    /// running process knew, then nothing.
    fn render_frame(&self, frame: &Value) -> String {
        let mut out = String::new();
        push_hex(&mut out, frame.get("addr").and_then(Value::as_address));
        out.push_str(": ");

        let resolved = frame.get("function").and_then(Value::as_str);
        let recorded = frame.get("symbol").and_then(Value::as_str);
        match (resolved, recorded) {
            (Some(name), _) => heapscope::output::push_display(&mut out, name),
            (None, Some(name)) => {
                // Still mangled in the file, because the format keeps the
                // linker's own spelling. Rendered the way every other reader of
                // this crate renders it.
                let mut demangled = String::new();
                if heapscope::demangle(name, &mut demangled) {
                    heapscope::output::push_display(&mut out, &demangled);
                } else {
                    heapscope::output::push_display(&mut out, name);
                }
                if let Some(offset) = frame.get("symbolOffset").and_then(Value::as_u64) {
                    if offset != 0 {
                        out.push_str(&format!("+{offset:#x}"));
                    }
                }
            }
            (None, None) => out.push_str("???"),
        }

        if let (Some(module), Some(file_address)) = (
            frame
                .get("module")
                .and_then(Value::as_u64)
                .and_then(|at| usize::try_from(at).ok())
                .and_then(|at| self.modules.get(at)),
            frame.get("fileAddr").and_then(Value::as_address),
        ) {
            out.push_str(" (");
            heapscope::output::push_display(&mut out, &module.path);
            out.push_str(&format!("+{file_address:#x})"));
        }
        out
    }

    /// The profile as folded stacks, counted by `metric`.
    ///
    /// The same file [`Snapshot::write_folded`](heapscope::Snapshot::write_folded)
    /// writes, produced from the profile rather than from an engine — which is
    /// the whole reason this exists: on a platform where in-process
    /// symbolization names nothing, the flame graph worth drawing is the one
    /// made *after* the addresses have been resolved.
    ///
    /// Trimmed by the crate's own rule, and that is a strict improvement on
    /// trimming at record time: the rule reads frame names, so on Linux, where
    /// `dladdr` names almost nothing, it had nothing to work with and left every
    /// stack whole. Here the names exist.
    pub fn to_folded(&self, metric: &str) -> Result<String, String> {
        if !METRICS.contains(&metric) {
            return Err(format!(
                "`{metric}` is not a metric; expected one of {}",
                METRICS.join(", ")
            ));
        }
        let frames = self.frames();
        let rendered: Vec<String> = frames
            .iter()
            .map(|frame| self.render_frame(frame))
            .collect();

        let mut totals: Vec<(String, u64)> = Vec::new();
        let mut index: BTreeMap<String, usize> = BTreeMap::new();
        let mut stack = String::new();

        for point in self
            .root
            .get("points")
            .and_then(Value::as_array)
            .unwrap_or(&[])
        {
            // Absent in a mode that has no such measurement, which is the
            // format omitting rather than zeroing. Nothing to draw either way.
            let Some(count) = point.get(metric).and_then(Value::as_u64) else {
                continue;
            };
            if count == 0 {
                continue;
            }

            let indices: Vec<usize> = point
                .get("frames")
                .and_then(Value::as_array)
                .unwrap_or(&[])
                .iter()
                .filter_map(Value::as_u64)
                .filter_map(|at| usize::try_from(at).ok())
                .filter(|&at| at < rendered.len())
                .collect();
            let shown: Vec<String> = indices.iter().map(|&at| rendered[at].clone()).collect();
            let keep = heapscope::symbol::trim::worth_showing(&shown);

            stack.clear();
            // Outermost first, which is where a flame graph puts its root.
            for frame in shown[keep].iter().rev() {
                if !stack.is_empty() {
                    stack.push(';');
                }
                push_frame(&mut stack, frame);
            }
            if stack.is_empty() {
                push_frame(
                    &mut stack,
                    match point.get("kind").and_then(Value::as_str) {
                        Some("overflow") => OVERFLOW_FRAME,
                        _ => UNWALKABLE_FRAME,
                    },
                );
            }

            match index.get(&stack) {
                Some(&at) => totals[at].1 = totals[at].1.saturating_add(count),
                None => {
                    index.insert(stack.clone(), totals.len());
                    totals.push((stack.clone(), count));
                }
            }
        }

        let mut out = String::new();
        for (stack, count) in &totals {
            out.push_str(stack);
            out.push_str(&format!(" {count}\n"));
        }
        Ok(out)
    }
}

/// The two labels the library's emitters give a point with no frames. Repeated
/// as text rather than shared because they are `pub(super)` there — and because
/// what has to match is the *file*, which a test compares.
const OVERFLOW_FRAME: &str =
    "[overflow]: allocations recorded after the program-point table filled up";
const UNWALKABLE_FRAME: &str = "[unwalkable]: no frame pointer chain at this allocation";

fn push_hex(out: &mut String, address: Option<u64>) {
    match address {
        Some(address) => out.push_str(&format!("{address:#x}")),
        None => out.push_str("0x?"),
    }
}

/// Appends one frame with the separator escaped, exactly as the library's folded
/// emitter does. See `src/output/folded.rs` for why `;` is the one character
/// this handles and why the escape is not reversible.
fn push_frame(out: &mut String, frame: &str) {
    for character in frame.chars() {
        if character == ';' {
            out.push_str("\\u{3b}");
        } else {
            out.push(character);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::Frame;

    fn a_profile() -> String {
        String::from(
            r#"{
  "format":"heapscope-profile","formatVersion":1,
  "somethingThisToolHasNeverHeardOf":{"keep":"me"},
  "frames":[
    {"addr":"0x1100","module":0,"fileAddr":"0x100"},
    {"addr":"0x1200","module":0,"fileAddr":"0x200","symbol":"_ZN4core3fmt5write17hb1f9a4a7f2f1a0c9E","symbolOffset":16},
    {"addr":"0x9999"}
  ],
  "points":[
    {"kind":"recorded","totalBytes":4096,"totalBlocks":2,"frames":[0,1]},
    {"kind":"recorded","totalBytes":1024,"totalBlocks":1,"frames":[2]}
  ],
  "modules":[{"path":"/bin/program","load":"0x1000","start":"0x1000","size":4096,"bias":"0x1000"}]
}"#,
        )
    }

    fn resolution(function: &str) -> Resolution {
        Resolution {
            frames: vec![Frame {
                function: String::from(function),
                file: Some(String::from("/src/main.rs")),
                line: Some(42),
            }],
        }
    }

    #[test]
    fn a_file_of_another_format_is_refused_by_name() {
        let error = Profile::parse(r#"{"dhatFileVersion":2,"pps":[]}"#).expect_err("refused");
        assert!(error.contains("DHAT"), "{error}");
        assert!(error.contains("Output::native"), "{error}");

        let error = Profile::parse(r#"{"format":"something-else"}"#).expect_err("refused");
        assert!(error.contains("something-else"), "{error}");
    }

    /// The other half of the rule every profile states about itself.
    #[test]
    fn an_unknown_format_version_is_refused_rather_than_attempted() {
        let error = Profile::parse(r#"{"format":"heapscope-profile","formatVersion":99}"#)
            .expect_err("refused");
        assert!(error.contains("99"), "{error}");
    }

    /// **The property.** Everything this tool did not set is still there, in
    /// order, after a resolve and a render.
    #[test]
    fn resolving_preserves_every_field_the_tool_never_looked_at() {
        let before = json::parse(&a_profile()).expect("parses");
        let mut profile = Profile::parse(&a_profile()).expect("a native profile");
        profile.resolve_frame(0, &resolution("program::churn"));

        let after = json::parse(&profile.to_json()).expect("the rendering parses");
        assert_eq!(
            after.get("somethingThisToolHasNeverHeardOf"),
            before.get("somethingThisToolHasNeverHeardOf"),
            "a member this tool has no opinion about was changed"
        );
        assert_eq!(after.get("points"), before.get("points"));
        assert_eq!(after.get("modules"), before.get("modules"));

        // Every key that was there is still there, at least as often.
        let census = json::key_census(&after);
        for (key, count) in json::key_census(&before) {
            assert!(
                census.get(&key).copied().unwrap_or(0) >= count,
                "`{key}` appeared {count} times and now appears {:?}",
                census.get(&key)
            );
        }
    }

    /// `symbol` is what the running process knew; `function` is what the file
    /// says. Keeping both is what makes a resolve against the wrong binary
    /// visible instead of silent.
    #[test]
    fn what_the_process_knew_is_not_overwritten_by_what_the_file_says() {
        let mut profile = Profile::parse(&a_profile()).expect("a native profile");
        profile.resolve_frame(1, &resolution("something::else"));
        let after = json::parse(&profile.to_json()).expect("parses");
        let frame = &after
            .get("frames")
            .and_then(Value::as_array)
            .expect("frames")[1];

        assert_eq!(
            frame.get("symbol").and_then(Value::as_str),
            Some("_ZN4core3fmt5write17hb1f9a4a7f2f1a0c9E")
        );
        assert_eq!(
            frame.get("function").and_then(Value::as_str),
            Some("something::else")
        );
        assert_eq!(frame.get("line").and_then(Value::as_u64), Some(42));
    }

    #[test]
    fn inlined_callers_are_recorded_only_when_there_are_some() {
        let mut profile = Profile::parse(&a_profile()).expect("a native profile");
        profile.resolve_frame(0, &resolution("only::one"));
        profile.resolve_frame(
            1,
            &Resolution {
                frames: vec![
                    Frame {
                        function: String::from("inner"),
                        file: None,
                        line: None,
                    },
                    Frame {
                        function: String::from("outer"),
                        file: Some(String::from("/src/a.rs")),
                        line: Some(7),
                    },
                ],
            },
        );
        let after = json::parse(&profile.to_json()).expect("parses");
        let frames = after
            .get("frames")
            .and_then(Value::as_array)
            .expect("frames");

        assert!(
            frames[0].get("inlinedBy").is_none(),
            "an empty `inlinedBy` cannot be told from a tool that does not report inlining"
        );
        let inlined = frames[1]
            .get("inlinedBy")
            .and_then(Value::as_array)
            .expect("one inlined caller");
        assert_eq!(inlined.len(), 1);
        assert_eq!(
            inlined[0].get("function").and_then(Value::as_str),
            Some("outer")
        );
    }

    /// The batches decide which number each tool is asked about, and the two are
    /// equal exactly when an image's bias is zero — so a fixture with a non-zero
    /// bias is the only one where getting it wrong shows.
    #[test]
    fn each_tool_is_asked_about_the_address_it_understands() {
        let profile = Profile::parse(&a_profile()).expect("a native profile");

        let by_file = profile.batches(false);
        assert_eq!(by_file[&0], vec![(0, 0x100), (1, 0x200)]);

        let by_runtime = profile.batches(true);
        assert_eq!(by_runtime[&0], vec![(0, 0x1100), (1, 0x1200)]);
    }

    /// An address in no image has nothing to be resolved against, and a module
    /// index past the end of the map is a profile to distrust rather than to
    /// index with.
    #[test]
    fn a_frame_in_no_image_is_asked_about_nowhere() {
        let profile = Profile::parse(&a_profile()).expect("a native profile");
        let batches = profile.batches(false);
        let asked: Vec<usize> = batches
            .values()
            .flat_map(|frames| frames.iter().map(|&(at, _)| at))
            .collect();
        assert_eq!(asked, vec![0, 1], "frame 2 is in no image");
    }

    /// Running the tool twice does no work the second time, and cannot replace a
    /// good answer with a worse one.
    #[test]
    fn a_frame_that_is_already_resolved_is_not_asked_about_again() {
        let mut profile = Profile::parse(&a_profile()).expect("a native profile");
        assert_eq!(profile.batches(false)[&0].len(), 2);
        profile.resolve_frame(0, &resolution("program::churn"));
        assert_eq!(profile.batches(false)[&0], vec![(1, 0x200)]);
        assert_eq!(profile.resolved_frames(), 1);
    }

    #[test]
    fn a_folded_rendering_uses_the_best_name_available() {
        let mut profile = Profile::parse(&a_profile()).expect("a native profile");
        profile.resolve_frame(0, &resolution("program::churn"));
        let folded = profile.to_folded("totalBytes").expect("a metric");

        // Frame 0 resolved, frame 1 only known in-process and demangled from
        // what the file carries, frame 2 known nowhere.
        assert!(folded.contains("program::churn"), "{folded}");
        assert!(folded.contains("core::fmt::write+0x10"), "{folded}");
        assert!(folded.contains("0x9999: ???"), "{folded}");
        // Both points survive, and the counts are the profile's own.
        assert_eq!(folded.lines().count(), 2, "{folded}");
        assert!(folded.contains(" 4096\n"), "{folded}");
        assert!(folded.contains(" 1024\n"), "{folded}");
    }

    /// The name is added to the image and offset rather than replacing them,
    /// which is what keeps a symbolized profile resolvable all over again.
    #[test]
    fn resolving_a_name_never_costs_the_attribution_underneath_it() {
        let mut profile = Profile::parse(&a_profile()).expect("a native profile");
        profile.resolve_frame(0, &resolution("program::churn"));
        let folded = profile.to_folded("totalBytes").expect("a metric");
        assert!(folded.contains("(/bin/program+0x100)"), "{folded}");
    }

    #[test]
    fn a_metric_the_profile_does_not_carry_is_named_rather_than_guessed() {
        let profile = Profile::parse(&a_profile()).expect("a native profile");
        let error = profile.to_folded("atGmax").expect_err("not a metric");
        assert!(error.contains("totalBytes"), "{error}");

        // A metric that *is* one, but which this profile omits — an ad hoc run
        // does exactly that — is an empty rendering rather than an error.
        assert_eq!(profile.to_folded("atEndBytes").expect("a metric"), "");
    }

    /// A path is whatever the filesystem allows, and `;` is the folded format's
    /// only structure.
    #[test]
    fn a_separator_in_a_path_does_not_invent_a_frame() {
        let text = a_profile().replace("/bin/program", "/bin/we;ird");
        let profile = Profile::parse(&text).expect("a native profile");
        let folded = profile.to_folded("totalBytes").expect("a metric");
        let first = folded.lines().next().expect("a line");
        let stack = first.rsplit_once(' ').expect("a count").0;
        assert_eq!(stack.split(';').count(), 2, "{folded}");
        assert!(folded.contains(r"we\u{3b}ird"), "{folded}");
    }

    /// Names come out of somebody else's symbol table, and reach a terminal
    /// through a flame graph. Screened by the same rule the library applies.
    #[test]
    fn a_hostile_name_is_screened_before_it_reaches_the_output() {
        let mut profile = Profile::parse(&a_profile()).expect("a native profile");
        profile.resolve_frame(
            0,
            &Resolution {
                frames: vec![Frame {
                    function: String::from("evil\u{1b}[2J\u{202e}gnp.eslaf"),
                    file: None,
                    line: None,
                }],
            },
        );
        let folded = profile.to_folded("totalBytes").expect("a metric");
        assert!(!folded.contains('\u{1b}'), "an escape survived: {folded}");
        assert!(!folded.contains('\u{202e}'), "an override survived");
        assert!(folded.contains(r"\u{202e}"), "{folded}");
    }

    /// A profile with no module map is the degraded case, not a crash.
    #[test]
    fn a_profile_with_no_modules_still_renders() {
        let text = a_profile().replace(r#""modules":[{"path":"/bin/program","load":"0x1000","start":"0x1000","size":4096,"bias":"0x1000"}]"#, r#""modules":[]"#);
        let profile = Profile::parse(&text).expect("a native profile");
        assert!(profile.batches(false).is_empty());
        assert!(!profile
            .to_folded("totalBytes")
            .expect("a metric")
            .is_empty());
    }
}
