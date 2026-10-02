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

/// The format this tool reads, and the versions of it this tool knows.
///
/// Refused rather than attempted on anything else, which is the second half of
/// the compatibility rule every profile states about itself: *ignore unknown
/// fields; refuse an unknown `formatVersion`*. A tool that tried anyway would be
/// writing frame indices into a file whose frame table may mean something else.
///
/// Version 2 is a run whose counts were restarted, which changes what its
/// totals cover and nothing about its frames, modules or addresses: the only
/// parts of the file this tool reads or writes. So it is known here, and read
/// exactly as version 1 is.
const FORMAT: &str = "heapscope-profile";
const FORMAT_VERSIONS: std::ops::RangeInclusive<u64> = 1..=2;

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
            Some(version) if FORMAT_VERSIONS.contains(&version) => {}
            Some(other) => {
                return Err(format!(
                    "formatVersion {other}, and this tool knows versions {} to {}. \
                     A profile says a reader must refuse a version it does not know",
                    FORMAT_VERSIONS.start(),
                    FORMAT_VERSIONS.end()
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
    /// Either way it is the
    /// [`call_site`](heapscope::symbol::call_site) of what was recorded, not the
    /// recorded number itself, which is the rule the library's own lookups
    /// follow.
    ///
    /// Frames a symbolizer already answered for are skipped, so running this
    /// tool twice over one profile does no work the second time and cannot
    /// overwrite a better answer with a worse one. An answer is a `function`,
    /// or a `file` where the tool placed the address without naming it.
    pub fn batches(&self, runtime_addresses: bool) -> BTreeMap<usize, Vec<(usize, u64)>> {
        let mut batches: BTreeMap<usize, Vec<(usize, u64)>> = BTreeMap::new();
        for (at, frame) in self.frames().iter().enumerate() {
            if frame.get("function").is_some() || frame.get("file").is_some() {
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
            if let Some(address) = address.and_then(heapscope::symbol::call_site) {
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

        // Absent where the tool found a file and line but no name, rather than
        // a placeholder: every reader of `function` takes it as a name, and a
        // placeholder would outrank the `symbol` the running process recorded
        // and draw unrelated frames as one.
        if let Some(function) = &innermost.function {
            frame.set("function", Value::String(function.clone()));
        }
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
                    let mut members = Vec::new();
                    if let Some(function) = &frame.function {
                        members.push((String::from("function"), Value::String(function.clone())));
                    }
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
            .filter(|frame| resolved_name(frame).is_some())
            .count()
    }

    /// The profile, rendered back to JSON.
    pub fn to_json(&self) -> String {
        json::render(&self.root)
    }

    /// One frame, in the shape the trimming rules read, once for every function
    /// its address is in, innermost first: the one the symbolizer named, then
    /// each caller that inlined it.
    ///
    /// ```text
    /// 0x1044c81f0: profile_a_program::churn (/path/to/program+0x2c1f0)
    /// ```
    ///
    /// The shape is not cosmetic. It is what
    /// [`trim::worth_showing`](heapscope::symbol::trim::worth_showing) parses,
    /// so deciding which frames to keep from some other rendering would
    /// silently disable trimming. A frame with a recorded name and no resolved
    /// one is exactly what [`Symbolized`](heapscope::symbol::Symbolized) writes,
    /// so the rules see what they would have seen at record time.
    ///
    /// Every entry carries the frame's own address and image, because every one
    /// of them is a true answer about that one instruction.
    fn render_for_trimming(&self, frame: &Value) -> Vec<String> {
        functions_of(frame)
            .iter()
            .map(|function| self.render_addressed(frame, function))
            .collect()
    }

    /// The same functions as [`render_for_trimming`](Self::render_for_trimming),
    /// in the same order, as a flame graph wants them: the name and nothing that
    /// would keep two return addresses in one function apart.
    ///
    /// ```text
    /// profile_a_program::churn
    /// [libsystem_malloc.dylib+0x1a2b4]
    /// ```
    ///
    /// What [`FunctionNames`](heapscope::symbol::FunctionNames) writes in the
    /// library's own folded output, with a name a symbolizer resolved taking
    /// precedence over the one the running process knew. A function with no
    /// name is the image's label and the frame's `fileAddr` in brackets, or its
    /// runtime address where there is no image; `labels` is
    /// [`image_labels`](heapscope::internals::image_labels) over this profile's
    /// module map, so an image is labelled here exactly as it would have been at
    /// record time.
    fn render_for_display(&self, frame: &Value, labels: &[&str]) -> Vec<String> {
        functions_of(frame)
            .iter()
            .map(|function| {
                let mut out = String::new();
                if push_name(function, &mut out) {
                    return out;
                }
                out.push('[');
                let image = self
                    .module_of(frame)
                    .map(|at| labels[at])
                    .filter(|label| !label.is_empty())
                    .zip(frame.get("fileAddr").and_then(Value::as_address));
                match image {
                    Some((label, file_address)) => {
                        heapscope::output::push_display(&mut out, label);
                        out.push_str(&format!("+{file_address:#x}"));
                    }
                    None => push_hex(&mut out, frame.get("addr").and_then(Value::as_address)),
                }
                out.push(']');
                out
            })
            .collect()
    }

    /// `function` as the frame `0xADDR: name (image+0xfileaddress)`.
    fn render_addressed(&self, frame: &Value, function: &Function) -> String {
        let mut out = String::new();
        push_hex(&mut out, frame.get("addr").and_then(Value::as_address));
        out.push_str(": ");

        if push_name(function, &mut out) {
            // An offset belongs to a name the running process found, not to
            // one a symbolizer resolved for this exact address.
            if matches!(function, Function::Recorded(_)) {
                if let Some(offset) = frame.get("symbolOffset").and_then(Value::as_u64) {
                    if offset != 0 {
                        out.push_str(&format!("+{offset:#x}"));
                    }
                }
            }
        } else {
            out.push_str("???");
        }

        if let (Some(module), Some(file_address)) = (
            self.module_of(frame).map(|at| &self.modules[at]),
            frame.get("fileAddr").and_then(Value::as_address),
        ) {
            out.push_str(" (");
            heapscope::output::push_display(&mut out, &module.path);
            out.push_str(&format!("+{file_address:#x})"));
        }
        out
    }

    /// Where in the module map `frame`'s image is, if it names one that exists.
    fn module_of(&self, frame: &Value) -> Option<usize> {
        frame
            .get("module")
            .and_then(Value::as_u64)
            .and_then(|at| usize::try_from(at).ok())
            .filter(|&at| at < self.modules.len())
    }

    /// The profile as folded stacks, counted by `metric`.
    ///
    /// The same file [`Snapshot::write_folded`](heapscope::Snapshot::write_folded)
    /// writes, produced from the profile rather than from an engine — which is
    /// the whole reason this exists: on a platform where in-process
    /// symbolization names nothing, the flame graph worth drawing is the one
    /// made *after* the addresses have been resolved. So frames are function
    /// names alone, as [`render_for_display`](Self::render_for_display)
    /// describes, and two points that render alike are one line with their
    /// counts summed.
    ///
    /// Trimmed by the crate's own rule, and that is a strict improvement on
    /// trimming at record time: the rule reads frame names, so on Linux, where
    /// `dladdr` names almost nothing, it had nothing to work with and left every
    /// stack whole. Here the names exist.
    ///
    /// Inlined frames take part: see [`worth_showing_inlined`] for how a frame
    /// is judged and which of its functions names it. The judgement is made on
    /// the [`render_for_trimming`](Self::render_for_trimming) shape, which the
    /// rule parses, and the function it chooses is then written by name alone.
    pub fn to_folded(&self, metric: &str) -> Result<String, String> {
        if !METRICS.contains(&metric) {
            return Err(format!(
                "`{metric}` is not a metric; expected one of {}",
                METRICS.join(", ")
            ));
        }
        let rendered: Vec<Vec<String>> = self
            .frames()
            .iter()
            .map(|frame| self.render_for_trimming(frame))
            .collect();
        let paths: Vec<&str> = self
            .modules
            .iter()
            .map(|module| module.path.as_str())
            .collect();
        let labels = heapscope::internals::image_labels(&paths);
        // Parallel to `rendered`, function for function, so that the pair
        // `worth_showing_inlined` chooses indexes both.
        let named: Vec<Vec<String>> = self
            .frames()
            .iter()
            .map(|frame| self.render_for_display(frame, &labels))
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

            let chains: Vec<&[String]> =
                indices.iter().map(|&at| rendered[at].as_slice()).collect();
            let shown = worth_showing_inlined(&chains);

            stack.clear();
            // Outermost first, which is where a flame graph puts its root.
            for &(frame, function) in shown.iter().rev() {
                if !stack.is_empty() {
                    stack.push(';');
                }
                push_frame(&mut stack, &named[indices[frame]][function]);
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

/// Which frames of a stack are worth showing, and which function names each.
///
/// `stack` is innermost first, one entry per frame, and each entry is every
/// function that frame's address is in — innermost first, rendered in the shape
/// [`worth_showing`](heapscope::symbol::trim::worth_showing) reads, as
/// `Profile::render_for_trimming` makes them. The answer is innermost first too: one
/// `(frame, function)` pair per frame kept, indexing into `stack`.
///
/// # Judged by function
///
/// With inlining one address is in several functions at once, and judging a
/// frame by any single one of them is wrong somewhere. A `Vec::with_capacity`
/// call in `tests/symbolize.rs` is one frame that is `RawVec::with_capacity_in`,
/// inlined into `Vec::with_capacity_in`, inlined into `Vec::with_capacity`
/// **\[measured, Linux and Windows\]**. Judged by the innermost, the frame is the
/// allocation path and goes, taking with it the call the program wrote. So the
/// stack is expanded to the one the source describes and the crate's rule
/// applies to that unchanged: where the cut falls no longer depends on where
/// the optimiser happened to leave a frame boundary. The same holds at the
/// other end, where on Windows and Linux `std`'s runtime marker arrives inlined
/// into its caller **\[measured\]** and is found there.
///
/// # Named by the outermost function kept
///
/// A frame that keeps any of its functions is shown by the outermost of them,
/// which is the function the address physically lies in whenever that
/// survives. That is the name `atos`, `dladdr` and `SymFromAddr` give the same
/// frame, and the name the library writes at record time, so a profile
/// symbolized here and one named in-process agree about it. The innermost would
/// not: a program function with `Vec::with_capacity` inlined into it would be
/// shown as `Vec::with_capacity_in` and the program's own name would vanish
/// from the stack. Where the physical function is itself cut — the runtime
/// marker, inlined into the frame that calls the thread's closure — the
/// outermost function kept is the one inside it.
fn worth_showing_inlined(stack: &[&[String]]) -> Vec<(usize, usize)> {
    let mut functions: Vec<String> = Vec::new();
    let mut owners: Vec<(usize, usize)> = Vec::new();
    for (frame, chain) in stack.iter().enumerate() {
        for (function, rendered) in chain.iter().enumerate() {
            functions.push(rendered.clone());
            owners.push((frame, function));
        }
    }

    // The kept range is contiguous and the owners ascend, so a frame's kept
    // functions are adjacent and the last of them is its outermost.
    let mut shown: Vec<(usize, usize)> = Vec::new();
    for at in heapscope::symbol::trim::worth_showing(&functions) {
        let (frame, function) = owners[at];
        match shown.last_mut() {
            Some(last) if last.0 == frame => last.1 = function,
            _ => shown.push((frame, function)),
        }
    }
    shown
}

/// The two labels the library's emitters give a point with no frames. Repeated
/// as text rather than shared because they are `pub(super)` there — and because
/// what has to match is the *file*, which a test compares.
const OVERFLOW_FRAME: &str =
    "[overflow]: allocations recorded after the program-point table filled up";
const UNWALKABLE_FRAME: &str = "[unwalkable]: no frame pointer chain at this allocation";

/// One of the functions a frame's address is in, and what is known of its name.
enum Function<'a> {
    /// What a symbolizer resolved, already demangled.
    Resolved(&'a str),
    /// What the running process knew the frame by, still mangled.
    Recorded(&'a str),
    /// A function the symbolizer placed the address in without naming.
    Unnamed,
}

/// Every function `frame`'s address is in, innermost first: the one the frame
/// is named by, then each caller that inlined it.
///
/// Each level is what this tool resolved for it. Where it resolved nothing,
/// the outermost level, the function the address physically lies in, falls
/// back to what the running process knew: `symbol` is the name the image's
/// symbol table gives that address, which is that function's, and it is what
/// the library writes for the same frame. A level inside it has no such
/// fallback, since the recorded name is not its name.
///
/// An inner level with no name is kept, as [`Function::Unnamed`], rather than
/// dropped. It is a real level of the call chain, and the trimming rules treat
/// a level they cannot read as program code: dropping it would let the leading
/// run of allocation-path functions continue through code that might be the
/// program's own and remove it.
fn functions_of(frame: &Value) -> Vec<Function<'_>> {
    let callers = frame
        .get("inlinedBy")
        .and_then(Value::as_array)
        .unwrap_or(&[]);
    let entries = std::iter::once(frame).chain(callers);
    let outermost = callers.len();
    entries
        .enumerate()
        .map(|(level, entry)| match resolved_name(entry) {
            Some(name) => Function::Resolved(name),
            None if level == outermost => frame
                .get("symbol")
                .and_then(Value::as_str)
                .map_or(Function::Unnamed, Function::Recorded),
            None => Function::Unnamed,
        })
        .collect()
}

/// Appends `function`'s name, screened, and says whether there was one.
///
/// A recorded name is still mangled in the file, because the format keeps the
/// linker's own spelling, and is demangled the way every other reader of this
/// crate does it. A name that comes out empty is no name: a frame of nothing
/// would be a nameless level in a flame graph.
fn push_name(function: &Function, out: &mut String) -> bool {
    let before = out.len();
    match *function {
        Function::Resolved(name) => heapscope::output::push_display(out, name),
        Function::Recorded(symbol) => {
            let mut demangled = String::new();
            if heapscope::demangle(symbol, &mut demangled) {
                heapscope::output::push_display(out, &demangled);
            } else {
                heapscope::output::push_display(out, symbol);
            }
        }
        Function::Unnamed => {}
    }
    out.len() > before
}

/// The name a symbolizer resolved for `entry`, a frame or one of its
/// `inlinedBy` callers, if it resolved one.
///
/// This tool now leaves `function` out where a symbolizer placed an address
/// without naming it. Version 0.1.0 wrote `"???"` there instead, in frames and
/// in `inlinedBy` entries alike, and profiles it rewrote are on disk, so that
/// spelling is read as what it meant: no name. Taken as a name it would outrank
/// the `symbol` the process recorded and draw every such frame as one.
fn resolved_name(entry: &Value) -> Option<&str> {
    entry
        .get("function")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty() && *name != UNNAMED_IN_0_1_0)
}

/// What version 0.1.0 of this tool wrote as the `function` of a frame it had a
/// location for and no name. See [`resolved_name`].
const UNNAMED_IN_0_1_0: &str = "???";

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
                function: Some(String::from(function)),
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

    /// Version 2 is a restarted run, and nothing this tool does depends on what
    /// the totals cover, so it is read like version 1 rather than refused.
    #[test]
    fn a_restarted_run_is_read_like_any_other() {
        let restarted = a_profile().replace(r#""formatVersion":1"#, r#""formatVersion":2"#);
        assert_ne!(
            restarted,
            a_profile(),
            "the fixture has a version to change"
        );
        let mut profile = Profile::parse(&restarted).expect("a version 2 profile");
        profile.resolve_frame(0, &resolution("program::churn"));
        assert!(profile.to_json().contains(r#""formatVersion":2"#));
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
                        function: Some(String::from("inner")),
                        file: None,
                        line: None,
                    },
                    Frame {
                        function: Some(String::from("outer")),
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
        assert_eq!(by_file[&0], vec![(0, 0xff), (1, 0x1ff)]);

        let by_runtime = profile.batches(true);
        assert_eq!(by_runtime[&0], vec![(0, 0x10ff), (1, 0x11ff)]);
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
        assert_eq!(profile.batches(false)[&0], vec![(1, 0x1ff)]);
        assert_eq!(profile.resolved_frames(), 1);
    }

    #[test]
    fn a_folded_rendering_uses_the_best_name_available() {
        let mut profile = Profile::parse(&a_profile()).expect("a native profile");
        profile.resolve_frame(0, &resolution("program::churn"));
        let folded = profile.to_folded("totalBytes").expect("a metric");

        // Frame 0 resolved, frame 1 only known in-process and demangled from
        // what the file carries, frame 2 known nowhere and in no image. Names
        // alone, as the library's folded output has them: no address, no
        // offset from the symbol, no image.
        assert_eq!(
            folded,
            "core::fmt::write;program::churn 4096\n[0x9999] 1024\n"
        );
    }

    /// A frame nothing named keeps what tells it apart from its neighbours,
    /// the image and the file address, with the image cut to its file name.
    #[test]
    fn an_unnamed_frame_is_its_image_file_name_and_file_address() {
        let profile = Profile::parse(&a_profile()).expect("a native profile");
        let folded = profile.to_folded("totalBytes").expect("a metric");
        assert!(
            folded.starts_with("core::fmt::write;[program+0x100] "),
            "{folded}"
        );
    }

    /// The flame graph merge, from a profile: two points that differ only in
    /// which return address in one function they passed through are one stack.
    /// Their counts are summed, so the file still adds up to the profile.
    #[test]
    fn two_return_addresses_in_one_function_are_one_line() {
        let text = r#"{
  "format":"heapscope-profile","formatVersion":1,
  "frames":[
    {"addr":"0x1100","module":0,"fileAddr":"0x100","symbol":"_ZN7program5churn17h0123456789abcdefE","symbolOffset":16},
    {"addr":"0x1140","module":0,"fileAddr":"0x140","symbol":"_ZN7program5churn17h0123456789abcdefE","symbolOffset":80},
    {"addr":"0x1200","module":0,"fileAddr":"0x200","symbol":"_ZN7program4main17h0123456789abcdefE","symbolOffset":8}
  ],
  "points":[
    {"kind":"recorded","totalBytes":4096,"frames":[0,2]},
    {"kind":"recorded","totalBytes":1024,"frames":[1,2]}
  ],
  "modules":[{"path":"/bin/program","load":"0x1000","start":"0x1000","size":4096,"bias":"0x1000"}]
}"#;
        let profile = Profile::parse(text).expect("a native profile");
        assert_eq!(
            profile.to_folded("totalBytes").expect("a metric"),
            "program::main;program::churn 5120\n"
        );
    }

    /// Two frames a symbolizer placed in a file without naming, under one
    /// caller. Neither has a name, so neither may be drawn as one: the first
    /// falls back to the name the process recorded, the second to its image and
    /// file address, and the two stay two stacks.
    #[test]
    fn a_location_without_a_name_is_not_a_name() {
        let location = Resolution {
            frames: vec![Frame {
                function: None,
                file: Some(String::from("/src/main.rs")),
                line: Some(7),
            }],
        };
        let text = r#"{
  "format":"heapscope-profile","formatVersion":1,
  "frames":[
    {"addr":"0x1100","module":0,"fileAddr":"0x100","symbol":"_ZN7program5churn17h0123456789abcdefE","symbolOffset":16},
    {"addr":"0x1140","module":0,"fileAddr":"0x140"},
    {"addr":"0x1200","module":0,"fileAddr":"0x200"}
  ],
  "points":[
    {"kind":"recorded","totalBytes":4096,"frames":[0,2]},
    {"kind":"recorded","totalBytes":1024,"frames":[1,2]}
  ],
  "modules":[{"path":"/bin/program","load":"0x1000","start":"0x1000","size":4096,"bias":"0x1000"}]
}"#;
        let mut profile = Profile::parse(text).expect("a native profile");
        profile.resolve_frame(0, &location);
        profile.resolve_frame(1, &location);
        profile.resolve_frame(2, &resolution("program::main"));

        assert_eq!(
            profile.to_folded("totalBytes").expect("a metric"),
            "program::main;program::churn 4096\nprogram::main;[program+0x140] 1024\n"
        );
        // The location is kept, and is not counted as a name.
        let after = json::parse(&profile.to_json()).expect("parses");
        let frames = after
            .get("frames")
            .and_then(Value::as_array)
            .expect("frames");
        assert_eq!(frames[1].get("function"), None);
        assert_eq!(frames[1].get("line").and_then(Value::as_u64), Some(7));
        assert_eq!(profile.resolved_frames(), 1);
        // And a frame that has its answer is not asked about again.
        assert!(profile.batches(false).is_empty());
    }

    /// Version 0.1.0 of this tool wrote `"???"` as the `function` of a frame
    /// it had only a location for, and profiles it rewrote exist. That spelling
    /// is read as no name, or every such frame would be drawn as one.
    #[test]
    fn the_placeholder_an_older_version_wrote_is_not_a_name() {
        let text = a_profile().replace(
            r#"{"addr":"0x1100","module":0,"fileAddr":"0x100"}"#,
            r#"{"addr":"0x1100","module":0,"fileAddr":"0x100","function":"???","file":"/src/main.rs","line":7}"#,
        );
        let profile = Profile::parse(&text).expect("a native profile");
        assert_eq!(
            profile.to_folded("totalBytes").expect("a metric"),
            "core::fmt::write;[program+0x100] 4096\n[0x9999] 1024\n"
        );
        assert_eq!(profile.resolved_frames(), 0);
    }

    /// Two images with one file name would put their unnamed frames under one
    /// label, and two such frames at one file address would merge into a frame
    /// that is neither. The library's rule keeps them apart, and this tool uses
    /// the library's rule.
    #[test]
    fn images_that_share_a_file_name_keep_their_paths() {
        let text = r#"{
  "format":"heapscope-profile","formatVersion":1,
  "frames":[
    {"addr":"0x1100","module":0,"fileAddr":"0x100"},
    {"addr":"0x3100","module":1,"fileAddr":"0x100"}
  ],
  "points":[
    {"kind":"recorded","totalBytes":4096,"frames":[0]},
    {"kind":"recorded","totalBytes":1024,"frames":[1]}
  ],
  "modules":[
    {"path":"/opt/one/libsame.so","load":"0x1000","start":"0x1000","size":4096,"bias":"0x1000"},
    {"path":"/opt/two/libsame.so","load":"0x3000","start":"0x3000","size":4096,"bias":"0x3000"}
  ]
}"#;
        let profile = Profile::parse(text).expect("a native profile");
        assert_eq!(
            profile.to_folded("totalBytes").expect("a metric"),
            "[/opt/one/libsame.so+0x100] 4096\n[/opt/two/libsame.so+0x100] 1024\n"
        );
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
                    function: Some(String::from("evil\u{1b}[2J\u{202e}gnp.eslaf")),
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

    /// A resolution naming `functions`, innermost first.
    fn inlined(functions: &[&str]) -> Resolution {
        Resolution {
            frames: functions
                .iter()
                .map(|function| Frame {
                    function: Some(String::from(*function)),
                    file: None,
                    line: None,
                })
                .collect(),
        }
    }

    /// A profile of one point whose stack is `depth` frames, all in one image.
    fn a_stack(depth: usize) -> Profile {
        let frames: Vec<String> = (0..depth)
            .map(|at| {
                format!(
                    r#"{{"addr":"{:#x}","module":0,"fileAddr":"{at:#x}"}}"#,
                    0x1000 + at
                )
            })
            .collect();
        let indices: Vec<String> = (0..depth).map(|at| at.to_string()).collect();
        Profile::parse(&format!(
            r#"{{"format":"heapscope-profile","formatVersion":1,
                "frames":[{}],
                "points":[{{"kind":"recorded","totalBytes":1,"frames":[{}]}}],
                "modules":[{{"path":"/bin/program","load":"0x1000"}}]}}"#,
            frames.join(","),
            indices.join(",")
        ))
        .expect("a native profile")
    }

    /// The names a folded stack shows, innermost first. A folded frame is a
    /// name and nothing else, so the frames are the names.
    fn shown(profile: &Profile) -> Vec<String> {
        let folded = profile.to_folded("totalBytes").expect("a metric");
        let stack = folded
            .lines()
            .next()
            .expect("a line")
            .rsplit_once(' ')
            .expect("a count")
            .0;
        stack.rsplit(';').map(String::from).collect()
    }

    /// Allocation machinery inlined into the call the program wrote, as
    /// `tests/symbolize.rs` measures it. Judged by its innermost function the
    /// frame would go, and `Vec::with_capacity` with it. It stays, and is named
    /// as `atos` and the library name it: by the function it lies in.
    #[test]
    fn a_frame_with_machinery_inlined_into_it_keeps_its_own_name() {
        let mut profile = a_stack(3);
        profile.resolve_frame(0, &inlined(&["__rustc::__rust_alloc"]));
        profile.resolve_frame(
            1,
            &inlined(&[
                "<alloc::raw_vec::RawVec<u8>>::with_capacity_in",
                "<alloc::vec::Vec<u8>>::with_capacity_in",
                "<alloc::vec::Vec<u8>>::with_capacity",
            ]),
        );
        profile.resolve_frame(2, &inlined(&["program::grow"]));
        assert_eq!(
            shown(&profile),
            ["<alloc::vec::Vec<u8>>::with_capacity", "program::grow"]
        );
    }

    /// The case that decides between innermost and outermost: the program's own
    /// function is the physical frame, with `Vec` and its machinery inlined into
    /// it — what `rustc -O` makes of an `#[inline(always)]` helper calling
    /// `Vec::with_capacity` from an `#[inline(never)]` one. Named by the
    /// innermost function kept, the stack would read `Vec::with_capacity_in` and
    /// `t::outer` would vanish from it.
    #[test]
    fn a_program_function_with_vec_inlined_into_it_is_shown_by_its_own_name() {
        let mut profile = a_stack(2);
        profile.resolve_frame(
            0,
            &inlined(&[
                "alloc::alloc::alloc",
                "<alloc::alloc::Global as core::alloc::Allocator>::allocate",
                "<alloc::raw_vec::RawVecInner>::try_allocate_in",
                "<alloc::raw_vec::RawVec<u8>>::with_capacity_in",
                "<alloc::vec::Vec<u8>>::with_capacity_in",
                "<alloc::vec::Vec<u8>>::with_capacity",
                "t::inner",
                "t::outer",
            ]),
        );
        profile.resolve_frame(1, &inlined(&["t::main"]));
        assert_eq!(shown(&profile), ["t::outer", "t::main"]);
    }

    /// A frame that is machinery all the way out goes whole, however much was
    /// inlined into it.
    #[test]
    fn a_frame_that_is_machinery_throughout_is_trimmed() {
        let mut profile = a_stack(2);
        profile.resolve_frame(
            0,
            &inlined(&[
                "alloc::alloc::alloc",
                "<alloc::alloc::Global as core::alloc::Allocator>::allocate",
                "<alloc::raw_vec::RawVecInner>::finish_grow",
            ]),
        );
        profile.resolve_frame(1, &inlined(&["program::grow"]));
        assert_eq!(shown(&profile), ["program::grow"]);
    }

    /// A frame nothing was trimmed from is named by the function it lies in,
    /// as every in-process lookup names it.
    #[test]
    fn a_kept_frame_is_named_by_the_function_it_lies_in() {
        let mut profile = a_stack(1);
        profile.resolve_frame(0, &inlined(&["program::helper", "program::outer"]));
        assert_eq!(shown(&profile), ["program::outer"]);
    }

    /// The runtime marker, found where Windows puts it: inlined into the frame
    /// that calls the thread's closure. The frame stays, because a function
    /// inside the marker is in it, and is named by that function, the outermost
    /// one kept; everything outside goes.
    #[test]
    fn an_inlined_runtime_marker_still_ends_the_stack() {
        let mut profile = a_stack(3);
        profile.resolve_frame(0, &inlined(&["program::allocates"]));
        profile.resolve_frame(
            1,
            &inlined(&[
                "program::main",
                "std::sys::backtrace::__rust_begin_short_backtrace::<fn()>",
            ]),
        );
        profile.resolve_frame(2, &inlined(&["std::rt::lang_start_internal"]));
        assert_eq!(shown(&profile), ["program::allocates", "program::main"]);
    }

    /// One level of a resolution, named or not.
    fn level(function: Option<&str>) -> Frame {
        Frame {
            function: function.map(String::from),
            file: Some(String::from("/src/main.rs")),
            line: Some(3),
        }
    }

    /// An inlined level the symbolizer could not name is still a level. In the
    /// middle of a chain of machinery it stops the leading run of
    /// allocation-path functions, as any frame the rules cannot read does, so
    /// whatever program code it may be is not trimmed away. Without it this
    /// frame would be machinery throughout and go.
    #[test]
    fn an_unnamed_middle_level_is_kept_and_stops_the_trim() {
        let mut profile = a_stack(2);
        profile.resolve_frame(
            0,
            &Resolution {
                frames: vec![
                    level(Some("alloc::alloc::alloc")),
                    level(None),
                    level(Some("<alloc::raw_vec::RawVecInner>::finish_grow")),
                ],
            },
        );
        profile.resolve_frame(1, &inlined(&["program::main"]));
        assert_eq!(
            shown(&profile),
            [
                "<alloc::raw_vec::RawVecInner>::finish_grow",
                "program::main"
            ]
        );
    }

    /// The outermost level is the function the address physically lies in,
    /// and the name the running process recorded for the address is that
    /// function's. Where the symbolizer left that level unnamed, the recorded
    /// name stands in, as it does in the library's own output.
    #[test]
    fn an_unnamed_outermost_level_takes_the_recorded_name() {
        let text = r#"{
  "format":"heapscope-profile","formatVersion":1,
  "frames":[
    {"addr":"0x1100","module":0,"fileAddr":"0x100","symbol":"_ZN7program5churn17h0123456789abcdefE","symbolOffset":16},
    {"addr":"0x1200","module":0,"fileAddr":"0x200"}
  ],
  "points":[{"kind":"recorded","totalBytes":64,"frames":[0,1]}],
  "modules":[{"path":"/bin/program","load":"0x1000","start":"0x1000","size":4096,"bias":"0x1000"}]
}"#;
        let mut profile = Profile::parse(text).expect("a native profile");
        profile.resolve_frame(
            0,
            &Resolution {
                frames: vec![level(Some("alloc::alloc::alloc")), level(None)],
            },
        );
        profile.resolve_frame(1, &inlined(&["program::main"]));
        assert_eq!(shown(&profile), ["program::churn", "program::main"]);
    }

    /// Version 0.1.0 wrote `"???"` into `inlinedBy` entries as well as into
    /// frames. Read as no name there too, it is an unnamed level like the ones
    /// above, and not a function called `???`.
    #[test]
    fn the_placeholder_an_older_version_wrote_into_an_inlined_caller_is_not_a_name() {
        let text = r#"{
  "format":"heapscope-profile","formatVersion":1,
  "frames":[
    {"addr":"0x1100","module":0,"fileAddr":"0x100","function":"alloc::alloc::alloc","inlinedBy":[{"function":"???","file":"/src/main.rs","line":3}]},
    {"addr":"0x1200","module":0,"fileAddr":"0x200","function":"program::main"}
  ],
  "points":[{"kind":"recorded","totalBytes":64,"frames":[0,1]}],
  "modules":[{"path":"/bin/program","load":"0x1000","start":"0x1000","size":4096,"bias":"0x1000"}]
}"#;
        let profile = Profile::parse(text).expect("a native profile");
        assert_eq!(
            profile.to_folded("totalBytes").expect("a metric"),
            "program::main;[program+0x100] 64\n"
        );
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
