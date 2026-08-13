//! `heapscope-symbolize` — resolve a native profile's addresses against the
//! binaries on disk.
//!
//! # Why this exists
//!
//! A profile records return addresses and a module map, and gives each frame the
//! name the running process knew it by. How many frames get one is a property of
//! the platform rather than of this crate: Apple's loader reads an image's whole
//! symbol table, while `dladdr` on ELF sees only what an image *exports*, which
//! for a Rust executable is very nearly nothing. **Measured on one program: 52
//! of 52 frames named on macOS aarch64, 0 of 70 on Linux aarch64.**
//!
//! So on Linux the profile a run writes is a wall of `??? (/path/to/program+0x…)`.
//! Everything needed to fix that is already in the file — that is what the
//! module map and the file addresses are *for* — and until now fixing it meant
//! running `llvm-symbolizer` by hand and reading the answers back into the
//! profile yourself. This is that, done properly and in one pass.
//!
//! It also unlocks trimming. The rule that drops the allocation path above a
//! stack and the runtime entry below it reads frame *names*, so where nothing is
//! named nothing is trimmed. Resolving first is what lets the folded output
//! below cut the same 93-of-144 frames the macOS profile cuts at record time.
//!
//! # What it does not do
//!
//! It reads the **native** profile, not the DHAT v2 one. A DHAT frame is a
//! string, so a DHAT file has no addresses left to resolve — which is one of the
//! things `Output::native` exists for. Ask for both; they come from one reading
//! of the engine.
//!
//! It does not rewrite the bundled HTML page. The page carries the native
//! profile verbatim and renders from display names chosen when it was written,
//! so a symbolized profile shows its new names in the JSON and in the folded
//! output rather than in that page.

mod json;
mod profile;
mod tool;

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use profile::Profile;
use tool::Tool;

const USAGE: &str = "\
heapscope-symbolize — resolve a native profile's addresses against the binaries on disk

USAGE:
    heapscope-symbolize [OPTIONS] <PROFILE>

ARGS:
    <PROFILE>              A profile written by `Output::native`, or `-` for stdin.

OPTIONS:
    -o, --output <PATH>    Where to write. Default: standard output.
    -f, --format <FORMAT>  `native` (default) writes the profile back with names
                           added; `folded` writes folded stacks for a flame graph.
        --metric <NAME>    Which counter `--format folded` carries. One of
                           totalBytes (default), totalBlocks, atGmaxBytes, atEndBytes.
        --tool <TOOL>      atos, llvm-symbolizer, or addr2line. Default: whichever
                           of those is installed, best for this platform first.
        --binary <OLD=NEW> Resolve the image recorded as OLD using the file at NEW.
                           Repeatable. This is how a profile recorded on one
                           machine is symbolized on another, against an archived
                           build, which is what the recorded build identity is for.
    -q, --quiet            Do not report progress on standard error.
    -h, --help             Print this.
    -V, --version          Print the version.

EXAMPLES:
    heapscope-symbolize profile.native.json -o resolved.json
    heapscope-symbolize profile.native.json -f folded | inferno-flamegraph > heap.svg
    heapscope-symbolize profile.native.json --binary /build/app=./target/release/app
";

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(complaint) => {
            eprintln!("heapscope-symbolize: {complaint}");
            ExitCode::FAILURE
        }
    }
}

struct Options {
    input: PathBuf,
    output: Option<PathBuf>,
    folded: bool,
    metric: String,
    tool: Option<Tool>,
    /// Recorded image path to the file that should be read instead.
    replacements: BTreeMap<String, PathBuf>,
    quiet: bool,
}

fn run() -> Result<ExitCode, String> {
    let Some(options) = parse_arguments()? else {
        // `--help` and `--version` already printed what they had to say.
        return Ok(ExitCode::SUCCESS);
    };

    let text = read_input(&options.input)?;
    let mut profile =
        Profile::parse(&text).map_err(|error| format!("{}: {error}", options.input.display()))?;

    let tool = match options.tool {
        Some(tool) => {
            if !tool.available() {
                return Err(format!("{} is not installed", tool.name()));
            }
            tool
        }
        None => *Tool::preference()
            .iter()
            .find(|tool| tool.available())
            .ok_or_else(|| {
                format!(
                    "no symbolizer is installed. This needs one of: {}",
                    Tool::preference()
                        .iter()
                        .map(|tool| tool.name())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })?,
    };

    let outcome = symbolize(&mut profile, tool, &options);
    report(&outcome, tool, &profile, &options);

    let rendered = if options.folded {
        profile.to_folded(&options.metric)?
    } else {
        profile.to_json()
    };
    write_output(options.output.as_deref(), &rendered)?;

    verdict(&outcome, tool)?;
    Ok(ExitCode::SUCCESS)
}

/// Whether a finished run is a failure despite having written a file.
///
/// A run that resolved nothing at all, from images that were actually
/// consulted, is a failure with a zero exit status unless it says so. The usual
/// cause is a binary that is not the one the profile was recorded against —
/// which produces exactly this: no errors, no names, and a file that looks fine.
///
/// Separate and pure so that it can be tested. The condition it exists for needs
/// a binary that is the wrong build for a profile, which is not something a test
/// can conjure portably, so an end-to-end test of it does not exist — and the
/// end-to-end test that was believed to cover it turned out to be reporting a
/// non-zero exit from an unwritable `-o /dev/null` instead. A rule nothing
/// exercises is the failure this crate keeps finding; the remedy here is to make
/// the rule a thing a unit test can hold.
fn verdict(outcome: &Outcome, tool: Tool) -> Result<(), String> {
    if outcome.resolved == 0 && outcome.asked > 0 {
        return Err(format!(
            "{} resolved none of the {} addresses asked about. \
             The images may not be the build this profile was recorded against; \
             `--binary OLD=NEW` points at another copy",
            tool.name(),
            outcome.asked
        ));
    }
    Ok(())
}

#[derive(Default)]
struct Outcome {
    /// Addresses an image actually answered about.
    ///
    /// Not addresses this run would have liked to know. An image that is not on
    /// this disk contributes none of them — that arm has always `continue`d
    /// before counting — and neither does one the symbolizer could not read.
    /// Both appear in `skipped` instead.
    ///
    /// The distinction decides the exit status, through [`verdict`]. Counting
    /// an image that errored made a run whose only remaining work was such an
    /// image report that it resolved none of the addresses it asked about,
    /// which is the sentence reserved for a profile pointed at the wrong build.
    /// **Measured on Windows**, where the system images have no PDB and never
    /// will, so a second run over an already-resolved profile has nothing else
    /// left and failed the idempotence half of `tests/symbolize.rs`.
    asked: usize,
    resolved: usize,
    /// Images that could not be consulted, and why.
    skipped: Vec<(String, String)>,
}

/// Resolves every frame the profile has an image for.
fn symbolize(profile: &mut Profile, tool: Tool, options: &Options) -> Outcome {
    let mut outcome = Outcome::default();
    let batches = profile.batches(tool.wants_runtime_addresses());

    for (module, frames) in batches {
        let recorded = profile.modules()[module].clone();
        let image = options
            .replacements
            .get(&recorded.path)
            .cloned()
            .unwrap_or_else(|| PathBuf::from(&recorded.path));

        // Checked before spawning, so that the ordinary case — a profile
        // recorded elsewhere, or a system image this machine does not have —
        // reports one clear line rather than whatever the tool says about a
        // path it cannot open.
        if !image.exists() {
            outcome
                .skipped
                .push((recorded.path.clone(), String::from("no such file here")));
            continue;
        }

        let addresses: Vec<u64> = frames.iter().map(|&(_, address)| address).collect();

        match tool::resolve(tool, &image.to_string_lossy(), recorded.load, &addresses) {
            Ok(answers) => {
                // Counted here, not before the call: see `Outcome::asked`.
                outcome.asked += addresses.len();
                for (&(at, _), answer) in frames.iter().zip(&answers) {
                    if let Some(resolution) = answer {
                        profile.resolve_frame(at, resolution);
                        outcome.resolved += 1;
                    }
                }
            }
            // One image failing is not the run failing: a profile spans the
            // program and every library it loaded, and the program's own frames
            // are the ones worth having.
            Err(complaint) => outcome.skipped.push((recorded.path.clone(), complaint)),
        }
    }
    outcome
}

fn report(outcome: &Outcome, tool: Tool, profile: &Profile, options: &Options) {
    if options.quiet {
        return;
    }
    for (image, reason) in &outcome.skipped {
        eprintln!("heapscope-symbolize: skipped {image}: {reason}");
    }
    eprintln!(
        "heapscope-symbolize: {} resolved {} of {} addresses; {} of {} frames now named",
        tool.name(),
        outcome.resolved,
        outcome.asked,
        profile.resolved_frames(),
        profile.frame_count()
    );
}

fn read_input(path: &Path) -> Result<String, String> {
    if path == Path::new("-") {
        let mut text = String::new();
        return std::io::Read::read_to_string(&mut std::io::stdin().lock(), &mut text)
            .map(|_| text)
            .map_err(|error| format!("reading standard input: {error}"));
    }
    std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))
}

/// Writes `text` to `path`, or to standard output.
///
/// Written beside its destination and renamed into place, which is the promise
/// every `save_*` method in the library makes and the same reason: a full disk
/// or a killed process must leave the file that was already there intact rather
/// than replacing it with half a profile. That matters more here than there,
/// because the obvious way to use this tool is to point `--output` at the
/// profile it just read.
fn write_output(path: Option<&Path>, text: &str) -> Result<(), String> {
    let Some(path) = path else {
        return std::io::stdout()
            .lock()
            .write_all(text.as_bytes())
            // A closed pipe is what `| head` looks like, and is not an error
            // worth a message.
            .or_else(|error| match error.kind() {
                std::io::ErrorKind::BrokenPipe => Ok(()),
                _ => Err(format!("writing to standard output: {error}")),
            });
    };

    let temporary = match path.file_name() {
        Some(name) => {
            let mut name = name.to_os_string();
            name.push(format!(".{}.tmp", std::process::id()));
            path.with_file_name(name)
        }
        // No file name to write beside, so there is nothing to protect.
        None => return std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display())),
    };

    let written = std::fs::write(&temporary, text).and_then(|()| std::fs::rename(&temporary, path));
    if written.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    written.map_err(|error| format!("{}: {error}", path.display()))
}

/// Reads the command line.
///
/// Hand-written because the library has no dependencies and a binary that pulled
/// in an argument parser would be the first thing in this repository that did.
/// `Ok(None)` means the run is over and said so already — `--help`, `--version`.
fn parse_arguments() -> Result<Option<Options>, String> {
    let mut input: Option<PathBuf> = None;
    let mut output = None;
    let mut folded = false;
    let mut metric = String::from("totalBytes");
    let mut tool = None;
    let mut replacements = BTreeMap::new();
    let mut quiet = false;

    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let mut value = |name: &str| -> Result<String, String> {
            arguments
                .next()
                .ok_or_else(|| format!("{name} needs a value"))
        };
        match argument.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(None);
            }
            "-V" | "--version" => {
                println!("heapscope-symbolize {}", env!("CARGO_PKG_VERSION"));
                return Ok(None);
            }
            "-q" | "--quiet" => quiet = true,
            "-o" | "--output" => output = Some(PathBuf::from(value("--output")?)),
            "-f" | "--format" => {
                let requested = value("--format")?;
                folded = match requested.as_str() {
                    "native" => false,
                    "folded" => true,
                    other => {
                        return Err(format!(
                            "`{other}` is not a format; expected native or folded"
                        ))
                    }
                };
            }
            "--metric" => {
                metric = value("--metric")?;
                if !profile::METRICS.contains(&metric.as_str()) {
                    return Err(format!(
                        "`{metric}` is not a metric; expected one of {}",
                        profile::METRICS.join(", ")
                    ));
                }
            }
            "--tool" => {
                let requested = value("--tool")?;
                tool = Some(Tool::parse(&requested).ok_or_else(|| {
                    format!(
                        "`{requested}` is not a symbolizer this knows; expected one of {}",
                        Tool::preference()
                            .iter()
                            .map(|tool| tool.name())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })?);
            }
            "--binary" => {
                let mapping = value("--binary")?;
                // Split at the first `=`, because a path can contain one and the
                // recorded path is the half this has to match exactly.
                let (from, to) = mapping
                    .split_once('=')
                    .ok_or_else(|| format!("`--binary {mapping}` is not OLD=NEW"))?;
                replacements.insert(String::from(from), PathBuf::from(to));
            }
            other if other.starts_with('-') && other != "-" => {
                return Err(format!("unknown option `{other}`. `--help` lists them"));
            }
            path if input.is_none() => input = Some(PathBuf::from(path)),
            extra => return Err(format!("unexpected argument `{extra}`")),
        }
    }

    let Some(input) = input else {
        eprint!("{USAGE}");
        return Err(String::from("no profile given"));
    };
    Ok(Some(Options {
        input,
        output,
        folded,
        metric,
        tool,
        replacements,
        quiet,
    }))
}

#[cfg(test)]
mod tests {
    use super::{verdict, Outcome, Tool};

    fn skipped(reason: &str) -> Vec<(String, String)> {
        vec![(String::from("/some/image"), String::from(reason))]
    }

    /// The condition the rule exists for: images were consulted and answered
    /// with nothing. A profile pointed at the wrong build looks exactly like
    /// this, and looks like success everywhere else.
    #[test]
    fn a_run_that_consulted_images_and_named_nothing_is_a_failure() {
        let outcome = Outcome {
            asked: 12,
            resolved: 0,
            skipped: Vec::new(),
        };
        let complaint = verdict(&outcome, Tool::LlvmSymbolizer)
            .expect_err("resolving none of twelve addresses reported success");
        assert!(
            complaint.contains("12") && complaint.contains("--binary"),
            "the complaint says neither how many nor what to do: {complaint}"
        );
    }

    /// The condition it must *not* fire on, which is what made it wrong before:
    /// every image was skipped, so nothing was ever asked. On Windows that is
    /// an ordinary second run — the system images have no PDB — and on any
    /// platform it is a profile whose binaries are elsewhere.
    #[test]
    fn a_run_with_nothing_it_could_consult_is_not_a_failure() {
        let outcome = Outcome {
            asked: 0,
            resolved: 0,
            skipped: skipped("no such file here"),
        };
        assert!(
            verdict(&outcome, Tool::LlvmSymbolizer).is_ok(),
            "a run that could consult nothing was reported as the wrong build"
        );
    }

    /// Partial success is success. A profile spans the program and every
    /// library it loaded, and the images this machine cannot symbolize are
    /// routine rather than exceptional — which is why one image failing is not
    /// the run failing.
    #[test]
    fn resolving_some_addresses_is_not_a_failure_however_many_images_were_skipped() {
        let outcome = Outcome {
            asked: 30,
            resolved: 1,
            skipped: skipped("no debug info"),
        };
        assert!(
            verdict(&outcome, Tool::LlvmSymbolizer).is_ok(),
            "a run that named a frame was reported as having named none"
        );
    }

    /// A run with no work at all — a profile with no frames, or one already
    /// wholly resolved — is not a failure either.
    #[test]
    fn a_run_with_no_work_is_not_a_failure() {
        assert!(verdict(&Outcome::default(), Tool::LlvmSymbolizer).is_ok());
    }
}
