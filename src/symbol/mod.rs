//! Turning addresses into names.
//!
//! Nothing here runs on the allocation path. The hot path stores return
//! addresses and nothing else; everything that gives them meaning happens at
//! output time, or later, or on a different machine.
//!
//! # Why offline resolution is the primary path
//!
//! The obvious approach is to call `dladdr` while the process is alive. It does
//! not work on the binaries people actually ship: on a stripped image `dladdr`
//! returns *success* with a null symbol name, and `strip = true` is common in
//! release profiles. A profiler that symbolizes only in-process therefore
//! produces its worst output for exactly the builds most worth profiling.
//!
//! So the profile carries the [module map](modules) — image paths, load
//! addresses, and build identities — and renders frames as `image + offset`.
//! That is resolvable afterwards by `atos`, `addr2line`, or `llvm-symbolizer`,
//! against a build with symbols, on any machine. In-process `dladdr` arrives
//! later as a convenience layer on top, not as the foundation.
//!
//! # Every lookup is at the call, not the return address
//!
//! A recorded frame is where execution would *resume*: the instruction after a
//! call, which belongs to whatever the compiler placed next. Across inlining that
//! is routinely another function, and at the end of a function that never
//! returns it is the next function in the image. So everything that asks what a
//! frame *is* — which image holds it, which symbol names it — asks about
//! `call_site` instead, one byte earlier, inside the call itself. What is
//! *written* is still the recorded number: a frame's address, its file address,
//! and a symbol's offset are all measured from the return address, so the
//! format's numbers mean exactly what they did and a reader resolving one by
//! hand starts from what the stack walk saw.
//!
//! Once a frame has a name, [`trim`] can tell the frames that are about the
//! program from the ones every stack has — the allocation path above and the
//! runtime entry below — and leave the second kind out. That is downstream of
//! naming by construction, and inert wherever naming finds nothing.

pub mod demangle;
#[cfg(all(unix, not(miri)))]
mod dl;
pub mod dynamic;
pub(crate) mod labels;
pub mod modules;
pub mod trim;

use std::cell::RefCell;
use std::collections::HashMap;

use crate::output::FrameFormat;
use dynamic::Symbol;
use labels::image_labels;
use modules::Module;

pub use demangle::demangle;
pub use modules::capture as capture_modules;
pub use trim::Trimmed;

/// One address, resolved as far as this process can resolve it, in parts.
///
/// The same three questions [`Symbolized`] answers, before the answers are
/// joined into a line of text. Text is what a viewer built for Valgrind's format
/// has a column for; the native format writes the parts instead, so that a tool
/// reading it can sort by image, group by symbol, or ignore the name entirely
/// and resolve the file address itself against a build with symbols.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Resolved {
    /// Which image the address is in, as an index into the module map it was
    /// resolved against.
    ///
    /// `None` for an address in no image at all, which is what a truncated or
    /// misaligned stack walk produces.
    pub module: Option<usize>,
    /// The address as it appears in that image's file on disk.
    ///
    /// This is the number `addr2line` and `llvm-symbolizer` take. Not an offset
    /// from the load address, which is a different number on Mach-O and on a
    /// non-PIE ELF executable.
    pub file_address: Option<usize>,
    /// What the running process calls it, still mangled.
    ///
    /// `None` on a stripped image, on Linux for almost everything (`dladdr`
    /// reads `.dynsym`, which a Rust executable barely populates), and for any
    /// address in no image. Mangled because demangling is a rendering decision
    /// and this is not a rendering: `heapscope::demangle` is public, and a
    /// reader that wants the raw linker name would have no way back to it.
    pub symbol: Option<Symbol>,
}

/// Resolves `address` against `modules` and this process's own symbol tables.
///
/// The module map is consulted **first**, and that ordering is load-bearing
/// rather than an optimisation: the platform lookup cannot be trusted to refuse.
/// See [`Symbolized`], whose rendering path documents the measurement — on macOS
/// 15 arm64, `dladdr((void *)-1)` returns success and names whichever symbol is
/// last in the main executable, and `(void *)-1` is precisely what a bad stack
/// walk produces.
///
/// Located and named at its `call_site`; see the [module
/// documentation](self). The file address and the symbol offset are still
/// those of `address` itself.
pub fn resolve(modules: &[Module], address: usize) -> Resolved {
    resolve_with(modules, address, dynamic::lookup)
}

/// The address to look up for a recorded frame: one byte before it, inside the
/// call instruction rather than after it.
///
/// Every frame a stack walk records is a return address — the frame-pointer
/// walk, `backtrace`, and `RtlCaptureStackBackTrace` all report where execution
/// will *resume* — and that instruction belongs to whatever the compiler placed
/// after the call. Any byte of the call instruction would name the call, and
/// the last is the one known without decoding anything, on every
/// architecture. It is the adjustment the `backtrace` crate makes before
/// symbolizing, and so `std`'s own backtraces.
///
/// Measured on why it matters. `std` ends `RawVecInner::finish_grow` by calling
/// the allocator and then `map_err` on the result, so the return address lies
/// in the inlined `map_err`, and offline the frame read as
/// `<core::result::Result<…>>::map_err` — which trimming rightly does not
/// recognise as the allocation path. One byte earlier the same frame is
/// `alloc::alloc::alloc`, inlined through `Global::allocate` into
/// `finish_grow`. Elsewhere in the same profile a thread's entry frame read as
/// `core::mem::size_of_val_raw` and a `read_to_end` frame as `Vec::len`
/// **\[measured, Linux x86_64, rustc 1.98, binutils 2.42\]**.
///
/// `heapscope-symbolize` applies the same rule to the profile's numbers, so a
/// name found in-process and one found offline describe the same instruction.
///
/// # The one exception: a frame interrupted by a signal
///
/// An allocation made inside a signal handler has a stack that crosses the
/// signal, and two of its frames are not return addresses. The frame the signal
/// interrupted is recorded at its program counter, the instruction that had not
/// yet run; one byte earlier is the instruction before it, and where the
/// interrupted instruction was a function's first, the end of the previous
/// function. And the kernel enters the signal trampoline — `__restore_rt` on
/// glibc — by a return to its first byte, so one byte earlier names whatever
/// precedes it in the image. The profile format records no marker for a signal
/// frame, so neither can be told apart from a call; both are looked up as
/// though they were one. Only stacks that cross a signal handler are affected,
/// and allocating in one is not async-signal-safe to begin with.
///
/// `None` for zero, which no stack walk records as a return address.
///
/// <div class="warning">
///
/// `#[doc(hidden)]` and **not part of the supported surface**. It is public
/// only so that `heapscope-symbolize` applies this rule rather than a copy of it.
///
/// </div>
#[doc(hidden)]
pub fn call_site(return_address: u64) -> Option<u64> {
    return_address.checked_sub(1)
}

/// `call_site` for an address of this process.
fn call_site_of(return_address: usize) -> Option<usize> {
    // Lossless both ways: the result is no larger than the argument, which
    // was a `usize`.
    call_site(return_address as u64).map(|at| at as usize)
}

/// Which image a recorded frame is in, and its file address there.
///
/// The image is the one holding the `call_site`: a return address one past
/// the end of an image's code belongs to that image, whose last instruction
/// made the call, and one at an image's first byte does not. The file address
/// is the return address's own, translated, because that is what the format
/// records.
fn locate(modules: &[Module], address: usize) -> Option<(usize, usize)> {
    let call = call_site_of(address)?;
    let at = modules::index_containing(modules, call)?;
    let file_address = modules[at].file_address(call)? + (address - call);
    Some((at, file_address))
}

/// Names the call a recorded frame made, with the offset of the frame itself.
///
/// The symbol is the one holding the `call_site`. The offset is measured from
/// the return address, so a rendered `name+0x24` and a native `symbolOffset`
/// still say how far the *recorded* address is past the symbol: the offset
/// `lookup` reported, plus the byte stepped back.
fn name_call(lookup: fn(usize) -> Option<Symbol>, address: usize) -> Option<Symbol> {
    let call = call_site_of(address)?;
    let mut symbol = lookup(call)?;
    symbol.offset += address - call;
    Some(symbol)
}

/// Resolves using `lookup` instead of asking the platform. Testing hook.
///
/// The same hook [`Symbolized::with_lookup`] has, and for the same reason: what
/// the gate above prevents is *platform-dependent*. On macOS 15 arm64 the
/// measurement is that `dladdr((void *)-1)` succeeds; on Linux the same call
/// finds nothing, so a test asserting that an address in no image goes unnamed
/// would pass there whether or not the gate existed. A supplied lookup that
/// names everything makes the rule observable on every platform.
fn resolve_with(
    modules: &[Module],
    address: usize,
    lookup: fn(usize) -> Option<Symbol>,
) -> Resolved {
    let Some((module, file_address)) = locate(modules, address) else {
        return Resolved::default();
    };
    Resolved {
        module: Some(module),
        file_address: Some(file_address),
        symbol: name_call(lookup, address),
    }
}

/// Renders frames as an address plus the image it belongs to and the offset
/// within it.
///
/// ```text
/// 0x1044c81f0: ??? (/path/to/program+0x2c1f0)
/// ```
///
/// The three parts each earn their place. The runtime address is what `atos`
/// consumes, given the image's base from the module map. The path names the file
/// to resolve against. The last number is the address **as it appears in the
/// file**, which is what `addr2line` and `llvm-symbolizer` take — not an offset
/// from the image base, which is a different number on Mach-O, where file
/// addresses start at 0x1_0000_0000, and on a non-PIE ELF executable, where they
/// start at 0x400000.
///
/// An address in no known image keeps the bare form, because inventing an
/// attribution would be worse than saying nothing.
#[derive(Clone, Copy, Debug)]
pub struct ModuleOffsets<'a> {
    modules: &'a [Module],
}

impl<'a> ModuleOffsets<'a> {
    /// Renders against `modules`, which must be sorted by load address —
    /// [`modules::capture`] returns them that way.
    pub fn new(modules: &'a [Module]) -> Self {
        Self { modules }
    }
}

impl FrameFormat for ModuleOffsets<'_> {
    fn format(&self, address: usize, out: &mut String) {
        crate::output::RawAddresses.format(address, out);
        push_image(self.modules, address, out);
    }
}

/// Appends ` (path+0xfileaddress)` for the image `address` is in, or nothing.
///
/// Shared by both renderers, because the part of a frame that says *which file
/// to resolve against* is the part that has to be there whether or not a name
/// was found — it is what makes the frame answerable later, by a different tool,
/// on a different machine.
fn push_image(modules: &[Module], address: usize, out: &mut String) {
    let Some((module, file_address)) = locate(modules, address) else {
        return;
    };
    out.push_str(" (");
    out.push_str(&modules[module].path);
    out.push('+');
    crate::output::push_hex(out, file_address);
    out.push(')');
}

/// Renders frames with the name the running process knows them by, falling back
/// to exactly what [`ModuleOffsets`] would have said.
///
/// ```text
/// 0x1044c81f0: core::fmt::write+0x1c (/path/to/program+0x2c1f0)
/// 0x1044c9330: ??? (/path/to/program+0x2d330)
/// ```
///
/// This is tier 1 of PLAN.md section 6.1, and the shape above is the whole
/// design: the name is *added to* the module and offset rather than replacing
/// them. A profile rendered this way is readable now, by the person who ran it,
/// and still resolvable later by `atos`, `addr2line`, or `llvm-symbolizer`
/// against a build with full symbols — which matters because the names available
/// in-process are the ones the dynamic symbol table happens to export, and that
/// is a small fraction of the ones a debug build has. Dropping the offset in
/// favour of a name would trade a complete answer for a partial one.
///
/// A flame graph is the one place that trade is worth making, because it
/// merges frames by their text and the address keeps apart what the picture
/// should merge. [`FunctionNames`] makes it, and is what folded output uses.
///
/// Where a name is not available, and on a stripped binary that is everywhere,
/// the rendering is byte-for-byte what [`ModuleOffsets`] produces, so nothing is
/// lost by choosing this.
///
/// # Cost
///
/// Symbol lookup is per address and Windows charges a lock and a dbghelp call
/// for each one, so renderings are cached by address. A profile's frames repeat
/// heavily — every stack shares its outermost frames with every other — and the
/// cache is what turns a lookup per frame into a lookup per distinct address.
/// It lives as long as the renderer, which is one output operation.
pub struct Symbolized<'a> {
    modules: &'a [Module],
    /// Indirected so that tests can render against a symbol table they control.
    /// A real one has whatever this build happens to export in it, which is not
    /// something a test can assert about.
    lookup: fn(usize) -> Option<Symbol>,
    cache: Renderings,
}

impl<'a> Symbolized<'a> {
    /// Renders against the running process and `modules`, which must be sorted
    /// by load address — [`modules::capture`] returns them that way.
    pub fn new(modules: &'a [Module]) -> Self {
        Self::with_lookup(modules, dynamic::lookup)
    }

    /// Renders using `lookup` instead of asking the platform. Testing hook.
    fn with_lookup(modules: &'a [Module], lookup: fn(usize) -> Option<Symbol>) -> Self {
        Self {
            modules,
            lookup,
            cache: Renderings::new(),
        }
    }

    fn render(&self, address: usize) -> Box<str> {
        let mut out = String::new();
        crate::output::push_hex(&mut out, address);
        out.push_str(": ");

        // The module map decides whether the address is worth naming, and it is
        // consulted first because the platform lookup cannot be trusted to
        // refuse. Measured on macOS 15, arm64: `dladdr((void *)-1)` returns
        // *success*, attributes the address to the main executable, and names
        // whichever symbol is last in it.
        //
        // ```text
        // usize::MAX     rc=1 sname=_MergedGlobals.1385 saddr=0x104bc8d28 off=0xfffffffefb4372d7
        // usize::MAX-1   rc=0 sname=<null>
        // ```
        //
        // Only that one value — dyld uses it as a sentinel — but it is precisely
        // the value a truncated or misaligned stack walk produces, so the
        // in-process symbolizer's confident wrong answer would land on exactly
        // the frames least able to be checked. The map has this process's own
        // measured bounds for each image, so an address outside all of them gets
        // no name, matching what `ModuleOffsets` already documents about not
        // inventing an attribution.
        //
        // How tight those bounds are is a per-platform fact and worth not
        // overstating. On Unix they are the executable segments. On Windows they
        // are the whole image, because `K32EnumProcessModules` reports a base
        // and a size and `modules.rs` does not walk the section table — so an
        // address in a PE's `.data` passes this gate and dbghelp will name it.
        // The gate rules out addresses in *no* image, which is the case that
        // produced a confident wrong answer; it is not a claim that everything
        // it admits is code.
        //
        // It also means a garbage address costs no lookup at all, which on
        // Windows is a lock and a dbghelp call saved per bad frame.
        //
        // The consequence, accepted rather than overlooked: where the module map
        // came back empty, nothing is named even if the platform would have
        // named it. A profile with no module map cannot be resolved offline
        // either, so it is already the degraded case — and a special rule that
        // fires only in a degraded state is a rule nothing routinely exercises.
        let symbol = locate(self.modules, address).and_then(|_| name_call(self.lookup, address));

        match symbol {
            Some(symbol) => {
                push_symbol_name(&symbol.name, &mut out);
                if symbol.offset != 0 {
                    out.push('+');
                    crate::output::push_hex(&mut out, symbol.offset);
                }
            }
            // The same three characters `ModuleOffsets` uses, so that a frame
            // with no name looks the same however the profile was rendered.
            None => out.push_str("???"),
        }

        push_image(self.modules, address, &mut out);
        out.into_boxed_str()
    }
}

impl FrameFormat for Symbolized<'_> {
    fn format(&self, address: usize, out: &mut String) {
        self.cache.append(address, out, || self.render(address));
    }
}

impl std::fmt::Debug for Symbolized<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Symbolized")
            .field("modules", &self.modules.len())
            .field("cached", &self.cache.len())
            .finish()
    }
}

/// Renders frames as the name of the function they are in, and nothing else.
///
/// ```text
/// core::fmt::write
/// <alloc::vec::Vec<u8>>::with_capacity
/// [program+0x2d330]
/// ```
///
/// This is what [`Snapshot::write_folded`](crate::Snapshot::write_folded)
/// renders with, and the reason is how a flame graph is built. Its tools merge
/// frames by their **text**, so whatever a renderer puts in a frame beyond the
/// function decides what the picture keeps apart. [`Symbolized`] puts in the
/// runtime address and the offset from the symbol, which is right for a record
/// and wrong for a picture: a function that allocates through two of its calls
/// is on the stack at two return addresses, and was drawn as two frames side by
/// side where a reader expects one. It also repeats the image's absolute path
/// in every frame, which made the files large and every label in `inferno` or
/// `speedscope` a path first and a name somewhere after it.
///
/// Folded output used [`Symbolized`], wrapped in [`Trimmed`] as the default
/// `trim_frames` setting asks, up to 0.1.0. The default changed before 1.0,
/// which is when a default can still change. That rendering is one argument
/// away, through
/// [`Snapshot::write_folded_with`](crate::Snapshot::write_folded_with), as
/// `&Trimmed::new(Symbolized::new(&snapshot.modules))`, for a flame graph that
/// has to keep every return address apart.
///
/// # The name
///
/// Found exactly as [`Symbolized`] finds it: the module map is asked first, so
/// an address in no image is never named however willing the platform is, and
/// the symbol is then demangled, or printed as the linker wrote it where the
/// demangler refuses. So a frame here is always the name part of what
/// [`Symbolized`] says about the same address, byte for byte. Demangling drops
/// the hash a legacy-mangled name ends in, which is what makes `core::fmt::write`
/// one function across a whole profile rather than one per build.
///
/// Generic arguments are **kept**. `<alloc::vec::Vec<u8>>::push` and
/// `<alloc::vec::Vec<String>>::push` are separate machine code with separate
/// callees, and merging them is a further and lossy decision that a viewer's
/// search can make at reading time and a file cannot undo. The merge this type
/// exists for is between return addresses inside one function, and those share
/// their generic arguments by construction. Keeping them also keeps the text
/// [`Trimmed`]'s rules were measured against: `<alloc::boxed::Box<` is one of
/// its prefixes. (A legacy-mangled name carries no instantiation, only the hash
/// that demangling drops, so code from a toolchain that still emits those, this
/// crate's MSRV among them, has its instantiations share a frame regardless.)
///
/// # A frame with no name
///
/// Written `[image+0xfileaddress]`: the image's file name and the address as it
/// appears in that file, the second half of what [`ModuleOffsets`] says, in
/// brackets. As there, the image is the one that made the call and the number
/// is the recorded return address's, not the call site's, so it is the number
/// the native profile carries. Distinct addresses stay distinct, because
/// merging every unnamed frame in an image into one, as `stackcollapse-perf.pl`
/// does with its `[module]`, would draw call paths that never happened. The
/// brackets are that convention's, and they mark the frame as something other
/// than a function name to a reader and to [`name_of`](FrameFormat::name_of).
/// No demangled name
/// begins with one, and no symbol a compiler emits does; a garbage symbol from a
/// damaged table that did would only be left untrimmed.
///
/// The image is named by its **file name**, not its path, which is most of what
/// made the old labels unreadable. A file name can be ambiguous where a path is
/// not, and an ambiguous label is worse than a long one: two images both called
/// `libfoo.so`, in two directories, would put their unnamed frames under one
/// label, and two such frames at the same file address would merge into a frame
/// that is neither. So an image whose file name is shared with another image in
/// the module map keeps its whole path, and the labels stay as distinct as the
/// paths are. An address in no image, or in an image with no path, is written
/// as its runtime address, `[0x1044c81f0]`, which is distinct by definition.
///
/// # What this gives up
///
/// Two things [`Symbolized`] keeps, and both deliberately:
///
/// - **The offset from the symbol.** That number is a reader's only clue that a
///   name is not to be believed: `dladdr` names the nearest preceding symbol it
///   can see, and on an image with only its exported symbols left, a private
///   function is reported under whatever exported one precedes it. Here such a
///   function is drawn as part of that one. That is the cost of a merged
///   picture, and the same trade `stackcollapse-perf.pl` makes when it strips
///   the offsets from `perf` output.
/// - **Resolvability.** The runtime address and file attribution are gone from
///   every named frame, so a folded file cannot be symbolized afterwards. It is
///   a picture of a profile, not a record of one; the native profile is the
///   record, and `heapscope-symbolize` can turn it into a folded file with
///   names an offline symbolizer found.
///
/// And one consequence that is the flame graph convention rather than a loss:
/// a function name is the merge key, so two images that each contain a function
/// of the same name draw it as one frame.
///
/// # Cost
///
/// The same as [`Symbolized`], and for the same reason cached by address for as
/// long as the renderer lives.
pub struct FunctionNames<'a> {
    modules: &'a [Module],
    /// What an unnamed frame in each image is labelled, by index into
    /// `modules`. Empty for an image that has no path to label it by.
    labels: Vec<&'a str>,
    /// Indirected for the same reason as [`Symbolized`]'s.
    lookup: fn(usize) -> Option<Symbol>,
    cache: Renderings,
}

impl<'a> FunctionNames<'a> {
    /// Renders against the running process and `modules`, which must be sorted
    /// by load address — [`modules::capture`] returns them that way.
    pub fn new(modules: &'a [Module]) -> Self {
        Self::with_lookup(modules, dynamic::lookup)
    }

    /// Renders using `lookup` instead of asking the platform. Testing hook.
    fn with_lookup(modules: &'a [Module], lookup: fn(usize) -> Option<Symbol>) -> Self {
        let paths: Vec<&str> = modules.iter().map(|module| module.path.as_str()).collect();
        Self {
            modules,
            labels: image_labels(&paths),
            lookup,
            cache: Renderings::new(),
        }
    }

    fn render(&self, address: usize) -> Box<str> {
        let mut out = String::new();

        // The module map first, for the reason `Symbolized::render` measures:
        // the platform lookup names `(void *)-1`, which is what a bad stack walk
        // produces, and only the map knows that address is in nothing. Both
        // ask about the call site, as every lookup here does; see the module
        // documentation.
        let located = locate(self.modules, address);
        let symbol = located.and_then(|_| name_call(self.lookup, address));

        if let Some(symbol) = symbol {
            push_symbol_name(&symbol.name, &mut out);
            // A name that came back empty cannot be a frame: the line would
            // carry a level of the flame graph with nothing in it. `lookup`
            // already refuses an empty name; this holds the rule for a lookup
            // that does not, rather than trusting every one to.
            if !out.is_empty() {
                return out.into_boxed_str();
            }
        }

        // The image is the one that made the call and the file address is the
        // recorded one's, exactly the pair `ModuleOffsets` writes, so the
        // number here is the one a reader would look up by hand.
        out.push('[');
        let image = located
            .map(|(at, file_address)| (self.labels[at], file_address))
            .filter(|(label, _)| !label.is_empty());
        match image {
            Some((label, file_address)) => {
                out.push_str(label);
                out.push('+');
                crate::output::push_hex(&mut out, file_address);
            }
            None => crate::output::push_hex(&mut out, address),
        }
        out.push(']');
        out.into_boxed_str()
    }
}

impl FrameFormat for FunctionNames<'_> {
    fn format(&self, address: usize, out: &mut String) {
        self.cache.append(address, out, || self.render(address));
    }

    /// The whole frame, unless it is one of the bracketed stand-ins for a
    /// frame with no name.
    fn name_of<'f>(&self, frame: &'f str) -> Option<&'f str> {
        (!frame.starts_with('[')).then_some(frame)
    }
}

impl std::fmt::Debug for FunctionNames<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionNames")
            .field("modules", &self.modules.len())
            .field("cached", &self.cache.len())
            .finish()
    }
}

/// Appends `name` demangled, or as the linker wrote it where it cannot be.
///
/// Demangling refuses on anything it does not fully understand, which includes
/// every C and C++ name in the process as well as a Rust name read out of a
/// damaged table. The raw symbol is then the best available answer: ugly, but
/// what the linker actually wrote. Neither branch is screened here — the
/// emitter screens the finished frame, which is the only place that also covers
/// a `FrameFormat` this crate did not write.
///
/// The `truncate` is belt and braces: `demangle` documents and tests that it
/// leaves `out` untouched when it refuses. It is one instruction, and the
/// failure it guards against is a half-parsed name attributing an allocation to
/// code that did not make it, which is the one output error this crate has no
/// way to make visible to a reader.
///
/// Shared by [`Symbolized`] and [`FunctionNames`], whose claim to name a frame
/// exactly as the other does rests on there being one copy of this.
fn push_symbol_name(name: &str, out: &mut String) {
    let before = out.len();
    if !demangle(name, out) {
        out.truncate(before);
        out.push_str(name);
    }
}

/// Renderings already made, by address.
///
/// Symbol lookup is per address, and on Windows each one is a lock and a
/// dbghelp call. A profile's frames repeat heavily — every stack shares its
/// outermost frames with every other — so remembering each answer is what
/// turns a lookup per frame into a lookup per distinct address. Lives as long
/// as the renderer that owns it, which is one output operation.
struct Renderings(RefCell<HashMap<usize, Box<str>>>);

impl Renderings {
    fn new() -> Self {
        Self(RefCell::new(HashMap::new()))
    }

    /// Appends the rendering of `address` to `out`, making it with `render`
    /// the first time it is asked for.
    fn append(&self, address: usize, out: &mut String, render: impl FnOnce() -> Box<str>) {
        if let Some(cached) = self.0.borrow().get(&address) {
            out.push_str(cached);
            return;
        }
        // Deliberately outside the borrow above: `render` calls into the
        // platform, and holding a `RefCell` borrow across a foreign call is the
        // kind of thing that is fine until someone adds a lookup that renders.
        let rendered = render();
        out.push_str(&rendered);
        self.0.borrow_mut().insert(address, rendered);
    }

    fn len(&self) -> usize {
        self.0.borrow().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn module(path: &str, start: usize, size: usize) -> Module {
        Module {
            path: String::from(path),
            start,
            size,
            // A bias of zero makes the file address and the runtime address the
            // same, which keeps these tests about the rendering.
            bias: 0,
            image_base: start,
            build_id: None,
        }
    }

    fn render(modules: &[Module], address: usize) -> String {
        let mut out = String::new();
        ModuleOffsets::new(modules).format(address, &mut out);
        out
    }

    #[test]
    fn an_address_is_named_by_its_image_and_offset() {
        let modules = vec![module("/bin/program", 0x1000, 0x1000)];
        assert_eq!(
            render(&modules, 0x1234),
            "0x1234: ??? (/bin/program+0x1234)"
        );
    }

    #[test]
    fn a_lookup_is_at_the_call_rather_than_where_it_returns() {
        assert_eq!(call_site(0x1c3ca0), Some(0x1c3c9f));
        assert_eq!(call_site(0), None, "zero is no return address");
    }

    /// An image is chosen by the call a frame made, not by where it returns
    /// to. A return address one past the end of an image's code is that image's
    /// last instruction calling out, and one at its first byte is a call made
    /// from whatever precedes it.
    #[test]
    fn a_frame_belongs_to_the_image_that_made_the_call() {
        let modules = vec![module("/bin/program", 0x1000, 0x1000)];
        assert_eq!(
            render(&modules, 0x2000),
            "0x2000: ??? (/bin/program+0x2000)"
        );
        assert_eq!(render(&modules, 0x1000), "0x1000: ???");
        // The file address is the recorded one's, not the call's.
        assert_eq!(
            render(&modules, 0x1001),
            "0x1001: ??? (/bin/program+0x1001)"
        );
    }

    #[test]
    fn an_address_in_no_image_is_left_bare_rather_than_guessed_at() {
        let modules = vec![module("/bin/program", 0x1000, 0x1000)];
        assert_eq!(render(&modules, 0x9999), "0x9999: ???");
        assert_eq!(render(&[], 0x9999), "0x9999: ???");
    }

    #[test]
    fn the_right_image_is_chosen_when_several_are_loaded() {
        let modules = vec![
            module("/lib/first.so", 0x1000, 0x100),
            module("/lib/second.so", 0x2000, 0x100),
        ];
        assert!(render(&modules, 0x2010).contains("second.so+0x2010"));
        assert!(render(&modules, 0x1010).contains("first.so+0x1010"));
    }

    /// A path is whatever the filesystem allows, and it lands in a JSON string.
    #[test]
    fn an_awkward_path_survives_rendering() {
        let modules = vec![module("/tmp/a b\"c\\d", 0x1000, 0x100)];
        assert_eq!(
            render(&modules, 0x1004),
            "0x1004: ??? (/tmp/a b\"c\\d+0x1004)"
        );
    }

    // ---- Symbolized ----
    //
    // Against a symbol table the test supplies. The real one holds whatever
    // this build happened to export, which is a different set on every platform
    // and no set at all on a stripped one; `dynamic.rs` tests the platform call
    // itself.

    /// A symbol table of two functions, `core::fmt::write` at `0x1000` and a
    /// C function at `0x1020`, and nothing from `0x1028` on, so that one
    /// renderer covers the found and not-found paths in the same profile.
    fn fake_lookup(address: usize) -> Option<Symbol> {
        let (name, start) = match address {
            0x1000..0x1020 => ("_ZN4core3fmt5write17hb1f9a4a7f2f1a0c9E", 0x1000),
            0x1020..0x1028 => ("a_c_function_no_demangler_will_touch", 0x1020),
            _ => return None,
        };
        Some(Symbol {
            name: String::from(name),
            offset: address - start,
        })
    }

    fn symbolize(modules: &[Module], address: usize) -> String {
        let mut out = String::new();
        Symbolized::with_lookup(modules, fake_lookup).format(address, &mut out);
        out
    }

    #[test]
    fn a_named_address_is_rendered_with_the_demangled_name() {
        let modules = vec![module("/bin/program", 0x1000, 0x1000)];
        assert_eq!(
            symbolize(&modules, 0x1004),
            "0x1004: core::fmt::write+0x4 (/bin/program+0x1004)"
        );
    }

    /// Named by the call, so a call that is the last instruction of one
    /// function is not credited to the next one, which is where it returns to.
    /// The offset is still the recorded address's.
    #[test]
    fn a_frame_is_named_by_the_function_that_made_the_call() {
        let modules = vec![module("/bin/program", 0x1000, 0x1000)];
        assert_eq!(
            symbolize(&modules, 0x1020),
            "0x1020: core::fmt::write+0x20 (/bin/program+0x1020)"
        );
    }

    /// The offset is the reader's only defence against `dladdr` matching a
    /// symbol that is nowhere near the address. See `dynamic.rs`.
    #[test]
    fn the_distance_from_the_symbol_is_shown_when_there_is_any() {
        let modules = vec![module("/bin/program", 0x1000, 0x1000)];
        assert_eq!(
            symbolize(&modules, 0x1010),
            "0x1010: core::fmt::write+0x10 (/bin/program+0x1010)"
        );
    }

    /// A name the demangler refuses is still a name. Printing nothing because
    /// the symbol is not Rust would hide every C and C++ frame in the process.
    #[test]
    fn a_name_no_demangler_understands_is_printed_as_the_linker_wrote_it() {
        let modules = vec![module("/bin/program", 0x1000, 0x1000)];
        assert_eq!(
            symbolize(&modules, 0x1024),
            "0x1024: a_c_function_no_demangler_will_touch+0x4 (/bin/program+0x1024)"
        );
    }

    /// The claim on [`Symbolized`]: choosing it never costs anything, because
    /// where it finds no name it says exactly what [`ModuleOffsets`] says.
    ///
    /// This is the property that makes it safe as the default. If it were to
    /// drop the image and offset when a name was found, a profile from a
    /// machine with symbols would stop being resolvable on one without.
    #[test]
    fn an_unnamed_address_renders_exactly_as_module_offsets_would() {
        let modules = vec![module("/bin/program", 0x1000, 0x1000)];
        for address in [0x1030, 0x1500, 0x1FFF, 0x9999, 0] {
            assert_eq!(
                symbolize(&modules, address),
                render(&modules, address),
                "the two renderers disagreed about {address:#x}"
            );
        }
    }

    /// Whatever the name, the part a later tool resolves against must survive.
    #[test]
    fn the_image_and_file_offset_are_kept_whether_or_not_a_name_was_found() {
        let modules = vec![module("/bin/program", 0x1000, 0x1000)];
        for address in [0x1004, 0x1010, 0x1024, 0x1030] {
            let symbolized = symbolize(&modules, address);
            let bare = render(&modules, address);
            let (runtime_address, image) = bare
                .split_once(": ???")
                .expect("ModuleOffsets renders the address, `: ???`, then the image");
            assert!(
                symbolized.starts_with(runtime_address),
                "`{symbolized}` lost the runtime address `{runtime_address}`"
            );
            assert!(
                symbolized.ends_with(image),
                "`{symbolized}` lost the image attribution `{image}`"
            );
        }
    }

    /// A lookup that names anything it is asked about, which is what `dladdr`
    /// turns out to be for one address.
    fn credulous_lookup(address: usize) -> Option<Symbol> {
        Some(Symbol {
            name: format!("a_name_for_{address:#x}"),
            offset: 0x20,
        })
    }

    /// The module map decides what may be named, and it is asked first.
    ///
    /// This is the check the whole tier-1 design rests on — `dladdr` reports
    /// *success* for `(void *)-1`, naming a real symbol in a real image at an
    /// offset of 18 quintillion, and that value is exactly what a truncated
    /// stack walk produces. See `dynamic.rs`.
    ///
    /// It needs a lookup that succeeds where the map refuses, which no other
    /// test here has: `fake_lookup` only names addresses that are inside the
    /// module the tests supply, so with it the gate cannot be observed at all.
    /// Deleting the gate left the entire suite green until this existed.
    #[test]
    fn an_address_outside_every_image_is_not_named_however_willing_the_platform_is() {
        let modules = vec![module("/bin/program", 0x1000, 0x1000)];
        let format = Symbolized::with_lookup(&modules, credulous_lookup);

        for address in [0, 1, 0x999, 0x1000, 0x2001, usize::MAX] {
            let mut out = String::new();
            format.format(address, &mut out);
            assert!(
                !out.contains("a_name_for"),
                "{address:#x} is in no image in the map, and was named anyway: `{out}`"
            );
            // And it renders as the bare address, which is what `ModuleOffsets`
            // says for the same input.
            assert_eq!(out, render(&modules, address));
        }

        // The same renderer still names what the map does vouch for, so this
        // passes by asking the map rather than by never naming anything.
        let mut inside = String::new();
        format.format(0x1500, &mut inside);
        assert_eq!(
            inside,
            "0x1500: a_name_for_0x14ff+0x21 (/bin/program+0x1500)"
        );
    }

    /// The same gate, on the structured path the native format takes.
    ///
    /// `resolve` is a second implementation of the rule above, and it had no
    /// test of its own: removing its module check left the whole suite green,
    /// because the integration test that looks for it uses a real `dladdr`,
    /// which refuses a nonsense address on Linux whether or not the gate is
    /// there. So this supplies a lookup that names everything, which is what
    /// macOS measurably does for `(void *)-1`.
    #[test]
    fn resolving_an_address_outside_every_image_asks_the_platform_nothing() {
        let modules = vec![module("/bin/program", 0x1000, 0x1000)];

        for address in [0, 1, 0x999, 0x1000, 0x2001, usize::MAX] {
            let resolved = resolve_with(&modules, address, credulous_lookup);
            assert_eq!(
                resolved,
                Resolved::default(),
                "{address:#x} is in no image in the map and was resolved anyway"
            );
        }

        // And it still answers for an address the map does vouch for, so this
        // passes by asking the map rather than by never resolving anything.
        let inside = resolve_with(&modules, 0x1500, credulous_lookup);
        assert_eq!(inside.module, Some(0));
        assert_eq!(inside.file_address, Some(0x1500));
        assert_eq!(
            inside.symbol,
            Some(Symbol {
                name: String::from("a_name_for_0x14ff"),
                offset: 0x21,
            }),
            "named at the call, with the offset of the recorded address"
        );
    }

    /// The file address is the number `addr2line` takes: the runtime address
    /// minus the image's bias. Not an offset from the load address, which is a
    /// different number on Mach-O and on a non-PIE ELF executable — and the
    /// two are equal exactly when the bias is zero, which is why every other
    /// module in these tests has one.
    #[test]
    fn a_resolved_file_address_is_the_address_the_file_has() {
        let modules = vec![Module {
            path: String::from("/bin/program"),
            start: 0x1_0000_5000,
            size: 0x1000,
            bias: 0x5000,
            image_base: 0x1_0000_5000,
            build_id: None,
        }];
        let resolved = resolve(&modules, 0x1_0000_5100);
        assert_eq!(resolved.module, Some(0));
        assert_eq!(resolved.file_address, Some(0x1_0000_0100));
    }

    /// A profile resolves the same address once per program point that contains
    /// it, and the outermost frames are shared by every stack in the process.
    /// On Windows each of those is a lock and a dbghelp call.
    #[test]
    fn a_repeated_address_is_only_looked_up_once() {
        use std::cell::Cell;

        thread_local! {
            static CALLS: Cell<usize> = const { Cell::new(0) };
        }

        fn counting_lookup(address: usize) -> Option<Symbol> {
            CALLS.with(|calls| calls.set(calls.get() + 1));
            fake_lookup(address)
        }

        let modules = vec![module("/bin/program", 0x1000, 0x1000)];
        let format = Symbolized::with_lookup(&modules, counting_lookup);

        let mut first = String::new();
        format.format(0x1004, &mut first);
        for _ in 0..32 {
            let mut again = String::new();
            format.format(0x1004, &mut again);
            assert_eq!(again, first);
        }
        // The address that resolves to nothing is worth caching too: it is the
        // common one on a stripped binary, and it costs the same to find out.
        for _ in 0..32 {
            let mut nothing = String::new();
            format.format(0x1030, &mut nothing);
        }

        assert_eq!(
            CALLS.with(Cell::get),
            2,
            "two distinct addresses should mean two lookups"
        );
    }

    /// `format` appends. A renderer that cleared its output would silently drop
    /// whatever the caller had already written.
    #[test]
    fn rendering_appends_rather_than_replacing() {
        let modules = vec![module("/bin/program", 0x1000, 0x1000)];
        let format = Symbolized::with_lookup(&modules, fake_lookup);
        let mut out = String::from("before ");
        format.format(0x1004, &mut out);
        // Twice, because the second call takes the cached path, which is a
        // different line of code and just as able to get this wrong.
        format.format(0x1004, &mut out);
        assert_eq!(
            out,
            "before 0x1004: core::fmt::write+0x4 (/bin/program+0x1004)\
             0x1004: core::fmt::write+0x4 (/bin/program+0x1004)"
        );
    }

    // ---- FunctionNames ----
    //
    // Against the same supplied symbol tables as `Symbolized`, for the same
    // reason, and in places against `Symbolized` itself: the claim is that a
    // name here is the name there.

    fn name(modules: &[Module], address: usize) -> String {
        let mut out = String::new();
        FunctionNames::with_lookup(modules, fake_lookup).format(address, &mut out);
        out
    }

    /// A module whose file addresses start at zero, so that two of them can
    /// put different code at the same file address, which is what two
    /// different builds of one library do.
    fn image_at(path: &str, start: usize) -> Module {
        Module {
            bias: start,
            ..module(path, start, 0x1000)
        }
    }

    #[test]
    fn a_named_address_is_rendered_as_the_function_name_alone() {
        let modules = vec![module("/bin/program", 0x1000, 0x1000)];
        assert_eq!(name(&modules, 0x1004), "core::fmt::write");
    }

    /// The point of the type. Two return addresses in one function were two
    /// frames under `Symbolized`, by their addresses and their offsets, and
    /// are one here.
    #[test]
    fn two_return_addresses_in_one_function_render_alike() {
        let modules = vec![module("/bin/program", 0x1000, 0x1000)];
        assert_eq!(name(&modules, 0x1004), name(&modules, 0x1010));
        assert_ne!(
            symbolize(&modules, 0x1004),
            symbolize(&modules, 0x1010),
            "the fixture no longer has two addresses `Symbolized` keeps apart"
        );
    }

    /// Named by the call, as every lookup is: a call that is the last
    /// instruction of `core::fmt::write` returns to the first byte of the next
    /// function, and is still `core::fmt::write`. An image is chosen the same
    /// way, so a return address at an image's first byte is not that image's.
    #[test]
    fn a_frame_is_named_by_the_function_that_made_the_call_here_too() {
        let modules = vec![module("/bin/program", 0x1000, 0x1000)];
        assert_eq!(name(&modules, 0x1020), "core::fmt::write");
        assert_eq!(name(&modules, 0x1000), "[0x1000]");
        // The fallback's file address is the recorded one's, not the call's.
        assert_eq!(name(&modules, 0x2000), "[program+0x2000]");
    }

    /// The name is found, demangled, and refused exactly as `Symbolized` does
    /// it, so a frame in a flame graph can be searched for in the text summary
    /// or the DHAT file of the same run and found.
    #[test]
    fn a_name_is_the_name_symbolized_shows_for_the_same_address() {
        let modules = vec![module("/bin/program", 0x1000, 0x1000)];
        for address in [0x1004, 0x1010, 0x1020, 0x1024] {
            let alone = name(&modules, address);
            let symbolized = symbolize(&modules, address);
            // `0xADDR: NAME[+0xOFFSET] (IMAGE+0xFILEADDR)`, taken apart from
            // both ends so that the comparison is with the whole name and
            // nothing else. A prefix check would pass on a truncated name.
            let after_address = symbolized
                .split_once(": ")
                .expect("Symbolized renders the address, then `: `")
                .1;
            let named = after_address
                .rsplit_once(" (")
                .expect("and the image after the name")
                .0;
            let symbolized_name = named
                .rsplit_once("+0x")
                .map_or(named, |(name, _offset)| name);
            assert_eq!(alone, symbolized_name, "in `{symbolized}`");
        }
    }

    /// The legacy hash names a build, not a function. It is the demangler that
    /// drops it, and this pins that the flame graph gets the demangler's answer.
    #[test]
    fn the_legacy_hash_is_not_part_of_the_name() {
        let modules = vec![module("/bin/program", 0x1000, 0x1000)];
        let rendered = name(&modules, 0x1004);
        assert!(!rendered.contains("::h"), "{rendered}");
        assert!(!rendered.contains("hb1f9a4a7"), "{rendered}");
    }

    #[test]
    fn a_name_no_demangler_understands_is_the_name_the_linker_wrote() {
        let modules = vec![module("/bin/program", 0x1000, 0x1000)];
        assert_eq!(
            name(&modules, 0x1024),
            "a_c_function_no_demangler_will_touch"
        );
    }

    /// A frame with no name keeps the half of the `ModuleOffsets` rendering
    /// that tells two such frames apart: the image and the file address. The
    /// path is cut to its file name, which is what made every label unreadable.
    #[test]
    fn an_unnamed_frame_is_its_image_file_name_and_file_address() {
        let modules = vec![module("/very/long/path/to/program", 0x1000, 0x1000)];
        assert_eq!(name(&modules, 0x1030), "[program+0x1030]");
        assert_eq!(name(&modules, 0x1040), "[program+0x1040]");
    }

    /// Two images with one file name, in two directories, each with code at
    /// file address 0x100. Shortened to the file name, those two frames would
    /// be one frame in the flame graph, made of two unrelated functions.
    #[test]
    fn an_image_whose_file_name_another_image_shares_keeps_its_whole_path() {
        let modules = vec![
            image_at("/opt/one/libsame.so", 0x1000),
            image_at("/opt/two/libsame.so", 0x3000),
            image_at("/usr/lib/libother.so", 0x5000),
        ];
        let first = name(&modules, 0x1100);
        let second = name(&modules, 0x3100);
        assert_ne!(first, second, "two images' frames merged");
        assert_eq!(first, "[/opt/one/libsame.so+0x100]");
        assert_eq!(second, "[/opt/two/libsame.so+0x100]");
        // And the ambiguity is per file name, not a reason to stop shortening
        // the images that have none.
        assert_eq!(name(&modules, 0x5100), "[libother.so+0x100]");
    }

    #[test]
    fn an_address_in_no_image_is_its_runtime_address() {
        let modules = vec![module("/bin/program", 0x1000, 0x1000)];
        assert_eq!(name(&modules, 0x9999), "[0x9999]");
        assert_eq!(name(&[], 0x1000), "[0x1000]");
    }

    /// An image with no path has nothing to be labelled by, and an empty label
    /// is the same label for every such image. The runtime address is distinct
    /// by definition.
    #[test]
    fn an_image_with_no_path_is_labelled_by_the_address_instead() {
        let modules = vec![image_at("", 0x1000), image_at("", 0x3000)];
        assert_eq!(name(&modules, 0x1100), "[0x1100]");
        assert_eq!(name(&modules, 0x3100), "[0x3100]");
    }

    /// The same gate `Symbolized` has, held here separately because it is a
    /// separate line of code: an address in no image is never named.
    #[test]
    fn an_address_outside_every_image_is_not_named_here_either() {
        let modules = vec![module("/bin/program", 0x1000, 0x1000)];
        let format = FunctionNames::with_lookup(&modules, credulous_lookup);
        for address in [0, 1, 0x999, 0x1000, 0x2001, usize::MAX] {
            let mut out = String::new();
            format.format(address, &mut out);
            assert!(
                !out.contains("a_name_for"),
                "{address:#x} is in no image in the map, and was named anyway: `{out}`"
            );
        }
        let mut inside = String::new();
        format.format(0x1500, &mut inside);
        assert_eq!(inside, "a_name_for_0x14ff");
    }

    /// A name that comes back empty would be an empty frame, which a folded
    /// line shows as a nameless level. `dynamic::lookup` refuses one; a lookup
    /// that did not would still not produce it.
    #[test]
    fn an_empty_name_is_no_name() {
        fn empty_lookup(_: usize) -> Option<Symbol> {
            Some(Symbol {
                name: String::new(),
                offset: 0,
            })
        }
        let modules = vec![module("/bin/program", 0x1000, 0x1000)];
        let mut out = String::new();
        FunctionNames::with_lookup(&modules, empty_lookup).format(0x1004, &mut out);
        assert_eq!(out, "[program+0x1004]");
    }

    /// `Trimmed` reads names where the renderer says they are. Without that,
    /// it would look for the `0xADDR: ` this rendering does not have, find no
    /// names, and trim nothing, silently.
    #[test]
    fn a_stand_in_is_not_a_name_and_a_name_is_all_of_it() {
        let format = FunctionNames::with_lookup(&[], fake_lookup);
        assert_eq!(format.name_of("core::fmt::write"), Some("core::fmt::write"));
        // A const generic argument renders with `": "` in it, which is why
        // the default reading cannot be used here.
        assert_eq!(
            format.name_of("program::f::<{program::S { a: 1 }}>"),
            Some("program::f::<{program::S { a: 1 }}>")
        );
        assert_eq!(format.name_of("[program+0x1030]"), None);
        assert_eq!(format.name_of("[0x9999]"), None);
    }

    /// Names in the shape of a real stack: allocation path inside, runtime
    /// entry outside, and the program between.
    fn stack_lookup(address: usize) -> Option<Symbol> {
        let name = match address {
            0x1000..0x1100 => "__rust_alloc",
            0x1100..0x1200 => "_ZN7program5churn17h0123456789abcdefE",
            0x1200..0x1300 => "_ZN7program4main17h0123456789abcdefE",
            0x1300..0x1400 => "std::sys::backtrace::__rust_begin_short_backtrace",
            0x1400..0x1500 => "main",
            _ => return None,
        };
        Some(Symbol {
            name: String::from(name),
            offset: 8,
        })
    }

    fn kept<F: FrameFormat>(format: &F, stack: &[usize]) -> Vec<String> {
        let frames: Vec<String> = stack
            .iter()
            .map(|&address| {
                let mut out = String::new();
                format.format(address, &mut out);
                out
            })
            .collect();
        frames[format.keep(&frames)].to_vec()
    }

    /// The same rules cut the same frames whichever of the two renderers a
    /// stack is in, which is what makes the default folded output trimmed at
    /// all.
    #[test]
    fn trimming_cuts_the_same_frames_from_bare_names() {
        let modules = vec![module("/bin/program", 0x1000, 0x1000)];
        let stack = [0x1008, 0x1108, 0x1208, 0x1308, 0x1408];

        let names = kept(
            &Trimmed::new(FunctionNames::with_lookup(&modules, stack_lookup)),
            &stack,
        );
        assert_eq!(names, ["program::churn", "program::main"]);

        let symbolized = kept(
            &Trimmed::new(Symbolized::with_lookup(&modules, stack_lookup)),
            &stack,
        );
        assert_eq!(symbolized.len(), names.len(), "{symbolized:?}");
        for (bare, full) in names.iter().zip(&symbolized) {
            assert!(full.contains(bare.as_str()), "{full} is not {bare}");
        }
    }

    #[test]
    fn a_repeated_address_is_only_looked_up_once_for_names_too() {
        use std::cell::Cell;

        thread_local! {
            static CALLS: Cell<usize> = const { Cell::new(0) };
        }

        fn counting_lookup(address: usize) -> Option<Symbol> {
            CALLS.with(|calls| calls.set(calls.get() + 1));
            fake_lookup(address)
        }

        let modules = vec![module("/bin/program", 0x1000, 0x1000)];
        let format = FunctionNames::with_lookup(&modules, counting_lookup);
        for _ in 0..32 {
            for address in [0x1004, 0x1030] {
                format.format(address, &mut String::new());
            }
        }
        assert_eq!(CALLS.with(Cell::get), 2);
    }

    #[test]
    fn rendering_names_appends_rather_than_replacing() {
        let modules = vec![module("/bin/program", 0x1000, 0x1000)];
        let format = FunctionNames::with_lookup(&modules, fake_lookup);
        let mut out = String::from("before ");
        format.format(0x1004, &mut out);
        format.format(0x1004, &mut out);
        assert_eq!(out, "before core::fmt::writecore::fmt::write");
    }
}
