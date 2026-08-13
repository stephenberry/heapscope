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
//! Once a frame has a name, [`trim`] can tell the frames that are about the
//! program from the ones every stack has — the allocation path above and the
//! runtime entry below — and leave the second kind out. That is downstream of
//! naming by construction, and inert wherever naming finds nothing.

pub mod demangle;
#[cfg(all(unix, not(miri)))]
mod dl;
pub mod dynamic;
pub mod modules;
pub mod trim;

use std::cell::RefCell;
use std::collections::HashMap;

use crate::output::FrameFormat;
use dynamic::Symbol;
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
pub fn resolve(modules: &[Module], address: usize) -> Resolved {
    resolve_with(modules, address, dynamic::lookup)
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
    let Some(module) = modules::index_containing(modules, address) else {
        return Resolved::default();
    };
    Resolved {
        module: Some(module),
        file_address: modules[module].file_address(address),
        symbol: lookup(address),
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
        push_image(modules::containing(self.modules, address), address, out);
    }
}

/// Appends ` (path+0xfileaddress)` for `module`, or nothing.
///
/// Shared by both renderers, because the part of a frame that says *which file
/// to resolve against* is the part that has to be there whether or not a name
/// was found — it is what makes the frame answerable later, by a different tool,
/// on a different machine.
fn push_image(module: Option<&Module>, address: usize, out: &mut String) {
    let Some(module) = module else {
        return;
    };
    let Some(file_address) = module.file_address(address) else {
        return;
    };
    out.push_str(" (");
    out.push_str(&module.path);
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
    cache: RefCell<HashMap<usize, Box<str>>>,
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
            cache: RefCell::new(HashMap::new()),
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
        let module = modules::containing(self.modules, address);
        let symbol = module.and_then(|_| (self.lookup)(address));

        match symbol {
            Some(symbol) => {
                // Demangling refuses on anything it does not fully understand,
                // which includes every C and C++ name in the process as well as
                // a Rust name read out of a damaged table. The raw symbol is
                // then the best available answer: ugly, but what the linker
                // actually wrote. Neither branch is screened here — the emitter
                // screens the finished frame, which is the only place that also
                // covers a `FrameFormat` this crate did not write.
                //
                // The `truncate` is belt and braces: `demangle` documents and
                // tests that it leaves `out` untouched when it refuses. It is
                // one instruction, and the failure it guards against is a
                // half-parsed name attributing an allocation to code that did
                // not make it, which is the one output error this crate has no
                // way to make visible to a reader.
                let before = out.len();
                if !demangle(&symbol.name, &mut out) {
                    out.truncate(before);
                    out.push_str(&symbol.name);
                }
                if symbol.offset != 0 {
                    out.push('+');
                    crate::output::push_hex(&mut out, symbol.offset);
                }
            }
            // The same three characters `ModuleOffsets` uses, so that a frame
            // with no name looks the same however the profile was rendered.
            None => out.push_str("???"),
        }

        push_image(module, address, &mut out);
        out.into_boxed_str()
    }
}

impl FrameFormat for Symbolized<'_> {
    fn format(&self, address: usize, out: &mut String) {
        if let Some(cached) = self.cache.borrow().get(&address) {
            out.push_str(cached);
            return;
        }
        // Deliberately outside the borrow above: `render` calls into the
        // platform, and holding a `RefCell` borrow across a foreign call is the
        // kind of thing that is fine until someone adds a lookup that renders.
        let rendered = self.render(address);
        out.push_str(&rendered);
        self.cache.borrow_mut().insert(address, rendered);
    }
}

impl std::fmt::Debug for Symbolized<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Symbolized")
            .field("modules", &self.modules.len())
            .field("cached", &self.cache.borrow().len())
            .finish()
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
    fn the_first_byte_of_an_image_has_a_zero_offset() {
        let modules = vec![module("/bin/program", 0x1000, 0x1000)];
        assert_eq!(
            render(&modules, 0x1000),
            "0x1000: ??? (/bin/program+0x1000)"
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

    /// Names two addresses and refuses everything else, so that one renderer
    /// covers the found and not-found paths in the same profile.
    fn fake_lookup(address: usize) -> Option<Symbol> {
        match address {
            0x1000 => Some(Symbol {
                name: String::from("_ZN4core3fmt5write17hb1f9a4a7f2f1a0c9E"),
                offset: 0,
            }),
            0x1010 => Some(Symbol {
                name: String::from("_ZN4core3fmt5write17hb1f9a4a7f2f1a0c9E"),
                offset: 0x10,
            }),
            0x1020 => Some(Symbol {
                name: String::from("a_c_function_no_demangler_will_touch"),
                offset: 4,
            }),
            _ => None,
        }
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
            symbolize(&modules, 0x1000),
            "0x1000: core::fmt::write (/bin/program+0x1000)"
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
            symbolize(&modules, 0x1020),
            "0x1020: a_c_function_no_demangler_will_touch+0x4 (/bin/program+0x1020)"
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
        for address in [0x1000, 0x1010, 0x1020, 0x1030] {
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

        for address in [0, 1, 0x999, 0x2000, usize::MAX] {
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
            "0x1500: a_name_for_0x1500+0x20 (/bin/program+0x1500)"
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

        for address in [0, 1, 0x999, 0x2000, usize::MAX] {
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
            inside.symbol.map(|symbol| symbol.name),
            Some(String::from("a_name_for_0x1500"))
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
        format.format(0x1000, &mut first);
        for _ in 0..32 {
            let mut again = String::new();
            format.format(0x1000, &mut again);
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
        format.format(0x1000, &mut out);
        // Twice, because the second call takes the cached path, which is a
        // different line of code and just as able to get this wrong.
        format.format(0x1000, &mut out);
        assert_eq!(
            out,
            "before 0x1000: core::fmt::write (/bin/program+0x1000)\
             0x1000: core::fmt::write (/bin/program+0x1000)"
        );
    }
}
