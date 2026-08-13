//! Where each image is loaded, and how to identify it later.
//!
//! A profile records return addresses. An address on its own is meaningless the
//! moment the process exits: address space layout randomization means the same
//! function is at a different address on the next run, and on a stripped binary
//! `dladdr` returns success with a null symbol name, so in-process symbolization
//! cannot rescue it either.
//!
//! The module map is what makes an address mean something afterwards. For each
//! mapped image it records the path, where its code is, the bias that converts a
//! runtime address to an address in the file, and the build identity the linker
//! stamped in.
//!
//! # Three numbers, because three tools want three different things
//!
//! | field | is | consumed by |
//! |---|---|---|
//! | [`Module::bias`] | subtract from a runtime address to get the address *in the file* | `llvm-symbolizer`, `addr2line` |
//! | [`Module::image_base`] | where the image itself begins | `atos -l` |
//! | [`Module::start`], [`Module::size`] | the image's **executable** region | deciding which image a return address is in |
//!
//! On Windows the third row is the whole image rather than its executable
//! sections: `K32EnumProcessModules` reports a base and a size, and this does
//! not walk the PE section table. Attribution is unaffected, because Windows
//! images do not overlap — but an address in a PE's `.data` or headers is
//! *inside* a module by this map's reckoning, which matters to anything using
//! containment as evidence that an address is code.
//!
//! Collapsing these into one number is a mistake that has been made here twice.
//! They coincide on Mach-O and on an ELF position-independent executable, and
//! they diverge on a non-PIE — where the bias is 0 while the image begins at
//! 0x400000 — and on Mach-O, where the bias is the slide while file addresses
//! start at 0x1_0000_0000.
//!
//! # Why the recorded region is the executable one
//!
//! The map exists to attribute *return addresses*, which are always in code. An
//! image's full virtual span is the wrong thing to record: on Linux the kernel
//! places `[vdso]` inside the gap between `ld.so`'s two `PT_LOAD` segments, so
//! spans overlap and an address resolves to whichever image is asked first.
//! Recording only the executable region makes the map non-overlapping, which is
//! what lets a lookup bisect.
//!
//! # Build identity
//!
//! | platform | source | what it is |
//! |---|---|---|
//! | Apple | `LC_UUID` load command | 16-byte image UUID |
//! | Linux | `PT_NOTE` / `NT_GNU_BUILD_ID` | usually a 20-byte SHA-1 |
//! | Windows | not yet captured | the PDB GUID needs the PE debug directory |
//!
//! It is what tells you whether the binary you are symbolizing against is the
//! one that produced the profile. Without it, a stale build gives you confident,
//! wrong function names.
//!
//! # When this runs
//!
//! Output time only, never from the allocation path. The enumerations are not
//! synchronized against a concurrent `dlopen`, so a library loaded during the
//! capture may be missed; the addresses already recorded stay correct, and a
//! frame in a missing image renders unresolved rather than wrong.

/// One image mapped into this process.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Module {
    /// The path the image was loaded from.
    pub path: String,
    /// The lowest runtime address of the image's executable region.
    pub start: usize,
    /// How far that region extends.
    pub size: usize,
    /// Subtract from a runtime address to get the address as it appears in the
    /// file, which is what `llvm-symbolizer` and `addr2line` resolve.
    ///
    /// On Windows this is the image base, so the difference is a relative
    /// virtual address and `llvm-symbolizer` needs `--relative-address`.
    ///
    /// **Not resolvable for an image in the dyld shared cache**, which on macOS
    /// is almost everything under `/usr/lib`. A cache image's segments are laid
    /// out at cache addresses rather than at the addresses in the file at its
    /// path, so subtracting the slide gives an address in the cache. Measured:
    /// `/usr/lib/dyld`'s `start` is at file address `0x1e994`, its in-memory
    /// `__TEXT` reports `vmaddr 0x180114000` against `0x0` on disk, and this
    /// field yields `0x1801344e4` for a frame the file places at `0x204e4` —
    /// with the UUIDs matching, so nothing warns the reader.
    ///
    /// [`image_base`](Self::image_base) is unaffected, which is why `atos` — the
    /// tool that takes it — resolves those frames correctly and the ELF tools do
    /// not. Fixing this needs the link-time address from the file on disk, which
    /// means reading a Mach-O rather than asking the loader: tier 3, M8.
    pub bias: usize,
    /// Where the image begins, which is what `atos -l` wants.
    pub image_base: usize,
    /// The linker's build identity, lower-case hexadecimal.
    ///
    /// `None` where the platform or the build did not provide one; a stripped
    /// binary built without `--build-id` genuinely has nothing to report.
    pub build_id: Option<String>,
}

impl Module {
    /// Whether `address` falls in this image's executable region.
    pub fn contains(&self, address: usize) -> bool {
        address >= self.start && address - self.start < self.size
    }

    /// `address` as it appears in the file on disk.
    pub fn file_address(&self, address: usize) -> Option<usize> {
        self.contains(address)
            .then(|| address.wrapping_sub(self.bias))
    }
}

/// Every image mapped into this process, ordered by the start of its code.
///
/// Ordered so that a lookup can bisect: a profile with a million program points
/// resolves several million frames.
pub fn capture() -> Vec<Module> {
    let mut modules = imp::capture();
    // An image with no executable bytes cannot hold a return address, and one
    // with a zero-length region would make `contains` a permanent `false`
    // anyway. Dropping them keeps the map to what it is for.
    modules.retain(|module| module.size > 0);
    modules.sort_by_key(|module| module.start);
    modules
}

/// The image containing `address`, if any.
pub fn containing(modules: &[Module], address: usize) -> Option<&Module> {
    index_containing(modules, address).map(|at| &modules[at])
}

/// Where in `modules` the image containing `address` is, if any.
///
/// The index rather than the image, for a caller that has to *refer* to it —
/// the native format writes one module map and has every frame point into it,
/// so that a profile of a program with a thousand images does not repeat a
/// thousand paths per stack.
pub fn index_containing(modules: &[Module], address: usize) -> Option<usize> {
    // `partition_point` finds the first module starting after `address`; the one
    // before it is the only candidate, because executable regions do not
    // overlap. See the module documentation for why the region is the
    // executable one rather than the image's whole span.
    let at = modules.partition_point(|module| module.start <= address);
    (at > 0 && modules[at - 1].contains(address)).then(|| at - 1)
}

/// Formats `bytes` as lower-case hexadecimal.
///
/// Only the Unix backends have a build identity to format; the PE debug
/// directory that Windows would need is not read yet.
#[cfg(unix)]
fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        text.push(DIGITS[usize::from(byte >> 4)] as char);
        text.push(DIGITS[usize::from(byte & 0xF)] as char);
    }
    text
}

#[cfg(all(target_vendor = "apple", not(miri)))]
mod imp {
    use super::{hex, Module};
    use crate::symbol::dl::{self, DlInfo};
    use std::ffi::c_void;

    // The dyld image APIs, in libSystem, which every process links. `dladdr`
    // lives there too, and is reached through `symbol::dl` — the same binding
    // the symbolizer uses.
    extern "C" {
        fn _dyld_image_count() -> u32;
        fn _dyld_get_image_name(index: u32) -> *const std::ffi::c_char;
        fn _dyld_get_image_header(index: u32) -> *const MachHeader64;
        fn _dyld_get_image_vmaddr_slide(index: u32) -> isize;
    }

    const MH_MAGIC_64: u32 = 0xfeed_facf;
    const LC_SEGMENT_64: u32 = 0x19;
    const LC_UUID: u32 = 0x1b;
    /// `VM_PROT_EXECUTE`.
    const EXECUTABLE: i32 = 0x4;

    #[repr(C)]
    struct MachHeader64 {
        magic: u32,
        cpu_type: i32,
        cpu_subtype: i32,
        file_type: u32,
        commands: u32,
        commands_size: u32,
        flags: u32,
        reserved: u32,
    }

    #[repr(C)]
    struct LoadCommand {
        kind: u32,
        size: u32,
    }

    #[repr(C)]
    struct UuidCommand {
        kind: u32,
        size: u32,
        uuid: [u8; 16],
    }

    #[repr(C)]
    struct SegmentCommand64 {
        kind: u32,
        size: u32,
        name: [u8; 16],
        vm_address: u64,
        vm_size: u64,
        file_offset: u64,
        file_size: u64,
        max_protection: i32,
        initial_protection: i32,
        sections: u32,
        flags: u32,
    }

    pub(super) fn capture() -> Vec<Module> {
        // SAFETY: no arguments, no failure mode. The count can change under a
        // concurrent `dlopen`, which is documented on `super::capture`.
        let count = unsafe { _dyld_image_count() };
        let mut modules = Vec::with_capacity(count as usize + 1);

        for index in 0..count {
            // SAFETY: `index` is below the count dyld just reported. The header
            // is null if the image was unloaded in between, checked in
            // `describe`.
            let (name, header, slide) = unsafe {
                (
                    _dyld_get_image_name(index),
                    _dyld_get_image_header(index),
                    _dyld_get_image_vmaddr_slide(index),
                )
            };
            // SAFETY: dyld returns a NUL-terminated path or null.
            let path = unsafe { dl::string_at(name) }.unwrap_or_default();
            if let Some(module) = describe(header, slide as usize, path) {
                modules.push(module);
            }
        }

        // dyld is not in its own list, and it holds the outermost frames of
        // every stack — `start`, the image initializers, lazy binding.
        if let Some(header) = loader_header() {
            let header = header as *const MachHeader64;
            if !modules.iter().any(|m| m.image_base == header as usize) {
                // The kernel gives the address; `dladdr` gives the path. An
                // image with no path is one nothing can be resolved against, so
                // it is dropped rather than entered under a guessed name.
                let path = DlInfo::of(header as *const c_void)
                    .map(|info| info.image_path())
                    .unwrap_or_default();
                if !path.is_empty() {
                    if let Some(module) = describe_with_derived_slide(header, path) {
                        modules.push(module);
                    }
                }
            }
        }

        modules
    }

    /// `TASK_DYLD_INFO`, the flavour of `task_info` that reports where the
    /// process's image records live.
    const TASK_DYLD_INFO: u32 = 17;

    /// `task_dyld_info_data_t`.
    #[repr(C)]
    struct TaskDyldInfo {
        all_image_info_addr: u64,
        all_image_info_size: u64,
        all_image_info_format: i32,
    }

    /// The leading fields of `dyld_all_image_infos`.
    ///
    /// Deliberately a prefix: the structure has grown across releases and only
    /// ever by appending, so a declaration of the first seven fields reads
    /// correctly against every version that has existed. `all_image_info_size`
    /// is checked against this before anything is read, so a hypothetical
    /// shorter one is refused rather than read past.
    #[repr(C)]
    struct AllImageInfos {
        version: u32,
        info_array_count: u32,
        info_array: *const c_void,
        notification: *const c_void,
        process_detached_from_shared_region: bool,
        lib_system_initialized: bool,
        loader_load_address: *const MachHeader64,
    }

    extern "C" {
        /// The Mach port for this task. A variable rather than a function:
        /// `mach_task_self()` is a macro over it.
        static mach_task_self_: u32;
        fn task_info(task: u32, flavour: u32, out: *mut u32, count: *mut u32) -> i32;
    }

    /// Where dyld itself is loaded, according to the kernel.
    ///
    /// `_dyld_image_count` enumerates what dyld *loaded*, which does not include
    /// dyld. The previous attempt at this asked `dladdr` where
    /// `_dyld_image_count` lives, on the reasoning that it is one of dyld's own
    /// functions. Measured, it is not: it resolves to
    /// `/usr/lib/system/libdyld.dylib`, an ordinary image that is already in the
    /// list, so the duplicate check discarded the result and the map stayed
    /// exactly as incomplete as before — silently, because nothing asserted the
    /// loader was in it.
    ///
    /// ```text
    /// stack frame 1     0x187a484e4  sname=start  fname=/usr/lib/dyld
    /// dyld image [8]    0x1879f3000               /usr/lib/system/libdyld.dylib
    /// _dyld_image_count lives in     0x1879f3000  /usr/lib/system/libdyld.dylib
    /// ```
    ///
    /// So this asks the kernel, which is where a debugger gets it. `None` if the
    /// call is refused — inside a sandbox that denies task introspection, for
    /// instance — in which case the outermost frame of each stack renders as a
    /// bare address, which is what it did before.
    pub(super) fn loader_header() -> Option<usize> {
        let mut info = TaskDyldInfo {
            all_image_info_addr: 0,
            all_image_info_size: 0,
            all_image_info_format: 0,
        };
        let mut count = (std::mem::size_of::<TaskDyldInfo>() / std::mem::size_of::<u32>()) as u32;
        // SAFETY: `info` is a live `task_dyld_info_data_t` and `count` says how
        // many words of it may be written. `mach_task_self_` is the port for
        // this process, which needs no right to be acquired or released.
        let result = unsafe {
            task_info(
                mach_task_self_,
                TASK_DYLD_INFO,
                (&mut info as *mut TaskDyldInfo).cast(),
                &mut count,
            )
        };
        // `KERN_SUCCESS`.
        if result != 0 || info.all_image_info_addr == 0 {
            return None;
        }
        if (info.all_image_info_size as usize) < std::mem::size_of::<AllImageInfos>() {
            return None;
        }

        let infos = info.all_image_info_addr as *const AllImageInfos;
        // SAFETY: the kernel reported this address as holding a structure of at
        // least `all_image_info_size` bytes, checked above to cover the prefix
        // declared here. It is in this process's own address space and stays
        // mapped for the life of the process.
        let (version, header) = unsafe { ((*infos).version, (*infos).loader_load_address) };
        // `loader_load_address` has been present since version 1.
        if version == 0 || header.is_null() {
            return None;
        }
        Some(header as usize)
    }

    /// Builds a module from a mach header known to be `slide` bytes from its
    /// link-time address.
    fn describe(header: *const MachHeader64, slide: usize, path: String) -> Option<Module> {
        let image = read_image(header)?;
        Some(Module {
            path,
            start: image.code_start.wrapping_add(slide),
            size: image.code_size,
            bias: slide,
            image_base: header as usize,
            build_id: image.uuid,
        })
    }

    /// The same, for an image whose slide dyld will not report — which is dyld
    /// itself. The slide is the difference between where the header actually is
    /// and the link-time address of the segment that contains it.
    ///
    /// That segment is `__TEXT`, and `__TEXT` is the lowest-addressed executable
    /// segment of every image Apple's linker produces, so `code_start` is its
    /// link-time address and no separate field is needed to find it.
    ///
    /// The rule this replaced was "the segment mapping file offset zero", which
    /// is the textbook one and is wrong for both images it could meet here.
    /// dyld is mapped from the shared cache, where file offsets are offsets into
    /// the cache, so its `__TEXT` reports `fileoff=0x8c000` and nothing matched
    /// at all — leaving the slide as the entire header address and dyld's
    /// recorded code region 6 GiB away from where its code is:
    ///
    /// ```text
    /// seg __TEXT  vmaddr=0x180114000 vmsize=0xb3500 fileoff=0x8c000 initprot=5
    /// header at 0x187a28000, so the slide is 0x7914000 — the same as every
    /// other image in the shared region, and not the 0x187a28000 that rule gave
    /// ```
    ///
    /// A main executable would have gone wrong the other way, because
    /// `__PAGEZERO` maps file offset zero at link-time address zero.
    fn describe_with_derived_slide(header: *const MachHeader64, path: String) -> Option<Module> {
        let image = read_image(header)?;
        if image.code_size == 0 {
            // No executable segment means no link-time address to measure the
            // slide against, and nothing worth recording either.
            return None;
        }
        let slide = (header as usize).wrapping_sub(image.code_start);
        Some(Module {
            path,
            start: image.code_start.wrapping_add(slide),
            size: image.code_size,
            bias: slide,
            image_base: header as usize,
            build_id: image.uuid,
        })
    }

    fn read_image(header: *const MachHeader64) -> Option<Image> {
        if header.is_null() {
            return None;
        }
        // SAFETY: dyld's headers stay mapped for the life of the image, and a
        // null pointer was just excluded.
        if unsafe { (*header).magic } != MH_MAGIC_64 {
            // A 32-bit image in a 64-bit process is not a thing that happens;
            // skipping is more honest than guessing at a layout.
            return None;
        }
        // SAFETY: the magic number confirms a mapped 64-bit Mach-O header.
        Some(unsafe { walk_load_commands(header) })
    }

    /// What the load commands say about an image, in link-time addresses.
    #[derive(Default)]
    struct Image {
        uuid: Option<String>,
        /// Lowest link-time address of an executable segment.
        code_start: usize,
        /// Extent of the executable segments.
        code_size: usize,
    }

    /// # Safety
    ///
    /// `header` must point to a mapped 64-bit Mach-O header whose load commands
    /// follow it contiguously, which is what dyld guarantees for a loaded image.
    unsafe fn walk_load_commands(header: *const MachHeader64) -> Image {
        // SAFETY: the caller guarantees a mapped header.
        let (commands, commands_size) = unsafe { ((*header).commands, (*header).commands_size) };

        let mut image = Image::default();
        let mut code_start = usize::MAX;
        let mut code_end = 0usize;
        let mut cursor = header.wrapping_add(1) as *const u8;
        let end = (header as *const u8)
            .wrapping_add(std::mem::size_of::<MachHeader64>() + commands_size as usize);

        for _ in 0..commands {
            if cursor.wrapping_add(std::mem::size_of::<LoadCommand>()) > end {
                break;
            }
            // SAFETY: the bounds check above keeps the read inside the load
            // command region dyld mapped with the header. The read is unaligned
            // so that nothing here depends on a producer having honoured the
            // 8-byte alignment load commands are supposed to have.
            let command = unsafe { std::ptr::read_unaligned(cursor as *const LoadCommand) };
            let size = command.size as usize;
            if size < std::mem::size_of::<LoadCommand>() || cursor.wrapping_add(size) > end {
                break;
            }

            match command.kind {
                LC_UUID if size >= std::mem::size_of::<UuidCommand>() => {
                    // SAFETY: the kind and size together say this is a
                    // `UuidCommand`, and it lies within the mapped region.
                    let command = unsafe { std::ptr::read_unaligned(cursor as *const UuidCommand) };
                    image.uuid = Some(hex(&command.uuid));
                }
                LC_SEGMENT_64 if size >= std::mem::size_of::<SegmentCommand64>() => {
                    // SAFETY: as above, for a segment command.
                    let segment =
                        unsafe { std::ptr::read_unaligned(cursor as *const SegmentCommand64) };
                    let address = segment.vm_address as usize;
                    let extent = segment.vm_size as usize;

                    // Executable segments are where return addresses live:
                    // `__TEXT`, and `__TEXT_EXEC` where a platform uses one.
                    if segment.initial_protection & EXECUTABLE != 0 && extent > 0 {
                        code_start = code_start.min(address);
                        code_end = code_end.max(address.saturating_add(extent));
                    }
                }
                _ => {}
            }
            cursor = cursor.wrapping_add(size);
        }

        if code_start != usize::MAX && code_end > code_start {
            image.code_start = code_start;
            image.code_size = code_end - code_start;
        }
        image
    }
}

#[cfg(all(unix, not(target_vendor = "apple"), not(miri)))]
mod imp {
    use super::{hex, Module};
    use crate::symbol::dl;
    use std::ffi::{c_char, c_int, c_void};

    extern "C" {
        fn dl_iterate_phdr(
            callback: extern "C" fn(*mut DlPhdrInfo, usize, *mut c_void) -> c_int,
            data: *mut c_void,
        ) -> c_int;
    }

    /// The prefix of `struct dl_phdr_info` that has been stable since glibc 2.2.
    ///
    /// The real struct has more fields after these, and different libcs add
    /// different ones. Only the prefix is declared, and only ever read behind a
    /// pointer the loader owns, so the trailing fields are none of our business.
    #[repr(C)]
    struct DlPhdrInfo {
        /// The load bias: a runtime address minus this is the address in the
        /// file, which is what `addr2line` and `llvm-symbolizer` expect.
        address: usize,
        name: *const c_char,
        headers: *const ProgramHeader,
        header_count: u16,
    }

    #[cfg(target_pointer_width = "64")]
    #[repr(C)]
    struct ProgramHeader {
        kind: u32,
        flags: u32,
        offset: u64,
        virtual_address: u64,
        physical_address: u64,
        file_size: u64,
        memory_size: u64,
        alignment: u64,
    }

    // `p_flags` moves from the second field to the seventh between the two
    // classes. Reusing the 64-bit order with narrower fields is the mistake this
    // separate definition exists to avoid.
    #[cfg(target_pointer_width = "32")]
    #[repr(C)]
    struct ProgramHeader {
        kind: u32,
        offset: u32,
        virtual_address: u32,
        physical_address: u32,
        file_size: u32,
        memory_size: u32,
        flags: u32,
        alignment: u32,
    }

    const PT_LOAD: u32 = 1;
    const PT_NOTE: u32 = 4;
    const PF_X: u32 = 1;
    const NT_GNU_BUILD_ID: u32 = 3;

    #[repr(C)]
    struct NoteHeader {
        name_size: u32,
        description_size: u32,
        kind: u32,
    }

    pub(super) fn capture() -> Vec<Module> {
        let mut modules: Vec<Module> = Vec::new();
        // SAFETY: `visit` matches the callback signature, and the context
        // pointer is a live `Vec` for the duration of the call.
        unsafe {
            dl_iterate_phdr(visit, std::ptr::addr_of_mut!(modules) as *mut c_void);
        }

        // glibc reports the main executable with an empty name — always, not in
        // some configurations. Left empty, the one image whose frames matter
        // most is the one that cannot be resolved, so it is filled in from the
        // kernel's own answer.
        if modules.iter().any(|module| module.path.is_empty()) {
            if let Ok(executable) = std::fs::read_link("/proc/self/exe") {
                let executable = executable.to_string_lossy().into_owned();
                for module in modules.iter_mut().filter(|m| m.path.is_empty()) {
                    module.path.clone_from(&executable);
                }
            }
        }
        modules
    }

    /// Called once per loaded object by the dynamic loader.
    ///
    /// `extern "C"` rather than `extern "C-unwind"` deliberately: a panic here
    /// would have to cross the loader's frames, and aborting is the only defined
    /// outcome. Nothing in the body panics on purpose.
    extern "C" fn visit(info: *mut DlPhdrInfo, _size: usize, data: *mut c_void) -> c_int {
        // SAFETY: the loader passes a valid `dl_phdr_info` for the duration of
        // the callback, and `data` is the `Vec` handed to `dl_iterate_phdr`.
        let (info, modules) = unsafe { (&*info, &mut *(data as *mut Vec<Module>)) };

        let mut lowest = usize::MAX;
        let mut code_start = usize::MAX;
        let mut code_end = 0usize;
        let mut build_id = None;

        for index in 0..usize::from(info.header_count) {
            // SAFETY: the loader guarantees `header_count` headers at `headers`.
            let header = unsafe { &*info.headers.add(index) };
            let address = header.virtual_address as usize;
            let extent = header.memory_size as usize;
            match header.kind {
                PT_LOAD => {
                    lowest = lowest.min(address);
                    // Only the executable part. An image's whole span is the
                    // wrong region: the kernel places `[vdso]` inside the gap
                    // between `ld.so`'s two `PT_LOAD`s, so spans overlap.
                    if header.flags & PF_X != 0 && extent > 0 {
                        code_start = code_start.min(address);
                        code_end = code_end.max(address.saturating_add(extent));
                    }
                }
                PT_NOTE if build_id.is_none() => {
                    let start = info.address.wrapping_add(address);
                    // SAFETY: a `PT_NOTE` segment is mapped at the biased
                    // virtual address for `memory_size` bytes.
                    build_id = unsafe { find_build_id(start as *const u8, extent) };
                }
                _ => {}
            }
        }

        if code_start == usize::MAX || code_end <= code_start {
            // Nothing executable: not an image a return address can be in.
            return 0;
        }

        modules.push(Module {
            // SAFETY: the loader's name is NUL-terminated, or null.
            path: unsafe { dl::string_at(info.name) }.unwrap_or_default(),
            start: info.address.wrapping_add(code_start),
            size: code_end - code_start,
            bias: info.address,
            image_base: info
                .address
                .wrapping_add(if lowest == usize::MAX { 0 } else { lowest }),
            build_id,
        });
        0
    }

    /// Scans a `PT_NOTE` segment for the GNU build identity.
    ///
    /// # Safety
    ///
    /// `start` must point to `len` readable bytes.
    unsafe fn find_build_id(start: *const u8, len: usize) -> Option<String> {
        let header_size = std::mem::size_of::<NoteHeader>();
        let mut at = 0usize;
        while at + header_size <= len {
            // SAFETY: the bounds check keeps this inside the mapped segment.
            // Read unaligned: notes are usually 4-aligned, but a producer that
            // aligns to 8 would otherwise make this undefined behaviour rather
            // than merely a miss.
            let note = unsafe { std::ptr::read_unaligned(start.add(at) as *const NoteHeader) };
            let name_size = align4(note.name_size as usize)?;
            let description_size = note.description_size as usize;
            let name_at = at.checked_add(header_size)?;
            let description_at = name_at.checked_add(name_size)?;
            if description_at.checked_add(description_size)? > len {
                return None;
            }

            if note.kind == NT_GNU_BUILD_ID && note.name_size as usize >= 4 {
                // SAFETY: within the segment, per the bounds checks above.
                let name = unsafe { std::slice::from_raw_parts(start.add(name_at), 4) };
                if name == b"GNU\0" {
                    // SAFETY: as above, for the description.
                    let id = unsafe {
                        std::slice::from_raw_parts(start.add(description_at), description_size)
                    };
                    return Some(hex(id));
                }
            }

            at = description_at.checked_add(align4(description_size)?)?;
        }
        None
    }

    /// Rounds up to the 4-byte alignment ELF notes use. `None` on overflow.
    fn align4(value: usize) -> Option<usize> {
        value.checked_add(3).map(|rounded| rounded & !3)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Builds a note the way a linker lays one out.
        fn note(kind: u32, name: &[u8], description: &[u8]) -> Vec<u8> {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&(name.len() as u32).to_ne_bytes());
            bytes.extend_from_slice(&(description.len() as u32).to_ne_bytes());
            bytes.extend_from_slice(&kind.to_ne_bytes());
            bytes.extend_from_slice(name);
            while bytes.len() % 4 != 0 {
                bytes.push(0);
            }
            bytes.extend_from_slice(description);
            while bytes.len() % 4 != 0 {
                bytes.push(0);
            }
            bytes
        }

        fn scan(bytes: &[u8]) -> Option<String> {
            // SAFETY: `bytes` is a live slice of exactly this length.
            unsafe { find_build_id(bytes.as_ptr(), bytes.len()) }
        }

        #[test]
        fn a_build_id_note_is_found() {
            let bytes = note(NT_GNU_BUILD_ID, b"GNU\0", &[0xDE, 0xAD, 0xBE, 0xEF]);
            assert_eq!(scan(&bytes).as_deref(), Some("deadbeef"));
        }

        #[test]
        fn a_build_id_after_other_notes_is_found() {
            let mut bytes = note(4, b"GNU\0", &[1, 2, 3, 4, 5]);
            bytes.extend(note(1, b"Xen\0", &[9; 9]));
            bytes.extend(note(NT_GNU_BUILD_ID, b"GNU\0", &[0xAB; 20]));
            assert_eq!(scan(&bytes).as_deref(), Some(&"ab".repeat(20)[..]));
        }

        #[test]
        fn a_note_from_another_producer_is_not_mistaken_for_one() {
            // The same type number in a different namespace means something
            // else entirely.
            let bytes = note(NT_GNU_BUILD_ID, b"Go\0\0", &[1, 2, 3, 4]);
            assert_eq!(scan(&bytes), None);
        }

        #[test]
        fn a_truncated_note_is_refused_rather_than_read_past() {
            let full = note(NT_GNU_BUILD_ID, b"GNU\0", &[0xAB; 20]);
            for length in 0..full.len() {
                // Must not read beyond `length`, and must not spin. Miri and
                // the sanitizers are what make this assertion mean something.
                let _ = scan(&full[..length]);
            }
        }

        #[test]
        fn a_zero_sized_note_does_not_spin() {
            let bytes = note(7, b"", b"");
            assert_eq!(scan(&bytes), None);
        }

        #[test]
        fn a_note_claiming_an_enormous_name_is_refused() {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&u32::MAX.to_ne_bytes());
            bytes.extend_from_slice(&u32::MAX.to_ne_bytes());
            bytes.extend_from_slice(&NT_GNU_BUILD_ID.to_ne_bytes());
            bytes.extend_from_slice(&[0; 32]);
            assert_eq!(scan(&bytes), None);
        }

        #[test]
        fn an_unaligned_segment_is_read_rather_than_faulted_on() {
            // Some producers align notes to 8, which this walk does not follow.
            // Missing the identity is acceptable; undefined behaviour is not.
            let bytes = note(NT_GNU_BUILD_ID, b"GNU\0", &[0xCD; 8]);
            let mut shifted = vec![0u8];
            shifted.extend_from_slice(&bytes);
            let _ = scan(&shifted[1..]);
        }
    }
}

#[cfg(all(windows, not(miri)))]
mod imp {
    use super::Module;
    use std::ffi::c_void;

    // The `K32`-prefixed forms live in kernel32, which every process already
    // links, so this needs no psapi import library.
    #[link(name = "kernel32", kind = "raw-dylib")]
    extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn K32EnumProcessModules(
            process: *mut c_void,
            modules: *mut *mut c_void,
            size: u32,
            needed: *mut u32,
        ) -> i32;
        fn K32GetModuleFileNameExW(
            process: *mut c_void,
            module: *mut c_void,
            filename: *mut u16,
            size: u32,
        ) -> u32;
        fn K32GetModuleInformation(
            process: *mut c_void,
            module: *mut c_void,
            info: *mut ModuleInformation,
            size: u32,
        ) -> i32;
    }

    #[repr(C)]
    struct ModuleInformation {
        base: *mut c_void,
        size: u32,
        entry_point: *mut c_void,
    }

    pub(super) fn capture() -> Vec<Module> {
        // SAFETY: returns a pseudo-handle; no failure mode.
        let process = unsafe { GetCurrentProcess() };

        // Room for 256 modules first, which answers both the size question and
        // the contents question in one call for any ordinary process. A module
        // loaded between the two calls simply does not appear, which is the same
        // race the Apple path documents.
        let mut handles: Vec<*mut c_void> = vec![std::ptr::null_mut(); 256];
        loop {
            let capacity = std::mem::size_of_val(handles.as_slice()) as u32;
            let mut needed = 0u32;
            // SAFETY: `handles` has room for `capacity` bytes.
            let listed = unsafe {
                K32EnumProcessModules(process, handles.as_mut_ptr(), capacity, &mut needed)
            };
            if listed == 0 {
                return Vec::new();
            }
            let wanted = needed as usize / std::mem::size_of::<*mut c_void>();
            if needed <= capacity {
                handles.truncate(wanted);
                break;
            }
            handles.resize(wanted, std::ptr::null_mut());
        }

        let mut modules = Vec::with_capacity(handles.len());
        for handle in handles {
            let mut information = ModuleInformation {
                base: std::ptr::null_mut(),
                size: 0,
                entry_point: std::ptr::null_mut(),
            };
            // SAFETY: `handle` came from `K32EnumProcessModules`, and the
            // structure is the size the call is told it is.
            let described = unsafe {
                K32GetModuleInformation(
                    process,
                    handle,
                    &mut information,
                    std::mem::size_of::<ModuleInformation>() as u32,
                )
            };
            if described == 0 {
                continue;
            }

            // Not `MAX_PATH`: long paths exist, and a truncated path resolves
            // against the wrong file or none at all.
            let mut name = [0u16; 1024];
            // SAFETY: `name` is a writable buffer of the length passed.
            let length = unsafe {
                K32GetModuleFileNameExW(process, handle, name.as_mut_ptr(), name.len() as u32)
            };

            let base = information.base as usize;
            modules.push(Module {
                path: String::from_utf16_lossy(&name[..length as usize]),
                start: base,
                // Sections are not enumerated, so the region is the whole image.
                // Windows images do not overlap, so this costs nothing in
                // attribution accuracy.
                size: information.size as usize,
                // Subtracting the base gives a relative virtual address, which
                // `llvm-symbolizer` takes with `--relative-address`.
                bias: base,
                image_base: base,
                // The PDB signature lives in the PE debug directory, which means
                // reading an object file rather than asking the OS. Deferred
                // rather than guessed at.
                build_id: None,
            });
        }
        modules
    }
}

/// No platform loader to ask: a target this crate has no backend for, or Miri.
///
/// Miri is the interesting case. It cannot execute `_dyld_image_count`,
/// `dl_iterate_phdr`, or `K32EnumProcessModules` — they are foreign functions
/// with no shim — and reaching one aborts the whole run rather than failing a
/// test. That made every test that takes a `Snapshot` unrunnable under Miri,
/// which is most of the output layer, and it went unnoticed because the module
/// map's own tests were correctly marked `#[cfg_attr(miri, ignore)]` while the
/// tests that reach it *transitively* were not.
///
/// Reporting no images is honest: under Miri there is no address space layout
/// to describe. The tests that exist to check the real backends are ignored
/// under Miri anyway, so nothing that was being verified stops being verified.
#[cfg(any(miri, not(any(unix, windows))))]
mod imp {
    use super::Module;

    pub(super) fn capture() -> Vec<Module> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn module(start: usize, size: usize) -> Module {
        Module {
            path: format!("/image/at/{start:x}"),
            start,
            size,
            bias: 0,
            image_base: start,
            build_id: None,
        }
    }

    #[test]
    fn an_image_contains_its_first_byte_but_not_the_one_past_its_last() {
        let image = module(0x1000, 0x100);
        assert!(image.contains(0x1000));
        assert!(image.contains(0x10FF));
        assert!(!image.contains(0x1100));
        assert!(!image.contains(0x0FFF));
        assert_eq!(image.file_address(0x1080), Some(0x1080));
        assert_eq!(image.file_address(0x1100), None);
    }

    /// The three numbers are different numbers, and this is where that is
    /// pinned down.
    #[test]
    fn a_file_address_is_the_runtime_address_less_the_bias() {
        // An ELF position-independent executable: bias is where it was mapped,
        // so file addresses are small.
        let mut elf = module(0x7F00_0000, 0x1000);
        elf.bias = 0x7F00_0000;
        assert_eq!(elf.file_address(0x7F00_0234), Some(0x234));

        // Mach-O: the bias is the slide, and file addresses start at
        // 0x1_0000_0000 — so a file address is much larger than the offset
        // from the image base, which is exactly what `llvm-symbolizer` needs.
        let mut apple = module(0x1_0400_0000, 0x1000);
        apple.bias = 0x0400_0000;
        assert_eq!(apple.file_address(0x1_0400_0234), Some(0x1_0000_0234));

        // An ELF non-PIE: no bias at all, so the runtime address *is* the file
        // address.
        let mut fixed = module(0x40_0000, 0x1000);
        fixed.bias = 0;
        assert_eq!(fixed.file_address(0x40_0234), Some(0x40_0234));
    }

    #[test]
    fn an_empty_image_contains_nothing() {
        // A platform that declines to report an extent must not swallow every
        // address that happens to be above its base.
        let image = module(0x1000, 0);
        assert!(!image.contains(0x1000));
        assert_eq!(image.file_address(0x1000), None);
    }

    #[test]
    fn lookup_finds_the_image_an_address_belongs_to() {
        let modules = vec![
            module(0x1000, 0x100),
            module(0x2000, 0x100),
            module(0x3000, 0x100),
        ];
        assert_eq!(containing(&modules, 0x2050).unwrap().start, 0x2000);
        assert_eq!(containing(&modules, 0x1000).unwrap().start, 0x1000);
        assert_eq!(containing(&modules, 0x30FF).unwrap().start, 0x3000);
    }

    #[test]
    fn an_address_in_no_image_resolves_to_nothing() {
        let modules = vec![module(0x1000, 0x100), module(0x3000, 0x100)];
        // Below every image, in the gap between two, and past the last.
        assert!(containing(&modules, 0x500).is_none());
        assert!(containing(&modules, 0x2000).is_none());
        assert!(containing(&modules, 0x4000).is_none());
        assert!(containing(&[], 0x1000).is_none());
    }

    #[cfg(unix)]
    #[test]
    fn hexadecimal_is_lower_case_and_zero_padded() {
        assert_eq!(hex(&[0x00, 0x0f, 0xa0, 0xff]), "000fa0ff");
        assert_eq!(hex(&[]), "");
    }

    /// The whole point: an address from this very binary must land in a module,
    /// in an image with a name, at a file address a symbolizer could resolve.
    #[test]
    #[cfg_attr(miri, ignore = "enumerating loaded images needs the platform's loader")]
    fn this_binary_appears_in_its_own_module_map() {
        let modules = capture();
        assert!(!modules.is_empty(), "no images were found at all");

        let here = this_binary_appears_in_its_own_module_map as *const () as usize;
        let image = containing(&modules, here)
            .unwrap_or_else(|| panic!("no image contains this function at {here:#x}"));

        assert!(
            !image.path.is_empty(),
            "the image at {:#x} has no path, so nothing in it can be symbolized",
            image.start
        );
        assert!(
            image.file_address(here).is_some(),
            "no file address for {here:#x}"
        );
        assert!(image.start > 0, "an image at address zero is not a thing");
        assert!(
            image.image_base > 0,
            "an image based at zero is not a thing"
        );
    }

    /// The loader is in the map, and its recorded code region is where its code
    /// actually is.
    ///
    /// dyld holds the outermost frames of every stack — `start`, the image
    /// initializers, lazy binding — and it does not appear in the image list it
    /// maintains, so it has to be found some other way. It has now been found
    /// the wrong way twice, and neither failure was visible in a passing suite:
    /// first by asking `dladdr` about `_dyld_image_count`, which lives in
    /// `libdyld.dylib` and is therefore already in the list, so the answer was
    /// discarded as a duplicate; then by deriving its slide from the segment
    /// that maps file offset zero, which no shared-cache image has, so the
    /// slide came out as the whole header address and the region landed 6 GiB
    /// from the code.
    ///
    /// The kernel is the oracle for *where* rather than a path — a test that
    /// looked for `/usr/lib/dyld` would be checking Apple's naming rather than
    /// this code.
    ///
    /// # The assertion that has to be on `bias`
    ///
    /// `describe_with_derived_slide` computes `slide = header - code_start` and
    /// then `start = code_start + slide`, which is `header` for *any* value of
    /// `code_start`. So `contains(header)` is a tautology dressed as a check: it
    /// reduces to `size > 0`, which the function already guarantees. The one
    /// field that carries the derivation is `bias`, and it needs an oracle this
    /// code does not share.
    ///
    /// It has one. Every image mapped from the shared cache is slid by the same
    /// amount, dyld is one of them, and the *other* cache images get their bias
    /// straight from `_dyld_get_image_vmaddr_slide`. Comparing against the slide
    /// most images agree on therefore checks the derivation against dyld's own
    /// answer, without naming a path or a number. Measured on macOS 15.6 arm64:
    /// 44 of 46 images share a slide of `0x7914000`, and that is dyld's.
    #[test]
    #[cfg(all(target_vendor = "apple", not(miri)))]
    fn the_dynamic_loader_is_in_the_map() {
        let Some(header) = imp::loader_header() else {
            // A sandbox that denies task introspection. Documented on
            // `loader_header` as leaving stack bases unattributed rather than
            // wrong, so there is nothing to assert.
            return;
        };
        let modules = capture();
        let loader = modules
            .iter()
            .find(|module| module.image_base == header)
            .unwrap_or_else(|| {
                panic!(
                    "the loader at {header:#x} is not in the map, so the outermost \
                     frame of every stack is unattributable"
                )
            });
        assert!(!loader.path.is_empty());

        // The slide the rest of the shared cache is at, taken from the biases
        // dyld reported for the images it does enumerate.
        let mut tally: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for module in &modules {
            if module.image_base != header {
                *tally.entry(module.bias).or_default() += 1;
            }
        }
        let Some((&shared_slide, &agreeing)) = tally.iter().max_by_key(|&(_, count)| count) else {
            return;
        };
        if agreeing < 3 {
            // Not a shared cache: nothing here to compare dyld against. A
            // platform where system libraries are separately slid would land
            // here, and this test has no oracle for that case rather than a
            // failing one.
            eprintln!("skipping the loader slide check: no shared slide among {agreeing} images");
            return;
        }

        assert_eq!(
            loader.bias, shared_slide,
            "the loader's bias is {:#x} while the {agreeing} images dyld enumerated \
             from the same shared cache are slid by {shared_slide:#x}. `bias` is what \
             `llvm-symbolizer` and `addr2line` subtract, so a wrong one resolves every \
             frame in the loader to the wrong place — and `start` cannot show it, \
             because the derivation makes `start == header` whatever the slide is.",
            loader.bias
        );
    }

    /// A lookup bisects, so the map must be sorted; and it takes the nearest
    /// image at or below the address, so an overlap would silently attribute a
    /// frame to whichever image happens to come later.
    #[test]
    #[cfg_attr(miri, ignore = "enumerating loaded images needs the platform's loader")]
    fn the_map_is_ordered_and_free_of_overlaps() {
        let modules = capture();
        for pair in modules.windows(2) {
            assert!(
                pair[0].start <= pair[1].start,
                "the map is not sorted: {:#x} then {:#x}",
                pair[0].start,
                pair[1].start
            );
            assert!(
                pair[1].start >= pair[0].start + pair[0].size,
                "{} at {:#x}+{:#x} overlaps {} at {:#x}",
                pair[0].path,
                pair[0].start,
                pair[0].size,
                pair[1].path,
                pair[1].start
            );
        }
    }

    /// A build identity is what tells you the binary you are symbolizing against
    /// is the one that produced the profile.
    #[test]
    #[cfg_attr(
        any(miri, windows),
        ignore = "Apple and Linux stamp a build identity; the PE debug directory is not read yet"
    )]
    fn images_carry_a_build_identity() {
        let modules = capture();
        let here = images_carry_a_build_identity as *const () as usize;
        let image = containing(&modules, here).expect("the image containing this test");
        let build_id = image
            .build_id
            .as_deref()
            .expect("this binary should carry a build identity");
        assert!(
            build_id.len() >= 32,
            "a UUID is 32 hex digits and a GNU build id is usually 40: {build_id}"
        );
        assert!(
            build_id.bytes().all(|b| b.is_ascii_hexdigit()),
            "{build_id}"
        );
    }

    /// Every image the loader reports should be resolvable, not just this one.
    #[test]
    #[cfg_attr(miri, ignore = "enumerating loaded images needs the platform's loader")]
    fn every_image_in_the_map_has_a_path_and_a_size() {
        for image in capture() {
            assert!(
                !image.path.is_empty(),
                "an image at {:#x} has no path",
                image.start
            );
            assert!(image.size > 0, "{} has no extent", image.path);
        }
    }
}
