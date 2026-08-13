//! Deliberate defects, for proving that a sanitizer can see this crate.
//!
//! Every case allocates through heapscope's shim first, so what the sanitizer
//! is asked to see is memory that reached `malloc` the way a profiled program's
//! memory does. That is the whole question: heapscope installs a
//! `#[global_allocator]` between the program and `malloc`, which is where ASan
//! puts its redzones, and TSan has to reason about happens-before through a
//! crate whose subject is hand-rolled locking. If either composition were
//! broken the tool would report nothing, the suite would pass, and the job
//! would be green for having watched nothing at all.
//!
//! Run by `ci/sanitizers.sh`, which requires each case below to be reported and
//! fails the job when one is not.

// The point of this program is to do what these lints forbid.
#![allow(clippy::undocumented_unsafe_blocks, unknown_lints)]

#[global_allocator]
static ALLOC: heapscope::Alloc<std::alloc::System> = heapscope::Alloc::system();

fn main() {
    match std::env::args().nth(1).unwrap_or_default().as_str() {
        // ASan: a read from a block that has been released.
        "use-after-free" => {
            let mut bytes: Vec<u8> = Vec::with_capacity(64);
            bytes.push(7);
            let released = bytes.as_ptr();
            drop(bytes);
            // SAFETY: none, deliberately. This is the control.
            println!("{}", unsafe { std::ptr::read_volatile(released) });
        }

        // ASan: a read one byte past the end of a live block. Distinct from the
        // case above because it exercises the redzone rather than the
        // quarantine, and a shim could plausibly break one and not the other.
        "overflow" => {
            let mut bytes: Vec<u8> = Vec::with_capacity(64);
            bytes.push(7);
            // SAFETY: none, deliberately.
            let past = unsafe { bytes.as_ptr().add(64) };
            // SAFETY: none, deliberately.
            println!("{}", unsafe { std::ptr::read_volatile(past) });
        }

        // TSan: two threads writing one cell with nothing ordering them.
        //
        // The cell is heap memory rather than a `static`, so the raced bytes
        // are bytes the shim handed out — which is the composition in question.
        // The address travels as a `usize` because the point is to defeat the
        // ownership rules that would otherwise make this unwriteable.
        "race" => {
            let cell = Box::leak(Box::new(0u64)) as *mut u64 as usize;
            let writers: Vec<_> = (0..2)
                .map(|value| {
                    std::thread::spawn(move || {
                        for _ in 0..10_000 {
                            // SAFETY: none, deliberately. Two threads, one
                            // cell, no synchronisation between them.
                            unsafe { std::ptr::write_volatile(cell as *mut u64, value) };
                        }
                    })
                })
                .collect();
            for writer in writers {
                writer.join().expect("a writer thread panicked");
            }
            println!("done");
        }

        // The control's own control. A sanitizer that reports something here is
        // reporting on a program with nothing wrong in it, and its verdict on
        // the cases above would be worth as little as its silence.
        _ => {
            let bytes: Vec<u8> = vec![7; 64];
            println!("{}", bytes[0]);
        }
    }
}
