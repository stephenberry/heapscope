//! Guard slots go back to the table when their threads exit.
//!
//! A binary of its own, with one `#[test]`, because what it reads is
//! process-wide: `guard::stats().claimed` counts the slots held by live threads
//! anywhere in the process. As a unit test it ran beside others that hold
//! sixteen slots at once, and a drift allowance wide enough for them was
//! exceeded on a loaded runner while proving nothing about leaks. Alone, the
//! only threads claiming slots are the ones this test starts, so the count can
//! be compared exactly.
//!
//! No `#[global_allocator]`: with the shim installed, the harness's own threads
//! would claim slots whenever they allocated.

use heapscope::internals::guard;

#[test]
fn slots_are_reclaimed_when_threads_exit() {
    #[cfg(miri)]
    const ROUNDS: usize = 8;
    #[cfg(not(miri))]
    const ROUNDS: usize = 200;

    let before = guard::stats();
    for _ in 0..ROUNDS {
        std::thread::spawn(|| {
            let _guard = guard::enter().expect("a fresh thread gets a slot");
        })
        .join()
        .expect("the thread exits cleanly");
    }
    let after = guard::stats();

    // `join` returns after the thread's thread-local destructors have run,
    // and releasing the slot is one of them, so nothing is still in flight.
    assert_eq!(
        after.claimed, before.claimed,
        "slots leaked: {before:?} -> {after:?}"
    );
    assert_eq!(after.refused, before.refused, "no refusal was expected");
}
