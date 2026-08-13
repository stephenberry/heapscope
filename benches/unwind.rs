//! Stack-capture cost.
//!
//! This is the number that decides the architecture. PLAN.md section 5.1 puts a
//! baseline allocation at 15.8 ns and `_Unwind_Backtrace` at depth 32 at
//! ~8,335 ns; if the frame-pointer walk did not land near the former, the whole
//! premise of capturing a stack on *every* allocation would collapse and the
//! design would have to move to sampling by default.
//!
//! Comparison strategies are benchmarked alongside so the ratio is measured on
//! the machine running it rather than quoted from the plan. Being 400x apart is
//! the kind of claim that should be re-derived, not inherited.

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use heapscope::internals::stack;
use heapscope::unwind::frame_pointer::{self, RealStack};
use heapscope::unwind::{Capture, Outcome};

/// Puts a known number of frames on the stack, then runs `f` at the bottom.
///
/// `#[inline(never)]` plus `black_box` on the return value defeats both inlining
/// and tail-call elimination, either of which would silently flatten the stack
/// this is trying to build.
#[inline(never)]
fn at_depth<R>(depth: usize, f: &mut impl FnMut() -> R) -> R {
    if depth == 0 {
        return black_box(f());
    }
    black_box(at_depth(depth - 1, f))
}

/// Establishes what `at_depth` itself costs, so the capture numbers below can
/// be read as capture cost rather than capture-plus-scaffolding.
///
/// The control returns the same type the real benchmark returns. That is not
/// pedantry: `at_depth` is generic over its return type, so a control returning
/// `usize` while the measurement returns `Capture` moves a different amount of
/// data back through every one of the frames and is not a control at all.
fn scaffolding_control(c: &mut Criterion) {
    let mut group = c.benchmark_group("capture/control");
    group.measurement_time(Duration::from_secs(3));

    let sentinel = Capture {
        len: 0,
        outcome: Outcome::Complete,
    };

    for depth in [1usize, 4, 12, 28] {
        group.bench_with_input(BenchmarkId::from_parameter(depth), &depth, |b, &depth| {
            b.iter(|| black_box(at_depth(depth, &mut || black_box(sentinel))));
        });
    }

    group.finish();
}

fn frame_pointer_walk(c: &mut Criterion) {
    let mut group = c.benchmark_group("capture/frame_pointer");
    group.measurement_time(Duration::from_secs(3));

    let source = RealStack::new(stack::current_bounds());

    for depth in [1usize, 4, 12, 28] {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(depth), &depth, |b, &depth| {
            let mut out = [0usize; 64];
            b.iter(|| {
                let capture = at_depth(depth, &mut || {
                    frame_pointer::capture(&source, 0, black_box(&mut out))
                });
                black_box(capture)
            });
        });
    }

    group.finish();
}

/// Cost per *frame*, measured as a slope.
///
/// Subtracting a control from an absolute time is fragile — the control is
/// never quite the same code. Varying only the output buffer size avoids that
/// entirely: the scaffolding, the call, and the fixed setup are byte-identical
/// across these runs, so whatever differs between `cap=1` and `cap=32` is
/// exactly the cost of walking 31 more frames. The intercept is the fixed cost
/// of starting a capture.
///
/// The stack is made deliberately deeper than the largest cap so that the walk
/// is always cut short by the buffer rather than by running out of frames,
/// which would flatten the slope at the top end and understate the cost.
fn cost_per_frame(c: &mut Criterion) {
    let mut group = c.benchmark_group("capture/per_frame");
    group.measurement_time(Duration::from_secs(3));

    let source = RealStack::new(stack::current_bounds());
    let mut storage = [0usize; 64];

    for cap in [1usize, 2, 4, 8, 16, 32] {
        group.bench_with_input(BenchmarkId::from_parameter(cap), &cap, |b, &cap| {
            b.iter(|| {
                let out = &mut storage[..cap];
                let capture = at_depth(40, &mut || {
                    frame_pointer::capture(&source, 0, black_box(out))
                });
                black_box(capture)
            });
        });
    }

    group.finish();
}

/// The same walk with stack bounds withheld.
///
/// Bounds cost one comparison pair per frame. If that turned out to be
/// expensive it would be worth caching them more aggressively; this measures
/// whether it is.
fn bounds_checking_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("capture/bounds");
    group.measurement_time(Duration::from_secs(3));

    let with_bounds = RealStack::new(stack::current_bounds());
    let without_bounds = RealStack::new(None);

    let mut out = [0usize; 64];
    group.bench_function("with_bounds", |b| {
        b.iter(|| {
            let capture = at_depth(12, &mut || {
                frame_pointer::capture(&with_bounds, 0, black_box(&mut out))
            });
            black_box(capture)
        });
    });
    group.bench_function("without_bounds", |b| {
        b.iter(|| {
            let capture = at_depth(12, &mut || {
                frame_pointer::capture(&without_bounds, 0, black_box(&mut out))
            });
            black_box(capture)
        });
    });

    group.finish();
}

/// What the profiler would cost per allocation if it used the standard library.
///
/// Kept as a benchmark rather than a footnote because it is the claim most
/// likely to be doubted, and because `std::backtrace` is what a reader would
/// otherwise assume we should have used.
fn std_backtrace_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("capture/std_backtrace");
    // Each sample costs microseconds, so a shorter window still gathers plenty.
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(30);

    group.bench_function("force_capture", |b| {
        b.iter(|| {
            let bt = at_depth(12, &mut || std::backtrace::Backtrace::force_capture());
            black_box(bt)
        });
    });

    group.finish();
}

/// The platform's own unwinder, which is what [`Strategy::System`] selects.
///
/// This is the number that justifies the opt-in being an opt-in. The plan
/// quotes libc `backtrace` at 157 ns and `_Unwind_Backtrace` at 8,335 ns from a
/// separate probe; measuring it here makes the ratio this machine's finding
/// rather than an inherited claim, and it is the same shape of check that
/// caught the frame-pointer numbers being right.
///
/// On Windows this measures `RtlCaptureStackBackTrace`, which is the *default*
/// there rather than an escape hatch — so on that platform this benchmark is
/// measuring the hot path, not the alternative to it.
fn system_unwinder(c: &mut Criterion) {
    let mut group = c.benchmark_group("capture/system");
    // A capture here can cost microseconds, so a shorter window still gathers
    // plenty of samples.
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(50);

    for depth in [1usize, 12] {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(depth), &depth, |b, &depth| {
            let mut out = [0usize; 64];
            b.iter(|| {
                let capture = at_depth(depth, &mut || {
                    heapscope::unwind::system::capture(0, black_box(&mut out))
                });
                black_box(capture)
            });
        });
    }

    group.finish();
}

/// The floor: what an allocation costs before any profiling is added.
///
/// Every capture number above is only meaningful relative to this.
fn baseline_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("capture/baseline");
    group.measurement_time(Duration::from_secs(3));

    group.bench_function("malloc_free_64b", |b| {
        b.iter(|| {
            let v: Vec<u8> = Vec::with_capacity(black_box(64));
            black_box(&v);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    baseline_allocation,
    scaffolding_control,
    frame_pointer_walk,
    cost_per_frame,
    bounds_checking_overhead,
    std_backtrace_comparison,
    system_unwinder,
);
criterion_main!(benches);
