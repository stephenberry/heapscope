//! What the `save_*` methods promise about the file that was already there.
//!
//! [`Snapshot::save_dhat_v2`], [`save_native`], [`save_html`] and
//! [`save_folded`] all document the same guarantee in the same words: the
//! profile "is written beside its destination and renamed into place, so that a
//! full disk, a write error, or a process killed mid-write leaves the previous
//! profile intact rather than replacing it with a truncated file that no viewer
//! will open."
//!
//! They share one implementation, `save_with`, so the cases below drive it
//! through `save_dhat_v2` rather than four times over. The one arm that is not
//! shared is in `tests/folded_output.rs`: folded output is the only emitter that
//! can refuse, and a refusal reaches the cleanup path *after* the temporary has
//! been created, which no failure here does.
//!
//! That is a durability promise about someone else's data, and until M7 chunk M
//! nothing checked any part of it. Neither did anything check the reason
//! `temporary_path` mixes a process id *and* a per-write counter into the name,
//! which its own doc comment records as a defect that was found and fixed: two
//! concurrent writes to one destination — `Profiler::drop` racing the exit
//! handler, or two threads calling `save_dhat_v2` — created and truncated the
//! same temporary and wrote into it at independent offsets, "producing an
//! interleaved profile that no viewer will open". A fixed defect with no test is
//! a defect that can be reintroduced silently.
//!
//! [`save_native`]: heapscope::output::Snapshot::save_native
//! [`save_html`]: heapscope::output::Snapshot::save_html
//! [`save_folded`]: heapscope::output::Snapshot::save_folded

use heapscope::output::{Counters, PointKind, ProgramPoint, Snapshot};

/// A profile big enough that two writers sharing one file would interleave.
///
/// Size is the whole point of the shape: the defect being guarded against is
/// two `write` calls landing at independent offsets, which a file that fits in
/// one buffer flush can hide. Eight frames per point and four thousand points
/// is around a megabyte of JSON, which is hundreds of writes.
fn a_large_profile() -> Snapshot {
    const POINTS: usize = 4_000;

    let mut snapshot = Snapshot::default();
    snapshot.command = String::from("saving-test");
    snapshot.points = (0..POINTS)
        .map(|point| ProgramPoint {
            kind: PointKind::Recorded,
            // Distinct per point, so the emitter's fold cannot merge them and
            // shrink the file this test depends on being large.
            frames: (0..8)
                .map(|frame| 0x10_0000 + point * 512 + frame * 8)
                .collect(),
            counters: Counters {
                total_bytes: 4096 + point as u64,
                total_blocks: 1 + point as u64 % 7,
                total_lifetime: 17 * point as u64,
                curr_bytes: 128,
                curr_blocks: 1,
                max_bytes: 4096,
                max_blocks: 1,
                at_gmax_bytes: 64,
                at_gmax_blocks: 1,
            },
            unretired_lifetime: 3,
        })
        .collect();
    snapshot
}

/// Every `*.tmp` sibling the save path could have left behind.
fn leftovers(directory: &std::path::Path) -> Vec<String> {
    std::fs::read_dir(directory)
        .expect("reading the directory back")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".tmp"))
        .collect()
}

/// A write that fails leaves the profile that was already there untouched.
///
/// This is the promise in its literal form, and the only way to make the write
/// fail without reaching into the crate is to take away permission to create the
/// temporary. That is a Unix-only lever, and it does not work as root — where
/// the mode bits on a directory do not stop the owner — so the test proves it
/// can still fail before it concludes anything from passing.
#[test]
#[cfg(unix)]
#[cfg_attr(miri, ignore = "needs a real filesystem and real permission bits")]
fn a_failed_write_leaves_the_previous_profile_where_it_was() {
    use std::os::unix::fs::PermissionsExt;

    const PREVIOUS: &[u8] = b"{\"this\":\"is the profile that was already there\"}";

    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("profile.json");
    std::fs::write(&path, PREVIOUS).expect("writing the previous profile");

    let mode = std::fs::metadata(directory.path())
        .expect("reading the directory's mode")
        .permissions();
    std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o555))
        .expect("making the directory read-only");

    // Root ignores the bits just set, so the write would succeed and the
    // assertions below would be checking nothing.
    let writable = std::fs::File::create(directory.path().join("probe")).is_ok();
    if writable {
        std::fs::set_permissions(directory.path(), mode).expect("restoring the mode");
        eprintln!("skipping: this user can write to a read-only directory, so no write can be made to fail");
        return;
    }

    let outcome = a_large_profile().save_dhat_v2(&path);
    std::fs::set_permissions(directory.path(), mode).expect("restoring the mode");

    assert!(
        outcome.is_err(),
        "the save reported success from a directory it could not write to"
    );
    assert_eq!(
        std::fs::read(&path).expect("reading the previous profile back"),
        PREVIOUS,
        "a failed save replaced the profile that was already there"
    );
}

/// A write that fails takes its temporary with it.
///
/// A destination that is a directory is the portable way to fail *late*: the
/// temporary is created and written in full, and only the rename fails. So this
/// reaches the cleanup arm specifically, which the test above does not — there,
/// nothing was created to clean up.
#[test]
#[cfg_attr(miri, ignore = "needs a real filesystem")]
fn a_failed_write_leaves_no_temporary_behind() {
    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("profile.json");
    std::fs::create_dir(&path).expect("putting a directory where the profile would go");

    let outcome = a_large_profile().save_dhat_v2(&path);

    assert!(
        outcome.is_err(),
        "the save reported success writing over a directory"
    );
    assert_eq!(
        leftovers(directory.path()),
        Vec::<String>::new(),
        "a failed save left its temporary file in the reader's directory"
    );
}

/// Two threads saving to one path produce one whole profile, not a mixture.
///
/// The interleaved-write defect `temporary_path` was fixed for is invisible to
/// any single-threaded test, and invisible to a multi-threaded one whose profile
/// is small enough to be written in a single call. Both threads save the *same*
/// snapshot, so the destination has exactly one correct content and any
/// difference from it is bytes from two writers in one file.
///
/// It also asserts the file is not merely well-formed but complete: a truncated
/// profile is a plausible outcome of two writers sharing a file, and truncated
/// JSON is still JSON up to the point it stops.
#[test]
#[cfg_attr(miri, ignore = "needs a real filesystem and several threads")]
fn concurrent_saves_to_one_path_do_not_interleave() {
    const WRITERS: usize = 8;

    let directory = tempfile::tempdir().expect("a temporary directory");
    let path = directory.path().join("profile.json");
    let snapshot = a_large_profile();

    let expected = {
        let alone = directory.path().join("reference.json");
        snapshot.save_dhat_v2(&alone).expect("the reference save");
        std::fs::read(&alone).expect("reading the reference back")
    };
    assert!(
        expected.len() > 512 * 1024,
        "the profile is {} bytes, which is small enough for one write to place \
         it whole — so two writers sharing a file would not show here",
        expected.len()
    );

    std::thread::scope(|scope| {
        for _ in 0..WRITERS {
            let (path, snapshot) = (&path, &snapshot);
            scope.spawn(move || {
                snapshot.save_dhat_v2(path).expect("a concurrent save");
            });
        }
    });

    let written = std::fs::read(&path).expect("reading the profile back");
    assert_eq!(
        written.len(),
        expected.len(),
        "{WRITERS} writers left a profile of {} bytes where one writer leaves {}",
        written.len(),
        expected.len()
    );
    assert!(
        written == expected,
        "{WRITERS} writers left a profile that is the right length and the wrong \
         bytes, which is two writers' output in one file"
    );
    assert_eq!(
        leftovers(directory.path()),
        Vec::<String>::new(),
        "a temporary survived the writes that created it"
    );
}
