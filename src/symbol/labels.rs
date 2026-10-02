//! How an image is named in a frame that has no function name.
//!
//! Private to the crate, and reachable from outside only as
//! `heapscope::internals::image_labels`, which carries no stability promise.
//! The rule exists for [`FunctionNames`](super::FunctionNames) and for
//! `heapscope-symbolize`, which writes the same frames from a profile's module
//! map, and it is not something a user of `heapscope::symbol` should find there
//! and depend on.

use std::collections::{HashMap, HashSet};

/// What each of `paths` is called in a frame with no name: its file name,
/// unless another of `paths` has the same one, and then the whole path.
///
/// <div class="warning">
///
/// **Not part of the supported surface.** Public only so that
/// `heapscope-symbolize`, which writes folded files from a profile's module map
/// rather than a live one, labels images by this rule rather than by a copy of
/// it. The rule is what keeps two images apart, and a copy that drifted would
/// merge frames in one tool's output and not the other's.
///
/// </div>
///
/// The answer is as distinct as the paths are, which is the whole point.
/// Shortened labels cannot collide with each other, because a file name shared
/// by two distinct paths is never shortened. And one cannot collide with a
/// path kept whole: a label has no separator in it, so a path equal to it is its
/// own file name, which would make the file name shared, and then neither is
/// shortened. The same path given twice is one image, not two, and is
/// shortened.
///
/// Both `/` and `\` separate, whatever this platform's own rule is, so that a
/// Windows path reads the same on any machine that renders it. On Unix a
/// backslash is a legal file-name character, and splitting on one costs only a
/// shorter label: the distinctness argument above does not depend on where the
/// split falls.
///
/// A path with no file name, the empty one included, is given back whole, which
/// is the empty string for the empty path. Labelling that is the caller's
/// decision; [`FunctionNames`](super::FunctionNames) writes the runtime address
/// instead.
pub fn image_labels<'p>(paths: &[&'p str]) -> Vec<&'p str> {
    fn file_name(path: &str) -> &str {
        path.rsplit(['/', '\\']).next().unwrap_or(path)
    }

    // Each file name, and the first path seen with it. A file name met again
    // with a *different* path is shared, and both keep their paths.
    let mut first_path: HashMap<&str, &str> = HashMap::new();
    let mut shared: HashSet<&str> = HashSet::new();
    for &path in paths {
        let name = file_name(path);
        match first_path.get(name) {
            Some(&seen) if seen != path => {
                shared.insert(name);
            }
            Some(_) => {}
            None => {
                first_path.insert(name, path);
            }
        }
    }

    paths
        .iter()
        .map(|&path| {
            let name = file_name(path);
            if name.is_empty() || shared.contains(name) {
                path
            } else {
                name
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_unique_file_name_is_the_label() {
        assert_eq!(
            image_labels(&["/usr/lib/libc.so.6", "/bin/program"]),
            ["libc.so.6", "program"]
        );
    }

    /// Both separators, because a Windows profile is rendered wherever it is
    /// read.
    #[test]
    fn a_windows_path_is_cut_at_its_own_separator() {
        assert_eq!(
            image_labels(&[r"C:\Windows\System32\ntdll.dll"]),
            ["ntdll.dll"]
        );
    }

    #[test]
    fn a_shared_file_name_keeps_both_paths_whole() {
        assert_eq!(
            image_labels(&["/a/libfoo.so", "/b/libfoo.so", "/c/libbar.so"]),
            ["/a/libfoo.so", "/b/libfoo.so", "libbar.so"]
        );
    }

    /// One image listed twice is not two images, and is no reason to fall back
    /// to the path.
    #[test]
    fn the_same_path_twice_is_still_shortened() {
        assert_eq!(image_labels(&["/lib/a.so", "/lib/a.so"]), ["a.so", "a.so"]);
    }

    /// The case the distinctness argument has to cover explicitly: a path that
    /// is nothing but a file name, equal to the label another path would get.
    #[test]
    fn a_bare_file_name_cannot_collide_with_a_shortened_path() {
        let labels = image_labels(&["/x/libfoo.so", "libfoo.so"]);
        assert_ne!(labels[0], labels[1], "{labels:?}");
        assert_eq!(labels, ["/x/libfoo.so", "libfoo.so"]);
    }

    #[test]
    fn a_path_with_no_file_name_is_given_back_whole() {
        assert_eq!(image_labels(&["", "/x/dir/"]), ["", "/x/dir/"]);
    }
}
