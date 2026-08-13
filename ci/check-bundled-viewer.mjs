// Runs the bundled viewer's own logic against a profile it carries.
//
// The page is one file with no build step, so there is no module to import and
// no test runner to point at it. What there is instead is a deliberate seam:
// everything in `viewer.html` above `HEAPSCOPE` is pure, touches no DOM, and is
// assigned to `module.exports` when a `module` happens to exist. This script is
// what makes one exist. The rendering half checks `typeof document`, finds none
// here, and does not run.
//
// So this covers the arithmetic — tree building, folding, ordering, the
// warnings a profile raises about itself — without a browser, a DOM shim, or a
// dependency. What it cannot cover is that the page *looks* right, which is a
// human's job and is why `Snapshot::save_html` writes something you can
// double-click.
//
// Usage: node ci/check-bundled-viewer.mjs <page.html> [...]

import { readFileSync } from "node:fs";
import vm from "node:vm";

let failures = 0;

function check(condition, description) {
  if (!condition) {
    console.error(`  FAIL ${description}`);
    failures += 1;
  }
}

/** The two data blocks and the viewer script, lifted back out of the page. */
function dismantle(page) {
  const data = {};
  const dataPattern = /<script type="application\/json" id="heapscope-([a-z]+)">([\s\S]*?)<\/script>/g;
  let match;
  while ((match = dataPattern.exec(page)) !== null) {
    data[match[1]] = match[2];
  }
  const scriptPattern = /<script>\n([\s\S]*?)\n<\/script>/;
  const script = scriptPattern.exec(page);
  return { data, script: script && script[1] };
}

/** Depth of the deepest node, counting the root as zero. */
function depth(node) {
  let deepest = 0;
  for (const child of node.children.values()) {
    deepest = Math.max(deepest, 1 + depth(child));
  }
  return deepest;
}

function everyNode(node, visit) {
  for (const child of node.children.values()) {
    visit(child, node);
    everyNode(child, visit);
  }
}

function checkPage(path) {
  console.log(`checking ${path}`);
  const page = readFileSync(path, "utf8");
  const { data, script } = dismantle(page);

  check(script !== null && script !== undefined, "the page carries a viewer script");
  check(data.profile !== undefined, "the page carries a profile");
  check(data.display !== undefined, "the page carries a display sidecar");
  if (!script || !data.profile || !data.display) return;

  // The escape that keeps a symbol or a path from ending the script element.
  // Rust generic names contain `<` constantly, so this is not a rare path: a
  // profile of almost any Rust program exercises it thousands of times.
  check(!data.profile.includes("<"), "no raw < survives in the embedded profile");
  check(!data.display.includes("<"), "no raw < survives in the sidecar");

  const profile = JSON.parse(data.profile);
  const display = JSON.parse(data.display);

  const context = { module: { exports: {} }, console };
  vm.createContext(context);
  vm.runInContext(script, context, { filename: path });
  const HEAPSCOPE = context.module.exports;

  check(typeof HEAPSCOPE.buildTree === "function", "the pure half is exported");
  if (typeof HEAPSCOPE.buildTree !== "function") return;

  check(profile.formatVersion === 1, "the profile is a version this viewer knows");
  check(
    display.names.length === profile.frames.length,
    `a name per frame (${display.names.length} names, ${profile.frames.length} frames)`
  );
  check(
    display.keep.length === profile.points.length,
    `a kept range per point (${display.keep.length} ranges, ${profile.points.length} points)`
  );
  display.keep.forEach(function (range, at) {
    const frames = profile.points[at].frames || [];
    const usable = range[0] >= 0 && range[1] <= frames.length && (frames.length === 0 || range[0] < range[1]);
    check(usable, `point ${at}'s kept range ${JSON.stringify(range)} fits its ${frames.length} frames`);
  });

  // The tree accounts for every byte the points recorded: this is arithmetic
  // over a partition, so anything but equality is a bug in the fold.
  const tree = HEAPSCOPE.buildTree(profile, display, {});
  const expected = profile.points.reduce(function (sum, point) { return sum + (point.totalBytes || 0); }, 0);
  check(
    tree.totals.totalBytes === expected,
    `the tree totals ${tree.totals.totalBytes}, the points total ${expected}`
  );
  const expectedBlocks = profile.points.reduce(function (sum, point) { return sum + (point.totalBlocks || 0); }, 0);
  check(tree.totals.totalBlocks === expectedBlocks, "block counts agree too");

  // The absence the page reads to decide whether this run has live blocks at
  // all. It is a data check rather than a rendering one because the rendering
  // half needs a `document` and this harness has none — but it is the load
  // bearing half: the page asks `totals.maxBytes !== undefined`, so a writer
  // that started emitting zeroes here would put `peak 0 B` back in the header
  // of every ad hoc profile with nothing to notice. That is the state this
  // check was written after finding.
  const lifetimes = profile.run.mode === "heap";
  for (const key of ["maxBytes", "maxBlocks", "currBytes", "currBlocks"]) {
    check(
      (profile.totals[key] !== undefined) === lifetimes,
      `a ${profile.run.mode} run ${lifetimes ? "reports" : "omits"} totals.${key}`
    );
  }

  check(depth(tree) >= 1, "the tree has at least one level");
  everyNode(tree, function (child, parent) {
    check(
      child.totals.totalBytes <= parent.totals.totalBytes,
      `a child (${child.totals.totalBytes}) never exceeds its parent (${parent.totals.totalBytes})`
    );
    check(Number.isFinite(child.totals.totalBytes), "every node's total is a number");
    check(typeof child.name === "string" && child.name.length > 0, "every node is named");
  });

  // Showing the trimmed frames can only add levels, never remove them.
  const untrimmed = HEAPSCOPE.buildTree(profile, display, { untrimmed: true });
  check(depth(untrimmed) >= depth(tree), "untrimmed stacks are at least as deep as trimmed ones");
  check(
    untrimmed.totals.totalBytes === tree.totals.totalBytes,
    "trimming changes what is shown, never what is counted"
  );

  for (const metric of HEAPSCOPE.METRICS) {
    const sorted = HEAPSCOPE.sortTree(HEAPSCOPE.buildTree(profile, display, {}), metric.key);
    let ordered = true;
    everyNode(sorted, function (node) {
      for (let at = 1; at < node.sorted.length; at += 1) {
        if (node.sorted[at - 1].totals[metric.key] < node.sorted[at].totals[metric.key]) ordered = false;
      }
    });
    for (let at = 1; at < sorted.sorted.length; at += 1) {
      if (sorted.sorted[at - 1].totals[metric.key] < sorted.sorted[at].totals[metric.key]) ordered = false;
    }
    check(ordered, `sorting by ${metric.key} puts the heaviest first at every level`);
  }

  // The fold's subtlest rule, checked against the closed form rather than
  // against itself.
  //
  // Maxima do not add. Two points that each peaked at 4 MB, at different
  // moments, did not jointly peak at 8 MB, and a viewer that says so has
  // invented a number the run does not support. What *is* provable about a
  // subtree is the largest of three things: the biggest peak any one point
  // below it reached, and the sums at t-gmax and at t-end, where everything
  // below it demonstrably was live at the same instant.
  //
  // This recomputes that from the points rather than from the tree, so a fold
  // that quietly starts adding is caught by arithmetic that never added.
  const attributed = new Map();
  profile.points.forEach(function (point, index) {
    const frames = point.frames || [];
    const range = display.keep[index] || [0, frames.length];
    const shown = frames.slice(range[0], range[1]);
    const short = HEAPSCOPE.shortenNames(profile, display.names);
    const path = shown.length > 0
      ? shown.map(function (frame) { return short[frame] || "0x?"; }).reverse()
      : [point.kind === "overflow" ? display.labels.overflow : display.labels.unwalkable];

    // The root first: it is the node every point reaches, so it is where a
    // fold that adds instead of bounding shows up. Every other node on a
    // chain has one point beneath it, where the two rules agree.
    let node = tree;
    for (let step = 0; step <= path.length; step += 1) {
      if (step > 0) {
        node = node.children.get(path[step - 1]);
        if (!node) return;
      }
      let seen = attributed.get(node);
      if (!seen) {
        seen = { peak: 0, atGmax: 0, atEnd: 0 };
        attributed.set(node, seen);
      }
      seen.peak = Math.max(seen.peak, point.maxBytes || 0);
      seen.atGmax += point.atGmaxBytes || 0;
      seen.atEnd += point.atEndBytes || 0;
    }
  });
  let unattributed = 0;
  function checkPeak(node, what) {
    const seen = attributed.get(node);
    if (!seen) {
      unattributed += 1;
      return;
    }
    const bound = Math.max(seen.peak, seen.atGmax, seen.atEnd);
    check(
      node.totals.maxBytes === bound,
      `${what} peak is bounded, not summed (${node.totals.maxBytes} against ${bound})`
    );
  }
  checkPeak(tree, "the whole tree's");
  everyNode(tree, function (node) { checkPeak(node, "a subtree's"); });
  check(unattributed === 0, `every node came from a point (${unattributed} did not)`);

  const warnings = HEAPSCOPE.warnings(profile);
  check(Array.isArray(warnings), "the profile's warnings are a list");
  warnings.forEach(function (warning) {
    check(typeof warning.title === "string" && warning.title.length > 0, "each warning has a title");
    check(typeof warning.body === "string" && warning.body.length > 0, "each warning has a body");
  });

  const sampled = Boolean(profile.settings && profile.settings.samplingInterval);
  check(
    sampled === warnings.some(function (warning) { return warning.title.startsWith("Sampled run"); }),
    "a sampled run says so, and an unsampled one does not"
  );

  // Shortening a label is exact: it removes the path the profile itself says
  // the frame's module was loaded from, and leaves every other label alone.
  const short = HEAPSCOPE.shortenNames(profile, display.names);
  check(short.length === display.names.length, "shortening keeps one label per frame");
  short.forEach(function (label, at) {
    const frame = profile.frames[at];
    const module = frame.module === undefined ? null : profile.modules[frame.module];
    if (module && module.path && module.path.includes("/")) {
      check(!label.includes(module.path), `frame ${at} no longer spells out its module's path`);
      const base = module.path.slice(module.path.lastIndexOf("/") + 1);
      check(label.includes(base), `frame ${at} still names the file it came from`);
    } else {
      check(label === display.names[at], `frame ${at} without a module path is untouched`);
    }
  });

  check(HEAPSCOPE.formatBytes(1024) === "1.00 KiB", "formatBytes scales");
  check(HEAPSCOPE.formatBytes(512) === "512 B", "formatBytes leaves small counts alone");
  check(HEAPSCOPE.formatCount(1234567) === "1,234,567", "formatCount groups digits");
  // Ad hoc weights are dimensionless, and calling them bytes would be a claim
  // the run does not support.
  check(HEAPSCOPE.makeAmountFormatter("ad-hoc")(4096) === "4,096", "ad hoc units are not bytes");
  check(HEAPSCOPE.makeAmountFormatter("heap")(4096) === "4.00 KiB", "heap amounts are bytes");
}

const pages = process.argv.slice(2);
if (pages.length === 0) {
  console.error("usage: node ci/check-bundled-viewer.mjs <page.html> [...]");
  process.exit(2);
}
pages.forEach(checkPage);

if (failures > 0) {
  console.error(`${failures} check(s) failed`);
  process.exit(1);
}
console.log(`ok: ${pages.length} page(s)`);
