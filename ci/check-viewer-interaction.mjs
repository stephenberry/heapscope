// Clicks the bundled viewer's controls in a real browser, and checks what they did.
//
// `ci/check-bundled-viewer.mjs` runs the page's script in a `vm` context with no
// `document`, so the whole block below `typeof document !== "undefined"` — the
// half that answers a click — is skipped there by construction. That half is
// most of the viewer: six tabs, a sort control, a trimming toggle, expand and
// collapse, and a twisty on every branch of the tree. M7 chunk L broke each of
// them in turn and watched the entire repository pass: a tab that reveals every
// panel at once, an expand that opens one level, a collapse that draws nothing,
// a twisty whose key is the node's name so two unrelated nodes open together.
// Seven mutations, no failures.
//
// So this drives the page the way a reader does. Chrome is launched headless
// with `--remote-debugging-pipe`, which speaks the DevTools protocol over a pair
// of file descriptors: no port, no WebSocket, no npm, and nothing added to the
// repository's dependency discipline (`tests/no_dependencies.rs` fails the build
// on a `package.json`). Buttons are pressed with `Input.dispatchMouseEvent` at
// the coordinates the control actually occupies, so the browser's own hit
// testing decides what was clicked — a control hidden behind another element
// fails here, which no synthetic `element.click()` could tell you.
//
// The oracle is the pure half. Every claim below has the form "what the page
// displays after this interaction is what `buildTree`/`sortTree` say it should
// be", and that arithmetic is checked independently, against closed forms, by
// `ci/check-bundled-viewer.mjs`. Neither harness is complete alone: one knows
// the numbers are right and cannot see the page, the other knows the page shows
// what the numbers say and takes the numbers on trust.
//
// What this deliberately does not claim to cover: that the page *looks* right.
// Layout, colour and legibility are a human's job, and `Snapshot::save_html`
// writes something you can double-click for exactly that reason.
//
// Usage: node ci/check-viewer-interaction.mjs <page.html> [...]
//
// Exit codes:
//   0  every control did what the page's own data says it should
//   1  a check failed
//   2  the check could not be run (no Chrome)

import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

let failures = 0;
let checks = 0;

function check(condition, description) {
  checks += 1;
  if (!condition) {
    console.error(`  FAIL ${description}`);
    failures += 1;
  }
}

/** Chrome, wherever this platform keeps it. */
function findChrome() {
  const named = process.env.CHROME || process.env.CHROME_PATH;
  if (named) return existsSync(named) ? named : null;
  const candidates = {
    darwin: [
      "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
      "/Applications/Chromium.app/Contents/MacOS/Chromium",
      "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
    ],
    win32: [
      "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
      "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
    ],
    linux: [
      "/usr/bin/google-chrome",
      "/usr/bin/google-chrome-stable",
      "/usr/bin/chromium",
      "/usr/bin/chromium-browser",
      "/snap/bin/chromium",
    ],
  }[process.platform] || [];
  return candidates.find(existsSync) || null;
}

/**
 * A DevTools client over Chrome's pipe transport.
 *
 * Messages are NUL-terminated JSON in both directions, on descriptors 3 and 4.
 * That is the whole protocol framing, which is why this needs no library.
 */
class Browser {
  constructor(binary, profileDirectory) {
    this.child = spawn(binary, [
      "--headless=new",
      "--remote-debugging-pipe",
      "--disable-gpu",
      "--no-sandbox",
      "--no-first-run",
      "--no-default-browser-check",
      "--disable-extensions",
      // A container's /dev/shm is 64 MB by default, which Chrome outgrows and
      // then dies in a way that reads as "the viewer is broken".
      "--disable-dev-shm-usage",
      "--disable-background-networking",
      "--disable-component-update",
      // Large enough that every control is inside the viewport without
      // scrolling, because a click is dispatched at viewport coordinates.
      "--window-size=1600,1400",
      "--user-data-dir=" + profileDirectory,
      "about:blank",
    ], { stdio: ["ignore", "ignore", "pipe", "pipe", "pipe"] });

    this.stderr = "";
    this.child.stderr.on("data", (chunk) => { this.stderr += chunk.toString("utf8"); });
    this.writable = this.child.stdio[3];
    this.readable = this.child.stdio[4];
    this.pending = "";
    this.nextId = 1;
    this.waiting = new Map();
    this.readable.on("data", (chunk) => this.receive(chunk));
  }

  receive(chunk) {
    this.pending += chunk.toString("utf8");
    let at;
    while ((at = this.pending.indexOf("\0")) !== -1) {
      const raw = this.pending.slice(0, at);
      this.pending = this.pending.slice(at + 1);
      const message = JSON.parse(raw);
      const settle = message.id && this.waiting.get(message.id);
      if (settle) {
        this.waiting.delete(message.id);
        settle(message);
      }
    }
  }

  async send(method, params, sessionId) {
    const id = this.nextId++;
    const payload = { id, method, params: params || {} };
    if (sessionId) payload.sessionId = sessionId;
    this.writable.write(JSON.stringify(payload) + "\0");
    const message = await new Promise((resolve) => this.waiting.set(id, resolve));
    if (message.error) {
      throw new Error(`${method} failed: ${message.error.message}`);
    }
    return message.result;
  }

  async open(url) {
    const created = await this.send("Target.createTarget", { url });
    const attached = await this.send("Target.attachToTarget", {
      targetId: created.targetId,
      flatten: true,
    });
    return new Page(this, created.targetId, attached.sessionId);
  }

  /**
   * Waits for the browser to actually be gone, rather than only asking it.
   *
   * `kill()` sends a signal and returns; Chrome then takes its time flushing
   * the profile directory it was given. Removing that directory in the
   * meantime raced it, and on a Linux runner the race was lost: `ENOTEMPTY` on
   * `profile/Default`, thrown out of the `finally` block **after** every check
   * had already passed, so the job went red on its own cleanup.
   */
  close() {
    return new Promise((resolve) => {
      if (this.child.exitCode !== null || this.child.signalCode !== null) {
        resolve();
        return;
      }
      this.child.once("exit", resolve);
      this.child.kill();
    });
  }
}

/** One open page, and the two things a reader does to it: look, and click. */
class Page {
  constructor(browser, targetId, sessionId) {
    this.browser = browser;
    this.targetId = targetId;
    this.sessionId = sessionId;
  }

  /**
   * Evaluate an expression in the page and return its value.
   *
   * A thrown exception is re-thrown here rather than reported as a value. That
   * matters more than it looks: an oracle that silently evaluated to `undefined`
   * because a selector went stale would compare `undefined` against `undefined`
   * and pass, which is the failure mode this whole file exists to prevent.
   */
  async evaluate(expression) {
    const outcome = await this.browser.send("Runtime.evaluate", {
      expression,
      returnByValue: true,
      awaitPromise: true,
    }, this.sessionId);
    if (outcome.exceptionDetails) {
      const thrown = outcome.exceptionDetails.exception;
      throw new Error((thrown && thrown.description) || outcome.exceptionDetails.text);
    }
    return outcome.result.value;
  }

  /** Wait until the page's own script has run and drawn something. */
  async settle() {
    for (let attempt = 0; attempt < 100; attempt += 1) {
      const state = await this.evaluate(
        'document.readyState === "complete" && !!document.getElementById("tabs")'
      );
      if (state) return true;
      await new Promise((resume) => setTimeout(resume, 50));
    }
    return false;
  }

  /**
   * Click what `expression` evaluates to, at the coordinates it occupies.
   *
   * Returns what the browser says is at that point. `hit` is false when the
   * control is covered, transparent to pointer events, or scrolled out of the
   * viewport — three ways for a control to be present in the DOM and unusable,
   * none of which a synthetic `element.click()` can distinguish from working.
   */
  async click(expression) {
    const spot = await this.evaluate(`(function () {
      const element = (${expression});
      if (!element) return null;
      element.scrollIntoView({ block: "center", inline: "center" });
      const box = element.getBoundingClientRect();
      if (box.width === 0 || box.height === 0) return { hit: false, reason: "zero-sized" };
      const x = box.left + box.width / 2;
      const y = box.top + box.height / 2;
      const at = document.elementFromPoint(x, y);
      return {
        x, y,
        hit: at === element || element.contains(at),
        reason: at ? at.tagName + (at.className ? "." + at.className : "") : "nothing",
      };
    })()`);
    if (!spot || !spot.hit) {
      return { hit: false, reason: (spot && spot.reason) || "no such element" };
    }
    for (const type of ["mousePressed", "mouseReleased"]) {
      await this.browser.send("Input.dispatchMouseEvent", {
        type,
        x: spot.x,
        y: spot.y,
        button: "left",
        buttons: type === "mousePressed" ? 1 : 0,
        clickCount: 1,
      }, this.sessionId);
    }
    return { hit: true };
  }

  async close() {
    await this.browser.send("Target.closeTarget", { targetId: this.targetId });
  }
}

/**
 * Installed in the page once, and used by every check below.
 *
 * `HEAPSCOPE` is a top-level `const` in a classic script, so it is in the global
 * lexical scope and a later evaluation can see it. `profile` and `display` are
 * not — they are private to the rendering closure — so they are read back out of
 * the page the same way the page read them.
 */
const PROBE = `
window.__probe = (function () {
  const profile = JSON.parse(document.getElementById("heapscope-profile").textContent);
  const display = JSON.parse(document.getElementById("heapscope-display").textContent);

  /** The rows the page is showing, in the order it is showing them. */
  function rows() {
    return Array.from(document.querySelectorAll("#tree .node")).map(function (row) {
      const label = row.querySelector(".label");
      // The page writes paddingLeft as indent * 1.1 + 0.4 rem, so the depth a
      // reader sees is recoverable from what the page actually set.
      const padding = parseFloat(row.style.paddingLeft);
      return {
        name: label.querySelector(".text").textContent,
        depth: Math.round((padding - 0.4) / 1.1),
        glyph: row.querySelector(".twisty").textContent,
        amount: row.querySelector(".metric b").textContent,
        title: label.title,
      };
    });
  }

  /** What the pure half says a fully open tree looks like. */
  function model(metric, untrimmed) {
    const tree = HEAPSCOPE.sortTree(
      HEAPSCOPE.buildTree(profile, display, { untrimmed: !!untrimmed }),
      metric
    );
    const flat = [];
    (function walk(node, depth) {
      node.sorted.forEach(function (child) {
        flat.push({
          name: child.name,
          depth: depth,
          leaf: child.sorted.length === 0,
          value: child.totals[metric],
        });
        walk(child, depth + 1);
      });
    })(tree, 0);
    return flat;
  }

  function amountText(metric, value) {
    const chosen = HEAPSCOPE.METRICS.find(function (entry) { return entry.key === metric; });
    const amount = HEAPSCOPE.makeAmountFormatter(profile.run && profile.run.mode);
    return chosen && chosen.amount ? amount(value) : HEAPSCOPE.formatCount(value);
  }

  /** Which panel is on screen, and which tab says it is. */
  function tabState() {
    const panels = Array.from(document.querySelectorAll("main section")).map(function (panel) {
      return {
        id: panel.id,
        shown: !panel.hidden,
        elements: panel.querySelectorAll("*").length,
        text: panel.textContent.replace(/\\s+/g, " ").trim().length,
      };
    });
    const buttons = Array.from(document.getElementById("tabs").children).map(function (button) {
      return { label: button.textContent, selected: button.getAttribute("aria-selected") };
    });
    return { panels: panels, buttons: buttons };
  }

  return {
    mode: profile.run && profile.run.mode,
    rows: rows,
    model: model,
    amountText: amountText,
    tabState: tabState,
    metrics: HEAPSCOPE.METRICS.map(function (metric) { return metric.key; }),
    options: function () {
      return Array.from(document.getElementById("metric").options).map(function (o) { return o.value; });
    },
    /**
     * The first branch row whose name is also a branch somewhere else, or -1.
     *
     * This is the row the twisty check must click. Closing a node whose name
     * occurs once behaves identically whether the open set is keyed by path or
     * by name, so clicking such a row proves nothing -- and the row the check
     * used to take, the first branch, is the outermost frame every stack
     * shares and therefore occurs exactly once by construction. Measured on a
     * real macOS profile: 23 rows, 14 branches, the clicked row unique, and
     * the only two rows that could have caught the defect at indices 17 and 21.
     * The check passed anyway, on the platform where it was believed to work.
     *
     * Both occurrences have to be branches. A leaf twin has no twisty and no
     * descendants to hide, so a name-keyed open set changes nothing on screen
     * and the check is vacuous again one step further in.
     */
    duplicateBranch: function (metric) {
      const flat = model(metric, false);
      const branches = new Map();
      flat.forEach(function (row) {
        if (!row.leaf) branches.set(row.name, (branches.get(row.name) || 0) + 1);
      });
      return flat.findIndex(function (row) {
        return !row.leaf && branches.get(row.name) > 1;
      });
    },
  };
})();
true`;

/** Same rows, same order, same depths: what the page shows against what it means. */
function sameRows(shown, expected, what) {
  check(
    shown.length === expected.length,
    `${what}: ${shown.length} rows shown, the tree has ${expected.length}`
  );
  const upto = Math.min(shown.length, expected.length);
  let mismatch = -1;
  for (let at = 0; at < upto; at += 1) {
    if (shown[at].name !== expected[at].name || shown[at].depth !== expected[at].depth) {
      mismatch = at;
      break;
    }
  }
  check(
    mismatch === -1,
    mismatch === -1 ? what : `${what}: row ${mismatch} shows ` +
      `${JSON.stringify(shown[mismatch].name)} at depth ${shown[mismatch].depth}, ` +
      `the tree says ${JSON.stringify(expected[mismatch].name)} at depth ${expected[mismatch].depth}`
  );
}

async function checkPage(browser, path) {
  const url = pathToFileURL(resolve(path)).href;
  console.log(`clicking through ${path}`);
  const page = await browser.open(url);
  try {
    check(await page.settle(), "the page finishes loading and draws its tabs");
    await page.evaluate(PROBE);
    const mode = await page.evaluate("window.__probe.mode");

    // ---- the state a reader arrives in ----

    const arrival = await page.evaluate("window.__probe.tabState()");
    check(arrival.buttons.length === 6, `six tabs (${arrival.buttons.length})`);
    check(
      arrival.panels.filter((panel) => panel.shown).length === 1 &&
        arrival.panels.find((panel) => panel.shown).id === "panel-points",
      "one panel is on screen on arrival, and it is the program points"
    );
    check(
      arrival.buttons.filter((button) => button.selected === "true").length === 1 &&
        arrival.buttons[0].selected === "true",
      "the first tab is the one marked selected on arrival"
    );

    // The heaviest path is open on arrival, which is the page's own promise
    // about what a reader sees before touching anything.
    const arrived = await page.evaluate("window.__probe.rows()");
    check(arrived.length > 1, `the tree arrives partly open (${arrived.length} rows)`);

    // ---- tabs ----

    for (let index = 0; index < arrival.buttons.length; index += 1) {
      const label = arrival.buttons[index].label;
      const landed = await page.click(`document.getElementById("tabs").children[${index}]`);
      check(landed.hit, `the ${label} tab is clickable where it is drawn (${landed.reason || ""})`);
      const state = await page.evaluate("window.__probe.tabState()");
      const shown = state.panels.filter((panel) => panel.shown);
      check(
        shown.length === 1 && shown[0].id === state.panels[index].id,
        `clicking ${label} shows exactly its own panel ` +
          `(${shown.length} shown: ${shown.map((panel) => panel.id).join(", ")})`
      );
      const selected = state.buttons.filter((button) => button.selected === "true");
      check(
        selected.length === 1 && state.buttons[index].selected === "true",
        `clicking ${label} moves the selection to it (${selected.length} marked selected)`
      );
      check(shown.length === 1 && shown[0].elements > 0, `the ${label} panel has something in it`);
    }

    // Back to the tree for the rest.
    await page.click('document.getElementById("tabs").children[0]');

    // ---- expand, collapse, and the twisty ----

    const metric = (await page.evaluate("window.__probe.options()"))[0];

    const collapsed = await page.click('document.getElementById("collapse")');
    check(collapsed.hit, `Collapse all is clickable where it is drawn (${collapsed.reason || ""})`);
    const afterCollapse = await page.evaluate("window.__probe.rows()");
    const topLevel = (await page.evaluate(`window.__probe.model(${JSON.stringify(metric)}, false)`))
      .filter((row) => row.depth === 0);
    sameRows(afterCollapse, topLevel, "collapse all leaves exactly the top level");
    check(
      afterCollapse.every((row) => row.glyph !== "\u25be"),
      "no row is still drawn open after collapse all"
    );

    const expanded = await page.click('document.getElementById("expand")');
    check(expanded.hit, `Expand all is clickable where it is drawn (${expanded.reason || ""})`);
    const afterExpand = await page.evaluate("window.__probe.rows()");
    const whole = await page.evaluate(`window.__probe.model(${JSON.stringify(metric)}, false)`);
    sameRows(afterExpand, whole, "expand all shows every node in the tree");
    check(
      afterExpand.every((row, at) => (whole[at] && whole[at].leaf ? row.glyph === "\u00b7" : row.glyph === "\u25be")),
      "every branch is drawn open and every leaf is drawn as a leaf"
    );

    // Every row's number is the number the tree holds for that node.
    const expectedAmounts = await page.evaluate(
      `window.__probe.model(${JSON.stringify(metric)}, false).map(function (row) {
         return window.__probe.amountText(${JSON.stringify(metric)}, row.value);
       })`
    );
    const wrongAmount = afterExpand.findIndex((row, at) => row.amount !== expectedAmounts[at]);
    check(
      wrongAmount === -1,
      wrongAmount === -1
        ? "every row shows the amount its node holds"
        : `row ${wrongAmount} (${afterExpand[wrongAmount].name}) shows ` +
          `${afterExpand[wrongAmount].amount}, its node holds ${expectedAmounts[wrongAmount]}`
    );

    // One twisty, from a fully open tree. A defect that keys the open set by
    // the node's name instead of its path opens or closes two unrelated nodes
    // at once, so the invariant is not "the subtree disappeared" but "exactly
    // one row's state changed".
    const branch = await page.evaluate(`window.__probe.duplicateBranch(${JSON.stringify(metric)})`);
    check(
      branch !== -1,
      "the tree has a branch whose name is a branch elsewhere, so a per-name " +
        "open set is distinguishable by closing it"
    );
    if (branch !== -1) {
      check(
        afterExpand[branch] && afterExpand[branch].glyph === "\u25be",
        `the row the model chose (${branch}) is drawn as an open branch`
      );
      const closed = await page.click(
        `document.querySelectorAll("#tree .node")[${branch}].querySelector(".twisty")`
      );
      check(closed.hit, `a twisty is clickable where it is drawn (${closed.reason || ""})`);
      const afterClose = await page.evaluate("window.__probe.rows()");
      const subtree = whole.slice(branch + 1).findIndex((row) => row.depth <= whole[branch].depth);
      const hidden = subtree === -1 ? whole.length - branch - 1 : subtree;
      check(
        afterExpand.length - afterClose.length === hidden,
        `closing ${afterExpand[branch].name} hides its ${hidden} descendants and nothing else ` +
          `(${afterExpand.length - afterClose.length} rows went away)`
      );
      check(
        afterClose[branch] && afterClose[branch].glyph === "\u25b8",
        "the row that was clicked is the row that closed"
      );
      // Everything outside that node's subtree is untouched \u2014 same rows, same
      // order, same open state. This is where a per-name open set is caught:
      // it closes every node sharing the clicked node's name, and those rows
      // are somewhere else in this list.
      const disturbed = [];
      for (let at = 0; at < branch; at += 1) {
        const after = afterClose[at];
        if (!after || after.name !== afterExpand[at].name || after.glyph !== afterExpand[at].glyph) {
          disturbed.push(afterExpand[at].name);
        }
      }
      for (let at = branch + 1 + hidden; at < afterExpand.length; at += 1) {
        const after = afterClose[at - hidden];
        if (!after || after.name !== afterExpand[at].name || after.glyph !== afterExpand[at].glyph) {
          disturbed.push(afterExpand[at].name);
        }
      }
      check(
        disturbed.length === 0,
        `closing one branch left every other row alone (${disturbed.length} changed: ` +
          `${disturbed.slice(0, 3).join(", ")})`
      );
      // Re-opening restores exactly what was there.
      await page.click(`document.querySelectorAll("#tree .node")[${branch}].querySelector(".twisty")`);
      const reopened = await page.evaluate("window.__probe.rows()");
      sameRows(reopened, whole, "re-opening the same branch restores the tree");
    }

    // ---- sorting ----

    const options = await page.evaluate("window.__probe.options()");
    const lifetimes = mode === "heap";
    check(
      options.length === (lifetimes ? 6 : 2),
      `a ${mode} run offers ${lifetimes ? 6 : 2} sort orders (${options.length}: ${options.join(", ")})`
    );
    const reachable = await page.evaluate(`(function () {
      const select = document.getElementById("metric");
      const box = select.getBoundingClientRect();
      const at = document.elementFromPoint(box.left + box.width / 2, box.top + box.height / 2);
      return at === select || select.contains(at);
    })()`);
    check(reachable, "the sort control is where it is drawn");
    for (const option of options) {
      // The dropdown itself belongs to the browser, not to the page, and cannot
      // be operated headlessly. What the page owns is the change handler, so
      // that is what is exercised: a real change event on a real selection.
      await page.evaluate(`(function () {
        const select = document.getElementById("metric");
        select.focus();
        select.value = ${JSON.stringify(option)};
        select.dispatchEvent(new Event("change", { bubbles: true }));
      })()`);
      const sorted = await page.evaluate("window.__probe.rows()");
      const expected = await page.evaluate(`window.__probe.model(${JSON.stringify(option)}, false)`);
      sameRows(sorted, expected, `sorting by ${option} reorders the tree`);
    }
    await page.evaluate(`(function () {
      const select = document.getElementById("metric");
      select.value = ${JSON.stringify(metric)};
      select.dispatchEvent(new Event("change", { bubbles: true }));
    })()`);

    // ---- trimmed frames ----

    const toggled = await page.click('document.getElementById("untrimmed")');
    check(toggled.hit, `the trimming checkbox is clickable where it is drawn (${toggled.reason || ""})`);
    check(await page.evaluate('document.getElementById("untrimmed").checked'), "clicking it ticks it");
    await page.click('document.getElementById("expand")');
    const untrimmedShown = await page.evaluate("window.__probe.rows()");
    const untrimmedModel = await page.evaluate(`window.__probe.model(${JSON.stringify(metric)}, true)`);
    sameRows(untrimmedShown, untrimmedModel, "showing trimmed frames shows the untrimmed tree");
    check(
      untrimmedShown.length >= afterExpand.length,
      `the untrimmed tree is at least as large (${untrimmedShown.length} against ${afterExpand.length})`
    );
    await page.click('document.getElementById("untrimmed")');
    await page.click('document.getElementById("expand")');
    sameRows(
      await page.evaluate("window.__probe.rows()"),
      whole,
      "unticking it puts the trimmed tree back"
    );

    return { mode, rows: whole.length };
  } finally {
    await page.close();
  }
}

/**
 * A profile from the future must be refused, not guessed at.
 *
 * Every writer in this crate stamps `formatVersion`, and the page carries the
 * one reader guaranteed to be looking at it. The refusal is drawn rather than
 * computed — it hides the tabs and the whole of `main` — so it lives here and
 * nowhere else.
 */
async function checkVersionRefusal(browser, path, directory) {
  console.log("checking that an unknown format is refused rather than guessed at");
  const page = readFileSync(path, "utf8");
  const patched = page.replace('"formatVersion":1', '"formatVersion":9');
  if (patched === page) {
    check(false, "the page carries a formatVersion to patch");
    return;
  }
  const forged = join(directory, "from-the-future.html");
  writeFileSync(forged, patched);
  const open = await browser.open(pathToFileURL(forged).href);
  try {
    check(await open.settle(), "the refusing page still loads");
    const state = await open.evaluate(`(function () {
      return {
        banners: document.getElementById("banners").textContent,
        tabsHidden: document.getElementById("tabs").hidden,
        mainHidden: document.querySelector("main").hidden,
        rows: document.querySelectorAll("#tree .node").length,
      };
    })()`);
    check(state.tabsHidden, "an unknown format hides the tabs");
    check(state.mainHidden, "an unknown format hides everything below them");
    check(state.rows === 0, `an unknown format draws no tree (${state.rows} rows)`);
    check(
      state.banners.includes("formatVersion 9"),
      "the banner says which version it was handed"
    );
  } finally {
    await open.close();
  }
}

const pages = process.argv.slice(2);
if (pages.length === 0) {
  console.error("usage: node ci/check-viewer-interaction.mjs <page.html> [...]");
  process.exit(2);
}

const binary = findChrome();
if (!binary) {
  console.error("skip: no Chrome or Chromium found, so nothing can click the viewer");
  process.exit(2);
}
console.log(`driving ${binary}`);

const directory = mkdtempSync(join(tmpdir(), "heapscope-viewer-"));
const browser = new Browser(binary, join(directory, "profile"));
let summaries = [];
try {
  for (const path of pages) {
    summaries.push(await checkPage(browser, path));
  }
  await checkVersionRefusal(browser, pages[0], directory);
} catch (failure) {
  console.error(`  FAIL ${failure.message}`);
  failures += 1;
} finally {
  await browser.close();
  // Belt and braces behind the awaited exit: `maxRetries` is node's own answer
  // to exactly this class (`ENOTEMPTY`, `EBUSY`, `EPERM`), and Chrome can leave
  // a helper process writing for a moment after the one we spawned is gone.
  //
  // And a failure here is reported rather than thrown. By this point every
  // check has run and been counted, so the exit code below is the verdict on
  // the viewer; a temporary directory that would not delete says nothing about
  // it, and turning that into a red build is how a green suite gets ignored.
  try {
    rmSync(directory, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
  } catch (cleanup) {
    console.error(`  note: could not remove ${directory}: ${cleanup.message}`);
  }
}

// The guard this replaces asked whether a duplicate name existed *anywhere*,
// which is not the precondition the twisty check needs -- it needs the row it
// clicks to be one. `duplicateBranch` is that guard now, per page and per
// metric, and it fails naming what is missing rather than counting.

if (failures > 0) {
  console.error(`${failures} of ${checks} check(s) failed`);
  process.exit(1);
}
console.log(`ok: ${checks} checks over ${pages.length} page(s)`);
