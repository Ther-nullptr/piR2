/* Exercise the real view/controller with controlled network completion order. */
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const D = require("./replay-dynamic.js");
const Store = require("./replay-dynamic-store.js");
const T = require("./replay-timeline.js");

function element(id = "") {
  let html = "",
    children = new Map();
  const listeners = new Map();
  return {
    id,
    dataset: {},
    style: { setProperty() {} },
    value: "",
    hidden: false,
    textContent: "",
    disabled: false,
    set innerHTML(value) {
      html = value;
      children = new Map();
      const first = value.match(/<option value="([^"]*)"/);
      if (first)
        this.value = first[1]
          .replaceAll("&quot;", '"')
          .replaceAll("&amp;", "&");
    },
    get innerHTML() {
      return html;
    },
    setAttribute() {},
    closest() {
      return element();
    },
    querySelector(selector) {
      if (!children.has(selector)) children.set(selector, element());
      return children.get(selector);
    },
    querySelectorAll(selector) {
      if (!children.has(selector)) {
        const pattern =
          selector === ".precision-card"
            ? /<article\b/g
            : selector === ".dynamic-run"
              ? /<select class="dynamic-run" data-index="(\d+)"/g
              : /<button class="dynamic-compare" data-index="(\d+)"/g;
        children.set(
          selector,
          [...html.matchAll(pattern)].map((match) => {
            const child = element();
            child.dataset.index = match[1];
            return child;
          }),
        );
      }
      return children.get(selector);
    },
    addEventListener(name, fn) {
      listeners.set(name, fn);
    },
    async dispatch(name) {
      return listeners.get(name)?.();
    },
  };
}
function fixture() {
  const nodes = new Map();
  const $ = (id) => {
    if (!nodes.has(id)) nodes.set(id, element(id));
    return nodes.get(id);
  };
  $("error-banner").hidden = true;
  $("window").value = "auto";
  const runs = ["a", "b", "c", "d"].map((id, i) => ({
    schema: "pir2.replay.run",
    version: 1,
    id,
    kind: "dynamic",
    label: id,
    precision: i % 2 ? "w8a8" : "bf16",
    protocol: "default_vla_flow4",
    episodes: [0, 1].map((seed) => ({
      id: `${id}-${seed}`,
      suite: "libero_spatial",
      task_id: 0,
      level: "l1",
      traj_index: 0,
      speed_mps: 0.001,
      seed,
      init_state_id: 0,
      protocol: "default_vla_flow4",
      timing_mode: "native",
      execution_prefix_steps: 2,
      scene_drive: id === "c" ? "frozen" : "moving",
      duration_s: 1,
      status: "ok",
      events: [],
      requests: [],
      frames: [],
      media: {},
      motion_events: [],
      warnings: [],
    })),
  }));
  const catalog = {
    schema: "pir2.replay.catalog",
    version: 1,
    runs: runs.map((run) => ({
      id: run.id,
      kind: run.kind,
      label: run.label,
      precision: run.precision,
      protocol: run.protocol,
      episodes: run.episodes.length,
      episode_summaries: run.episodes.map(D.episodeSummary),
    })),
  };
  const calls = [],
    pending = [],
    disposed = [],
    presented = [],
    frames = [];
  const fetch = async (url, options = {}) => {
    calls.push(url);
    if (url.endsWith("/catalog"))
      return { ok: true, json: async () => catalog };
    const run = runs.find((r) => url.endsWith("/" + r.id));
    return new Promise((resolve, reject) =>
      pending.push({
        id: run.id,
        signal: options.signal,
        resolve: (value = run) =>
          resolve({ ok: true, json: async () => value }),
        reject,
      }),
    );
  };
  const window = {
    DynamicReplay: D,
    DynamicReplayStore: Store,
    ReplayTimeline: T,
    ReplayMedia: {
      update: (card) => presented.push(card),
      dispose: (card) => disposed.push(card),
    },
  };
  vm.runInNewContext(
    fs.readFileSync(require.resolve("./replay-dynamic-view.js"), "utf8"),
    {
      window,
      document: {
        body: { dataset: {} },
        getElementById: $,
        querySelector: () => element(),
      },
      fetch,
      requestAnimationFrame(callback) {
        frames.push(callback);
      },
      URL,
      Blob,
      setTimeout,
    },
  );
  return {
    $,
    runs,
    calls,
    pending,
    disposed,
    presented,
    frames,
    view: window.DynamicReplayView,
  };
}
const turn = () => new Promise((resolve) => setImmediate(resolve));

test("view loads only selected configurations, disposes old cards and keeps full-run links", async () => {
  const f = fixture();
  const initialized = f.view.initialize();
  await turn();
  assert.deepEqual(f.calls, [
    "/api/replay/catalog",
    "/api/replay/run/a",
    "/api/replay/run/b",
  ]);
  f.pending.forEach((request) => request.resolve());
  await initialized;
  assert.equal(f.$("error-banner").hidden, true);
  const grid = f.$("precision-cards"),
    oldCards = grid.querySelectorAll(".precision-card");
  const disposedBefore = f.disposed.length;
  assert.match(grid.innerHTML, /\/api\/replay\/run\/a/);
  const choice = grid.querySelectorAll(".dynamic-run")[0];
  choice.value = "c";
  const changed = choice.dispatch("change");
  await turn();
  assert.deepEqual(f.disposed.slice(disposedBefore), oldCards);
  assert.doesNotMatch(grid.innerHTML, /\/api\/replay\/run\/a/);
  assert.equal(f.$("inspector").textContent, "");
  assert.equal(f.pending.at(-1).id, "c");
  assert.equal(f.calls.filter((url) => url.endsWith("/b")).length, 1);
  f.pending.at(-1).resolve();
  await changed;
  assert.match(grid.innerHTML, /\/api\/replay\/run\/c/);
  assert.match(grid.innerHTML, /外驱冻结/);
  assert.doesNotMatch(f.$("sample").innerHTML, /冻结/);
});

test("late responses and abort errors cannot repaint or show an error over a new condition", async () => {
  for (const reject of [false, true]) {
    const f = fixture();
    const initialized = f.view.initialize();
    await turn();
    f.$("precision").value = "bf16";
    const changed = f.$("precision").dispatch("change");
    // Same a request is shared; b is deselected and its response arrives late.
    f.pending[0].resolve();
    await changed;
    const html = f.$("precision-cards").innerHTML;
    assert.equal(
      f.frames.length,
      1,
      "active playback must not wait for a superseded request",
    );
    assert.equal(f.pending[1].signal.aborted, true);
    if (reject)
      f.pending[1].reject(
        Object.assign(new Error("aborted"), { name: "AbortError" }),
      );
    else f.pending[1].resolve();
    await initialized;
    assert.equal(f.$("precision-cards").innerHTML, html);
    assert.equal(f.$("error-banner").hidden, true);
  }
});

test("an active run identity error is visible and keeps index choices available for recovery", async () => {
  const f = fixture();
  const initialized = f.view.initialize();
  await turn();
  f.pending[0].resolve({ ...f.runs[0], id: "wrong" });
  f.pending[1].resolve();
  await initialized;
  assert.equal(f.$("error-banner").hidden, false);
  assert.match(f.$("error-banner").textContent, /identity mismatch/);
  assert.equal(f.presented.length, 0);
  const choice = f.$("precision-cards").querySelectorAll(".dynamic-run")[0];
  assert.ok(choice, "a failed load must leave a configuration selector");
  choice.value = "c";
  const changed = choice.dispatch("change");
  await turn();
  for (const request of f.pending.slice(2)) request.resolve();
  await changed;
  assert.equal(f.$("error-banner").hidden, true);
  assert.match(f.$("precision-cards").innerHTML, /\/api\/replay\/run\/c/);
});

test("condition changes during load reuse run requests but render only the latest seed", async () => {
  const f = fixture();
  const initialized = f.view.initialize();
  await turn();
  assert.equal(
    f.$("precision-cards").querySelectorAll(".dynamic-run").length,
    2,
  );
  f.$("sample").value = D.conditionKey(f.runs[0].episodes[1]);
  const changed = f.$("sample").dispatch("change");
  assert.equal(f.calls.length, 3);
  f.pending.forEach((request) => request.resolve());
  await changed;
  await initialized;
  assert.match(f.$("precision-cards").innerHTML, /seed 1/);
  assert.doesNotMatch(f.$("precision-cards").innerHTML, /seed 0/);
});
