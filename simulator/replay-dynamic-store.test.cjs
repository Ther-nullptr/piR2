const test = require("node:test");
const assert = require("node:assert/strict");
const D = require("./replay-dynamic.js");

function run(id, precision = "bf16", seed = 0) {
  return {
    schema: "pir2.replay.run",
    version: 1,
    id,
    kind: "dynamic",
    precision,
    protocol: "default_vla_flow4",
    label: id,
    mode: "libero_dynamic",
    episodes: [
      {
        id: `${id}-episode`,
        suite: "libero_spatial",
        task_id: 0,
        level: "l1",
        traj_index: 0,
        speed_mps: 0.06,
        seed,
        init_state_id: 0,
        protocol: "default_vla_flow4",
        timing_mode: "native",
        execution_prefix_steps: 2,
        requests: [{ id: `${id}-request`, service_duration_ms: 31.2 }],
        frames: [{ frame_id: `${id}-frame` }],
        events: [],
        trace: ["original"],
      },
    ],
  };
}
function descriptor(full, indexed = true) {
  const { id, kind, precision, label, mode, protocol } = full;
  return {
    id,
    kind,
    precision,
    label,
    mode,
    path: `runs/${id}.json`,
    episodes: full.episodes.length,
    ...(indexed
      ? { protocol, episode_summaries: full.episodes.map(D.episodeSummary) }
      : {}),
  };
}
function fixture({ indexed = true, deferred = false } = {}) {
  const { Store } = require("./replay-dynamic-store.js");
  const runs = [
    run("a"),
    run("b", "w8a8"),
    run("c"),
    run("d", "w8a8"),
    run("other", "bf16", 1),
  ];
  const catalog = {
    schema: "pir2.replay.catalog",
    version: 1,
    runs: runs.map((r) => descriptor(r, indexed)),
  };
  const calls = [],
    waiting = [];
  const response = (value) => ({ ok: true, json: async () => value });
  const fetch = (url, options = {}) => {
    calls.push(url);
    if (url.endsWith("/catalog")) return Promise.resolve(response(catalog));
    const full = runs.find((r) => url.endsWith("/" + r.id));
    if (!deferred) return Promise.resolve(response(full));
    return new Promise((resolve, reject) =>
      waiting.push({
        id: full.id,
        signal: options.signal,
        resolve: (value = full) => resolve(response(value)),
        reject,
      }),
    );
  };
  const store = new Store({ fetch });
  const row = (id) => {
    const run = store.index.find((r) => r.id === id);
    return {
      run,
      episode: run.episode_summaries[0],
      columnKey: D.groupKey(run),
    };
  };
  return { store, runs, calls, waiting, row, catalog };
}
const turn = () => new Promise((resolve) => setImmediate(resolve));

test("indexed initialization only fetches selected full runs and selection releases the old cache", async () => {
  const f = fixture();
  await f.store.initialize();
  assert.deepEqual(f.calls, ["/api/replay/catalog"]);
  const rows = D.rows(
    f.store.index,
    D.conditionKey(f.runs[0].episodes[0]),
    "all",
    {},
  );
  const loaded = await f.store.select(rows);
  assert.deepEqual(f.calls, [
    "/api/replay/catalog",
    "/api/replay/run/a",
    "/api/replay/run/b",
  ]);
  assert.equal(loaded[0].run, f.runs[0]);
  assert.equal(loaded[0].episode.requests[0].id, "a-request");
  assert.deepEqual([...f.store.runs.keys()], ["a", "b"]);
  await f.store.select([f.row("c"), f.row("b")]);
  assert.deepEqual([...f.store.runs.keys()].sort(), ["b", "c"]);
  assert.equal(f.calls.filter((url) => url.endsWith("/b")).length, 1);
  assert.equal(f.store.index[0].episode_summaries[0].requests, undefined);
});

test("overlapping pending requests are shared and late superseded responses cannot cache or publish", async () => {
  const f = fixture({ deferred: true });
  await f.store.initialize();
  const first = f.store.select([f.row("a"), f.row("b")]);
  const second = f.store.select([f.row("b"), f.row("c")]);
  assert.equal(f.waiting.find((r) => r.id === "a").signal.aborted, true);
  assert.equal(f.waiting.find((r) => r.id === "b").signal.aborted, false);
  assert.equal(f.calls.filter((url) => url.endsWith("/b")).length, 1);
  f.waiting.find((r) => r.id === "b").resolve();
  f.waiting.find((r) => r.id === "c").resolve();
  assert.deepEqual(
    (await second).map((r) => r.run.id),
    ["b", "c"],
  );
  // Simulate a transport that completes despite abort, after the new pair renders.
  f.waiting.find((r) => r.id === "a").resolve();
  assert.equal(await first, null);
  assert.deepEqual([...f.store.runs.keys()].sort(), ["b", "c"]);
  assert.equal(f.store.pending.size, 0);
});

test("superseded abort errors cannot replace the active selection with an error", async () => {
  const f = fixture({ deferred: true });
  await f.store.initialize();
  const old = f.store.select([f.row("a")]);
  const current = f.store.select([f.row("c")]);
  f.waiting[0].reject(
    Object.assign(new Error("aborted"), { name: "AbortError" }),
  );
  assert.equal(await old, null);
  f.waiting[1].resolve();
  assert.equal((await current)[0].run.id, "c");
  assert.deepEqual([...f.store.runs.keys()], ["c"]);
});

test("an aborted earlier request for the same run cannot overwrite a new request", async () => {
  const f = fixture({ deferred: true });
  await f.store.initialize();
  const old = f.store.select([f.row("a")]);
  await f.store.select([]);
  const current = f.store.select([f.row("a")]);
  f.waiting[0].resolve();
  assert.equal(await old, null);
  assert.equal(f.store.runs.size, 0);
  assert.equal(f.store.pending.size, 1);
  f.waiting[1].resolve();
  assert.equal((await current)[0].run.id, "a");
  assert.equal(f.store.pending.size, 0);
});

for (const defect of [
  "id",
  "kind",
  "precision",
  "protocol",
  "episode",
  "seed",
  "trajectory",
  "condition",
  "drive",
  "duplicate",
]) {
  test(`run/episode identity mismatch rejects selected data: ${defect}`, async () => {
    const f = fixture({ deferred: true });
    await f.store.initialize();
    const loaded = f.store.select([f.row("a")]);
    const altered = structuredClone(f.runs[0]);
    if (["id", "kind", "precision", "protocol"].includes(defect))
      altered[defect] = "mismatch";
    else if (defect === "episode") altered.episodes[0].id = "another-episode";
    else if (defect === "seed") altered.episodes[0].seed = 10;
    else if (defect === "trajectory") altered.episodes[0].traj_index = 1;
    else if (defect === "condition")
      altered.episodes[0].condition_id = "another-condition";
    else if (defect === "drive") altered.episodes[0].scene_drive = "frozen";
    else altered.episodes.push(structuredClone(altered.episodes[0]));
    f.waiting[0].resolve(altered);
    await assert.rejects(loaded, /identity|episode|condition|metadata/i);
    assert.equal(f.store.runs.size, 0);
    assert.equal(f.store.pending.size, 0);
  });
}

test("legacy catalogs derive summaries sequentially without retaining all full runs", async () => {
  const f = fixture({ indexed: false, deferred: true });
  const ready = f.store.initialize();
  await turn();
  for (let i = 0; i < f.runs.length; i++) {
    assert.equal(f.waiting.length, i + 1);
    assert.equal(f.store.runs.size, 0);
    f.waiting[i].resolve();
    await turn();
  }
  await ready;
  assert.equal(f.store.index.length, f.runs.length);
  assert.ok(
    f.store.index.every(
      (r) => r.episodes === 1 && r.episode_summaries.length === 1,
    ),
  );
  assert.ok(
    f.store.index.every((r) => r.episode_summaries[0].frames === undefined),
  );
  const loaded = f.store.select([f.row("c")]);
  f.waiting.at(-1).resolve();
  assert.equal((await loaded)[0].run.id, "c");
  assert.deepEqual([...f.store.runs.keys()], ["c"]);
});

test("selection is bounded to four rows and never silently replaces a missing indexed episode", async () => {
  const f = fixture();
  await f.store.initialize();
  await assert.rejects(
    f.store.select(f.runs.map((r) => f.row(r.id))),
    /four|4/i,
  );
  const row = f.row("a");
  await assert.rejects(
    f.store.select([{ ...row, episode: { ...row.episode, id: "unlisted" } }]),
    /episode|identity/i,
  );
  await assert.rejects(
    f.store.select([{ ...row, episode: { ...row.episode, seed: 9 } }]),
    /condition|identity/i,
  );
  assert.deepEqual(f.calls, ["/api/replay/catalog"]);
});

test("an invalid selection clears any previous cache and pending requests", async () => {
  const f = fixture();
  await f.store.initialize();
  await f.store.select([f.row("a")]);
  const row = f.row("a");
  await assert.rejects(
    f.store.select([{ ...row, episode: { ...row.episode, seed: 9 } }]),
    /identity/,
  );
  assert.equal(f.store.runs.size, 0);
  assert.equal(f.store.pending.size, 0);
});
