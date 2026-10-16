const test = require("node:test");
const assert = require("node:assert/strict");
const {
  StreamStore,
  StreamConnection,
  StreamProtocolError,
} = require("./replay-stream.js");

// Protocol fixtures only: no model run, measured timings, or invented kernel evidence.
function snapshot(runId = "contract-run") {
  return {
    schema: "pir2.replay.snapshot",
    version: 1,
    run_id: runId,
    last_sequence: 0,
    stream_available: true,
    run: {
      schema: "pir2.replay.run",
      version: 1,
      id: runId,
      kind: "model",
      precision: "w8a8",
      status: "incomplete",
      observations: [],
      clean_requests: [],
      profile_requests: [],
      kernels: [],
    },
    state: {
      status: "running",
      dropped_events: 0,
      recorded_events: 0,
      error: null,
    },
    clocks: { "host-monotonic": { unit: "s", domain: "host_monotonic" } },
  };
}
function event(sequence, changes = {}) {
  return {
    schema: "pir2.replay.event",
    version: 1,
    run_id: "contract-run",
    sequence,
    type: "model.stage.started",
    occurrence: {
      value: 10 + sequence / 10,
      unit: "s",
      clock_id: "host-monotonic",
      domain: "host_monotonic",
    },
    sources: { request_id: "request-fixture", observation_id: null },
    payload: { stage: "vlm", end: null, timing_boundary: "host_submission" },
    projection: null,
    ...changes,
  };
}
const code = (expected) => (error) =>
  error instanceof StreamProtocolError && error.code === expected;

test("golden snapshot and normalized append retain immutable originals and explicit unknowns", () => {
  const initial = snapshot();
  const store = new StreamStore("contract-run").snapshot(initial);
  const first = event(1, {
    type: "observation.loaded",
    sources: { observation_id: "obs-fixture" },
    projection: {
      op: "append",
      collection: "observations",
      items: [{ observation_id: "obs-fixture", sha256: null }],
    },
  });
  assert.equal(store.event(first), true);
  assert.equal(store.last_sequence, 1);
  assert.equal(store.state.recorded_events, 1);
  assert.equal(store.run.observations[0].sha256, null);
  assert.equal(store.events[0].payload.end, null);
  assert.deepEqual(store.events[0].occurrence, first.occurrence);
  assert.deepEqual(initial.run.observations, []);
  first.sources.observation_id = "mutated";
  assert.equal(store.events[0].sources.observation_id, "obs-fixture");
  assert.ok(Object.isFrozen(store.run.observations[0]));
  assert.ok(Object.isFrozen(store.events));
});

test("known duplicates compare semantic JSON; conflicting duplicates and gaps never mutate state", () => {
  const store = new StreamStore("contract-run").snapshot(snapshot());
  const first = event(1);
  store.event(first);
  assert.equal(
    store.event({
      ...first,
      payload: { end: null, timing_boundary: "host_submission", stage: "vlm" },
    }),
    false,
  );
  assert.throws(
    () => store.event(event(1, { payload: {} })),
    code("duplicate_conflict"),
  );
  assert.throws(() => store.event(event(3)), code("sequence_gap"));
  assert.equal(store.last_sequence, 1);
  assert.equal(store.events.length, 1);
});

test("bounded recent history deduplicates old cursors without claiming an unknown fingerprint matches", () => {
  const store = new StreamStore("contract-run", 2).snapshot(snapshot());
  for (let sequence = 1; sequence <= 4; sequence++)
    store.event(event(sequence));
  assert.deepEqual(
    store.events.map((row) => row.sequence),
    [3, 4],
  );
  assert.equal(store.event(event(1)), false);
  assert.equal(store.state.recorded_events, 4);
});

test("clock registry is explicit, permits null and backdated occurrences, and rejects undeclared clocks", () => {
  const store = new StreamStore("contract-run").snapshot(snapshot());
  const first = event(1);
  first.occurrence.value = null;
  store.event(first);
  const second = event(2);
  second.occurrence.value = -10;
  store.event(second);
  for (const change of [
    { clock_id: "other" },
    { clock_id: ["host-monotonic"] },
    { unit: "ms" },
    { domain: "cuda_event" },
    { value: NaN },
  ]) {
    assert.throws(
      () =>
        store.event(
          event(3, { occurrence: { ...event(3).occurrence, ...change } }),
        ),
      StreamProtocolError,
    );
  }
  assert.equal(store.last_sequence, 2);
  assert.equal(store.events[0].occurrence.value, null);
  assert.equal(store.events[1].occurrence.value, -10);
});

test("append collisions and replacement identity changes are rejected atomically", () => {
  const store = new StreamStore("contract-run").snapshot(snapshot());
  const projection = {
    op: "append",
    collection: "kernels",
    items: ["actual-kernel-name"],
  };
  store.event(event(1, { projection }));
  assert.throws(
    () =>
      store.event(
        event(2, {
          projection: {
            op: "append",
            collection: ["observations"],
            items: [{ observation_id: "wrong-type" }],
          },
        }),
      ),
    StreamProtocolError,
  );
  assert.throws(
    () => store.event(event(2, { projection })),
    code("identity_collision"),
  );
  for (const change of [
    { id: "other" },
    { kind: "deployment" },
    { precision: "bf16" },
    { version: 2 },
  ]) {
    assert.throws(
      () =>
        store.event(
          event(2, {
            projection: { op: "replace", run: { ...store.run, ...change } },
          }),
        ),
      StreamProtocolError,
    );
  }
  const replacement = { ...store.run, status: "complete" };
  store.event(event(2, { projection: { op: "replace", run: replacement } }));
  assert.equal(store.run.status, "complete");
  assert.equal(store.last_sequence, 2);
});

test("source IDs, finite JSON, run identity and sequence boundaries are validated", () => {
  const store = new StreamStore("contract-run").snapshot(snapshot());
  for (const change of [
    { run_id: "other" },
    { sequence: 0 },
    { sequence: 1.5 },
    { sequence: Number.MAX_SAFE_INTEGER + 1 },
    { sources: { path: "anything" } },
    { payload: { duration: Infinity } },
    { payload: null },
  ])
    assert.throws(() => store.event(event(1, change)), StreamProtocolError);
  assert.throws(() => new StreamStore("../private"), StreamProtocolError);
  assert.throws(() => new StreamStore("run\n"), StreamProtocolError);
  assert.throws(
    () =>
      store.snapshot({
        ...snapshot(),
        run: { ...snapshot().run, id: "other" },
      }),
    StreamProtocolError,
  );
  assert.equal(store.last_sequence, 0);
});

test("drops and terminal recording failure preserve explicit accounting without changing model status", () => {
  const store = new StreamStore("contract-run").snapshot(snapshot());
  store.event(event(1, { type: "stream.dropped", payload: { count: 3 } }));
  assert.throws(
    () =>
      store.event(event(2, { type: "stream.dropped", payload: { count: 0 } })),
    StreamProtocolError,
  );
  store.event(
    event(2, {
      type: "stream.failed",
      payload: {
        error: "disk unavailable",
        scope: "event_recording",
        model_failure: false,
      },
    }),
  );
  assert.equal(store.state.dropped_events, 3);
  assert.equal(store.state.status, "failed");
  assert.equal(store.state.error, "disk unavailable");
  assert.equal(store.run.status, "incomplete");
  assert.throws(() => store.event(event(3)), code("terminal_stream"));
  assert.equal(
    store.event(
      event(2, {
        type: "stream.failed",
        payload: {
          error: "disk unavailable",
          scope: "event_recording",
          model_failure: false,
        },
      }),
    ),
    false,
  );
});

class Source {
  static instances = [];
  constructor(url) {
    this.url = url;
    this.listeners = new Map();
    this.closed = false;
    Source.instances.push(this);
  }
  addEventListener(type, callback) {
    this.listeners.set(type, callback);
  }
  close() {
    this.closed = true;
  }
  emit(type, data) {
    this.listeners.get(type)?.({ data: JSON.stringify(data) });
  }
}
function connection(fetchSnapshot = async () => snapshot()) {
  const statuses = [],
    snapshots = [],
    events = [];
  const client = new StreamConnection({
    fetchSnapshot,
    EventSource: Source,
    onStatus: (value) => statuses.push(value),
    onSnapshot: (store, envelope) =>
      snapshots.push([store.run_id, envelope.last_sequence]),
    onEvent: (value, store) =>
      events.push([value.sequence, store.last_sequence]),
  });
  return { client, statuses, snapshots, events };
}
const flush = () => new Promise((resolve) => setImmediate(resolve));

test("connection uses snapshot cursor and native reconnection, then explicitly closes terminal events", async () => {
  const { client, statuses, events } = connection();
  await client.start("contract-run");
  const source = Source.instances.at(-1);
  assert.equal(source.url, "/api/replay/events/contract-run?after=0");
  source.emit("replay", event(1));
  source.onerror({});
  assert.equal(statuses.at(-1).status, "reconnecting");
  assert.equal(source.closed, false);
  source.emit("replay", event(1));
  assert.deepEqual(events, [[1, 1]]);
  source.emit("replay", event(2, { type: "run.ended", payload: {} }));
  assert.equal(source.closed, true);
  assert.equal(statuses.at(-1).status, "ended");
  assert.equal(client.store.state.status, "ended");
});

test("explicit end reconciles terminal status with an authoritative snapshot", async () => {
  let calls = 0;
  const { client, statuses } = connection(async () => {
    const value = snapshot();
    if (++calls > 1) value.state.status = "ended";
    return value;
  });
  await client.start("contract-run");
  const source = Source.instances.at(-1);
  source.emit("end", {
    run_id: "contract-run",
    last_sequence: 0,
    status: "ended",
  });
  await flush();
  assert.equal(source.closed, true);
  assert.equal(statuses.at(-1).status, "ended");
  assert.equal(client.store.state.status, "ended");
});

test("a permanently closed native EventSource reports transport failure instead of endless reconnecting", async () => {
  const { client, statuses } = connection();
  await client.start("contract-run");
  const source = Source.instances.at(-1);
  source.readyState = 2;
  source.onerror({});
  assert.equal(source.closed, true);
  assert.equal(statuses.at(-1).status, "error");
  assert.equal(statuses.at(-1).code, "transport_closed");
  assert.equal(client.store.state.status, "running");
});

test("resnapshot control closes old transport and fresh snapshot supersedes its callbacks", async () => {
  let count = 0;
  const fresh = snapshot();
  fresh.last_sequence = 5;
  fresh.state.recorded_events = 5;
  const { client, snapshots } = connection(async () =>
    ++count === 1 ? snapshot() : fresh,
  );
  await client.start("contract-run");
  const old = Source.instances.at(-1);
  old.emit("resnapshot", {
    run_id: "contract-run",
    reason: "sequence_gap",
    snapshot_url: "https://untrusted.invalid/ignored",
  });
  old.emit("replay", event(1));
  await flush();
  assert.equal(old.closed, true);
  assert.equal(client.store.last_sequence, 5);
  assert.equal(
    Source.instances.at(-1).url,
    "/api/replay/events/contract-run?after=5",
  );
  assert.deepEqual(snapshots, [
    ["contract-run", 0],
    ["contract-run", 5],
  ]);
  client.close();
});

test("gaps resnapshot but conflicting duplicate evidence is surfaced as protocol failure", async () => {
  const { client, statuses } = connection();
  await client.start("contract-run");
  let source = Source.instances.at(-1);
  source.emit("replay", event(2));
  await flush();
  assert.ok(statuses.some((row) => row.status === "resnapshotting"));
  source = Source.instances.at(-1);
  source.emit("replay", event(1));
  source.emit("replay", event(1, { payload: {} }));
  assert.equal(statuses.at(-1).status, "error");
  assert.equal(statuses.at(-1).code, "duplicate_conflict");
  assert.equal(source.closed, true);
});

test("asynchronous run selection and close ignore late snapshots and old-source callbacks", async () => {
  let resolveOld;
  const { client, snapshots, events } = connection((runId) =>
    runId === "old"
      ? new Promise((resolve) => {
          resolveOld = resolve;
        })
      : Promise.resolve(snapshot(runId)),
  );
  const oldStart = client.start("old");
  await client.start("new");
  const source = Source.instances.at(-1);
  resolveOld(snapshot("old"));
  await oldStart;
  assert.equal(client.store.run_id, "new");
  assert.deepEqual(snapshots, [["new", 0]]);
  client.close();
  source.emit("replay", event(1, { run_id: "new" }));
  assert.deepEqual(events, []);
  let resolveClosed;
  const pending = connection(
    () =>
      new Promise((resolve) => {
        resolveClosed = resolve;
      }),
  );
  const start = pending.client.start("contract-run");
  pending.client.close();
  resolveClosed(snapshot());
  await start;
  assert.deepEqual(pending.snapshots, []);
  assert.equal(pending.statuses.at(-1).status, "closed");
});

test("historical, terminal and 404 snapshots do not open transports or claim model failure", async () => {
  const before = Source.instances.length;
  for (const status of ["historical", "ended", "failed"]) {
    const initial = snapshot();
    initial.state.status = status;
    initial.stream_available = status !== "historical";
    const { client, statuses } = connection(async () => initial);
    await client.start("contract-run");
    assert.equal(statuses.at(-1).status, status);
  }
  const { client, statuses } = connection(async () => ({
    status: 404,
    ok: false,
    json: async () => ({ error: "not found" }),
  }));
  await client.start("contract-run");
  assert.equal(statuses.at(-1).status, "historical");
  assert.equal(statuses.at(-1).unavailable, true);
  assert.equal(client.store.run, null);
  assert.equal(Source.instances.length, before);
});
