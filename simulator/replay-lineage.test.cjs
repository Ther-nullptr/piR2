const test = require("node:test"),
  assert = require("node:assert/strict");
let L;
try {
  L = require("./replay-lineage.js");
} catch {
  L = {};
}
const s2 = {
  id: "v2",
  role: "S2",
  feature_sequence: 2,
  source_tick: 3,
  capture_s: 0.2,
  start_s: 0.21,
  end_s: 0.25,
  published_s: 0.26,
};
const s1 = {
  id: "a7",
  role: "S1",
  request_tick: 7,
  feature_sequence: 2,
  feature_source_tick: 3,
  feature_capture_s: 0.2,
  state_tick: 7,
  state_capture_s: 0.35,
  start_s: 0.36,
  use_s: 0.37,
  completed_s: 0.4,
  end_s: 0.4,
  adopted_s: 0.45,
};
const tick = {
  id: "tick8",
  tick: 8,
  relative_s: 0.46,
  executed_slot: { producer_kind: "request", producer_request_tick: 7 },
};
const episode = {
  id: "ep",
  ticks: [tick],
  queue: {
    duration_s: 1,
    requests: [s2, s1],
    feature_updates: [{ feature_sequence: 2, t: 0.26 }],
    initial_feature: {},
    ticks: [{ ...tick, t: 0.46 }],
    camera_events: [],
  },
};
test("actual VLM publication, feature read, action output and execution form traceable arrows", () => {
  assert.equal(typeof L.deployment, "function");
  const graph = L.deployment(episode, 0.5);
  assert.deepEqual(
    graph.edges.map((e) => e.from + ">" + e.to),
    [
      "observation>vlm",
      "vlm>cache",
      "cache>action",
      "action>output",
      "output>control",
    ],
  );
  assert.equal(graph.nodes.find((n) => n.key === "action").request.id, "a7");
  assert.match(graph.nodes.find((n) => n.key === "action").title, /S1\[/);
  assert.equal(graph.nodes.find((n) => n.key === "vlm").request.id, "v2");
});
test("unknown, unpublished and future execution links remain absent", () => {
  const bad = structuredClone(episode);
  bad.queue.feature_updates = [];
  bad.queue.requests[0].published_s = null;
  const graph = L.deployment(bad, 0.38, { request: s1 });
  assert.ok(!graph.edges.some((e) => e.from === "cache"));
  assert.ok(!graph.edges.some((e) => e.to === "control"));
  assert.equal(graph.nodes.find((n) => n.key === "control").known, false);
});
test("request selection follows its own consumed feature, never the latest cache", () => {
  const e = structuredClone(episode);
  e.queue.feature_updates.push({ feature_sequence: 8, t: 0.3 });
  const graph = L.deployment(e, 0.5, { request: s1 });
  assert.equal(graph.featureSequence, 2);
  assert.equal(graph.nodes.find((n) => n.key === "cache").feature_sequence, 2);
});

test("fallback control never acquires a model dependency from stale slot metadata", () => {
  const e = structuredClone(episode);
  e.ticks[0].fallback = true;
  const graph = L.deployment(e, 0.5);
  assert.ok(!graph.edges.some((edge) => edge.to === "control"));
});

test("missing S2 never borrows the S1 state observation ID as visual source", () => {
  const e = structuredClone(episode);
  e.queue.requests = e.queue.requests.filter((r) => r.role !== "S2");
  e.queue.requests[0].observation_id = "state-tick-7";
  e.queue.requests[0].feature_source_tick = 3;
  const graph = L.deployment(e, 0.5);
  const observation = graph.nodes.find((node) => node.key === "observation");
  assert.equal(observation.source_tick, 3);
  assert.equal(observation.observation_id, null);
});

test("preparation-only and unknown boundaries cannot invent neural execution chains", () => {
  const run = {
    boundaries: {
      prepared_model: "full VLM + Flow4",
      preparation_transfer: "raw -> resident inputs",
    },
  };
  for (const scope of ["preparation_transfer", "unknown"]) {
    const graph = L.model(
      run,
      { id: "one", scope, observation_id: "obs", instrumented: false },
      10,
    );
    assert.ok(
      !graph.edges.some(
        (edge) =>
          ["vlm", "action"].includes(edge.from) ||
          ["vlm", "action"].includes(edge.to),
      ),
    );
    assert.equal(graph.nodes.find((node) => node.key === "vlm").known, false);
  }
});
