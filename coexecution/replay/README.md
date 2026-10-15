# Measured model and robot replay

The catalog builder and browser/server read recorded evidence; neither starts a
model or simulator. The separate, explicit `record-model` command runs the
existing prepared CUDA measurement engine and publishes a bounded event stream.
Importing the
catalog needs only the standard library. Building a model catalog lazily reads
NumPy action archives; it never sends an NPZ archive to the browser.

## Capture and build

Add `--replay-assets` to `python -m coexecution.libero.measurement` to export the
actual raw camera PNGs and observation state before any measured model calls.
Use `--measurement-boundary model-only` for independent resident-input replay.
Clean samples contain exact retained actions. `--profile-rounds 1` adds separate,
instrumented invocations: their seeds, request annotations and actual Chrome
kernel timestamps are retained. Their outputs are explicitly unavailable unless
that exact instrumented invocation records them; clean outputs are never
substituted for a profiler call.

For actual robot responses, use the existing deployment evaluator with
`--record-replay --replay-max-frames 721`. The experiment-plan equivalent is
`"record_replay": true, "replay_max_frames": 721, "task_ids": [0]`. Omitting
`task_ids` keeps all ten Spatial tasks. This debug flag is deployment-only and
records each evaluated episode's own cameras and robot states. `--no-video`
continues to control the legacy MP4 recorder and does not disable an explicitly
requested replay. Replay capture supersedes duplicate legacy video recording.

The bounded recorder stores owned copies of both cameras and before/after
position, rotation and gripper state. Encoding WebM/VP9 and writing JSON happen
after control ends. Frame/byte overflow and encoding failures remain explicit in
the episode metadata. Camera copies are committed as a complete group; allocation
failure stops recording, preserves prior complete frames and leaves control
running. The first frame is the actual initial observation, which
may precede control; subsequent frames map to `next_observation_computed_s`
after the physical step, not the tick's start. Media timestamps follow the
encoded 20 fps index; wall-clock seeking must use the frame mapping. Recording
adds array-copy overhead and marks the run instrumented. Physical control,
real cameras, real VLM and the 20 Hz environment remain unchanged.

`--execution-clock ready` (or plan/case `"execution_clock": "ready"`) records
GR00T Flow execution without an imposed wall grid. The physics step remains
0.05 s; actual wall execution Hz and simulated-time/wall-time ratio are separate
metadata. Completed plans are adopted only between atomic environment steps,
with a one-step committed prefix. Ready deadline/lateness/validity fields are
null, and its records cannot seed the fixed-rate teaching simulator. This is
StreamingVLA-inspired scheduling with GR00T, not its AFM/AEO algorithm.

```bash
python -m coexecution.replay build \
  --model-run /path/to/model_only \
  --deployment /path/to/evaluation \
  --output /path/to/new-replay-catalog \
  --clock-cohort audited-device-and-clock-session
```

Both input options can be repeated. A deployment input may be one deployment
directory, its condition directory, or an experiment root containing condition
subdirectories. The output must be new or empty. Inputs are read only; generated
catalogs, media and profiles stay outside Git. The optional clock cohort is an
explicit operator declaration based on hardware/telemetry review, not a measured
clock inferred by the builder. Omit it if unknown; same-input comparison keys
will then be unavailable.

For catalogs containing both cadences, add `--default-execution-clock ready`
to choose the new recordings explicitly. The builder rejects a requested clock
that is absent from the evidence. The frontend's execution-clock selector
switches all precisions together; `?execution_clock=paced` or `ready` overrides
the catalog default. Legacy catalogs remain paced. Each video keeps its own
frame-to-wall mapping; the selector never retimes an old trajectory.

## Explicit recording and incremental delivery

Use the PR28 recipes in [platform-handoff.md](../../docs/platform-handoff.md)
for the current official Spatial runtime, including Flow4/H40 and optional
intra-call K/V reuse. Keep independent processes and output directories for
different operator profiles. For example, a manually launched BF16 recording is:

```bash
python -m coexecution.replay record-model \
  --catalog-output /path/to/live-catalog \
  --checkpoint /path/to/libero_spatial --dataset /path/to/groot-libero-spatial \
  --upstream /path/to/Isaac-GR00T --output /path/to/new-measurement \
  --precisions bf16 --modes kv_reuse --action-cuda-graph --operator-fusion \
  --measurement-boundary model-only --observations 3 --rounds 3 \
  --warmup 2 --profile-rounds 1 --replay-assets
```

This command delegates all model options and numerical/Graph checks to the
existing measurement entrypoint and enables runtime identity receipts for the
recording. Ordinary `coexecution.libero.measurement` captures those additional
source and loaded-binary identities only with `--runtime-receipt`, outside its
measurement timers. Recording requires exactly one runtime mode. The
catalog and one journal per precision are registered before inference; actual
inputs and completed outputs are appended, and final canonical evidence replaces
the partial run. A separate caller may attest a runtime receipt only with
matching identity and retained GPU validation evidence. A new date, filename or
loaded profile alone is not such an attestation.

The optional catalog `streams` list registers `run_id`, `snapshot_path`,
`journal_path` and `max_event_bytes` as allowlisted relative resources. The server
exposes `/api/replay/snapshot/<id>` and `/api/replay/events/<id>?after=N` in addition
to the original APIs. Snapshots use `pir2.replay.snapshot` v1 and retain the v1
normalized run, `last_sequence`, stream state and explicit clocks. SSE `replay`
events use `pir2.replay.event` v1, a monotonically increasing sequence, occurrence
clock/unit/domain, type, source IDs and an optional append/replace projection.
`Last-Event-ID` takes precedence over the query cursor. Gaps or stale cursors
require a new snapshot; `end` closes a terminal stream. Browser receipt time is
never an inference timestamp or an AoI input.

The writer has event-count and byte bounds; loss and recorder errors remain
visible. The default event limit is 1 MiB, explicit registrations support up to
8 MiB, and tail batches are bounded to 16 MiB. `flush()` is an untimed
finalization barrier, including pending loss records. Partial trailing JSONL is
not delivered. Catalog/media refresh retains the path/MIME/NOFOLLOW allowlist.
Model and video generation still require explicit external commands.

Instrumented live host-stage events describe submission, not individual GPU
completion. Real kernel intervals become available only from the correlated
profile export. Clean requests retain their own wall/output evidence and do not
acquire an invented VLM/DiT split. Robot sink hooks announce owned frame groups
and encoding completion; the standard recorder still encodes after control.

## Version 1 contract

`catalog.json` uses `schema: "pir2.replay.catalog"`, `version: 1`, and `runs` with
`id`, `kind`, `precision`, `mode`, `label`, `path`, request/episode counts and
warnings. Each `runs/<id>.json` uses `schema: "pir2.replay.run"` and keeps source
paths, SHA-256 digests, gates and configuration. `media.json` uses
`schema: "pir2.replay.media"`; its `assets` list maps opaque IDs to copied,
relative media paths with MIME type and original source digest. Asset paths must
remain under their explicitly declared input root. A server should expose only
these manifest IDs, never arbitrary filesystem paths.

Model runs contain `observations`, `clean_requests`, `profile_requests`,
`kernels`, `profile_summary` and `validation`. Clean requests retain measured
wall latency, seed, observation identity and action fields as `{shape, values}`.
Optional `difference_from_bf16` uses matched observation hash, seed, scope,
known shared runtime configuration and declared clock cohort; it describes
numerical differences, not task-quality equivalence. Action arrays use decoded
prediction-horizon coordinates, not executed control ticks.

Instrumented requests have their own IDs and seeds. Their `timeline` contains
actual `start_ms`, `duration_ms`, `stage`, `category`, `kernel_id`, device and
stream, relative to the CPU request annotation. `kernel_id` indexes `kernels`.
CUDA launch correlation links kernels to requests and VLM/action stages; temporal
proximity is never substituted for missing correlation. CPU and GPU copies of a
Chrome user annotation are distinct; only host annotations define request scope.
Unknown activities remain unclassified or unlinked, with explicit warnings.
Aggregates are never rendered as an invented timeline. Clean wall, instrumented
host wall, CUDA span, summed GPU work and active union are nonadditive metrics.

Deployment runs contain `episodes`. Each has original `metadata`, normalized
`queue` evidence from the existing queue reporter, actual `ticks` and `robot`
recording metadata. Tick/observation/request/feature/frame IDs preserve causal
ownership. `robot.frames` includes `wall_s`, `relative_s`, `media_time_s`, index,
phase and source tick; each camera maps to its own media ID. Legacy runs without
recorded state or video show unavailable evidence. Paired task/init/seed fixtures
can be identified, but closed-loop trajectories diverge and never receive a
same-input comparison key.

Robot positions are meters. The pinned fields named roll/pitch/yaw actually
contain an axis-angle rotation vector in radians, not Euler angles. Gripper
state is physical joint position; executed commands are normalized LIBERO
controller commands and must not be labelled Cartesian meter displacements.

Finite values and chronological intervals are checked. Missing values remain
null/unavailable. Failed or missing gates are preserved. Historical sidecar
inconsistencies are disclosed, with complete `results.json` evidence preferred
and both source hashes retained. A numerical, RNG or graph failure may stop before
`results.json` exists. The builder then reads actual retained sidecars and marks
runs `failed` or `incomplete`; no missing configuration or latency is inferred.
`retained_outputs` exposes archived invocations that have no measured sample row,
with null wall time/seed/comparison key. Saved nonfinite arrays keep their shape
and nonfinite count with unavailable values, never fabricated finite outputs.
Deployment task IDs are integers in 0–9 and episode IDs are nonnegative integers;
trace paths and external symlinks must remain under the declared input root.


Legacy local measurement packages may lack the newer observation/request IDs.
The importer derives labels only from persisted observation SHA-256 plus list
index, and the stored round/mode/observation/scope fields used in action-archive
keys. Each derivation is marked in `identity_derivation`. It preserves original
wall times, action arrays and any recorded seed; missing seeds, settings and raw
camera assets remain unknown. A missing `instrumented` flag remains unknown even
when `results.profiler` is false (CUDA-event measurement is a separate concern),
so these historical rows are excluded from Clean P50/P95. Their own raw request
wall/output remains viewable. Historical profile aggregates without request
annotations stay in `profile_summary`; the importer never converts them into a
per-request kernel timeline.
