# Measured model and robot replay

This package builds local evidence for a debugger. It does not run a model or
simulate a robot. Neural inference remains in the prepared CUDA engine; the
browser/server reads prebuilt JSON and an explicit media manifest. Importing the
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
