# Two timing protocols

Both protocols use real VLM image features, current robot state, the same checkpoint/data identities and paired initial conditions. They answer different questions and their scores must not be pooled.

| Protocol | Observation/action timing | Purpose |
|---|---|---|
| Algorithm | Image age fixed at 3 ticks (150ms); action release fixed at 1 tick (50ms); current state | Compare behavior at identical simulated delays |
| Deployment | Absolute wall-clock 20Hz control; asynchronous real VLM and action workers | Measure behavior with actual processing/transport delays |

In the algorithm protocol, computation waits do not advance physics. Each method executes the committed prefix, then uses the next legal segment. The initial missing image history is explicitly padded. A GPU contract check adds 100ms wait per call and compares actions and final images; this is a timing check, not a success-rate evaluation.

Deployment keeps control independent from unfinished RPCs. An observation computed early is withheld until its physical tick. Committed commands cannot be overwritten, expired slots are discarded, and buffer exhaustion uses zero Cartesian deltas with the previous gripper command. Flow publishes the unexpired native clean plan; πR² publishes only its newly clean segment. πR² normally uses one DiT evaluation per request; full-buffer recovery is explicitly counted as four bootstrap evaluations plus one normal update.

Warm-up and full request calibration occur outside scored control. Control frequency, policy query frequency and plan publication frequency are separate metrics. Record action/VLM latency, actual cache age, state/image timestamp difference, control lateness, expired/fallback slots and recovery counts. Quantiles are calculated from raw requests, not averaged episode quantiles.

Deployment optionally accepts `--vision-max-hz 15` to limit scored S2 admissions. The default zero is unlimited. The first admission is immediate; each later admission is at least `1 / vision_max_hz` after the previous actual admission, with no catch-up after a late wake or slow request. Camera offers continue at physical 20Hz. The newest pending frame is selected only at admission, so newer frames can replace it throughout a wakeable rate wait. Completed features publish without an admission-rate hold; the independent service floors below still apply. Bootstrap and solo warmup/calibration remain outside this limit. Experiment plans may set `vision_max_hz` globally or per condition; a condition value, including zero, overrides the plan, and only deployment evaluator commands receive it.

`vision_admission_wait` traces the eligible time and replaceable mailbox; `vision_admitted` records `admitted_s` under the mailbox lock, separately from actual RPC `started_s`. Request rows retain both timestamps, `vision_max_hz` and elapsed `admission_wait_seconds` (from first rate wait through admission, including scheduling). `vlm_work` reports offered/admitted/completed/replaced/dropped counts, admission starts and waits, and `actual_admission_hz` as `(admissions - 1) / (last - first)` when defined. An interrupted final wait has no admission row; its wait and stop events remain in the trace. `vision_stop_requested` records when stop was observed under the same lock and retains the controller's separate cutoff. No new admission follows observed stop; an already admitted RPC may begin or finish afterward, and its timestamps and post-control counts are retained.

For a hardware scan, the evaluator accepts `--task-ids 0 3 --episodes-per-task 1 --max-steps 300 --no-video`. The default still selects all ten Spatial tasks. A selected subset is labelled `selected_task_subset`, and its expected episode count uses only those tasks. Episodes retain ordinary first-success/environment-termination semantics: `physical_control_seconds` and the deployment `controlled_start_s`/`controlled_end_s` report actual exposure; a 300-tick cap does not promise 15 seconds after an early success. Subset success is not a full-suite result.

Deployment calibration stores all 12 action samples in `calibration.json`, marking the first two as excluded; `action_rpc_seconds` and `action_server_seconds` contain the ten included samples. The action budget remains `min(5, ceil((p95 RPC + 0.005) / 0.05))`, with an explicit over-budget flag. `slow-warmup.json` similarly keeps every visual RPC/server sample and marks the first two excluded. `--slow-warmup-calls` defaults to 12. These are solo warm calibration measurements; concurrent-control latency comes from the episode trace.

Feature transport defaults to `--feature-transport numpy_fp32`, the upstream NumPy dictionary. Opt in to `--feature-transport numpy_bf16` on **both services and the deployment evaluator** to send existing BF16 features as explicitly tagged raw uint16 words in a versioned envelope. This changes their representation, with no new quantization: BF16 bits, other floating dtypes, bool/int masks, scalar shapes and metadata are preserved. Imported tensors own their storage and finish transfer on the endpoint's CUDA stream before planning. Unknown schemas, dtype tags and malformed shapes are rejected before cache installation. Service identities declare the encoding/source; VLM metadata and action cache audits report `feature_wire` with actual field dtypes and `payload_nbytes` (array storage bytes, excluding serialization headers).

The evaluator additionally accepts `--feature-rpc combined` to install a newly selected feature inside the same `plan` request. The default `separate` sends `install` then `plan`. Both modes select at the existing control boundary and deduplicate the same feature sequence; combined mode advances the client's installed sequence only after a successful reply. Bootstrap, calibration and scored workers use the declared mode, while action service floors still begin at submission and include installation. Experiment plans accept `feature_transport` and `feature_rpc` globally or per condition, independently of weight-precision flags. Nondefault evaluator feature controls require deployment; resume configuration and calibration bind these settings. The physical 20Hz loop and the optional 15Hz S2 admission cap are unchanged.

Timing version 2 adds explicit server boundaries to both reference and candidate requests. `plan_rpc_started_s` remains the actual client network-call start; in combined mode it precedes installation. `server_install_started_s` and `server_install_completed_s` describe the most recent imported cache, while `feature_installed_in_request` distinguishes a combined install from reuse. `server_plan_started_s` is recorded after optional installation and stream readiness, before server planning; it is a host boundary, not an exact neural-kernel timestamp. `server_seconds` measures planning from that boundary; `server_request_seconds` also includes combined installation. Whole-call `raw_service_seconds` includes queueing, network, installation and planning. None of these wall-time differences is a CPU-work measurement.

Queue reports and feature-usage audits prefer `server_plan_started_s` for `use_s` and cache age, record `use_boundary`, and retain legacy `plan_rpc_started_s` semantics when the new timestamp is absent. Existing `cache_age_at_request_ms` remains age at RPC entry; `cache_age_at_server_plan_ms` is the comparable post-install boundary. These loopback timestamps share the host monotonic clock. Feature publication, selection and physical execution keep their existing timestamps and source lineage.

Run `PIR2_GPU_TESTS=1 python -m pytest -q tests/gpu/test_feature_transport_gpu.py` in the pinned CUDA environment for all 65,536 BF16 bit patterns, other dtype/shape preservation, fresh storage and stream-readiness gates. These codec checks perform no neural inference and do not establish LIBERO task success.

Same-process Flow workers are an explicit `--server-layout paired` option. Start
one service with `--device cuda:0 --vlm-device cuda:0 --port 5570 --vlm-port 5572`
and select `--server-layout paired` on the deployment evaluator. The service loads
two independent official policies, each with its own CUDA stream and endpoint.
The default layout remains `split`. Experiment plans accept `server_layout`
globally or per condition; paired plans start and own one process, wait for both
endpoints, and require the same declared physical GPU.

With paired Flow workers on one physical device, `--feature-transport cuda_local`
replaces tensor serialization with versioned, owner-qualified feature handles.
Each admitted frame still runs the real VLM. Publication follows the completed
producer event; the consumer waits on that event and records its stream before
installing independent references in its policy cache. A bounded pool (default
16 leases, service option `--feature-max-leases`) retains owned tensor snapshots.
It rejects capacity exhaustion rather than evicting an in-flight reader. Handles
bind the exact metadata, dtype, shape, device, process instance and client session;
stale or modified handles fail without substituting older neural outputs.

The client holds a lease while the feature is latest, selected, pending, or kept
as the action worker's fallback. Last-reference releases are batched on the next
vision RPC. Shutdown first joins action consumers, then closes the owned feature
session over a fresh RPC connection, including features whose replies were lost.
Reset invalidates the producer session while existing reader references remain
safe. `feature_wire.payload_nbytes` is zero tensor bytes for this transport;
`resident_nbytes` reports retained tensor storage. Handle serialization still has
metadata overhead. Pool counts appear in VLM responses and runtime identities.

Paired optimization installs adapters independently for each policy. Ordinary
CUDA submissions can overlap; first VLM/DiT graph captures take a process-wide
exclusive guard after quiescing prior GPU work. Simultaneous capture upgrades
drop their shared submission slots before waiting. Startup, reset, inherited
policy endpoints and teardown also participate. Paired whole-action graph mode
is rejected; per-DiT and whole-VLM graph options are supported. S2 reset does not
reseed S1's RNG. Graph capture/replay/failure deltas remain separate from physical
forward hooks, and `scored_cold_capture_requests` exposes captures during scored
control. Cold warmup costs stay outside steady measurement.

Condition-wide solo calibration prepares only its initial task metadata. Each
episode therefore also runs one real S2 preparation call for its current task,
before the fresh S1 bootstrap capture and scored control. This applies equally
to split and paired layouts and every feature transport. The call requests no
feature payload; its outputs never enter the scored mailbox or S1 cache. Every
subsequent admitted frame still runs the real VLM. Per-episode
`vision_preparation` records its actual source and wall times, physical VLM count
and graph deltas separately. It does not change the calibrated delay, the
nondecreasing delay rule, camera admission or scored completion timestamps.

Compare resident transport against **paired `numpy_bf16`** as well as split
`numpy_bf16`, with identical precision/runtime flags, 20Hz control and S2 admission
settings. This separates transport cost from process/context layout effects.
Run `PIR2_GPU_TESTS=1 python -m pytest -q tests/gpu/test_resident_features_gpu.py`
for stream readiness, owned storage, allocator lifetime and two independent tiny
pinned CUDA policies with simultaneous cold captures, fresh images and RNG
checks. Official-checkpoint equality, measured server latency and full Spatial
quality remain separate gates; W4 systems measurements do not establish quality.

Measured queue traces retain the existing `control_tick`, `request_completed`, `unused_result_at_episode_end` and `vision_request` row kinds. New lifecycle rows add `schema_version: 1`, a unique insertion-order `seq`, and `t_s` in the same host monotonic-clock domain as worker RPC and controller timestamps. Threads collect detached metadata in memory; JSON/file I/O occurs after timed control and worker drain. Row order is insertion order, so timeline viewers use each event's timestamp instead of assuming timestamps increase with `seq`.

| Event / field | Meaning |
|---|---|
| `control_start`, `control_end` | Bounds of the scored wall-clock interval; drain work can complete after its end |
| `camera_offered`, `camera_replaced`, `camera_dropped` | Actual latest-frame mailbox transitions, keyed by `source_tick`/`capture_s`; replacement identifies `replacement_source_tick` |
| `vision_started`, `feature_published` | In-flight frame ownership and feature availability; publication records `feature_sequence`, actual publish `t_s`, and separate `rpc_completed_s` |
| `camera_queue` | Waiting/in-flight ticks and capture times, plus the asynchronous worker's published feature sequence/source/capture time |
| `action_submitted`, `action_started`, `action_completed` | Request lifecycle keyed by `request_tick`; every stage records state and selected/consumed feature dependencies; completion uses the server's actual cache audit |
| `action_adopted` | Controller tick accepting a result, including installed, expired and protected slot counts |
| `control_tick.executed_slot` | Producer metadata for the command just applied |
| `control_tick.action_buffer` | Detached remaining-slot ownership after execution, with `committed_until` and `slots` |

Each slot reports `tick`, `status` (`future`, `committed` or `fallback`), `committed`, `fallback`, `producer_kind` and `producer_request_tick`. Bootstrap and fallback slots have no model request producer; a fallback's `reserved_at_tick` records when the hold command was committed. Sequence 0 in `control_start` and action dependencies denotes the bootstrap feature; the asynchronous camera worker begins at sequence 1. Request metadata includes `state_tick`, `state_capture_s`, `feature_sequence`, `feature_source_tick`, `feature_capture_s`, `delay_ticks` and `deadline_s`. Traces contain metadata and executed actions, never images or embeddings.

Strict timing validity requires mean frequency within 2% of 20Hz, maximum control lateness at most 5ms and adjacent interval error at most 5ms. All episodes remain in the success-rate denominator, including timing-invalid episodes. Deployment timing runs without concurrent training; the adopted-client supervisor temporarily suspends training and waits for its GPU to become quiet before deployment, then resumes it.

The local LIBERO wrapper clears accumulated object-property samplers before hard model reload. This fixes non-reproducible fixture placement without changing task success predicates. Actual fixture-model, settled simulator-state and initial RGB hashes are compared across methods, rather than assuming seed/init-state equality suffices.

Artifacts are generated below `artifacts/libero-protocols/<label>/`: per-condition config, episodes, traces and summary; stage comparison, initial-condition audit and report. They remain local. The publication checkout extracts two unchanged observation/video helpers from the obsolete evaluator into `libero_observations.py`; implementation hashes therefore differ from earlier live-workspace runs. Do not relabel those older results as a rerun of this publication checkout.

## Controlled completion-release experiments

Deployment can run the real model first and then delay worker release using `--action-min-service-ms` and `--vision-min-service-ms` (both default to zero). `--deployment-fixed-delay` optionally fixes both bootstrap and runtime d in 1..5; omitted preserves adaptive scheduling. These controls are rejected for the algorithm protocol and are part of resume identity.

```bash
python scripts/evaluate_libero_protocol.py \
  --protocol deployment --variant pir2 --output .local/experiments/release-floor \
  --task-ids 0 3 --episodes-per-task 1 --no-video \
  --action-min-service-ms 80 --vision-min-service-ms 160 \
  --deployment-fixed-delay 2
```

The action floor starts at submission and includes worker queueing, feature installation and the real plan RPC. The vision floor starts at its worker RPC timestamp. Calibration uses the same floors; fixed d changes bootstrap and runtime scheduling while retaining deadline misses and per-slot rejection accounting. An over-target raw request is released without additional waiting, with `floor_miss` and raw/total overrun recorded. No extra model forward is introduced.

`raw_completed_s` and `raw_service_seconds` precede the hold. `released_s`/`completed_s` are timestamps recorded just after the hold, **before** metadata recording, Future completion or the VLM publication lock. They are not exact consumer-visible availability timestamps. Use `feature_published` for cache visibility and `action_adopted` for actual controller adoption; S1 Future completion is not separately timestamped. `added_wait_seconds` includes bookkeeping and scheduling as well as sleeping.

This intervention controls a minimum recorded release duration, not raw GPU computation, GPU frequency, or a hardware/quantization speedup. Sleeping releases CPU/GPU execution resources, so its contention pattern differs from slowing down an executing GPU kernel. Report raw and recorded-release durations, floor overruns, actual clocks, control validity, source lineage and AoI coverage separately. Failed or timing-invalid attempts remain evidence and do not silently increase paired-repeat counts.
