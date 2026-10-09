# Two timing protocols

Both protocols use real VLM image features, current robot state, the same checkpoint/data identities and paired initial conditions. They answer different questions and their scores must not be pooled.

| Protocol | Observation/action timing | Purpose |
|---|---|---|
| Algorithm | Image age fixed at 3 ticks (150ms); action release fixed at 1 tick (50ms); current state | Compare behavior at identical simulated delays |
| Deployment | Absolute wall-clock 20Hz control; asynchronous real VLM and action workers | Measure behavior with actual processing/transport delays |

In the algorithm protocol, computation waits do not advance physics. Each method executes the committed prefix, then uses the next legal segment. The initial missing image history is explicitly padded. A GPU contract check adds 100ms wait per call and compares actions and final images; this is a timing check, not a success-rate evaluation.

Deployment keeps control independent from unfinished RPCs. An observation computed early is withheld until its physical tick. Committed commands cannot be overwritten, expired slots are discarded, and buffer exhaustion uses zero Cartesian deltas with the previous gripper command. Flow publishes the unexpired native clean plan; πR² publishes only its newly clean segment. πR² normally uses one DiT evaluation per request; full-buffer recovery is explicitly counted as four bootstrap evaluations plus one normal update.

Warm-up and full request calibration occur outside scored control. Control frequency, policy query frequency and plan publication frequency are separate metrics. Record action/VLM latency, actual cache age, state/image timestamp difference, control lateness, expired/fallback slots and recovery counts. Quantiles are calculated from raw requests, not averaged episode quantiles.

For a hardware scan, the evaluator accepts `--task-ids 0 3 --episodes-per-task 1 --max-steps 300 --no-video`. The default still selects all ten Spatial tasks. A selected subset is labelled `selected_task_subset`, and its expected episode count uses only those tasks. Episodes retain ordinary first-success/environment-termination semantics: `physical_control_seconds` and the deployment `controlled_start_s`/`controlled_end_s` report actual exposure; a 300-tick cap does not promise 15 seconds after an early success. Subset success is not a full-suite result.

Deployment calibration stores all 12 action samples in `calibration.json`, marking the first two as excluded; `action_rpc_seconds` and `action_server_seconds` contain the ten included samples. The action budget remains `min(5, ceil((p95 RPC + 0.005) / 0.05))`, with an explicit over-budget flag. `slow-warmup.json` similarly keeps every visual RPC/server sample and marks the first two excluded. `--slow-warmup-calls` defaults to 12. These are solo warm calibration measurements; concurrent-control latency comes from the episode trace.

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
