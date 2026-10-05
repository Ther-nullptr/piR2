# Two timing protocols

Both protocols use real VLM image features, current robot state, the same checkpoint/data identities and paired initial conditions. They answer different questions and their scores must not be pooled.

| Protocol | Observation/action timing | Purpose |
|---|---|---|
| Algorithm | Image age fixed at 3 ticks (150ms); action release fixed at 1 tick (50ms); current state | Compare behavior at identical simulated delays |
| Deployment | Absolute wall-clock 20Hz control; asynchronous real VLM and action workers | Measure behavior with actual processing/transport delays |

In the algorithm protocol, computation waits do not advance physics. Each method executes the committed prefix, then uses the next legal segment. The initial missing image history is explicitly padded. A GPU contract check adds 100ms wait per call and compares actions and final images; this is a timing check, not a success-rate evaluation.

Deployment keeps control independent from unfinished RPCs. An observation computed early is withheld until its physical tick. Committed commands cannot be overwritten, expired slots are discarded, and buffer exhaustion uses zero Cartesian deltas with the previous gripper command. Flow publishes the unexpired native clean plan; πR² publishes only its newly clean segment. πR² normally uses one DiT evaluation per request; full-buffer recovery is explicitly counted as four bootstrap evaluations plus one normal update.

Warm-up and full request calibration occur outside scored control. Control frequency, policy query frequency and plan publication frequency are separate metrics. Record action/VLM latency, actual cache age, state/image timestamp difference, control lateness, expired/fallback slots and recovery counts. Quantiles are calculated from raw requests, not averaged episode quantiles.

Strict timing validity requires mean frequency within 2% of 20Hz, maximum control lateness at most 5ms and adjacent interval error at most 5ms. All episodes remain in the success-rate denominator, including timing-invalid episodes. Deployment timing runs without concurrent training; the adopted-client supervisor temporarily suspends training and waits for its GPU to become quiet before deployment, then resumes it.

The local LIBERO wrapper clears accumulated object-property samplers before hard model reload. This fixes non-reproducible fixture placement without changing task success predicates. Actual fixture-model, settled simulator-state and initial RGB hashes are compared across methods, rather than assuming seed/init-state equality suffices.

Artifacts are generated below `artifacts/libero-protocols/<label>/`: per-condition config, episodes, traces and summary; stage comparison, initial-condition audit and report. They remain local. The publication checkout extracts two unchanged observation/video helpers from the obsolete evaluator into `libero_observations.py`; implementation hashes therefore differ from earlier live-workspace runs. Do not relabel those older results as a rerun of this publication checkout.
