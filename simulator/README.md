# Interactive πR² queue simulator

This browser tool explores πR² queueing, cached visual features and fixed-rate action execution. It accepts VLM and action-request inference times directly and compares independent resources, shared-GPU contention and serial GPU scheduling. It is an auxiliary explanation tool, not a GR00T-LIBERO evaluator, robot simulator or predictor of task success on new hardware.

Open [index.html](index.html) directly, or preview over loopback from the repository root:

```bash
python3 simulator/serve.py --port 8765
```

The preview serves only this tool and selected stable documentation. It does not expose runtime artifacts, private configuration or the project root. The numerical engine is in `model.js`; interface code is in `app.js`. Extend those modules rather than creating a new simulator per experiment.

## Queue simulation

Enter complete **solo** VLM and action-request service times in milliseconds, plus the control period and camera rate. The timeline can be played, stepped by control tick or scrubbed to inspect individual waiting frames, feature versions, in-flight requests, completed results and committed execution slots. `V0` produces `F0`; `I0a`, `I0b`, etc. consume that version. A small `×` appears only after an output is actually superseded, protected or expired; a result awaiting future execution is not rejected merely because the display window ends.

Output-slot freshness uses **playback time minus the producing request's state-capture time**, not the target execution time or the latest VLM cache. Light/medium/dark age bands indicate at most one control period, one to two periods, and more than two periods; each slot also shows its numerical state age. The latest adopted batch has a separate outline/badge. Completion, publication, commitment, execution and rejection remain separate states: an older action can still be valid. Tooltips include the age of the visual feature actually read by that request and time since result completion. Bootstrap slots have a distinct initial-state style and an age lower bound, because their observation predates the displayed window.

The default adaptive mode models the repository's LIBERO wall-clock scheduling: a seeded set of 12 warm request durations discards the first two, then selects `d = ceil((P95 + margin) / control_period)`, capped at the selected maximum of at most 5. A late result can increase subsequent `d`. Each request reserves its committed prefix and produces `d` actions for ticks `[r+d, r+2d)`. Publication rejects individual expired/protected slots and preserves remaining valid output. Completed results wait for the control thread's next tick.

For example, a deterministic 80 ms action request at a 50 ms control period selects `d=2` with the default 5 ms margin. The first result executes at 100 and 150 ms while the controller uses bootstrap actions at 0 and 50 ms. Fixed-d mode is an explicit contrast for studying insufficient budgets, buffer exhaustion and fallback; it is not the default πR² behavior.

`queueDefaults`, `simulatePolicyQueues(config)` and `policyQueueSnapshot(result, timeMs)` expose the reusable model. JSON exports include inputs, calibration, all events, feature dependencies, per-slot adoption/execution and the selected snapshot. Expiration metrics distinguish in-window publication from final draining; genuine future slots remain censored at the window boundary.

The queue view uses one latest waiting camera frame. It samples bounded uniform service-time jitter and treats the supplied action time as the complete solo request cost, including any preprocessing, feature installation or transport that the user intends to include. It starts with an initialized visual cache and configurable clean action slots. It does not run a neural policy, evaluate action values, or add neural bootstrap/recovery computation. The 40-position rolling-buffer display shows scheduling regions, not denoised tensors. The official robot deployment's continuous-worker timing differs from this repository's tick-aligned wrapper; see [timing protocols](../docs/timing-protocols.md).

### Single-GPU contention

The queue view supports three resource modes through `computeMode`:

- `independent`: workers can overlap without slowing one another; this is the original queue model and remains the default.
- `shared`: both workers may compute on one GPU. During GPU overlap, their remaining GPU work advances at `1 / (1 + slowdownA)` and `1 / (1 + slowdownV)` of solo speed. Each worker immediately returns to solo speed when the other leaves the GPU.
- `serial`: GPU tasks run without overlap or preemption; a ready action task wins a simultaneous task-boundary tie. Non-GPU portions may still overlap.

`gpuShareA` and `gpuShareV` split each sampled solo request into a non-GPU prefix and a GPU work budget. They default to 1; lower the fraction when CPU work or transport is included in the entered request time. Those portions are not multiplied by the GPU slowdown factor. This prefix placement is a modeling assumption, not a reconstruction of a real framework's kernels or transfers.

For an isolated pair of full-GPU tasks starting together, action work of 20 ms and VLM work of 100 ms with both slowdown coefficients set to 1 finish at 40 ms and 120 ms respectively. Only the first 40 ms overlap. Multiplying both whole requests by 2 would incorrectly predict 200 ms for the VLM.

Resource segments and per-job accounting distinguish non-GPU work, GPU work, shared-GPU slowdown and GPU queue waiting. Their actual completion times drive feature publication, action deadlines, adaptive `d`, and execution-slot expiration. Active work drains after the control window; no new camera/request work is introduced during draining. Initial calibration deliberately uses solo request samples; it is not a contended warm-up measurement. Runtime misses can therefore increase `d` after contention begins.

Slowdown coefficients are editable assumptions or values to fit from measured solo/concurrent traces. Do not enter an already-contended latency and then apply the same slowdown again. This model does not derive occupancy or bandwidth contention from a GPU name, and simulated overlap does not prove CUDA kernel overlap. See [measurement boundaries](../docs/coexecution.md) and the [CUDA asynchronous-execution guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html).

## Checks

```bash
node --test simulator/model.test.cjs
npm ci --prefix tools/ui-check
tools/ui-check/node_modules/.bin/playwright install chromium
node simulator/browser-check.cjs
```

Browser dependencies are development-only. Playwright 1.51.1 was used with the original Ubuntu 20.04 workspace; newer hosts may require a reviewed dependency update. Browser checks cover layout, controls and exported records, not CPU model inference. Screenshots and downloads are written to ignored `artifacts/simulator/`. Lightweight CI runs only the Node numerical contracts; browser checks are explicit local validation.
