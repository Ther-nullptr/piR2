# Single-GPU S1/S2 execution tools

The S1/S2 replay tools below are auxiliary GR00T inference and profiling tools. Their workload loads the SO100 replay checkpoint `outputs/pir2-so100-smoke/checkpoint-10` and real prerecorded SO100 observations. The separate opt-in GR00T fusion/integer adapters described below target standard LIBERO Flow inference. Neither replay profiling nor adapter tests replace closed-loop evaluation.

The two workers run a real VLM (S2) and a rolling DiT action head (S1) on one physical GPU. Feature handoff uses completion events and reader leases. Optional fusion, static input handling and CUDA Graph variants are explicit A/B choices. Numerical checks restore a paired starting buffer and compare with the eager path; periodic replay also records deadlines and feature ages.

## Commands

Prepare the pinned GR00T source and GPU environment as described in [environment setup](../docs/environment.md). Run the separate SO100 demonstration training entrypoint only if you need this replay checkpoint. Select an idle GPU; the launcher performs a preflight occupancy check:

```bash
PIR2_GPU=0 bash scripts/run_single_gpu_baseline.sh measure
PIR2_GPU=0 bash scripts/run_single_gpu_baseline.sh runtime
PIR2_GPU=0 bash scripts/run_single_gpu_baseline.sh protocols
```

Run one command at a time. Use `PIR2_OUTPUT` for a new result directory. Trace modes require Nsight Systems on PATH or an explicit `PIR2_NSYS` executable. The `test_*_gpu.py` modules require CUDA and are opt-in; CI only runs timeline/launch-attribution accounting tests.

See [measurement definitions](../docs/coexecution.md). Raw JSON, plots, SQLite exports, logs and Nsight traces stay local. These tools do not by themselves demonstrate LIBERO accuracy, physical control stability, quantization quality or performance on another GPU.

## Experimental GR00T fusion and integer inference

这些适配器来自独立的 GR00T-N1.7-LIBERO 算子实验，默认关闭。它们提供 BF16 融合、W8A8/W4A4、调制到量化的融合以及可选 DiT CUDA Graph；依赖固定版本的 Speedup Paradox 整数后端。当前范围是单提交者、冻结权重、B1 的标准 Flow 推理。流式 πR²、并发调用和量化闭环质量尚未验证，服务入口拒绝把这些实验选项用于 `--variant pir2`。

The modules provide instance-local, reversible adaptations. Weight packing happens during installation; activation scales, quantization and packing run online. Integer GEMM uses INT32 accumulation and a fused scale/bias/BF16 output epilogue. Compatible QKV/KV and gate/up projections share preparation. The native dispatch can combine activation preparation with GELU or SiLU-times-up; AdaLN modulation can produce the packed input directly. Fused SDPA remains floating point. No checkpoint, convolution layout, sampling schedule, observation or VLM-cache policy is changed by enabling quantization.

Prepare the pinned `robotics-kernels` source and its GPU dependencies using [environment setup](../docs/environment.md). Use the existing `serve_libero_protocol.py` entrypoint with explicit options, for example:

```bash
export PYTHONPATH="$PWD:$PWD/upstream/learning/Isaac-GR00T:$PWD/third_party/robotics-the-speedup-paradox/src"
python scripts/serve_libero_protocol.py \
  --checkpoint models/GR00T-N1.7-LIBERO/libero_10 \
  --variant flow --role action --output .local/integer-server \
  --operator-fusion --inference-precision w8a8 \
  --quantization-scope all --quantization-category-id 2
```

This is a server invocation, not a completed closed-loop protocol. The existing protocol orchestrator retains its BF16 default; select matching precision/configuration on any separately started VLM server as well. `--dit-cuda-graph` is a separate serial-only option: first calls and new tensor signatures capture graphs and must be excluded from timing. The current reference Graph helper is not suitable for concurrent S1/S2 capture. Enabled experimental optimizations therefore reject paired `--vlm-device` workers in one process; default BF16 paired workers remain supported. Close the optimization scope before changing weights or using the policy concurrently.

- `--operator-fusion`: BF16 RoPE/RMSNorm, AdaLN modulation, and shared DiT condition/mask preparation. Unsupported attention configurations are rejected at installation.
- `--inference-precision {bf16,w8a8,w4a4}`: BF16 is the default; W4A4 is experimental and can substantially change actions.
- `--quantization-scope transformer`: quantize selected vision/text/DiT transformer projections. `all` additionally selects other BF16 Linear modules, including conditioning and vocabulary projections; it does not skip logits or cache condition calculations.
- `--quantization-category-id N`: with `scope=all`, quantize the fixed embodiment's category projections. Each call checks B1 and the actual ID; omitting the option leaves these projections in BF16. ID 2 is the LIBERO_PANDA mapping in the pinned model, not a universal embodiment ID.
- `--group-conditioning`: with integer precision, fusion and `scope=all`, concatenate the DiT blocks' condition projections and final output-modulation projection using the reference `IntegerProjectionGroup`. They consume the same activated condition in this fused forward, so one preparation and GEMM serves all projections. Shared `[B,D]` and per-token `[B,T,D]` conditions retain their values; this is within one forward, with no reuse across denoising steps. Concatenated weights are packed at installation and require additional packed-weight storage. The option defaults off, requires complete coverage of the group to take effect, and records actual groups in the identity metadata. It does not establish streaming checkpoint or closed-loop quality. Measure both its complete group and full inference; the speedup of these small projections is not the speedup of every model GEMM.
- `--quantization-coverage FILE`: optionally select recorded executed Linear sites using a JSON `linears` mapping with input `shape` fields. Without this file, the selected model inventory is used.
- `--quantization-tactics FILE`: optional `{"8": {"module.name": 0}, "4": {...}}` mapping. `--quantization-group-tactics FILE` accepts rows with `bits`, `members` and `tactic`; legacy shape rows additionally require explicit coverage. Tactics are integers 0–7. Unmatched entries use reference tactic 0, not a measured optimum. No past experiment directory is searched automatically.

For an already loaded eval-mode BF16 CUDA policy, the same public interface is:

```python
from coexecution.groot_optimization import GrootOptimizations, OptimizationConfig

config = OptimizationConfig(precision="w8a8", fusion=True, scope="all", category_id=2)
with GrootOptimizations(policy, config) as optimized:
    actions, info = policy.get_action(
        observation,
        options={"force_nonstreaming": True, "num_inference_timesteps": 4},
    )
    coverage = optimized.evidence()
# Original module forwards/processors are restored, including on failure.
```

An optional RTX 6000 Ada preset for the pinned `GR00T-N1.7-LIBERO/libero_10` model is provided as [coverage](../configs/quantization/groot-libero10-ada/coverage.json), [single-projection tactics](../configs/quantization/groot-libero10-ada/tactics.json) and [group tactics](../configs/quantization/groot-libero10-ada/group-tactics.json). These are executable configuration inputs, not timing reports. Add the following options to the standard Flow invocation above, selecting `w4a4` instead for INT4:

```bash
--inference-precision w8a8 --operator-fusion --dit-cuda-graph \
--quantization-scope all --quantization-category-id 2 --group-conditioning \
--quantization-coverage configs/quantization/groot-libero10-ada/coverage.json \
--quantization-tactics configs/quantization/groot-libero10-ada/tactics.json \
--quantization-group-tactics configs/quantization/groot-libero10-ada/group-tactics.json
```

The preset covers 469 executed Linear sites and seven fixed-category projections, with separate choices for W8A8 and W4A4. Calibration used B1/H40 standard Flow4, the pinned model/backend, SDPA, shared fusion, DiT Graph and an experimental channels-last-3D Conv3D layout. The preset does not change that layout or model weights. Other input lengths, checkpoints or hardware require remeasurement; the configuration is never selected automatically. The backend performs INT8-by-INT8 or packed INT4-by-INT4 Tensor Core GEMM with INT32 accumulation. Floating scales, bias and BF16 output conversion are fused into its epilogue; floating SDPA is outside the integer GEMM claim.

To recalibrate, collect executed input shapes on a representative full call, then compare reference tactics 0–7 on actual packed activations and weights using CUDA Graph event timing. Retain raw repetitions and check every candidate against the same-precision reference before choosing a tactic. Export module-name entries and exact group-member entries in the formats above. Recheck complete policy outputs after loading the files in a fresh process, and measure BF16, previous and candidate configurations in one randomized, matched GPU session. Isolated repeated-GEMM timings have different cache behavior from a full policy; a microbenchmark winner alone is not evidence of a full-policy improvement.

The service records the configuration, replaced sites and tactic-file hashes in its local identity output. A complete comparison must include online activation preparation and checks, use matching fusion/layout/Graph settings for BF16 and integer variants, and record checkpoint/input identity. Profiled GPU duration sums, isolated GEMM timing, complete Linear timing and synchronized CPU-observation-to-action latency are different metrics. Installation, compilation, packing and Graph capture are reported separately. Old experiment measurements are not automatically performance claims for a different entrypoint or configuration.

Lightweight public CI runs only the configuration and restoration contracts. Explicit local GPU checks use no model download:

```bash
PIR2_GPU_TESTS=1 python -m pytest -q \
  coexecution/test_groot_fusion_gpu.py coexecution/test_quantization_gpu.py
```

Kernel agreement at a fixed precision does not establish quantization quality versus BF16 or robot-task success. Preserve raw evidence outside Git and report unexecuted streaming/closed-loop checks in the PR. Source and license attribution is recorded in [third-party notices](../THIRD_PARTY_NOTICES.md).

## Measured LIBERO queue reports

`queue_report` is a separate offline consumer of real LIBERO closed-loop trace artifacts. It loads no model and launches no GPU work. Install Matplotlib for static exports, and make Node.js available for the optional illustrative queue-model comparison:

```bash
python -m coexecution.queue_report \
  --input .local/experiments/queue-scan \
  --output .local/experiments/queue-report
python -m pytest -q coexecution/test_queue_report.py
```

The input contains `layout/condition/{hardware.json,telemetry.jsonl,calibration.json,slow-warmup.json,episodes.jsonl,task*-episode*-trace.jsonl}`; a single condition directory also works. Each completed control window contributes to the report. The output includes a self-contained offline `index.html`, PNG/SVG capacity and hardware curves, representative per-condition trace plots, `summary.json`, `summary.csv`, complete replay data, and an explicitly labelled `simulation-comparison.json`. Smoke, clock-probe and profile conditions are excluded by default; `--include-diagnostics` enables diagnostic previews.

The viewer replays camera waiting/in-flight/latest-feature state, S1 host requests and completed results awaiting adoption, and committed/future/bootstrap/fallback action slots with their producing request. Real capacity is the reciprocal of the mean complete solo RPC duration, never inferred from a requested clock. Hardware summaries report effective per-role clocks from telemetry. Requested settings need not take effect equally on different GPUs.

Latency percentiles use requests admitted and completed within control windows. End-crossing requests remain censored; outputs targeting future control ticks are not expired. Energy integrates board-power samples only within each control window and reports NVML energy-counter differences and sample coverage separately. Board energy includes rendering or other work on those GPUs. Different success-terminated durations remain visible, along with paired initial-state hashes. Host RPC overlap does not establish CUDA kernel overlap, and selected episodes do not establish full-suite success. The embedded simulator is a separate illustrative reference, with explicit sharing assumptions and its original synthetic calibration/drain accounting.

### Feature consumption accounting

```bash
python -m coexecution.feature_usage \
  --report .local/experiments/queue-report/report-data.json \
  --output .local/experiments/feature-usage \
  --raw-root .local/experiments/queue-scan
```

The optional raw-root audit verifies published and consumed source identities against the original traces. Counts always use the actual feature sequence, never the nearest publication to an RPC timestamp: a request can hold an older snapshot while a new version arrives during installation. Outputs contain per-feature DiT reads, distinct directly executed producer requests, execution ticks, zero/one/multiple-read distributions and count-weighted valid-episode summaries. Bootstrap and right-censored final features are separate; unknown sources explicitly mark accounting incomplete. Direct execution does not represent all causal contributions through the neural rolling buffer.

Use `queue_report --no-figures` to export replay without Matplotlib. The optional Source Han webfont remains a separately downloaded local asset; generated reports inline it with its license when present.
