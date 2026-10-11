# Single-GPU S1/S2 execution tools

The S1/S2 replay tools below are auxiliary GR00T inference and profiling tools. Their workload loads the SO100 replay checkpoint `outputs/pir2-so100-smoke/checkpoint-10` and real prerecorded SO100 observations. The separate opt-in GR00T fusion/integer adapters described below target standard LIBERO Flow and serial πR² inference. Neither replay profiling nor adapter tests replace closed-loop evaluation.

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

这些适配器来自独立的 GR00T-N1.7-LIBERO 算子实验，默认关闭。它们提供 BF16 融合、W8A8/W4A4、调制到量化的融合以及可选的纯 DiT CUDA Graph；依赖固定版本的 Speedup Paradox 整数后端。当前范围是单提交者、冻结权重、B1 的标准 Flow 或流式 πR² 推理。条件投影分组在流式模式下仍被拒绝，同进程并发优化服务也不支持。随机权重动作头测试不代表已训练检查点或量化闭环质量通过。

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

This is a server invocation, not a completed closed-loop protocol. The existing protocol orchestrator retains its BF16 default; select matching precision/configuration on any separately started VLM server as well. `--dit-cuda-graph` is a separate serial-only option for Flow and πR². It captures only the pure DiT forward; rolling buffers, noise, action decoding and the VLM remain eager: first calls and new tensor signatures capture graphs and must be excluded from timing. The current reference Graph helper is not suitable for concurrent S1/S2 capture. Enabled experimental optimizations therefore reject paired `--vlm-device` workers in one process; default BF16 paired workers remain supported. Close the optimization scope before changing weights or using the policy concurrently.

- `--operator-fusion`: BF16 RoPE/RMSNorm, AdaLN modulation, and shared DiT condition/mask preparation. Unsupported attention configurations are rejected at installation.
- `--vision-channels-last`: independently opt into channels-last-3D inputs and weights for the vision patch Conv3D. This can avoid the `SlowDilated3d` fallback; it preserves tensor values but can change BF16 rounding. The original weight storage, layout and hooks are restored on exit. Compare fusion and integer modes with the same explicit layout setting and measure its action differences separately.
- `--norm-modulation-quant`: experimental fusion of non-affine DiT LayerNorm, AdaLN modulation, dynamic scale and low-bit packing. Requires `w8a8`, `w4a4`, `fp8` or `fp4` precision and `--operator-fusion`; shared and per-token conditions are supported. Integer modes use the pinned reference Triton kernel; floating modes use the separate local producers described in [Thor FP8 and FP4](#experimental-thor-fp8-and-fp4). FP32 reduction order differs from native LayerNorm, so this is not a bitwise-preserving option.
- `--residual-norm-quant`: additionally fuse the attention-output residual addition, FFN input LayerNorm and low-bit packing. Requires `--norm-modulation-quant` and a quantized FFN input projection. GR00T has no gate/modulation at this boundary: integer modes use unit gate and zero scale/shift with the reference residual kernel, while floating modes use a separate residual producer. Both preserve the residual equation; the FFN output residual remains separate. Only non-affine LayerNorm and blocks without positional embeddings are supported. Both options default off and restore original forwards on exit; checkpoint quality and GPU savings require target-device validation.
- `--inference-precision {bf16,w8a8,w4a4,fp8,fp4}`: BF16 is the default; low-bit modes are experimental and can change actions. The floating modes require the separate Thor environment below.
- `--quantization-scope transformer`: quantize selected vision/text/DiT transformer projections. `all` additionally selects other BF16 Linear modules, including conditioning and vocabulary projections; it does not skip logits or cache condition calculations.
- `--quantization-category-id N`: with `scope=all`, quantize the fixed embodiment's category projections. Each call checks B1 and the actual ID; omitting the option leaves these projections in BF16. ID 2 is the LIBERO_PANDA mapping in the pinned model, not a universal embodiment ID.
- `--group-conditioning`: with integer precision, fusion and `scope=all`, concatenate the DiT blocks' condition projections and final output-modulation projection using the reference `IntegerProjectionGroup`. They consume the same activated condition in this fused forward, so one preparation and GEMM serves all projections. Shared `[B,D]` and per-token `[B,T,D]` conditions retain their values; this is within one forward, with no reuse across denoising steps. Concatenated weights are packed at installation and require additional packed-weight storage. The option defaults off, requires complete coverage of the group to take effect, and records actual groups in the identity metadata. It does not establish streaming checkpoint or closed-loop quality. Measure both its complete group and full inference; the speedup of these small projections is not the speedup of every model GEMM.
- `--quantization-coverage FILE`: optionally select ordinary Linear sites using a JSON `linears` mapping keyed by module name. Input `shape` fields are needed only for legacy shape-based tactic lookup. Without this file, the selected model inventory is used; category projections are selected separately by `--quantization-category-id`.
- `--quantization-tactics FILE`: optional `{"8": {"module.name": 0}, "4": {...}}` mapping. `--quantization-group-tactics FILE` accepts rows with `bits`, `members` and `tactic`; legacy shape rows additionally require explicit coverage. Tactics are integers 0–7. Unmatched entries use reference tactic 0, not a measured optimum. No past experiment directory is searched automatically.
- `--quantization-shape-tactics FILE`: optional rows such as `[{"bits": 8, "shape": [41, 6144, 1536, true], "tactic": 2}]`, with shape `[M, K, N, has_bias]`. An exact match overrides the module/group tactic for that call; other shapes retain their configured tactic. `M` is the flattened input row count, `K`/`N` are the integer projection's logical widths (a grouped projection uses its combined padded output width). This selects existing reference kernels for native, prepared-input and grouped paths without changing quantization or weights. Load the table before CUDA Graph capture; close and recreate the optimization scope to change it. Calibrate on the target device and checkpoint, then validate full streaming replay; an example row is not a recommended preset.

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

The preset covers 469 executed Linear sites and seven fixed-category projections, with separate choices for W8A8 and W4A4. Calibration used B1/H40 standard Flow4, the pinned model/backend, SDPA, shared fusion, DiT Graph and a channels-last-3D Conv3D layout. The preset does not select that layout; it requires the separate `--vision-channels-last` option. Other input lengths, checkpoints or hardware require remeasurement; the configuration is never selected automatically. The backend performs INT8-by-INT8 or packed INT4-by-INT4 Tensor Core GEMM with INT32 accumulation. Floating scales, bias and BF16 output conversion are fused into its epilogue; floating SDPA is outside the integer GEMM claim.

To recalibrate, collect executed input shapes on a representative full call, then compare reference tactics 0–7 on actual packed activations and weights using CUDA Graph event timing. Retain raw repetitions and check every candidate against the same-precision reference before choosing a tactic. Export module-name entries and exact group-member entries in the formats above. Recheck complete policy outputs after loading the files in a fresh process, and measure BF16, previous and candidate configurations in one randomized, matched GPU session. Isolated repeated-GEMM timings have different cache behavior from a full policy; a microbenchmark winner alone is not evidence of a full-policy improvement.

The service records the configuration, replaced sites and tactic-file hashes in its local identity output. A complete comparison must include online activation preparation and checks, use matching fusion/layout/Graph settings for BF16 and integer variants, and record checkpoint/input identity. Profiled GPU duration sums, isolated GEMM timing, complete Linear timing and synchronized CPU-observation-to-action latency are different metrics. Installation, compilation, packing and Graph capture are reported separately. Old experiment measurements are not automatically performance claims for a different entrypoint or configuration.

Lightweight public CI runs only the configuration and restoration contracts. Explicit local GPU checks use no model download:

```bash
PIR2_GPU_TESTS=1 python -m pytest -q \
  coexecution/test_groot_fusion_gpu.py coexecution/test_quantization_gpu.py \
  coexecution/test_groot_vision_layout_gpu.py
```

Kernel agreement at a fixed precision does not establish quantization quality versus BF16 or robot-task success. Preserve raw evidence outside Git and report unexecuted streaming/closed-loop checks in the PR. Source and license attribution is recorded in [third-party notices](../THIRD_PARTY_NOTICES.md).

### Serial streaming validation

Use a Spatial πR² checkpoint produced by the [training recipe](../docs/libero.md),
then compare the same checkpoint with BF16, fusion BF16, W8A8 and W4A4. Supply
`--variant pir2` to the server above, omit `--group-conditioning`, and regenerate
packed weights from the selected checkpoint. Compare eager execution first, then
enable `--dit-cuda-graph` explicitly. Keep the same Graph setting across precision
comparisons; changed input signatures recapture and require separate warmup.
The Long/Flow tactic preset is not a calibrated Spatial/streaming configuration.

Before model-quality evaluation, the following opt-in check runs the actual
pinned action-head implementation with a small AlternateVLDiT and synthetic
inputs. It exercises bootstrap, changing slide sizes and image delays, fresh
visual features, reset and restoration. Graph checks also change context length and
mask values, preserve earlier outputs, and verify normal and exceptional cleanup.
Fusion is compared with unfused execution
at the same precision; the trace checks INT8/INT4 kernel execution. It bypasses
the VLM and uses random weights, so it establishes neither task accuracy nor
production-model speedup:

```bash
PIR2_GPU_TESTS=1 python -m pytest -q coexecution/test_groot_streaming_gpu.py
```

Explicit checks for the experimental norm/residual options:

```bash
PIR2_GPU_TESTS=1 python -m pytest -q coexecution/test_groot_norm_quant_gpu.py
```

Shape-tactic dispatch has separate opt-in CUDA checks for both precisions, prepared inputs, grouped projections and changing-shape Graph capture:

```bash
PIR2_GPU_TESTS=1 python -m pytest -q coexecution/test_shape_tactics_gpu.py
```

These checks separate native-reduction control-flow equivalence from fused
reduction error, exercise streaming/reset/Graph and restoration, and verify true
integer and fused kernel execution. They do not establish task success. Compare
the existing integer path, norm fusion, then residual fusion at the same precision,
checkpoint, inputs, layout and Graph setting. Report repeated profiled GPU kernel /
memcpy / memset duration sums separately from CUDA-event elapsed time and host
latency; inspect action error against both the previous integer path and BF16.

## Experimental Thor FP8 and FP4

The serial floating adapter requires Thor SM110, eval-mode BF16 CUDA weights,
and the explicitly built [Blackwell dependencies](../docs/environment.md#optional-thor-floating-inference).
It uses the same reversible `GrootOptimizations` interface:

```python
config = OptimizationConfig(
    precision="fp8", scope="all", dit_graph=True, fp8_fast_quant=True
)
with GrootOptimizations(policy, config) as optimized:
    # Run the normal streaming bootstrap / vision / plan interface here.
    coverage = optimized.evidence()
```

The equivalent server flags are `--inference-precision fp8 --quantization-scope all
--dit-cuda-graph --fp8-fast-quant`. Select `fp4` and omit `--fp8-fast-quant` for FP4.
All options default off. Close the scope before changing weights or configuration.

- FP8 uses E4M3 weights and activations with one FP32 scale per tensor.
  `--fp8-fast-quant` selects a two-pass Triton activation packer: partial absmax,
  followed by scale finalization and conversion. Both passes refresh every call,
  including Graph replay. The default uses the pinned reference packer.
- `--fp-shared-inputs` optionally packs the common activation once per known
  QKV/KV or gate/up module invocation. Only fully selected groups participate;
  every member keeps its original quantized weight, scale, GEMM and tactic.
  Reuse requires the identical, unmodified input tensor in that invocation.
  Owner exit (including exceptions) clears reuse state, and direct projection
  calls pack independently. Graph capture records fresh packing for every replay.
  This option supports serial FP8/FP4 inference and defaults off.
- `--fp-grouped` merges fully selected QKV/KV and gate/up projections into one
  GEMM per owner invocation, taking precedence over `--fp-shared-inputs`.
  FP8 repacks the concatenated weight with one tensor scale, so it can change
  quantization error compared with independent weights. FP4 retains block-16
  row scales. Output views are retained except where text Q/K normalization
  requires contiguous inputs. Owner exit and exceptions clear the group state.
- `--fp-swiglu` fuses the text MLP's native BF16 SiLU lookup, multiply and input
  packing. The SiLU output is rounded to BF16 before multiplying, preserving
  the original PyTorch boundary. This differs from the reference's alternative
  SwiGLU rounding policies. FP8 uses a producer/partial-amax pass and one global
  scale/cast pass; FP4 emits packed E2M1 and swizzled block scales directly.
- With `--operator-fusion`, floating modes also accept `--norm-modulation-quant`
  and `--residual-norm-quant`. These connect non-affine DiT LayerNorm/AdaLN and
  attention residual + FFN LayerNorm to floating packed inputs. The residual
  and materialized intermediate boundaries remain BF16, but Triton LayerNorm's
  reduction order differs from native Torch. Measure numerical error against
  both the same-precision unfused path and BF16. These flags default off and do
  not enable the integer packed-buffer path or alter attention's precision.
- FP4 uses E2M1 data and UE4M3 scales per 16 elements in the CUTLASS swizzled
  layout. This is the reference block-scaled FP4 GEMM path, not MXFP8 or an INT4
  buffer. Both floating GEMMs accumulate in FP32 and output BF16; neither
  dequantizes its operands to run a BF16 GEMM.
- FP8 chooses from a small, measured Thor table keyed by exact runtime
  `(M, K, N, has_bias)`, falling back to explicit prefill tactic 2. FP4 uses
  bias-aware tactic 0 for M>1 and tactic 3 for M=1. The pinned FP4 backend's
  nonzero prefill tactics omit the bias epilogue and are deliberately avoided.
  Ada integer tactic files, category selection and condition grouping are rejected.
- `scope=all` covers eligible ordinary Linear projections, including vocabulary
  and conditioning projections. FP8 requires K/N divisible by 16; FP4 by 32.
  CategorySpecificLinear and excluded/unaligned sites remain BF16 and are listed
  in coverage metadata. QK, softmax and PV attention stay floating point.
- Original parameters remain allocated so exit and failure can restore BF16.
  Packed weights, scales, activation workspaces and Graph pools add storage;
  quantization does not imply lower total resident model memory here.

The targeted validation uses checkpoint-500 of Spatial πR² adaptation, B1/H40,
fixed demonstration observations and committed prefixes. GPU tests check packed
references, bias, changing Graph inputs, RNG preservation and restoration:

```bash
PIR2_GPU_TESTS=1 python -m pytest -q coexecution/test_floating_quantization_gpu.py coexecution/test_floating_fusion_gpu.py
```

Keep these checks separate from model replay and LIBERO closed-loop evaluation.
Compare BF16/FP8/FP4 with identical checkpoint, attention, layout, fusion and Graph
settings. Report bootstrap, vision and plan separately, along with full replay;
exclude offline weight packing, compilation and capture from steady timings;
include online activation packing. Action/cache errors
and finite outputs do not establish task quality. No learned-policy quality
threshold or FP4 deployment recommendation is supplied by this adapter.

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
