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

### Module ownership / 模块职责

GR00T optimization implementations live under `coexecution/optimization`:

| Path | Responsibility / 职责 |
| --- | --- |
| `config.py`, `tactics.py`, `operators.py` | GPU-free configuration, tactic parsing and operator inventory/profiles; 配置与算子选择 |
| `capabilities.py` | Selected-backend preflight and runtime identity; 后端能力与来源 |
| `session.py` | Reversible installation, restoration and evidence; 安装生命周期 |
| `groot/projections.py` | Shared model inventory; 共用模型节点发现 |
| `groot/integer.py`, `groot/floating.py` | Model projection replacement; 模型适配 |
| `groot/*_connections.py` | GR00T norm/residual producer boundaries; 模型连接 |
| `execution/graph.py` | Shared pure-DiT graph capture; 共用 Graph |
| `execution/cache.py` | Invocation-local cache ownership and serial guards; 调用边界 |
| `execution/preparation.py`, `execution/tokenization.py` | Scoped input routing and exact-text token preparation; 输入准备 |
| `execution/guards.py`, `execution/patching.py` | Host metadata guards and instance-local callable adaptation; 状态校验 |
| `groot/vlm.py` | Static VLM metadata and optional complete VLM capture; 完整 VLM 执行 |
| `backends/integer/linear.py` | Integer packing, GEMM and dispatch; 整数算子 |
| `backends/thor/linear.py`, `backends/thor/packing.py` | Thor floating formats and execution; 浮点算子 |

Configuration and default scope imports do not load Torch or GPU kernels. A
session imports only its selected implementation. Backends do not import LIBERO
schedulers or RPC services; integer and floating implementations share model
inventory and Graph handling, not packed formats. Actual CUDA sources and the
selected shared Triton primitives now live under `coexecution/kernels`: `common`,
`ampere_ada` and `blackwell`. The official `vendor/cutlass` submodule is the default
header dependency; a compatible `CUTLASS_ROOT` can select existing headers.
Model-specific connections remain in `optimization/groot`; Python entrypoints do not resolve an
external robotics source tree or prebuilt shared-library path.

原有 `coexecution.groot_optimization`、`quantization`、`floating_quantization`
等导入路径作为兼容转发保留；新增实现使用上述目录。已有默认精度、packing、scale、
tactic 和融合公式保留；新增 profile 与模式参数用于显式选择。不同硬件共用模型与
调度语义，不复制整套推理服务。

Optimization contracts are in `tests/contracts`, explicit CUDA tests in
`tests/gpu`, and reusable fixtures in `tests/helpers`. GPU tests require
`PIR2_GPU_TESTS=1`; public CI checks configuration, scheduling and accounting only.
The separate SO100/queue tools retain their existing modules and tests.

### Common policy runtime / 公共推理路径

Common runtime options apply to the same BF16 or quantized CUDA policy:
`--input-preparation` routes only consumed tensor fields to each model component;
`--tokenizer-cache` caches exact expanded text after fresh image processing;
`--static-vlm` prepares fixed text/grid/mask metadata without device-side scalar
reads; `--vlm-cuda-graph` captures the complete VLM and requires `--static-vlm`.
`--image-array-preparation` replaces the pinned GR00T processor's PIL conversion
with owned contiguous uint8 RGB arrays. Its original conversation, chat template,
crop/resize, state and Qwen collation still run; every image buffer is fresh.
Scope entry checks pinned GR00T/Transformers source and live method code and
admits the stock Qwen2VL fast or slow pixel processor. Replacement processors or
methods require closing the scope; unsupported implementations fail admission.
Heights of 1 or 3 pixels are rejected because ndarray channel inference would be
ambiguous; normal task images retain their original preprocessing.
`--compiled-host-guards` compiles the exact Python guard source with Cython and
binds its predicate methods to each guard instance. Discovery and all frozen
execution checks retain the Python reference behavior, including tracked weight
mutations, methods, hooks, closures, defaults, configuration and topology.
Images, robot state and VLM features are recomputed for every observation.
These options default off. Graph and metadata caches are bounded, and new input
signatures require warmup. Runtime changes through `optimized.set_runtime(...)`
retain the selected precision controller and the separate DiT graph cache.

Compiled guards require CPython, a working host C compiler and the optional
`Cython==3.1.2` and `setuptools` packages in the model environment. Install them
with `python -m pip install Cython==3.1.2 setuptools==84.0.0` when using that flag.
Startup builds a native extension from the unchanged guard source; generated
source, binaries and build logs stay in `~/.cache/pir2/host-guards`, or the directory
specified by `PIR2_HOST_GUARD_CACHE`. The cache is locked across processes and
keyed by exact source, CPython ABI, platform, compiler identity and build settings.
Evidence records those inputs and the compiled binary hash. Missing dependencies
or build failures fail explicitly. Normal imports, CLI help and the default
Python backend never invoke a compiler.

Use `python -m coexecution.libero.measurement --help` for paired measurements of
raw-observation policy wall time, prepared-input complete-model time, CUDA event
spans and separately collected GPU activity. These timing scopes overlap and
must not be added or subtracted as a CPU/GPU decomposition. The module validates
same-precision actions and checks that timed graph calls replay without capture.
It requires CUDA, the official Spatial task checkpoint and its dataset; profiler
output and numerical archives stay local.
The cumulative modes are `baseline`, `prepared`, `tokenized`, `static`, `graph`,
`images` and `compiled`; the last two add image arrays and compiled guards to the
preceding mode. Keep `--action-cuda-graph` and precision settings fixed for each
paired run. Use the same runtime flags for BF16, W8A8 and W4A4 comparisons.
The default mode list remains `baseline` through `graph`; explicitly select
`--modes graph images compiled` to measure the new host optimizations.
Same-precision action checks use `baseline` when requested, otherwise the first
requested mode (`graph` in this example); results record that reference explicitly.

For independent quantization debugging, select `--measurement-boundary model-only`.
This prepares and owns the selected real observation input trees once per
precision/runtime block (bounded by `--max-prepared-observations`, default 16),
collects raw-policy action/RNG references, then warms and replays those resident
inputs. Each recorded call recomputes the complete VLM and all four Flow steps;
no neural output is cached. Observation selection, image conversion, collation,
input copies, RPC, simulation, action decoding, numerical checks and file writes
are outside the model timer. Raw/prepared comparisons remain available with the
default `--measurement-boundary comparison`; their separate wall probes are not
an additive decomposition. Model-only groups runtime modes into blocks rather
than interleaving preparation and raw calls with recorded model calls.

`coexecution.libero.inference.PreparedInferenceEngine` exposes `vlm`, `action`
and `full` on caller-owned resident inputs. Preparation and decoding are separate
adapters, and decoder state is opaque to the engine. The engine returns the
upstream output directly and retains no outputs: snapshot retained results
outside timing according to the component's lifetime contract. Callers establish
CUDA, inference mode, stream readiness and serial ownership; CUDA completion is
measured at the runner boundary. Imports and CLI help do not load a model.
Preparation copies mutable inputs before upstream preparation registers static
metadata. Registered metadata identities stay intact and retain their mutation
guards; keep the corresponding optimization scope and bindings alive throughout
the prepared replay. No input copy or metadata registration occurs in the engine.

Clean latency runs do not allocate timing events or profiler scopes. Explicit
`--cuda-events` adds CUDA span and host-return detail; `--profile-rounds N` runs
a separate instrumented pass with actual request/VLM/action annotations. In
model-only profiles, the fixed input cohort is prepared before profiling starts.
Samples retain precision, observation identity, request identity and seed; trace
requests retain the corresponding annotation and input identity. Instrumented
GPU work, CUDA span and complete host wall latency remain different metrics.
The runner checks exact same-precision decoded actions and CPU/CUDA RNG against
raw policy calls before accepting model-only results. These are execution gates,
not LIBERO success or quantization-quality claims.

Run `PIR2_GPU_TESTS=1 python -m pytest tests/gpu/test_prepared_inference_gpu.py`
in the pinned CUDA environment for actual fresh-input ownership, original model
output/RNG equality and output-lifetime gates, including static VLM and VLM graph
composition with input separation. Host CI uses interface objects,
never CPU neural inference.

Run `PIR2_NATIVE_TESTS=1 python -m pytest tests/native/test_compiled_host_guards.py`
for native predicate/mutation gates, and
`PIR2_GPU_TESTS=1 python -m pytest tests/gpu/test_image_preparation_gpu.py` in the
pinned environment for actual image/collator equality and ownership checks.
The latter performs deployment preprocessing only; complete policy action/RNG
gates still require CUDA.

`--action-cuda-graph` additionally captures S1 visual/state conditioning and the
unchanged four-step Flow action head. It supports BF16, W8A8 and W4A4, requires
non-streaming H40, and preserves the policy's decoded horizon and output API.
The adapter calls the pinned upstream algorithm with one fresh explicit noise
tensor per request, including the first capture. Features, state, embodiment ID
and masks are copied freshly; validated integer image delay is part of the
bounded eight-entry graph signature along with shape, dtype, stride and stream.
Inpaint, rolling/streaming requests and unknown options or hooks are rejected.
Scheduler state, cache timestamps, noise generation and policy decoding remain
outside the graph. Returned tensors have independent storage.

When both graph flags are selected, the action graph supersedes the inner
`--dit-cuda-graph`; evidence reports this explicitly. Runtime mode changes close
and recapture action graphs after installing new serial boundaries. Warm each
signature before measuring. Capture failures propagate and the adapter stays
installed until explicit scope close, with no automatic eager fallback. Close
before replacing weights, methods or attention settings; tracked mutations are
rejected, while untracked `.data`/foreign writes are outside the frozen scope
contract. CUDA callers must establish input/weight readiness on their stream.

Service audits preserve `python_forward_counts` as physical hook observations.
For the larger graph, `forward_counts.dit` is the logical NFE proved by four DiT
calls in every captured Flow body and explicit replay counts; `dit_count_source`
and `action_graph` identify that source. Warmup/capture hooks and logical request
work are separate. The measurement and profiler modules require one whole-action
replay and four logical evaluations per timed call. Select the same graph flag
for BF16 and quantized comparisons; these execution checks establish neither
quantization quality nor closed-loop success.

`--action-kv-reuse` additionally reuses each cross-attention layer's K/V outputs
within that unchanged Flow4 body. It requires `--action-cuda-graph`, non-streaming
H40/NFE4, the pinned interleaved AlternateVLDiT, frozen eval execution and an
admitted SDPA processor. BF16, W8A8 and W4A4 support both standard
`AttnProcessor2_0` and `PreparedAttention`; operator fusion is independent. Q,
self-attention, masks, state/noise and the four denoising evaluations stay fresh.
Installed integer grouping, tactics and producer packs remain in effect.

The scope starts empty for every warmup/captured body. Each layer's first K/V
computation is captured, so every replay recomputes it from the newly copied
observation. The remaining three evaluations use those outputs. All Python
K/V references are released on body exit, including exceptions, independently
of per-DiT projection-cache resets. There is no reuse across observations or
reliance on feature pointers as freshness evidence. Closing restores projection
forwards; changing the flag through `set_runtime(action_kv_reuse=...)` closes and
recaptures action graphs while keeping precision and weights resident.

For paired timing, explicitly select `--modes compiled kv_reuse` with
`--action-cuda-graph` and the same precision/fusion/tactic flags for both modes.
Only K/V reuse differs between those two modes. The measurement command owns
the switch, so it rejects a global `--action-kv-reuse`; the standalone runtime
and service commands accept that flag. Existing default measurement modes are
unchanged. For example:

```bash
python -m coexecution.libero.measurement \
  --checkpoint /path/to/libero_spatial --dataset /path/to/groot-libero-spatial \
  --output .local/kv-paired --precisions bf16 w8a8 w4a4 \
  --modes compiled kv_reuse --action-cuda-graph --operator-fusion \
  --observations 3 --rounds 20 --warmup 2
```

The comparison boundary randomizes mode order per round using identical inputs
and seeds. `--measurement-boundary model-only` retains its resident mode blocks;
record or reverse block order separately when evaluating timing drift. Warm
every shape/stream signature before timing. `action_graph.kv_reuse` records
per-layer captured-body proofs and expected/executed/avoided projection counts
from explicit replay counts. These count K/V projection evaluations, not GEMM
or kernel launches: an installed K/V group can execute one GEMM for two
projections. Python projection counters describe successful warmup/capture
bodies only; replay bypasses those counters and hooks. Timing audits require
the proof and exact work deltas. No speedup or task-quality claim follows from
these execution counts alone.

Run `PIR2_GPU_TESTS=1 python -m pytest tests/gpu/test_action_graph_gpu.py` in the
pinned CUDA environment for full-action/RNG, freshness, stride/stream, output
lifetime, failure, frozen-execution and restoration gates.

The opt-in numerical suite is
`PIR2_GPU_TESTS=1 python -m pytest tests/gpu/test_policy_vlm_gpu.py` in the pinned
CUDA environment. This checks execution behavior, not closed-loop task success.
For full Spatial evaluation, see the declared-condition runner in
[LIBERO](../docs/libero.md). Separate action and VLM processes can share an
explicitly declared GPU. The opt-in `--server-layout paired` service uses two
independent policy instances/streams and coordinates cold VLM/DiT capture across
all endpoint CUDA submissions. See [transport and placement](../docs/timing-protocols.md)
for bounded resident feature leases and the paired NumPy control.

这些适配器来自独立的 GR00T-N1.7-LIBERO 算子实验，默认关闭。它们提供 BF16 融合、W8A8/W4A4、调制到量化的融合以及可选的纯 DiT CUDA Graph；整数后端源码来自固定版本的 Speedup Paradox，现由本仓集中管理。当前范围是每个策略单提交者、冻结权重、B1 的标准 Flow 或流式 πR² 推理。条件投影分组在流式模式下仍被拒绝。配对服务使用独立策略实例与流，首次 VLM/DiT capture 通过进程共享协调器排他执行；配对 whole-action graph 仍被拒绝。随机权重动作头测试不代表已训练检查点或量化闭环质量通过。

The modules provide instance-local, reversible adaptations. Weight packing happens during installation; activation scales, quantization and packing run online. Integer GEMM uses INT32 accumulation and a fused scale/bias/BF16 output epilogue. Compatible QKV/KV and gate/up projections share preparation. The native dispatch can combine activation preparation with GELU or SiLU-times-up; AdaLN modulation can produce the packed input directly. Fused SDPA remains floating point. No checkpoint, convolution layout, sampling schedule, observation or VLM-cache policy is changed by enabling quantization.

Verify the repository-owned `robotics-kernels` manifest and prepare the GPU environment using [environment setup](../docs/environment.md). Use the existing `serve_libero_protocol.py` entrypoint with explicit options, for example:

```bash
export PYTHONPATH="$PWD:$PWD/upstream/learning/Isaac-GR00T${PYTHONPATH:+:$PYTHONPATH}"
python scripts/serve_libero_protocol.py \
  --checkpoint models/GR00T-N1.7-LIBERO/libero_10 \
  --variant flow --role action --output .local/integer-server \
  --operator-fusion --inference-precision w8a8 \
  --quantization-scope all --quantization-category-id 2
```

This is a server invocation, not a completed closed-loop protocol. The existing protocol orchestrator retains its BF16 default; select matching precision/configuration on any separately started VLM server as well. `--dit-cuda-graph` requires serial submissions to each Flow or πR² policy. It captures only the pure DiT forward; rolling buffers, noise and action decoding stay outside that graph. First calls and new tensor signatures capture graphs and must be excluded from steady timing. In paired service mode, a process-wide submission coordinator gives cold VLM/DiT graph calls exclusive CUDA access without changing the pinned graph implementation. Steady handlers can overlap. Close the optimization scope before changing weights or using the same policy concurrently.

- `--operator-fusion`: enable BF16 fusion. The default mode `all` retains RoPE/RMSNorm, AdaLN modulation and shared DiT condition/mask preparation. Unsupported attention configurations are rejected at installation.
- `--operator-fusion-mode {vision_rope,text_rope,rope,rmsnorm,rope_rms,all}`: select the BF16 component set while fusion is enabled. `rope` selects both RoPE paths; only `all` installs DiT pointwise preparation. A selective mode requires `--operator-fusion`; condition grouping and norm/residual producer flags require `all`. The option defaults to `all` and does not enable fusion by itself.
- `--operator-profile FILE`: explicitly override operator implementations for one architecture and precision. Omitted families retain their existing choices; no candidate profile is selected automatically. It requires fusion or a low-bit precision and validates before model patches. See the inventory and profile contract below.
- `--vision-channels-last`: independently opt into channels-last-3D inputs and weights for the vision patch Conv3D. This can avoid the `SlowDilated3d` fallback; it preserves tensor values but can change BF16 rounding. The original weight storage, layout and hooks are restored on exit. Compare fusion and integer modes with the same explicit layout setting and measure its action differences separately.
- `--norm-modulation-quant`: experimental fusion of non-affine DiT LayerNorm, AdaLN modulation, dynamic scale and low-bit packing. Requires `w8a8`, `w4a4`, `fp8` or `fp4` precision and `--operator-fusion`; shared and per-token conditions are supported. Integer modes use the pinned reference Triton kernel; floating modes use the separate local producers described in [Thor FP8 and FP4](#experimental-thor-fp8-and-fp4). FP32 reduction order differs from native LayerNorm, so this is not a bitwise-preserving option.
- `--residual-norm-quant`: additionally fuse the attention-output residual addition, FFN input LayerNorm and low-bit packing. Requires `--norm-modulation-quant` and a quantized FFN input projection. GR00T has no gate/modulation at this boundary: integer modes use unit gate and zero scale/shift with the reference residual kernel, while floating modes use a separate residual producer. Both preserve the residual equation; the FFN output residual remains separate. Only non-affine LayerNorm and blocks without positional embeddings are supported. Both options default off and restore original forwards on exit; checkpoint quality and GPU savings require target-device validation.
- `--inference-precision {bf16,w8a8,w4a4,fp8,fp4}`: BF16 is the default; low-bit modes are experimental and can change actions. The floating modes require the separate Thor environment below.
- `--quantization-scope transformer`: quantize selected vision/text/DiT transformer projections. `all` additionally selects other BF16 Linear modules, including conditioning and vocabulary projections; it does not skip logits or cache condition calculations.
- `--quantization-component {all,vlm,dit}`: intersect the selected scope and optional coverage with S2 backbone or S1 action-head projections. S1 includes `action_head.vl_self_attention`. The default is `all`; this selects quantization coverage, not the scope of general BF16 fusion. Category, condition-group and DiT norm-producer options cannot be combined with `vlm` alone. The same selection applies to integer and Thor floating precision.
- `--quantization-category-id N`: with `scope=all`, quantize the fixed embodiment's category projections. Each call checks B1 and the actual ID; omitting the option leaves these projections in BF16. ID 2 is the LIBERO_PANDA mapping in the pinned model, not a universal embodiment ID.
- `--group-conditioning`: with integer precision, fusion and `scope=all`, concatenate the DiT blocks' condition projections and final output-modulation projection using the reference `IntegerProjectionGroup`. They consume the same activated condition in this fused forward, so one preparation and GEMM serves all projections. Shared `[B,D]` and per-token `[B,T,D]` conditions retain their values; this is within one forward, with no reuse across denoising steps. Concatenated weights are packed at installation and require additional packed-weight storage. The option defaults off, requires complete coverage of the group to take effect, and records actual groups in the identity metadata. It does not establish streaming checkpoint or closed-loop quality. Measure both its complete group and full inference; the speedup of these small projections is not the speedup of every model GEMM.
- `--quantization-coverage FILE`: optionally select ordinary Linear sites using a JSON `linears` mapping keyed by module name. Input `shape` fields are needed only for legacy shape-based tactic lookup. Without this file, the selected model inventory is used; category projections are selected separately by `--quantization-category-id`.
- `--quantization-tactics FILE`: optional `{"8": {"module.name": 0}, "4": {...}}` mapping. `--quantization-group-tactics FILE` accepts rows with `bits`, `members` and `tactic`; legacy shape rows additionally require explicit coverage. Tactics are integers 0–7. Unmatched entries use reference tactic 0, not a measured optimum. No past experiment directory is searched automatically.
- `--quantization-shape-tactics FILE`: optional rows such as `[{"bits": 8, "shape": [41, 6144, 1536, true], "tactic": 2}]`, with shape `[M, K, N, has_bias]`. An exact match overrides the module/group tactic for that call; other shapes retain their configured tactic. `M` is the flattened input row count, `K`/`N` are the integer projection's logical widths (a grouped projection uses its combined padded output width). This selects existing reference kernels for native, prepared-input and grouped paths without changing quantization or weights. Load the table before CUDA Graph capture; close and recreate the optimization scope to change it. Calibrate on the target device and checkpoint, then validate full streaming replay; an example row is not a recommended preset.

### Owned operators and explicit profiles

The source layout separates a shared model contract from platform-specific
implementations:

| Location | Implementation or dependency |
| --- | --- |
| `kernels/common/csrc/` | Native vision RoPE and AdaLN BF16 fusion |
| `kernels/common/fused.py`, `graph.py` | Shared Triton primitives and graph lifecycle |
| `kernels/ampere_ada/csrc/integer_gemm.cu` | Canonical INT4/INT8 CUTLASS GEMM |
| `kernels/ampere_ada/csrc/integer_dispatch.cu` | Native activation preparation plus integer dispatch |
| `kernels/ampere_ada/` | Integer formats, modulation producers and optional Ada candidates |
| `kernels/blackwell/csrc/` | FP8/FP4 CUDA GEMM and packing, with focused lazy loaders |
| `kernels/vendor/cutlass/` | Default official CUTLASS submodule; compatible `CUTLASS_ROOT` also supported |

For the default dependency, initialize CUTLASS from the repository root, then
inspect GPU-free source and build metadata:

```bash
git submodule update --init --recursive coexecution/kernels/vendor/cutlass
python tools/prepare_sources.py --component robotics-kernels
python -m coexecution.kernels.build --help
python -m coexecution.kernels.build --backend integer --describe
```

The build command accepts `--backend` with `integer`, `integer_dispatch`,
`bf16_fusion`, `fp8`, or `fp4`, plus `--describe` and `--verbose`. Omit `--describe`
to explicitly build/load that backend with the installed CUDA toolkit. Torch's
extension cache stores generated binaries outside the tracked source tree;
`TORCH_EXTENSIONS_DIR` can point to a personal `.local/` directory. Runtime
backends also load lazily when explicitly enabled. Build and warm all selected
kernels before timing or CUDA Graph capture. The source manifest records imported
and adapted operator hashes and the default CUTLASS submodule revision. Actual
selected dependency and loaded binary identity are separate runtime evidence.
The Git link records `982748aa7356fa838c2ea4994ddcb0b2a4b4cefa` for reproducibility;
it does not require every author to run that exact SHA.

An existing compatible CUTLASS clone or complete headers-only installation can
be used directly:

```bash
export CUTLASS_ROOT=/path/to/cutlass
```

With this variable set, omit the submodule initialization command above.
The selected root needs the full required `include/` tree, including `cutlass`
and `cute`, plus `tools/util/include/`. Its APIs and CUDA/C++ toolchain must
support the selected operators; arbitrary versions are not promised to work.
Tools record the actual Git revision when present rather than rejecting a
source solely for differing from the default. The CUDA operator implementations
remain local piR2 files regardless of the general CUTLASS header location.

New environments can use `git clone --recurse-submodules`. Repeat the submodule
update command after pulling or switching branches when selecting that branch's
default. Compatible local submodule revisions are also usable; preserve those
intentional choices and keep their personal Git-link changes out of unrelated
commits. `--reference /path/to/existing/cutlass --dissociate` is an optional
first-clone download optimization, not a prerequisite for reusing an existing
installation. See [CUTLASS setup](../docs/environment.md#cutlass-submodule).
Routine default setup does not need `--remote`; changing the project's shared
default pin remains a separate reviewed change.

The inventory API reports implementation language, source, architecture,
precision, shape/layout constraints and numerical contract without importing
Torch, Triton or the model:

```python
from dataclasses import asdict
from coexecution.optimization.operators import inventory

for implementation in inventory("integer_gemm"):
    print(asdict(implementation))
# inventory() lists every family; it does not benchmark or rank implementations.
```

The newly consolidated native Ada candidates are explicit opt-ins for SM89:
INT4 GEMV (`integer_gemm=int4_gemv`), small-M INT4 tiles (`small_m`), integer
split-K (`split_k`), LayerNorm packing (`layernorm_pack=native`), native RMS
statistics (`rmsnorm=native`), RMSNorm packing (`rmsnorm_pack=native`), residual
epilogues (`integer_residual=native`) and grouped output stores
(`group_writeback=native`). Their per-call guards fall back to the configured
same-precision reference for unsupported shapes or runtime contracts. Native
LayerNorm/RMS candidates additionally enforce their pinned PyTorch requirements;
see [environment boundaries](../docs/environment.md#optional-gr00t-fusion-and-integer-inference).
The already existing BF16 and activation-pack implementations keep their own
inventory contracts; platform restrictions are not inferred from language alone.

An operator profile is a partial override. The following is a schema example,
not a measured preset or a recommendation for every W4A4 shape:

```json
{
  "schema_version": 1,
  "architecture": "sm_89",
  "precision": "w4a4",
  "implementations": {
    "integer_gemm": "small_m",
    "activation_pack": "native"
  },
  "parameters": {
    "integer_gemm": {"tactic": 0}
  }
}
```

Save a selected profile locally and pass `--operator-profile FILE` to the same
service command, or set `OptimizationConfig(operator_profile=Path(...))`.
`small_m` parameters accept tactic 0–5; `split_k` accepts `slices` 2/4/8 and
`tile` 0–2. These parameters belong to those implementations and are distinct
from canonical integer GEMM tactics 0–7. Unknown families, implementations,
parameters, architecture/precision mismatches and incompatible activation flags
are rejected. Each family must also be applicable to the selected fusion/precision
configuration. The profile hash and actual installed choices are retained in
optimization evidence; the profile does not change checkpoint weights, default
precision or unrelated fusion flags.

DiT LayerNorm, residual, shared-context and final-modulation profile choices
require full fusion and a DiT-containing component. RMS packing requires text
RMSNorm fusion. Overlapping producer replacements are rejected: profile
LayerNorm with `--norm-modulation-quant`, profile residual with
`--residual-norm-quant`, or profile LayerNorm with final-modulation packing.
Select one producer implementation for each boundary.

CUDA and Triton alternatives are not globally ranked. The GPU-free
`read_measurements(path)` and `select_measured(records, baseline="...")` API in
`coexecution.optimization.operators` consumes retained, normalized measurements
with an explicit same-precision numerical gate. Comparisons must agree on
architecture, precision, exact shape, measurement cohort, checkpoint/input
workload, device/environment identity, measured boundary, timing method and
statistic. Failed numerical candidates are excluded; mismatched contexts are
rejected. `deployment=True` additionally requires full-policy wall latency.
The result applies to that exact context and never changes runtime defaults.
No new benchmark, cross-platform winner or LIBERO success result is supplied by
this source organization; historical experiments require matched remeasurement
after selecting a new combination. Keep records and generated profiles local.

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

For a focused BF16 operator experiment, `NonGemmAdapters` supports `baseline`,
`vision_rope`, `text_rope`, `rope` (both RoPE paths), `rmsnorm`, `rope_rms`, and
`all` (RoPE/RMSNorm plus AdaLN modulation). These select existing kernels;
`baseline` restores the original forwards. The low-level adapter does not install
DiT condition/mask preparation, Graphs or quantization. Use one adapter per
private eval-mode model, change modes only after its work has completed, and
restore it with `close()` in `finally`. The service exposes the same enabled
component choices through `--operator-fusion-mode`; disable `--operator-fusion`
for the baseline. Neither interface relaxes the paired-worker guard. Test
component selection, instance isolation
and restoration explicitly with `tests/gpu/test_groot_fusion_modes_gpu.py`;
measure performance on the target GPU with matched inputs and invocation counts.

```python
from coexecution.groot_fusion import NonGemmAdapters

adapter = NonGemmAdapters(policy)
try:
    adapter.set("vision_rope")
    # Run the existing inference path on this privately owned model.
finally:
    adapter.close()
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

The service records the configuration, replaced sites and tactic-file hashes in its local identity output. Before changing model methods or layouts, an enabled scope checks the selected backend's exact operator APIs, including the FP8 packed-tactic and FP4 weight-packing APIs. Runtime evidence separates original upstream revisions, the owned source-tree fingerprint and loaded-library hashes, and records Torch/CUDA/Triton, device architecture and quantization format. Preloaded or ambiguous binary ownership is explicitly `unknown`; a requested environment-variable path is not proof that its library executed.

The canonical integer backend retains its original arithmetic/API contract, with SM89 separately recorded as historical validated architecture. Additional native Ada candidates are explicitly restricted to SM89. Successfully loading elsewhere does not establish arithmetic or task quality. Thor floating precision retains its existing SM110 restriction; BF16 has no new architecture whitelist. Missing APIs and loader failures propagate rather than silently selecting BF16. A complete comparison must include online activation preparation and checks, use matching fusion/layout/Graph settings for BF16 and low-bit variants, and record checkpoint/input identity. Profiled GPU duration sums, isolated GEMM timing, complete Linear timing and synchronized CPU-observation-to-action latency are different metrics. Installation, compilation, packing, preflight and Graph capture are reported separately.

Shape-tactic files still accept the legacy row list. A versioned integer profile can additionally bind the rows to the actual backend, clean source revision, architecture and quantization format:

```json
{
  "schema_version": 1,
  "backend_id": "pir2_integer",
  "backend_revision": "sha256:<source_sha256 from runtime backend evidence>",
  "device_arch": "sm_89",
  "quant_format": "w8a8-symmetric-row-v1",
  "rows": [{"bits": 8, "shape": [41, 6144, 1536, true], "tactic": 2}]
}
```

This is a schema example, not a recommended tuning result; replace the revision placeholder with the recorded owned-source fingerprint from backend evidence. Old bound `robotics_integer` profiles do not automatically match the new owned namespace/source identity. Profiles with missing, unknown or mismatched runtime identity are rejected before installation; legacy lists remain explicitly unbound. W4A4 uses `w4a4-symmetric-row-v1`. Thor FP8/FP4 use their distinct floating formats and cannot consume integer profiles. Close and rebuild the optimization scope after changing precision, weights or tactics.

优化根调用与 reset 共享串行保护，调用开始、结束和异常时清理投影缓存；Graph 异常恢复 eager 后仍保留该边界。该锁只管理 Python 提交，不是 CUDA 跨流同步，也不开放同进程双优化 worker。组件选择、缓存与后端契约的新增测试使用元数据/fake 对象；旧闭环结果不能作为新组合的质量结论。对应 CUDA 回归需显式执行 `tests/gpu/test_streaming_component_cache_gpu.py`，包含 eager/Graph 的 W8/W4 异常重试与 reset。本次结构和契约整理未重跑这些 GPU 测试。

Lightweight public CI runs only the configuration and restoration contracts. Explicit local GPU checks use no model download:

```bash
PIR2_GPU_TESTS=1 python -m pytest -q \
  tests/gpu/test_groot_fusion_gpu.py tests/gpu/test_quantization_gpu.py \
  tests/gpu/test_groot_vision_layout_gpu.py
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
PIR2_GPU_TESTS=1 python -m pytest -q tests/gpu/test_groot_streaming_gpu.py
```

Explicit checks for the experimental norm/residual options:

```bash
PIR2_GPU_TESTS=1 python -m pytest -q tests/gpu/test_groot_norm_quant_gpu.py
```

Shape-tactic dispatch has separate opt-in CUDA checks for both precisions, prepared inputs, grouped projections and changing-shape Graph capture:

```bash
PIR2_GPU_TESTS=1 python -m pytest -q tests/gpu/test_shape_tactics_gpu.py
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
and the repository-owned [Blackwell operators](../docs/environment.md#optional-thor-floating-inference).
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
  including Graph replay. The default uses the owned native packer with the pinned reference convention.
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

`--fp4-swap-ab` (Python: `OptimizationConfig(precision="fp4", fp4_swap_ab=True)`)
is an additional opt-in SM110 dispatch. It uses the repository-owned FP4 CUDA
backend described in [environment setup](../docs/environment.md). The unified
operator profile can select the same implementation with
`"floating_gemm": "swap_ab"`; `native` retains the original floating dispatch. Only
`(M,K,N,has_bias)=(41,256,1536,true)` and `(41,1536,1536,true)` use the new API;
all other signatures, including biasless and M=1 calls, keep the previous path.
The selected tile is `(128,64,256)` after swapping W and X, with cluster
`(2,1,1)`/two stages for K=256 and `(4,1,1)`/automatic stages for K=1536.
Both use the reference dynamic scheduler. Column-major output writes directly
to the original contiguous `[M,N]` result; there is no output transpose kernel.
Weights/activations remain E2M1 with block-16 UE4M3 scales, FP32 accumulation
and BF16 output. Packing, masks and attention are unchanged.

The two shapes were selected by repeated CUDA Graph event timing over actual
module weights, followed by numerical and full-replay checks. Their measured
operator gains do not establish a complete-policy speedup: paired full-replay
wall timing did not show a consistent improvement. BF16 rounding boundaries
may differ on arbitrary inputs, and existing FP4 quantization error remains;
this option is not LIBERO closed-loop quality acceptance. Disable the flag to
restore the original FP4 dispatch, or leave the optimization scope to restore
BF16. Public CI does not execute CUDA; run the explicit tests on Thor:

```bash
PIR2_GPU_TESTS=1 python -m pytest -q tests/gpu/test_fp4_swap_ab_gpu.py
```

These tests cover dynamic Graph inputs, nonzero bias, unsupported-shape fallback,
buffer rejection and exception restoration. Recalibrate before expanding the
signature table or claiming gains on a different checkpoint or workload.
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
PIR2_GPU_TESTS=1 python -m pytest -q tests/gpu/test_floating_quantization_gpu.py tests/gpu/test_floating_fusion_gpu.py
```

Keep these checks separate from model replay and LIBERO closed-loop evaluation.
Compare BF16/FP8/FP4 with identical checkpoint, attention, layout, fusion and Graph
settings. Report bootstrap, vision and plan separately, along with full replay;
exclude offline weight packing, compilation and capture from steady timings;
include online activation packing. Action/cache errors
and finite outputs do not establish task quality. No learned-policy quality
threshold or FP4 deployment recommendation is supplied by this adapter.

### Calibrated SmoothQuant channel balancing

`--fp-smoothquant PROFILE.pt` optionally balances activation/weight input
channels before the existing Thor FP8 or FP4 GEMM. Profiles are precision
specific and bound to SHA-256 hashes of the original BF16 projection weights.
An incompatible weight, channel count or shared-input group is rejected.
Missing profile sites keep their configured low-bit quantization; they do not
silently fall back to BF16. Coverage records smoothed and unsmoothed sites.

The calibration statistic and alpha formula follow the official
[SmoothQuant revision](https://github.com/mit-han-lab/smoothquant/tree/c61476d728e42ae0d8a35e7e78494edcac3237b5).
Its published INT8 language-model results do not establish accuracy of this
GR00T FP8/FP4 adaptation. For PyTorch weights `[N,K]`, the transformation is
`(X / s) @ (W * s).T + bias`. Original weights/biases are retained unchanged;
the packed weight copy contains the compensation. Finite-precision rounding
means this is not a claim of bitwise equivalence to unsmoothed BF16 inference.

Input compensation is fused into packing, including the opt-in SwiGLU,
GELU, RMSNorm, AdaLN and residual/normalization producers. AdaLN compensation is applied
after both conditional scale and shift. QKV/KV/gate-up consumers share one
channel scale, preserving shared-packing and merged-projection compatibility.
The profile is loaded and verified before Graph capture; activation-dependent
quantization scales still refresh on every invocation.

For an existing Thor launch command, retain its execution/precision flags and
add the matching profile:

```bash
# Existing FP8 command: --inference-precision fp8 --fp-smoothquant /path/fp8.pt
# Existing FP4 command: --inference-precision fp4 --fp-smoothquant /path/fp4.pt
# Optional activation producers: --fp-gelu --fp-swiglu
# Optional text MLP RMSNorm producer: --fp-rmsnorm
```

All options remain disabled by default. `--fp-gelu` uses the existing native
BF16 GELU lookup (including its approximation mode), then applies channel
balancing and packs the result without writing a separate smoothed tensor.
Diffusers' intervening eval Dropout accepts the packed carrier; training
Dropout is rejected and installation is rolled back. `--fp-rmsnorm` targets
Qwen3VL text MLP input normalization only when both gate/up consumers are
quantized. Its reduction order may differ from the native normalization;
validate action error separately before enabling it. These options do not
alter attention masks, image resolution, S1/S2 scheduling or GEMM tactics.

FP4 writes packed E2M1 data and block-16 UE4M3 scales directly. FP8 still
requires a tensor-wide maximum followed by scaling/casting; the fusion does
not remove that global dependency or promise a single kernel. LayerNorm and
RMSNorm retain explicit BF16 output stages but are not claimed bitwise equal
to the native reductions. Original forwards are restored when the scope exits.

The reusable Python APIs `ActivationCalibration`, `channel_scales` and
`projection_groups` live in
`coexecution.optimization.backends.thor.smoothquant`. Collect eager BF16
activation statistics from representative images, language, robot state and
denoising steps. Keep calibration, parameter-selection and final-validation
episodes disjoint. The collector owns its RNG and removes hooks on exit.
Export a `pir2.thor.smoothquant.v1` tensor-only profile with `precision`,
`upstream_revision`, calibration metadata and `linears[name]` records containing
FP32 `scale` tensors and `weight_sha256`. Shared-input members must all be
present with identical scales, or all absent. A scale of one and no profile
entry are different: absence preserves the original packing path.

Report action deviations against matched BF16, gripper disagreements and
complete-call latency separately. Calibration/operator error improvements are
not proof of lower held-out action error or LIBERO success. Profile collection,
weight packing, Graph capture and warmup are outside inference timing. The
GPU tests in `tests/gpu/test_smoothquant_gpu.py` exercise compensation, producer
compatibility, Graph updates, weight identity and restoration; they do not
establish task-quality acceptance.

On an otherwise configured Thor environment, run these explicit GPU checks:

```bash
PIR2_GPU_TESTS=1 python -m pytest -q \
  tests/gpu/test_smoothquant_gpu.py \
  tests/gpu/test_smoothquant_producers_gpu.py \
  tests/gpu/test_floating_fusion_gpu.py
```

Keep profiles, calibration samples, traces and measured figures in ignored
local directories. Complete-policy results that include other Graph or
transport optimizations cannot be attributed to these producers alone.

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

### Model and robot replay evidence

The [replay package](replay/README.md) builds a framework-free catalog from the
existing independent model benchmark and actual deployment traces. Optional
`--replay-assets` exports raw model input images outside timing;
`--record-replay` records each precision's own bounded robot state and two-camera
response, with encoding after control. The catalog keeps clean model timing,
actual instrumented kernel timelines and divergent closed-loop trajectories
separate. This is debugger evidence, not a replacement for LIBERO quality tests.
