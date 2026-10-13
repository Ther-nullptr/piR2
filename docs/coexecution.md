# S1/S2 measurement boundaries

## LIBERO VLM/DiT contention profiling

Use `scripts/profile_libero_contention.py` to measure a trained H40 streaming
πR² checkpoint with real LIBERO observations. Shared code lives in
`coexecution/profiling`; the script only dispatches commands. This opt-in
diagnostic is separate from closed-loop success evaluation and from the older
SO100 replay described below.

**中文：** 此入口比较 VLM 独跑、DiT 独跑、串行、同优先级并发、DiT
高优先级以及 VLM 层边界协作让出。输入、随机数和动作缓冲固定，DiT
读取上一代已经完成的特征，下一代特征只在 CUDA 完成事件之后发布。
每个配置先检查同精度下的串行与并发输出是否逐值一致，再预热和采集。
它不运行机器人闭环，不能把速度或一致性检查当作任务成功率。

The supported diagnostic precisions are BF16, W8A8 and W4A4, with quantization
selected for both components, VLM only or DiT only. Streaming inference uses
H40, one DiT evaluation per call, slide length one and image-delay embedding
three. The common BF16 bootstrap uses four DiT evaluations and is excluded from
measurement. Observations and state preprocessing, RPC, action decoding and
simulator execution are also excluded. Integer quantization can change policy
quality; a W4A4 hardware measurement is not deployment acceptance.
The diagnostic controller retains the original BF16 parameters and prepares
integer weights, so allocated/reserved memory is the diagnostic process footprint,
not the minimum deployment footprint of a pure W8/W4 model. Use the independent
DRAM counters to compare actual transferred bytes.

CUDA stream priority is a scheduling hint, not proof of interrupting a running
kernel. Cooperative mode explicitly waits for a completed VLM layer boundary,
runs DiT and resumes VLM; `vlm_only_cooperative` measures the layer-fence overhead
without DiT. Arrival offsets and one/two-block boundaries are explicit inputs.

### Set up and collect

Use the CUDA environment and pinned GR00T setup from
[environment setup](environment.md) and [upstream provenance](upstream.md).
Initialize or select compatible CUTLASS headers as described in
[Contributing](../CONTRIBUTING.md#cutlass-dependency--cutlass-依赖).
Supply a trained streaming checkpoint and the corresponding LeRobot dataset;
the tool does not download them or train a model. CPU CI only exercises metadata,
scheduling and parsing contracts.

Select an idle physical GPU explicitly. The campaign waits for no compute
processes, zero reported GPU utilization and at most 200 MiB used memory, then
keeps the same GPU UUID throughout the run. If another process appears, only the
campaign's own processes are stopped and the attempt is marked invalid. No
clock or power settings are changed.

```bash
# Inspect device occupancy and use the intended profiler versions.
nvidia-smi
nsys --version
ncu --version

# Creates a frozen local plan; it does not start a model.
.venv/bin/python scripts/profile_libero_contention.py campaign prepare \
  --plan artifacts/contention/plan.json \
  --checkpoint outputs/my-pir2/checkpoint-10000 \
  --dataset datasets/groot-libero-spatial \
  --groot-root upstream/learning/Isaac-GR00T \
  --gpus 2 --offline --analyze

# Copy the exact plan hash printed by prepare. The two commands validate it.
.venv/bin/python scripts/profile_libero_contention.py campaign check \
  --plan artifacts/contention/plan.json --plan-sha256 PLAN_SHA256
.venv/bin/python scripts/profile_libero_contention.py campaign run \
  --plan artifacts/contention/plan.json --plan-sha256 PLAN_SHA256 \
  --output artifacts/contention/run
```

`--nsys`, `--ncu`, `--python`, `--cpu-set`, `--stages`, `--nsys-metrics-set` and
`--ncu-metrics` select tools and collection settings explicitly. Inspect each
command's `--help` for details. Profiler binaries otherwise resolve through
`PATH`; confirm the selected versions. The hardware-counter path was developed
with Nsight Systems 2025.3, Nsight Compute 2025.1 and RTX 6000 Ada. Other devices
need their own supported metrics and kernel backends; missing counters are not
silently filled with zero.

For a custom matrix, `--cases-json` accepts a list such as:

```json
[
  {
    "id": "w8-vlm-priority-offset5",
    "precision": "w8a8",
    "component": "vlm",
    "mode": "concurrent_dit_priority",
    "dit_offset_ms": 5,
    "vlm_chunk_blocks": 1
  }
]
```

Precision/component choices and modes appear in `worker --help`. The lower-level
worker requires an explicit GPU mask; the campaign owns admission, telemetry,
process ownership checks, frozen source/checkpoint evidence and profiler
launches. For long collections, run the campaign in a persistent terminal session
and retain its output directory. Resume with the same plan, hash and output:
only completed records whose evidence still matches are skipped, and failed
attempts remain available. The public tool does not automatically retry a
failed campaign or modify other users' processes or profiler lock files.

### Read the results

| Measurement | What it establishes | Limitation |
| --- | --- | --- |
| Unprofiled response time | Request release to this worker observing its own CUDA completion, including cooperative waiting | Includes CPU submission/wakeup; it is not kernel busy time |
| CUDA event span | Elapsed time on a stream | Excludes permit waiting before the start event and includes submission gaps |
| Nsight Systems | Actual kernel overlap, launch attribution and sampled SM/Tensor/DRAM activity | Concurrent counters describe the whole GPU, not separate model shares |
| Nsight Compute app-range | DRAM reads/writes per captured model call or pair | Replay changes timing; its GB/s is not normal-service bandwidth |
| NVML telemetry | Device/process occupancy, clocks, temperature and power context | `utilization.memory` is a busy-time fraction, not GB/s |

Nsight Systems and Compute run separately. The NCU application-range path uses
the worker's `cudaProfilerStart/Stop` markers and does not pass the unsupported
`--profile-from-start` option. Kernel replay would change concurrency. Clock and
cache controls are explicitly disabled for NCU; all replay throughput remains
labelled as replay-window throughput. The optional private temporary directory
is used only with a system-wide check for other NCU processes, following
[NVIDIA's lock-file guidance](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#faq).

```bash
# Parse an existing capture without loading CUDA or requiring matplotlib.
.venv/bin/python scripts/profile_libero_contention.py analyze \
  --measurement artifacts/contention/case/worker/measurement.json \
  --nsys artifacts/contention/case/trace.sqlite \
  --out artifacts/contention/case/report --no-plots

# Aggregate valid completed attempts; failed and pending attempts stay excluded.
.venv/bin/python scripts/profile_libero_contention.py report \
  --root artifacts/contention/run --out artifacts/contention/report --no-plots
```

For figures, install matplotlib, omit `--no-plots`, and use an installed Source
Han/Noto CJK font or `--font /path/to/font.ttf`. Plots and JSON retain source
provenance and units. Different hardware, model/fixture, implementation or timing
cohorts are kept separate. Raw traces, generated reports and local plans belong
under ignored artifact directories, not in Git.

## Auxiliary SO100 replay

The original `coexecution.run` replay workload uses the SO100 GR00T example, H16, BF16 and preprocessed observations on GPU. It does not load the main LIBERO task checkpoint by default and is not a drop-in LIBERO benchmark. Its purpose is to validate execution mechanisms before applying them to the real closed-loop task.

Measure S1 alone, S2 alone, the serial pair and the concurrent pair. Use the measured serial pair as the speedup denominator, not a sum of isolated medians. Preserve input, cache version, RNG and rolling-buffer state across paired comparisons. Report model loading, preprocessing, CPU action decoding and robot/simulator I/O exclusions explicitly.

CUDA event spans are not proof of kernel overlap. Export Nsight data to SQLite and use `coexecution.analyze_trace` to intersect actual kernel intervals; `coexecution.operator_analysis` joins kernel launches with the innermost NVTX range on the correct CPU thread. Profiled latency and profiler-free latency are separate measurements.

```bash
nsys export --type=sqlite --output=artifacts/coexecution/run/nsys.sqlite \
  artifacts/coexecution/run/nsys.nsys-rep
.venv/bin/python -m coexecution.analyze_trace \
  artifacts/coexecution/run/nsys.sqlite --output artifacts/coexecution/run/overlap.json
```

The periodic runner retains the configured release period, bounded in-flight work and actual feature versions. Deadline denominators include dropped requests. Cache age is calculated at consumption, and release/completion/drop conservation is checked. Fixed-pair profiling and free-running periodic replay answer different questions; neither substitutes for the LIBERO wall-clock control protocol.

Analysis produces local machine-readable records. Plotting, if requested through the optional `--plot` flag, additionally requires matplotlib. One-off rendering scripts with historical hardware/trial constants, generated figures, optimization ledgers and snapshots remain local. Do not interpret trace coverage as demonstrated quantization quality or task accuracy.
