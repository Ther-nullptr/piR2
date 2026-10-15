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

### Official Spatial Flow4 scheduling diagnostic

本节用于官方 **非 streaming Spatial、H40、Flow4（NFE=4）** 权重的调度诊断。
`flow` 读取真实图像/状态/文本 fixture；每个请求对重置输入，S1/DiT 消费已完成的旧特征，
S2/VLM 计算下一个 fixture，S1 使用下一个 fixture 的状态。它不运行 LIBERO 闭环；
调度时间、数值漂移和量化覆盖率均不能证明任务成功率或策略质量。原有 `worker`、`campaign`
保留上文的 streaming πR² 协议。

**1. 准备环境和路径。** 按[固定上游来源](upstream.md)、[CUDA 模型环境](environment.md)
准备完整任务 checkpoint、对应 LeRobot Spatial 数据集与 processor/config 缓存。这里不下载或训练模型。
外部 CUTLASS 选择与编译缓存见[环境配置](environment.md#optional-gr00t-fusion-and-integer-inference)。
当前 W8 full split-K profile 使用 Ada SM89 候选和严格 Torch guard（包括 `2.7.1+cu126`、
Torch revision/header 检查）；W4 使用现有 grouping/norm producers。两者保留 category 2、
site/group tactics，W8 另含 shape tactics。请核对环境文档，不能据此承诺 Jetson 支持。

每次调用都会加载**两份独立完整策略**，未执行组件的权重也常驻；整数配置保留 BF16 权重，
并准备 W8/W4 两种格式。下面两个独立进程合计加载四份完整策略。显存记录是诊断进程占用，
不能解释为纯 INT8/INT4 部署的压缩比例。模型计算只在 CUDA 环境运行。

以下代码块在同一个 Bash 会话中执行；先替换路径和 GPU 编号：

```bash
cd /path/to/piR2
export PIR2_MODEL_PY=/path/to/model-env/bin/python
CHECKPOINT=/path/to/GR00T-N1.7-LIBERO/libero_spatial
DATASET=/path/to/groot-libero-spatial
GROOT_ROOT=/path/to/Isaac-GR00T
export PYTHONPATH="$PWD:$GROOT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export GROOT_HF_LOCAL_FIRST=1 GROOT_PATCH_MISTRAL=1
export TORCH_EXTENSIONS_DIR=/path/to/local-cache/torch-extensions
GPU_S1=2
GPU_S2=1
PRECISION=bf16  # 也可为 w8a8 或 w4a4
CPU_PREFIX=()
taskset -pc $$  # 可选绑核：从允许集合选择 CPU_SET，再设 CPU_PREFIX=(taskset -c "$CPU_SET")
mkdir -p .local
BATCH=$(mktemp -d "$PWD/.local/flow4-XXXXXXXX")
COMMON=(scripts/profile_libero_contention.py flow
  --checkpoint "$CHECKPOINT" --dataset "$DATASET" --groot-root "$GROOT_ROOT"
  --fixtures 3 --episodes 0 --start-step 10 --step-stride 1 --seed 1000
  --torch-threads 2)
```

这明确选取 episode 位置 0 的 step 10/11/12；每个 fixture 使用 `seed + fixture_index`。
可选 CPU affinity 必须来自当前允许集合，并在可比较条件间保持一致。
`--output` 必须是**尚不存在的新目录**：只创建外层 `BATCH`，不要提前 `mkdir` 下文的输出子目录。
重跑、更换精度或 group 时使用新输出；保留失败目录、JSON、日志和 PID，不覆盖旧证据。

**2. 检查设备并记录上下文。** `flow` 是低层入口，**不负责 GPU admission、预留或外来进程监控**；
上文 `campaign` 的保护不适用于这里。Ada 计时准入要求所选卡无 compute PID、GPU 利用率 0%、
`memory.used <= 200 MiB`，并有足够空闲显存容纳两份完整策略；调用者检查这些条件并确认 UUID 映射，
运行期间持续检查占用并记录频率。发现外来负载时保留证据，将受影响时段标为不可比较。
不要为本诊断修改全局频率/MPS，也不要停止其他用户的任务。

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,clocks.sm,clocks.mem --format=csv > "$BATCH/gpus-before.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv > "$BATCH/processes-before.csv"
cat "$BATCH/gpus-before.csv" "$BATCH/processes-before.csv"
GPU_S1_UUID=$(nvidia-smi -i "$GPU_S1" --query-gpu=uuid --format=csv,noheader)
GPU_S2_UUID=$(nvidia-smi -i "$GPU_S2" --query-gpu=uuid --format=csv,noheader)
printf 'S1=%s\nS2=%s\n' "$GPU_S1_UUID" "$GPU_S2_UUID" > "$BATCH/selected-gpus.txt"
nohup nvidia-smi -i "$GPU_S1,$GPU_S2" \
  --query-gpu=timestamp,index,uuid,memory.used,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw \
  --format=csv -l 1 > "$BATCH/telemetry.csv" 2> "$BATCH/telemetry.log" &
TELEMETRY_PID=$!
printf '%s\n' "$TELEMETRY_PID" > "$BATCH/telemetry.pid"
```

确认两个查询成功并返回不同的完整 `GPU-...` UUID；物理索引仅用于 `nvidia-smi`，所有 CUDA 启动使用 UUID，
避免把 `nvidia-smi` 索引当作 CUDA 默认枚举序号。采样器记录同一批物理设备；仍需定期复查 compute processes。
仅跑单卡时可省略 S2 UUID 查询，并把采样器 `-i` 改成 `"$GPU_S1"`；结果中 `hardware` 的逻辑设备 UUID
必须与 `selected-gpus.txt` 核对：先去掉可选的 `GPU-` 前缀，再比较完整 UUID 主体（Torch 记录可能不含此前缀）。
单个 UUID mask 中的 `cuda:0` 就是该 UUID 对应设备。

**3. 同 GPU：先跑完整匹配矩阵。** 下例含每 case 30 次 clean 请求、2 轮预热及独立短 profile。
所有 fixture 会在实际执行线程/stream（含高优先级 DiT stream）上预热并 capture，另逐 case 预热；
这些调用、特征准备、RNG reset、快照和数值检查均不计入请求计时。

```bash
GROUP=optimized
OUT="$BATCH/same-$PRECISION-$GROUP"
nohup env CUDA_VISIBLE_DEVICES="$GPU_S1_UUID" "${CPU_PREFIX[@]}" \
  "$PIR2_MODEL_PY" "${COMMON[@]}" --output "$OUT" \
  --precision "$PRECISION" --group "$GROUP" --iterations 30 --warmup 2 \
  --offsets-ms 0 5 --profile-iterations 1 --action-device cuda:0 --vlm-device cuda:0 \
  > "$OUT.log" 2>&1 &
PID=$!
printf '%s\n' "$PID" > "$OUT.pid"
tail -n 20 "$OUT.log"
# status.json 在加载开始后出现；后续可重复查看。
cat "$OUT/status.json"
# 在启动它的 shell 中等待；确认退出后再在同卡启动下一条件。
if wait "$PID"; then RUN_RC=0; else RUN_RC=$?; fi
printf '%s\n' "$RUN_RC" > "$OUT.exit-code"
```

默认 `optimized` 是 7 cases，`cooperative` 是 15 cases（并发模式各有 0/5 ms 两个 offset，
alone/serial 各一次）。`--modes` 可选子集；完整对比需保留相同精度、placement、group 的 alone/serial。

| `--modes` | 测量内容 |
| --- | --- |
| `vlm_only` / `dit_only` | S2 / S1 独跑 |
| `serial` | S1 先完成，再启动 S2 |
| `concurrent_equal` / `concurrent_dit_priority` | 等优先级 / DiT 高优先级并发 |
| `vlm_only_fenced` / `serial_fenced` | 带已完成层边界 fence 的独跑/串行控制 |
| `concurrent_equal_fenced` / `concurrent_dit_priority_fenced` | 保留 fence 成本的并发控制 |
| `cooperative_dit_priority` | 在已完成的 VLM 层边界让出，执行 DiT 后恢复 |

`optimized` 开启 whole VLM graph；两组均使用 whole action graph、K/V reuse、静态准备与编译 guard。
将 `GROUP=cooperative` 并换新输出后重跑同一命令，整个组统一关闭 whole VLM graph、保留相同空操作层 hook；
仅 fenced/cooperative cases 同步层边界。用匹配 fenced controls 区分 fence 成本和让出收益；
`--vlm-chunk-blocks 1`（默认）或 `2` 设置边界粒度。不要把跨 group 的差值直接称为调度收益。
更换 `PRECISION` 后顺序重跑；同卡不同条件不要同时启动。CUDA priority 是提示，不是硬抢占保证。

**4. 同进程双 GPU。** 上一轮退出、两卡预检通过后运行。完整 UUID 按 `GPU_S1_UUID,GPU_S2_UUID`
顺序列入 visible mask 后，分别映射为进程内 `cuda:0,cuda:1`；保留该 placement 自己的 alone/serial 基线与两卡频率记录：

```bash
OUT="$BATCH/dual-$PRECISION-optimized"
CUDA_VISIBLE_DEVICES="$GPU_S1_UUID,$GPU_S2_UUID" "${CPU_PREFIX[@]}" \
  "$PIR2_MODEL_PY" "${COMMON[@]}" --output "$OUT" \
  --precision "$PRECISION" --group optimized --iterations 30 --warmup 2 \
  --offsets-ms 0 5 --profile-iterations 0 --action-device cuda:0 --vlm-device cuda:1 \
  > "$OUT.log" 2>&1
```

旧特征在计时前复制到 S1 GPU 的独立存储；S2 ready 仅表示 VLM GPU 上完成，**新特征传输被排除**。
因此它不是同一帧的 VLM→DiT 端到端延迟。不同物理卡/频率下的独跑时间不能直接作为隔离收益分母。

**5. 两个独立进程、每进程一张 GPU。** 此例连续回放各 600 次，关闭 profiler；每个进程内部两份
策略都位于本进程的 `cuda:0`。先等双方预热并确认 CUDA idle，再发布 release 文件；start gate
仅协调开始时刻，不预留 GPU。先放行 VLM，读取首条完成样本，再以其实际 `s2_release_ns + 3 s`
为目标放行 DiT。每次重跑都创建新外层目录、两个新 gate 路径，禁止复用文件或 symlink。

```bash
PAIR=$(mktemp -d "$BATCH/replay-XXXXXXXX")
REPLAY_ARGS=(--precision "$PRECISION" --group optimized --iterations 600 --warmup 2
  --profile-iterations 0 --action-device cuda:0 --vlm-device cuda:0 --gate-timeout-s 600)
nohup env CUDA_VISIBLE_DEVICES="$GPU_S1_UUID" "${CPU_PREFIX[@]}" \
  "$PIR2_MODEL_PY" "${COMMON[@]}" "${REPLAY_ARGS[@]}" --modes dit_only \
  --output "$PAIR/dit" --start-gate "$PAIR/release-dit" > "$PAIR/dit.log" 2>&1 &
DIT_PID=$!
printf '%s\n' "$DIT_PID" > "$PAIR/dit.pid"
nohup env CUDA_VISIBLE_DEVICES="$GPU_S2_UUID" "${CPU_PREFIX[@]}" \
  "$PIR2_MODEL_PY" "${COMMON[@]}" "${REPLAY_ARGS[@]}" --modes vlm_only \
  --output "$PAIR/vlm" --start-gate "$PAIR/release-vlm" > "$PAIR/vlm.log" 2>&1 &
VLM_PID=$!
printf '%s\n' "$VLM_PID" > "$PAIR/vlm.pid"
set -o pipefail
python3 - "$PAIR" "$DIT_PID" "$VLM_PID" <<'PY' 2>&1 | tee "$PAIR/coordinator.log"
import json, os, sys, time
from pathlib import Path
root = Path(sys.argv[1])
pids = dict(zip(("dit", "vlm"), map(int, sys.argv[2:])))
deadline = time.monotonic() + 600
def check():
    if time.monotonic() >= deadline:
        raise TimeoutError("Coordinator exceeded 600 s; inspect worker logs/status")
    for role, pid in pids.items():
        failure = root / role / "failure.json"
        if failure.exists():
            raise RuntimeError(f"{role}: {failure.read_text()}")
        os.kill(pid, 0)  # 检查自己启动的 PID，不终止进程。
        status = root / role / "status.json"
        try:
            state = json.loads(status.read_text())["state"]
        except (FileNotFoundError, json.JSONDecodeError):
            continue  # 短暂缺失或不完整 JSON 留到下次有界轮询。
        if state == "complete":
            raise RuntimeError(f"{role} finished before both gates were released")
def poll(read):
    while True:
        check()
        value = read()
        if value is not None:
            return value
        time.sleep(0.05)
poll(lambda: True if all((root / r / "start-ready.json").is_file() for r in pids) else None)
with (root / "release-vlm").open("x") as stream:
    stream.write("release\n")  # 排他创建普通文件，已有文件/链接会报错。
def first_sample():
    path = root / "vlm" / "samples.jsonl"
    if path.exists():
        with path.open() as stream:
            line = stream.readline()
        if line.endswith("\n"):
            return json.loads(line)
row = poll(first_sample)
target = row["s2_release_ns"] + 3_000_000_000
if time.perf_counter_ns() >= target:
    raise RuntimeError("First VLM sample arrived too late for the 3 s target; rerun")
while time.perf_counter_ns() < target:
    check()
    time.sleep(min(0.01, max(0, (target - time.perf_counter_ns()) / 1e9)))
check()
with (root / "release-dit").open("x") as stream:
    stream.write("release\n")
print(json.dumps({"first_s2_release_ns": row["s2_release_ns"],
                  "target_ns": target, "gate_created_ns": time.perf_counter_ns()}))
PY
```

轮询失败/超时会显示 traceback 并返回非零；先检查 `coordinator.log`、两份日志/status/failure，
不要继续把这次运行当成有效样本。worker 的 600 s gate 超时从各自 ready 开始；协调器另有 600 s 总限。
成功后可在启动 shell 分别 `wait "$DIT_PID"`、`wait "$VLM_PID"` 并记录各退出码；需要中止时，
核对 PID/命令仍属于本次启动，再只终止这些跟踪进程。采样结束同样只停止自己的 `TELEMETRY_PID`。

gate 轮询与调度会产生偏差，实际间隔要用两份 `samples.jsonl` 的 release 时间戳计算。
在同一主机的 `perf_counter_ns` 时钟上，令 `L=max(双方首条 release)`、`R=min(双方最后 ready)`。
co-running 窗口两端各裁去 50 ms，仅保留整个 `[release, ready]` 落在 `[L+50 ms,R−50 ms]` 内的请求；
solo head 要求请求完成早于对方首条 release 减 50 ms。solo tail 则要求请求开始晚于**已观测对方进程退出**
加 50 ms：最后 ready 后仍可能有验证、写文件和资源清理，这些 CPU 工作不能归入 solo tail。
上述协调器提供开始证据；tail 还需父进程 `wait` 或 PID 状态观测，并以同一 `perf_counter_ns` 时钟记录
逐 PID 的退出观测时间，不能用 `status=complete` 代替。没有退出证据时，tail 保持未分类或使用独立基线。
剔除跨边界请求；裁剪后空窗口没有 co-running 样本。此窗口只表示两个回放循环都活跃，不证明 kernel 重叠。
这是**连续回放**，不是锁步 0/5 ms 请求对、进程间特征 IPC 或机器人 AoI 测试。

**6. 读结果和检查边界。** 先确认正常退出、`status.json` 为 `complete` 且无 `failure.json`；
异常后 status 可能保留旧状态，缺少最终结果不能视为完成。保留以下本地产物：

| 文件/字段 | 用途 |
| --- | --- |
| `status.json`、`failure.json`、外层 `.log`/`.pid` | 当前阶段、失败原因及进程归属 |
| `start-ready.json` / `start-gate.json`（指定 gate 时） | 完成预热后的 ready 与实际放行记录 |
| `measurement.json`、`samples.jsonl` | schema `libero-flow4-contention` v1；汇总与逐请求记录 |
| `fixtures.json`、`features.json`、导出的 observation | 输入身份、旧特征 generation/hash/准备时间 |
| `executed-coverage.json`、`graph_audits` | 量化执行覆盖、capture/replay 与 action K/V 审计 |
| `provenance`、`hardware`、`runtime`、`memory` | 源码/checkpoint 配置 hash、UUID、软件/绑核、常驻和峰值显存 |

覆盖计数是预热期间的 adapter visits，不是 timed kernel 数；static feature 路径不调用的已知
`backbone.model.lm_head` 单列为 inactive，不能当成漏量化，也不能略过其他 missing projection。
冷 capture、launcher cache 改变、特征存储变化、RNG 消耗不符或输出生命周期违规都会使诊断失败。
数值 MAE/max-abs/relative-L2 对照同精度/placement/group 的 serial；不设逐值相等否决门槛，
但 keys、shape、finite、RNG 与旧特征 freshness 严格检查。数值差异仍需独立质量评估。

只读汇总无需 Torch；将 `OUT` 指向要检查的已完成输出：

```bash
python3 - "$OUT/measurement.json" <<'PY'
import json, sys
from pathlib import Path
m = json.loads(Path(sys.argv[1]).read_text())
assert (m["schema"], m["schema_version"]) == ("libero-flow4-contention", 1)
for case, values in m["summary"].items():
    print(case)
    for key in ("s1_response_ms", "s2_response_ms", "pair_makespan_ms"):
        if key in values:
            x = values[key]
            print(f"  {key}: P50={x['p50']:.3f}, P95={x['p95']:.3f} ms (n={x['count']})")
PY
```

response 是 release 到本 worker 观察到 CUDA output ready，含协作等待和 CPU 提交；makespan
含 host dispatch/串行交接。`gpu_span_ms` 是 CUDA event 跨度，含队列/提交空隙，**不等于 kernel busy time**。
host submission 区间可能与 GPU 执行重叠，不能直接把它们相加。用 `s1_release_ns-s2_release_ns`
核对并发 case 的实际 offset 与 `offset_ms` 目标；`old_feature_ready_ns` 是准备完成时间，不能当图像采集 AoI。
旧 streaming `analyze`/`report` 不消费 Flow4 schema。

`--profile-iterations 1` 在全部 clean cases 之后另跑每 case 的短 Torch profiler window，
保存在 `profiles` 与 `trace-*.chrome.json`，不混入 clean samples；这是已验证的导出流程。
高级开关 `--cuda-profiler-range` 仅在 `--profile-iterations > 0` 时可用，标记单独的 profile window。
外部 Nsight 可按 `--capture-range=cudaProfilerApi --capture-range-end=repeat` 捕获各 case 的范围，
但此分支仍启用 Torch profiler，二者组合尚未经过集成验证。尝试时使用新输出，将整个 Nsight 调用
保留为 profiler 证据，不将其计时混入独立无 profiler 运行。NVTX 标注实际 S1/S2 request/fixture；
应检查真实 CUDA activity 与 stream ID 才能判断 kernel 重叠。

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
