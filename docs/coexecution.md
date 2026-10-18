# 量化收益、模型时延与 S1/S2 争用实验

本项目用三层证据连接算子加速与机器人响应：在固定真实输入上测模型，在独立 S1/S2 工作负载上解释争用，最后在 LIBERO 闭环中观察动作是否及时生效。每层保持明确的输入、工作量和计时边界，因而能够区分量化收益、运行时开销与调度收益。

| 要回答的问题                                      | 入口                                        | 输入与结果                                                         |
| ------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------ |
| 量化后完整模型快多少，输入准备占多少？            | `python -m coexecution.libero.measurement`  | 官方 Spatial 真实输入；独立模型/原始策略计时与分开采集的 profile   |
| 优化后的 Flow4 在串行、并发、协作调度下怎样变化？ | `scripts/profile_libero_contention.py flow` | 官方非 streaming H40/NFE4；S1/S2 response、pair makespan、数值漂移 |
| πR² 的一步 DiT 与真实 VLM 如何争用 GPU？          | 同脚本 `campaign` / `worker`                | 训练后的 streaming H40；固定输入/rolling buffer、Nsight 与硬件计数 |
| 这些变化是否改善真实动作的及时性与任务表现？      | [LIBERO 声明条件 runner](libero.md)         | 匹配初态的闭环；成功率、实际延迟、AoI、采用/执行与 fallback        |

本页给出实验操作；开关与实现细节集中在[算子参考](../coexecution/README.md)，控制节拍、动作提交与缓存语义集中在[时序协议](timing-protocols.md)。代码帮助和离线分析不加载模型，模型计算使用 CUDA。

## 比较口径

对照固定 checkpoint、真实输入或闭环初态、随机种子、精度覆盖、attention、layout、fusion、Graph 设置与硬件状态。改变运行时开关时保持精度不变；比较精度时共用相同的系统优化。稳态计时包含 online activation quant/pack，安装、编译、权重 packing、预热和 capture 单列。

| 指标                            | 含义                                             | 使用方式                                           |
| ------------------------------- | ------------------------------------------------ | -------------------------------------------------- |
| Raw policy wall                 | 原始观测进入策略，到同步完成并得到动作           | 衡量输入准备、模型和解码合在一起的调用成本         |
| Prepared/model-only wall        | 常驻准备输入上的完整模型调用，到同步完成         | 隔离输入准备、传输、解码、RPC 与仿真               |
| S1/S2 response                  | release 到 worker 观察到本次 CUDA output ready   | 含提交、唤醒和协作等待，用于评估请求响应           |
| Pair makespan                   | 一对 S1/S2 请求从开始到共同完成                  | 联合评估调度，使用同条件 serial pair 为分母        |
| CUDA event span                 | stream 上两个 event 的跨度                       | 含队列/提交空隙，不等于 GPU 活跃时间               |
| GPU activity sum / union / span | 实际 kernel/memcpy/memset 区间的总和、并集和跨度 | 在独立 profile 中解释工作量与重叠                  |
| LIBERO 控制指标                 | 真实请求、提交、采用和执行时间及来源             | 将模型加速关联到动作等待、AoI、fallback 与任务结果 |

这些范围有重叠，不能把 wall 减 event span 当作纯 CPU 时间，也不能将孤立 S1/S2 中位数相加代替 serial pair。host RPC 重叠与 CUDA stream priority 均不能证明 kernel 抢占；实际重叠需检查 CUDA activity。缺失计数/时间戳保留 unknown 或 null，不填零。

先按[环境配置](environment.md)准备固定源、完整任务权重及数据。计时使用空闲物理 GPU，保留 UUID、compute processes、实际时钟、软件和 profiler 版本；有外来负载的时段单独标记。各入口的 GPU 准入责任不同，见下文。结果、计划和原始 trace 留在忽略的 `.local/` 或 `artifacts/`。

## Independent model measurement

`coexecution.libero.measurement` 同时支持两种边界：

| `--measurement-boundary` | 实验组织                                                                                                       |
| ------------------------ | -------------------------------------------------------------------------------------------------------------- |
| `model-only`             | 每个 precision/runtime block 一次准备并持有固定真实输入，预热后回放；每次重算完整 VLM 与 Flow4，不缓存神经输出 |
| `comparison`（默认）     | 分别测 raw-policy、准备/传输与 prepared-model probes；每轮随机模式/观测顺序，相同输入和 seed 配对              |

model-only 将观测选择、图像处理、collation、输入复制、RPC、仿真、动作解码、数值验证和写文件全部移出模型 timer；最多常驻 16 个观测，可用 `--max-prepared-observations` 调整。runtime block 内不穿插 raw 调用，评估时间漂移时另行记录或反转 block 顺序。

每次使用尚不存在的 `--output` 目录，保留上一条件的原始记录。以下示例假定已选择空闲 CUDA 设备并准备好模型环境：

```bash
python -m coexecution.libero.measurement \
  --checkpoint /path/to/GR00T-N1.7-LIBERO/libero_spatial \
  --dataset /path/to/groot-libero-spatial \
  --output .local/model-paired --precisions bf16 w8a8 \
  --measurement-boundary model-only --modes compiled kv_reuse \
  --action-cuda-graph --operator-fusion \
  --observations 3 --rounds 20 --warmup 2 \
  --profile-rounds 1 --replay-assets --runtime-receipt
```

此例在同一精度内比较 `compiled` 与 `kv_reuse`，两者只相差 Flow4 内的 cross-attention K/V reuse。测量入口通过 mode 管理开关，因此拒绝全局 `--action-kv-reuse`；服务入口则接受该 flag。支持 BF16/W8A8/W4A4，各精度应按实际研究范围显式列出。

runtime modes 按下表逐项累加；默认运行 `baseline` 到 `graph`，新增 host 优化需显式选择：

| Mode        | 新增机制                                      |
| ----------- | --------------------------------------------- |
| `baseline`  | 原始运行时                                    |
| `prepared`  | 输入字段分离                                  |
| `tokenized` | 精确文本 token 缓存                           |
| `static`    | 固定 VLM 元数据                               |
| `graph`     | 完整 VLM Graph                                |
| `images`    | ndarray 图像准备                              |
| `compiled`  | 编译 host guards                              |
| `kv_reuse`  | Flow4 内 K/V 复用，要求 `--action-cuda-graph` |

`baseline` 存在时作为同精度参考，否则用所选列表的第一个 mode，并记录其身份。model-only 接纳前比较 raw policy 的 decoded actions 与 CPU/CUDA RNG；timed graph 调用必须只有 replay，没有 capture。whole-action 模式要求每次完整 replay 对应四次逻辑 DiT 工作。

clean latency 不分配 timing events 或 profiler scopes。`--cuda-events` 显式增加 CUDA span/host-return 信息；`--profile-rounds N` 另开带 request/VLM/action 标注的 profile pass，model-only 输入在 profile 开始前准备好。样本保留 precision、observation、request 和 seed；`--replay-assets` 在计时外导出真实输入，`--runtime-receipt` 记录源码和已加载算子身份。前端使用方式见[回放合同](../coexecution/replay/README.md)。

## LIBERO VLM/DiT contention profiling

streaming 入口测训练后的 πR² H40 检查点：每次 S1 使用一次 DiT、slide=1、image-delay embedding=3，S2 使用真实 LIBERO 观测。每对请求恢复相同输入、RNG 和动作缓冲；S1 消费上一代已完成的特征，S2 的新特征经 CUDA completion event 后发布。共同 BF16 bootstrap 使用四次 DiT，位于计时外。

支持 BF16/W8A8/W4A4，以及 `component=all/vlm/dit`。量化保留原 BF16 参数和整数格式，memory 字段表示诊断进程占用；实际传输字节由独立 DRAM counters 给出。该入口排除观测/状态预处理、RPC、动作解码和仿真，结果用于解释组件争用。

调度包括独跑、serial、同优先级 concurrent、DiT priority 和 VLM 层边界 cooperative。cooperative 等待已完成层边界，运行 DiT 后恢复 VLM；`vlm_only_cooperative` 测纯 fence 开销，offset 和一/两 block 粒度为显式参数。这个入口在计时前要求同精度 serial/concurrent tensors 逐值相等；下节 Flow4 入口则保留结构、finite、RNG 门槛并报告数值漂移，二者准入规则分别记录。

### 冻结计划并采集

campaign 管理 GPU 准入与 owned processes：等待无 compute process、GPU utilization=0%、memory.used≤200 MiB，之后固定 UUID；检测到外来 process 时停止自身实验进程并将 attempt 标为 invalid。campaign 不修改频率或功率。低层 `worker` 要求显式 GPU mask，调用方自行承担准入。

```bash
nvidia-smi
nsys --version
ncu --version

.venv/bin/python scripts/profile_libero_contention.py campaign prepare \
  --plan artifacts/contention/plan.json \
  --checkpoint outputs/my-pir2/checkpoint-10000 \
  --dataset datasets/groot-libero-spatial \
  --groot-root upstream/learning/Isaac-GR00T \
  --gpus 2 --offline --analyze

# 用 prepare 输出的准确 hash 替换 PLAN_SHA256。
.venv/bin/python scripts/profile_libero_contention.py campaign check \
  --plan artifacts/contention/plan.json --plan-sha256 PLAN_SHA256
.venv/bin/python scripts/profile_libero_contention.py campaign run \
  --plan artifacts/contention/plan.json --plan-sha256 PLAN_SHA256 \
  --output artifacts/contention/run
```

`prepare` 创建冻结计划，不启动模型。`--nsys`、`--ncu`、`--python`、`--cpu-set`、`--stages`、`--nsys-metrics-set`、`--ncu-metrics` 显式选择工具/采集配置，其他选项见 `--help`。硬件计数路径的已用环境为 Nsight Systems 2025.3、Nsight Compute 2025.1、RTX 6000 Ada；其他硬件选择其实际支持的 metrics/backend。

用 `--cases-json` 指定矩阵，例如：

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

长采集放入持久终端会话。相同 plan/hash/output 可恢复：只有完成且证据仍匹配的记录被跳过，失败 attempt 保留，失败 campaign 不自动重试。其他用户进程与 profiler lock 不归该工具管理。

### 分开解释 timing 与 profiler

| 证据                     | 解释                                                                                          |
| ------------------------ | --------------------------------------------------------------------------------------------- |
| 无 profiler 的 response  | 含 CPU submission/wakeup 和 cooperative waiting 的实际请求响应                                |
| Nsight Systems           | kernel overlap、launch attribution、采样 SM/Tensor/DRAM activity；并发 counters 是全 GPU 总量 |
| Nsight Compute app-range | 每个 captured model call/pair 的 DRAM read/write；replay-window GB/s 与正常服务带宽分开       |
| NVML                     | process/设备占用、实际 clocks、温度和功率；`utilization.memory` 是 busy-time 比例             |

NSYS 与 NCU 独立采集。NCU application-range 使用 worker 的 `cudaProfilerStart/Stop`，不传不支持的 `--profile-from-start`；clock/cache control 显式关闭，以免 kernel replay 改变并发解释。可选 private temporary directory 只有在系统范围确认无其他 NCU process 时使用，遵循 [NVIDIA lock-file 说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#faq)。

```bash
.venv/bin/python scripts/profile_libero_contention.py analyze \
  --measurement artifacts/contention/case/worker/measurement.json \
  --nsys artifacts/contention/case/trace.sqlite \
  --out artifacts/contention/case/report --no-plots
.venv/bin/python scripts/profile_libero_contention.py report \
  --root artifacts/contention/run --out artifacts/contention/report --no-plots
```

分析不加载 CUDA。report 聚合有效 completed attempts，排除 failed/pending，按硬件、模型/fixture、实现和计时 cohort 分组。绘图需 Matplotlib，移除 `--no-plots`；中文字体使用已安装 Source Han/Noto CJK 或 `--font /path/to/font.ttf`。图表和 JSON 保留单位及来源。

## Official Spatial Flow4 scheduling diagnostic

此入口使用官方非 streaming Spatial、H40、Flow4（NFE=4）权重，比较完整优化路径的 S1 响应、S2 响应和请求对 makespan。`flow` 读取真实图像/状态/文本 fixture，每个请求对重置输入：S1 消费已完成的旧特征和下一个 fixture 的状态，S2 计算下一个 fixture 的视觉特征。相同输入安排适用于各调度模式。

### 1. 准备输入与路径

按[固定上游来源](upstream.md)、[CUDA 模型环境](environment.md)
准备完整任务 checkpoint、对应 LeRobot Spatial 数据集与 processor/config 缓存。完整任务权重可直接用于本诊断。
CUTLASS 选择与编译缓存见[环境配置](environment.md#optional-gr00t-fusion-and-integer-inference)。
当前 W8 full split-K profile 使用 Ada SM89 候选和严格 Torch guard（包括 `2.7.1+cu126`、
Torch revision/header 检查）；W4 使用现有 grouping/norm producers。两者保留 category 2、
site/group tactics，W8 另含 shape tactics。其他平台按[迁移指南](platform-handoff.md)重新选择后端与校准。

每次命令启动加载两份独立完整 CUDA 策略，未执行组件的权重也常驻；整数配置保留 BF16 权重，并准备 W8/W4 格式。双独立进程共加载四份策略，因此显存字段表示该诊断配置的进程占用。

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

### 2. 检查设备并记录上下文

`flow` 是低层入口，**不负责 GPU admission、预留或外来进程监控**；
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

### 3. 同 GPU 的匹配调度矩阵

下例含每 case 30 次 clean 请求、2 轮预热及独立短 profile。
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

| `--modes`                                                    | 测量内容                                   |
| ------------------------------------------------------------ | ------------------------------------------ |
| `vlm_only` / `dit_only`                                      | S2 / S1 独跑                               |
| `serial`                                                     | S1 先完成，再启动 S2                       |
| `concurrent_equal` / `concurrent_dit_priority`               | 等优先级 / DiT 高优先级并发                |
| `vlm_only_fenced` / `serial_fenced`                          | 带已完成层边界 fence 的独跑/串行控制       |
| `concurrent_equal_fenced` / `concurrent_dit_priority_fenced` | 保留 fence 成本的并发控制                  |
| `cooperative_dit_priority`                                   | 在已完成的 VLM 层边界让出，执行 DiT 后恢复 |

`optimized` 开启 whole VLM graph；两组均使用 whole action graph、K/V reuse、静态准备与编译 guard。
将 `GROUP=cooperative` 并换新输出后重跑同一命令，整个组统一关闭 whole VLM graph、保留相同空操作层 hook；
仅 fenced/cooperative cases 同步层边界。用匹配 fenced controls 区分 fence 成本和让出收益；
`--vlm-chunk-blocks 1`（默认）或 `2` 设置边界粒度。不要把跨 group 的差值直接称为调度收益。
更换 `PRECISION` 后顺序重跑；同卡不同条件不要同时启动。CUDA priority 是提示，不是硬抢占保证。

### 4. 同进程双 GPU

上一轮退出、两卡预检通过后运行。完整 UUID 按 `GPU_S1_UUID,GPU_S2_UUID`
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

<details>
<summary>可选：独立进程连续回放与错开放行</summary>

此例连续回放各 600 次，关闭 profiler；每个进程内部两份
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

</details>

### 5. 读取结果

先确认正常退出、`status.json` 为 `complete` 且无 `failure.json`；
异常后 status 可能保留旧状态，缺少最终结果不能视为完成。保留以下本地产物：

| 文件/字段                                              | 用途                                                       |
| ------------------------------------------------------ | ---------------------------------------------------------- |
| `status.json`、`failure.json`、外层 `.log`/`.pid`      | 当前阶段、失败原因及进程归属                               |
| `start-ready.json` / `start-gate.json`（指定 gate 时） | 完成预热后的 ready 与实际放行记录                          |
| `measurement.json`、`samples.jsonl`                    | schema `libero-flow4-contention` v1；汇总与逐请求记录      |
| `fixtures.json`、`features.json`、导出的 observation   | 输入身份、旧特征 generation/hash/准备时间                  |
| `executed-coverage.json`、`graph_audits`               | 量化执行覆盖、capture/replay 与 action K/V 审计            |
| `provenance`、`hardware`、`runtime`、`memory`          | 源码/checkpoint 配置 hash、UUID、软件/绑核、常驻和峰值显存 |

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
本节使用 `libero-flow4-contention` schema；campaign 的 `analyze`/`report` 使用另一份 streaming schema。

`--profile-iterations 1` 在全部 clean cases 之后另跑每 case 的短 Torch profiler window，
保存在 `profiles` 与 `trace-*.chrome.json`，不混入 clean samples。
高级开关 `--cuda-profiler-range` 仅在 `--profile-iterations > 0` 时可用，标记单独的 profile window。
外部 Nsight 可按 `--capture-range=cudaProfilerApi --capture-range-end=repeat` 捕获各 case 的范围，
该分支仍启用 Torch profiler；与外部 Nsight 联用需要独立集成验证。使用新输出保留整个调用的 profiler 证据，clean timing 则单独采集。NVTX 标注实际 S1/S2 request/fixture；
应检查真实 CUDA activity 与 stream ID 才能判断 kernel 重叠。

## Auxiliary SO100 replay

SO100 H16 BF16 录制观测用于检查 S1/S2 执行机制。`coexecution.run` 默认读取 `outputs/pir2-so100-smoke/checkpoint-10`，以及已准备在 GPU 的 SO100 观测；Spatial 任务实验使用前述入口。

```bash
PIR2_GPU=0 bash scripts/run_single_gpu_baseline.sh measure
PIR2_GPU=0 bash scripts/run_single_gpu_baseline.sh runtime
PIR2_GPU=0 bash scripts/run_single_gpu_baseline.sh protocols
```

launcher 先检查占用；同卡逐条运行，用 `PIR2_OUTPUT` 指定新目录。trace mode 使用 PATH 上的 Nsight Systems 或 `PIR2_NSYS`。paired 对照恢复 input、cache version、RNG 与 rolling buffer，速度分母用实测 serial pair；model loading、预处理、CPU 解码及机器人 I/O 位于本 workload 计时外。

```bash
nsys export --type=sqlite --output=artifacts/coexecution/run/nsys.sqlite \
  artifacts/coexecution/run/nsys.nsys-rep
.venv/bin/python -m coexecution.analyze_trace \
  artifacts/coexecution/run/nsys.sqlite --output artifacts/coexecution/run/overlap.json
```

`analyze_trace` 对实际 kernel intervals 求交；`operator_analysis` 把 launch 归到正确 CPU thread 的最内层 NVTX range。profile latency 与 profiler-free latency 分开，`--plot` 可选 Matplotlib 输出。

periodic replay 保留 release period、有界 in-flight 和实际 feature versions；deadline 分母包含 dropped requests，消费时计算 cache age，并检查 release/completion/drop 守恒。固定请求对解释资源争用，periodic replay 解释排队；真实任务表现仍由 LIBERO wall-clock 协议测量。
