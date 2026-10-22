# 时序协议：推理结果何时成为动作

本项目用两类对照连接模型速度与控制质量：**algorithm 固定图像/动作延迟，deployment 测量实际墙钟调度。** 两者都使用真实 VLM 图像特征，并记录 checkpoint、初始条件和动作来源。评测结果按协议、coupling 和 clock 分组。

## 先选择推理与执行关系

| 模式                                | 模型工作                                                 | 动作执行                               | 主要问题                      |
| ----------------------------------- | -------------------------------------------------------- | -------------------------------------- | ----------------------------- |
| `algorithm`                         | 固定历史图像、当前状态；默认图像延迟 3 步、动作延迟 1 步 | 计算等待期间物理时间暂停               | 相同延迟下的策略能力          |
| deployment / `asynchronous + paced` | 独立 S2 VLM 与 S1 动作 worker，消费最新可用缓存          | 绝对墙钟 20 Hz，合法动作缓冲           | 实际延时、缓存年龄与 deadline |
| deployment / `asynchronous + ready` | 同样的真实 VLM 缓存控制器，Flow                          | 原子物理步结束后立即接纳合法结果并继续 | 去掉墙钟 pacing 后的动作时效  |
| deployment / `synchronous`          | 当前观测的完整 VLM + Flow4                               | 推理结束后执行 K 步，再开始下一轮      | 推理与执行串行的基线          |
| deployment / `one_step_async`       | 完整 VLM + Flow4，提前一个物理步推理                     | K=1、D=1，结果未就绪则等待             | 单步推理/执行重叠             |
| deployment / `chunked_async`        | 完整 VLM + Flow4，在段边界前 D 步启动                    | 每段执行 K 步，要求 1 ≤ D ≤ K          | 动作前缀与异步深度的权衡      |

当前 evaluator 固定 LIBERO 每步推进 **0.05 秒仿真时间**，并要求 `--control-hz 20`。这不保证每个 `env.step` 恰好耗费 50 ms 墙钟时间：

- `paced` 缓存异步有全局 20 Hz 墙钟 deadline。
- `synchronous + paced` 只在执行段内按 20 Hz 运行，推理段另占时间。
- `ready` 不额外等待 50 ms，实际频率由计算、环境步进和等待决定。
- 同步及 ready 模式的全局固定频率有效性为 `null / not applicable`。

`--execution-prefix-steps` 是 `--synchronous-chunk-steps` 的别名，默认 K=8；plan 使用对应下划线字段，同层冲突报错。K 的上限是 checkpoint 的**原生解码长度**，由 processor modality 的 action `delta_indices` 决定，不能用训练 H40 替代。plan 在 GPU 准入前检查全部条件：优先 root metadata，缺失时回退 `processor/`；服务加载后再次核对 identity。

## 固定延迟算法对照

默认图像年龄为 3 ticks（150 ms），动作放行为 1 tick（50 ms），状态取当前物理步。计算期间不推进物理；每种方法保留已承诺前缀并使用下一合法段，缺失的起始图像历史显式补齐。

`--artificial-compute-wait` 用于检查该时间边界。GPU contract 给每次请求增加 100 ms 等待后比较动作与最终图像，以验证计算等待没有改变物理协议。任务成功率仍由正式闭环评测给出。

## 缓存异步部署

S2 持续处理真实图像并发布特征，S1 使用当前状态、选定特征和不可改写的动作前缀。控制线程独立于未完成 RPC；提前计算出的观测到物理 tick 才释放。结果只写入合法未来槽位：已执行槽位过期、已承诺槽位受保护，缓冲耗尽时使用零笛卡尔增量与上一夹爪命令。

Flow 发布未过期的原生干净计划；πR² 发布本次新变干净的段。πR² 常规请求使用一次 DiT evaluation；整缓冲恢复单独计为 4 次 bootstrap 加 1 次正常更新。控制频率、请求频率和有效发布频率分别统计。

### 校准与 episode 准备

solo 预热和校准在 scored control 外完成：

| 记录                 | 规则                                                                        |
| -------------------- | --------------------------------------------------------------------------- |
| `calibration.json`   | 保存全部 12 个动作样本，前 2 个标记 excluded；RPC/server 数组使用其余 10 个 |
| 初始动作预算 d       | `min(5, ceil((P95 RPC + 0.005) / 0.05))`，越过训练预算另记标志              |
| `slow-warmup.json`   | 保存全部视觉样本及 excluded 标记；`--slow-warmup-calls` 默认 12             |
| `vision_preparation` | 每 episode 用当前任务做一次真实 S2 准备调用，再进行新鲜 S1 bootstrap        |

episode 准备调用不返回特征载荷，其输出不进入 scored mailbox/S1 cache；记录实际来源、wall 时间、forward 和 graph 增量。这样任务切换也能在控制开始前准备相应路径。正式准入的每帧仍运行真实 VLM。并发控制期间的 latency 使用 episode trace，区别于 solo 校准值。

### Responsive πR² scheduling (opt in)

`--deployment-scheduling-policy conservative` 为默认：d 校准后单调不减，结果在 tick 边界处理。`responsive` 支持 paced、cached asynchronous 的 deployment πR²，且不与 fixed-delay override 组合。plan 支持全局或逐 condition 设置，配置和 resume identity 记录选择。

responsive 在保留初始 solo 校准的基础上，使用每 episode 最近 8 个有效完成的 S1 请求时长：

1. 样本为 `completed_s - submitted_s`，包含执行器排队、安装缓存与 RPC。P95 按 `(n - 1) × 0.95` 线性插值。
2. 目标为 `ceil((P95 + 0.005) / physical_period)`，限制在 1..5。
3. 窗口满且连续 3 次目标低于当前 d 时，每次降低 1 tick 并重置计数；无效/非有限/负样本排除并重置降低计数。
4. deadline miss 立即提高到当前 d+1、最新请求时长预算、从原定物理边界到完成的预算三者最大值，再限制到 5。第三项包含启动迟延；超过 5 的需求仍显式记录。

Future 回调只发 Event。控制器在当前原子 `env.step` 后的 pacing 空闲期处理结果，以**下一未执行物理步**为发布边界，等待截止于下一物理 deadline。传感器仍按 tick 释放，新 S1 请求仍须到达前一后端 anchor `r + old_d`。在飞请求、结果原有效区间和已承诺前缀保持其语义；减小 d 影响后续请求，不改变普通 πR² 的 DiT NFE。

`delay_policy_decision` 保存窗口、候选/选定预算、降低计数、原因和越界标志。`request_completed` 保存 `rpc_completed_s`、`controller_ready_observed_s`、`controller_processed_s` 及发布计数、缓冲快照。更早处理可能只减少入队等待，首次执行 tick 仍受合法槽位约束。终止后排空的结果标记 unused。

### S2 准入频率

`--vision-max-hz` 默认 0（不限），例如 15 表示 scored S2 准入间隔至少 1/15 秒。首帧立即准入，此后基于实际前次准入计算，不补发迟到的请求。paced 摄像头仍按物理 20 Hz 提供图像；等待期间 latest-frame mailbox 可被更新图像替换，真正准入时才取最新帧。已完成特征立即进入发布路径；独立服务时长下限仍适用。

`vision_admission_wait` 记录可准入时间和等待，`vision_admitted.admitted_s` 在 mailbox 锁内记录，与实际 RPC 的 `started_s` 分开。`vlm_work` 汇总 offered/admitted/completed/replaced/dropped，实际准入频率在有定义时为 `(admissions-1)/(last-first)`。stop 在同一锁内观测，随后不再准入；已经准入的 RPC 可以稍后开始或完成，控制后的工作保留。准备、bootstrap 和 solo 校准在此限频之外。

<a id="controlled-completion-release-experiments"></a>

### Client-visible release floors

`--action-min-service-ms`、`--vision-min-service-ms` 默认 0；设置后先运行真实模型，再把结果释放延迟到最小时长。`--deployment-fixed-delay` 可把 bootstrap 和运行期 d 固定在 1..5；这些控制只用于缓存异步 deployment，并进入 resume identity。

```bash
python scripts/evaluate_libero_protocol.py \
  --protocol deployment --variant pir2 --output .local/experiments/release-floor \
  --task-ids 0 3 --episodes-per-task 1 --no-video \
  --action-min-service-ms 80 --vision-min-service-ms 160 \
  --deployment-fixed-delay 2
```

动作 floor 从提交开始，含排队、特征安装和真实 plan RPC；视觉 floor 从 worker RPC 时间戳开始。校准使用同样的 floor。原始请求已超目标时直接释放，记录 `floor_miss` 及 raw/total overrun，模型 forward 数保持不变。

`raw_completed_s`、`raw_service_seconds` 位于 hold 前；`released_s/completed_s` 位于 hold 后、元数据记录和 Future 完成之前。它们不是精确的消费者可见时刻，`added_wait_seconds` 也包含 bookkeeping 和线程调度。sleep 释放执行资源，与实际降频中的 GPU 争用不同；实验分别报告 raw/release 时长、真实时钟、overrun 与控制质量。

### Ready-driven GR00T execution

`--execution-clock ready` 支持 deployment Flow 缓存异步控制。每个原子 `env.step` 后接纳合法结果、取得当前状态并立即执行下一动作，无额外 wall pacing 或末尾 padding。每个请求保留一个逻辑承诺步；过期、受保护和缺失槽位按同样规则计数。

该模式的 `action_delay_policy=one_step_committed`，solo RPC 校准作为独立记录保留；预算不由墙钟延时除以物理周期推导。`actual_control_hz`、`simulated_elapsed_seconds`、`controlled_wall_seconds` 和 `real_time_factor` 分别记录，deadline/lateness/固定频率有效性为 null。

前端标签“就绪即接纳（StreamingVLA式调度，GR00T）”指 GR00T 干净动作块上的调度对照；使用的权重、动作生成和算法身份仍为 GR00T，未接入 StreamingVLA 的 AFM/AEO。ready 不支持 πR² 或 fixed-delay。

## 特征传输与 worker 布局

这些选项服务于缓存异步控制，独立于权重量化精度。选用相同 precision/runtime flags 比较传输，先固定布局，再比较 paired 与 split。

| 配置                                     | 数据路径与契约                                                                     |
| ---------------------------------------- | ---------------------------------------------------------------------------------- |
| `--feature-transport numpy_fp32`（默认） | 上游 NumPy 字典传输                                                                |
| `numpy_bf16`                             | 已有 BF16 按带版本标签的 uint16 原始位传输；保留其他 dtype、mask、标量形状和元数据 |
| `cuda_local`                             | 同 GPU、paired Flow 两份 policy 间传递带所有权的驻留特征 handle                    |
| `--feature-rpc separate`（默认）         | 新特征先 install 再 plan                                                           |
| `--feature-rpc combined`                 | 新特征安装合入同一次 plan 请求；成功回复后才推进客户端 installed sequence          |
| `--server-layout split`（默认）          | 独立服务进程                                                                       |
| `--server-layout paired`                 | 一个进程，两份独立 policy、CUDA stream 和 endpoint                                 |

传输编码必须同时设置在服务与 evaluator。NumPy 导入获得独立存储，CUDA stream 就绪后才规划；未知 schema、dtype 或错误 shape 在安装前拒绝。`feature_wire.payload_nbytes` 是数组存储字节，不含序列化头。BF16 codec 只改变传输表示，不新增数值量化。combined/separate 均在既有控制边界选择特征并按 sequence 去重。

paired 服务使用 `--device cuda:0 --vlm-device cuda:0 --port 5570 --vlm-port 5572`。plan 管理一个进程并等待两 endpoint，要求同一物理 GPU。每份 policy 串行提交；常规 GPU 工作可以重叠，冷 VLM/DiT 图捕获通过进程级 exclusive guard 和 quiescence 协调。startup、reset、继承 endpoint 和 teardown 参与协调；S2 reset 不重置 S1 RNG。支持 per-DiT/whole-VLM graph，拒绝 paired whole-action graph。

`cuda_local` 在真实 VLM producer event 完成后发布，consumer 等待 event、登记消费 stream 后安装独立引用。默认 16 个 lease（`--feature-max-leases`）保存 owned snapshot；容量耗尽报错，保持在飞读者有效。handle 绑定 metadata、dtype、shape、device、进程和 client session；过期或改写 handle 拒绝使用。

客户端在 latest、selected、pending 或 fallback 引用期间持有 lease，最后引用释放在下次视觉 RPC 批处理。停止时先 join action consumer，再用新 RPC 关闭 session，清理包括丢失回复在内的特征。producer reset 失效 session，已有读者引用仍安全。`payload_nbytes=0` 表示没有张量序列化，`resident_nbytes` 给出驻留存储；handle 元数据仍有传输开销。

CUDA 契约检查（在准备好的模型环境显式运行）：

```bash
PIR2_GPU_TESTS=1 python -m pytest -q tests/gpu/test_feature_transport_gpu.py
PIR2_GPU_TESTS=1 python -m pytest -q tests/gpu/test_resident_features_gpu.py
```

前者覆盖全部 65,536 种 BF16 位型、其他 dtype/shape 和独立存储；后者检查 stream 就绪、allocator 生命周期、两份独立 tiny CUDA policy 的并发冷捕获、新图像与 RNG。官方 checkpoint 数值和 Spatial 闭环质量另由对应模型实验报告。

## Synchronous inference and action execution / 同步推理与动作执行

`--execution-coupling synchronous` 用一个普通 Flow action endpoint，每轮获取当前图像与状态，完成真实 VLM + Flow4，再执行至多 K 个干净动作。推理时暂停物理，执行段不启动神经推理；首次成功或环境终止立即停止。原生解码后缀按未执行数量记录，下轮重新取观测。

`paced` 在执行段维持 20 Hz，包含最后动作的完整 dwell，然后开始下一轮推理；`ready` 不增加等待。`actual_control_hz = control_steps / controlled_wall_seconds` 包含首轮推理等待，两种 clock 均没有全局 20 Hz 有效性声明。

启动 action 服务后，使用仿真环境执行：

```bash
python scripts/evaluate_libero_protocol.py \
  --protocol deployment --variant flow --port 5570 \
  --execution-coupling synchronous --execution-clock paced \
  --execution-prefix-steps 8 --task-ids 0 3 --episodes-per-task 1 \
  --max-steps 300 --record-replay --replay-max-frames 301 \
  --output .local/synchronous-examples/deployment
```

完整策略模式（synchronous / one_step_async / chunked_async）使用单个普通 Flow endpoint，保持默认特征接口；paired/resident、缓存服务 floor、视觉限频、fixed-delay 和人工计算等待不适用。每任务在同 endpoint 做完整 warmup，再重置一次 episode RNG，正式周期不调用重置 seed 的 bootstrap。冷 JIT warmup 超时默认 900 秒，可用 `--synchronous-warmup-timeout-seconds` 调整；随后恢复原客户端/socket 的 120 秒设置。plan 支持对应下划线字段。

`cycle_phase` 保存真实 inference/execution 起止、物理步范围和暂停状态；`synchronous_cycle` 保存图像 hash、原生计划、VLM/DiT 段时间、forward/graph 计数和丢弃后缀。`inference_wall_seconds` 是包含输入准备的完整客户端调用，`synchronous_rpc_started_s/completed_s` 表示该完整调用边界。服务端 `vlm_stage_seconds/action_stage_seconds` 到达对应 stream 完成边界；前端标为 `server_vlm_stage/server_action_stage`，与独立 RPC 或 GPU kernel 区间分开。

## One-step asynchronous policy/execution / 单步异步策略与动作执行

`one_step_async + ready` 固定 K=1、D=1。一次计入 scored control 的 bootstrap 用观测 0 生成动作 0；执行动作 k 前，主线程复制当前观测 k 交给专属模型线程，计算下一目标步 k+1 的完整 VLM + Flow4，同时执行当前计划第一个动作。原子步结束后若下一结果未完成，明确等待，不执行旧后缀或 fallback。

```bash
python scripts/evaluate_libero_protocol.py \
  --protocol deployment --variant flow --port 5570 \
  --execution-coupling one_step_async --execution-clock ready \
  --execution-prefix-steps 1 --task-ids 0 3 --episodes-per-task 1 \
  --max-steps 300 --record-replay --replay-max-frames 301 \
  --output .local/one-step-async/deployment
```

bootstrap 的 source/target 为 0/0，随后为 0/1、1/2；图像和状态共同滞后一个物理步，未引入未来状态预测。该调度受 [The Speedup Paradox §3.1](https://arxiv.org/pdf/2606.28529v3) 启发。重叠预算来自实际 `env.step` wall 区间，并包含渲染/模型同卡争用；物理一步 0.05 秒不能直接充当 wall overlap。

后台线程独占自己的 PolicyClient/ZeroMQ context，主线程独占 MuJoCo，跨线程输入为独立只读数组。正式请求 send/receive 分别保留 120 秒上限，Future 等待覆盖两者加 30 秒余量，关闭也有上限。超时或模型/仿真异常保留不完整 trace 并令实验失败。

`request_id`（兼容 `request_tick`）唯一标识请求，`target_tick` 与 `source_tick` 分别标识目标和源观测。bootstrap request0 与预取 request1 都可使用观测0，feature_sequence 按 request_id 区分。`policy_request_submitted` 与 `policy_rpc_started/completed` 分开记录提交和实际 RPC；`one_step_async_request` 保存计划与接纳状态，`one_step_async_execution/control_tick` 证明执行。

## Chunked asynchronous lookahead / 动作块长度与异步深度

`chunked_async` 独立设置前缀 K 和提前量 D。在目标段 T=K、2K… 的边界前，于物理步 T−D 取得观测并发起下一次完整 VLM + Flow4；当前段按原生 offset 0…K−1 执行完后再换段。结果提前就绪仍保留当前承诺段，未就绪则等待。

```bash
python scripts/evaluate_libero_protocol.py \
  --protocol deployment --variant flow --port 5570 \
  --execution-coupling chunked_async --execution-clock ready \
  --execution-prefix-steps 8 --async-overlap-steps 4 \
  --task-ids 0 3 --episodes-per-task 1 --max-steps 300 \
  --record-replay --replay-max-frames 301 \
  --output .local/chunked-async/deployment
```

要求 `1 <= D <= K <= native decoded horizon`，只支持 deployment Flow、ready 和单 action endpoint。D=0 使用 synchronous；旧 one_step_async 保留 K1/D1 的 v1 接口。可比较 K1/D0,1；K2/D0,1,2；K4/D0,1,2,4；K8/D0,1,4,8。普通段内步骤复用前一 env.step 观测，bootstrap 和指定预取边界强制刷新摄像头。

bootstrap 的 offset j 滞后 j 物理步，后续段 offset j 滞后 D+j；wall AoI 使用真实 capture-to-apply。成功或终止可发生在任意 offset；max_steps 截断接纳长度，例如 K8、上限300时 target296 接纳4步。接纳但未执行、原生未接纳后缀和最终完全 unused 请求分别计数。

v2 `chunked_async_request` 在段结束后写最终计数（`t_s=finalized_s`），接纳/接收时刻为 `adopted_s/received_s`。`request_id=request_tick=target_tick` 按 K 递增；`execution_id=tick` 唯一标识动作并保留 `executed_offset`。`chunked_async_boundary` 记录边界是否就绪及等待。尾部 unused target 可以大于终止步；drain 的 physics tick 使用实际 `control_steps`。

## 时间、来源与汇总合同

### 完成、发布、执行是三个边界

| 证据                                        | 能说明什么                                                      |
| ------------------------------------------- | --------------------------------------------------------------- |
| `rpc_completed_s` / `completed_s`           | 客户端记录的完成/释放时刻，早于后续 Future bookkeeping          |
| `controller_ready_observed_s`               | 控制器实际观察到 readiness；Future 首次变 done 的精确时刻未测量 |
| `controller_processed_s` / `action_adopted` | 控制器处理结果；兼容 `action_adopted` 可能 installed_slots=0    |
| `installed_slots` 与 publication 后 buffer  | 写入合法未来动作槽位                                            |
| `feature_published`                         | 视觉特征对消费者可见                                            |
| `control_tick.executed_slot`                | 已执行动作与其精确 producer                                     |

因此 `action_ready_to_adopt_seconds` 和 `action_ready_to_first_apply_seconds` 分别度量处理与首次执行等待，并按 producer request 关联。slot 保存 tick、future/committed/fallback、producer_kind/request_tick；bootstrap/fallback 没有普通模型请求 producer，fallback 的 `reserved_at_tick` 给出承诺时间。bootstrap feature sequence 为0，异步视觉 worker 从1开始。

### RPC 与缓存年龄

timing version 2 在可用时优先使用 `server_plan_started_s` 作为 cache use 边界：

| 字段                                                      | 范围                                                              |
| --------------------------------------------------------- | ----------------------------------------------------------------- |
| `plan_rpc_started_s`                                      | 实际客户端网络调用起点，combined 时位于安装之前                   |
| `server_install_started_s/completed_s`                    | 最近一次缓存导入；`feature_installed_in_request` 区分新安装和复用 |
| `server_plan_started_s`                                   | 可选安装与 stream 就绪之后、服务规划之前的 host 边界              |
| `server_seconds`                                          | 从规划边界开始的服务时间                                          |
| `server_request_seconds`                                  | 还包含 combined 安装                                              |
| `raw_service_seconds`                                     | 队列、网络、安装和规划的完整服务 wall                             |
| `cache_age_at_request_ms` / `cache_age_at_server_plan_ms` | 分别在 RPC 入口/安装后规划边界计算的年龄                          |

这些 loopback 时间戳使用同一 host monotonic clock。旧记录缺少新字段时保留 `plan_rpc_started_s` 与显式 `use_boundary`。wall 边界差额不等于 CPU 活动或 kernel 时间。

同步完整调用包含 worker 输入准备；整策略异步的 `inference_request_seconds` 从主线程提交到响应，`inference_wall_seconds` 从 worker 准备输入后开始 RPC。比较整体延时需使用匹配边界。真实重叠由 RPC/执行区间的交集计算，服务端 VLM/DiT 交集另列；提交顺序本身不证明 GPU 重叠。

### 终止后的工作与回放

整策略异步首次成功、环境终止或 max_steps 后停止执行；已提交请求排空并标 unused，已知最后一步不再额外预取。`control_end` 截止控制，`async_work_end` 截止全部工作。policy_calls、forward_counts 和完整 RPC 分布包含末尾未用请求，adopted_policy_calls、执行步数与后缀丢弃分别列出。

queue report 的 `requests` 表示控制窗口与 censoring，`work_requests` 表示全部 bootstrap/预取/drain；`coexecution.async_evidence` 核对每个请求的提交、图像 hash、阶段、source/target、接纳或 unused 处置与真实动作。`replay_requests` 保留完整阶段，`work_end_s` 保留尾部。视频在推理暂停或末尾排空时保持已有真实帧，模型轨道继续显示实际工作。

trace 的 `seq` 是唯一插入序号，`t_s` 属于共享 monotonic clock；跨线程插入顺序不保证时间递增。线程收集独立元数据，JSON 写盘在控制和 worker drain 之后。常见生命周期包括 control_start/end、camera_offered/replaced/dropped、vision_started、feature_published、action_submitted/started/completed 和缓冲快照。queue trace 保存来源元数据与已执行动作；启用机器人录制时，图像保存在独立媒体资产中，embedding 不持久化。详细 schema 和视频映射见[回放接口](../coexecution/replay/README.md)。

### 配对与统计

- paced 缓存异步的严格时序有效性要求平均频率距 20 Hz 不超过2%，最大迟延与相邻间隔误差各不超过5 ms。时序无效 episode 仍计入成功率分母。
- 默认全部 Spatial 10任务；`--task-ids 0 3` 等子集标记 selected_task_subset，单独报告。首次成功即结束，300步上限不保证每 episode 暴露15秒。
- 初始条件配对核对 fixture 模型、settled state 和初始 RGB 哈希。LIBERO wrapper 在 hard reload 前清理累积的 object-property sampler，保持成功谓词。
- K/D 对照报告成功率、actual_control_hz、总控制 wall/动作步数、段边界等待和 gap P95、AoI 与实测重叠。`boundary_gap_seconds` 是前段末步完成到后段首步开始；`adjacent_action_gap_seconds` 是相邻动作起点间隔，较大 K 的段内短间隔会影响其分布。
- 分位数从原始请求/动作汇总，保留 count 和 sum；不平均 episode 分位数。记录相机拷贝和 trace 的插桩成本，视频编码在控制结束后执行。

运行资源与 plan 见 [LIBERO 指南](libero.md)。各条件的配置、episode、trace、summary 和初始条件审计保留在本地输出目录，并绑定实际源码、权重和硬件身份；训练与计时部署使用独立时段或空闲设备。
