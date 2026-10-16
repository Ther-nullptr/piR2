# Two timing protocols

Both protocols use real VLM image features, the same checkpoint/data identities and paired initial conditions. Their deployment coupling selects the image/state age and inference/execution relationship; the table describes the original cached-worker deployment baseline. Scores from different protocols or coupling modes must not be pooled.

| Protocol | Observation/action timing | Purpose |
|---|---|---|
| Algorithm | Image age fixed at 3 ticks (150ms); action release fixed at 1 tick (50ms); current state | Compare behavior at identical simulated delays |
| Deployment (cached asynchronous baseline) | Absolute wall-clock 20Hz control; asynchronous real VLM and action workers | Measure behavior with actual processing/transport delays |

In the algorithm protocol, computation waits do not advance physics. Each method executes the committed prefix, then uses the next legal segment. The initial missing image history is explicitly padded. A GPU contract check adds 100ms wait per call and compares actions and final images; this is a timing check, not a success-rate evaluation.

Deployment keeps control independent from unfinished RPCs. An observation computed early is withheld until its physical tick. Committed commands cannot be overwritten, expired slots are discarded, and buffer exhaustion uses zero Cartesian deltas with the previous gripper command. Flow publishes the unexpired native clean plan; πR² publishes only its newly clean segment. πR² normally uses one DiT evaluation per request; full-buffer recovery is explicitly counted as four bootstrap evaluations plus one normal update.

Warm-up and full request calibration occur outside scored control. Control frequency, policy query frequency and plan publication frequency are separate metrics. Record action/VLM latency, actual cache age, state/image timestamp difference, control lateness, expired/fallback slots and recovery counts. Quantiles are calculated from raw requests, not averaged episode quantiles.

Deployment optionally accepts `--vision-max-hz 15` to limit scored S2 admissions. The default zero is unlimited. The first admission is immediate; each later admission is at least `1 / vision_max_hz` after the previous actual admission, with no catch-up after a late wake or slow request. Camera offers continue at physical 20Hz. The newest pending frame is selected only at admission, so newer frames can replace it throughout a wakeable rate wait. Completed features publish without an admission-rate hold; the independent service floors below still apply. Bootstrap and solo warmup/calibration remain outside this limit. Experiment plans may set `vision_max_hz` globally or per condition; a condition value, including zero, overrides the plan, and only deployment evaluator commands receive it.

`vision_admission_wait` traces the eligible time and replaceable mailbox; `vision_admitted` records `admitted_s` under the mailbox lock, separately from actual RPC `started_s`. Request rows retain both timestamps, `vision_max_hz` and elapsed `admission_wait_seconds` (from first rate wait through admission, including scheduling). `vlm_work` reports offered/admitted/completed/replaced/dropped counts, admission starts and waits, and `actual_admission_hz` as `(admissions - 1) / (last - first)` when defined. An interrupted final wait has no admission row; its wait and stop events remain in the trace. `vision_stop_requested` records when stop was observed under the same lock and retains the controller's separate cutoff. No new admission follows observed stop; an already admitted RPC may begin or finish afterward, and its timestamps and post-control counts are retained.

For a hardware scan, the evaluator accepts `--task-ids 0 3 --episodes-per-task 1 --max-steps 300 --no-video`. The default still selects all ten Spatial tasks. A selected subset is labelled `selected_task_subset`, and its expected episode count uses only those tasks. Episodes retain ordinary first-success/environment-termination semantics: `physical_control_seconds` and the deployment `controlled_start_s`/`controlled_end_s` report actual exposure; a 300-tick cap does not promise 15 seconds after an early success. Subset success is not a full-suite result.

Deployment calibration stores all 12 action samples in `calibration.json`, marking the first two as excluded; `action_rpc_seconds` and `action_server_seconds` contain the ten included samples. The action budget remains `min(5, ceil((p95 RPC + 0.005) / 0.05))`, with an explicit over-budget flag. `slow-warmup.json` similarly keeps every visual RPC/server sample and marks the first two excluded. `--slow-warmup-calls` defaults to 12. These are solo warm calibration measurements; concurrent-control latency comes from the episode trace.

Feature transport defaults to `--feature-transport numpy_fp32`, the upstream NumPy dictionary. Opt in to `--feature-transport numpy_bf16` on **both services and the deployment evaluator** to send existing BF16 features as explicitly tagged raw uint16 words in a versioned envelope. This changes their representation, with no new quantization: BF16 bits, other floating dtypes, bool/int masks, scalar shapes and metadata are preserved. Imported tensors own their storage and finish transfer on the endpoint's CUDA stream before planning. Unknown schemas, dtype tags and malformed shapes are rejected before cache installation. Service identities declare the encoding/source; VLM metadata and action cache audits report `feature_wire` with actual field dtypes and `payload_nbytes` (array storage bytes, excluding serialization headers).

The evaluator additionally accepts `--feature-rpc combined` to install a newly selected feature inside the same `plan` request. The default `separate` sends `install` then `plan`. Both modes select at the existing control boundary and deduplicate the same feature sequence; combined mode advances the client's installed sequence only after a successful reply. Bootstrap, calibration and scored workers use the declared mode, while action service floors still begin at submission and include installation. Experiment plans accept `feature_transport` and `feature_rpc` globally or per condition, independently of weight-precision flags. Nondefault evaluator feature controls require deployment; resume configuration and calibration bind these settings. The physical 20Hz loop and the optional 15Hz S2 admission cap are unchanged.

Timing version 2 adds explicit server boundaries to both reference and candidate requests. `plan_rpc_started_s` remains the actual client network-call start; in combined mode it precedes installation. `server_install_started_s` and `server_install_completed_s` describe the most recent imported cache, while `feature_installed_in_request` distinguishes a combined install from reuse. `server_plan_started_s` is recorded after optional installation and stream readiness, before server planning; it is a host boundary, not an exact neural-kernel timestamp. `server_seconds` measures planning from that boundary; `server_request_seconds` also includes combined installation. Whole-call `raw_service_seconds` includes queueing, network, installation and planning. None of these wall-time differences is a CPU-work measurement.

Queue reports and feature-usage audits prefer `server_plan_started_s` for `use_s` and cache age, record `use_boundary`, and retain legacy `plan_rpc_started_s` semantics when the new timestamp is absent. Existing `cache_age_at_request_ms` remains age at RPC entry; `cache_age_at_server_plan_ms` is the comparable post-install boundary. These loopback timestamps share the host monotonic clock. Feature publication, selection and physical execution keep their existing timestamps and source lineage.

Run `PIR2_GPU_TESTS=1 python -m pytest -q tests/gpu/test_feature_transport_gpu.py` in the pinned CUDA environment for all 65,536 BF16 bit patterns, other dtype/shape preservation, fresh storage and stream-readiness gates. These codec checks perform no neural inference and do not establish LIBERO task success.

Same-process Flow workers are an explicit `--server-layout paired` option. Start
one service with `--device cuda:0 --vlm-device cuda:0 --port 5570 --vlm-port 5572`
and select `--server-layout paired` on the deployment evaluator. The service loads
two independent official policies, each with its own CUDA stream and endpoint.
The default layout remains `split`. Experiment plans accept `server_layout`
globally or per condition; paired plans start and own one process, wait for both
endpoints, and require the same declared physical GPU.

With paired Flow workers on one physical device, `--feature-transport cuda_local`
replaces tensor serialization with versioned, owner-qualified feature handles.
Each admitted frame still runs the real VLM. Publication follows the completed
producer event; the consumer waits on that event and records its stream before
installing independent references in its policy cache. A bounded pool (default
16 leases, service option `--feature-max-leases`) retains owned tensor snapshots.
It rejects capacity exhaustion rather than evicting an in-flight reader. Handles
bind the exact metadata, dtype, shape, device, process instance and client session;
stale or modified handles fail without substituting older neural outputs.

The client holds a lease while the feature is latest, selected, pending, or kept
as the action worker's fallback. Last-reference releases are batched on the next
vision RPC. Shutdown first joins action consumers, then closes the owned feature
session over a fresh RPC connection, including features whose replies were lost.
Reset invalidates the producer session while existing reader references remain
safe. `feature_wire.payload_nbytes` is zero tensor bytes for this transport;
`resident_nbytes` reports retained tensor storage. Handle serialization still has
metadata overhead. Pool counts appear in VLM responses and runtime identities.

Paired optimization installs adapters independently for each policy. Ordinary
CUDA submissions can overlap; first VLM/DiT graph captures take a process-wide
exclusive guard after quiescing prior GPU work. Simultaneous capture upgrades
drop their shared submission slots before waiting. Startup, reset, inherited
policy endpoints and teardown also participate. Paired whole-action graph mode
is rejected; per-DiT and whole-VLM graph options are supported. S2 reset does not
reseed S1's RNG. Graph capture/replay/failure deltas remain separate from physical
forward hooks, and `scored_cold_capture_requests` exposes captures during scored
control. Cold warmup costs stay outside steady measurement.

Condition-wide solo calibration prepares only its initial task metadata. Each
episode therefore also runs one real S2 preparation call for its current task,
before the fresh S1 bootstrap capture and scored control. This applies equally
to split and paired layouts and every feature transport. The call requests no
feature payload; its outputs never enter the scored mailbox or S1 cache. Every
subsequent admitted frame still runs the real VLM. Per-episode
`vision_preparation` records its actual source and wall times, physical VLM count
and graph deltas separately. It does not change the calibrated delay, the
nondecreasing delay rule, camera admission or scored completion timestamps.

Compare resident transport against **paired `numpy_bf16`** as well as split
`numpy_bf16`, with identical precision/runtime flags, 20Hz control and S2 admission
settings. This separates transport cost from process/context layout effects.
Run `PIR2_GPU_TESTS=1 python -m pytest -q tests/gpu/test_resident_features_gpu.py`
for stream readiness, owned storage, allocator lifetime and two independent tiny
pinned CUDA policies with simultaneous cold captures, fresh images and RNG
checks. Official-checkpoint equality, measured server latency and full Spatial
quality remain separate gates; W4 systems measurements do not establish quality.

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

### Ready-driven GR00T execution

`--execution-clock ready` is an opt-in deployment Flow mode. The default
`paced` mode retains its 20 Hz wall target and timing tests. Experiment plans
accept `execution_clock` globally or per condition. Ready rejects πR² and
`--deployment-fixed-delay` rather than silently changing their timing semantics.

Ready keeps the existing real VLM worker and GR00T clean action chunks. At each
boundary after an atomic `env.step`, the controller publishes any completed
result into legal, uncommitted future slots, takes a fresh state observation,
and executes the next command immediately. There is no additional 50 ms wall
wait or final period padding. An in-flight environment step cannot be replaced;
one logical action step is committed for each request. Results arriving after
their original target steps retain expired/protected counts, and missing legal
actions remain explicit zero-delta/gripper-hold fallback commands.

Each `env.step` still advances **0.05 simulated seconds**. `actual_control_hz`
measures wall execution rate; `physics_step_seconds`, `nominal_physics_hz`,
`simulated_elapsed_seconds`, `controlled_wall_seconds` and `real_time_factor`
keep physics units distinct. Request and effective adoption rates are separate.
`action_ready_to_adopt_seconds` and `action_ready_to_first_apply_seconds` use
the producing request's recorded `completed_s`, never temporal matching.
As documented below, that completion precedes Future bookkeeping; the measured
gap includes that bookkeeping and waiting for the next legal actuator boundary.

Ready has no wall-grid deadline, target frequency, lateness or fixed-rate
validity; those values are **null / not applicable**, not zero, pass or failure.
Solo RPC calibration is retained as evidence, while effective alignment stays
one step with `action_delay_policy=one_step_committed`; it is not derived by
dividing wall latency by the nominal physics period. The frontend labels this
“就绪即接纳（StreamingVLA式调度，GR00T）”. It is a scheduling comparison using
GR00T weights, not StreamingVLA AFM/AEO, paper weights or intermediate noisy actions.

### Client-visible release floors

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

### Synchronous inference and action execution / 同步推理与动作执行

`--execution-coupling synchronous` 与 `--execution-clock paced|ready` 独立。
默认仍为 `asynchronous`。同步模式使用一个普通 Flow action server：每周期
采集当前真实观测，依次完成真实 VLM 和 Flow4，然后执行至多
`--synchronous-chunk-steps 8` 个干净动作。推理期间不调用 `env.step`，执行段不
启动神经推理；首次成功或环境终止立即停止后续动作。每步仍推进 0.05 秒物理时间。

`paced` 只在执行段内按 20 Hz 节拍运行，包含最终动作的完整 dwell；下一轮推理
从该执行段结束后开始。`ready` 不增加等待。两者的全局控制频率都会反映真实推理、
环境和记录成本；同步模式不声明全局 20 Hz deadline，有效性字段为 null。
`actual_control_hz = control_steps / controlled_wall_seconds` 包含首轮推理等待。
原生解码长度保持模型配置，必须覆盖请求的执行段长度；未执行的计划后缀显式记录，
不挪用为下一周期的动作。

启动单个模型服务（CUDA 环境、checkpoint 与优化参数按已有环境文档配置），然后
使用仿真环境运行现有 evaluator；无需第二个 VLM 服务：

```bash
python scripts/evaluate_libero_protocol.py \
  --protocol deployment --variant flow --port 5570 \
  --execution-coupling synchronous --execution-clock paced \
  --synchronous-chunk-steps 8 --task-ids 0 3 --episodes-per-task 1 \
  --max-steps 300 --record-replay --replay-max-frames 301 \
  --output .local/synchronous-examples/deployment
```

现有 `coexecution.libero.experiments` plan 支持全局或逐 condition 的
`execution_coupling`、`execution_clock`、`synchronous_chunk_steps`，同步 condition
只启动 action worker。同步模式拒绝 πR²、algorithm protocol、paired/resident
feature exchange、服务时长下限、视觉限频和 fixed-delay 控制。每任务先做本 endpoint
完整 warmup，再重置一次 episode RNG；正式周期不调用会重新设 seed 的 bootstrap。
同步 warmup 的 socket 超时默认 900 秒，可用
`--synchronous-warmup-timeout-seconds` 或 plan 中同名下划线字段调整；只覆盖冷启动
JIT/预热调用，结束或异常时恢复原客户端及当前 socket 的超时。reset 与正式推理仍用
原来的 120 秒设置，预热耗时不计入 scored control。
可选 `gpu_uuids` 必须提供 action/vlm/simulation 的完整 GPU UUID，准入先核对实际
`nvidia-smi` index/UUID，保留 `gpu-admission.json` 与 hardware 中的 verified mapping。

每轮 `cycle_phase` 保存真实 inference/execution 起止、物理步范围与暂停状态，
`synchronous_cycle` 保存原生解码动作、源图像 hash、VLM/DiT 分段时间、forward/graph
计数和丢弃后缀。兼容现有 feature→request→executed-slot 的来源审计；同步的 S1/S2
区间标为 `server_action_stage` / `server_vlm_stage`，不是两条独立 RPC，也不是 GPU
kernel 时间。`inference_wall_seconds` 是单次 VLM+Flow4 RPC 的完整客户端 wall；
`action_stage_seconds` 和 `vlm_stage_seconds` 是服务端 stream 完成边界。
完整同步调用使用 `synchronous_rpc_started_s` / `synchronous_rpc_completed_s`，
不与单独 action RPC 的提交边界混用。导入器兼容第一批记录中的旧字段名称，但保留
原文件并在规范化输出中明确完整同步 RPC 的范围；显示的各阶段事件必须逐一匹配
同周期 audit、source IDs 和执行动作。

```bash
python -m coexecution.replay build \
  --deployment .local/synchronous-examples/deployment \
  --default-execution-coupling synchronous --default-execution-clock paced \
  --output .local/synchronous-replay
python simulator/serve.py --replay .local/synchronous-replay/catalog.json
```

回放按 execution coupling 和 clock 筛选匹配运行，显示 `Synchronous`、真实 Phase
轨道以及 `Inference / simulation paused` / `Executing` 状态。不同精度共用 wall
游标，各自 physics clock 仅在实际 `env.step` 完成后前进。视频只使用记录的 wall→frame
映射，推理期间保持已有帧，不伪造运动；VLM→DiT→已执行动作仍可追溯。
相机拷贝与 trace 收集属于 instrumented debug 成本，视频编码在控制结束后执行。
少量任务样例不代表完整 Spatial 套件成功率。


### One-step asynchronous policy/execution / 单步异步策略与动作执行

`--execution-coupling one_step_async --execution-clock ready
--execution-prefix-steps 1` 是独立协议。它保留一个普通 Flow action server，每次请求
在同一服务内完整运行真实 VLM 和 Flow4。原有 `asynchronous` 仍指独立 VLM/DiT
缓存控制器，`synchronous` 仍按推理、执行顺序运行；三者不能混称。

先用观测0完成一次计入 scored control 的 bootstrap 请求，得到动作0。执行动作 k
之前，主线程复制当前观测 k，将它交给专属模型线程计算动作 k+1；主线程同时执行
当前结果的**第一个动作**。该原子步结束后，若下一结果未完成，控制器明确等待，
不补执行旧计划后缀，也不使用 fallback。bootstrap 的源/目标为0/0，随后为0/1、
1/2等：图像和状态一起滞后一个**物理步**，没有未来状态预测或补偿算法。
原生解码长度保持 checkpoint 配置，未采纳的动作后缀计入丢弃数量。

每个 `env.step` 仍推进0.05秒仿真时间，但 `ready` 不为它额外等待50ms。
因此重叠预算来自实际的 `env.step` wall 区间，不等同于50ms；后端渲染和模型同卡
争用也包含在实测值中。`observation_lag_ticks` 与 capture-to-apply wall AoI 分别
报告，一个旧物理步不等于额外50ms wall AoI。此实现受
[The Speedup Paradox §3.1](https://arxiv.org/pdf/2606.28529v3) 启发，但使用真实并发
RPC/仿真执行；论文中的历史观测抽象和理想周期公式不能替代此处的测量。

```bash
python scripts/evaluate_libero_protocol.py \
  --protocol deployment --variant flow --port 5570 \
  --execution-coupling one_step_async --execution-clock ready \
  --execution-prefix-steps 1 --task-ids 0 3 --episodes-per-task 1 \
  --max-steps 300 --record-replay --replay-max-frames 301 \
  --output .local/one-step-async/deployment
```

`--execution-prefix-steps` 是 `--synchronous-chunk-steps` 的兼容别名。experiment plan
支持同名下划线字段，逐 condition 覆盖全局；同层别名冲突会报错。该模式只支持
`ready`、prefix1、deployment Flow 和单个普通 action endpoint；paced、πR²、paired/
resident 特征传输、服务等待下限、视觉限频与 fixed-delay 均明确拒绝。

专属后台线程创建、使用并销毁自己的 PolicyClient/ZeroMQ context；主线程独占
MuJoCo 观测和步进，提交的是独立只读数组。每任务同 endpoint 完整 warmup 后只
重置一次 RNG；warmup socket 超时默认900秒，正式请求恢复120秒的 send/receive
各自上限。等待 Future 的上限覆盖两次 socket 上限和30秒余量，关闭线程也有上限。
超时、模型或仿真异常保留不完整 trace 并令实验失败，不会被当作成功 episode。

`request_id`（兼容 `request_tick`）标识唯一策略请求，`target_tick` 是它可能执行的
目标步；`source_tick` 独立标识源观测步。bootstrap request0与预取 request1 都可源于
观测0，VLM feature_sequence 使用唯一 request_id，不能用 source_tick 合并它们。
`policy_request_submitted`、`policy_rpc_started/completed` 分别记录排队提交和模型线程
实际 RPC 边界；VLM/DiT 的 server stage 时间另存，均不是 kernel-only 时间。
`cycle_phase` 包含 inference、execution、wait、drain；异步 inference 不标为全程
暂停仿真。`one_step_async_request` 保存原生计划和采纳状态，
`one_step_async_execution` 与 `control_tick` 才证明实际动作执行。

首次成功、环境终止或 max_steps 后不再执行动作。已在飞行的最终推测请求必须排空，
记录为 unused；已知最后一步达到 max_steps 时不额外预取。`control_end` 截止控制
窗口，`async_work_end` 截止全部已提交策略工作，末尾 drain 分开报告。`policy_calls`、
forward_counts 和完整 RPC 分布包含最终未用请求；`adopted_policy_calls`、实际执行
步数、丢弃后缀另列。queue report 的 `requests` 保留控制窗口的阶段暴露和 censoring，
`work_requests` 保留整个 bootstrap/推测/排空过程，不把 drain 隐去。
`measured_inference_execution_overlap_seconds` 是实际 RPC 与实际执行区间的交集；
服务端 VLM/DiT 交集另列，不能从“提交在步进之前”推断一定存在 GPU 重叠。


回放导入与 `feature_usage.audit_raw` 共用 `coexecution.async_evidence` 的全工作审计。
每个规范化 `work_requests` 项都要独立匹配原始 submission、图像 hash、RPC 和服务端
阶段、唯一 request/target/source、adopted 或 unused 处置及实际动作；最终不用的计算
同样验证。缺失、重复、错误时序或计数不一致会拒绝结果。控制窗口 `requests` 不变，
供前端使用的 `replay_requests` 保留完整 VLM/DiT 阶段，`work_end_s` 保留排空尾部。
前端把 Policy RPC 与 Execute/Wait/Drain 分成可同时可见的轨道；执行动作关联其精确的
DiT→VLM 来源，并同时显示物理步滞后和 wall AoI。末尾机器人保持真实末帧，模型轨道
继续展示未执行的最终结果。当前为调试记录，非无插桩性能测试或完整套件成功率。

比较完整策略延时时，异步 `inference_request_seconds` 从主线程提交至响应；其
`inference_wall_seconds` 仅为 worker 输入准备之后的 RPC。同步的完整调用起点在
输入准备之前。前端分别标注这些边界，不能直接把异步 RPC 与同步完整调用做加速比。

### Chunked asynchronous lookahead / 动作块长度与异步深度

`chunked_async` 将执行前缀长度 K 与提前推理步数 D 分开：
`--execution-prefix-steps K --async-overlap-steps D`，要求
`1 <= D <= K <= checkpoint 原生解码长度`。它复用单步协议的专属 RPC worker，
保持真实 VLM＋Flow4，每段依次执行原生动作 offset0…K−1。下一段目标 T 为
K、2K 等；在物理步 T−D 之前捕获当前图像和状态并提交请求，执行完当前段承诺
的所有动作后才采用下一结果。结果提前就绪也不会提前换段；未就绪时明确等待。
D=0 使用现有 `synchronous`，而非退化的异步模式。旧 `one_step_async` 保留其
v1 接口和记录，等价于 K=1、D=1；旧 `asynchronous` 缓存控制器不变。

```bash
python scripts/evaluate_libero_protocol.py \
  --protocol deployment --variant flow --port 5570 \
  --execution-coupling chunked_async --execution-clock ready \
  --execution-prefix-steps 8 --async-overlap-steps 4 \
  --task-ids 0 3 --episodes-per-task 1 --max-steps 300 \
  --record-replay --replay-max-frames 301 \
  --output .local/chunked-async/deployment
```

experiment plan 支持 `async_overlap_steps`，condition 中的值覆盖全局。
启动实验前，runner 读取每个 checkpoint 的 `processor_config.json` 中
`processor_kwargs.modality_configs.libero_sim.action.delta_indices`，按其实际长度
检查所有 condition 的有效 K；它不会把模型的 H40 当作原生解码长度，也不会加载模型。
root metadata 优先，缺失时才按上游 loader 回退到 `processor/` 子目录；错误或缺失的
metadata 在 GPU admission 和子进程启动之前报错。服务加载后的 identity 检查仍保留。
对照矩阵可采用 K1/D0,1；K2/D0,1,2；K4/D0,1,2,4；K8/D0,1,4,8。
D0 condition 设置 `execution_coupling=synchronous`；D>0 设置 `chunked_async`。
该版本仅支持 deployment Flow、ready、一个普通 action endpoint。输入快照独立
只读，MuJoCo 仅由主线程操作；普通段内步骤复用上一 env.step 的观测，只有 bootstrap
和指定的预取边界才强制刷新摄像头。没有未来状态预测或异步补偿算法。

bootstrap 的 offset j 来自物理观测0，滞后 j 步；后续每段 offset j 滞后 D+j
物理步。wall AoI 由真实 capture-to-apply 测量，与仿真0.05秒/步分别报告。
成功或终止发生在任何 offset 都立即停止。max_steps 截断最后一段的接纳长度，
例如 K8、上限300时 target296 只接纳4步。接纳但因提前成功未执行的步骤、未接纳
的原生后缀，以及最终推测请求完全未采用的动作分别计数，不把“接纳”写成“执行”。

v2 `chunked_async_request` 在段完成后保存最终计数，`t_s=finalized_s`；实际接纳
时间使用 `adopted_s`，响应接收使用 `received_s`。`request_id=request_tick=target_tick`
按 K 递增，不能假设连续；每个 `chunked_async_execution` 与 execution phase 使用
唯一 `execution_id=tick`，并记录 `executed_offset`。多个物理步关联同一个请求。
尾部 unused 请求的 target 可能大于实际控制结束步；drain 的 physics tick 始终使用
实际 `control_steps`。`chunked_async_boundary` 记录结果是否在边界就绪，完整工作
审计验证对应 wait，以及所有原生 offset、执行动作、源观测和终止后未用计算。

主要比较成功/失败、实际 `actual_control_hz`、完整控制窗口除以动作步数的
`amortized_wall_seconds_per_action`、段边界等待与 gap 的 P95、AoI 和实测重叠。
`boundary_gap_seconds` 是上一段最后一步 env.step 完成到下一段第一步开始的间隔；
`adjacent_action_gap_seconds` 是相邻动作开始时刻的间隔。K 较大时段内短间隔会压低
后一指标的 P50，不能仅凭它声称整体吞吐提升。两种间隔均保留 count 和 sum，原始
逐动作/逐阶段记录用于汇总分位数，不能把每 episode 的分位数再平均。
