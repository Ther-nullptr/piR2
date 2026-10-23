# 实测模型与机器人回放合同

`coexecution.replay` 把模型测量、控制 trace 和机器人视频归一成同一个可追溯目录，供多精度调试前端与新的流式平台复用。它保留**输入 → VLM 特征 → DiT 请求 → 动作槽 → 执行**的真实来源，同时分开模型计算、请求响应和物理推进时间。

| 入口                                        | 职责                                                   |
| ------------------------------------------- | ------------------------------------------------------ |
| `python -m coexecution.replay build`        | 读取显式指定的既有证据，生成 catalog、run 和媒体清单   |
| `python -m coexecution.replay refresh-index` | 从已有 run JSON 刷新动态回放选择器索引                 |
| `python -m coexecution.replay record-model` | 显式启动 CUDA 模型测量，并增量发布 snapshot / SSE 事件 |
| `python simulator/serve.py --replay …`      | 在 loopback 提供目录、媒体和注册事件流                 |
| [桌面调试前端](../../simulator/README.md)   | 比较模型输出、依赖时间线和各精度实际机器人响应         |

Builder 与浏览器服务不启动模型或仿真。目录读取使用标准库；模型导入器按需读取 NumPy 动作档案，再输出 JSON，NPZ 不进入浏览器。模型和控制启动配方分别见 [性能实验](../../docs/coexecution.md) 与 [LIBERO](../../docs/libero.md)。

<a id="capture-and-build"></a>

## 采集与构建

### 固定输入模型

给 `python -m coexecution.libero.measurement` 加 `--replay-assets`，在计时前保存真实原始相机 PNG 和状态。`--measurement-boundary model-only` 测已驻留输入上的完整模型；Clean 请求保留精确动作。`--profile-rounds 1` 额外执行带 request/VLM/action 注解的 profile 调用，保留其 seed、相关联的 Chrome CUDA 时间和自身输出状态。

### 机器人响应

部署评估器使用 `--record-replay --replay-max-frames 721`；实验计划对应字段为：

```json
{ "record_replay": true, "replay_max_frames": 721, "task_ids": [0] }
```

省略 `task_ids` 评估完整 Spatial 十任务。Replay 为 deployment 专用录制，记录每回合自己的双相机与动作前后机器人状态。`--no-video` 控制旧 MP4，不关闭显式 replay；启用 replay 时避免重复旧视频录制。

有界 recorder 在控制阶段复制完整相机组及状态，控制结束后才编码 WebM/VP9 并写 JSON。数组复制属于 instrumentation；帧数/字节上限、内存分配与编码失败写入 metadata。录制失败保留此前完整帧，控制可以继续。初始帧来自真实初始观测；后续帧对齐物理步后的 `next_observation_computed_s`。

### 生成目录

```bash
python -m coexecution.replay build \
  --model-run /path/to/model-only \
  --deployment /path/to/evaluation \
  --output /path/to/new-replay-catalog \
  --clock-cohort audited-device-and-clock-session
python simulator/serve.py --port 8765 --replay /path/to/new-replay-catalog/catalog.json
```

两个输入参数均可重复。Deployment 输入可以是 deployment 目录、其条件目录或含多个条件的实验根目录。输出目录必须新建或为空；输入只读。`--clock-cohort` 是操作者依据硬件/遥测审计作出的声明，未知时省略，此时不给出跨精度同输入比较键。

可用 `--default-execution-clock paced|ready`、`--default-execution-coupling`、`--default-execution-prefix-steps` 与 `--default-async-overlap-steps` 选择默认机器人组合；该组合必须出现在记录中。旧 catalog 默认 paced。前端选择器只切换记录，不改变旧轨迹的时间或调度。

### 移动物体协议：LIBERO-Dynamic

```bash
python -m coexecution.replay build \
  --dynamic-evaluation /path/to/dynamic-evaluation \
  --output /path/to/new-dynamic-catalog
python simulator/serve.py --port 8765 --replay /path/to/new-dynamic-catalog/catalog.json
```

打开 `/simulator/replay.html?view=dynamic`。`--dynamic-evaluation` 可重复，也可与原有模型或部署输入一起构建。每个目录包含评估器的 `evaluation.json`、`manifest.jsonl`、policy audit、物理 trace 和 MP4。Builder 读取已有文件，沿用媒体白名单，不启动模型。

动态目录在每个描述符中保留 `episodes` 计数，并增加 `protocol` 与小型
`episode_summaries`。浏览器用索引显示条件和配置，只读取当前所选运行的完整 JSON，
最多保留四列；切换后释放未选运行与播放器。索引中的
`request_duration_bounds_ms` 分别记录全部请求、S1、S2 的实际时长上下界；
该组没有请求或有缺失时长时为 `null`。原始请求、视频、wall 阶段和 Run JSON 链接保留。

已有目录可直接刷新索引，无需重新导入实验或编码媒体：

```bash
python -m coexecution.replay refresh-index --catalog /path/to/dynamic-catalog/catalog.json
```

命令逐个校验已列出的 run 路径与身份，原子替换目录索引，保留其它目录字段和
媒体/流登记；完整 run 与媒体文件不改写。旧目录没有摘要时，浏览器逐个读取运行来
提取小型索引，因此旧目录仍可阅读，但建议先刷新索引以缩短首次加载。

Dynamic run 使用 `kind=dynamic`。主轴 `rollout_sim_time_s` 从首次策略观测开始，包含机器人保持、目标继续运动的推理等待。每个 hold 与动作块通过 `chunk_id → policy_request_sequence` 关联本回合的 scored request；warmup 请求不画进主时间轴。选择请求后，独立 wall 面板展示实际 Fresh VLM → Flow DiT 的 stream-complete 起止时间、RPC 和原始测量边界，物理 τ 与实测 wall 延迟分别呈现。

导入 MP4 需要本机 `ffmpeg` / `ffprobe`。Builder 生成浏览器可解码的无损 VP9/WebM 副本；保留原 MP4、源 SHA-256、编码参数与逐帧时间戳，不插帧或重采样。WebM 容器按毫秒舍入 PTS，因此保留 audit 的 `source_media_time_s`，并将 `media_time_s` 更新为 ffprobe 读出的实际 PTS。双相机帧数和 PTS 必须一致。

视频逐帧使用 `video_samples.sample_sim_time_s → media_time_s`，保留等待期间真实采集的运动帧。视频 fps 与控制 Hz 分别显示；`source_tick` 是执行动作计数，不含物理等待子步。反事实固定 τ 标记为 counterfactual，实测校准保留样本数与应用 τ；每回合模型热身和延迟校准热身单独显示。选择器按套件、任务、轨迹、速度、seed 和初始状态配对；每个精度的配置独立选择，未录制精度不生成结果。

导入器验证媒体位于评估根目录、帧时间单调且归属对应 chunk/阶段/动作步、动作来自对应请求前缀、trace 与物理事件一致，并保留源路径/SHA-256。缺失媒体、计时与失败回合保持不可用或未知。浏览器使用同一个机器人播放器和传输控件。

<a id="explicit-recording-and-incremental-delivery"></a>

## 显式记录与增量交付

准备目标 CUDA 环境后，使用现有测量引擎记录单一 runtime mode：

```bash
python -m coexecution.replay record-model \
  --catalog-output /path/to/live-catalog \
  --checkpoint /path/to/libero_spatial --dataset /path/to/groot-libero-spatial \
  --upstream /path/to/Isaac-GR00T --output /path/to/new-measurement \
  --precisions bf16 --modes kv_reuse --action-cuda-graph --operator-fusion \
  --measurement-boundary model-only --observations 3 --rounds 3 \
  --warmup 2 --profile-rounds 1 --replay-assets
```

命令沿用 measurement 的模型选项及数值/RNG/Graph 检查，并在计时外采集 runtime receipt。普通 measurement 使用 `--runtime-receipt` 才额外记录源码与已加载二进制身份。每次录制要求新 measurement 输出目录和一个 mode；不同 operator profile 使用独立进程。Catalog 和每精度 journal 在推理前注册，输入与已完成输出随后追加，完成时用 canonical run 替换临时投影。Receipt 记录实际运行身份；验证背书还要求完全匹配的身份与保留的 GPU 验证证据。

实时 host-stage 事件描述主机提交，相关 CUDA profile 导入后才具备真实 kernel 完成时间。Clean 调用保持自己的 wall/output 证据。机器人 recorder 已提供帧组与编码完成 sink hook；标准机器人视频仍在控制结束后编码，模型 SSE 与实时视频编码是不同能力。

### HTTP 与 SSE

| 路由                              | 返回内容                                            |
| --------------------------------- | --------------------------------------------------- |
| `/api/replay/catalog`             | 当前目录与已注册 streams                            |
| `/api/replay/run/<id>`            | v1 normalized run                                   |
| `/api/replay/media/<id>`          | 清单内媒体，支持 HEAD / 有界 Range                  |
| `/api/replay/snapshot/<id>`       | run、`last_sequence`、stream state 和 clocks        |
| `/api/replay/events/<id>?after=N` | 已注册流的 SSE；`Last-Event-ID` 优先于 query cursor |

Catalog 的可选 `streams` 项登记 `run_id`、`snapshot_path`、`journal_path`、`max_event_bytes`，路径为经过 allowlist 校验的相对资源。客户端先取 snapshot，再从其 `last_sequence` 接续；SSE `replay` 按 sequence 应用，重复去重，`resnapshot` 表示需要重新取权威快照，`end` 表示终止。心跳不改变证据时间。

| 合同                      | 内容                                                                                                |
| ------------------------- | --------------------------------------------------------------------------------------------------- |
| `pir2.replay.snapshot` v1 | normalized v1 run、最后 sequence、stream availability/state、显式 clock registry                    |
| `pir2.replay.event` v1    | run、递增 sequence、发生时间的 clock/unit/domain、事件类型、来源 ID、可选 append/replace projection |
| 时间关系                  | 入队 sequence 与发生时间分别保留；浏览器接收时间不用于推理耗时或 AoI                                |
| 有界交付                  | 队列按事件数和字节限制；默认单事件 1 MiB，显式注册上限 8 MiB，tail batch 上限 16 MiB                |
| 完成与丢失                | `flush()` 为计时外最终屏障；loss、recorder error 和 terminal 状态可见；不交付末尾未完整 JSONL       |

路径、ID、MIME、NOFOLLOW 和媒体清单在刷新时继续校验。绝对路径、目录穿越、外部 symlink 及未列资源拒绝访问。完成的 run/资产保持不可变；注册的 snapshot/journal/media manifest 按 recorder 合同更新。HTML/API 使用 no-store，媒体允许私有缓存一小时。

<a id="version-1-contract"></a>

## v1 数据结构

| 文件             | schema 与核心字段                                                                                                          |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `catalog.json`   | `pir2.replay.catalog`，version 1；runs 的 id/kind/precision/mode/label/path、计数、warnings；media manifest 与可选 streams |
| `runs/<id>.json` | `pir2.replay.run`；来源路径与 SHA-256、配置、验证状态及 normalized model/deployment 数据                                   |
| `media.json`     | `pir2.replay.media`；opaque asset ID → 相对路径、MIME、原来源哈希                                                          |

当前 `PRECISIONS` 为 `bf16`、`w8a8`、`w4a4`。FP8/FP4 的接入范围见 [平台精度支持](../../docs/platform-handoff.md#precision-support)。新前端应通过 manifest ID 访问媒体，保留资产声明的输入根目录约束。

### 模型输入与输出

Model run 含 `observations`、`clean_requests`、`profile_requests`、`retained_outputs`、`kernels`、`profile_summary` 与 `validation`。Clean 请求保留 wall latency、seed、observation identity 和动作 `{shape, values}`；动作坐标表示原生解码预测步，不表示已执行控制 tick。

`difference_from_bf16` 要求观测哈希、seed、scope、已知共享运行配置和已声明 clock cohort 同时匹配。它报告数值差异，任务质量由闭环实验衡量。Clean 与 Profile 各有自己的 ID、seed 和输出，Profile 未保存动作时保持 unavailable。

Profile `timeline` 的 `start_ms` / `duration_ms` 相对该 CPU request annotation，另有 stage/category/kernel_id/device/stream；`kernel_id` 索引 `kernels`。只有真实 CUDA launch correlation 才连接请求与内核，CPU/GPU 两份 Chrome annotation 以 host annotation 定义请求范围。未知活动保留未归类或未关联状态。

| 量                     | 定义与用途                                            |
| ---------------------- | ----------------------------------------------------- |
| Clean wall             | 同边界、确认未 instrumentation 的请求延迟，供 P50/P95 |
| Instrumented host wall | profile 或其他插桩调用的实际主机范围                  |
| GPU work               | 各 CUDA 活动时长之和；并发时可以超过墙钟              |
| GPU active union       | CUDA 时间区间并集；除以窗口可表示时间覆盖率           |
| Device span            | 首个至末个设备活动跨度，包含空隙                      |

上述量边界重叠，不作可加分解；GPU 时间覆盖率也不等于 SM occupancy。聚合 profile 只有汇总时保留在 `profile_summary`，不生成逐请求内核轨道。

### 机器人状态与时间映射

Deployment run 含 `episodes`，各回合保留原 `metadata`、既有 queue reporter 归一证据、实际 `ticks` 和 `robot` 录制信息。帧包含 `wall_s`、`relative_s`、`media_time_s`、index、phase、source tick，每路相机对应自己的媒体 ID。编码视频为 20 fps，墙钟 seek 必须使用逐帧映射，不能用帧号/名义控制频率替代。

Tick、observation、request、feature、frame ID 保持因果归属。回合命名空间中的 `observation-{tick}`、`S1-{request_tick}`、`S2-{source_tick}`、`feature-{feature_sequence}` 和 `tick-{tick}` 连到原始记录。轨迹可配对 task/初始化/seed，但闭环分叉后不分配同输入模型比较键。

位置单位为米。上游字段名 roll/pitch/yaw 实际保存弧度轴角旋转向量；夹爪为物理关节位置。执行命令为归一化 LIBERO 控制命令，不能标为笛卡尔米位移。状态在计算完成时生效，命令在 `applied_s` 生效；物理步区间为 `applied_s` 到 `next_observation_computed_s`。

Wall、simulation 与 media 使用各自时钟；多设备扩展需保存经验证的 clock-domain 映射。当前 LIBERO 评估器固定 `control_hz=20`，物理步保持 0.05 s；`paced` 采用 50 ms 墙钟目标节拍，同步模式只在执行阶段 pacing。`ready` 在原子仿真步结束后接纳完成结果，实际墙钟执行率可变。Ready 的固定频率 deadline/lateness/validity 为 null，也不用于校准固定节拍机制页。

<a id="controller-policy-and-publication-evidence"></a>

### 控制器、模型与动作采用

`deployment_scheduling_policy` / `controller_identity` 描述控制器，`policy_variant` 描述模型。全局 conservative/responsive 选择器绑定精度卡片；SSE 替换继续核对控制器、模型与 K/D cohort。历史 trace 只有显式 `adaptive_nondecreasing` 或 fixed delay policy 可证明 conservative；其余保持 null。

Cached-worker 请求的 `publication` 来自 `request_completed` 与 `action_adopted` 原始计数。读取结果、安装、过期、保护和执行各自计数；安装数为零不会标成 adopted。执行严格使用 `control_tick.executed_slot.producer_request_tick` 且 producer kind 为 request。Bootstrap/fallback 不按时间邻近补出模型来源。

时间线分开 RPC ready 等待控制器、已入槽等待执行、部分/已执行、丢弃及最终未用结果。合法区间、承诺前缀和延迟决定保留可检查来源；between-tick 缓冲变化读取 `publication_snapshots`。历史计数或执行覆盖不足时保持 unknown。

### 同步与前缀异步的完整工作

`episode.phases` 保留实际 inference/execution/wait/drain 区间及 producer，决定时间线重叠。`one_step_async` 与 `chunked_async` 的 `work_requests` 通过共享完整工作校验：逐项核对原提交、采集/哈希、RPC/服务端阶段、结果去向和物理执行；缺失、重复或矛盾的终端工作拒绝导入。控制窗口统计与包含终端 drain 的 `replay_requests` 分开。

K/D 的请求 ID 是目标 tick 0、K、2K…，每个物理动作保留本身 tick 与 native offset。同一前缀的所有执行都链接同一个 DiT→VLM producer。`adopted_s` 表示采用；v2 的 `finalized_s` 在实际执行后记录，不作推理结束时间。请求结束可晚于 `controlled_end_s`，回放延伸至 `work_end_s`，机器人则保持终止状态。

Async RPC 在 worker 输入准备后开始，同步完整调用包含准备；比较完整策略延迟时使用 submission-to-response。模型原生 horizon 与实际执行前缀分别保留，提前成功或步数上限产生的短前缀保留实际长度。具体实验定义见 [时序协议](../../docs/timing-protocols.md)。

## 历史记录、失败与缺失证据

导入器检查有限值与时间顺序；未知值保留 null/unavailable，失败 gate 和未完成状态进入目录。任务 ID 为 0–9 的整数，episode ID 为非负整数。完整 `results.json` 优先；历史 sidecar 矛盾时披露差异并保留双方哈希。数值/RNG/Graph 失败可能发生在 results 写出前，已存 sidecar 仍以 failed/incomplete 导入。

`retained_outputs` 保存没有测量样本行的实际输出，其 wall、seed、comparison key 可为 null。非有限数组保留 shape 和 nonfinite count，数值显示 unavailable。旧档案缺 ID 时只从已保存的 observation SHA-256、列表索引以及 round/mode/observation/scope 生成标识，并记录 `identity_derivation`；缺 seed、设置或相机资产保持未知。

缺 `instrumented` 的历史行即使 `results.profiler=false` 也不进入 Clean P50/P95，因为 CUDA-event 插桩是另一条件；原始 wall/output 仍可查看。缺控制 trace 的回合保留任务身份和状态，但不因 metadata 中有 success 就播放成完整控制记录。

## 实现与验证入口

| 模块                                                                                             | 职责                                            |
| ------------------------------------------------------------------------------------------------ | ----------------------------------------------- |
| [catalog.py](catalog.py)、[schema.py](schema.py)                                                 | 目录组装、身份与支持范围                        |
| [model.py](model.py)、[deployment.py](deployment.py)、[async_deployment.py](async_deployment.py) | 模型、闭环、异步完整工作归一                    |
| [recording.py](recording.py)、[producer.py](producer.py)                                         | 有界机器人帧组与来源事件                        |
| [live.py](live.py)、[stream.py](stream.py)                                                       | 显式模型录制、journal、snapshot、投影与丢失处理 |
| [media.py](media.py)、[http.py](http.py)                                                         | 媒体身份、清单和只读交付                        |

公共合同检查按 [CONTRIBUTING](../../CONTRIBUTING.md)；桌面播放和 SSE 检查见 [前端说明](../../simulator/README.md)。模型验证与真实 LIBERO 结果分别由 CUDA 测量和闭环评估产生。模型、原始实验、视频、profile 和生成目录均留在忽略的本地路径。

### πR² 动态回放

`build --dynamic-evaluation` 同样接受 `evaluate-pir2` 生成的
`pir2.libero_dynamic.streaming_evaluation.v1` 目录。默认 Flow4 与 πR² 即使
精度相同也各占一列；每列可以切换配置，使用“添加同协议对照”同时查看另一个
已记录配置。场景、速度、轨迹、seed 和初态必须相同才会配对。

πR² 时间线有 S2、S1、动作三轨。点击 S1 高亮其实际采用的 S2 特征，点击动作
追溯实际生产请求；特征序号 0 属于 bootstrap，fallback 动作没有虚构的模型来源。
S1 面板展示 normal1/recovery5、有效动作窗口、接纳/过期/保护槽数及特征 AoI。
控制器事件、原始请求和物理位姿保留在 Run JSON 中。

主轴是实际物理仿真秒数，包含启动时物体继续运动的 bootstrap hold。
逻辑推理时间只减去记录中的 `logical_epoch_s`。native录制逐请求核对
`service_wall`完整调用起止时间与主轴区间，显示各自S2/S1实测范围，不对齐不同精度
的延迟。fixed录制保留反事实标签。两者的实际GPU/物理计算均分阶段串行，独立
逻辑通道不表示同卡墙钟并发。独立wall面板同时显示完整调用、其内的RPC和模型
阶段；这些嵌套区间不能相加。只有时长而没有绝对边界的server记录不绘制虚构位置。

视频仍使用同一媒体播放器和无损 VP9 派生文件，按解码后的真实 PTS 定位。新录制
支持逐相机 frame map；录制异常时只保留明确写入成功的帧，缺失的腕部帧不会用
另一相机的帧数补齐。原始视频和时间戳不修改。

πR² 规范化数据保留单份证据：控制器事件在 `episode.trace`，S1/S2 时间线通过
`request_id` 引用 `episode.requests`，主相机帧在 `episode.frames`，腕部等其它
映射在 `frames_by_camera`。前端同时兼容旧的内联 request 与重复主相机映射。
入队槽数表示发布时写入队列的数量，已执行动作只统计实际物理动作记录。
