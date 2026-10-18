# 量化推理与机器人闭环调试前端

这个前端把**模型计算、特征依赖和机器人响应放在同一条可追溯时间线上**。先在固定真实输入下比较 BF16/W8A8/W4A4 的推理，再查看各精度在 LIBERO 中实际产生的动作、视频和信息年龄，定位模型加速如何影响控制结果。

| 入口           | 适合回答的问题                                      | 数据来源                                   |
| -------------- | --------------------------------------------------- | ------------------------------------------ |
| 独立模型回放   | VLM、DiT 和在线量化各花多少时间？动作输出如何变化？ | 显式加载的模型测量与 CUDA profile          |
| 机器人闭环回放 | 结果何时就绪、何时采用、执行了哪一条动作？          | 每个精度自己的控制 trace、状态和双相机视频 |
| 机制仿真       | 服务时间、共享资源和调度假设如何改变队列与 AoI？    | 可编辑的离散事件模型                       |

前两项读取实验记录；机制页用于解释调度，不运行神经网络或物理仿真。数据生成与接口合同见 [replay 模块](../coexecution/replay/README.md)，控制协议见 [时序协议](../docs/timing-protocols.md)。

<a id="quantization-and-robot-debugger"></a>

## 打开实测回放

在仓库根目录启动已有 catalog：

```bash
python simulator/serve.py --port 8765 --replay /path/to/catalog.json
```

浏览器打开 `http://127.0.0.1:8765/`，或直接打开 `/simulator/replay.html`。`--replay` 也接受包含 `catalog.json` 的目录。有机器人记录时默认显示机器人视频；`?view=model` 打开固定输入模型页，`?view=deployment` 打开闭环页。模型页的相机图是该请求的静态输入，机器人页播放各精度自己的视频。

1. 选择模型/机器人页，再选择实验、观测或任务与回合。
2. 在机器人页统一选择 cadence、coupling、控制器策略及已有 K/D 组合；精度卡片共用这些条件，每张卡片再选重复实验。
3. 播放、暂停、拖动游标，或用前后事件按钮定位一次请求和执行。模型时间单位为毫秒，机器人时间单位为秒。
4. 点击 VLM/S2、DiT/S1 或执行块，查看其输入来源、时间戳、动作与 AoI；展开内核区查看实际 profile。
5. 使用“运行来源”和导出功能保存当前证据选择。切换条件时所有卡片一同更新，过期 HTTP/媒体响应被丢弃。

页面只提供 catalog 中已有的条件。未知身份、缺视频、失败和未完成回合保留状态及原因；加载新条件期间禁用次级控件。跨精度使用相同任务与初始化可以比较闭环响应，但轨迹分叉后各自的观测不同。

## 阅读模型时间线

| 测量边界               | 页面含义                               |
| ---------------------- | -------------------------------------- |
| `policy`               | 原始输入准备到解码动作                 |
| `preparation_transfer` | 输入准备与传输，不显示神经计算块       |
| `prepared_model`       | 已驻留输入上的模型计算                 |
| `preparation_model`    | profile 的准备与模型调用，解码在边界外 |

Clean P50/P95 使用同一边界中已确认无 instrumentation 的样本。精确动作在该请求完成后显示；Profile 调用保留自己的 seed 和输出。跨精度数值比较要求观测哈希、seed、scope、共享配置和已审计时钟组同时匹配；单条未配对记录仍可查看。

Profile 的 VLM/DiT 块覆盖实际归因 CUDA 活动的首尾区间，并保留其中空隙。内核表提供 stage、category、stream 和完整 kernel 名称。Host submission、GPU work sum、GPU active union、device span 与 wall latency 各有边界；Clean 请求缺少组件时间戳时，其组件轨道显示 unavailable。聚合耗时不能还原一条内核时间线。

## 阅读机器人响应与依赖

共享游标使用回合开始后的**控制墙钟时间**。视频通过逐帧 `wall_s` / `media_time_s` 映射跟随游标，状态在记录的计算完成时刻出现，命令在实际执行时出现。回合结束后保留最后一帧；推理末尾 drain 可以继续显示，机器人和物理时钟保持停止。

VLM 为 S2，DiT 为 S1。`S1[n].k` 表示消费第 n 份 S2 特征的第 k 次 S1 请求。点击执行块可沿 `executed_slot` 找到精确 DiT→VLM 来源；点击 S1 可查看 feature ID、图像/状态 tick、承诺前缀、滚动缓冲、shift 与恢复标记。来源高亮直接画在时间线上。bootstrap、fallback 和缺失来源按各自状态显示。

| 轨道 / 指标                                         | 应如何解释                                                                           |
| --------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Cached-asynchronous 的 S2/S1 块                     | 实测 RPC 起止；服务端未记录分段时不拆成内核                                          |
| 同步推理的 VLM/DiT 块                               | 一次完整 RPC 内的服务端阶段                                                          |
| `Action execution / gap`                            | `applied_s` 到 `next_observation_computed_s` 的实测 `env.step`；每步物理推进时间另列 |
| `Between steps`                                     | 相邻执行间隔，包含节拍等待及控制开销                                                 |
| `Inference / Execution` 或 `Execute / Wait / Drain` | 记录的推理、动作、等待和收尾阶段；保持实际重叠关系                                   |
| cache / image / state AoI                           | 该请求或已执行动作真正使用的信息年龄；未知来源为空                                   |
| XY / state / command 曲线                           | 位置为米，旋转为弧度轴角向量，夹爪状态为两关节位置；归一化控制命令单独显示           |

模型身份 `policy_variant` 与控制器策略 `deployment_scheduling_policy` 分开。πR² 使用 S2 特征缓存和 S1 滚动去噪；正常请求 DiT forward 1 次，bootstrap 的 VLM 1 / DiT 4 和整缓冲恢复 DiT 5 分开计数。完整 VLM+Flow4 属于另一模型路径。回合 `policy_evidence` 保留已记录请求和末尾未采用工作的计数；预热、独占校准另列。检查点配置只有与 worker SHA-256 一致才显示，训练步数来自保留的训练状态。

就绪、控制器读取、安装、执行、过期和保护是不同状态。结果安装零条动作不会显示为已采用；已生成动作也可能等待合法执行槽。控制器选择器把 conservative/responsive 绑定到所有精度卡片，详细采用规则见 [时序协议](../docs/timing-protocols.md)。

## 选择推理与执行协议

| coupling         | 页面展示                                             | K/D                                |
| ---------------- | ---------------------------------------------------- | ---------------------------------- |
| `asynchronous`   | 独立 S2 缓存与 S1 请求；推理可与动作重叠             | 使用记录的延迟与槽，不显示固定 K/D |
| `synchronous`    | 一次完整 VLM+Flow4 后执行动作前缀；推理时物理暂停    | D=0，K 为最大执行前缀              |
| `one_step_async` | 完整下一次推理与当前一个物理步重叠，结果未就绪时等待 | K=1 / D=1 的独立协议               |
| `chunked_async`  | 当前前缀剩 D 步时提交下一次完整推理                  | 页面从实测记录提供 K/D 组合        |

K 是每个采用结果最多执行的原生动作数；D 是下次采集观测并提交推理时，当前前缀剩余的动作数。它们不改变模型原生解码长度。卡片同时显示原生 horizon、接纳前缀和实际执行前缀；提前成功或达到步数上限可缩短最后一段。

当前 LIBERO 评估器固定 `control_hz=20`，每个物理步推进 0.05 仿真秒。`paced` 以 50 ms 为墙钟目标节拍；同步模式的节拍只约束执行阶段，全回合仍包含推理暂停。`ready` 在原子仿真步边界尽快采用可用结果，不额外等待墙钟网格，因此实际墙钟执行率可以变化。页面分别显示执行 Hz、请求率、计划更新率、就绪至首次执行等待和仿真/墙钟倍率；Ready 的固定频率 deadline/lateness/validity 为 null。这里的就绪调度沿用 GR00T 权重。

分块实验优先查看 `control wall / actual actions`、有效执行 Hz、前缀边界 gap P95 和显式等待。K 较大时，相邻步 gap P50 常由块内动作主导，不能代表整回合吞吐。各 K/D、成功/失败回合和提前终止保留各自记录。

构建矩阵回放并指定默认组合：

```bash
python -m coexecution.replay build \
  --deployment /path/to/matrix/evaluation \
  --default-execution-coupling chunked_async --default-execution-clock ready \
  --default-execution-prefix-steps 8 --default-async-overlap-steps 1 \
  --output .local/matrix-replay
python simulator/serve.py --replay .local/matrix-replay/catalog.json
```

URL 参数 `execution_coupling`、`execution_clock`、`execution_prefix_steps`、`async_overlap_steps` 与 `deployment_scheduling_policy` 保留选择。K/D 切换只加载匹配的 cohort；catalog 刷新保留仍有效的组合，SSE 更新也必须保持该模型、控制器和调度身份。

## 在线更新与本地服务

已注册的流先加载 snapshot，再消费有序 SSE；重复事件去重，断档重新取 snapshot，终止事件关闭连接。实时更新保留游标、媒体和选择，“跟随最新事件”需显式启用。Catalog 刷新可发现新记录，不启动实验。事件合同和 `record-model` 用法见 [在线记录](../coexecution/replay/README.md#explicit-recording-and-incremental-delivery)。

预览绑定 loopback，只开放所选 catalog、run、media 和注册 stream。路径、ID、MIME 与资源清单经过校验，媒体支持 HEAD 和有界 Range；目录列表关闭。HTML/API 使用 no-store，媒体允许私有缓存一小时。完成的实验资产保持不变，在线 journal/snapshot/media manifest 按 recorder 合同更新。

## 机制仿真与实测报告

从 `/simulator/` 打开机制页；没有 catalog 时也可直接启动：

```bash
python simulator/serve.py --port 8765
# 可选：连接已有实验的自包含报告。
python -m coexecution.queue_report --input /path/to/experiment --output /path/to/report --no-figures
python simulator/serve.py --port 8765 --report /path/to/report/index.html
```

实测报告保留频率、功率、重复条件、S1/S2 GPU 映射和原始有效性。点击条件可将其**独占 RPC 均值**载入机制页，作为服务时间起点；主机占比和争用系数需单独拟合。输入完整 solo 服务时间、控制周期和相机频率，再选择 GPU 拓扑及可选共享主机服务。

| 资源模式      | 模型行为                                                  |
| ------------- | --------------------------------------------------------- |
| `independent` | GPU 工作可并行且互不减速；主机策略独立配置                |
| `shared`      | 只在 GPU 重叠期间按各自 slowdown 降速，另一任务离开后恢复 |
| `serial`      | GPU 非抢占串行；同刻任务边界由动作请求优先                |

`gpuShareA/V` 将 solo 请求拆成主机前缀与 GPU 工作，默认 1。精度标签本身不改变服务时间；显式 GPU 加速倍率、主机额外开销和 VLM 接纳率才影响模型。等待相机槽始终保留最新帧，0 接纳率上限表示无限制。

网页默认 `paper` 机制参考，实测导入使用 `libero`；这里的 `libero` 表示保守调度模型，真实 responsive 控制器以实测回放为准。两种机制模式可随时切换：

| 行为     | `paper`                                              | `libero`                                             |
| -------- | ---------------------------------------------------- | ---------------------------------------------------- |
| 请求启动 | 上次完成后连续发起                                   | 控制 tick 接纳，至多一个未消费请求                   |
| d        | 显式初值；滚动响应均值/周期，round-to-even，可升可降 | 12 次独占预热弃前 2，P95+margin 向上取整；迟到后上调 |
| 结果采用 | 当前干净段耗尽后换段                                 | 发布到未过期、未承诺的绝对槽                         |
| 动作耗尽 | 重复最后一个干净动作，保留原来源                     | 零位移并保持夹爪，无模型来源的 fallback              |

`paper` 参考[论文滚动 d](https://arxiv.org/html/2607.26055v1#A3) 与[作者固定版本超时处理](https://github.com/pi-r2-flow/pi-r2-flow/blob/3af52ca400a6ec7d141416879aa531a6f62f697a/deployment/apps/run_policy.py#L896-L910)，使用冻结的每请求干净段宽度、40 格示意缓冲和可配置最大 d；`libero` 模型的 d 上限至多 5。它们是调度模型，不包含神经 bootstrap/recovery 计算，也不预测任务成功率。页面的加速、接纳反馈和微小超时预设均为合成机制示例。

`queueDefaults`、`simulatePolicyQueues(config)` 与 `policyQueueSnapshot(result, timeMs)` 是可复用接口；API 默认保持 `libero`，网页默认单独设置为 `paper`。JSON 导出保留配置、校准、全部事件、特征依赖、逐槽采用/执行及当前快照。

## 模块与桌面验证

| 职责                 | 文件                                                        |
| -------------------- | ----------------------------------------------------------- |
| 回放证据、选择与展示 | `replay-model.js`、`replay-view.js`                         |
| 媒体、时间线、SSE    | `replay-media.js`、`replay-timeline.js`、`replay-stream.js` |
| 实测来源解析         | `measured-model.js`、`measured.js`                          |
| 机制引擎与场景       | `model.js`、`scenarios.js`                                  |
| 机制交互与公共样式   | `app.js`、`lab.js`、`shared.css`                            |

这些模块复用同一套来源和显示规则；扩展实验时增加参数与数据合同。前端无外部框架依赖，服务读取证据时不导入 Torch 或启动仿真。

```bash
node --test simulator/*.test.cjs
python -m pytest -q simulator/test_serve.py
# 可选桌面浏览器依赖与检查；读取已有数据，不启动模型。
npm ci --prefix tools/ui-check
tools/ui-check/node_modules/.bin/playwright install chromium
node simulator/replay-browser-check.cjs --catalog=/path/to/catalog.json --no-screenshots
node simulator/replay-stream-browser-check.cjs
```

实测浏览器检查使用含三精度模型/机器人、双相机及保留失败例子的 catalog，覆盖真实视频像素变化、播放/暂停/反向 seek、依赖选择、条件切换和过期响应。SSE 检查使用桌面协议 fixture。它们验证界面合同，任务质量由 CUDA 与 LIBERO 实验评价。

公共字体优先 Source Han Sans CN VF/SC，后备 Noto Sans CJK SC；本地 WOFF2 位于忽略的 `simulator/fonts/`。来源和哈希见 [sources.lock.json](../sources.lock.json)，许可见 [Source Han Sans 许可证](../licenses/SOURCE_HAN_SANS_LICENSE)。离线报告内联已有字体和许可，缺省时使用本机回退。原始记录、报告、截图与字体下载留在忽略目录。
