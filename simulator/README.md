# Interactive πR² queue simulator

This browser tool explores πR² queueing, cached visual features and fixed-rate action execution. It accepts VLM and action-request inference times directly and compares independent resources, shared-GPU contention and serial GPU scheduling. It is an auxiliary explanation tool, not a GR00T-LIBERO evaluator, robot simulator or predictor of task success on new hardware.

Open [index.html](index.html) directly, or preview over loopback from the repository root:

```bash
python3 simulator/serve.py --port 8765
```

The preview serves this tool, selected stable documentation, and an optional explicitly configured report HTML. It does not expose its parent artifact directory, private configuration or the project root. Common typography, colors, cards, controls and chart tokens live in `shared.css`, using the measured replay as the visual reference. The report exporter inlines that same stylesheet for offline use; `style.css` and the measured template retain view-specific layout only. The numerical engine is in `model.js`; interface code is in `app.js`. Extend those modules rather than creating a new simulator per experiment.

## 实测与机制联动

前端有两个入口：**实测回放**展示显式加载的实验记录；**机制仿真**接受可编辑的服务时间和调度假设。连接自包含实测报告：

```bash
python -m coexecution.queue_report --input /path/to/experiment --output /path/to/report --no-figures
python simulator/serve.py --port 8765 --report /path/to/report/index.html
```

浏览器打开 `http://127.0.0.1:8765/simulator/`。预览只把该 HTML 映射到 `/experiment-report.html`，不公开实验目录；未提供报告时仍能离线仿真。报告保留全部频率、功率与重复条件，显示实际 S1/S2 GPU 映射及有效频率。数据窗口和有效性限制沿用原始记录，短回合不能当作长期稳定性或完整套件成功率。

实测时间线默认覆盖整个回合，也可选局部时间窗。VLM 为 S2，DiT 为 S1；第一段可见 S2 从 `S2-00` 编号，图前来源标为 `pre`。`S1[n].k` 的 `n` 是实际消费的 S2 特征编号，`k` 是该特征的第几次使用，依赖箭头按完成记录中的特征来源连接。点击请求可高亮来源。缓存 AoI、DiT 读入特征龄、执行动作的图像/状态 AoI 使用各自的时间戳；未知来源留空，不填零。

实测页可把当前条件的 **独占 RPC 均值** 载入机制页。导入不把并发 RPC 当作独占计算，也不按频率线性换算速度。完整 RPC 暂按 GPU 比例 1 建模，主机占比与争用参数需要独立拟合。这是校准起点，不是实测曲线的拟合结果。

机制页新增：

- VLM/DiT 独立的精度标签、GPU 部分加速倍率和新增主机/转换耗时。标签本身不改变速度，也不预测任务精度或成功率。
- VLM 接纳上限：限制请求启动间隔，相机仍按原频率采集，等待位继续保留最新帧。零表示无限制。短窗口的请求数/时长可略高于上限，因为起点即时接纳。
- 可选共享主机服务：两侧非 GPU 前缀非抢占串行，同刻就绪时动作优先；主机服务与 GPU 拓扑分开配置。主机前缀是 CPU、转换、传输的简化假设，不是内核 trace。
- 首次 DiT 额外派发开销：只作用于首次请求。LIBERO 初始独占校准不含它，在线 d 只升不降；论文参考模式的在线滚动窗口包含它，后续 d 可以回落。
- 固定 P95 预算下的四种状态，与自适应 `d` 的真实截止期违约分开统计。同策略原始耗时对照和独立 GPU 对照也分别展示。

“VLM 加速与接纳率反馈”使用本地 LIBERO 调度作为机制示例：GPU 部分假设加速 2 倍，使 VLM 接纳从 5 Hz 升至 8 Hz；共享主机等待使 `d` 从 1 升至 2。设置 5 Hz 接纳上限后保持 `d=1`，相机仍为 20 Hz。该数值不是 INT8 实测收益。“微小超时”示例用 48 ms 基础动作耗时与 3 ms 首次派发开销，演示 50 ms 截止期被跨过后 `d` 保持 2。

模块职责：`model.js` 为调度引擎；`scenarios.js` 为 GPU 部分耗时变换、AoI 与依赖编号；`app.js`/`lab.js` 为交互；`measured-model.js`/`measured.js` 为实测来源解析与回放；`coexecution/queue_report.py` 负责构建可离线分享的报告。原始实验数据和生成报告不进入 Git。

## 两种动作执行与 d 更新口径

机制页默认选择 **论文机制参考**；时间控制条可随时切换到 **本地 LIBERO**。从实测页载入校准时会选择本地 LIBERO，避免把实验记录误认为论文部署产生的数据。两种模式共用 GPU/主机资源模型，比较表使用同一套耗时与量化假设。

| 行为 | 论文机制参考 `paper` | 本地 LIBERO `libero` |
| --- | --- | --- |
| 动作请求 | 上一次完成后连续发起，读取最近控制 tick 的状态 | 控制 tick 接纳，至多一个未消费请求 |
| 初始 d | `paperInitialDelay` 显式初值 | 独占预热 P95 + margin 向上取整 |
| 在线 d | 最近 `delayWindow` 次完整响应均值 / 周期，round-to-even，可升可降 | 消费超时结果时上调，后续变快不下降 |
| 结果暂存 | 一个最新 ready 槽，较新完成结果可覆盖尚未接纳的旧结果 | 完成结果等待控制 tick 领取 |
| 采用新结果 | 当前干净动作段消耗完后，在控制 tick 换段 | 发布到合法、未过期、未承诺的绝对时间槽 |
| 旧动作 | 继续执行当前段剩余向量；每段宽度固定为产生它的请求 d | 继续执行已有未来槽中的动作 |
| 段／槽耗尽 | 重复最后一个已执行的干净动作 | 零位移并保持夹爪；作为无模型来源的 fallback |
| 信息年龄 | 重复保留原 producer、offset、图像/状态来源，AoI 持续增长 | 可追溯槽按其来源计算；无来源 fallback 留空 |

论文参考模式结合[论文 §3.3 与 Algorithm 2](https://arxiv.org/html/2607.26055v1#A3)的滚动 d、连续动作请求和[作者固定版本的超时处理](https://github.com/pi-r2-flow/pi-r2-flow/blob/3af52ca400a6ec7d141416879aa531a6f62f697a/deployment/apps/run_policy.py#L896-L910)。它是调度机制参考，**不宣称逐行复刻作者部署**：明确冻结每请求的干净段宽度，使用统一 40 格示意缓冲、可配置最大 d 和简化主机阶段；不执行神经网络或物理控制。尚未去噪的远端缓冲不作为可执行动作。

`queueDefaults` 保持 `executionPolicy='libero'`，使已有工具调用与实测报告的模型对照向后兼容；网页自身默认 `paper`。新增配置为 `delayWindow=20`、`paperInitialDelay=1`；`delayMode='fixed'` 在两种模式中均可作为固定预算实验。`maxDelay` 是模拟保护上限，不能把它与论文 Algorithm 2 的无上限公式等同。

执行事件新增 `executionMode`、`repeated`。重复事件保留原始 `origin.requestId/offset/producerTargetTick`，不覆盖生产输出的首次 `executedAtMs`。论文模式的输出在实际换段前 `targetTick=null`，窗口结束或 ready 覆盖不会被伪造为过期动作。快照分别展示当前计算、最新 ready、当前动作段与可重复的动作来源。

页面上的“旧段续播与末动作重复”示例固定 d=2、动作耗时160 ms、周期50 ms；执行色条区分新段、旧段剩余向量与重复末动作，AoI 图用淡紫底标记重复时段。“滚动 d 升高后回落”示例以40 ms基础耗时、首轮100 ms附加开销、两次滚动窗口得到 `1 → 3 → 2 → 1`。两个示例都是合成机制说明。

## Queue simulation

Enter complete **solo** VLM and action-request service times in milliseconds, plus the control period and camera rate. The timeline can be played, stepped by control tick or scrubbed to inspect individual waiting frames, feature versions, in-flight requests, completed results and committed execution slots. The timeline displays `S2-00` and its actual consumers `S1[00].1`, `S1[00].2`, etc.; exported engine IDs remain unchanged. A small `×` appears only after an output is actually superseded, protected or expired; a result awaiting future execution is not rejected merely because the display window ends.

Output-slot freshness uses **playback time minus the producing request's state-capture time**, not the target execution time or the latest VLM cache. Light/medium/dark age bands indicate at most one control period, one to two periods, and more than two periods; each slot also shows its numerical state age. The latest adopted batch has a separate outline/badge. Completion, publication, commitment, execution and rejection remain separate states: an older action can still be valid. Tooltips include the age of the visual feature actually read by that request and time since result completion. Bootstrap slots have a distinct initial-state style and an age lower bound, because their observation predates the displayed window.

With `executionPolicy="libero"`, adaptive mode models the repository's LIBERO wall-clock scheduling: a seeded set of 12 warm request durations discards the first two, then selects `d = ceil((P95 + margin) / control_period)`, capped at the selected maximum of at most 5. A late result can increase subsequent `d`. Each request reserves its committed prefix and produces `d` actions for ticks `[r+d, r+2d)`. Publication rejects individual expired/protected slots and preserves remaining valid output. Completed results wait for the control thread's next tick.

For example, under the LIBERO policy a deterministic 80 ms action request at a 50 ms control period selects `d=2` with the default 5 ms margin. The first result executes at 100 and 150 ms while the controller uses bootstrap actions at 0 and 50 ms. Fixed-d mode is an explicit contrast for studying insufficient budgets, buffer exhaustion and fallback; it is an explicit budget-mismatch experiment.

`queueDefaults`, `simulatePolicyQueues(config)` and `policyQueueSnapshot(result, timeMs)` expose the reusable model. JSON exports include inputs, calibration, all events, feature dependencies, per-slot adoption/execution and the selected snapshot. Expiration metrics distinguish in-window publication from final draining; genuine future slots remain censored at the window boundary.

Both scheduling policies use one latest waiting camera frame. It samples bounded uniform service-time jitter and treats the supplied action time as the complete solo request cost, including any preprocessing, feature installation or transport that the user intends to include. It starts with an initialized visual cache and configurable clean action slots. It does not run a neural policy, evaluate action values, or add neural bootstrap/recovery computation. The 40-position rolling-buffer display shows scheduling regions, not denoised tensors. The official robot deployment's continuous-worker timing differs from this repository's tick-aligned wrapper; see [timing protocols](../docs/timing-protocols.md).

### Single-GPU contention

The queue view supports three resource modes through `computeMode`:

- `independent`: GPU phases overlap without slowing one another; the separately selected host policy may still serialize host phases. Default host service remains independent.
- `shared`: both workers may compute on one GPU. During GPU overlap, their remaining GPU work advances at `1 / (1 + slowdownA)` and `1 / (1 + slowdownV)` of solo speed. Each worker immediately returns to solo speed when the other leaves the GPU.
- `serial`: GPU tasks run without overlap or preemption; a ready action task wins a simultaneous task-boundary tie. Non-GPU portions may still overlap when host service is independent.

`gpuShareA` and `gpuShareV` split each sampled solo request into a non-GPU prefix and a GPU work budget. They default to 1; lower the fraction when CPU work or transport is included in the entered request time. Those portions are not multiplied by the GPU slowdown factor. This prefix placement is a modeling assumption, not a reconstruction of a real framework's kernels or transfers.

For an isolated pair of full-GPU tasks starting together, action work of 20 ms and VLM work of 100 ms with both slowdown coefficients set to 1 finish at 40 ms and 120 ms respectively. Only the first 40 ms overlap. Multiplying both whole requests by 2 would incorrectly predict 200 ms for the VLM.

Resource segments and per-job accounting distinguish non-GPU work, GPU work, shared-GPU slowdown and GPU queue waiting. Their actual completion times drive feature publication, action deadlines, adaptive `d`, and execution-slot expiration. Active work drains after the control window; no new camera/request work is introduced during draining. LIBERO initial calibration deliberately uses solo request samples; it is not a contended warm-up measurement. Runtime misses can therefore increase its `d` after contention begins. The paper-reference policy instead starts from an explicit d estimate and recomputes it from its rolling response window.

Slowdown coefficients are editable assumptions or values to fit from measured solo/concurrent traces. Do not enter an already-contended latency and then apply the same slowdown again. This model does not derive occupancy or bandwidth contention from a GPU name, and simulated overlap does not prove CUDA kernel overlap. See [measurement boundaries](../docs/coexecution.md) and the [CUDA asynchronous-execution guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html).

## Checks

```bash
node --test simulator/*.test.cjs
npm ci --prefix tools/ui-check
tools/ui-check/node_modules/.bin/playwright install chromium
node simulator/browser-check.cjs --no-screenshots
# Optional measured-report integration (preview already running):
node simulator/browser-check.cjs --no-screenshots --url=http://127.0.0.1:8765/simulator/
```

Browser dependencies are development-only. Playwright 1.51.1 was used with the original Ubuntu 20.04 workspace; newer hosts may require a reviewed dependency update. Browser checks cover layout, controls and exported records, not CPU model inference. `--no-screenshots` disables image capture; browser assertions still cover controls, provenance, exports and mobile overflow. The optional integration check expects a report with single/dual `core1200` conditions. Downloads and any explicitly enabled screenshots are written to ignored `artifacts/simulator/`. Lightweight CI runs only the Node numerical contracts; browser checks are explicit local validation.

## 中文字体

公共样式优先使用思源黑体（Source Han Sans CN VF / SC），后备为同源的 Noto Sans CJK SC。字体文件是本地资源，位于被忽略的 `simulator/fonts/SourceHanSansCN-VF.ttf.woff2`；来源、固定版本和 SHA-256 见 `sources.lock.json`，许可见 `licenses/SOURCE_HAN_SANS_LICENSE`。使用固定版本仓库内 `Variable/WOFF2/TTF/Subset/SourceHanSansCN-VF.ttf.woff2` 与 `LICENSE.txt`，保持字体原样。

预览从同源地址加载字体；报告导出器将本地 WOFF2 和许可证一并内联，离线打开也可使用。未下载字体时，报告保留本机字体回退。时序与 AoI 的简短图例位于各自图内，详细调度说明折叠显示。

## Quantization and robot debugger

模型协议与动作调度分开标识。`policy_variant=pir2` 来自评测配置和 S1/S2
worker 的一致身份；`execution_coupling=asynchronous` 本身不能证明使用 πR²。
πR² 回放使用独立 S2 VLM 缓存和 S1 滚动去噪，常态每次 1 次 DiT forward，
初始 bootstrap 的 VLM 1 / DiT 4 与整缓冲恢复的 DiT 5 单独显示。完整
VLM + Flow4 的同步、单步预取、K/D 分块预取属于不同策略路径；πR² 缓存模式
不显示 K/D 控件。预测及原生解码长度从 worker identity 读取，不固定写成 16。

导入器可读取 checkpoint 的 `config.json`，但必须与 worker 记录的 SHA256
一致，才显示 streaming/schedule 字段；同目录 `trainer_state.json` 的
`global_step` 和文件哈希作为训练来源保留，不据此宣称任务质量。缺少来源时
训练状态留空，不从路径名猜测。跨精度卡片和后续 SSE 投影绑定同一 policy
variant，包括首次加载时解析的旧版 descriptor；未知身份不能替代已知 πR²。

点击 S1 块可检查真实消费的 feature ID、视觉来源、本体状态 tick、承诺前缀、
`buffer_origin_after`、额外 shift 和恢复标记。轨道沿用实际 RPC 起止时间，
没有服务端分段就不绘制假 VLM/DiT 内核区间。每回合 `policy_evidence` 保存
bootstrap 与所有已记录请求（含末尾未采用结果）的计数审计。旧版 S2 事件
可能没保存 forward counter：只有精确匹配 publication/source/feature sequence
的 consumed cache 能补充计数，同一缓存的多次消费只证明一次 S2 工作；其余
显示 unknown，并与已记录 RPC 数区分。预热、独占校准不计入回合 forward 总数。

`paced` 的 20 Hz 是现实墙钟的目标节拍：第 k 步目标为 `t₀ + k × 50 ms`。
提前完成的推理仍等待执行时刻，实际迟到会按日志保留；VLM/DiT 可与动作执行重叠。
缓存策略的 `Action execution / gap` 轨道使用 `applied_s` 到
`next_observation_computed_s` 的实测 `env.step` 区间，每步推进的仿真时间单独显示。
`Between steps` 是相邻执行之间的实测间隔，包含节拍等待和控制开销，不能视为纯 sleep。
点击执行块可按 `executed_slot` 追溯 DiT→VLM；bootstrap、fallback、缺失或歧义来源保持明确。
缺少完成时间不会推算成 50 ms；缺失动作记录之间不会补画空闲。

机器人闭环视图可切换 `paced` 20 Hz 目标墙钟节拍与 `ready`
“就绪即接纳（StreamingVLA式调度，GR00T）”。后者使用真实重新录制的
GR00T 轨迹，在原子仿真步结束后接纳完成结果；物理步长仍为 0.05 s。
实际墙钟执行率、S1 请求率、计划更新率、就绪至首次执行等待与仿真/墙钟倍率
分别显示。Ready 没有固定频率有效性，不会用零代替未知或不适用的值。
依赖链与视频继续按本精度实际来源和采集时间回放；没有 AFM/AEO 权重适配，
也不将旧视频加速当作新执行结果。Catalog 可用
`--default-execution-clock ready` 显式设置默认组，URL 的
`?execution_clock=paced|ready` 可切换；旧 catalog 保持 paced。

Serve an explicitly built v1 replay catalog with the existing preview server:

```bash
python simulator/serve.py --port 8765 --replay /path/to/catalog.json
# A directory containing catalog.json is accepted too.
```

Open `http://127.0.0.1:8765/` (defaults to the debugger when configured), or
`/simulator/replay.html` directly. Recorded robot runs open the robot-video view by
default; `?view=model` explicitly opens fixed-input model replay. The two prominent
tabs distinguish actual robot video from the static model input images. The existing queue teaching and measured-report
views remain at `/simulator/`; `--report` and `--replay` can be used together.
Build recording/catalog inputs with the [replay evidence package](../coexecution/replay/README.md).

The interface follows the compact header, experiment controls, precision cards,
shared cursor and event inspector of the VLM realtime-caption viewer. It contains
no external assets or framework dependency. `replay-model.js` interprets evidence;
`replay-view.js` renders it. `replay-media.js` owns bounded media decoding and
keeps the last decoded frame visible during forward playback; backwards seeks
hide any frame newer than the cursor. `replay-timeline.js` uses measured task
intervals and the existing S2/S1 source aliases; it does not guess component
durations from a complete request's wall time. The server imports only standard-library evidence
helpers, never Torch, NumPy archives, a simulator or a model.

The two evidence tabs are the single view switch. Initial loading and catalog
refresh share experiment filtering and run-option rendering; the card grid follows
the number of visible precisions. Recorded source dependencies are interpreted by
`measured-model.js` and `replay-timeline.js`.

- **Independent model**: select a measurement boundary, real input and seed.
  `policy` includes raw-input preparation and action decoding;
  `preparation_transfer` only prepares/transfers inputs and shows no neural
  task blocks; `prepared_model` measures the resident-input model body.
  Profile preparation+model calls explicitly exclude decoding and are a separate
  `preparation_model` boundary. Statistics and cross-precision matches never mix
  scopes. Unknown boundaries remain inspectable without inferred neural stages.
  An exact record in its own run can be viewed even without a declared clock cohort;
  it is labelled unpaired and other precisions remain unmatched. Missing seeds,
  observations or cohorts never authorize a cross-precision match.
 Clean wall P50/P95 use only
  uninstrumented samples within the selected boundary; their exact decoded action sequence appears after that
  request completes. Profile records expose the actual VLM/action GPU events,
  stage, category, stream and full kernel name. Profile calls have their own seeds
  and outputs; unavailable outputs remain unavailable. GPU work, device span and
  host wall are different, nonadditive boundaries. Filtering a module/category
  never fabricates a timeline for Clean samples.
- **Robot closed loop**: each precision plays its own main/wrist video and actual
  state/command history. The shared cursor is elapsed control wall time. Frame
  timestamps map to explicit encoded-media timestamps; playback does not assume
  wall time equals frame index / nominal control rate. Initial frames may predate
  control start. States become available at their recorded computation time;
  commands become available at actual execution. Ended episodes hold their last
  available frame with an ended label. Missing videos, failed episodes, invalid
  control timing and unmatched runs stay visible.
- The XY projection uses measured end-effector position in metres. State rotation
  is an axis-angle **rotation vector**, not Euler angles; gripper state has two
  joints. Normalized controller commands are separate selectable channels.
  Selecting a command exposes its request, source frame/feature and cache/image/
  state AoI through the existing measured-queue helper. Diverged robot trajectories
  are never presented as the same-input numerical comparison.
- The primary timeline has separate VLM/S2 and DiT/S1 tracks for each precision.
  Thin task bars move through a following time window, with explicit not-started,
  running, completed and failed/unknown states. `S1[n].k` names the actual consumed
  S2 feature, never the newest cache by proximity. Click a bar to inspect its
  source and timestamps; recorded upstream tasks are highlighted with a dashed
  outline. Detailed kernels remain in the expandable section.
  Cached-asynchronous deployment bars are RPC intervals. Synchronous VLM/DiT bars are
  server stages within one full inference RPC, not separate RPCs or GPU kernel
  durations. A third `Inference / Execution` track uses only recorded
  `episode.phases` intervals on the control wall clock: inference pauses physics;
  execution includes the configured action-phase pacing. Execution source links
  use the phase's recorded producer request identity, never the closest task in
  time. Missing phases or producer identities stay unavailable. The coupling
  selector is independent of cadence; synchronous `paced` means 20 Hz within
  execution phases only, not a global fixed control frequency. Cards show the
  current phase, wall/physics clocks and retained episode outcome. Videos continue
  to use recorded frame timestamps, including held frames during inference.
  Model Profile bars span the actual first
  and last attributed CUDA activity, including gaps; host-submission intervals
  remain explicitly labelled. Clean records without component timestamps show
  unavailable tracks. Bar shading follows the recorded playback cursor, not an
  estimated fraction of neural work. There is no separate dependency diagram.
- **One-step asynchronous** is a separate `one_step_async` coupling option, available
  with `ready` cadence. It overlaps a complete VLM+Flow4 request for the next action
  with one current physical step. The first bootstrap uses observation 0 for action
  0; the next request also captures observation 0, but has its own request/feature
  identity. Subsequent labels show `Obs k → action k+1`. A physical observation lag
  of one step and wall-clock AoI are distinct metrics. The model's native decoded
  horizon stays unchanged; only the first command is executed from each result.
  Separate `Policy RPC` and `Execute / Wait / Drain` tracks expose overlapping
  intervals without hiding bars. Waiting never consumes extra old actions. Clicking
  execution highlights the exact prior DiT and VLM producers, even while the next
  inference runs. The auto window retains these sources. Cards can show
  `Inferring + executing`, explicit waiting or bootstrap, and stopped-robot terminal
  drain. Playback extends through `work_end_s`; the robot itself freezes at
  `controlled_end_s`, while final unused inference remains visible. Frame and
  physics progression still use actual post-step timestamps, never nominal FPS.
  The importer and raw feature auditor share a full-work evidence gate: every
  `work_requests` row is matched to the original submission, capture/hash,
  RPC/server stages, result disposition and physical execution. Missing, duplicate
  or contradictory terminal work rejects the catalog. The control-window queue
  statistics stay separate from the full `replay_requests` task display.
  Async RPC begins **after** worker input preparation; synchronous full-call
  timing includes preparation. Compare submission-to-response boundaries when
  assessing complete policy latency. Both are instrumented host measurements,
  not GPU kernel activity or full-suite success-rate evidence.

  ```bash
  python -m coexecution.replay build \
    --deployment /path/to/one-step-async/deployment \
    --deployment /path/to/matched-synchronous/deployment \
    --default-execution-coupling one_step_async --default-execution-clock ready \
    --output .local/one-step-replay
  python simulator/serve.py --replay .local/one-step-replay/catalog.json
  ```

  The catalog default or `?execution_coupling=one_step_async&execution_clock=ready`
  selects the new protocol; catalog refresh, task/camera changes and SSE retain the
  same coupling filter. `ready` here means unpaced physical steps with explicit
  result waits, not 20 Hz wall control or StreamingVLA weights.
- **Prefix / overlap matrix** adds `chunked_async` with synchronized **K** and
  **D** selectors. K is the maximum number of native decoded actions executed
  from each accepted result; D is the number of current-prefix actions remaining
  when the next observation is captured and its full VLM+Flow4 request is submitted.
  D=0 uses `synchronous`; D>0 uses `chunked_async`. The older `one_step_async`
  K=1/D=1 and cached-worker `asynchronous` remain separate protocols. Cached-worker
  recordings have no invented fixed K/D. Choices come from measured catalog rows,
  not a hardcoded grid: all precision cards share one cadence/coupling/K/D, and
  run selectors only choose repetitions of that schedule. A K or coupling change
  retains the old D when available, otherwise selects the lowest recorded D for
  that K. Catalog refresh preserves a valid selection; a removed combination
  falls back deterministically and loads a new cohort. Only that cohort's run
  JSON is loaded. Stream updates that change its identity or schedule are withheld.

  Request/feature IDs are target ticks 0, K, 2K, …; they are not array indices.
  Each physical execution block has its own tick and native offset, and all steps
  of one prefix link to the same exact DiT→VLM producer. The auto window keeps the
  producer visible at the end of a long prefix. Bootstrap lag equals the native
  offset; later lag equals D+offset physical steps. Wall AoI is measured separately.
  Cards show native decoded horizon, accepted prefix and actual executed prefix:
  success can stop midway, and a finite step cap can shorten the final prefix.
  The normalized request's `adopted_s` is adoption; v2 `finalized_s` is recorded
  after actual execution and must not become a false response or inference end.
  Wait and terminal drain preserve the stopped robot and physics clock; the full
  unused final VLM/DiT computation remains visible after control ends. Every imported
  v2 record passes the shared strict full-work evidence validator.

  Use **control wall / actual actions (ms/action)** and **effective execution rate
  (Hz)** for matrix timing, alongside prefix-boundary gap P95, adjacent gap P95 and
  explicit wait count/time. An adjacent-gap P50 is dominated by block-internal
  steps when K>1 and is not whole-episode throughput. Each card describes its
  selected episode and outcome; no distinct K/D, successful/failed trajectories
  or max-step terminations are pooled. These are recorded host bounds with camera
  overhead, not GPU-kernel measurements or suite accuracy.

  ```bash
  python -m coexecution.replay build \
    --deployment /path/to/matrix/evaluation \
    --default-execution-coupling chunked_async --default-execution-clock ready \
    --default-execution-prefix-steps 8 --default-async-overlap-steps 1 \
    --output .local/matrix-replay
  python simulator/serve.py --replay .local/matrix-replay/catalog.json
  ```

  URL query parameters `execution_prefix_steps` and `async_overlap_steps` select
  the measured schedule together with `execution_coupling` and `execution_clock`.
  The URL and exported evidence selection retain all four dimensions. Old catalogs
  missing K/D descriptors remain inspectable, but incompatible loaded prefixes are
  withheld rather than compared; rebuild such catalogs to enable their selectors.
- Declared live streams use `replay-stream.js`: snapshot first, then ordered SSE
  events, duplicate/gap checks, resnapshot and explicit terminal closure. Live
  updates preserve media, cursor and user selection; following the latest event
  is opt-in. Catalog refresh discovers newly recorded runs without starting them.
  Missing clocks, recorder loss and failures remain visible; browser receipt
  time does not replace an observation, GPU or execution timestamp. Catalogs may
  set `default_view` and `default_record_type`; explicit URL choices take priority.
- Incomplete/unavailable controller traces keep their task identity, status and
  reason. They cannot play a normal control episode even if metadata contains a
  success flag. All secondary controls are disabled while a new run set loads.
- Use playback speed, scrub, previous/next event, window and precision controls to
  inspect a run. Model time is milliseconds (default slow playback); robot time is
  seconds. Configuration changes commit all cards/plots together and discard stale
  HTTP/media responses. Export downloads only the current evidence selection;
  “运行来源” and catalog/run JSON preserve source hashes and validation status.

The preview exposes only the selected catalog and its explicitly listed run/media
files through `/api/replay/catalog`, `/api/replay/run/<id>` and
`/api/replay/media/<id>`. Media supports bounded byte-range reads and HEAD. IDs,
paths and MIME types are validated; absolute paths, traversal, symlinks and
unlisted files are rejected. Directory listings are disabled. APIs and HTML/JS/CSS use no-store (static conditional requests return fresh
content); media can be privately cached for one hour. Keep configured artifacts unchanged
for immutable completed runs. Registered stream snapshots/journals and their
media manifests may grow through the explicit recorder protocol. The preview binds to loopback; it is not a public artifact
hosting service.

```bash
node --test simulator/replay-model.test.cjs
python -m pytest -q simulator/test_serve.py
# Explicit browser integration with real prebuilt evidence; never launches inference:
node simulator/replay-browser-check.cjs --catalog=/path/to/catalog.json
# Or against an existing preview:
node simulator/replay-browser-check.cjs --url=http://127.0.0.1:8765/simulator/replay.html
# Desktop transport contracts using protocol fixtures, not measured model data:
node simulator/replay-stream-browser-check.cjs
```

The optional browser check expects a three-precision model/robot debug catalog
with main/wrist videos and a retained failed, timing-invalid example. It verifies
actual WebM pixel changes during several seconds of PLAY, visible decoded frames,
pause/resume, backwards seek, delayed media, dependency selection, mode/camera
changes, stale requests, secondary controls during loading, incomplete records
and desktop layout. It does not run mobile-browser checks. Screenshots go to ignored `artifacts/simulator/`; append
`--no-screenshots` to skip images. These are interface checks, not inference or
LIBERO task-quality validation.
