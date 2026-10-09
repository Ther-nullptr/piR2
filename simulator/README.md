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
