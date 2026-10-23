# LIBERO-Dynamic：移动目标下的速度与控制质量

本模块接入 [LIBERO-Dynamic](https://arxiv.org/abs/2610.24579) 的移动目标环境，用同一组任务、轨迹和初始条件比较默认 VLA、量化推理与 πR²。核心观测是目标在等待模型期间移动了多少，以及返回动作能否完成抓取和放置。`inventory` 固定完整条件清单，`evaluate` 将已有 GR00T 服务接入上游物理载体、任务判定和录像循环。速度—成功率实验及 πR² 动态适配分别记录实际验收结果。

## 基准与模型

动态基准覆盖四套 LIBERO、40 个任务；每任务3个轨迹家族、5条路径、3档速度（0.03/0.06/0.09 m/s），合计1800个条件。Spatial 单套为450个条件。570 条轨迹变体通过载体接触摩擦驱动物体，保留抓取后脱离载体的物理行为；Goal task 0 和 task 7 的另外30条变体使用柜体旋转/炉具平移，沿用上游无载体 fixture-body 路线。环境配置与资源不进入本仓库。

模型基线使用本项目已经保存并使用过的 NVIDIA GR00T-N1.7-LIBERO 对应套件任务权重：默认完整 VLM + Flow4，预测长度40、原生解码16、执行前缀8。πR² 使用独立的已有适配权重。服务指定本地checkpoint并设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`；不下载新的VLA权重。当前阶段不使用论文的数据生成补偿器或微调方法，也不把低比特模型预设为更快或更差；静态质量与动态任务成功率分别实测。

## 准备与清单

```bash
python tools/prepare_sources.py --component libero-dynamic
python -m coexecution.libero_dynamic inventory \
  --upstream third_party/libero_dynamic \
  --output .local/libero-dynamic/conditions.json
```

源码固定在 `1e5845720643da281c47bf4423ae99e115976a13`。清单验证上游评审配置哈希，保存任务、轨迹、速度和来源；它只读取配置，不执行模型或仿真。仿真使用独立 Python3.10 环境、robosuite1.4.0、MuJoCo2.3.7及EGL，模型继续使用现有 CUDA 环境。依赖见 [requirements-simulation.txt](requirements-simulation.txt)，其中 termcolor 显式补齐 robosuite 的日志依赖。PyTorch 使用适合目标驱动的 CUDA 构建，模型环境与仿真环境分开。初始化、接触和位姿校验通过后再开展模型评测。

默认使用 `seed=0`、`init_state_id=0`。配对其他已测种子时，可显式传入 `--seed 8000`；支持0至2³²−1的整数，初态索引仍限定为0。非零seed记录为 `seed_scope=custom`，条件ID追加 `__seed8000` 并保留原模板来源，前端显示实际seed。官方1800项清单和默认ID保持不变；自定义seed结果单独统计。例如原始静态入口对task8使用基础seed0加task_id×1000，动态跟进应显式设8000，不能将seed0录像标成同一随机种子的配对。

## 必须显式区分的时间与静态对照

论文附录将延迟量化为 `ceil(latency ×20)` 个控制周期；发布代码的载体 preview 路线按精确延迟秒数推进 MuJoCo 子步，并锁住机器人关节。60ms 在两者中分别对应100ms和60ms，实验身份须注明选择。默认执行预算600步只统计控制动作，同步等待另计物理时间。

无载体 fixture 路线的上游实现把等待放在下一次 `env.step` 内，每块推进10ms，余数向上取整。本模块的 `NativeFixtureTiming` 将等待物理独立推进，保留10ms的 fixture 更新粒度，最后一块及 MuJoCo 子步按实际余数缩短；61.3ms 的等待因此推进61.3ms，原上游会推进70ms。该扩展标记为 `latency_backend=owned_fixture_exact_hold`，记录原生取整时长、实际物理时钟、fixture elapsed 和每块驱动事件。初始化、40步 pre-policy control、动作步的 fixture 驱动及任务成功判定仍走原生 non-preview 路线；`motion_phase_s` 读取真实 fixture elapsed，包含源码 start_step 生效后的阶段。`paper_quantized_sync` 的50ms整倍数等待同时是10ms整倍数。

Goal task 0 的柜体 yaw 使用 `alpha_yaw_rad_per_s`；修改名义 `speed_mps`/`alpha_mps` 不会改变该角速度。其三档名义速度条件应按实际角速度识别，不能作为三档物体运动速度曲线。`fixture_driver` 保存有效运动类型、实际速度参数/单位以及 action/latency 的完整运动配置，区分角速度与线速度；这里保留源配置的速度语义。

等待只冻结源码默认选出的机器人关节，自由物体与抽屉/按钮等任务关节继续参与物理。等待通知消费一次；下一动作步不会再次执行等待。录像按实际物理子步采样，独立等待后修正上游非 preview 回调的旧偏移。宿主合同测试覆盖这些时钟与状态约束；真实 Goal fixture 闭环结果需单独验收，不能由配置清单或宿主测试推断。

发布代码中的部分轨迹函数会把 `alpha_mps=0` 回退为0.055m/s，因此不能用改零速度宣称静态对照。静态控制需要明确停用或冻结载体运动，并核对目标实际位移。动态的40步初始相位必须真实推进场景；推理延迟期间也保留目标运动。

评测记录同时保留仿真时间、墙钟时间、校准延迟、实际目标位姿、动作来源、成功率和空队列事件。视频覆盖等待期间的移动过程；默认 VLA、量化与 πR² 的对照使用明确一致的评测条件，不从固定输入误差推断动态任务排名。

### 比较结果时使用的口径

| 实验 | 推理与物理时间 | 应比较什么 |
| --- | --- | --- |
| 默认 VLA，`--native-latency` | 每轮新鲜 VLM + Flow4；移动世界先等待该次完整调用，再执行前 K 步 | 相同任务/轨迹/初态下的成功、请求耗时、等待总量、实际动作数和物理完成时间 |
| πR²，`--timing-mode native` | 启动 hold 后每步推进50ms；S1/S2 逻辑完成时间来自各自实测调用，实际 CUDA/物理串行 | 成功、特征年龄、过期动作、fallback、承诺窗口；同卡真实墙钟争用需另做部署实验 |
| 固定延迟控制 | 模型仍真实执行，物理等待或逻辑服务使用显式指定的时长 | 隔离时序或动作数值的影响；结果以固定控制身份保留 |
| 原始静态 LIBERO | 使用原始场景；同步推理期间仿真暂停 | 现有权重的静态任务质量；与动态载体场景分别标识 |

默认 VLA 的平均物理动作频率为 `N / (sum(request_latency) + 0.05N)`，其中 N 是实际执行步数。减小 K 会增加重规划频率，也增加单位动作数对应的完整 VLM/Flow4 调用；必须同时报告等待量和任务结果。K 不改变预测长度40、原生解码16或 Flow4 的去噪次数。

静态质量排名和动态质量排名分别由配对结果给出。只有静态对照确实观察到低精度质量下降，才能把动态结果解释为“速度弥补静态质量损失”；不能从精度位数推断该前提。代表性视频保留所选条件的成功与失败，完整套件按全部预定条件统计；未完成批次只报告覆盖进度。

### 保持动态布局的外驱冻结对照

发布配置可能通过 `initial_scene_adjustments` 调整物体起始位置。降低 `alpha_mps` 后，这些布局调整和载体接触条件仍然存在，因此低速动态场景与原始静态 LIBERO 分别标识。

`--scene-drive frozen` 是诊断用的自定义对照。初始化完整执行原配置及40个 pre-policy 动作；随后读取实际载体位置，在本回合的状态副本中将路径改为一个零偏移点。原来的正速度只定义冻结前的初始化过程，控制阶段的外驱保持在捕获位置。原生推理耗时照常用于真实物理等待，动作步仍推进50ms；抓取物的 qpos/qvel 不由冻结适配器改写，重力、接触和机器人作用继续生效。

当前对照支持 preview 路径中的物理 slide-joint 载体；其他表示会在初始化时明确拒绝。`drive_freeze` 保存冻结阶段、载体关节和目标状态、原始/有效路径及捕获前后的状态指纹。条件ID追加 `__drive_frozen`，`parent_dynamic_condition_id` 指向同布局、速度、seed的移动条件；默认移动实验和官方清单不变。验收核对配对初始观测、载体实际漂移和自由物体可操作性，前端将它标为“外驱冻结对照”，不作为原始静态基准统计。

## 运行已有模型的默认动态评测

在模型环境中启动已有套件权重的普通 Flow 服务。以下路径变量应指向本机现有目录，输出目录为本地实验目录。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m coexecution.libero.service \
  --checkpoint "$LOCAL_SPATIAL_CHECKPOINT" --variant flow --role action \
  --port 19170 --device cuda:0 --inference-precision bf16 \
  --runtime-receipt --output .local/dynamic-model-bf16
```

在独立仿真环境中选择套件、任务、轨迹和速度，连接上述服务。每个输出目录只能创建一次，重跑须用新目录。

```bash
python -m coexecution.libero_dynamic evaluate \
  --upstream third_party/libero_dynamic \
  --output .local/dynamic-spatial-bf16-v006 \
  --suite libero_spatial --task-ids 0 3 --levels l1 --traj-indices 0 \
  --speed-mps 0.06 --model-port 19170 \
  --expected-checkpoint "$LOCAL_SPATIAL_CHECKPOINT" --precision bf16 \
  --chunk-steps 8 --max-steps 600 --native-latency \
  --timing-rule release_continuous_sync
```

省略任务、轨迹家族和路径参数时，会运行该套件当前速度下的全部150个条件；三档速度分别运行。`libero_object`、`libero_goal`、`libero_10` 必须连接各自的已有套件权重，不能复用 Spatial 权重后称为套件基线。所有神经模型在 CUDA 服务上运行。

`--native-latency` 使用每个正式请求的原生实测耗时：复用上游 `measured_wallclock` 采样器，以 `perf_counter` 计完整 `get_action_chunk`（输入处理、RPC、输出处理）及返回结果的 NumPy 规范化。未舍入的 `chunk.policy_elapsed_s` 直接作为该次物理等待时长，机器人保持、物体继续运动；不取共同常数、中位数或均值，不按50ms控制步取整。每回合预热后重置策略，预热、身份/重置RPC和物理计算不进入正式请求计时，预热也不推进动态世界。

原生结果标记 `timing_mode=native`、`latency_kind=measured_per_request`，保存明确的 `latency_boundary`。每条请求记录 `controller_policy_elapsed_s`，并通过动作块ID关联实际等待起止和物体位姿。适配器、RPC和服务端的 monotonic 时间单独保留；采样器只提供时长，不据此伪造完整调用的绝对时间端点。该模式不创建 `calibration.json`，批次固定延迟字段为 null，且与 `--calibrate`、`--calibration`、`--latency-seconds` 互斥，拒绝 `paper_quantized_sync`。

历史校准实验仍通过显式 `--calibrate` 选择。校准使用第一个选定场景的真实初始观测，8次预热后保留88次完整请求，取中位数作为该批次固定延迟；完整请求包含输入处理、RPC、模型和动作输出处理。正式开始前重置策略随机状态。每个后续场景仍有预热，物理场景在预热和校准期间保持初始状态，正式推理等待才推进动态时间。实际请求耗时与施加的校准延迟分别保留。`--calibration` 可复用匹配执行身份的实测校准；`--latency-seconds` 是明确标记的延迟控制实验，不能作为模型速度实测。

少于8次预热或88次保留调用的校准标为 `diagnostic`，样本数随结果保存。校准身份核对权重内容、精度、执行前缀、输入形状、GPU和可用的源码/运行时证据；锁频与功耗配置仍需由实验方保存，改变这些条件后重新校准。正式延迟测量使用无其他任务争用的GPU。

`evaluation-summary.json` 给出成功数、时序规则和校准身份；各场景的 `policy_audits` 保存完整模型输出、执行前缀、服务阶段、动作块与物体位姿。只读 `observation_registry` 补全源目标、coupled objects、passengers、场景自由物体、fixture 根和关节，位姿记录包含 qvel 与全部已知关节状态；加入观测集合不会增加驱动目标或冻结对象。上游 step trace 保留逐动作队列和任务状态。`video_samples` 将每帧映射到仿真时间及推理等待/动作阶段。视频为物理场景相机画面，GR00T 实际输入另经官方图像转换；视频60fps与控制20Hz分别记录。

## 执行前缀与量化配对

`--chunk-steps 8`、`2`、`1` 选择每轮实际执行的前缀。三者都重新读取观测，执行完整 VLM + Flow4，再从原生16步输出中取前 K 步；改变 K 不会缩短预测长度或减少去噪次数。同步时序为 `观测 → 推理等待τ → K个50ms动作 → 下一次观测`，等待与动作期间物体都继续运动。

固定τ时，完整周期的平均动作频率为 `K / (τ + 0.05K)`；原生模式按已执行动作数除以实际累计物理时间统计，其中每次请求的等待分别累加。因此缩短前缀既增加重新观察次数，也增加单位动作数对应的等待总量。报告任务成功、动作数、物理完成时间和等待总时长，才能区分这两种影响；600动作预算下，不同K的物理时长可能不同。

隔离前缀的影响时，固定场景、初态、模型和 `--latency-seconds`，只改变 `--chunk-steps`，每次写入新输出目录。采用旧K8实测值作为K2/K1的固定等待时，结果仍属于反事实时间控制。比较各K的原生耗时与动态表现时分别使用 `--native-latency`；每个请求使用本次完整调用耗时，包含实际波动。历史固定校准实验使用 `--calibrate`，校准身份包含执行前缀。

量化配对再增加精度这一维：同一已有checkpoint分别启动BF16和W8A8服务，评测端的 `--precision` 与服务一致。原生对照分别使用 `--native-latency`，由两种精度各自的逐请求耗时决定物理等待，保留精度、速度与波动的联合效果。仅在显式时间控制实验中固定相同等待以比较动作数值变化。前端用K、精度与等待来源标识配置，原生配置展示完整调用的实测范围及每次Hold，验证其与本次请求耗时一致；两路视频按物理时间同步。

## 默认 GR00T 策略适配器

`GrootChunkPolicy` 使用本项目的 `synchronous_plan` 服务入口，执行完整新鲜 VLM 和四次 DiT，再取默认8步前缀。图像和状态转换复用 NVIDIA 官方封装；模型输出完整长度单独保留，夹爪0/1约定只在送入原生LIBERO前转换一次。接口与宿主合同测试已实现，真实动态模型闭环结果单独报告。

```python
from coexecution.libero_dynamic.policy import GrootChunkPolicy

policy = GrootChunkPolicy(port=5570, chunk_steps=8, seed=0)
# These calls run on the same inference thread:
identity = policy.prepare()  # Connect and reset outside the measured request.
actions = policy.get_action_chunk(raw_observation, task_language)
policy.close()
```

运行时 `PYTHONPATH` 需包含本仓库、`scripts/` 和固定 GR00T 源码。连接在 `prepare()` 或第一次推理的线程创建，后续使用和关闭保持在该线程；reset可以由控制线程请求，在下一次 `prepare()` 或推理前执行一次。上游控制器退出前，先排空请求，再通过其executor执行 `policy.close()`，最后关闭executor。`policy.records` 在内存保存完整模型输出、执行前缀和RPC/服务端阶段，不在模型计时路径写盘。

观测可携带 `__pir2_capture` 的来源步、步计数类型、仿真时间和实际capture时间。动态评测的 `source_tick_kind=action_step` 统计已执行动作，不包含推理等待子步；完整物理时间单独记录。缺少来源时标为请求序号/策略入口时间，环境接入须在上游观测快照阶段保留这些字段。

## 已有 πR² 权重的动态控制实验

`evaluate-pir2` 使用本项目已有的 πR² checkpoint，连接两个独立 policy 实例的 action/VLM 服务。服务必须声明 `variant=pir2`、原生预测和解码长度40、`numpy_bf16` 特征传输及 combined install/plan；两侧权重哈希和请求精度必须一致。普通 Flow 权重不能直接改名为 πR²。客户端复用 `Backend.bootstrap/vision/plan` 的真实神经推理和滚动缓冲，不重写模型算法。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m coexecution.libero.service \
  --checkpoint "$LOCAL_PIR2_SPATIAL_CHECKPOINT" --variant pir2 --role action \
  --port 19180 --device cuda:0 --vlm-port 19182 --vlm-device cuda:0 \
  --feature-transport numpy_bf16 --inference-precision bf16 \
  --runtime-receipt --output .local/dynamic-pir2-model

python -m coexecution.libero_dynamic evaluate-pir2 \
  --upstream third_party/libero_dynamic \
  --output .local/dynamic-pir2-spatial-v003 \
  --suite libero_spatial --task-ids 0 3 --levels l1 --traj-indices 0 \
  --speed-mps 0.03 --action-port 19180 --vision-port 19182 \
  --expected-checkpoint "$LOCAL_PIR2_SPATIAL_CHECKPOINT" --precision bf16 \
  --timing-mode native --max-steps 600
```

`--timing-mode native` 是默认模式：每次bootstrap、S2、S1的完整适配器调用分别实测，包含输入准备、RPC和返回处理；该次耗时原值直接决定该请求的逻辑完成时间。两种精度各用自己的逐请求耗时与波动，不取共同常数或均值，不添加最低服务时长。`service_wall` 保存独立的monotonic起止时间、耗时及边界，原始RPC/server时间另存。结果标记 `measured_per_request` 与 `measured_request_discrete_event`。

实际CUDA调用与物理计算仍分阶段串行，S1/S2具有独立逻辑服务通道；这是逐请求实测时长驱动的事件实验，资源模型为 `independent_stage_service`、计算方式为 `serialized_gpu_simulation`。它保留每次真实调用时长，但不等于同卡真实墙钟并发部署。50ms是LIBERO动作的物理步长，模型请求不会被拉长或缩短到50ms。

配对录像可加 `--startup-release-seconds 0.4`，将两种精度的控制起点共同设为首次观测后的400ms。bootstrap的真实调用耗时仍单独保留，差额记为 `startup_barrier_seconds`；物体在整段等待期间持续运动，物理层只推进一次总等待，避免两种调用时长改变子步切分。若实际bootstrap超过指定起点，该回合保留耗时和物理证据并在执行动作前报错；不会截短调用或声称对齐。S1/S2后续请求保持各自实测时长。验收配对时同时检查初始与启动后观测指纹，不能只看两个总等待数值相等。前端分别显示 Bootstrap hold 与 Startup barrier，实际视频帧时间保持原值。

较慢物体的示例使用 `--speed-mps 0.015 --custom-speed` 或 `--speed-mps 0.01 --custom-speed`。这是实际物体运动速度，不是网页播放倍速；所有目标的速度配置在副本中覆盖。自定义条件标记 `speed_scope=custom`，使用独立条件ID，并保留官方模板来源；不计入官方0.03/0.06/0.09m/s的1800项评测清单。零速度不是静态开关。相同场景、速度和起点下的BF16/W8A8录像用于检查动态响应，完整套件成功率仍须单独统计。

历史固定延迟控制仍通过显式 `--timing-mode fixed` 使用，并且必须给齐 `--bootstrap-latency-seconds`、`--s1-latency-seconds`、`--s2-latency-seconds`、`--recovery-latency-seconds`。该模式标记 `fixed_counterfactual` / `counterfactual_discrete_event`；native模式拒绝这些固定延迟参数。

πR² 的量化对照仍加载同一已有πR² checkpoint。paired服务设置 `--inference-precision w8a8 --quantization-scope all`，评测设置 `--precision w8a8`，并使用独立输出目录。默认原生对照保持承诺策略、场景和初态一致，每种精度使用各自真实请求耗时；启动等待也各自实测，因此启动后世界状态可以不同，并会记录新的观测指纹。需要单独隔离动作数值变化时才显式采用fixed控制。不能沿用默认Flow4的整轮延迟代替πR²阶段耗时。RPC身份中的 `inference_optimization.coverage` 记录实际转换的VLM和动作侧模块。πR²可以使用分DiT CUDA Graph；整段 `--action-cuda-graph` 和 `--group-conditioning` 属于Flow路径，服务会拒绝将它们用于πR²。

初始化沿用同一任务、轨迹、速度、所选seed（默认0）/init0 和40个真实 pre-policy 步。默认每场景预热8轮；responsive 调度逐轮覆盖 d=1..5，`--fixed-delay d` 只预热该 d。每次预热执行真实 bootstrap→S2→S1，世界时间保持不变，记录均标为 warmup。`--warmup-requests 0` 可用于明确不预热的诊断。正式开始时重新 bootstrap/reset seed；启动的真实 VLM1/DiT4 按该次实测耗时（fixed模式按指定参数）只冻结机器人并推进移动世界，随后强制读取新状态。bootstrap 图像保留原 capture，不能将它改写为启动等待后的新时间。

之后每50ms执行一个真实动作，物体持续运动；S1/S2 等待与这些物理步重叠，不再额外增加 hold。S1 使用已发布的 S2 特征和当前状态，正常1NFE，全 H40 缓冲失效时恢复5NFE；native模式直接测量本次恢复调用，fixed模式才使用独立的 `recovery-latency-seconds`。每次先承诺 `[r,r+d)` 前缀，输出只属于原始 `[r+d,r+2d)`；过期动作原位丢弃，已承诺动作受保护，队列耗尽执行零位移并保持夹爪的 fallback，仍推进真实物理。

S2 只有一个在途请求和一个可覆盖的 waiting frame。恰好在控制边界完成时，先发布缓存，再提供该边界的新图像并开始下次请求；例如150ms S2 在50ms控制周期下取 tick0、3、6。中途完成只能取之前已发布的历史帧，不能看宿主已计算但逻辑上尚未到达的下一帧。结束边界不会再启动新请求，未完成结果保留为 unapplied。

`physical.py` 共享默认 Flow 与 πR² 的物理初始化、动作、等待和录像采样。默认 Flow 仍使用上游 `run()` 的 fixture 回调补偿；πR² 自有循环使用实际物理时间原点，显式禁用该旧补偿。视频默认60fps，包含 startup hold 和动作期间的真实子步画面；双相机与腕部各保存 MP4。`video_samples` 保存每帧物理采样时间、动作步与阶段，控制图像采样仍为20Hz。

输出包括批次 `evaluation.json`、`manifest.jsonl`、`evaluation-summary.json`，以及逐场景 manifest、完整 streaming 请求/动作/缓存血缘、真实 RPC 审计、物体位姿、源物理诊断、视频映射和原始录像。每个成功返回的动作步必须推进50ms，完整物理时长必须等于 startup hold 加实际动作数×50ms。异常保留已完成请求和动作、在途状态、实际部分物理时间与录像，并在同一线程关闭两条 RPC、摄像机和环境。前端导入与真实闭环效果需按对应实验另行验收。

异常回合的 `steps` 按实际完成的50ms物理动作计数，`controller_recorded_steps` 单列核心成功接收的动作记录数，便于识别物理已完成而后处理失败的情况。每个 `video_samples` 条目保留采样时间和双路 `streams` 写入状态；某一路编码失败时，另一侧已写帧的索引和时间映射仍保留。批次中断会先登记该回合的条件身份与失败产物，再传播原中断。
