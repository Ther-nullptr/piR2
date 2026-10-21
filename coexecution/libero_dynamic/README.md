# LIBERO-Dynamic：移动目标下的速度与控制质量

本模块接入 [LIBERO-Dynamic](https://arxiv.org/abs/2610.24579) 的移动目标环境，用同一组任务、轨迹和初始条件比较默认 VLA、量化推理与 πR²。核心观测是目标在等待模型期间移动了多少，以及返回动作能否完成抓取和放置。当前里程碑固定源码、完整任务清单和默认 GR00T RPC 适配器；动态闭环、速度—成功率实验与代表性模型视频按后续阶段验收。

## 基准与模型

动态基准覆盖四套 LIBERO、40 个任务；每任务3个轨迹家族、5条路径、3档速度（0.03/0.06/0.09 m/s），合计1800个条件。Spatial 单套为450个条件。物体通过载体接触摩擦运动，保留抓取后脱离载体的物理行为。环境配置与资源不进入本仓库。

模型基线使用本项目已经保存并使用过的 NVIDIA GR00T-N1.7-LIBERO 对应套件任务权重：默认完整 VLM + Flow4，预测长度40、原生解码16、执行前缀8。πR² 使用独立的已有适配权重。服务指定本地checkpoint并设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`；不下载新的VLA权重。当前阶段不使用论文的数据生成补偿器或微调方法，也不把低比特模型预设为更快或更差；静态质量与动态任务成功率分别实测。

## 准备与清单

```bash
python tools/prepare_sources.py --component libero-dynamic
python -m coexecution.libero_dynamic inventory \
  --upstream third_party/libero_dynamic \
  --output .local/libero-dynamic/conditions.json
```

源码固定在 `1e5845720643da281c47bf4423ae99e115976a13`。清单验证上游评审配置哈希，保存任务、轨迹、速度和来源；它只读取配置，不执行模型或仿真。仿真使用独立 Python3.10 环境、robosuite1.4.0、MuJoCo2.3.7及EGL，模型继续使用现有 CUDA 环境。依赖见 [requirements-simulation.txt](requirements-simulation.txt)，其中 termcolor 显式补齐 robosuite 的日志依赖。PyTorch 使用适合目标驱动的 CUDA 构建，模型环境与仿真环境分开。初始化、接触和位姿校验通过后再开展模型评测。

## 必须显式区分的时间与静态对照

论文附录将延迟量化为 `ceil(latency ×20)` 个控制周期；当前发布代码的正式同步入口则按精确延迟秒数推进 MuJoCo 子步，并锁住机器人关节。60ms 在两者中分别对应100ms和60ms，实验身份须注明选择。默认执行预算600步只统计控制动作，同步等待另计物理时间。

发布代码中的部分轨迹函数会把 `alpha_mps=0` 回退为0.055m/s，因此不能用改零速度宣称静态对照。静态控制需要明确停用或冻结载体运动，并核对目标实际位移。动态的40步初始相位必须真实推进场景；推理延迟期间也保留目标运动。

后续记录同时保留仿真时间、墙钟时间、校准延迟、实际目标位姿、动作来源、成功率和空队列事件。视频覆盖等待期间的移动过程；默认 VLA、量化与 πR² 的对照使用明确一致的评测条件，不从固定输入误差推断动态任务排名。

## 默认 GR00T 策略适配器

`GrootChunkPolicy` 使用本项目的 `synchronous_plan` 服务入口，执行完整新鲜 VLM 和四次 DiT，再取默认8步前缀。图像和状态转换复用 NVIDIA 官方封装；模型输出完整长度单独保留，夹爪0/1约定只在送入原生LIBERO前转换一次。接口与宿主合同测试已实现，真实动态模型闭环结果单独报告。

```python
from coexecution.libero_dynamic.policy import GrootChunkPolicy

policy = GrootChunkPolicy(port=5570, chunk_steps=8, seed=0)
# These calls run on the same inference thread:
actions = policy.get_action_chunk(raw_observation, task_language)
policy.close()
```

运行时 `PYTHONPATH` 需包含本仓库、`scripts/` 和固定 GR00T 源码。连接在第一次推理的线程创建，后续使用和关闭保持在该线程；reset可以由控制线程请求，在下一次推理前执行一次。上游控制器退出前，先排空请求，再通过其executor执行 `policy.close()`，最后关闭executor。`policy.records` 在内存保存完整模型输出、执行前缀和RPC/服务端阶段，不在模型计时路径写盘。

观测可携带 `__pir2_capture` 的物理步、仿真时间和实际capture时间；缺少时明确标为请求序号/策略入口时间，不能伪装成物理观测年龄。环境接入必须在上游观测快照阶段保留这些字段。
