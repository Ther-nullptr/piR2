# LIBERO-Spatial：从现成权重到真实 VLM 闭环

从官方 Spatial 任务权重即可建立 Flow 基线。研究 πR² 时，再用同一任务数据适配动作头，并在匹配初始条件下比较时序与控制质量。评测始终把真实仿真图像送入 VLM；训练时冻结视觉和语言骨干，更新动作头与相应投影模块。

参考入口：[NVIDIA 任务权重](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO) · [NVIDIA LIBERO 示例](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/LIBERO/README.md) · [模型清单](../configs/libero/model-family.json)。当前流程覆盖 Spatial 全部 10 个任务、432 条示范和 52,970 帧；Object、Goal、Long 的接入目标单独记录。

<a id="assets-and-environment"></a>

## 1. 准备任务资产

按[环境指南](environment.md)建立模型环境 `.venv` 和仿真环境 `.venv-pi05`，在本机完成模型访问授权与 Hugging Face 登录，然后从仓库根目录运行：

```bash
python3 tools/prepare_sources.py --component groot
.venv/bin/python scripts/fetch_groot_spatial.py
.venv-pi05/bin/python tools/prepare_libero_config.py --assets ~/.cache/libero/assets
```

下载器固定模型和数据版本，准备 Spatial 推理权重、全部 Spatial 示范与官方 modality 配置；清单写入本地 `artifacts/libero-pir2/assets-manifest.json`。如果本机已经具有匹配的完整任务权重，直接在后续 plan 中指向该目录。

完整任务 checkpoint 提供骨干权重；离线运行还需要本地可用的骨干配置和 processor 资产。仿真配置工具使用已下载的 LIBERO assets 创建独立配置，并保留现有配置。EGL 选择实际渲染 GPU；主机需要的 libexpat 修正可通过 `PIR2_LIBERO_LD_PRELOAD` 或 plan 的 `simulation_preload` 设置。

## 2. 按研究问题选择入口

| 目标                            | 入口                                                                                       | 输出                                        |
| ------------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------- |
| 建立普通任务基线                | 官方示例的 `LIBERO_PANDA`、`--use-sim-policy-wrapper`，示例执行前缀为 `--n-action-steps 8` | 官方示例协议下的任务表现                    |
| 测量量化、融合与系统优化        | [性能实验指南](coexecution.md)，固定真实 Spatial 观测                                      | 模型 wall 延时、分段和 kernel profile       |
| 比较 πR² 与 Flow 的延迟适应能力 | [algorithm 协议](timing-protocols.md)：固定观测/动作延迟                                   | 同一模拟延迟下的任务表现                    |
| 比较部署、异步与动作接纳策略    | [deployment 协议](timing-protocols.md)及下述 JSON plan                                     | 成功率、实际延时、执行 AoI、deadline 和视频 |

普通基线、固定延迟评测和墙钟部署各自使用明确的时间语义，结果按协议分别汇总。任务 checkpoint 的原生解码长度、训练 horizon 和每轮执行前缀 K 分别记录；改变 K 的实验见[时序协议](timing-protocols.md)。

## Fixed inference image resolution / 固定推理图像分辨率

**中文：** 在原有服务命令后添加 `--image-resolution 192`（或 `128`），
即可保留 checkpoint 的原有裁剪/视野，再对双相机图像做 `INTER_AREA`
缩放。未指定参数时完全沿用 checkpoint；这不是默认 256。
该选项会改变视觉信息及可能的任务质量，与数值优化配置独立。

**English:** Append `--image-resolution 192` or `128` to an existing
`python -m coexecution.libero.service` command (the compatibility
`scripts/serve_libero_protocol.py` entrypoint also accepts it). Both cameras keep
the checkpoint's complete eval crop/transform, followed by an area resize.
All Qwen pixel consumers, including the collators, receive the matching pixel
budget so they cannot silently upscale the image again. Omitting the argument
leaves the checkpoint preprocessing untouched. Explicit sizes must be positive,
no larger than the square checkpoint target, and divisible by the actual Qwen
`patch_size * merge_size`. The current override requires eval-mode
albumentations preprocessing; unsupported pipelines fail instead of falling back.

An experiment plan can declare `image_resolution` globally or per condition;
a condition's explicit `null` restores checkpoint behavior for that condition.
For example, a condition in an otherwise complete plan can contain:

```json
{
  "name": "bf16-r192",
  "checkpoint": "models/spatial-checkpoint",
  "variant": "pir2",
  "image_resolution": 192,
  "flags": ["--input-preparation", "--static-vlm", "--vlm-cuda-graph"]
}
```

The runner passes the same setting to both independently owned policies in
split or paired layouts. The evaluator records it in each server's stable
identity, binding resumes to that configuration. Vision response metadata also
records the observed `actual_grid_thw`, visual tokens per image and feature
tensor shapes. Do not equate the requested size with the observed token count.

The reusable API is separate from `OptimizationConfig`:

```python
from coexecution.libero.preprocessing import ImageResolutionConfig, ImageResolutionScope
from coexecution.optimization.session import GrootOptimizations

with ImageResolutionScope(policy, ImageResolutionConfig(size=192)):
    with GrootOptimizations(policy, optimization_config):
        # Collate, prepare inputs, warm every shape, then run serial inference.
        ...
```

The resolution scope must enclose input preparation and optimization/Graph
scopes. Exit quiescently after those scopes close; the original processors and
eval transform are restored, including on setup failure. Live resolution
switching and nested ownership are unsupported. Training transforms, model
weights and checkpoint metadata are not modified. Lower-resolution features
can also reduce DiT attention work, so measure the matching solo DiT baseline
before attributing a concurrent improvement entirely to reduced contention.

`coexecution.libero.measurement`, the streaming contention `worker` and
`campaign prepare`, and the Flow contention `flow` command accept the same
`--image-resolution` option. They apply it before preparing resident inputs and
keep the actual-grid evidence in their results; an already prepared tensor
cannot be resized by changing a later service argument.

Lightweight configuration/lifecycle checks run with
`python -m pytest -q tests/contracts/test_image_resolution.py`.
The explicit pinned-processor/tiny-CUDA gate is
`PIR2_GPU_TESTS=1 python -m pytest -q tests/gpu/test_image_resolution_gpu.py`.
It checks fast/slow Qwen processors, reduced token/feature lengths, Flow4 and
streaming bootstrap/one-step updates, and eager/Graph agreement at each size.
These contracts do not establish trained-policy task success or a speedup.

<a id="evaluation"></a>

## 3. 用 plan 运行闭环对照

`coexecution.libero.experiments` 管理自己启动的服务、端口、GPU 准入和评测进程。推荐在本地忽略目录保存 plan，并为每份配置使用新的输出目录：

```bash
python -m coexecution.libero.experiments \
  --plan .local/experiments/spatial.json \
  --output artifacts/libero-protocols/spatial-comparison
```

下面是双精度 deployment plan 的结构示例。把 `/path/to/piR2` 替换为准备好环境和资产的实验目录，GPU 编号改为实际可用的物理卡；本例顺序运行两个条件，默认覆盖全部 10 个任务：

```json
{
  "runtime_root": "/path/to/piR2",
  "model_python": "/path/to/piR2/.venv/bin/python",
  "evaluation_python": "/path/to/piR2/.venv-pi05/bin/python",
  "upstream": "/path/to/piR2/upstream/learning/Isaac-GR00T",
  "libero_config": "/path/to/piR2/.libero-config-pi05",
  "gpus": { "action": 0, "vlm": 1, "simulation": 2 },
  "ports": { "action": 5570, "vlm": 5572 },
  "protocols": ["deployment"],
  "episodes_per_task": 20,
  "max_steps": 720,
  "seed": 1000,
  "conditions": [
    {
      "name": "flow-bf16",
      "checkpoint": "/path/to/piR2/models/GR00T-N1.7-LIBERO/libero_spatial",
      "variant": "flow",
      "flags": ["--inference-precision", "bf16"]
    },
    {
      "name": "flow-w8a8",
      "checkpoint": "/path/to/piR2/models/GR00T-N1.7-LIBERO/libero_spatial",
      "variant": "flow",
      "flags": ["--inference-precision", "w8a8"]
    }
  ]
}
```

量化条件先按[整数 backend 配置](environment.md#optional-gr00t-fusion-and-integer-inference)构建。上例用于匹配基本精度配置；融合、图捕获和输入准备等系统优化应在两个条件中明确配对，配置参考[性能指南](coexecution.md)。

常用扩展字段如下；调度、传输等支持逐 condition 覆盖的字段以该 condition 为准：

| 字段                                                                                     | 用途                                                                      |
| ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `task_ids`                                                                               | 选择任务子集，如 `[0, 3]`；输出标记 `selected_task_subset`                |
| `record_replay`、`replay_max_frames`                                                     | deployment 记录机器人状态和双相机，用于[前端回放](../simulator/README.md) |
| `execution_coupling`、`execution_clock`、`execution_prefix_steps`、`async_overlap_steps` | 选择同步/缓存异步/整策略异步及 K、D                                       |
| `deployment_scheduling_policy`                                                           | πR² 缓存异步的 conservative / responsive 接纳策略                         |
| `server_layout`、`feature_transport`、`feature_rpc`、`vision_max_hz`                     | 配置 worker 布局、特征传输和 S2 准入                                      |
| `gpu_uuids`                                                                              | 绑定 action / vlm / simulation 三角色的完整 GPU UUID，并核对 index 映射   |

runner 在启动前检查配置和 checkpoint 元数据，每个条件准入空闲设备，保留调用日志，并仅清理自己启动的进程。跨条件核对 fixture 模型、settled state 和初始 RGB 哈希。缓存异步的 paced 部署基线以 20 Hz 控制；其他 coupling/clock 的频率含义见[时序协议](timing-protocols.md)。

<a id="training"></a>

## 4. 按需进行 πR² 适配训练

现成任务权重用于 Flow 推理；πR² 的滚动控制与延迟条件通过适配训练接入。匹配训练采用相同数据、seed 1000、有效 batch 64、H40 和 10,000 次更新，单卡即可运行。以下卡号替换为空闲 GPU，两种方法顺序执行：

```bash
export PYTHONPATH="$PWD/upstream/learning/Isaac-GR00T"
export GROOT_HF_LOCAL_FIRST=1 GROOT_PATCH_MISTRAL=1 HF_HUB_OFFLINE=1
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train_groot_spatial.py \
  --variant pir2 --steps 10000
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train_groot_spatial.py \
  --variant flow --steps 10000
```

`--microbatch` 默认 4，必须整除有效 batch 64；梯度累积保持总 batch。需要阶段检查时加 `--stop-at 500`，随后去掉该参数继续同一 10k schedule。checkpoint 保存 optimizer、scheduler 和 RNG；上游 trainer 从恢复的 global step 重新设置数据顺序，配对训练应使用相同恢复边界，记录续跑方式。

πR² 配方包含 0–5 步随机图像延迟和 delay embedding；普通 Flow 对照没有图像延迟增强。因此这里比较的是完整训练配方。若研究滚动调度本身的因果贡献，需要另设匹配训练条件的消融实验。

<a id="runtime-modules--运行时模块"></a>

## 模块与兼容入口

| 模块                                                                                                        | 责任                                               |
| ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| [loading.py](../coexecution/libero/loading.py)                                                              | 完整任务 checkpoint 的骨干构造、配置和原生解码长度 |
| [backend.py](../coexecution/libero/backend.py)                                                              | 特征缓存、滚动缓冲、每个 worker 的私有 CUDA stream |
| [service.py](../coexecution/libero/service.py)                                                              | 严格权重加载、endpoint 注册与 worker 管理          |
| [measurement.py](../coexecution/libero/measurement.py) / [profiling.py](../coexecution/libero/profiling.py) | 匹配输入测量、GPU 活动去重与归因                   |
| [experiments.py](../coexecution/libero/experiments.py)                                                      | plan 校验、进程所有权、初始条件和执行年龄审计      |

每份 policy 串行提交；`split` 为默认布局，`paired` 使用同进程两份独立 policy 和共享冷捕获协调。Flow 的布局对照见性能指南；`cuda_local` 传输要求 paired Flow 同卡运行。配套约束见[传输与布局](timing-protocols.md)。旧脚本 `libero_model_loading.py`、`libero_inference_backend.py` 和 `serve_libero_protocol.py` 保留导入/命令行兼容性，评测身份同时记录兼容入口和核心实现哈希。

历史固定拓扑入口 `scripts/run_libero_protocols.py --step 500` 保留 GPU0 action、GPU2 VLM、GPU3 rendering 的四条件顺序评测，每条件 10 任务 × 20 episodes。新实验优先使用可声明资源的 plan。`run_libero_protocol_pipeline.py` 保留阶段续训管理，要求已有 `protocol-v4-integration` 和 `check_libero_timing_gpu.py` 检查记录；`--adopt-step500-client` 用于停止旧 supervisor 后转接已知 client。同一输出只运行一个 supervisor，续训保留 step500 归档。

环境、数据、权重和实验结果留在本地；源码检查与贡献流程见 [CONTRIBUTING.md](../CONTRIBUTING.md)。SO100 演示与 Leap 重建另有[辅助实验说明](../README.md#算法来源与辅助实验)。
