# Leap 状态策略：数据收集、训练与评测

本模块用公开 MuJoCo Playground 专家采集独立示范，训练并比较 Flow、训练期 RTC 和 πR² 状态策略。它提供一条可追踪的“专家 → 数据 → 学生策略”实验路径；GR00T 的真实 VLM 闭环主线见 [LIBERO 指南](../docs/libero.md)。

## 任务与模型配置

| 项目                    | 配置                               |
| ----------------------- | ---------------------------------- |
| 控制频率与 episode 上限 | 50 Hz，600 步                      |
| 观测                    | 41 维状态，连续两帧                |
| 动作                    | 16 维，预测长度 H=16               |
| 方法                    | Flow、训练期 RTC、πR²              |
| 数据来源                | 公开 Playground 专家，本地重新采集 |

物理推进使用长度为 1 的 JAX scan 保持已检查的仿真边界。已执行动作先裁剪，再重新归一化后进入前缀条件。这里的状态策略与独立采集数据有自己的实验身份；论文原始四专家数据集和三种子训练不在本模块资产中。

<a id="setup-and-execution"></a>

## 运行流程

按[环境指南](../docs/environment.md)准备独立 `.venv-sim`，使用 CUDA JAX 0.5.3、PyTorch 2.7.1、MuJoCo/MJX 3.14.0、ONNX、Flax、Optax 和 NumPy。以下命令从仓库根目录执行，卡号替换为已检查为空闲的 GPU：

```bash
python3 tools/prepare_sources.py --component playground
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv-sim/bin/python -m simulation.collect --episodes 200 --seed 0 --save-data \
  --output outputs/simulation/demonstrations
CUDA_VISIBLE_DEVICES=0 .venv-sim/bin/python -m simulation.train \
  --method pir2 --seed 0 --epochs 800 --output outputs/simulation/pir2-seed0
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv-sim/bin/python -m simulation.evaluate \
  --checkpoint outputs/simulation/pir2-seed0/epoch-0800.pt \
  --output artifacts/simulation/pir2-seed0-evaluation.json
```

训练续跑使用 `--resume` 指向 `latest.pt`。[scripts/run_leap_seed.sh](../scripts/run_leap_seed.sh)提供分阶段日志、每种子锁和断点续跑；通过 `PIR2_GPU`、`PIR2_SEED` 指定运行资源与种子。

## 结果与检查

`simulation.test_gpu_simulation` 和 `simulation.test_policy_gpu` 是显式 CUDA 检查入口。评测记录环境版本、专家与数据身份、学生 checkpoint 和共用评测种子，分别报告专家与学生成功率。示范、权重、视频和原始结果保存在本地忽略目录，主机 CI 检查范围见 [CONTRIBUTING.md](../CONTRIBUTING.md)。
