# 源码关系与版本追踪

本仓库把作者的 πR²/Flow 实现、NVIDIA 任务权重和本地量化/调度工具连接为 LIBERO 实验流程。算法来源、任务权重和实验工程分别追踪，便于复刻同一配置或替换其中一层。

## 各层来源

| 层次                     | 来源与固定版本                                                                                                                                                                   | 在本项目中的位置                                                |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| πR² 发布仓库             | [pi-r2-flow/pi-r2-flow](https://github.com/pi-r2-flow/pi-r2-flow)，`3af52ca400a6ec7d141416879aa531a6f62f697a`                                                                    | 本地 `upstream/`                                                |
| πR² / Flow 模型实现      | 作者 GR00T fork，`2a39d591a3af24c42bb1d1c9cd708a2dddac0600`                                                                                                                      | `upstream/learning/Isaac-GR00T`                                 |
| LIBERO 任务权重          | [NVIDIA GR00T-N1.7-LIBERO](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO)，版本见[模型清单](../configs/libero/model-family.json)                                               | 本地 `models/`                                                  |
| 整数及 Thor 浮点算子参考 | [The Speedup Paradox](https://github.com/thu-ee-acts-lab/robotics-the-speedup-paradox/tree/239b4a3ef2268048571c9f508ad5700f98398c2f)，`239b4a3ef2268048571c9f508ad5700f98398c2f` | [仓库内算子](../coexecution/kernels/)及其修改                   |
| CUTLASS 通用依赖         | [NVIDIA/CUTLASS](https://github.com/NVIDIA/cutlass)，默认 `982748aa7356fa838c2ea4994ddcb0b2a4b4cefa`                                                                             | Git submodule 或 `CUTLASS_ROOT`                                 |
| 实验工程                 | 本仓库维护                                                                                                                                                                       | LIBERO 适配、模型服务、融合与量化适配、时序评测、回放及修正补丁 |

`Ther-nullptr/piR2` 是独立维护的实验仓库，与作者仓库之间通过固定源码和补丁建立关系；GitHub 上没有 fork 关系。作者发布面向 xArm6 + XHand 部署及 GR00T 训练，本仓库的 LIBERO 适配和 Leap 数据重建使用各自的任务资产与评测协议。

<a id="low-bit-backend-reference--低比特量化参考实现"></a>

## 修改放在哪里

- **模型修正**：以 [GR00T 补丁](../patches/groot-reproduction-fixes.patch)追踪，应用于固定作者版本。
- **CUDA 算子**：直接维护在 [coexecution/kernels](../coexecution/kernels/)。[源码清单](../coexecution/kernels/SOURCE_MANIFEST.json)记录参考版本、原文件与仓库修改。
- **模型适配与 Triton producer**：维护在 [coexecution/optimization](../coexecution/optimization/)，负责算子选择、张量契约、图捕获和模型接入。
- **依赖与许可**：[sources.lock.json](../sources.lock.json)记录源码固定版本；[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)及随附许可保留各组件条款。

例如，FP4 的两种带 bias、M=41 的 Swap-AB 配置已进入仓库内 CUDA 实现；`--fp4-swap-ab` 与 `floating_gemm=swap_ab` 选择该分支，其他形状按原有路径处理。更新实现后重编译对应 backend 即可，使用方无需另行应用外部仓库补丁。

## 准备源码与复现实验

```bash
python3 tools/prepare_sources.py --component groot
python3 tools/prepare_sources.py --component robotics-kernels
```

第一条准备固定模型源码和本地补丁。第二条核验仓库内算子文件，并记录实际选择的 CUTLASS；运行前需按[环境指南](environment.md#cutlass-submodule)准备默认 submodule 或兼容头文件安装。核验过程保留现有源码和依赖选择。

复现实验时，记录源码/补丁哈希、checkpoint、输入或数据集、执行配置、实际 CUTLASS 版本和硬件环境。整数 W8A8/W4A4 与 Thor FP8/FP4 使用各自的数据格式和适配契约；可用接口及验证步骤见[算子参考](../coexecution/README.md)和[平台交接](platform-handoff.md)。任务质量由对应 checkpoint 的 LIBERO 闭环结果给出，算子参考项目和辅助回放保留各自的结果归属。
