# 环境配置与算子构建

模型、仿真和算子构建使用明确的环境边界，便于复刻相同输入、精度与执行路径。先准备 GR00T CUDA 环境和 LIBERO 仿真环境，再按实验所需构建整数或 Thor 浮点 backend。

## 环境选择

| 用途                               | 本地目录约定       | 已记录的环境                                                                                                    |
| ---------------------------------- | ------------------ | --------------------------------------------------------------------------------------------------------------- |
| GR00T 训练、Ada 模型回放与闭环服务 | `.venv/`           | Python 3.10，PyTorch `2.7.1+cu126`，编译 toolkit CUDA 12.8，Transformers 4.57.3，Diffusers 0.35.1，NumPy 1.26.4 |
| LIBERO 仿真客户端                  | `.venv-pi05/`      | Python 3.12，hf-libero 0.1.4，robosuite 1.4.0，MuJoCo 3.8.1；另需 NumPy、PyAV、Pillow、pyzmq、msgpack           |
| Leap 状态策略与仿真                | `.venv-sim/`       | CUDA JAX 0.5.3，PyTorch 2.7.1，MuJoCo/MJX 3.14.0，ONNX、Flax、Optax、NumPy                                      |
| Thor 浮点推理                      | 独立环境和构建缓存 | 见 [Thor 配置](#optional-thor-floating-inference)                                                               |

这些版本记录已使用过的组合；复刻时同时保存实际依赖清单、GPU 架构、Torch CUDA runtime 与 `nvcc` 版本。当前 Ada 回归使用表中 Torch 2.7.1；早期算子开发记录的 `2.9.0+cu128` 是另一套环境。`.venv-pi05` 是历史目录名，当前用途是 LIBERO 仿真。

模型计算使用 CUDA。启动前检查占用并显式指定 `CUDA_VISIBLE_DEVICES`。环境、权重、数据、构建缓存和 `.libero-config-pi05/` 留在本地；主机相关 EGL、libexpat 设置通过本地环境配置。

## 模型源码、任务权重与配置

```bash
python3 tools/prepare_sources.py --component groot
```

按固定 GR00T fork 的依赖说明安装模型环境，再按 [LIBERO 指南](libero.md)准备任务权重和仿真资产。源码准备工具负责固定源码与补丁；模型访问授权和 Hugging Face 登录在本机完成。

Spatial 入口从完整 GR00T 任务 checkpoint 加载骨干权重，同时从配置构造 Qwen3-VL。离线服务需要可用的骨干配置与 processor 资产，无需额外保存一份骨干基座权重。scoped loader 保留请求的 dtype 和 attention 实现，严格检查缺失、多余与形状不匹配的权重；训练适配器显式初始化的新 streaming 参数单独处理。该加载路径的输入是完整任务 checkpoint。

Leap 使用另一份固定依赖：

```bash
python3 tools/prepare_sources.py --component playground
```

## CUTLASS submodule

CUDA 算子实现维护在 `coexecution/kernels`，通用头文件由官方 [NVIDIA/CUTLASS](https://github.com/NVIDIA/cutlass) 提供。以下两种配置选择其一。

**使用仓库默认版本：**

```bash
git submodule update --init --recursive coexecution/kernels/vendor/cutlass
```

Git link 默认记录 `982748aa7356fa838c2ea4994ddcb0b2a4b4cefa`。新克隆也可以使用 `git clone --recurse-submodules`。切换分支后，若要使用该分支记录的依赖，再运行上述 update；先保留有意选择的本地版本与未提交修改。

**复用兼容安装：**

```bash
export CUTLASS_ROOT=/path/to/cutlass
```

`CUTLASS_ROOT` 优先于 submodule，目录需具有完整的 `include/cutlass`、`include/cute` 和 `tools/util/include/`。兼容性由算子使用的 API 和 CUDA/C++ 工具链决定，SHA 可以与默认版本不同。纯头文件安装也可使用；工具记录实际来源，并在有 Git 元数据时记录 revision。取消该变量即可返回本地 submodule。

若希望用已有独立 Git clone 加速默认 submodule 的首次下载：

```bash
git submodule update --init --recursive \
  --reference /path/to/existing/cutlass --dissociate \
  coexecution/kernels/vendor/cutlass
```

`--reference` 复用已有 Git objects，`--dissociate` 将所需 objects 留在新克隆中；缺失部分从配置的官方远端获取。直接复用头文件时使用 `CUTLASS_ROOT` 即可。

个人版本选择记录在实验身份中。修改项目共享默认版本时，同步更新 Git link、`sources.lock.json` 和算子 manifest，遵循 [CONTRIBUTING.md](../CONTRIBUTING.md)。保留所选 CUTLASS 安装的许可与文件声明；常规默认初始化使用固定 Git link，无需 `--remote`。

## Optional GR00T fusion and integer inference

整数 W8A8/W4A4 和融合 backend 在当前模型环境内构建。需要 `nvcc`、兼容 C++17 编译器、Ninja 和匹配 Torch 的 Triton；编译与首次 JIT 放在计时和图捕获之前。

先准备 CUTLASS，再核验源码、查看构建输入：

```bash
python tools/prepare_sources.py --component robotics-kernels
export PYTHONPATH="$PWD:$PWD/upstream/learning/Isaac-GR00T${PYTHONPATH:+:$PYTHONPATH}"
python -m coexecution.kernels.build --help
python -m coexecution.kernels.build --backend integer --describe
```

准备命令核验仓库内算子并记录实际 CUTLASS。metadata 入口不导入 Torch/Triton，可在没有 GPU 的环境读取。运行时直接导入 `coexecution.kernels`，算子代码和构建均来自本仓库。

在目标 CUDA 模型环境构建所需 backend：

```bash
export TORCH_EXTENSIONS_DIR="$PWD/.local/torch-extensions"
python -m coexecution.kernels.build --backend integer
python -m coexecution.kernels.build --backend integer_dispatch
python -m coexecution.kernels.build --backend bf16_fusion
```

各实现的准入条件如下，具体选择见[算子 inventory 与 profile](../coexecution/README.md#owned-operators-and-explicit-profiles)：

| 实现                             | 适用条件                                                                                            |
| -------------------------------- | --------------------------------------------------------------------------------------------------- |
| 通用整数 GEMM                    | 使用 SM80 兼容整数 kernel；当前 GR00T 接入记录在 Ada，其他平台按[迁移流程](platform-handoff.md)验证 |
| native Ada 候选                  | SM89；shape/layout guard 不满足时回到参考路径                                                       |
| native LayerNorm packing         | SM89，PyTorch 2.7.1                                                                                 |
| native RMS statistics            | SM89，`2.7.1+cu126`，记录的 Torch Git revision 和精确 reduction header 哈希                         |
| Flow / 串行 πR² 适配             | 显式选择优化；可用纯 DiT CUDA Graph                                                                 |
| paired worker                    | 两份独立 policy，捕获阶段协调；whole-action graph/KV reuse 不支持                                   |
| πR² streaming condition grouping | 当前适配不支持                                                                                      |

构建成功、张量契约、模型数值检查和闭环任务质量是不同验证阶段；操作命令见[算子参考](../coexecution/README.md)，主机 CI 范围见 [CONTRIBUTING.md](../CONTRIBUTING.md)。许可边界见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。

## Optional Thor floating inference

FP8/FP4 backend 面向 Jetson AGX Thor（SM110，aarch64）。此前 Thor 环境记录如下，迁移当前实现时按[平台交接](platform-handoff.md)逐层检查：

| 层次                    | 已记录配置                                                    |
| ----------------------- | ------------------------------------------------------------- |
| 系统与 toolkit          | JetPack 7.2 / L4T 39.2，CUDA toolkit 13.2                     |
| Python / Torch / Triton | Python 3.12，PyTorch `2.10.0+cu130`，Triton 3.6.0             |
| 模型依赖                | Transformers 4.57.3，Diffusers 0.35.1，SDPA attention         |
| 数据依赖                | NumPy 1.26.4，PyArrow 20.0.0，tifffile 2024.9.20，PyAV 16.1.0 |

使用独立 checkout、环境和缓存。正常用户需有平台 GPU 设备节点的访问权限，包括实测机器上的 `/dev/nvmap`；这由机器的运行环境提供。具体 FP8/FP4 服务、测量、ActionGraph 和回放接口支持见[精度能力表](platform-handoff.md#precision-support)。

准备 CUTLASS 后，在兼容 CUDA toolkit 下构建：

```bash
python tools/prepare_sources.py --component robotics-kernels
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}"
export TORCH_EXTENSIONS_DIR="$PWD/.local/torch-extensions"
export PYTHONPATH="$PWD:$PWD/upstream/learning/Isaac-GR00T${PYTHONPATH:+:$PYTHONPATH}"
python -m coexecution.kernels.build --backend fp8
python -m coexecution.kernels.build --backend fp4
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas"
export TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
```

构建目标为 `sm_110a`；`require_fp8_backend` / `require_fp4_backend` 在按需构建前检查 SM110。Triton 3.6 的已记录安装自带 assembler 无法识别 `sm_110a`，以上两个变量使普通与 Blackwell 路径均使用兼容 toolkit 的 `ptxas`，无需修改 vendor 文件。

`PIR2_FP8_DECODE_POLICY`、`PIR2_FP4_DECODE_POLICY` 默认 `auto`；`PIR2_FP8_SWAP_AB=1` 为默认设置。GR00T 投影 tactic 由适配器显式选择。可选 `--fp4-swap-ab` 位于仓库内 `blackwell/csrc/fp4_cutlass_linear.cu`，针对两种带 bias 的 M=41 形状；其他形状保留原 FP4 路径。源码更新后重新构建 `fp4` backend。

具体格式、参数与 GPU 检查见 [Thor 算子参考](../coexecution/README.md#experimental-thor-fp8-and-fp4)。冷启动、编译、预热与 capture 单独记录，稳态推理使用完成这些步骤后的测量。
