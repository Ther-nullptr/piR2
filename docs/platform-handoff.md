# Jetson 与流式回放交接 / Platform handoff

本文交接两项后续工作：在 **Jetson AGX Orin** 与 **Jetson AGX Thor** 上复现并重新调优推理，以及基于真实测量构建多精度流式回放平台。当前 Ada 上的 CUDA 回归和官方 Spatial 模型配对回放是起点；本文没有新增 Jetson 实机结果或 LIBERO 闭环成功率结论。性能数值以对应提交、输入、设备和时钟组的本地测量记录为准。

主线使用官方 `nvidia/GR00T-N1.7-LIBERO/libero_spatial` 与真实 VLM，保持 B1、Flow4/NFE4、预测 H40 和原生解码 16 步。启动时核对 `predicted_horizon`、`decoded_horizon` 与实际配置，不能通过减少步数、跳过视觉或复用旧观测来替代系统优化。当前仅 Spatial 的十任务完整管线已经实现；Object、Goal、Long 尚不能据此宣称通过。训练后的 πR²、Leap、SO100 和浏览器机制模拟各有不同边界，见 [LIBERO](libero.md) 与 [时序协议](timing-protocols.md)。

## 入口与可追溯身份

| 工作 | 现有入口 / 模块 | 交接边界 |
|---|---|---|
| 源码和权重 | [sources.lock.json](../sources.lock.json)、[fetch_groot_spatial.py](../scripts/fetch_groot_spatial.py)、[环境说明](environment.md) | 固定作者 fork、被忽略的 GR00T 源码 checkout 与补丁、官方任务权重和数据版本；本仓 Git 子模块仅 CUTLASS |
| 常驻输入推理 | [inference.py](../coexecution/libero/inference.py) 的 `PreparedInferenceEngine` | `vlm`、`action`、`full`；准备与解码在外部，调用者管理 CUDA stream 和输入生命周期 |
| 服务与测量 | `python -m coexecution.libero.service`、`python -m coexecution.libero.measurement` | 严格加载检查点；模型计算与模型数值验证仅在 CUDA 上执行 |
| 闭环实验 | [evaluate_libero_protocol.py](../scripts/evaluate_libero_protocol.py)、`python -m coexecution.libero.experiments` | 固定延迟算法与真实部署分开；实验计划复用现有服务和评估器 |
| 优化与算子 | [optimization](../coexecution/optimization)、[kernels](../coexecution/kernels) | 实例内安装/恢复、显式配置、实际后端与源码身份 |
| 测量回放 | [replay](../coexecution/replay/README.md)、[simulator](../simulator/README.md) | 导入保留证据、构建 v1 目录、桌面查看；不运行模型 |

从仓库根目录设置自己的本地路径。模型环境与仿真环境分别准备，不能把 x86_64 wheel、Ada 扩展 `.so` 或缓存复制到 aarch64 后视为可用：

~~~bash
export PIR2_MODEL_PY=/path/to/model-env/bin/python
export PIR2_EVAL_PY=/path/to/libero-env/bin/python
export PIR2_CHECKPOINT=/path/to/GR00T-N1.7-LIBERO/libero_spatial
export PIR2_DATASET=/path/to/groot-libero-spatial
export PIR2_UPSTREAM="$PWD/upstream/learning/Isaac-GR00T"
export PIR2_RUN_ROOT="$PWD/.local/platform-runs"
export PYTHONPATH="$PWD:$PIR2_UPSTREAM"
export TORCH_EXTENSIONS_DIR="$PWD/.local/torch-extensions"
export PIR2_HOST_GUARD_CACHE="$PWD/.local/host-guards"
~~~

首次设置按 [CUTLASS 说明](environment.md#cutlass-submodule) 初始化默认子模块；也可用 `CUTLASS_ROOT` 选择完整且兼容的头文件树，包括 `include/cutlass`、`include/cute` 与 `tools/util/include`。保留实际版本和许可证；不要为个人依赖选择改动公共 Git link。自有 CUDA 实现仍来自本仓：

~~~bash
python3 tools/prepare_sources.py --component groot
git submodule update --init --recursive coexecution/kernels/vendor/cutlass
python3 tools/prepare_sources.py --component robotics-kernels
"$PIR2_MODEL_PY" -m coexecution.kernels.build --backend integer --describe
~~~

使用外部兼容 `CUTLASS_ROOT` 时省略子模块更新命令。上述 `--describe` 是元数据检查；实际编译、设备加载和模型 gate 仍需在目标 CUDA 环境中分别完成。编译、权重打包、首次加载、预热和 Graph capture 均留在稳态计时之外。compiled 模式还需要 CPython、主机 C 编译器和记录的 Cython 3.1.2 / setuptools 84.0.0，详见 [公共推理路径](../coexecution/README.md#common-policy-runtime--公共推理路径)。

每个实验保存以下身份，不只保存命令行：

- 本仓提交及脏文件指纹、`sources.lock.json`、GR00T 上游提交和补丁哈希、实际 CUTLASS 来源、算子源码指纹与已加载二进制哈希。无法确认的二进制所有权保留 `unknown`。
- 检查点配置、权重分片哈希、处理器/统计量/模态配置；固定官方模型 revision `2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21`。Spatial 数据集为 `IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot`，固定 revision `bf14d6258218d12c2e3c1a3b9922e163cdf6455d`。
- 数据文件身份、真实观测的图像/状态/文本 SHA-256、episode/step、seed、attention 实现、精度覆盖、布局、Graph、融合、tactic/profile 哈希。
- Python、Torch 及其 CUDA runtime、`nvcc --version`、Triton、Transformers、Diffusers、驱动/JetPack/L4T、实际设备/SM 数、显存、功耗模式、CPU/GPU/EMC 时钟、温度和后台负载。

`measurement` 的 `results.json`、`mode_evidence`、`runtime-source` 和服务的 `identity.json` / `weight-loading.json` 提供其中部分证据；目标平台的工具链和遥测需另外保存在同一实验目录。它们都只留本地。CLI/元数据可在无 GPU 主机检查；CPU 神经网络 smoke test 不能代替 CUDA 加载、数值或质量验证。

## 平台与环境

| 平台 | 架构 | 当前证据 / 首要工作 |
|---|---|---|
| RTX 6000 Ada 参考机 | SM89 / Ada，x86_64 | 当前 CUDA 回归与官方 Spatial 配对模型回放；保留每个设备和时钟组的独立结果 |
| Jetson AGX Orin | SM87 / Ampere，aarch64 | 本交接未实机重跑；先建立目标环境与同设置 BF16，再验证 canonical INT8/INT4 |
| Jetson AGX Thor | SM110 / Blackwell，aarch64 | 仓库有既往 FP8/FP4 路径；本次 Ada 改动尚需在 Thor 上验证，另建平台浮点实验线 |

计算能力以 [NVIDIA CUDA GPU 表](https://developer.nvidia.com/cuda/gpus) 和 [Thor CUDA 指南](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_cuda.html) 为依据；Orin 的 Ampere 规格见 [官方产品页](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)。不要从“较少 SM 上占用率更高”推出绝对延迟更低。读取实际目标设备的 SM 数、可用内存和运行配置，再测模型完整延迟、部署 P50/P95、执行 AoI、功耗/每请求能耗与热稳定性。TOPS 比值和跨硬件推算都不是实测加速。

Jetson 的 CPU/GPU 共享 SoC LPDDR 与 EMC 带宽；不同内存类型仍有不同缓存、一致性与访问成本，见 [NVIDIA CUDA for Tegra](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#memory-management)。桌面端减少 host preparation、使用 pinned buffer 或重叠拷贝的收益必须在设备上重测，不能按独立显存/PCIe 模型外推。当前 split 的 action/VLM worker 可各自实例化完整 policy；[TransformerINT](../coexecution/optimization/groot/integer.py) 保留原 BF16 参数，并同时构造 8-bit/4-bit 权重包和分组权重包。还需计入 activation workspace、Graph pool、常驻输入、相机/录制与仿真缓冲。分别记录启动峰值、稳态系统 RAM 和 CUDA allocated/reserved；它们在共享内存设备上不可直接相加。位宽降低不等于总内存减少，也不能据此假定 32 GB 设备能容纳双 worker 配方。

当前 Ada CUDA 回归与官方模型配对使用 **Python 3.10、Torch 2.7.1+cu126、CUDA 12.8 编译工具链、Transformers 4.57.3、Diffusers 0.35.1**。历史算子实验的 Torch 2.9.0+cu128 环境是另一条记录。仓库此前记载的 Thor 环境为 JetPack 7.2 / L4T 39.2、CUDA toolkit 13.2、Torch 2.10.0+cu130、Python 3.12、Triton 3.6.0；它不表示本次重验通过。Torch wheel 所带 runtime 与实际扩展编译器版本必须分别记录。

平台安装依据目标设备的官方兼容矩阵选择。NVIDIA 的 [JetPack 7.2 发布说明](https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2) 列出 Orin/Thor、L4T 39.2 和 CUDA 13.2.1；不同 JetPack 版本的支持范围不同。选择匹配 JetPack、Python ABI 和 aarch64 的 [Jetson PyTorch 包](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)，核对 [JetPack 归档](https://developer.nvidia.com/embedded/jetpack-archive)，不把桌面环境版本当成升级指令。Thor 浮点扩展当前编译目标为 `sm_110a`；参考 [Thor 构建与 ptxas 配置](environment.md#optional-thor-floating-inference)。

在批准的设备执行环境中读取状态，并在整个测量窗口保留遥测：

~~~bash
nvpmodel -q --verbose
jetson_clocks --show
tegrastats --interval 1000
~~~

这些命令分别用于查看功耗模式、时钟和运行状态，依据 [NVIDIA 设备验证文档](https://docs.nvidia.com/jetson/archives/r38.4/DeveloperGuide/SD/TestPlanValidation.html#nvpmodel-power-modes)。权限不足时由设备管理员提供执行环境；本指南不要求修改共享功耗模式、锁频、以 root 跑模型或修改设备节点权限。先确认没有竞争性 GPU 工作，再测基线与候选。跨功耗、DVFS、温度、设备或运行窗口的结果分开成组，不能拼接相乘。

## 哪些可复用，哪些必须移植

输入字段准备、文本元数据缓存、图像数组准备、主机 guard 编译、VLM/动作 CUDA Graph 和调用内 K/V 复用是系统层候选。它们仍依赖固定模型/处理器/attention 契约，需在目标平台严格检查有限输出、形状、fresh-input、RNG、生命周期、调度语义与 Graph gate；数值差异按下述速度与质量准则评价。BF16 与量化路径使用相同适用的系统配置，才能分开系统收益与量化增量。

| 实现 | 代码中的约束 | Orin / Thor 处理 |
|---|---|---|
| canonical INT8/INT4 | [integer_gemm.cu](../coexecution/kernels/ampere_ada/csrc/integer_gemm.cu) 使用 SM80-compatible CUTLASS，历史集成验证为 Ada | 在目标工具链重新编译、检查实际整数 kernel、重测数值和完整路径；源码兼容不等于已有平台结果 |
| INT4 GEMV/small-M、split-K、residual/group writeback | [候选设备 guard](../coexecution/kernels/ampere_ada/_candidate_support.py)、[inventory](../coexecution/optimization/operators.py) 显式限定 SM89，另有 shape/stride/对齐约束 | 架构不匹配的 profile 会被拒绝；不合格调用保留同精度 reference。目标移植需新构建目标、指令/布局审查与设备测试 |
| native LayerNorm / RMS | [norm.py](../coexecution/kernels/ampere_ada/norm.py) 要求 Torch 2.7.1；[rms_statistics.py](../coexecution/kernels/ampere_ada/rms_statistics.py) 还绑定 2.7.1+cu126、Torch Git 与 reduction 头文件哈希 | ARM wheel 即使版本号相近也需核对实际归约实现；不能删除 guard 来“启用加速” |
| Thor FP8/FP4 | [Blackwell 源码](../coexecution/kernels/blackwell)、SM110 loader、独立 pack/scale 与 tactic 约定 | 只在匹配 Thor 环境中构建；Ada 的整数 profile 不适用，目标格式与调参单独验收 |

2026-10-14 核对时，[PR #27](https://github.com/Ther-nullptr/piR2/pull/27) 的 AGX INT8/INT4 整理仍为开放提案。其描述区分了 Orin 原型证据与尚待运行的集成验证；不能把该 PR 的接口、速度或质量记成当前主线/本交接的已验收能力。后续采用时先确认合并状态、实际提交和新的验证记录。

<a id="precision-support"></a>
## 精度支持与待补接口

| 路径 | BF16 / W8A8 / W4A4 | FP8 / FP4 |
|---|---|---|
| 服务配置、后端 | 已有显式选项；每个平台仍需 CUDA gate | 已有 Thor 专用服务/内核选项 |
| `--dit-cuda-graph` | 有；纯 DiT 范围 | 有；需各自验证 |
| `--action-cuda-graph` 与 `--action-kv-reuse` | 仅非流式 Flow4，H40；paired 服务拒绝 whole-action Graph | 当前配置拒绝，不能直接启用 |
| `measurement --precisions` | 仅 `bf16 w8a8 w4a4` | 尚未接入 |
| replay v1 导入与查看器 | `replay.schema.PRECISIONS` 仅三种精度 | 尚未形成完整支持 |

限制来自 [OptimizationConfig](../coexecution/optimization/config.py)、[measurement parser](../coexecution/libero/measurement.py) 和 [replay schema](../coexecution/replay/schema.py)，不是命名差异。W8A8/W4A4 是 INT8/INT4 整数点积、INT32 累加；当前 Thor FP8 为 E4M3 数据、每 tensor FP32 scale，FP4 为 E2M1 数据与 block-16 UE4M3 scale，二者 FP32 累加并输出 BF16。注意力等未覆盖组件仍是浮点；INT4 不等于 FP4，INT8 不等于 FP8。

安排两条实验线：第一条在后端通过验证的平台上比较同配置 BF16/W8A8/W4A4；第二条在 Thor 比较 BF16/FP8/FP4 的平台原生浮点实现。不能把浮点结果改标签导入整数 viewer。扩展浮点链路需要同时修改精度 CLI、runtime/gate、测量元数据、schema/importer、比较键、前端筛选/图例和测试；若支持混合精度，还需真实的逐组件/逐模块格式与覆盖记录。ActionGraph 和 K/V 复用的浮点支持必须增加相应实现和同精度输出/RNG/Graph 验证，不能只放宽 choices。每种精度应先与自身未改动的路径比较，再单独评价它相对 BF16 的量化误差和闭环质量。

### 速度与质量的推广准则

允许以适当精度损失换取速度。逐值一致用于诊断和保守对照，不是所有优化候选的硬性否决条件；声明逐值保持的路径仍需如实核验该合同。有数值差异的融合、近似或量化候选可以继续评估，但必须明确标记差异，不能把数值变化直接等同于任务质量下降，也不能由误差较小推断闭环质量已通过。

在相同输入与 seed 下，分别对照同精度 reference 和 BF16，报告一致归一化动作表示下的相对 L2、RMSE/最大绝对误差及分通道/预测步统计；记录归一化方式、分母和近零 reference 的处理。推广还需真实 VLM、配对 task/初始化/seed 的 LIBERO 闭环结果，连同失败、时序有效性、延迟/P95 和执行 AoI 一起权衡。可接受的任务损失范围由实验目标明确记录，本文不预设成功率阈值。闭环未完成的候选保持“质量待验证”，不因速度更快而自动升级为质量已验收。

有限值、输出键/形状、观测与特征新鲜度、RNG 消耗和状态、NFE/调度语义、资源生命周期以及计时边界仍是严格条件。现有 `measurement` 的逐值 gate 继续按其实际 reference 记录通过或失败；它提供执行诊断，不是上述速度与任务质量权衡的替代品。允许误差的评估必须另行保留误差政策和真实闭环证据，不能将原本失败的逐值检查改标为通过。

## K/V 复用与可执行测量

`--action-kv-reuse` 将**一次完整 Flow4 调用内、每层 cross-attention 的 K/V 投影从四次计算减到一次**。每个新观测的 Graph replay 都重新计算第一次 K/V；没有跨观测缓存。Q、self-attention、mask、状态/噪声、四次 NFE 和 RNG 消耗保持原契约。纯 `--dit-cuda-graph` 不覆盖完整 Flow4 生命周期，因此不足以启用该复用。源码与计数单位见 [kv_reuse.py](../coexecution/optimization/groot/kv_reuse.py)：投影次数不等于 grouped GEMM 数或 kernel launch 数，更不表示全模型获得 4× 加速。

在目标 CUDA 环境中先完成 build/warmup 与显式回归，再运行下例。它使用真实 Spatial 固定输入，配对 BF16 和整数路径的共同系统优化；未选择任何未经本设备测量的候选 profile：

~~~bash
"$PIR2_MODEL_PY" -m coexecution.libero.measurement \
  --checkpoint "$PIR2_CHECKPOINT" --dataset "$PIR2_DATASET" \
  --upstream "$PIR2_UPSTREAM" --output "$PIR2_RUN_ROOT/model-only" \
  --device cuda:0 --precisions bf16 w8a8 w4a4 \
  --modes compiled kv_reuse --action-cuda-graph \
  --operator-fusion --quantization-scope all --vision-channels-last \
  --measurement-boundary model-only \
  --observations 3 --rounds 20 --warmup 2 --seed 127 --replay-assets
~~~

此处 `compiled` 与 `kv_reuse` 只差调用内 K/V 复用；测量命令管理开关，因此**不要**再传全局 `--action-kv-reuse`。在独立服务中才直接传 `--action-cuda-graph --action-kv-reuse`。上例的两种 mode 均已包含输入准备、tokenizer cache、static VLM、VLM Graph、图像数组和编译 guard。跨精度比较保留相同 attention、布局、融合及 Graph；基线不能少用共同优化。

`model-only` 预先保存/准备一组 CUDA 输入，每次重新运行完整 VLM + Flow4。计时不含输入准备/拷贝、解码、RPC、仿真或写盘；该模式按 mode 分块，应另存反向 mode 顺序的实验检查漂移。`--measurement-boundary comparison` 则按轮随机 mode 顺序，分别测 `policy`（原始 CPU 观测至解码动作）、`preparation_transfer`、`prepared_model`。这些是独立且重叠的测量边界，不能相加/相减当作 CPU/GPU 分解。

`results.json`、`samples.jsonl`、`actions.npz`、数值/RNG gate 和 `graph-timing-audit.json` 保留原始证据。增加 `--profile-rounds 1` 会另做 instrumented 调用，保存真实 request/VLM/action 注解和 Chrome kernel 时间；它们不替代 clean 延迟。显式 `--cuda-events` 也属于 instrumentation。所有模型 gate 必须在 CUDA 上执行。

### 固定延迟算法与真实部署

先按服务接口分别启动 action 和 VLM worker；两者可以在明确声明的同一 GPU 上运行独立进程。例中 `PIR2_PRECISION` 从 `bf16`、`w8a8`、`w4a4` 选择；分别启动并等待 ready，再运行评估器。基线与候选的 worker 布局、特征传输和设备分配必须相同：

~~~bash
export PIR2_PRECISION=bf16
"$PIR2_MODEL_PY" -m coexecution.libero.service \
  --checkpoint "$PIR2_CHECKPOINT" --variant flow --role action \
  --device cuda:0 --port 5570 --output "$PIR2_RUN_ROOT/action-$PIR2_PRECISION" \
  --inference-precision "$PIR2_PRECISION" --quantization-scope all \
  --operator-fusion --action-cuda-graph --action-kv-reuse
~~~

另一个已设置相同路径与精度的终端启动 VLM：

~~~bash
"$PIR2_MODEL_PY" -m coexecution.libero.service \
  --checkpoint "$PIR2_CHECKPOINT" --variant flow --role vlm \
  --device cuda:0 --port 5572 --output "$PIR2_RUN_ROOT/vlm-$PIR2_PRECISION" \
  --inference-precision "$PIR2_PRECISION" --quantization-scope all \
  --operator-fusion --input-preparation --tokenizer-cache \
  --static-vlm --vlm-cuda-graph --image-array-preparation --compiled-host-guards
~~~

此服务示例是独立的部署配方，不等同于上面的整模型计时配置。为模型速度建立部署因果对照时，固定两个 worker 的全部配置，仅改变待比较的选项。完整十任务例子如下；仿真环境需先按 [LIBERO 环境](libero.md#assets-and-environment) 配好资源、配置和 EGL：

~~~bash
"$PIR2_EVAL_PY" scripts/evaluate_libero_protocol.py \
  --protocol algorithm --variant flow --port 5570 --slow-port 5572 \
  --visual-delay 3 --action-delay 1 --control-hz 20 \
  --episodes-per-task 20 --max-steps 720 --seed 1000 --no-video \
  --output "$PIR2_RUN_ROOT/algorithm-$PIR2_PRECISION"

"$PIR2_EVAL_PY" scripts/evaluate_libero_protocol.py \
  --protocol deployment --variant flow --port 5570 --slow-port 5572 \
  --server-layout split --execution-clock paced --control-hz 20 \
  --episodes-per-task 20 --max-steps 720 --seed 1000 --no-video \
  --output "$PIR2_RUN_ROOT/deployment-$PIR2_PRECISION"
~~~

算法协议固定图像年龄 3 tick/150 ms、动作释放 1 tick/50 ms，计算等待不推进物理；部署协议使用实际墙钟异步 VLM/动作服务。两者不合并分数。不同条件配对 task、seed、初始化及 fixture/settled-state/initial-RGB 哈希；更换训练后的 πR² 检查点时使用 `--variant pir2` 并移除不支持的 whole-action Graph/K/V 选项。

批量条件用现有 `python -m coexecution.libero.experiments --plan PLAN.json --output OUTPUT`，计划字段见 [运行时模块](libero.md#runtime-modules--运行时模块)。不为平台扫描另建一套脚本。若只是 `--task-ids 0 3 --episodes-per-task 1`，报告为子集诊断；所有失败和 timing-invalid episode 仍留在分母。每次更换配方使用新的输出目录。

## KDA 复现与目标调参

1. 在完整官方模型调用中采集真实模块名、执行次数、`[M,K,N,has_bias]`、stride、padding、dtype、量化格式与 backend/source identity。H40 的 DiT 行数可能包含状态 token；VLM 长度取决于真实图像和文本，不能由模型参数量推算形状。保留实际输入/权重，不用仅同形随机张量代替最终模型验证。
2. 先通过有限输出、形状、RNG、fresh-input、输出独立存储、异常恢复、冻结权重、调度语义和 Graph 无 recapture gate。记录同精度数值差异；允许误差的候选按动作误差与闭环结果继续评价，不因非逐值一致直接淘汰。随后单独测冷加载/编译/打包/capture 与暖稳态。延迟和吞吐测量使用干净 GPU；在 Jetson 记录 CPU/GPU/EMC、温度和功耗变化。
3. 对真实形状分别测裸 GEMM，以及在线 prepare/scale/pack + GEMM + epilogue 的完整 Linear；再测完整 VLM + Flow4 和部署路径。INT4/INT8 的准备成本、分组输出写回和额外 workspace 都必须计入实际边界。
4. 以匹配输入、seed、设置和时钟组做重复/交错或反向顺序测量。NCU 用独立短采样定位**实际 kernel** 的源代码、tile、occupancy、带宽、stall、寄存器和 shared-memory 限制；记录采样对象与 replay 方式。NCU per-kernel 指标与完整模型的 clean wall latency 分开，不能把诊断时长当作稳态延迟。
5. 在目标设备重新扫描 tile、warps、stages、split-K、grouping、layout 和工作区策略；核对它们在全模型调用次数中的贡献，再决定是否保留。Ada 候选的值只能作为起点。SM 数少可能提高占用比例，也可能降低并行能力或增加带宽/热限制，最终由完整延迟和质量判断。
6. 仅显式加载已测配置：`--operator-profile FILE` 选择实现，`--quantization-tactics FILE` / `--quantization-group-tactics FILE` / `--quantization-shape-tactics FILE` 选择整数 tactic。新 profile 绑定架构、精度、精确 shape、backend revision、量化格式及测量组；legacy shape row list 是 **unbound**，不可当成可追溯最优解。改变权重/精度/tactic 后关闭并重建 scope 和 Graph。

配置 schema、可用参数和源文件见 [operator inventory](../coexecution/README.md#owned-operators-and-explicit-profiles)。例如 canonical tactic 0–7 与 small-M tactic 0–5、split-K slices 2/4/8 是不同参数空间。历史 `groot-libero10-ada` 配置面向 Long/Flow，不能直接充当 Spatial 或 Jetson 校准结果。这里不指定跨平台“最佳 profile”；保留候选原始重复值、完整模型增益、同精度误差与严格语义检查后再选。量化与有数值差异的融合路径还需独立的 BF16 误差及真实 VLM 闭环评估，用实测任务质量决定精度与速度的取舍。

整模型 operator-profile case sweep 优先让每个 case 使用新进程。`service` / `measurement` 启动在 `load_core` 前设置 Torch interop 线程数；这是进程级初始化，不能在已经启动并行工作的解释器内反复运行整套启动/加载入口。另一种受控方式是单次加载 policy，串行排空 GPU 工作后正确关闭/重建优化 scope、Graph 与缓存，并重新检查每个 case 的有限输出/形状/RNG 及数值差异；不能仅反复调用 `load_core` 当作 case 隔离。启动与重新 capture 的成本仍单独记录。

### Ada 专用 W8 候选配方

以下显式候选使用 [site tactics](../configs/quantization/groot-libero-spatial-ada/site-tactics.json)、[group tactics](../configs/quantization/groot-libero-spatial-ada/group-tactics.json)、[W8 shape tactics](../configs/quantization/groot-libero-spatial-ada/w8-shape-tactics.json) 和 [split-K/writeback profile](../configs/quantization/groot-libero-spatial-ada/w8-splitk-writeback.json)。它只面向官方 Spatial、SM89 和 W8A8，完整配方的重复测量仍需按对应提交验收；文件名不代表全局最优、已通过闭环或已移植 Jetson。

~~~bash
"$PIR2_MODEL_PY" -m coexecution.libero.measurement \
  --checkpoint "$PIR2_CHECKPOINT" --dataset "$PIR2_DATASET" \
  --upstream "$PIR2_UPSTREAM" --output "$PIR2_RUN_ROOT/ada-w8-candidate" \
  --device cuda:0 --precisions w8a8 --modes kv_reuse \
  --action-cuda-graph --operator-fusion --dit-cuda-graph \
  --quantization-scope all --quantization-category-id 2 \
  --group-conditioning --norm-modulation-quant --residual-norm-quant \
  --quantization-tactics configs/quantization/groot-libero-spatial-ada/site-tactics.json \
  --quantization-group-tactics configs/quantization/groot-libero-spatial-ada/group-tactics.json \
  --quantization-shape-tactics configs/quantization/groot-libero-spatial-ada/w8-shape-tactics.json \
  --operator-profile configs/quantization/groot-libero-spatial-ada/w8-splitk-writeback.json \
  --measurement-boundary model-only \
  --observations 3 --rounds 20 --warmup 2 --seed 127
~~~

BF16 对照另起进程，将 `--precisions` 改为 `bf16`、使用新的输出目录并移除 W8 专用的 `--operator-profile`，保留相同共同系统选项。measurement 的 `_config` 会为 BF16 关闭 `group_conditioning`、`norm_modulation_quant`、`residual_norm_quant`；整数 tactic 文件不作用于 BF16。两个 Graph flag 同时出现时 whole-action Graph 取代内部 DiT Graph，仍必须记录实际 replay/capture。另以同精度 W8 reference 量化候选数值差异与完整延迟；BF16 对照不能替代这一误差归因，非零差异按上述质量权衡继续评估。

该 shape-tactic 文件当前是 legacy row list，明确为 **unbound**；operator profile 绑定 `sm_89` / `w8a8`，选择 split-K `slices=8, tile=0` 与 native group writeback，但本身没有把整套配方绑定到运行源码哈希。保存全部四个文件及实际 backend/source/输入身份再复测。Orin/Thor 继续保留原架构 guard，不能通过删除 SM89 条件复用该 profile。

允许数值差异的另一条候选使用 [W8 全融合 profile](../configs/quantization/groot-libero-spatial-ada/w8-splitk-full.json)，加入 native residual、LayerNorm/RMS packing、RMS statistics 与共享 context packing。它**质量待验证**，不能视为已通过 LIBERO。显式替换命令如下：保留 shape tactics，移除旧的 `--norm-modulation-quant --residual-norm-quant`，避免与 profile 的 producer 替换重叠；使用新输出目录。

~~~bash
"$PIR2_MODEL_PY" -m coexecution.libero.measurement \
  --checkpoint "$PIR2_CHECKPOINT" --dataset "$PIR2_DATASET" \
  --upstream "$PIR2_UPSTREAM" --output "$PIR2_RUN_ROOT/ada-w8-full-candidate" \
  --device cuda:0 --precisions w8a8 --modes kv_reuse \
  --action-cuda-graph --operator-fusion --dit-cuda-graph \
  --quantization-scope all --quantization-category-id 2 --group-conditioning \
  --quantization-tactics configs/quantization/groot-libero-spatial-ada/site-tactics.json \
  --quantization-group-tactics configs/quantization/groot-libero-spatial-ada/group-tactics.json \
  --quantization-shape-tactics configs/quantization/groot-libero-spatial-ada/w8-shape-tactics.json \
  --operator-profile configs/quantization/groot-libero-spatial-ada/w8-splitk-full.json \
  --measurement-boundary model-only \
  --observations 3 --rounds 20 --warmup 2 --seed 127
~~~

这条命令仅测模型路径；保留与保守 W8、同配置 BF16 的动作误差和完整延迟，再按相同协议评估真实闭环。全融合 profile 仍受 SM89 与各原生归约的 Torch/头文件约束，不能直接用于 Orin/Thor；未完成的平台与任务质量验证继续明确标注。

## 流式平台接入

先复用 [replay/schema.py](../coexecution/replay/schema.py)、[catalog.py](../coexecution/replay/catalog.py)、[model.py](../coexecution/replay/model.py)、[deployment.py](../coexecution/replay/deployment.py)、[media.py](../coexecution/replay/media.py)、[http.py](../coexecution/replay/http.py) 和 [现有前端](../simulator/README.md#quantization-and-robot-debugger)。按“原始输入 → 准备/传输 → GPU 执行 → 动作解码 → 执行/记录”分离职责；事件与媒体记录放入独立队列，显式界定同步、拷贝与计时范围。队列积压、替换、丢弃和记录失败都要保留。异步写盘不等于零开销，干净计时与录制实验分开。

为机器人回放另跑 deployment，使用新输出目录并加 `--record-replay --replay-max-frames 721`；它在控制阶段复制两路相机/状态，控制结束后编码 WebM/VP9。录制会标记 instrumented。`--no-video` 只关闭旧 MP4，不关闭显式 replay。然后构建并查看：

~~~bash
python3 -m coexecution.replay build \
  --model-run "$PIR2_RUN_ROOT/model-only" \
  --deployment /path/to/deployment-replay \
  --output "$PIR2_RUN_ROOT/replay-catalog" \
  --clock-cohort audited-device-and-clock-session

python3 simulator/serve.py --port 8765 \
  --replay "$PIR2_RUN_ROOT/replay-catalog/catalog.json"
~~~

`--clock-cohort` 是操作者依据设备/遥测作出的声明，不是 builder 自动推测；未知时省略，跨设备或时钟组不得用同一个标签制造配对。输出目录需新建或为空，输入只读；本地访问 `http://127.0.0.1:8765/`。

### 现有 v1 数据合同

| 数据 | 必须保留的内容 / 时间单位 |
|---|---|
| `pir2.replay.catalog` / version 1 | `runs` 的 id、kind、precision、mode、label、path、计数、warnings；`media_manifest` |
| `pir2.replay.run` / version 1 | 原始 source 路径与 SHA-256、配置、验证状态；model 的 observations/clean_requests/profile_requests/kernels 或 deployment 的 episodes |
| `pir2.replay.media` / version 1 | opaque asset id、相对路径、MIME、原始来源哈希；不把任意文件路径作为 API |
| model clean | 精确 request/observation ID、seed、scope、`wall_ms`、instrumented 与实际动作 `{shape, values}`；P50/P95 只用已确认 clean 的同边界样本 |
| instrumented profile | 独立 request ID/seed/output 状态；实际 `start_ms`、`duration_ms`、stage/category/kernel_id/device/stream，相对该 CPU request 注解 |
| deployment | 原始单调墙钟 `*_s`、control start/end、真实 `applied_s`、`next_observation_computed_s`；tick/observation/request/feature/frame 血缘 |
| robot media | 每帧 `wall_s`、`relative_s`、`media_time_s`、index、phase、source tick；每路相机自己的 media ID |

model 观测 ID 为 `obs-{index}-{sha256前16位}`，测量 request ID 包含 round、precision-mode、observation 和 scope。deployment importer 将 episode 命名空间内的 `observation-{tick}`、`S1-{request_tick}`、`S2-{source_tick}`、`feature-{feature_sequence}`、`tick-{tick}` 保留下来。沿真实 `feature_sequence/source_tick/capture_s`、`producer_request_tick` 和 `executed_slot` 连接 **相机 → VLM → 已发布特征 → DiT 请求 → 动作槽 → 实际执行**；bootstrap/fallback 或缺失来源保持无 model producer，不能用“最近完成的请求”补链。

AoI 使用实际已执行动作的生产者所用观测时间与真实执行墙钟；同时区分 state/image/cache 年龄、发布、安装、服务计划开始、接纳和执行。不能用当前最新相机帧或 NFE×kernel 均值反推。多设备/多主机将来需要明确 clock domain 与经验证的时钟映射；同名字段不代表时钟天然可减。

clean 与 profiler 调用各有自己的 ID、seed 和动作，不能借 clean 输出填充未保存的 instrumented 输出。只有真实 CUDA launch correlation 能把 kernel 连到请求；聚合 ms 不能画成虚构的 kernel 时间线。GPU work 是各活动时长之和；active union 是时间区间并集；device span 与 host wall 又有各自边界。并发流的 work sum 可以超过墙钟，稳态 GPU union/稳态窗口可报告时间覆盖率，但不能当作 SM occupancy 或神经计算延迟。

同输入模型比较必须同时匹配原始观测哈希、seed、scope、共享配置哈希与已审计 clock cohort。闭环最多配对任务/初始化/seed；精度改变动作后轨迹自然分叉，不能标为同输入数值比较。未知值保持 `null`，失败、未完成、缺视频、录制溢出和不适用状态保留；零不能代替未知。机器人 position 用米，roll/pitch/yaw 命名的字段实际是弧度轴角旋转向量；归一化控制命令不能标成米位移。

`paced` 保持 20 Hz 墙钟；`ready` 在原子仿真步结束后接纳完成计划，不额外等待墙钟网格，每步仍推进 0.05 仿真秒。实际 wall Hz、仿真/墙钟倍率、请求率和计划接纳率分别展示；ready 的 deadline/lateness/固定频率有效性为 null。视频用各自 frame-to-wall/media 映射，不能给旧轨迹加速播放后称为新的 ready 实验。

### 待实现的在线事件合同

现有服务只在 loopback 提供 `/api/replay/catalog`、`/api/replay/run/<id>`、`/api/replay/media/<id>`；校验 manifest allowlist、路径和 MIME，媒体支持 HEAD/有界 Range。它读取完成的证据文件，**不是** live WebSocket 推理 API。

流式平台的下一步是新增版本化、只追加的事件合同，同时复用 v1 快照与媒体身份。以下是设计要求，尚未实现：

- 每个事件具有 schema/version、run/event ID、递增 seq、明确 clock domain 的时间、类型、来源 observation/request/feature/action ID；保留采集、准备、GPU、解码、发布、执行和记录边界，不混用时钟。
- snapshot 带已包含的最后 seq；客户端用 reconnect cursor 恢复，按 ID 去重。历史窗口过期时明确返回缺口并取新 snapshot，不能伪装为连续时间线。事件入队顺序与实际时间可能不同。
- 媒体段完成后追加 manifest-ready 事件，保留不可变来源哈希与每帧墙钟/媒体映射；缓冲和背压有界，缺帧、断流、终止和 recorder 错误可见。
- 浮点/混合精度扩展增加设备与环境身份、cohort、component/module precision、实际格式、覆盖范围及来源 profile 哈希。一次运行的标签不能遮盖 BF16 attention、部分量化或不同 worker 精度。
- 先做桌面端事件顺序、断线续传、过期请求丢弃、暂停/反向 seek、媒体切换和血缘测试；浏览器测试读取已生成证据，不启动模型。本交接不包含移动端验收。

## 接收检查表

- [ ] 每个平台 CUDA 可访问、检查点严格完整加载、实际 device/backend/source/toolchain 身份齐全；没有用 CPU 模型测试替代。
- [ ] 每种精度严格通过有限输出、键/形状、RNG、fresh-input、调度语义、跨请求生命周期、异常恢复和稳态 Graph 无 recapture gate。
- [ ] 逐值一致作为诊断结果保留；有数值差异的候选报告归一化动作误差与真实闭环任务结果，明确速度/质量取舍及尚未验证的范围。
- [ ] 基线/候选使用匹配共同配置，冷/暖、裸 kernel/完整 Linear/整模型/部署边界分开；时钟、热、功耗和 P95 原始样本可追溯。
- [ ] 实际格式与覆盖明确；Orin/Thor 的新端口、FP8/FP4 与混合精度接口缺口已完成实现和测试后才报告支持。
- [ ] 真实 VLM、十任务 Spatial 闭环质量独立验收，初始化配对，失败与 timing-invalid 记录完整；微基准收益不替代任务成功。
- [ ] 回放保留精确 ID/血缘、未知/失败状态和各帧时间映射；在线扩展通过版本、cursor、snapshot、媒体与计时契约检查。

公共检查按 [CONTRIBUTING](../CONTRIBUTING.md) 执行；GPU 用例仍显式 opt-in，例如 `tests/gpu/test_action_graph_gpu.py`、`test_prepared_inference_gpu.py`、`test_shape_tactics_gpu.py`，Thor 另有 `test_floating_quantization_gpu.py` / `test_floating_fusion_gpu.py`。测试选择须覆盖所移植实现，不能将跳过的架构用例算成通过。原始实验、模型、数据、视频、profile、生成图表和过程记录保留在忽略的本地目录。
