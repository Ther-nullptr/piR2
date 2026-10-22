# Jetson 复现与流式平台交接

本项目把量化优化接到完整控制链：**先降低真实 VLM+DiT 的计算成本，再测它能否减少动作等待与信息年龄，并用机器人回放解释变化。** 迁移到 Jetson AGX Orin / Thor 时，复用这套模型、调度和证据接口，按目标设备重新选择内核与运行配置。

交接分两项：在目标设备建立可重复的 BF16 与低精度实验；让新的流式平台直接消费现有 catalog、snapshot、SSE 和媒体映射。具体使用方式见 [LIBERO](libero.md)、[性能实验](coexecution.md)、[前端](../simulator/README.md)；本页集中回答平台迁移顺序与接入工作。

## 复现对象与现有支持

官方 Spatial 基线为 `nvidia/GR00T-N1.7-LIBERO/libero_spatial`，B1、Flow4/NFE4、预测 H40，当前原生解码 16 步。启动时从 worker identity 核对实际 `predicted_horizon` / `decoded_horizon`。训练后的 πR² 采用独立 S2 缓存与 S1 滚动去噪，使用对应 checkpoint 和 `--variant pir2`，按 [时序协议](timing-protocols.md) 单独比较。

| 平台            | 架构           | 可作为起点的证据                       | 本次接收目标                                     |
| --------------- | -------------- | -------------------------------------- | ------------------------------------------------ |
| RTX 6000 Ada    | SM89，x86_64   | CUDA 回归、官方 Spatial 配对模型回放   | 保留提交、输入、设备与时钟组的参考记录           |
| Jetson AGX Orin | SM87，aarch64  | SM80-compatible 整数源码与公共运行接口 | 建立本机 BF16，重新编译并验证 INT8/INT4 集成     |
| Jetson AGX Thor | SM110，aarch64 | 既有 FP8/FP4 内核和服务接口            | 在本机重验当前源码；分别建立整数与原生浮点实验线 |

当前完整套件管线是 Spatial 十任务。表中的源码/接口支持与平台实机验收分开记录；Ada 结果不自动成为 Orin/Thor 结果。目标设备的较少 SM 可能改变占用比例，也同时改变计算、带宽和热约束，选型依据完整延迟、P95、执行 AoI、功耗与任务质量。

<a id="precision-support"></a>

## 精度与执行路径

| 路径                                        | BF16 / W8A8 / W4A4                                    | Thor FP8 / FP4              |
| ------------------------------------------- | ----------------------------------------------------- | --------------------------- |
| 服务与 backend                              | 有显式配置；目标平台执行 CUDA 验证                    | 有 Thor 专用配置和内核      |
| `--dit-cuda-graph`                          | 纯 DiT 范围                                           | 有对应入口，按平台验证      |
| `--action-cuda-graph` / `--action-kv-reuse` | 非流式 Flow4、H40；paired 服务拒绝 whole-action Graph | 当前配置拒绝                |
| `measurement --precisions`                  | 已接入                                                | 尚需接入                    |
| replay v1 / viewer                          | 已接入                                                | 尚需贯通 schema、导入和显示 |

支持范围由 [OptimizationConfig](../coexecution/optimization/config.py)、[measurement parser](../coexecution/libero/measurement.py) 和 [replay schema](../coexecution/replay/schema.py) 定义。W8A8/W4A4 使用 INT8/INT4 点积、INT32 累加；Thor FP8 使用 E4M3 数据和每 tensor FP32 scale，FP4 使用 E2M1 与 block-16 UE4M3 scale，均 FP32 累加、输出 BF16。注意力等未覆盖组件保留浮点，运行记录须保留实际覆盖。

浮点实验单独比较 BF16/FP8/FP4。贯通时同时扩展精度 CLI、runtime/gate、测量元数据、schema/importer、比较键、前端筛选/图例与测试；混合精度增加逐组件格式和 profile 来源。ActionGraph/K/V 复用还需相应实现与输出/RNG/Graph 验证，不能只新增精度名称。

## 第一步：建立目标设备环境与身份

按 [环境说明](environment.md) 分别准备 CUDA 模型环境和 LIBERO 仿真环境。aarch64 使用匹配 JetPack、Python ABI 和系统库的 [Jetson PyTorch 包](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)；选择平台版本时核对 [JetPack 归档](https://developer.nvidia.com/embedded/jetpack-archive)。Ada wheel、扩展 `.so` 和编译缓存须在目标环境重建。

参考 Ada 记录使用 Python 3.10、Torch 2.7.1+cu126、CUDA 12.8 编译器、Transformers 4.57.3、Diffusers 0.35.1。既往 Thor 记录使用 JetPack 7.2 / L4T 39.2、CUDA toolkit 13.2、Torch 2.10.0+cu130、Python 3.12、Triton 3.6.0；它们是复现记录，目标环境以实际兼容性和验证结果为准。Thor 的 `sm_110a` 编译与 ptxas 配置见 [浮点环境](environment.md#optional-thor-floating-inference)。

从仓库根目录设置自己的路径：

```bash
export PIR2_MODEL_PY=/path/to/model-env/bin/python
export PIR2_EVAL_PY=/path/to/libero-env/bin/python
export PIR2_CHECKPOINT=/path/to/GR00T-N1.7-LIBERO/libero_spatial
export PIR2_DATASET=/path/to/groot-libero-spatial
export PIR2_UPSTREAM="$PWD/upstream/learning/Isaac-GR00T"
export PIR2_RUN_ROOT="$PWD/.local/platform-runs"
export PYTHONPATH="$PWD:$PIR2_UPSTREAM"
export TORCH_EXTENSIONS_DIR="$PWD/.local/torch-extensions"
export PIR2_HOST_GUARD_CACHE="$PWD/.local/host-guards"

python3 tools/prepare_sources.py --component groot
# 默认 CUTLASS 流程；使用外部兼容 CUTLASS_ROOT 时省略此行。
git submodule update --init --recursive coexecution/kernels/vendor/cutlass
python3 tools/prepare_sources.py --component robotics-kernels
"$PIR2_MODEL_PY" -m coexecution.kernels.build --backend integer --describe
```

`--describe` 检查元数据，实际构建、加载和模型验证在目标 CUDA 环境执行。CUTLASS 所需头文件与来源记录见 [CUTLASS 设置](environment.md#cutlass-submodule)。编译、权重打包、首次加载、预热与 Graph capture 计入冷启动记录，放在稳态计时外。`compiled` mode 的 Cython/主机编译环境见 [公共推理路径](../coexecution/README.md#common-policy-runtime--公共推理路径)。

每组实验保留以下身份：

| 类别       | 保留内容                                                                                                           |
| ---------- | ------------------------------------------------------------------------------------------------------------------ |
| 代码与后端 | 仓库提交/脏文件指纹、GR00T pin 与补丁、实际 CUTLASS、算子源码及已加载二进制哈希                                    |
| 模型与输入 | checkpoint 配置/分片、processor/统计量/模态、模型与数据 revision、真实图像/状态/文本哈希、episode/step/seed        |
| 执行配置   | attention、组件精度覆盖、布局、Graph、融合、tactic/profile 哈希                                                    |
| 平台与窗口 | Python/Torch/CUDA runtime/nvcc/Triton、JetPack/L4T/驱动、device/SM/RAM、CPU/GPU/EMC 时钟、功耗模式、温度及后台负载 |

固定来源由 [sources.lock.json](../sources.lock.json) 与 [任务下载器](../scripts/fetch_groot_spatial.py) 管理。`results.json`、`mode_evidence`、`runtime-source`、`identity.json` 和 `weight-loading.json` 提供运行证据，设备遥测放入同一实验目录。无法确认的二进制来源保留 unknown。

在可使用的设备环境中查看状态并采集测量窗口遥测：

```bash
nvpmodel -q --verbose
jetson_clocks --show
tegrastats --interval 1000
```

测量使用空闲 GPU；功耗、时钟、温度或设备不同的窗口分组报告。Jetson 的 CPU/GPU 共享 SoC LPDDR/EMC，内存访问成本见 [CUDA for Tegra](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#memory-management)。分别记录启动峰值、稳态系统 RAM 与 CUDA allocated/reserved；这些数值不能直接相加。

Split worker 可能各自持有完整 policy；当前 [TransformerINT](../coexecution/optimization/groot/integer.py) 还保留 BF16 参数及整数/分组权重包。加上 activation workspace、Graph pool、常驻输入、相机和仿真缓冲后，按实际容量决定 worker 布局。部署内存不能从权重位宽直接推算。

## 第二步：复用系统优化，移植目标内核

文本元数据缓存、输入/图像准备、主机 guard 编译、VLM/动作 Graph 和调用内 K/V 复用都属于可复用系统路径。BF16 与量化使用相同适用配置，再评价量化的增量。

| 实现                                            | 当前约束                                                                                                                                                                             | 目标设备工作                                                        |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------- |
| canonical INT8/INT4                             | [integer_gemm.cu](../coexecution/kernels/ampere_ada/csrc/integer_gemm.cu) 使用 SM80-compatible CUTLASS                                                                               | 目标工具链重编译，核对实际整数 kernel，测完整 Linear 和整模型       |
| small-M/GEMV、split-K、residual/group writeback | [设备 guard](../coexecution/kernels/ampere_ada/_candidate_support.py) 限定 SM89，另有 shape/stride/对齐条件                                                                          | 新架构构建、布局/指令审查与设备测试；不合格调用保留同精度 reference |
| native LayerNorm / RMS                          | [norm.py](../coexecution/kernels/ampere_ada/norm.py) 要求 Torch 2.7.1；[rms_statistics.py](../coexecution/kernels/ampere_ada/rms_statistics.py) 另核对 cu126、Torch Git 和归约头文件 | 检查目标 ARM wheel 的归约实现，适配后验证，不通过删 guard 启用      |
| Thor FP8/FP4                                    | [Blackwell 内核](../coexecution/kernels/blackwell) 有独立 pack/scale/tactic 合同                                                                                                     | 使用匹配 SM110 环境，单独验收格式、调参和数值                       |

`--action-kv-reuse` 在**一次完整 Flow4 调用内**将各层 cross-attention K/V 投影从四次降为一次。每个新观测重新计算第一次 K/V；Q、self-attention、mask、四次 NFE 和 RNG 合同保持。它依赖 whole-action Graph，纯 DiT Graph 不覆盖该生命周期。实现见 [kv_reuse.py](../coexecution/optimization/groot/kv_reuse.py)；投影次数、GEMM 数、kernel launch 数与整模型加速分别测量。

### 按真实形状调优

1. 从官方模型调用采集模块名、执行次数、`[M,K,N,has_bias]`、stride、padding、dtype、格式和 backend 身份，并保留真实输入与权重。
2. 检查有限输出、键/形状、fresh-input、RNG、独立输出存储、异常恢复、冻结权重、调度语义与 Graph 无 recapture；另存同精度误差。
3. 分别测裸 GEMM、在线 prepare/scale/pack+GEMM+epilogue 的完整 Linear、完整 VLM+Flow4 和部署。在线打包、写回与 workspace 进入实际边界。
4. 在目标设备扫描 tile、warps、stages、split-K、grouping、layout 与 workspace。独立短 NCU 采样定位实际 kernel 的 occupancy、带宽、stall、寄存器和 shared memory；用 clean wall 确认完整收益。
5. 用相同输入/seed/设置/时钟组进行重复、交错或反向顺序测量，保存原始值，再选择显式 profile。每个整模型 case 优先使用新进程，避免进程级 interop 初始化与缓存相互影响。

`--operator-profile` 选择实现；`--quantization-tactics`、`--quantization-group-tactics`、`--quantization-shape-tactics` 选择整数 tactic。参数空间和 schema 见 [算子清单](../coexecution/README.md#owned-operators-and-explicit-profiles)。Profile 绑定架构、精度、shape、backend revision、格式与测量组；legacy shape row list 保持 unbound。变更权重/精度/tactic 后重建 scope 与 Graph。

Ada Spatial 的已有候选供定位与迁移参考：

| 配置                                                                                                                                                                     | 用途与边界                                                                            |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| [site-tactics](../configs/quantization/groot-libero-spatial-ada/site-tactics.json)、[group-tactics](../configs/quantization/groot-libero-spatial-ada/group-tactics.json) | 真实调用位置与分组 tactic                                                             |
| [w8-shape-tactics](../configs/quantization/groot-libero-spatial-ada/w8-shape-tactics.json)                                                                               | legacy unbound shape rows，需连同运行身份保存                                         |
| [w8-splitk-writeback](../configs/quantization/groot-libero-spatial-ada/w8-splitk-writeback.json)                                                                         | SM89/W8A8，split-K slices=8、tile=0 与 native group writeback                         |
| [w8-splitk-full](../configs/quantization/groot-libero-spatial-ada/w8-splitk-full.json)                                                                                   | 增加 residual、LayerNorm/RMS packing/statistics 与 shared context packing；质量待验证 |

全融合 profile 替换旧 `--norm-modulation-quant --residual-norm-quant` producer，避免重复安装；仍保留 shape tactics。BF16 对照移除 W8 专用 profile，使用相同共同系统优化。SM89 及原生归约 guard 保留，文件名不作为 Jetson 验收或全局最优的证据。

## 第三步：建立模型与闭环对照

下面的固定真实输入实验比较共同系统优化上的 K/V 增量，使用新的输出目录：

```bash
"$PIR2_MODEL_PY" -m coexecution.libero.measurement \
  --checkpoint "$PIR2_CHECKPOINT" --dataset "$PIR2_DATASET" \
  --upstream "$PIR2_UPSTREAM" --output "$PIR2_RUN_ROOT/model-only" \
  --device cuda:0 --precisions bf16 w8a8 \
  --modes compiled kv_reuse --action-cuda-graph \
  --operator-fusion --quantization-scope all --vision-channels-last \
  --measurement-boundary model-only \
  --observations 3 --rounds 20 --warmup 2 --seed 127 --replay-assets
```

`compiled` 与 `kv_reuse` 只差调用内 K/V 复用，均含输入准备、tokenizer cache、static VLM、VLM Graph、图像数组和 compiled guard。Measurement 通过 mode 管理复用，命令中不再加全局 `--action-kv-reuse`；独立服务才直接使用该开关。

`model-only` 预先准备 CUDA 输入，每次重新运行完整 VLM+Flow4，计时外是准备/传输、解码、RPC、仿真和写盘；按 mode 分块测量，可另存反向 mode 顺序检查漂移。`comparison` 则按轮随机 mode 顺序，分别报告 policy、preparation_transfer、prepared_model。各边界相互重叠，不能相加或相减当作完整分解。

`results.json`、`samples.jsonl`、`actions.npz`、数值/RNG gate 与 `graph-timing-audit.json` 保留原始证据。`--profile-rounds 1` 增加独立 instrumented 调用；`--cuda-events` 也属于 instrumentation。它们解释 kernel 行为，Clean 样本衡量稳态延迟。

### 从同配置模型推进到闭环

按下面配方分别启动两个终端中的 action/VLM worker，等待 ready 后启动评估；切换精度时使用新输出目录并保持设备分配和 worker 布局一致：

```bash
export PIR2_PRECISION=bf16
"$PIR2_MODEL_PY" -m coexecution.libero.service \
  --checkpoint "$PIR2_CHECKPOINT" --variant flow --role action \
  --device cuda:0 --port 5570 --output "$PIR2_RUN_ROOT/action-$PIR2_PRECISION" \
  --inference-precision "$PIR2_PRECISION" --quantization-scope all \
  --operator-fusion --action-cuda-graph --action-kv-reuse
```

```bash
"$PIR2_MODEL_PY" -m coexecution.libero.service \
  --checkpoint "$PIR2_CHECKPOINT" --variant flow --role vlm \
  --device cuda:0 --port 5572 --output "$PIR2_RUN_ROOT/vlm-$PIR2_PRECISION" \
  --inference-precision "$PIR2_PRECISION" --quantization-scope all \
  --operator-fusion --input-preparation --tokenizer-cache \
  --static-vlm --vlm-cuda-graph --image-array-preparation --compiled-host-guards
```

仿真环境按 [资源与环境](libero.md#assets-and-environment) 配好 LIBERO/EGL，然后运行：

```bash
"$PIR2_EVAL_PY" scripts/evaluate_libero_protocol.py \
  --protocol deployment --variant flow --port 5570 --slow-port 5572 \
  --server-layout split --execution-clock paced --control-hz 20 \
  --episodes-per-task 20 --max-steps 720 --seed 1000 --no-video \
  --output "$PIR2_RUN_ROOT/deployment-$PIR2_PRECISION"
```

这是独立部署配方，服务配置与整模型测量分别记录。当前 LIBERO 评估器要求 `control_hz=20`，物理步为 0.05 s。算法对照改用 `--protocol algorithm --visual-delay 3 --action-delay 1`，固定图像年龄和动作释放；部署对照使用真实墙钟，按 [时序协议](timing-protocols.md) 分开汇总。πR² 使用训练后的 checkpoint、`--variant pir2`，移除 whole-action Graph/K/V 选项。

批量条件用 `python -m coexecution.libero.experiments --plan PLAN.json --output OUTPUT`，计划字段见 [运行时模块](libero.md#runtime-modules--运行时模块)。配对 task、seed、初始化、fixture/settled-state/initial-RGB 哈希；子集诊断注明 task_ids，失败与 timing-invalid 回合保留。更换配方使用新的输出目录。

### 速度与任务质量共同验收

允许适当数值差异换取速度。逐值一致作为诊断与明确声明的合同保留；有差异的融合/量化候选继续报告同精度 reference 和 BF16 下的相对 L2、RMSE、最大绝对误差及分通道/预测步统计，注明归一化与近零分母处理。

候选的推广依据真实 VLM、配对初始化的 LIBERO 任务结果，联合延迟/P95、执行 AoI 与任务成功权衡。可接受损失由实验目标记录；未完成闭环的候选标记质量待验证。有限值、形状、fresh-input、RNG、调度/NFE、资源生命周期和计时边界保持严格。现有逐值 gate 的失败如实保存，允许误差的质量评估单独留证据。

## 第四步：接入新的流式播放平台

复用 [replay 模块](../coexecution/replay/README.md) 作为后端证据层，把平台组织为观测来源、输入准备、模型执行、输出处理和异步记录五层。模型测量与仿真/RPC/编码解耦，共同使用精确来源 ID 连接回放。

| 能力         | 当前实现                                                | 新平台应复用或补充的内容                        |
| ------------ | ------------------------------------------------------- | ----------------------------------------------- |
| 完成实验目录 | `build`、catalog/run/media v1                           | 复用 schema/importer/媒体身份，新增展示层       |
| 在线模型记录 | `record-model`、runtime receipt、bounded journal        | 复用 producer 和显式启动入口                    |
| 增量传输     | snapshot + SSE、Last-Event-ID、resnapshot、end          | 按 sequence 恢复，保留时钟域与错误状态          |
| 桌面播放     | 共用游标、视频 seek、依赖高亮、K/D 与控制器 cohort、SSE | 复用选择和时间映射，扩展所需视图                |
| 机器人视频   | 控制时复制完整帧组，结束后编码；已有 sink hooks         | 真正实时视频需独立有界编码/分段媒体 producer    |
| Thor FP8/FP4 | 专用计算 backend                                        | 按精度支持表贯通测量与 replay，保存实际格式覆盖 |

静态目录构建与预览：

```bash
python3 -m coexecution.replay build \
  --model-run "$PIR2_RUN_ROOT/model-only" \
  --deployment /path/to/deployment-replay \
  --output "$PIR2_RUN_ROOT/replay-catalog" \
  --clock-cohort audited-device-and-clock-session
python3 simulator/serve.py --port 8765 \
  --replay "$PIR2_RUN_ROOT/replay-catalog/catalog.json"
```

机器人记录来自另一次部署的 `--record-replay --replay-max-frames 721`，计入录制 instrumentation。在线模型命令、HTTP 路由和完整 schema 见 [增量交付合同](../coexecution/replay/README.md#explicit-recording-and-incremental-delivery)；已实现的 `/api/replay/snapshot/<id>` 与 `/api/replay/events/<id>` 可直接接入。

新平台保持以下观察关系：

- 用 observation/request/feature/action ID 连接相机→VLM→特征→DiT→执行；bootstrap/fallback 和缺来源显示真实状态。
- AoI 来自已执行动作生产者使用的图像/状态时间；wall、simulation、media 与多设备 clock domain 分开。每帧保持自身墙钟/媒体映射。
- Clean 与 Profile 各用自己的调用、seed 和输出；kernel 只按实际 launch correlation 关联，host stage 不能替代 CUDA span。
- 同输入模型比较匹配观测、seed、scope、共享配置和 audited cohort；闭环配对初始化，轨迹各自回放。
- 事件/媒体队列有界，显示丢失、替换、缺帧和终止；unknown/null 与真实零值分开。新增实时编码单独测复制、排队和编码开销。

桌面端验收覆盖断线续传、重复/断档、暂停/反向 seek、真实视频变化、精度/协议切换与 producer 高亮。已有浏览器检查读取生成证据，不启动模型；入口见 [前端验证](../simulator/README.md)。

## 交接产物

| 产物           | 验收依据                                                                  |
| -------------- | ------------------------------------------------------------------------- |
| 每平台运行清单 | CUDA 加载、源码/工具链/模型身份、功耗与时钟遥测                           |
| 模型比较       | 相同共同配置、独立冷/暖边界、原始 latency 与动作误差、严格语义/Graph 检查 |
| LIBERO 比较    | 真实 VLM、配对初始化、成功/失败和时序有效性、P95 与执行 AoI               |
| 回放目录       | 完整来源、状态、frame mapping、模型/控制器/调度身份                       |
| 在线接入       | snapshot/cursor/sequence/resnapshot/end 与有界媒体/事件合同               |

公共检查按 [CONTRIBUTING](../CONTRIBUTING.md)。目标 CUDA 用例按移植范围选择，例如 `test_action_graph_gpu.py`、`test_prepared_inference_gpu.py`、`test_shape_tactics_gpu.py`；Thor 另有 floating quantization/fusion 用例。架构 skip 单独记录。模型、数据、实验原件、视频、profile 和过程文档留在忽略的本地目录，Git 保留实现、配置和稳定说明。
