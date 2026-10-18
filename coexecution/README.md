# GR00T 推理、量化与算子参考

`coexecution` 把 GR00T 的真实 VLM（S2）、DiT 动作头（S1）、执行调度和测量边界拆成可组合模块。共享运行时优化用于减少输入准备与提交开销，量化和融合用于减少模型计算与中间读写；两类优化通过同一策略接口进入 LIBERO 闭环。

本页回答“实现在哪里、开关如何组合、算子合同是什么”。实验步骤见[测量与争用实验](../docs/coexecution.md)，任务运行见 [LIBERO](../docs/libero.md)，调度与特征传输见[时序协议](../docs/timing-protocols.md)，环境与编译见[环境配置](../docs/environment.md)。主线使用官方 Spatial 任务权重及其 πR² 适配检查点。

## 模块地图

| 路径                                                              | 职责                                              |
| ----------------------------------------------------------------- | ------------------------------------------------- |
| `optimization/config.py`、`tactics.py`、`operators.py`            | 不加载 GPU 的配置、tactic、算子清单与 profile     |
| `optimization/capabilities.py`、`session.py`                      | 后端准入、可恢复安装、运行身份与证据              |
| `optimization/groot/projections.py`、`integer.py`、`floating.py`  | 共用模型节点发现与投影替换                        |
| `optimization/groot/*_connections.py`                             | GR00T norm、residual 等 producer 到打包算子的连接 |
| `optimization/execution/graph.py`、`cache.py`                     | 纯 DiT Graph、调用内缓存与串行所有权              |
| `optimization/execution/preparation.py`、`tokenization.py`        | 输入路由与精确文本缓存                            |
| `optimization/execution/guards.py`、`patching.py`、`groot/vlm.py` | 冻结执行校验、实例级适配与完整 VLM Graph          |
| `optimization/backends/integer/linear.py`                         | 整数激活打包、GEMM 与 dispatch                    |
| `optimization/backends/thor/linear.py`、`packing.py`              | Thor 浮点格式与执行                               |
| `libero/inference.py`、`measurement.py`、`service.py`             | 常驻输入模型接口、独立测量与策略服务              |
| `profiling/`、`replay/`                                           | VLM/DiT 争用实验、模型及机器人回放证据            |

实际 CUDA/Triton 实现在 `kernels/`；模型连接留在 `optimization/groot/`。后端不依赖 LIBERO 调度器或 RPC。整数与浮点后端共享模型发现及 Graph 生命周期，各自拥有 packed format。配置导入不加载 Torch/GPU，安装时只加载选中的实现。旧 `coexecution.groot_optimization`、`quantization`、`floating_quantization` 导入路径保留兼容转发；新增实现使用上述目录。

<a id="common-policy-runtime--公共推理路径"></a>

## 共用策略接口

服务开关默认为 BF16、优化关闭。先准备 [CUDA 环境](../docs/environment.md)和完整任务检查点，再显式选择精度与算子：

```bash
export PYTHONPATH="$PWD:$PWD/upstream/learning/Isaac-GR00T${PYTHONPATH:+:$PYTHONPATH}"
python scripts/serve_libero_protocol.py \
  --checkpoint models/GR00T-N1.7-LIBERO/libero_spatial \
  --variant flow --role action --output .local/integer-server \
  --operator-fusion --inference-precision w8a8 \
  --quantization-scope all --quantization-category-id 2
```

该命令启动 action 服务。运行完整协议时，为独立 VLM 服务选择对应配置，或使用声明条件的实验 runner。每份策略由一个提交者串行调用；`--server-layout paired` 使用两份独立策略与流，并协调所有端点的冷 VLM/DiT capture，稳定执行可重叠。配对服务支持纯 DiT Graph，拒绝 whole-action Graph；特征租约与 placement 见[时序协议](../docs/timing-protocols.md)。

已有 eval-mode BF16 CUDA 策略可直接使用可恢复作用域：

```python
from coexecution.groot_optimization import GrootOptimizations, OptimizationConfig

config = OptimizationConfig(precision="w8a8", fusion=True, scope="all", category_id=2)
with GrootOptimizations(policy, config) as optimized:
    actions, info = policy.get_action(
        observation,
        options={"force_nonstreaming": True, "num_inference_timesteps": 4},
    )
    coverage = optimized.evidence()
```

作用域退出时恢复原 forwards/processors。更换权重、方法、attention 配置或精度前关闭作用域；支持的运行时开关可经 `optimized.set_runtime(...)` 更新，保留精度控制器，并按需关闭和重新 capture Graph。调用方负责 CUDA stream 上的输入/权重就绪。Python 串行保护不替代跨流同步；root 调用、reset 及异常清理共用投影缓存边界。

## 与精度共用的运行时优化

这些开关保持每次观测的图像、状态和 VLM 计算更新，使 BF16 与量化模式能够使用相同的系统路径。

| 开关                        | 行为与条件                                                                                                          |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `--input-preparation`       | 向各组件路由实际消费的 tensor 字段                                                                                  |
| `--tokenizer-cache`         | 图像处理后缓存精确展开文本的 tokenization                                                                           |
| `--static-vlm`              | 准备固定 text/grid/mask 元数据，避免 device scalar 读取                                                             |
| `--vlm-cuda-graph`          | 捕获完整 VLM，要求 `--static-vlm`                                                                                   |
| `--image-array-preparation` | 用新鲜 contiguous uint8 RGB array 代替 PIL 转换；保留 conversation、模板、crop/resize、state 与 Qwen collator       |
| `--compiled-host-guards`    | 用 Cython 编译同一 Python guard 源码，保留权重、方法、hook、closure、defaults、配置和拓扑校验                       |
| `--vision-channels-last`    | 显式选择 vision patch Conv3D 的 channels-last-3D 输入与权重布局；可改变 BF16 舍入，退出恢复原 storage、布局和 hooks |

image-array 路径校验固定 GR00T/Transformers 源码和当前方法，接纳标准 Qwen2VL fast/slow pixel processor；替换 processor/method 需要关闭作用域。高度为 1 或 3 的图像因 ndarray channel 推断歧义而被拒绝。每次使用独立图像缓冲。

compiled guards 需要 CPython、host C compiler、`Cython==3.1.2` 和 setuptools；已记录的 setuptools 兼容版本为 84.0.0，构建保存实际版本。缓存默认位于 `~/.cache/pir2/host-guards`，可用 `PIR2_HOST_GUARD_CACHE` 指定；缓存加锁，并按源码、ABI、平台、编译器和编译选项标识。构建失败显式报错；默认 Python 路径及 CLI help 不触发编译。构建身份和 binary hash 写入证据。

### 常驻输入接口

`coexecution.libero.inference.PreparedInferenceEngine` 在调用方持有的 GPU 输入上提供 `vlm`、`action` 和 `full`。输入准备及动作解码是独立 adapter，decoder state 对 engine 不透明。engine 原样返回上游输出、不保留输出，也不执行输入复制或静态元数据注册；需要留存结果时，在计时外按其生命周期做 snapshot。

准备阶段先复制可变输入，再由上游注册静态元数据。已注册对象的 identity 和 mutation guards 保持有效；回放期间保持对应优化作用域与 bindings 存活。调用方建立 inference mode、stream readiness、串行所有权，并在 runner 边界测 CUDA 完成。完整实验与计时范围见[独立模型测量](../docs/coexecution.md#independent-model-measurement)。

### Graph 层级与新鲜度

| 开关                  | 捕获范围                                            | 适用路径                                                                         |
| --------------------- | --------------------------------------------------- | -------------------------------------------------------------------------------- |
| `--dit-cuda-graph`    | 纯 DiT forward；rolling buffer、noise、解码在图外   | 每策略串行的 Flow 与 πR²；配对服务协调冷 capture                                 |
| `--action-cuda-graph` | S1 visual/state conditioning 与完整四步 Flow 动作头 | BF16/W8A8/W4A4，非 streaming H40/NFE4                                            |
| `--action-kv-reuse`   | 同一 Flow4 body 内复用每层 cross-attention K/V 投影 | 要求 whole-action Graph、固定 interleaved AlternateVLDiT 与受支持 SDPA processor |

whole-action Graph 每个请求生成一次新显式 noise，包括首次 capture；features、state、embodiment ID 和 masks 均新鲜复制。整数 image delay 与 shape/dtype/stride/stream 共同组成最多八项的 Graph signature。inpaint、rolling/streaming、未知 options/hooks 被拒绝。调度器状态、cache 时间戳、noise 生成与动作解码留在图外；输出持有独立 storage。

同时开启两个 action 相关 Graph 开关时，whole-action Graph 替代内层 DiT Graph，并在证据中标明。新 signature 先预热；whole-action capture 失败向调用方传播，不静默切换 eager。冻结校验可检测受追踪的修改；`.data` 或外部直接写入属于调用方必须避免的未追踪修改。

K/V reuse 支持 `AttnProcessor2_0` 和 `PreparedAttention`，与 operator fusion 独立。每层第一次 K/V 计算仍被捕获，随后三次复用，故每次 replay 都从新观测重算 K/V；Q、self-attention、mask、state/noise 与四次 denoising 保持更新。复用状态在 body 退出（含异常）清空，无跨观测复用。整数 grouping/tactics/producer packs 继续生效。

审计区分物理 Python hooks 和逻辑工作量：`python_forward_counts` 是实际 hook 计数；whole-action 的 `forward_counts.dit` 由每个 captured body 的四次调用及 replay 次数证明。`dit_count_source`、`action_graph` 与 `action_graph.kv_reuse` 保留来源和每层投影计数。K/V projection 次数与 GEMM/kernel 次数不同：分组 K/V 可用一次 GEMM 完成两次投影。

## 整数量化与 producer 融合

权重在安装时量化并打包；**online quant** 指每次调用的激活尺度计算、量化和打包。INT8×INT8 或 packed INT4×INT4 GEMM 使用 INT32 accumulation，scale、bias 与 BF16 输出转换进入 epilogue。兼容的 QKV/KV、gate/up 投影共享输入准备；GELU、SiLU-times-up 或 AdaLN producer 可直接生成 packed input，减少中间读写。SDPA 的 QK/softmax/PV 保持浮点路径。

量化覆盖、后端选择和调度独立：启用量化不改变 checkpoint、采样步数、观测或 VLM cache 策略。原 BF16 参数保留以支持恢复，加上 packed weights、workspace 与 Graph pools 后，诊断进程总显存不等于纯低比特模型的最小部署显存。

| 开关                                                                       | 合同                                                                                                                                                                |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--inference-precision {bf16,w8a8,w4a4,fp8,fp4}`                           | BF16 默认；浮点低比特使用下节 Thor 后端                                                                                                                             |
| `--operator-fusion`                                                        | 启用 BF16 RoPE/RMSNorm、AdaLN 与共享 DiT condition/mask 准备；不支持的 attention 配置安装时拒绝                                                                     |
| `--operator-fusion-mode {vision_rope,text_rope,rope,rmsnorm,rope_rms,all}` | 默认 `all`；`rope` 包括两种 RoPE，只有 `all` 安装 DiT 点算子准备；选择模式本身不启用 fusion                                                                         |
| `--quantization-scope {transformer,all}`                                   | `transformer` 选 transformer projections；`all` 加入其余合适 BF16 Linear，包括 condition/vocabulary                                                                 |
| `--quantization-component {all,vlm,dit}`                                   | 与 scope/coverage 求交；VLM 为 backbone，DiT 为整个 action head（含 `vl_self_attention`）；不控制一般 BF16 fusion 范围                                              |
| `--quantization-category-id N`                                             | `scope=all` 时量化固定 embodiment 的 category projections；逐次检查 B1 与实际 ID，LIBERO_PANDA 的固定映射为 2                                                       |
| `--group-conditioning`                                                     | 整数精度 + fusion `all` + scope `all`；拼接各 DiT condition 与最终 modulation 投影，一次准备/GEMM 服务共享或 per-token condition；需完整组覆盖，流式 πR² 拒绝此选项 |
| `--norm-modulation-quant`                                                  | 非仿射 DiT LayerNorm、AdaLN、动态 scale 与 packing 融合；要求低比特 + fusion `all`，支持共享/per-token condition                                                    |
| `--residual-norm-quant`                                                    | attention residual + FFN 输入 LayerNorm + packing；要求上一选项和量化 FFN 输入投影；只支持非仿射 LayerNorm、无 positional embeddings 的 blocks                      |

`component=vlm` 不能组合 category、condition grouping 或 DiT norm producers。condition grouping 的 packed 拼接权重增加存储，只复用当前 forward 的相同激活，不跨 denoising 步缓存。residual 边界没有 gate/modulation，整数实现用 unit gate、zero scale/shift 保持原 residual 方程；FFN 输出 residual 仍独立。norm 融合改变 FP32 reduction 顺序，应记录相对同精度未融合路径及 BF16 的动作误差。

### Coverage 与 tactic

| 输入                                | 格式与选择顺序                                                                                                   |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `--quantization-coverage FILE`      | `linears` 以 module name 为键；无文件时使用模型 inventory，category 单独选择；旧 shape lookup 才需要输入 `shape` |
| `--quantization-tactics FILE`       | `{"8": {"module.name": 0}, "4": {...}}`，canonical tactics 为 0–7                                                |
| `--quantization-group-tactics FILE` | rows 含 `bits`、`members`、`tactic`；旧 shape rows 额外要求 coverage                                             |
| `--quantization-shape-tactics FILE` | 精确 `[M,K,N,has_bias]` 覆盖 site/group tactic；M 为展开行数，group 的 N 为合并 padded width                     |

未匹配项使用 reference tactic 0。shape 选择作用于 native、prepared 和 grouped 路径，保持量化与权重不变；表必须在 Graph capture 前载入，修改后重建作用域。工具不会从旧实验目录自动寻找“最佳”配置。

shape 文件可为旧 row list，也可绑定实际后端身份：

```json
{
  "schema_version": 1,
  "backend_id": "pir2_integer",
  "backend_revision": "sha256:<runtime source_sha256>",
  "device_arch": "sm_89",
  "quant_format": "w8a8-symmetric-row-v1",
  "rows": [{ "bits": 8, "shape": [41, 6144, 1536, true], "tactic": 2 }]
}
```

这是 schema 示例，revision 应来自实际 backend evidence。W4A4 使用 `w4a4-symmetric-row-v1`。身份缺失、unknown 或不匹配的绑定 profile 被拒绝；旧 list 明确为 unbound，旧 `robotics_integer` 绑定不会自动匹配 `pir2_integer`。Thor 浮点使用独立格式。

[libero_10 Ada 配置](../configs/quantization/groot-libero10-ada/coverage.json)保留为显式的历史校准输入：469 个 executed Linear sites 和 7 个 fixed-category projections，B1/H40 Flow4、SDPA、fusion、DiT Graph、channels-last-3D 条件。它不是 Spatial streaming preset。使用时同时选择 [site tactics](../configs/quantization/groot-libero10-ada/tactics.json)、[group tactics](../configs/quantization/groot-libero10-ada/group-tactics.json)，并显式启用 `--vision-channels-last`；配置不会自动选择该布局。

重新校准时，采集完整真实调用的执行 shape，在实际 packed inputs/weights 上比较 tactics 0–7，保留重复测量与同精度误差，再于新进程验证完整输出和 policy wall time。不同 checkpoint、长度、硬件或 producer 组合需要各自的校准；重复 GEMM 的 cache 行为与完整策略不同。

## Owned operators and explicit profiles

| 源码位置                                      | 实现                                                |
| --------------------------------------------- | --------------------------------------------------- |
| `kernels/common/csrc/`                        | native vision RoPE、AdaLN BF16 fusion               |
| `kernels/common/fused.py`、`graph.py`         | 共享 Triton primitives 与 Graph 生命周期            |
| `kernels/ampere_ada/csrc/integer_gemm.cu`     | canonical INT4/INT8 CUTLASS GEMM                    |
| `kernels/ampere_ada/csrc/integer_dispatch.cu` | native 激活准备与 integer dispatch                  |
| `kernels/ampere_ada/`                         | 整数格式、modulation producers、Ada 候选            |
| `kernels/blackwell/csrc/`                     | FP8/FP4 CUDA GEMM/packing 与 lazy loaders           |
| `kernels/vendor/cutlass/`                     | 默认官方 CUTLASS 依赖，可由兼容 `CUTLASS_ROOT` 替代 |

源码归属与版本见[上游记录](../docs/upstream.md)，CUTLASS 初始化、替代安装和 CUDA 编译见[环境配置](../docs/environment.md#cutlass-submodule)。以下入口读取源码/构建元数据，不加载模型：

```bash
python tools/prepare_sources.py --component robotics-kernels
python -m coexecution.kernels.build --help
python -m coexecution.kernels.build --backend integer --describe
```

build 接受 `integer`、`integer_dispatch`、`bf16_fusion`、`fp8`、`fp4`；移除 `--describe` 才显式构建/加载后端，`--verbose` 显示构建信息。二进制写入 Torch extension cache（可设 `TORCH_EXTENSIONS_DIR`），在计时和 capture 前完成构建/预热。

算子 inventory 提供语言、源码、architecture、precision、shape/layout 和数值合同，不依赖 Torch/Triton：

```python
from dataclasses import asdict
from coexecution.optimization.operators import inventory

for implementation in inventory("integer_gemm"):
    print(asdict(implementation))
```

SM89 的显式候选包括 `integer_gemm=int4_gemv/small_m/split_k`、`layernorm_pack=native`、`rmsnorm=native`、`rmsnorm_pack=native`、`integer_residual=native`、`group_writeback=native`。每次调用检查 shape/runtime 合同，不适用时回到同精度 reference。native LayerNorm/RMS 还有[固定 Torch 条件](../docs/environment.md#optional-gr00t-fusion-and-integer-inference)。其他 implementation 以 inventory 自身合同为准。

`--operator-profile FILE` 是某 architecture/precision 的部分覆盖，未列 family 保留默认选择；需启用 fusion 或低比特精度。示例：

```json
{
  "schema_version": 1,
  "architecture": "sm_89",
  "precision": "w4a4",
  "implementations": { "integer_gemm": "small_m", "activation_pack": "native" },
  "parameters": { "integer_gemm": { "tactic": 0 } }
}
```

`small_m` tactic 为 0–5，`split_k` 的 `slices` 为 2/4/8、`tile` 为 0–2，与 canonical tactic 0–7 分开。示例描述格式，不提供测量推荐。profile 在模型安装前校验未知字段、architecture/precision、flag 与 family 适用性，保留文件 hash 和实际选择。

DiT LayerNorm/residual/shared-context/final-modulation choices 要求 full fusion 和 DiT component；RMS packing 要求 text RMSNorm fusion。每个 producer 边界只选择一个实现：profile LayerNorm 与 `--norm-modulation-quant`、profile residual 与 `--residual-norm-quant`、profile LayerNorm 与 final-modulation packing 不可重叠。

`read_measurements(path)`、`select_measured(records, baseline="...")` 消费保留的标准测量记录：必须匹配 architecture、precision、shape、cohort、checkpoint/input、设备/环境、计时边界、方法与统计量，并通过显式同精度数值门槛。`deployment=True` 额外要求完整 policy wall latency。选择结果只适用于该上下文，不改变运行默认值。

运行证据分别记录 upstream revision、owned-source fingerprint、loaded-library hash、Torch/CUDA/Triton、设备及 quant format。无法确认的预加载或 binary ownership 标为 `unknown`；环境变量路径不等于实际执行证明。缺 API 或加载失败直接报错。canonical integer 保持其算术/API 合同，SM89 为历史验证架构；新增 Ada candidates 限定 SM89，浮点限定 SM110，BF16 不新增架构白名单。

聚焦单个 BF16 fusion 时可使用 `NonGemmAdapters`：

```python
from coexecution.groot_fusion import NonGemmAdapters

adapter = NonGemmAdapters(policy)
try:
    adapter.set("vision_rope")
    # 在私有 eval 模型上执行推理；等待完成后才能换 mode。
finally:
    adapter.close()
```

支持 `baseline`、`vision_rope`、`text_rope`、`rope`、`rmsnorm`、`rope_rms`、`all`；这里的 `all` 为 RoPE/RMSNorm/AdaLN，不安装高层 DiT condition/mask、Graph 或量化。服务通过 `--operator-fusion-mode` 选择对应组件，baseline 则关闭 `--operator-fusion`。

## Experimental Thor FP8 and FP4

Thor SM110 浮点后端使用相同的可恢复接口和 eval-mode BF16 CUDA 原始权重。软件/编译准备见 [Thor 环境](../docs/environment.md#optional-thor-floating-inference)，迁移步骤见[平台交接](../docs/platform-handoff.md#precision-support)。

```python
config = OptimizationConfig(
    precision="fp8", scope="all", dit_graph=True, fp8_fast_quant=True
)
with GrootOptimizations(policy, config) as optimized:
    coverage = optimized.evidence()
```

服务对应 `--inference-precision fp8 --quantization-scope all --dit-cuda-graph --fp8-fast-quant`；FP4 选择 `fp4` 并省略 `--fp8-fast-quant`。

| 机制   | 格式和执行合同                                                                                                                                             |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| FP8    | E4M3 权重/激活，每 tensor 一个 FP32 scale；K/N 需被 16 整除                                                                                                |
| FP4    | E2M1 数据与每 16 元素 UE4M3 scale，CUTLASS swizzled layout；K/N 需被 32 整除                                                                               |
| GEMM   | FP32 accumulation、BF16 输出；低比特 operands 直接进入 GEMM                                                                                                |
| 覆盖   | `scope=all` 包括 eligible 普通 Linear；CategorySpecificLinear、excluded/unaligned sites 保持 BF16 并记入 coverage；attention 保持浮点                      |
| tactic | FP8 查精确 `(M,K,N,has_bias)` Thor 表，fallback 为 prefill tactic 2；FP4 的 M>1 用 bias-aware 0、M=1 用 3，避免缺 bias epilogue 的 nonzero prefill tactics |

浮点模式拒绝整数 tactic files、category selection 与 condition grouping。可选融合如下，默认关闭：

| 开关                                               | 行为                                                                                                                     |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `--fp8-fast-quant`                                 | 两遍 Triton 激活打包：partial absmax，再 finalize scale/cast；每次调用及 replay 都刷新，默认是 native packer             |
| `--fp-shared-inputs`                               | 同一已知 QKV/KV 或 gate/up owner invocation 中，共用未修改的同一 activation packing；独立调用仍独立 pack                 |
| `--fp-grouped`                                     | 完整选中组拼成一次 GEMM，优先于 shared-inputs；FP8 拼接权重后共用 tensor scale，可改变误差；FP4 保持 block-16 row scales |
| `--fp-swiglu`                                      | native BF16 SiLU lookup、乘法与 packing 融合，保留 SiLU 输出先 round BF16 的边界；FP8 两遍，FP4 直接输出 E2M1/scales     |
| `--norm-modulation-quant`、`--residual-norm-quant` | 配合 `--operator-fusion` 使用独立 floating producers；residual/materialized 边界仍为 BF16，LayerNorm reduction 顺序改变  |

shared/grouped 状态在 owner 退出或异常时清空，Graph 记录每次 replay 的 fresh packing；group outputs 通常保留 views，text Q/K normalization 需要时产生 contiguous inputs。原参数、packed weights、scales、workspaces 与 Graph pools共同常驻。

### FP4 Swap-AB

`--fp4-swap-ab` 或 operator profile 的 `"floating_gemm": "swap_ab"` 显式选择 SM110 新 dispatch；`native` 保留原路径。命中条件限定为 biased `(M,K,N)=(41,256,1536)` 和 `(41,1536,1536)`，其他 signature（含 biasless/M=1）沿原路径。

交换 W/X 后 tile 为 `(128,64,256)`；K=256 使用 cluster `(2,1,1)`、两 stages，K=1536 使用 `(4,1,1)`、自动 stages，均为 reference dynamic scheduler。column-major store 直接写原 contiguous `[M,N]` 输出，无额外 transpose kernel。格式、packing、mask 与 attention 保持原合同。

这两个 shape 的选择来自实际权重上的重复 CUDA Graph timing 及数值/完整回放检查；已有配对完整回放未观察到稳定的 policy wall 提升，因此此开关供该 shape 的算子研究。任意输入的 BF16 rounding 可不同，FP4 误差与闭环质量需独立评估。关闭开关恢复原 FP4 dispatch，关闭整个作用域恢复 BF16。

## 验证与证据

验证分为执行合同、真实检查点回放和 LIBERO 闭环三个层次。固定精度的算子一致性解释实现行为；精度间动作误差描述数值变化；任务成功率由[闭环 runner](../docs/libero.md)给出。完整比较固定 checkpoint/input、attention、布局、fusion 与 Graph，并分别报告 bootstrap、vision、plan 和完整调用。steady latency 包含 online activation packing，安装/编译/离线权重打包/capture 单列。

下列测试需在匹配的 CUDA 环境显式运行；公共 CI 使用 `tests/contracts`、元数据和 fake objects 检查接口，不执行 CPU 神经模型：

| 测试文件（`tests/gpu/`）                                                                        | 主要覆盖                                                                        |
| ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `test_prepared_inference_gpu.py`、`test_policy_vlm_gpu.py`                                      | fresh-input ownership、真实输出/RNG、VLM Graph 与生命周期                       |
| `test_image_preparation_gpu.py`                                                                 | 图像/collator 一致性与 ownership；完整动作/RNG 另由 CUDA policy gates 验证      |
| `test_action_graph_gpu.py`                                                                      | Flow4/RNG、K/V freshness、stride/stream、异常及恢复                             |
| `test_groot_fusion_gpu.py`、`test_groot_fusion_modes_gpu.py`、`test_groot_vision_layout_gpu.py` | fusion、组件选择、布局、隔离和恢复                                              |
| `test_quantization_gpu.py`、`test_shape_tactics_gpu.py`                                         | 整数 kernel、prepared/grouped dispatch、变 shape Graph                          |
| `test_groot_streaming_gpu.py`、`test_streaming_component_cache_gpu.py`                          | bootstrap、slide/image delay、reset、eager/Graph 异常清理；前者使用小随机动作头 |
| `test_groot_norm_quant_gpu.py`                                                                  | norm/residual producer、reduction 误差与实际融合 kernel                         |
| `test_floating_quantization_gpu.py`、`test_floating_fusion_gpu.py`、`test_fp4_swap_ab_gpu.py`   | Thor packing、bias、Graph freshness、fallback、恢复                             |

```bash
PIR2_GPU_TESTS=1 python -m pytest -q tests/gpu/test_action_graph_gpu.py
PIR2_NATIVE_TESTS=1 python -m pytest -q tests/native/test_compiled_host_guards.py
```

Spatial streaming 验证使用同一个 πR² checkpoint，服务设 `--variant pir2`，省略 `--group-conditioning`；先比较 eager，再显式加 `--dit-cuda-graph`，每个新 signature 预热。Thor 的已有目标回放条件为 Spatial πR² checkpoint-500、B1/H40、固定 demonstration observations 与 committed prefixes，不能自动外推到其他检查点。

## 闭环报告与辅助工具

`queue_report` 读取真实 LIBERO traces，不加载模型：

```bash
python -m coexecution.queue_report \
  --input .local/experiments/queue-scan --output .local/experiments/queue-report
python -m coexecution.feature_usage \
  --report .local/experiments/queue-report/report-data.json \
  --output .local/experiments/feature-usage --raw-root .local/experiments/queue-scan
```

输入可为单 condition 或 `layout/condition/`，包含 hardware、telemetry、calibration、warmup、episodes 和逐 episode traces。输出有离线 HTML、PNG/SVG、JSON/CSV 与完整 replay。默认排除 smoke/clock-probe/profile，`--include-diagnostics` 纳入诊断预览；`--no-figures` 可免 Matplotlib。可选 Node queue comparison 标为 simulation，字体为本地下载资产并保留许可。

报告的 capacity 来自真实 solo RPC 均值的倒数，时钟来自 telemetry。latency 只统计控制窗口内入场且完成的请求；跨终点请求保留删失标记，未来 tick 的结果不视为 expired。能量只积分控制窗口的 board power，能量计数差值与样本覆盖单列；board energy 包含同卡渲染。成功提前终止的时长、初态 hashes 和未知源保留。

feature usage 依据实际 consumed sequence，而非最近 publication 时间；输出每 feature 的 DiT reads、直接执行 producer、ticks 和零/一/多次消费分布。bootstrap、最终右删失特征单列，unknown 来源使 accounting 标为 incomplete。直接执行计数不覆盖 rolling buffer 中的全部神经因果贡献。

[模型与机器人回放](replay/README.md)统一展示独立模型输入、clean latency、真实 kernel timeline 以及每个精度自己的闭环轨迹。`--replay-assets` 在计时外导出模型输入，`--record-replay` 有界记录机器人状态/双摄像头，控制完成后编码。

SO100 的 H16 BF16 录制观测用于辅助执行机制诊断，入口与分析方式见[辅助 SO100 回放](../docs/coexecution.md#auxiliary-so100-replay)。模型、数据、trace、图表和报告留在忽略目录；源码与许可见[第三方记录](../THIRD_PARTY_NOTICES.md)。
