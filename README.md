# GR00T-N1.7-LIBERO: πR² and real-time control experiments

这是基于 [πR² 官方实现](https://github.com/pi-r2-flow/pi-r2-flow) 的复现与实验扩展仓库，由本仓维护者独立维护，不是论文作者的官方发布，也不是 GitHub fork。

This independently maintained repository studies **NVIDIA GR00T-N1.7-LIBERO** task checkpoints, πR² adaptation, real VLM closed-loop control and deployment efficiency. It separates upstream source revisions and local fixes from experiment code. Local results are not automatically claims of reproducing the paper.

主要研究对象为 [官方 GR00T-N1.7-LIBERO 权重](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO)，常规基线以 [NVIDIA LIBERO 示例](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/LIBERO/README.md) 为参考。先从已适配任务的权重建立基线，再比较 πR²、时序协议与部署优化；不默认重做通用基座的任务微调。

## Scope / 范围

- [LIBERO-Spatial adaptation](docs/libero.md) of public GR00T task weights and real VLM closed-loop evaluation.
- Separate [fixed-delay algorithm and wall-clock deployment comparisons](docs/timing-protocols.md).
- Four-suite research targets are recorded in [the model-family manifest](configs/libero/model-family.json); the current implemented training/evaluation pipeline covers the complete Spatial suite.
- [Leap reconstruction](simulation/README.md), SO100 profiling and browser teaching tools are auxiliary historical work, reported separately from the GR00T-LIBERO main line.
- [Single-GPU execution tools](coexecution/README.md) support auxiliary SO100 replay profiling; their mechanisms require separate LIBERO validation before contributing main-line performance claims.
- [Interactive timing tool](simulator/README.md) explains queueing and cache-age assumptions without predicting task success.

Modules are introduced through focused pull requests. Model weights, datasets, environments and generated results are intentionally excluded from Git.

## Start / 开始

Read [source provenance](docs/upstream.md), [environment boundaries](docs/environment.md) and [contribution rules](CONTRIBUTING.md). New users can initialize the default official CUTLASS dependency with the clone:

```bash
git clone --recurse-submodules https://github.com/Ther-nullptr/piR2.git
cd piR2
```

For an existing piR2 checkout using the default CUTLASS version, run
`git submodule update --init --recursive coexecution/kernels/vendor/cutlass`
after pulling or switching branches. The Git link records
`982748aa7356fa838c2ea4994ddcb0b2a4b4cefa` as the reproducible default. Authors may
instead set `CUTLASS_ROOT=/path/to/cutlass` to use an existing compatible full
header tree, or keep a compatible local submodule revision; the exact SHA need
not match. Actual CUDA operator implementations remain in this repository.
[Setup instructions](docs/environment.md#cutlass-submodule) describe the required
header structure, toolchain compatibility and optional Git object-cache reuse.
Fetch the separately pinned model source when needed:

```bash
python3 tools/prepare_sources.py --component groot
```

Model training and inference require CUDA. Lightweight CI checks repository hygiene and scheduling/accounting logic; it never evaluates a model on CPU, downloads model weights or uses laboratory GPU runners.

## Paper / 论文

Sungjae Park and Shubham Tulsiani, *πR²: Reactive Real-time Flow Policies*, 2026. [Paper](https://arxiv.org/abs/2607.26055) · [Official implementation](https://github.com/pi-r2-flow/pi-r2-flow).
