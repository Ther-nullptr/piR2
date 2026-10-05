# πR² reproduction and experiments

这是基于 [πR² 官方实现](https://github.com/pi-r2-flow/pi-r2-flow) 的复现与实验扩展仓库，由本仓维护者独立维护，不是论文作者的官方发布，也不是 GitHub fork。

This independently maintained repository builds on the official πR² implementation. It separates upstream source revisions and local fixes from experiment code. Local results are not automatically claims of reproducing the paper.

## Scope / 范围

- [Leap GPU simulation reconstruction](simulation/README.md), with explicit differences from the paper's data and experts.
- [LIBERO-Spatial adaptation](docs/libero.md) of public GR00T task weights and real VLM closed-loop evaluation.
- Separate [fixed-delay algorithm and wall-clock deployment comparisons](docs/timing-protocols.md).
- Single-GPU S1/S2 inference tools and a browser teaching simulator, reported separately from robot task success.

Modules are introduced through focused pull requests. Model weights, datasets, environments and generated results are intentionally excluded from Git.

## Start / 开始

Read [source provenance](docs/upstream.md), [environment boundaries](docs/environment.md) and [contribution rules](CONTRIBUTING.md). Fetch pinned upstream source only when needed:

```bash
python3 tools/prepare_sources.py --component groot
```

Model training and inference require CUDA. Lightweight CI checks repository hygiene and scheduling/accounting logic; it never evaluates a model on CPU, downloads model weights or uses laboratory GPU runners.

## Paper / 论文

Sungjae Park and Shubham Tulsiani, *πR²: Reactive Real-time Flow Policies*, 2026. [Paper](https://arxiv.org/abs/2607.26055) · [Official implementation](https://github.com/pi-r2-flow/pi-r2-flow).
