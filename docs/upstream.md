# Upstream relationship / 与作者代码的关系

`Ther-nullptr/piR2` is an independent experiment repository, not the paper authors' official repository and not a GitHub fork. The runtime directory `upstream/` is a clone of [the official parent repository](https://github.com/pi-r2-flow/pi-r2-flow), whose `learning/Isaac-GR00T` submodule supplies the actual πR²/Flow implementation.

The parent is pinned to `3af52ca400a6ec7d141416879aa531a6f62f697a`, and the GR00T fork to `2a39d591a3af24c42bb1d1c9cd708a2dddac0600`. Our repository owns the experiment entrypoints, LIBERO adaptation, timing/evaluation tooling and explicitly recorded corrections. It does not present the algorithm as newly invented here.

Source locations and commits live in [sources.lock.json](../sources.lock.json). Model and simulation dependencies are downloaded into ignored directories without nested Git history. The operator sources used at runtime are owned under `coexecution/kernels/`, with explicit origin revisions, file hashes and retained notices. GR00T corrections remain patches against its pinned model source; kernel updates modify the owned implementation directly.

The official release focuses on xArm6 + XHand deployment and GR00T training. Our LIBERO experiments and reconstructed Leap data therefore require separate validation; they are not the authors' original experiment assets.

## Low-bit backend reference / 低比特量化参考实现

The `robotics-kernels` origin is [Speedup Paradox](https://github.com/thu-ee-acts-lab/robotics-the-speedup-paradox/tree/239b4a3ef2268048571c9f508ad5700f98398c2f)
at `239b4a3ef2268048571c9f508ad5700f98398c2f`. The selected integer/common
runtime, CUDA GEMMs, Thor FP8/FP4 kernels and pinned CUTLASS headers now live in
[coexecution/kernels](../coexecution/kernels/). Their source manifest distinguishes
original upstream bytes from repository adaptations. The preparation command
`python tools/prepare_sources.py --component robotics-kernels` verifies these
owned files; it does not fetch an external runtime checkout or apply patches.

[PR #21](https://github.com/Ther-nullptr/piR2/pull/21)'s two biased M=41 FP4
Swap-AB configurations are implemented directly in the owned FP4 CUDA source.
The `--fp4-swap-ab` flag and exact-shape fallback remain available; the unified
operator profile also accepts `floating_gemm=swap_ab`. No external repository
patch or prebuilt-library path is needed. This relocation does not establish new
hardware performance or closed-loop quality.

The model adapters and floating Triton producers remain in
`coexecution/optimization/`, separate from hardware kernels. Integer and floating
formats retain distinct contracts. Source revisions and notices are recorded in
[sources.lock.json](../sources.lock.json) and
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). See
[integer setup](environment.md#optional-gr00t-fusion-and-integer-inference) and
[Thor setup](environment.md#optional-thor-floating-inference) for toolchain and
hardware requirements.

这里复用参考实现的量化计算路径，不把它描述为新量化算法，也不将参考项目的任务成功率
视为 GR00T 或 πR² 的质量结论。当前接入范围为标准 Flow 及串行 πR²；真实动作头的
随机权重滚动测试不等于训练后检查点或 LIBERO 闭环已验证。性能和质量报告必须注明真实模型、检查点、
输入、执行路径与测量范围。
