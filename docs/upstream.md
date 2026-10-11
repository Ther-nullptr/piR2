# Upstream relationship / 与作者代码的关系

`Ther-nullptr/piR2` is an independent experiment repository, not the paper authors' official repository and not a GitHub fork. The runtime directory `upstream/` is a clone of [the official parent repository](https://github.com/pi-r2-flow/pi-r2-flow), whose `learning/Isaac-GR00T` submodule supplies the actual πR²/Flow implementation.

The parent is pinned to `3af52ca400a6ec7d141416879aa531a6f62f697a`, and the GR00T fork to `2a39d591a3af24c42bb1d1c9cd708a2dddac0600`. Our repository owns the experiment entrypoints, LIBERO adaptation, timing/evaluation tooling and explicitly recorded corrections. It does not present the algorithm as newly invented here.

Source locations and commits live in [sources.lock.json](../sources.lock.json). Dependencies are downloaded to ignored directories rather than importing nested `.git` histories or copying complete source trees into commits. Local corrections are tracked as patches with the relevant upstream license and notice; the preparation command checks the exact base and fails on incompatible changes.

The official release focuses on xArm6 + XHand deployment and GR00T training. Our LIBERO experiments and reconstructed Leap data therefore require separate validation; they are not the authors' original experiment assets.

## Low-bit backend reference / 低比特量化参考实现

The optional `robotics-kernels` component pins
[Speedup Paradox](https://github.com/thu-ee-acts-lab/robotics-the-speedup-paradox/tree/239b4a3ef2268048571c9f508ad5700f98398c2f)
to `239b4a3ef2268048571c9f508ad5700f98398c2f`. Fetch it with
`python tools/prepare_sources.py --component robotics-kernels`; the existing
preparation tool verifies the revision and leaves a mismatched checkout untouched.
It has no local patches and remains in ignored
`third_party/robotics-the-speedup-paradox/`.

GR00T-specific bindings, reversible installation and the local integer CUDA
dispatch wrapper live in this repository. INT4/INT8 GEMM, integer
weight/activation packing, modulation packing and the reference graph utility
remain imports from that external source. The Thor adapter calls the same pinned
checkout's `blackwell.fp8_linear` and `blackwell.fp4_linear` CUTLASS operators,
using independent floating packed formats and dispatch choices.

The local Triton producers in `coexecution/floating_packing.py` implement FP8
activation packing and BF16 SwiGLU, normalization, modulation and residual
boundaries that emit FP8/FP4 inputs. `coexecution/floating_connections.py` binds
the normalization and residual producers to DiT. These producers follow the
pinned floating encodings and scale layouts; they do not use integer packed
buffers or replace the external floating GEMM. The dependency remains unmodified.

The bundled CUTLASS snapshot stays inside the dependency with its original
license and hash manifest. Attribution for the integer and floating paths is
recorded in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). See
[integer environment setup](environment.md#optional-gr00t-fusion-and-integer-inference)
and [Thor floating environment setup](environment.md#optional-thor-floating-inference)
for their separate build and runtime requirements.

这里复用参考实现的量化计算路径，不把它描述为新量化算法，也不将参考项目的任务成功率
视为 GR00T 或 πR² 的质量结论。当前接入范围为标准 Flow 及串行 πR²；真实动作头的
随机权重滚动测试不等于训练后检查点或 LIBERO 闭环已验证。性能和质量报告必须注明真实模型、检查点、
输入、执行路径与测量范围。
