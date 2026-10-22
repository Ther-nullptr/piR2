# Third-party provenance

- **SmoothQuant**: https://github.com/mit-han-lab/smoothquant at
  `c61476d728e42ae0d8a35e7e78494edcac3237b5`, copyright 2022 MIT HAN Lab.
  The per-input-channel absmax statistic, shared-consumer weight maximum and
  alpha balancing formula in `coexecution/optimization/backends/thor/smoothquant.py`
  adapt `smoothquant/calibration.py` and `smoothquant/smooth.py` under the
  [retained MIT license](licenses/SMOOTHQUANT_LICENSE). The Thor implementation
  retains native FP8/FP4 GEMMs and fuses input compensation into local packing
  producers; it does not use upstream INT8 model classes or assert that the
  upstream LLM accuracy results transfer to GR00T or FP4.
  The local GELU, RMSNorm, AdaLN, residual and SwiGLU producer adapters in
  `coexecution/optimization/backends/thor/packing.py` preserve the explicit
  BF16 rounding boundaries before smoothing and packing. Their installation
  is implemented in `coexecution/optimization/groot/floating.py`; no external
  SmoothQuant runtime or robotics-kernels checkout is required.

- **πR² official repository**: https://github.com/pi-r2-flow/pi-r2-flow at `3af52ca400a6ec7d141416879aa531a6f62f697a`. The official parent source is downloaded separately, not redistributed here. Preserve its original attribution.
- **πR² Isaac-GR00T fork**: https://github.com/pi-r2-flow/Isaac-GR00T at `2a39d591a3af24c42bb1d1c9cd708a2dddac0600`, derived from NVIDIA Isaac-GR00T. Apache-2.0; [LICENSE](licenses/ISAAC_GROOT_LICENSE) and [NOTICE](licenses/ISAAC_GROOT_NOTICE) accompany `patches/groot-reproduction-fixes.patch`. The local patch preserves clean prefixes under jitter, aligns warm-start time/delay conditioning, zero-initializes newly missing delay embeddings after Hugging Face loading, and retains padded final action frames. The runtime adapters in `coexecution/groot_pointwise.py` and `coexecution/quantization_connections.py` also follow this fork's `gr00t/model/modules/dit.py` and `embodiment_conditioned_mlp.py`: they replace selected inference forwards while retaining the loaded model weights, attention sequence and native normalization. These are modified adapters, not unmodified upstream files.
- **MuJoCo Playground**: https://github.com/google-deepmind/mujoco_playground at `ef4fefc13033c0468af4ef651847f5348af0c7d7`. Downloaded separately under its own terms.
- **LIBERO-Dynamic**: https://github.com/renshaojie233/LIBERO-Dynamic at `1e5845720643da281c47bf4423ae99e115976a13`, MIT license inherited from LIBERO. Downloaded separately into the ignored `third_party/libero_dynamic` directory; preserve its original LICENSE and task/asset notices. The reviewed motion configurations and physical carrier implementations remain upstream files. This repository owns the benchmark selection, GR00T integration and experiment accounting; it does not redistribute dynamic demonstrations or adopt the paper's fine-tuning recipe.


The project does not redistribute model weights or training datasets. Obtain them from their publishers under the applicable terms. A project-wide license for independently written code has not been selected; third-party licenses do not automatically license the whole repository.


- **Diffusion Policy ConditionalUnet1D**: https://github.com/real-stanford/diffusion_policy. The architecture in `simulation/model.py` is adapted from this MIT-licensed implementation; the original notice is preserved in [licenses/DIFFUSION_POLICY_LICENSE](licenses/DIFFUSION_POLICY_LICENSE). The model is an independent reconstruction, not the unpublished Leap training code.

- **Transformers 4.57.3 Qwen3-VL**: https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py. Copyright 2025 The Qwen Team and The HuggingFace Inc. team. The vision/attention paths in `coexecution/static_s2.py` adapt this Apache-2.0 code to cache static metadata; see [licenses/TRANSFORMERS_LICENSE](licenses/TRANSFORMERS_LICENSE). The RoPE/RMSNorm bindings in `coexecution/groot_fusion.py` and gated-MLP forward in `coexecution/groot_pointwise.py` also follow this pinned implementation, replacing selected operations with local inference kernels. Fusion adapters preserve the underlying source's attribution and compare against that pinned implementation.

- **GR00T image preparation**: `coexecution/optimization/execution/images.py` clones the pinned fork's `Gr00tN1d7Processor._apply_vlm_processing` at runtime with an instance-local image conversion binding. It retains that original function's conversation and chat-template code and the pinned GR00T/Qwen collators. No upstream source file is changed or copied into the repository; the original Apache-2.0 attribution above applies.

- **Optional native host-guard build tools**: Cython 3.1.2 and the tested setuptools 84.0.0 are recorded in [sources.lock.json](sources.lock.json). These packages are installed separately under their own licenses. `coexecution/optimization/execution/compiled_guards.py` compiles this repository's exact Python guard source at opt-in startup; compiler packages, generated C, native binaries and build logs are not redistributed. Runtime evidence records the actual toolchain and source/binary hashes.

- **Source Han Sans / 思源黑体**: https://github.com/adobe-fonts/source-han-sans at `a4f7cf94edfb9d7ffbdfc4841de276358bd7e0f2`. Copyright 2014–2025 Adobe; SIL Open Font License 1.1, Reserved Font Name “Source”. [License](licenses/SOURCE_HAN_SANS_LICENSE). The unmodified CN variable WOFF2 is an optional separately downloaded local asset; its SHA-256 is recorded in `sources.lock.json`. Offline reports embed the font and its license when present.

## Repository-owned fusion and low-bit operators

The `robotics-kernels` record in [sources.lock.json](sources.lock.json) retains
[robotics-the-speedup-paradox](https://github.com/thu-ee-acts-lab/robotics-the-speedup-paradox/tree/239b4a3ef2268048571c9f508ad5700f98398c2f)
at `239b4a3ef2268048571c9f508ad5700f98398c2f` as its original source. Its mode is
`vendored`: selected runtime modules and CUDA translation units now reside in
[coexecution/kernels](coexecution/kernels/). No separate robotics runtime checkout
or prebuilt robotics shared library is used. The general CUTLASS header dependency
can be selected independently as described below.
[The owned source manifest](coexecution/kernels/SOURCE_MANIFEST.json) records the
original path, revision and SHA-256 separately from each adapted file's SHA-256.
`prepare_sources.py --component robotics-kernels` verifies this manifest without
cloning, resetting, compiling or replacing local files.

The selected copies comprise Ampere/Ada integer packing and projection modules,
AdaLN/norm/residual producers, common fusion and graph primitives, integer
CUTLASS GEMM, and the Blackwell FP8/FP4 CUDA translation units and sampling header.
Imports, build paths, registered operator namespaces and focused floating loaders
are adapted to `coexecution.kernels` and `pir2_*`; original numerical kernels are
retained. CUDA Toolkit, PyTorch and Triton remain installed framework dependencies.
Compiled extensions, weights, generated evidence and runtime caches are excluded.

The reference's [third-party notices](https://github.com/thu-ee-acts-lab/robotics-the-speedup-paradox/blob/239b4a3ef2268048571c9f508ad5700f98398c2f/THIRD_PARTY_NOTICES.md)
state that its independently written integer/common code has no selected
project-wide license. That statement is preserved in
[UPSTREAM_NOTICE.txt](coexecution/kernels/UPSTREAM_NOTICE.txt). Source ownership
and attribution do not assign a new license or treat a nested dependency license
as an unrestricted license for the imported reference code or this repository.

The official [NVIDIA/CUTLASS repository](https://github.com/NVIDIA/cutlass) is a
Git submodule at `coexecution/kernels/vendor/cutlass`, with piR2's Git link
recording `982748aa7356fa838c2ea4994ddcb0b2a4b4cefa` as the reproducible default.
This is the same revision previously used for the copied headers. CUTLASS files
are supplied by the submodule or a compatible `CUTLASS_ROOT` installation rather
than duplicated as piR2-tracked files. The owned manifest retains the default
submodule path and revision; runtime provenance records the source actually
selected, including its Git revision when available. Integer and floating CUDA
builds require a complete compatible `include/` tree and `tools/util/include/`.
Exact SHA equality with the default is not a compatibility requirement.
Copyright 2017–2026 NVIDIA CORPORATION & AFFILIATES; the original
[BSD-3-Clause license](https://github.com/NVIDIA/cutlass/blob/982748aa7356fa838c2ea4994ddcb0b2a4b4cefa/LICENSE.txt)
and file-specific notices remain intact. These terms cover CUTLASS, not the
reference's independently written integer operator or the entire piR2 repository.
Preserve the notices supplied with any alternate compatible installation.
See [CUTLASS setup](docs/environment.md#cutlass-submodule) for the default
submodule, direct reuse of complete headers, compatible local revisions and
optional Git object-cache reuse. CUTLASS APIs and the CUDA/C++ toolchain still
need to be compatible; selecting a different root is not a claim of validation.

Blackwell sources retain the [MIT license](coexecution/kernels/blackwell/LICENSE),
copyright 2025 MIT HAN Lab, and their preceding
[upstream provenance](coexecution/kernels/blackwell/UPSTREAM_PROVENANCE.json).
That provenance identifies
`streaming-inference-infra-code/vlm/streaming_vlm_kernels` at
`1c6f8b8ac31c49e401ee267647b3f967725cd26e`; the owned manifest separately identifies
the selected files as obtained from Speedup Paradox revision `239b4a3`.
The preceding pending-hardware-validation notice remains part of the source
history. Moving the files and changing their namespace is not a new device,
model-quality or performance result.

The local Triton producers in `coexecution/optimization/backends/thor/packing.py`
follow the pinned source's per-tensor E4M3 convention and block-16 E2M1/UE4M3
encoding with the CUTLASS `sfa_offset` scale layout. Their LayerNorm, modulation
and residual sequencing follows `ampere_ada/modulation.py`, adapted to floating
formats without integer packed buffers. Native BF16 SiLU lookup follows
`common/fused.py` and retains the original activation-rounding boundary.
The connection adapters remain modified inference forwards derived from the
pinned GR00T/Qwen sources described above.

The optional Ada CUDA candidates were consolidated from piR2's integrated
fastpath revision `6f64bbe0c71d6c73c8b0e00964b037e69ae92abf`; their original and
adapted hashes are in the owned manifest. `layernorm_pack.cu` derives its
vectorized Welford reduction from PyTorch 2.7.1
[`layer_norm_kernel.cu`](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/cuda/layer_norm_kernel.cu).
`rms_statistics.cu` instantiates the installed, SHA-256-verified PyTorch 2.7.1
[`Reduce.cuh`](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/cuda/Reduce.cuh)
under a pinned runtime contract. Both retain the original
[PyTorch license](licenses/PYTORCH_LICENSE). The grouped output visitor in
`integer_epilogue.cu` derives from the pinned CUTLASS
[`visitor_store.hpp`](https://github.com/NVIDIA/cutlass/blob/982748aa7356fa838c2ea4994ddcb0b2a4b4cefa/include/cutlass/epilogue/threadblock/fusion/visitor_store.hpp)
and retains its CUTLASS notice. Exact upstream hashes and license paths are also
recorded under `adapted_sources` in the lock file. These candidates remain
explicit opt-ins with guarded fallback; this organization does not establish a
new speedup or LIBERO closed-loop result.

## Owned Thor FP4 Swap-AB implementation

The implementation introduced by [piR2 PR #21](https://github.com/Ther-nullptr/piR2/pull/21)
at `4147186b973fb85b32045208b9ba4028342ceacb` is integrated directly into
`coexecution/kernels/blackwell/csrc/fp4_cutlass_linear.cu`. It adapts the pinned
Blackwell `fp4_mx_cutlass_linear.cu::mx_swap_gemm` layout and bias epilogue to
NVFP4 on both operands. Copyright 2025 MIT HAN Lab and the original
[MIT license](coexecution/kernels/blackwell/LICENSE) remain with the source.
The former external dependency patch is no longer applied or required. The
source manifest records this additional adaptation without replacing the
original Blackwell file provenance.
