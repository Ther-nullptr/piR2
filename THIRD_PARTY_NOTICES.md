# Third-party provenance

- **πR² official repository**: https://github.com/pi-r2-flow/pi-r2-flow at `3af52ca400a6ec7d141416879aa531a6f62f697a`. The official parent source is downloaded separately, not redistributed here. Preserve its original attribution.
- **πR² Isaac-GR00T fork**: https://github.com/pi-r2-flow/Isaac-GR00T at `2a39d591a3af24c42bb1d1c9cd708a2dddac0600`, derived from NVIDIA Isaac-GR00T. Apache-2.0; [LICENSE](licenses/ISAAC_GROOT_LICENSE) and [NOTICE](licenses/ISAAC_GROOT_NOTICE) accompany `patches/groot-reproduction-fixes.patch`. The local patch preserves clean prefixes under jitter, aligns warm-start time/delay conditioning, zero-initializes newly missing delay embeddings after Hugging Face loading, and retains padded final action frames. The runtime adapters in `coexecution/groot_pointwise.py` and `coexecution/quantization_connections.py` also follow this fork's `gr00t/model/modules/dit.py` and `embodiment_conditioned_mlp.py`: they replace selected inference forwards while retaining the loaded model weights, attention sequence and native normalization. These are modified adapters, not unmodified upstream files.
- **MuJoCo Playground**: https://github.com/google-deepmind/mujoco_playground at `ef4fefc13033c0468af4ef651847f5348af0c7d7`. Downloaded separately under its own terms.

The project does not redistribute model weights or training datasets. Obtain them from their publishers under the applicable terms. A project-wide license for independently written code has not been selected; third-party licenses do not automatically license the whole repository.


- **Diffusion Policy ConditionalUnet1D**: https://github.com/real-stanford/diffusion_policy. The architecture in `simulation/model.py` is adapted from this MIT-licensed implementation; the original notice is preserved in [licenses/DIFFUSION_POLICY_LICENSE](licenses/DIFFUSION_POLICY_LICENSE). The model is an independent reconstruction, not the unpublished Leap training code.

- **Transformers 4.57.3 Qwen3-VL**: https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py. Copyright 2025 The Qwen Team and The HuggingFace Inc. team. The vision/attention paths in `coexecution/static_s2.py` adapt this Apache-2.0 code to cache static metadata; see [licenses/TRANSFORMERS_LICENSE](licenses/TRANSFORMERS_LICENSE). The RoPE/RMSNorm bindings in `coexecution/groot_fusion.py` and gated-MLP forward in `coexecution/groot_pointwise.py` also follow this pinned implementation, replacing selected operations with local inference kernels. Fusion adapters preserve the underlying source's attribution and compare against that pinned implementation.

## Optional Speedup Paradox integer dependency

The `robotics-kernels` component in [sources.lock.json](sources.lock.json) fetches
[robotics-the-speedup-paradox](https://github.com/thu-ee-acts-lab/robotics-the-speedup-paradox/tree/239b4a3ef2268048571c9f508ad5700f98398c2f)
at `239b4a3ef2268048571c9f508ad5700f98398c2f` into the ignored dependency directory.
No backend source tree or compiled library is redistributed in this repository.
The optional GR00T adapters call its `robotics_kernels.ampere_ada.integer`
(`IntegerLinear`, projection groups and activation packing), `ampere_ada.modulation`,
`common.fused` and `common.graph` modules. The local CUDA dispatch wrapper prepares
activations and invokes the dependency's registered integer GEMM; it does not
replace that GEMM implementation. Local BF16 fusion follows the reference's
native-rounding and activation-lookup approach.

The pinned reference's [third-party notices](https://github.com/thu-ee-acts-lab/robotics-the-speedup-paradox/blob/239b4a3ef2268048571c9f508ad5700f98398c2f/THIRD_PARTY_NOTICES.md)
state that its independently written code has no selected project-wide license.
This integration does not assign one or treat a nested dependency license as
permission to redistribute the whole reference project. Its original notices
remain with the separately obtained checkout.

The integer backend includes unmodified NVIDIA CUTLASS headers at
`982748aa7356fa838c2ea4994ddcb0b2a4b4cefa`, limited to `include/`,
`tools/util/include/` and `LICENSE.txt`. Copyright 2017–2026 NVIDIA CORPORATION &
AFFILIATES; BSD-3-Clause with file-specific notices. The external checkout retains
the original [CUTLASS license](https://github.com/thu-ee-acts-lab/robotics-the-speedup-paradox/blob/239b4a3ef2268048571c9f508ad5700f98398c2f/src/robotics_kernels/ampere_ada/third_party/cutlass/LICENSE.txt)
and [provenance with per-file hashes](https://github.com/thu-ee-acts-lab/robotics-the-speedup-paradox/blob/239b4a3ef2268048571c9f508ad5700f98398c2f/src/robotics_kernels/ampere_ada/third_party/cutlass/PROVENANCE.json).
These terms apply to that header snapshot, not the reference's independent
integer operator or this entire repository. No CUTLASS headers or binaries are
redistributed here.
