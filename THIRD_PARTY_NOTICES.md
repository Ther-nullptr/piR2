# Third-party provenance

- **πR² official repository**: https://github.com/pi-r2-flow/pi-r2-flow at `3af52ca400a6ec7d141416879aa531a6f62f697a`. The official parent source is downloaded separately, not redistributed here. Preserve its original attribution.
- **πR² Isaac-GR00T fork**: https://github.com/pi-r2-flow/Isaac-GR00T at `2a39d591a3af24c42bb1d1c9cd708a2dddac0600`, derived from NVIDIA Isaac-GR00T. Apache-2.0; [LICENSE](licenses/ISAAC_GROOT_LICENSE) and [NOTICE](licenses/ISAAC_GROOT_NOTICE) accompany `patches/groot-reproduction-fixes.patch`. The local patch preserves clean prefixes under jitter, aligns warm-start time/delay conditioning, zero-initializes newly missing delay embeddings after Hugging Face loading, and retains padded final action frames.
- **MuJoCo Playground**: https://github.com/google-deepmind/mujoco_playground at `ef4fefc13033c0468af4ef651847f5348af0c7d7`. Downloaded separately under its own terms.

The project does not redistribute model weights or training datasets. Obtain them from their publishers under the applicable terms. A project-wide license for independently written code has not been selected; third-party licenses do not automatically license the whole repository.

- **Diffusion Policy ConditionalUnet1D**: https://github.com/real-stanford/diffusion_policy. The architecture in `simulation/model.py` is adapted from this MIT-licensed implementation; the original notice is preserved in [licenses/DIFFUSION_POLICY_LICENSE](licenses/DIFFUSION_POLICY_LICENSE). The model is an independent reconstruction, not the unpublished Leap training code.
