# Environment boundaries

Use separate environments for GR00T training/inference, LIBERO simulation, and JAX/MJX Leap simulation. Model computations require CUDA. The source preparation tool downloads Git repositories only; it does not install a working GPU environment or download gated weights.

```bash
python3 tools/prepare_sources.py --component groot
python3 tools/prepare_sources.py --component playground
```

The GR00T environment used Python 3.10, PyTorch 2.7.1, Transformers 4.57.3, Diffusers 0.35.1 and NumPy 1.26.4. Install the pinned fork following its own dependency instructions. The LIBERO environment used Python 3.12, hf-libero 0.1.4, robosuite 1.4.0 and MuJoCo 3.8.1. Leap used a separate CUDA JAX 0.5.3 / MuJoCo-MJX 3.14.0 environment. These are observed compatibility versions, not a claim that all transitive dependencies have been tested on a new machine.

Keep `.venv/`, `.venv-pi05/`, `.venv-sim/`, `.libero-config-pi05/`, weights and datasets local. The `.venv-pi05` name is historical: the current LIBERO evaluator uses it as a simulator environment; it does not imply π0.5 model training.

Set CUDA_VISIBLE_DEVICES explicitly after inspecting GPU occupancy. Supply credentials through the standard local Hugging Face login, never through committed configuration. Host-specific EGL or libexpat fixes belong in the local shell environment, not shared source. GPU evaluation results must state source revisions, checkpoint identity, dataset, initialization and timing protocol.
