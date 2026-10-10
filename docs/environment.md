# Environment boundaries

Use separate environments for GR00T training/inference, LIBERO simulation, and JAX/MJX Leap simulation. Model computations require CUDA. The source preparation tool downloads Git repositories only; it does not install a working GPU environment or download gated weights.

```bash
python3 tools/prepare_sources.py --component groot
python3 tools/prepare_sources.py --component playground
```

The GR00T environment used Python 3.10, PyTorch 2.7.1, Transformers 4.57.3, Diffusers 0.35.1 and NumPy 1.26.4. Install the pinned fork following its own dependency instructions. The LIBERO environment used Python 3.12, hf-libero 0.1.4, robosuite 1.4.0 and MuJoCo 3.8.1. Leap used a separate CUDA JAX 0.5.3 / MuJoCo-MJX 3.14.0 environment. These are observed compatibility versions, not a claim that all transitive dependencies have been tested on a new machine.

Keep `.venv/`, `.venv-pi05/`, `.venv-sim/`, `.libero-config-pi05/`, weights and datasets local. The `.venv-pi05` name is historical: the current LIBERO evaluator uses it as a simulator environment; it does not imply π0.5 model training.

Set CUDA_VISIBLE_DEVICES explicitly after inspecting GPU occupancy. Supply credentials through the standard local Hugging Face login, never through committed configuration. Host-specific EGL or libexpat fixes belong in the local shell environment, not shared source. GPU evaluation results must state source revisions, checkpoint identity, dataset, initialization and timing protocol.

## Optional GR00T fusion and integer inference

Use a separate Linux CUDA model environment for the experimental GR00T fusion and
W8A8/W4A4 adapters. The measured development configuration used an NVIDIA RTX 6000
Ada (SM89), PyTorch `2.9.0+cu128` and CUDA 12.8, with BF16 model weights. This is
separate from the PyTorch 2.7.1 training/inference environment above; compatibility
with that environment has not been established. Keep the pinned GR00T source and
its model dependencies, including Transformers 4.57.3 and Diffusers 0.35.1. Do not
upgrade the working model environment merely to install the optional backend.

The extension requires a CUDA toolkit with `nvcc`, a compatible C++17 compiler,
Ninja, and the Torch-compatible Triton package. Compilation happens on first
use and belongs outside timed inference. The reference uses SM80-compatible
integer kernels on Ampere/Ada; the GR00T integration's recorded GPU validation is
on Ada, not a claim of validation on every supported architecture.

From the repository root, with the intended model environment activated:

```bash
python tools/prepare_sources.py --component robotics-kernels
export PYTHONPATH="$PWD:$PWD/third_party/robotics-the-speedup-paradox/src:$PWD/upstream/learning/Isaac-GR00T${PYTHONPATH:+:$PYTHONPATH}"
```

This source-path setup preserves the packaged CUDA sources and CUTLASS headers.
Alternatively, install the same checkout as an editable package in that model
environment, retaining the GR00T source path:

```bash
python -m pip install --no-deps --no-build-isolation -e third_party/robotics-the-speedup-paradox
```

The pinned `pyproject.toml` names the distribution `robotics-inference-bench`,
requires Python >=3.10 and setuptools >=68, and declares no runtime dependencies.
Neither command installs or replaces Torch, Triton, CUDA or the model packages;
install compatible prerequisites deliberately in the chosen environment. Keep
the editable dependency checkout in place: its CUDA build reads the bundled
headers and provenance at runtime. Review its [source and licensing boundaries](../THIRD_PARTY_NOTICES.md).

The adapters are opt-in and currently scoped to the standard Flow entrypoint;
they do not establish streaming πR² compatibility or closed-loop task quality.
See [execution configuration and validation](../coexecution/README.md). GPU/model checks
remain explicit local commands, not public CI dependencies.
