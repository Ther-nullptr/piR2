# Environment boundaries

Use separate environments for GR00T training/inference, LIBERO simulation, and JAX/MJX Leap simulation. Model computations require CUDA. The source preparation tool downloads Git repositories only; it does not install a working GPU environment or download gated weights.

```bash
python3 tools/prepare_sources.py --component groot
python3 tools/prepare_sources.py --component playground
```

The GR00T environment used Python 3.10, PyTorch 2.7.1, Transformers 4.57.3, Diffusers 0.35.1 and NumPy 1.26.4. Install the pinned fork following its own dependency instructions. The LIBERO environment used Python 3.12, hf-libero 0.1.4, robosuite 1.4.0 and MuJoCo 3.8.1. Leap used a separate CUDA JAX 0.5.3 / MuJoCo-MJX 3.14.0 environment. These are observed compatibility versions, not a claim that all transitive dependencies have been tested on a new machine.

Keep `.venv/`, `.venv-pi05/`, `.venv-sim/`, `.libero-config-pi05/`, weights and datasets local. The `.venv-pi05` name is historical: the current LIBERO evaluator uses it as a simulator environment; it does not imply π0.5 model training.

Set CUDA_VISIBLE_DEVICES explicitly after inspecting GPU occupancy. Supply credentials through the standard local Hugging Face login, never through committed configuration. Host-specific EGL or libexpat fixes belong in the local shell environment, not shared source. GPU evaluation results must state source revisions, checkpoint identity, dataset, initialization and timing protocol.

The Spatial training and protocol entrypoints construct the Qwen3-VL backbone from
its configuration while loading a complete GR00T task checkpoint. Keep the backbone
configuration and processor assets available; a separate copy of its base weights
is unnecessary. The scoped loader preserves the requested dtype and attention
implementation. Full-checkpoint validation still rejects missing, unexpected or
mismatched weights, apart from the training adapter's explicitly initialized new
streaming parameters. Do not use this loading context for a bare backbone checkpoint.

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

The adapters are opt-in for standard Flow and serial πR² inference, with optional pure-DiT CUDA Graph.
Streaming condition grouping and paired optimized workers remain unsupported.
Synthetic streaming checks do not establish trained-checkpoint or closed-loop quality.
See [execution configuration and validation](../coexecution/README.md). GPU/model checks
remain explicit local commands, not public CI dependencies.

## Optional Thor floating inference

The FP8/FP4 adapter targets NVIDIA Thor (SM110, aarch64). The tested environment
uses JetPack 7.2 / L4T 39.2, CUDA toolkit 13.2, PyTorch `2.10.0+cu130`, Triton 3.6.0,
Python 3.12, Transformers 4.57.3 and Diffusers 0.35.1. It uses SDPA attention.
Keep NumPy 1.26.4 with compatible dependencies; the model environment used
PyArrow 20.0.0, tifffile 2024.9.20 and PyAV 16.1.0. These are observed versions,
not an instruction to upgrade a shared installation.

Use a personal checkout, environment, build directory and caches. CUDA requires
access to the platform GPU device nodes (including `/dev/nvmap` on the tested
machine). A successful SSH login or Torch import does not establish that access.
Have the administrator provide an approved GPU execution context; do not run the
model as root or change shared device permissions from the project setup.

With the pinned complete `robotics-kernels` checkout and a compatible existing
CUDA toolkit, build both backends explicitly for Thor:

```bash
ref="$PWD/third_party/robotics-the-speedup-paradox"
cutlass="$ref/src/robotics_kernels/ampere_ada/third_party/cutlass"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}"
for precision in fp8 fp4; do
  python "$ref/tools/build_blackwell.py" --backend "$precision" --arch 11.0a \
    --cutlass-root "$cutlass" --build-dir "$PWD/.local/build/$precision"
done
export ROBOTICS_CUTLASS_FP8_SO="$PWD/.local/build/fp8/robotics_cutlass_fp8_ext.so"
export ROBOTICS_CUTLASS_FP4_SO="$PWD/.local/build/fp4/robotics_cutlass_fp4_ext.so"
export PYTHONPATH="$PWD:$ref/src:$PWD/upstream/learning/Isaac-GR00T${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas"
export TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
```

Triton 3.6 selects a separate Blackwell assembler; on the tested installation its
bundled binary did not recognize `sm_110a`. The second assembler variable is needed
for the optional fast FP8 packer and uses the already installed compatible
toolkit. Do not edit Triton's vendor files. Build and warmup/capture costs belong
outside steady inference. See [floating formats, flags and tests](../coexecution/README.md#experimental-thor-fp8-and-fp4).
