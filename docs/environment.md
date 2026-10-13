# Environment boundaries

Use separate environments for GR00T training/inference, LIBERO simulation, and JAX/MJX Leap simulation. Model computations require CUDA. The source preparation tool downloads pinned Git repositories or verifies repository-owned source manifests; it does not install a GPU environment or download gated weights.

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

## CUTLASS submodule

The CUDA implementations remain ordinary source files under
`coexecution/kernels`. Their general CUTLASS dependency can come from the official
[NVIDIA/CUTLASS](https://github.com/NVIDIA/cutlass) submodule or an existing
compatible header tree. The submodule lives at
`coexecution/kernels/vendor/cutlass`; its Git link and source metadata record
`982748aa7356fa838c2ea4994ddcb0b2a4b4cefa` as the default reproducible version.
Other authors do not have to use that exact SHA. The selected headers must
provide the APIs used by these kernels and work with the installed CUDA/C++
toolchain; this does not promise compatibility with every CUTLASS release.

For a new environment, use the default submodule:

```bash
git clone --recurse-submodules https://github.com/Ther-nullptr/piR2.git
cd piR2
```

For an existing piR2 clone using the default dependency, initialize it from the
repository root:

```bash
git submodule update --init --recursive coexecution/kernels/vendor/cutlass
```

Authors with an existing CUTLASS clone or complete compatible header installation
can select it directly:

```bash
export CUTLASS_ROOT=/path/to/cutlass
```

`CUTLASS_ROOT` takes precedence over the submodule. It points at the CUTLASS root,
with the complete required `include/` tree (including `include/cutlass` and
`include/cute`) and `tools/util/include/`. A headers-only installation is usable
when that structure and the needed APIs are present; Git metadata is optional.
The tools record the actual selected source and Git revision when available.
They do not reject a dependency solely because its SHA differs from the default
pin. This environment path selects the CUTLASS dependency; piR2's CUDA operator
implementations continue to build from this repository. Unset `CUTLASS_ROOT` to
return to the local submodule.

You may also keep a compatible local revision checked out inside the submodule.
Record that actual revision with the experiment, and leave personal Git-link
changes out of unrelated piR2 commits. After pulling or switching piR2 branches,
run the submodule update command when you want that branch's default dependency;
it will select the recorded Git link. Preserve local work and any intentional
version choice before doing so. Routine default setup does not need `--remote`,
which follows a remote branch tip instead. Updating the project's shared default
pin remains a separate reviewed change to the Git link and dependency metadata.

An existing independent CUTLASS Git clone can alternatively serve as a download
cache during first initialization of the default submodule:

```bash
git submodule update --init --recursive \
  --reference /path/to/existing/cutlass --dissociate \
  coexecution/kernels/vendor/cutlass
```

`--reference` reuses available Git objects to reduce network transfer;
`--dissociate` makes the needed local copies so the new submodule stops borrowing
objects afterward. Missing objects are fetched from the configured official
remote. This optional optimization leaves the existing clone unchanged and
applies only when cloning the submodule. A headers-only installation has no Git
object cache, but can still be selected directly with `CUTLASS_ROOT` as above.

## Optional GR00T fusion and integer inference

Use a separate Linux CUDA model environment for the experimental GR00T fusion and
W8A8/W4A4 adapters. The measured development configuration used an NVIDIA RTX 6000
Ada (SM89), PyTorch `2.9.0+cu128` and CUDA 12.8, with BF16 model weights. This is
separate from the PyTorch 2.7.1 training/inference environment above; compatibility
with that environment has not been established. Keep the pinned GR00T source and
its model dependencies, including Transformers 4.57.3 and Diffusers 0.35.1. Do not
upgrade the working model environment merely to install the optional backend.

The extension requires a CUDA toolkit with `nvcc`, a compatible C++17 compiler,
Ninja, and the Torch-compatible Triton package. Explicit builds or the first
opted-in backend load use Torch's extension cache;
compilation belongs outside timed inference. The integer source uses SM80-compatible
integer kernels on Ampere/Ada; the GR00T integration's recorded GPU validation is
on Ada, not a claim of validation on every supported architecture.

From the repository root, with the intended model environment activated, the
default-submodule setup is:

```bash
git submodule update --init --recursive coexecution/kernels/vendor/cutlass
python tools/prepare_sources.py --component robotics-kernels
export PYTHONPATH="$PWD:$PWD/upstream/learning/Isaac-GR00T${PYTHONPATH:+:$PYTHONPATH}"
python -m coexecution.kernels.build --help
python -m coexecution.kernels.build --backend integer --describe
```

The Git command initializes the default CUTLASS checkout. When using an existing
compatible installation, set `CUTLASS_ROOT` and omit that Git command. The
preparation command verifies owned operator sources and records the selected
CUTLASS dependency; it does not fetch, reset or overwrite either source tree.
Make one usable CUTLASS root available before requesting build-input metadata
or compiling.
`--help`, `--describe` and the operator inventory are CPU-only metadata access and
do not import Torch or Triton. CUDA implementations are resolved inside the
repository, CUTLASS comes from the selected dependency root, and runtime adapters
import `coexecution.kernels` directly. No
`robotics-inference-bench` installation, external robotics source path or
`ROBOTICS_CUTLASS_*_SO` setting is needed or read by the owned loaders.

Build the selected native backends before timed inference or Graph capture:

```bash
export TORCH_EXTENSIONS_DIR="$PWD/.local/torch-extensions"
python -m coexecution.kernels.build --backend integer
python -m coexecution.kernels.build --backend integer_dispatch
python -m coexecution.kernels.build --backend bf16_fusion
```

The two metadata commands above can run without CUDA. These three build commands
require the intended CUDA toolkit and model environment; they do not execute a
model or choose an optimization profile. The default CUTLASS submodule or the
selected compatible `CUTLASS_ROOT` supplies headers under its original terms.
Review the
[source and licensing boundaries](../THIRD_PARTY_NOTICES.md), including the
upstream integer/common code's unselected project license.

Additional native Ada candidates are optional and restricted to SM89. Their
shape/layout guards retain the reference path when ineligible. Native LayerNorm
packing additionally requires PyTorch 2.7.1; native RMS statistics requires
`2.7.1+cu126`, the recorded Torch Git revision and the exact installed reduction
header hash. These narrow contracts do not establish compatibility for all
candidates in the historical PyTorch 2.9 environment. Use the
[operator inventory and explicit profiles](../coexecution/README.md#owned-operators-and-explicit-profiles)
to inspect each implementation before enabling it; organization has not added
new measured speedups or task-quality evidence.

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

With the owned sources and an existing compatible CUDA toolkit, build the
floating backends explicitly. The build definitions target Thor `sm_110a` and
use the same CUTLASS root-selection rules. The default-submodule setup is:

```bash
git submodule update --init --recursive coexecution/kernels/vendor/cutlass
python tools/prepare_sources.py --component robotics-kernels
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}"
export TORCH_EXTENSIONS_DIR="$PWD/.local/torch-extensions"
export PYTHONPATH="$PWD:$PWD/upstream/learning/Isaac-GR00T${PYTHONPATH:+:$PYTHONPATH}"
python -m coexecution.kernels.build --backend fp8
python -m coexecution.kernels.build --backend fp4
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas"
export TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"
```

An existing compatible `CUTLASS_ROOT` can replace the initialization command
above. The loaders build this repository's operators and do not load an external
prebuilt robotics shared library. The runtime's `require_fp8_backend` /
`require_fp4_backend` loaders validate SM110 before an
on-demand build. Metadata access remains GPU-free. The retained decode-policy
settings use `PIR2_FP8_DECODE_POLICY` / `PIR2_FP4_DECODE_POLICY` (default `auto`),
with `PIR2_FP8_SWAP_AB=1` by default; ordinary GR00T projection tactics remain
explicit in the Thor adapter.

The optional `--fp4-swap-ab` path is implemented directly in the owned
`coexecution/kernels/blackwell/csrc/fp4_cutlass_linear.cu`, including its new
registered operator. Rebuild the local `fp4` backend after updating this source;
there is no external patch to apply. The two exact biased M=41 signatures retain
their explicit opt-in dispatch; other signatures use the existing FP4 path.

Triton 3.6 selects a separate Blackwell assembler; on the tested installation its
bundled binary did not recognize `sm_110a`. The second assembler variable is needed
for the optional fast FP8 packer and uses the already installed compatible
toolkit. Do not edit Triton's vendor files. Build and warmup/capture costs belong
outside steady inference. See [floating formats, flags and tests](../coexecution/README.md#experimental-thor-fp8-and-fp4).
