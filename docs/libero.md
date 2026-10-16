# LIBERO-Spatial training and real VLM evaluation

Use the official `nvidia/GR00T-N1.7-LIBERO` Spatial task checkpoint and all 432 demonstrations / 52,970 frames across ten Spatial tasks. This is GR00T adaptation, not π0.5 training. The VLM backbone remains frozen in training and active on real simulator images in evaluation.

The repository's main research references are the [NVIDIA task-weight repository](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO) and [official LIBERO example](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/LIBERO/README.md). The model-family manifest records Spatial, Object, Goal and Long separately. Current scripts implement Spatial only; extending a suite requires the matching checkpoint, data, initialization and full task evaluation.

For the ordinary NVIDIA baseline, the official example loads the local suite subdirectory, uses `LIBERO_PANDA` and `--use-sim-policy-wrapper`, and illustrates `--n-action-steps 8`. Our fixed-delay and wall-clock protocols intentionally change execution timing; their scores are separate from that ordinary baseline. The πR² author fork supplies the algorithm changes, not NVIDIA's published fine-tuned weights. Pin the actual source and model versions instead of silently tracking a moving `main` branch.

## Assets and environment

Prepare and patch GR00T source with `python3 tools/prepare_sources.py --component groot`. Follow its installation instructions in a Python 3.10 CUDA environment at `.venv/`. Model access may require accepting the publisher's terms and logging into Hugging Face locally. Then:

```bash
.venv/bin/python scripts/fetch_groot_spatial.py
```

The downloader fixes both model and dataset revisions and copies the official modality configuration. It does not download every suite or training optimizer state. Cosmos configuration and processor assets must also be available through the authorized local cache before offline serving. The training and protocol entrypoints load backbone weights from the complete task checkpoint; see the [loading boundary](environment.md).

Create a separate Python 3.12 simulator environment at `.venv-pi05/` with hf-libero 0.1.4, robosuite 1.4.0, MuJoCo 3.8.1, NumPy, PyAV, Pillow, pyzmq and msgpack. Prepare LIBERO assets through its normal package instructions, then create the isolated local config:

```bash
.venv-pi05/bin/python tools/prepare_libero_config.py --assets ~/.cache/libero/assets
```

The policy server uses `.venv`; the simulator client uses `.venv-pi05` and the pinned GR00T source via PYTHONPATH. Set EGL to the selected physical rendering GPU. If the host requires a libexpat preload, set `PIR2_LIBERO_LD_PRELOAD` to a local compatible library; no user's home directory is baked into the runner.

## Runtime modules / 运行时模块

`coexecution/libero/loading.py` owns scoped task-checkpoint backbone construction;
`backend.py` owns feature-cache installation, rolling-buffer updates and each
worker's private CUDA stream; `service.py` owns strict checkpoint loading, endpoint
registration and worker supervision. Their imports use repository-qualified module
paths. The `scripts/libero_model_loading.py`, `libero_inference_backend.py` and
`serve_libero_protocol.py` entrypoints retain their prior imports and command line;
direct script calls resolve the repository root without adding `scripts/` to
`PYTHONPATH`. Keep the pinned GR00T source and environment settings described above.

Evaluation provenance hashes both the compatibility entrypoints and the canonical
runtime implementations. This source-tracking update does not change scheduling,
episode accounting or protocol semantics. Lightweight contracts live in
`tests/contracts/`; they use import checks and doubles rather than executing models.
Each policy requires serial submissions. Separate processes remain the default;
`--server-layout paired` also supports independent action/VLM policies in one
process with shared cold-capture coordination. Paired whole-action graphs remain
unsupported. See [feature transport and placement](timing-protocols.md) for the
optional bounded `cuda_local` handoff and matched paired/split NumPy controls.

`coexecution/libero/measurement.py` measures official Spatial observations with
matched runtime modes and precision settings. `profiling.py` accounts for each
recorded device activity once and keeps GPU work separate from elapsed time.
`experiments.py` runs a declared JSON plan with owned workers, distinct ports,
all ten Spatial tasks and paired initial-state hashes. It validates actual
executed image/state ages against feature publication and execution records.
These modules reuse the existing evaluator and timing protocols.

Run `python -m coexecution.libero.experiments --plan PLAN.json --output OUTPUT`
from the prepared runtime. The plan declares `runtime_root`, `model_python`,
`evaluation_python`, `upstream`, `libero_config`, the action/VLM/simulation
`gpus`, action/VLM `ports`, `protocols`, `episodes_per_task`, `max_steps`, `seed`
and `conditions`. Each condition supplies a unique `name`, `checkpoint`,
`variant` and explicit optimization `flags`; optional `simulation_preload`
selects a compatible local library. The runner requires clean devices before
each condition, preserves invocation logs and stops only its own processes.
Use a new output directory when changing the plan. Deployment comparison fixes
real control at 20 Hz and reports success, deadline misses and executed AoI;
kernel timing alone does not establish these results.

## Fixed inference image resolution / 固定推理图像分辨率

**中文：** 在原有服务命令后添加 `--image-resolution 192`（或 `128`），
即可保留 checkpoint 的原有裁剪/视野，再对双相机图像做 `INTER_AREA`
缩放。未指定参数时完全沿用 checkpoint；这不是默认 256。
该选项会改变视觉信息及可能的任务质量，与数值优化配置独立。

**English:** Append `--image-resolution 192` or `128` to an existing
`python -m coexecution.libero.service` command (the compatibility
`scripts/serve_libero_protocol.py` entrypoint also accepts it). Both cameras keep
the checkpoint's complete eval crop/transform, followed by an area resize.
All Qwen pixel consumers, including the collators, receive the matching pixel
budget so they cannot silently upscale the image again. Omitting the argument
leaves the checkpoint preprocessing untouched. Explicit sizes must be positive,
no larger than the square checkpoint target, and divisible by the actual Qwen
`patch_size * merge_size`. The current override requires eval-mode
albumentations preprocessing; unsupported pipelines fail instead of falling back.

An experiment plan can declare `image_resolution` globally or per condition;
a condition's explicit `null` restores checkpoint behavior for that condition.
For example, a condition in an otherwise complete plan can contain:

```json
{
  "name": "bf16-r192",
  "checkpoint": "models/spatial-checkpoint",
  "variant": "pir2",
  "image_resolution": 192,
  "flags": ["--input-preparation", "--static-vlm", "--vlm-cuda-graph"]
}
```

The runner passes the same setting to both independently owned policies in
split or paired layouts. The evaluator records it in each server's stable
identity, binding resumes to that configuration. Vision response metadata also
records the observed `actual_grid_thw`, visual tokens per image and feature
tensor shapes. Do not equate the requested size with the observed token count.

The reusable API is separate from `OptimizationConfig`:

```python
from coexecution.libero.preprocessing import ImageResolutionConfig, ImageResolutionScope
from coexecution.optimization.session import GrootOptimizations

with ImageResolutionScope(policy, ImageResolutionConfig(size=192)):
    with GrootOptimizations(policy, optimization_config):
        # Collate, prepare inputs, warm every shape, then run serial inference.
        ...
```

The resolution scope must enclose input preparation and optimization/Graph
scopes. Exit quiescently after those scopes close; the original processors and
eval transform are restored, including on setup failure. Live resolution
switching and nested ownership are unsupported. Training transforms, model
weights and checkpoint metadata are not modified. Lower-resolution features
can also reduce DiT attention work, so measure the matching solo DiT baseline
before attributing a concurrent improvement entirely to reduced contention.

`coexecution.libero.measurement`, the streaming contention `worker` and
`campaign prepare`, and the Flow contention `flow` command accept the same
`--image-resolution` option. They apply it before preparing resident inputs and
keep the actual-grid evidence in their results; an already prepared tensor
cannot be resized by changing a later service argument.

Lightweight configuration/lifecycle checks run with
`python -m pytest -q tests/contracts/test_image_resolution.py`.
The explicit pinned-processor/tiny-CUDA gate is
`PIR2_GPU_TESTS=1 python -m pytest -q tests/gpu/test_image_resolution_gpu.py`.
It checks fast/slow Qwen processors, reduced token/feature lengths, Flow4 and
streaming bootstrap/one-step updates, and eager/Graph agreement at each size.
These contracts do not establish trained-policy task success or a speedup.

## Training

After selecting an idle GPU, train both methods with the same dataset, seed 1000, effective batch 64, H40 and 10,000-update schedule:

```bash
export PYTHONPATH="$PWD/upstream/learning/Isaac-GR00T"
export GROOT_HF_LOCAL_FIRST=1 GROOT_PATCH_MISTRAL=1 HF_HUB_OFFLINE=1
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/train_groot_spatial.py \
  --variant pir2 --steps 10000 --stop-at 500
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/train_groot_spatial.py \
  --variant flow --steps 10000 --stop-at 500
```

Run sequentially when sharing the selected GPU. The stop-at checkpoint contains optimizer/scheduler/RNG state; remove `--stop-at` to continue the same 10k schedule. The upstream trainer reseeds data order from the resumed global step, so continuation is not bitwise equivalent to uninterrupted training. Both methods use the same resume boundary.

πR² uses random image delays 0–5 and a delay embedding; the current ordinary Flow control uses no image-delay augmentation. Accordingly, the comparison tests complete recipes, not an isolated causal effect of the rolling schedule.

## Evaluation

Read [timing protocols](timing-protocols.md). With both checkpoints available:

```bash
python3 scripts/run_libero_protocols.py --step 500
```

This runner reserves physical GPU0 for the action server, GPU2 for the VLM server and GPU3 for rendering, waits for clean devices, and runs the four conditions sequentially. Each condition covers ten tasks × twenty episodes. Use a new `--label` for a deliberately changed experiment; outputs bind checkpoint and implementation hashes and must not be mixed across versions.

`run_libero_protocol_pipeline.py` is the longer staged supervisor. Its default mode first requires the completed `protocol-v4-integration` run and the explicit GPU contracts from `check_libero_timing_gpu.py`; it does not synthesize these checks. The `--adopt-step500-client` mode is only for an operator who has stopped the old supervisors and is deliberately transferring a known live client. Never start two supervisors over the same outputs. Checkpoints at step500 are archived before rolling retention during continuation.

The SO100 `train_pir2_demo.sh` / `infer_pir2_demo.py` scripts are separate real-observation replay checks used by profiling tools; they are not LIBERO success-rate evaluation.
