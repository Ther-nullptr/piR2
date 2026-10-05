# LIBERO-Spatial training and real VLM evaluation

Use the official `nvidia/GR00T-N1.7-LIBERO` Spatial task checkpoint and all 432 demonstrations / 52,970 frames across ten Spatial tasks. This is GR00T adaptation, not π0.5 training. The VLM backbone remains frozen in training and active on real simulator images in evaluation.

The repository's main research references are the [NVIDIA task-weight repository](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO) and [official LIBERO example](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/LIBERO/README.md). The model-family manifest records Spatial, Object, Goal and Long separately. Current scripts implement Spatial only; extending a suite requires the matching checkpoint, data, initialization and full task evaluation.

For the ordinary NVIDIA baseline, the official example loads the local suite subdirectory, uses `LIBERO_PANDA` and `--use-sim-policy-wrapper`, and illustrates `--n-action-steps 8`. Our fixed-delay and wall-clock protocols intentionally change execution timing; their scores are separate from that ordinary baseline. The πR² author fork supplies the algorithm changes, not NVIDIA's published fine-tuned weights. Pin the actual source and model versions instead of silently tracking a moving `main` branch.

## Assets and environment

Prepare and patch GR00T source with `python3 tools/prepare_sources.py --component groot`. Follow its installation instructions in a Python 3.10 CUDA environment at `.venv/`. Model access may require accepting the publisher's terms and logging into Hugging Face locally. Then:

```bash
.venv/bin/python scripts/fetch_groot_spatial.py
```

The downloader fixes both model and dataset revisions and copies the official modality configuration. It does not download every suite or training optimizer state. The Cosmos backbone must also be available through the authorized local cache before offline serving.

Create a separate Python 3.12 simulator environment at `.venv-pi05/` with hf-libero 0.1.4, robosuite 1.4.0, MuJoCo 3.8.1, NumPy, PyAV, Pillow, pyzmq and msgpack. Prepare LIBERO assets through its normal package instructions, then create the isolated local config:

```bash
.venv-pi05/bin/python tools/prepare_libero_config.py --assets ~/.cache/libero/assets
```

The policy server uses `.venv`; the simulator client uses `.venv-pi05` and the pinned GR00T source via PYTHONPATH. Set EGL to the selected physical rendering GPU. If the host requires a libexpat preload, set `PIR2_LIBERO_LD_PRELOAD` to a local compatible library; no user's home directory is baked into the runner.

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
