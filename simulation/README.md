# Leap GPU reconstruction

This is a state-based reconstruction using a public MuJoCo Playground expert and independently collected demonstrations. It is separate from GR00T/LIBERO, uses no VLM, and does not reproduce the paper's original four-expert dataset or three training seeds.

The task runs at 50 Hz with a 600-step budget. The student observes 41 dimensions, uses two observation frames, predicts 16-dimensional actions over H=16, and supports Flow, train-time RTC and πR². Physics stepping uses a length-one JAX scan to preserve the validated simulator execution boundary. Clipped executed actions are re-normalized before prefix conditioning.

## Setup and execution

Prepare `playground` with `tools/prepare_sources.py`, then install the pinned Playground in a separate `.venv-sim` CUDA environment following its own dependency instructions. Required packages include CUDA JAX 0.5.3, PyTorch 2.7.1, MuJoCo/MJX 3.14.0, ONNX, Flax, Optax and NumPy. Run from the repository root after checking for an idle GPU:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv-sim/bin/python -m simulation.collect --episodes 200 --seed 0 --save-data \
  --output outputs/simulation/demonstrations
CUDA_VISIBLE_DEVICES=0 .venv-sim/bin/python -m simulation.train \
  --method pir2 --seed 0 --epochs 800 --output outputs/simulation/pir2-seed0
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv-sim/bin/python -m simulation.evaluate \
  --checkpoint outputs/simulation/pir2-seed0/epoch-0800.pt \
  --output artifacts/simulation/pir2-seed0-evaluation.json
```

Use `--resume` with the saved `latest.pt` for training continuation. `scripts/run_leap_seed.sh` provides stage logs, a per-seed lock and resumable orchestration. Set `PIR2_GPU` and `PIR2_SEED` explicitly; inspect the budget before starting a long job. The older duplicate launcher and one-off replay diagnosis remain local.

`simulation.test_gpu_simulation` and `simulation.test_policy_gpu` are explicit GPU checks. They are excluded from lightweight CI. Checkpoints, demonstrations, videos and raw results remain in ignored local directories. Report environment versions, expert/data identity, selected checkpoint and shared evaluation seeds with any success rate; do not mix expert success with student success.
