# Single-GPU S1/S2 execution tools

These are auxiliary GR00T inference and profiling tools. The current workload loads the SO100 replay checkpoint `outputs/pir2-so100-smoke/checkpoint-10` and real prerecorded SO100 observations. It is not the LIBERO closed-loop evaluator. Adapting an optimization to the repository's GR00T-N1.7-LIBERO main line requires validating the suite-specific input, action horizon, state, embodiment and task success separately.

The two workers run a real VLM (S2) and a rolling DiT action head (S1) on one physical GPU. Feature handoff uses completion events and reader leases. Optional fusion, static input handling and CUDA Graph variants are explicit A/B choices. Numerical checks restore a paired starting buffer and compare with the eager path; periodic replay also records deadlines and feature ages.

## Commands

Prepare the pinned GR00T source and GPU environment as described in [environment setup](../docs/environment.md). Run the separate SO100 demonstration training entrypoint only if you need this replay checkpoint. Select an idle GPU; the launcher performs a preflight occupancy check:

```bash
PIR2_GPU=0 bash scripts/run_single_gpu_baseline.sh measure
PIR2_GPU=0 bash scripts/run_single_gpu_baseline.sh runtime
PIR2_GPU=0 bash scripts/run_single_gpu_baseline.sh protocols
```

Run one command at a time. Use `PIR2_OUTPUT` for a new result directory. Trace modes require Nsight Systems on PATH or an explicit `PIR2_NSYS` executable. The `test_*_gpu.py` modules require CUDA and are opt-in; CI only runs timeline/launch-attribution accounting tests.

See [measurement definitions](../docs/coexecution.md). Raw JSON, plots, SQLite exports, logs and Nsight traces stay local. These tools do not by themselves demonstrate LIBERO accuracy, physical control stability, quantization quality or performance on another GPU.
