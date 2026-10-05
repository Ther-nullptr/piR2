# S1/S2 measurement boundaries

The current replay workload uses the SO100 GR00T example, H16, BF16 and preprocessed observations on GPU. It does not load the main LIBERO task checkpoint by default and is not a drop-in LIBERO benchmark. Its purpose is to validate execution mechanisms before applying them to the real closed-loop task.

Measure S1 alone, S2 alone, the serial pair and the concurrent pair. Use the measured serial pair as the speedup denominator, not a sum of isolated medians. Preserve input, cache version, RNG and rolling-buffer state across paired comparisons. Report model loading, preprocessing, CPU action decoding and robot/simulator I/O exclusions explicitly.

CUDA event spans are not proof of kernel overlap. Export Nsight data to SQLite and use `coexecution.analyze_trace` to intersect actual kernel intervals; `coexecution.operator_analysis` joins kernel launches with the innermost NVTX range on the correct CPU thread. Profiled latency and profiler-free latency are separate measurements.

```bash
nsys export --type=sqlite --output=artifacts/coexecution/run/nsys.sqlite \
  artifacts/coexecution/run/nsys.nsys-rep
.venv/bin/python -m coexecution.analyze_trace \
  artifacts/coexecution/run/nsys.sqlite --output artifacts/coexecution/run/overlap.json
```

The periodic runner retains the configured release period, bounded in-flight work and actual feature versions. Deadline denominators include dropped requests. Cache age is calculated at consumption, and release/completion/drop conservation is checked. Fixed-pair profiling and free-running periodic replay answer different questions; neither substitutes for the LIBERO wall-clock control protocol.

Analysis produces local machine-readable records. Plotting, if requested through the optional `--plot` flag, additionally requires matplotlib. One-off rendering scripts with historical hardware/trial constants, generated figures, optimization ledgers and snapshots remain local. Do not interpret trace coverage as demonstrated quantization quality or task accuracy.
