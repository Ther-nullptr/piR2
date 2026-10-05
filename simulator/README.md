# Interactive timing teaching tool

This browser tool explores queueing, cache age, fixed-rate control, assumed compute/bandwidth and hypothetical quantization tradeoffs. It is an auxiliary explanation tool, not a GR00T-LIBERO evaluator, robot simulator or predictor of task success on new hardware.

Open [index.html](index.html) directly, or preview over loopback from the repository root:

```bash
python3 simulator/serve.py --port 8765
```

The preview serves only this tool and selected stable documentation. It does not expose runtime artifacts, private configuration or the project root. The numerical engine is in `model.js`; interface code is in `app.js`. Extend those modules rather than creating a new simulator per experiment.

Hardware multipliers, contention and error assumptions are editable. The historical SO100 replay values in `evidence.js` are a compact illustrative snapshot; original traces and generated reports are not bundled. They are not LIBERO measurements. Low-bit speedups and AR(1) error behavior in the interactive model are assumptions, not measured kernels or task accuracy.

## Checks

```bash
node --test simulator/model.test.cjs
npm ci --prefix tools/ui-check
tools/ui-check/node_modules/.bin/playwright install chromium
node simulator/browser-check.cjs
```

Browser dependencies are development-only. Playwright 1.51.1 was used with the original Ubuntu 20.04 workspace; newer hosts may require a reviewed dependency update. Browser checks cover layout, controls and exported records, not CPU model inference. Screenshots and downloads are written to ignored `artifacts/simulator/`. Lightweight CI runs only the Node numerical contracts; browser checks are explicit local validation.
