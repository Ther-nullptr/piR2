# Repository working agreement

Read README.md and CONTRIBUTING.md. Use focused topic branches and bilingual commit/PR titles, with merge commits after checks. Explicit user authorization to merge these batches persists; do not ask repeatedly for the same authorization. Never force-push or bypass protection.

Keep models/data/environment directories, generated experiments, research drafts and process notes out of Git. Documents use the .gitignore allowlist. Preserve running experiments; organize publication in an isolated checkout. Stage explicit paths and inspect them for credentials, personal paths and oversized artifacts.

Use the validation commands in CONTRIBUTING.md. Public CI checks code and scheduling/accounting only: no CPU model validation, GPU workloads, gated-model downloads or laboratory runners. Describe actual validation limits. Do not conflate Leap reconstruction, SO100 replay, browser simulation or single-GPU profiling with LIBERO closed-loop success.

Record dependency pins and local patches in sources.lock.json and THIRD_PARTY_NOTICES.md. Do not silently edit or update upstream source. Work-in-progress plans and generated evidence stay local or in PR discussion.
