# Contributing / 贡献

Use a topic branch and a focused PR. Titles use `type: 中文摘要 / English summary`; PR bodies use the four bilingual sections in the template. Base dependent work on its prerequisite branch and state that dependency explicitly. Do not mix unrelated experiments into one PR.

代码、配置、稳定说明通过主题分支和 PR 进入 `main`，默认使用 merge commit。首次空仓库只建立空的 main 基线；后续内容均通过 PR。用户明确授权的批次合并，在相关检查通过、讨论解决后执行；普通开发请求不自动授权合并。不强推、不绕过保护，不把代理自身审阅冒充独立 GitHub approval。

## CUTLASS dependency / CUTLASS 依赖

Keep actual CUDA implementations in `coexecution/kernels`. The official NVIDIA
CUTLASS submodule at `coexecution/kernels/vendor/cutlass` records
`982748aa7356fa838c2ea4994ddcb0b2a4b4cefa` as the project's reproducible default.
Initialize that default from an existing piR2 checkout with:

```bash
git submodule update --init --recursive coexecution/kernels/vendor/cutlass
```

An author may instead set `CUTLASS_ROOT=/path/to/cutlass` to reuse a complete
compatible Git checkout or headers-only installation, or keep a compatible
local submodule revision. Required `include/` (including `cutlass` and `cute`)
and `tools/util/include/` content must be available. Compatibility depends on
the kernel APIs and CUDA/C++ toolchain, not exact equality with the default SHA;
the tools record the actual selected source and Git revision when available.
Do not add personal absolute paths or local Git-link version choices to an
unrelated commit. General CUTLASS headers can come from the environment while
piR2's operator implementations stay in this repository.

For the default-submodule workflow, repeat the update command after pulling or
switching branches. Preserve local work and deliberate version choices first;
that command returns the submodule to the branch's recorded Git link. Routine
default setup does not need `--force` or `--remote`. If an existing independent
Git clone is available, `--reference /path/to/existing/cutlass --dissociate` is
an optional first-clone download optimization. It is not required for direct
reuse through `CUTLASS_ROOT`. Full instructions are in
[environment setup](docs/environment.md#cutlass-submodule).

Change the shared default CUTLASS pin in a separate reviewed change, updating
the Git link and its records in `sources.lock.json` and
`coexecution/kernels/SOURCE_MANIFEST.json` together. Describe the API/toolchain
reason and actual validation scope. Local compatibility choices do not require
changing that default. Preserve the
[default upstream license](https://github.com/NVIDIA/cutlass/blob/982748aa7356fa838c2ea4994ddcb0b2a4b4cefa/LICENSE.txt)
and the applicable notices of any alternate installation.

## Validation / 验证

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python tools/check_repository.py
python tools/run_checks.py
git diff --check
```

CI runs syntax, metadata, scheduling and accounting checks only. GPU/model tests are explicit opt-in commands in the relevant module documentation. Preserve real hardware, timing, initial-state and checkpoint evidence locally; report the validation scope in the PR. Do not interpret an interface check as task success.

只版本管理代码、必要配置、测试、固定依赖版本与稳定使用说明。模型、数据、缓存、日志、视频、profile、计划、研究草稿、过程文档与生成图表留本地。使用明确路径暂存，提交前检查 staged diff；不要使用 `git add .` 导入实验工作区。

## Review and landing / 审阅与合并

Use GitHub-hosted `Repository checks`, resolve relevant review discussions, and retain merge commits. Request a real independent reviewer when available; never fabricate an approval. No particular reviewer or service-side protection is claimed until configured and observed. The initial reference repository has no protected main branch; local templates are not server enforcement.

Keep GR00T changes as reviewed patches against the pinned model source; edit owned operator implementations locally and manage the shared CUTLASS default through its reviewed submodule pin. Preserve third-party notices. Do not choose a new license for third-party source or publish private machine paths, credentials, local symlinks or raw experimental artifacts.
