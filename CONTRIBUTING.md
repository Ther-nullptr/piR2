# Contributing / 贡献

Use a topic branch and a focused PR. Titles use `type: 中文摘要 / English summary`; PR bodies use the four bilingual sections in the template. Base dependent work on its prerequisite branch and state that dependency explicitly. Do not mix unrelated experiments into one PR.

代码、配置、稳定说明通过主题分支和 PR 进入 `main`，默认使用 merge commit。首次空仓库只建立空的 main 基线；后续内容均通过 PR。用户明确授权的批次合并，在相关检查通过、讨论解决后执行；普通开发请求不自动授权合并。不强推、不绕过保护，不把代理自身审阅冒充独立 GitHub approval。

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

Keep upstream changes as reviewed patches against the pinned source. Preserve third-party notices. Do not choose a new license for third-party source or publish private machine paths, credentials, local symlinks or raw experimental artifacts.
