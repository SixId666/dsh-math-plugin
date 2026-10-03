# 贡献指南（Contributing）

感谢关注 dsh-math！欢迎提交 Issue 与 Pull Request。

## 仓库结构速览

```
engine/        引擎源码（唯一事实来源，开发与调试都在这里改）
plugin/        可安装插件包（自包含 bundle；plugin/engine/ 由脚本同步生成）
tests/         验收测试与专项测试
scripts/       同步 / 打包 / 契约校验脚本
docs/          交付文档（中文为权威版本）
```

> **重要**：`plugin/engine/` 是由 `engine/` 同步生成的镜像，**不要直接修改**。
> `plugin/` 必须自包含：安装进 DSH profile 后不会再回来读仓库其它目录。

## 开发流程

### 场景 A：改引擎（新增/修复数学算子）

```powershell
# 1. 在 engine/mathkit/ 下改代码
# 2. 同步到 plugin/engine/（白名单：worker.py、mathkit/*.py、_selftest_*.py）
python scripts/_sync_engine.py

# 3. 跑测试
python -X utf8 tests/_test_acceptance.py     # 111 例验收
python -X utf8 tests/_test_special.py        # 8 个行为专项
python -X utf8 engine/_selftest_calculus.py  # 对应模块的引擎自检
```

### 场景 B：改工具接口（`plugin/lib/tools.js`）

工具表里的算子名与参数名必须与引擎签名一致，改完请跑契约校验：

```powershell
# 1. 重新导出工具→算子映射表（Node 不在 PATH 时可用 Electron 充当 Node）
$env:ELECTRON_RUN_AS_NODE=1; & "DeepSeek Harness.exe" scripts/_dump_specs.mjs > scripts/_specs.json

# 2. 契约校验（会抓出拼错的算子名、不被接受的参数名、必需参数缺失）
python -X utf8 scripts/_check_contract.py
```

### 场景 C：打包发布

```powershell
# 1. 提升 plugin/package.json 的 version（pnpm 对同路径 file: tarball 会复用旧副本）
# 2. 打包
python scripts/_pack.py    # 生成 _dist/dsh-math-<version>.tgz
# 3. 安装后重启 DSH（宿主解析表要求进程重启）
```

## 提交要求

- 新增能力或修复缺陷时，请在 `tests/_test_acceptance.py`（数学正确性）或
  `tests/_test_special.py`（行为语义）中补充可复现用例。
- 不要把「SymPy 未求值表达式」当成功返回——这是本插件的硬性语义（见 `docs/工具说明.md`）。
- 验证语义必须诚实：`unverified` 就写 `unverified` 并说明原因，不编造验证方法。
- 提交信息建议使用英文，一行主题 + 正文说明动机。
- PR 请对照模板里的自测清单逐项勾选。

## 报告缺陷

请使用 Issue 模板（bug report），附上：发给 Agent 的原话、工具名、返回文本、
Python 版本与操作系统。安全漏洞请勿公开提交，见 [SECURITY.md](SECURITY.md)。

## 文档语言

`docs/` 与代码注释以中文为权威版本；README 提供中英双语。修改文档时请保持
两个语言版本同步。
