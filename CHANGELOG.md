# 更新日志（Changelog）

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 风格，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [1.0.2+pkg] - 2026-10-19

仅打包布局与仓库元数据的变更，**插件行为与 1.0.2 完全相同**，已安装的 1.0.2 无需升级，
也无需重启（引擎、工具表、缓存语义均未改动）；tarball 里的 `package.json` 版本仍是 `1.0.2`。

- 可安装包由 `plugin/` 移到 `plugins/dsh-math/`：awesome-dsh-plugin 市场 CI 只在**仓库根**或
  `packages/` / `plugins/` / `apps/` 子包读取 `dsh.bundle` manifest，`plugin/` 不在其列。
- `package.json` 去掉 `"private": true`，可发布到 npm（可选）。
- `_dist/` 不再入库；tarball 改为 `Release` 工作流产出的 Release 资产，资产名固定为 `dsh-math.tgz`
  （不带版本号 —— 市场用 `releases/latest/download/dsh-math.tgz`，带版本号的文件名下次发版就 404）。
- 新增 `docs/submission/`：市场条目文件（`data/plugins/…yml`）与投稿步骤。
- `_verify/` 下的启动脚本改为读环境变量 `DSH_HOME`，不再硬编码 Harness 安装路径。
- `Release` 工作流改为「先建 **draft** → 上传资产 → 最后一步才发布」。本仓库开启了
  「不可变 release」（immutable releases）：release 一经发布，就不能再追加或替换资产
  （REST 返回 `422 Cannot upload assets to an immutable release.`），tag 名也会被永久占用
  （删掉 release 也换不回同名 tag）。所以资产必须在 draft 阶段补齐。

## [1.0.2] - 2026-10-03

首个公开发布版本（GitHub 开源）。

### 特性

- 14 个 `math_*` 工具，接入 214 个数学算子（SymPy / mpmath / NumPy / SciPy），
  覆盖考研数学一：高等数学、线性代数、概率论与数理统计。
- 独立验证机制：重要结果用与计算不同的途径复核，`verification.status` 区分
  `independent / cross / numeric / unverified` 四级。
- 诚实返回：`status` 严格区分 `ok / partial / unsolved / error`，
  绝不把 SymPy 未求值表达式当成功。
- 成功缓存 + 失败记账：相同参数命中缓存；失败参数被拦下避免无意义重试（可显式 `retry: true`）。
- 常驻 Python 引擎：JSON lines over stdio，崩溃自动重建（最多 2 次）。
- 四类任务模式（纯计算 / 验证找错 / 思路分析 / 完整解题）由自然语言自动识别。

### 测试期间修复的缺陷（节选，详见 docs/测试报告.md）

- 关系运算符 `>=` / `<=` / `!=` 被静默解析成未定义函数（`ast.py` 的 `_RESERVED` 缺 `Ge/Le/Ne/Gt/Lt`）。
- `abs(x)` 被隐式乘法拆成 `a*b*s*x`。
- `is_unevaluated` 引用了不存在的 `sp.Solve`，导致未求值表达式漏拦。
- `result.approx` 从第 13 位起数值错误。
- `numeric_agree` 多变量时取同一点导致漏判。
- 含参二次型 `.equals()` 挂死。
- `series_sum` 只返回 `Piecewise` 导致幂级数被判 `unsolved`。
- `y'(x)` 撇号解析失败；`implicit_diff` 变量角色歧义。

### 测试结论

- 端到端验收 `tests/_test_acceptance.py`：111/111 通过，exit 0。
- 八个专项 `tests/_test_special.py`：8/8 通过，exit 0。
- 引擎自检 `engine/_selftest_*.py`：7 个脚本全部 exit 0。

### 仓库整理（发布时）

- 测试脚本移入 `tests/`，构建/同步脚本移入 `scripts/`。
- 清理 `_verify/` 下的临时探针脚本与中间产物，保留可复现的证据文件。
- 补充双语 README、MIT LICENSE、CHANGELOG、requirements.txt、CI、贡献与安全政策。

## [1.0.1] / [1.0.0]

内部迭代版本，未另行发布与记录。
