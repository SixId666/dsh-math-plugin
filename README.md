<div align="center">

# dsh-math

### DSH（DeepSeek Harness）考研数学一智能计算与验证插件

[![CI](https://github.com/SixId666/dsh-math-plugin/actions/workflows/ci.yml/badge.svg)](https://github.com/SixId666/dsh-math-plugin/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![SymPy](https://img.shields.io/badge/powered%20by-SymPy-3B5584?logo=sympy&logoColor=white)](https://www.sympy.org/)
[![Version](https://img.shields.io/badge/version-1.0.2-success)]()

**[English](README.en.md)** | 简体中文

</div>

> **核心原则：计算交给工具，思考交给模型，验证交给独立机制。**

本插件把 SymPy / mpmath / NumPy / SciPy 驱动的 **214 个数学算子** 接进 DSH（DeepSeek Harness），
注册为 **14 个 `math_*` 工具**，让 Agent 在正常对话中被自动调用（Agent Tool Calling），完成考研数学一
（高等数学、线性代数、概率论与数理统计）的精确计算、结果验证、思路分析与完整解题——
而不必为一次积分临时写一个 Python 脚本。

---

## 特性

- **覆盖考研数学一三大模块**：14 个工具、214 个引擎算子（`calculus` 43、`linalg` 36、`probability` 49、
  `symbolic_extra` 36、`numeric` 25、`vector_calc` 16、`verifier` 9），覆盖化简、方程、极限、导数、积分、
  级数、微分方程、线性代数、概率统计、数值计算、曲线曲面积分与三大公式、函数分析。
- **独立验证机制**：重要结果由与计算**不同**的途径复核（求导回代、代回方程、矩阵乘积、归一化、
  mpmath 数值求积、矩母函数），并区分 `independent / cross / numeric / unverified` 四个级别，
  `unverified` 会如实说明原因，不编造验证。
- **诚实返回**：`status` 严格区分 `ok / partial / unsolved / error`；SymPy 求不出解析解（如 `∫x^x dx`）
  时如实返回 `unsolved`，**绝不把未求值表达式当成功**。
- **缓存 + 失败记账**：相同参数命中缓存；失败过的完全相同参数被账本拦下，避免无意义重试
  （改参数/换方法，或显式 `retry: true` 才会重试）。
- **常驻引擎**：Python 子进程常驻，JSON lines over stdio；首次调用约 1–2 s，之后毫秒级；
  引擎崩溃自动重建（最多 2 次）。
- **零 Node 依赖**：插件 Node 侧无任何外部依赖；Python 侧仅需 `sympy`、`mpmath`
  （可选 `numpy` / `scipy` 增强数值能力）。

## 快速开始

### 1. 环境要求

| 项目 | 要求 | 说明 |
| --- | --- | --- |
| DSH | 本机桌面版（desktop profile） | 插件以 bundle + cordis patch 的形式注册 |
| Python | 3.10 及以上 | 引擎以常驻子进程运行 |
| Python 包 | `sympy`、`mpmath`（必需）；`numpy` / `scipy`（可选） | 已在 Anaconda 环境验证 |

一键安装全部 Python 依赖：`pip install -r requirements.txt`。

Python 解释器按以下顺序解析（`lib/bridge.js: resolvePython`）：

1. 环境变量 `DSH_MATH_PYTHON` → `DSH_PYTHON` → `PYTHON` → `PYTHON_EXE`（取第一个「文件确实存在」的）；
2. `PATH` 上的 `python` / `python3` / `py`；
3. 已知安装位置兜底（Anaconda / miniconda / `C:\Python3xx` / `/usr/bin/python3` 等）；
4. 全部失败则报错并提示用 `DSH_MATH_PYTHON` 指定解释器绝对路径。

也可以在插件配置里显式写死解释器（见「配置」）。

### 2. 安装

产物：`_dist/dsh-math-1.0.2.tgz`（npm pack 格式）。

```
plugin_manager install_bundle target: "E:\dsh-math\_dist\dsh-math-1.0.2.tgz"
plugin_manager list_bundles
```

或在 DSH 插件界面选择「从本地文件安装」。手工登记到 profile 的方式见
[plugin/README.md](plugin/README.md#2-安装)。

> **注意**：pnpm 对同路径的 `file:` tarball 会按 lockfile 复用已解包副本——改了插件内容后请提升
> `package.json` 的 `version` 并重新打包，再安装一次。

### 3. 启用并重启 DSH（必要步骤）

```
plugin_manager set_bundle target: "dsh-math" enabled: true
```

**安装或升级后必须重启 DSH**：宿主的模块解析表在进程启动时建立，运行中的进程里新增/替换 bundle
会被要求「进程重启」，因此新插件会停在 `inactive`，`math_*` 工具不会出现。这不是插件缺陷——
已用宿主自身的 loader 在隔离进程里验证过完整启动链可正常导入并注册 14 个工具（证据见
[docs/交付说明.md](docs/交付说明.md#三证据关键实测记录)）。

### 4. 直接用自然语言提问

```
计算 ∫₀¹ x² dx
求 lim_{x→0} sin(x)/x
解方程 x² - 3x + 2 = 0
判断 A = [[2,1],[1,2]] 是否可对角化
X ~ N(0,1)，求 P(-1 < X < 1)
求 y' = x*y 的通解
验证一下：∫ x e^x dx = (x-1)e^x 对不对？
分析 f(x) = x³ - 3x 的单调区间、极值和渐近线
```

无需手动切换模式，插件按自然语言自动识别**计算 / 验证 / 思路分析 / 完整解题**四类任务，
行为规范见 [docs/使用示例.md](docs/使用示例.md)。

### 配置

配置写在 profile 的 `cordis.patch.yml` 对应条目里，四个键都有默认值，`config: {}` 即可直接使用：

```yaml
- id: dsh-math
  config:
    python: 'E:\anaconda3\python.exe'   # 指定解释器（缺省走上面的探测顺序）
    workerPath: ''                      # 覆盖 engine/worker.py 的路径（一般留空）
    timeoutMs: 60000                    # 单次调用超时（毫秒），默认 60000
    cache: true                         # 是否启用结果缓存与失败记账，默认 true
```

插件不导出 Config schema（依赖宿主的 `@deepseek-ai/schemastery`），由插件自己归一化与校验：
非法的 `timeoutMs` 回落到 60000，`cache` 只认布尔语义。

## 工具清单（14 个）

| 工具 | 用途 |
| --- | --- |
| `math_simplify` | 化简 / 展开 / 因式分解 / 通分 / 部分分式 |
| `math_solve` | 方程、方程组、不等式求解 |
| `math_limit` | 极限（含单侧极限、数列极限） |
| `math_derivative` | 导数 / 高阶导 / 偏导 / 梯度 / 雅可比 / 海森 / 隐函数 / 参数方程 |
| `math_integrate` | 不定积分 / 定积分 / 反常积分（重积分算子 `integrate_multi` 已在引擎注册，未在工具层暴露参数入口，见 [docs/未实现与限制.md](docs/未实现与限制.md)） |
| `math_series` | 泰勒与幂级数展开、级数求和与判敛 |
| `math_dsolve` | 常微分方程（含初值问题、方程组） |
| `math_matrix` | 行列式 / 逆 / 秩 / 特征值与特征向量 / 对角化 / 二次型 / 线性方程组 |
| `math_probability` | 概率、分布（pdf/pmf/cdf/分位数）、期望方差协方差相关系数、矩、参数估计、假设检验、中心极限定理 |
| `math_numeric` | 高精度数值、数值积分、数值求根、优化、数值比较 |
| `math_vector_calculus` | 曲线积分 / 曲面积分 / 格林公式 / 高斯公式 / 斯托克斯公式 |
| `math_verify` | 结果验证（用与计算不同的途径复核） |
| `math_analysis` | 函数分析：定义域 / 单调凹凸 / 渐近线 / 极值 / 多元极值 / 拉格朗日乘子 / 不等式 |
| `math_status` | 引擎状态、算子清单、缓存统计 |

所有工具返回**统一的 JSON 结构**（由插件渲染成可读文本）：
`success / operation / input / result / conditions / verification / method / error / warnings`。

### 验证语义（`verification.status`）

| 取值 | 含义 |
| --- | --- |
| `independent` | 有独立途径复核（求导回代、代回方程、另一套公式/数值求积等），且与主结果一致 |
| `cross` | 有交叉验证（例如符号结果与数值结果一致） |
| `numeric` | 只有数值证据，**不构成解析证明** |
| `unverified` | 未能验证（工具会如实说明原因，不编造） |

## 工作原理

```
用户自然语言提问
      │
      ▼
DSH Agent ── 系统提示 section（MATH_SECTION_ORDER = 2500）说明四类任务模式与调用时机
      │  tool calling
      ▼
dsh-math 插件（Node，零依赖）
  entry.js      注册 14 个工具、注入系统提示、配置归一化
  tools.js      工具的 name / description / parameters
  bridge.js     PythonBridge：常驻子进程、请求超时、缓存与失败记账
  render.js     统一结果 JSON → 可读文本
      │  {id, op, args} 一行 JSON → stdin；stdout 逐行回 {id, status, result, verification, …}
      ▼
Python 常驻引擎（worker.py + mathkit/，214 个算子）
  ast / symbolic_core / calculus / linalg / probability / numeric /
  vector_calc / verifier / result …（SymPy / mpmath / NumPy / SciPy）
      │
      ▼
统一结果 JSON（success / operation / input / result / conditions / verification / method / …）
```

引擎协议：stderr 只用于日志（插件保留尾部 64 KiB 供排错）。`math_status` 可查看引擎状态、
算子清单与缓存统计。

## 仓库结构

```
dsh-math-plugin/
├── README.md / README.en.md   # 中英双语说明
├── LICENSE  CHANGELOG.md  CONTRIBUTING.md  SECURITY.md  requirements.txt
├── docs/                      # 交付文档（中文为权威版本）
│   ├── 工具说明.md             # 14 个工具的参数、子操作、返回结构与示例
│   ├── 使用示例.md             # 四类任务模式的真实对话示例
│   ├── 测试报告.md             # 测试分层、真实运行记录、复现命令
│   ├── 已实现功能清单.md        # 已实现功能 + 运行依赖清单
│   ├── 未实现与限制.md          # 数一考纲缺口与环境限制
│   └── 交付说明.md             # 十项交付物索引 + 11 条验收标准对照 + 实测证据
├── plugin/                    # 可安装插件包（自包含 bundle，pack 源）
│   ├── package.json  cordis.patch.yml  icon.svg  README.md
│   ├── locale/{zh,en}.json
│   ├── lib/{entry,define_tool,bridge,tools,render}.js
│   └── engine/                # 由 engine/ 同步生成（勿直接修改，见 CONTRIBUTING.md）
├── engine/                    # 引擎源码（唯一事实来源，改这里再同步）
│   ├── worker.py              # 常驻进程入口（JSON lines over stdio）
│   ├── mathkit/               # 14 个 .py：13 个功能模块 + __init__.py，214 个算子
│   └── _selftest_*.py         # 7 个引擎自检脚本
├── tests/                     # 测试（CI 自动运行）
│   ├── _test_acceptance.py    # 端到端验收（111 例）+ _test_acceptance_result.json
│   └── _test_special.py       # 八个专项测试 + _test_special_result.json
├── scripts/                   # 构建与校验脚本
│   ├── _sync_engine.py        # engine/ → plugin/engine 同步（白名单）
│   ├── _pack.py               # 生成 _dist/*.tgz（npm pack 格式）
│   ├── _dump_specs.mjs        # 导出 14 个工具的真实参数表 → _specs.json
│   ├── _check_contract.py     # 契约校验：工具表里的算子名/参数名能否被引擎接受
│   └── _specs.json            # 工具→算子映射导出（_dump_specs.mjs 产物）
├── _verify/                   # 宿主 loader 内真机验证脚本与输出（证据）
└── _dist/dsh-math-1.0.2.tgz   # 安装包
```

## 文档

| 文档 | 内容 |
| --- | --- |
| [docs/工具说明.md](docs/工具说明.md) | 14 个工具的名称、参数、子操作分派、返回结构与典型调用 |
| [docs/使用示例.md](docs/使用示例.md) | 四类任务模式（计算/验证/思路分析/完整解题）的真实对话示例 |
| [docs/测试报告.md](docs/测试报告.md) | 测试分层、真实运行记录、复现命令、期间修复的缺陷 |
| [docs/已实现功能清单.md](docs/已实现功能清单.md) | 已实现功能与运行依赖 |
| [docs/未实现与限制.md](docs/未实现与限制.md) | 数一考纲缺口与环境限制 |
| [docs/交付说明.md](docs/交付说明.md) | 十项交付物索引、11 条验收标准逐条对照、实测证据 |
| [plugin/README.md](plugin/README.md) | 随 npm 包分发的安装/启用/配置/排错说明 |

## 常用问法

| 任务类型 | 问法示例 | 期望行为 |
| --- | --- | --- |
| 纯计算 | 「计算 ∫₀^{π} sin x dx」「求 A 的特征值」 | 直接调用对应工具，给出精确结果与条件 |
| 验证/找错 | 「验证 ∫x e^x dx = (x-1)e^x」「我这样算对吗：…」 | 用 `math_verify` 独立复核，**指出具体错误步骤** |
| 思路分析 | 「这道题该从哪下手」「为什么这里要用拉格朗日乘子」 | 以结构化结论为支撑讲思路，不做无效计算 |
| 完整解题 | 「完整解一下这道题：…」 | 八项齐全：题目分析、考点识别、解题思路、关键公式、具体计算、结果验证、最终答案、同类识别技巧 |
| 追问 | 「第二步再展开讲讲」 | 只讲该步，复用上一轮已验证结果，不重算整题 |

## FAQ

**Q：安装后对话里看不到 `math_*` 工具？**
安装后必须重启 DSH（见「快速开始」第 3 步）。重启前 bundle 停在 `inactive` 是宿主解析表的正常行为，不是插件故障。

**Q：报「找不到可用的 Python 解释器」？**
用环境变量 `DSH_MATH_PYTHON` 指定 python.exe 绝对路径，或在插件配置里写 `python:` 键；并确认该环境已 `pip install sympy mpmath`。

**Q：第一次调用很慢？**
首次调用要拉起常驻引擎并加载模块（约 1–2 秒），之后同进程内为毫秒级。`math_status` 可查看引擎状态与缓存统计。

**Q：`∫x^x dx` 返回 unsolved，是 bug 吗？**
不是。没有初等原函数（如 `∫x^x dx`）或无闭式分位数（χ²/F/t）时，工具如实返回 `unsolved`，不会编造结果；可改用 `math_numeric` 求数值解。

**Q：如何参与贡献 / 复现测试？**
见 [CONTRIBUTING.md](CONTRIBUTING.md)；CI 每次推送会自动运行全套测试。

## 已知限制（摘要）

完整列表见 [docs/未实现与限制.md](docs/未实现与限制.md)。

1. **安装/升级后需重启 DSH** 才能激活（宿主解析表要求进程重启）。
2. **图片输入依赖 DSH 原生多模态**：插件不做 OCR；若当前环境的多模态能力不可用，需用户把题目转成文本。
3. **数一考纲仍有缺口**：抽样分布与统计量的精确分布、条件分布、随机变量函数的分布（卷积）、二维正态、
   协方差矩阵与随机向量、全期望/全方差公式、大数定律、顺序统计量、χ²/F/t 的分位数等。
4. **`unsolved` 是诚实结果**：例如 `∫x^x dx`、χ²/F/t 分位数（无闭式反函数）——工具不会编造闭式。

## 测试与证据（摘要）

全部结论均可由仓库内脚本复现（命令与真实输出见 [docs/测试报告.md](docs/测试报告.md)），
GitHub Actions 每次推送自动重跑同一套测试（见上方 CI 徽章）：

- 端到端验收 `tests/_test_acceptance.py`：**111/111 通过，exit 0**，明细 `tests/_test_acceptance_result.json`；
- 八个专项 `tests/_test_special.py`：**8/8 通过，exit 0**（214 个算子全部注册、缓存语义、未求值表达式拦截、
  验证级别断言等）；
- 引擎自检 `engine/_selftest_*.py`：7 个脚本全部 **exit 0**；
- 契约校验 `scripts/_check_contract.py`：工具表里的算子名/参数名全部能被引擎接受；
- 宿主 loader 内真机验证：`_verify/scratch_boot_full.mjs` → `BOOT OK`，14 个工具全部注册并真机执行成功。

关键实测记录与验收标准逐条对照见 [docs/交付说明.md](docs/交付说明.md)。

## License

[MIT](LICENSE) © [SixId666](https://github.com/SixId666)
