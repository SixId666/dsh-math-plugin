# dsh-math —— DSH 考研数学一智能计算与验证插件

> 「计算交给工具，思考交给模型，验证交给独立机制。」

本插件把 SymPy / mpmath / NumPy / SciPy 驱动的数学计算引擎接进 DSH，注册 **14 个数学工具**，
让 Agent 在正常对话里直接调用它们完成考研数学一（高等数学、线性代数、概率论与数理统计）的
精确计算与独立验证，而不必为一次积分临时写一个 Python 脚本。

本文件是随 npm 包一起分发的插件说明（安装 / 启用 / 配置 / 工具清单 / 排错）。
更完整的交付文档在仓库根目录：`README.md`、`docs/`。

---

## 1. 环境要求

| 项目 | 要求 | 说明 |
| --- | --- | --- |
| DSH | 本机桌面版（desktop profile） | 插件以 bundle + cordis patch 的形式注册 |
| Python | 3.10 及以上 | 引擎以常驻子进程运行 |
| Python 包 | `sympy`（必需）、`mpmath`（必需）、`numpy` / `scipy`（可选，增强数值能力） | 本机已用 Anaconda 环境验证 |

Python 解释器的解析顺序（`lib/bridge.js: resolvePython`）：

1. 环境变量 `DSH_MATH_PYTHON` → `DSH_PYTHON` → `PYTHON` → `PYTHON_EXE`（取第一个「文件确实存在」的）；
2. `PATH` 上的 `python` / `python3` / `py`（经 DSH 的 `subprocess.resolveExecutable` 解析）；
3. 已知安装位置兜底：`E:\anaconda3\python.exe`、`C:\ProgramData\Anaconda3\python.exe`、
   `C:\ProgramData\miniconda3\python.exe`、`C:\Python313\python.exe`、`C:\Python312\python.exe`、
   `/usr/bin/python3`、`/usr/local/bin/python3`；
4. 全部失败则报错：`找不到可用的 Python 解释器。请安装 Python 3.10+ 并安装 sympy，或用环境变量 DSH_MATH_PYTHON 指定 python.exe 的绝对路径。`

也可以在插件配置里显式写死解释器（见第 4 节）。

---

## 2. 安装

### 方式 A：用打包好的 tarball 安装（推荐）

产物：`_dist/dsh-math-1.0.2.tgz`（`npm pack` 格式，内部路径前缀 `package/`）。

在 DSH 的插件管理界面选择「从本地文件安装」，或让 Agent 调用插件管理工具：

```
plugin_manager install_bundle target: "E:\dsh-math\_dist\dsh-math-1.0.2.tgz"
```

安装完成后确认已登记：

```
plugin_manager list_bundles
```

### 方式 B：手工登记到 profile

编辑 `C:\Users\HP\.dsh\profiles\desktop\package.json`：

```json
{
  "dependencies": {
    "dsh-math": "file:E:/dsh-math/_dist/dsh-math-1.0.2.tgz"
  },
  "dsh": {
    "profile": {
      "bundles": ["...", "dsh-math"]
    }
  }
}
```

然后在 profile 目录执行该 profile 所用的包管理器安装（`pnpm install`），或直接让 DSH 的插件管理器
重新同步。

> **注意（实测坑）**：pnpm 对同路径的 `file:` tarball 会按 lockfile 复用已解包的副本，
> **内容变了但文件名/版本号没变时不会重新导入**。改了插件内容后请提升 `package.json` 的 `version`
> 并重新打包（得到新的文件名），再安装一次。

---

## 3. 启用 / 停用

本插件的补丁文件 `cordis.patch.yml` 会插入一个条目：

```yaml
- insert:
    - id: dsh-math
      name: 'dsh-math'
      config: {}
```

启用/停用可以直接在 DSH 插件界面操作，或：

```
plugin_manager set_bundle target: "dsh-math" enabled: true
plugin_manager set_bundle target: "dsh-math" enabled: false
```

> **重要**：插件的解析表（resolution）在 DSH 进程启动时建立，运行中的进程里新增/替换 bundle 时
> `ResolutionRouter.replace()` 会要求「进程重启」。因此**安装或改版本号之后必须重启 DSH**，
> 新插件才会从 `inactive` 变为 `active` 并在对话中可用。重启前工具列表里不会出现 `math_*`。

---

## 4. 配置

配置写在 profile 的 `cordis.patch.yml` 对应条目里：

```yaml
- id: dsh-math
  config:
    python: 'E:\anaconda3\python.exe'   # 指定解释器（缺省走第 1 节的探测顺序）
    workerPath: ''                      # 覆盖 engine/worker.py 的路径（一般留空）
    timeoutMs: 60000                    # 单次调用超时（毫秒），默认 60000
    cache: true                         # 是否启用结果缓存与失败记账，默认 true
```

四个键都有默认值，`config: {}` 即可直接使用。插件**不导出 Config schema**（DSH 的 Config 描述依赖
`@deepseek-ai/schemastery`，而 linked bundle 在本构建里解析不到宿主包），因此这里由插件自己归一化与
校验：非法的 `timeoutMs` 会被忽略并回落到 60000，`cache` 只认布尔语义。

---

## 5. 工具清单（14 个）

| 工具 | 用途 |
| --- | --- |
| `math_simplify` | 化简 / 展开 / 因式分解 / 通分 / 部分分式 |
| `math_solve` | 方程、方程组、不等式求解 |
| `math_limit` | 极限（含单侧极限、数列极限） |
| `math_derivative` | 导数 / 高阶导 / 偏导 / 梯度 / 雅可比 / 海森 / 隐函数 / 参数方程 |
| `math_integrate` | 不定积分 / 定积分 / 反常积分（重积分算子 `integrate_multi` 已在引擎注册，但未在工具层暴露参数入口，见 `未实现与限制.md`） |
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

```
success / operation / input / result / conditions / verification / method / error / warnings
```

`status` 取值区分四种情况：`ok`（成功）、`partial`（部分成功）、`unsolved`（当前方法求不出解析解）、
`error`（执行失败）。`unsolved` 与 `error` 都会带上失败原因，并且会被缓存账本记为失败，
**相同参数不会被无意义地重复尝试**（除非显式传 `retry: true`）。

### 验证语义（`verification.status`）

| 取值 | 含义 |
| --- | --- |
| `independent` | 有独立途径复核（求导回代、代回方程、另一套公式/数值求积等），且与主结果一致 |
| `cross` | 有交叉验证（例如符号结果与数值结果一致） |
| `numeric` | 只有数值证据，**不构成解析证明** |
| `unverified` | 未能验证（工具会如实说明原因，不编造） |

---

## 6. 安装/启用的实测注意事项

1. **必须重启 DSH**：安装或升级版本号后，活进程里该 bundle 会停在 `inactive`（工具也不会注册）。
   这是因为 DSH 的解析表要求「entry 变化需重启进程」，不是插件本身出错。
   已用宿主自己的 loader 在隔离进程里验证：`loadProfileDirectory` + `createRuntimeResolution` +
    `readProfilePatches` + `boot` 全流程可正常导入本插件、注册 14 个工具并真机执行
   （证据脚本 `_verify/scratch_boot_full.mjs`，输出 `_verify/scratch_boot_full_out.txt`）。
2. **首次调用会拉起引擎**：第一次调用任意 `math_*` 工具时会启动常驻 Python 子进程并加载模块，
   耗时约 1–2 秒；之后同一进程内的调用是毫秒级。`math_status` 可以看到引擎是否已启动。
3. **缓存与失败记账**：相同参数再次调用会命中缓存（`math_status` 的 `cache_stats` 可见命中数）；
   失败过的完全相同参数默认被拦下，需要换参数/换方法，或显式 `retry: true`。
4. **引擎崩溃自愈**：引擎进程异常退出时最多自动重建 2 次（`MAX_AUTO_RESTARTS`），避免
   「反复崩溃 → 反复重启」的无意义循环。

---

## 7. 包内容

```
package/
├── package.json          # name=dsh-math, version, exports → ./lib/entry.js, dsh.bundle.patch
├── cordis.patch.yml      # 插件插入条目
├── icon.svg
├── locale/               # 本地化文案
├── lib/
│   ├── entry.js          # Host 插件入口：注册 14 个工具、注入系统提示、配置归一化
│   ├── define_tool.js    # 与 DSH tools 服务对接的最小适配层
│   ├── bridge.js         # PythonBridge（常驻子进程、请求超时、账本）、resolvePython
│   ├── tools.js          # 14 个工具的 name/description/parameters
│   └── render.js         # 统一结果 → 可读文本渲染
└── engine/
    ├── worker.py         # 常驻引擎入口（JSON lines over stdio）
    └── mathkit/          # 计算与验证实现（ast/numeric/calculus/linalg/probability/…）
```

引擎协议：插件把 `{id, op, args}` 以一行 JSON 写入子进程 stdin，引擎逐行读、逐行把
`{id, status, result, verification, …}` 写回 stdout；stderr 只用于日志（插件保留尾部 64 KiB 供排错）。
