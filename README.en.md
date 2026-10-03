<div align="center">

# dsh-math

### Smart Computation & Verification Plugin for Postgraduate Entrance Exam Mathematics I, for DSH (DeepSeek Harness)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![SymPy](https://img.shields.io/badge/powered%20by-SymPy-3B5584?logo=sympy&logoColor=white)](https://www.sympy.org/)
[![Version](https://img.shields.io/badge/version-1.0.2-success)]()

English | **[简体中文](README.md)**

</div>

> **Core principle: computation goes to tools, thinking goes to the model, verification goes to an independent mechanism.**

This plugin wires a math engine driven by SymPy / mpmath / NumPy / SciPy — **214 operators** — into
DSH (DeepSeek Harness) and registers **14 `math_*` tools**, so that the Agent can call them automatically
during ordinary conversations (Agent Tool Calling). It performs exact computation, result verification,
approach analysis and full problem solving for the Chinese postgraduate entrance examination
Mathematics I (Advanced Mathematics, Linear Algebra, Probability & Statistics) — no more throwing
together a throwaway Python script just to evaluate one integral.

---

## Features

- **Covers the three major modules of Math I**: 14 tools and 214 engine operators
  (`calculus` 43, `linalg` 36, `probability` 49, `symbolic_extra` 36, `numeric` 25, `vector_calc` 16,
  `verifier` 9) — simplification, equations, limits, derivatives, integrals, series, ODEs, linear algebra,
  probability & statistics, high-precision numerics, line/surface integrals with the three major
  formulas, and function analysis.
- **Independent verification**: important results are re-checked through a *different* route than the
  computation (differentiate-and-substitute, substitute back into equations, matrix products,
  normalization, mpmath numerical quadrature, moment generating functions), classified into four levels
  `independent / cross / numeric / unverified`. When `unverified`, the tool honestly states why —
  it never fabricates verification.
- **Honest results**: `status` strictly distinguishes `ok / partial / unsolved / error`. When SymPy
  cannot produce a closed form (e.g. `∫x^x dx`), the tool honestly returns `unsolved` and **never
  reports an unevaluated expression as success**.
- **Cache + failure ledger**: identical parameters hit the cache; parameters that previously failed are
  blocked by the ledger to avoid meaningless retries (retry requires different parameters, a different
  method, or an explicit `retry: true`).
- **Resident engine**: a persistent Python subprocess speaking JSON lines over stdio; the first call
  takes about 1–2 s, afterwards calls are millisecond-level. The engine auto-restarts after crashes
  (up to 2 times).
- **Zero Node dependencies**: the Node side of the plugin has no external dependencies at all; the
  Python side only needs `sympy` and `mpmath` (optional `numpy` / `scipy` for enhanced numerics).

## Quick Start

### 1. Requirements

| Item | Requirement | Notes |
| --- | --- | --- |
| DSH | Local desktop build (desktop profile) | The plugin registers as a bundle + cordis patch |
| Python | 3.10+ | The engine runs as a resident subprocess |
| Python packages | `sympy`, `mpmath` (required); `numpy` / `scipy` (optional) | Verified with an Anaconda environment |

The Python interpreter is resolved in the following order (`lib/bridge.js: resolvePython`):

1. Environment variables `DSH_MATH_PYTHON` → `DSH_PYTHON` → `PYTHON` → `PYTHON_EXE`
   (the first one whose file actually exists);
2. `python` / `python3` / `py` found on `PATH`;
3. Fallback to well-known install locations (Anaconda / miniconda / `C:\Python3xx` / `/usr/bin/python3`, etc.);
4. If all fail, an error is raised suggesting the absolute interpreter path via `DSH_MATH_PYTHON`.

You can also hard-code the interpreter in the plugin config (see "Configuration").

### 2. Install

Artifact: `_dist/dsh-math-1.0.2.tgz` (npm pack format).

```
plugin_manager install_bundle target: "E:\dsh-math\_dist\dsh-math-1.0.2.tgz"
plugin_manager list_bundles
```

Or choose "Install from local file" in the DSH plugin UI. To register it manually in a profile, see
[plugin/README.md](plugin/README.md#2-安装) (Chinese).

> **Note**: pnpm reuses the unpacked copy of a same-path `file:` tarball according to the lockfile —
> after changing the plugin, bump `version` in `package.json` and re-pack before installing again.

### 3. Enable and restart DSH (required)

```
plugin_manager set_bundle target: "dsh-math" enabled: true
```

**You must restart DSH after installing or upgrading**: the host builds its module resolution table at
process startup; adding/replacing a bundle in a running process triggers a "process restart" requirement,
so the new bundle stays `inactive` and the `math_*` tools will not appear. This is not a plugin defect —
the full boot chain has been verified with the host's own loader in an isolated process, registering all
14 tools successfully (evidence in [docs/交付说明.md](docs/交付说明.md#三证据关键实测记录), Chinese).

### 4. Just ask in natural language

```
Compute ∫₀¹ x² dx
Find lim_{x→0} sin(x)/x
Solve x² - 3x + 2 = 0
Is A = [[2,1],[1,2]] diagonalizable?
X ~ N(0,1), find P(-1 < X < 1)
Find the general solution of y' = x*y
Verify: is ∫ x e^x dx = (x-1)e^x correct?
Analyze f(x) = x³ - 3x: monotonic intervals, extrema and asymptotes
```

No manual mode switching: the plugin automatically recognizes one of the four task types
**computation / verification / approach analysis / full solving**; the behavior spec is in
[docs/使用示例.md](docs/使用示例.md) (Chinese).

### Configuration

The config lives in the profile's `cordis.patch.yml` entry. All four keys have defaults, so
`config: {}` works out of the box:

```yaml
- id: dsh-math
  config:
    python: 'E:\anaconda3\python.exe'   # interpreter (defaults to the detection order above)
    workerPath: ''                      # override the path to engine/worker.py (normally empty)
    timeoutMs: 60000                    # per-call timeout in ms, default 60000
    cache: true                         # enable result cache and failure ledger, default true
```

The plugin does not export a Config schema (that would depend on the host's
`@deepseek-ai/schemastery`); it normalizes and validates the values itself: an invalid `timeoutMs`
falls back to 60000, and `cache` only accepts boolean semantics.

## Tool List (14)

| Tool | Purpose |
| --- | --- |
| `math_simplify` | Simplify / expand / factor / combine / partial fractions |
| `math_solve` | Equations, equation systems, inequalities |
| `math_limit` | Limits (one-sided limits, sequence limits) |
| `math_derivative` | Derivatives / higher order / partial / gradient / Jacobian / Hessian / implicit / parametric |
| `math_integrate` | Indefinite / definite / improper integrals (the multiple-integral operator `integrate_multi` is registered in the engine but not exposed as a tool-level entry, see [docs/未实现与限制.md](docs/未实现与限制.md)) |
| `math_series` | Taylor & power series expansion, series summation and convergence |
| `math_dsolve` | Ordinary differential equations (IVPs, systems) |
| `math_matrix` | Determinant / inverse / rank / eigenvalues & eigenvectors / diagonalization / quadratic forms / linear systems |
| `math_probability` | Probability, distributions (pdf/pmf/cdf/quantiles), expectation/variance/covariance/correlation, moments, estimation, hypothesis tests, CLT |
| `math_numeric` | High-precision numerics, numerical integration, root finding, optimization, numeric comparison |
| `math_vector_calculus` | Line integrals / surface integrals / Green's / Gauss's / Stokes' formulas |
| `math_verify` | Result verification (re-checked via a different route than the computation) |
| `math_analysis` | Function analysis: domain / monotonicity & concavity / asymptotes / extrema / multivariable extrema / Lagrange multipliers / inequalities |
| `math_status` | Engine status, operator list, cache statistics |

Every tool returns a **unified JSON structure** (rendered into readable text by the plugin):
`success / operation / input / result / conditions / verification / method / error / warnings`.

### Verification semantics (`verification.status`)

| Value | Meaning |
| --- | --- |
| `independent` | An independent route exists (differentiate-and-substitute, substitute back into the equation, another formula / numerical quadrature, etc.) and agrees with the main result |
| `cross` | Cross-validation exists (e.g. symbolic result agrees with numeric result) |
| `numeric` | Only numeric evidence — **does not constitute a symbolic proof** |
| `unverified` | Could not be verified (the tool honestly says why, without fabricating) |

## How It Works

```
User's natural-language question
      │
      ▼
DSH Agent ── a system-prompt section (MATH_SECTION_ORDER = 2500) explains the four
      │       task modes and when to call each tool
      │  tool calling
      ▼
dsh-math plugin (Node, zero dependencies)
  entry.js      registers the 14 tools, injects the system prompt, normalizes config
  tools.js      tool name / description / parameters
  bridge.js     PythonBridge: resident subprocess, request timeout, cache & failure ledger
  render.js     unified result JSON → readable text
      │  {id, op, args} one JSON line → stdin; stdout replies line by line with
      │  {id, status, result, verification, …}
      ▼
Python resident engine (worker.py + mathkit/, 214 operators)
  ast / symbolic_core / calculus / linalg / probability / numeric /
  vector_calc / verifier / result … (SymPy / mpmath / NumPy / SciPy)
      │
      ▼
Unified result JSON (success / operation / input / result / conditions / verification / method / …)
```

Engine protocol: stderr is used for logs only (the plugin keeps the last 64 KiB for troubleshooting).
`math_status` reports engine status, the operator list and cache statistics.

## Repository Layout

```
dsh-math-plugin/
├── README.md / README.en.md   # bilingual documentation
├── LICENSE                    # MIT
├── docs/                      # delivery documents (Chinese)
│   ├── 工具说明.md             # tool reference: parameters, sub-operations, returns, examples
│   ├── 使用示例.md             # real conversation examples for the four task modes
│   ├── 测试报告.md             # test layers, real run records, reproduction commands
│   ├── 已实现功能清单.md        # implemented features + runtime dependencies
│   ├── 未实现与限制.md          # syllabus gaps and environment limitations
│   └── 交付说明.md             # deliverables index + 11 acceptance criteria + evidence
├── plugin/                    # installable plugin package (pack source)
│   ├── package.json  cordis.patch.yml  icon.svg  README.md
│   ├── locale/{zh,en}.json
│   ├── lib/{entry,define_tool,bridge,tools,render}.js
│   └── engine/                # synced from engine/ (_sync_engine.py)
├── engine/                    # engine source (development; edit here, then sync)
│   ├── worker.py              # resident process entry (JSON lines over stdio)
│   ├── mathkit/               # 14 .py files: 13 feature modules + __init__.py, 214 operators
│   └── _selftest_*.py         # 7 engine self-test scripts
├── _dist/dsh-math-1.0.2.tgz   # installable tarball
├── _test_acceptance.py        # end-to-end acceptance (111 cases) + result JSON
├── _test_special.py           # 8 special tests + result JSON
├── _sync_engine.py            # engine/ → plugin/engine sync (whitelist)
├── _pack.py                   # builds _dist/*.tgz (npm pack format)
├── _dump_specs.mjs            # dumps the real parameter tables of the 14 tools → _specs.json
├── _check_contract.py         # contract check: operator/parameter names vs. engine
└── _verify/                   # in-host-loader verification scripts and outputs
```

## Documentation

| Document | Content |
| --- | --- |
| [docs/工具说明.md](docs/工具说明.md) | Tool reference: names, parameters, sub-operation dispatch, returns, typical calls |
| [docs/使用示例.md](docs/使用示例.md) | Real conversation examples for the four task modes |
| [docs/测试报告.md](docs/测试报告.md) | Test layers, real run records, reproduction commands, bugs fixed along the way |
| [docs/已实现功能清单.md](docs/已实现功能清单.md) | Implemented features and runtime dependencies |
| [docs/未实现与限制.md](docs/未实现与限制.md) | Syllabus gaps and environment limitations |
| [docs/交付说明.md](docs/交付说明.md) | Deliverables index, 11 acceptance criteria with evidence, measurement records |
| [plugin/README.md](plugin/README.md) | npm-distributed install/enable/config/troubleshooting guide |

> The `docs/` files are written in Chinese; they remain the authoritative, code-accurate reference.

## Typical Prompts

| Task type | Example prompt | Expected behavior |
| --- | --- | --- |
| Pure computation | "Compute ∫₀^{π} sin x dx", "find the eigenvalues of A" | Calls the matching tool directly, returns the exact result with conditions |
| Verification / error hunting | "Verify ∫x e^x dx = (x-1)e^x", "is my solution correct: …" | Re-checks with `math_verify`, **pinpoints the exact wrong step** |
| Approach analysis | "How should I start this problem?", "why use Lagrange multipliers here?" | Explains the approach backed by structured conclusions, without wasted computation |
| Full solving | "Fully solve this problem: …" | All eight parts: problem analysis, exam-point identification, approach, key formulas, computation, result verification, final answer, pattern-recognition tips |
| Follow-up | "Expand on step 2" | Explains only that step, reusing verified results instead of re-solving |

## Known Limitations (Summary)

The complete list is in [docs/未实现与限制.md](docs/未实现与限制.md).

1. **Restart DSH after install/upgrade** to activate the bundle (the host's resolution table requires a process restart).
2. **Image input relies on DSH native multimodality**: the plugin does no OCR; if multimodal capability is unavailable, the user must transcribe the problem.
3. **Remaining syllabus gaps**: exact distributions of sampling statistics, conditional distributions, distributions of functions of random variables (convolution), bivariate normal, covariance matrices & random vectors, law of total expectation/variance, law of large numbers, order statistics, quantiles of χ²/F/t, etc.
4. **`unsolved` is an honest answer**: e.g. `∫x^x dx`, or quantiles of χ²/F/t (no closed-form inverse) — the tool never invents a closed form.

## Tests & Evidence (Summary)

All conclusions are reproducible from scripts in this repository (commands and real outputs in
[docs/测试报告.md](docs/测试报告.md)):

- End-to-end acceptance `_test_acceptance.py`: **111/111 passed, exit 0**, details in `_test_acceptance_result.json`;
- Eight special tests `_test_special.py`: **8/8 passed, exit 0** (all 214 operators registered, cache semantics, unevaluated-expression blocking, verification-level assertions, etc.);
- Engine self-tests `engine/_selftest_*.py`: all 7 scripts **exit 0**;
- In-host-loader verification: `_verify/scratch_boot_full.mjs` → `BOOT OK`, all 14 tools registered and executed successfully.

Measurement records and the criterion-by-criterion acceptance table are in
[docs/交付说明.md](docs/交付说明.md).

## License

[MIT](LICENSE) © [SixId666](https://github.com/SixId666)
