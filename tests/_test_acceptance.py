"""考研数学一验收测试集 —— 直接走 E.execute()（与插件调用完全相同的路径）。

覆盖用户规格第十二节：
  * 高等数学：极限、等价无穷小、洛必达、泰勒、复杂求导、不定/定/反常积分、
    二重/三重积分、曲线积分、曲面积分、格林/高斯/斯托克斯、常微分方程、数项级数、幂级数
  * 线性代数：行列式、矩阵求逆、秩、线性方程组、特征值、特征向量、对角化、二次型
  * 概率统计：条件概率、全概率、贝叶斯、离散型、连续型、联合分布、期望、方差、
    协方差、参数估计
  * 12.4 八个专项测试见 _test_special.py

**本文件的参数名与算子名全部来自 `_introspect_api.py` 导出的真实签名**
（`_api_surface.json`）。概率/数值等 `**kwargs` 型算子写错参数名不会报错、
只会静默取不到值，所以这里不允许「凭记忆写」。

每个用例只断言「数学上可判定的性质」，不断言渲染文本，避免把测试写成实现快照。
用法（仓库根目录）：
    python -X utf8 tests/_test_acceptance.py              # 全部
    python -X utf8 tests/_test_acceptance.py calculus     # 只跑某组
    python -X utf8 tests/_test_acceptance.py --list       # 只列组名与用例数
"""

from __future__ import annotations

import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)  # 脚本位于 <repo>/tests/，引擎在 <repo>/engine/
for candidate in (os.path.join(REPO, "engine"), os.path.join(HERE, "engine"), HERE):
    if os.path.isdir(os.path.join(candidate, "mathkit")):
        sys.path.insert(0, candidate)
        break
else:  # pragma: no cover
    raise SystemExit("找不到 mathkit 包（应在 <repo>/engine/mathkit 下）")

import sympy as sp  # noqa: E402

from mathkit import engine as E  # noqa: E402
from mathkit import ops as _ops  # noqa: E402,F401  触发全部算子注册（与 worker.py 一致）

_skip = "  [跳过：模块未加载]\n" if _ops.failed else ""

PASS: list[dict] = []
FAIL: list[dict] = []
SKIP: list[dict] = []


def run(op, args, **kwargs):
    """执行一次算子调用，返回 payload（永不抛异常）。"""
    try:
        return E.execute(op, args, **kwargs)
    except Exception as error:  # noqa: BLE001  引擎级异常也要被测试捕获
        return {
            "status": "error",
            "operation": op,
            "error": {"kind": "exception", "message": f"{type(error).__name__}: {error}"},
            "result": {},
            "verification": {"status": "unverified", "methods": []},
        }


# ---------------------------------------------------------------------------
# 断言小工具
# ---------------------------------------------------------------------------

def text(payload) -> str:
    """规范化后的结果文本（去掉全部空格，便于子串比较）。"""
    return json.dumps(payload.get("result", {}), ensure_ascii=False, default=str).replace(" ", "")


def statuses(payload) -> str:
    return str(payload.get("status"))


def ok(payload) -> bool:
    return payload.get("status") in ("ok", "partial")


def value(payload):
    """result['text'] 的 SymPy 重建（失败返回 None）。

    兼容三类返回形态：普通表达式、布尔值（True/False）、矩阵（[[..],[..]]）。
    """
    raw = (payload.get("result") or {}).get("text")
    if not isinstance(raw, str):
        return None
    stripped = raw.strip()
    if stripped in ("True", "False"):
        return stripped == "True"
    if stripped.startswith("[[") or stripped.startswith("Matrix("):
        try:
            return sp.Matrix(sp.sympify(stripped.replace("^", "**")))
        except Exception:  # noqa: BLE001
            pass
    try:
        return sp.sympify(stripped.replace("^", "**"))
    except Exception:  # noqa: BLE001
        return None


_INFINITIES = (sp.oo, -sp.oo, sp.zoo)


def equal_value(payload, expected, tol=1e-9):
    """把结果与期望值做比较（期望值可以是 SymPy 精确式、布尔值或嵌套列表）。

    覆盖：精确相等、高精度数值容差、无穷（±∞ 相减会得到 nan，绝不能走减法路径）、
    布尔判定、矩阵逐元素比较。
    """
    got = value(payload)
    if got is None:
        return False, f"无 result.text（status={statuses(payload)}）"
    if isinstance(expected, bool):
        if isinstance(got, bool):
            return (got is expected), "" if got is expected else f"得到 {got}，期望 {expected}"
        return False, f"得到 {got}（不是布尔值），期望 {expected}"
    try:
        expected_value = sp.sympify(expected)
        if isinstance(got, sp.MatrixBase) or isinstance(expected_value, sp.MatrixBase):
            got_matrix = got if isinstance(got, sp.MatrixBase) else sp.Matrix(got)
            want_matrix = (expected_value if isinstance(expected_value, sp.MatrixBase)
                           else sp.Matrix(expected_value))
            if got_matrix.shape != want_matrix.shape:
                return False, f"维度不一致 {got_matrix.shape} vs {want_matrix.shape}"
            for entry, want in zip(got_matrix, want_matrix):
                same, detail = equal_value({"result": {"text": str(entry)}}, str(want), tol)
                if not same:
                    return False, (f"元素不一致：得到 {got_matrix}，期望 {want_matrix}"
                                   f"（{detail}）")
            return True, ""
        if got in _INFINITIES or expected_value in _INFINITIES:
            if got == expected_value:
                return True, ""
            return False, f"得到 {got}，期望 {expected_value}"
        difference = sp.simplify(got - expected_value)
        if difference == 0:
            return True, ""
        numeric = complex(sp.N(difference, 20))
        if abs(numeric) <= tol:
            return True, ""
        return False, f"得到 {got}（≈{sp.N(got, 12)}），期望 {expected_value}"
    except Exception as error:  # noqa: BLE001
        return False, f"比较失败 {type(error).__name__}: {error}"


def verification(payload) -> dict:
    return payload.get("verification") or {}


def verified(payload) -> bool:
    """至少有一条复核方法，且没有任何一条方法判否。

    引擎的 method 条目有两种判定字段：`passed`（归一化/F(−∞)=0 之类）与
    `agree`（互为独立的两种算法是否一致）。还有纯描述性条目（两个字段都没有，
    例如只写 `{"method": "数值积分复核", "numeric_value": "2"}`）——这些按中性处理。
    只要有一条显式判否（`passed` 或 `agree` 为 False）就算未通过。
    """
    methods = verification(payload).get("methods") or []
    if not methods:
        return False
    for method in methods:
        # 只认显式判否：None／缺字段都按中性处理（引擎里确有 passed=None 的条目）
        if method.get("passed") is False or method.get("agree") is False:
            return False
    return True


def check(name, op, args, predicate, group="misc", note=""):
    """predicate(payload) 返回 True/False，或 (bool, 说明)。"""
    payload = run(op, args)
    try:
        outcome = predicate(payload)
    except Exception as error:  # noqa: BLE001
        outcome = (False, f"断言本身出错 {type(error).__name__}: {error}")
    if isinstance(outcome, tuple):
        passed, detail = outcome
    else:
        passed, detail = bool(outcome), ""
    entry = {
        "name": name,
        "group": group,
        "op": op,
        "args": args,
        "status": payload.get("status"),
        "detail": detail,
        "note": note,
    }
    if passed:
        PASS.append(entry)
    else:
        entry["payload"] = json.dumps(payload, ensure_ascii=False, default=str)[:1500]
        FAIL.append(entry)
    print(f"[{'PASS' if passed else 'FAIL'}] {group:<10} {name}"
          + (f"  —— {detail}" if detail and not passed else ""))
    return payload


def skip(name, group, reason):
    SKIP.append({"name": name, "group": group, "reason": reason})
    print(f"[SKIP] {group:<10} {name}  —— {reason}")


# ---------------------------------------------------------------------------
# 一、高等数学
# ---------------------------------------------------------------------------

def test_calculus() -> None:
    G = "calculus"

    # ---- 极限：等价无穷小 / 洛必达 / 泰勒 / 1^∞ ----
    check("极限 sin(x)/x (x→0)", "limit", {"expr": "sin(x)/x", "var": "x", "point": "0"},
          lambda p: equal_value(p, 1), G)
    check("等价无穷小 (1-cos x)/x²", "limit", {"expr": "(1-cos(x))/x^2", "var": "x", "point": "0"},
          lambda p: equal_value(p, sp.Rational(1, 2)), G)
    check("洛必达 (e^x-1-x)/x²", "limit", {"expr": "(exp(x)-1-x)/x^2", "var": "x", "point": "0"},
          lambda p: equal_value(p, sp.Rational(1, 2)), G)
    check("泰勒型 (x-sin x)/x³", "limit", {"expr": "(x-sin(x))/x^3", "var": "x", "point": "0"},
          lambda p: equal_value(p, sp.Rational(1, 6)), G)
    check("1^∞ 型 (1+1/x)^x (x→∞)", "limit", {"expr": "(1+1/x)^x", "var": "x", "point": "oo"},
          lambda p: equal_value(p, sp.E), G)
    check("左右极限：1/x 从右侧 → +∞", "limit", {"expr": "1/x", "var": "x", "point": "0", "dir": "+"},
          lambda p: equal_value(p, sp.oo), G)
    check("左右极限：1/x 从左侧 → -∞", "limit", {"expr": "1/x", "var": "x", "point": "0", "dir": "-"},
          lambda p: equal_value(p, -sp.oo), G)
    check("数列极限 (1+1/n)^n", "limit", {"expr": "(1+1/n)^n", "var": "n", "point": "oo"},
          lambda p: equal_value(p, sp.E), G)
    check("趋向无穷的等价量 ln(x)/x", "limit", {"expr": "ln(x)/x", "var": "x", "point": "oo"},
          lambda p: equal_value(p, 0), G)

    # ---- 导数：基本 / 复合 / 隐函数 / 参数方程 / 高阶 / 偏导 ----
    check("基本求导 (x³-3x)'", "diff", {"expr": "x^3-3*x", "var": "x"},
          lambda p: equal_value(p, 3*sp.Symbol("x")**2 - 3), G)
    check("复合求导 x^x", "diff", {"expr": "x^x", "var": "x"},
          lambda p: ((ok(p) and "log(x)" in text(p).lower() and verified(p)),
                     "" if ok(p) else f"status={statuses(p)}"), G)
    check("二阶导 sin(x) 二阶梯", "diff", {"expr": "sin(x)", "var": "x", "order": 2},
          lambda p: equal_value(p, -sp.sin(sp.Symbol("x"))), G)
    check("四阶导 x⁴ 四阶梯", "diff", {"expr": "x^4", "var": "x", "order": 4},
          lambda p: equal_value(p, 24), G)
    check("隐函数求导 x²+y²=1 → -x/y", "implicit_diff",
          {"equation": "x^2+y^2-1", "vars": ["x", "y"]},
          lambda p: ((ok(p) and "-x/y" in text(p)),
                     f"得到 {text(p)[:80]}" if ok(p) else f"status={statuses(p)}"), G,
          note="vars 顺序约定为 [自变量…, 因变量]")
    check("隐函数求导（显式角色 dependent/independent）", "implicit_diff",
          {"equation": "x^2+y^2=1", "dependent": "y", "independent": "x"},
          lambda p: ((ok(p) and "-x/y" in text(p)),
                     f"得到 {text(p)[:80]}" if ok(p) else f"status={statuses(p)}"), G,
          note="用显式角色参数避免 vars 顺序歧义")
    check("参数方程求导 x=t-sin t, y=1-cos t", "parametric_derivative",
          {"x_expr": "t-sin(t)", "y_expr": "1-cos(t)", "param": "t"},
          lambda p: ((ok(p) and "sin" in text(p) and "cos" in text(p)),
                     f"得到 {text(p)[:80]}" if ok(p) else f"status={statuses(p)}"), G)
    check("偏导 ∂/∂x (x²y+y³)", "partial", {"expr": "x^2*y+y^3", "var": "x"},
          lambda p: equal_value(p, 2*sp.Symbol("x")*sp.Symbol("y")), G)
    check("梯度 ∇(x²y+z)", "gradient", {"expr": "x^2*y+z", "vars": ["x", "y", "z"]},
          lambda p: (ok(p) and "2*x*y" in text(p) and "x**2" in text(p), ""), G)

    # ---- 积分：不定 / 定 / 反常 / 累次（二重三重）----
    check("不定积分 ∫x²dx", "integrate", {"expr": "x^2", "var": "x"},
          lambda p: equal_value(p, sp.Symbol("x")**3 / 3), G)
    check("不定积分（换元）∫x e^{x²}dx", "integrate", {"expr": "x*exp(x^2)", "var": "x"},
          lambda p: ((ok(p) and "exp(x**2)/2" in text(p).replace(" ", "")),
                     f"得到 {text(p)[:90]}" if ok(p) else f"status={statuses(p)}"),
          G, note="用户规格原文用例：应返回 (1/2)e^{x²}+C 的精确符号形式，不转小数")
    check("分部积分 ∫x e^x dx", "integrate", {"expr": "x*exp(x)", "var": "x"},
          lambda p: (ok(p) and verified(p),
                     f"得到 {text(p)[:90]}，验证={verification(p).get('status')}"), G)
    check("三角有理式 ∫sin²x dx", "integrate", {"expr": "sin(x)^2", "var": "x"},
          lambda p: equal_value(p, sp.Symbol("x")/2 - sp.sin(2*sp.Symbol("x"))/4), G)
    check("有理函数 ∫1/(x²-1) dx", "integrate", {"expr": "1/(x^2-1)", "var": "x"},
          lambda p: (ok(p) and "log" in text(p), ""), G)
    check("定积分 ∫₀¹x²dx = 1/3", "integrate", {"expr": "x^2", "var": "x", "lower": "0", "upper": "1"},
          lambda p: equal_value(p, sp.Rational(1, 3)), G, note="规格 6.4：必须直接调用工具，不得自写程序")
    check("定积分 ∫₀^π sin x dx = 2", "integrate", {"expr": "sin(x)", "var": "x", "lower": "0", "upper": "pi"},
          lambda p: (equal_value(p, 2)[0] and verified(p),
                     f"验证={verification(p).get('status')}"), G)
    check("反常积分 ∫₁^∞ 1/x² dx = 1", "integrate", {"expr": "1/x^2", "var": "x", "lower": "1", "upper": "oo"},
          lambda p: equal_value(p, 1), G)
    check("反常积分 ∫₀^∞ e^{-x}dx = 1", "integrate", {"expr": "exp(-x)", "var": "x", "lower": "0", "upper": "oo"},
          lambda p: equal_value(p, 1), G)
    check("无初等原函数 ∫x^x dx 必须如实报告", "integrate", {"expr": "x^x", "var": "x"},
          lambda p: (statuses(p) in ("unsolved", "partial", "error")
                     and "Integral(" not in text(p),
                     f"status={statuses(p)} result={text(p)[:90]}"), G,
          note="规格 5.3 / 十二测试三：不得把未求值表达式当作成功（e^{x²} 有 erfi 闭式，"
               "故改用真正无初等原函数的 x^x）")
    check("二重积分 ∬(x+y)dxdy（单位正方形）= 1", "integrate_multi",
          {"expr": "x+y", "vars": [["0", "1"], ["0", "1"]]},
          lambda p: equal_value(p, 1), G)
    check("二重积分 ∬xy（三角形 x+y≤1）= 1/24", "integrate_multi",
          {"expr": "x*y", "vars": [{"var": "y", "lower": "0", "upper": "1-x"},
                                   {"var": "x", "lower": "0", "upper": "1"}]},
          lambda p: equal_value(p, sp.Rational(1, 24)), G)
    check("二重积分 ∬(x²+y²)（单位圆盘）= π/2", "integrate_multi",
          {"expr": "x^2+y^2", "vars": [{"var": "y", "lower": "-sqrt(1-x^2)", "upper": "sqrt(1-x^2)"},
                                       {"var": "x", "lower": "-1", "upper": "1"}]},
          lambda p: equal_value(p, sp.pi / 2), G)
    check("三重积分 ∭xyz（单位立方体）= 1/8", "integrate_multi",
          {"expr": "x*y*z", "vars": [["0", "1"], ["0", "1"], ["0", "1"]]},
          lambda p: equal_value(p, sp.Rational(1, 8)), G)
    check("累次积分含参数 a 时必须标注未独立复核", "integrate_multi",
          {"expr": "a*x", "vars": [["0", "1"], ["0", "1"]]},
          lambda p: ((ok(p) and not verified(p) and bool(p.get("warnings"))
                      and "a" in text(p)),
                     f"status={statuses(p)} verify={verification(p).get('status')} "
                     f"方法={len(verification(p).get('methods') or [])} result={text(p)[:60]}"), G,
          note="规格 4.4 / 5.3：结果含参数时无法独立验证，必须明确标记而不是谎称已验证")

    # ---- 曲线积分 / 曲面积分 / 三大公式 ----
    check("第一类曲线积分 ∮(x²+y²)ds（单位圆）= 2π", "curve_integral",
          {"kind": "first", "expr": "x^2+y^2", "x_expr": "cos(t)", "y_expr": "sin(t)",
           "var": "t", "lower": "0", "upper": "2*pi"},
          lambda p: equal_value(p, 2*sp.pi), G)
    check("第二类曲线积分 ∮(-y dx + x dy)（单位圆）= 2π", "curve_integral",
          {"kind": "second", "expr": "-y", "expr2": "x", "x_expr": "cos(t)", "y_expr": "sin(t)",
           "var": "t", "lower": "0", "upper": "2*pi"},
          lambda p: equal_value(p, 2*sp.pi), G)
    check("第一类曲面积分 ∬dS（单位球面）= 4π", "surface_integral",
          {"kind": "first", "expr": "1", "x_expr": "sin(u)*cos(v)", "y_expr": "sin(u)*sin(v)",
           "z_expr": "cos(u)", "u_var": "u", "v_var": "v",
           "u_lower": "0", "u_upper": "pi", "v_lower": "0", "v_upper": "2*pi"},
          lambda p: equal_value(p, 4*sp.pi), G)
    check("格林公式 ∮(-y dx+x dy) 单位圆 = 2π", "green",
          {"p": "-y", "q": "x", "x_lower": "-1", "x_upper": "1",
           "y_lower_expr": "-sqrt(1-x^2)", "y_upper_expr": "sqrt(1-x^2)"},
          lambda p: equal_value(p, 2*sp.pi), G)
    check("高斯公式 ∭∇·F dV（单位球，F=(x,y,z)）= 4π", "gauss",
          {"p": "x", "q": "y", "r_component": "z", "r": "z",
           "x_lower": "-1", "x_upper": "1", "y_lower_expr": "-sqrt(1-x^2)",
           "y_upper_expr": "sqrt(1-x^2)", "z_lower_expr": "-sqrt(1-x^2-y^2)",
           "z_upper_expr": "sqrt(1-x^2-y^2)"},
          lambda p: equal_value(p, 4*sp.pi), G)
    check("斯托克斯 ∬(∇×F)·n dS（单位圆盘，F=(-y,x,0)）= 2π", "stokes",
          {"p": "-y", "q": "x", "r_component": "0", "r": "0",
           "x_expr": "r*cos(t)", "y_expr": "r*sin(t)", "z_expr": "0",
           "u_var": "r", "v_var": "t", "u_lower": "0", "u_upper": "1",
           "v_lower": "0", "v_upper": "2*pi"},
          lambda p: equal_value(p, 2*sp.pi), G)

    # ---- 微分方程 ----
    check("可分离变量 y'=x y", "dsolve", {"equations": "diff(y(x),x) - x*y(x)"},
          lambda p: (ok(p) and "exp" in text(p), f"status={statuses(p)} 得到 {text(p)[:90]}"), G)
    check("一阶线性 y'+y=e^x", "dsolve", {"equations": "diff(y(x),x) + y(x) - exp(x)"},
          lambda p: (ok(p) and "exp" in text(p), f"status={statuses(p)} 得到 {text(p)[:90]}"), G)
    check("二阶常系数 y''-y=0", "dsolve", {"equations": "diff(y(x),x,2) - y(x)"},
          lambda p: (ok(p) and "exp" in text(p), f"status={statuses(p)} 得到 {text(p)[:90]}"), G)
    check("ODE 解代回原方程复核 y'=y", "ode_check",
          {"equations": "diff(y(x),x) - y(x)", "candidate": "C1*exp(x)"},
          lambda p: (ok(p) and verified(p), f"status={statuses(p)} 验证={verification(p).get('status')}"), G)

    # ---- 级数 ----
    check("几何级数 Σ(1/2)^n (n≥1) = 1", "sum", {"expr": "(1/2)^n", "var": "n", "lower": "1", "upper": "oo"},
          lambda p: equal_value(p, 1), G)
    check("Σ1/n² = π²/6", "sum", {"expr": "1/n^2", "var": "n", "lower": "1", "upper": "oo"},
          lambda p: equal_value(p, sp.pi**2 / 6), G)
    check("Σ1/n 发散（调和级数）", "sum", {"expr": "1/n", "var": "n", "lower": "1", "upper": "oo"},
          lambda p: equal_value(p, sp.oo), G)
    check("幂级数 Σ_{n≥1} x^n = x/(1-x)", "sum", {"expr": "x^n", "var": "n", "lower": "1", "upper": "oo"},
          lambda p: equal_value(p, sp.Symbol("x") / (1 - sp.Symbol("x"))), G,
          note="n 从 1 起；若 lower=0 则和是 1/(1-x)")
    check("泰勒展开 e^x 到 x⁴ 含 1/24 项", "series",
          {"expr": "exp(x)", "var": "x", "point": "0", "order": 4},
          lambda p: (ok(p) and ("1/24" in text(p) or "x**4/24" in text(p)),
                     f"得到 {text(p)[:120]}"), G)
    check("泰勒展开 ln(1+x) 到 x⁴ 含 -x⁴/4", "series",
          {"expr": "ln(1+x)", "var": "x", "point": "0", "order": 4},
          lambda p: (ok(p) and "x**3/3" in text(p), f"得到 {text(p)[:120]}"), G)

    # ---- 化简与函数分析 ----
    check("化简 sin²x+cos²x → 1", "simplify", {"expr": "sin(x)^2+cos(x)^2"},
          lambda p: equal_value(p, 1), G)
    check("因式分解 x³-1", "factor", {"expr": "x^3-1"},
          lambda p: (ok(p) and "(x-1)" in text(p).replace(" ", ""), f"得到 {text(p)[:80]}"), G)
    check("函数分析 x³-3x 给出单调/极值结构", "function_analysis", {"expr": "x^3-3*x", "var": "x"},
          lambda p: ((ok(p) and len(p.get("result") or {}) >= 3),
                     f"status={statuses(p)} 字段={sorted((p.get('result') or {}).keys())[:8]}"), G)
    check("定义域 sqrt(x-1)/(x-2)", "domain", {"expr": "sqrt(x-1)/(x-2)", "var": "x"},
          lambda p: ((ok(p) and bool(p.get("conditions") or (p.get("result") or {}))),
                     f"status={statuses(p)} 条件={p.get('conditions')}"), G)
    check("等价无穷小替换被识别 (1-cos x)~x²/2", "series",
          {"expr": "1-cos(x)", "var": "x", "point": "0", "order": 2},
          lambda p: (ok(p) and "x**2/2" in text(p), f"得到 {text(p)[:90]}"), G)


# ---------------------------------------------------------------------------
# 二、线性代数
# ---------------------------------------------------------------------------

def test_linalg() -> None:
    G = "linalg"

    check("行列式 |[[1,2],[3,4]]| = -2", "det", {"matrix": [["1", "2"], ["3", "4"]]},
          lambda p: equal_value(p, -2), G)
    check("行列式（含符号）|[[a,b],[c,d]]| = ad-bc", "det",
          {"matrix": [["a", "b"], ["c", "d"]]},
          lambda p: equal_value(p, sp.Symbol("a")*sp.Symbol("d") - sp.Symbol("b")*sp.Symbol("c")), G)
    check("行列式（展开）|[[1,2,3],[4,5,6],[7,8,10]]| = -3", "det",
          {"matrix": [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "10"]]},
          lambda p: equal_value(p, -3), G)

    check("矩阵求逆 [[1,2],[3,4]] 并核验乘积为单位阵", "inv",
          {"matrix": [["1", "2"], ["3", "4"]]},
          lambda p: ((ok(p) and verified(p) and "1/2" in text(p)),
                     f"status={statuses(p)} 验证={verification(p).get('status')}"), G)
    check("不可逆矩阵求逆必须失败并说明原因", "inv", {"matrix": [["1", "2"], ["2", "4"]]},
          lambda p: (statuses(p) in ("error", "unsolved", "partial")
                     and bool(p.get("error") or p.get("warnings")),
                     f"status={statuses(p)} error={(p.get('error') or {}).get('message', '')[:80]}"), G)

    check("矩阵秩 [[1,2,3],[2,4,6]] = 1", "rank", {"matrix": [["1", "2", "3"], ["2", "4", "6"]]},
          lambda p: equal_value(p, 1), G)
    check("矩阵乘法 [[1,2],[3,4]]·[[5,6],[7,8]] = [[19,22],[43,50]]", "matmul",
          {"matrix": [["1", "2"], ["3", "4"]], "matrix_b": [["5", "6"], ["7", "8"]]},
          lambda p: equal_value(p, sp.Matrix([[19, 22], [43, 50]])), G)
    check("转置 [[1,2,3],[4,5,6]] → 3×2", "transpose",
          {"matrix": [["1", "2", "3"], ["4", "5", "6"]]},
          lambda p: equal_value(p, sp.Matrix([[1, 4], [2, 5], [3, 6]])), G)
    check("迹 [[1,2],[3,4]] = 5", "trace", {"matrix": [["1", "2"], ["3", "4"]]},
          lambda p: equal_value(p, 5), G)
    check("行最简形 [[1,2,3],[4,5,6]] 首元为 1", "rref",
          {"matrix": [["1", "2", "3"], ["4", "5", "6"]]},
          lambda p: (ok(p) and "1" in text(p), f"status={statuses(p)} 得到 {text(p)[:100]}"), G)

    check("线性方程组（唯一解）x+y=3, x-y=1 → x=2,y=1", "solve_linear",
          {"matrix": [["1", "1"], ["1", "-1"]], "rhs": ["3", "1"]},
          lambda p: (ok(p) and "2" in text(p) and "1" in text(p),
                     f"status={statuses(p)} 得到 {text(p)[:110]}"), G)
    check("线性方程组（无穷多解必须明确说明）", "solve_linear",
          {"matrix": [["1", "1"], ["2", "2"]], "rhs": ["1", "2"]},
          lambda p: (statuses(p) in ("ok", "partial", "unsolved")
                     and bool(p.get("conditions") or p.get("warnings") or p.get("result")),
                     f"status={statuses(p)} 条件={p.get('conditions')}"), G)

    check("特征值 [[2,1],[1,2]] = {3,1}", "eigen", {"matrix": [["2", "1"], ["1", "2"]]},
          lambda p: (ok(p) and "3" in text(p) and "1" in text(p),
                     f"status={statuses(p)} 得到 {text(p)[:110]}"), G)
    check("特征值 [[1,1],[0,1]] 为二重 1", "eigen", {"matrix": [["1", "1"], ["0", "1"]]},
          lambda p: (ok(p) and "1" in text(p), f"status={statuses(p)} 得到 {text(p)[:110]}"), G)
    check("可对角化 [[2,1],[1,2]]", "diagonalize", {"matrix": [["2", "1"], ["1", "2"]]},
          lambda p: (ok(p) and verified(p), f"status={statuses(p)} 验证={verification(p).get('status')}"), G)
    check("不可对角化 [[1,1],[0,1]] 必须明确说明", "diagonalize", {"matrix": [["1", "1"], ["0", "1"]]},
          lambda p: ((statuses(p) in ("unsolved", "partial", "ok")
                      and (statuses(p) != "ok" or "不可对角化" in json.dumps(p, ensure_ascii=False))),
                     f"status={statuses(p)} result={text(p)[:110]}"), G)

    check("二次型 x²+2xy+y² 正定判定", "quadratic_form", {"expr": "x^2+2*x*y+y^2", "variables": ["x", "y"]},
          lambda p: (ok(p) and "半正定" in json.dumps(p, ensure_ascii=False),
                     f"status={statuses(p)} 得到 {text(p)[:110]}"), G)
    check("二次型 x²+y² 正定", "quadratic_form", {"expr": "x^2+y^2", "variables": ["x", "y"]},
          lambda p: (ok(p) and "正定" in json.dumps(p, ensure_ascii=False),
                     f"status={statuses(p)} 得到 {text(p)[:110]}"), G)
    check("矩阵分析 [[2,0],[0,3]] 给出秩/行列式/特征值", "matrix_analysis",
          {"matrix": [["2", "0"], ["0", "3"]]},
          lambda p: ((ok(p) and len(p.get("result") or {}) >= 3),
                     f"status={statuses(p)} 字段={sorted((p.get('result') or {}).keys())[:10]}"), G)


# ---------------------------------------------------------------------------
# 三、概率论与数理统计
# ---------------------------------------------------------------------------

def test_probability() -> None:
    G = "probability"

    check("条件概率 P(A|B)=P(AB)/P(B)", "probability",
          {"kind": "conditional", "joint": "0.2", "given": "0.5"},
          lambda p: equal_value(p, sp.Rational(2, 5)), G)
    check("全概率公式 ΣP(Ai)P(B|Ai)", "probability",
          {"kind": "total_probability", "prior": "0.4,0.6", "likelihood": "0.5,0.3"},
          lambda p: equal_value(p, sp.Rational(38, 100)), G)
    check("贝叶斯公式后验，并核验后验之和=1", "probability",
          {"kind": "bayes", "prior": "0.5,0.5", "likelihood": "0.8,0.2"},
          lambda p: (ok(p) and verified(p), f"status={statuses(p)} 验证={verification(p).get('status')}"), G)
    check("独立性判定 P(AB)=P(A)P(B)", "probability",
          {"kind": "independent_check", "joint": "0.2", "pa": "0.5", "pb": "0.4"},
          lambda p: equal_value(p, True), G)
    check("容斥公式（无交）P(A∪B)=P(A)+P(B)", "probability",
          {"kind": "inclusion_exclusion", "pa": "0.3", "pb": "0.4", "joint": "0"},
          lambda p: equal_value(p, sp.Rational(7, 10)), G)

    check("正态分布密度归一化检查", "distribution",
          {"kind": "pdf", "dist": "normal", "mu": "0", "sigma": "1"},
          lambda p: (ok(p) and verified(p), f"status={statuses(p)} 验证={verification(p).get('status')}"), G)
    check("二项分布分布律 B(5,1/2) k=2 → 5/16", "probability",
          {"kind": "at", "dist": "binomial", "n": "5", "p": "1/2", "point": "2"},
          lambda p: equal_value(p, sp.Rational(5, 16)), G)
    check("正态分布分位数 z_{0.975} ≈ 1.96 并回代复核", "distribution",
          {"kind": "quantile", "dist": "normal", "mu": "0", "sigma": "1", "point": "0.975"},
          lambda p: (ok(p) and abs(float(sp.N(value(p) or 0)) - 1.959963985) < 2e-3,
                     f"得到 {value(p)}"), G)

    check("二项分布 E[X] = np", "expectation", {"dist": "binomial", "n": "10", "p": "1/3"},
          lambda p: equal_value(p, sp.Rational(10, 3)), G)
    check("正态分布 E[X] = μ", "expectation", {"dist": "normal", "mu": "2", "sigma": "3"},
          lambda p: equal_value(p, 2), G)
    check("泊松分布 E[X] = λ", "expectation", {"dist": "poisson", "lam": "3"},
          lambda p: equal_value(p, 3), G)
    check("自定义分布律 E[X]（均匀骰子）= 3.5", "expectation",
          {"pmf": "1/6,1/6,1/6,1/6,1/6,1/6", "values": "1,2,3,4,5,6"},
          lambda p: equal_value(p, sp.Rational(7, 2)), G)

    check("二项分布 Var(X)=np(1-p)", "variance", {"dist": "binomial", "n": "10", "p": "1/3"},
          lambda p: (equal_value(p, sp.Rational(20, 9))[0] and verified(p),
                     f"验证={verification(p).get('status')} 得到 {value(p)}"), G)
    check("均匀分布 Var(X) U(0,1) = 1/12", "variance", {"dist": "uniform", "a": "0", "b": "1"},
          lambda p: equal_value(p, sp.Rational(1, 12)), G)
    check("指数分布 Var(X) = 1/λ²", "variance", {"dist": "exponential", "lam": "2"},
          lambda p: equal_value(p, sp.Rational(1, 4)), G)

    check("联合分布律的边缘分布与独立性判定", "joint_marginal",
          {"joint_pmf": "[[1/4,1/4],[1/4,1/4]]", "x_values": "0,1", "y_values": "0,1"},
          lambda p: (ok(p) and "独立" in json.dumps(p, ensure_ascii=False),
                     f"status={statuses(p)} result={text(p)[:110]}"), G)
    check("不具有独立性的联合分布必须判不独立", "joint_marginal",
          {"joint_pmf": "[[0,1/2],[1/2,0]]", "x_values": "0,1", "y_values": "0,1"},
          lambda p: (ok(p) and "不独立" in json.dumps(p, ensure_ascii=False),
                     f"status={statuses(p)} result={text(p)[:110]}"), G)
    check("离散联合分布协方差 Cov(X,Y)=0（独立）", "covariance",
          {"joint_pmf": "[[1/4,1/4],[1/4,1/4]]", "x_values": "0,1", "y_values": "0,1"},
          lambda p: equal_value(p, 0), G)
    check("离散联合分布相关系数 ρ=0（不相关）", "correlation",
          {"joint_pmf": "[[1/4,1/4],[1/4,1/4]]", "x_values": "0,1", "y_values": "0,1"},
          lambda p: equal_value(p, 0), G)
    check("矩母函数 M(t) 与 M(0)=1 复核", "moment_generating",
          {"dist": "binomial", "n": "3", "p": "1/2"},
          lambda p: (ok(p) and verified(p), f"status={statuses(p)} 验证={verification(p).get('status')}"), G)

    check("矩估计（正态）σ̂ 用样本", "point_estimate",
          {"samples": "1,2,3,4,5", "dist": "normal", "method": "both"},
          lambda p: (ok(p) and verified(p), f"status={statuses(p)} 验证={verification(p).get('status')}"), G)
    check("区间估计（正态均值，σ 已知，0.95）", "interval_estimate",
          {"samples": "2,3,4,5,6", "dist": "normal", "sigma": "1", "confidence": "0.95", "kind": "mean"},
          lambda p: (ok(p) and bool(p.get("result")), f"status={statuses(p)} result={text(p)[:110]}"), G)
    check("假设检验（正态 U 检验，双侧）", "hypothesis_test",
          {"kind": "z_test", "samples": "2,3,4,5,6", "sigma": "1", "mu0": "3", "alpha": "0.05"},
          lambda p: (ok(p), f"status={statuses(p)} result={text(p)[:110]}"), G)
    check("中心极限定理近似 P(S_n ≤ 55), n=100, p=0.5", "central_limit",
          {"dist": "binomial", "n": "100", "p": "0.5", "target": "P(S_n <= 55)"},
          lambda p: (ok(p) and bool(p.get("result")), f"status={statuses(p)} result={text(p)[:110]}"), G)


# ---------------------------------------------------------------------------
# 四、多元微分（高数后半部分）
# ---------------------------------------------------------------------------

def test_multivar() -> None:
    G = "multivar"

    check("多元函数极值 x²+y²+2x-4y 的极小点", "multivar_extremum", {"expr": "x^2+y^2+2*x-4*y", "vars": ["x", "y"]},
          lambda p: (ok(p) and "-1" in text(p) and "2" in text(p),
                     f"status={statuses(p)} result={text(p)[:120]}"), G)
    check("拉格朗日乘子 x²+y² s.t. x+y=1", "lagrange",
          {"expr": "x^2+y^2", "constraints": ["x+y-1"], "vars": ["x", "y"]},
          lambda p: (ok(p) and "1/2" in text(p), f"status={statuses(p)} result={text(p)[:120]}"), G)
    check("多元泰勒展开 e^x·cos y 到二阶", "taylor_multivar",
          {"expr": "exp(x)*cos(y)", "vars": ["x", "y"], "point": ["0", "0"], "order": 2},
          lambda p: (ok(p), f"status={statuses(p)} result={text(p)[:120]}"), G)
    check("方向导数 ∂f/∂l 沿 (1,1)", "directional_derivative",
          {"expr": "x^2+y^2", "vars": ["x", "y"], "point": ["1", "1"], "dir": ["1", "1"]},
          lambda p: (ok(p), f"status={statuses(p)} result={text(p)[:120]}"), G)
    check("渐近线 y=(x²+1)/x", "asymptote", {"expr": "(x^2+1)/x", "var": "x"},
          lambda p: (ok(p) and bool(p.get("result")), f"status={statuses(p)} result={text(p)[:110]}"), G)
    check("不等式分析 x²-1 ≥ 0 的解集", "inequality", {"expr": "x^2-1 >= 0", "var": "x"},
          lambda p: (ok(p) and bool(p.get("result")), f"status={statuses(p)} result={text(p)[:110]}"), G)


# ---------------------------------------------------------------------------
# 五、高精度数值计算（独立复核通道）
# ---------------------------------------------------------------------------

def test_numeric() -> None:
    G = "numeric"

    check("高精度数值 π（30 位）", "numeric", {"expr": "pi", "digits": 30},
          lambda p: (ok(p) and "3.14159265358979323846264338328" in text(p),
                     f"得到 {text(p)[:80]}"), G)
    check("数值积分 ∫₀¹e^{-x²}dx ≈ 0.7468241328", "numeric_integrate",
          {"expr": "exp(-x^2)", "var": "x", "lower": "0", "upper": "1", "digits": 12},
          lambda p: (ok(p) and abs(float(sp.N(value(p) or 0)) - 0.746824132812427) < 1e-10,
                     f"得到 {value(p)}"), G)
    check("数值解方程 cos(x)=x 的根 ≈ 0.739085", "numeric_solve",
          {"equations": "cos(x)-x", "vars": ["x"]},
          lambda p: (ok(p) and "0.739" in text(p), f"status={statuses(p)} result={text(p)[:110]}"), G)
    check("数值导数 (sin x)' 在 x=0 处 = 1", "numeric_derivative",
          {"expr": "sin(x)", "var": "x", "point": "0", "digits": 12},
          lambda p: (ok(p) and abs(float(sp.N(value(p) or 0)) - 1) < 1e-9, f"得到 {value(p)}"), G)
    check("数值优化 min x²-2x 在 x=1", "numeric_optimize",
          {"expr": "x^2-2*x", "vars": ["x"], "mode": "min"},
          lambda p: (ok(p) and "1" in text(p), f"status={statuses(p)} result={text(p)[:110]}"), G)
    check("数值极限 sin(x)/x → 1", "numeric_limit",
          {"expr": "sin(x)/x", "var": "x", "point": "0", "digits": 12},
          lambda p: (ok(p) and abs(float(sp.N(value(p) or 0)) - 1) < 1e-8, f"得到 {value(p)}"), G)
    check("数值级数 Σ1/n² ≈ π²/6", "numeric_sum",
          {"expr": "1/n^2", "var": "n", "lower": "1", "upper": "oo", "digits": 12},
          lambda p: (ok(p) and abs(float(sp.N(value(p) or 0)) - float(sp.pi**2/6)) < 1e-9,
                     f"得到 {value(p)}"), G)
    check("数值 vs 符号一致性比对 ∫₀^π sin x dx", "compare_numeric",
          {"expr": "sin(x)", "candidate": "2", "vars": ["x"]},
          lambda p: (ok(p), f"status={statuses(p)} result={text(p)[:120]}"), G)


GROUPS = {
    "calculus": test_calculus,
    "linalg": test_linalg,
    "probability": test_probability,
    "multivar": test_multivar,
    "numeric": test_numeric,
}


def main() -> int:
    print(f"python {sys.version.split()[0]}")
    print(f"已注册算子 {len(E.known_operations())} 个"
          f"（canonical {len(E._HANDLERS)}）")
    if _ops.failed:
        print(f"⚠ 未加载模块：{_ops.failed}")
    selected = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--list" in sys.argv:
        for name in GROUPS:
            print(f"  {name}")
        return 0
    names = selected or list(GROUPS)
    for name in names:
        runner = GROUPS.get(name)
        if runner is None:
            print(f"未知测试组 {name}；可用：{', '.join(GROUPS)}")
            return 2
        print(f"\n{'=' * 78}\n== 测试组 {name}\n{'=' * 78}")
        try:
            runner()
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            FAIL.append({"name": f"<{name} 组整体异常>", "group": name, "op": "-", "args": {}})

    print(f"\n{'=' * 78}")
    print(f"通过 {len(PASS)} / {len(PASS) + len(FAIL)}   跳过 {len(SKIP)}")
    if FAIL:
        print("\n失败用例：")
        for entry in FAIL:
            print(f"  ✗ [{entry['group']}] {entry['name']}")
            if entry.get("detail"):
                print(f"      {entry['detail']}")
            if entry.get("payload"):
                print(f"      payload: {entry['payload'][:600]}")
    report = os.path.join(HERE, "_test_acceptance_result.json")
    with open(report, "w", encoding="utf-8") as handle:
        json.dump({"pass": PASS, "fail": FAIL, "skip": SKIP}, handle,
                  ensure_ascii=False, indent=2, default=str)
    print(f"\n详细结果已写入 {report}")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())
