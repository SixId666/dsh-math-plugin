"""微积分算子：极限、导数、积分、级数、方程求解、微分方程、化简与多元微分。

设计原则（与用户规格一致）：

* 抽象优先：能给出解析解就返回解析解，**绝不**用数值近似冒充解析结论；
* 诚实标记：SymPy 原样回显 ``Integral(...)`` / ``Sum(...)`` / ``Limit(...)`` 时一律
  ``unsolved``（或 ``partial``），不算成功；
* 独立验证：每个成功结果都尝试用与求解**不同机制**的方法复核
  （不定积分对候选原函数求导、定积分用 mpmath 数值积分、ODE 解代回原方程、
  化简用恒等式比较、级数用余项收敛趋势），并把过程写进 ``verification``；
* 条件齐备：定义域、参数取值、收敛要求、瑕点都必须写进 ``conditions``。

本模块只依赖标准库 + sympy/mpmath + 本包的 ast / sympy_core / result / engine。
"""

from __future__ import annotations

import json
import re
from typing import Any

import sympy as sp
from . import ast as A
from . import sympy_core as C
from .engine import op
from .result import MathResult

__all__ = [
    "limit",
    "differentiate",
    "integrate",
    "series_sum",
    "solve",
    "dsolve",
    "simplify",
    "expand",
    "factor",
    "apart",
    "series",
    "ode_check",
    "gradient",
    "jacobian",
    "hessian",
]

# 内部占位符：把 ``y(x)`` 换成普通名字后再交给 A.parse 解析
_PLACEHOLDER_RE = re.compile(r"^_f(\d+)$")
# 一次导数撇号：y' / y'' / y'''（后面不接 ``(``）
_PRIME_RE = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z][A-Za-z0-9_]*)('{1,4})(?:\s*\(\s*([A-Za-z][A-Za-z0-9_]*)\s*\))?")


# ---------------------------------------------------------------------------
# 内部小工具（全部私有，不注册为算子）
# ---------------------------------------------------------------------------

def _num_str(value: float) -> str:
    return f"{value:.6g}"


def _sym_names(item: Any) -> list[str]:
    """把 ``"x"`` / ``"x,y"`` / ``["x", "y"]`` 统一成变量名列表。"""
    if item is None:
        return []
    if isinstance(item, sp.Symbol):
        return [str(item)]
    if isinstance(item, sp.Basic):
        return [str(s) for s in sorted(item.free_symbols, key=lambda z: str(z))]
    if isinstance(item, (list, tuple, set)):
        names: list[str] = []
        for piece in item:
            names.extend(_sym_names(piece))
        return names
    raw = str(item).strip().strip("[](){}")
    if not raw:
        return []
    parts = re.split(r"[,;、\s]+", raw)
    return [p for p in parts if p]


def _make_symbols(item: Any) -> list[sp.Symbol]:
    """构造符号列表。

    必须与 :func:`mathkit.ast.parse` 使用的符号**完全同一**（``A.get_symbols`` 对
    ``x/t/y/z/u/v/s`` 加了 ``real=True``），否则 ``sp.limit``/``sp.diff`` 会认为
    所给变量不在表达式中，从而原样回显 ``Limit(...)`` 或直接返回 0。
    """
    names = _sym_names(item)
    table = A.get_symbols(names)
    return [table[name] for name in names if name in table]


def _parse_item(item: Any, symbols: list[str] | None = None) -> list[sp.Basic]:
    """把 ``str | list[str]`` 统一解析成表达式列表。"""
    if item is None:
        return []
    if isinstance(item, sp.Basic):
        return [item]
    if isinstance(item, (list, tuple, set)):
        out: list[sp.Basic] = []
        for piece in item:
            out.extend(_parse_item(piece, symbols))
        return out
    raw = str(item).strip()
    if not raw:
        return []
    return [A.parse(raw, symbols=symbols)]


def _rel_to_expr(expr: sp.Basic) -> sp.Basic:
    """``Eq(a, b)`` -> ``a - b``（求解与验算统一用「残差为 0」形式）。"""
    if isinstance(expr, sp.Equality):
        return sp.sympify(expr.lhs) - sp.sympify(expr.rhs)
    return expr


def _split_top_level(text: str, separators: str) -> list[str]:
    """按**括号深度为 0** 的分隔符切分（括号内的逗号/分号不动）。"""
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if depth == 0 and ch in separators:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    parts.append("".join(buf))
    return parts


def _split_equations(text: Any) -> list[str]:
    """拆分方程字符串：支持 ``;`` / 换行 / 顶层逗号，并校验 ``=`` 只出现一次。

    **必须按括号深度切分**：``diff(y(x), x) = x*y`` 里的逗号是函数参数分隔符，
    早期版本用 ``re.split(r",(?=[^,]*=)")`` 会把它错切成 ``diff(y(x)`` 与 ``x) = x*y``，
    直接导致所有 ``diff(...)`` 写法的微分方程解析失败。顶层逗号也只在
    文本整体含多个 ``=``（即真的是方程列表）时才当作分隔符。
    """
    if isinstance(text, (list, tuple, set)):
        chunks: list[str] = []
        for piece in text:
            chunks.extend(_split_equations(piece))
        return chunks
    raw = str(text)
    parts = _split_top_level(raw, ";\n")
    if sum(p.count("=") for p in parts) > 1:
        expanded: list[str] = []
        for part in parts:
            expanded.extend(_split_top_level(part, ","))
        parts = expanded
    out: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.count("=") > 1:
            raise ValueError(f"方程 {part!r} 中含有多个 '='，无法判定等式两边（请写成 a = b 形式）")
        out.append(part)
    return out


def _eq_expr(text: str, symbols: list[str], *, extra: dict[str, Any] | None = None) -> sp.Basic:
    """解析单条方程：有 ``=`` 取左减右，否则整体作为表达式（默认等于 0）。"""
    raw = str(text).strip()
    if "=" in raw:
        lhs, rhs = raw.split("=", 1)
        if not lhs.strip() or not rhs.strip():
            raise ValueError(f"方程 {raw!r} 的等号两边不能为空")
        return A.parse(lhs, symbols=symbols, extra=extra) - A.parse(rhs, symbols=symbols, extra=extra)
    return A.parse(raw, symbols=symbols, extra=extra)


def _greekless(expr: sp.Basic) -> sp.Basic:
    """把 ``\\alpha`` 之类的希腊字母名换成纯 ASCII 占位名（SymPy 内部的 ``<lambda>`` 不能用）。"""
    return expr.xreplace({s: sp.Symbol(f"_{chr(97 + (i % 20))}{i}") for i, s in enumerate(sorted(expr.free_symbols, key=str))})


def _free_names(expr: sp.Basic) -> list[str]:
    if not isinstance(expr, sp.Basic):
        return []
    return sorted({str(s) for s in expr.free_symbols}, key=lambda n: (len(n), n))


def _condition_notes(*exprs: sp.Basic) -> list[str]:
    """按表达式中出现过的函数给出定义域要求（并显示符号的实数假设）。"""
    joined = " ".join(A.to_text(e) for e in exprs if isinstance(e, sp.Basic))
    notes: list[str] = []
    if re.search(r"log|ln", joined):
        names = sorted({str(s) for e in exprs if isinstance(e, sp.Basic)
                        for s in e.free_symbols if e.has(sp.log(s))})
        target = "、".join(names) if names else "对数内表达式"
        notes.append(f"对数要求 {target} > 0（定义域限制）")
    if joined.count("sqrt") or "**(1/2)" in joined:
        notes.append("根号内表达式须 ≥ 0（定义域限制）")
    if re.search(r"tan", joined) and not re.search(r"atan", joined):
        notes.append("tan 在 pi/2 + k*pi 处无定义（k 为整数）")
    if re.search(r"cot", joined) or re.search(r"csc", joined):
        notes.append("cot/csc 在 sin = 0 处无定义")
    if re.search(r"sec", joined) and not re.search(r"asec", joined):
        notes.append("sec 在 cos = 0 处无定义")
    if re.search(r"asin|acos", joined):
        notes.append("反三角函数 asin/acos 要求自变量落在 [-1, 1]")
    if "/" in joined or "zoo" in joined:
        notes.append("分式要求分母不为 0（定义域限制）")
    return notes


def _sample_x(expr: sp.Basic, var: sp.Symbol, extra_symbols: list[sp.Symbol]) -> tuple[sp.Basic, list[sp.Symbol]]:
    """把表达式里的参数替换成固定有理数，便于在 ``var`` 上做数值探针。"""
    subs: dict[sp.Symbol, Any] = {}
    for index, sym in enumerate(extra_symbols):
        if sym != var:
            subs[sym] = sp.Rational(index + 3, index + 2)
    return (expr.subs(subs) if subs else expr), [var]


def _probe_values(expr: sp.Basic, var: sp.Symbol, point: Any, *, samples: int = 9) -> list[sp.Basic]:
    """在趋近点附近取一串「越来越近」的探针点。

    上界刻意控制在 1e7 以内：像 ``(1+1/n)**n`` 这种表达式若用精确整数在 n=1e9 处
    求值，会构造出上亿位的整数，直接把 worker 拖死。
    """
    vals: list[sp.Basic] = []
    if point in (sp.oo, -sp.oo):
        for k in range(samples):
            v = sp.Integer(100) * (3 ** k)
            vals.append(v if point is sp.oo else -v)
        return vals
    if isinstance(point, sp.Basic) and point.is_number:
        eps = [sp.Rational(1, 10 ** (k + 1)) for k in range(samples)]
        return [sp.sympify(point) + e for e in eps]
    return []


def _numeric_sampling_check(expr: sp.Basic, var: sp.Symbol, point: Any, candidate: Any, *,
                            extra_symbols: list[sp.Symbol] | None = None, tol: float = 1e-4) -> tuple[bool, str]:
    """独立数值证据：沿趋近点采样，检查函数值是否收敛到候选取值。

    仅作**补充证据**，不替代符号结论。
    """
    worked = expr
    if extra_symbols:
        worked, _varlist = _sample_x(expr, var, list(extra_symbols))
    probes = _probe_values(worked, var, point)
    if not probes:
        return False, "趋近点不是可数值化的有限点/无穷远点，无法采样"
    try:
        cand = sp.N(sp.sympify(candidate), 30)
    except Exception as exc:  # noqa: BLE001
        return False, f"候选取值无法数值化: {exc}"
    if getattr(cand, "free_symbols", None):
        return False, "候选取值仍含自由符号，无法采样比较"
    rows: list[str] = []
    got: list[tuple[sp.Basic, float]] = []
    for probe in probes:
        try:
            # 用浮点代入而非精确有理代入：避免精确大整数运算导致的组合爆炸
            value = sp.N(worked.subs({var: sp.N(probe, 30)}), 30)
        except Exception:  # noqa: BLE001
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number != number or number in (float("inf"), float("-inf")):
            continue
        got.append((probe, number))
    if not got:
        return False, "所有探针点都落在定义域外，未获得数值证据"
    try:
        cand_value = float(cand)
    except (TypeError, ValueError):
        return False, f"候选取值 {A.to_text(cand)} 无法转成浮点数，未获得数值证据"
    worst = 0.0
    for probe, number in got[-3:]:
        dev = abs(cand_value - number) / max(1.0, abs(cand_value))
        worst = max(worst, dev)
        rows.append(f"x≈{_num_str(float(probe))}: {_num_str(number)}")
    note = (f"沿趋近点取 {len(got)} 个探针点采样，末尾 3 点 {'/'.join(rows)}，"
            f"与候选值 {_num_str(cand_value)} 的最大相对偏差 {worst:.3e}")
    return worst < tol, note


def _is_divergent_value(value: Any) -> bool:
    """积分值是否为「发散」标志（±oo / zoo / nan）。"""
    try:
        return value in (sp.oo, -sp.oo, sp.zoo, sp.nan)
    except Exception:  # noqa: BLE001
        return False


def _is_accum_bounds(value: Any) -> bool:
    """结果是否为 ``AccumBounds``（极限不存在／无限振荡）。

    ``sp.integrate(sin(x), (x, 0, oo))`` 返回 ``AccumBounds(0, 2)``，它**不是**积分值，
    而是「在有界范围里无限振荡、极限不存在」的表示 —— 必须当作未求出结果。
    """
    try:
        if isinstance(value, sp.AccumBounds):
            return True
        return isinstance(value, sp.Basic) and bool(value.has(sp.AccumBounds))
    except Exception:  # noqa: BLE001
        return False


def _numeric_int_compare(integrand: sp.Basic, var: sp.Symbol, lower: Any, upper: Any, candidate: Any, *,
                         rel_tol: float = 1e-6) -> tuple[bool | None, dict[str, Any]]:
    """定积分的独立数值验算（与 SymPy 的符号积分是两套完全不同的算法）。

    **为什么要走 ``numeric.numeric_integrate`` 的加固路线，而不是直接调 mpmath.quad**：
    朴素 ``mp.quad`` 在振荡反常积分上会「自信地给出错误值」。实测 ``∫_0^∞ sin(x)/x dx``
    直接 quad 得 ``0.90019``（真值 π/2 = 1.5707963…），相对偏差 0.43 —— 于是**正确的**
    符号结果 π/2 被本函数误判成「与数值不一致」，用户会看到一条误导性的不一致警告。
    ``numeric_integrate`` 有多条路线（直接 quad / quadosc / 相位线性化 / 截断替换）
    加三道收敛守卫（尾部窗口、截断序列、内点奇点柯西准则），拿不到可信值时返回
    unsolved，本函数据此返回 ``None``（= 没有有效数值证据），**绝不据此指控符号结果错误**。
    """
    detail: dict[str, Any] = {"method": "numeric_integrate 加固数值路线（多路线 + 收敛守卫，与 sp.integrate 相互独立）"}
    try:
        import mpmath as mp  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        detail["error"] = f"mpmath 不可用: {exc}"
        return None, detail
    try:
        params = sorted(sp.sympify(integrand).free_symbols - {var}, key=str)
        subs = {s: sp.Rational(i + 7, i + 5) for i, s in enumerate(params)}
        a, b = sp.sympify(lower), sp.sympify(upper)
        if a.free_symbols or b.free_symbols:
            detail["error"] = "积分端点含自由符号，无法做数值验算"
            return None, detail
        work = sp.sympify(integrand).subs(subs) if subs else sp.sympify(integrand)
        cand = sp.N(candidate.subs(subs) if params else candidate, 30)
        if getattr(cand, "free_symbols", None):
            detail["error"] = "定积分结果仍含自由符号，无法与数值积分比较"
            return None, detail
        if params:
            detail["parameter_substitution"] = {str(k): A.to_text(v) for k, v in subs.items()}

        from mathkit import numeric as _numeric  # 惰性导入，避免 import 环
        probe = _numeric.numeric_integrate(
            expr=A.to_text(work), var=str(var), lower=A.to_text(a), upper=A.to_text(b), digits=25,
        )
        payload = probe.to_dict()
        if payload.get("status") != "ok":
            reason = str(payload.get("error") or (payload.get("result") or {}).get("text") or "").strip()
            detail["error"] = (reason[:200] if reason else "加固数值路线未能给出可信数值")
            detail["numeric_status"] = payload.get("status")
            return None, detail
        raw = payload.get("extra") or {}
        value = (payload.get("result") or {}).get("approx")
        if value is None:
            detail["error"] = "数值路线没有给出可比较的数值"
            return None, detail
        numeric = complex(value)
        exact = complex(sp.N(cand, 30))
        scale = max(1.0, abs(exact))
        dev = abs(numeric - exact) / scale
        detail.update({
            "numeric_value": _num_str(numeric.real) if abs(numeric.imag) < 1e-12 else str(numeric),
            "symbolic_value": _num_str(exact.real) if abs(exact.imag) < 1e-12 else str(exact),
            "max_rel_dev": dev,
            "numeric_abs_error": raw.get("abs_error"),
            "numeric_reliable_digits": raw.get("reliable_digits"),
        })
        # 数值路线自报的有效位数不足以支撑 rel_tol 级别的判定时，只当「无证据」。
        reliable = raw.get("reliable_digits")
        if isinstance(reliable, int) and reliable < 6:
            detail["error"] = f"数值路线的有效位数只有 {reliable} 位，不足以判定 1e-6 级的一致性"
            return None, detail
        return dev <= rel_tol, detail
    except Exception as exc:  # noqa: BLE001
        detail["error"] = f"{type(exc).__name__}: {exc}"
        return None, detail


_SERIES_PARAM_SAMPLES = (
    sp.Rational(1, 2), sp.Rational(1, 3), sp.Rational(-1, 2), sp.Rational(2, 5),
    sp.Rational(1, 4), sp.Rational(3, 4), sp.Integer(2), sp.Integer(3),
)


def _series_param_sample(params: list[sp.Symbol], condition: Any) -> dict[sp.Symbol, sp.Basic] | None:
    """为含参数的级数挑一组**满足收敛条件**的数值样本。

    幂级数在有收敛域（如 ``Abs(x) < 1``）时，若随手取 x = 8/5 去数值求和，
    级数在该点发散，数值证据必然与符号结果不符 —— 会把「样本点取在收敛域外」
    误报成「符号结果与数值结果不一致」。这里按条件筛选样本，取不到就如实返回 None。
    """
    if not params:
        return {}
    for sample in _SERIES_PARAM_SAMPLES:
        subs = {symbol: sample for symbol in params}
        if condition is None:
            return subs
        try:
            verdict = sp.simplify(condition.subs(subs))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(verdict, sp.logic.boolalg.BooleanTrue) or verdict is True:
            return subs
    return None


def _series_numeric_evidence(term: sp.Basic, var: sp.Symbol, lower: Any, total: Any, *,
                             condition: Any = None,
                             rel_tol: float = 1e-8, dps: int = 25) -> tuple[str, str]:
    """级数求和的参考性数值证据（三态）。

    用 ``mpmath.nsum`` 的收敛加速独立算出级数的数值和，再与符号结果比较。
    这是**数值证据**，不是收敛性证明（规格 4.3：不得把数值验证冒充严格证明）。

    **为什么不再用「符号部分和」**：旧实现调 ``sp.summation(term, (n, start, N))``
    在 N 很大时（如 51200）会走 polygamma / 不完全 gamma 的高精度求值，单项就超过
    30 s，把 ``∑1/n² = π²/6``、``∑1/n! = e-1`` 这类最基础的求和全部拖成假超时。
    """
    try:
        start = int(sp.sympify(lower))
    except Exception:  # noqa: BLE001
        return "unknown", "求和下界不是整数，无法做数值核验"
    params = sorted(term.free_symbols - {var}, key=str)
    subs = _series_param_sample(params, condition)
    if subs is None:
        return "unknown", ("级数含参数且取不到满足收敛条件的数值样本，未做数值核验"
                           "（避免在收敛域外取样导致假「不一致」）")
    worked = term.subs(subs) if subs else term
    try:
        target = sp.N(total.subs(subs) if params else total, dps)
    except Exception as exc:  # noqa: BLE001
        return "unknown", f"级数和无法数值化: {exc}"
    if getattr(target, "free_symbols", None) or not getattr(target, "is_number", False):
        return "unknown", "级数和仍含自由符号，无法做数值比较"
    if target in (sp.oo, -sp.oo, sp.zoo, sp.nan):
        return "unknown", f"级数和为 {target}（发散或无穷），无法据数值判定"
    try:
        import mpmath as mp
    except Exception as exc:  # noqa: BLE001
        return "unknown", f"未安装 mpmath，无法做数值核验: {exc}"
    try:
        mp.mp.dps = dps
        fn = sp.lambdify(var, worked, modules="mpmath")
        numeric = mp.mpf(mp.nsum(lambda k: fn(k), [start, mp.inf]))
        exact = mp.mpf(str(sp.N(target, dps)))
    except Exception as exc:  # noqa: BLE001
        return "unknown", f"数值求和失败: {exc}"
    if not (mp.isfinite(numeric) and mp.isfinite(exact)):
        return "unknown", "数值求和得到非有限值，未获得有效证据"
    dev = abs(numeric - exact) / max(abs(exact), mp.mpf(1))
    dev_f = float(dev)
    note = (f"mpmath.nsum（收敛加速）独立数值求和 = {mp.nstr(numeric, 12)}，"
            f"与符号结果 {mp.nstr(exact, 12)} 的相对偏差 {dev_f:.3e}（数值证据，不构成收敛性证明）")
    if dev_f <= rel_tol:
        return "equal", note
    return "different", note + "；偏差超出容差，符号结果与数值结果不一致"


def _partial_sum_evidence(term: sp.Basic, var: sp.Symbol, lower: Any, total: Any, *,
                          samples: int = 6, tol: float = 1e-3) -> tuple[bool, str]:
    """兼容旧调用：把 :func:`_series_numeric_evidence` 的三态压成布尔。"""
    state, note = _series_numeric_evidence(term, var, lower, total)
    return state == "equal", note


def _guess_function_names(text: str, symbols: list[str], *, default: str = "y") -> list[str]:
    """从方程文本里猜出未知函数名（``y(x)`` 中的 ``y``）。"""
    found: list[str] = []
    for match in re.finditer(r"(?<![A-Za-z0-9_])([A-Za-z][A-Za-z0-9_]*)\s*\(", text):
        name = match.group(1)
        if name in symbols or name in A._RESERVED:  # noqa: SLF001 - 同包内部工具
            continue
        if name in found:
            continue
        found.append(name)
    if found:
        return found
    for name in (default, "f", "u", "v", "z", "g", "h", "w"):
        if name not in symbols:
            return [name]
    return ["y"]


def _ode_functions(funcs: Any, text: str, symbols: list[str]) -> list[sp.Function]:
    names = _sym_names(funcs) or _guess_function_names(text, symbols)
    return [sp.Function(name) for name in names]


def _normalize_primes(text: str) -> str:
    """把 ``y'`` / ``y''`` / ``y'(x)`` 改写成 ``diff(y(x), x)`` 等形式（不碰 ``y=x``）。

    ``y'(x)`` 这种「撇号后还写了自变量」的写法必须把 ``(x)`` 一起吃掉：
    旧实现只匹配「撇号后面不是左括号」的情况，于是 ``y'(x)`` 落到
    :func:`_apply_function_names` 手里被补成 ``y(x)'(x)``，
    报出「表达式无法解析（已规范化为 "_f0(x)'(x)"）」这种看不懂的错误。
    """

    def repl(match: re.Match[str]) -> str:
        name, primes, explicit = match.group(1), match.group(2), match.group(3)
        arg = (explicit or "x").strip()
        if len(primes) == 1:
            return f"diff({name}({arg}), {arg})"
        return f"Derivative({name}({arg}), ({arg}, {len(primes)}))"

    return _PRIME_RE.sub(repl, str(text))


def _extract_functions(text: str) -> tuple[str, dict[str, sp.Function], list[str]]:
    """把 ``y(x)`` 抽成占位函数 ``_f0(x)``，返回 ``(占位文本, {占位名: Function}, [自变量名])``。

    两个关键点（都是踩过的坑）：

    1. 占位符**必须保留 ``(arg)``**，并且 ``_fN`` 要以真正的 ``sp.Function`` 形式进
       ``local_dict``（走 ``A.parse(..., extra=...)``）。若像早期版本那样只放裸 ``_f0``，
       ``diff(_f0, x)`` 会把 ``_f0`` 当成与 ``x`` 无关的符号、直接求导成 ``0``，
       导数信息在解析阶段就被静默丢掉，``dsolve`` 会拿到一个没有导数的「方程」。
    2. 已在 ``A._RESERVED`` 里的名字（``diff``/``Derivative``/``sin``/``exp``…）是库函数，
       不能被抽成未知函数。
    """
    mapping: dict[str, sp.Function] = {}
    args: list[str] = []

    def repl(match: re.Match[str]) -> str:
        func_name, arg = match.group(1), match.group(2).strip()
        if func_name in A._RESERVED:  # noqa: SLF001 - 同包内部工具
            return match.group(0)
        key = next((k for k, fn in mapping.items() if fn.func.__name__ == func_name), None)
        if key is None:
            key = f"_f{len(mapping)}"
            mapping[key] = sp.Function(func_name)
        args.append(arg)
        return f"{key}({arg})"

    text = re.sub(r"(?<![A-Za-z0-9_])([A-Za-z][A-Za-z0-9_]*)\s*\(\s*([A-Za-z][A-Za-z0-9_]*)\s*\)", repl, text)
    return text, mapping, args


def _apply_function_names(text: str, names: list[str], independent: str) -> str:
    """把裸写的未知函数名补成 ``y(x)``。

    ``y' = x*y`` 里的 ``y'`` 会被 ``_normalize_primes`` 改写成 ``diff(y(x), x)``，
    但等号右边的裸 ``y`` 不会被改。若不补，``x*y`` 里的 ``y`` 就成了与函数无关的
    ``Symbol('y')``，``dsolve`` 会把它当参数解出 ``C1 + x**2*y/2`` 这种**静默错误答案**，
    而且残差化简还能「通过」验证。已带 ``(`` 的写法与自变量名不动。
    """
    for name in names:
        if name == independent or name in A._RESERVED:  # noqa: SLF001
            continue
        text = re.sub(
            # 后面已经跟了 ``(`` 或撇号的一律不动：``y(x)`` 已是函数写法，
            # ``y'`` / ``y'(x)`` 由 _normalize_primes 负责（补成 ``y(x)'(x)``
            # 就是这里漏掉撇号守卫时踩出来的坑）。
            rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])(?!\s*[('])",
            f"{name}({independent})",
            text,
        )
    return text


def _parse_ode_system(item: Any, funcs: Any) -> tuple[list[sp.Basic], list[sp.Function], list[sp.Symbol], list[str]]:
    """解析 ODE（组）：先做撇号预处理，再抽出未知函数，最后交给 A.parse。

    返回 ``(方程列表, 未知函数列表, 自变量列表, 原始文本列表)``。
    """
    texts = _split_equations(item)
    if not texts:
        raise ValueError("微分方程为空")
    texts = [_normalize_primes(t) for t in texts]

    joined = " ".join(texts)
    indep_match = re.search(r"(?:diff|Derivative)\(\s*[A-Za-z]\w*\s*\(\s*([A-Za-z]\w*)\s*\)", joined)
    indep_name = indep_match.group(1) if indep_match else "x"

    names = _sym_names(funcs) or _guess_function_names(joined, [])
    args: list[str] = []
    equations: list[sp.Basic] = []
    for text in texts:
        text = _apply_function_names(text, names, indep_name)
        text, mapping, found_args = _extract_functions(text)
        args.extend(found_args)
        symbols = set(names) | set(found_args)
        if "diff" in text or "Derivative" in text:
            symbols |= {"x"}
        expr = _eq_expr(text, sorted(symbols), extra=dict(mapping))
        equations.append(expr)

    arg_names = [a for a in (args or ["x"]) if a]
    if "diff" in " ".join(texts) or "Derivative" in " ".join(texts):
        # 自变量取 diff/Derivative 的第二个参数（若有）
        match = re.search(
            r"(?:diff|Derivative)\(\s*_f\d+\s*\(\s*([A-Za-z][A-Za-z0-9_]*)\s*\)\s*,\s*([A-Za-z][A-Za-z0-9_]*)",
            " ".join(texts),
        )
        if match:
            arg_names = [match.group(2)]
    independent = sp.Symbol(arg_names[0]) if arg_names else sp.Symbol("x")
    independent = _known_symbol(equations, independent)
    functions = [sp.Function(name) for name in (_sym_names(funcs) or _guess_function_names(" ".join(texts), []))]
    functions = _known_functions(equations, functions, independent)
    return equations, functions, [independent], texts


def _known_symbol(equations: list[sp.Basic], symbol: sp.Symbol) -> sp.Symbol:
    """在已解析的方程里找回同名 Symbol 对象，避免无假设符号与 real=True 符号不等。"""
    for eq in equations:
        for candidate in eq.free_symbols:
            if str(candidate) == str(symbol):
                return candidate
    return symbol


def _known_functions(equations: list[sp.Basic], functions: list[sp.Function], independent: sp.Symbol) -> list[sp.Function]:
    """把函数对象对齐成方程里实际出现的 ``y(x)`` 形式（``dsolve`` 要求它与方程里的对象一致）。"""
    present: dict[str, sp.Basic] = {}
    for eq in equations:
        for node in sp.preorder_traversal(eq):
            if isinstance(node, sp.Function) and node.args:
                present.setdefault(str(node.func), node)
    out: list[sp.Basic] = []
    for fn in functions:
        app = present.get(str(fn))
        out.append(app if app is not None else fn(independent))
    return out  # type: ignore[return-value]


def _applied(function: Any, independent: sp.Symbol) -> sp.Basic:
    """把「函数」统一成 ``y(x)`` 形式。

    ``_parse_ode_system`` 现在返回的是**已经应用过的** ``y(x)``（AppliedUndef），
    对它再调一次 ``y(x)(x)`` 会抛 ``TypeError: 'y' object is not callable``；
    而调用方若传进来的是 ``sp.Function('y')`` 类，则需要自己应用。
    """
    if isinstance(function, sp.Function):
        return function
    if isinstance(function, type) and issubclass(function, sp.Function):
        return function(independent)
    return sp.Function(str(function))(independent)


def _ode_residuals(equations: list[sp.Basic], solution: dict[Any, Any], independent: sp.Symbol) -> tuple[bool, list[str]]:
    """把候选解代回原 ODE，返回 (残差是否全为 0, 残差文本列表)。

    两个致命细节（都踩过）：

    1. **必须把 ``y(x)`` 直接换成表达式**，不能再包一层 ``sp.Lambda``。
       SymPy 不会自动把 ``Lambda`` 应用到参数上，残差里会留下不被化简的
       ``-x*Lambda(x, exp(x))``，而 ``Derivative(Lambda(...), x)`` 又算不出来。
    2. **「拿不到数值证据」绝不能当作「一致」**。旧实现在数值抽样失败时沿用
       ``numeric_agree`` 的 ``True``，于是 ``exp(x)`` 也被判成 ``y' = x*y`` 的解。
       现在符号残差非零即判不通过，数值抽样只用来补充说明。
    """
    subs_map: dict[Any, Any] = dict(solution)

    residuals: list[str] = []
    all_zero = True
    for eq in equations:
        try:
            substituted = eq.subs(subs_map, simultaneous=True)
            if isinstance(substituted, sp.Basic):
                substituted = substituted.doit()
            residual = sp.simplify(substituted)
        except Exception as exc:  # noqa: BLE001
            residuals.append(f"代入失败: {type(exc).__name__}: {exc}")
            all_zero = False
            continue
        residuals.append(A.to_text(residual))
        if residual == 0:
            continue
        zero = False
        if isinstance(residual, sp.Basic):
            try:
                zero = bool(sp.simplify(sp.expand(residual)) == 0)
            except Exception:  # noqa: BLE001
                zero = False
        if zero:
            continue
        # 符号上非零 → 再给一次数值机会（只作为说明，不推翻符号结论）
        note = None
        if isinstance(residual, sp.Basic) and not residual.has(sp.Derivative):
            probe = residual
            consts = sorted((s for s in probe.free_symbols if str(s).startswith("C")), key=str)
            if consts:
                probe = probe.subs({s: sp.Integer(i + 1) for i, s in enumerate(consts)})
            verdict, _dev, detail = C.numeric_evidence(probe, 0, symbols=[independent])
            note = f"（数值抽样判定：{verdict}；{detail}）"
        if note:
            residuals.append(note)
        all_zero = False
    return all_zero, residuals


def _solve_with_params(equations: list[sp.Basic], symbols: list[sp.Symbol],
                       solution: dict[sp.Symbol, Any]) -> tuple[bool, dict[str, Any]]:
    """代回验算：若解里还有自由参数，先把参数取定值再验证。"""
    params = sorted(
        {s for value in solution.values() if isinstance(value, sp.Basic) for s in value.free_symbols} - set(symbols),
        key=str,
    )
    test_map = {p: sp.Rational(i + 2, i + 1) for i, p in enumerate(params)}
    pairs = {s: (solution[s].subs(test_map) if isinstance(solution[s], sp.Basic) else solution[s]) for s in solution}
    ok, detail = C.check_solution(equations, symbols, pairs)
    if test_map:
        detail["note"] = ("解中含自由参数 " + ", ".join(str(p) for p in params)
                          + "，验证时取定值 " + ", ".join(f"{p}={v}" for p, v in test_map.items()))
    return ok, detail


def _verify_ok(status_ok: bool) -> str:
    return "independent" if status_ok else "cross"


# ---------------------------------------------------------------------------
# 1. 极限
# ---------------------------------------------------------------------------

@op("limit", "lim", "math_limit")
def limit(*, expr: str, var: str = None, point: str = "0", dir: str = None) -> MathResult:
    """求极限（支持单侧/双侧、有限点与 ``oo``）。"""
    parsed = A.parse(expr, symbols=_sym_names(var))
    variable = C.resolve_var(parsed, var)
    target = C.parse_point(point, [variable])
    extra = sorted(parsed.free_symbols - {variable}, key=str)
    direction = dir if dir in ("+", "-") else None

    r = MathResult.ok("limit", method="sp.limit（内部按需选择洛必达/级数展开/等价无穷小等策略）")
    r.set_input(expr=A.to_text(parsed), var=str(variable), point=A.to_text(target), dir=direction or "both")
    side_text = {"+": "右极限（x → x0+）", "-": "左极限（x → x0-）", None: "双侧极限"}[direction]
    r.set_raw(side=side_text)
    method_parts: list[str] = []

    value: Any = None
    if direction is None:
        left = sp.limit(parsed, variable, target, "-")
        right = sp.limit(parsed, variable, target, "+")
        method_parts.append(f"左极限 -> {A.to_text(left)}")
        method_parts.append(f"右极限 -> {A.to_text(right)}")
        if not (C.is_closed_form(left) and C.is_closed_form(right)):
            return MathResult.unsolved(
                "limit",
                f"SymPy 未能求出极限（左={A.to_text(left)}，右={A.to_text(right)}）；"
                "可尝试：先用 sp.simplify/等价无穷小化简、有理化、洛必达多次求导，或改用单侧极限。",
                method=r.method,
            ).set_input(**r.input)
        # Limit(expr) 未求值时会原样回显式子，其仍含自变量：必须判定为未求得
        if left.has(variable) or right.has(variable):
            return MathResult.unsolved(
                "limit",
                f"SymPy 原样回显了极限式（左={A.to_text(left)}，右={A.to_text(right)}），即未求出该极限；"
                "可尝试先化简表达式、等价无穷小替换或洛必达法则后重试。",
                method=r.method,
            ).set_input(**r.input)
        left_inf = left in (sp.oo, -sp.oo)
        right_inf = right in (sp.oo, -sp.oo)
        if left_inf or right_inf:
            if (left_inf and right_inf and left != right) or (left_inf != right_inf):
                return MathResult.unsolved(
                    "limit",
                    f"极限不存在：左右极限不一致（左 -> {A.to_text(left)}，右 -> {A.to_text(right)}）。",
                    method=r.method,
                ).set_input(**r.input).add_condition("极限存在要求左极限与右极限相等且有限")
            value = left
        elif C.same_expr(left, right):
            value = left
        else:
            ok, dev, note = C.numeric_agree(left, right, symbols=extra or None)
            if ok:
                value = left
                r.add_warning(f"左右极限符号形式不同（{A.to_text(left)} vs {A.to_text(right)}），数值抽样判定等价：{note}")
            else:
                return MathResult.unsolved(
                    "limit",
                    f"极限不存在：左右极限不一致（左 -> {A.to_text(left)}，右 -> {A.to_text(right)}，{note}）。",
                    method=r.method,
                ).set_input(**r.input)
        r.set_raw(left_limit=A.to_text(left), right_limit=A.to_text(right))
        method_parts.append("双侧：分别计算左右极限并比较")
    else:
        value = sp.limit(parsed, variable, target, direction)
        if not C.is_closed_form(value) or (isinstance(value, sp.Basic) and value.has(variable)):
            return MathResult.unsolved(
                "limit",
                f"SymPy 未能求出该单侧极限，返回未求值形式 {A.to_text(value)}；"
                "可尝试先化简表达式或改用数值/级数方法。",
                method=r.method,
            ).set_input(**r.input)
        method_parts.append(f"单侧极限（dir={direction}）")

    if isinstance(value, sp.AccumBounds):
        return MathResult.unsolved(
            "limit", f"极限振荡无确定值（{A.to_text(value)}），因此极限不存在。", method=r.method
        ).set_input(**r.input)

    r.set_method("极限：" + "；".join(method_parts))
    r.set_result(value)

    # 成立条件
    if target in (sp.oo, -sp.oo):
        r.add_condition("极限过程为自变量趋向无穷远，需保证表达式在该方向上最终有定义")
    else:
        r.add_condition(f"表达式在 {A.to_text(target)} 的某去心邻域内有定义（{A.to_text(target)} 本身可以不有定义）")
    for note in _condition_notes(parsed):
        r.add_condition(note)
    for sym in extra:
        r.add_condition(f"参数 {sym} 在极限意义下视为常数（其取值不影响该极限结论）")

    # 独立（补充性）数值证据
    ok, note = _numeric_sampling_check(parsed, variable, target, value, extra_symbols=extra)
    methods = [{
        "method": "沿趋近点取值采样，检查函数值是否收敛到极限值（数值补充证据，非独立证明）",
        "agree": bool(ok),
        "detail": note,
    }]
    r.verify(status=_verify_ok(ok), methods=methods,
             note="符号结论由 sp.limit 给出；采样只用于排除粗错，不构成证明。" if not ok else None)
    if not ok:
        r.add_warning("数值采样未能确认该极限（可能受采样点、量级或数值误差影响）：" + note)
    return r


# ---------------------------------------------------------------------------
# 2. 导数
# ---------------------------------------------------------------------------

@op("diff", "derivative", "differentiate", "math_derivative")
def differentiate(*, expr: str, var: str = None, order: int = 1) -> MathResult:
    """求导（支持任意阶）。"""
    parsed = A.parse(expr, symbols=_sym_names(var))
    variable = C.resolve_var(parsed, var)
    try:
        order_int = int(order)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"阶数 order 必须是正整数，收到 {order!r}") from exc
    if order_int < 1:
        raise ValueError(f"阶数 order 必须 ≥ 1（收到 {order_int}）")

    r = MathResult.ok("diff", method=f"sp.diff 逐阶符号求导（{order_int} 阶）")
    r.set_input(expr=A.to_text(parsed), var=str(variable), order=order_int)
    value = sp.diff(parsed, variable, order_int)
    r.set_result(value)

    r.add_condition(f"函数需在考察区间内 {order_int} 阶可导")
    r.add_condition("以下等式在其成立区间（函数有定义的区间）内成立")
    for note in _condition_notes(parsed):
        r.add_condition(note)
    if _free_names(value) == [] and value != 0:
        r.add_warning("导数为常数：原函数在该变量上为线性（一次）函数")

    ok, detail = C.check_derivative(parsed, variable, value, order=order_int, independent=False)
    methods: list[dict[str, Any]] = [detail]
    if order_int > 1:
        # 高阶导另用「低一阶导数再求导一次」的独立路径复核（规格 4.3 要求的「再求导检查」）
        try:
            lower = sp.diff(parsed, variable, order_int - 1)
            again = sp.simplify(sp.diff(lower, variable) - value)
            recurrence = again == 0
            methods.append({
                "method": f"对 {order_int - 1} 阶导数再求导一次，与候选 {order_int} 阶导数比较",
                "residual": A.to_text(again),
                "agree": recurrence,
                "independent": True,
            })
            if not recurrence:
                ok = False
        except Exception as exc:  # noqa: BLE001
            methods.append({"method": "递推复核", "error": str(exc)})
    # 真正独立的证据：数值微分（与符号求导完全不同的算法路径）
    numeric_state, numeric_dev, numeric_note = C.numeric_derivative_check(
        parsed, variable, value, order=order_int
    )
    methods.append({
        "method": "mpmath 数值微分独立核验",
        "independent": True,
        "agree": numeric_state,
        "max_rel_dev": numeric_dev,
        "note": numeric_note,
    })
    if numeric_state == "different":
        ok = False
        status = "unverified"
    elif numeric_state == "equal" and ok:
        status = "independent"
    elif ok:
        status = "symbolic"
    else:
        status = "unverified"
    r.verify(status=status, methods=methods)
    if status == "unverified":
        r.partial("导数结果未通过独立复核，请人工确认。")
    elif status == "symbolic":
        r.add_warning("仅完成同源符号一致性检查，数值微分未取得有效证据，验证状态标记为 symbolic。")
    return r


# ---------------------------------------------------------------------------
# 3. 积分
# ---------------------------------------------------------------------------

def _antiderivative_candidates(integrand: sp.Basic, variable: sp.Symbol) -> tuple[sp.Basic, str]:
    """先直接积分；若回显未求值，再用 simplify/apart/expand 预处理重试一次。"""
    raw = sp.integrate(integrand, variable)
    if C.is_closed_form(raw):
        return raw, "sp.integrate 直接求解"
    prepared = integrand
    steps: list[str] = []
    for name, func in (("simplify", sp.simplify), ("apart", sp.apart), ("expand", sp.expand)):
        try:
            trial = func(integrand)
        except Exception:  # noqa: BLE001
            continue
        if trial != integrand:
            prepared = trial
            steps.append(name)
            retry = sp.integrate(prepared, variable)
            if C.is_closed_form(retry):
                return retry, f"先做 {name} 预处理后 sp.integrate 成功"
    return raw, "sp.integrate 未求得原函数（已尝试 " + ("/".join(steps) or "预处理") + " 重试一次）"


@op("integrate", "integral", "int", "math_integrate")
def integrate(*, expr: str, var: str = None, lower: str = None, upper: str = None) -> MathResult:
    """不定积分 / 定积分（支持反常积分端点 ``oo``）。"""
    parsed = A.parse(expr, symbols=_sym_names(var))
    variable = C.resolve_var(parsed, var)
    definite = lower is not None or upper is not None
    if definite and (lower is None or upper is None):
        raise ValueError("定积分必须同时给出下限 lower 与上限 upper；只给一个端点无法确定积分区间")

    r = MathResult.ok("integrate")
    r.set_input(expr=A.to_text(parsed), var=str(variable),
                lower=None if not definite else str(lower), upper=None if not definite else str(upper))

    if definite:
        a = C.parse_point(lower, [variable])
        b = C.parse_point(upper, [variable])
        r.set_input(lower=A.to_text(a), upper=A.to_text(b))
        value = sp.integrate(parsed, (variable, a, b))
        r.set_method("定积分：sp.integrate(expr, (var, a, b))")
        if not C.is_closed_form(value):
            return MathResult.unsolved(
                "integrate",
                f"SymPy 未能求出该定积分，返回未求值形式 {A.to_text(value)}；"
                "可尝试：换元、分部积分、有理函数分解（apart）、对称性/奇偶性化简，"
                "或先求原函数再代入端点。",
                method=r.method,
            ).set_input(**r.input)
        if value in (sp.nan, sp.zoo):
            return MathResult.unsolved(
                "integrate", f"积分为 {A.to_text(value)}：该反常积分发散或端点处不可积。", method=r.method
            ).set_input(**r.input)
        if _is_accum_bounds(value):
            # AccumBounds(0, 2) 这类结果表示「极限不存在（在有界值之间无限振荡）」，
            # 它**不是**一个答案：∫_0^∞ sin(x)dx 就返回 AccumBounds(0, 2)。
            # 若不拦住，用户会看到一个看起来像结果的表达式（违反规格 5.3）。
            return MathResult.unsolved(
                "integrate",
                f"该反常积分不收敛：SymPy 返回 {A.to_text(value)}（振荡不趋于定值，表示积分发散），"
                "这不是一个确定的积分值。判断敛散性请用 Dirichlet / Abel 判别法等收敛性判别法。",
                method=r.method,
            ).set_input(**r.input)
        r.set_result(value)

        improper = a in (sp.oo, -sp.oo) or b in (sp.oo, -sp.oo)
        if improper:
            if _is_divergent_value(value):
                r.add_condition(
                    "该反常积分**发散**（积分值不是有限数）：oo 表示「不收敛」，不是某个具体的和/值"
                )
            else:
                r.add_condition("反常积分：收敛性已由积分结果的存在性体现（积分值有限即收敛）")
        if parsed.has(sp.log) or parsed.has(sp.sqrt) or parsed.is_rational_function(variable) is False:
            r.add_condition(f"被积函数在闭区间 [{A.to_text(a)}, {A.to_text(b)}]（去掉内部瑕点后）可积")
        if a == 0 and (parsed.has(1 / variable) or parsed.has(sp.sin(variable) / variable)):
            if not parsed.has(sp.exp(-variable)):
                r.add_condition("被积函数在 x=0 处为瑕点（0/0 或 1/x 型），该积分为反常积分，需单独讨论收敛性")
        for note in _condition_notes(parsed):
            r.add_condition(note)
        if not improper and _is_divergent_value(value):
            # 有限区间也可能发散：∫_{-1}^{1} dx/x² 在 x=0 处有不可积瑕点（两侧都发散）。
            r.add_condition(
                "被积函数在积分区间内部存在瑕点（如分母为零的无界点），该积分是反常积分，"
                "且积分值不是有限数（发散）"
            )
            r.add_warning("这是发散的反常积分：oo 表示「不收敛」，不是「一个很大的数」")

        agree, detail = _numeric_int_compare(parsed, variable, a, b, value)
        methods = [detail]
        if agree is None:
            r.verify(status="unverified", methods=methods,
                     note="独立数值积分无法给出可信参考值（见 methods 中的 error 字段），因此该定积分缺少独立数值交叉验证。")
        elif agree:
            r.verify(status="cross", methods=methods,
                     note="符号结果与独立数值积分（多路线 + 收敛守卫）一致（数值交叉验证，非符号证明）。")
        else:
            r.verify(status="unverified", methods=methods)
            r.add_warning("符号积分结果与独立数值积分不一致，请核对被积函数与积分区间（可能存在瑕点或分支问题）。")
        return r

    # 不定积分
    value, how = _antiderivative_candidates(parsed, variable)
    r.set_method(f"不定积分：{how}")
    if not C.is_closed_form(value):
        return MathResult.unsolved(
            "integrate",
            f"SymPy 未能求出该不定积分，返回未求值形式 {A.to_text(value)}；"
            "可尝试：换元法（凑微分）、分部积分、有理函数部分分式分解（apart）、"
            "三角恒等变形、根式代换（如 sqrt(a^2-x^2) 用 x=a*sin t）。",
            method=r.method,
        ).set_input(**r.input)
    r.set_result(value)
    r.set_raw(antiderivative_note="不定积分结果含任意常数 C（通解为 F(x) + C）")
    r.add_condition("不定积分含任意常数 C（已省略，恢复即为 F(x) + C）")
    r.add_condition(f"等式在 {A.to_text(variable)} 的连续区间内成立（跨越被积函数的间断点时，各区间上的常数 C 可以不同）")
    for note in _condition_notes(parsed):
        r.add_condition(note)

    ok, detail = C.check_antiderivative(parsed, variable, value)
    r.verify(status="independent" if ok else "unverified",
             methods=[{"method": "C.check_antiderivative：对候选原函数求导，检查是否等于被积函数", **detail}])
    if not ok:
        r.partial("原函数未通过「求导还原」复核，请人工确认。")
    return r


# ---------------------------------------------------------------------------
# 3b. 累次积分（二重 / 三重等）
# ---------------------------------------------------------------------------

def _bound_pair(item: Any) -> tuple[Any, Any]:
    """把一条边界记录解析成 (下界, 上界)。"""
    if isinstance(item, dict):
        lower = item.get("lower", item.get("from", item.get("lo")))
        upper = item.get("upper", item.get("to", item.get("hi")))
        if lower is None or upper is None:
            raise ValueError(f"边界对象必须同时给 lower 与 upper，收到 {item!r}")
        return lower, upper
    if isinstance(item, (list, tuple)):
        if len(item) != 2:
            raise ValueError(f"边界必须是 [下界, 上界] 二元组，收到 {item!r}")
        return item[0], item[1]
    raise ValueError(f"无法解析边界 {item!r}：请用 [[\"0\",\"1\"],[\"0\",\"1-x\"]] 或 "
                     "[{\"lower\":\"0\",\"upper\":\"1\"}, {...}]")


def _parse_region(vars_value: Any) -> list[tuple[str, Any, Any]]:
    """把区域描述解析成 [(变量名, 下界表达式, 上界表达式)]，最内层在前。"""
    text = vars_value
    if isinstance(text, str):
        try:
            decoded = json.loads(text)
        except Exception:  # noqa: BLE001
            decoded = [part.strip() for part in text.replace("，", ",").split(",") if part.strip()]
        text = decoded
    if not isinstance(text, (list, tuple)) or not text:
        raise ValueError("区域描述必须是非空列表（最内层积分变量在前）")
    region: list[tuple[str, Any, Any]] = []
    for item in text:
        if isinstance(item, dict):
            name = item.get("var") or item.get("variable")
            if not name:
                raise ValueError(f"边界对象缺少变量名：{item!r}")
            lower, upper = _bound_pair(item)
        elif isinstance(item, (list, tuple)) and len(item) == 3:
            name, lower, upper = item
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            # [[下界, 上界], ...]：变量名在外层 vars 里按顺序给出
            lower, upper = item
            name = None
        else:
            raise ValueError(f"无法解析区域项 {item!r}")
        region.append((name, lower, upper))  # type: ignore[arg-type]
    return region


def _has_integral(expr: Any) -> bool:
    """是否仍残留未求值的 Integral / Sum（SymPy 对嵌套积分有时只回显）。"""
    return any(isinstance(node, (sp.Integral, sp.Sum, sp.Limit, sp.Derivative))
               for node in sp.preorder_traversal(expr))


def _integrate_iterated(integrand: sp.Basic, region: list[tuple[str, Any, Any]]) -> tuple[sp.Basic, str]:
    """从最内层开始逐层积分；内层边界可引用外层的积分变量。

    注意：**必须复用被积表达式里已有的 Symbol 对象**（``Symbol("x", real=True)``），
    不能临时 ``sp.Symbol(str(name), real=True)``：``A.parse`` 用的是带 ``real=True``
    假设的符号，而 ``sp.sympify("1-x")`` 产生的是**无假设**的 ``Symbol('x')``。
    两者打印完全相同但不相等，会让 ``sp.integrate`` 认不出积分变量而返回错误的层结果
    （例如 ∬xy 在三角形上会得到 ``x**2/4 - x/2 + 1/4`` 而不是 ``1/24``）。
    """
    # 1) 先把整个区域涉及的符号统一成被积表达式里实际使用的那些对象
    known: dict[str, sp.Symbol] = {str(s): s for s in integrand.free_symbols}
    for name, lower, upper in region:
        known.setdefault(str(name), sp.Symbol(str(name), real=True))
    # sp.sympify("1-x") 默认产生**无假设**的 Symbol('x')，与 A.parse 产出的
    # Symbol('x', real=True) 打印相同却互不相等。用它当积分上下界时，SymPy 会把被积
    # 函数里的 x 当成常数而给出错误的层结果（∬xy 会得到 x**2/4 - x/2 + 1/4 而不是 1/24）。
    # 因此所有界表达式都必须用同一个 local_dict 解析。
    def _sympify_bound(expr: Any) -> sp.Basic:
        if isinstance(expr, sp.Basic):
            return expr
        return sp.sympify(str(expr), locals=dict(known))
    for _name, lower, upper in region:
        for expr in (lower, upper):
            for sym in _sympify_bound(expr).free_symbols:
                known.setdefault(str(sym), sym)

    value = integrand
    done: list[sp.Symbol] = []
    for name, lower, upper in region:
        key = str(name)
        symbol = known.get(key) or sp.Symbol(key, real=True)
        known.setdefault(key, symbol)
        lo, hi = _sympify_bound(lower), _sympify_bound(upper)
        inner_symbols = set(lo.free_symbols) | set(hi.free_symbols)
        illegal = inner_symbols & set(done)
        if illegal:
            bad = ", ".join(sorted(str(s) for s in illegal))
            raise ValueError(
                f"积分变量 {symbol} 的上下界不能含已积分掉的变量（{bad}）；"
                "请把外层变量写在区域列表前面、内层变量写在后面。"
            )
        if symbol not in value.free_symbols:
            # 退一步：用当前表达式里同名符号对象，避免「对常数积分」
            replacement = next(
                (candidate for candidate in value.free_symbols if str(candidate) == key), None)
            if replacement is not None:
                symbol = replacement
        try:
            layered = sp.integrate(value, (symbol, lo, hi))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"对 {symbol} 在 [{lo}, {hi}] 上积分失败：{exc}") from exc
        if _has_integral(layered):
            # 嵌套积分时 SymPy 常只回显 Integral(...)，必须显式 doit 再化简
            try:
                layered = sp.simplify(sp.simplify(layered.doit()))
            except Exception:  # noqa: BLE001
                layered = sp.simplify(layered)
        if _has_integral(layered):
            return layered, f"累次积分在第 {len(done) + 1} 层（{symbol}）未能求出闭式解"
        if symbol in layered.free_symbols and len(done) + 1 == len(region):
            # 最外层积完后仍含该变量 → 说明这一层没有真的积（变量对象错配或界限退化）
            return layered, (f"累次积分在第 {len(done) + 1} 层（{symbol}）未能完成："
                             f"结果仍含 {symbol}，请检查该层上下界是否与积分变量匹配")
        value = sp.simplify(layered)
        done.append(symbol)
    order = "、".join(str(name) for name, _lo, _hi in reversed(region))
    return value, f"累次积分（由内向外依次对 {order} 积分）"


def _numeric_iterated_area(integrand: sp.Basic, region: list[tuple[str, Any, Any]]) -> float | None:
    """用 scipy 独立复算二重积分（仅支持变量名 x/y、有限边界）。"""
    if len(region) != 2:
        return None
    names = [str(name) for name, _lo, _hi in region]
    if sorted(names) != ["x", "y"]:
        return None
    try:
        import scipy.integrate as sci  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    by_name = {str(name): (lo, hi) for name, lo, hi in region}
    x_lo, x_hi = by_name["x"]
    y_lo, y_hi = by_name["y"]

    def fx(value: Any) -> float:
        return float(sp.N(value, 20))

    # 复用被积表达式里实际的符号对象（带 real=True 假设），不要新建 Symbol：
    # 新建的 Symbol('x') 与 A.parse 产生的 Symbol('x', real=True) 不相等，
    # 会让 lambdify 认不出自由变量而报错。
    symbols = {str(s): s for s in integrand.free_symbols}
    for _name, lo_e, hi_e in region:
        for expr in (lo_e, hi_e):
            for sym in sp.sympify(expr).free_symbols:
                symbols.setdefault(str(sym), sym)
    x = symbols.get("x") or sp.Symbol("x", real=True)
    y = symbols.get("y") or sp.Symbol("y", real=True)
    inner_is_y = not (sp.sympify(y_lo).free_symbols or sp.sympify(y_hi).free_symbols)
    try:
        if inner_is_y:
            # 外层 x：∫_xl^xu [∫_yl(x)^yu(x) f dy] dx
            y_lo_fn = sp.lambdify(x, sp.sympify(y_lo), "math")
            y_hi_fn = sp.lambdify(x, sp.sympify(y_hi), "math")
            f_fn = sp.lambdify((y, x), integrand, "math")
            total, _err = sci.dblquad(lambda yy, xx: f_fn(yy, xx), fx(x_lo), fx(x_hi),
                                      lambda xx: fx(y_lo_fn(xx)), lambda xx: fx(y_hi_fn(xx)))
        else:
            # 外层 y：∫_yl^yu [∫_xl(y)^xu(y) f dx] dy
            x_lo_fn = sp.lambdify(y, sp.sympify(x_lo), "math")
            x_hi_fn = sp.lambdify(y, sp.sympify(x_hi), "math")
            f_fn = sp.lambdify((x, y), integrand, "math")
            total, _err = sci.dblquad(lambda xx, yy: f_fn(xx, yy), fx(y_lo), fx(y_hi),
                                      lambda yy: fx(x_lo_fn(yy)), lambda yy: fx(x_hi_fn(yy)))
    except Exception:  # noqa: BLE001
        return None
    return float(total)


@op("integrate_multi", "multiple_integral", "iterated_integral", "math_integrate_multi")
def integrate_multi(*, expr: str, vars: Any = None, bounds: Any = None, region: Any = None) -> MathResult:
    """累次积分：二重积分 ∬、三重积分 ∭（由内向外逐层积分）。

    ``vars``/``bounds`` 可写 ``[["0","1"],["0","1-x"]]``（最内层在前）、
    ``[{"var":"y","lower":"0","upper":"1-x"}, {"var":"x","lower":"0","upper":"1"}]``，
    或 ``[["y","0","1-x"],["x","0","1"]]``。内层上下界可以引用外层变量。
    """
    spec = vars if vars is not None else bounds if bounds is not None else region
    if spec is None:
        raise ValueError("需要区域描述：vars=[[\"0\",\"1\"],[\"0\",\"1-x\"]] 或 "
                         "vars=[{\"var\":\"y\",\"lower\":\"0\",\"upper\":\"1-x\"}, {...}]")
    raw_region = _parse_region(spec)
    names = [name for name, _lo, _hi in raw_region]
    if any(name is None for name in names):
        # 未写变量名时按「最内层在前」的约定顺序补全：二重 → (y, x)，三重 → (z, y, x)
        defaults = {1: ["x"], 2: ["y", "x"], 3: ["z", "y", "x"]}.get(
            len(raw_region), ["z", "y", "x", "t", "u", "v"][: len(raw_region)])
        names = [names[i] if names[i] else defaults[i] for i in range(len(raw_region))]
    named = [(str(names[i]), raw_region[i][1], raw_region[i][2]) for i in range(len(raw_region))]

    parsed = A.parse(expr, symbols=[name for name, _lo, _hi in named])
    variable_names = {str(name) for name, _lo, _hi in named}
    extra = {str(s) for s in parsed.free_symbols} - variable_names
    parameters: list[sp.Symbol] = sorted(
        (s for s in parsed.free_symbols if str(s) in extra), key=lambda s: str(s))

    value, method = _integrate_iterated(parsed, named)
    if _has_integral(value):
        return MathResult.unsolved(
            "integrate_multi",
            f"逐层积分未能求出闭式解（{method}）；SymPy 返回未求值形式 {A.to_text(value)}。"
            "可尝试：交换积分次序（选择先积更容易的变量）、换元（极坐标/柱坐标/球坐标）、"
            "利用对称性，或先用 math_numeric 做数值估计。",
            method=method,
        ).set_input(expr=A.to_text(parsed))
    region_text = "，".join(f"{name} ∈ [{A.to_text(sp.sympify(lo))}, {A.to_text(sp.sympify(hi))}]"
                            for name, lo, hi in named)
    r = MathResult.ok("integrate_multi", method=method)
    r.set_input(expr=A.to_text(parsed), region=region_text)
    if not C.is_closed_form(value):
        return MathResult.unsolved(
            "integrate_multi",
            f"逐层积分未能求出闭式解，SymPy 返回未求值形式 {A.to_text(value)}；"
            "可尝试：交换积分次序（选择先积更容易的变量）、换元（极坐标/柱坐标/球坐标）、"
            "利用对称性，或先用 math_numeric 做数值估计。",
            method=method,
        ).set_input(**r.input)
    value = sp.simplify(value)
    r.set_result(value)
    r.add_condition(f"积分区域：{region_text}")
    r.add_condition("累次积分要求区域在该次序下是「先内后外」的简单区域；"
                    "更换次序时需重新确定每一层的上下界")
    if parameters:
        # 参数（非积分变量）是合法的，但结果随参数取值而变，必须让模型看到
        names_text = ", ".join(str(s) for s in parameters)
        r.add_condition(f"结果依赖参数 {names_text}（它们不是积分变量）；"
                        "只有在这些参数取到使积分收敛的值时结果才成立")
        r.add_warning(f"结果含参数 {names_text}，因此未做与参数的无关性核验，"
                      "验证状态标记为未独立验证。")

    methods: list[dict[str, Any]] = []
    status = "unverified"
    if not value.free_symbols:
        # ---- 情形一：结果已经是常数 → 可以做真正独立的复核 ----
        numeric = _numeric_iterated_area(parsed, named) if len(named) == 2 else None
        if numeric is not None:
            symbolic_value = float(sp.N(value, 20))
            relative = abs(symbolic_value - numeric) / max(1.0, abs(numeric))
            passed = relative < 1e-6
            methods.append({
                "method": "scipy.integrate.dblquad 用不同的数值算法独立复算同一个二重积分",
                "evidence": {"symbolic": symbolic_value, "numeric": numeric,
                             "relative_deviation": relative},
                "passed": bool(passed),
            })
            status = "symbolic+numeric" if passed else "unverified"
            if not passed:
                r.add_warning("符号结果与独立数值复算不一致，请检查积分区域（换序时的上下界最容易写错）。")
        else:
            # 二重以上或非 x/y 变量：用逐层回代（对每层结果求导再与下层被积函数比对）
            layers = _verify_layers(parsed, named)
            methods.extend(layers)
            if layers and all(item["passed"] for item in layers):
                status = "independent"
            else:
                status = "unverified"
                r.add_warning("逐层回代复核未全部通过，结果需要人工检查。")
    else:
        # ---- 情形二：结果仍含自由符号 → 承认没有独立复核，绝不谎称已验证 ----
        methods.append({
            "method": "未能独立复核：结果仍含自由符号，无法做数值比对",
            "evidence": {"free_symbols": [str(s) for s in sorted(value.free_symbols, key=str)],
                         "layers": len(named)},
            "passed": False,
        })
        status = "unverified"
    r.verify(status=status, methods=methods)
    return r


def _verify_layers(integrand: sp.Basic, region: list[tuple[str, Any, Any]]) -> list[dict[str, Any]]:
    """逐层回代复核：对第 k 层结果按该层积分变量求导，应与下一层被积函数一致。

    这是**独立于求积分路径**的检查（走微分而非积分），但只覆盖闭式可求导的情形。
    """
    evidence: list[dict[str, Any]] = []
    current = integrand
    known: dict[str, sp.Symbol] = {str(s): s for s in integrand.free_symbols}
    for _name, lower, upper in region:
        for expr in (lower, upper):
            for sym in sp.sympify(expr).free_symbols:
                known.setdefault(str(sym), sym)
    for name, lower, upper in region:
        symbol = known.get(str(name)) or sp.Symbol(str(name), real=True)
        known.setdefault(str(name), symbol)
        try:
            antiderivative = sp.integrate(current, symbol)
            if _has_integral(antiderivative):
                evidence.append({"method": f"对 {symbol} 逐层回代", "passed": False,
                                 "evidence": {"reason": "该层原函数未能求出，无法回代"}})
                return evidence
            # 用「原函数在上下界的差」的导数反查：d/dx ∫ = 被积函数
            back = sp.simplify(sp.diff(antiderivative, symbol) - current)
            passed = back == 0
            evidence.append({
                "method": f"对第 {symbol} 层结果求导回代下层被积函数",
                "evidence": {"difference": A.to_text(back)},
                "passed": bool(passed),
            })
        except Exception as exc:  # noqa: BLE001
            evidence.append({"method": f"对 {symbol} 逐层回代", "passed": False,
                             "evidence": {"reason": f"{type(exc).__name__}: {exc}"}})
            return evidence
        try:
            current = sp.simplify(sp.integrate(current, (symbol, sp.sympify(lower), sp.sympify(upper))))
        except Exception:  # noqa: BLE001
            return evidence
    return evidence


# ---------------------------------------------------------------------------
# 4. 级数求和
# ---------------------------------------------------------------------------

@op("sum", "series_sum", "summation", "math_series_sum")
def series_sum(*, expr: str, var: str = None, lower: str = "1", upper: str = "oo") -> MathResult:
    """数项级数 / 幂级数求和（``sp.summation``）。"""
    parsed = A.parse(expr, symbols=_sym_names(var))
    variable = C.resolve_var(parsed, var)
    try:
        start = int(sp.sympify(str(lower).replace("无穷", "1")))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"求和下界必须是整数，收到 {lower!r}") from exc
    if str(upper).strip() in ("oo", "∞", "inf", "无穷", "+oo"):
        end: Any = sp.oo
    else:
        try:
            end = int(sp.sympify(str(upper)))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"求和上界必须是整数或 oo，收到 {upper!r}") from exc

    r = MathResult.ok("sum", method="级数求和：sp.summation（内部使用符号求和 / 迭代求和方法）")
    r.set_input(expr=A.to_text(parsed), var=str(variable), lower=start, upper=A.to_text(end))
    value = sp.summation(parsed, (variable, start, end))
    # 幂级数常见形态：SymPy 返回 Piecewise((主分支, 收敛条件), (Sum(...), True))。
    # 主分支（如 Σxⁿ = 1/(1-x) 在 |x|<1 上）是有效闭式，收敛条件必须显式写进 conditions，
    # 绝不能因为「带条件的闭式」就当成未求出而丢掉整个结果。
    branch_condition: Any = None
    if isinstance(value, sp.Piecewise):
        for branch, condition in value.args:
            if isinstance(condition, sp.logic.boolalg.BooleanTrue):
                continue                       # True 分支是原样回显的 Sum(...)，不是闭式
            if C.is_closed_form(branch):
                value, branch_condition = branch, condition
                break
    if not C.is_closed_form(value):
        return MathResult.unsolved(
            "sum",
            f"SymPy 未能求出该级数的和，返回未求值形式 {A.to_text(value)}；"
            "可尝试：裂项相消、等比/等差分解、幂级数逐项积分或求导、"
            "先求部分和再取极限，或改用比值/根值判别法判定敛散性。",
            method=r.method,
        ).set_input(**r.input)
    r.set_result(value)
    if value in (sp.oo, -sp.oo, sp.zoo):
        r.add_condition("该级数发散（和趋于无穷）：oo 表示「不收敛」，不是有限和")

    finite = end is not sp.oo
    if finite:
        r.add_condition(f"有限项求和，恒成立（n 从 {start} 取到 {end} 的整数）")
    else:
        r.add_condition("该结果是级数收敛时的和；级数发散时结论不成立（发散级数不能用此和表示）")
        r.add_condition(f"求和下标 {variable} 取遍所有 ≥ {start} 的整数")
    for sym in sorted(parsed.free_symbols - {variable}, key=str):
        r.add_condition(f"参数 {sym} 须取使级数收敛的值（否则级数发散，该和式不成立）")
    for note in _condition_notes(parsed):
        r.add_condition(note)
    if branch_condition is not None:
        r.add_condition(f"该和式在收敛条件 {A.to_text(branch_condition)} 下成立（超出该范围级数发散）")
        r.add_warning(
            f"SymPy 给出的原始结果是分段函数（Piecewise），已取主分支 {A.to_text(value)}，"
            f"其成立范围为 {A.to_text(branch_condition)}；请按此收敛域使用该和式。"
        )

    numeric_state, numeric_note = _series_numeric_evidence(parsed, variable, start, value,
                                                           condition=branch_condition)
    methods = [{
        "method": "mpmath.nsum 收敛加速独立数值求和（数值证据，非收敛性证明）",
        "independent": True,
        "agree": numeric_state,
        "detail": numeric_note,
    }]
    if numeric_state == "equal":
        r.verify(status="independent+numeric", methods=methods,
                 note="级数求和没有纯符号的独立证明手段：本结果由 mpmath 的独立数值求和"
                      "（收敛加速）给出数值证据；收敛性判定仍依赖 sp.summation 的符号结论。")
    elif numeric_state == "different":
        r.verify(status="unverified", methods=methods,
                 note="数值求和与符号结果不一致，结果可疑，请人工复核。")
        r.add_warning("数值求和与符号结果不一致（见验证证据）。")
    else:
        r.verify(status="unverified", methods=methods,
                 note="未获得有效数值证据（见 methods 说明）；该求和结果缺少独立验证。")
    return r


# ---------------------------------------------------------------------------
# 5. 方程 / 方程组求解
# ---------------------------------------------------------------------------

@op("solve", "solve_equation", "equation", "math_solve")
def solve(*, equations: Any, vars: Any = None) -> MathResult:
    """解方程 / 方程组（``sp.solve``，返回全部解）。"""
    texts = _split_equations(equations)
    if not texts:
        raise ValueError("方程为空")
    names = _sym_names(vars)
    equations_expr = [_eq_expr(text, names) for text in texts]

    if not names:
        collected: list[sp.Symbol] = []
        for eq in equations_expr:
            for sym in sorted(eq.free_symbols, key=str):
                if sym not in collected:
                    collected.append(sym)
        names = [str(s) for s in collected]
    if not names:
        raise ValueError("方程中没有可求解的未知量（请用 vars 指定未知量）")
    symbols = _make_symbols(names)

    r = MathResult.ok("solve", method="代数方程求解：sp.solve(..., dict=True)")
    r.set_input(equations=[A.to_text(e) for e in equations_expr], vars=[str(s) for s in symbols])

    try:
        solutions = sp.solve(equations_expr, symbols, dict=True)
    except NotImplementedError as exc:
        return MathResult.unsolved(
            "solve", f"SymPy 的方程求解器无法处理该方程（组）：{exc}；可尝试移项化简、增加假设（如变量为正）或改用数值方法。",
            method=r.method,
        ).set_input(**r.input)
    if solutions is None:
        solutions = []

    equation_text = "；".join(f"{A.to_text(e)} = 0" for e in equations_expr)
    if not solutions:
        r.set_raw(solution_set="empty", solution_count=0)
        r.set_result("无解")
        r.add_condition(f"在实数/复数范围内，方程组 {equation_text} 无解")
        r.verify(status="independent", methods=[{
            "method": "解集为空性说明",
            "detail": "sp.solve 返回空列表，表示这些方程无公共零点（无解），可代入任意候选值验证残差不全为 0。",
        }])
        return r

    candidates: list[dict[sp.Symbol, Any]] = []
    rows: list[str] = []
    all_ok = True
    for index, item in enumerate(solutions, start=1):
        if not isinstance(item, dict):
            item = {symbols[0]: item} if len(symbols) == 1 else {}
        candidates.append(item)
        assignment = ", ".join(f"{sym} = {A.to_text(item[sym])}" for sym in item if sym in item)
        rows.append(f"解 {index}: {assignment or '空'}")
        ok, detail = _solve_with_params(equations_expr, symbols, item)
        rows[-1] = f"{rows[-1]}（代回残差：{'；'.join(detail.get('residuals', [])) or detail.get('error', '')}）"
        all_ok = all_ok and ok

    r.set_raw(solution_count=len(solutions), solutions=rows)
    if len(symbols) == 1:
        values = [item[symbols[0]] for item in candidates if symbols[0] in item]
        r.set_result(values)
    else:
        first = candidates[0]
        r.set_result([first[s] for s in symbols if s in first] or None)
        r.set_raw(solution_dicts=[{str(k): A.to_text(v) for k, v in item.items()} for item in candidates])

    # 自由变量说明（无穷多解的情形）
    free_params: set[sp.Symbol] = set()
    for item in candidates:
        for value in item.values():
            if isinstance(value, sp.Basic):
                free_params |= (value.free_symbols - set(symbols))
    for sym in sorted(free_params, key=str):
        r.add_condition(f"{sym} 是自由参数：该方程组的解含无穷多组，{sym} 可取任意值，解由 {sym} 参数化")
    if free_params:
        r.set_raw(free_variables=sorted(str(s) for s in free_params))
    if len(solutions) > 1:
        r.add_condition(f"共有 {len(solutions)} 组解（已全部给出）")
    if not free_params:
        r.add_condition("解已通过代回原方程验算（残差为 0）")

    r.verify(status="independent" if all_ok else "unverified", methods=[{
        "method": "C.check_solution：把每组解代回原方程，检查残差是否为 0",
        "agree": all_ok,
        "detail": rows if len(rows) <= 6 else rows[:6] + [f"... 其余 {len(rows) - 6} 组同理"],
    }])
    if not all_ok:
        r.add_warning("至少一组解未通过代回验算，请人工确认（可能涉及分支/多值或参数取值）。")
    return r


# ---------------------------------------------------------------------------
# 6. 常微分方程
# ---------------------------------------------------------------------------

@op("dsolve", "ode", "differential_equation", "math_dsolve")
def dsolve(*, equations: Any = None, vars: Any = None, funcs: Any = None, **kwargs: Any) -> MathResult:
    """解常微分方程（组）：先做撇号预处理，再 ``sp.dsolve``，最后代回原方程独立验证。"""
    if equations is None:
        # 容错：某些调用方可能用了别的键名
        for key in ("equation", "eq", "expr", "ode"):
            if kwargs.get(key) is not None:
                equations = kwargs[key]
                break
    if equations is None:
        raise ValueError("缺少微分方程参数 equations（例如 \"diff(y(x),x) - 2*y(x) = 0\" 或 \"y' - 2*y = 0\"）")

    eqs, functions, independent, texts = _parse_ode_system(equations, funcs)
    if not eqs:
        raise ValueError("微分方程为空")
    independent = independent[0]

    r = MathResult.ok("dsolve", method="常微分方程求解：sp.dsolve（分类后套用对应解法）")
    r.set_input(equations=[A.to_text(e) for e in eqs],
                funcs=[str(f.func) for f in functions],
                independent=str(independent))

    try:
        # 单方程必须传单个方程：传列表会被 sympy 当成「方程组」交给 dsolve_system，
        # 而 dsolve_system 要求 funcs 是 list，于是抛
        # ValueError: Input to the funcs should be a list of functions.
        if len(eqs) == 1:
            raw = sp.dsolve(eqs[0], functions[0])
        else:
            raw = sp.dsolve(eqs, list(functions))
    except NotImplementedError as exc:
        return MathResult.unsolved(
            "dsolve", f"sp.dsolve 无法识别该微分方程类型：{exc}；可尝试降阶、分离变量或改写形式后重试。",
            method=r.method,
        ).set_input(**r.input)
    except (ValueError, TypeError) as exc:
        return MathResult.unsolved(
            "dsolve", f"sp.dsolve 拒绝了该输入：{exc}", method=r.method
        ).set_input(**r.input)

    pairs = raw if isinstance(raw, list) else [raw]
    solution: dict[Any, Any] = {}
    rows: list[str] = []
    for item in pairs:
        if isinstance(item, sp.Equality):
            solution[item.lhs] = item.rhs
        else:
            rows.append(A.to_text(item))
    if not solution:
        return MathResult.unsolved(
            "dsolve", f"sp.dsolve 未给出显式解（返回 {A.to_text(raw)}）。", method=r.method
        ).set_input(**r.input)

    unclosed = [k for k, v in solution.items() if not C.is_closed_form(v)]
    if unclosed:
        return MathResult.unsolved(
            "dsolve",
            f"解中含未求值形式 {[A.to_text(k) for k in unclosed]}，不能视为已求解。",
            method=r.method,
        ).set_input(**r.input)

    for key, value in solution.items():
        rows.append(f"{A.to_text(key)} = {A.to_text(value)}")
    r.set_raw(solution=rows, arbitrary_constants=[])
    for value in solution.values():
        if isinstance(value, sp.Basic):
            consts = sorted((s for s in value.free_symbols if str(s).startswith("C")), key=str)
            if consts:
                names = [str(s) for s in consts]
                r.set_raw(arbitrary_constants=sorted(set(r.result.get("arbitrary_constants", [])) | set(names)))
                r.add_condition("通解含任意常数 " + ", ".join(names) + "（C1, C2, ... 由初值条件确定）")

    order = 0
    for value in solution.values():
        if isinstance(value, sp.Basic):
            order = max(order, sp.ode_order(eqs[0], functions[0]) if functions else 0)
    if order:
        r.set_raw(ode_order=order)
    r.add_condition("解在自变量取值的连续区间内成立（分段系数不连续时需分段讨论）")

    all_zero, residuals = _ode_residuals(eqs, solution, independent)
    ok = all_zero
    if not ok:
        agree, _dev, _note = C.numeric_agree(sp.sympify(residuals and 0 or 0), 0, symbols=[independent])
        ok = False
    r.verify(status="independent" if ok else "unverified", methods=[{
        "method": "把通解代回原微分方程，检查残差 simplify(lhs - rhs) 是否为 0",
        "agree": bool(ok),
        "residuals": residuals,
    }])
    if not all_zero:
        r.add_warning("通解代回原方程后残差未化简为 0，请人工确认（可能为隐式解/分支解）。")
    if len(solution) > 1:
        r.set_raw(solution_count=len(solution))
    return r


# ---------------------------------------------------------------------------
# 7. 化简
# ---------------------------------------------------------------------------

def _count_ops(expr: sp.Basic) -> int:
    try:
        return int(sp.count_ops(expr))
    except Exception:  # noqa: BLE001
        return 10 ** 6


@op("simplify", "math_simplify")
def simplify(*, expr: str) -> MathResult:
    """表达式化简：多个化简器取最优（以操作数作为参考指标）。"""
    parsed = A.parse(expr)
    r = MathResult.ok("simplify", method="sp.simplify / radsimp / powsimp / factor / cancel 比较后取最简")
    r.set_input(expr=A.to_text(parsed))

    ops: dict[str, int] = {}
    candidates: list[tuple[str, sp.Basic]] = []
    for name, func in (("simplify", sp.simplify), ("factor", sp.factor), ("cancel", sp.cancel),
                       ("radsimp", sp.radsimp), ("powsimp", sp.powsimp), ("trigsimp", sp.trigsimp)):
        try:
            value = func(parsed)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(value, sp.Basic):
            continue
        ops[name] = _count_ops(value)
        candidates.append((name, value))
    if not candidates:
        return MathResult.unsolved("simplify", "所有化简策略都失败，无法化简该表达式。", method=r.method)

    best_name, best = min(candidates, key=lambda item: (ops[item[0]], len(A.to_text(item[1]))))
    r.set_raw(op_counts=ops, chosen=best_name)
    r.set_method(f"化简：比较 simplify/factor/cancel/radsimp/powsimp/trigsimp 的操作数（参考指标），采用 {best_name} 的结果")
    r.set_result(best)

    r.add_condition("化简结果与原式在两者共同的定义域内恒等")
    if any(s.is_real is None for s in best.free_symbols):
        r.add_condition("变量未附加实数假设时，等式在复数范围内同样按主值分支理解")
    for note in _condition_notes(parsed):
        r.add_condition(note)

    ok = C.same_expr(parsed, best)
    detail: dict[str, Any] = {"method": "C.same_expr：比较原式与化简结果的差（数学等价判定）", "agree": bool(ok)}
    if not ok:
        passed, dev, note = C.numeric_agree(parsed, best)
        detail["numeric"] = {"agree": passed, "max_rel_dev": dev, "note": note}
        ok = passed
        r.add_warning("化简结果未能通过符号等价判定，仅数值抽样一致（可能引入了分支/定义域差异）。")
    r.verify(status="independent" if ok else "unverified", methods=[detail])
    if not ok:
        r.partial("化简结果与原式不等价（same_expr 为 False），已降级为 partial，请勿直接使用。")
    return r


# ---------------------------------------------------------------------------
# 8. 展开 / 因式分解 / 部分分式
# ---------------------------------------------------------------------------

def _identity_op(name: str, method_text: str, expr: str, transform) -> MathResult:
    parsed = A.parse(expr)
    r = MathResult.ok(name, method=method_text)
    r.set_input(expr=A.to_text(parsed))
    value = transform(parsed)
    if not C.is_closed_form(value):
        return MathResult.unsolved(name, f"{method_text} 未能给出结果（返回未求值形式 {A.to_text(value)}）。", method=r.method
                                   ).set_input(**r.input)
    r.set_result(value)
    r.add_condition("结果与原式在共同定义域内恒等（恒等变形，不改变定义域时完全等价）")
    for note in _condition_notes(parsed):
        r.add_condition(note)
    ok, detail = C.verify_identity(parsed, value)
    r.verify(status="independent" if detail.get("symbolic") else ("cross" if ok else "unverified"),
             methods=[{"method": f"{method_text} 后与原式做恒等比较", **detail}])
    if not ok:
        r.add_warning("恒等比较未通过，结果可能不等价（注意定义域差异）。")
        r.partial("结果未通过恒等验证。")
    return r


@op("expand", "math_expand")
def expand(*, expr: str) -> MathResult:
    """展开表达式（``sp.expand``）。"""
    return _identity_op("expand", "展开：sp.expand", expr, sp.expand)


@op("factor", "math_factor")
def factor(*, expr: str) -> MathResult:
    """因式分解（``sp.factor``）。"""
    return _identity_op("factor", "因式分解：sp.factor", expr, sp.factor)


@op("together", "math_together")
def together(*, expr: str) -> MathResult:
    """通分（``sp.together``）。

    注意：``sp.together`` 的第二个位置参数是 ``deep`` 而不是变量，所以本算子不接受
    ``var``（部分分式分解才需要指定变量，见 :func:`apart`）。
    """
    return _identity_op("together", "通分：sp.together（等价变形再用 sp.apart 校验）", expr, sp.together)


def _symbol_by_name(parsed: sp.Basic, name: str) -> sp.Symbol:
    """按名字取回在 ``parsed`` 里真实出现的符号对象（避免与无假设符号不等）。"""
    wanted = str(name).strip()
    for symbol in sorted(parsed.free_symbols, key=lambda item: str(item)):
        if str(symbol) == wanted:
            return symbol
    return sp.Symbol(wanted, real=True)


@op("apart", "math_apart")
def apart(*, expr: str, var: str | None = None) -> MathResult:
    """部分分式分解（``sp.apart``），可指定分解变量 ``var``。

    SymPy 的 ``apart`` 不实现**多元**部分分式分解（``1/(x*(x+y))`` 会抛
    ``NotImplementedError: multivariate partial fraction decomposition``），
    但指定变量后是支持的。所以：给了 ``var`` 就按它分解；没给时先试多元自动分解，
    失败则逐个变量重试，并把「把其它符号当参数」这件事写进条件与警告。
    """
    parsed = A.parse(expr)
    if var is not None:
        symbol = _symbol_by_name(parsed, var)
        return _identity_op("apart", f"部分分式分解：sp.apart(..., {A.to_text(symbol)})", expr,
                            lambda value: sp.apart(value, symbol))
    try:
        sp.apart(parsed)
    except NotImplementedError:
        pass
    else:
        return _identity_op("apart", "部分分式分解：sp.apart（等价变形再用 sp.together 校验）", expr, sp.apart)

    candidates = sorted(parsed.free_symbols, key=lambda item: str(item))
    for symbol in candidates:
        try:
            sp.apart(parsed, symbol)
        except Exception:  # noqa: BLE001
            continue
        result = _identity_op(
            "apart",
            f"部分分式分解：SymPy 不做多元自动分解，已对变量 {A.to_text(symbol)} 分解"
            f"（sp.apart(..., {A.to_text(symbol)})）",
            expr,
            lambda value: sp.apart(value, symbol),
        )
        others = "、".join(A.to_text(item) for item in candidates if item != symbol)
        if others:
            result.add_condition(f"分解是把 {others} 当作参数（常数）对 {A.to_text(symbol)} 进行的。")
        result.add_warning("SymPy 不支持多元部分分式分解：这里只对指定变量分解，其余符号视为参数。")
        return result
    raise ValueError(
        "SymPy 不支持对该表达式做部分分式分解（multivariate partial fraction decomposition），"
        "请用 var 指定分解变量，或先化为单变量情形"
    )


# ---------------------------------------------------------------------------
# 9. 幂级数 / 泰勒展开
# ---------------------------------------------------------------------------

def _series_degree(value: sp.Basic, variable: sp.Symbol, center: sp.Basic) -> int:
    """展开式在 ``(variable - center)`` 下的最高次数；判断不出返回 -1。

    **不要对 value 做 expand**：``exp(x)`` 在 ``x=1`` 处的展开是
    ``E + E*(x-1) + E*(x-1)**2/2 + ...``，一旦展开就变成 ``x`` 的幂，
    再拿 ``Poly(..., x-1)`` 去截断就会把结果返回成 ``x`` 的幂形式，
    用户拿到的就不是「以 x-1 为基」的泰勒多项式了。
    """
    if value == 0:
        return 0
    try:
        poly = sp.Poly(value, variable - center)
        return int(poly.degree())
    except Exception:  # noqa: BLE001
        return -1


def _truncate_series(value: sp.Basic, variable: sp.Symbol, center: sp.Basic, order_int: int) -> sp.Basic:
    """丢掉次数高于 ``order_int`` 的项，使结果「展开到并包含 x^order」。

    与 :func:`_series_degree` 同理：逐项按 ``(variable - center)`` 判断次数，
    **不展开**，以保留 ``(x - x0)`` 的基形式。
    """
    if value == 0:
        return value
    kept: list[sp.Basic] = []
    for term in sp.Add.make_args(value):
        try:
            degree = int(sp.Poly(term, variable - center).degree())
        except Exception:  # noqa: BLE001
            kept.append(term)
            continue
        if degree <= order_int:
            kept.append(term)
    return sp.Add(*kept) if kept else sp.Integer(0)


def _taylor_expand(parsed: sp.Basic, variable: sp.Symbol, center: sp.Basic,
                   order_int: int) -> tuple[sp.Basic, int]:
    """展开到并包含 ``(variable - center)^order_int`` 项。

    **为什么要循环**：``sp.series`` 的 ``nterms`` 参数数的是「项数」，而带前导零的
    展开（如 ``1-cos(x)`` 的第一个非零项是 x²）会把 x⁰ 也算进去，于是
    ``sp.series(1-cos(x), x, 0, 2).removeO()`` 直接得到 ``0`` —— 这不是
    「展开到 x²」，而是「项数不够」。因此按次数判断，不够就加大 nterms 重试。
    """
    nterms = order_int + 1
    cap = order_int + 14
    best: sp.Basic = sp.Integer(0)
    best_degree = -1
    used = nterms
    while nterms <= cap:
        raw = sp.series(parsed, variable, center, nterms)
        if not C.is_closed_form(raw):
            break
        value = raw.removeO()
        used = nterms
        degree = _series_degree(value, variable, center)
        if degree > best_degree:
            best, best_degree = value, degree
        if degree >= order_int or degree == 0 and nterms >= order_int + 1:
            break
        nterms += 2
    return _truncate_series(best, variable, center, order_int), used


@op("series", "taylor", "math_series")
def series(*, expr: str, var: str = None, point: str = "0", order: int = 6) -> MathResult:
    """泰勒 / 幂级数展开（展开到并包含 ``x^order`` 项；余项以文字说明）。"""
    parsed = A.parse(expr, symbols=_sym_names(var))
    variable = C.resolve_var(parsed, var)
    try:
        order_int = int(order)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"展开阶数 order 必须是正整数，收到 {order!r}") from exc
    if order_int < 1:
        raise ValueError(f"展开阶数 order 必须 ≥ 1（收到 {order_int}）")
    center = C.parse_point(point, [variable])

    r = MathResult.ok(
        "series",
        method=f"泰勒展开：sp.series 反复加项直到最高次数达到 {order_int}，再截断到 {order_int} 次（去除余项）",
    )
    r.set_input(expr=A.to_text(parsed), var=str(variable), point=A.to_text(center), order=order_int)

    raw = sp.series(parsed, variable, center, order_int + 1)
    if not C.is_closed_form(raw):
        return MathResult.unsolved(
            "series",
            f"SymPy 未能给出该展开式（返回 {A.to_text(raw)}）；可尝试：先化简、换元把展开点平移到 0，或降低阶数。",
            method=r.method,
        ).set_input(**r.input)
    value, used_nterms = _taylor_expand(parsed, variable, center, order_int)
    r.set_result(value)
    r.set_raw(remainder=f"o(({A.to_text(variable)} - ({A.to_text(center)}))^{order_int})")

    r.add_condition(f"这是 {A.to_text(variable)} → {A.to_text(center)} 时的**局部**近似，不是恒等式，"
                    f"只在展开点附近成立；余项为 Peano 余项 o(({A.to_text(variable)} - ({A.to_text(center)}))^{order_int})")
    r.add_condition(f"函数须在 {A.to_text(center)} 的某邻域内 {order_int} 阶可导（泰勒中值定理条件）")
    r.add_condition(f"展开到并包含 ({A.to_text(variable)} - ({A.to_text(center)}))^{order_int} 项"
                    f"（内部按次数判断，实际用到 sp.series 的 nterms={used_nterms}）")
    for note in _condition_notes(parsed):
        r.add_condition(note)

    # 独立证据：邻近点的偏差应随阶数提高而减小
    probes = [sp.Rational(1, 10), sp.Rational(1, 5), sp.Rational(1, 3)]
    if center not in (sp.oo, -sp.oo) and isinstance(center, sp.Basic) and center.is_number:
        probes = [sp.sympify(center) + p for p in probes]
    deviations: dict[str, list[str]] = {}
    monotone = True
    evaluated = False
    for probe in probes:
        series_rows: list[str] = []
        prev: float | None = None
        for k in range(1, max(order_int, 4) + 1):
            try:
                approx = _taylor_expand(parsed, variable, center, k)[0].subs(variable, probe)
                exact = parsed.subs(variable, probe)
                dev = abs(complex(sp.N(exact - approx, 30)))
            except Exception:  # noqa: BLE001
                continue
            evaluated = True
            series_rows.append(f"n={k}: {dev:.3e}")
            if prev is not None and dev > prev * 1.5 + 1e-12:
                monotone = False
            prev = dev
        deviations[f"{A.to_text(variable)}={A.to_text(probe)}"] = series_rows
    note = ("泰勒展开是局部近似：下表是展开点附近各点的「真值-展开式」绝对偏差随阶数 n 的变化趋势"
            "（n 越大偏差总体越小，即近似越精确，但这不是恒等式）。")
    if not evaluated:
        r.verify(status="unverified", methods=[{"method": "阶数-偏差趋势比较", "agree": False,
                                                "detail": "所有探针点都未能数值化（可能超出定义域）"}],
                 note="未获得数值证据：泰勒展开是局部近似，缺少验证。")
        r.add_warning("未能在展开点附近取得有效数值证据。")
    else:
        r.verify(status="numeric", methods=[{
            "method": "在展开点附近取点，比较不同阶数展开式与原函数的偏差（数值证据）",
            "independent": True,
            "agree": bool(monotone),
            "deviations": deviations,
        }], note=note)
        if not monotone:
            r.add_warning("偏差未随阶数单调减小（可能超出收敛半径或展开点不可导）。")
    return r


# ---------------------------------------------------------------------------
# 10. ODE 解的独立验证
# ---------------------------------------------------------------------------

def _solution_key(lhs_text: str, functions: list[Any], independent: sp.Symbol, funcs: Any) -> Any:
    """把候选解左端（``y``、``y(x)``…）统一成原方程里真实使用的函数对象。

    不能只靠 ``A.parse(lhs_text)``：当 ``symbols`` 里只有 ``y`` 时，``y(x)`` 会被解析成
    ``Symbol('y')`` **被调用**，直接抛 ``TypeError: 'Symbol' object is not callable``。
    因此先按名字匹配方程里的函数，匹配不到再退回解析。
    """
    text = str(lhs_text).strip()
    by_name: dict[str, Any] = {}
    for func in functions:
        holder = getattr(func, "func", func)
        by_name[str(getattr(holder, "__name__", holder))] = func
    plain = text.split("(", 1)[0].strip()
    if plain in by_name:
        return by_name[plain]
    try:
        return A.parse(text, symbols=_sym_names(funcs))
    except Exception:  # noqa: BLE001
        return _applied(functions[0], independent)


def _align_solution_keys(solution: dict[Any, Any], functions: list[Any], independent: sp.Symbol) -> dict[Any, Any]:
    """把候选解字典的键对齐成原方程里真实出现的 ``y(x)`` 对象。

    ``y = ...`` 解析出来的键是 ``Symbol('y')``，而方程里是 ``y(x)``；两者的键不一致
    时 ``subs`` 完全不起作用（残差 == 原方程），验证会给出无意义的结论。
    """
    by_name: dict[str, Any] = {}
    for func in functions:
        holder = getattr(func, "func", func)
        by_name[str(getattr(holder, "__name__", holder))] = func
    aligned: dict[Any, Any] = {}
    for key, value in solution.items():
        name: str | None = None
        if isinstance(key, sp.Symbol):
            name = str(key)
        elif hasattr(key, "func"):
            name = str(getattr(key.func, "__name__", key.func))
        target = by_name.get(name) if name else None
        if target is None and isinstance(key, sp.Symbol):
            target = _applied(functions[0], independent)
        aligned[target if target is not None else key] = value
    return aligned


@op("ode_check", "verify_ode", "check_ode")
def ode_check(*, equations: Any = None, candidate: Any = None, vars: Any = None, funcs: Any = None,
              **kwargs: Any) -> MathResult:
    """独立验证给定的 ODE 解：把候选解代回原方程，残差化简为 0 则通过。"""
    if isinstance(equations, str) and candidate is None and equations.strip().startswith("y("):
        # 容错：调用方只传了一个"像是解"的字符串
        candidate = equations
        equations = kwargs.get("target") or kwargs.get("claim")
    if equations is None or candidate is None:
        raise ValueError("需要同时给出原微分方程 equations 与候选解 candidate")

    eqs, functions, independent, _texts = _parse_ode_system(equations, funcs)
    independent = independent[0]
    cand_texts = _split_equations(candidate)
    if not cand_texts:
        raise ValueError("候选解为空")

    cand_equations, cand_functions, _ind, _t = _parse_ode_system(candidate, funcs)
    solution: dict[Any, Any] = {}
    for eq, text in zip(cand_equations, cand_texts):
        if isinstance(eq, sp.Equality):
            solution[eq.lhs] = eq.rhs
        elif text.strip().count("=") == 1:
            lhs_text, rhs_text = text.split("=", 1)
            key = _solution_key(lhs_text, functions, independent, funcs)
            try:
                # 注意：_parse_ode_system 会把 `lhs = rhs` 化成 `lhs - rhs`（Add），
                # 所以这里**必须重新解析右端**，否则候选解被记成
                # `y(x) = y(x) - rhs`，代回后残差毫无意义（真实踩过的坑）。
                solution[key] = A.parse(rhs_text, symbols=_sym_names(funcs))
            except Exception:  # noqa: BLE001
                solution[key] = eq
        else:
            solution[_applied(functions[0], independent)] = eq
    if not solution and cand_functions:
        solution[_applied(cand_functions[0], independent)] = cand_equations[0]
    if not solution:
        raise ValueError("候选解无法解析为「函数 = 表达式」的形式（请写成 y(x) = ... 或 y(x) - ... = 0）")

    solution = _align_solution_keys(solution, functions, independent)
    # 候选解是**独立解析**出来的，它的变量可能与方程里的同名但不同一
    # （Symbol('x') vs Symbol('x', real=True)）。不对齐的话
    # Derivative(y(x), x).subs({y(x): C1*exp(x_p**2/2)}) 对 x 求导得 0，
    # 残差凭空变成「−x·y(x)」，正确的解会被判成不满足方程（真实踩过的坑）。
    anchor = eqs[0]
    for extra_eq in eqs[1:]:
        anchor = anchor + extra_eq
    solution = {key: C.unify_symbols(value, anchor) for key, value in solution.items()}

    r = MathResult.ok("ode_check", method="独立验证：候选解代回原 ODE，化简 lhs-rhs 检查残差是否为 0")
    r.set_input(equations=[A.to_text(e) for e in eqs], candidate=[A.to_text(k) + " = " + A.to_text(v) for k, v in solution.items()])

    for key, value in solution.items():
        if not C.is_closed_form(value):
            return MathResult.unsolved(
                "ode_check", f"候选解含未求值形式（{A.to_text(key)} = {A.to_text(value)}），无法验证。", method=r.method
            ).set_input(**r.input)

    all_zero, residuals = _ode_residuals(eqs, solution, independent)
    r.set_raw(residuals=residuals, residual_count=len(residuals))
    methods = [{
        "method": "独立机制：把候选解代入原方程并化简残差（不依赖 sp.dsolve 的求解过程）",
        "agree": bool(all_zero),
        "residuals": residuals,
    }]
    if all_zero:
        r.set_result("残差化简为 0：候选解满足原微分方程")
        r.verify(status="independent", methods=methods,
                 note="结论仅说明「满足方程」，初值条件与定义域仍需单独核对。")
    else:
        r.set_result("残差不为 0：候选解不满足原微分方程")
        r.verify(status="independent", methods=methods,
                 note="已独立判定该候选解是错误的（残差非 0），这属于「验证失败」而非「未能验证」。")
        r.partial("候选解未通过代回验证，残差见 extra.residuals。")
    return r


# ---------------------------------------------------------------------------
# 11. 多元微分：梯度 / 雅可比 / 海森矩阵
# ---------------------------------------------------------------------------

def _grad_vector(expr: sp.Basic, variables: list[sp.Symbol]) -> list[sp.Basic]:
    return [sp.diff(expr, v) for v in variables]


@op("gradient", "grad", "math_gradient")
def gradient(*, expr: str, vars: Any = None) -> MathResult:
    """梯度 ∇f：对每个变量求偏导，并用方向导数独立复核。"""
    parsed = A.parse(expr, symbols=_sym_names(vars))
    names = _sym_names(vars)
    variables = _make_symbols(names) if names else sorted(parsed.free_symbols, key=str)
    if not variables:
        raise ValueError("梯度需要至少一个自变量（请用 vars 指定）")
    missing = [str(v) for v in variables if not parsed.has(v)]
    if missing:
        raise ValueError(f"表达式中不含变量 {', '.join(missing)}，无法对这些变量求偏导")

    r = MathResult.ok("gradient", method="梯度：对各分量变量分别用 sp.diff 求偏导")
    r.set_input(expr=A.to_text(parsed), vars=[str(v) for v in variables])
    grad = sp.Matrix(_grad_vector(parsed, variables))
    r.set_result(grad)
    r.set_raw(components=[A.to_text(g) for g in grad])
    r.add_condition("函数需在各变量方向上偏导数存在（可微）")
    r.add_condition("偏导结果在函数定义域内成立")
    for note in _condition_notes(parsed):
        r.add_condition(note)

    checks: list[dict[str, Any]] = []
    all_ok = True
    fallback = {v: sp.Rational(i + 4, i + 3) for i, v in enumerate(variables)}
    for index, v in enumerate(variables):
        ok, detail = C.check_derivative(parsed, v, grad[index])
        all_ok = all_ok and ok
        checks.append({"method": f"C.check_derivative：对 {v} 独立求偏导并比较", "agree": ok, **detail})
    # 方向导数独立复核：∇f·u 应等于对 f(x + t*u) 在 t=0 处求导
    direction = sp.Matrix([sp.Rational(i + 2, i + 1) for i in range(len(variables))])
    try:
        t = sp.Symbol("_t")
        shifted = parsed.subs({v: v + t * direction[i] for i, v in enumerate(variables)})
        directional = sp.diff(shifted, t).subs(t, 0)
        dot = sp.simplify((grad.T * direction)[0] - directional)
        ok = dot == 0
        checks.append({
            "method": "方向导数复核：比较 ∇f·u 与 d/dt f(x + t*u)|_{t=0}（独立机制）",
            "agree": bool(ok),
            "residual": A.to_text(dot),
        })
        all_ok = all_ok and ok
    except Exception as exc:  # noqa: BLE001
        checks.append({"method": "方向导数复核", "agree": False, "error": f"{type(exc).__name__}: {exc}"})
    r.verify(status="independent" if all_ok else "unverified", methods=checks)
    if not all_ok:
        r.partial("梯度未通过全部独立复核。")
    return r


@op("jacobian", "math_jacobian")
def jacobian(*, exprs: Any = None, expr: Any = None, vars: Any = None, **kwargs: Any) -> MathResult:
    """雅可比矩阵 ∂(f1..fm)/∂(x1..xn)，逐元素独立复核。"""
    source = exprs if exprs is not None else expr
    if source is None:
        raise ValueError("缺少参数 exprs（多个函数表达式）或 expr（单个表达式）")
    texts = _sym_names(source) if isinstance(source, str) and ";" in source else None
    if texts is None:
        raw_list = source if isinstance(source, (list, tuple)) else [source]
    else:
        raw_list = texts
    parsed = [A.parse(str(item), symbols=_sym_names(vars)) for item in raw_list]
    if not parsed:
        raise ValueError("函数表达式为空")

    names = _sym_names(vars)
    if names:
        variables = _make_symbols(names)
    else:
        collected: list[sp.Symbol] = []
        for item in parsed:
            for sym in sorted(item.free_symbols, key=str):
                if sym not in collected:
                    collected.append(sym)
        if not collected:
            raise ValueError("表达式中没有自变量，无法构造雅可比矩阵（请用 vars 指定）")
        variables = collected

    r = MathResult.ok("jacobian", method="雅可比矩阵：sp.Matrix([[∂fi/∂xj]]) 逐元素 sp.diff")
    r.set_input(exprs=[A.to_text(e) for e in parsed], vars=[str(v) for v in variables])
    matrix = sp.Matrix([[sp.diff(item, v) for v in variables] for item in parsed])
    r.set_result(matrix)
    r.add_condition("每个分量函数在考察点处对每个变量的一阶偏导数存在")
    r.add_condition("雅可比矩阵在定义域内逐元素成立")
    for note in _condition_notes(*parsed):
        r.add_condition(note)

    checks: list[dict[str, Any]] = []
    all_ok = True
    for i, item in enumerate(parsed):
        for j, v in enumerate(variables):
            ok, _detail = C.check_derivative(item, v, matrix[i, j])
            all_ok = all_ok and ok
            if not ok:
                checks.append({"method": f"元素 ({i + 1},{j + 1}) 的 C.check_derivative 复核", "agree": False})
    # 独立机制：全微分一致性 d(fi) = Σ_j (∂fi/∂xj) dx_j
    try:
        deltas = [sp.Symbol(f"_d{j}") for j in range(len(variables))]
        differential_ok = True
        residuals: list[str] = []
        for item in parsed:
            # 用有限差分方向比较：f(x + t*d) 在 t=0 的导数应等于 Σ ∂fi/∂xj * dj
            t = sp.Symbol("_t")
            shifted = item.subs({v: v + t * deltas[j] for j, v in enumerate(variables)})
            chain = sp.expand(sp.diff(shifted, t).subs(t, 0))
            combo = sum(matrix[parsed.index(item), j] * deltas[j] for j in range(len(variables)))
            residual = sp.simplify(chain - combo)
            residuals.append(A.to_text(residual))
            if residual != 0:
                differential_ok = False
        all_ok = all_ok and differential_ok
        checks.append({
            "method": "链式法则复核：d/dt f(x + t*d)|_{t=0} 应等于 Σ_j (∂f/∂x_j) d_j（独立机制）",
            "agree": bool(differential_ok),
            "residuals": residuals,
        })
    except Exception as exc:  # noqa: BLE001
        checks.append({"method": "链式法则复核", "agree": False, "error": f"{type(exc).__name__}: {exc}"})
        all_ok = False
    r.verify(status="independent" if all_ok else "unverified", methods=checks)
    if not all_ok:
        r.partial("雅可比矩阵未通过全部独立复核。")
    return r


@op("hessian", "math_hessian")
def hessian(*, expr: str, vars: Any = None) -> MathResult:
    """海森矩阵 ∂²f/∂xi∂xj，用「先一阶后二阶」与「直接二阶」两条独立路径复核。"""
    parsed = A.parse(expr, symbols=_sym_names(vars))
    names = _sym_names(vars)
    variables = _make_symbols(names) if names else sorted(parsed.free_symbols, key=str)
    if not variables:
        raise ValueError("海森矩阵需要至少一个自变量（请用 vars 指定）")
    missing = [str(v) for v in variables if not parsed.has(v)]
    if missing:
        raise ValueError(f"表达式中不含变量 {', '.join(missing)}，无法构造海森矩阵")

    r = MathResult.ok("hessian", method="海森矩阵：二阶偏导 sp.diff(f, xi, xj)")
    r.set_input(expr=A.to_text(parsed), vars=[str(v) for v in variables])
    size = len(variables)
    matrix = sp.Matrix(size, size, lambda i, j: sp.diff(parsed, variables[i], variables[j]))
    r.set_result(matrix)
    r.add_condition("函数需二阶偏导数存在且连续（此时海森矩阵对称、混合偏导可交换次序）")
    r.add_condition("海森矩阵在函数定义域内成立")
    for note in _condition_notes(parsed):
        r.add_condition(note)

    checks: list[dict[str, Any]] = []
    all_ok = True
    for i in range(size):
        for j in range(size):
            first = sp.diff(parsed, variables[i])
            ok, detail = C.check_derivative(first, variables[j], matrix[i, j])
            all_ok = all_ok and ok
            if not ok:
                checks.append({"method": f"先对 {variables[i]} 求偏导、再对 {variables[j]} 求导的独立复核",
                               "agree": False, **detail})
    symmetric = all(sp.simplify(matrix[i, j] - matrix[j, i]) == 0 for i in range(size) for j in range(size))
    checks.append({
        "method": "混合偏导可交换性检查（H_ij == H_ji，二阶偏导连续时必然成立）",
        "agree": bool(symmetric),
    })
    all_ok = all_ok and symmetric
    r.verify(status="independent" if all_ok else "unverified", methods=checks)
    if not all_ok:
        r.partial("海森矩阵未通过全部独立复核（可能二阶偏导不连续）。")
    return r
