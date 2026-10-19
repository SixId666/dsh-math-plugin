"""mathkit.numeric —— 高精度数值计算模块（数值求值 / 数值积分 / 求根 / 优化 / 数值验证）。

设计原则（与整个引擎一致）：
    * 计算交给工具、思考交给模型、验证交给独立机制；
    * 符号计算是主力，本模块负责 ① 高精度数值求值，② 为符号结果提供**独立**数值验证，
      ③ 符号引擎求不出时的数值兜底；
    * **数值结果永远不是解析证明**：所有算子的 ``verification.status`` 只会是
      ``"numeric"``（或 ``"unverified"``），并在 ``note`` 里写明「这是数值近似/数值证据」；
    * 每个数值结果都带上 ``digits``（目标有效位数）与 ``abs_error``（误差界或独立路线之差），
      误差估计较大时 ``add_warning`` 说明可信度下降。

实现要点（踩过的坑，勿踩）：
    * SymPy 的 Symbol 带假设，``Symbol('x') != Symbol('x', real=True)``；
      用不匹配的 Symbol 去做 diff/limit/subs 会**静默**给出 0 或原样回显。
      因此本模块一律先 ``A.parse``，再从 ``expr.free_symbols`` 里取真实 Symbol。
    * ``mpmath.quad`` 对振荡半无穷积分（如 sin(x)/x 在 [0,∞)）会给出**错误**结果并报误差 1.0，
      必须改用 ``mpmath.quadosc``；对有限区间的内点奇性要把它加进 ``points``。
    * ``mpmath.nsum(method="d"/"s")`` 对 Σ1/n² 分别偏 2.8e-3 / 5.6e-5，``method="r"`` 才准。
    * Euler–Maclaurin 尾项符号：Σ_{n=N+1}^∞ f = ∫_N^∞ f − f(N)/2 − f'(N)/12 + f'''(N)/720 − f⁽⁵⁾/30240 + …
"""

from __future__ import annotations

from typing import Any

import math
import re

import mpmath
import sympy as sp

from . import ast as A
from . import sympy_core as C
from .engine import op
from .result import MathResult

__all__ = [
    "numeric",
    "numeric_integrate",
    "numeric_solve",
    "numeric_optimize",
    "numeric_derivative",
    "compare_numeric",
    "numeric_limit",
    "numeric_sum",
    "numeric_matrix",
]

# --------------------------------------------------------------------------------------
# 通用常量与工具
# --------------------------------------------------------------------------------------

_NUMERIC_CAVEAT = (
    "数值结果不是解析证明：以上是有限精度的高精度数值近似/数值证据，"
    "不能替代符号推导与严格证明。"
)

#: 与 sympy_core.numeric_agree 内部一致的抽样网格（正有理数，避开 0/1 与奇点）
_GRID = (
    sp.Rational(1, 3),
    sp.Rational(1, 2),
    sp.Rational(2, 3),
    sp.Rational(3, 2),
    sp.Rational(5, 4),
    sp.Rational(7, 5),
    sp.Rational(9, 7),
    sp.Rational(11, 6),
    sp.Rational(4, 3),
)
_GRID_STR = tuple(str(sp.N(v, 30)) for v in _GRID)

_MAX_DIGITS = 200


def _check_digits(digits: Any) -> int:
    """校验有效位数，越界抛 ValueError（调度器转 invalid_input）。"""
    if isinstance(digits, bool) or digits is None:
        raise ValueError(f"digits 必须是 1..{_MAX_DIGITS} 之间的整数，收到 {digits!r}")
    try:
        d = int(digits)
    except (TypeError, ValueError):
        raise ValueError(f"digits 必须是 1..{_MAX_DIGITS} 之间的整数，收到 {digits!r}") from None
    if not 1 <= d <= _MAX_DIGITS:
        raise ValueError(f"digits 必须在 1..{_MAX_DIGITS} 之间，收到 {d}")
    return d


def _check_order(order: Any, *, name: str = "order", maximum: int = 10) -> int:
    try:
        k = int(order)
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须是 1..{maximum} 之间的整数，收到 {order!r}") from None
    if not 1 <= k <= maximum:
        raise ValueError(f"{name} 必须在 1..{maximum} 之间，收到 {k}")
    return k


def _num(value: Any) -> Any:
    """把数值转成 JSON 友好的 float；下溢/非有限时保留字符串以免丢精度。"""
    if value is None:
        return None
    try:
        f = float(value)
    except Exception:  # noqa: BLE001
        return str(value)
    if f == 0.0:
        try:
            if mpmath.mpf(value) != 0:
                return mpmath.nstr(mpmath.mpf(value), 6)
        except Exception:  # noqa: BLE001
            pass
        return 0.0
    if not math.isfinite(f):
        return mpmath.nstr(value, 8) if isinstance(value, mpmath.mpf) else str(value)
    return f


def _sp_float(value: Any, digits: int) -> sp.Basic:
    """把 mpmath 数值（实数或复数）转成 SymPy 对象。

    **不要用 ``str(mpf)``**：``str`` 的位数取决于**当前** ``mpmath.mp.dps``，
    离开 ``workdps`` 高精度块之后再 str 会把结果截断到 15 位（实测
    ``numeric_sum(..., digits=40)`` 曾给出 ``1.644934066848230000…``，
    真值 1.64493406684822643647241516665）。``mpmath.nstr(x, n)`` 与全局 dps 无关。
    """
    import mpmath as mp

    try:
        if isinstance(value, mp.mpc):
            return sp.Float(mp.nstr(mp.re(value), digits + 5), digits) + sp.I * sp.Float(
                mp.nstr(mp.im(value), digits + 5), digits
            )
        if isinstance(value, mp.mpf):
            return sp.Float(mp.nstr(value, digits + 5), digits)
        if isinstance(value, float):
            return sp.Float(value, digits)
        return sp.Float(str(value), digits)
    except Exception:  # noqa: BLE001
        try:
            return sp.N(sp.sympify(value), digits)
        except Exception:  # noqa: BLE001
            return sp.Symbol("nan")  # pragma: no cover - 兜底，正常不会走到


def _mpf(value: Any, digits: int = 30) -> Any:
    """把 SymPy 数值 / Python 数值 / 字符串端点转成 mpmath 数值（支持 ±oo）。"""
    import mpmath as mp

    if isinstance(value, bool):
        raise ValueError(f"无法把布尔值 {value!r} 当作数值")
    if isinstance(value, (int, float)):
        return mp.mpf(value)
    if isinstance(value, mp.mpf):
        return value
    if isinstance(value, sp.Basic):
        if value is sp.oo:
            return mp.inf
        if value is -sp.oo:
            return -mp.inf
        if value in (sp.zoo, sp.nan):
            raise ValueError(f"{value} 不是有限数值")
        if value.free_symbols:
            names = ", ".join(sorted(str(s) for s in value.free_symbols))
            raise ValueError(f"数值 {value} 含自由符号（{names}），必须给出具体数值")
        if value.has(sp.oo, -sp.oo):
            raise ValueError(f"{value} 含无穷，无法作为有限数值使用")
        try:
            return mp.mpf(str(sp.N(value, max(30, digits))))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"无法把 {value} 转成高精度数值: {exc}") from None
    try:
        return mp.mpf(str(value))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"无法把 {value!r} 解析成数值: {exc}") from None


def _safe(f: Any, *args: Any, allow_inf: bool = False) -> Any:
    """调用 mpmath 函数并吞掉定义域错误，失败返回 None。"""
    import mpmath as mp

    try:
        v = f(*args)
    except Exception:  # noqa: BLE001
        return None
    try:
        v = mp.mpmathify(v)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(v, mp.mpc):
        if mp.im(v) == 0:
            v = mp.re(v)
        else:
            return v if allow_inf else v
    if not allow_inf and not mp.isfinite(v):
        return None
    return v


def _is_real_mp(value: Any) -> bool:
    return not isinstance(value, mpmath.mpc) or mpmath.im(value) == 0


def _rel_dev(a: Any, b: Any) -> float:
    """相对偏差 |a-b| / max(1,|b|)（sympy_core.numeric_agree 的同一口径）。"""
    try:
        num = abs(complex(mpmath.mpmathify(a)) - complex(mpmath.mpmathify(b)))
        scale = max(1.0, abs(complex(mpmath.mpmathify(b))))
        return float(num / scale)
    except Exception:  # noqa: BLE001
        return float("inf")


def _reliable_digits(value: Any, abs_error: Any, digits: int) -> int | None:
    """按误差界估计「可靠有效位数」：abs_error / max(1,|value|) ≈ 10^-k ⇒ 约 k 位。

    ``digits`` 是**目标**精度，``abs_error`` 才是可信度；``reliable_digits`` 把两者
    联系起来，避免使用者把 40 位输出里的后若干位垃圾当成有效数字。
    """
    try:
        e = abs(float(abs_error))
        v = abs(float(value))
    except Exception:  # noqa: BLE001
        return None
    if not math.isfinite(e):
        return None
    scale = max(1.0, v)
    if e <= 0:
        return digits
    rel = e / scale
    if rel >= 1:
        return 0
    k = int(math.floor(-math.log10(rel)))
    return max(0, min(digits, k))


def _name_list(var: Any) -> list[str]:
    """把 var/vars 参数统一成变量名列表。"""
    if var is None:
        return []
    if isinstance(var, sp.Symbol):
        return [var.name]
    if isinstance(var, dict):
        return [str(k) for k in var.keys()]
    if isinstance(var, (list, tuple, set)):
        out: list[str] = []
        for item in var:
            out.extend(_name_list(item))
        return out
    text = str(var).strip().strip("[](){}")
    if not text:
        return []
    return [p.strip() for p in re.split(r"[,;\s]+", text) if p.strip()]


def _as_seq(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, sp.Basic):
        return [value]
    text = str(value).strip()
    if not text:
        return []
    return [p for p in re.split(r"[,;\s]+", text) if p]


def _symbol_for(expr: Any, name: str) -> Any:
    """从表达式自带的 free_symbols 里取同名 Symbol（避免假设不匹配的陷阱）。"""
    if isinstance(expr, sp.Basic):
        for s in expr.free_symbols:
            if s.name == name:
                return s
    return None


def _symbols_for(exprs: list[Any], names: list[str]) -> list[Any]:
    out: list[Any] = []
    for name in names:
        sym = None
        for e in exprs:
            sym = _symbol_for(e, name)
            if sym is not None:
                break
        if sym is None:
            sym = A.get_symbols([name])[0]
        out.append(sym)
    return out


_FUNC_MAP = {"abs": "Abs", "max": "Max", "min": "Min"}
_FUNC_NAME_HINTS = {
    "abs": "Abs(x) 或 |x|（本模块已自动改名；若仍报错请检查空格/大小写）",
    "max": "Max(a, b)",
    "min": "Min(a, b)",
}


def _pre_normalize_special(text: str) -> str:
    """把 ``A.parse`` 会误解析的常见写法改成 SymPy 能识别的名字。

    ``mathkit.ast.parse`` 的变换里含 ``implicit_multiplication_application``，
    未识别的多字母函数名会被 ``split_symbols`` 拆成单字母乘积：实测
    ``A.parse("abs(x)", symbols=["x"])`` 得到 ``a*b*s*x``（三个凭空出现的自由变量！），
    而 ``A.parse("|x|")`` 直接抛语法错误。绝对值在考研数学里极常见，因此这里
    先把 ``abs(`` / ``max(`` / ``min(`` 改名成 ``Abs(`` / ``Max(`` / ``Min(``，
    并把 ``|expr|`` 改写成 ``Abs(expr)``；之后仍然一律交给 ``A.parse`` 解析。
    """
    out = text
    if "|" in out and "Abs" not in out:
        out = re.sub(r"\|([^|]+)\|", r"Abs(\1)", out)
    out = re.sub(
        r"(?<![A-Za-z0-9_])(abs|max|min)\s*\(",
        lambda m: _FUNC_MAP[m.group(1).lower()] + "(",
        out,
        flags=re.IGNORECASE,
    )
    return out


def _split_name_hint(text: str, extras: list[str]) -> str | None:
    """识别「多字母函数名被拆成单字母乘积」的污染（只报告，不改写表达式）。"""
    if not extras:
        return None
    if len(extras) == 1 and extras[0].lower() in _FUNC_NAME_HINTS:
        return (
            f"其中 {extras[0]}(...) 是函数名而不是变量名，被解析成了自由变量 {extras[0]}；"
            f"正确写法示例：{_FUNC_NAME_HINTS[extras[0].lower()]}"
        )
    if len(extras) < 2 or not all(len(e) == 1 for e in extras):
        return None
    letters = "".join(sorted(extras))
    for token in re.findall(r"[A-Za-z]{2,}", text):
        if len(token) == len(extras) and "".join(sorted(token.lower())) == letters:
            suggest = _FUNC_NAME_HINTS.get(token.lower(), f"{token.capitalize()}(...)")
            return (
                f"其中 {token}(...) 很可能被解析器拆成了单字母乘积 {'*'.join(sorted(extras))}"
                f"（mathkit.ast 的隐式乘法变换所致），正确写法示例：{suggest}"
            )
    return None


def _parse_expr(text: Any, names: list[str] | None) -> Any:
    raw = str(text)
    parsed = A.parse(_pre_normalize_special(raw), symbols=list(names) if names else None)
    if names and isinstance(parsed, sp.Basic):
        allowed = {str(n) for n in names}
        extras = sorted({str(s) for s in parsed.free_symbols if str(s) not in allowed})
        hint = _split_name_hint(raw, extras)
        if hint:
            raise ValueError(f"表达式 {raw!r} 解析结果可疑（得到 {A.to_text(parsed)}）：{hint}")
    return parsed


def _equation_expr(text: Any, names: list[str]) -> Any:
    """把 "x^3-x-1=0" / "x^2+y^2=1" 变成 f(x)=0 的表达式（左边-右边）。"""
    raw = str(text).strip()
    if not raw:
        raise ValueError("方程不能为空")
    for sep in ("==", "="):
        if sep in raw:
            left, right = raw.split(sep, 1)
            if not left.strip() or not right.strip():
                raise ValueError(f"方程 {text!r} 缺少等号一侧")
            return _parse_expr(left, names) - _parse_expr(right, names)
    return _parse_expr(raw, names)


def _lambdify_mp(syms: list[Any], expr: Any) -> Any:
    try:
        return sp.lambdify(syms, expr, modules=["mpmath"])
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"无法把表达式转成 mpmath 可调用对象（可能含未求值的符号函数）: {exc}") from None


def _eval_mp(expr: Any, syms: list[Any], point: tuple[Any, ...]) -> Any:
    """高精度求值；syms 为空表示常数表达式。"""
    import mpmath as mp

    if syms:
        return _safe(sp.lambdify(syms, expr, modules=["mpmath"]), *point)
    dummy = sp.Dummy()
    return _safe(sp.lambdify([dummy], expr, modules=["mpmath"]), mp.mpf(0))


def _verify_numeric(r: MathResult, methods: list[str], note: str) -> None:
    r.verify(status="numeric", methods=list(methods), note=note)


def _nsimplify_hint(value: Any, digits: int) -> str | None:
    """给数值结果一个「疑似闭式」提示（**只是猜测**，不作为结果）。"""
    try:
        val = sp.N(value, min(max(digits, 12), 25))
        if not val.is_number or val.free_symbols:
            return None
        tol = sp.Float(10) ** (-max(6, min(digits, 20) - 4))
        consts = [sp.pi, sp.E, sp.EulerGamma, sp.log(2), sp.sqrt(2), sp.sqrt(3), sp.sqrt(5), sp.sqrt(sp.pi)]
        cand = sp.nsimplify(sp.Float(val, 20), consts, tolerance=tol)
        if cand is None or cand == val:
            return None
        if sp.count_ops(cand) > 6:
            return None
        dev = abs(complex(sp.N(cand - val, 20)))
        if dev > float(tol) * 10:
            return None
        return A.to_text(sp.simplify(cand))
    except Exception:  # noqa: BLE001
        return None


def _first_available(routes: list[tuple[str, Any]], priority: list[str]) -> tuple[str, Any] | None:
    for name in priority:
        for rname, rval in routes:
            if rname == name and rval is not None:
                return rname, rval
    for rname, rval in routes:
        if rval is not None:
            return rname, rval
    return None


# --------------------------------------------------------------------------------------
# 1. numeric —— 高精度数值求值
# --------------------------------------------------------------------------------------


@op("numeric", "eval", "n", "math_numeric")
def numeric(expr: str, vars: dict | None = None, digits: int = 15) -> MathResult:
    """把表达式中的自由变量代入后做高精度数值求值。

    返回 ``result.text`` 为高精度十进制串、``result.approx`` 为双精度近似；
    ``result.exact_form`` 是代入后的**精确形式**，``result.is_approximate`` 标出哪个是近似。
    """
    d = _check_digits(digits)
    subs_in = vars if vars is not None else {}
    if not isinstance(subs_in, dict):
        raise ValueError('vars 必须是形如 {"x": 1.5} 或 {"x": "pi/4"} 的字典')

    names = [str(k) for k in subs_in.keys()]
    parsed = _parse_expr(expr, names or None)
    exact = parsed
    for name in names:
        sym = _symbol_for(parsed, name)
        if sym is None:
            continue
        try:
            val = _parse_expr(subs_in[name], None)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"vars[{name!r}] 的值 {subs_in[name]!r} 无法解析: {exc}") from None
        if isinstance(val, sp.Basic) and val.free_symbols:
            raise ValueError(f"vars[{name!r}] 的值 {subs_in[name]!r} 仍含自由符号，必须给出具体数值")
        exact = exact.subs(sym, val)

    input_kwargs = {"expr": str(expr), "vars": {str(k): str(v) for k, v in subs_in.items()}, "digits": d}
    if not isinstance(exact, sp.Basic):
        return MathResult.unsolved("numeric", "表达式无法解析为数学对象")

    free = sorted(str(s) for s in exact.free_symbols)
    if free:
        bad = MathResult.unsolved(
            "numeric",
            "表达式仍含自由变量 " + ", ".join(free) + '，请用 vars 给出数值（如 {"x": 1.5} 或 {"x": "pi/4"}）',
        )
        bad.set_input(**input_kwargs)
        bad.add_condition("把全部自由变量代入后才是定值：" + ", ".join(free))
        return bad

    if exact.has(sp.zoo, sp.nan) or exact in (sp.oo, -sp.oo):
        bad = MathResult.unsolved("numeric", f"表达式在代入点无有限数值（{A.to_text(exact)}）")
        bad.set_input(**input_kwargs)
        return bad

    method = f"高精度数值求值：SymPy evalf 为主路线（目标 {d} 位有效数字），mpmath 为独立交叉路线"
    r = MathResult.ok("numeric", method=method)
    r.set_input(**input_kwargs)

    # 路线 1：SymPy evalf（主力）
    v_sympy = None
    try:
        cand = sp.N(exact, d)
        if (
            isinstance(cand, sp.Basic)
            and cand.is_number
            and not cand.free_symbols
            and not cand.has(sp.zoo, sp.nan, sp.oo, -sp.oo)
        ):
            v_sympy = cand
    except Exception:  # noqa: BLE001
        v_sympy = None

    # 路线 2：mpmath（独立路线，同时用于误差估计）
    v_mp = None
    try:
        with mpmath.workdps(d + 20):
            v_mp = _eval_mp(exact, [], ())
    except Exception:  # noqa: BLE001
        v_mp = None

    if v_sympy is None and v_mp is None:
        bad = MathResult.unsolved("numeric", "SymPy evalf 与 mpmath 两条路线都无法对该表达式数值化")
        bad.set_input(**input_kwargs)
        bad.add_condition("可能含未实现的特殊函数，或表达式的数值化需要额外参数")
        return bad

    exact_rational = isinstance(exact, sp.Rational)
    primary: sp.Basic
    primary_mp: Any
    if v_sympy is not None:
        primary = v_sympy
    else:
        primary = _sp_float(v_mp, d)
    if v_mp is not None:
        primary_mp = v_mp
    else:
        try:
            primary_mp = mpmath.mpf(str(sp.N(v_sympy, d + 20)))
        except Exception:  # noqa: BLE001
            primary_mp = None

    routes: dict[str, str] = {}
    if v_sympy is not None:
        routes["sympy_evalf"] = str(sp.N(v_sympy, min(d + 5, _MAX_DIGITS)))
    if v_mp is not None:
        routes["mpmath"] = mpmath.nstr(v_mp, min(d + 5, _MAX_DIGITS + 5))

    errors: list[Any] = []
    if v_sympy is not None and v_mp is not None:
        try:
            cross = abs(mpmath.mpf(str(sp.N(v_sympy, d + 20))) - v_mp)
            errors.append(cross)
            r.set_raw(cross_route_deviation=_num(cross))
        except Exception:  # noqa: BLE001
            pass
    if not exact_rational and primary_mp is not None:
        errors.append(abs(primary_mp) * mpmath.mpf(10) ** (-d) + mpmath.mpf(10) ** (-d))
    abs_err = max(errors) if errors else None

    r.set_result(primary)
    r.set_raw(
        exact_form=A.to_text(exact),
        approximate_value=A.to_text(sp.N(primary, d)),
        is_approximate=not exact_rational,
        is_exact=exact_rational,
        digits=d,
        abs_error=_num(abs_err),
        routes=routes,
        suspected_closed_form=_nsimplify_hint(primary, d),
        error_bound_note=(
            "abs_error = max(两条独立高精度路线之差, 按 digits 位报告值时的舍入界)；"
            "输入为精确有理数时无舍入误差"
        ),
    )
    _verify_numeric(
        r,
        ["SymPy evalf 高精度求值", "mpmath 高精度求值（独立交叉路线）"],
        f"数值求值：{d} 位有效数字下的近似值；两路线偏差 "
        + (f"{_num(errors[0])}" if (v_sympy is not None and v_mp is not None and errors) else "不可用")
        + "。" + _NUMERIC_CAVEAT,
    )
    if v_sympy is None:
        r.add_warning(f"SymPy evalf 不可用，已降级为 mpmath 路线（digits={d}）")
    return r


# --------------------------------------------------------------------------------------
# 2. numeric_integrate —— 高精度数值积分（符号积分的独立交叉验算）
# --------------------------------------------------------------------------------------


def _oscillation_period(expr: Any, sym: Any) -> Any:
    """若被积函数含 sin/cos(线性于积分变量)，返回其振荡周期（mpf），否则 None。"""
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Function) and node.func in (sp.sin, sp.cos):
            if len(node.args) != 1:
                continue
            try:
                w = sp.diff(node.args[0], sym)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(w, sp.Basic) and w.is_number and not w.free_symbols and w != 0:
                try:
                    wv = abs(float(sp.N(w, 25)))
                except Exception:  # noqa: BLE001
                    continue
                if wv > 0:
                    return mpmath.mpf(str(sp.N(2 * sp.pi / sp.Abs(w), 25)))
    return None


def _has_trig(expr: Any) -> bool:
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Function) and node.func in (sp.sin, sp.cos):
            return True
    return False


def _mag_at(f: Any, t: Any) -> Any:
    """|f(t)|，定义域外/非实数一律当作 +inf（视作更奇）。"""
    v = _safe(f, t, allow_inf=True)
    if v is None or not _is_real_mp(v):
        return mpmath.inf
    return abs(v)


def _refine_spike(f: Any, lo: Any, hi: Any, iters: int = 200) -> Any:
    """用黄金分割在 [lo,hi] 上最大化 |f|，把疑似奇点位置精化。

    iters 必须足够大：60 次只能把 0.005 的初始区间缩到约 1e-15，
    断点离真实奇点 1e-15 时 tanh-sinh 会退化（实测 1/sqrt(|x-0.3|) 只能到
    1e-13 而非 1e-21）。200 次把区间缩到 ~1e-44，相当于精确命中。
    """
    gr = (mpmath.sqrt(5) - 1) / 2

    def val(t: Any) -> Any:
        v = _safe(f, t, allow_inf=True)
        if v is None or not _is_real_mp(v):
            return mpmath.inf
        return abs(v)

    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    fc, fd = val(c), val(d)
    for _ in range(iters):
        if fc > fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = val(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = val(d)
    return (lo + hi) / 2


def _scan_interior_singularities(f: Any, a: Any, b: Any, n: int = 200) -> list[Any]:
    """粗扫有限区间，找出**确认过的**内点奇点位置（已精化）。

    两处经验教训：
    - 阈值不能太高：原用中位数的 1e4 倍，实测漏掉 1/sqrt(|x-0.3|)——
      200 点网格上最近格点离奇点约 0.003，此时 |f| 只有约 18。现用 100 倍。
    - 命中后必须做「逼近增长率」确认：幂型奇点满足 |f| ∝ |x-t0|^{-p}，
      把距离减半时 |f| 按固定比例增大；光滑尖峰（如 e^{-x²} 的峰）不会。
      否则会把正常的光滑大值误报成奇点。
    """
    flagged: list[Any] = []
    samples: list[tuple[Any, Any]] = []
    for i in range(1, n):
        t = a + (b - a) * mpmath.mpf(i) / n
        v = _safe(f, t, allow_inf=True)
        if v is None or isinstance(v, mpmath.mpc) or not mpmath.isfinite(v):
            flagged.append((t, True))
            continue
        samples.append((t, abs(v)))
    if samples:
        mags = sorted(m for _, m in samples)
        med = mags[len(mags) // 2]
        thresh = max(med, mpmath.mpf(1)) * mpmath.mpf(10) ** 2
        for t, m in samples:
            if m > thresh:
                flagged.append((t, False))
    span = abs(b - a)
    h = span / n
    out: list[Any] = []
    for t, definitely in sorted(flagged):
        lo = max(a, t - h)
        hi = min(b, t + h)
        if definitely:
            loc = t
        else:
            def mag(u: Any) -> Any:
                v = _safe(f, u, allow_inf=True)
                if v is None or not _is_real_mp(v):
                    return mpmath.inf
                return abs(v)

            m0 = mag(t)
            m1 = mag(t + h / 2)
            m2 = mag(t + h / 4)
            m3 = mag(t - h / 2)
            m4 = mag(t - h / 4)
            grew = (
                mpmath.isfinite(m0)
                and mpmath.isfinite(m1)
                and mpmath.isfinite(m2)
                and mpmath.isfinite(m3)
                and mpmath.isfinite(m4)
            )
            ratios: list[Any] = []
            if grew:
                for base, near in ((m1, m2), (m3, m4)):
                    if base > 0 and near > 0:
                        ratios.append(near / base)
            # 幂型奇点：靠近时 |f| 按固定比例增大（>1.15 才算增长，避免光滑峰误报）
            if not (len(ratios) == 2 and all(r > mpmath.mpf("1.15") for r in ratios)):
                continue
            loc = _refine_spike(f, lo, hi)
        # 精化后若能用小分母有理数表示，就吸附回去：
        # 断点与真实奇点差 1e-15 时 tanh-sinh 会退化，精确的 3/10 则能算到 20 位以上。
        # 但**只有当吸附点比精化点更奇（|f| 更大）时**才吸附：用户写 abs(x-0.3) 时
        # 0.3 是 double(0.29999999999999998889…)，真实奇点就在 double 处，硬吸附成
        # 十进制 3/10 反而离真奇点 1.1e-17，实测精度从 1e-20 掉到 1e-13。
        try:
            snap = sp.nsimplify(
                sp.Float(mpmath.nstr(loc, 25)),
                rational=True,
                tolerance=sp.Float("1e-12"),
            )
            if (
                snap.is_Rational
                and abs(snap.q) <= 10**4
                and abs(sp.N(snap - sp.Float(mpmath.nstr(loc, 25)), 25)) < sp.Float("1e-12")
                and _mag_at(f, mpmath.mpf(str(sp.N(snap, 30)))) >= _mag_at(f, loc)
            ):
                loc = snap
        except Exception:  # noqa: BLE001
            pass
        out.append(loc)
    result: list[Any] = []
    for loc in out:
        try:
            # 保留工作精度：把 loc 截断到 30 位会让分点离真实奇点约 1e-30，
            # tanh-sinh 于是丢掉宽度 ~1e-30 的奇异薄片（误差 ~2*sqrt(1e-30)），
            # 实测 1/sqrt(|x-0.3|) 只能到 13 位而不是 20 位。
            lv = loc if isinstance(loc, mpmath.mpf) else mpmath.mpf(str(sp.N(loc, mpmath.mp.dps + 5)))
        except Exception:  # noqa: BLE001
            continue
        if not mpmath.isfinite(lv) or lv <= a or lv >= b:
            continue
        if result and abs(lv - result[-1]) < span / (10 * n):
            continue
        result.append(lv)
    return result


def _analytic_breakpoints(expr: Any, sym: Any, a: Any, b: Any) -> list[Any]:
    """解析求出 Abs/Max/Min 造成的折点（落在 (a,b) 内）。

    分段求积对这些折点是必需的：不分段时 ∫_{-1}^{1}|x|dx 只能算到
    0.999993725049396（误差 6e-6，因为 tanh-sinh 在尖点处收敛慢）。
    """
    eqs: list[Any] = []
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Abs) and len(node.args) == 1:
            eqs.append(sp.Eq(node.args[0], 0))
        elif isinstance(node, (sp.Max, sp.Min)) and len(node.args) == 2:
            eqs.append(sp.Eq(node.args[0], node.args[1]))
    out: list[Any] = []
    for eq in eqs:
        try:
            sols = sp.solve(eq, sym)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(sols, list):
            sols = [sols]
        for s in sols:
            try:
                if not isinstance(s, sp.Basic) or s.free_symbols or not s.is_real:
                    continue
                fv = float(sp.N(s, 30))
            except Exception:  # noqa: BLE001
                continue
            if a < fv < b and not any(abs(fv - float(sp.N(o, 30))) < 1e-14 for o in out):
                out.append(s)
    return out


def _mpf_breakpoints(points: list[Any], dps: int) -> list[Any]:
    """把断点转成按工作精度精确表示的 mpf，排序去重。

    必须用**十进制字符串**转换：``mpmath.mpf(0.3)`` 得到的是
    0.29999999999999998889776975374843…（二进制 double），奇点就不落在分点上，
    tanh-sinh 会剧烈退化——实测 quad(f,[0,0.3,1]) 只有约 9 位有效数字，
    而 quad(f,[0,mpf('0.3'),1]) 有 21 位。
    """
    out: list[Any] = []
    for p in points:
        try:
            q = p if isinstance(p, mpmath.mpf) else mpmath.mpf(str(sp.N(p, dps + 5)))
        except Exception:  # noqa: BLE001
            continue
        if not mpmath.isfinite(q):
            continue
        out.append(q)
    out.sort()
    # 去重必须**排序之后**做，且用容差而不是严格相等：扫描出的精化奇点与解析折点
    # 往往只差 1e-17~1e-40（都是同一个点），严格相等判定会把它们都留下，
    # 排完序就变成一个零宽度分片 → quad 的每个结点都恰好落在奇点上 →
    # ZeroDivisionError（实测 1/sqrt(|x-0.3|) 的分段路线就是这样整个失败退化的）。
    deduped: list[Any] = []
    tol_rel = mpmath.mpf(10) ** (-(max(dps, 20) - 2))
    for q in out:
        if deduped and abs(q - deduped[-1]) <= tol_rel * max(mpmath.mpf(1), abs(q)):
            continue
        deduped.append(q)
    return deduped


def _truncation_guard(f: Any, a: Any, b: Any, dps: int) -> dict[str, Any] | None:
    """截断收敛检验：在 R = 2,4,…,256 的截断区间上求 I(R)，判断尾巴是否趋于 0。

    用途：无穷端点的振荡积分上，直接 quad 会给出**自信的错误值**——
    实测 ∫_0^∞ sin(x²)dx 直接 quad = -5.6e36（真值 0.6266570686577501），
    ∫_0^∞ sin(x)dx 用 quadosc = 1.0（该积分发散）。本检验给出诚实的收敛判断。
    """
    left_inf = bool(mpmath.isinf(a) and a < 0)
    right_inf = bool(mpmath.isinf(b))
    if not (left_inf or right_inf):
        return None
    vals: list[tuple[Any, Any]] = []
    for k in range(1, 9):
        R = mpmath.mpf(2) ** k
        lo = -R if left_inf else a
        hi = R if right_inf else b
        try:
            v, e = mpmath.quad(f, [lo, hi], error=True)
        except Exception:  # noqa: BLE001
            return None
        if not _is_real_mp(v):
            return None
        vals.append((R, v))
    diffs = [abs(vals[i + 1][1] - vals[i][1]) for i in range(len(vals) - 1)]
    last = diffs[-1]
    scale = max(mpmath.mpf(1), abs(vals[-1][1]))
    finite = all(mpmath.isfinite(v) for _, v in vals) and all(mpmath.isfinite(dd) for dd in diffs)
    # 判据用「相对于最早一次跳变」而不是绝对阈值：条件收敛的振荡积分尾项只有 O(1/R)，
    # 例如 ∫_0^∞ sin(x)/x dx 在 R=256 处相邻两次仍差 5e-3（绝对阈值 1e-3·π/2 会误判不收敛）；
    # 但它的跳变确实在按 R 成比例衰减（早期跳变 0.23），而发散/垃圾序列的跳变不会衰减。
    converged = bool(
        finite
        and diffs
        and last <= diffs[0] / 10
        and last <= mpmath.mpf("0.1") * scale
    )
    return {
        "converged": converged,
        "value": vals[-1][1],
        "error": max(last, scale * mpmath.mpf(10) ** (-dps)),
        "last_gap": last,
        "note": (
            f"截断序列 I(R)（R=2,4,…,256）最后两次相差 {mpmath.nstr(last, 6)}"
            + ("，已收敛" if converged else "，**没有收敛迹象**")
        ),
    }


def _tail_window_test(f: Any, a: Any, b: Any) -> dict[str, Any] | None:
    """尾部窗口检验：窗口积分 T(N) = ∫_N^{2N} f（左无穷取 ∫_{-2N}^{-N}）。

    收敛的必要条件是 T(N) → 0。这条检验专门抓「单调发散」的积分：
    mpmath.quad 对 ∫_1^∞ dx/x 会**自信地**给出 102.02044304415191546（自报误差仅 0.1，
    相对误差 1e-3，不足以触发 partial），而 T(N) = ln2 恒定不变，立刻暴露发散。
    振荡积分不归它管（由 _truncation_guard 负责）。
    """
    left_inf = bool(mpmath.isinf(a) and a < 0)
    right_inf = bool(mpmath.isinf(b))
    if not (left_inf or right_inf):
        return None
    windows: list[tuple[Any, Any]] = []
    try:
        for k in range(1, 11):
            N = mpmath.mpf(2) ** k
            tot = mpmath.mpf(0)
            if right_inf:
                v, _ = mpmath.quad(f, [N, 2 * N], error=True)
                if not _is_real_mp(v):
                    return None
                tot += v
            if left_inf:
                v, _ = mpmath.quad(f, [-2 * N, -N], error=True)
                if not _is_real_mp(v):
                    return None
                tot += v
            windows.append((N, abs(tot)))
    except Exception:  # noqa: BLE001
        return None
    if len(windows) < 2:
        return None
    first = windows[0][1]
    prev = windows[-2][1]
    last = windows[-1][1]
    if not (mpmath.isfinite(first) and mpmath.isfinite(last)):
        return None
    ratio = (last / prev) if prev > 0 else mpmath.inf
    # 「不衰减」的强证据：窗口值几乎没变小，或最后一个窗口仍占总量的显著份额。
    decaying = bool(last <= mpmath.mpf("0.5") * first)
    return {
        "decaying": decaying,
        "first_window": mpmath.nstr(first, 6),
        "last_window": mpmath.nstr(last, 6),
        "ratio": _num(ratio),
        "note": (
            f"窗口积分 T(N)=∫_N^(2N)f 从 {mpmath.nstr(first, 6)}（N=2）到 {mpmath.nstr(last, 6)}（N=1024），"
            f"相邻比 {mpmath.nstr(ratio, 6)}"
            + ("，尾部在衰减" if decaying else "，**尾部几乎不衰减**：该积分很可能发散（或收敛极慢）")
        ),
    }


def _singularity_divergence_test(f: Any, p: Any, a: Any, b: Any) -> dict[str, Any] | None:
    """在**内点奇点** p 处逐段检验单侧反常积分是否收敛（不可积奇点）。

    做法：取该侧可用步长 Δ = min(p-a, b-p)/4，把下界从 p+Δ·10^-1 一路推到 p+Δ·10^-10，
    得到 J_k = ∫_{p+Δ·10^-k}^{p+Δ} f（奇点在端点，tanh-sinh 对端点奇性很准）。
    判据用**柯西准则**：J_k 收敛 ⟺ 增量 ΔJ_k 必须趋于 0。
      - 可积奇点（1/√|x-p|、x^{-0.9}、ln|x-p|）：增量按几何级数衰减（比值 ≈0.3 / 0.79 / 0.1）；
      - 不可积奇点（1/(x-p)、1/(x-p)²）：增量不衰减（比值 =1.0 / 10）。
    实测：∫_{-1}^{1} dx/x² 的分段求积会「自信地」报出 5.05417063636375e+39（自报误差很小），
    而这里能识破（1/x² 的增量每步放大 10 倍），把它降级为 unsolved。
    """
    if not (mpmath.isfinite(a) and mpmath.isfinite(b) and mpmath.isfinite(p)):
        return None
    delta = min(p - a, b - p) / 4
    if delta <= 0:
        return None
    out: dict[str, Any] = {"point": mpmath.nstr(p, 12), "sides": {}, "divergent": False, "note": ""}
    hit: list[str] = []
    for sign_, name in ((1, "右"), (-1, "左")):
        vals: list[Any] = []
        for k in range(1, 11):
            lo = p + sign_ * delta * mpmath.mpf(10) ** (-k)
            hi = p + sign_ * delta
            try:
                v = mpmath.quad(f, [lo, hi])
            except Exception:  # noqa: BLE001
                continue
            if v is None or not mpmath.isfinite(v):
                continue
            vals.append(v)
        if len(vals) < 5:
            continue
        incs = [abs(vals[i] - vals[i - 1]) for i in range(1, len(vals))]
        out["sides"][name] = {
            "partial_integrals": [mpmath.nstr(v, 6) for v in vals],
            "increments": [mpmath.nstr(i, 6) for i in incs],
        }
        if incs[-2] == 0:
            continue
        decay = incs[-1] / incs[-2]
        big_tail = incs[-1] > mpmath.mpf("0.05") * max(mpmath.mpf(1), abs(vals[-1]))
        out["sides"][name]["increment_decay"] = _num(decay)
        if decay >= mpmath.mpf("0.9") or big_tail:
            out["divergent"] = True
            hit.append(name)
    if out["divergent"]:
        out["note"] = (
            f"内点奇点 x={out['point']} 的{'、'.join(hit)}侧单侧积分不满足柯西准则"
            "（增量不趋于 0）：该反常积分发散，任何有限「数值结果」都是假的"
        )
    return out


def _phase_linear_route(
    f: Any, parsed: Any, sym: Any, a: Any, b: Any, dps: int
) -> tuple[Any, Any, str] | None:
    """非线性相位的振荡积分：用 t = 相位(x) 替换把相位线性化后交给 quadosc。

    例如 ∫_0^∞ sin(x²)dx：t = x², x = √t, dx = dt/(2√t)，得 ∫_0^∞ sin(t)/(2√t) dt，
    相位变成线性的（周期 2π），quadosc 可用。**注意必须把第一个周期用直接求积
    算掉**：g(t)=sin(t)/(2√t) 在 t=0 处有 u^{-1/2} 型导数奇性，整段交给 quadosc
    只有约 1e-8 精度；拆成「[0,2π] 直接求积 + [2π,∞) quadosc」实测能到 25 位。
    """
    if not (mpmath.isfinite(a) and mpmath.isinf(b)) or not _has_trig(parsed):
        return None
    t = sp.Symbol("_phase_t", positive=True)
    for node in sp.preorder_traversal(parsed):
        if not (isinstance(node, sp.Function) and node.func in (sp.sin, sp.cos)):
            continue
        if len(node.args) != 1:
            continue
        phase = node.args[0]
        if phase.free_symbols != {sym}:
            continue
        try:
            dphase = sp.diff(phase, sym)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(dphase, sp.Basic) or dphase.is_number or dphase.free_symbols != {sym}:
            continue
        try:
            sols = sp.solve(sp.Eq(phase, t), sym)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(sols, list):
            sols = [sols]
        x_test = a + max(mpmath.mpf(1), abs(a))
        try:
            t_test = mpmath.mpf(str(sp.N(phase.subs(sym, sp.Float(mpmath.nstr(x_test, 25))), 25)))
        except Exception:  # noqa: BLE001
            continue
        best_root = None
        best_gap = None
        for root in sols:
            if not isinstance(root, sp.Basic) or root.free_symbols - {t}:
                continue
            try:
                val = root.subs(t, sp.Float(mpmath.nstr(t_test, 25)))
                if not val.is_real:
                    continue
                xv = mpmath.mpf(str(sp.N(val, 25)))
            except Exception:  # noqa: BLE001
                continue
            gap = abs(xv - x_test)
            if best_gap is None or gap < best_gap:
                best_gap, best_root = gap, root
        if best_root is None or best_gap is None or best_gap > max(mpmath.mpf(1), abs(x_test)):
            continue
        try:
            dxdt = sp.diff(best_root, t)
            g_expr = sp.simplify(parsed.subs(sym, best_root) * dxdt)
            t_lo = mpmath.mpf(str(sp.N(phase.subs(sym, sp.Float(mpmath.nstr(a, 25))), 25)))
        except Exception:  # noqa: BLE001
            continue
        if not (mpmath.isfinite(t_lo) and mpmath.isfinite(best_gap)):
            continue
        try:
            g_fun = _lambdify_mp([t], g_expr)
            two_pi = 2 * mpmath.pi
            head, e_head = mpmath.quad(g_fun, [t_lo, t_lo + two_pi], error=True)
            tail = mpmath.quadosc(g_fun, [t_lo + two_pi, mpmath.inf], period=two_pi)
            whole = mpmath.quadosc(g_fun, [t_lo, mpmath.inf], period=two_pi)
        except Exception:  # noqa: BLE001
            continue
        v1 = head + tail
        err = max(abs(v1 - whole), abs(e_head))
        return (
            v1,
            err,
            f"相位线性化替换 t = {A.to_text(phase)}（x = {A.to_text(best_root)}）+ quadosc(period=2π)",
        )
    return None


def _tail_integrand(f: Any, N: Any, sign: int) -> tuple[Any, Any]:
    """∫_N^∞ f（sign=+1）或 ∫_{-∞}^{-N} f（sign=-1），用 x = sign·(N + t/(1-t)) 替换。

    注意：sign=-1 时 x 必须从 -N 出发走向 -∞（即 x = -N - t/(1-t)），
    写成 x = N - t/(1-t) 会错误地覆盖 (-∞, N]，实测会把结果放大约一倍。
    """

    def g(t: Any) -> Any:
        x = sign * (N + t / (1 - t))
        return f(x) / (1 - t) ** 2

    return mpmath.quad(g, [0, 1], error=True)


def _substitution_route(f: Any, a: Any, b: Any) -> tuple[Any, Any, str] | None:
    """独立路线：无穷区间 → 有限截断 + 变量替换处理尾部。"""
    left_inf = bool(mpmath.isinf(a) and a < 0)
    right_inf = bool(mpmath.isinf(b))
    if not left_inf and not right_inf:
        return None
    lo = None if left_inf else a
    hi = None if right_inf else b
    ref = None
    for probe in (lo, hi):
        if probe is not None and mpmath.isfinite(probe):
            v = _safe(f, probe)
            if v is not None and _is_real_mp(v):
                ref = abs(v)
    if ref is None:
        ref = mpmath.mpf(1)
    target = mpmath.mpf(10) ** (-3) * max(ref, mpmath.mpf(1))
    N = mpmath.mpf(2)
    ok = False
    for _ in range(24):
        good = True
        for probe in ((-N if left_inf else None), (N if right_inf else None)):
            if probe is None:
                continue
            v = _safe(f, probe)
            if v is None or not _is_real_mp(v) or abs(v) > target:
                good = False
                break
        if good:
            ok = True
            break
        N *= 2
    if not ok:
        return None
    total = mpmath.mpf(0)
    err = mpmath.mpf(0)
    mid_lo = -N if left_inf else lo
    mid_hi = N if right_inf else hi
    try:
        v, e = mpmath.quad(f, [mid_lo, mid_hi], error=True)
        total += v
        err += abs(e)
        if right_inf:
            v, e = _tail_integrand(f, N, 1)
            total += v
            err += abs(e)
        if left_inf:
            v, e = _tail_integrand(f, N, -1)
            total += v
            err += abs(e)
    except Exception:  # noqa: BLE001
        return None
    return total, err, f"截断 + 变量替换（x = ±N ∓ t/(1-t)，截断点 N={mpmath.nstr(N, 6)}）"


@op("numeric_integrate", "quad", "math_numeric_integrate")
def numeric_integrate(
    expr: str,
    var: str = "x",
    lower: Any = None,
    upper: Any = None,
    digits: int = 15,
    points: Any = None,
) -> MathResult:
    """用 mpmath 做高精度数值积分，支持无穷端点与可积奇点。

    这是给符号积分做**独立交叉验证**的关键算子：返回 (值, 误差界)，
    多条数值路线（直接 quad / quadosc / 截断替换）互相验算，偏差写进 abs_error。
    """
    d = _check_digits(digits)
    if lower is None or upper is None:
        raise ValueError("numeric_integrate 需要 lower 与 upper（积分上下限，可为 oo / -oo / 无穷）")

    names = _name_list(var) or ["x"]
    parsed = _parse_expr(expr, names)
    sym = _symbols_for([parsed], names)[0]
    lo_sp = C.parse_point(lower, [sym])
    hi_sp = C.parse_point(upper, [sym])
    user_pts: list[Any] = []
    for p in _as_seq(points):
        user_pts.append(C.parse_point(p, [sym]))

    r = MathResult.ok(
        "numeric_integrate",
        method=f"mpmath 高精度数值积分（tanh-sinh / 高斯-勒让德求积，目标 {d} 位有效数字）",
    )
    r.set_input(
        expr=str(expr),
        var=str(sym),
        lower=str(lower),
        upper=str(upper),
        digits=d,
        points=[str(p) for p in user_pts] or None,
    )

    with mpmath.workdps(d + 20):
        try:
            a = _mpf(lo_sp, d)
            b = _mpf(hi_sp, d)
        except ValueError as exc:
            raise ValueError(f"积分限无法数值化：{exc}") from None
        if mpmath.isnan(a) or mpmath.isnan(b):
            raise ValueError("积分限不能是 nan")
        if a == b:
            r.set_result(sp.Integer(0))
            r.set_raw(digits=d, abs_error=0.0, is_approximate=False, note="上下限相同，积分为 0（精确）")
            _verify_numeric(r, ["积分区间退化"], "区间退化为一点，积分值恒为 0（这是精确结论）")
            return r
        if not mpmath.isfinite(a) and not mpmath.isfinite(b) and (a == b or mpmath.sign(a) == mpmath.sign(b)):
            raise ValueError("积分限不能是同号的无穷")

        f = _lambdify_mp([sym], parsed)

        tries: list[dict[str, Any]] = []
        fails: list[str] = []

        def _add(value: Any, error: Any, method_note: str, note: str, kind: str = "other") -> None:
            if value is None:
                return
            # 非有限值（±∞/NaN）绝不能当成结果路线：区间内的奇点会让 quad 取到 x=奇点
            # 上的函数值（tanh-sinh 的节点含区间中点，如 log|x| 在 [-1,1] 上会命中 x=0），
            # 于是整条路线变成 ±∞，而「自报误差」也是 ∞ 或 NaN，_score 排序会把它蒙过去。
            if not (_is_real_mp(value) and mpmath.isfinite(value)):
                fails.append(f"{method_note} 给出非有限值（{mpmath.nstr(value, 6)}），已弃用该路线")
                return
            tries.append(
                {"value": value, "error": error, "method": method_note, "note": note, "kind": kind, "reliable": True}
            )

        infinite = bool(mpmath.isinf(a) or mpmath.isinf(b))
        period = _oscillation_period(parsed, sym)

        # 路线 1：直接 quad
        try:
            v, e = mpmath.quad(f, [a, b], error=True)
            _add(v, e, "mpmath.quad([a,b]) 直接求积（含无穷端点的区间变换）", "对整个区间一次求积", kind="direct")
        except Exception as exc:  # noqa: BLE001
            fails.append(f"quad 直接求积失败: {exc}")

        # 路线 2：分段求积（内点奇点 / Abs-Max-Min 折点 / 用户给定断点）
        suspicious: list[Any] = []
        kinks: list[Any] = []
        if mpmath.isfinite(a) and mpmath.isfinite(b):
            try:
                suspicious = _scan_interior_singularities(f, a, b)
            except Exception:  # noqa: BLE001
                suspicious = []
            try:
                kinks = _analytic_breakpoints(parsed, sym, a, b)
            except Exception:  # noqa: BLE001
                kinks = []
        pts_all: list[Any] = []
        for p in list(user_pts) + suspicious + kinks:
            if mpmath.isfinite(p) and a < p < b:
                pts_all.append(p)
        segs = _mpf_breakpoints([a] + pts_all + [b], d)

        # 内点奇点的敛散性检验（不可积奇点，如 ∫_{-1}^{1} dx/x²）：
        # 分段求积会把 1/x² 这种非可积奇点也算成一个很大的有限数（实测 5.05e39）。
        divergent_pts: list[Any] = []
        sing_checks: list[dict[str, Any]] = []
        if suspicious:
            for p in suspicious:
                try:
                    chk = _singularity_divergence_test(f, p, a, b)
                except Exception:  # noqa: BLE001
                    chk = None
                if chk is None:
                    continue
                sing_checks.append(chk)
                if chk["divergent"]:
                    divergent_pts.append(p)
        if len(segs) > 2:
            try:
                v, e = mpmath.quad(f, segs, error=True)
                _add(
                    v,
                    e,
                    "mpmath.quad 分段求积（在奇点/折点处切开后逐段 tanh-sinh）",
                    "分段点：" + ", ".join(mpmath.nstr(p, 8) for p in segs[1:-1]),
                    kind="segmented",
                )
            except Exception as exc:  # noqa: BLE001
                fails.append(f"quad(分段) 失败: {exc}")

        # 路线 3：振荡半无穷积分 → quadosc
        if infinite and period is not None:
            try:
                v = mpmath.quadosc(f, [a, b], period=period)
                _add(
                    v,
                    None,
                    f"mpmath.quadosc（振荡积分专用，period={mpmath.nstr(period, 12)}）",
                    "利用振荡周期做级数加速；仅适用于该类振荡积分",
                    kind="quadosc",
                )
            except Exception as exc:  # noqa: BLE001
                fails.append(f"quadosc(period=...) 失败: {exc}")

        # 路线 4：非线性相位的振荡积分 → 相位线性化 + quadosc
        guard: dict[str, Any] | None = None
        if infinite and period is None and _has_trig(parsed):
            try:
                phase_route = _phase_linear_route(f, parsed, sym, a, b, d)
            except Exception as exc:  # noqa: BLE001
                phase_route = None
                fails.append(f"相位线性化路线失败: {exc}")
            if phase_route is not None:
                _add(phase_route[0], phase_route[1], phase_route[2], "把非线性相位换成线性相位后加速", kind="phase")

        # 路线 5：无穷区间 → 截断 + 变量替换（非振荡时才是独立可信路线）
        if infinite and period is None:
            try:
                sub = _substitution_route(f, a, b)
                if sub is not None:
                    _add(sub[0], sub[1], "mpmath.quad 分段：中间段 + " + sub[2], "无穷尾部的独立处理路线", kind="substitution")
            except Exception as exc:  # noqa: BLE001
                fails.append(f"截断+替换路线失败: {exc}")

        # 路线 6：有限区间再加一条不同求积法（高斯-勒让德）
        if mpmath.isfinite(a) and mpmath.isfinite(b):
            try:
                v, e = mpmath.quad(f, [a, b], method="gauss-legendre", error=True)
                _add(v, e, "mpmath.quad(method='gauss-legendre') 高斯-勒让德求积", "与主路线不同的求积法则", kind="gauss")
            except Exception as exc:  # noqa: BLE001
                fails.append(f"gauss-legendre 路线失败: {exc}")

        # 路线 7：振荡积分的截断收敛检验（≤R 的截断值是否稳定下来）
        oscillatory = bool(infinite and _has_trig(parsed))
        guard = None
        if oscillatory:
            try:
                guard = _truncation_guard(f, a, b, d)
            except Exception as exc:  # noqa: BLE001
                guard = None
                fails.append(f"截断收敛检验失败: {exc}")
            if guard is not None and guard["converged"]:
                _add(
                    guard["value"],
                    guard["error"],
                    "截断收敛检验 I(R), R=2,4,…,256（截断值已稳定）",
                    guard["note"],
                    kind="truncation",
                )
            elif guard is not None:
                # 截断序列不收敛：普通 quad / quadosc 在振荡积分上会「自信地给出错误值」，
                # 它们的自报误差不可信，必须作废（相位线性化路线不受此限制）。
                for t in tries:
                    if t["kind"] in ("direct", "quadosc"):
                        t["reliable"] = False
                        try:
                            t["error"] = max(abs(t["value"]), mpmath.mpf(1))
                        except Exception:  # noqa: BLE001
                            t["error"] = mpmath.mpf(1)

        # 路线 8：非振荡无穷积分的「尾部窗口」发散检验。
        # 单调发散的积分（如 ∫_1^∞ dx/x）会让 quad 报出一个小误差，必须靠它识破。
        window = None
        if infinite and not oscillatory:
            try:
                window = _tail_window_test(f, a, b)
            except Exception as exc:  # noqa: BLE001
                window = None
                fails.append(f"尾部窗口检验失败: {exc}")
            if window is not None and not window["decaying"]:
                for t in tries:
                    t["reliable"] = False
                    try:
                        t["error"] = max(abs(t["value"]), mpmath.mpf(1))
                    except Exception:  # noqa: BLE001
                        t["error"] = mpmath.mpf(1)

        # 路线 9：内点不可积奇点 → 全部路线作废（分段求积对 1/x² 会给出 5.05e39 的假结果）
        if divergent_pts:
            for t in tries:
                t["reliable"] = False
                try:
                    t["error"] = max(abs(t["value"]), mpmath.mpf(1))
                except Exception:  # noqa: BLE001
                    t["error"] = mpmath.mpf(1)

        if not tries:
            bad = MathResult.unsolved("numeric_integrate", "所有数值积分路线都失败：" + "；".join(fails[:4]))
            bad.set_input(expr=str(expr), var=str(sym), lower=str(lower), upper=str(upper), digits=d)
            bad.add_condition("常见原因：被积函数在区间内不可积/发散、端点奇异且无法变换、表达式无法数值化")
            return bad

        def _score(t: dict[str, Any]) -> Any:
            e = t["error"]
            if e is None or not mpmath.isfinite(e):
                return mpmath.mpf(10) ** (-(d - 3)) * max(mpmath.mpf(1), abs(t["value"]))
            return abs(e)

        best = min(tries, key=_score)
        ref = best["value"]
        best_score = _score(best)
        # 只有「自称精度与最优路线可比」的路线才参与偏差统计；
        # 否则一条已知不可靠的路线（如振荡积分的直接 quad，误差估计 1.0）会把
        # abs_error 抬到结果量级，把正确的数值结论误判为不可信。
        threshold = max(best_score * 100, mpmath.mpf(10) ** (-(d - 2)) * max(mpmath.mpf(1), abs(ref)))
        comparable = [t for t in tries if t is not best and _score(t) <= threshold]
        outliers = [t for t in tries if t is not best and _score(t) > threshold]
        spreads = [abs(t["value"] - ref) for t in comparable]
        spread = max(spreads) if spreads else mpmath.mpf(0)
        errs = [abs(best["error"])] if best["error"] is not None else []
        errs.append(spread)
        abs_err = max(errs) if errs else abs(ref) * mpmath.mpf(10) ** (-d)
        if abs_err == 0 and best["error"] is None:
            abs_err = abs(ref) * mpmath.mpf(10) ** (-d)

        if not best.get("reliable", True):
            # 唯一「胜出」的路线本身已被判定不可信（截断序列不收敛的振荡积分 /
            # 尾部不衰减的发散积分）：宁可 unsolved，也不能把数值垃圾当成结果报出去。
            reasons: list[str] = []
            if guard is not None and not guard["converged"]:
                reasons.append("振荡积分的截断序列不收敛：" + str(guard["note"]))
            if window is not None and not window["decaying"]:
                reasons.append("尾部窗口检验未通过：" + str(window["note"]))
            if divergent_pts:
                reasons.extend(str(c["note"]) for c in sing_checks if c["divergent"])
            _scope = "无穷区间上的数值积分" if infinite else "区间内存在不可积奇点时的数值积分"
            bad = MathResult.unsolved(
                "numeric_integrate",
                _scope
                + "不可信"
                + ("（" + "；".join(reasons) + "）" if reasons else "（所有数值路线都被判定不可靠）"),
            )
            bad.set_input(expr=str(expr), var=str(sym), lower=str(lower), upper=str(upper), digits=d, points=[str(p) for p in user_pts] or None)
            bad.set_raw(
                digits=d,
                routes=[{"method": t["method"], "value": mpmath.nstr(t["value"], min(d + 3, 30)), "error": _num(t["error"])} for t in tries],
                truncation_check=guard,
                tail_window_check=window,
                singularity_checks=sing_checks or None,
            )
            if divergent_pts:
                bad.add_condition(
                    "反常积分的敛散性必须单独判定：把不可积奇点两侧分别写成极限 ∫_p^b = lim_{ε→0+}∫_{p+ε}^b，"
                    "再判断该极限是否存在（如 ∫_{-1}^{1} dx/x² 两侧都发散）"
                )
                bad.add_warning(
                    "分段求积对**不可积**内点奇点会「自信地」给出一个很大的有限值：实测 ∫_{-1}^{1} dx/x² "
                    "= 5.05417063636375e+39，但该积分发散。数值结果在这里没有意义。"
                )
            elif oscillatory:
                bad.add_condition("发散或收敛极慢的振荡积分没有数值值；请用收敛性判别法（如 Dirichlet/Abel）判断敛散性")
                bad.add_warning(
                    "直接 quad / quadosc 在振荡积分上可能「自信地给出错误值」：实测 ∫_0^∞ sin(x²)dx 直接 quad = -5.6e36"
                    "（真值 0.6266570686577501），∫_0^∞ sin(x)dx 用 quadosc = 1.0（该积分发散）"
                )
            else:
                bad.add_condition("尾部窗口积分 T(N)=∫_N^(2N)f 不随 N 增大而衰减：被积函数在无穷远处衰减不够快（如 1/x）")
                bad.add_warning(
                    "mpmath.quad 对单调发散的无穷积分会「自信地给出错误值」：实测 ∫_1^∞ dx/x 得到 102.02044304415191546，"
                    "自报误差仅 0.1（相对误差 1e-3），但该积分发散"
                )
            return bad
        rel_err = float(abs_err / max(mpmath.mpf(1), abs(ref))) if _is_real_mp(ref) else float(abs_err)

    r.set_result(_sp_float(ref, d))
    r.set_raw(
        digits=d,
        abs_error=_num(abs_err),
        reliable_digits=_reliable_digits(ref, abs_err, d),
        relative_error=_num(rel_err) if math.isfinite(rel_err) else str(rel_err),
        is_approximate=True,
        routes=[
            {
                "method": t["method"],
                "value": mpmath.nstr(t["value"], min(d + 3, 30)),
                "error": _num(t["error"]),
                "kind": t["kind"],
                "reliable": t.get("reliable", True),
            }
            for t in tries
        ],
        chosen_route=best["method"],
        chosen_route_note=best["note"],
        failed_routes=fails[:4] or None,
        route_spread=_num(spread),
        discarded_routes=[
            {"method": t["method"], "value": mpmath.nstr(t["value"], min(d + 3, 30)), "error": _num(t["error"])}
            for t in outliers
        ]
        or None,
        truncation_check=guard,
        tail_window_check=window,
        suspected_closed_form=_nsimplify_hint(ref, d),
        abs_error_note="abs_error = max(所选路线自身的求积误差估计, 与所选路线精度可比的其他路线之间的最大偏差)；误差估计明显偏大的路线只作弃用记录，不计入误差界",
    )

    methods = [t["method"] for t in tries]
    note = (
        f"数值积分：{len(tries)} 条路线（"
        + "；".join(t["method"].split("（")[0] for t in tries)
        + f"）；采用「{best['method']}」，误差界 {_num(abs_err)}"
    )
    if len(tries) > 1:
        note += f"，各路线最大偏差 {_num(spread)}"
    _verify_numeric(
        r,
        methods,
        note + "。不同路线一致只说明数值上自洽，**不构成解析证明**；请与符号积分(integrate)交叉确认。",
    )

    r.add_condition("数值积分给出的是定积分近似值；被积函数的原函数是否存在、积分是否收敛需另行判定")
    if period is not None:
        r.add_warning(
            "被积函数在区间上振荡：直接 quad 对该类积分不可靠（实测会给出错误值并报误差 1.0），"
            "已改用 quadosc（依赖振荡周期，周期检测失败时不适用）"
        )
    if best["kind"] == "phase":
        r.add_warning(
            "被积函数的相位非线性（如 sin(x²)）：已用相位替换 t = 相位(x) 把它化成线性相位后交给 quadosc；"
            "这类积分的数值精度低于常规积分，误差界已如实给出"
        )
    if guard is not None:
        r.add_condition("振荡积分的截断收敛检验：" + guard["note"])
    if suspicious:
        r.add_warning(
            "区间内检出内点奇点（已在该点切开后分段求积）："
            + ", ".join(mpmath.nstr(p, 10) for p in suspicious)
        )
    if kinks:
        r.add_warning(
            "被积函数含 Abs/Max/Min 的折点（已在折点处分段求积）："
            + ", ".join(mpmath.nstr(mpmath.mpf(str(sp.N(k, 25))), 10) for k in kinks)
        )
    if outliers and period is None:
        worst = max(outliers, key=lambda t: abs(t["value"] - ref))
        r.add_warning(
            "另一条数值路线给出明显不同的结果，已弃用："
            f"{worst['method']} → {mpmath.nstr(worst['value'], 12)}"
            f"（与所选结果相差 {mpmath.nstr(abs(worst['value'] - ref), 6)}，"
            f"其自身误差估计 {mpmath.nstr(worst['error'], 6) if worst['error'] is not None else '未知'}）"
        )
    if infinite:
        r.add_condition("区间含无穷端点：数值结果反映的是收敛后的极限值，发散时数值方法只会给出无意义的大数")
    if rel_err > 1e-2:
        r.partial(f"数值积分误差界 {_num(abs_err)} 与结果量级相当（相对误差 {_num(rel_err)}），结果不可信")
    elif rel_err > 1e-8:
        r.add_warning(f"数值积分误差界偏大（相对误差 {_num(rel_err)}），可信度下降")
    return r


# --------------------------------------------------------------------------------------
# 3. numeric_solve —— 数值求根 / 解方程组
# --------------------------------------------------------------------------------------


def _guess_mpf(guesses: Any, name: str, default: Any = 1) -> Any:
    if isinstance(guesses, dict) and name in guesses:
        return _mpf(_parse_expr(guesses[name], None))
    if isinstance(guesses, (list, tuple)) and guesses:
        return _mpf(guesses[0])
    return mpmath.mpf(default)


@op("numeric_solve", "find_root", "math_numeric_solve")
def numeric_solve(
    equations: Any,
    vars: Any,
    guesses: dict | None = None,
    digits: int = 15,
) -> MathResult:
    """数值求根 / 解方程组，并把**残差**写进 verification。

    残差超容差会 ``partial``/``unsolved``，绝不把未收敛的解当成功。
    """
    d = _check_digits(digits)
    names = _name_list(vars)
    if not names:
        raise ValueError("numeric_solve 需要 vars 指定未知量，如 vars='x' 或 vars=['x','y']")
    eq_texts = _as_seq(equations) if not isinstance(equations, str) or ";" in equations or "\n" in equations else [equations]
    eq_texts = [t for t in eq_texts if str(t).strip()]
    if not eq_texts:
        raise ValueError("numeric_solve 需要至少一个方程")
    if len(eq_texts) > len(names):
        raise ValueError(f"方程数({len(eq_texts)})多于未知量数({len(names)})，请补充 vars")

    exprs = [_equation_expr(t, names) for t in eq_texts]
    syms = _symbols_for(exprs, names)
    missing = [n for n, s in zip(names, syms) if not any(_symbol_for(e, n) is not None for e in exprs)]
    if missing and len(missing) == len(names):
        raise ValueError("方程里没有出现任何指定未知量 " + ", ".join(names) + "，请检查 vars 与方程")

    given_guess = guesses is not None
    r = MathResult.ok("numeric_solve", method=f"数值求根：mpmath.findroot（高精度牛顿法，目标 {d} 位有效数字）")
    r.set_input(equations=[str(t) for t in eq_texts], vars=list(names), guesses=({str(k): str(v) for k, v in guesses.items()} if given_guess else None), digits=d)
    if missing:
        r.add_warning("以下未知量未出现在方程中：" + ", ".join(missing) + "（方程可能是欠定的）")

    tol = mpmath.mpf(10) ** (-(d - 4))

    # ---------------- 单变量单方程 ----------------
    if len(exprs) == 1 and len(names) == 1:
        sym = syms[0]
        fmp = _lambdify_mp([sym], exprs[0])
        with mpmath.workdps(d + 20):
            try:
                x0 = _guess_mpf(guesses, names[0])
            except ValueError as exc:
                raise ValueError(f"初始点无法数值化：{exc}") from None
            starts = [x0] if given_guess else [x0, mpmath.mpf(0), mpmath.mpf(-1), mpmath.mpf(2), mpmath.mpf("0.5")]
            root = None
            resid0 = None
            used = None
            for st in starts:
                cand = None
                try:
                    cand = mpmath.findroot(fmp, st)
                except Exception:  # noqa: BLE001
                    cand = None
                if cand is None:
                    continue
                try:
                    cand = mpmath.mpmathify(cand)
                except Exception:  # noqa: BLE001
                    continue
                if isinstance(cand, mpmath.mpc):
                    if mpmath.im(cand) != 0:
                        cand = mpmath.re(cand)
                val = _safe(fmp, cand)
                if val is None:
                    continue
                res = abs(val)
                if resid0 is None or res < resid0:
                    root, resid0, used = cand, res, st
                if res <= tol:
                    break
            if root is None:
                bad = MathResult.unsolved("numeric_solve", "mpmath.findroot 在所有初值下都未能收敛到根")
                bad.set_input(equations=[str(t) for t in eq_texts], vars=list(names), digits=d)
                bad.add_condition("可尝试给出更好的初值 guesses，或先用符号方法分析方程")
                return bad

            resid = resid0
            # 独立路线：多项式 → 数值特征根
            poly_match = None
            try:
                if exprs[0].is_polynomial(sym):
                    proots = sp.nroots(sp.Poly(exprs[0], sym), n=min(max(d, 15), 30))
                    real_roots = [sp.re(rt) for rt in proots if abs(float(sp.im(rt))) < 1e-12]
                    if real_roots:
                        near = min(real_roots, key=lambda rt: abs(float(rt) - float(root)))
                        poly_match = abs(complex(sp.N(near, 25)) - complex(root))
            except Exception:  # noqa: BLE001
                poly_match = None
            # 独立路线：符号 nsolve
            nsolve_dev = None
            try:
                sn = sp.nsolve(exprs[0], sym, float(root), prec=min(max(d, 15), 40))
                nsolve_dev = abs(complex(sp.N(sn, 25)) - complex(root))
            except Exception:  # noqa: BLE001
                nsolve_dev = None

        r.set_result(_sp_float(root, d))
        converged = bool(resid <= tol)
        r.set_raw(
            digits=d,
            abs_error=_num(resid),
            residual=_num(resid),
            residual_tolerance=_num(tol),
            converged=converged,
            initial_guess=str(used),
            is_approximate=True,
            cross_checks={
                "sympy.nroots_closest_real_root_deviation": _num(poly_match),
                "sympy.nsolve_deviation": _num(nsolve_dev),
            },
            error_bound_note="abs_error = 把数值根代回方程后的残差 |f(x*)|（残差小只说明该点是数值根，不是解析证明）",
        )
        methods = ["mpmath.findroot 高精度牛顿法"]
        if poly_match is not None:
            methods.append("SymPy nroots 特征值法（多项式，独立路线）")
        if nsolve_dev is not None:
            methods.append("SymPy nsolve（独立路线）")
        note = f"残差 |f(x*)| = {_num(resid)}（容差 {_num(tol)}），已收敛={converged}"
        if poly_match is not None:
            note += f"；与 nroots 最接近的实根相差 {_num(poly_match)}"
        _verify_numeric(r, methods, note + "。数值求根依赖初值，**可能有其他根**，且不是解析证明。")
        r.add_condition("数值根依赖初值与算法，只能给出初值附近的根；请用符号 solve/factor 确认是否有其他根")
        if not given_guess:
            r.add_condition("未提供初值，已用默认初值 " + str(used) + "；结果依赖初值，可能有其他根")
        if resid > tol:
            r.partial(f"残差 {_num(resid)} 超过容差 {_num(tol)}，该数值根不可信")
        return r

    # ---------------- 多变量方程组 ----------------
    import numpy as np
    from scipy.optimize import fsolve

    def _f_np(v: Any) -> Any:
        f = sp.lambdify(syms, exprs, modules=["numpy"])
        out = f(*[float(x) for x in v])
        return np.asarray(out, dtype=float).reshape(-1)

    fmp_multi = sp.lambdify(syms, exprs, modules=["mpmath"])

    def _f_mp(v: Any) -> list[Any]:
        vals = fmp_multi(*list(v))
        if not isinstance(vals, (list, tuple)):
            vals = [vals]
        return [mpmath.mpmathify(x) for x in vals]

    base = []
    for name in names:
        try:
            base.append(_guess_mpf(guesses, name))
        except ValueError as exc:
            raise ValueError(f"初始点无法数值化：{exc}") from None
    if not given_guess:
        base = [mpmath.mpf(1)] * len(names)

    starts_mp = [tuple(base)]
    if not given_guess:
        for i in range(len(names)):
            for delta in (mpmath.mpf(2), mpmath.mpf("-1"), mpmath.mpf("0.3")):
                alt = list(base)
                alt[i] = alt[i] + delta
                starts_mp.append(tuple(alt))

    best = None
    with mpmath.workdps(d + 20):
        for st in starts_mp:
            x0 = [float(x) for x in st]
            try:
                sol, info, ier, msg = fsolve(_f_np, x0, full_output=True)
            except Exception:  # noqa: BLE001
                continue
            if ier != 1:
                continue
            vec = [mpmath.mpf(str(float(x))) for x in np.asarray(sol).reshape(-1)]
            try:
                polished = mpmath.findroot(_f_mp, vec, solver="mnewton", tol=mpmath.mpf(10) ** (-(d + 5)), maxsteps=200)
            except Exception:  # noqa: BLE001
                polished = None
            if polished is not None:
                try:
                    tl = polished.tolist()
                    flat = [mpmath.mpmathify(r0[0] if isinstance(r0, (list, tuple)) else r0) for r0 in tl]
                    if len(flat) == len(names):
                        vec = flat
                except Exception:  # noqa: BLE001
                    pass
            try:
                resid = max(abs(x) for x in _f_mp(vec))
            except Exception:  # noqa: BLE001
                continue
            if best is None or resid < best[1]:
                best = (vec, resid, [float(x) for x in st])
            if resid <= tol:
                break

    if best is None:
        bad = MathResult.unsolved("numeric_solve", "scipy.fsolve 与 mpmath 高精度牛顿法都未能给出数值解")
        bad.set_input(equations=[str(t) for t in eq_texts], vars=list(names), digits=d)
        bad.add_condition("可尝试给出更接近解的初值 guesses")
        return bad

    vec, resid, _st = best
    converged = bool(resid <= tol)
    sol_dict = {name: _sp_float(val, d) for name, val in zip(names, vec)}
    r.set_result(sol_dict)
    r.set_raw(
        digits=d,
        abs_error=_num(resid),
        residual=_num(resid),
        residual_tolerance=_num(tol),
        converged=converged,
        initial_guess=[str(x) for x in _st],
        equations_verified=[A.to_text(e) for e in exprs],
        is_approximate=True,
        error_bound_note="abs_error = max|F(x*)| 残差向量的最大分量（代回原方程组）；残差小不等于解唯一",
    )
    _verify_numeric(
        r,
        ["scipy.optimize.fsolve（双精度）", "mpmath.findroot mnewton 高精度抛光"],
        f"残差 max|F(x*)| = {_num(resid)}（容差 {_num(tol)}），已收敛={converged}；"
        "数值解依赖初值，可能有其他解，且不是解析证明。",
    )
    r.add_condition("数值求根依赖初值，可能有其他解；请用符号 solve 确认解的完整性")
    if not given_guess:
        r.add_condition("未提供初值，已用默认初值 (1,1,...)；结果依赖初值，可能有其他解")
    if resid > tol:
        r.partial(f"残差 {_num(resid)} 超过容差 {_num(tol)}，该数值解不可信")
    return r


# --------------------------------------------------------------------------------------
# 4. numeric_optimize —— 数值极值
# --------------------------------------------------------------------------------------


def _bound_value(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip().replace("无穷", "oo").replace("∞", "oo")
        if s in ("oo", "+oo", "inf", "+inf", "-oo", "-inf"):
            return None
    if isinstance(x, sp.Basic):
        if x in (sp.oo, -sp.oo, sp.zoo):
            return None
        if x.free_symbols:
            raise ValueError(f"边界 {x} 含自由符号，必须给出具体数值")
        return float(sp.N(x, 25))
    try:
        f = float(x)
    except Exception:  # noqa: BLE001
        raise ValueError(f"无法把边界 {x!r} 转成数值") from None
    if math.isinf(f):
        return None
    return f


def _bounds_pairs(bounds: Any, n: int) -> list[tuple[float | None, float | None]]:
    if bounds is None:
        return [(None, None)] * n
    seq = bounds
    if isinstance(seq, dict):
        seq = [seq.get(i) for i in range(n)] if all(i in seq for i in range(n)) else list(seq.values())
    if isinstance(seq, (list, tuple)):
        if n == 1 and len(seq) == 2 and not any(isinstance(x, (list, tuple, dict)) for x in seq):
            seq = [seq]
        out: list[tuple[float | None, float | None]] = []
        for item in seq:
            if isinstance(item, dict):
                lo = item.get("lo", item.get("lower", item.get("min")))
                hi = item.get("hi", item.get("upper", item.get("max")))
                out.append((_bound_value(lo), _bound_value(hi)))
            elif isinstance(item, (list, tuple)):
                if len(item) != 2:
                    raise ValueError("bounds 的每一项必须是 [下界, 上界]")
                out.append((_bound_value(item[0]), _bound_value(item[1])))
            elif item is None:
                out.append((None, None))
            else:
                raise ValueError("bounds 必须是 [[lo,hi], ...] 形式")
        if len(out) != n:
            raise ValueError(f"bounds 的维度({len(out)})与变量个数({n})不一致")
        for lo, hi in out:
            if lo is not None and hi is not None and lo > hi:
                raise ValueError(f"bounds 的下界 {lo} 大于上界 {hi}")
        return out
    raise ValueError("bounds 必须是 [[lo,hi], ...] 形式")


@op("numeric_optimize", "minimize", "maximize", "math_optimize")
def numeric_optimize(
    expr: str,
    vars: Any,
    bounds: Any = None,
    mode: str = "min",
    digits: int = 15,
) -> MathResult:
    """数值求极值（scipy.optimize.minimize）；并在极值点附近做扰动采样做验证。

    注意：数值优化通常只给出**局部**极值，且受边界影响；这一点会写进 conditions。
    """
    d = _check_digits(digits)
    names = _name_list(vars) or ["x"]
    parsed = _parse_expr(expr, names)
    syms = _symbols_for([parsed], names)
    mode_txt = str(mode).strip().lower()
    if mode_txt in ("max", "maximum", "最大", "极大", "最大值"):
        maximize = True
    elif mode_txt in ("min", "minimum", "最小", "极小", "最小值"):
        maximize = False
    else:
        raise ValueError(f"mode 只能是 'min' 或 'max'，收到 {mode!r}")

    pairs = _bounds_pairs(bounds, len(names))
    scipy_bounds = [(lo, hi) for lo, hi in pairs]

    r = MathResult.ok(
        "numeric_optimize",
        method="scipy.optimize.minimize 数值优化"
        + ("（max 模式对 -expr 求最小）" if maximize else "")
        + "；极值点附近扰动采样验证",
    )
    r.set_input(expr=str(expr), vars=list(names), bounds=[[lo, hi] for lo, hi in pairs], mode=mode_txt, digits=d)

    import numpy as np
    from scipy.optimize import minimize

    try:
        f_np = sp.lambdify(syms, parsed, modules=["numpy"])
        f_np(*[0.0] * len(syms))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"表达式无法转成 numpy 可调用对象: {exc}") from None

    def _obj(v: Any) -> float:
        arr = np.asarray([float(x) for x in v], dtype=float)
        val = f_np(*list(arr))
        if np.iscomplexobj(val):
            val = np.real(val)
        return float(val)

    def _target(v: Any) -> float:
        val = _obj(v)
        return -val if maximize else val

    # 初值：优先边界中点
    base: list[float] = []
    for lo, hi in pairs:
        if lo is not None and hi is not None:
            base.append(0.5 * (lo + hi))
        elif lo is not None:
            base.append(lo + 1.0)
        elif hi is not None:
            base.append(hi - 1.0)
        else:
            base.append(0.0)
    starts: list[list[float]] = [list(base)]
    for i in range(len(names)):
        for delta in (1.0, -1.0, 0.5):
            alt = list(base)
            alt[i] = alt[i] + delta
            starts.append(alt)
    starts.append([1.0] * len(names))
    starts.append([-1.0] * len(names))
    del starts[12:]

    def _clip(v: list[float]) -> list[float]:
        out = []
        for x, (lo, hi) in zip(v, pairs):
            if lo is not None and x < lo:
                x = lo
            if hi is not None and x > hi:
                x = hi
            out.append(x)
        return out

    best_x = None
    best_val = None
    failures: list[str] = []
    bounded = any(b != (None, None) for b in scipy_bounds)
    method_order = ["L-BFGS-B", "Nelder-Mead", "Powell"] if bounded else ["BFGS", "Nelder-Mead", "Powell"]
    for st in starts:
        st = _clip(st)
        for method in method_order:
            use_bounds = scipy_bounds if method in ("L-BFGS-B", "TNC", "SLSQP", "Powell", "Nelder-Mead") else None
            try:
                res = minimize(
                    _target,
                    np.asarray(st, dtype=float),
                    method=method,
                    bounds=use_bounds,
                    options={"maxiter": 2000},
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{method}: {exc}")
                continue
            if not np.all(np.isfinite(res.x)):
                continue
            xin = _clip([float(x) for x in np.asarray(res.x).reshape(-1)])
            val = float(_target(xin))
            if best_val is None or val < best_val - 1e-14:
                best_x, best_val = xin, val

    if best_x is None or best_val is None:
        bad = MathResult.unsolved("numeric_optimize", "所有数值优化尝试都失败：" + "；".join(failures[:3]))
        bad.set_input(expr=str(expr), vars=list(names), mode=mode_txt)
        return bad

    f_star = _obj(best_x)

    # 扰动采样验证：邻域内不应有更优值
    gap = 0.0
    samples = 0
    worse_free = True
    worst_val = best_val
    for i in range(len(names)):
        scale = max(1.0, abs(best_x[i]))
        for delta in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
            for sgn in (1.0, -1.0):
                cand = list(best_x)
                cand[i] = cand[i] + sgn * delta * scale
                cand = _clip(cand)
                if cand == best_x:
                    continue
                try:
                    val = _target(cand)
                except Exception:  # noqa: BLE001
                    continue
                samples += 1
                gap = max(gap, abs(val - best_val))
                if val < best_val - 1e-9 * max(1.0, abs(best_val)):
                    worse_free = False
                    worst_val = min(worst_val, val)

    # 独立路线：符号梯度在最优点处是否为零（内点驻点判据）
    grad_norm = None
    try:
        grad = [sp.diff(parsed, s) for s in syms]
        gvals = []
        with mpmath.workdps(max(d, 20) + 10):
            for g in grad:
                gg = sp.lambdify(syms, g, modules=["mpmath"])
                v = _safe(gg, *[mpmath.mpf(x) for x in best_x])
                gvals.append(None if v is None else float(abs(v)))
        if all(v is not None for v in gvals):
            grad_norm = max(gvals)
    except Exception:  # noqa: BLE001
        grad_norm = None

    boundary_hits = []
    for i, (lo, hi) in enumerate(pairs):
        if lo is not None and abs(best_x[i] - lo) <= 1e-6 * max(1.0, abs(lo)):
            boundary_hits.append(f"{names[i]} 取下界 {lo}")
        if hi is not None and abs(best_x[i] - hi) <= 1e-6 * max(1.0, abs(hi)):
            boundary_hits.append(f"{names[i]} 取上界 {hi}")

    r.set_result(
        {
            "point": [_sp_float(mpmath.mpf(str(x)), min(d, 15)) for x in best_x],
            "value": _sp_float(mpmath.mpf(str(f_star)), min(d, 15)),
            "mode": "max" if maximize else "min",
        }
    )
    r.set_raw(
        digits=d,
        abs_error=_num(gap),
        is_approximate=True,
        objective_value=_num(f_star),
        internal_minimized_value=_num(best_val),
        point=[_num(x) for x in best_x],
        perturbation_samples=samples,
        nearby_worse_ok=worse_free,
        nearest_better_value=_num(worst_val),
        symbolic_gradient_max_abs=_num(grad_norm),
        boundary_hits=boundary_hits or None,
        failed_attempts=failures[:3] or None,
        abs_error_note="abs_error = 极值点 δ 邻域内目标函数的采样变化量（表征极值平坦程度/数值分辨率，不是严格误差界）",
    )
    methods = ["scipy.optimize.minimize（L-BFGS-B/BFGS/Nelder-Mead，多初值）", "极值点 δ 邻域扰动采样（确认不更优）"]
    if grad_norm is not None:
        methods.append("符号梯度 |∇f| 在最优点处的数值（驻点判据，独立路线）")
    note = f"扰动采样 {samples} 点，邻域内无更优值={worse_free}"
    if grad_norm is not None:
        note += f"，|∇f(x*)|max = {_num(grad_norm)}"
    _verify_numeric(r, methods, note + "。" + _NUMERIC_CAVEAT)
    r.add_condition("数值优化只给出**局部**极值（受初值与边界影响），可能存在其他极值点；全局性需另行论证")
    if boundary_hits:
        r.add_warning("最优解落在边界上（" + "；".join(boundary_hits) + "），该极值是**受边界约束**的结果，不是自由极值")
    if not worse_free:
        r.partial(f"扰动采样发现更优点（目标值 {_num(worst_val)} 优于 {_num(best_val)}），该结果不是极值")
    if grad_norm is not None and grad_norm > 1e-6:
        r.add_warning(f"最优点处符号梯度范数 {_num(grad_norm)} 不可忽略：可能未收敛，或最优点在边界/非光滑处")
    return r


# --------------------------------------------------------------------------------------
# 5. numeric_derivative —— 数值导数（符号求导的独立验证）
# --------------------------------------------------------------------------------------


def _richardson_central(f: Any, x0: Any, order: int, dps: int) -> Any:
    """两点/三点中心差分 + Richardson 外推（order=1,2 时与 mpmath.diff 独立）。"""
    h1 = mpmath.mpf(10) ** (-max(2, dps // (order + 2)))

    def d(h: Any) -> Any:
        if order == 1:
            return (f(x0 + h) - f(x0 - h)) / (2 * h)
        return (f(x0 + h) - 2 * f(x0) + f(x0 - h)) / h ** 2

    v1 = d(h1)
    v2 = d(h1 / 2)
    return (4 * v2 - v1) / 3


@op("numeric_derivative", "math_numeric_derivative")
def numeric_derivative(
    expr: str,
    var: str = "x",
    point: Any = None,
    order: int = 1,
    digits: int = 15,
) -> MathResult:
    """在某点求数值导数（mpmath.diff 为主，中心差分/符号导数为独立路线）。"""
    d = _check_digits(digits)
    k = _check_order(order)
    if point is None:
        raise ValueError("numeric_derivative 需要 point（求导点，如 point='1' 或 point='pi/4'）")
    names = _name_list(var) or ["x"]
    parsed = _parse_expr(expr, names)
    sym = _symbols_for([parsed], names)[0]
    pt = C.parse_point(point, [sym])
    if pt in (sp.oo, -sp.oo, sp.zoo):
        raise ValueError("数值导数不支持无穷远点（请用符号或极限手段）")

    r = MathResult.ok("numeric_derivative", method=f"数值 {k} 阶导数：mpmath.diff（高精度自适应差分）为主路线")
    r.set_input(expr=str(expr), var=str(sym), point=str(point), order=k, digits=d)

    with mpmath.workdps(d + 20):
        x0 = _mpf(pt, d)
        f = _lambdify_mp([sym], parsed)
        fv = _safe(f, x0, allow_inf=True)
        if fv is not None and not _is_real_mp(fv):
            r.add_warning("被积/被求导函数在该点取复数值，导数为复数值")
        if fv is not None and abs(fv) > mpmath.mpf(10) ** 10:
            r.add_warning(f"求导点附近函数值极大（|f|≈{mpmath.nstr(abs(fv), 6)}），可能是奇点附近，数值差分不可信")

        v1 = None
        try:
            v1 = mpmath.diff(f, x0, k)
        except Exception:  # noqa: BLE001
            v1 = None
        if v1 is None:
            try:
                v1 = _richardson_central(f, x0, k, d + 20) if k <= 2 else None
            except Exception:  # noqa: BLE001
                v1 = None

        v2 = None
        try:
            if k <= 2:
                v2 = _richardson_central(f, x0, k, d + 20)
        except Exception:  # noqa: BLE001
            v2 = None

        v3 = None
        symbolic_text = None
        nonsmooth = bool(
            parsed.has(sp.sign, sp.Abs, sp.Heaviside, sp.Max, sp.Min, sp.floor, sp.ceiling)
        )
        try:
            sd = sp.diff(parsed, sym, k)
            if isinstance(sd, sp.Basic) and not isinstance(sd, sp.Derivative):
                symbolic_text = A.to_text(sd)
                nonsmooth = bool(
                    sd.has(sp.sign, sp.Abs, sp.Heaviside, sp.Max, sp.Min, sp.floor, sp.ceiling)
                )
                fsd = sp.lambdify([sym], sd, modules=["mpmath"])
                v3 = _safe(fsd, x0)
        except Exception:  # noqa: BLE001
            v3 = None

        v_repeat = None
        try:
            v_repeat = mpmath.diff(f, x0, k)
        except Exception:  # noqa: BLE001
            v_repeat = None

        if v1 is None:
            bad = MathResult.unsolved("numeric_derivative", "mpmath.diff 与中心差分都失败（函数可能在该点不可导或无法数值化）")
            bad.set_input(expr=str(expr), var=str(sym), point=str(point), order=k, digits=d)
            return bad

        devs = []
        route_info: dict[str, Any] = {"mpmath.diff": mpmath.nstr(v1, min(d + 3, 30))}
        for label, other in (("中心差分+Richardson", v2), ("符号导数数值化", v3)):
            if other is None:
                route_info[label] = None
                continue
            route_info[label] = mpmath.nstr(mpmath.mpmathify(other), min(d + 3, 30))
            try:
                devs.append(abs(mpmath.mpmathify(other) - v1))
            except Exception:  # noqa: BLE001
                pass
        if not devs:
            devs.append(abs(v1) * mpmath.mpf(10) ** (-d) + mpmath.mpf(10) ** (-d))
            route_info["_note"] = "无独立路线可用，误差用 digits 位舍入界估计"
        abs_err = max(devs)

    _err_note = "abs_error = max(各独立数值/符号路线相对 mpmath.diff 的最大偏差)"
    if nonsmooth:
        # 角点/折点处三条数值路线会「一致地」给出对称差商（甚至都为 0），偏差为 0 并不代表精确。
        # 此处误差界本身没有意义，给一个与 digits 相关的分辨率下限，绝不谎报 abs_error = 0。
        abs_err = max(
            abs_err,
            abs(v1) * mpmath.mpf(10) ** (-(d - 3)) + mpmath.mpf(10) ** (-(d - 3)),
        )
        _err_note += (
            "；但表达式含非光滑成分（Abs/Max/Min/sign/…），该点可能是角点：各路线给出的只是对称差商，"
            "误差界已按 digits 给分辨率下限，不能作为可导性/导数值的证明"
        )
    r.set_result(_sp_float(v1, d))
    r.set_raw(
        digits=d,
        abs_error=_num(abs_err),
        is_approximate=True,
        order=k,
        point=str(point),
        derivative_value=mpmath.nstr(v1, min(d + 5, 30)),
        routes=route_info,
        symbolic_derivative=symbolic_text,
        nonsmooth=nonsmooth,
        suspected_closed_form=_nsimplify_hint(v1, d),
        abs_error_note=_err_note,
    )
    methods = ["mpmath.diff 高精度自适应差分"]
    if v2 is not None:
        methods.append("中心差分 + Richardson 外推（独立路线）")
    if v3 is not None:
        methods.append("SymPy 符号求导后高精度数值化（独立路线）")
    note = f"{k} 阶数值导数 = {mpmath.nstr(v1, min(d + 5, 25))}，" + f"独立路线最大偏差 {_num(abs_err)}"
    if symbolic_text is not None:
        note += f"；符号导数为 {symbolic_text}"
    _verify_numeric(r, methods, note + "。" + _NUMERIC_CAVEAT)
    if v3 is not None and abs_err > mpmath.mpf(10) ** (-(d - 5)):
        r.add_warning(
            f"数值导数与符号导数偏差较大（{_num(abs_err)}）：可能是差分步长/截断误差导致，符号结果优先"
        )
    if k >= 3 and abs_err > mpmath.mpf(10) ** (-(d - 6)):
        r.add_warning(f"高阶（{k} 阶）数值差分的舍入误差随阶数迅速放大，偏差 {_num(abs_err)} 属正常范围，可信度下降")
    if nonsmooth:
        r.add_warning(
            "被求导表达式含 Abs/Max/Min/sign/Heaviside/floor 等非光滑成分：该点可能是角点、折点或尖点，"
            "此处可能根本不可导；数值差分给出的只是（对称）差商，不构成可导性证明，请分别检查左右导数"
        )
        r.add_condition(
            "含绝对值/分段（非光滑）的表达式，导数请分区间讨论并把折点单独拿出来验证左右导数"
        )
    return r


# --------------------------------------------------------------------------------------
# 6. compare_numeric —— 独立数值交叉验证统一入口
# --------------------------------------------------------------------------------------


@op("compare_numeric", "math_compare", "numeric_compare")
def compare_numeric(
    expr: str,
    candidate: str,
    vars: Any = None,
    trials: int = 9,
    samples: int | None = None,
    rel_tol: float = 1e-9,
) -> MathResult:
    """高精度抽样比较两个表达式是否数值一致（数值证据，不构成符号证明）。

    说明：``samples`` 会被引擎别名表改名为 ``trials``，两个名字都接受。
    """
    n_samples = trials if trials is not None else (samples if samples is not None else 9)
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, float)):
        raise ValueError(f"samples 必须是正整数，收到 {n_samples!r}")
    n_samples = int(n_samples)
    if n_samples < 1 or n_samples > 200:
        raise ValueError(f"samples 必须在 1..200 之间，收到 {n_samples}")
    if isinstance(rel_tol, bool) or not isinstance(rel_tol, (int, float)) or not (0 < float(rel_tol) < 1):
        raise ValueError(f"rel_tol 必须是 (0,1) 内的小数，收到 {rel_tol!r}")
    rel_tol = float(rel_tol)

    names = _name_list(vars)
    a = _parse_expr(expr, names or None)
    b = _parse_expr(candidate, names or None)
    if not isinstance(a, sp.Basic) or not isinstance(b, sp.Basic):
        raise ValueError("expr / candidate 必须是可用 SymPy 解析的表达式")

    all_names = sorted({s.name for s in (a.free_symbols | b.free_symbols)} | set(names))
    symbols = _symbols_for([a, b], all_names)

    r = MathResult.ok(
        "compare_numeric",
        method="高精度抽样比较（数值证据，不构成符号证明）",
    )
    r.set_input(expr=str(expr), candidate=str(candidate), vars=list(all_names) or None, samples=n_samples, rel_tol=rel_tol)

    # 官方入口：C.numeric_agree（必须传表达式自带的真实 Symbol 对象）
    agree, dev, note = C.numeric_agree(a, b, symbols=symbols or None, rel_tol=rel_tol, samples=n_samples)
    no_evidence = ("未获得有效数值证据" in str(note)) or ("无有效" in str(note))

    # 自建抽样明细（与 numeric_agree 同一网格；多变量时各变量取**不同**值以避开对角线切片）
    detail: list[dict[str, Any]] = []
    checked = 0
    worst_mine = 0.0
    if all_names:
        fa = sp.lambdify(symbols, a, modules=["mpmath"])
        fb = sp.lambdify(symbols, b, modules=["mpmath"])
        with mpmath.workdps(30):
            for idx in range(min(n_samples, len(_GRID))):
                pts = tuple(mpmath.mpf(_GRID_STR[(idx + j) % len(_GRID)]) for j in range(len(symbols)))
                try:
                    va = _safe(fa, *pts)
                    vb = _safe(fb, *pts)
                except Exception:  # noqa: BLE001
                    va = vb = None
                entry = {"point": {s.name: _GRID_STR[(idx + j) % len(_GRID)] for j, s in enumerate(symbols)}}
                if va is None or vb is None:
                    entry["status"] = "定义域外/无法求值"
                else:
                    d_rel = _rel_dev(va, vb)
                    entry["expr_value"] = mpmath.nstr(va, 15)
                    entry["candidate_value"] = mpmath.nstr(vb, 15)
                    entry["rel_dev"] = _num(d_rel)
                    entry["consistent"] = bool(d_rel <= rel_tol)
                    checked += 1
                    worst_mine = max(worst_mine, d_rel)
                detail.append(entry)
    else:
        # 纯常数比较
        with mpmath.workdps(30):
            va = _eval_mp(a, [], ())
            vb = _eval_mp(b, [], ())
        if va is not None and vb is not None:
            d_rel = _rel_dev(va, vb)
            checked = 1
            worst_mine = d_rel
            detail.append({"point": {}, "expr_value": mpmath.nstr(va, 15), "candidate_value": mpmath.nstr(vb, 15), "rel_dev": _num(d_rel)})

    if no_evidence or checked == 0:
        r.set_result({"agree": None, "max_rel_dev": None, "note": str(note), "samples_used": 0})
        r.set_raw(sampling_detail=detail, samples_requested=n_samples, rel_tol=rel_tol)
        _verify_numeric(
            r,
            ["SymPy 高精度抽样比较（numeric_agree）", "mpmath 30 位精度独立抽样"],
            "没有任何抽样点落在两个表达式的公共定义域内 → **无法判定**（注意：numeric_agree 在这种情况下会返回 agree=True 并注明「未获得有效数值证据」，本算子已识别该情形并拒绝据此宣布一致）。",
        )
        r.partial("所有抽样点都超出定义域或无法求值，没有有效数值证据，不能判定两个表达式是否一致")
        return r

    r.set_result({"agree": bool(agree), "max_rel_dev": _num(dev), "note": str(note), "samples_used": n_samples})
    r.set_raw(
        agree=bool(agree),
        max_rel_dev=_num(dev),
        sampling_detail=detail,
        samples_requested=n_samples,
        samples_checked=checked,
        rel_tol=rel_tol,
        max_rel_dev_numeric_agree=_num(dev),
        max_rel_dev_local=worst_mine,
        variables=[s.name for s in symbols],
        note="抽样明细使用各变量不同的网格取值（避免 numeric_agree 对所有变量取同一值形成的对角线切片）",
    )
    _verify_numeric(
        r,
        ["SymPy 高精度(30 位)抽样比较（sympy_core.numeric_agree）", "mpmath 30 位精度独立抽样（各变量取不同值）"],
        f"{note}；本地独立抽样 {checked} 点，最大相对偏差 {worst_mine:.3e}。"
        "抽样一致是**数值证据**，有限个点上的吻合不能排除两者在别处不同，也不构成符号证明。",
    )
    if len(symbols) > 1:
        r.add_warning(
            "多变量情形：sympy_core.numeric_agree 在抽样时对所有自由变量取**同一个值**（在 x=y 的对角线上采样），"
            "可能漏判（例如 x-y 与 0 在对角线上完全一致）；本算子的本地抽样已用各变量不同取值补充验证"
        )
    if not agree and worst_mine <= rel_tol:
        r.add_warning("官方接口判定不一致，但本地抽样未发现明显偏差：可能是抽样点/口径差异，请人工确认")
    if agree and worst_mine > rel_tol:
        r.add_warning("官方接口判定一致，但本地抽样发现偏差超过 rel_tol，结论存疑")
    return r


# --------------------------------------------------------------------------------------
# 7. numeric_limit —— 数值极限估计
# --------------------------------------------------------------------------------------


def _aitken(v1: Any, v2: Any, v3: Any) -> tuple[Any, Any]:
    """Aitken Δ² 外推 + 收敛比 r（误差 ~ C·h^p 时按几何级数外推）。

    注意：v 为**线性**序列（Δ²=0，例如 log(x) 在 0+ 的 -k·log2）时 Aitken 公式是 0/0，
    此时必须把收敛比报成 1（不可外推、未收敛），绝不能报 0 —— 报 0 会被上层
    `converged = ratio < 0.9` 误判成「已收敛」，从而把一个发散到 -∞ 的序列
    当成正常极限返回（实测 log(x)→-∞ 曾被判定 ok + 值 -9.41）。
    """
    denom = v3 - 2 * v2 + v1
    try:
        if abs(denom) == 0:
            return v3, mpmath.mpf(1)
        est = v3 - (v3 - v2) ** 2 / denom
    except Exception:  # noqa: BLE001
        return v3, mpmath.mpf(1)
    d1 = abs(v2 - v1)
    d2 = abs(v3 - v2)
    if d1 == 0:
        # 常数序列：已经收敛（r=0）；只有末段又变了才视为未收敛。
        return est, (mpmath.mpf(0) if d2 == 0 else mpmath.mpf(1))
    return est, d2 / d1


def _richardson(seq: list[Any], levels: int = 5) -> tuple[Any, Any] | None:
    """经典 Richardson（Neville）幂级数外推，h 每次减半。

    只要 v(h) = L + a₁h + a₂h² + …（有限点处的 Taylor 展开、无穷远处的 1/x 展开
    都是这种形式），Richardson 表就能逐级消掉各阶项。比 Aitken Δ² 更合适：
    Aitken 只按几何级数模型消掉主项，实测 (1+1/x)^x（展开里同时有 1/x 与 1/x² 项）
    的 Aitken 估计有 3e-7 偏差，Richardson 表能到 1e-15 以上。
    对振荡/发散序列本函数会给出垃圾——调用方必须用收敛比等判据把好关。
    """
    vals = list(seq)
    if len(vals) < 3:
        return None
    vals = vals[-(levels + 1):]
    table: list[list[Any]] = [vals]
    for j in range(1, len(vals)):
        prev = table[-1]
        new: list[Any] = []
        for i in range(len(prev) - 1):
            try:
                new.append(prev[i + 1] + (prev[i + 1] - prev[i]) / (mpmath.mpf(2) ** j - 1))
            except Exception:  # noqa: BLE001
                return None
        if not new:
            break
        table.append(new)
    best = table[-1][0]
    stability = abs(table[-1][0] - table[-2][0]) if len(table) >= 2 else None
    return best, stability


@op("numeric_limit", "math_numeric_limit")
def numeric_limit(
    expr: str,
    var: str = "x",
    point: Any = None,
    dir: str | None = None,
    digits: int = 15,
) -> MathResult:
    """用序列逼近 + Aitken 外推给出**数值**极限估计（不是解析证明）。"""
    d = _check_digits(digits)
    if point is None:
        raise ValueError("numeric_limit 需要 point（趋近点，如 0 / oo / -oo）")
    names = _name_list(var) or ["x"]
    parsed = _parse_expr(expr, names)
    sym = _symbols_for([parsed], names)[0]
    p_sp = C.parse_point(point, [sym])
    direction = None
    if dir is not None:
        ds = str(dir).strip()
        if ds in ("-", "left", "左", "左极限", "below", "lt"):
            direction = "-"
        elif ds in ("+", "right", "右", "右极限", "above", "gt"):
            direction = "+"
        elif ds in ("", "both", "双侧", "two-sided", "+-"):
            direction = None
        else:
            raise ValueError(f"dir 只能是 '+'/'-'（或 left/right/左/右），收到 {dir!r}")

    r = MathResult.ok(
        "numeric_limit",
        method="序列逼近 + Richardson/Aitken 外推（mpmath 高精度，双底数 2 与 10 交叉验证）",
    )
    r.set_input(expr=str(expr), var=str(sym), point=str(point), dir=direction, digits=d)

    def _sequence(side: int, base: int) -> list[tuple[Any, Any]]:
        seq: list[tuple[Any, Any]] = []
        for k in range(1, 13):
            if mpmath.isfinite(c):
                x = c + side * mpmath.mpf(base) ** (-k)
            else:
                x = side * mpmath.mpf(base) ** k
            v = _safe(f, x)
            if v is None:
                continue
            seq.append((x, v))
        return seq

    with mpmath.workdps(d + 30):
        c = _mpf(p_sp, d)
        f = _lambdify_mp([sym], parsed)

        sides = []
        if direction in (None, "+"):
            sides.append((+1, "右" if direction == "+" else "两侧(右)"))
        if direction in (None, "-"):
            sides.append((-1, "左" if direction == "-" else "两侧(左)"))

        results: dict[str, Any] = {}
        for side, label in sides:
            ests = []
            for base in (2, 10):
                seq = _sequence(side, base)
                if len(seq) < 3:
                    continue
                vals = [v for _, v in seq]
                est, ratio = _aitken(vals[-3], vals[-2], vals[-1])
                est2 = None
                if len(vals) >= 4:
                    est2, _ = _aitken(vals[-4], vals[-3], vals[-2])
                rich = _richardson(vals)
                # 优先用 Richardson：它对 v(h)=L+a₁h+a₂h²+… 型序列能消掉多阶项，
                # 而 Aitken 只按几何级数消主项。两者都记录，差值计入误差界。
                if rich is not None and (est2 is None or rich[1] is None or rich[1] <= abs(est - est2) * 100 + abs(est) * mpmath.mpf(10) ** (-d)):
                    chosen_est, chosen_stab, used = rich[0], rich[1], "richardson"
                else:
                    chosen_est, chosen_stab, used = est, (None if est2 is None else abs(est - est2)), "aitken"
                ests.append(
                    {
                        "base": base,
                        "estimate": chosen_est,
                        "method": used,
                        "ratio": ratio,
                        "last": vals[-1],
                        "stability": chosen_stab,
                        "aitken": _num(est),
                        "richardson": None if rich is None else _num(rich[0]),
                        "method_deviation": _num(abs(rich[0] - est)) if rich is not None else None,
                        # _num() 对溢出会返回字符串（"inf"），**不能**拿去做数值比较；
                        # 需要参与比较的数值一律另存 mpf 版本。
                        "method_deviation_mpf": (abs(rich[0] - est) if rich is not None else None),
                        "points": [mpmath.nstr(x, 10) for x, _ in seq[-4:]],
                        "values": [mpmath.nstr(v, min(d + 3, 25)) for _, v in seq[-4:]],
                        # 整条序列的量级放大倍数：单调发散时可达 1e3 以上，而 Aitken 的
                        # 逐步收敛比可能只有 2（如 1/x 在 0+：2,4,8,… 每步只翻倍）。
                        "amplification": _num(
                            abs(vals[-1]) / max(abs(vals[0]), mpmath.mpf(10) ** (-(d + 5)))
                        ),
                    }
                )
            if not ests:
                results[label] = None
                continue
            main = ests[0]
            alt = ests[1] if len(ests) > 1 else None
            cross = abs(main["estimate"] - alt["estimate"]) if alt is not None else None
            results[label] = {"main": main, "alt": alt, "cross": cross}

        ok_sides = {k: v for k, v in results.items() if v is not None}
        if not ok_sides:
            bad = MathResult.unsolved("numeric_limit", "在该点附近无法取得有效的函数值序列（定义域/奇点问题）")
            bad.set_input(expr=str(expr), var=str(sym), point=str(point), dir=direction, digits=d)
            return bad

        # 选择估计：优先双侧中较稳定的
        label, info = next(iter(ok_sides.items()))
        main = info["main"]
        est = main["estimate"]
        ratio = main["ratio"]
        _dev = main["method_deviation_mpf"]
        _cands = [x for x in [main["stability"], info["cross"]] if x is not None]
        if _dev is not None:
            if main["method"] == "richardson" and main["stability"] is not None:
                # 选了 Richardson 且它自身逐级稳定：此时 Aitken 的偏差是**模型误差**
                # （Aitken 只按几何级数消主项，(1+1/x)^x 这种含 1/x 与 1/x² 两项的
                # 序列会差 3e-7）。只把它的很小一部分计入误差界，原值仍完整记录在 raw。
                _cands.append(min(_dev, 100 * main["stability"] + abs(est) * mpmath.mpf(10) ** (-d)))
            else:
                _cands.append(_dev)
        abs_err = max(_cands or [abs(main["last"] - est)])
        if abs_err == 0:
            abs_err = abs(est) * mpmath.mpf(10) ** (-d) + mpmath.mpf(10) ** (-d)

        both_side_check = None
        if len(ok_sides) == 2:
            lv = [v["main"]["estimate"] for v in ok_sides.values()]
            both_side_check = abs(lv[0] - lv[1])

        # 符号 limit 独立对照（必须用表达式自带的 Symbol）
        sym_limit_text = None
        sym_limit_dev = None
        try:
            sdir = "+-" if direction is None else direction
            sl = sp.limit(parsed, sym, p_sp, sdir)
            # 注：不要用 A.is_unevaluated —— 该函数在当前 SymPy(1.13.3) 下必抛 AttributeError
            # （内部引用了不存在的 sp.Solve），这里自行判断是否仍含未求值的 Limit/Integral 等。
            unevaluated = isinstance(sl, (sp.Integral, sp.Derivative, sp.Limit, sp.Sum, sp.Product))
            if isinstance(sl, sp.Basic) and not unevaluated and not sl.free_symbols:
                sym_limit_text = A.to_text(sl)
                if sl not in (sp.oo, -sp.oo, sp.zoo) and _is_real_mp(est):
                    try:
                        sym_limit_dev = abs(complex(sp.N(sl, 30)) - complex(est))
                    except Exception:  # noqa: BLE001
                        sym_limit_dev = None
        except Exception:  # noqa: BLE001
            sym_limit_text = None

        # 符号 limit 是可信的独立结论：数值估计与它的偏差本身就是数值误差的实测值。
        if sym_limit_dev is not None and sym_limit_dev > abs_err:
            abs_err = sym_limit_dev

        growth = None
        try:
            if _is_real_mp(est):
                first = abs(info["main"]["last"])
                growth = float(abs(est) / max(mpmath.mpf(1), first))
        except Exception:  # noqa: BLE001
            growth = None

    converged = bool(ratio < mpmath.mpf("0.9"))
    # _num() 会把溢出值保留成字符串（"inf"），这里必须转回 mpf 再比较，否则
    # `str > mpf` 会抛 TypeError 变成 internal 错误（实测 exp(x) 在 +∞ 触发过）。
    _amp = main.get("amplification")
    try:
        _amp_v = None if _amp is None else mpmath.mpf(str(_amp))
    except Exception:  # noqa: BLE001
        _amp_v = None
    if not converged and (
        (growth is not None and growth > 1e3)
        or (_amp_v is not None and _amp_v > 10)
    ):
        # 量级还在放大 + 序列不收敛：函数在该点发散/趋于无穷。
        # 这时任何有限「极限估计」都是垃圾（实测 1/x 在 0+ 会给出 0.0），宁 unsolved。
        _amp_txt = "未知" if _amp_v is None else mpmath.nstr(_amp_v, 4)
        _sym_txt = "" if sym_limit_text is None else f"（符号 limit 给出 {sym_limit_text}）"
        bad = MathResult.unsolved(
            "numeric_limit",
            f"逼近序列的量级在持续放大（整段放大约 {_amp_txt} 倍）且收敛比 {_num(ratio)} ≥ 0.9："
            f"函数在该点附近发散或趋于无穷，数值极限没有意义{_sym_txt}",
        )
        bad.set_input(expr=str(expr), var=str(sym), point=str(point), dir=direction, digits=d)
        bad.set_raw(
            digits=d,
            convergence_ratio=_num(ratio),
            growth_factor=None if growth is None else _num(growth),
            amplification=None if _amp_v is None else _num(_amp_v),
            sequence_tail={k: v["main"]["values"] for k, v in ok_sides.items()},
            symbolic_limit=sym_limit_text,
            extrapolation_estimate=mpmath.nstr(est, min(d + 5, 25)),
        )
        bad.add_condition("发散/趋于无穷的情形请用符号 limit，或用无穷大量的比较判别法（等价无穷小/大）")
        bad.add_warning("数值「极限」在发散点上会给出看似合理的有限值，必须靠序列量级是否放大来识别")
        return bad
    r.set_result(_sp_float(est, d))
    r.set_raw(
        digits=d,
        abs_error=_num(abs_err),
        is_approximate=True,
        limit_estimate=mpmath.nstr(est, min(d + 5, 30)),
        convergence_ratio=_num(ratio),
        converged=converged,
        side=label,
        sequence_tail={k: v["main"]["values"] for k, v in ok_sides.items()},
        base2_vs_base10_deviation=_num(info["cross"]),
        extrapolation_method=main["method"],
        aitken_estimate=main["aitken"],
        richardson_estimate=main["richardson"],
        extrapolation_deviation=main["method_deviation"],
        symbolic_limit=sym_limit_text,
        symbolic_limit_deviation=_num(sym_limit_dev),
        both_sides_deviation=_num(both_side_check),
        suspected_closed_form=_nsimplify_hint(est, d),
        abs_error_note="abs_error = max(所选外推法的稳定性, 底数 2 与 10 两条序列估计之差, Richardson 与 Aitken 两法之差)；比值<1 才认为序列收敛",
    )
    methods = [
        "mpmath 序列逼近（x→点，h=2^-k / 10^-k）",
        f"{'Richardson/Neville 幂级数外推' if main['method'] == 'richardson' else 'Aitken Δ² 外推'}（主）",
        "Aitken/Richardson 双法交叉验证",
        "双底数序列交叉验证",
    ]
    note = f"数值极限估计 = {mpmath.nstr(est, min(d + 5, 25))}（收敛比 {_num(ratio)}，外推误差估计 {_num(abs_err)}）"
    if sym_limit_text is not None:
        methods.append("SymPy 符号 limit（独立路线）")
        note += f"；符号 limit = {sym_limit_text}"
    _verify_numeric(r, methods, note + "。**这是数值估计，不是解析证明。** 只有符号 limit 或 ε-δ 论证才构成证明。")

    if sym_limit_text is None:
        r.add_warning("符号引擎未给出解析极限：以下结论**只是数值估计**，收敛慢或不收敛时代码可能给出假象")
    elif sym_limit_dev is not None and sym_limit_dev > min(1e-4, 10.0 ** (-(min(d, 12) - 3))) * max(1.0, float(abs(complex(est))) if _is_real_mp(est) else 1.0):
        r.add_warning(
            f"数值极限 {mpmath.nstr(est, 12)} 与符号 limit {sym_limit_text} 不一致（偏差 {_num(sym_limit_dev)}）："
            "以符号结论为准；数值方法在收敛慢/振荡时容易给出错误估计"
        )
    _achieved = None
    if abs_err is not None and abs_err > 0:
        try:
            _achieved = int(mpmath.floor(-mpmath.log10(abs_err / max(mpmath.mpf(1), abs(est)))))
        except Exception:  # noqa: BLE001
            _achieved = None
    if _achieved is not None and _achieved < min(d, 12) - 3:
        r.add_warning(
            f"数值极限只稳定到约 {_achieved} 位有效数字（请求 digits={d}）：序列收敛慢或不是幂级数型，"
            "误差界已如实给出；请以符号 limit 为准"
        )
    if not converged:
        r.partial(f"逼近序列未收敛（收敛比 {_num(ratio)} ≥ 0.9），数值极限不可信")
    elif both_side_check is not None and both_side_check > 1e-6 * max(1.0, float(abs(complex(est))) if _is_real_mp(est) else 1.0):
        r.add_warning(
            f"左极限与右极限相差 {_num(both_side_check)}：双侧极限可能不存在，请分别查看单侧极限"
        )
    if growth is not None and growth > 1e3:
        r.add_warning(f"序列量级放大约 {growth:.3g} 倍：函数可能趋于无穷或发散，数值极限意义有限")
    return r


# --------------------------------------------------------------------------------------
# 8. numeric_sum —— 数值求和（有限项精确 / 无限项加速）
# --------------------------------------------------------------------------------------


def _int_or_inf(value: Any, sym: Any, *, name: str) -> int | None:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是整数或 oo，收到 {value!r}")
    if isinstance(value, (int,)):
        return int(value)
    if isinstance(value, float):
        if float(value).is_integer():
            return int(value)
        raise ValueError(f"{name} 必须是整数，收到 {value!r}")
    if isinstance(value, sp.Basic):
        if value is sp.oo:
            return None
        if value.is_Integer:
            return int(value)
        raise ValueError(f"{name} 必须是整数或 oo，收到 {value}")
    text = str(value).strip()
    if text in ("∞", "无穷", "inf", "Inf", "oo", "+oo"):
        return None
    if text in ("-∞", "-无穷", "-inf", "-oo"):
        raise ValueError(f"{name} 不支持负无穷（求和下标必须是递增整数）")
    norm = A.normalize(text)
    if norm in ("oo", "+oo"):
        return None
    try:
        e = _parse_expr(norm, [sym.name])
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{name} 无法解析为整数: {exc}") from None
    if isinstance(e, sp.Basic) and e.is_Integer:
        return int(e)
    raise ValueError(f"{name} 必须是整数或 oo，收到 {value!r}")


def _is_smooth_in(expr: Any, sym: Any) -> bool:
    """Euler–Maclaurin 尾项要求被积函数对求和指标是光滑函数。"""
    bad_types = (sp.floor, sp.ceiling, sp.sign, sp.Mod, sp.factorial, sp.binomial, sp.RisingFactorial, sp.FallingFactorial)
    for node in sp.preorder_traversal(expr):
        if isinstance(node, bad_types):
            return False
        if isinstance(node, sp.Pow):
            base, exp = node.args
            if base == -1:
                return False
            if isinstance(exp, sp.Basic) and exp.has(sym):
                if base.is_negative:
                    return False
                if base.is_negative is None and isinstance(base, sp.Basic) and base.is_number:
                    try:
                        if float(sp.N(base)) < 0:
                            return False
                    except Exception:  # noqa: BLE001
                        return False
        if isinstance(node, sp.Abs):
            return False
    return True


def _euler_maclaurin_tail(f: Any, N: Any) -> tuple[Any, Any]:
    """Σ_{n=N+1}^∞ f(n) 的 Euler–Maclaurin 估计（返回 值, 误差界）。

    注意尾积分的变量替换必须写成 ``x = N + t/(1-t)``：写成 ``x = N - t/(1-t)``
    会覆盖 (-∞, N] 而不是 [N, ∞)，实测把结果放大约一倍。
    误差界取「尾积分自报误差」与「渐近级数末项量级 ×10」的较大者 —— EM 是渐近级数，
    真正的截断误差由**首个未计入项**主导，量级不超过最后一个已计入的修正项。
    """
    tail, err = mpmath.quad(lambda t: f(N + t / (1 - t)) / (1 - t) ** 2, [0, 1], error=True)
    corr = -f(N) / 2
    last_term = abs(corr)
    for k, coef in ((1, mpmath.mpf(-1) / 12), (3, mpmath.mpf(1) / 720), (5, mpmath.mpf(-1) / 30240), (7, mpmath.mpf(1) / 1209600)):
        try:
            dk = mpmath.diff(f, N, k)
        except Exception:  # noqa: BLE001
            break
        if not mpmath.isfinite(dk):
            break
        term = coef * dk
        corr += term
        last_term = abs(term)
    bound = max(abs(err), last_term * 10)
    return tail + corr, bound


@op("numeric_sum", "math_numeric_sum")
def numeric_sum(
    expr: str,
    var: str = "n",
    lower: Any = "1",
    upper: Any = "oo",
    digits: int = 15,
    order: int | None = None,
    terms: int | None = None,
) -> MathResult:
    """数值求和：有限项高精度精确求和；无限项用 mpmath.nsum 加速 + Euler–Maclaurin 尾项交叉验证。

    ``terms`` 会被引擎别名表改名为 ``order``，两个名字都接受（默认 200）。
    """
    d = _check_digits(digits)
    n_terms = order if order is not None else (terms if terms is not None else 200)
    try:
        n_terms = int(n_terms)
    except (TypeError, ValueError):
        raise ValueError(f"terms/order 必须是整数，收到 {n_terms!r}") from None
    if not 10 <= n_terms <= 100000:
        raise ValueError(f"terms/order 必须在 10..100000 之间，收到 {n_terms}")

    names = _name_list(var) or ["n"]
    parsed = _parse_expr(expr, names)
    sym = _symbols_for([parsed], names)[0]
    extra = sorted(s.name for s in parsed.free_symbols if s is not sym and s != sym)
    if extra:
        raise ValueError("求和表达式含额外自由变量 " + ", ".join(extra) + "，请先代入具体数值")

    lo = _int_or_inf(lower, sym, name="lower")
    hi = _int_or_inf(upper, sym, name="upper")
    if lo is None:
        raise ValueError("lower 必须是具体整数（不能是无穷）")
    if hi is not None and hi < lo:
        raise ValueError(f"upper({hi}) 小于 lower({lo})")
    if hi is not None and hi - lo + 1 > 1000000:
        raise ValueError(f"有限求和项数({hi - lo + 1})过大（上限 1000000），请改用无穷上界或缩小范围")

    r = MathResult.ok("numeric_sum", method=f"高精度数值求和（mpmath，目标 {d} 位有效数字）")
    r.set_input(expr=str(expr), var=str(sym), lower=str(lower), upper=str(upper), digits=d, terms=n_terms)

    with mpmath.workdps(d + 25):
        f = _lambdify_mp([sym], parsed)

        # ---------------- 有限项 ----------------
        if hi is not None:
            values = []
            bad_terms = 0
            for k in range(lo, hi + 1):
                v = _safe(f, mpmath.mpf(k))
                if v is None:
                    bad_terms += 1
                    continue
                values.append(v)
            if not values:
                bad = MathResult.unsolved("numeric_sum", "求和区间内没有任何可求值的项（定义域问题）")
                bad.set_input(expr=str(expr), var=str(sym), lower=str(lower), upper=str(upper), digits=d)
                return bad
            total_fsum = mpmath.fsum(values)
            total_exact = None
            if (hi - lo + 1) <= 120:
                try:
                    exact_terms = [parsed.subs(sym, k) for k in range(lo, hi + 1)]
                    if all(isinstance(t, sp.Basic) and t.is_rational for t in exact_terms):
                        total_exact = sp.N(sp.Add(*exact_terms), d)
                except Exception:  # noqa: BLE001
                    total_exact = None
            total = total_fsum
            # 只有 SymPy 精确有理求和成功时才能把误差定为 0；否则即使通项是"精确有理数"，
            # mpmath 的 mpf 仍是二进制浮点（如 1/3 不可精确表示），必须给舍入界。
            if total_exact is not None:
                abs_err = abs(mpmath.mpmathify(str(total_exact)) - total_fsum)
            else:
                abs_err = abs(total_fsum) * mpmath.mpf(10) ** (-d) + mpmath.mpf(10) ** (-d)
            r.set_result(_sp_float(total, d))
            r.set_raw(
                digits=d,
                abs_error=_num(abs_err),
                is_approximate=total_exact is None,
                terms_summed=len(values),
                skipped_terms=bad_terms or None,
                exact_rational_sum=(A.to_text(total_exact) if total_exact is not None else None),
                method="mpmath.fsum 高精度逐项求和" + ("；另用 SymPy 精确有理数求和交叉验证" if total_exact is not None else ""),
                abs_error_note="abs_error = mpmath.fsum 与 SymPy 精确有理求和之差（无精确路线时为 digits 位舍入界）",
            )
            methods = ["mpmath.fsum 逐项高精度求和"]
            if total_exact is not None:
                methods.append("SymPy 精确有理数求和（独立路线）")
            _verify_numeric(
                r,
                methods,
                f"有限项求和：{len(values)} 项，结果 {mpmath.nstr(total, min(d + 5, 25))}"
                + (f"，与精确有理求和偏差 {_num(abs_err)}" if total_exact is not None else "")
                + "。有限项求和是确定的（只受浮点舍入影响）。",
            )
            if bad_terms:
                r.add_warning(f"{bad_terms} 项因定义域问题被跳过，结果不完整")
                r.partial(f"{bad_terms} 项无法求值，求和结果不完整")
            return r

        # ---------------- 无限项 ----------------
        N = n_terms
        # 一次遍历同时取两个部分和（S(N) 与 S(2N)），避免重复数值求值
        short_vals: list[Any] = []
        long_vals: list[Any] = []
        for k in range(lo, 2 * N + 1):
            v = _safe(f, mpmath.mpf(k))
            if v is None:
                continue
            if k <= N:
                short_vals.append(v)
            long_vals.append(v)
        partial = mpmath.fsum(short_vals)
        partial2 = mpmath.fsum(long_vals)

        routes: list[tuple[str, Any, Any, str]] = []  # name, value, err, note
        fails: list[str] = []

        for method, label in (("r", "Richardson 外推"), ("s", "Shanks 变换"), ("d", "直接求和")):
            try:
                v = mpmath.nsum(f, [lo, mpmath.inf], method=method)
                routes.append((f"nsum[{method}]", v, None, f"mpmath.nsum method='{method}'（{label}）"))
            except Exception as exc:  # noqa: BLE001
                fails.append(f"nsum[{method}]: {exc}")

        smooth = _is_smooth_in(parsed, sym)
        if smooth:
            try:
                tail, terr = _euler_maclaurin_tail(f, mpmath.mpf(N))
                em = partial + tail
                routes.append(("euler_maclaurin", em, terr, f"部分和({N} 项) + Euler–Maclaurin 尾项（∫_N^∞ − f(N)/2 − f'(N)/12 + f'''(N)/720 − …）"))
            except Exception as exc:  # noqa: BLE001
                fails.append(f"Euler–Maclaurin: {exc}")

        usable = [(n, v, e, note) for n, v, e, note in routes if v is not None and mpmath.isfinite(v)]
        if not usable:
            bad = MathResult.unsolved("numeric_sum", "所有无限求和路线都失败：" + "；".join(fails[:4]))
            bad.set_input(expr=str(expr), var=str(sym), lower=str(lower), upper="oo", digits=d)
            bad.add_condition("常见原因：级数发散（数值加速只对收敛级数有效），或通项无法数值化")
            return bad

        check = partial2 - partial
        slow = bool(abs(check) > mpmath.mpf(10) ** (-6) * max(mpmath.mpf(1), abs(partial)))

        priority = ["euler_maclaurin", "nsum[r]", "nsum[s]", "nsum[d]"]
        chosen_name, chosen_val, chosen_err, chosen_note = None, None, None, None
        for p in priority:
            for name, val, err, note in usable:
                if name == p:
                    chosen_name, chosen_val, chosen_err, chosen_note = name, val, err, note
                    break
            if chosen_name:
                break

        # 路线可信度分级：nsum 的 'd'（直接求和）与 's'（Shanks 变换）对慢收敛级数
        # 只有 3~5 位精度（实测 Σ1/n²：d 偏 2.8e-3、s 偏 5.6e-5），而 'r'（Richardson）
        # 与 Euler–Maclaurin 尾项可达 20 位以上。低精度路线只作参考记录、不参与误差聚合，
        # 否则会把正确的 EM/Richardson 结果误标成"不可信"（实测 abs_error 被抬到 2.2e-3）。
        trusted_names = {"euler_maclaurin", "nsum[r]"}
        trusted = [(n, v) for n, v, _e, _n in usable if n in trusted_names]
        reference = [
            {
                "route": n,
                "value": mpmath.nstr(v, min(d + 3, 30)),
                "error": _num(e),
                "used_for_error": n in trusted_names,
            }
            for n, v, e, _note in usable
        ]
        if len(trusted) >= 2:
            spread = max([abs(v - chosen_val) for n, v in trusted if n != chosen_name], default=mpmath.mpf(0))
        else:
            spread = mpmath.mpf(0)  # 只有一条高精度路线可用时不虚构偏差
        # 低精度路线的"确认"判定：nsum[s]/nsum[d] 的固有精度有限，只要它们的偏差
        # 落在各自固有精度内，就说明结果被独立复现（不抬高误差界，但可解除慢收敛警报）。
        rough_tol = {"nsum[s]": mpmath.mpf("1e-4"), "nsum[d]": mpmath.mpf("1e-2")}
        confirmations = [
            n
            for n, v, _e, _note in usable
            if n in rough_tol and abs(v - chosen_val) <= rough_tol[n] * max(mpmath.mpf(1), abs(chosen_val))
        ]
        errs = [e for e in [chosen_err, spread] if e is not None]
        abs_err = max(errs) if errs else abs(chosen_val) * mpmath.mpf(10) ** (-d) + mpmath.mpf(10) ** (-d)
        if abs_err == 0:
            abs_err = abs(chosen_val) * mpmath.mpf(10) ** (-d) + mpmath.mpf(10) ** (-d)
        rel_err = float(abs_err / max(mpmath.mpf(1), abs(chosen_val)))
        # 慢收敛告警只在"既没有第二条高精度路线一致确认、也没有低精度路线复现"时才升级，
        # 避免对 Σ1/n² 这类刻意慢收敛的收敛级数误报。
        confirmed_routes = [n for n, _v in trusted if n != chosen_name] + confirmations
        trusted_consistent = spread <= mpmath.mpf("1e-6") * max(mpmath.mpf(1), abs(chosen_val))
        slow_suspect = slow and not (trusted_consistent and confirmed_routes)

    r.set_result(_sp_float(chosen_val, d))
    r.set_raw(
        digits=d,
        abs_error=_num(abs_err),
        relative_error=_num(rel_err),
        is_approximate=True,
        terms_used=N,
        routes=reference,
        chosen_route=chosen_note,
        trusted_routes=[n for n, _v in trusted],
        confirming_routes=confirmations,
        failed_routes=fails[:4] or None,
        partial_sum_at_N=mpmath.nstr(partial, min(d + 3, 30)),
        partial_sum_growth=_num(check),
        slow_convergence=slow,
        suspected_closed_form=_nsimplify_hint(chosen_val, d),
        abs_error_note="abs_error = max(所选路线误差界, 高精度路线(nsum[r]/Euler–Maclaurin)之间的最大偏差)；nsum[s]/nsum[d] 只作参考，不参与误差聚合",
    )
    methods = [note for _n, _v, _e, note in usable]
    _verify_numeric(
        r,
        methods,
        f"数值求和：采用「{chosen_note}」= {mpmath.nstr(chosen_val, min(d + 5, 25))}，"
        f"各路线最大偏差 {_num(spread)}，误差界 {_num(abs_err)}。"
        "**数值加速只对收敛级数有效**，发散级数的任何截断和都没有意义；结论不构成解析证明。",
    )
    r.add_condition("mpmath.nsum 的加速（Richardson/Shanks）只对**收敛**级数有效；发散级数会给出无意义的数值")
    r.add_condition("Euler–Maclaurin 尾项要求通项对求和指标是光滑函数（含 floor/Mod/(-1)^n/factorial 时自动跳过）")
    if not smooth:
        r.add_warning("通项不是光滑函数（含 floor/Mod/(-1)^n 等），已跳过 Euler–Maclaurin 尾项路线，仅用 nsum 加速")
    if slow:
        if slow_suspect:
            what = "且高精度路线之间不一致" if not trusted_consistent else "且没有别的路线复现"
            r.add_warning(f"部分和仍在增长（S(2N)-S(N) ≈ {_num(check)}），收敛很慢{what}，级数可能发散，结果可信度下降")
        else:
            r.add_warning(f"部分和仍在增长（S(2N)-S(N) ≈ {_num(check)}），收敛很慢；已有独立路线（" + "、".join(confirmed_routes) + "）交叉确认")
    if slow_suspect:
        r.add_condition("慢收敛/发散级数：数值加速结果只是截断近似，发散级数下无意义")
    if rel_err > 1e-2:
        r.partial(f"各求和路线偏差过大（相对 {_num(rel_err)}），数值结果不可信")
    return r


# --------------------------------------------------------------------------------------
# 9. numeric_matrix —— numpy 浮点线代（符号线代结果的交叉验证）
# --------------------------------------------------------------------------------------


@op("numeric_matrix", "math_numeric_matrix")
def numeric_matrix(matrix: Any = None, digits: int = 15) -> MathResult:
    """用 numpy 浮点线代（行列式/逆/特征值/秩/条件数）交叉验证符号线代结果。"""
    d = _check_digits(digits)
    if matrix is None:
        raise ValueError('numeric_matrix 需要 matrix，如 matrix="1,2;3,4" 或 matrix=[[1,2],[3,4]]')
    try:
        M = A.parse_matrix(text=matrix) if isinstance(matrix, str) else A.parse_matrix(rows=matrix)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"矩阵无法解析: {exc}") from None
    if not isinstance(M, sp.MatrixBase):
        raise ValueError("matrix 参数无法解析成矩阵")

    r = MathResult.ok("numeric_matrix", method="numpy 浮点数值计算（LAPACK：LU/QR/特征值分解）；与 SymPy 精确值交叉对比")
    r.set_input(matrix=[[A.to_text(x) for x in row] for row in M.tolist()], digits=d)

    if M.free_symbols:
        bad = MathResult.unsolved("numeric_matrix", "矩阵含自由符号 " + ", ".join(str(s) for s in M.free_symbols) + "，numeric_matrix 只处理数值矩阵")
        bad.set_input(matrix=[[A.to_text(x) for x in row] for row in M.tolist()], digits=d)
        return bad

    import numpy as np

    try:
        arr = np.array([[float(sp.N(x, 25)) for x in row] for row in M.tolist()], dtype=float)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"矩阵元素无法数值化: {exc}") from None

    square = arr.shape[0] == arr.shape[1]
    info: dict[str, Any] = {"shape": [int(arr.shape[0]), int(arr.shape[1])], "digits": d}
    methods = ["numpy 浮点数值计算"]
    det_sym_dev = None
    eig_resid = None

    if square:
        det_np = float(np.linalg.det(arr))
        info["determinant"] = _num(det_np)
        try:
            det_sym = sp.N(sp.det(M), 30)
            info["determinant_exact"] = A.to_text(det_sym)
            det_sym_dev = abs(complex(det_sym) - det_np)
            info["determinant_deviation"] = _num(det_sym_dev)
            methods.append("SymPy 精确行列式（独立路线）")
        except Exception:  # noqa: BLE001
            pass
        info["rank"] = int(np.linalg.matrix_rank(arr))
        try:
            info["condition_number"] = _num(float(np.linalg.cond(arr)))
        except Exception:  # noqa: BLE001
            pass
        eigvals = None
        try:
            eigvals = np.linalg.eigvals(arr)
            vals = [complex(x) for x in np.asarray(eigvals).reshape(-1)]
            info["eigenvalues"] = [(_num(v.real) if abs(v.imag) < 1e-12 * max(1.0, abs(v.real)) else [ _num(v.real), _num(v.imag)]) for v in vals]
            # 独立验证：把特征值代回 det(A - λI) 检查（特征多项式残差）
            worst = 0.0
            for v in vals:
                lam = sp.Float(v.real, 20) + sp.I * sp.Float(v.imag, 20) if abs(v.imag) > 0 else sp.Float(v.real, 20)
                try:
                    resid = abs(complex(sp.N(sp.det(M - lam * sp.eye(M.shape[0])), 25)))
                except Exception:  # noqa: BLE001
                    continue
                scale = max(1.0, abs(complex(det_np)))
                worst = max(worst, float(resid) / scale)
            eig_resid = worst
            info["eigen_residual_max"] = _num(worst)
            methods.append("特征多项式残差 det(A−λI)（独立验证）")
        except Exception:  # noqa: BLE001
            eigvals = None
        if det_np != 0.0:
            try:
                inv = np.linalg.inv(arr)
                info["inverse"] = [[_num(x) for x in row] for row in np.asarray(inv).tolist()]
                eye = arr @ inv
                info["inverse_residual"] = _num(float(np.max(np.abs(eye - np.eye(arr.shape[0])))))
                methods.append("A·A⁻¹−I 残差（独立验证）")
            except Exception as exc:  # noqa: BLE001
                info["inverse"] = None
                info["inverse_error"] = str(exc)
        else:
            info["inverse"] = None
            info["inverse_error"] = "行列式为 0，矩阵奇异，无逆"
    else:
        info["rank"] = int(np.linalg.matrix_rank(arr))
        info["determinant"] = None
        info["determinant_note"] = "非方阵无行列式"

    # 方阵的主结果是行列式，非方阵退化为秩（都是数值近似，raw 里有全套量）
    if square and info.get("determinant") is not None:
        r.set_result(_sp_float(mpmath.mpf(str(info["determinant"])), d))
    else:
        r.set_result(sp.Integer(int(info.get("rank", 0))))
    # 精度与误差信息（约定：数值结果必须自带 digits 与 abs_error）
    resid_candidates = [x for x in (det_sym_dev, eig_resid, info.get("inverse_residual")) if x is not None]
    if resid_candidates:
        abs_err = max(float(x) for x in resid_candidates)
        err_note = "abs_error = 各独立残差（SymPy 精确行列式偏差 / det(A−λI) 残差 / A·A⁻¹−I 残差）的最大值"
    else:
        abs_err = float(np.finfo(float).eps) * max(1.0, float(np.max(np.abs(arr))))
        err_note = "abs_error = float64 机器精度 × 矩阵元素最大绝对值（无可用独立残差）"
    info["abs_error"] = _num(abs_err)
    info["abs_error_note"] = err_note
    r.set_raw(**info)
    _verify_numeric(
        r,
        methods,
        "numpy 浮点线代结果（双精度，约 15–16 位）；与 SymPy 精确行列式/特征多项式残差交叉对比："
        f"行列式偏差 {_num(det_sym_dev)}，特征值残差 {_num(eig_resid)}。"
        "浮点线代对病态矩阵（条件数大）会严重失真，**不是解析证明**。",
    )
    cond = info.get("condition_number")
    if cond is not None:
        try:
            # _num 会把 inf 保留成字符串 "inf"，直接对字符串用 :.3g 会抛 ValueError
            cond_f = float(cond)
        except (TypeError, ValueError):
            cond_f = float("inf") if str(cond).strip().lower().endswith("inf") else None
        if cond_f is not None and cond_f > 1e10:
            r.add_warning(f"矩阵条件数约 {cond_f:.3g}，浮点结果高度不可信，请以符号计算为准")
    if not square:
        r.add_condition("非方阵：只给出秩等基本量，无行列式/特征值")
    return r
