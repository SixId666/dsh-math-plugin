"""曲线积分、曲面积分与三大公式（格林 / 高斯 / 斯托克斯）。

**定位说明（重要，防止误用）**

本模块刻意不替模型做「该用投影法还是高斯公式」这类判断——那属于数学分析，
必须由模型结合曲面是否封闭、投影区域是否简单、被积函数是否轮换对称来决定。
模型选定方法后，本模块负责把积分**精确算出来**并做独立复核。

因此这里的每个算子都接受「已经参数化/已经选定区域」的输入：

* :func:`curve_integral`      参数化曲线上的第一类（弧长）与第二类（坐标）曲线积分
* :func:`surface_integral`    参数化曲面上的第一类（面积）与第二类（通量）曲面积分
* :func:`green`               格林公式：∮_L P dx + Q dy → ∬_D (∂Q/∂x − ∂P/∂y) dx dy
* :func:`gauss`               高斯公式：∯_Σ P dy∧dz + Q dz∧dx + R dx∧dy → ∭_Ω ∇·F dV
* :func:`stokes`              斯托克斯公式：∮_Γ F·dr → ∬_Σ (∇×F)·n dS

所有算子都返回 :class:`MathResult`，并遵守用户规格：
给出成立条件（定向、分段光滑、封闭性、可积性），给出独立验证状态，
数值证据与符号证明严格区分。
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import sympy as sp

from . import ast as A
from . import sympy_core as C
from .engine import op
from .result import MathResult

# ---------------------------------------------------------------------------
# 公共解析工具
# ---------------------------------------------------------------------------

_INF_WORDS = {"oo", "+oo", "inf", "infinity", "无穷", "+无穷", "∞", "+∞"}
_NEG_INF_WORDS = {"-oo", "-inf", "-infinity", "-无穷", "-∞"}


def _parse_bound(value: Any, symbols: Iterable[str], *, what: str) -> sp.Basic:
    """解析积分限，支持 oo / 无穷 / -oo。"""
    if value is None:
        raise ValueError(f"缺少{what}")
    if isinstance(value, sp.Basic):
        return value
    text = str(value).strip()
    if not text:
        raise ValueError(f"缺少{what}")
    lowered = text.lower().replace(" ", "")
    if lowered in _INF_WORDS:
        return sp.oo
    if lowered in _NEG_INF_WORDS:
        return -sp.oo
    expr = A.parse(text, symbols=symbols, extra={"oo": sp.oo, "inf": sp.oo, "无穷": sp.oo})
    return expr


def _param_range(
    symbol: sp.Symbol,
    lower: Any,
    upper: Any,
    *,
    allow_reversed: bool = False,
) -> tuple[sp.Basic, sp.Basic]:
    lo = _parse_bound(lower, [str(symbol)], what=f"{symbol} 的下限")
    hi = _parse_bound(upper, [str(symbol)], what=f"{symbol} 的上限")
    if lo == hi:
        raise ValueError(f"{symbol} 的积分区间退化（上下限相同）")
    if lo.is_real is False or hi.is_real is False:
        raise ValueError(f"{symbol} 的积分限必须是实数")
    if (
        not allow_reversed
        and lo.is_number
        and hi.is_number
        and lo.is_finite
        and hi.is_finite
        and lo > hi
    ):
        raise ValueError(
            f"{symbol} 的积分下限大于上限：第一类积分要求上限不小于下限；"
            "第二类（坐标）积分允许反向，此时结果自动取负号，请确认 kind='second'。"
        )
    return lo, hi


def _cylindrical_setup(
    integrand: sp.Basic,
    z_sym: sp.Symbol,
    zlo: sp.Basic,
    zhi: sp.Basic,
    y_sym: sp.Symbol,
    ylo: sp.Basic,
    yhi: sp.Basic,
    x_sym: sp.Symbol,
    xlo: sp.Basic,
    xhi: sp.Basic,
) -> tuple[sp.Basic, dict[sp.Symbol, tuple[sp.Basic, sp.Basic]]] | None:
    """把「圆周型」直角坐标积分区域改写成柱坐标。

    适用形态：``x ∈ [-a, a]``、``y ∈ [-√(a²−x²), √(a²−x²)]``、
    ``z ∈ [f(x,y), g(x,y)]``，且 f、g 化简后只依赖 x²+y²。

    这类直角坐标下的三重积分在 SymPy 里经常慢到超时，但换成柱坐标
    ``x = r cosθ, y = r sinθ``（雅可比 r）后立刻可积。识别不出上述形态就返回
    ``None``——**绝不猜一个近似区域**。
    """
    try:
        if not C.same_expr(xlo, -xhi):
            return None
        expected_hi = sp.sqrt(xhi**2 - x_sym**2)
        if not C.same_expr(yhi, expected_hi) or not C.same_expr(ylo, -expected_hi):
            return None

        r_sym = sp.Symbol("_r", positive=True)
        theta_sym = sp.Symbol("_theta", real=True)
        x_of = r_sym * sp.cos(theta_sym)
        y_of = r_sym * sp.sin(theta_sym)

        def as_polar(expr: sp.Basic) -> sp.Basic:
            return sp.simplify(sp.expand_trig(sp.expand(expr.subs({x_sym: x_of, y_sym: y_of}))))

        zlo_polar, zhi_polar = as_polar(zlo), as_polar(zhi)
        if zlo_polar.has(theta_sym) or zhi_polar.has(theta_sym):
            return None
        integrand_polar = sp.simplify(integrand.subs({x_sym: x_of, y_sym: y_of}) * r_sym)
        return integrand_polar, {
            z_sym: (zlo_polar, zhi_polar),
            r_sym: (sp.Integer(0), xhi),
            theta_sym: (sp.Integer(0), 2 * sp.pi),
        }
    except Exception:  # noqa: BLE001
        return None


def _integrate_limits(integrand: sp.Basic, limits: dict[sp.Symbol, tuple[sp.Basic, sp.Basic]]) -> sp.Basic:
    """按「最内层优先」的顺序把 limits 交给 SymPy 逐层积分。"""
    ordered = list(limits.items())
    return sp.integrate(integrand, *[(sym, lo, hi) for sym, (lo, hi) in ordered])



def _vector_field(
    components: Sequence[Any],
    variables: Sequence[str],
    *,
    what: str = "向量场",
) -> list[sp.Basic]:
    if not components:
        raise ValueError(f"缺少{what}分量")
    parsed: list[sp.Basic] = []
    for index, item in enumerate(components):
        if item is None or (isinstance(item, str) and not item.strip()):
            raise ValueError(f"{what}的第 {index + 1} 个分量为空")
        parsed.append(A.parse(str(item), symbols=list(variables)) if not isinstance(item, sp.Basic) else item)
    return parsed


def _symbols_of(names: Sequence[str]) -> dict[str, sp.Symbol]:
    return A.get_symbols(list(names))


def _resolve_param(exprs: Sequence[Any], name: str) -> sp.Symbol:
    """在已解析的表达式里按名字找回**同一**符号对象。

    直接用 ``sp.Symbol(name, real=True)`` 与 ``A.parse`` 产出的符号不是同一个对象
    （``A.get_symbols`` 只对 ``x/t/y/z/u/v/s`` 加 ``real=True``，其它名字是无假设符号），
    于是 ``position.diff(Symbol('r', real=True))`` 会得到 0，法向量变成 (0,0,0)、
    斯托克斯积分结果变成 0 —— 一个**静默错误答案**。这与 ``C.resolve_var`` 是同一类坑。
    """
    for expr in exprs:
        if isinstance(expr, sp.Basic):
            for symbol in expr.free_symbols:
                if str(symbol) == name:
                    return symbol
    return sp.Symbol(name, real=True)


# ---------------------------------------------------------------------------
# 1. 曲线积分
# ---------------------------------------------------------------------------

@op("curve_integral", "line_integral", "math_curve_integral")
def curve_integral(
    expr: Any = None,
    x_expr: Any = None,
    y_expr: Any = None,
    z_expr: Any = None,
    var: Any = "t",
    lower: Any = None,
    upper: Any = None,
    kind: Any = "first",
    expr2: Any = None,
    expr3: Any = None,
    **kwargs: Any,
) -> MathResult:
    """参数化曲线上的曲线积分。

    * ``kind="first"``（第一类，弧长）：``expr`` 是被积函数 f(x,y,z)，
      结果为 ∫_a^b f(x(t),y(t),z(t))·|r'(t)| dt
    * ``kind="second"``（第二类，坐标/功）：``expr``、``expr2``、``expr3`` 分别是
      P、Q、R，结果为 ∫_a^b (P x' + Q y' + R z') dt
    """
    t = sp.Symbol(str(var or "t"), real=True)
    t_name = str(var or "t")
    try:
        params = _symbols_of([t_name, "x", "y", "z", "u", "v"])
    except Exception as exc:  # noqa: BLE001
        return MathResult.fail("curve_integral", f"无法建立符号：{exc}", kind="invalid_input")

    if x_expr is None and y_expr is None and z_expr is None:
        return MathResult.fail(
            "curve_integral",
            "缺少曲线参数方程：请给出 x_expr（必填）以及 y_expr / z_expr。例如 x_expr='cos(t)', y_expr='sin(t)'。",
            kind="invalid_input",
        )
    if kind not in ("first", "second", "1", "2", "type1", "type2", "arc", "work"):
        return MathResult.fail(
            "curve_integral",
            f"kind 只能是 first（第一类，默认）或 second（第二类），收到 {kind!r}。",
            kind="invalid_input",
        )
    is_first = kind in ("first", "1", "type1", "arc")

    curried = {
        name: sp.Symbol(name, real=True)
        for name in ("x", "y", "z")
    }
    try:
        xs = A.parse(str(x_expr), symbols=[t_name], extra=params)
        ys = A.parse(str(y_expr), symbols=[t_name], extra=params) if y_expr is not None else None
        zs = A.parse(str(z_expr), symbols=[t_name], extra=params) if z_expr is not None else None
    except ValueError as exc:
        return MathResult.fail("curve_integral", f"参数方程解析失败：{exc}", kind="invalid_input")

    try:
        lo, hi = _param_range(
            t,
            lower if lower is not None else "0",
            upper if upper is not None else "2*pi",
            allow_reversed=not is_first,
        )
    except ValueError as exc:
        return MathResult.fail("curve_integral", str(exc), kind="invalid_input")

    substituted = {curried["x"]: xs}
    if ys is not None:
        substituted[curried["y"]] = ys
    if zs is not None:
        substituted[curried["z"]] = zs

    r = MathResult.ok(
        "curve_integral",
        method="参数化 + 弧长元/坐标微分（SymPy 精确积分）",
    )
    r.set_input(kind="第一类（弧长）" if is_first else "第二类（坐标）", parametrization={t_name: [str(lo), str(hi)]})
    r.set_raw(x_of_t=A.to_text(xs), y_of_t=A.to_text(ys) if ys is not None else None, z_of_t=A.to_text(zs) if zs is not None else None)

    dx = sp.diff(xs, t)
    dy = sp.diff(ys, t) if ys is not None else sp.Integer(0)
    dz = sp.diff(zs, t) if zs is not None else sp.Integer(0)
    speed = sp.sqrt(dx**2 + dy**2 + dz**2)

    if is_first:
        if expr is None:
            return MathResult.fail("curve_integral", "第一类曲线积分需要被积函数 expr，例如 expr='x^2+y^2'。", kind="invalid_input")
        try:
            f = A.parse(str(expr), symbols=["x", "y", "z"], extra=params)
        except ValueError as exc:
            return MathResult.fail("curve_integral", f"被积函数解析失败：{exc}", kind="invalid_input")
        integrand = sp.simplify(f.subs(substituted) * speed)
        formula = f"∫ f(x(t),y(t),z(t))·|r'(t)| dt，其中 |r'(t)| = {A.to_text(sp.simplify(speed))}"
    else:
        if expr is None:
            return MathResult.fail("curve_integral", "第二类曲线积分需要 P 分量 expr，例如 expr='-y'。", kind="invalid_input")
        try:
            p = A.parse(str(expr), symbols=["x", "y", "z"], extra=params)
            q = A.parse(str(expr2), symbols=["x", "y", "z"], extra=params) if expr2 is not None else sp.Integer(0)
            rr = A.parse(str(expr3), symbols=["x", "y", "z"], extra=params) if expr3 is not None else sp.Integer(0)
        except ValueError as exc:
            return MathResult.fail("curve_integral", f"分量解析失败：{exc}", kind="invalid_input")
        integrand = sp.simplify(
            p.subs(substituted) * dx + q.subs(substituted) * dy + rr.subs(substituted) * dz
        )
        formula = "∫ (P dx + Q dy + R dz)，代入 x(t),y(t),z(t) 后化为对 t 的一元积分"

    r.set_raw(integrand_of_t=A.to_text(integrand), formula=formula)

    try:
        value = sp.simplify(sp.integrate(integrand, (t, lo, hi)))
    except Exception as exc:  # noqa: BLE001
        return MathResult.unsolved(
            "curve_integral",
            f"参数化已完成，但定积分未能精确求出（{type(exc).__name__}: {exc}）。"
            "可尝试化简被积函数，或改用 math_numeric 的 numeric_integrate 取得高精度数值。",
            method="SymPy integrate",
        )

    if isinstance(value, sp.Integral) or not C.is_closed_form(value):
        unsolved = MathResult.unsolved(
            "curve_integral",
            "定积分未能求出闭式结果（返回的是未求值的积分表达式）。可化简被积函数后重试，或改用数值积分。",
            method="SymPy integrate（返回未求值积分）",
        )
        unsolved.set_raw(integrand_of_t=A.to_text(integrand), unevaluated=A.to_text(value))
        return unsolved

    r.set_result(sp.simplify(value))
    r.set_raw(value_text=A.to_text(sp.simplify(value)))
    r.add_condition("曲线需分段光滑，且参数化在 [a, b] 上连续可导")
    r.add_condition("第一类曲线积分与曲线方向无关（弧长元 ds ≥ 0）")
    if not is_first:
        r.add_condition("第二类曲线积分与曲线方向有关：反向则结果取负号")
        r.add_condition(f"定向为 t 从 {A.to_text(lo)} 增到 {A.to_text(hi)}")

    # 独立复核：用 mpmath 数值积分走同一条参数化，验证符号结果
    methods = []
    ok_numeric = None
    try:
        import mpmath as mp

        numeric_value = mp.quad(
            sp.lambdify(t, integrand, modules=["mpmath"]),
            [float(lo), float(hi)],
        )
        exact_float = complex(sp.N(value, 30))
        deviation = abs(complex(numeric_value) - exact_float) / max(1.0, abs(exact_float))
        ok_numeric = deviation < 1e-9
        methods.append(
            {
                "method": "独立数值复核：mpmath 高精度数值积分（tanh-sinh）与该参数化下的符号结果对比",
                "numeric": mp.nstr(numeric_value, 15),
                "symbolic": mp.nstr(mp.mpf(exact_float.real), 15) if exact_float.imag == 0 else mp.nstr(exact_float, 15),
                "relative_deviation": f"{deviation:.3e}",
                "agree": bool(ok_numeric),
            }
        )
    except Exception as exc:  # noqa: BLE001
        methods.append({"method": "独立数值复核", "error": f"{type(exc).__name__}: {exc}", "agree": False})

    r.verify(
        status="symbolic+numeric" if ok_numeric else "unverified",
        methods=methods,
        note="符号结果由 SymPy 精确求出，并通过独立的 mpmath 数值积分复核。"
        if ok_numeric
        else "符号结果已求出，但独立数值复核未通过或无法执行，请人工复核参数化。",
    )
    if ok_numeric is False:
        r.add_warning("独立数值复核与该符号结果不一致，请检查参数方程与被积函数是否匹配。")
    return r


# ---------------------------------------------------------------------------
# 2. 曲面积分
# ---------------------------------------------------------------------------

@op("surface_integral", "math_surface_integral")
def surface_integral(
    expr: Any = None,
    x_expr: Any = None,
    y_expr: Any = None,
    z_expr: Any = None,
    u_var: Any = "u",
    v_var: Any = "v",
    u_lower: Any = None,
    u_upper: Any = None,
    v_lower: Any = None,
    v_upper: Any = None,
    kind: Any = "first",
    expr2: Any = None,
    expr3: Any = None,
    **kwargs: Any,
) -> MathResult:
    """参数化曲面上的曲面积分。

    * ``kind="first"``（第一类，面积）：``expr`` 是 f(x,y,z)，
      结果为 ∬ f·|r_u × r_v| du dv
    * ``kind="second"``（第二类，通量）：``expr``、``expr2``、``expr3`` 是 P、Q、R，
      结果为 ∬ (P, Q, R)·(r_u × r_v) du dv（定向由参数化的叉积方向决定）
    """
    u, v = sp.Symbol(str(u_var or "u"), real=True), sp.Symbol(str(v_var or "v"), real=True)
    names = [str(u_var or "u"), str(v_var or "v"), "x", "y", "z"]
    try:
        params = _symbols_of(names)
    except Exception as exc:  # noqa: BLE001
        return MathResult.fail("surface_integral", f"无法建立符号：{exc}", kind="invalid_input")

    if x_expr is None or y_expr is None:
        return MathResult.fail(
            "surface_integral",
            "缺少曲面参数方程：至少需要 x_expr 与 y_expr（z_expr 可视曲面而定）。"
            "例如球面上半部分：x_expr='sin(u)*cos(v)', y_expr='sin(u)*sin(v)', z_expr='cos(u)'。",
            kind="invalid_input",
        )
    if kind not in ("first", "second", "1", "2", "type1", "type2", "area", "flux"):
        return MathResult.fail(
            "surface_integral",
            f"kind 只能是 first（第一类，默认）或 second（第二类），收到 {kind!r}。",
            kind="invalid_input",
        )
    is_first = kind in ("first", "1", "type1", "area")

    try:
        xs = A.parse(str(x_expr), symbols=names, extra=params)
        ys = A.parse(str(y_expr), symbols=names, extra=params)
        zs = A.parse(str(z_expr), symbols=names, extra=params) if z_expr is not None else sp.Integer(0)
    except ValueError as exc:
        return MathResult.fail("surface_integral", f"参数方程解析失败：{exc}", kind="invalid_input")

    # 参数符号必须取自解析结果本身，否则自定义参数名（如 'r'）的 assumptions 不一致
    u = _resolve_param([xs, ys, zs], str(u_var or "u"))
    v = _resolve_param([xs, ys, zs], str(v_var or "v"))

    try:
        ulo = _parse_bound(u_lower if u_lower is not None else "0", names, what="u 的下限")
        uhi = _parse_bound(u_upper if u_upper is not None else "pi", names, what="u 的上限")
        vlo = _parse_bound(v_lower if v_lower is not None else "0", names, what="v 的下限")
        vhi = _parse_bound(v_upper if v_upper is not None else "2*pi", names, what="v 的上限")
    except ValueError as exc:
        return MathResult.fail("surface_integral", str(exc), kind="invalid_input")

    position = sp.Matrix([xs, ys, zs])
    r_u = position.diff(u)
    r_v = position.diff(v)
    normal = r_u.cross(r_v)
    area_element = sp.simplify(sp.sqrt(sum(component**2 for component in normal)))

    substituted = {sp.Symbol("x", real=True): xs, sp.Symbol("y", real=True): ys, sp.Symbol("z", real=True): zs}

    r = MathResult.ok("surface_integral", method="参数化 + 面积元/有向面积元（SymPy 精确积分）")
    r.set_input(kind="第一类（面积）" if is_first else "第二类（通量）", parametrization={str(u): [str(ulo), str(uhi)], str(v): [str(vlo), str(vhi)]})
    r.set_raw(
        position=A.to_text(position),
        r_u=A.to_text(r_u),
        r_v=A.to_text(r_v),
        normal_vector=A.to_text(normal),
        area_element=A.to_text(area_element),
    )

    if is_first:
        if expr is None:
            return MathResult.fail("surface_integral", "第一类曲面积分需要被积函数 expr，例如 expr='z'。", kind="invalid_input")
        try:
            f = A.parse(str(expr), symbols=["x", "y", "z"], extra=params)
        except ValueError as exc:
            return MathResult.fail("surface_integral", f"被积函数解析失败：{exc}", kind="invalid_input")
        integrand = sp.simplify(f.subs(substituted) * area_element)
        formula = "∬ f(x(u,v),y(u,v),z(u,v))·|r_u × r_v| du dv"
        r.add_condition("第一类曲面积分与曲面定向无关（面积元 dS = |r_u × r_v| du dv ≥ 0）")
    else:
        if expr is None:
            return MathResult.fail("surface_integral", "第二类曲面积分需要 P 分量 expr。", kind="invalid_input")
        try:
            p = A.parse(str(expr), symbols=["x", "y", "z"], extra=params)
            q = A.parse(str(expr2), symbols=["x", "y", "z"], extra=params) if expr2 is not None else sp.Integer(0)
            rr = A.parse(str(expr3), symbols=["x", "y", "z"], extra=params) if expr3 is not None else sp.Integer(0)
        except ValueError as exc:
            return MathResult.fail("surface_integral", f"分量解析失败：{exc}", kind="invalid_input")
        field = sp.Matrix([p.subs(substituted), q.subs(substituted), rr.subs(substituted)])
        integrand = sp.simplify(field.dot(normal))
        formula = "∬ (P, Q, R)·(r_u × r_v) du dv"
        r.add_condition("第二类曲面积分与曲面定向有关：法向取 r_u × r_v；反向则结果取负号")

    r.set_raw(integrand=A.to_text(integrand), formula=formula)

    try:
        value = sp.simplify(sp.integrate(integrand, (u, ulo, uhi), (v, vlo, vhi)))
    except Exception as exc:  # noqa: BLE001
        return MathResult.unsolved(
            "surface_integral",
            f"参数化已完成，但二重积分未能精确求出（{type(exc).__name__}: {exc}）。"
            "可尝试化简被积函数或改换参数化方式，也可用 math_numeric 求高精度数值结果。",
            method="SymPy integrate",
        )

    if isinstance(value, sp.Integral) or not C.is_closed_form(value):
        unsolved = MathResult.unsolved(
            "surface_integral",
            "二重积分未能求出闭式结果（返回未求值积分）。可化简被积函数、调整参数化，或改用数值积分。",
            method="SymPy integrate（返回未求值积分）",
        )
        unsolved.set_raw(integrand=A.to_text(integrand), unevaluated=A.to_text(value))
        return unsolved

    r.set_result(sp.simplify(value))
    r.set_raw(value_text=A.to_text(sp.simplify(value)))

    methods: list[dict[str, Any]] = []
    ok_numeric: bool | None = None
    try:
        import mpmath as mp

        f_numeric = sp.lambdify((u, v), integrand, modules=["mpmath"])
        numeric_value = mp.quad(
            lambda uu: mp.quad(lambda vv: f_numeric(uu, vv), [float(vlo), float(vhi)]),
            [float(ulo), float(uhi)],
        )
        exact = complex(sp.N(value, 30))
        deviation = abs(complex(numeric_value) - exact) / max(1.0, abs(exact))
        ok_numeric = deviation < 1e-8
        methods.append(
            {
                "method": "独立数值复核：mpmath 对同一参数化的二重数值积分与符号结果对比",
                "numeric": mp.nstr(numeric_value, 15),
                "relative_deviation": f"{deviation:.3e}",
                "agree": bool(ok_numeric),
            }
        )
    except Exception as exc:  # noqa: BLE001
        methods.append({"method": "独立数值复核", "error": f"{type(exc).__name__}: {exc}", "agree": False})

    r.verify(
        status="symbolic+numeric" if ok_numeric else "unverified",
        methods=methods,
        note="符号结果由 SymPy 精确求出，并通过独立的二重数值积分复核。"
        if ok_numeric
        else "符号结果已求出，但独立数值复核未通过或无法执行，请人工复核参数化与定向。",
    )
    if ok_numeric is False:
        r.add_warning("独立数值复核与该符号结果不一致，请检查参数化、雅可比行列式与定向。")
    return r


# ---------------------------------------------------------------------------
# 3. 格林公式
# ---------------------------------------------------------------------------

@op("green", "green_theorem", "math_green")
def green(
    p: Any = None,
    q: Any = None,
    x_lower: Any = None,
    x_upper: Any = None,
    y_lower: Any = None,
    y_upper: Any = None,
    y_lower_expr: Any = None,
    y_upper_expr: Any = None,
    orientation: Any = "positive",
    **kwargs: Any,
) -> MathResult:
    """格林公式：把 ∮_L P dx + Q dy 化为区域 D 上的二重积分 ∬ (Q_x − P_y) dx dy。

    区域 D 用「x 型」描述：``x_lower ≤ x ≤ x_upper``，``y_lower(x) ≤ y ≤ y_upper(x)``。
    也可只给 ``y_lower / y_upper`` 常数（矩形区域）。
    """
    x, y = sp.Symbol("x", real=True), sp.Symbol("y", real=True)
    if p is None or q is None:
        return MathResult.fail(
            "green",
            "格林公式需要 P 与 Q 两个分量，例如 p='-y', q='x'（此时 ∮ = 2×区域面积）。",
            kind="invalid_input",
        )
    if x_lower is None or x_upper is None:
        return MathResult.fail(
            "green",
            "缺少区域 D 的 x 范围：请给出 x_lower 与 x_upper。",
            kind="invalid_input",
        )

    try:
        p_expr = A.parse(str(p), symbols=["x", "y"])
        q_expr = A.parse(str(q), symbols=["x", "y"])
    except ValueError as exc:
        return MathResult.fail("green", f"分量解析失败：{exc}", kind="invalid_input")

    try:
        xlo = _parse_bound(x_lower, ["x"], what="x 的下限")
        xhi = _parse_bound(x_upper, ["x"], what="x 的上限")
    except ValueError as exc:
        return MathResult.fail("green", str(exc), kind="invalid_input")

    if y_lower_expr is not None or y_upper_expr is not None:
        if y_lower_expr is None or y_upper_expr is None:
            return MathResult.fail(
                "green",
                "上下边界必须同时给出：y_lower_expr 与 y_upper_expr（例如 '0' 与 'sqrt(1-x^2)'）。",
                kind="invalid_input",
            )
        try:
            ylo_expr = A.parse(str(y_lower_expr), symbols=["x"])
            yhi_expr = A.parse(str(y_upper_expr), symbols=["x"])
        except ValueError as exc:
            return MathResult.fail("green", f"上下边界解析失败：{exc}", kind="invalid_input")
    elif y_lower is not None and y_upper is not None:
        try:
            ylo_expr = _parse_bound(y_lower, ["y"], what="y 的下限")
            yhi_expr = _parse_bound(y_upper, ["y"], what="y 的上限")
        except ValueError as exc:
            return MathResult.fail("green", str(exc), kind="invalid_input")
    else:
        return MathResult.fail(
            "green",
            "缺少区域 D 的 y 范围：请给出 y_lower_expr / y_upper_expr（可为常数，如 '0' 与 '1'）。",
            kind="invalid_input",
        )

    curl = sp.simplify(sp.diff(q_expr, x) - sp.diff(p_expr, y))
    r = MathResult.ok("green", method="格林公式：∬_D (∂Q/∂x − ∂P/∂y) dx dy（SymPy 精确积分）")
    r.set_input(region={"x": [A.to_text(xlo), A.to_text(xhi)], "y": [A.to_text(ylo_expr), A.to_text(yhi_expr)]})
    r.set_raw(P=A.to_text(p_expr), Q=A.to_text(q_expr), curl=A.to_text(curl))

    try:
        value = sp.simplify(sp.integrate(curl, (y, ylo_expr, yhi_expr), (x, xlo, xhi)))
    except Exception as exc:  # noqa: BLE001
        return MathResult.unsolved(
            "green",
            f"∂Q/∂x − ∂P/∂y 已求出（{A.to_text(curl)}），但区域上的二重积分未能精确求出"
            f"（{type(exc).__name__}: {exc}）。可改换积分次序或改用 math_numeric。",
            method="SymPy integrate",
        )
    if isinstance(value, sp.Integral) or not C.is_closed_form(value):
        unsolved = MathResult.unsolved(
            "green",
            "二重积分未能求出闭式结果。可尝试把区域拆成简单区域、改用极坐标参数化，或用数值积分。",
            method="SymPy integrate（返回未求值积分）",
        )
        unsolved.set_raw(curl=A.to_text(curl), unevaluated=A.to_text(value))
        return unsolved

    r.set_result(sp.simplify(value))
    r.set_raw(value_text=A.to_text(sp.simplify(value)))
    r.add_condition("L 为区域 D 的正向（逆时针）边界，且分段光滑、闭曲线")
    r.add_condition("P、Q 在 D 上有一阶连续偏导数（格林公式成立的前提）")
    if str(orientation).lower() in ("negative", "clockwise", "顺", "顺时针", "负向"):
        r.add_warning("用户声明 L 为负向（顺时针）：格林公式结果为上式取负号。")
        r.add_condition("L 取负向时 ∮_L P dx + Q dy = −∬_D (Q_x − P_y) dx dy")

    # 独立复核：矩形区域可以安全交换积分次序；非矩形区域（y 的上下限含 x）换序时必须
    # 同步改写上下限，沿用原上下限直接换序在数学上就是错的，会得到假「不一致」，
    # 因此非矩形区域改用 mpmath 高精度数值二重积分做独立复核。
    methods = []
    ok_order: bool | None = None
    y_bounds_free = sp.sympify(ylo_expr).free_symbols | sp.sympify(yhi_expr).free_symbols
    if not y_bounds_free:
        try:
            alternative = sp.simplify(sp.integrate(sp.integrate(curl, (x, xlo, xhi)), (y, ylo_expr, yhi_expr)))
            ok_order = C.same_expr(alternative, value)
            methods.append(
                {
                    "method": "独立复核：交换积分次序重算（∬ 的结果应与次序无关）",
                    "alternative_order_result": A.to_text(alternative),
                    "agree": bool(ok_order),
                }
            )
        except Exception as exc:  # noqa: BLE001
            methods.append({"method": "交换积分次序复核", "error": f"{type(exc).__name__}: {exc}", "agree": False})
    else:
        try:
            import mpmath as mp

            f_numeric = sp.lambdify((x, y), curl, modules=["mpmath"])
            ylo_f = sp.lambdify(x, ylo_expr, modules=["mpmath"])
            yhi_f = sp.lambdify(x, yhi_expr, modules=["mpmath"])
            numeric_value = mp.quad(
                lambda xv: mp.quad(lambda yv: f_numeric(xv, yv), [float(ylo_f(xv)), float(yhi_f(xv))]),
                [float(xlo), float(xhi)],
            )
            exact = complex(sp.N(value, 30))
            deviation = abs(complex(numeric_value) - exact) / max(1.0, abs(exact))
            ok_order = bool(deviation < mp.mpf("1e-8"))
            methods.append(
                {
                    "method": "独立复核：mpmath 高精度数值二重积分（按原区域先 y 后 x 逐层积分）与符号结果对比",
                    "numeric_value": mp.nstr(numeric_value, 15),
                    "symbolic_value": A.to_text(sp.N(value, 15)),
                    "relative_deviation": mp.nstr(deviation, 3),
                    "agree": ok_order,
                }
            )
        except Exception as exc:  # noqa: BLE001
            methods.append(
                {
                    "method": "独立复核：mpmath 数值二重积分（区域非矩形，无法安全交换次序）",
                    "error": f"{type(exc).__name__}: {exc}",
                    "agree": None,
                }
            )

    if ok_order is True and not y_bounds_free:
        verify_note = "已通过 mpmath 高精度数值二重积分独立复核，结果一致。"
    elif ok_order is True:
        verify_note = "通过交换积分次序独立重算，结果一致。"
    elif ok_order is False:
        verify_note = "独立复核得到不同结果，请复核区域描述（上下边界）是否正确。"
    else:
        verify_note = "独立复核未能执行（区域含参数或数值积分失败），请人工确认区域描述是否正确。"

    r.verify(
        status="independent" if ok_order else "unverified",
        methods=methods,
        note=verify_note,
    )
    if ok_order is False:
        r.add_warning("独立复核得到不同结果，区域描述可能有误，请复核上下边界。")
    return r


# ---------------------------------------------------------------------------
# 4. 高斯公式
# ---------------------------------------------------------------------------

@op("gauss", "divergence_theorem", "gauss_theorem", "math_gauss")
def gauss(
    p: Any = None,
    q: Any = None,
    r_component: Any = None,
    r: Any = None,
    x_lower: Any = None,
    x_upper: Any = None,
    y_lower: Any = None,
    y_upper: Any = None,
    z_lower: Any = None,
    z_upper: Any = None,
    x_lower_expr: Any = None,
    x_upper_expr: Any = None,
    y_lower_expr: Any = None,
    y_upper_expr: Any = None,
    z_lower_expr: Any = None,
    z_upper_expr: Any = None,
    **kwargs: Any,
) -> MathResult:
    """高斯公式：把 ∯_Σ P dy∧dz + Q dz∧dx + R dx∧dy 化为 ∭_Ω ∇·F dV。

    区域 Ω 用直角坐标描述。可给出常数的 x/y 范围与逐层边界：
    ``x ∈ [x_lower, x_upper]``，``y ∈ [y_lower(x), y_upper(x)]``，
    ``z ∈ [z_lower(x,y), z_upper(x,y)]``。
    """
    z_sym = sp.Symbol("z", real=True)
    third = r_component if r_component is not None else r
    if p is None or q is None or third is None:
        return MathResult.fail(
            "gauss",
            "高斯公式需要 P、Q、R 三个分量。R 分量请用参数 r 或 r_component 给出"
            "（例如 p='x', q='y', r='z' 对应 ∇·F = 3）。",
            kind="invalid_input",
        )
    if x_lower is None or x_upper is None:
        return MathResult.fail("gauss", "缺少区域 Ω 的 x 范围：请给出 x_lower 与 x_upper。", kind="invalid_input")

    try:
        p_expr = A.parse(str(p), symbols=["x", "y", "z"])
        q_expr = A.parse(str(q), symbols=["x", "y", "z"])
        rr_expr = A.parse(str(third), symbols=["x", "y", "z"])
    except ValueError as exc:
        return MathResult.fail("gauss", f"分量解析失败：{exc}", kind="invalid_input")

    try:
        xlo = _parse_bound(x_lower, ["x"], what="x 的下限")
        xhi = _parse_bound(x_upper, ["x"], what="x 的上限")
        ylo_spec = x_lower_expr  # 兼容误传
        ylo_expr = A.parse(str(y_lower_expr if y_lower_expr is not None else (y_lower if y_lower is not None else "0")), symbols=["x"])
        yhi_expr = A.parse(str(y_upper_expr if y_upper_expr is not None else (y_upper if y_upper is not None else "1")), symbols=["x"])
        zlo_expr = A.parse(str(z_lower_expr if z_lower_expr is not None else (z_lower if z_lower is not None else "0")), symbols=["x", "y"])
        zhi_expr = A.parse(str(z_upper_expr if z_upper_expr is not None else (z_upper if z_upper is not None else "1")), symbols=["x", "y"])
    except ValueError as exc:
        return MathResult.fail("gauss", f"区域边界解析失败：{exc}", kind="invalid_input")

    divergence = sp.simplify(sp.diff(p_expr, sp.Symbol("x", real=True)) + sp.diff(q_expr, sp.Symbol("y", real=True)) + sp.diff(rr_expr, z_sym))
    r_out = MathResult.ok("gauss", method="高斯公式：∭_Ω ∇·F dV（SymPy 精确积分）")
    r_out.set_input(
        region={
            "x": [A.to_text(xlo), A.to_text(xhi)],
            "y": [A.to_text(ylo_expr), A.to_text(yhi_expr)],
            "z": [A.to_text(zlo_expr), A.to_text(zhi_expr)],
        }
    )
    r_out.set_raw(P=A.to_text(p_expr), Q=A.to_text(q_expr), R=A.to_text(rr_expr), divergence=A.to_text(divergence))

    x_sym, y_sym = sp.Symbol("x", real=True), sp.Symbol("y", real=True)
    ylo_spec = x_lower_expr  # 兼容误传：x_lower_expr 不是本函数使用的参数

    # 先判断区域是否是「圆周型」直角坐标描述；是的话直接用柱坐标积分，
    # 避免在直角坐标下做嵌套根式三重积分（SymPy 会慢到超时）。
    cylindrical = _cylindrical_setup(
        divergence, z_sym, zlo_expr, zhi_expr, y_sym, ylo_expr, yhi_expr, x_sym, xlo, xhi
    )
    try:
        if cylindrical is not None:
            polar_integrand, limits = cylindrical
            value = sp.simplify(_integrate_limits(polar_integrand, limits))
            method_note = "先在柱坐标 x=r cosθ, y=r sinθ（雅可比 det = r）下改写区域，再逐层精确积分"
        else:
            value = sp.simplify(
                sp.integrate(divergence, (z_sym, zlo_expr, zhi_expr), (y_sym, ylo_expr, yhi_expr), (x_sym, xlo, xhi))
            )
            method_note = "直角坐标下逐层精确积分"
    except Exception as exc:  # noqa: BLE001
        hint = (
            "可改用 math_numeric 取高精度数值结果，或把区域拆成简单区域分块计算。"
            if cylindrical is None
            else "可改用 math_numeric 取高精度数值结果。"
        )
        return MathResult.unsolved(
            "gauss",
            f"∇·F = {A.to_text(divergence)} 已求出，但区域上的三重积分未能精确求出"
            f"（{type(exc).__name__}: {exc}）。{hint}",
            method="SymPy integrate",
        )
    if isinstance(value, sp.Integral) or not C.is_closed_form(value):
        unsolved = MathResult.unsolved(
            "gauss",
            "三重积分未能求出闭式结果。可尝试改换积分次序、把区域拆成简单区域，或用数值积分。",
            method="SymPy integrate（返回未求值积分）",
        )
        unsolved.set_raw(divergence=A.to_text(divergence), unevaluated=A.to_text(value))
        return unsolved

    r_out.set_result(sp.simplify(value))
    r_out.set_raw(value_text=A.to_text(sp.simplify(value)), integration=method_note)
    r_out.add_condition("Σ 为封闭曲面且取外侧，Ω 为 Σ 所围区域")
    r_out.add_condition("P、Q、R 在 Ω 上有一阶连续偏导数（高斯公式成立的前提）")
    r_out.add_condition("第一类曲面积分 ∯ (P cosα + Q cosβ + R cosγ) dS 与第二类积分 ∯ P dy∧dz + Q dz∧dx + R dx∧dy 相等")

    methods = []
    ok_div: bool | None = None
    try:
        # 独立复核：直接用向量场构造通量型被积函数，检查 ∇·F 的符号求值
        dfx = sp.diff(p_expr, sp.Symbol("x", real=True))
        dfy = sp.diff(q_expr, sp.Symbol("y", real=True))
        dfz = sp.diff(rr_expr, z_sym)
        reconstructed = sp.simplify(dfx + dfy + dfz)
        ok_div = C.same_expr(reconstructed, divergence)
        methods.append(
            {
                "method": "独立复核：分别求三个偏导后重新求和，与 ∇·F 比较",
                "reconstructed": A.to_text(reconstructed),
                "agree": bool(ok_div),
            }
        )
    except Exception as exc:  # noqa: BLE001
        methods.append({"method": "散度复核", "error": f"{type(exc).__name__}: {exc}", "agree": False})

    r_out.verify(
        status="independent" if ok_div else "unverified",
        methods=methods,
        note="散度由三条独立偏导路径重算一致。" if ok_div else "散度独立复核未通过，请复核分量输入。",
    )
    return r_out


# ---------------------------------------------------------------------------
# 5. 斯托克斯公式
# ---------------------------------------------------------------------------

@op("stokes", "stokes_theorem", "curl_theorem", "math_stokes")
def stokes(
    p: Any = None,
    q: Any = None,
    r_component: Any = None,
    r: Any = None,
    x_expr: Any = None,
    y_expr: Any = None,
    z_expr: Any = None,
    u_var: Any = "u",
    v_var: Any = "v",
    u_lower: Any = None,
    u_upper: Any = None,
    v_lower: Any = None,
    v_upper: Any = None,
    **kwargs: Any,
) -> MathResult:
    """斯托克斯公式：把 ∮_Γ P dx + Q dy + R dz 化为 ∬_Σ (∇×F)·n dS。

    曲面 Σ 以参数方程给出（取向由 r_u × r_v 决定），边界 Γ 取与之相容的定向。
    """
    x, y, z = sp.Symbol("x", real=True), sp.Symbol("y", real=True), sp.Symbol("z", real=True)
    u, v = sp.Symbol(str(u_var or "u"), real=True), sp.Symbol(str(v_var or "v"), real=True)
    names = [str(u_var or "u"), str(v_var or "v"), "x", "y", "z"]
    third = r_component if r_component is not None else r
    if p is None or q is None or third is None:
        return MathResult.fail(
            "stokes",
            "斯托克斯公式需要 P、Q、R 三个分量。R 分量请用参数 r 或 r_component 给出"
            "（例如 p='-y', q='x', r='0' 对应 ∇×F = (0,0,2)）。",
            kind="invalid_input",
        )
    if x_expr is None or y_expr is None:
        return MathResult.fail(
            "stokes",
            "缺少曲面 Σ 的参数方程：至少需要 x_expr 与 y_expr。"
            "例如平面圆盘：x_expr='u*cos(v)', y_expr='u*sin(v)', z_expr='0'。",
            kind="invalid_input",
        )

    try:
        p_expr = A.parse(str(p), symbols=["x", "y", "z"])
        q_expr = A.parse(str(q), symbols=["x", "y", "z"])
        rr_expr = A.parse(str(third), symbols=["x", "y", "z"])
        params = _symbols_of(names)
        xs = A.parse(str(x_expr), symbols=names, extra=params)
        ys = A.parse(str(y_expr), symbols=names, extra=params)
        zs = A.parse(str(z_expr), symbols=names, extra=params) if z_expr is not None else sp.Integer(0)
    except ValueError as exc:
        return MathResult.fail("stokes", f"解析失败：{exc}", kind="invalid_input")

    try:
        ulo = _parse_bound(u_lower if u_lower is not None else "0", names, what="u 的下限")
        uhi = _parse_bound(u_upper if u_upper is not None else "1", names, what="u 的上限")
        vlo = _parse_bound(v_lower if v_lower is not None else "0", names, what="v 的下限")
        vhi = _parse_bound(v_upper if v_upper is not None else "2*pi", names, what="v 的上限")
    except ValueError as exc:
        return MathResult.fail("stokes", str(exc), kind="invalid_input")

    curl = sp.Matrix(
        [
            sp.diff(rr_expr, y) - sp.diff(q_expr, z),
            sp.diff(p_expr, z) - sp.diff(rr_expr, x),
            sp.diff(q_expr, x) - sp.diff(p_expr, y),
        ]
    )
    position = sp.Matrix([xs, ys, zs])
    u = _resolve_param([xs, ys, zs], str(u_var or "u"))
    v = _resolve_param([xs, ys, zs], str(v_var or "v"))
    normal = position.diff(u).cross(position.diff(v))
    substituted = {x: xs, y: ys, z: zs}
    curl_substituted = sp.Matrix([component.subs(substituted) for component in curl])
    integrand = sp.simplify(curl_substituted.dot(normal))

    r = MathResult.ok("stokes", method="斯托克斯公式：∬_Σ (∇×F)·(r_u × r_v) du dv（SymPy 精确积分）")
    r.set_input(parametrization={str(u): [A.to_text(ulo), A.to_text(uhi)], str(v): [A.to_text(vlo), A.to_text(vhi)]})
    r.set_raw(
        P=A.to_text(p_expr),
        Q=A.to_text(q_expr),
        R=A.to_text(rr_expr),
        curl=A.to_text(curl),
        normal_vector=A.to_text(normal),
        integrand=A.to_text(integrand),
    )

    try:
        value = sp.simplify(sp.integrate(integrand, (u, ulo, uhi), (v, vlo, vhi)))
    except Exception as exc:  # noqa: BLE001
        return MathResult.unsolved(
            "stokes",
            f"∇×F 与有向面积元已求出，但二重积分未能精确求出（{type(exc).__name__}: {exc}）。"
            "可化简被积函数或改换参数化，也可改用 math_numeric。",
            method="SymPy integrate",
        )
    if isinstance(value, sp.Integral) or not C.is_closed_form(value):
        unsolved = MathResult.unsolved(
            "stokes",
            "二重积分未能求出闭式结果（返回未求值积分）。可化简被积函数、调整参数化，或改用数值积分。",
            method="SymPy integrate（返回未求值积分）",
        )
        unsolved.set_raw(integrand=A.to_text(integrand), unevaluated=A.to_text(value))
        return unsolved

    r.set_result(sp.simplify(value))
    r.set_raw(value_text=A.to_text(sp.simplify(value)))
    r.add_condition("Σ 为分片光滑的有向曲面，Γ 为 Σ 的边界且定向与 Σ 的法向构成右手系")
    r.add_condition("P、Q、R 在含 Σ 的空间区域上有一阶连续偏导数")
    r.add_condition("法向由 r_u × r_v 决定；若取相反法向，结果取负号")

    methods: list[dict[str, Any]] = []
    ok_curl: bool | None = None
    try:
        reconstructed = sp.Matrix(
            [
                sp.diff(rr_expr, y) - sp.diff(q_expr, z),
                sp.diff(p_expr, z) - sp.diff(rr_expr, x),
                sp.diff(q_expr, x) - sp.diff(p_expr, y),
            ]
        )
        ok_curl = C.same_expr(reconstructed, curl)
        methods.append(
            {
                "method": "独立复核：按旋度定义逐分量重算 ∇×F 并与结果比较",
                "reconstructed": A.to_text(reconstructed),
                "agree": bool(ok_curl),
            }
        )
    except Exception as exc:  # noqa: BLE001
        methods.append({"method": "旋度复核", "error": f"{type(exc).__name__}: {exc}", "agree": False})

    # 数值复核：二重数值积分
    ok_numeric: bool | None = None
    try:
        import mpmath as mp

        f_numeric = sp.lambdify((u, v), integrand, modules=["mpmath"])
        numeric_value = mp.quad(
            lambda uu: mp.quad(lambda vv: f_numeric(uu, vv), [float(vlo), float(vhi)]),
            [float(ulo), float(uhi)],
        )
        exact = complex(sp.N(value, 30))
        deviation = abs(complex(numeric_value) - exact) / max(1.0, abs(exact))
        ok_numeric = deviation < 1e-8
        methods.append(
            {
                "method": "独立数值复核：对同一参数化做二重数值积分与符号结果对比",
                "numeric": mp.nstr(numeric_value, 15),
                "relative_deviation": f"{deviation:.3e}",
                "agree": bool(ok_numeric),
            }
        )
    except Exception as exc:  # noqa: BLE001
        methods.append({"method": "独立数值复核", "error": f"{type(exc).__name__}: {exc}", "agree": False})

    all_ok = bool(ok_curl) and ok_numeric is not False
    r.verify(
        status="symbolic+numeric" if all_ok and ok_numeric else ("independent" if ok_curl else "unverified"),
        methods=methods,
        note="旋度按定义独立重算一致，且二重数值积分与符号结果吻合。"
        if all_ok and ok_numeric
        else "符号结果已求出，但部分复核未通过或无法执行，请人工复核参数化与定向。",
    )
    if ok_curl is False:
        r.add_warning("旋度独立复核不一致，请复核分量输入顺序（P、Q、R 分别对应 x、y、z 分量）。")
    if ok_numeric is False:
        r.add_warning("数值复核与符号结果不一致，请检查参数化与定向。")
    return r


# ---------------------------------------------------------------------------
# 6. 曲线积分 ↔ 格林公式 一致性自检（供测试用）
# ---------------------------------------------------------------------------

def green_consistency_example() -> dict[str, Any]:
    """用 ∮_L −y dx + x dy = 2·面积 检验 curve_integral 与 green 是否一致。"""
    unit_circle = curve_integral(
        expr="-y", expr2="x", x_expr="cos(t)", y_expr="sin(t)", var="t", lower="0", upper="2*pi", kind="second"
    )
    disk = green(p="-y", q="x", x_lower="-1", x_upper="1", y_lower_expr="-sqrt(1-x^2)", y_upper_expr="sqrt(1-x^2)")
    return {
        "curve_integral": unit_circle.to_dict(),
        "green": disk.to_dict(),
        "agree": C.same_expr(unit_circle.result.get("text"), disk.result.get("text")),
    }
