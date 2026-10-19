"""符号计算核心：化简、求导、积分、极限、级数。

这一层不直接对外注册工具，而是把「严谨性」集中实现一次，供各模块复用：

* :func:`same_expr` —— 只认数学等价，绝不靠字符串比较
* :func:`check_derivative` / :func:`check_antiderivative` —— 独立复核
* :func:`numeric_agree` —— 高精度数值交叉验算（不作为证明）
* :func:`is_closed_form` —— 识别 SymPy 的未求值回显

设计原则（用户规格 4.1 / 4.3 / 十一）：能给出解析解就给解析解；给不出就明确
说明「未求得」，绝不用数值近似冒充解析结果。
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import sympy as sp

from . import ast as A

# 未求值对象的类（SymPy 原样回显时不代表成功）
_UNEVALUATED = (sp.Integral, sp.Derivative, sp.Limit, sp.Sum, sp.Product)


# ---------------------------------------------------------------------------
# 等价性判定
# ---------------------------------------------------------------------------

def same_expr(a: Any, b: Any) -> bool:
    """数学等价判定：先结构化比较，再化简比较，最后（仅数值）高精度抽样比较。"""
    try:
        ea, eb = sp.sympify(a), sp.sympify(b)
    except Exception:  # noqa: BLE001
        return str(a).strip() == str(b).strip()
    # 矩阵逐元素（先处理，避免把矩阵丢给 equals/simplify 展开）
    if isinstance(ea, sp.MatrixBase) and isinstance(eb, sp.MatrixBase):
        if ea.shape != eb.shape:
            return False
        return all(same_expr(x, y) for x, y in zip(ea, eb))
    try:
        diff = sp.simplify(ea - eb)
        if diff == 0:
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        if sp.expand(ea - eb) == 0:
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        if sp.nsimplify(sp.simplify(ea - eb)) == 0:
            return True
    except Exception:  # noqa: BLE001
        pass
    # 含自由符号（参数）时**不要**调 equals()：它对符号表达式会走 is_constant → simplify
    # → cancel → 无限 expand，实测在 a*x**2+2*b*x*y+c*y**2 这类表达式上直接挂死。
    free = set(getattr(ea, "free_symbols", ()) or ()) | set(getattr(eb, "free_symbols", ()) or ())
    if free:
        return False
    try:
        # equals() 内部会做数值抽查，仅对纯数／无自由符号表达式安全
        verdict = ea.equals(eb)
        if verdict is True:
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def numeric_agree(
    a: Any,
    b: Any,
    *,
    symbols: Sequence[sp.Symbol] | None = None,
    rel_tol: float = 1e-9,
    samples: int = 9,
) -> tuple[bool, float, str]:
    """在若干随机点上高精度比较两个表达式。

    返回 ``(是否一致, 最大相对偏差, 说明)``。
    只在表达式含自由变量时才有意义；纯数值直接比较。
    """
    try:
        ea, eb = sp.sympify(a), sp.sympify(b)
    except Exception as exc:  # noqa: BLE001
        return False, float("inf"), f"无法转为表达式: {exc}"

    free = sorted((ea.free_symbols | eb.free_symbols), key=lambda s: str(s)) if isinstance(ea, sp.Basic) and isinstance(eb, sp.Basic) else []
    if symbols:
        free = list(symbols)
    if not free and isinstance(ea, sp.Basic) and isinstance(eb, sp.Basic):
        try:
            va = sp.N(ea, 30)
            vb = sp.N(eb, 30)
            if va.is_real and vb.is_real:
                if va == vb:
                    return True, 0.0, "纯数值相等"
                dev = abs(float(va - vb)) / max(1.0, abs(float(vb)))
                return dev < rel_tol, dev, f"纯数值相对偏差 {dev:.3e}"
            return (va == vb), 0.0, "复数比较"
        except Exception as exc:  # noqa: BLE001
            return False, float("inf"), f"数值化失败: {exc}"

    if not free:
        return True, 0.0, "无可变点，未做抽样比较"

    rng = [sp.Rational(1, 3), sp.Rational(1, 2), sp.Rational(2, 3), sp.Rational(3, 2),
           sp.Rational(5, 4), sp.Rational(7, 5), sp.Rational(9, 7), sp.Rational(11, 6),
           sp.Rational(4, 3)]
    # 各变量必须取**不同**的值：早先的实现给所有自由变量取同一个值（对角线切片），
    # 于是 x-y、x/y-1 这类「只在 x=y 时为零」的差式在每个抽样点都恰好为 0，
    # numeric_agree("x-y", "0", symbols=["x","y"]) 会误判一致（漏判）。
    offsets = [sp.Rational(1, 7), sp.Rational(-1, 5), sp.Rational(2, 9), sp.Rational(-3, 11),
               sp.Rational(5, 13), sp.Rational(-4, 15)]
    worst = 0.0
    checked = 0
    failures: list[str] = []
    for index in range(max(1, min(samples, len(rng)))):
        base = rng[index % len(rng)]
        point = {}
        for position, symbol in enumerate(free):
            if position == 0:
                point[symbol] = base
            else:
                step = offsets[(position + index) % len(offsets)]
                candidate = base + step
                point[symbol] = candidate if candidate != 0 else base + sp.Rational(1, 3)
        try:
            va = sp.N(ea.subs(point), 30)
            vb = sp.N(eb.subs(point), 30)
        except Exception:  # noqa: BLE001
            continue
        if not (va.is_real and vb.is_real):
            continue
        if va in (sp.nan, sp.zoo, sp.oo, -sp.oo) or vb in (sp.nan, sp.zoo, sp.oo, -sp.oo):
            continue
        try:
            fa, fb = float(va), float(vb)
        except (TypeError, ValueError):
            # 结果里仍残留无法数值化的成分（未求值积分、未定义函数代入点等）：
            # 这不是「不一致」，而是「没有有效数值证据」，直接跳过该点。
            continue
        checked += 1
        if fa == fb:
            continue
        scale = max(1.0, abs(fb))
        dev = abs(fa - fb) / scale
        worst = max(worst, dev)
        if dev > rel_tol and len(failures) < 3:
            failures.append(f"{point} 处 {fa:.12g} vs {fb:.12g}")
    if checked == 0:
        return False, float("inf"), "所有抽样点都拿不到有效数值（未获得数值证据，不能据此判定一致）"
    if failures:
        return False, worst, "抽样点不一致：" + "；".join(failures)
    return True, worst, f"{checked} 个抽样点一致（最大相对偏差 {worst:.3e}）"


def unify_symbols(expr: Any, reference: Any) -> Any:
    """把 ``expr`` 里与 ``reference`` **同名但不同一**的符号换成 reference 里的对象。

    解析是分多次进行的（原式、候选解、积分上下界、参数…），每次 ``get_symbols``
    造出的 ``Symbol`` 可能带不同假设（``real=True`` vs 无假设）。它们打印完全相同
    但 ``!=``，SymPy 于是把变量当常数、把导数当 0：

    * ``sp.integrate(x**2, sp.Symbol('x'))`` → ``x*x**2``（被积函数被当成常数）
    * ``Derivative(y(x), x).subs({y(x): C1*exp(x_p**2/2)})`` → 对 x 求导得 0

    这是本项目最隐蔽的一类静默错误（不报错、结果看似合理），因此在「两段独立解析
    的结果相遇」的地方都要先对齐符号。
    """
    if not isinstance(expr, sp.Basic) or not isinstance(reference, sp.Basic):
        return expr
    ref_by_name = {str(s): s for s in reference.free_symbols}
    replacements: dict[sp.Symbol, sp.Symbol] = {}
    for symbol in expr.free_symbols:
        target = ref_by_name.get(str(symbol))
        if target is not None and target is not symbol:
            replacements[symbol] = target
    if replacements:
        return expr.subs(replacements)
    return expr


# 「没有拿到有效数值证据」的说明标记。调用方必须把这种返回值当作
# 「未知」而不是「一致」——把证据缺失当通过是最危险的假阳性。
_NO_EVIDENCE_MARKERS = (
    "未获得数值证据",
    "未做抽样比较",
    "数值化失败",
    "无法转为表达式",
)


def numeric_evidence(a: Any, b: Any, **kwargs: Any) -> tuple[str, float, str]:
    """带三态判定的数值比较：``"equal"`` / ``"different"`` / ``"unknown"``。

    ``numeric_agree`` 为了兼容旧调用方，在拿不到数值时会返回 ``True``（视为「没有反例」）。
    做验证结论时必须用本函数，否则「无法数值化」会被当成为「一致」，
    错的候选解也能判成通过。
    """
    ok, dev, note = numeric_agree(a, b, **kwargs)
    if any(marker in note for marker in _NO_EVIDENCE_MARKERS):
        return "unknown", dev, note
    return ("equal" if ok else "different"), dev, note


# ---------------------------------------------------------------------------
# 未求值识别
# ---------------------------------------------------------------------------

def is_closed_form(expr: Any) -> bool:
    """True 表示结果已经求值，不是 SymPy 的原样回显。"""
    if expr is None:
        return False
    if isinstance(expr, _UNEVALUATED):
        return False
    if isinstance(expr, sp.Basic):
        for node in sp.preorder_traversal(expr):
            if isinstance(node, _UNEVALUATED):
                return False
    if isinstance(expr, (list, tuple, set)):
        return all(is_closed_form(x) for x in expr)
    return True


# ---------------------------------------------------------------------------
# 复核（独立验证）原语
# ---------------------------------------------------------------------------

def check_derivative(
    expr: Any,
    var: sp.Symbol,
    candidate: Any,
    *,
    order: int = 1,
    independent: bool = True,
) -> tuple[bool, dict[str, Any]]:
    """复核导数：把候选结果与解析求导作符号比较。

    ``order`` 为求导阶数 —— 高阶导必须传对阶数，否则会把正确结果判为不一致
    （``diff(x**4, x, 4)`` 的候选是常数 24，若仍拿 ``diff(x**4, x)`` 去比就永远不等）。

    ``independent=False`` 表示调用方与本次计算同源（都用 ``sp.diff``），此时该结果
    只能算「一致性检查」而非独立验证 —— 真正的独立证据由
    :func:`numeric_derivative_check` 用数值微分给出。detail 里会如实标注。
    """
    try:
        order_int = max(1, int(order))
        expected = sp.diff(sp.sympify(expr), var, order_int)
        ok = same_expr(expected, candidate)
        detail: dict[str, Any] = {
            "method": "解析求导并与候选结果作符号比较",
            "independent": bool(independent),
            "note": (
                "与主计算同源（均为符号求导），仅作一致性检查"
                if not independent
                else "独立符号路径"
            ),
            "order": order_int,
            "expected": A.to_text(expected),
            "expected_latex": A.to_latex(expected),
            "agree": ok,
        }
        if not ok:
            passed, dev, note = numeric_agree(expected, candidate, symbols=[var])
            detail["numeric"] = {"agree": passed, "max_rel_dev": dev, "note": note}
            ok = passed
        return ok, detail
    except Exception as exc:  # noqa: BLE001
        return False, {"method": "独立求导", "error": str(exc)}


def numeric_derivative_check(
    expr: Any,
    var: sp.Symbol,
    candidate: Any,
    *,
    order: int = 1,
    rel_tol: float = 1e-6,
    samples: int = 5,
) -> tuple[str, float | None, str]:
    """用**数值微分**独立核验 ``candidate`` 是否为 ``expr`` 的 ``order`` 阶导数。

    这是与符号求导完全不同的算法路径（``mpmath.diff`` 的中心差分），因此结论可以
    标为独立证据。返回三态 ``("equal" | "different" | "unknown", 最大相对偏差, 说明)``。

    含其它自由符号（参数）时不做抽样 —— 参数未定则无法给出数值证据，如实返回
    ``unknown``，绝不因此判通过。
    """
    try:
        expr_s = sp.sympify(expr)
        cand_s = sp.sympify(candidate)
    except Exception as exc:  # noqa: BLE001
        return "unknown", None, f"无法转为表达式: {exc}"
    if not isinstance(expr_s, sp.Basic) or not isinstance(cand_s, sp.Basic):
        return "unknown", None, "非标量表达式，未做数值微分核验"
    extra = (expr_s.free_symbols | cand_s.free_symbols) - {var}
    if extra:
        names = ", ".join(sorted(str(s) for s in extra))
        return "unknown", None, f"表达式含未定参数（{names}），未做数值微分核验"
    try:
        import mpmath as mp
    except Exception as exc:  # noqa: BLE001
        return "unknown", None, f"未安装 mpmath，无法做数值微分核验: {exc}"
    try:
        f = sp.lambdify(var, expr_s, modules="mpmath")
        g = sp.lambdify(var, cand_s, modules="mpmath")
    except Exception as exc:  # noqa: BLE001
        return "unknown", None, f"数值化失败: {exc}"
    order_int = max(1, int(order))
    rng = [sp.Rational(1, 3), sp.Rational(1, 2), sp.Rational(2, 3),
           sp.Rational(5, 4), sp.Rational(7, 5), sp.Rational(4, 3)]
    worst = mp.mpf(0)
    checked = 0
    for rational in rng[: max(1, samples)]:
        x0 = mp.mpf(int(rational.p)) / mp.mpf(int(rational.q))
        try:
            got = mp.mpf(g(x0))
            want = mp.mpf(mp.diff(f, x0, order_int))
        except Exception:  # noqa: BLE001
            continue
        if not (mp.isfinite(got) and mp.isfinite(want)):
            continue
        scale = max(abs(want), mp.mpf(1))
        dev = abs(got - want) / scale
        if not mp.isfinite(dev):
            continue
        worst = max(worst, dev)
        checked += 1
    if checked == 0:
        return "unknown", None, "所有抽样点都拿不到有效数值（未获得数值证据）"
    worst_f = float(worst)
    if worst_f <= rel_tol:
        return "equal", worst_f, (
            f"在 {checked} 个抽样点上用 mpmath 数值微分（中心差分，{order_int} 阶）独立核验，"
            f"最大相对偏差 {worst_f:.3e}"
        )
    return "different", worst_f, (
        f"数值微分独立核验不一致：{checked} 个抽样点最大相对偏差 {worst_f:.3e}（超出容差 {rel_tol:.0e}）"
    )


def check_antiderivative(integrand: Any, var: sp.Symbol, candidate: Any) -> tuple[bool, dict[str, Any]]:
    """独立复核不定积分：对候选结果求导，必须回到被积函数。"""
    try:
        back = sp.simplify(sp.diff(sp.sympify(candidate), var))
        target = sp.sympify(integrand)
        ok = same_expr(back, target)
        detail: dict[str, Any] = {
            "method": "对候选原函数求导，检查是否等于被积函数",
            "derivative_of_candidate": A.to_text(back),
            "derivative_latex": A.to_latex(back),
            "agree": ok,
        }
        if not ok:
            passed, dev, note = numeric_agree(back, target, symbols=[var])
            detail["numeric"] = {"agree": passed, "max_rel_dev": dev, "note": note}
            ok = passed
        return ok, detail
    except Exception as exc:  # noqa: BLE001
        return False, {"method": "对候选原函数求导", "error": str(exc)}


def check_solution(equations: Iterable[Any], var: Sequence[sp.Symbol], candidate: Any) -> tuple[bool, dict[str, Any]]:
    """独立复核方程解：代回原方程检查残差。"""
    try:
        eqs = [sp.sympify(e) for e in equations]
        residuals: list[str] = []
        ok = True
        subs_pairs = _solution_pairs(var, candidate)
        if subs_pairs is None:
            return False, {"method": "代回原方程", "error": "解的结构无法与变量匹配"}
        for eq in eqs:
            try:
                lhs_lhs = sp.simplify(eq.subs(subs_pairs))
            except Exception as exc:  # noqa: BLE001
                residuals.append(f"代入失败: {exc}")
                ok = False
                continue
            residuals.append(A.to_text(lhs_lhs))
            if sp.sympify(lhs_lhs) != 0:
                agree, dev, _ = numeric_agree(lhs_lhs, 0)
                if not (agree or (lhs_lhs.is_number and abs(complex(sp.N(lhs_lhs, 30))) < 1e-9)):
                    ok = False
        return ok, {"method": "代回原方程检查残差", "residuals": residuals}
    except Exception as exc:  # noqa: BLE001
        return False, {"method": "代回原方程", "error": str(exc)}


def _solution_pairs(var: Sequence[sp.Symbol], candidate: Any) -> dict[sp.Symbol, Any] | None:
    syms = list(var)
    cand = candidate
    if isinstance(cand, dict):
        pairs: dict[sp.Symbol, Any] = {}
        for key, value in cand.items():
            key_sym = sp.Symbol(str(key)) if not isinstance(key, sp.Symbol) else key
            pairs[key_sym] = value
        return pairs
    if isinstance(cand, (list, tuple)):
        if len(cand) != len(syms):
            return None
        return dict(zip(syms, cand))
    if len(syms) == 1:
        return {syms[0]: cand}
    return None


def verify_identity(a: Any, b: Any, *, symbols: Sequence[sp.Symbol] | None = None) -> tuple[bool, dict[str, Any]]:
    """恒等式复核：符号化简优先，数值抽样仅作补充证据。"""
    try:
        ea, eb = sp.sympify(a), sp.sympify(b)
    except Exception as exc:  # noqa: BLE001
        return False, {"method": "恒等式比较", "error": str(exc)}
    detail: dict[str, Any] = {}
    try:
        diff = sp.simplify(ea - eb)
        detail["simplified_difference"] = A.to_text(diff)
        detail["method"] = "差式化简为 0"
        if diff == 0:
            detail["symbolic"] = True
            return True, detail
        detail["symbolic"] = False
    except Exception as exc:  # noqa: BLE001
        detail["simplify_error"] = str(exc)
    passed, dev, note = numeric_agree(ea, eb, symbols=symbols)
    detail["numeric"] = {"agree": passed, "max_rel_dev": dev, "note": note}
    detail["method"] = "差式未能符号化简为 0；改用高精度抽样比较（仅为数值证据）"
    return passed, detail


# ---------------------------------------------------------------------------
# 共享的解析助手
# ---------------------------------------------------------------------------

def resolve_var(expr: sp.Basic, var: str | None, *, fallback: str = "x") -> sp.Symbol:
    """确定对哪个变量操作：显式指定优先，否则取表达式中唯一的自由变量。

    **关键**：显式指定时必须优先复用表达式里已有的 Symbol 对象。
    ``A.parse`` 产生的符号带 ``real=True`` 等假设，而 ``sp.Symbol("x")`` 不带假设；
    两者打印完全相同却不相等（``sp.Symbol('x') != sp.Symbol('x', real=True)``）。
    把不带假设的符号当作积分/求导变量传给 SymPy，SymPy 会认为该变量不在表达式里，
    于是把被积函数当常数处理，直接返回被积函数本身乘以区间长度之类的错误结果
    （实测 ``sp.integrate(x**2, Symbol('x'))`` 会返回 ``x*x**2``）。
    """
    if var:
        name = str(var).strip()
        if isinstance(expr, sp.Basic):
            for candidate in expr.free_symbols:
                if str(candidate) == name:
                    return candidate
        return sp.Symbol(name)
    free = sorted(expr.free_symbols, key=lambda s: (len(str(s)), str(s))) if isinstance(expr, sp.Basic) else []
    if len(free) == 1:
        return free[0]
    if len(free) > 1:
        for candidate in free:
            if str(candidate) == fallback:
                return candidate
        raise ValueError(
            f"表达式含多个变量 {[str(s) for s in free]}，请用 var 明确指定对哪个变量求导/积分/求极限"
        )
    return sp.Symbol(fallback)


def parse_point(point: Any, symbols: Iterable[sp.Symbol]) -> Any:
    """把极限点/代入点解析成 SymPy 对象，支持限号 oo / -oo / 无穷。"""
    if point is None:
        raise ValueError("缺少趋近点")
    if isinstance(point, (int, float, sp.Basic)):
        return sp.sympify(point)
    raw = str(point).strip()
    normalized = raw.replace("无穷", "oo").replace("∞", "oo").replace("inf", "oo").replace("Inf", "oo")
    normalized = normalized.replace("+oo", "oo")
    return A.parse(normalized, symbols=[str(s) for s in symbols])
