"""``math_verify``：统一独立复核入口（用户规格 5.2）。

设计原则（硬要求）：

* **计算交给工具，思考交给模型，验证交给独立机制**：本模块不重算原题，
  而是用一套与计算路径无关的原语（``sympy_core`` 的符号复核原语、数值抽样、
  有限差分、蒙特卡洛、scipy 精确分布函数）去复核别人给出的结果。
* **绝不为「通过」放宽条件**：符号证据不足时只给低级别（``numeric``）结论；
  连数值证据都拿不到时返回 ``unsolved``，绝不返回 ``passed=True``。
* **条件必须被真正使用**，用不上的条件一律进入 ``unhandled_conditions``。
* 复核失败必须给出**反例或差异位置**（抽样点数值、差式化简结果、残差）。
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Sequence

import sympy as sp

from . import ast as A
from . import sympy_core as C
from .engine import op
from .result import MathResult

# ---------------------------------------------------------------------------
# 常量与小工具
# ---------------------------------------------------------------------------

LEVEL_SYMBOLIC = "symbolic"
LEVEL_NUMERIC = "numeric"
LEVEL_NONE = "none"

_TOL = 1e-7

#: 抽样点：避开 0、±1 等平凡点，兼顾负数（用于发现定义域问题）
_BASE_POINTS: tuple[sp.Basic, ...] = (
    sp.Rational(1, 3), sp.Rational(1, 2), sp.Rational(2, 3), sp.Rational(3, 2),
    sp.Rational(5, 4), sp.Rational(2), sp.Rational(7, 3), sp.Rational(4),
    sp.Rational(-1, 3), sp.Rational(-1, 2), sp.Rational(-3, 2), sp.Rational(-2),
)


class _CannotVerify(Exception):
    """内部信号：确实无法验证（转成 unsolved，并写明原因）。"""


def _t(value: Any) -> str:
    try:
        return A.to_text(value)
    except Exception:  # noqa: BLE001
        return str(value)


def _points_text(subs: dict[sp.Symbol, Any]) -> str:
    return "{" + ", ".join(f"{_t(k)}: {_t(v)}" for k, v in subs.items()) + "}"


def _parse_expr(text: Any, *, what: str = "表达式", extra: dict[str, Any] | None = None) -> sp.Basic:
    if text is None:
        raise ValueError(f"缺少{what}")
    if isinstance(text, sp.Basic):
        return text
    try:
        return A.parse(str(text), extra=extra)
    except ValueError as exc:
        raise ValueError(f"{what}无法解析：{exc}") from exc


def _parse_expr_soft(text: Any, *, what: str = "表达式", extra: dict[str, Any] | None = None) -> sp.Basic | None:
    try:
        return _parse_expr(text, what=what, extra=extra)
    except ValueError:
        return None


def _split_equality(text: Any) -> tuple[str, str] | None:
    """把 ``a = b`` 拆成左右两边（跳过 ``==`` / ``<=`` / ``>=`` / ``!=``）。"""
    s = str(text)
    for i, ch in enumerate(s):
        if ch != "=":
            continue
        prev = s[i - 1] if i > 0 else ""
        nxt = s[i + 1] if i + 1 < len(s) else ""
        if prev in "<>=!" or nxt == "=":
            continue
        return s[:i], s[i + 1:]
    return None


def _equation_exprs(texts: Any) -> list[sp.Basic]:
    """把方程（字符串或列表）统一成「左边 − 右边」形式的表达式列表。"""
    if texts is None:
        return []
    items = texts if isinstance(texts, (list, tuple)) else [texts]
    out: list[sp.Basic] = []
    for it in items:
        if it is None:
            continue
        if isinstance(it, sp.Basic):
            out.append(it)
            continue
        pair = _split_equality(it)
        if pair is None:
            out.append(_parse_expr(it, what="方程"))
        else:
            lhs = _parse_expr(pair[0], what="方程左端")
            rhs = _parse_expr(pair[1], what="方程右端")
            out.append(sp.sympify(lhs) - sp.sympify(rhs))
    return out


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _resolve_symbols(vars_arg: Any, default_syms: Sequence[sp.Symbol]) -> list[sp.Symbol]:
    names: list[str] = []
    for item in _as_list(vars_arg):
        if isinstance(item, sp.Symbol):
            names.append(item.name)
        else:
            names.extend(str(item).replace(",", " ").replace(";", " ").split())
    if not names:
        return list(default_syms)
    table = A.get_symbols(names)
    return [table[n] for n in names if n in table]


def _resolve_matrix(value: Any, *, what: str = "矩阵") -> sp.Matrix | None:
    if value is None:
        return None
    if isinstance(value, sp.MatrixBase):
        return sp.Matrix(value)
    if isinstance(value, (list, tuple)):
        rows = []
        for row in value:
            if isinstance(row, (list, tuple)):
                rows.append([_parse_expr(c, what=what) for c in row])
            elif isinstance(row, str) and ("," in row or ";" in row):
                # ['1,2','3,4'] 这种「每个元素是一行」的常见写法：整体交给矩阵解析器。
                # 若按「列表元素 = 单元格」处理，会把 "1,2" 解析成 Tuple，得到非方阵并抛
                # NonSquareMatrixError（message 为空），把「写法特殊」误报成内部错误。
                return A.parse_matrix(text=";".join(str(r) for r in value))
            else:
                rows.append([_parse_expr(row, what=what)])
        return sp.Matrix(rows)
    return A.parse_matrix(text=str(value))


# ---------------------------------------------------------------------------
# 数值抽样比较
# ---------------------------------------------------------------------------

def _ev(expr: Any, subs: dict[sp.Symbol, Any], *, digits: int = 20) -> float | None:
    """在给定取值处求实数数值；不是实数 / 无法求值 / 含自由符号 → None。"""
    if expr is None:
        return None
    try:
        value = sp.sympify(expr).subs(subs)
        if getattr(value, "free_symbols", None):
            return None
        num = sp.N(value, digits)
        if getattr(num, "free_symbols", None):
            return None
        if num.is_real is not True:
            return None
        if num.is_finite is not True:
            return None
        return float(num)
    except Exception:  # noqa: BLE001
        return None


def _rel_true(value: Any) -> bool | None:
    """判定关系式代入具体数值后的真假。

    sympy 的 ``(x > 0).subs(x, 1/4)`` 会返回 **Python 的 True/False**（而不是
    ``sp.true``），所以不能直接用 ``is sp.true`` 判断。
    """
    if value is True or value is sp.true:
        return True
    if value is False or value is sp.false:
        return False
    if isinstance(value, sp.logic.boolalg.BooleanTrue):
        return True
    if isinstance(value, sp.logic.boolalg.BooleanFalse):
        return False
    return None


def _ev_complex(expr: Any, subs: dict[sp.Symbol, Any]) -> Any:
    try:
        value = sp.sympify(expr).subs(subs)
        if getattr(value, "free_symbols", None):
            return None
        if value.is_real is True:
            return value
        return sp.N(value, 12)
    except Exception:  # noqa: BLE001
        return None


def _numeric_compare(
    a: Any,
    b: Any,
    symbols: Sequence[sp.Symbol],
    *,
    extra_points: Sequence[sp.Basic] | None = None,
    tol: float = _TOL,
) -> dict[str, Any]:
    """在若干抽样点比较两个表达式，返回最大相对偏差与首个反例点。

    多符号时给每个符号错开取值，避免「所有变量取同一个值」导致的假一致。
    """
    syms = [s for s in symbols]
    pts = list(extra_points or []) + list(_BASE_POINTS)
    rows: list[dict[str, Any]] = []
    worst: dict[str, Any] | None = None
    max_dev = 0.0
    checked = 0
    if not syms:
        va, vb = _ev(a, {}), _ev(b, {})
        if va is not None and vb is not None:
            denom = max(1.0, abs(va), abs(vb))
            dev = abs(va - vb) / denom
            rows.append({"point": {}, "a": va, "b": vb, "rel_dev": dev, "text": "无自由变量（单点求值）"})
            max_dev, worst, checked = dev, rows[0], 1
    else:
        for k in range(len(pts)):
            subs = {s: pts[(k + i) % len(pts)] for i, s in enumerate(syms)}
            va, vb = _ev(a, subs), _ev(b, subs)
            if va is None or vb is None:
                continue
            denom = max(1.0, abs(va), abs(vb))
            dev = abs(va - vb) / denom
            row = {"point": subs, "a": va, "b": vb, "rel_dev": dev}
            rows.append(row)
            checked += 1
            if dev > max_dev:
                max_dev, worst = dev, row
    agree = None if checked == 0 else bool(max_dev <= tol)
    note = None
    if agree is None:
        note = "没有获得任何有效的实数抽样点（表达式含未定义函数、参数或处处非实数）"
    elif agree:
        note = f"在 {checked} 个抽样点上数值一致（最大相对偏差 {max_dev:.3e}）——仅为数值证据，不是符号证明"
    else:
        note = (f"抽样点不一致：{_points_text(worst['point'])} 处 {worst['a']:.12g} vs {worst['b']:.12g}"
                f"（相对偏差 {max_dev:.6g}）")
    return {"agree": agree, "max_rel_dev": max_dev, "worst": worst, "checked": checked,
            "rows": rows, "note": note}


def _symbolic_zero(diff: Any) -> tuple[bool | None, sp.Basic, str]:
    """判断差式是否恒为 0。返回 (结论, 化简后的差式, 依据)。"""
    d = sp.sympify(diff)
    if d == 0:
        return True, sp.Integer(0), "差式构造上即为 0"
    if d.is_zero is True:
        return True, sp.Integer(0), "is_zero 判定为 0"
    trials: list[tuple[str, Callable[[sp.Basic], Any]]] = [
        ("sp.simplify", sp.simplify),
        ("sp.expand", sp.expand),
        ("sp.cancel(sp.together())", lambda e: sp.cancel(sp.together(e))),
        ("sp.factor", sp.factor),
        ("sp.trigsimp", sp.trigsimp),
        ("sp.powsimp", sp.powsimp),
        ("sp.ratsimp", sp.ratsimp),
    ]
    best = d
    for name, fn in trials:
        try:
            r = fn(d)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(r, sp.Basic):
            if r == 0 or r.is_zero is True:
                return True, sp.Integer(0), f"{name} 化简为 0"
            try:
                if sp.count_ops(r) < sp.count_ops(best):
                    best = r
            except Exception:  # noqa: BLE001
                pass
    try:
        eq = d.equals(0)
    except Exception:  # noqa: BLE001
        eq = None
    if eq is True:
        return True, sp.Integer(0), "sp.equals(0) 判定为真"
    if eq is False:
        return False, best, "sp.equals(0) 判定为假：差式不恒为 0"
    return None, best, "符号化简未收敛，无法判定差式是否恒为 0"


def _decide_equality(
    a: Any,
    b: Any,
    symbols: Sequence[sp.Symbol],
    *,
    numeric: dict[str, Any] | None = None,
    allow_numeric_pass: bool = True,
) -> dict[str, Any]:
    """用「符号差式 + 数值抽样」判定 a 与 b 是否相等，并给出证据与分级。"""
    diff = sp.sympify(a) - sp.sympify(b)
    zero, simplified, how = _symbolic_zero(diff)
    num = numeric if numeric is not None else _numeric_compare(a, b, symbols)
    evidence: dict[str, Any] = {
        "difference": _t(simplified),
        "difference_latex": None,
        "symbolic_basis": how,
        "numeric": {"agree": num.get("agree"), "max_rel_dev": num.get("max_rel_dev"),
                    "checked_points": num.get("checked"), "note": num.get("note")},
    }
    try:
        evidence["difference_latex"] = A.to_latex(simplified)
    except Exception:  # noqa: BLE001
        pass
    if num.get("worst") and num.get("agree") is False:
        w = num["worst"]
        evidence["counterexample"] = (
            f"在 {_points_text(w['point'])} 处：左边 = {w['a']:.12g}，右边 = {w['b']:.12g}"
            f"（相对偏差 {w['rel_dev']:.6g}）")
    if zero is True:
        return {"passed": True, "level": LEVEL_SYMBOLIC, "method": f"符号差式复核（{how}）",
                "evidence": evidence, "cannot": None}
    if num.get("agree") is True and allow_numeric_pass:
        return {"passed": True, "level": LEVEL_NUMERIC,
                "method": "符号差式化简未能归零，改用高精度数值抽样逐点比较（仅为数值证据，不是符号证明）",
                "evidence": evidence, "cannot": None}
    if num.get("agree") is False:
        return {"passed": False, "level": LEVEL_NUMERIC if zero is None else LEVEL_SYMBOLIC,
                "method": "符号差式不为 0，且数值抽样逐点不一致（已给出反例点）",
                "evidence": evidence, "cannot": None}
    evidence["failure"] = "符号差式未能化简为 0，且没有得到任何有效的数值抽样点"
    return {"passed": None, "level": LEVEL_NONE,
            "method": "符号差式无法归零，数值抽样也失效",
            "evidence": evidence,
            "cannot": "既无法获得符号级证明（差式化简不收敛），也拿不到有效的数值抽样点"
                      "（表达式含未定义函数/参数或处处非实数），因此无法验证",
            }


# ---------------------------------------------------------------------------
# 定义域分析
# ---------------------------------------------------------------------------

def _point_values(candidate: Any) -> tuple[list[sp.Basic], bool]:
    """把 ``sp.singularities`` 的结果转成有限个点；第二个返回值表示是否完整。"""
    out: list[sp.Basic] = []
    incomplete = False
    if candidate is None:
        return out, incomplete
    try:
        if candidate is sp.EmptySet or candidate == sp.EmptySet:
            return out, incomplete
        if isinstance(candidate, sp.FiniteSet):
            return list(candidate), incomplete
        if isinstance(candidate, sp.Union):
            for item in candidate.args:
                pts, inc = _point_values(item)
                out.extend(pts)
                incomplete = incomplete or inc
            return out, incomplete
        if isinstance(candidate, sp.ImageSet):
            try:
                lam = candidate.lamda
                var = lam.variables[0]
                for n0 in range(-2, 3):
                    val = lam(n0)
                    if val.free_symbols:
                        incomplete = True
                        break
                    out.append(sp.simplify(val))
            except Exception:  # noqa: BLE001
                incomplete = True
            return out, True
        if isinstance(candidate, sp.Set):
            return out, True
        return [candidate], incomplete
    except Exception:  # noqa: BLE001
        return out, True


def _domain_restrictions(expr: Any, var: sp.Symbol | None) -> dict[str, Any]:
    """分析表达式的自然定义域：分母、偶次根式、对数、反三角、tan/cot 极点等。"""
    restrictions: list[dict[str, Any]] = []
    excluded: list[tuple[sp.Symbol, sp.Basic]] = []
    complete = True
    if expr is None:
        return {"restrictions": restrictions, "excluded": excluded, "complete": complete}
    e = sp.sympify(expr)
    if var is None:
        syms = sorted(e.free_symbols, key=lambda s: s.name)
        var = syms[0] if syms else None

    # 分母不为零
    try:
        den = sp.denom(sp.together(e))
    except Exception:  # noqa: BLE001
        den = sp.Integer(1)
    if den != 1 and var is not None:
        try:
            roots = sp.solve(sp.Eq(den, 0), var)
        except Exception:  # noqa: BLE001
            roots = []
        pts = []
        for r in roots:
            if isinstance(r, sp.Set):
                sub, inc = _point_values(r)
                pts.extend(sub)
                complete = complete and not inc
            else:
                pts.append(r)
        for p in pts:
            excluded.append((var, p))
        restrictions.append({
            "kind": "denominator",
            "text": f"分母不为零：{_t(den)} ≠ 0" + (f" ⇒ " + "、".join(
                f"{_t(var)} ≠ {_t(p)}" for p in pts) if pts else ""),
            "subexpr": den, "points": pts, "must_be_nonzero": True,
        })

    # 对数
    for lg in e.atoms(sp.log):
        arg = lg.args[0]
        restrictions.append({"kind": "log", "text": f"对数要求参数 > 0：{_t(arg)} > 0",
                             "subexpr": arg, "must_be_positive": True})
    # 偶次根式
    for rt in e.atoms(sp.sqrt):
        arg = rt.args[0]
        restrictions.append({"kind": "sqrt", "text": f"偶次根式要求被开方式 ≥ 0：{_t(arg)} ≥ 0",
                             "subexpr": arg, "must_be_nonnegative": True})
    # 反三角
    for fn, name in ((sp.asin, "arcsin"), (sp.acos, "arccos")):
        for node in e.atoms(fn):
            arg = node.args[0]
            restrictions.append({"kind": name, "text": f"{name} 要求 −1 ≤ {_t(arg)} ≤ 1",
                                 "subexpr": arg, "must_be_in_unit": True})
    # tan / cot / sec / csc 极点
    for fn, name, wording in ((sp.tan, "tan", "cos"), (sp.cot, "cot", "sin"),
                              (sp.sec, "sec", "cos"), (sp.csc, "csc", "sin")):
        for node in e.atoms(fn):
            arg = node.args[0]
            restrictions.append({"kind": name, "text": f"{name} 的极点：{wording}({_t(arg)}) ≠ 0",
                                 "subexpr": arg, "pole_fn": fn})

    # 兜底：sympy 的奇点分析（捕捉上面没覆盖到的情形）
    if var is not None:
        try:
            sing = sp.singularities(sp.together(e), var)
            pts, inc = _point_values(sing)
            known = {(str(s), str(p)) for s, p in excluded}
            for p in pts:
                if (str(var), str(p)) not in known:
                    excluded.append((var, p))
            if pts:
                restrictions.append({
                    "kind": "singularity",
                    "text": "表达式在 " + "、".join(f"{_t(var)} = {_t(p)}" for p in pts) + " 处无定义（奇点分析）",
                    "points": pts,
                })
            if inc:
                complete = False
        except Exception:  # noqa: BLE001
            complete = False
    return {"restrictions": restrictions, "excluded": excluded, "complete": complete}


def _point_domain_status(expr: Any, var: sp.Symbol | None, point: sp.Basic) -> dict[str, Any]:
    """判断指定点是否落在定义域内，并给出原因与该点处的取值。"""
    reasons: list[str] = []
    info = _domain_restrictions(expr, var)
    subs = {var: point} if var is not None else {}
    illegal = False
    for res in info["restrictions"]:
        kind = res.get("kind")
        try:
            if kind == "singularity":
                if any(sp.simplify(point - p) == 0 for p in res.get("points", [])):
                    illegal = True
                    reasons.append(res["text"])
                continue
            sub = res.get("subexpr")
            if sub is None:
                continue
            val = _ev_complex(sub, subs)
            if val is None:
                continue
            if res.get("must_be_nonzero") and sp.simplify(val) == 0:
                illegal = True
                reasons.append(f"{res['text']}；在 {_t(var)} = {_t(point)} 处分母为 0")
            if res.get("must_be_positive"):
                num = sp.N(val, 12)
                if num.is_real is not True:
                    illegal = True
                    reasons.append(f"{res['text']}；在 {_t(var)} = {_t(point)} 处该参数为 {num}（非实数）")
                elif num <= 0:
                    illegal = True
                    reasons.append(f"{res['text']}；在 {_t(var)} = {_t(point)} 处该参数为 {_t(sp.simplify(val))} ≤ 0")
            if res.get("must_be_nonnegative"):
                num = sp.N(val, 12)
                if num.is_real is not True:
                    illegal = True
                    reasons.append(f"{res['text']}；在 {_t(var)} = {_t(point)} 处该参数为 {num}（非实数）")
                elif num < 0:
                    illegal = True
                    reasons.append(f"{res['text']}；在 {_t(var)} = {_t(point)} 处该参数为 {_t(sp.simplify(val))} < 0")
            if res.get("must_be_in_unit"):
                num = sp.N(val, 12)
                if num.is_real is not True or num < -1 or num > 1:
                    illegal = True
                    reasons.append(f"{res['text']}；在 {_t(var)} = {_t(point)} 处该参数为 {num}")
            if res.get("pole_fn") is not None:
                fn = res["pole_fn"]
                other = sp.cos if fn is sp.tan or fn is sp.sec else sp.sin
                if sp.simplify(other(val)) == 0:
                    illegal = True
                    reasons.append(f"{res['text']}；在 {_t(var)} = {_t(point)} 处取到极点")
        except Exception:  # noqa: BLE001
            continue
    value = None if illegal else _ev_complex(expr, subs)
    if not illegal and value is None and expr is not None and not getattr(
            sp.sympify(expr).subs(subs), "free_symbols", None):
        illegal = None  # 类型不确定
    return {"legal": (False if illegal is True else (None if illegal is None else True)),
            "reasons": reasons, "value": value, "restrictions": info["restrictions"],
            "atomic_singularities_complete": info["complete"]}


def _removable_caveats(lhs: Any, rhs: Any, var: sp.Symbol | None) -> list[dict[str, Any]]:
    """发现「差式化简为 0，但某个式子在该点无定义」的定义域陷阱（如 x≠1）。"""
    caveats: list[dict[str, Any]] = []
    if var is None:
        return caveats
    for side, other in ((lhs, rhs), (rhs, lhs)):
        if side is None or other is None:
            continue
        try:
            pts, _inc = _point_values(sp.singularities(sp.together(sp.sympify(side)), var))
        except Exception:  # noqa: BLE001
            continue
        for p in pts:
            if _ev(other, {var: p}) is None:
                continue
            caveats.append({
                "point": {var: p},
                "text": (f"{_t(var)} = {_t(p)} 处 {_t(sp.sympify(side))} 无定义，而 {_t(sp.sympify(other))} 有定义："
                         f"等式只在 {_t(var)} ≠ {_t(p)} 上成立（约去公因子改变了定义域）"),
                "kind": "removable_singularity",
            })
    return caveats


def _branch_caveats(integrand: Any, var: sp.Symbol | None, candidate: Any) -> list[dict[str, Any]]:
    """原函数的分支/定义域问题：如 1/x 的原函数要写成 log|x| 或分段。"""
    caveats: list[dict[str, Any]] = []
    if var is None or candidate is None:
        return caveats
    cand = sp.sympify(candidate)
    for lg in cand.atoms(sp.log):
        arg = lg.args[0]
        found = None
        for p in (sp.Rational(-1, 2), sp.Rational(-1), sp.Rational(-2), sp.Rational(-3, 2)):
            av = _ev(arg, {var: p})
            iv = _ev(integrand, {var: p})
            if av is not None and iv is not None and av < 0:
                found = (p, av, iv)
                break
        if found is not None:
            p, av, iv = found
            caveats.append({
                "point": {var: p},
                "text": (f"候选原函数含 log({_t(arg)})，而 log 只在 {_t(arg)} > 0 时有定义；"
                         f"被积函数 {_t(sp.sympify(integrand))} 在 {_t(var)} = {_t(p)} 处仍有定义"
                         f"（取值 {iv:.6g}）⇒ 完整定义域上的原函数应写成 log|{_t(arg)}| 或分段形式"
                         f"（例如 F(x) = log(x) + C₁ (x > 0)、log(−x) + C₂ (x < 0)）。"
                         "本次验证只证明了在 log 的自然定义域内导数正确"),
                "kind": "branch",
            })
    for rt in cand.atoms(sp.sqrt):
        arg = rt.args[0]
        for p in (sp.Rational(-1, 2), sp.Rational(-2)):
            av = _ev(arg, {var: p})
            iv = _ev(integrand, {var: p})
            if av is not None and iv is not None and av < 0:
                caveats.append({
                    "point": {var: p},
                    "text": (f"候选原函数含 sqrt({_t(arg)})，要求 {_t(arg)} ≥ 0；被积函数在 {_t(var)} = {_t(p)} "
                             "处仍有定义 ⇒ 原函数的定义域小于被积函数，需要分段或另行说明"),
                    "kind": "branch",
                })
                break
    return caveats


# ---------------------------------------------------------------------------
# 独立数值机制：有限差分 / 蒙特卡洛
# ---------------------------------------------------------------------------

def _finite_difference_table(
    expr: Any, var: sp.Symbol, candidate: Any, *, h: sp.Basic | None = None
) -> dict[str, Any]:
    """用中心差分独立估计 d(expr)/dvar，并与候选值比较（与符号求导完全独立的机制）。"""
    if var is None or expr is None or candidate is None:
        return {"agree": None, "note": "有限差分需要自变量与被复核的两个表达式"}
    step = h or sp.Rational(1, 10 ** 6)
    rows: list[dict[str, Any]] = []
    max_dev = 0.0
    worst = None
    for x0 in (sp.Rational(1, 3), sp.Rational(1, 2), sp.Rational(2, 3), sp.Rational(3, 2), sp.Rational(7, 3)):
        fp = _ev(expr, {var: x0 + step})
        fm = _ev(expr, {var: x0 - step})
        cv = _ev(candidate, {var: x0})
        if fp is None or fm is None or cv is None:
            continue
        fd = (fp - fm) / (2 * float(step))
        denom = max(1.0, abs(fd), abs(cv))
        dev = abs(fd - cv) / denom
        row = {"point": {var: x0}, "finite_difference": fd, "candidate": cv, "rel_dev": dev}
        rows.append(row)
        if dev > max_dev:
            max_dev, worst = dev, row
    if not rows:
        return {"agree": None, "note": "没有可用的有限差分抽样点", "rows": rows}
    agree = bool(max_dev <= 1e-4)
    note = (f"中心差分（步长 {_t(step)}）在 {len(rows)} 个点上与候选导数"
            + ("一致" if agree else "不一致")
            + f"，最大相对偏差 {max_dev:.3e}")
    if worst is not None and not agree:
        note += (f"；反例：{_points_text(worst['point'])} 处差分导数 {worst['finite_difference']:.12g} "
                 f"vs 候选 {worst['candidate']:.12g}")
    return {"agree": agree, "max_rel_dev": max_dev, "rows": rows, "note": note, "worst": worst}


def _make_rng(seed: int = 20240517) -> random.Random:
    return random.Random(seed)


def _sampler(dist: str, params: dict[str, Any], var: sp.Symbol) -> Callable[[random.Random], float] | None:
    d = (dist or "").lower()
    if d in ("binomial", "bernoulli"):
        m = int(params.get("m") or params.get("n") or 1)
        p = float(params.get("p"))

        def samp(rng: random.Random) -> float:
            return float(sum(1 for _ in range(m) if rng.random() < p))
        return samp
    if d == "poisson":
        lam = float(params.get("lam"))

        def samp(rng: random.Random) -> float:  # Knuth
            if lam <= 0:
                return 0.0
            if lam > 30:  # 正态近似，避免超长循环
                return max(0.0, rng.gauss(lam, math.sqrt(lam)))
            limit, k, prod = math.exp(-lam), 0, 1.0
            while True:
                prod *= rng.random()
                if prod <= limit:
                    return float(k)
                k += 1
        return samp
    if d == "geometric":
        p = float(params.get("p"))

        def samp(rng: random.Random) -> float:
            u = max(rng.random(), 1e-12)
            return float(math.ceil(math.log(u) / math.log(1 - p))) if p < 1 else 1.0
        return samp
    if d == "uniform":
        lo = float(params.get("lower") if params.get("lower") is not None else 0)
        hi = float(params.get("upper") if params.get("upper") is not None else 1)

        def samp(rng: random.Random) -> float:
            return rng.uniform(lo, hi)
        return samp
    if d in ("exponential", "exp"):
        lam = float(params.get("lam"))

        def samp(rng: random.Random) -> float:
            u = max(rng.random(), 1e-12)
            return -math.log(u) / lam
        return samp
    if d in ("normal", "gauss", "正态"):
        mu = float(params.get("mu") or 0)
        sigma = float(params.get("sigma") or 1)

        def samp(rng: random.Random) -> float:
            return rng.gauss(mu, sigma)
        return samp
    return None


def _monte_carlo(
    sampler: Callable[[random.Random], float],
    g: Callable[[float], float],
    *,
    trials: int = 200_000,
    seed: int = 20240517,
) -> dict[str, Any]:
    rng = _make_rng(seed)
    total = 0.0
    total2 = 0.0
    n = 0
    for _ in range(trials):
        try:
            v = g(sampler(rng))
        except Exception:  # noqa: BLE001
            continue
        if v is None or not math.isfinite(v):
            continue
        total += v
        total2 += v * v
        n += 1
    if n == 0:
        return {"trials": 0, "mean": None, "standard_error": None}
    mean = total / n
    var = max(total2 / n - mean * mean, 0.0)
    se = math.sqrt(var / n)
    return {"trials": n, "mean": mean, "standard_error": se,
            "ci95": [mean - 1.96 * se, mean + 1.96 * se]}


# ---------------------------------------------------------------------------
# 条件解析与使用
# ---------------------------------------------------------------------------

_REL_TWO = (("!=", "ne"), ("≠", "ne"), ("==", "eq"))
_REL_ONE = (("=", "eq"),)


def _split_relation_text(text: str) -> tuple[str | None, str | None, str | None]:
    """把 ``x != 0`` / ``x = 2`` 这类文本拆成 (算子, 左, 右)。

    只处理 sympy 的 ``parse_expr`` 会「提前结构化求真」而失真的 ``=`` 与 ``!=``
    （``sp.Symbol('x') == 2`` 返回 False，``!= 0`` 返回 True），
    ``< > <= >=`` 交给 ``A.parse`` 原生处理。
    """
    for op, code in _REL_TWO:
        if op in text:
            left, _, right = text.partition(op)
            return code, left.strip(), right.strip()
    for op, code in _REL_ONE:
        if op in text:
            left, _, right = text.partition(op)
            return code, left.strip(), right.strip()
    return None, None, None


def _parse_relation_side(text: str) -> sp.Basic:
    try:
        return A.parse(text)
    except Exception:  # noqa: BLE001
        return sp.sympify(text)


def _condition_entries(conditions: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for raw in _as_list(conditions):
        text = str(raw).strip()
        if not text:
            continue
        entry: dict[str, Any] = {"text": text, "rel": None, "kind": "unparsed", "used": False, "how": None}
        low = text.lower()
        if any(k in text for k in ("可逆", "满秩", "非奇异")) or "invertib" in low:
            entry["kind"] = "matrix_invertible"
        elif "正态" in text or "normal" in low:
            entry["kind"] = "distribution"
        elif "独立" in text or "iid" in low or "同分布" in text:
            entry["kind"] = "distribution"
        else:
            code, left, right = _split_relation_text(text)
            if code is not None and left and right:
                try:
                    lobj, robj = _parse_relation_side(left), _parse_relation_side(right)
                    entry["rel"] = sp.Eq(lobj, robj) if code == "eq" else sp.Ne(lobj, robj)
                    entry["kind"] = "relation"
                except Exception:  # noqa: BLE001
                    entry["kind"] = "unparsed"
            else:
                try:
                    rel = A.parse(text)
                    if isinstance(rel, sp.Rel):
                        entry["rel"] = rel
                        entry["kind"] = "relation"
                    elif rel is sp.true:
                        entry["kind"] = "tautology"
                    elif rel is sp.false:
                        entry["kind"] = "contradiction"
                    else:
                        entry["rel"] = rel
                        entry["kind"] = "expression"
                except ValueError:
                    entry["kind"] = "unparsed"
        entries.append(entry)
    return entries


def _condition_assumptions(entries: Sequence[dict[str, Any]]) -> list[Any]:
    """把用户条件转成 sympy 新假设（供 refine 使用）。"""
    out: list[Any] = []
    for e in entries:
        rel = e.get("rel")
        if not isinstance(rel, sp.Rel):
            continue
        try:
            lhs = rel.lhs
            if not isinstance(lhs, sp.Symbol):
                continue
            if isinstance(rel, sp.StrictGreaterThan) and rel.rhs == 0:
                out.append(sp.Q.positive(lhs))
            elif isinstance(rel, sp.GreaterThan) and rel.rhs == 0:
                out.append(sp.Q.nonnegative(lhs))
            elif isinstance(rel, sp.StrictLessThan) and rel.rhs == 0:
                out.append(sp.Q.negative(lhs))
            elif isinstance(rel, sp.LessThan) and rel.rhs == 0:
                out.append(sp.Q.nonpositive(lhs))
        except Exception:  # noqa: BLE001
            continue
    return out


def _use_conditions_on_difference(diff: Any, entries: list[dict[str, Any]]) -> tuple[sp.Basic, list[str]]:
    """尝试用用户条件（refine）辅助化简差式；返回 (化简结果, 真正用到的条件文本)。"""
    used: list[str] = []
    assumptions = _condition_assumptions(entries)
    d = sp.sympify(diff)
    if not assumptions:
        return d, used
    refined = d
    for assume in assumptions:
        try:
            new = sp.refine(refined, assume)
        except Exception:  # noqa: BLE001
            continue
        if new == refined:
            continue
        # 条件确实改变了化简结果 → 视为被使用
        entry_text = None
        for e in entries:
            rel = e.get("rel")
            if isinstance(rel, sp.Rel) and _t(rel.lhs) == _t(getattr(assume, "arg", assume)):
                entry_text = e["text"]
                break
        if entry_text:
            used.append(entry_text)
        refined = new
    return refined, used


def _caveats_covered(caveats: Sequence[dict[str, Any]], entries: Sequence[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """把定义域陷阱分成「已被用户条件覆盖」与「未处理」两类。"""
    covered: list[str] = []
    unhandled: list[str] = []
    for cav in caveats:
        matched = None
        pts = cav.get("point", {})
        for e in entries:
            rel = e.get("rel")
            if not isinstance(rel, sp.Rel):
                continue
            try:
                lhs, rhs = rel.lhs, rel.rhs
                if isinstance(rel, (sp.StrictGreaterThan, sp.GreaterThan, sp.StrictLessThan, sp.LessThan)):
                    continue
                eq_lhs = sp.simplify(lhs)
                for sym, val in pts.items():
                    if eq_lhs == sym and sp.simplify(rhs - val) == 0:
                        matched = e["text"]
            except Exception:  # noqa: BLE001
                continue
        if matched:
            covered.append(f"{cav['text']}（已由用户条件「{matched}」覆盖）")
        else:
            unhandled.append(f"{cav['text']}（未处理：用户未声明该限制）")
    return covered, unhandled


# ---------------------------------------------------------------------------
# 精确概率 / 期望（独立于其他 handler 模块，直接用 scipy）
# ---------------------------------------------------------------------------

def _scipy_stats() -> Any:
    try:
        from scipy import stats as st  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise _CannotVerify(f"未安装 scipy，无法做精确分布计算：{exc}") from exc
    return st


def _need_float(params: dict[str, Any], key: str, *, alias: str | None = None) -> float:
    value = params.get(key)
    if value is None and alias:
        value = params.get(alias)
    if value is None:
        raise _CannotVerify(f"缺少分布参数 {key}")
    try:
        return float(sp.N(_parse_expr(value, what=f"参数 {key}")))
    except Exception as exc:  # noqa: BLE001
        raise _CannotVerify(f"分布参数 {key} 无法转成数值：{exc}") from exc


def _exact_prob(dist: str, params: dict[str, Any], spec: dict[str, Any]) -> tuple[float | None, str]:
    """用 scipy 精确分布函数独立重算概率。spec: {kind, b1, b2, bar, n}"""
    st = _scipy_stats()
    d = (dist or "").lower()
    kind = spec["kind"]
    b1, b2 = spec.get("b1"), spec.get("b2")
    if d in ("binomial", "bernoulli"):
        m = params.get("m")
        m = int(float(sp.N(_parse_expr(m)))) if m is not None else 1
        p = _need_float(params, "p")
        rv = st.binom(m, p)
        method = f"B({m}, {p}) 精确分布函数（scipy）"
    elif d == "poisson":
        lam = _need_float(params, "lam", alias="lambda")
        rv = st.poisson(lam)
        method = f"P({lam}) 精确分布函数（scipy）"
    elif d in ("geometric",):
        p = _need_float(params, "p")
        rv = st.geom(p)
        method = f"Geom({p}) 精确分布函数（scipy）"
    elif d in ("normal", "gauss", "正态"):
        mu = _need_float(params, "mu") if params.get("mu") is not None else 0.0
        sigma = _need_float(params, "sigma") if params.get("sigma") is not None else 1.0
        rv = st.norm(mu, sigma)
        method = f"N({mu}, {sigma}²) 精确分布函数（scipy）"
    elif d in ("uniform", "均匀"):
        lo = _need_float(params, "lower") if params.get("lower") is not None else 0.0
        hi = _need_float(params, "upper") if params.get("upper") is not None else 1.0
        rv = st.uniform(lo, hi - lo)
        method = f"U({lo}, {hi}) 精确分布函数（scipy）"
    elif d in ("exponential", "exp"):
        lam = _need_float(params, "lam", alias="lambda")
        rv = st.expon(scale=1.0 / lam)
        method = f"Exp(λ={lam}) 精确分布函数（scipy）"
    else:
        return None, f"未知分布 {dist!r}，无法独立重算"
    try:
        if spec.get("bar"):
            n = int(spec.get("n") or 0)
            if n <= 0:
                return None, "样本均值记号 X̄ 需要给出样本量 n"
            mu1, var1 = float(rv.mean()), float(rv.var())
            se = math.sqrt(var1 / n)
            approx = st.norm(mu1, se)
            val = _apply_prob(approx, kind, b1, b2, discrete=False, corr=1.0 / (2 * n))
            return float(val), method + f"；样本均值用 CLT 正态近似 N({mu1}, {var1}/{n})（近似）"
        discrete = d in ("binomial", "bernoulli", "poisson", "geometric")
        corr = 0.5 if discrete else 0.0
        val = _apply_prob(rv, kind, b1, b2, discrete=discrete, corr=corr)
        return float(val), method
    except Exception as exc:  # noqa: BLE001
        return None, f"精确分布计算失败：{type(exc).__name__}: {exc}"


def _apply_prob(rv: Any, kind: str, b1: Any, b2: Any, *, discrete: bool, corr: float) -> float:
    f1 = None if b1 is None else float(sp.N(_parse_expr(b1, what="概率界")))
    f2 = None if b2 is None else float(sp.N(_parse_expr(b2, what="概率界")))
    if kind == "eq":
        if discrete:
            k = f1
            return float(rv.cdf(k) - rv.cdf(k - 1))
        return 0.0
    if kind == "le":
        return float(rv.cdf(f1 + corr))
    if kind == "lt":
        return float(rv.cdf(f1 - (0.5 if discrete else 0.0)))
    if kind == "ge":
        return float(rv.sf(f1 - 1 - corr))
    if kind == "gt":
        return float(rv.sf(f1 + (0.5 if discrete else 0.0)))
    if kind == "between":
        return float(rv.cdf(f2 + corr) - rv.cdf(f1 - 1 - corr if discrete else f1))
    raise _CannotVerify(f"不支持的概率记号：{kind}")


def _parse_prob_spec(text: str) -> dict[str, Any] | None:
    """解析 ``P(X <= 3)`` / ``P(2 <= X <= 5)`` / ``P(X = 2)`` / ``P(|X| > 1)`` 之类记号。"""
    s = str(text).replace(" ", "")
    m = None
    import re  # noqa: PLC0415

    mm = re.match(r"^[Pp]\s*[（(](.+)[)）]$", str(text).strip())
    if not mm:
        mm = re.match(r"^[Pp]\s*[（(](.+)[)）]$", s)
    if not mm:
        return None
    body = mm.group(1)
    bar = ("̄" in body) or ("bar" in body.lower()) or ("mean" in body.lower()) or ("X̄" in body)
    body = body.replace("̄", "").replace("X̄", "X").replace("x̄", "x")
    body = body.replace("X_bar", "X").replace("x_bar", "x")
    body = body.replace("bar", "")
    # |X| 形式
    absm = re.match(r"^\|([A-Za-z])\|([<>]=?)(.+)$", body)
    if absm:
        c = absm.group(3)
        if absm.group(2).startswith("<"):
            return {"kind": "between", "b1": f"-({c})", "b2": c, "bar": bar}
        return None
    # a <= X <= b
    m3 = re.match(r"^(.+?)(<=|<)([A-Za-z])(<=|<)(.+)$", body)
    if m3:
        return {"kind": "between", "b1": m3.group(1), "b2": m3.group(5), "bar": bar}
    # X op c
    m2 = re.match(r"^([A-Za-z])(<=|>=|<|>|=)(.+)$", body)
    if m2:
        op = {"<=": "le", "<": "lt", ">=": "ge", ">": "gt", "=": "eq"}[m2.group(2)]
        return {"kind": op, "b1": m2.group(3), "b2": None, "bar": bar}
    # c op X
    m4 = re.match(r"^(.+?)(<=|>=|<|>)([A-Za-z])$", body)
    if m4:
        op = {"<=": "ge", ">=": "le", "<": "gt", ">": "lt"}[m4.group(2)]
        return {"kind": op, "b1": m4.group(1), "b2": None, "bar": bar}
    return None


def _dist_descriptor(dist: str, params: dict[str, Any], x: sp.Symbol) -> dict[str, Any]:
    """把分布参数转成 sympy 的 pdf/支撑/均值/方差（用于期望与方差复核）。"""
    d = (dist or "").lower()
    k = sp.Symbol("k", integer=True, nonnegative=True)
    if d in ("binomial", "bernoulli"):
        m = params.get("m")
        m = int(float(sp.N(_parse_expr(m)))) if m is not None else 1
        p = _parse_expr(params.get("p"), what="p")
        pdf = sp.binomial(m, k) * p ** k * (1 - p) ** (m - k)
        return {"discrete": True, "sum_var": k, "pdf": pdf, "support": (sp.Integer(0), sp.Integer(m)),
                "mean": m * p, "variance": m * p * (1 - p), "name": f"B({m}, {_t(p)})"}
    if d == "poisson":
        lam = _parse_expr(params.get("lam") if params.get("lam") is not None else params.get("lambda"), what="λ")
        pdf = sp.exp(-lam) * lam ** k / sp.factorial(k)
        return {"discrete": True, "sum_var": k, "pdf": pdf, "support": (sp.Integer(0), sp.oo),
                "mean": lam, "variance": lam, "name": f"P({_t(lam)})"}
    if d == "geometric":
        p = _parse_expr(params.get("p"), what="p")
        pdf = (1 - p) ** (k - 1) * p
        return {"discrete": True, "sum_var": k, "pdf": pdf, "support": (sp.Integer(1), sp.oo),
                "mean": 1 / p, "variance": (1 - p) / p ** 2, "name": f"Geom({_t(p)})"}
    if d == "uniform":
        lo = _parse_expr(params.get("lower") if params.get("lower") is not None else 0, what="a")
        hi = _parse_expr(params.get("upper") if params.get("upper") is not None else 1, what="b")
        pdf = 1 / (hi - lo)
        return {"discrete": False, "sum_var": x, "pdf": pdf, "support": (lo, hi),
                "mean": (lo + hi) / 2, "variance": (hi - lo) ** 2 / 12, "name": f"U({_t(lo)}, {_t(hi)})"}
    if d in ("exponential", "exp"):
        lam = _parse_expr(params.get("lam") if params.get("lam") is not None else params.get("lambda"), what="λ")
        pdf = lam * sp.exp(-lam * x)
        return {"discrete": False, "sum_var": x, "pdf": pdf, "support": (sp.Integer(0), sp.oo),
                "mean": 1 / lam, "variance": 1 / lam ** 2, "name": f"Exp(λ={_t(lam)})"}
    if d in ("normal", "gauss", "正态"):
        mu = _parse_expr(params.get("mu") if params.get("mu") is not None else 0, what="μ")
        sigma = _parse_expr(params.get("sigma") if params.get("sigma") is not None else 1, what="σ")
        pdf = sp.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * sp.sqrt(2 * sp.pi))
        return {"discrete": False, "sum_var": x, "pdf": pdf, "support": (-sp.oo, sp.oo),
                "mean": mu, "variance": sigma ** 2, "name": f"N({_t(mu)}, {_t(sigma)}²)"}
    raise _CannotVerify(f"未知分布 {dist!r}（支持 binomial/poisson/geometric/uniform/exponential/normal）")


def _dist_expectation(desc: dict[str, Any], g: sp.Basic, *, allow_symbolic: bool = True) -> dict[str, Any]:
    """独立计算 E[g(X)]：符号积分/求和 + scipy 数值 + 蒙特卡洛的兜底。"""
    out: dict[str, Any] = {"symbolic": None, "numeric": None, "method": []}
    v = desc["sum_var"]
    lo, hi = desc["support"]
    integrand = sp.sympify(desc["pdf"]) * sp.sympify(g)
    if allow_symbolic:
        try:
            if desc["discrete"]:
                val = sp.summation(integrand, (v, lo, hi))
            else:
                val = sp.Integral(integrand, (v, lo, hi)).doit()
            val = sp.simplify(val)
            if not getattr(val, "free_symbols", None) and val.is_finite is not False:
                out["symbolic"] = val
                out["method"].append("符号求和/积分（sympy）")
        except Exception:  # noqa: BLE001
            pass
    return out


def _dist_numeric_expectation(desc: dict[str, Any], g: sp.Basic, *, trials: int = 200_000) -> dict[str, Any]:
    """用 scipy 数值积分 + 蒙特卡洛算 E[g(X)]（与符号路径独立的机制）。"""
    v = desc["sum_var"]
    lo, hi = desc["support"]
    numeric = None
    method = None
    try:
        from scipy import integrate as si  # noqa: PLC0415
        pdf = sp.lambdify(v, desc["pdf"], "math")
        gg = sp.lambdify(v, sp.sympify(g), "math")
        if desc["discrete"]:
            numeric = None  # 离散型改用蒙特卡洛/符号
        else:
            lo_f = -math.inf if lo is -sp.oo or lo == -sp.oo else float(lo)
            hi_f = math.inf if hi == sp.oo else float(hi)

            def integrand(x: float) -> float:
                try:
                    return pdf(x) * gg(x)
                except Exception:  # noqa: BLE001
                    return 0.0
            val, err = si.quad(integrand, lo_f, hi_f, limit=200)
            numeric = (val, err)
            method = f"scipy.quad 数值积分（误差估计 {err:.2e}）"
    except Exception:  # noqa: BLE001
        numeric = None
    return {"numeric": numeric, "method": method}


# ---------------------------------------------------------------------------
# 各 kind 的复核实现
# ---------------------------------------------------------------------------

def _base_result(passed: bool | None, level: str, method: str, evidence: dict[str, Any],
                 *, cannot: str | None = None, unhandled: list[str] | None = None,
                 extra: dict[str, Any] | None = None, used_conditions: list[str] | None = None,
                 covered_conditions: list[str] | None = None) -> dict[str, Any]:
    return {"passed": passed, "level": level, "method": method, "evidence": evidence,
            "cannot": cannot, "unhandled": list(unhandled or []),
            "extra": dict(extra or {}), "used_conditions": list(used_conditions or []),
            "covered_conditions": list(covered_conditions or [])}


def _need(ctx: dict[str, Any], *names: str) -> None:
    missing = [n for n in names if ctx.get(n) is None]
    if missing:
        raise ValueError("复核缺少必需参数：" + "、".join(missing))


def _h_derivative(ctx: dict[str, Any]) -> dict[str, Any]:
    _need(ctx, "expr", "cand")
    expr, cand = ctx["expr"], ctx["cand"]
    var = ctx["var"] or C.resolve_var(expr, None)
    ok, detail = C.check_derivative(expr, var, cand)
    expected = sp.diff(expr, var)
    numeric = _numeric_compare(expected, cand, [var])
    fd = _finite_difference_table(expr, var, cand)
    decision = _decide_equality(expected, cand, [var], numeric=numeric)
    diff_refined, used = _use_conditions_on_difference(expected - cand, ctx["conditions"])
    if diff_refined == 0 and decision["passed"] is not True:
        decision = {"passed": True, "level": LEVEL_SYMBOLIC,
                    "method": "在用户声明的条件下符号差式化简为 0",
                    "evidence": decision["evidence"], "cannot": None}
    evidence = {
        "independent_derivative": _t(expected),
        "independent_derivative_latex": detail.get("expected_latex"),
        "sympy_core_method": detail.get("method"),
        "difference": decision["evidence"].get("difference"),
        "numeric": decision["evidence"].get("numeric"),
        "finite_difference": {"agree": fd.get("agree"), "max_rel_dev": fd.get("max_rel_dev"),
                              "note": fd.get("note")},
        "check_derivative_agree": bool(ok),
        "condition_refined_difference": _t(diff_refined),
    }
    if decision["evidence"].get("counterexample"):
        evidence["counterexample"] = decision["evidence"]["counterexample"]
    level = decision["level"]
    if decision["passed"] is False and fd.get("agree") is False and level == LEVEL_SYMBOLIC:
        level = LEVEL_SYMBOLIC
    methods = [{"method": "独立符号求导 + 差式化简（sympy_core.check_derivative）", "passed": bool(ok),
                "evidence": {"expected": detail.get("expected"), "difference": decision["evidence"].get("difference")}},
               {"method": "高精度数值抽样逐点比较", "passed": numeric.get("agree"),
                "evidence": {"max_rel_dev": numeric.get("max_rel_dev"), "note": numeric.get("note"),
                             "counterexample": numeric.get("worst", {}).get("point") if numeric.get("worst") else None}},
               {"method": "中心差分独立估计导数（与符号求导无关的机制）", "passed": fd.get("agree"),
                "evidence": {"max_rel_dev": fd.get("max_rel_dev"), "note": fd.get("note")}}]
    return _base_result(decision["passed"], level, decision["method"], evidence,
                        cannot=decision["cannot"], unhandled=_unhandled_from(ctx, used),
                        used_conditions=used, extra={"methods": methods})


def _h_integral(ctx: dict[str, Any]) -> dict[str, Any]:
    _need(ctx, "expr", "cand")
    integrand, cand = ctx["expr"], ctx["cand"]
    var = ctx["var"] or C.resolve_var(integrand, None)
    ok, detail = C.check_antiderivative(integrand, var, cand)
    derivative = sp.diff(cand, var)
    numeric = _numeric_compare(integrand, derivative, [var])
    fd = _finite_difference_table(cand, var, integrand)
    decision = _decide_equality(integrand, derivative, [var], numeric=numeric)
    if not ok and decision["passed"] is True:
        decision["passed"] = False
        decision["method"] += "；但 sympy_core.check_antiderivative 未通过，需人工核对"
    caveats = _branch_caveats(integrand, var, cand)
    covered, unhandled_cav = _caveats_covered(caveats, ctx["conditions"])
    diff_refined, used = _use_conditions_on_difference(integrand - derivative, ctx["conditions"])
    evidence = {
        "derivative_of_candidate": _t(derivative),
        "derivative_of_candidate_latex": detail.get("derivative_latex"),
        "sympy_core_method": detail.get("method"),
        "difference": decision["evidence"].get("difference"),
        "numeric": decision["evidence"].get("numeric"),
        "finite_difference": {"agree": fd.get("agree"), "max_rel_dev": fd.get("max_rel_dev"), "note": fd.get("note")},
        "check_antiderivative_agree": bool(ok),
        "domain_caveats": [c["text"] for c in caveats],
    }
    if decision["evidence"].get("counterexample"):
        evidence["counterexample"] = decision["evidence"]["counterexample"]
    methods = [{"method": "对候选原函数求导并与被积函数作符号比较（sympy_core.check_antiderivative）",
                "passed": bool(ok), "evidence": {"derivative_of_candidate": detail.get("derivative_of_candidate"),
                                                 "difference": decision["evidence"].get("difference")}},
               {"method": "高精度数值抽样逐点比较", "passed": numeric.get("agree"),
                "evidence": {"max_rel_dev": numeric.get("max_rel_dev"), "note": numeric.get("note")}},
               {"method": "中心差分独立估计候选原函数的导数", "passed": fd.get("agree"),
                "evidence": {"max_rel_dev": fd.get("max_rel_dev"), "note": fd.get("note")}}]
    if caveats:
        methods.append({"method": "定义域/分支完整性检查（原函数是否覆盖被积函数的完整定义域）",
                        "passed": False if unhandled_cav else True,
                        "evidence": {"caveats": [c["text"] for c in caveats], "covered": covered}})
    unhandled = _unhandled_from(ctx, used) + unhandled_cav
    return _base_result(decision["passed"], decision["level"], decision["method"], evidence,
                        cannot=decision["cannot"], unhandled=unhandled, used_conditions=used + covered,
                        extra={"methods": methods})


def _h_definite_integral(ctx: dict[str, Any]) -> dict[str, Any]:
    _need(ctx, "expr", "cand", "lower", "upper")
    expr, cand = ctx["expr"], ctx["cand"]
    var = ctx["var"] or C.resolve_var(expr, None)
    lo = C.parse_point(ctx["lower"], [var])
    hi = C.parse_point(ctx["upper"], [var])
    integral = sp.Integral(expr, (var, lo, hi))
    evidence: dict[str, Any] = {"integral": _t(integral), "lower": _t(lo), "upper": _t(hi)}
    methods: list[dict[str, Any]] = []
    symbol_ok: bool | None = None
    closed = None
    try:
        closed = integral.doit()
        if isinstance(closed, sp.Integral) or closed.has(sp.Integral):
            evidence["symbolic_note"] = "sympy 未能求出该定积分的闭式，改用数值复核"
            closed = None
        else:
            closed_s = sp.simplify(closed)
            diff = sp.simplify(closed_s - cand)
            evidence["closed_form"] = _t(closed_s)
            if diff == 0:
                symbol_ok = True
                evidence["difference"] = "0"
            else:
                z, dz, how = _symbolic_zero(diff)
                evidence["difference"] = _t(dz)
                evidence["symbolic_basis"] = how
                if z is True:
                    symbol_ok = True
                elif closed_s.is_number and sp.sympify(cand).is_number:
                    symbol_ok = False
                    closed_num = _ev(closed_s, {})
                    cand_num_ = _ev(cand, {})
                    if closed_num is not None and cand_num_ is not None:
                        evidence["counterexample"] = (
                            f"精确积分值为 {_t(closed_s)}（≈ {closed_num:.12g}），"
                            f"候选结果为 {_t(cand)}（≈ {cand_num_:.12g}）")
                    else:
                        evidence["counterexample"] = (f"精确积分值为 {_t(closed_s)}，"
                                                      f"候选结果为 {_t(cand)}（差式 {_t(dz)}）")
            methods.append({"method": "先求定积分的闭式再与候选作符号比较",
                            "passed": symbol_ok, "evidence": {"closed_form": _t(closed_s),
                                                              "difference": evidence.get("difference")}})
    except Exception as exc:  # noqa: BLE001
        evidence["symbolic_note"] = f"符号定积分失败：{type(exc).__name__}: {exc}"
    # 数值复核（与符号路径独立）
    num_int = _ev(sp.N(integral, 25), {}) if closed is None else _ev(closed, {})
    if num_int is None:
        try:
            num_int = float(sp.N(integral.evalf(25), 20))
        except Exception:  # noqa: BLE001
            num_int = None
    num_cand = _ev(cand, {})
    numeric_ok = None
    if num_int is not None and num_cand is not None:
        dev = abs(num_int - num_cand) / max(1.0, abs(num_int), abs(num_cand))
        numeric_ok = bool(dev <= 1e-6)
        evidence["numeric"] = {"integral": num_int, "candidate": num_cand, "rel_dev": dev,
                               "note": None if numeric_ok else
                               f"数值复核不一致：定积分 ≈ {num_int:.12g}，候选 ≈ {num_cand:.12g}（相对偏差 {dev:.6g}）"}
        if not numeric_ok and "counterexample" not in evidence:
            evidence["counterexample"] = evidence["numeric"]["note"]
        methods.append({"method": "高精度数值定积分（sympy evalf / 数值积分）独立重算", "passed": numeric_ok,
                        "evidence": evidence["numeric"]})
    passed = symbol_ok if symbol_ok is not None else numeric_ok
    level = LEVEL_SYMBOLIC if symbol_ok is not None else (LEVEL_NUMERIC if numeric_ok is not None else LEVEL_NONE)
    unhandled: list[str] = []
    used: list[str] = []
    if not lo.is_finite or not hi.is_finite:
        unhandled.append("积分区间无界：收敛性未单独证明，只验证了数值/符号结果一致")
    if closed is not None:
        _refined, used = _use_conditions_on_difference(sp.sympify(closed) - cand, ctx["conditions"])
    method = "定积分复核：符号闭式比较" + (" + 数值定积分重算" if numeric_ok is not None else "")
    cannot = None
    if passed is None:
        cannot = "定积分既无法求出闭式，也无法得到数值结果，无法复核"
    return _base_result(passed, level, method, evidence, cannot=cannot,
                        unhandled=_unhandled_from(ctx, used) + unhandled,
                        used_conditions=used, extra={"methods": methods})


def _h_limit(ctx: dict[str, Any]) -> dict[str, Any]:
    _need(ctx, "expr", "cand", "point")
    expr, cand = ctx["expr"], ctx["cand"]
    var = ctx["var"] or C.resolve_var(expr, None)
    point = C.parse_point(ctx["point"], [var])
    direction = ctx.get("dir")
    dirn = None if direction in (None, "", "both") else str(direction)
    evidence: dict[str, Any] = {"point": _t(point), "direction": dirn or "双侧"}
    methods: list[dict[str, Any]] = []
    sym_ok: bool | None = None
    lim = None
    try:
        lim = sp.limit(expr, var, point, dirn) if dirn else sp.limit(expr, var, point)
        if isinstance(lim, sp.Limit) or lim.has(sp.Limit):
            evidence["symbolic_note"] = "sympy 未能求出该极限，改用数值逼近"
            lim = None
        else:
            evidence["independent_limit"] = _t(lim)
            z, dz, how = _symbolic_zero(sp.sympify(lim) - sp.sympify(cand))
            if z is True:
                sym_ok = True
                evidence["difference"] = "0"
                evidence["symbolic_basis"] = how
            elif z is False:
                sym_ok = False
                evidence["difference"] = _t(dz)
                evidence["counterexample"] = f"独立求得的极限为 {_t(lim)}，候选为 {_t(cand)}；差式化简为 {_t(dz)}"
            methods.append({"method": "用 sympy 独立求极限后与候选作符号比较", "passed": sym_ok,
                            "evidence": {"independent_limit": _t(lim), "difference": evidence.get("difference")}})
    except Exception as exc:  # noqa: BLE001
        evidence["symbolic_note"] = f"符号求极限失败：{type(exc).__name__}: {exc}"
    # 数值逼近（独立机制）
    numeric_ok = None
    if point.is_finite:
        approach: list[sp.Basic] = []
        if dirn in (None, "+"):
            approach += [point + sp.Rational(1, 10 ** 5), point + sp.Rational(1, 10 ** 8)]
        if dirn in (None, "-"):
            approach += [point - sp.Rational(1, 10 ** 5), point - sp.Rational(1, 10 ** 8)]
        rows: list[dict[str, Any]] = []
        cand_val = _ev(cand, {})
        for p in approach:
            v = _ev(expr, {var: p})
            if v is None or cand_val is None:
                continue
            dev = abs(v - cand_val) / max(1.0, abs(v), abs(cand_val))
            rows.append({"point": _t(p), "value": v, "rel_dev": dev})
        if rows:
            worst = max(rows, key=lambda r: r["rel_dev"])
            numeric_ok = bool(worst["rel_dev"] <= 1e-4)
            evidence["numeric_approach"] = {"rows": rows, "max_rel_dev": worst["rel_dev"],
                                            "note": ("数值逼近在逐步靠近的取样点上与候选一致" if numeric_ok else
                                                     f"数值逼近不一致：在 {worst['point']} 处函数值 {worst['value']:.12g}，"
                                                     f"候选 {cand_val:.12g}（相对偏差 {worst['rel_dev']:.6g}）")}
            if not numeric_ok and "counterexample" not in evidence:
                evidence["counterexample"] = evidence["numeric_approach"]["note"]
            methods.append({"method": "数值逼近（自变量逐步趋近该点，独立于符号求极限）", "passed": numeric_ok,
                            "evidence": evidence["numeric_approach"]})
    passed = sym_ok if sym_ok is not None else numeric_ok
    level = LEVEL_SYMBOLIC if sym_ok is not None else (LEVEL_NUMERIC if numeric_ok is not None else LEVEL_NONE)
    unhandled = []
    if point.is_finite and not dirn:
        try:
            left = sp.limit(expr, var, point, "-")
            right = sp.limit(expr, var, point, "+")
            if left != right:
                unhandled.append(f"双侧极限不存在（左极限 {_t(left)} ≠ 右极限 {_t(right)}）；"
                                 "本次按双侧极限复核，未分别讨论左右极限")
        except Exception:  # noqa: BLE001
            unhandled.append("未验证左右极限是否相等（sympy 未能分别求出）")
    cannot = None if passed is not None else "极限既无法符号求出，也无法用数值逼近得到有效证据"
    method = "极限复核：独立求极限 + 数值逼近"
    return _base_result(passed, level, method, evidence, cannot=cannot, unhandled=unhandled,
                        extra={"methods": methods})


def _parse_solution(candidate: str, vars_: Sequence[sp.Symbol]) -> tuple[dict[sp.Symbol, Any], list[Any]]:
    """解析候选解：``2`` / ``x=2`` / ``x=2, y=3`` / ``2, 3``。"""
    text = str(candidate)
    parts = [p.strip() for p in text.split(",") if p.strip()]
    ordered: list[Any] = []
    subs: dict[sp.Symbol, Any] = {}
    for p in parts:
        pair = _split_equality(p)
        if pair is not None:
            name = pair[0].strip()
            sym = None
            for s in vars_:
                if s.name == name:
                    sym = s
                    break
            if sym is None and len(vars_) == 1:
                sym = vars_[0]
            if sym is None:
                raise ValueError(f"候选解中的变量 {name!r} 不在方程组的未知量 {[s.name for s in vars_]} 中")
            val = _parse_expr(pair[1], what="候选解取值")
            subs[sym] = val
            if sym not in ordered:
                ordered.append(val)
        else:
            val = _parse_expr(p, what="候选解")
            ordered.append(val)
    if not subs:
        if len(vars_) == 1:
            subs[vars_[0]] = ordered[0]
        elif len(ordered) == len(vars_):
            for s, v in zip(vars_, ordered):
                subs[s] = v
        elif len(ordered) > 0:
            raise ValueError("方程组有多个未知量，候选解必须写成 x=2, y=3 的形式")
    values = [subs.get(s, ordered[i] if i < len(ordered) else None) for i, s in enumerate(vars_)]
    return subs, values


def _h_equation_solution(ctx: dict[str, Any]) -> dict[str, Any]:
    eqs = ctx["equations"] or ([ctx["expr"]] if ctx.get("expr") is not None else [])
    if not eqs:
        raise ValueError("方程（组）复核需要 expr 或 equations 参数")
    eqs = [sp.sympify(e) for e in eqs]
    vars_ = ctx["symbols"] if ctx.get("symbols") else sorted(
        set().union(*[e.free_symbols for e in eqs]), key=lambda s: s.name)
    if not vars_:
        raise ValueError("方程中没有可解的未知量")
    cand_text = ctx.get("cand_text") or ctx["cand"]
    _need(ctx, "cand")
    subs, values = _parse_solution(str(cand_text), vars_)
    if any(v is None for v in values):
        raise ValueError("候选解缺少部分未知量的取值")
    ok, detail = C.check_solution(eqs, list(vars_), values)
    residuals = []
    for e in eqs:
        r = sp.simplify(e.subs(subs))
        residuals.append(r)
    exact_zero = all(_symbolic_zero(r)[0] is True for r in residuals)
    evidence: dict[str, Any] = {
        "candidate_substitution": {_t(k): _t(v) for k, v in subs.items()},
        "residuals": [_t(r) for r in residuals],
        "sympy_core_method": detail.get("method"),
    }
    methods: list[dict[str, Any]] = [{
        "method": "把候选解代回原方程逐条检查残差（sympy_core.check_solution）",
        "passed": bool(ok), "evidence": {"residuals": [_t(r) for r in residuals]}}]
    nonzero = [(eqs[i], residuals[i]) for i in range(len(eqs)) if _symbolic_zero(residuals[i])[0] is not True]
    if nonzero:
        parts = [f"第 {i + 1} 个方程代入后残差为 {_t(r)} ≠ 0" for i, (e, r) in enumerate(zip(eqs, residuals)) if
                 _symbolic_zero(r)[0] is not True]
        evidence["counterexample"] = "；".join(parts)
    # 独立机制：直接解方程，检查候选是否落在解集中
    passed = bool(ok) and exact_zero
    level = LEVEL_SYMBOLIC
    root_info = None
    try:
        if len(vars_) == 1:
            target = eqs[0] if len(eqs) == 1 else [sp.Eq(e, 0) for e in eqs]
            sols = sp.solve(target, vars_[0])
            flat: list[Any] = []
            for s in sols:
                if isinstance(s, (tuple, list)):
                    flat.extend(s)
                else:
                    flat.append(s)
            root_info = [_t(s) for s in flat]
            in_solutions = None
            if flat:
                in_solutions = any(
                    _symbolic_zero(sp.sympify(s) - sp.sympify(values[0]))[0] is True for s in flat)
            evidence["independent_solution_set"] = root_info
            methods.append({"method": "独立求解原方程并与候选解比对（与代回残差互相独立）",
                            "passed": in_solutions, "evidence": {"solutions": root_info}})
            if in_solutions is False and passed is False:
                evidence["independent_note"] = (f"独立求解得到的解集为 {root_info}，候选值不在其中"
                                                if root_info else "独立求解未得到解（可能无解）")
    except Exception as exc:  # noqa: BLE001
        evidence["independent_solution_note"] = f"独立求解失败：{type(exc).__name__}: {exc}"
    unhandled = []
    if len(vars_) > len(eqs):
        unhandled.append(f"未知量个数（{len(vars_)}）多于方程个数（{len(eqs)}）："
                         "验证的只是「该取值是解」，未讨论解集是否只有一个")
    if not nonzero and passed:
        unhandled.append("验证的是「候选值满足方程」，不涉及解的完备性（是否还有别的解）")
    return _base_result(passed, level,
                        "方程（组）复核：代回检查残差" + (" + 独立求解对照" if root_info else ""),
                        evidence, cannot=None, unhandled=unhandled, extra={"methods": methods})


def _h_identity(ctx: dict[str, Any]) -> dict[str, Any]:
    lhs, rhs = ctx["lhs"], ctx["rhs"]
    if lhs is None or rhs is None:
        pair = _split_equality(ctx["expr"]) if ctx.get("expr") is not None else None
        if pair is None:
            _need(ctx, "expr", "cand")
            lhs = _parse_expr(ctx["expr"], what="原表达式（左端）")
            rhs = _parse_expr(ctx["cand"], what="候选结果（右端）")
        else:
            lhs = _parse_expr(pair[0], what="等式左端")
            rhs = _parse_expr(pair[1], what="等式右端")
    var = ctx["var"] or (C.resolve_var(lhs, None) if lhs.free_symbols else None)
    ok, detail = C.verify_identity(lhs, rhs)
    numeric = _numeric_compare(lhs, rhs, [var] if var is not None else [])
    decision = _decide_equality(lhs, rhs, [var] if var is not None else [], numeric=numeric)
    if decision["passed"] is True and not ok:
        decision["method"] += "；sympy_core.verify_identity 未通过，请人工核对"
    caveats = _removable_caveats(lhs, rhs, var)
    covered, unhandled_cav = _caveats_covered(caveats, ctx["conditions"])
    diff_refined, used = _use_conditions_on_difference(lhs - rhs, ctx["conditions"])
    if diff_refined == 0 and decision["passed"] is not True:
        decision = {"passed": True, "level": LEVEL_SYMBOLIC,
                    "method": "在用户声明的条件下差式化简为 0",
                    "evidence": decision["evidence"], "cannot": None}
    evidence = {
        "simplified_difference": decision["evidence"].get("difference"),
        "sympy_core_method": detail.get("method"),
        "numeric": decision["evidence"].get("numeric"),
        "domain_caveats": [c["text"] for c in caveats],
        "condition_refined_difference": _t(diff_refined),
    }
    if decision["evidence"].get("counterexample"):
        evidence["counterexample"] = decision["evidence"]["counterexample"]
    methods = [{"method": "差式符号化简（sympy_core.verify_identity）", "passed": bool(ok),
                "evidence": {"simplified_difference": detail.get("simplified_difference")}},
               {"method": "高精度数值抽样逐点比较", "passed": numeric.get("agree"),
                "evidence": {"max_rel_dev": numeric.get("max_rel_dev"), "note": numeric.get("note")}}]
    if caveats:
        methods.append({"method": "定义域陷阱检查（约去公因子是否改变定义域）",
                        "passed": False if unhandled_cav else True,
                        "evidence": {"caveats": [c["text"] for c in caveats]}})
    return _base_result(decision["passed"], decision["level"], decision["method"], evidence,
                        cannot=decision["cannot"], unhandled=_unhandled_from(ctx, used) + unhandled_cav,
                        used_conditions=used + covered, extra={"methods": methods})


def _h_simplify(ctx: dict[str, Any]) -> dict[str, Any]:
    _need(ctx, "expr", "cand")
    orig = _parse_expr(ctx["expr"], what="原表达式")
    cand = _parse_expr(ctx["cand"], what="候选化简结果")
    syms = sorted(orig.free_symbols | cand.free_symbols, key=lambda s: s.name)
    numeric = _numeric_compare(orig, cand, syms)
    decision = _decide_equality(orig, cand, syms, numeric=numeric)
    caveats = _removable_caveats(orig, cand, ctx["var"] or (syms[0] if syms else None))
    covered, unhandled_cav = _caveats_covered(caveats, ctx["conditions"])
    diff_refined, used = _use_conditions_on_difference(orig - cand, ctx["conditions"])
    if diff_refined == 0 and decision["passed"] is not True:
        decision = {"passed": True, "level": LEVEL_SYMBOLIC,
                    "method": "在用户声明的条件下差式化简为 0", "evidence": decision["evidence"],
                    "cannot": None}
    try:
        ops_before, ops_after = sp.count_ops(orig), sp.count_ops(cand)
    except Exception:  # noqa: BLE001
        ops_before = ops_after = None
    evidence = {"difference": decision["evidence"].get("difference"),
                "numeric": decision["evidence"].get("numeric"),
                "complexity": {"original_ops": ops_before, "candidate_ops": ops_after,
                               "note": ("候选确实更简单" if (ops_before is not None and ops_after is not None
                                                             and ops_after <= ops_before)
                                        else "候选并不比原式简单，请确认化简目标")},
                "domain_caveats": [c["text"] for c in caveats]}
    if decision["evidence"].get("counterexample"):
        evidence["counterexample"] = decision["evidence"]["counterexample"]
    methods = [{"method": "原式与候选的差式符号化简", "passed": decision["passed"],
                "evidence": {"difference": decision["evidence"].get("difference")}},
               {"method": "高精度数值抽样逐点比较", "passed": numeric.get("agree"),
                "evidence": {"max_rel_dev": numeric.get("max_rel_dev"), "note": numeric.get("note")}}]
    if caveats:
        methods.append({"method": "定义域陷阱检查", "passed": False if unhandled_cav else True,
                        "evidence": {"caveats": [c["text"] for c in caveats]}})
    return _base_result(decision["passed"], decision["level"], decision["method"], evidence,
                        cannot=decision["cannot"], unhandled=_unhandled_from(ctx, used) + unhandled_cav,
                        used_conditions=used + covered, extra={"methods": methods})


def _h_factor(ctx: dict[str, Any]) -> dict[str, Any]:
    _need(ctx, "expr", "cand")
    orig = _parse_expr(ctx["expr"], what="原表达式")
    cand = _parse_expr(ctx["cand"], what="候选因式分解结果")
    expected = sp.factor(orig)
    syms = sorted(orig.free_symbols | cand.free_symbols, key=lambda s: s.name)
    numeric = _numeric_compare(expected, cand, syms)
    decision = _decide_equality(expected, cand, syms, numeric=numeric)
    evidence = {"sympy_factor_result": _t(expected), "difference": decision["evidence"].get("difference"),
                "numeric": decision["evidence"].get("numeric")}
    if decision["evidence"].get("counterexample"):
        evidence["counterexample"] = decision["evidence"]["counterexample"]
    methods = [{"method": "独立做因式分解（sympy factor）并与候选比较", "passed": decision["passed"],
                "evidence": {"expected": _t(expected), "difference": decision["evidence"].get("difference")}},
               {"method": "高精度数值抽样逐点比较", "passed": numeric.get("agree"),
                "evidence": {"max_rel_dev": numeric.get("max_rel_dev"), "note": numeric.get("note")}}]
    return _base_result(decision["passed"], decision["level"], "因式分解复核：" + decision["method"],
                        evidence, cannot=decision["cannot"], unhandled=_unhandled_from(ctx, []),
                        extra={"methods": methods})


def _h_expand(ctx: dict[str, Any]) -> dict[str, Any]:
    _need(ctx, "expr", "cand")
    orig = _parse_expr(ctx["expr"], what="原表达式")
    cand = _parse_expr(ctx["cand"], what="候选展开结果")
    expected = sp.expand(orig)
    syms = sorted(orig.free_symbols | cand.free_symbols, key=lambda s: s.name)
    numeric = _numeric_compare(expected, cand, syms)
    decision = _decide_equality(expected, cand, syms, numeric=numeric)
    evidence = {"sympy_expand_result": _t(expected), "difference": decision["evidence"].get("difference"),
                "numeric": decision["evidence"].get("numeric")}
    if decision["evidence"].get("counterexample"):
        evidence["counterexample"] = decision["evidence"]["counterexample"]
    methods = [{"method": "独立展开（sympy expand）并与候选比较", "passed": decision["passed"],
                "evidence": {"expected": _t(expected), "difference": decision["evidence"].get("difference")}},
               {"method": "高精度数值抽样逐点比较", "passed": numeric.get("agree"),
                "evidence": {"max_rel_dev": numeric.get("max_rel_dev"), "note": numeric.get("note")}}]
    return _base_result(decision["passed"], decision["level"], "展开复核：" + decision["method"],
                        evidence, cannot=decision["cannot"], unhandled=_unhandled_from(ctx, []),
                        extra={"methods": methods})


def _h_matrix_inverse(ctx: dict[str, Any]) -> dict[str, Any]:
    matrix = ctx["matrix"]
    if matrix is None:
        raise ValueError("矩阵求逆复核需要 matrix 参数")
    _need(ctx, "cand")
    M = ctx["matrix"]
    orig_text = _t(M)
    try:
        cand = A.parse_matrix(text=str(ctx["cand"]))
    except Exception as exc:  # noqa: BLE001
        return _base_result(None, LEVEL_NONE, "矩阵求逆复核",
                            {"original_matrix": orig_text, "parse_note": f"候选无法解析为矩阵：{exc}"},
                            cannot=f"候选结果无法解析为矩阵：{exc}")
    evidence: dict[str, Any] = {"original_matrix": orig_text, "candidate_matrix": _t(cand),
                                "determinant": _t(M.det())}
    conds = [e for e in ctx["conditions"] if e.get("kind") == "matrix_invertible"]
    used = [e["text"] for e in conds]
    if M.det() == 0:
        evidence["failure"] = "det(A) = 0，矩阵 A 不可逆，因此任何候选都不可能是它的逆矩阵"
        res = _base_result(False, LEVEL_SYMBOLIC, "行列式判定：A 不可逆", evidence)
        res["used_conditions"] = used
        res["extra"]["methods"] = [{"method": "行列式判定（det = 0 ⟹ 不可逆）", "passed": False,
                                    "evidence": {"determinant": "0"}}]
        return res
    product = M * cand
    diff = product - sp.eye(M.rows)
    ok = bool(diff.is_zero_matrix)
    evidence["product_M_times_candidate"] = _t(product)
    evidence["difference_MC_minus_I"] = _t(diff)
    if not ok:
        bad = [f"第({i + 1},{j + 1})项 = {_t(diff[i, j])}（应为 0）"
               for i in range(diff.rows) for j in range(diff.cols) if sp.simplify(diff[i, j]) != 0]
        evidence["counterexample"] = "M·C − I 不为零矩阵：" + "；".join(bad[:6])
    methods = [{"method": "把候选代回做矩阵乘法，检查 M·C 是否等于单位矩阵", "passed": ok,
                "evidence": {"product": _t(product), "difference": _t(diff)}}]
    # 数值核对（独立机制）
    try:
        from sympy.matrices import Matrix as _M  # noqa: PLC0415
        num_ok = None
        if not M.free_symbols and not cand.free_symbols:
            Mn = _M([[float(sp.N(x, 15)) for x in row] for row in M.tolist()])
            Cn = _M([[float(sp.N(x, 15)) for x in row] for row in cand.tolist()])
            dev = max(abs((Mn * Cn)[i, j] - (1.0 if i == j else 0.0))
                      for i in range(M.rows) for j in range(M.cols))
            num_ok = bool(dev <= 1e-9)
            evidence["numeric"] = {"max_abs_deviation_from_I": dev, "note":
                                   None if num_ok else f"数值核对：M·C 与单位矩阵最大偏差 {dev:.3e}"}
            methods.append({"method": "数值矩阵乘法核对（与符号乘法独立）", "passed": num_ok,
                            "evidence": evidence["numeric"]})
    except Exception:  # noqa: BLE001
        pass
    unhandled = []
    if not conds:
        unhandled.append("未声明「A 可逆」：已由 det(A) ≠ 0 自行判定可逆，故本次复核不依赖该条件")
    else:
        unhandled.append("已使用条件「A 可逆」（det ≠ 0 校验通过）；未验证候选是唯一的逆矩阵")
    return _base_result(ok, LEVEL_SYMBOLIC, "矩阵求逆复核：M·C 与单位矩阵作符号比较", evidence,
                        unhandled=unhandled, used_conditions=used + ["A 可逆（det ≠ 0）" if conds else ""],
                        extra={"methods": methods})


def _h_determinant(ctx: dict[str, Any]) -> dict[str, Any]:
    if ctx["matrix"] is None:
        raise ValueError("行列式复核需要 matrix 参数")
    _need(ctx, "cand")
    M = ctx["matrix"]
    det = sp.simplify(M.det())
    cand = _parse_expr(ctx["cand"], what="候选行列式")
    syms = sorted((det.free_symbols | cand.free_symbols), key=lambda s: s.name)
    numeric = _numeric_compare(det, cand, syms)
    decision = _decide_equality(det, cand, syms, numeric=numeric)
    evidence = {"independent_determinant": _t(det), "difference": decision["evidence"].get("difference"),
                "numeric": decision["evidence"].get("numeric"), "matrix": _t(M)}
    if decision["evidence"].get("counterexample"):
        evidence["counterexample"] = decision["evidence"]["counterexample"]
    methods = [{"method": "独立计算行列式（sympy det）并与候选比较", "passed": decision["passed"],
                "evidence": {"determinant": _t(det), "difference": decision["evidence"].get("difference")}},
               {"method": "高精度数值抽样逐点比较", "passed": numeric.get("agree"),
                "evidence": {"max_rel_dev": numeric.get("max_rel_dev"), "note": numeric.get("note")}}]
    if not det.free_symbols:
        try:
            import numpy as _np  # noqa: PLC0415
            arr = _np.array([[complex(sp.N(x, 15)) for x in row] for row in M.tolist()], dtype=complex)
            nv = _np.linalg.det(arr)
            evidence["numpy_determinant"] = {"value": [nv.real, nv.imag],
                                             "note": "numpy.linalg.det 数值核对（独立于 sympy）"}
            methods.append({"method": "numpy.linalg.det 数值核对", "passed": None,
                            "evidence": evidence["numpy_determinant"]})
        except Exception as exc:  # noqa: BLE001
            evidence["numpy_determinant"] = {"note": f"numpy 核对不可用：{exc}"}
    return _base_result(decision["passed"], decision["level"], "行列式复核：" + decision["method"],
                        evidence, cannot=decision["cannot"], unhandled=_unhandled_from(ctx, []),
                        extra={"methods": methods})


def _h_eigen(ctx: dict[str, Any]) -> dict[str, Any]:
    if ctx["matrix"] is None:
        raise ValueError("特征值复核需要 matrix 参数")
    _need(ctx, "cand")
    M = ctx["matrix"]
    var = ctx["var"] or sp.Symbol("lambda")
    cp = M.charpoly(var).as_expr()
    ev = M.eigenvals()
    ev_text = {_t(k): int(v) for k, v in ev.items()}
    items: list[tuple[sp.Basic, int | None]] = []
    raw_cand = str(ctx.get("cand_text") or ctx["cand"]).strip().strip("()[]{}")
    for raw in raw_cand.split(","):
        piece = raw.strip()
        if not piece:
            continue
        if ":" in piece or "×" in piece:
            sep = ":" if ":" in piece else "×"
            val_txt, mult_txt = piece.split(sep, 1)
            items.append((_parse_expr(val_txt, what="特征值"), int(float(sp.N(_parse_expr(mult_txt, what="重数"))))))
        else:
            items.append((_parse_expr(piece, what="特征值"), None))
    evidence: dict[str, Any] = {"characteristic_polynomial": _t(cp), "independent_eigenvalues": ev_text,
                                "candidate": [{"value": _t(v), "multiplicity": m} for v, m in items]}
    detail_rows = []
    ok = True
    counter = []
    for val, mult in items:
        resid = sp.simplify(cp.subs(var, val))
        is_root = _symbolic_zero(resid)[0] is True
        actual_mult = None
        for k, v in ev.items():
            if _symbolic_zero(sp.sympify(k) - val)[0] is True:
                actual_mult = int(v)
                break
        row = {"value": _t(val), "charpoly_at_value": _t(resid), "is_eigenvalue": is_root,
               "claimed_multiplicity": mult, "actual_multiplicity": actual_mult}
        if not is_root:
            ok = False
            counter.append(f"{_t(val)} 不是特征值：特征多项式在该点取值 {_t(resid)} ≠ 0（独立求得的特征值为 {ev_text}）")
        elif mult is not None and actual_mult is not None and mult != actual_mult:
            ok = False
            counter.append(f"{_t(val)} 的代数量数应为 {actual_mult}，候选写作 {mult}")
        detail_rows.append(row)
    if sum(m for _v, m in items if m is not None) and all(m is not None for _v, m in items):
        if sum(int(m) for _v, m in items) != M.rows:
            ok = False
            counter.append(f"候选给出的重数之和为 {sum(int(m) for _v, m in items)}，应等于矩阵阶数 {M.rows}")
    evidence["rows"] = detail_rows
    if counter:
        evidence["counterexample"] = "；".join(counter)
    methods = [{"method": "把候选特征值代入特征多项式检查是否为 0（独立于求解器）", "passed": ok,
                "evidence": {"characteristic_polynomial": _t(cp), "rows": detail_rows}},
               {"method": "独立求特征值并核对代数量数", "passed": ok,
                "evidence": {"eigenvalues": ev_text}}]
    unhandled = ["本次只复核特征值（及其重数），未复核特征向量"]
    if all(m is None for _v, m in items):
        unhandled.append("候选未给出代数重数，未核对重数")
    return _base_result(ok, LEVEL_SYMBOLIC if ok else LEVEL_SYMBOLIC, "特征值复核：特征多项式代入 + 独立特征值对照",
                        evidence, unhandled=unhandled, extra={"methods": methods})


def _h_probability(ctx: dict[str, Any]) -> dict[str, Any]:
    _need(ctx, "cand")
    expr_text = ctx.get("expr_text") or ctx.get("expr")
    if expr_text is None:
        raise ValueError("概率复核需要 expr（如 \"P(X<=3)\"）")
    spec = _parse_prob_spec(str(expr_text))
    if spec is None:
        # expr 不是 P(...) 记号 → 退化为一般表达式复核
        cand = _parse_expr(ctx["cand"], what="候选结果")
        target = _parse_expr(expr_text, what="原表达式")
        syms = sorted(target.free_symbols | cand.free_symbols, key=lambda s: s.name)
        numeric = _numeric_compare(target, cand, syms)
        decision = _decide_equality(target, cand, syms, numeric=numeric)
        evidence = {"note": "expr 不是 P(...) 形式的概率记号，已退化为一般表达式一致性复核",
                    "difference": decision["evidence"].get("difference"),
                    "numeric": decision["evidence"].get("numeric")}
        if decision["evidence"].get("counterexample"):
            evidence["counterexample"] = decision["evidence"]["counterexample"]
        methods = [{"method": "符号差式 + 数值抽样", "passed": decision["passed"], "evidence": evidence}]
        return _base_result(decision["passed"], decision["level"], "概率记号无法识别，" + decision["method"],
                            evidence, cannot=decision["cannot"], unhandled=_unhandled_from(ctx, []),
                            extra={"methods": methods})
    dist = ctx.get("dist") or (ctx["params"].get("dist") if ctx["params"].get("dist") else None)
    if dist is None:
        raise ValueError("概率复核需要给出分布类型（参数 dist，如 dist=\"binomial\"）")
    spec["n"] = ctx.get("order") or ctx["params"].get("n")
    params = dict(ctx["params"])
    if ctx.get("lower") is not None and "lower" not in params:
        params["lower"] = ctx["lower"]
    if ctx.get("upper") is not None and "upper" not in params:
        params["upper"] = ctx["upper"]
    exact, how = _exact_prob(str(dist), params, spec)
    cand = _parse_expr(ctx["cand"], what="候选概率")
    cand_num = _ev(cand, {})
    evidence: dict[str, Any] = {"probability_expression": str(expr_text), "distribution": str(dist),
                                "parsed_spec": {k: (None if v is None else str(v)) for k, v in spec.items()},
                                "independent_value": exact, "independent_method": how,
                                "candidate_value": cand_num}
    methods: list[dict[str, Any]] = []
    passed: bool | None = None
    level = LEVEL_NONE
    cannot = None
    if exact is not None and cand_num is not None:
        dev = abs(exact - cand_num)
        rel = dev / max(1.0, abs(exact))
        passed = bool(rel <= 1e-6 or dev <= 1e-9)
        level = LEVEL_NUMERIC
        evidence["deviation"] = {"absolute": dev, "relative": rel}
        if not passed:
            evidence["counterexample"] = (f"独立重算的概率为 {exact:.12g}，候选为 {cand_num:.12g}，"
                                          f"绝对偏差 {dev:.6g}")
        methods.append({"method": f"用 scipy 精确分布函数独立重算概率（{how}）——数值证据",
                        "passed": passed, "evidence": {"independent": exact, "candidate": cand_num,
                                                       "absolute_deviation": dev, "relative_deviation": rel}})
        samp = _sampler(str(dist), params, ctx.get("symbols", [sp.Symbol("x")])[0] if ctx.get("symbols") else sp.Symbol("x"))
        if samp is not None:
            mc = _monte_carlo(samp, lambda x: 1.0 if _prob_indicator(spec, x) else 0.0, trials=100_000)
            if mc.get("mean") is not None:
                mc_ok = abs(mc["mean"] - (cand_num if cand_num is not None else 0)) <= max(4 * mc["standard_error"], 0.01)
                evidence["monte_carlo"] = mc
                methods.append({"method": "蒙特卡洛重复试验（与解析重算相互独立的第三种机制）",
                                "passed": mc_ok,
                                "evidence": {"estimate": mc["mean"], "standard_error": mc["standard_error"],
                                             "candidate": cand_num}})
    else:
        cannot = (f"无法独立重算该概率：{how}") if exact is None else "候选概率无法转成数值"
    if passed is None:
        return _base_result(None, LEVEL_NONE, "概率复核", evidence, cannot=cannot,
                            unhandled=_unhandled_from(ctx, []), extra={"methods": methods})
    unhandled = []
    if dist and str(dist).lower() in ("binomial", "poisson", "geometric"):
        unhandled.append("离散型概率：精确分布函数本身精确，但样本均值记号（X̄）只能用 CLT 近似")
    return _base_result(passed, level, "概率复核：scipy 精确分布函数独立重算 + 蒙特卡洛对照", evidence,
                        unhandled=_unhandled_from(ctx, []) + unhandled, extra={"methods": methods})


def _prob_indicator(spec: dict[str, Any], x: float) -> bool:
    kind = spec["kind"]
    try:
        b1 = None if spec.get("b1") is None else float(sp.N(_parse_expr(spec["b1"])))
        b2 = None if spec.get("b2") is None else float(sp.N(_parse_expr(spec["b2"])))
    except Exception:  # noqa: BLE001
        return False
    if kind == "le":
        return x <= b1
    if kind == "lt":
        return x < b1
    if kind == "ge":
        return x >= b1
    if kind == "gt":
        return x > b1
    if kind == "eq":
        return abs(x - b1) < 1e-9
    if kind == "between":
        return b1 <= x <= b2
    return False


def _h_moment(ctx: dict[str, Any], mode: str) -> dict[str, Any]:
    dist = ctx.get("dist") or ctx["params"].get("dist")
    if dist is None:
        raise ValueError(f"{'期望' if mode == 'expectation' else '方差'}复核需要分布类型（参数 dist）")
    _need(ctx, "cand")
    g_text = ctx.get("expr")
    var = ctx["var"] or sp.Symbol("x")
    x_sym = sp.Symbol("X")
    g = _parse_expr(g_text, what="随机变量表达式", extra={"X": x_sym, "x": x_sym, "X_bar": x_sym}) if g_text else x_sym
    g = sp.sympify(g).subs(x_sym, var)
    desc = _dist_descriptor(str(dist), ctx["params"], var)
    cand = _parse_expr(ctx["cand"], what="候选结果")
    evidence: dict[str, Any] = {"distribution": desc["name"], "function_of_X": _t(g),
                                "support": [_t(desc["support"][0]), _t(desc["support"][1])]}
    methods: list[dict[str, Any]] = []
    exact_val: sp.Basic | None = None
    if mode == "expectation":
        ex = _dist_expectation(desc, g, allow_symbolic=desc["name"].startswith(("B", "U", "Exp")))
        exact_val = ex.get("symbolic")
        methods += [{"method": m, "passed": None, "evidence": {}} for m in ex.get("method", [])]
        target = exact_val if exact_val is not None else desc["mean"]
        if exact_val is None and _symbolic_zero(g - x_sym.subs(x_sym, var))[0] is True:
            exact_val = desc["mean"]
            methods.append({"method": "直接使用该分布的已知期望公式（独立于积分/求和）", "passed": None,
                            "evidence": {"mean": _t(desc["mean"])}})
    else:
        # Var(g(X)) = E[g²] − (E[g])²
        ex2 = _dist_expectation(desc, sp.expand(g ** 2))
        ex1 = _dist_expectation(desc, g)
        if ex2.get("symbolic") is not None and ex1.get("symbolic") is not None:
            exact_val = sp.simplify(ex2["symbolic"] - ex1["symbolic"] ** 2)
            methods.append({"method": "Var(g(X)) = E[g²] − (E[g])²（两项各自独立求和/积分）", "passed": None,
                            "evidence": {"E[g^2]": _t(ex2["symbolic"]), "E[g]": _t(ex1["symbolic"])}})
        elif _symbolic_zero(g - var)[0] is True:
            exact_val = desc["variance"]
            methods.append({"method": "直接使用该分布的已知方差公式（独立于积分/求和）", "passed": None,
                            "evidence": {"variance": _t(desc["variance"])}})
    num_parts = _dist_numeric_expectation(desc, g if mode == "expectation" else sp.expand(g ** 2))
    if num_parts.get("numeric"):
        evidence["numeric_quad"] = {"value": num_parts["numeric"][0], "error": num_parts["numeric"][1],
                                    "note": num_parts["method"]}
        methods.append({"method": f"数值积分核对（{num_parts['method']}）", "passed": None,
                        "evidence": evidence["numeric_quad"]})
    cand_num = _ev(cand, {})
    evidence["candidate"] = _t(cand)
    evidence["independent_value"] = _t(exact_val) if exact_val is not None else None
    if exact_val is not None:
        decision = _decide_equality(exact_val, cand, [])
        passed, level = decision["passed"], decision["level"]
        evidence["difference"] = decision["evidence"].get("difference")
        if decision["evidence"].get("counterexample"):
            evidence["counterexample"] = decision["evidence"]["counterexample"]
        methods.insert(0, {"method": "独立推导（符号求和/积分或已知矩公式）并与候选作差",
                           "passed": passed, "evidence": {"independent": _t(exact_val),
                                                           "difference": decision["evidence"].get("difference")}})
    elif cand_num is not None and evidence.get("numeric_quad"):
        indep = evidence["numeric_quad"]["value"]
        dev = abs(indep - cand_num) / max(1.0, abs(indep))
        passed = bool(dev <= 1e-4)
        level = LEVEL_NUMERIC
        evidence["deviation"] = dev
        if not passed:
            evidence["counterexample"] = f"数值核对给出 {indep:.12g}，候选为 {cand_num:.12g}（相对偏差 {dev:.6g}）"
        methods.insert(0, {"method": "数值积分独立重算（未获得符号级证明）", "passed": passed,
                           "evidence": {"independent_numeric": indep, "candidate": cand_num,
                                        "relative_deviation": dev}})
    else:
        return _base_result(None, LEVEL_NONE, "矩复核",
                            evidence, cannot="既无法符号求出也无法数值重算该矩，无法复核",
                            unhandled=_unhandled_from(ctx, []), extra={"methods": methods})
    # 蒙特卡洛第三种机制
    try:
        samp = _sampler(str(dist), ctx["params"], var)
        if samp is not None:
            f = sp.lambdify(var, g, "math")
            if mode == "variance":
                mc1 = _monte_carlo(samp, lambda x: float(f(x)))
                mc2 = _monte_carlo(samp, lambda x: float(f(x)) ** 2)
                if mc1.get("mean") is not None and mc2.get("mean") is not None:
                    est = mc2["mean"] - mc1["mean"] ** 2
                    evidence["monte_carlo"] = {"estimate": est, "trials": mc1["trials"]}
                    methods.append({"method": "蒙特卡洛估计 Var（第三种独立机制，粗粒度）", "passed": None,
                                    "evidence": evidence["monte_carlo"]})
            else:
                mc = _monte_carlo(samp, lambda x: float(f(x)))
                if mc.get("mean") is not None:
                    evidence["monte_carlo"] = mc
                    methods.append({"method": "蒙特卡洛估计 E[g(X)]（第三种独立机制，粗粒度）", "passed": None,
                                    "evidence": mc})
    except Exception:  # noqa: BLE001
        pass
    label = "期望" if mode == "expectation" else "方差"
    return _base_result(passed, level, f"{label}复核：独立推导 + 数值积分 + 蒙特卡洛交叉验证", evidence,
                        unhandled=_unhandled_from(ctx, []), extra={"methods": methods})


def _h_expectation(ctx: dict[str, Any]) -> dict[str, Any]:
    return _h_moment(ctx, "expectation")


def _h_variance(ctx: dict[str, Any]) -> dict[str, Any]:
    return _h_moment(ctx, "variance")


def _h_normalization(ctx: dict[str, Any]) -> dict[str, Any]:
    _need(ctx, "expr", "cand")
    var = ctx["var"] or C.resolve_var(_parse_expr(ctx["expr"], what="原表达式"), None)
    body = _parse_expr(ctx["expr"], what="待归一化的密度/分布律")
    lo = C.parse_point(ctx["lower"], [var]) if ctx.get("lower") is not None else None
    hi = C.parse_point(ctx["upper"], [var]) if ctx.get("upper") is not None else None
    if lo is None or hi is None:
        lo, hi = -sp.oo, sp.oo
    cand = _parse_expr(ctx["cand"], what="候选归一化常数")
    evidence: dict[str, Any] = {"integrand": _t(body), "domain": [_t(lo), _t(hi)]}
    # 候选语义：若候选不含自变量 → 视为归一化常数 C，检查 C·∫f = 1；否则视为完整密度
    cand_is_constant = not bool(cand.free_symbols)
    methods: list[dict[str, Any]] = []
    total = None
    total_note = None
    try:
        integral = sp.Integral(body, (var, lo, hi))
        value = integral.doit()
        if isinstance(value, sp.Integral) or value.has(sp.Integral):
            total_note = "sympy 未能求出该积分的闭式，改用数值积分"
        else:
            total = sp.simplify(value)
            evidence["total_integral"] = _t(total)
            evidence["total_integral_approx"] = _ev(total, {})
            methods.append({"method": "独立计算在整个支撑上的积分（sympy）", "passed": None,
                            "evidence": {"total": _t(total), "approx": _ev(total, {})}})
    except Exception as exc:  # noqa: BLE001
        total_note = f"符号积分失败：{type(exc).__name__}: {exc}"
    if total is None:
        try:
            val_num = float(sp.N(sp.Integral(body, (var, lo, hi)), 20))
            evidence["total_integral_approx"] = val_num
            evidence["total_integral"] = None
            total_note = total_note or ""
            methods.append({"method": "数值积分独立重算总测度", "passed": None,
                            "evidence": {"approx": val_num, "note": total_note}})
        except Exception as exc:  # noqa: BLE001
            return _base_result(None, LEVEL_NONE, "归一化复核", evidence,
                                cannot=f"无法计算该函数在整个支撑上的积分：{exc}",
                                unhandled=_unhandled_from(ctx, []), extra={"methods": methods})
    target = sp.sympify(1)
    if cand_is_constant:
        expr_check = sp.simplify(cand * (total if total is not None else sp.Float(evidence["total_integral_approx"], 20)))
        evidence["interpretation"] = "候选被解释为归一化常数 C，检查 C·∫f = 1"
        independent = (sp.simplify(1 / total) if (total is not None and total != 0) else None)
        if independent is not None:
            evidence["independent_constant"] = _t(independent)
    else:
        integral2 = sp.Integral(cand, (var, lo, hi))
        try:
            value2 = sp.simplify(integral2.doit())
            expr_check = sp.simplify(value2) if not isinstance(value2, sp.Integral) else None
            evidence["candidate_total_integral"] = _t(value2)
            evidence["interpretation"] = "候选被解释为完整密度，检查 ∫候选 = 1"
        except Exception:  # noqa: BLE001
            expr_check = None
    decision = _decide_equality(expr_check, target, []) if expr_check is not None else {
        "passed": None, "level": LEVEL_NONE, "method": "无法完成归一化检查",
        "evidence": {"difference": _t(expr_check) if expr_check is not None else None}, "cannot": None}
    evidence["check_expression"] = _t(expr_check) if expr_check is not None else None
    if decision.get("evidence", {}).get("counterexample"):
        evidence["counterexample"] = decision["evidence"]["counterexample"]
    elif decision["passed"] is False:
        evidence["counterexample"] = (f"归一化检查量为 {_t(expr_check)}（应为 1，差 "
                                      f"{decision['evidence'].get('difference')}）：该函数在给定支撑上的总测度为 "
                                      f"{_t(total) if total is not None else evidence.get('total_integral_approx')}")
    methods.append({"method": "归一化条件 C·∫f = 1（或 ∫候选 = 1）的符号/数值检查",
                    "passed": decision["passed"],
                    "evidence": {"check_quantity": evidence["check_expression"],
                                 "difference_from_1": decision["evidence"].get("difference")}})
    evidence["candidate"] = _t(cand)
    unhandled = []
    if lo == -sp.oo or hi == sp.oo:
        unhandled.append("积分区间无界：广义积分的收敛性未单独证明（只验证了积分结果等于 1）")
    if total is not None and total.free_symbols:
        unhandled.append(f"总测度含自由参数（{_t(total)}）：归一化只在特定参数取值下成立，"
                         "本次未验证参数范围")
    if not cand_is_constant:
        unhandled.append("候选被当作完整密度解释；若原意是归一化常数，请直接给出常数")
    return _base_result(decision["passed"], decision["level"], "归一化复核：" + decision["method"], evidence,
                        cannot=decision["cannot"], unhandled=_unhandled_from(ctx, []) + unhandled,
                        extra={"methods": methods})


def _h_general(ctx: dict[str, Any]) -> dict[str, Any]:
    kind = ctx.get("kind")
    note = none_note = None
    if kind and kind != "general":
        note = f"未知的计算类型 {kind!r}：已退化为通用复核（符号差式 + 数值抽样 + 蒙特卡洛），请确认 kind 是否正确"
    if ctx.get("expr") is None and ctx.get("cand") is None:
        raise ValueError("通用复核至少需要 candidate（待验证结果），最好同时给出 expr（原始表达式）")
    if ctx.get("expr") is None:
        cand = _parse_expr(ctx["cand"], what="候选结果")
        evidence = {"candidate": _t(cand), "note": "只给出了待验证结果，没有原始表达式/原方程，"
                                                 "缺少可独立复核的对象"}
        return _base_result(None, LEVEL_NONE, "通用复核", evidence,
                            cannot="只提供了待验证结果（candidate），没有给出原始表达式（expr）、方程或矩阵，"
                                   "无法构造独立的复核依据",
                            unhandled=["未提供 expr：任何「通过」结论都将缺乏依据，故不予通过"],
                            extra={"methods": [{"method": "缺少原始表达式，未能执行任何独立机制",
                                                "passed": None, "evidence": evidence}]})
    expr = _parse_expr(ctx["expr"], what="原表达式")
    cand = _parse_expr(ctx["cand"], what="候选结果")
    syms = sorted(expr.free_symbols | cand.free_symbols, key=lambda s: s.name)
    numeric = _numeric_compare(expr, cand, syms)
    decision = _decide_equality(expr, cand, syms, numeric=numeric)
    caveats = _removable_caveats(expr, cand, syms[0] if syms else None)
    covered, unhandled_cav = _caveats_covered(caveats, ctx["conditions"])
    diff_refined, used = _use_conditions_on_difference(expr - cand, ctx["conditions"])
    if diff_refined == 0 and decision["passed"] is not True:
        decision = {"passed": True, "level": LEVEL_SYMBOLIC,
                    "method": "在用户声明的条件下差式化简为 0", "evidence": decision["evidence"],
                    "cannot": None}
    evidence = {"difference": decision["evidence"].get("difference"),
                "numeric": decision["evidence"].get("numeric"),
                "domain_caveats": [c["text"] for c in caveats],
                "mechanisms_tried": ["符号差式化简（simplify/expand/cancel/factor/equals）",
                                     "高精度数值抽样逐点比较"]}
    if decision["evidence"].get("counterexample"):
        evidence["counterexample"] = decision["evidence"]["counterexample"]
    methods = [{"method": "符号差式化简", "passed": decision["passed"] if decision["level"] == LEVEL_SYMBOLIC else None,
                "evidence": {"difference": decision["evidence"].get("difference")}},
               {"method": "高精度数值抽样逐点比较（数值证据，不是符号证明）",
                "passed": numeric.get("agree"),
                "evidence": {"max_rel_dev": numeric.get("max_rel_dev"), "note": numeric.get("note")}}]
    unhandled = _unhandled_from(ctx, used) + unhandled_cav
    if note:
        unhandled.append(note)
    if decision["level"] == LEVEL_NONE:
        unhandled.append("未获得符号级证明，也未获得有效数值证据")
    return _base_result(decision["passed"], decision["level"],
                        "通用复核：" + decision["method"], evidence, cannot=decision["cannot"],
                        unhandled=unhandled, used_conditions=used + covered,
                        extra={"methods": methods})


def _unhandled_from(ctx: dict[str, Any], used: Sequence[str]) -> list[str]:
    """未被本次验证使用的用户条件 → unhandled_conditions。"""
    out: list[str] = []
    for e in ctx.get("conditions", []):
        text = e["text"]
        if text in used:
            continue
        if e.get("kind") == "unparsed":
            out.append(f"{text}（未处理：该条件无法解析成可计算的约束，未参与本次验证）")
        elif e.get("kind") == "distribution":
            out.append(f"{text}（未处理：属于总体/抽样前提，无法用当前表达式直接检验，未参与本次验证）")
        else:
            out.append(f"{text}（未处理：本次验证未使用该条件，结论在无此条件时同样成立）")
    return out


_HANDLERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "derivative": _h_derivative,
    "integral": _h_integral,
    "antiderivative": _h_integral,
    "indefinite_integral": _h_integral,
    "definite_integral": _h_definite_integral,
    "limit": _h_limit,
    "equation_solution": _h_equation_solution,
    "solution": _h_equation_solution,
    "identity": _h_identity,
    "simplify": _h_simplify,
    "factor": _h_factor,
    "expand": _h_expand,
    "matrix_inverse": _h_matrix_inverse,
    "inverse_matrix": _h_matrix_inverse,
    "determinant": _h_determinant,
    "eigen": _h_eigen,
    "eigenvalue": _h_eigen,
    "probability": _h_probability,
    "expectation": _h_expectation,
    "variance": _h_variance,
    "normalization": _h_normalization,
    "general": _h_general,
}

#: kind 归一化（把用户可能写的说法映射到内部 kind）
_KIND_ALIASES: dict[str, str] = {
    "derive": "derivative", "differentiate": "derivative", "导数": "derivative", "求导": "derivative",
    "integrate": "integral", "积分": "integral", "不定积分": "integral", "原函数": "integral",
    "definite": "definite_integral", "定积分": "definite_integral", "integral_definite": "definite_integral",
    "极限": "limit", "solve": "equation_solution", "roots": "equation_solution", "解方程": "equation_solution",
    "方程": "equation_solution", "方程的根": "equation_solution",
    "恒等式": "identity", "equal": "identity", "equality": "identity", "化简": "simplify",
    "因式分解": "factor", "分解": "factor", "展开": "expand", "逆矩阵": "matrix_inverse",
    "行列式": "determinant", "特征值": "eigen", "特征向量": "eigen", "概率": "probability",
    "期望": "expectation", "均值": "expectation", "方差": "variance", "归一化": "normalization",
    "归一": "normalization",
}


def _verify_status(level: str, methods: Sequence[dict[str, Any]]) -> str:
    """把「证据等级」映射成引擎统一的验证状态词。"""
    if level == LEVEL_SYMBOLIC:
        has_numeric = any(
            isinstance(m.get("method"), str) and any(
                k in m["method"] for k in ("数值", "抽样", "差分", "蒙特卡洛", "scipy"))
            for m in methods)
        return "cross" if has_numeric else "independent"
    if level == LEVEL_NUMERIC:
        return "numeric"
    return "unverified"


def _normalize_kind(kind: Any, claim: Any) -> str:
    text = None if kind is None else str(kind).strip()
    if not text and claim:
        low = str(claim)
        for key, target in (("导数", "derivative"), ("积分", "integral"), ("行列式", "determinant"),
                            ("逆矩阵", "matrix_inverse"), ("特征值", "eigen"), ("概率", "probability"),
                            ("期望", "expectation"), ("方差", "variance"), ("归一", "normalization"),
                            ("方程", "equation_solution"), ("极限", "limit"), ("化简", "simplify")):
            if key in low:
                text = target
                break
    if not text:
        return "general"
    low = text.lower()
    if low in _HANDLERS:
        return low
    if low in _KIND_ALIASES:
        return _KIND_ALIASES[low]
    if text in _KIND_ALIASES:
        return _KIND_ALIASES[text]
    return text


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

@op("verify", "check", "math_verify")
def verify(
    expr: Any = None,
    candidate: Any = None,
    kind: Any = None,
    var: Any = None,
    vars: Any = None,
    point: Any = None,
    lower: Any = None,
    upper: Any = None,
    equations: Any = None,
    matrix: Any = None,
    matrix_b: Any = None,
    conditions: Any = None,
    dist: Any = None,
    m: Any = None,
    p: Any = None,
    lam: Any = None,
    mu: Any = None,
    sigma: Any = None,
    a: Any = None,
    b: Any = None,
    n: Any = None,
    alpha: Any = None,
    order: Any = None,
    trials: Any = None,
    dir: Any = None,
    lhs: Any = None,
    rhs: Any = None,
    symbols: Any = None,
    claim: Any = None,
    target: Any = None,
    successes: Any = None,
    **kwargs: Any,
) -> MathResult:
    """独立复核一个计算结果。

    参数
    ----
    expr        原始表达式 / 被积函数 / 原方程（随 kind 而定）
    candidate   待验证的结果（别名 result_expr / answer）
    kind        计算类型（derivative/integral/definite_integral/limit/equation_solution/identity/
                simplify/factor/expand/matrix_inverse/determinant/eigen/probability/expectation/
                variance/normalization/general）
    var, vars   自变量 / 未知量
    point       极限点（或需要在定义域上检查的点）
    lower, upper 积分上下限 / 分布参数 a、b
    equations   方程（组），可写成 "x^2-3*x+2=0"
    matrix      矩阵（求逆 / 行列式 / 特征值）
    conditions  用户声明的附加条件，会在验证中真正尝试使用，用不上的进入 unhandled_conditions
    """
    kind_norm = _normalize_kind(kind, claim)
    if candidate is None and claim is not None:
        candidate = claim
    if expr is None and lhs is not None:
        expr = lhs
        if candidate is None and rhs is not None:
            candidate = rhs
    if candidate is None:
        raise ValueError("缺少必需参数 candidate（待验证的结果，别名 result_expr / answer）")

    # 条件
    cond_entries = _condition_entries(conditions)
    parsed_expr = _parse_expr_soft(expr, what="原始表达式")
    parsed_cand = _parse_expr_soft(candidate, what="候选结果")
    default_syms: list[sp.Symbol] = []
    for obj in (parsed_expr, parsed_cand):
        if obj is not None and getattr(obj, "free_symbols", None):
            default_syms.extend(sorted(obj.free_symbols, key=lambda s: s.name))
    if var is not None:
        try:
            resolved = A.get_symbols([str(var)])
            default_syms = [resolved[str(var)]] + [s for s in default_syms if s.name != str(var)]
        except Exception:  # noqa: BLE001
            pass
    var_sym: sp.Symbol | None = None
    if var is not None:
        try:
            var_sym = A.get_symbols([str(var)])[str(var)]
        except Exception:  # noqa: BLE001
            var_sym = sp.Symbol(str(var))
    elif default_syms:
        var_sym = default_syms[0]
    eq_exprs = _equation_exprs(equations if equations is not None else (expr if kind_norm in
                                                                       ("equation_solution",) else None))
    vars_syms = _resolve_symbols(vars, default_syms) if vars is not None else []
    if kind_norm in ("equation_solution", "solution"):
        vars_syms = _resolve_symbols(vars, default_syms) if vars is not None else sorted(
            set().union(*[e.free_symbols for e in eq_exprs]) if eq_exprs else set(), key=lambda s: s.name)
    matrix_obj = _resolve_matrix(matrix, what="矩阵 A") if matrix is not None else None
    matrix_b_obj = _resolve_matrix(matrix_b, what="矩阵 B") if matrix_b is not None else None
    if matrix_obj is None and kind_norm in ("matrix_inverse", "inverse_matrix", "determinant", "eigen",
                                            "eigenvalue"):
        raise ValueError("该计算类型需要 matrix 参数（矩阵 A）")
    if kind_norm in ("derivative", "integral", "antiderivative", "indefinite_integral",
                     "definite_integral") and expr is None:
        raise ValueError("该计算类型需要 expr（原始表达式 / 被积函数）")

    params: dict[str, Any] = {}
    for key, value in (("dist", dist), ("m", m), ("p", p), ("lam", lam if lam is not None else kwargs.get("lambda")),
                       ("mu", mu), ("sigma", sigma), ("lower", lower), ("upper", upper),
                       ("n", n if n is not None else order), ("alpha", alpha),
                       ("trials", trials), ("successes", successes), ("a", a), ("b", b)):
        if value is not None:
            params[key] = value
    for key in ("xmin", "xmax", "lambda", "beta", "var", "mean", "std", "sigma0", "mu0"):
        if kwargs.get(key) is not None and key not in params:
            params[key] = kwargs[key]

    ctx: dict[str, Any] = {
        "kind": kind_norm, "expr": parsed_expr, "expr_text": None if expr is None else str(expr),
        # candidate 解析成 SymPy 失败时（例如 "x=1,2" 这种「解集」写法、矩阵文本 "1,0;0,1"），
        # 不能把 ctx["cand"] 置空——那样下游会报「复核缺少必需参数：cand」，把「写法特殊」
        # 误报成「参数没给」。这里退回原始文本，由需要结构的处理器自己解析。
        "cand": parsed_cand if parsed_cand is not None else (None if candidate is None else str(candidate)),
        "cand_text": None if candidate is None else str(candidate),
        "var": var_sym, "symbols": vars_syms or default_syms, "point": point,
        "lower": lower, "upper": upper, "equations": eq_exprs, "matrix": matrix_obj,
        "matrix_b": matrix_b_obj, "conditions": cond_entries, "params": params,
        "dist": dist, "dir": dir, "order": order if order is not None else n, "lhs": None, "rhs": None,
        "kwargs": kwargs,
    }
    if lhs is not None or rhs is not None:
        ctx["lhs"] = _parse_expr_soft(lhs, what="等式左端")
        ctx["rhs"] = _parse_expr_soft(rhs, what="等式右端")
    elif kind_norm in ("identity", "simplify") and expr is not None:
        pair = _split_equality(expr)
        if pair is not None:
            ctx["lhs"] = _parse_expr_soft(pair[0], what="等式左端")
            ctx["rhs"] = _parse_expr_soft(pair[1], what="等式右端")

    handler = _HANDLERS.get(kind_norm, _h_general)
    try:
        outcome = handler(ctx)
    except ValueError:
        raise
    except _CannotVerify as exc:
        outcome = _base_result(None, LEVEL_NONE, f"{kind_norm} 复核", {"reason": str(exc)},
                               cannot=str(exc))
    except Exception as exc:  # noqa: BLE001
        outcome = _base_result(None, LEVEL_NONE, f"{kind_norm} 复核",
                               {"error": f"{type(exc).__name__}: {exc}"},
                               cannot=f"复核过程出错（{type(exc).__name__}: {exc}），未完成验证")

    passed = outcome["passed"]
    level = outcome["level"]
    method = outcome["method"]
    evidence = outcome["evidence"]
    cannot = outcome["cannot"]
    unhandled = outcome["unhandled"]
    used_conditions = outcome["used_conditions"]

    r = MathResult.ok("verify", method=f"独立复核（kind={kind_norm}）")
    r.set_input(kind=kind_norm, expr=None if expr is None else str(expr),
                candidate=str(candidate), var=None if var is None else str(var),
                conditions=None if conditions is None else "; ".join(
                    e["text"] for e in cond_entries))
    if point is not None:
        r.set_input(point=str(point))
    if lower is not None or upper is not None:
        r.set_input(lower=None if lower is None else str(lower), upper=None if upper is None else str(upper))
    if matrix_obj is not None:
        r.set_input(matrix=A.to_text(matrix_obj))

    if level == LEVEL_NONE or passed is None:
        r = MathResult.unsolved("verify", cannot or "证据不足，无法完成独立复核",
                                method=f"独立复核（kind={kind_norm}）")
        r.set_input(kind=kind_norm, candidate=str(candidate),
                    expr=None if expr is None else str(expr))
    verdict = ("通过" if passed is True else "不通过" if passed is False else "无法判定")
    r.set_result(f"复核结论：{verdict}（验证等级：{level}）；方式：{method}")
    methods = outcome["extra"].get("methods") or [{"method": method, "passed": passed, "evidence": evidence}]
    r.verify(status=_verify_status(level, methods), methods=methods)
    r.verification.update({
        "passed": passed,
        "level": level,
        "method": method,
        "evidence": evidence,
        "unhandled_conditions": unhandled,
        "cannot_verify_reason": cannot,
        "conditions_used": used_conditions,
    })
    r.set_raw(
        passed=passed,
        verification={"passed": passed, "method": method, "level": level, "evidence": evidence,
                      "unhandled_conditions": unhandled, "cannot_verify_reason": cannot,
                      "conditions_used": used_conditions, "methods": methods,
                      "grade_note": ("symbolic = 符号级证明；numeric = 仅为数值证据，"
                                     "不是符号证明；none = 无法验证")},
        unhandled_conditions=unhandled,
        cannot_verify_reason=cannot,
        conditions_used=used_conditions,
        **({"counterexample": evidence["counterexample"]} if evidence.get("counterexample") else {}),
    )
    for key, value in (outcome["extra"].items() if isinstance(outcome.get("extra"), dict) else []):
        if key == "methods":
            continue
        r.set_raw(**{key: value})
    if level == LEVEL_NUMERIC:
        r.add_condition("本次复核只获得数值证据（抽样/数值积分/精确分布函数求值），"
                        "不是符号级证明；数值证据可能被奇点、分支或特殊点掩盖")
    if level == LEVEL_SYMBOLIC:
        r.add_condition("本次复核获得符号级证据（差式化简 / 符号求导 / 特征多项式代入等）")
    if unhandled:
        r.add_condition(*[f"未处理条件：{u}" for u in unhandled])
    if passed is False:
        r.add_warning("复核未通过：" + (evidence.get("counterexample") or
                                     f"独立计算得到 {evidence.get('difference') or evidence.get('independent_value')}"))
    elif passed is None:
        r.add_warning(cannot or "证据不足")
    if used_conditions:
        r.set_raw(conditions_actually_used=used_conditions)
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        r.add_warning(f"以下参数未被识别，已忽略：{unknown}")
    return r


# ---------------------------------------------------------------------------
# 条件相容性检查
# ---------------------------------------------------------------------------

@op("conditional_check", "verify_conditions", "math_check_conditions")
def conditional_check(
    expr: Any = None,
    var: Any = None,
    conditions: Any = None,
    vars: Any = None,
    matrix: Any = None,
    point: Any = None,
    **kwargs: Any,
) -> MathResult:
    """逐条检验用户声明的条件是否与表达式相容。"""
    entries = _condition_entries(conditions)
    if not entries:
        raise ValueError("conditional_check 需要 conditions（如 \"x > 0\"）")
    expr_obj = _parse_expr_soft(expr, what="表达式")
    var_sym: sp.Symbol | None = None
    if var is not None:
        var_sym = A.get_symbols([str(var)])[str(var)]
    elif expr_obj is not None and expr_obj.free_symbols:
        var_sym = sorted(expr_obj.free_symbols, key=lambda s: s.name)[0]
    matrix_obj = _resolve_matrix(matrix) if matrix is not None else None

    results: list[dict[str, Any]] = []
    any_incompatible = False
    any_unknown = False
    for e in entries:
        text = e["text"]
        rel = e.get("rel")
        row: dict[str, Any] = {"condition": text, "status": "unknown", "evidence": [], "level": LEVEL_SYMBOLIC}
        if e["kind"] == "matrix_invertible":
            if matrix_obj is None:
                row["status"] = "unknown"
                row["evidence"].append("条件涉及矩阵可逆，但未提供 matrix 参数，无法检验")
                any_unknown = True
            else:
                det = sp.simplify(matrix_obj.det())
                ok = det != 0
                row["status"] = "compatible" if ok else "incompatible"
                row["evidence"].append(f"det(A) = {_t(det)}"
                                       + ("≠ 0，A 可逆，条件与给定矩阵相容" if ok
                                          else "= 0，A 不可逆，条件与给定矩阵矛盾"))
                if not ok:
                    any_incompatible = True
            results.append(row)
            continue
        if e["kind"] == "tautology":
            row["status"] = "compatible"
            row["evidence"].append("sympy 直接判定该条件在实数域上恒成立（例如 x² + 1 > 0）")
            results.append(row)
            continue
        if e["kind"] == "contradiction":
            row["status"] = "incompatible"
            any_incompatible = True
            row["evidence"].append("sympy 判定该条件恒不成立：条件本身自相矛盾")
            results.append(row)
            continue
        if isinstance(rel, sp.Rel):
            try:
                lhs, rhs = sp.sympify(rel.lhs), sp.sympify(rel.rhs)
            except Exception:  # noqa: BLE001
                lhs = rhs = None
            if isinstance(rel, (sp.Equality, sp.Unequality)) and var_sym is not None and expr_obj is not None:
                # 形如 x = a（要求该点在定义域内）或 x ≠ a（把该点排除在外）的条件
                try:
                    if lhs == var_sym and rhs.is_number:
                        status = _point_domain_status(expr_obj, var_sym, rhs)
                        if isinstance(rel, sp.Unequality):
                            # 排除某点：该点无定义是「相容」的（甚至是必要的），有定义也不矛盾
                            if status["legal"] is False:
                                row["status"] = "compatible"
                                row["evidence"].append(
                                    f"表达式在 {_t(var_sym)} = {_t(rhs)} 处无定义（"
                                    + "；".join(status["reasons"][:2])
                                    + "），排除该点的条件与表达式的自然定义域一致")
                            elif status["legal"] is True:
                                row["status"] = "compatible"
                                row["evidence"].append(
                                    f"表达式在 {_t(var_sym)} = {_t(rhs)} 处有定义（取值 {_t(status['value'])}）；"
                                    f"该条件只是把这一点排除在外，与表达式不矛盾")
                            else:
                                row["status"] = "unknown"
                                any_unknown = True
                                row["evidence"].append("无法判定该点是否在定义域内")
                        elif status["legal"] is False:
                            row["status"] = "incompatible"
                            any_incompatible = True
                            row["evidence"] += status["reasons"]
                        elif status["legal"] is True:
                            row["status"] = "compatible"
                            row["evidence"].append(
                                f"{_t(var_sym)} = {_t(rhs)} 落在 {_t(expr_obj)} 的定义域内，"
                                f"该点取值为 {_t(status['value'])}")
                        else:
                            row["status"] = "unknown"
                            any_unknown = True
                            row["evidence"].append("无法判定该点是否在定义域内")
                except Exception as exc:  # noqa: BLE001
                    row["status"] = "unknown"
                    any_unknown = True
                    row["evidence"].append(f"检验失败：{type(exc).__name__}: {exc}")
            if row["status"] == "unknown" and isinstance(rel, (sp.StrictGreaterThan, sp.GreaterThan,
                                                               sp.StrictLessThan, sp.LessThan)):
                # 在满足该条件的抽样点上检查表达式是否为实数
                checked = 0
                bad: list[str] = []
                for p in (sp.Rational(1, 4), sp.Rational(1, 2), sp.Rational(1), sp.Rational(2),
                          sp.Rational(3), sp.Rational(-1, 4), sp.Rational(-1), sp.Rational(-3)):
                    try:
                        if _rel_true(rel.subs(lhs, p)) is not True:
                            continue
                    except Exception:  # noqa: BLE001
                        continue
                    checked += 1
                    if expr_obj is None:
                        continue
                    val = _ev_complex(expr_obj, {var_sym: p} if var_sym is not None else {})
                    if val is None:
                        bad.append(f"{_t(var_sym)} = {_t(p)}（满足该条件）处表达式无法求值或无定义")
                    elif getattr(val, "is_real", None) is False:
                        bad.append(f"{_t(var_sym)} = {_t(p)}（满足该条件）处表达式取值为 {val}，不是实数")
                if checked == 0:
                    row["status"] = "unknown"
                    any_unknown = True
                    row["evidence"].append("在抽样点中找不到满足该条件的实数点，无法确认（条件可能自相矛盾）")
                elif bad:
                    row["status"] = "incompatible"
                    any_incompatible = True
                    row["evidence"] += bad[:4]
                else:
                    row["status"] = "compatible"
                    row["evidence"].append(f"在 {checked} 个满足该条件的抽样点上表达式均有实数定义")
            if row["status"] == "unknown" and not row["evidence"]:
                # 一般关系式：抽样检验
                try:
                    holds = 0
                    total = 0
                    for p in (sp.Rational(1, 4), sp.Rational(1, 2), sp.Rational(1), sp.Rational(2),
                              sp.Rational(3), sp.Rational(-1)):
                        try:
                            total += 1
                            if _rel_true(rel.subs(lhs, p)) is True:
                                holds += 1
                        except Exception:  # noqa: BLE001
                            continue
                    if total == 0:
                        row["status"] = "unknown"
                        any_unknown = True
                        row["evidence"].append("无法在抽样点上代入检验")
                    elif holds == 0:
                        row["status"] = "incompatible"
                        any_incompatible = True
                        row["evidence"].append("在所有抽样取值上该关系都不成立")
                    else:
                        row["status"] = "compatible"
                        row["evidence"].append(f"{holds}/{total} 个抽样取值满足该关系")
                except Exception as exc:  # noqa: BLE001
                    row["status"] = "unknown"
                    any_unknown = True
                    row["evidence"].append(f"检验失败：{type(exc).__name__}: {exc}")
        elif e["kind"] == "expression" and e.get("rel") is sp.true:
            row["status"] = "compatible"
            row["evidence"].append("sympy 判定该关系式恒成立（化简后为真）")
        else:
            row["status"] = "unknown"
            any_unknown = True
            row["level"] = LEVEL_NONE
            row["evidence"].append("该条件无法解析成可计算的等式/不等式（或属于总体前提），本次未检验")
        results.append(row)

    overall = "incompatible" if any_incompatible else ("unknown" if any_unknown else "compatible")
    level = LEVEL_SYMBOLIC if overall != "unknown" else LEVEL_NONE
    passed = None if overall == "unknown" else (overall == "compatible")
    r = MathResult.ok("conditional_check", method="逐条条件相容性检查（代入/抽样/行列式判定）")
    r.set_input(expr=None if expr is None else str(expr), var=None if var is None else str(var),
                conditions="; ".join(e["text"] for e in entries))
    r.set_result({"compatible": "条件与表达式相容（未发现矛盾）",
                  "incompatible": "存在与表达式矛盾的条件",
                  "unknown": "部分条件无法检验"}[overall])
    r.set_raw(overall=overall, passed=passed, per_condition=results,
              unparsed_conditions=[x["condition"] for x in results if x["level"] == LEVEL_NONE],
              verification={"passed": passed,
                            "method": "逐条条件相容性检查（代入/抽样/行列式判定）",
                            "level": level,
                            "evidence": {"per_condition": results},
                            "unhandled_conditions": [x["condition"] for x in results if x["status"] == "unknown"],
                            "cannot_verify_reason": None if passed is not None else "存在无法检验的条件"})
    r.verify(status="independent" if level == LEVEL_SYMBOLIC else "unverified",
             methods=[{"method": "逐条代入/抽样检验条件与表达式的相容性", "passed": passed,
                       "evidence": {"per_condition": results}}])
    r.verification.update({"passed": passed, "level": level, "unhandled_conditions": [
        x["condition"] for x in results if x["status"] == "unknown"],
        "cannot_verify_reason": None if passed is not None else "存在无法检验的条件"})
    if overall == "incompatible":
        r.add_warning("存在不相容的条件，请检查表达式与条件的搭配")
    r.add_condition("条件的相容性检查不等于数学证明：抽样点上成立只说明未发现矛盾")
    for row in results:
        if row["status"] == "incompatible":
            r.add_condition(f"条件「{row['condition']}」与表达式矛盾：" + "；".join(row["evidence"][:2]))
    if kwargs:
        r.add_warning("以下参数未被识别，已忽略：" + ", ".join(sorted(kwargs)))
    return r


# ---------------------------------------------------------------------------
# 定义域检查
# ---------------------------------------------------------------------------

@op("domain_check", "verify_domain", "math_domain_check")
def domain_check(expr: Any = None, var: Any = None, point: Any = None, **kwargs: Any) -> MathResult:
    """检查指定点是否在表达式的定义域内，并给出自然定义域的限制来源。"""
    if expr is None:
        raise ValueError("domain_check 需要 expr（表达式）")
    expr_obj = _parse_expr(expr, what="表达式")
    if var is not None:
        var_sym: sp.Symbol | None = A.get_symbols([str(var)])[str(var)]
    else:
        syms = sorted(expr_obj.free_symbols, key=lambda s: s.name)
        var_sym = syms[0] if syms else None
    info = _domain_restrictions(expr_obj, var_sym)
    restrictions_text = [res["text"] for res in info["restrictions"]]
    r = MathResult.ok("domain_check", method="自然定义域分析（分母/偶次根式/对数/反三角/tan 极点 + 奇点分析）")
    r.set_input(expr=str(expr), var=None if var is None else str(var),
                point=None if point is None else str(point))

    if point is None:
        r.set_result("自然定义域限制：" + ("；".join(restrictions_text) if restrictions_text
                                       else "该表达式在其自由变量的整个实轴上都有定义"))
        r.set_raw(restrictions=restrictions_text,
                  excluded_points=[f"{_t(s)} = {_t(p)}" for s, p in info["excluded"]],
                  point_status=None,
                  note="未指定 point：只给出自然定义域，未检查具体点")
        r.verify(status="independent" if info["complete"] else "numeric",
                 methods=[{"method": "对表达式做自然定义域分析（分母/根式/对数/反三角/极点 + 奇点）",
                           "passed": None, "evidence": {"restrictions": restrictions_text,
                                                        "complete": info["complete"]}}])
        r.verification.update({"passed": None, "level": LEVEL_SYMBOLIC if info["complete"] else LEVEL_NUMERIC,
                               "method": "自然定义域分析（分母/根式/对数/反三角/极点 + 奇点）",
                               "unhandled_conditions": [] if info["complete"] else
                               ["奇点集合可能不完整（含有周期极点或 sympy 无法完全解析的函数）"],
                               "cannot_verify_reason": None})
        r.add_condition(*restrictions_text or ["在整个实轴上定义"])
        if kwargs:
            r.add_warning("以下参数未被识别，已忽略：" + ", ".join(sorted(kwargs)))
        return r

    p = C.parse_point(point, [var_sym] if var_sym is not None else [])
    status = _point_domain_status(expr_obj, var_sym, p)
    legal = status["legal"]
    value = status["value"]
    if legal is False:
        r.set_result(f"{_t(var_sym)} = {_t(p)} 不在定义域内")
        r.add_warning("指定点不在定义域内：" + "；".join(status["reasons"][:3]))
    elif legal is True:
        r.set_result(f"{_t(var_sym)} = {_t(p)} 在定义域内，表达式在该点取值为 {_t(value)}")
    else:
        r.set_result(f"无法确定 {_t(var_sym)} = {_t(p)} 是否在定义域内")
    r.set_raw(passed=(None if legal is None else bool(legal)),
              restrictions=restrictions_text,
              excluded_points=[f"{_t(s)} = {_t(p)}" for s, p in info["excluded"]],
              point_status={"legal": legal, "reasons": status["reasons"],
                            "value": None if value is None else _t(value),
                            "value_approx": _ev(expr_obj, {var_sym: p}) if var_sym is not None else None},
              note="legal = true 表示该点满足所有判定得到的定义域限制")
    r.verify(status="independent" if legal is not None else "unverified",
             methods=[{"method": "逐条把指定点代入定义域限制（分母≠0、log 参数>0、根式≥0、反三角∈[-1,1]、极点）",
                       "passed": None if legal is None else bool(legal),
                       "evidence": {"point": _t(p), "reasons": status["reasons"],
                                    "value": None if value is None else _t(value),
                                    "restrictions": restrictions_text}}])
    r.verification.update({"passed": None if legal is None else bool(legal),
                           "level": LEVEL_SYMBOLIC if info["complete"] else LEVEL_NUMERIC,
                           "method": "把指定点逐条代入定义域限制判定（分母/根式/对数/反三角/极点）",
                           "evidence": {"point": _t(p), "reasons": status["reasons"],
                                        "value": None if value is None else _t(value),
                                        "restrictions": restrictions_text},
                           "unhandled_conditions": [] if info["complete"] else
                           ["奇点集合可能不完整（含有周期极点或 sympy 无法完全解析的函数）"],
                           "cannot_verify_reason": None if legal is not None else "无法判定该点是否在定义域内"})
    r.add_condition(*restrictions_text or ["在整个实轴上定义"])
    if kwargs:
        r.add_warning("以下参数未被识别，已忽略：" + ", ".join(sorted(kwargs)))
    return r
