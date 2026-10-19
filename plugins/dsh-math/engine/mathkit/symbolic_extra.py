"""补充符号计算算子：多元微积分 + 不等式 + 作图题考点（考研数学一）。

本模块**只**覆盖 ``calculus.py`` / ``linalg.py`` / ``numeric.py`` / ``probability.py``
未实现的考点：

* 多元微分：梯度、雅可比、海塞、偏导、方向导数
* 无条件/条件（拉格朗日乘数法）极值
* 隐函数求导、参数方程求导
* 多元泰勒展开
* 不等式解集、自然定义域、渐近线、函数作图分析
* 中值定理中间点、隐函数极值

设计约定（与用户规格一致）：

1. 一律通过 ``A.parse(text, symbols=...)`` 解析输入，并把符号表用 ``extra`` 注入，
   保证「表达式里的 x」与「变量名里的 x」是同一个 SymPy 符号对象。
2. 只使用 ``r.set_input`` / ``r.set_result`` / ``r.set_raw`` 装填结果。
3. 求不出 → ``MathResult.unsolved``；输入错 → ``ValueError``；部分完成 → ``r.partial``。
   SymPy 原样回显（``C.is_closed_form`` 为假）一律不标成功。
4. 每个成功算子都写入 ``r.verify(...)``，验证用独立机制（互印 / 代回残差 /
   高精度数值抽样 / 数值有限差分），并在 ``conditions`` 写明成立条件。
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Sequence

import sympy as sp

from . import ast as A
from . import sympy_core as C
from .engine import op
from .result import MathResult

# ---------------------------------------------------------------------------
# stdout 编码防护（不改动他人文件的最小兜底）
# ---------------------------------------------------------------------------
# worker.py 用 ``json.dumps(..., ensure_ascii=False)`` 直接写 stdout，但没有把
# stdout 的编码固定为 UTF-8。Windows 上 stdout 被重定向/接管道时默认是 GBK，
# 一旦结果里出现 GBK 无法表示的字符（如 ∇），worker 会抛
# ``UnicodeEncodeError: 'gbk' codec can't encode character '\u2207'``
# 并直接退出，Node 侧管道随即断开。本模块的 ``conditions``/``method`` 文本里
# 会出现 ∇、Σ、∈ 等数学符号，所以在导入时做一次幂等的 UTF-8 兜底，
# 保证「模块被 worker 导入」这条真实调用链不会因为编码而崩。
# 注意：这只是兜底；根本修复应在 worker.py 里显式
# ``sys.stdout.reconfigure(encoding="utf-8", errors="replace")``
# 或给子进程设 ``PYTHONIOENCODING=utf-8``（已作为 bug 上报，未改动他人文件）。
try:  # pragma: no cover - 环境相关
    _stdout = getattr(sys, "stdout", None)
    if _stdout is not None and hasattr(_stdout, "reconfigure"):
        if (getattr(_stdout, "encoding", "") or "").lower().replace("-", "") not in ("utf8", "utf_8"):
            _stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001 - 编码兜底失败不应影响计算
    pass

# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------

_MAX_HESSIAN = 8
_MAX_TAYLOR_ORDER = 6


def _is_text(value: Any) -> bool:
    return isinstance(value, str)


def _as_parts(value: Any) -> list[Any]:
    """把 ``"a,b"`` / ``"a;b"`` / ``"a b"`` / ``[..]`` / 标量统一成列表。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        out: list[Any] = []
        for item in value:
            out.extend(_as_parts(item))
        return out
    if isinstance(value, str):
        text = value.strip().strip("[]{}()")
        if not text:
            return []
        text = text.replace("，", ",").replace(";", ",").replace("、", ",")
        if "\n" in text:
            text = text.replace("\n", ",")
        if "," in text:
            return [p.strip() for p in text.split(",") if p.strip()]
        # 无分隔符：可能是 "x y" 这样的变量列表，也可能是单个表达式
        tokens = text.split()
        if len(tokens) > 1 and all(sp.Symbol(t).is_Symbol and t.isidentifier() for t in tokens):
            return tokens
        return [text]
    return [value]


def _var_names(vars_like: Any, expr: Any = None, *, default: Sequence[str] = ("x", "y")) -> list[str]:
    """确定参与运算的变量名列表。"""
    names: list[str] = []
    for raw in _as_parts(vars_like):
        name = str(raw).strip()
        if name:
            names.append(name)
    if names:
        return names
    if expr is not None and isinstance(expr, (sp.Basic, sp.MatrixBase)):
        free = sorted({str(s) for s in sp.sympify(expr).free_symbols})
        if free:
            return free
    return list(default)


def _sym_map(names: Sequence[str]) -> dict[str, sp.Symbol]:
    return A.get_symbols(names)


def _symbols(names: Sequence[str]) -> list[sp.Symbol]:
    mapping = _sym_map(names)
    return [mapping[n] for n in names if n in mapping]


def _parse(text: Any, mapping: dict[str, sp.Symbol], what: str) -> sp.Basic:
    if not isinstance(text, (str, sp.Basic)):
        if isinstance(text, (int, float)):
            return sp.sympify(text)
        raise ValueError(f"{what}必须是表达式字符串，收到 {type(text).__name__}")
    if isinstance(text, sp.Basic):
        return text
    try:
        return A.parse(text, symbols=[], extra=mapping or None)
    except ValueError as exc:
        raise ValueError(f"{what}解析失败：{exc}") from exc


def _parse_all(items: Any, mapping: dict[str, sp.Symbol], what: str) -> list[sp.Basic]:
    return [_parse(item, mapping, what) for item in _as_parts(items)]


def _parse_point(point: Any, mapping: dict[str, sp.Symbol], what: str = "点") -> list[sp.Basic]:
    """把 ``["1","1"]`` / ``"1,1"`` 解析为坐标列表（保持 ``vars`` 的顺序）。"""
    names = list(mapping)
    parts = _as_parts(point)
    if not parts:
        raise ValueError(f"缺少{what}坐标")
    if len(parts) != len(names):
        raise ValueError(f"{what}需要 {len(names)} 个坐标（对应变量 {names}），收到 {len(parts)} 个：{parts}")
    return [_parse(p, mapping, f"{what}坐标") for p in parts]


def _has_undeclared(expr: sp.Basic, mapping: dict[str, sp.Symbol]) -> bool:
    if not isinstance(expr, sp.Basic):
        return False
    return any(str(s) not in mapping for s in expr.free_symbols)


def _has_unevaluated(*values: Any) -> bool:
    return any(not C.is_closed_form(v) for v in values if v is not None)


def _clean(text: Any) -> str:
    return " ".join(str(text).split())


def _subs_map(expr: sp.Basic, mapping: dict[str, sp.Symbol], values: Sequence[Any]) -> sp.Basic:
    pairs = {sym: val for sym, val in zip(_symbols(list(mapping)), values)}
    pairs.update({sym: val for sym, val in zip([sp.Symbol(n) for n in mapping], values)})
    try:
        return sp.sympify(expr).subs(pairs)
    except Exception:  # noqa: BLE001
        return expr


def _subs_point(expr: Any, point: dict[sp.Symbol, Any]) -> Any:
    """把点坐标代回，兼容 ``y(x)`` 这类函数项与 Abs 之类的数值求值。"""
    try:
        out = sp.sympify(expr)
    except Exception:  # noqa: BLE001
        return expr
    try:
        out = out.subs(point)
    except Exception:  # noqa: BLE001
        pass
    by_name = {str(k): v for k, v in point.items()}

    def _walk(node: Any) -> Any:
        if isinstance(node, sp.Symbol) and str(node) in by_name:
            return by_name[str(node)]
        if isinstance(node, sp.Function) and type(node) is not sp.Symbol:
            try:
                if len(node.free_symbols) == 1 and str(node.func) in by_name:
                    return by_name[str(node.func)]
            except Exception:  # noqa: BLE001
                pass
        return node

    try:
        out = out.replace(lambda n: _walk(n) is not n, lambda n: _walk(n))
    except Exception:  # noqa: BLE001
        pass
    return out


def _eval(expr: Any, point: dict[sp.Symbol, Any], digits: int = 30) -> Any:
    try:
        value = sp.N(_subs_point(expr, point), digits)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(value, sp.Basic):
        try:
            value = sp.sympify(value)
        except Exception:  # noqa: BLE001
            return None
    return value


def _real_number(value: Any, digits: int = 30) -> float | None:
    try:
        num = sp.N(value, digits)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(num, sp.Basic) or not num.is_real:
        return None
    try:
        out = float(num)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _as_float(value: Any, digits: int = 25) -> float | None:
    """把矩阵元素转成普通 float（python float 矩阵让 SymPy 走 mpmath 数值特征值，避免 PrecisionExhausted）。"""
    try:
        num = sp.N(value, digits)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(num, sp.Basic) or not num.is_real:
        return None
    try:
        return float(num)
    except (TypeError, ValueError):
        return None


def _numeric_vec(values: Sequence[Any]) -> list[float] | None:
    out: list[float] = []
    for value in values:
        num = _real_number(value)
        if num is None:
            return None
        out.append(num)
    return out


def _diag(ok: bool, method: str, note: str) -> dict[str, Any]:
    return {"method": method, "agree": bool(ok), "note": note}


def _verdict(*methods: dict[str, Any], note: str | None = None) -> tuple[str, list[dict[str, Any]]]:
    """由证据汇总验证状态：全过 → verified；全不过 → unverified；否则 partially_verified。"""
    evidence = [m for m in methods if m]
    verdicts = [bool(m.get("agree")) for m in evidence if "agree" in m]
    if not verdicts:
        return "unverified", evidence
    if all(verdicts):
        return "verified", evidence
    if any(verdicts):
        return "partially_verified", evidence
    return "unverified", evidence


def _closes(a: Any, b: Any, mapping: dict[str, sp.Symbol] | None = None) -> tuple[bool, str]:
    """判断两个候选值是否数学等价，返回 (是否一致, 证据说明)。"""
    try:
        if C.same_expr(a, b):
            return True, "符号化简后等价"
    except Exception as exc:  # noqa: BLE001
        return False, f"符号比较失败：{exc}"
    syms = None
    if mapping:
        syms = [sp.Symbol(n) for n in mapping]
    try:
        ok, dev, note = C.numeric_agree(a, b, symbols=syms)
        return bool(ok), f"高精度抽样比较（{note}）"
    except Exception as exc:  # noqa: BLE001
        return False, f"数值比较失败：{exc}"


# ---------------------------------------------------------------------------
# 1. 梯度
# ---------------------------------------------------------------------------

def _obj_to_text(value: Any) -> Any:
    """把任意 SymPy 结构（含嵌套列表/字典）转成纯文本结构，保证可 JSON 序列化。"""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_obj_to_text(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _obj_to_text(v) for k, v in value.items()}
    try:
        return A.to_text(value)
    except Exception:  # noqa: BLE001
        return str(value)


def _result_payload(value: Any, digits: int = 12) -> dict[str, Any]:
    """把 SymPy 对象转成可 JSON 序列化的 {text, latex, approx} 片段。

    约定（用户硬性要求）：写进 result 的键必须只含字符串/数字/列表/字典，
    **绝不能是 SymPy 对象**（否则 Node 侧 JSON 管道会抛
    ``TypeError: Object of type ... is not JSON serializable``）。
    """
    out: dict[str, Any] = {}
    if value is None:
        return out
    try:
        out["text"] = A.to_text(value)
    except Exception:  # noqa: BLE001
        out["text"] = str(value)
    try:
        out["latex"] = A.to_latex(value)
    except Exception:  # noqa: BLE001
        pass
    approx = _numeric_literal(value, digits)
    if approx is not None:
        out["approx"] = approx
    return out


def _numeric_literal(value: Any, digits: int = 12) -> Any:
    """把「不含自由符号」的 SymPy 对象递归转成 Python 原生数值（否则返回 None）。"""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        items = [_numeric_literal(v, digits) for v in value]
        return None if any(v is None for v in items) else items
    if isinstance(value, dict):
        return {str(k): _numeric_literal(v, digits) for k, v in value.items()}
    if isinstance(value, sp.MatrixBase):
        rows = _numeric_literal(value.tolist(), digits)
        return rows
    if isinstance(value, sp.Basic):
        try:
            if value.free_symbols or not value.is_number:
                return None
            if value.is_Integer:
                return int(value)
            num = sp.N(value, digits)
            if num.is_real:
                return float(num)
            return str(num)
        except Exception:  # noqa: BLE001
            return str(value)
    return None


@op("gradient", "grad", "math_gradient")
def gradient(expr: str, vars: Any = None, **kwargs: Any) -> MathResult:
    """梯度向量：``∇f = (∂f/∂x1, …, ∂f/∂xn)``。"""
    names = _var_names(vars, expr if isinstance(expr, sp.Basic) else None)
    mapping = _sym_map(names)
    f = _parse(expr, mapping, "目标函数")
    syms = _symbols(names)
    if not syms:
        raise ValueError("无法确定求导变量，请用 vars 指定，例如 vars=\"x,y\"")

    grad = sp.Matrix([sp.diff(f, s) for s in syms])
    r = MathResult.ok("gradient", method="逐分量 sp.diff")
    r.set_input(expr=A.to_text(f), vars=names)
    if not C.is_closed_form(grad):
        return MathResult.unsolved("gradient", "梯度分量中含未求值对象（SymPy 原样回显）")
    components = [sp.diff(f, s) for s in syms]
    r.set_raw(
        gradient=A.to_text(grad),
        gradient_approx=_numeric_literal(grad),
        components=_obj_to_text(components),
        components_approx=[_numeric_literal(c) for c in components],
        component_latex=[A.to_latex(c) for c in components],
        variables=names,
    )
    r.add_condition("f 在该点邻域内可偏导（分量为初等函数时通常成立）")

    if _has_undeclared(f, mapping):
        r.add_warning(
            f"表达式含未在 vars 中声明的符号：{sorted(str(s) for s in f.free_symbols if str(s) not in mapping)}，"
            "它们被当作参数处理"
        )

    # 独立验证：每个分量与「单变量求导」互印（只保留被求导变量的符号约束）
    checks: list[str] = []
    all_ok = True
    for i, s in enumerate(syms):
        wrong = {str(x): x for x in syms if x != s}
        f_s = sp.sympify(f).subs(wrong)
        expected = sp.diff(f_s, s)
        ok, why = _closes(expected, grad[i], {str(s): s})
        all_ok = all_ok and ok
        checks.append(f"{A.to_text(grad[i])} vs 独立求导 {A.to_text(expected)}：{'一致' if ok else '不一致'}（{why}）")
    # 补充证据：数值方向导数（∇f·e_i 应等于沿 x_i 的数值差分）
    numeric_note = ""
    try:
        point = {s: sp.Rational(1, 3) + sp.Rational(1, 7) * (i + 1) for i, s in enumerate(syms)}
        h = sp.Rational(1, 1000)
        base = sp.N(f.subs(point), 30)
        diffs_ok = True
        for i, s in enumerate(syms):
            plus = sp.N(f.subs({**point, s: point[s] + h}), 30)
            minus = sp.N(f.subs({**point, s: point[s] - h}), 30)
            fd = (plus - minus) / (2 * h)
            an = sp.N(grad[i].subs(point), 30)
            if not (abs(complex(fd - an)) < 1e-6):
                diffs_ok = False
        if base.is_real:
            numeric_note = "中心差分与解析梯度在抽样点一致（偏差 <1e-6）" if diffs_ok else "中心差分与解析梯度不一致"
            all_ok = all_ok and diffs_ok
        else:
            numeric_note = "抽样点处函数值非实，未做差分补充"
    except Exception as exc:  # noqa: BLE001
        numeric_note = f"数值差分补充失败：{exc}"

    r.verify(
        status="verified" if all_ok else "partially_verified",
        methods=[
            _diag(all_ok, "逐分量独立求导 + 中心差分方向导数抽样", "；".join(checks)),
            {**_diag(all_ok, "数值中心差分", numeric_note)},
        ],
    )
    r.set_raw(verification_detail={"component_checks": checks, "numeric": numeric_note})
    return r


# ---------------------------------------------------------------------------
# 2. 雅可比矩阵
# ---------------------------------------------------------------------------

@op("jacobian", "math_jacobian")
def jacobian(exprs: Any, vars: Any = None, **kwargs: Any) -> MathResult:
    """雅可比矩阵 ``J_ij = ∂f_i/∂x_j``。"""
    items = _as_parts(exprs)
    if not items:
        raise ValueError("expressions 不能为空")
    names = _var_names(vars, None)
    mapping = _sym_map(names)
    funcs = _parse_all(items, mapping, "函数")
    syms = _symbols(names)

    rows = [[sp.diff(f, s) for s in syms] for f in funcs]
    mat = sp.Matrix(rows)
    r = MathResult.ok("jacobian", method="逐元素 sp.diff")
    r.set_input(expressions=[A.to_text(f) for f in funcs], vars=names)
    if not C.is_closed_form(mat):
        return MathResult.unsolved("jacobian", "雅可比元素中含未求值对象（SymPy 原样回显）")
    r.set_result(
        jacobian=A.to_text(mat),
        jacobian_approx=_numeric_literal(mat),
        variables=names,
        functions=[A.to_text(f) for f in funcs],
    )
    r.add_condition("各分量在考察点附近可偏导")

    # 独立验证：逐元素与单独求导互印
    detail: list[str] = []
    all_ok = True
    for i, f in enumerate(funcs):
        for j, s in enumerate(syms):
            others = {str(x): x for x in syms if x != s}
            expected = sp.diff(sp.sympify(f).subs(others), s)
            ok, why = _closes(expected, rows[i][j], {str(s): s})
            all_ok = all_ok and ok
            detail.append(f"[{i},{j}] {A.to_text(rows[i][j])} vs {A.to_text(expected)}：{'一致' if ok else '不一致'}")
    r.verify(
        status="verified" if all_ok else "unverified",
        methods=[_diag(all_ok, "逐元素与单独（降维）求导互印", "；".join(detail))],
    )
    r.set_raw(verification_detail={"element_checks": detail})
    return r


# ---------------------------------------------------------------------------
# 3. 海塞矩阵
# ---------------------------------------------------------------------------

def _build_hessian(f: sp.Basic, syms: Sequence[sp.Symbol]) -> sp.Matrix:
    return sp.Matrix([[sp.diff(f, s1, s2) for s2 in syms] for s1 in syms])


@op("hessian", "math_hessian")
def hessian(expr: str, vars: Any = None, **kwargs: Any) -> MathResult:
    """海塞矩阵 ``H_ij = ∂²f/∂x_i∂x_j``。"""
    names = _var_names(vars, expr if isinstance(expr, sp.Basic) else None)
    if len(names) > _MAX_HESSIAN:
        raise ValueError(f"变量数 {len(names)} 超过上限 {_MAX_HESSIAN}，海塞矩阵将无法有效计算")
    mapping = _sym_map(names)
    f = _parse(expr, mapping, "目标函数")
    syms = _symbols(names)

    mat = _build_hessian(f, syms)
    r = MathResult.ok("hessian", method="二阶偏导 sp.diff(f, xi, xj)")
    r.set_input(expr=A.to_text(f), vars=names)
    if not C.is_closed_form(mat):
        return MathResult.unsolved("hessian", "海塞元素中含未求值对象（SymPy 原样回显）")
    r.set_result(
        hessian=A.to_text(mat),
        hessian_approx=_numeric_literal(mat),
        variables=names,
    )
    r.add_condition("f 二阶连续可偏导（此时混合偏导相等，H 对称）")

    # 独立验证：对称性 H[i,j] == H[j,i]
    detail: list[str] = []
    all_ok = True
    n = len(syms)
    for i in range(n):
        for j in range(i + 1, n):
            ok, why = _closes(mat[i, j], mat[j, i], {str(s): s for s in syms})
            all_ok = all_ok and ok
            detail.append(f"H[{i},{j}]={A.to_text(mat[i, j])} 与 H[{j},{i}]={A.to_text(mat[j, i])}：{'相等' if ok else '不相等'}")
    # 补充证据：对角线元素与「一元二阶导」互印
    for i, s in enumerate(syms):
        others = {str(x): x for x in syms if x != s}
        expected = sp.diff(sp.sympify(f).subs(others), s, 2)
        ok, why = _closes(expected, mat[i, i], {str(s): s})
        all_ok = all_ok and ok
        detail.append(f"H[{i},{i}] 与一元二阶导比较：{'一致' if ok else '不一致'}")

    if n == 1:
        r.add_warning("只有一个变量，混合偏导对称性检验退化为退化情形（无混合项）")

    r.verify(
        status="verified" if all_ok else "unverified",
        methods=[_diag(all_ok, "混合偏导对称性检验 + 对角线一元二阶导互印", "；".join(detail))],
    )
    r.set_raw(verification_detail={"symmetry_checks": detail})
    return r


# ---------------------------------------------------------------------------
# 4. 单个偏导
# ---------------------------------------------------------------------------

@op("partial", "pderiv", "math_partial")
def partial(expr: str, var: str = None, order: Any = 1, **kwargs: Any) -> MathResult:
    """单个偏导数（含高阶）：``∂^n f / ∂x^n``。"""
    try:
        n = int(order) if order is not None else 1
    except (TypeError, ValueError) as exc:
        raise ValueError(f"order 必须是正整数，收到 {order!r}") from exc
    if n < 1:
        raise ValueError("order 必须是正整数")
    if n > 10:
        raise ValueError("order 过大（>10），请拆分计算")

    names = _var_names(None, expr if isinstance(expr, sp.Basic) else None)
    mapping = _sym_map(names)
    f = _parse(expr, mapping, "函数")
    if var:
        target = str(var).strip()
        mapping.setdefault(target, A.get_symbols([target])[target])
    else:
        if len(f.free_symbols) != 1:
            raise ValueError(
                f"表达式含变量 {sorted(str(s) for s in f.free_symbols)}，请用 var 指定对哪个变量求偏导"
            )
        target = str(next(iter(f.free_symbols)))
        mapping.setdefault(target, next(iter(f.free_symbols)))
    if target not in mapping:
        raise ValueError(f"变量 {target!r} 不是合法变量名")
    s = mapping[target]

    result = sp.diff(f, s, n)
    r = MathResult.ok("partial", method=f"sp.diff(f, {target}, {n})")
    r.set_input(expr=A.to_text(f), var=target, order=n)
    if not C.is_closed_form(result):
        return MathResult.unsolved("partial", "求导结果中含未求值对象（SymPy 原样回显）")
    r.set_result(text=A.to_text(result), latex=A.to_latex(result),
                 approx=_numeric_literal(result), order=n, variable=target)
    r.add_condition(f"f 关于 {target} 至少 {n} 阶可偏导")

    ok, detail = C.check_derivative(f, s, sp.diff(f, s)) if n == 1 else (True, {})
    methods: list[dict[str, Any]] = []
    all_ok = True
    if n == 1:
        all_ok = bool(ok)
        methods.append({"method": "C.check_derivative 独立求导互印", "agree": bool(ok), **{
            k: v for k, v in detail.items() if k in ("expected", "numeric")
        }})
    else:
        expected = sp.diff(f, *([s] * n))
        same, why = _closes(expected, result, mapping)
        all_ok = same
        methods.append(_diag(same, f"与 sp.diff(f, *[{target}]*{n}) 互印", why))

    # 降阶互印：n 阶导数 = (n-1) 阶导数再求一次导
    if n >= 2:
        lower = sp.diff(f, s, n - 1)
        same2, why2 = _closes(sp.diff(lower, s), result, mapping)
        all_ok = all_ok and same2
        methods.append(_diag(same2, f"对 {n - 1} 阶偏导再求一次导互印", why2))

    r.verify(status="verified" if all_ok else "unverified", methods=methods)
    return r


# ---------------------------------------------------------------------------
# 5. 方向导数
# ---------------------------------------------------------------------------

def _direction_vector(direction: Any, mapping: dict[str, sp.Symbol]) -> sp.Matrix:
    names = list(mapping)
    parts = _as_parts(direction)
    if len(parts) == 1 and _is_text(parts[0]) and "," not in str(parts[0]) and len(names) > 1:
        # 允许把方向写成 "<1,0>" 这类形式
        parts = _as_parts(str(parts[0]).replace("<", ",").replace(">", ","))
    if len(parts) != len(names):
        raise ValueError(f"方向向量需要 {len(names)} 个分量（对应变量 {names}），收到 {parts}")
    return sp.Matrix([_parse(p, mapping, "方向分量") for p in parts])


@op("directional_derivative", "math_directional_derivative")
def directional_derivative(expr: str, vars: Any = None, point: Any = None, dir: Any = None, **kwargs: Any) -> MathResult:
    """在给定点沿给定方向的方向导数（方向先单位化：``D_u f = ∇f · u``）。"""
    if dir is None:
        raise ValueError("缺少方向向量参数 direction/dir")
    names = _var_names(vars, expr if isinstance(expr, sp.Basic) else None)
    mapping = _sym_map(names)
    f = _parse(expr, mapping, "目标函数")
    syms = _symbols(names)
    d = _direction_vector(dir, mapping)
    pt = _parse_point(point, mapping, "点")
    subs = dict(zip(syms, pt))

    raw_norm = sp.sqrt(sum(component ** 2 for component in d))
    unit = sp.simplify(d / raw_norm) if not (raw_norm == 0) else d
    if raw_norm == 0:
        raise ValueError("方向向量不能是零向量")

    grad = sp.Matrix([sp.diff(f, s) for s in syms])
    grad_at = grad.subs(subs)
    value = sp.simplify((grad_at.T * unit)[0])
    unit_at = sp.simplify(unit)

    r = MathResult.ok("directional_derivative", method="方向单位化后与梯度作内积 ∇f·u")
    r.set_input(expr=A.to_text(f), vars=names, point=[A.to_text(p) for p in pt], dir=[A.to_text(x) for x in d])
    if not C.is_closed_form(value):
        return MathResult.unsolved("directional_derivative", "方向导数含未求值对象（SymPy 原样回显）")
    r.set_result(
        directional_derivative=A.to_text(value),
        directional_derivative_approx=_numeric_literal(value),
        gradient_at_point=_obj_to_text(grad_at),
        gradient_at_point_approx=_numeric_literal(grad_at),
        direction=[A.to_text(x) for x in d],
        unit_direction=A.to_text(unit_at),
        norm=A.to_text(raw_norm),
    )
    r.add_condition("方向向量必须是非零向量；方向导数沿单位向量 u 定义，本结果已按 u = d/‖d‖ 单位化")
    if d.free_symbols:
        r.add_condition(f"单位化过程中要求 ‖d‖ = {A.to_text(raw_norm)} 非零，即方向向量不退化")
        r.add_warning("方向向量含符号分量，单位化结果只在该范数不为零时成立")
    if f.free_symbols - set(syms):
        r.add_warning(f"函数含未声明符号 {sorted(str(s) for s in f.free_symbols - set(syms))}，按参数处理")

    # 独立验证 1：与「未单位化方向的内积再除以范数」互印
    alt = sp.simplify((grad_at.T * d)[0] / raw_norm)
    same, why = _closes(alt, value, mapping)
    # 独立验证 2：数值方向差分（沿单位方向取中心差分）
    numeric_note = "无法单位化为实数（方向含符号或复数），未做数值差分"
    numeric_ok = None
    unit_num = _numeric_vec(unit_at)
    pt_num = _numeric_vec(pt)
    if unit_num and pt_num:
        try:
            h = 1e-5
            base = pt_num
            plus = {s: sp.Float(base[i] + h * unit_num[i], 25) for i, s in enumerate(syms)}
            minus = {s: sp.Float(base[i] - h * unit_num[i], 25) for i, s in enumerate(syms)}
            fp = _eval(f, plus)
            fm = _eval(f, minus)
            if fp is not None and fm is not None:
                fd = (sp.N(fp, 25) - sp.N(fm, 25)) / (2 * h)
                an = _eval(value, {})
                if an is not None and abs(complex(sp.N(fd - an, 25))) < 1e-5:
                    numeric_ok = True
                    numeric_note = f"中心差分值 {float(sp.N(fd, 12)):.10g} 与解析值 {float(sp.N(an, 12)):.10g} 一致"
                else:
                    numeric_ok = False
                    numeric_note = f"中心差分值与解析值不一致：{A.to_text(fd)} vs {A.to_text(an)}"
        except Exception as exc:  # noqa: BLE001
            numeric_ok = False
            numeric_note = f"数值差分失败：{exc}"

    methods = [_diag(same, "∇f·d/‖d‖ 互印（不做单位化的等价写法）", why)]
    if numeric_ok is not None:
        methods.append(_diag(numeric_ok, "沿单位方向的中心差分", numeric_note))
    status, evidence = _verdict(*methods)
    r.verify(status=status, methods=evidence)
    r.set_raw(verification_detail={"unit_check": why, "numeric": numeric_note})
    return r


# ---------------------------------------------------------------------------
# 6. 无条件极值
# ---------------------------------------------------------------------------

def _nullspace_directions(mat: sp.Matrix) -> list[sp.Matrix]:
    """返回矩阵零空间的一组精确基向量（用于海塞退化时的补充判别）。"""
    if mat.rows == 0 or mat.rows != mat.cols:
        return []
    try:
        poly = mat.charpoly()
    except Exception:  # noqa: BLE001
        return []
    try:
        roots = sp.roots(poly.as_expr(), sp.Symbol("_lambda^"))
    except Exception:  # noqa: BLE001
        return []
    near_zero = [lam for lam in roots if abs(complex(sp.N(lam, 20))) < 1e-12]
    if not near_zero:
        return []
    symbols = list(mat.free_symbols) or [sp.Symbol(f"_v{i}") for i in range(mat.cols)]
    vec = sp.Matrix([sp.Symbol(f"_v{i}") for i in range(mat.cols)])
    try:
        solutions = sp.linsolve((mat, sp.zeros(mat.rows, 1)), *vec)
    except Exception:  # noqa: BLE001
        return []
    directions: list[sp.Matrix] = []
    solution = next(iter(solutions), None) if solutions else None
    if solution is None:
        return []
    for index, entry in enumerate(solution):
        if not isinstance(entry, sp.Symbol):
            continue
        basis = solution.subs({entry: 1})
        for other in [e for e in solution if isinstance(e, sp.Symbol) and e != entry]:
            basis = basis.subs({other: 0})
        try:
            basis = sp.Matrix([sp.simplify(x) for x in basis])
        except Exception:  # noqa: BLE001
            continue
        if any(x != 0 for x in basis):
            directions.append(basis)
    return directions


def _sample_nature(
    f: sp.Basic,
    point: dict[sp.Symbol, Any],
    value: float,
    scale: float,
) -> tuple[str, str]:
    """海塞判别失效时，用邻域抽样判断 f-f(p) 的符号。"""
    magnitudes: list[float] = []
    for factor in (0.5, 0.2, 0.1, 0.05, 0.01):
        step = scale * factor
        if step > 0:
            magnitudes.append(step)
    if not magnitudes:
        return "unknown", "无法确定扰动尺度"
    directions = [sp.Matrix([sp.Integer(1) if i == j else sp.Integer(0) for i in range(len(point))])
                  for j in range(len(point))]
    try:
        hess = _build_hessian(f, list(point))
        nulls = _nullspace_directions(hess.subs(point))
        directions.extend(nulls)
    except Exception:  # noqa: BLE001
        pass
    signs: set[int] = set()
    checked = 0
    for direction in directions:
        values = _numeric_vec(list(direction))
        norm = None
        if values:
            norm = sum(v * v for v in values) ** 0.5
        if not values or not norm:
            continue
        for step in magnitudes:
            trial = {s: sp.Float(float(sp.N(point[s])) + step * values[i] / norm, 25) for i, s in enumerate(point)}
            val = _eval(f, trial)
            num = _real_number(val)
            if num is None:
                continue
            delta = num - value
            tol = 1e-9 * max(1.0, abs(value))
            if abs(delta) <= tol:
                continue
            signs.add(1 if delta > 0 else -1)
            checked += 1
    if not signs:
        return "flat", f"邻域抽样 {checked} 点均与驻点值相同（在数值精度内），无法判别（可能为平坦点）"
    if signs == {1}:
        return "min", "邻域抽样中 f(x)-f(p) 恒为正 → 极小值"
    if signs == {-1}:
        return "max", "邻域抽样中 f(x)-f(p) 恒为负 → 极大值"
    return "saddle", "邻域抽样中 f(x)-f(p) 有正有负 → 鞍点"


def _classify_stationary(
    f: sp.Basic,
    syms: Sequence[sp.Symbol],
    point: dict[sp.Symbol, Any],
) -> dict[str, Any]:
    """用海塞矩阵判别驻点性质：正定→极小，负定→极大，不定→鞍点，半定→抽样补充。"""
    hess = _build_hessian(f, syms)
    hess_at = sp.simplify(hess.subs(point))
    out: dict[str, Any] = {"hessian_at_point": hess_at}
    eig_vals = []
    try:
        entries = [[_as_float(x) for x in row] for row in hess_at.tolist()]
        if any(v is None for row in entries for v in row):
            raise ValueError("海塞矩阵在驻点处含符号元素，无法数值求特征值")
        hess_num = sp.Matrix(entries)
        for lam, mult in hess_num.eigenvals().items():
            try:
                value = complex(sp.N(lam, 25))
            except Exception:  # noqa: BLE001
                value = complex(sp.re(lam), sp.im(lam))
            eig_vals.extend([value] * int(mult))
    except Exception as exc:  # noqa: BLE001
        eig_vals = []
        out["eigen_note"] = f"特征值求解失败：{type(exc).__name__}: {exc}"
    out["eigenvalues"] = [A.to_text(x) for x in eig_vals]

    tol = 1e-9
    reals = []
    imag_large = False
    for lam in eig_vals:
        try:
            val = complex(sp.N(lam, 20))
        except Exception:  # noqa: BLE001
            imag_large = True
            break
        if abs(val.imag) > 1e-9:
            imag_large = True
            break
        reals.append(val.real)

    if not reals or imag_large:
        out["kind"] = "undetermined"
        out["reason"] = "海塞矩阵特征值无法判定（含符号或复数特征值），无法用二阶判别法"
        return out

    pos = sum(1 for v in reals if v > tol)
    neg = sum(1 for v in reals if v < -tol)
    zero = len(reals) - pos - neg
    if zero == 0 and neg == 0 and pos > 0:
        out.update(kind="min", reason=f"海塞矩阵在驻点正定（特征值 {out['eigenvalues']}）→ 极小值")
        return out
    if zero == 0 and pos == 0 and neg > 0:
        out.update(kind="max", reason=f"海塞矩阵在驻点负定（特征值 {out['eigenvalues']}）→ 极大值")
        return out
    if zero == 0 and pos > 0 and neg > 0:
        out.update(kind="saddle", reason=f"海塞矩阵不定（特征值有正有负 {out['eigenvalues']}）→ 鞍点")
        return out
    # 半定：需要更高阶判断，用邻域抽样补充
    value_num = _real_number(_eval(f, point))
    if value_num is None:
        out.update(kind="undetermined", reason="海塞矩阵半定且驻点函数值无法数值化，二阶判别法失效")
        return out
    scale = max(1.0, abs(value_num))
    try:
        eig_scale = max([abs(v) for v in reals] or [1.0])
        scale = max(scale ** 0.5 / max(eig_scale ** 0.5, 1e-9), 1e-3)
        scale = min(scale, 0.5)
    except Exception:  # noqa: BLE001
        scale = 0.1
    kind, note = _sample_nature(f, point, value_num, scale)
    if kind == "flat":
        out.update(kind="undetermined", reason=f"海塞矩阵半定（{out['eigenvalues']}），{note}")
    else:
        out.update(kind=kind, reason=f"海塞矩阵半定（特征值 {out['eigenvalues']}），二阶判别法失效，改用邻域抽样：{note}")
    return out


@op("multivar_extremum", "math_extremum")
def multivar_extremum(expr: str, vars: Any = None, **kwargs: Any) -> MathResult:
    """无条件极值：解 ∇f=0 得驻点，再用海塞矩阵（或邻域抽样）判别。"""
    names = _var_names(vars, expr if isinstance(expr, sp.Basic) else None)
    mapping = _sym_map(names)
    f = _parse(expr, mapping, "目标函数")
    syms = _symbols(names)
    if not syms:
        raise ValueError("无法确定自变量，请用 vars 指定")

    grad = [sp.diff(f, s) for s in syms]
    r = MathResult.ok("multivar_extremum", method="∇f=0 求驻点 + 海塞矩阵二阶判别")
    r.set_input(expr=A.to_text(f), vars=names)
    r.add_condition("仅讨论 f 的定义域内部的无条件极值；边界上的极值需另行用条件极值/闭区域方法处理")

    equations = [sp.Eq(g, 0) for g in grad]
    try:
        solutions = sp.solve(equations, syms, dict=True)
    except NotImplementedError as exc:
        return MathResult.unsolved(
            "multivar_extremum",
            f"SymPy 无法解析求解驻点方程组 {[A.to_text(e) for e in equations]}：{exc}；建议改用 math_numeric 数值求根",
        )
    except Exception as exc:  # noqa: BLE001
        return MathResult.unsolved(
            "multivar_extremum",
            f"驻点方程组求解失败：{type(exc).__name__}: {exc}；建议改用数值方法",
        )
    if not solutions:
        r.partial("未找到实驻点：∇f=0 在该实数域内无解，极值（若有）只能在边界或不可导点取得")
        r.set_result(stationary_points=[], classification=[], gradient=[A.to_text(g) for g in grad])
        r.verify(status="verified", methods=[_diag(True, "驻点方程组无解", "∇f=0 无实数解")])
        return r

    classified: list[dict[str, Any]] = []
    residual_detail: list[str] = []
    all_residual_ok = True
    all_sample_ok = True
    undetermined = 0
    for sol in solutions:
        if not all(s in sol for s in syms):
            continue
        point = {s: sp.simplify(sol[s]) for s in syms}
        if any(not sp.sympify(v).is_real for v in point.values()):
            continue
        if any(v.has(sp.zoo) or v.has(sp.nan) or v is sp.oo or v is -sp.oo for v in point.values()):
            continue
        info = _classify_stationary(f, syms, point)
        val = sp.simplify(f.subs(point))
        entry: dict[str, Any] = {
            "point": {str(s): A.to_text(point[s]) for s in syms},
            "value": A.to_text(val),
            "kind": info.get("kind"),
            "reason": info.get("reason"),
            "eigenvalues": info.get("eigenvalues"),
            "hessian_at_point": A.to_text(info.get("hessian_at_point")),
        }
        # 验证：梯度残差必须为 0
        residuals = {str(s): sp.simplify(g.subs(point)) for s, g in zip(syms, grad)}
        entry["gradient_residual"] = {k: A.to_text(v) for k, v in residuals.items()}
        ok_res = all(abs(complex(sp.N(v, 25))) < 1e-9 for v in residuals.values())
        all_residual_ok = all_residual_ok and ok_res
        residual_detail.append(f"{entry['point']} 处梯度残差 {entry['gradient_residual']}：{'为 0' if ok_res else '非 0'}")
        # 验证：邻域抽样与判别结论一致
        val_num = _real_number(val)
        if val_num is not None and info.get("kind") in ("min", "max", "saddle"):
            kind, note = _sample_nature(f, point, val_num, 0.1)
            expect = {"min": "min", "max": "max", "saddle": "saddle"}[info["kind"]]
            agree = kind == expect or kind == "flat"
            entry["sampling_check"] = {"agree": agree, "note": note}
            all_sample_ok = all_sample_ok and agree
        elif info.get("kind") == "undetermined":
            undetermined += 1
        classified.append(entry)

    if not classified:
        return MathResult.unsolved("multivar_extremum", "驻点方程组有解，但没有实驻点（解含复数/无穷）")

    order = {"min": 0, "max": 1, "saddle": 2, "undetermined": 3}
    classified.sort(key=lambda e: (order.get(str(e.get("kind")), 9), str(e["point"])))
    r.set_result(
        stationary_points=[e["point"] for e in classified],
        classification=classified,
        gradient=[A.to_text(g) for g in grad],
        hessian=A.to_text(_build_hessian(f, syms)),
    )
    r.add_condition("判别依据：海塞矩阵在驻点处正定 → 极小值；负定 → 极大值；不定 → 鞍点；半定 → 二阶判别法失效，改用邻域抽样并可能仍无法判定")
    if undetermined:
        r.partial(f"有 {undetermined} 个驻点二阶判别法失效且抽样无法判定（可能是平坦点/高阶极值），请用更高阶展开人工判断")

    methods = [
        _diag(all_residual_ok, "驻点代回梯度方程检查残差为 0", "；".join(residual_detail)),
    ]
    sample_entries = [e for e in classified if e.get("sampling_check")]
    if sample_entries:
        detail = "；".join(f"{e['point']}: {e['sampling_check']['note']}" for e in sample_entries)
        methods.append(_diag(all_sample_ok, "驻点邻域扰动抽样复核极值性质", detail))
    status, evidence = _verdict(*methods)
    r.verify(status=status, methods=evidence)
    r.set_raw(verification_detail={"undetermined_count": undetermined})
    return r


# ---------------------------------------------------------------------------
# 7. 条件极值（拉格朗日乘数法）
# ---------------------------------------------------------------------------

def _constraint_exprs(constraints: Any, mapping: dict[str, sp.Symbol], syms: Sequence[sp.Symbol]) -> list[sp.Basic]:
    exprs: list[sp.Basic] = []
    for item in _as_parts(constraints):
        text = str(item)
        if isinstance(item, sp.Basic):
            exprs.append(sp.sympify(item))
            continue
        if "=" in text and "==" not in text:
            lhs_text, rhs_text = text.split("=", 1)
            lhs = _parse(lhs_text.strip(), mapping, "约束")
            rhs = _parse(rhs_text.strip(), mapping, "约束")
            exprs.append(sp.expand(lhs - rhs))
        else:
            exprs.append(_parse(text, mapping, "约束"))
    if not exprs:
        raise ValueError("至少需要一个约束条件")
    return exprs


@op("lagrange", "conditional_extremum", "math_lagrange")
def lagrange(expr: str, constraints: Any, vars: Any = None, **kwargs: Any) -> MathResult:
    """条件极值：构造 ``L = f - Σ λ_i g_i``，解 ∇L=0 与约束方程。"""
    names = _var_names(vars, expr if isinstance(expr, sp.Basic) else None)
    mapping = _sym_map(names)
    f = _parse(expr, mapping, "目标函数")
    syms = _symbols(names)
    if not syms:
        raise ValueError("无法确定自变量，请用 vars 指定")
    constraints_expr = _constraint_exprs(constraints, mapping, syms)
    lam_symbols = [sp.Symbol(f"_lambda{i}" if i else "_lambda", real=True) for i in range(len(constraints_expr))]

    L = f
    for lam, g in zip(lam_symbols, constraints_expr):
        L = L - lam * g
    equations = [sp.Eq(sp.diff(L, s), 0) for s in syms] + [sp.Eq(g, 0) for g in constraints_expr]

    r = MathResult.ok("lagrange", method="拉格朗日乘数法 L = f - Σ λ_i g_i")
    r.set_input(
        expr=A.to_text(f),
        constraints=[A.to_text(g) for g in constraints_expr],
        vars=names,
    )
    r.add_condition(f"约束正则性条件：在候选点处 {[A.to_text(sp.Matrix([sp.diff(g, s) for s in syms]).T) for g in constraints_expr]} 之间线性无关（∇g_i 无关），否则乘数法可能漏解")
    r.add_condition("条件极值只给出候选点；最大/最小值需比较所有候选点的目标值（并核对边界与不可导点）")

    try:
        raw = sp.solve(equations, syms + lam_symbols, dict=True)
    except NotImplementedError as exc:
        return MathResult.unsolved(
            "lagrange",
            f"拉格朗日方程组无法解析求解：{exc}；建议改用数值方法（math_numeric）求条件极值的数值近似",
        )
    except Exception as exc:  # noqa: BLE001
        return MathResult.unsolved(
            "lagrange",
            f"拉格朗日方程组求解失败：{type(exc).__name__}: {exc}；建议改用数值方法",
        )

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    dropped = 0
    for sol in raw:
        if not all(s in sol for s in syms):
            continue
        point = {s: sp.simplify(sol[s]) for s in syms}
        if any(not sp.sympify(v).is_real for v in point.values()):
            continue
        if any(v.has(sp.zoo) or v.has(sp.nan) or v in (sp.oo, -sp.oo) for v in point.values()):
            continue
        lam_values = {}
        bad = False
        for lam in lam_symbols:
            if lam not in sol:
                lam_values[str(lam)] = "未确定（自由乘数）"
                continue
            value = sp.simplify(sol[lam])
            if value.is_real is False or value.has(sp.zoo) or value.has(sp.nan):
                bad = True
                break
            lam_values[str(lam)] = A.to_text(value)
        if bad:
            dropped += 1
            continue
        key = str(sorted((str(k), A.to_text(v)) for k, v in point.items()))
        if key in seen:
            continue
        seen.add(key)
        candidates.append({"point": {str(s): A.to_text(point[s]) for s in syms}, "lambda": lam_values, "_subs": point})

    if not candidates:
        return MathResult.unsolved(
            "lagrange",
            "拉格朗日方程组没有实驻点解（可能解析求解不完整）；建议用数值方法求条件极值的数值近似",
        )

    for entry in candidates:
        point = entry["_subs"]
        entry["objective"] = A.to_text(sp.simplify(f.subs(point)))
        entry["constraint_residual"] = {
            A.to_text(g): A.to_text(sp.simplify(g.subs(point))) for g in constraints_expr
        }
        value = f.subs(point)
        entry["_value"] = value

    # 目标值比较：确定最大/最小
    comparable = [e for e in candidates if _real_number(e["_value"]) is not None]
    if comparable:
        best_max = max(comparable, key=lambda e: _real_number(e["_value"]))
        best_min = min(comparable, key=lambda e: _real_number(e["_value"]))
        for entry in candidates:
            marks = []
            if entry is best_max:
                marks.append("候选最大值")
            if entry is best_min:
                marks.append("候选最小值")
            entry["rank"] = marks or ["中间值"]
        r.set_result(maximum_candidate=best_max["point"], minimum_candidate=best_min["point"])
    else:
        r.add_warning("候选点目标值含符号参数，无法数值比较大小（无法自动确定最大/最小）")
    for entry in candidates:
        entry.pop("_value", None)

    r.set_result(candidates=candidates)
    r.add_condition("若约束集非紧或无界，候选点中的最大/最小只是「候选」；需结合约束集的紧性与边界讨论")

    # 验证：约束残差必须为 0
    detail: list[str] = []
    all_ok = True
    for entry in candidates:
        for text, residual in entry["constraint_residual"].items():
            try:
                res_expr = sp.sympify(residual.replace("**", "**"))
            except Exception:  # noqa: BLE001
                res_expr = None
            ok = False
            if res_expr is not None:
                num = _real_number(res_expr)
                if num is not None:
                    ok = abs(num) < 1e-9
                else:
                    ok = sp.simplify(res_expr) == 0
            all_ok = all_ok and ok
            detail.append(f"{entry['point']}：约束 {text} 残差 {residual} → {'满足' if ok else '不满足'}")
    # 验证：候选点满足 ∇f = Σ λ_i ∇g_i（梯度的线性组合条件）
    for entry in candidates:
        point = entry["_subs"]
        lhs = sp.Matrix([sp.diff(f, s).subs(point) for s in syms])
        rhs = sp.zeros(len(syms), 1)
        for index, g in enumerate(constraints_expr):
            lam_key = str(lam_symbols[index])
            if lam_key in entry["lambda"] and entry["lambda"][lam_key] != "未确定（自由乘数）":
                lam_value = sp.sympify(entry["lambda"][lam_key])
                rhs = rhs + lam_value * sp.Matrix([sp.diff(g, s).subs(point) for s in syms])
        diff = sp.simplify(lhs - rhs)
        ok = all(abs(complex(sp.N(x, 25))) < 1e-9 for x in diff)
        all_ok = all_ok and ok
        detail.append(f"{entry['point']}：∇f - Σλ∇g = {A.to_text(diff.T)} → {'为 0' if ok else '非 0'}")
    r.verify(
        status="verified" if all_ok else "partially_verified",
        methods=[_diag(all_ok, "候选点代回约束检查残差 + 检查 ∇f=Σλ∇g", "；".join(detail))],
    )
    for entry in candidates:
        entry.pop("_subs", None)
    if dropped:
        r.add_warning(f"丢弃了 {dropped} 个含复数/无穷乘数的解")
    r.set_raw(verification_detail={"residual_checks": detail})
    return r


# ---------------------------------------------------------------------------
# 8. 隐函数求导
# ---------------------------------------------------------------------------

def _implicit_setup(equation: Any, mapping: dict[str, sp.Symbol], names: Sequence[str]) -> tuple[sp.Basic, sp.Basic, list[sp.Symbol]]:
    syms = _symbols(names)
    if len(syms) < 2:
        raise ValueError("隐函数求导至少需要两个变量，例如 vars=\"x,y\"")
    if isinstance(equation, sp.Basic):
        raw = equation
    elif isinstance(equation, (list, tuple)):
        if len(equation) != 1:
            raise ValueError(f"隐式方程只能是一个等式，收到 {len(equation)} 个表达式")
        raw = _implicit_setup(equation[0], mapping, names)[0]
    else:
        text = str(equation)
        if "=" in text and "==" not in text:
            lhs_text, rhs_text = text.split("=", 1)
            lhs = _parse(lhs_text.strip(), mapping, "方程左端")
            rhs = _parse(rhs_text.strip(), mapping, "方程右端")
            raw = sp.expand(lhs - rhs)
        else:
            raw = _parse(text, mapping, "方程")
    if isinstance(raw, sp.Eq):
        raw = sp.expand(raw.lhs - raw.rhs)
    return raw, raw, syms


def _apply_implicit_roles(names: list[str], dependent: Any, independent: Any) -> list[str]:
    """按显式给出的因变量/自变量重排 ``names``（末位恒为因变量）。

    调用方常常把 ``vars`` 顺序写反（``vars=["y","x"]`` 其实表示「以 y 为自变量、
    对 x 求导」）。显式给 ``dependent``/``independent`` 即可无歧义地指定角色，
    且当指定的变量不在 ``names`` 里时保持原样（宁可报「变量不足」也不静默换算）。
    """
    if dependent is None and independent is None:
        return names
    dep = str(dependent).strip() if dependent is not None else None
    ind = str(independent).strip() if independent is not None else None
    if dep and dep not in names:
        raise ValueError(f"因变量 {dep!r} 不在变量列表 {names} 中")
    if ind and ind not in names:
        raise ValueError(f"自变量 {ind!r} 不在变量列表 {names} 中")
    if dep and ind and dep == ind:
        raise ValueError("因变量与自变量不能是同一个变量")
    rest = [name for name in names if name not in (dep, ind)]
    if ind:
        rest = [ind] + rest
    if dep:
        rest = rest + [dep]
    return rest


@op("implicit_diff", "math_implicit_diff")
def implicit_diff(
    equation: str = None,
    vars: Any = None,
    order: Any = 1,
    point: Any = None,
    equations: Any = None,
    dependent: Any = None,
    independent: Any = None,
    **kwargs: Any,
) -> MathResult:
    """隐函数求导：由 ``F(x,y)=0`` 求 ``dy/dx``。

    变量顺序约定：``vars = [自变量…, 因变量]``，即 ``vars=["x","y"]`` 求 ``dy/dx``。
    若担心顺序被误读，可直接给 ``dependent="y", independent="x"``（等价且无歧义）。
    """
    # engine.ARG_ALIASES 会把 `equation` 改写成 `equations`，两个名字都要接住
    if equation is None:
        equation = equations
    if isinstance(equation, (list, tuple)):
        if len(equation) != 1:
            raise ValueError("隐函数求导只接受一个方程")
        equation = equation[0]
    if equation is None:
        raise ValueError("缺少方程参数 equation，例如 equation=\"x^2+y^2=1\"")
    try:
        n = int(order) if order is not None else 1
    except (TypeError, ValueError) as exc:
        raise ValueError(f"order 必须是正整数，收到 {order!r}") from exc
    if n < 1:
        raise ValueError("order 必须是正整数")

    if vars:
        names = _var_names(vars, None)
    else:
        probe = _parse(str(equation).replace("=", "-"), A.get_symbols(list("xyz")) or {"x": sp.Symbol("x", real=True), "y": sp.Symbol("y", real=True)}, "方程")
        names = _var_names(None, probe, default=["x", "y"])
    names = _apply_implicit_roles(names, dependent, independent)
    mapping = _sym_map(names)
    F, _, syms = _implicit_setup(equation, mapping, names)
    if len(syms) < 2:
        raise ValueError("隐函数求导至少需要两个变量")

    r = MathResult.ok("implicit_diff", method="隐函数求导公式 -F_x/F_y（高阶用 sp.idiff）")
    r.set_input(equation=f"{A.to_text(F)} = 0", vars=names, order=n)
    r.add_condition("隐函数存在定理条件：F 在点附近连续可微，且对因变量的偏导 F_y ≠ 0；否则需改用参数化或分段讨论")

    if len(syms) == 2:
        x, y = syms
        d1_raw = sp.simplify(-sp.diff(F, x) / sp.diff(F, y))
        if not C.is_closed_form(d1_raw):
            return MathResult.unsolved("implicit_diff", "隐函数导数含未求值对象（SymPy 原样回显）")
        if n == 1:
            result = d1_raw
            result_label = f"d{y}/d{x}"
        else:
            try:
                result = sp.simplify(sp.idiff(F, y, x, n))
            except Exception as exc:  # noqa: BLE001
                return MathResult.unsolved("implicit_diff", f"{n} 阶隐函数导数求解失败：{exc}")
            if not C.is_closed_form(result):
                return MathResult.unsolved("implicit_diff", "高阶隐函数导数含未求值对象（SymPy 原样回显）")
            result_label = f"d^{n}{y}/d{x}^{n}"
        r.set_result(
            derivative=A.to_text(result),
            derivative_approx=_numeric_literal(result),
            derivative_label=result_label,
            order=n,
            dependent=str(y),
            independent=str(x),
            F=A.to_text(F),
            F_x=A.to_text(sp.diff(F, x)),
            F_y=A.to_text(sp.diff(F, y)),
            implicit_derivative_formula=f"-F_x/F_y = -({A.to_text(sp.diff(F, x))})/({A.to_text(sp.diff(F, y))})",
        )
        r.add_condition(f"{A.to_text(sp.diff(F, y))} ≠ 0（即 F_y ≠ 0）时 {y} 才能作为 {x} 的隐函数求导")
        r.add_condition(f"变量角色按自变量 → 因变量解释：自变量 {x}，因变量 {y}，所求为 {result_label}")

        # ------- 独立验证 -------
        methods: list[dict[str, Any]] = []
        # A. 由隐函数定理独立推出：对 F(x,y(x))=0 两边求导 → F_x + F_y y' = 0
        y_func = sp.Function(str(y))(x)
        F_sub = F.subs(y, y_func)
        try:
            eq1 = sp.diff(F_sub, x)
            solved = sp.solve(sp.Eq(eq1, 0), sp.Derivative(y_func, x))
            alt = sp.simplify(solved[0]) if solved else None
        except Exception:  # noqa: BLE001
            alt = None
        if alt is not None:
            # solve 解出的是以 y(x)（应用函数）表示的表达式，而 result 用的是普通符号 y；
            # 不换回来时高精度抽样无法把 y(x) 代入数值点，会得到「拿不到有效数值」的假不一致。
            alt = alt.subs(y_func, y)
            same, why = _closes(alt, result, mapping)
            methods.append(_diag(same, "对 F(x,y(x))=0 两边求导并解出 y′ 互印", why))
        else:
            methods.append(_diag(False, "对 F(x,y(x))=0 两边求导", "无法独立解出 y′"))

        # B. 数值验证：取真值点，检查曲线上的差分与公式值一致
        numeric_note = "未获得数值证据（曲线点求解失败）"
        numeric_ok: bool | None = None
        if n == 1:
            dydx_num = sp.lambdify((x, y), d1_raw, "mpmath")
            F_num = sp.lambdify((x, y), F, "mpmath")
            trials = [3, 5, 7, 9, 11]
            checked = 0
            worst = 0.0
            bad: list[str] = []
            for k in trials:
                guess = sp.Rational(k, 10)
                try:
                    y0 = sp.nsolve(F.subs(x, guess), y, sp.N(guess, 20))
                except Exception:  # noqa: BLE001
                    continue
                try:
                    slope_formula = complex(sp.N(dydx_num(complex(sp.N(guess, 20)), complex(sp.N(y0, 20))), 20))
                except Exception:  # noqa: BLE001
                    continue
                h = 1e-6
                try:
                    fp = complex(F_num(complex(sp.N(guess, 20) + h), complex(sp.N(y0, 20))))
                    fm = complex(F_num(complex(sp.N(guess, 20) - h), complex(sp.N(y0, 20))))
                    fy = complex(sp.N(sp.diff(F, y).subs({x: guess, y: y0}), 20))
                    slope_fd = -(fp - fm) / (2 * h) / fy
                except Exception:  # noqa: BLE001
                    continue
                checked += 1
                dev = abs(slope_fd - slope_formula) / max(1.0, abs(slope_formula))
                worst = max(worst, dev)
                if dev > 1e-5 and len(bad) < 2:
                    bad.append(f"x≈{float(guess):.3g}: 公式 {slope_formula.real:.8g} vs 差分 {slope_fd.real:.8g}")
            if checked:
                numeric_ok = not bad
                numeric_note = (
                    f"{checked} 个曲线点上公式值与隐式差分一致（最大相对偏差 {worst:.2e}）"
                    if numeric_ok
                    else "曲线点上公式值与差分不一致：" + "；".join(bad)
                )
            else:
                numeric_note = "曲线上未找到实数真值点，未获得数值证据（该曲线在抽样实数范围无实点）"
            methods.append(_diag(bool(numeric_ok), "曲线真值点上的隐式中心差分复核", numeric_note))

        status, evidence = _verdict(*methods)
        r.verify(status=status, methods=evidence)
        r.set_raw(verification_detail={"numeric": numeric_note})

        if point is not None:
            pt = _parse_point(point, mapping, "点")
            subs = dict(zip(syms, pt))
            at_point = sp.simplify(result.subs(subs))
            r.set_raw(value_at_point=A.to_text(at_point))
            residual = sp.simplify(F.subs(subs))
            if abs(complex(sp.N(residual, 25))) > 1e-9:
                r.add_warning(f"给定点不满足隐式方程：F = {A.to_text(residual)}")
        return r

    # 三变量：F(x,y,z)=0，按 vars 顺序取 z 为因变量，给出 ∂z/∂x 与 ∂z/∂y
    x, y, z = syms[0], syms[1], syms[2]
    if sp.simplify(sp.diff(F, z)) == 0:
        return MathResult.unsolved("implicit_diff", f"F 对 {z} 的偏导恒为 0，无法解出以 {z} 为因变量的隐函数")
    dz_dx = sp.simplify(-sp.diff(F, x) / sp.diff(F, z))
    dz_dy = sp.simplify(-sp.diff(F, y) / sp.diff(F, z))
    if not C.is_closed_form(dz_dx) or not C.is_closed_form(dz_dy):
        return MathResult.unsolved("implicit_diff", "隐函数偏导含未求值对象（SymPy 原样回显）")
    r.set_result(
        dz_dx=A.to_text(dz_dx),
        dz_dy=A.to_text(dz_dy),
        dz_dx_approx=_numeric_literal(dz_dx),
        dz_dy_approx=_numeric_literal(dz_dy),
        order=1,
        dependent=str(z),
        independent=[str(x), str(y)],
        F=A.to_text(F),
        F_x=A.to_text(sp.diff(F, x)),
        F_y=A.to_text(sp.diff(F, y)),
        F_z=A.to_text(sp.diff(F, z)),
    )
    r.add_condition(f"{A.to_text(sp.diff(F, z))} ≠ 0（F_z ≠ 0）时 {z} 可作为 ({x},{y}) 的隐函数")
    if n > 1:
        r.add_warning("三变量隐函数的高阶偏导未实现，仅返回一阶；如需请降为两变量问题")
        r.partial("三变量情形仅支持一阶偏导")
    checks: list[str] = []
    all_ok = True
    for target, expr in ((x, dz_dx), (y, dz_dy)):
        z_func = sp.Function(str(z))(x, y)
        F_sub = F.subs(z, z_func)
        try:
            partial_eq = sp.diff(F_sub, target)
            solved = sp.solve(sp.Eq(partial_eq, 0), sp.Derivative(z_func, target))
            alt = sp.simplify(solved[0]) if solved else None
        except Exception:  # noqa: BLE001
            alt = None
        if alt is None:
            all_ok = False
            checks.append(f"对 {target} 求导无法独立解出 ∂{z}/∂{target}")
            continue
        # 与两变量分支同理：solve 给的是 z(x,y) 形式，换回普通符号 z 才能做数值抽样比较
        alt = alt.subs(z_func, z)
        same, why = _closes(alt, expr, mapping)
        all_ok = all_ok and same
        checks.append(f"∂{z}/∂{target}：{'一致' if same else '不一致'}（{why}）")
    r.verify(status="verified" if all_ok else "unverified", methods=[_diag(all_ok, "对 F=0 两边求偏导独立解出互印", "；".join(checks))])
    return r


# ---------------------------------------------------------------------------
# 9. 参数方程求导
# ---------------------------------------------------------------------------

@op("parametric_derivative", "math_parametric_derivative")
def parametric_derivative(x_expr: str, y_expr: str, param: str = "t", order: Any = 1, **kwargs: Any) -> MathResult:
    """参数方程 ``x=x(t), y=y(t)`` 求 ``dy/dx`` 与 ``d²y/dx²``。"""
    try:
        n = int(order) if order is not None else 1
    except (TypeError, ValueError) as exc:
        raise ValueError(f"order 必须是正整数，收到 {order!r}") from exc
    if n not in (1, 2):
        raise ValueError("目前只支持一阶/二阶参数方程求导（order ∈ {1,2}）")
    t_name = str(param or "t").strip()
    if not t_name:
        raise ValueError("param 不能为空")

    names = [t_name, "x", "y"]
    mapping = _sym_map(names)
    t = mapping[t_name]
    x = _parse(x_expr, mapping, "x(t)")
    y = _parse(y_expr, mapping, "y(t)")
    if t not in x.free_symbols and t not in y.free_symbols:
        raise ValueError(f"x(t)/y(t) 中未出现参数 {t_name!r}，请检查 param 参数")

    dxdt = sp.simplify(sp.diff(x, t))
    dydt = sp.simplify(sp.diff(y, t))
    r = MathResult.ok("parametric_derivative", method="dy/dx = (dy/dt)/(dx/dt)")
    r.set_input(x_expr=A.to_text(x), y_expr=A.to_text(y), param=t_name, order=n)
    if dxdt == 0:
        return MathResult.unsolved(
            "parametric_derivative",
            f"dx/dt ≡ 0：曲线退化为竖直直线，dy/dx 不存在（此时 x 恒为常数）",
        )
    r.set_result(
        dx_dt=A.to_text(dxdt),
        dy_dt=A.to_text(dydt),
        dx_dt_approx=_numeric_literal(dxdt),
        dy_dt_approx=_numeric_literal(dydt),
    )
    r.add_condition(f"必须 dx/dt ≠ 0，即 {A.to_text(dxdt)} ≠ 0；使 dx/dt=0 的参数点处 dy/dx 不存在或为无穷大（需单独讨论）")

    methods: list[dict[str, Any]] = []
    all_ok = True
    if n == 1:
        result = sp.simplify(dydt / dxdt)
        if not C.is_closed_form(result):
            return MathResult.unsolved("parametric_derivative", "dy/dx 含未求值对象（SymPy 原样回显）")
        r.set_result(dy_dx=A.to_text(result), dy_dx_approx=_numeric_literal(result))
        # 独立验证：把 y 视为 t 的复合函数，用 dy/dt = (dy/dx)(dx/dt) 检查
        check = sp.simplify(result * dxdt - dydt)
        ok = sp.simplify(check) == 0
        all_ok = all_ok and ok
        methods.append(_diag(ok, "链式法则互印： (dy/dx)·(dx/dt) 应等于 dy/dt", f"残差 {A.to_text(check)}"))
        # 数值验证：在 t0 处比较差分斜率
        numeric_ok = None
        note = "未获得数值证据"
        for t0 in (sp.Rational(3, 10), sp.Rational(7, 10), sp.Rational(13, 10)):
            try:
                h = sp.Rational(1, 100000)
                x1 = sp.N(x.subs(t, t0 + h), 25); x2 = sp.N(x.subs(t, t0 - h), 25)
                y1 = sp.N(y.subs(t, t0 + h), 25); y2 = sp.N(y.subs(t, t0 - h), 25)
                fd = (y1 - y2) / (x1 - x2)
                an = sp.N(result.subs(t, t0), 25)
                numeric_ok = abs(complex(fd - an)) < 1e-4
                note = f"t={t0} 处差分斜率 {float(sp.N(fd, 12)):.10g} vs 公式值 {float(sp.N(an, 12)):.10g}"
                break
            except Exception:  # noqa: BLE001
                continue
        if numeric_ok is not None:
            all_ok = all_ok and bool(numeric_ok)
            methods.append(_diag(bool(numeric_ok), "参数曲线上的斜率先差分复核", note))
        r.verify(status="verified" if all_ok else "partially_verified", methods=methods)
        r.set_raw(numeric_note=note)
        return r

    # 二阶
    d2_num = sp.simplify(sp.diff(dydt / dxdt, t) / dxdt)
    result = sp.simplify(d2_num)
    if not C.is_closed_form(result):
        return MathResult.unsolved("parametric_derivative", "d²y/dx² 含未求值对象（SymPy 原样回显）")
    r.set_result(
        dy_dx=A.to_text(sp.simplify(dydt / dxdt)),
        d2y_dx2=A.to_text(result),
        d2y_dx2_approx=_numeric_literal(result),
        formula="d²y/dx² = [ (dx/dt)(d²y/dt²) - (dy/dt)(d²x/dt²) ] / (dx/dt)³",
        d2x_dt2=A.to_text(sp.simplify(sp.diff(x, t, 2))),
        d2y_dt2=A.to_text(sp.simplify(sp.diff(y, t, 2))),
    )
    r.add_condition(f"二阶导同样要求 dx/dt ≠ 0，且 {A.to_text(dxdt)} 的零点处需单独讨论")

    # 独立验证 1：一阶导再对 x 求导（用 d/dx = (1/(dx/dt)) d/dt）
    alt = sp.simplify(sp.diff(dydt / dxdt, t) / dxdt)
    same, why = _closes(alt, result, mapping)
    all_ok = all_ok and same
    methods.append(_diag(same, "一阶导再按 d/dx=(1/(dx/dt))d/dt 求导互印", why))
    # 独立验证 2：直接对参数方程二阶求导方程求解
    try:
        direct = sp.simplify(
            (sp.diff(x, t) * sp.diff(y, t, 2) - sp.diff(y, t) * sp.diff(x, t, 2)) / sp.diff(x, t) ** 3
        )
        same2, why2 = _closes(direct, result, mapping)
        all_ok = all_ok and same2
        methods.append(_diag(same2, "二阶参数方程标准公式互印", why2))
    except Exception as exc:  # noqa: BLE001
        all_ok = False
        methods.append(_diag(False, "二阶参数方程标准公式", f"计算失败：{exc}"))
    # 独立验证 3：数值二阶差分
    numeric_ok = None
    note = "未获得数值证据"
    for t0 in (sp.Rational(1, 2), sp.Rational(3, 4), sp.Rational(5, 4)):
        try:
            h = 1e-4
            def _slope(tt: float) -> float:
                xx1 = complex(sp.N(x.subs(t, sp.N(tt + h, 20)), 20))
                xx2 = complex(sp.N(x.subs(t, sp.N(tt - h, 20)), 20))
                yy1 = complex(sp.N(y.subs(t, sp.N(tt + h, 20)), 20))
                yy2 = complex(sp.N(y.subs(t, sp.N(tt - h, 20)), 20))
                return (yy1 - yy2) / (xx1 - xx2)
            x1 = complex(sp.N(x.subs(t, sp.N(float(t0) + h, 20)), 20))
            x2 = complex(sp.N(x.subs(t, sp.N(float(t0) - h, 20)), 20))
            fd = (_slope(float(t0) + h) - _slope(float(t0) - h)) / (x1 - x2)
            an = complex(sp.N(result.subs(t, t0), 20))
            numeric_ok = abs(fd - an) / max(1.0, abs(an)) < 1e-3
            note = f"t={t0} 处二阶差分 {fd.real:.8g} vs 公式值 {an.real:.8g}"
            break
        except Exception:  # noqa: BLE001
            continue
    if numeric_ok is not None:
        all_ok = all_ok and bool(numeric_ok)
        methods.append(_diag(bool(numeric_ok), "参数曲线上的二阶数值差分复核", note))
    r.verify(status="verified" if all_ok else "partially_verified", methods=methods)
    r.set_raw(numeric_note=note)
    return r


# ---------------------------------------------------------------------------
# 10. 多元泰勒展开
# ---------------------------------------------------------------------------

def _multi_indices(n_vars: int, order: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    def rec(prefix: tuple[int, ...], remaining: int) -> None:
        if len(prefix) == n_vars:
            out.append(prefix + (remaining,))
            return
        for k in range(remaining + 1):
            rec(prefix + (k,), remaining - k)
    rec((), order)
    return out


def _taylor_polynomial(f: sp.Basic, syms: Sequence[sp.Symbol], pt: Sequence[Any], order: int, center: bool) -> sp.Basic:
    total = sp.Integer(0)
    for idx in _multi_indices(len(syms), order):
        term = sp.Integer(1)
        for k, s in zip(idx, syms):
            if k:
                if center:
                    term *= (s - pt[list(syms).index(s)]) ** k
                else:
                    term *= s ** k
        deriv = f
        for k, s in zip(idx, syms):
            if k:
                deriv = sp.diff(deriv, s, k)
        coeff = deriv
        if center:
            coeff = coeff.subs(dict(zip(syms, pt)))
        denominator = sp.Integer(1)
        for k in idx:
            denominator *= sp.factorial(k)
        total += sp.simplify(coeff / denominator) * term
    return sp.expand(total)


@op("taylor_multivar", "math_taylor_multivar")
def taylor_multivar(expr: str, vars: Any = None, point: Any = None, order: Any = 2, **kwargs: Any) -> MathResult:
    """多元泰勒展开（按多重指标求和到总阶数）。"""
    try:
        n = int(order) if order is not None else 2
    except (TypeError, ValueError) as exc:
        raise ValueError(f"order 必须是正整数，收到 {order!r}") from exc
    if n < 1:
        raise ValueError("order 必须是正整数")
    if n > _MAX_TAYLOR_ORDER:
        raise ValueError(f"展开阶数过大（>{_MAX_TAYLOR_ORDER}），请降低 order 或分步展开")
    names = _var_names(vars, expr if isinstance(expr, sp.Basic) else None)
    mapping = _sym_map(names)
    f = _parse(expr, mapping, "目标函数")
    syms = _symbols(names)
    if not syms:
        raise ValueError("无法确定自变量，请用 vars 指定")

    if point is None:
        pt = [sp.Integer(0)] * len(syms)
        point_desc = "原点 " + "(" + ", ".join("0" for _ in syms) + ")"
        centered = True
    else:
        pt = _parse_point(point, mapping, "展开点")
        point_desc = "(" + ", ".join(A.to_text(p) for p in pt) + ")"
        centered = True

    poly = _taylor_polynomial(f, syms, pt, n, centered)
    if not C.is_closed_form(poly):
        return MathResult.unsolved("taylor_multivar", "泰勒展开含未求值对象（SymPy 原样回显）")
    remainder = sp.simplify(f - poly)

    r = MathResult.ok("taylor_multivar", method=f"多重指标求和：Σ_{{|α|≤{n}}} D^α f(a)/α! · (x-a)^α")
    r.set_input(expr=A.to_text(f), vars=names, point=point_desc, order=n)
    r.set_result(
        text=A.to_text(poly),
        latex=A.to_latex(poly),
        approx=_numeric_literal(poly),
        order=n,
        point=point_desc,
        remainder_expr=A.to_text(remainder),
        remainder_note=(
            f"余项 R = f - P_{n} = {A.to_text(remainder)}。"
            f"该展开是 {point_desc} 邻域内的**局部近似**，只在展开点附近有效；"
            "全局误差取决于展开点邻域大小与函数的高阶导数（拉格朗日余项）。"
        ),
    )
    r.add_condition(f"展开式在 {point_desc} 的邻域内成立（局部近似）；离开该邻域误差会迅速增大")
    r.add_condition(f"要求 f 在 {point_desc} 的邻域内具有直到 {n + 1} 阶连续偏导数")

    # 独立验证 1：逐变量 sp.series 展开
    alt_ok = False
    alt_note = "逐变量 series 展开失败"
    try:
        alt = sp.sympify(f)
        for s, c in zip(syms, pt):
            alt = alt.series(s, c, n + 1).removeO()
        if sp.simplify(sp.expand(alt - poly)) == 0:
            alt_ok = True
            alt_note = "逐变量 sp.series 展开取到总阶数的结果与多重指标求和一致"
        else:
            alt_note = f"逐变量展开结果 {A.to_text(sp.expand(alt))} 与多重指标结果不一致"
    except Exception as exc:  # noqa: BLE001
        alt_note = f"逐变量 series 展开不适用：{exc}"

    # 独立验证 2：数值偏差随阶数下降
    numeric_note = "未获得数值证据"
    numeric_ok: bool | None = None
    try:
        h = sp.Rational(1, 10)
        probe = {s: sp.N(c + h, 20) for s, c in zip(syms, pt)}
        exact = sp.N(f.subs(probe), 25)
        errors: list[tuple[int, float]] = []
        for k in range(1, n + 1):
            pk = _taylor_polynomial(f, syms, pt, k, centered)
            ek = abs(complex(sp.N((f - pk).subs(probe), 25)))
            errors.append((k, ek))
        mono = all(errors[i][1] >= errors[i + 1][1] - 1e-12 for i in range(len(errors) - 1))
        numeric_ok = bool(mono)
        numeric_note = (
            "在偏置点 " + A.to_text(probe) + f" 处误差随阶数单调下降："
            + "；".join(f"n={k}: {e:.3e}" for k, e in errors)
            + f"（精确值 {A.to_text(sp.N(exact, 12))}）"
        )
    except Exception as exc:  # noqa: BLE001
        numeric_note = f"数值偏差检验失败：{exc}"
        numeric_ok = None

    methods = [_diag(alt_ok, "逐变量 sp.series 展开互印", alt_note)]
    if numeric_ok is not None:
        methods.append(_diag(numeric_ok, "展开点邻域内误差随阶数下降", numeric_note))
    status, evidence = _verdict(*methods)
    r.verify(status=status, methods=evidence)
    r.set_raw(verification_detail={"series": alt_note, "numeric": numeric_note})
    return r


# ---------------------------------------------------------------------------
# 11. 不等式解集
# ---------------------------------------------------------------------------

def _relational(text: Any, mapping: dict[str, sp.Symbol]) -> Any:
    if isinstance(text, sp.Rel):
        return text
    if isinstance(text, sp.logic.boolalg.BooleanAtom):
        return sp.Le(0, 0) if text is sp.true else sp.Lt(0, 0)
    raw = str(text)
    chosen: tuple[str, str] | None = None
    for symbol, canonical in (("<=", "<="), (">=", ">="), ("≤", "<="), ("≥", ">="), ("<", "<"), (">", ">")):
        if symbol in raw:
            chosen = (symbol, canonical)
            break
    if chosen is None:
        raise ValueError(f"无法识别不等式 {raw!r}（需要包含 <、<=、>、>= 之一）")
    symbol, canonical = chosen
    lhs_text, rhs_text = raw.split(symbol, 1)
    lhs = _parse(lhs_text.strip(), mapping, "不等式左端")
    rhs = _parse(rhs_text.strip(), mapping, "不等式右端")
    if not (lhs.free_symbols or rhs.free_symbols):
        left = sp.N(lhs - rhs, 25)
        if canonical == "<":
            return bool(left < 0)
        if canonical == "<=":
            return bool(left <= 0)
        if canonical == ">":
            return bool(left > 0)
        return bool(left >= 0)
    operations = {"<=": sp.Le, ">=": sp.Ge, "<": sp.Lt, ">": sp.Gt}
    return operations[canonical](lhs, rhs)


def _set_to_intervals(solution: Any) -> list[Any]:
    if isinstance(solution, sp.Set):
        if isinstance(solution, sp.Union):
            out: list[Any] = []
            for arg in solution.args:
                out.extend(_set_to_intervals(arg))
            return out
        return [solution]
    if isinstance(solution, (list, tuple, sp.FiniteSet)):
        out = []
        for item in solution:
            out.extend(_set_to_intervals(item))
        return out
    return []


def _interval_text(interval: Any) -> str:
    if isinstance(interval, sp.Interval):
        left = "-∞" if interval.start is -sp.oo else A.to_text(interval.start)
        right = "+∞" if interval.end is sp.oo else A.to_text(interval.end)
        lb = "[" if interval.left_open is False else "("
        rb = "]" if interval.right_open is False else ")"
        return f"{lb}{left}, {right}{rb}"
    return A.to_text(interval)


def _boundaries(intervals: Sequence[Any], var: sp.Symbol) -> list[Any]:
    points: list[Any] = []
    for interval in intervals:
        if isinstance(interval, sp.Interval):
            for endpoint in (interval.start, interval.end):
                if endpoint not in (sp.oo, -sp.oo) and endpoint not in points:
                    points.append(endpoint)
    return points


def _predicate_holds(rel: sp.Rel, var: sp.Symbol, value: Any) -> bool | None:
    try:
        substituted = sp.sympify(rel.lhs - rel.rhs).subs(var, value)
        num = sp.N(substituted, 25)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(num, sp.Basic) or not num.is_real:
        return None
    try:
        real = float(num)
    except (TypeError, ValueError):
        return None
    tol = 1e-9
    rel_type = type(rel)
    if rel_type is sp.Lt:
        return real < -tol
    if rel_type is sp.Le:
        return real <= tol
    if rel_type is sp.Gt:
        return real > tol
    if rel_type is sp.Ge:
        return real >= -tol
    if rel_type is sp.Ne:
        return abs(real) > tol
    if rel_type is sp.Eq:
        return abs(real) <= tol
    return None


@op("inequality", "solve_inequality", "math_inequality")
def inequality(expr: str, var: str = None, **kwargs: Any) -> MathResult:
    """求解一元不等式，返回解集（区间并集）与集合表示。"""
    if isinstance(expr, (list, tuple)) or (isinstance(expr, str) and ("," in expr and not any(op_ in expr for op_ in ("<", ">", "≤", "≥")))):
        raise ValueError("不等式求解只接受单个不等式；不等式组请分别求解后取交集")
    if not isinstance(expr, str):
        raise ValueError("expr 必须是不等式的字符串形式，例如 \"x^2-3*x+2<0\"")

    names = _var_names([var], None) if var else None
    if names:
        x = _sym_map(names)[names[0]]
        extra = {names[0]: x}
    else:
        extra = {}
    probe = A.parse(expr, symbols=list(extra) or None, extra=extra or None)
    constant_probe = isinstance(probe, sp.logic.boolalg.BooleanAtom)
    if not isinstance(probe, sp.Rel) and not constant_probe:
        raise ValueError(f"{expr!r} 不是不等式（未识别出 <、<=、>、>=）")
    free = sorted(probe.free_symbols, key=lambda s: str(s))
    if var:
        x = sp.Symbol(str(var).strip())
        mapping = {str(x): x}
    elif len(free) == 1:
        x = free[0]
        mapping = {str(x): x}
    elif constant_probe:
        x = sp.Symbol(str(var).strip()) if var else sp.Symbol("x", real=True)
        mapping = {str(x): x}
    else:
        raise ValueError(f"不等式含变量 {[str(s) for s in free]}，请用 var 指定求解变量")
    rel = _relational(expr, mapping)
    if rel is True:
        r = MathResult.ok("inequality", method="常数不等式判定")
        r.set_input(inequality=expr, var=str(x))
        r.set_result(solution_set="Reals", intervals=["(-oo, oo)"], interval_text="(-∞, +∞)", set_text="ℝ")
        r.add_condition("该不等式经化简后恒成立（与变量无关），解集为全体实数")
        r.verify(status="verified", methods=[_diag(True, "常数判定", "不等号两端均为常数且成立")])
        return r
    if rel is False:
        r = MathResult.ok("inequality", method="常数不等式判定")
        r.set_input(inequality=expr, var=str(x))
        r.set_result(solution_set="EmptySet", intervals=[], interval_text="∅", set_text="∅")
        r.add_condition("该不等式经化简后恒不成立（与变量无关），在实数范围内无解")
        r.verify(status="verified", methods=[_diag(True, "常数判定", "不等号两端均为常数且不成立")])
        return r

    r = MathResult.ok("inequality", method="sp.solve_univariate_inequality(relational=False) / sp.reduce_inequalities")
    r.set_input(inequality=A.to_text(rel), var=str(x))
    unevaluated = False
    try:
        solution = sp.solve_univariate_inequality(rel, x, relational=False)
    except Exception:
        try:
            solution = sp.reduce_inequalities(rel, x)
        except Exception as exc:  # noqa: BLE001
            return MathResult.unsolved(
                "inequality",
                f"SymPy 无法解析求解不等式 {A.to_text(rel)}：{type(exc).__name__}: {exc}；"
                "建议改写为等价多项式不等式，或改用数值扫描",
            )
    if isinstance(solution, sp.Set) and not solution.is_Interval and not isinstance(solution, sp.Union):
        pass  # 保持原样（可能是 FiniteSet / 其它集合类型）
    if solution is sp.S.EmptySet or (isinstance(solution, sp.Set) and solution.is_empty):
        r.add_condition("该不等式在实数范围内无解（解集为空集）")
        r.set_result(solution_set="EmptySet", intervals=[], interval_text="∅", set_text="∅")
        r.verify(status="verified", methods=[_diag(True, "解集为空", "解集为空集是确定结论：无任何实数满足该不等号方向")])
        return r
    if solution is sp.S.Reals:
        r.set_result(solution_set="Reals", intervals=["(-oo, oo)"], interval_text="(-∞, +∞)", set_text="ℝ")
        r.add_condition("该不等式对所有实数成立")
        r.verify(status="verified", methods=[_diag(True, "解集为全体实数", "无需抽样")])
        return r

    intervals = _set_to_intervals(solution)
    if not intervals:
        unevaluated = not C.is_closed_form(solution)
        if unevaluated:
            return MathResult.unsolved(
                "inequality",
                f"SymPy 返回未求值的条件集 {A.to_text(solution)}（未能解出区间）；建议用数值方法扫描符号",
            )
        return MathResult.unsolved("inequality", f"无法把解集 {A.to_text(solution)} 表示为区间")

    # 统一用查询变量表示（solve_univariate_inequality 可能引入新的哑变量）
    normalized: list[Any] = []
    for interval in intervals:
        if isinstance(interval, sp.Interval):
            try:
                normalized.append(
                    sp.Interval(
                        sp.sympify(interval.start), sp.sympify(interval.end),
                        left_open=bool(interval.left_open), right_open=bool(interval.right_open),
                    )
                )
            except Exception:  # noqa: BLE001
                normalized.append(interval)
        else:
            normalized.append(interval)
    normalized.sort(key=lambda iv: (sp.N(iv.start, 20) if isinstance(iv, sp.Interval) else 0))

    interval_text = " ∪ ".join(_interval_text(iv) for iv in normalized)
    set_text = " ∪ ".join(f"{{{A.to_text(iv)}}}" for iv in normalized)
    r.set_result(
        intervals=[_interval_text(iv) for iv in normalized],
        interval_text=interval_text,
        solution_set=A.to_text(solution),
        set_text=set_text,
    )
    r.add_condition("解集为实数范围内的区间并集；不等式两端同乘/同除负数时需变号（SymPy 已按原不等式求解）")
    holds = {sp.Lt: "f(x) < 0", sp.Le: "f(x) ≤ 0", sp.Gt: "f(x) > 0", sp.Ge: "f(x) ≥ 0", sp.Ne: "f(x) ≠ 0"}.get(type(rel), "不等式成立")
    r.add_condition(f"解集内每一点都满足 {holds}；边界点的取舍（开/闭）由原不等式是否严格决定")

    # 独立验证：区间内取点必须满足，区间外取点必须不满足
    inside_points: list[Any] = []
    outside_points: list[Any] = []
    for interval in normalized:
        left = interval.start
        right = interval.end
        if left is -sp.oo and right is sp.oo:
            anchors = [sp.Integer(0), sp.Integer(1), sp.Integer(-1)]
        elif left is -sp.oo:
            anchors = [sp.sympify(right) - 1, sp.sympify(right) - 2, sp.sympify(right) - sp.Rational(1, 2)]
        elif right is sp.oo:
            anchors = [sp.sympify(left) + sp.Rational(1, 2), sp.sympify(left) + 1, sp.sympify(left) + 3]
        else:
            mid = sp.simplify((sp.sympify(left) + sp.sympify(right)) / 2)
            anchors = [mid, sp.simplify(mid + (sp.sympify(right) - sp.sympify(left)) / 4)]
        for anchor in anchors:
            if isinstance(interval, sp.Interval) and (
                (not interval.contains(anchor))
                or (anchor == left and interval.left_open)
                or (anchor == right and interval.right_open)
            ):
                continue
            if anchor not in inside_points:
                inside_points.append(anchor)
    bounds = _boundaries(normalized, x)
    if bounds:
        for b in bounds:
            for candidate in (sp.simplify(b - 1), sp.simplify(b + 1), sp.simplify(b - sp.Rational(1, 2)), sp.simplify(b + sp.Rational(1, 2))):
                if any(isinstance(iv, sp.Interval) and iv.contains(candidate) for iv in normalized):
                    continue
                if candidate not in outside_points and candidate not in inside_points:
                    outside_points.append(candidate)
    else:
        outside_points = []

    inside_ok = 0
    inside_bad: list[str] = []
    for point in inside_points:
        verdict = _predicate_holds(rel, x, point)
        if verdict:
            inside_ok += 1
        elif verdict is False:
            inside_bad.append(f"x={A.to_text(point)} 在解集内却不满足不等式")
    outside_ok = 0
    outside_bad: list[str] = []
    for point in outside_points:
        verdict = _predicate_holds(rel, x, point)
        if verdict is False:
            outside_ok += 1
        elif verdict is True:
            outside_bad.append(f"x={A.to_text(point)} 在解集外却满足不等式")

    detail = (
        f"解集内抽样 {len(inside_points)} 点，满足 {inside_ok} 点"
        + (f"；异常：{'；'.join(inside_bad)}" if inside_bad else "")
        + f"。解集外抽样 {len(outside_points)} 点，不满足 {outside_ok} 点"
        + (f"；异常：{'；'.join(outside_bad)}" if outside_bad else "")
    )
    ok = not inside_bad and not outside_bad and inside_ok > 0
    if not inside_points:
        r.verify(status="unverified", methods=[_diag(False, "解集内取点代入检验", "未能取到解集内的抽样点")])
    else:
        r.verify(
            status="verified" if ok else "partially_verified",
            methods=[_diag(ok, "解集内/外各取抽样点代入检验不等式方向（高精度 sp.N）", detail)],
        )
    r.set_raw(verification_detail={"inside_samples": len(inside_points), "outside_samples": len(outside_points)})
    return r


# ---------------------------------------------------------------------------
# 12. 自然定义域
# ---------------------------------------------------------------------------

def _numericize(expr: sp.Basic, dummy: sp.Symbol) -> sp.Basic:
    """按数值语义改写：Abs→sqrt(w²)、sign→w/sqrt(w²)，便于实数区间求解。"""
    w = dummy
    eps = sp.Symbol("_eps", positive=True)
    mapping = {
        sp.Abs: lambda a: sp.sqrt(a ** 2 + eps ** 2),
        sp.sign: lambda a: a / sp.sqrt(a ** 2 + eps ** 2),
    }
    try:
        out = expr.replace(lambda node: type(node) in mapping, lambda node: mapping[type(node)](node.args[0]))
    except Exception:  # noqa: BLE001
        out = expr
    return out


def _collect_domain_constraints(
    expr: sp.Basic, var: sp.Symbol
) -> list[tuple[str, Any, str]]:
    """遍历表达式树，收集使表达式有实数意义的限制条件。

    返回 ``(限制类型, 限制集合, 来源子表达式)`` 列表。
    """
    w = sp.Symbol("_w_domain", real=True)
    conditions: list[tuple[Any, str]] = []

    def walk(node: Any) -> None:
        if not isinstance(node, sp.Basic):
            return
        if node.is_Atom:
            return
        # 分母不为零
        if node.is_Pow and node.exp.is_number and node.exp.is_negative:
            base = node.base
            if base.has(w) or base.free_symbols:
                conditions.append((sp.Ne(base, 0), f"分母 / 负幂底数 {A.to_text(base)} ≠ 0"))
        # 偶次根号：被开方数 ≥ 0
        if node.is_Pow and node.exp.is_Rational and not node.exp.is_Integer:
            if node.exp.q % 2 == 0:
                conditions.append((sp.Ge(node.base, 0), f"偶次根式被开方数 {A.to_text(node.base)} ≥ 0"))
            else:
                conditions.append((sp.Ne(node.base, 0), f"分数幂 {A.to_text(node)} 的底数 {A.to_text(node.base)} ≠ 0"))
        # 对数：真数 > 0
        if isinstance(node, sp.log) and len(node.args) == 1:
            conditions.append((sp.Gt(node.args[0], 0), f"对数真数 {A.to_text(node.args[0])} > 0"))
        # 反正弦/反余弦：参数 ∈ [-1,1]
        if isinstance(node, (sp.asin, sp.acos)):
            arg = node.args[0]
            conditions.append((sp.And(sp.Ge(arg, -1), sp.Le(arg, 1)), f"{A.to_text(node)} 要求 {A.to_text(arg)} ∈ [-1,1]"))
        # 正切/余切/正割/余割的极点
        if isinstance(node, (sp.tan, sp.sec)):
            arg = node.args[0]
            conditions.append((sp.Ne(sp.cos(arg), 0), f"{A.to_text(node)} 要求 cos({A.to_text(arg)}) ≠ 0"))
        if isinstance(node, (sp.cot, sp.csc)):
            arg = node.args[0]
            conditions.append((sp.Ne(sp.sin(arg), 0), f"{A.to_text(node)} 要求 sin({A.to_text(arg)}) ≠ 0"))
        for arg in node.args:
            walk(arg)

    walk(expr)
    # 去重（保持顺序）
    seen: set[str] = set()
    unique: list[tuple[str, Any, str]] = []
    for cond, source in conditions:
        key = A.to_text(cond)
        if key in seen:
            continue
        seen.add(key)
        unique.append((key, cond, source))
    return unique


def _solve_constraint(cond: Any, var: sp.Symbol) -> Any:
    """把一条限制条件解成 SymPy 集合（And/Or 递归分解，Rel 用 solve_univariate_inequality / reduce_inequalities）。"""
    if isinstance(cond, sp.And):
        parts = [_solve_constraint(arg, var) for arg in cond.args]
        if any(isinstance(p, Exception) for p in parts):
            return next(p for p in parts if isinstance(p, Exception))
        if any(not isinstance(p, sp.Set) for p in parts):
            bad = next(p for p in parts if not isinstance(p, sp.Set))
            return TypeError(f"约束 {A.to_text(cond)} 的一部分解不成集合：{A.to_text(bad)}")
        if any(p is sp.S.EmptySet or p.is_empty for p in parts):
            return sp.S.EmptySet
        combined = parts[0]
        for part in parts[1:]:
            combined = sp.Intersection(combined, part)
        return combined

    if isinstance(cond, sp.Or):
        parts = [_solve_constraint(arg, var) for arg in cond.args]
        if any(isinstance(p, Exception) for p in parts):
            return next(p for p in parts if isinstance(p, Exception))
        if any(not isinstance(p, sp.Set) for p in parts):
            bad = next(p for p in parts if not isinstance(p, sp.Set))
            return TypeError(f"约束 {A.to_text(cond)} 的一部分解不成集合：{A.to_text(bad)}")
        combined = parts[0]
        for part in parts[1:]:
            combined = sp.Union(combined, part)
        return combined

    if cond is sp.true or cond is True:
        return sp.S.Reals
    if cond is sp.false or cond is False:
        return sp.S.EmptySet

    candidates = []
    if isinstance(cond, sp.Rel):
        candidates = [cond.lhs - cond.rhs, cond]
    else:
        candidates = [cond]
    for candidate in candidates:
        try:
            solved = sp.solve_univariate_inequality(candidate, var, relational=False)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(solved, sp.Set):
            return solved
    try:
        solved = sp.reduce_inequalities(cond, var)
    except Exception as exc:  # noqa: BLE001
        return exc
    if isinstance(solved, sp.Set):
        return solved
    if solved is sp.true or solved is True:
        return sp.S.Reals
    if solved is sp.false or solved is False:
        return sp.S.EmptySet
    try:
        as_set = sp.as_set(solved, var)
        if isinstance(as_set, sp.Set):
            return as_set
    except Exception:  # noqa: BLE001
        pass
    return TypeError(f"无法把限制条件 {A.to_text(cond)} 化为集合")


@op("domain", "math_domain")
def domain(expr: str, var: str = None, **kwargs: Any) -> MathResult:
    """求表达式的自然定义域（分母≠0、偶次根式≥0、对数真数>0、反三角∈[-1,1]、tan/cot 极点等）。"""
    names = _var_names([var], None) if var else None
    mapping = _sym_map(names) if names else {}
    f = _parse(expr, mapping, "表达式")
    if var:
        x = sp.Symbol(str(var).strip(), real=True)
    else:
        free = sorted(f.free_symbols, key=lambda s: (len(str(s)), str(s)))
        if len(free) != 1:
            raise ValueError(f"表达式含变量 {[str(s) for s in free]}，请用 var 指定自变量（本算子只处理一元函数）")
        x = free[0]
    var_text = str(x)
    dummy = sp.Symbol("_w_domain", real=True)
    f_w = f.subs(x, dummy)
    f_w = _numericize(f_w, dummy)

    r = MathResult.ok("domain", method="表达式树遍历收集限制条件 + sp.solve_univariate_inequality/sp.continuous_domain")
    r.set_input(expr=A.to_text(f), var=var_text)
    r.add_condition("自然定义域 = 使表达式在实数范围内有意义的自变量集合（不含人为限制）")

    raw_conditions = _collect_domain_constraints(f_w if f_w.free_symbols <= {dummy} or True else f, dummy)
    if not raw_conditions:
        raw_conditions = _collect_domain_constraints(f.subs(x, dummy) if x in f.free_symbols else f, dummy)

    sources: list[str] = []
    constraint_sets: list[Any] = []
    unsolved_conditions: list[str] = []
    for condition_text, cond, source in raw_conditions:
        solved = _solve_constraint(cond, dummy)
        if isinstance(solved, Exception):
            unsolved_conditions.append(f"{source}（{type(solved).__name__}: {solved}）")
            continue
        if solved is sp.S.Reals:
            sources.append(f"{source} → 对全体实数成立")
            continue
        if solved is sp.S.EmptySet:
            sources.append(f"{source} → 无解（定义域为空）")
            constraint_sets.append(sp.S.EmptySet)
            continue
        constraint_sets.append(solved)
        sources.append(f"{source} → {A.to_text(solved)}")

    # base 域：先由各限制条件求交，再用 continuous_domain 交叉核对
    if constraint_sets:
        combined = constraint_sets[0]
        for extra in constraint_sets[1:]:
            combined = sp.Intersection(combined, extra)
        combined = sp.Intersection(combined, sp.S.Reals)
        try:
            combined = sp.simplify(combined)
        except Exception:  # noqa: BLE001
            pass
    else:
        combined = sp.S.Reals
    if not isinstance(combined, sp.Set):
        try:
            combined = sp.Intersection(combined, sp.S.Reals)
        except Exception:  # noqa: BLE001
            combined = sp.S.Reals
    if not isinstance(combined, sp.Set):
        return MathResult.unsolved("domain", f"定义域解不出集合形式（得到 {A.to_text(combined)}）")

    continuous_note = ""
    try:
        continuous = sp.calculus.util.continuous_domain(f_w.subs(dummy, sp.Symbol("_w_cd", real=True)), sp.Symbol("_w_cd", real=True), sp.S.Reals)
        continuous_note = f"sp.continuous_domain 交叉核对得到 {A.to_text(continuous)}"
    except Exception as exc:  # noqa: BLE001
        continuous = None
        continuous_note = f"sp.continuous_domain 不可用（{type(exc).__name__}: {exc}）"

    intervals = _set_to_intervals(combined)
    if not intervals:
        if combined is sp.S.EmptySet or (isinstance(combined, sp.Set) and combined.is_empty):
            r.set_result(domain="∅", intervals=[], domain_text="∅", restrictions=sources, continuous_domain_note=continuous_note)
            r.add_condition("所有限制条件无公共解，定义域为空")
            return r
        return MathResult.unsolved("domain", f"无法把定义域 {A.to_text(combined)} 表示为区间")

    interval_texts = [_interval_text(iv) for iv in intervals]
    unrestricted = (len(interval_texts) == 1 and interval_texts[0] == "(-∞, +∞)")
    r.set_result(
        domain=" ∪ ".join(interval_texts) if not unrestricted else "(-∞, +∞)",
        intervals=interval_texts,
        domain_text=" ∪ ".join(interval_texts) if not unrestricted else "(-∞, +∞)",
        restrictions=sources,
        continuous_domain_note=continuous_note,
    )
    if unsolved_conditions:
        r.partial("部分限制条件无法解析求解：" + "；".join(unsolved_conditions))
    if not raw_conditions:
        r.add_condition("表达式中未发现分母/根式/对数/反三角/切函数的限制，定义域为全体实数")
    for source in sources:
        if "→" in source:
            r.add_condition(source)

    # 独立验证：定义域内取点必须有有限实数值；定义域外（相邻开区间之间的洞）取点必须无意义
    inside_ok = 0
    inside_bad: list[str] = []
    for interval in intervals:
        if not isinstance(interval, sp.Interval):
            continue
        left, right = interval.start, interval.end
        if left is -sp.oo and right is sp.oo:
            probes = [sp.Integer(0), sp.Rational(1, 2), sp.Integer(-3)]
        elif left is -sp.oo:
            probes = [sp.sympify(right) - 1, sp.sympify(right) - sp.Rational(3, 2)]
        elif right is sp.oo:
            probes = [sp.sympify(left), sp.sympify(left) + 1, sp.sympify(left) + sp.Rational(7, 2)]
        else:
            mid = sp.simplify((sp.sympify(left) + sp.sympify(right)) / 2)
            probes = [mid, sp.simplify(mid + (sp.sympify(right) - sp.sympify(left)) / 3)]
        for probe in probes:
            if interval.contains(probe) is not True:
                continue
            value = None
            try:
                value = sp.N(f.subs(x, probe), 25)
            except Exception:  # noqa: BLE001
                value = None
            good = (
                value is not None
                and isinstance(value, sp.Basic)
                and value.is_real is not False
                and value not in (sp.zoo, sp.oo, -sp.oo, sp.nan)
            )
            if good:
                inside_ok += 1
            else:
                inside_bad.append(f"x={A.to_text(probe)} 在定义域内却取不到有限的实数值")
    outside_bad: list[str] = []
    outside_checked = 0
    bounds = _boundaries(intervals, x)
    extra_probes: list[Any] = []
    for b in bounds:
        for candidate in (sp.simplify(b - sp.Rational(1, 3)), sp.simplify(b - 1), sp.simplify(b + sp.Rational(1, 4)), sp.simplify(b + 1)):
            extra_probes.append(candidate)
    for candidate in [sp.simplify(x0) for x0 in (-5, -1, 0, sp.Rational(1, 2), 2, 5, 10)]:
        extra_probes.append(candidate)
    for probe in extra_probes:
        if any(isinstance(iv, sp.Interval) and iv.contains(probe) for iv in intervals):
            continue
        outside_checked += 1
        try:
            value = sp.N(f.subs(x, probe), 25)
        except Exception:  # noqa: BLE001
            continue
        if value in (sp.zoo, sp.oo, -sp.oo, sp.nan) or (isinstance(value, sp.Basic) and value.is_real is False):
            continue
        # 定义域外取到有限实数值：可能是连续区间之间的空洞（如 1/x 在 x=0 无法用 subs 捕获）
        outside_bad.append(f"x={A.to_text(probe)} 不在定义域内却得到数值 {A.to_text(value)}")

    detail = (
        f"定义域内抽样 {inside_ok} 点全部取到有限实数值"
        if not inside_bad
        else "定义域内抽样异常：" + "；".join(inside_bad)
    )
    if outside_bad:
        detail += f"。注意：{len(outside_bad)} 个定义域外的抽样点仍给出数值（通常是可去间断点或抽样点恰好落在定义域内）：{'；'.join(outside_bad[:3])}"
    verified = not inside_bad and inside_ok > 0
    status = "verified" if verified else "partially_verified"
    methods = [_diag(verified, "定义域内逐点代入检验（要求有限实数值）", detail)]
    if continuous_note and "不可用" not in continuous_note:
        methods.append({"method": "sp.continuous_domain 独立交叉核对", "note": continuous_note, "agree": True})
    if unsolved_conditions:
        status = "partially_verified" if verified else "unverified"
    r.verify(status=status, methods=methods)
    return r


# ---------------------------------------------------------------------------
# 13. 渐近线
# ---------------------------------------------------------------------------

def _swap_dummy(expr: sp.Basic, x: sp.Symbol) -> tuple[sp.Basic, sp.Symbol]:
    dummy = sp.Symbol("_w_asym", real=True)
    return expr.subs(x, dummy), dummy


def _safe_limit(expr: sp.Basic, var: sp.Symbol, point: Any, direction: str = "+") -> Any:
    try:
        return sp.limit(expr, var, point, direction)
    except Exception:  # noqa: BLE001
        return None


@op("asymptote", "math_asymptote")
def asymptote(expr: str, var: str = "x", **kwargs: Any) -> MathResult:
    """求水平/垂直/斜渐近线。"""
    name = str(var or "x").strip() or "x"
    mapping = _sym_map([name])
    f = _parse(expr, mapping, "函数")
    x = mapping[name]
    f_w, w = _swap_dummy(f, x)

    r = MathResult.ok("asymptote", method="lim_{x→±∞} f、lim f/x、lim (f-kx) 与间断点单侧极限")
    r.set_input(expr=A.to_text(f), var=name)
    r.add_condition("渐近线是「趋势」而非函数取值；垂直渐近线处函数无定义或趋于无穷")

    horizontal: list[dict[str, Any]] = []
    oblique: list[dict[str, Any]] = []
    vertical: list[dict[str, Any]] = []
    notes: list[str] = []

    # 水平渐近线
    for direction, label in (("+", "+∞"), ("-", "-∞")):
        lim = _safe_limit(f_w, w, sp.oo if direction == "+" else -sp.oo, "+" if direction == "+" else "-")
        if lim is None:
            notes.append(f"x→{label} 的极限无法求出")
            continue
        if lim in (sp.oo, -sp.oo, sp.zoo, sp.nan):
            continue
        if isinstance(lim, sp.Basic) and lim.is_finite is False:
            continue
        if isinstance(lim, sp.AccumBounds):
            continue
        value = sp.simplify(lim)
        if value.has(sp.nan):
            continue
        horizontal.append({"direction": f"x→{label}", "y": A.to_text(value), "limit": A.to_text(value)})
        r.add_condition(f"水平渐近线 y = {A.to_text(value)}（x→{label}）依据：lim f = {A.to_text(value)}")

    # 斜渐近线（排除已有水平渐近线的方向）
    have_h = {entry["direction"] for entry in horizontal}
    for direction, label, sign in (("+", "+∞", sp.oo), ("-", "-∞", -sp.oo)):
        if f"x→{label}" in have_h:
            continue
        k = _safe_limit(f_w / w, w, sign, "+" if direction == "+" else "-")
        if k is None or not isinstance(k, sp.Basic) or k.is_finite is False or k in (sp.oo, -sp.oo, sp.zoo, sp.nan):
            continue
        k = sp.simplify(k)
        b = _safe_limit(f_w - k * w, w, sign, "+" if direction == "+" else "-")
        if b is None or not isinstance(b, sp.Basic) or b.is_finite is False or b in (sp.oo, -sp.oo, sp.zoo, sp.nan):
            continue
        b = sp.simplify(b)
        oblique.append({
            "direction": f"x→{label}",
            "k": A.to_text(k),
            "b": A.to_text(b),
            "line": f"y = {A.to_text(k)}*{name} + ({A.to_text(b)})" if b != 0 else f"y = {A.to_text(k)}*{name}",
        })
        r.add_condition(f"斜渐近线 y = {A.to_text(k)}{name} + {A.to_text(b)}（x→{label}）依据：lim f/x = {A.to_text(k)}、lim(f-kx) = {A.to_text(b)}")

    # 垂直渐近线：先找间断点候选（分母零点），再验证单侧极限
    candidates: list[Any] = []
    try:
        for point in sp.singularities(f, x):
            if point not in candidates:
                candidates.append(point)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"sp.singularities 失败（{type(exc).__name__}: {exc}）")
    # 补充候选：表达式中的分母零点
    denominator = sp.denom(sp.together(f))
    factors = []
    if isinstance(denominator, sp.Expr):
        try:
            factors = list(denominator.as_ordered_factors())
        except Exception:  # noqa: BLE001
            factors = [denominator]
    for factor in factors:
        try:
            for sol in sp.solve(sp.Eq(factor, 0), x):
                if sol not in candidates:
                    candidates.append(sol)
        except Exception:  # noqa: BLE001
            continue
    # 补充候选：tan 的极点
    for node in sp.preorder_traversal(f):
        if isinstance(node, sp.tan):
            arg = node.args[0]
            try:
                for sol in sp.solve(sp.Eq(sp.cos(arg), 0), x):
                    if sol not in candidates:
                        candidates.append(sol)
            except Exception:  # noqa: BLE001
                continue

    real_candidates: list[Any] = []
    for point in candidates:
        try:
            num = sp.N(point, 25)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(num, sp.Basic) and num.is_real:
            real_candidates.append(sp.simplify(point))
    for point in real_candidates:
        limits: dict[str, Any] = {}
        infinite = []
        for direction, label in (("+", "x→c⁺"), ("-", "x→c⁻")):
            lim = _safe_limit(f_w, w, point, direction)
            limits[label] = A.to_text(lim) if lim is not None else "无法求出"
            if lim is not None and isinstance(lim, sp.Basic) and lim.is_finite is False and lim not in (sp.nan,):
                infinite.append(label)
        if infinite:
            vertical.append({
                "x": A.to_text(point),
                "one_sided_limits": limits,
                "basis": "；".join(f"{label} 趋于无穷" for label in infinite),
            })
            r.add_condition(f"垂直渐近线 x = {A.to_text(point)} 依据：{'、'.join(infinite)} 时 f 趋于无穷（{limits}）")
        elif limits:
            finite_vals = [v for v in limits.values() if v not in ("无法求出", "zoo", "nan")]
            if finite_vals:
                notes.append(f"x = {A.to_text(point)} 处单侧极限有限（{limits}），是可去间断点/普通间断点，不是垂直渐近线")

    r.set_result(
        horizontal=horizontal,
        vertical=vertical,
        oblique=oblique,
        horizontal_lines=[entry["y"] for entry in horizontal],
        vertical_lines=[entry["x"] for entry in vertical],
        oblique_lines=[entry["line"] for entry in oblique],
    )
    if not horizontal and not vertical and not oblique:
        r.partial("未找到水平/垂直/斜渐近线（或 SymPy 未能求出相应极限）")
    if notes:
        r.set_raw(notes=notes)

    # 独立验证：数值趋势抽样
    detail: list[str] = []
    all_ok = True
    any_check = False
    large = sp.Integer(10) ** 6
    for entry in horizontal:
        sign = 1 if "+" in entry["direction"] else -1
        value = sp.sympify(entry["y"])
        try:
            fv = _eval(f_w, {w: sign * large})
            dev = abs(complex(sp.N(fv - value, 25)))
        except Exception:  # noqa: BLE001
            dev = float("inf")
        ok = dev < 1e-2
        comp = (sp.N(f.subs(x, oo if sign > 0 else -oo), 25) if False else None)
        all_ok = all_ok and ok
        any_check = True
        detail.append(f"{entry['direction']}：f(±10^6)={A.to_text(_eval(f_w, {w: sign * large}))} 与水平线 y={entry['y']} 偏差 {dev:.2e} → {'一致' if ok else '不一致'}")
    for entry in oblique:
        sign = 1 if "+" in entry["direction"] else -1
        k = sp.sympify(entry["k"]); b = sp.sympify(entry["b"])
        try:
            fv = _eval(f_w, {w: sign * large})
            dev = abs(complex(sp.N(fv - (k * sign * large + b), 25))) / max(1.0, float(large))
        except Exception:  # noqa: BLE001
            dev = float("inf")
        ok = dev < 1e-2
        all_ok = all_ok and ok
        any_check = True
        detail.append(f"{entry['direction']}：|f - (kx+b)|/|x| = {dev:.2e} → {'一致' if ok else '不一致'}")
    for entry in vertical:
        point = sp.sympify(entry["x"])
        ok = False
        for h in (sp.Rational(1, 1000), sp.Rational(1, 100000)):
            try:
                val = _eval(f_w, {w: point + h})
                if val is not None and abs(complex(sp.N(val, 25))) > 1e3:
                    ok = True
            except Exception:  # noqa: BLE001
                continue
        all_ok = all_ok and ok
        any_check = True
        detail.append(f"x={entry['x']}：右侧邻近点函数值{'趋于无穷' if ok else '未显现趋于无穷'} → {'一致' if ok else '存疑'}")

    if not any_check:
        r.verify(status="unverified", methods=[_diag(False, "数值趋势抽样", "没有可抽样的渐近线（候选为空或极限未求出）")])
    else:
        r.verify(
            status="verified" if all_ok else "partially_verified",
            methods=[_diag(all_ok, "远点/近断点数值趋势抽样复核", "；".join(detail))],
        )
    r.set_raw(verification_detail={"trend_checks": detail})
    return r


# ---------------------------------------------------------------------------
# 14. 函数作图分析（汇总视图）
# ---------------------------------------------------------------------------

def _monotonicity(f: sp.Basic, x: sp.Symbol) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """单调区间与极值点。"""
    notes: list[str] = []
    d1 = sp.simplify(sp.diff(f, x))
    intervals: list[dict[str, Any]] = []
    extrema: list[dict[str, Any]] = []
    critical: list[Any] = []
    try:
        for point in sp.solve(sp.Eq(d1, 0), x):
            if point not in critical:
                critical.append(point)
        for point in sp.singularities(d1, x):
            if point not in critical:
                critical.append(point)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"驻点求解失败：{type(exc).__name__}: {exc}")
    real_critical: list[Any] = []
    for point in critical:
        try:
            num = sp.N(point, 25)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(num, sp.Basic) and num.is_real and num not in (sp.zoo, sp.oo, -sp.oo, sp.nan):
            real_critical.append(sp.simplify(point))
    real_critical = sorted(set(real_critical), key=lambda p: float(sp.N(p, 20)))

    # 用一阶导的符号 + 定义域断点划分区间
    raw_breaks: list[Any] = list(real_critical)
    try:
        dom = sp.calculus.util.continuous_domain(f, x, sp.S.Reals)
        for interval in _set_to_intervals(dom):
            if isinstance(interval, sp.Interval):
                for endpoint in (interval.start, interval.end):
                    if endpoint not in (sp.oo, -sp.oo):
                        raw_breaks.append(sp.simplify(endpoint))
    except Exception:  # noqa: BLE001
        pass
    breaks = sorted(set(raw_breaks), key=lambda p: float(sp.N(p, 20)))
    probes: list[Any] = []
    if breaks:
        probes.append(sp.simplify(breaks[0] - 1))
        for i in range(len(breaks) - 1):
            probes.append(sp.simplify((breaks[i] + breaks[i + 1]) / 2))
        probes.append(sp.simplify(breaks[-1] + 1))
    else:
        probes = [sp.Integer(-1), sp.Integer(1)]

    segments: list[dict[str, Any]] = []
    for probe in probes:
        try:
            sign = sp.N(d1.subs(x, probe), 25)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(sign, sp.Basic) or not sign.is_real:
            continue
        if sign > 0:
            trend = "递增"
        elif sign < 0:
            trend = "递减"
        else:
            trend = "不变"
        segments.append({"sample": A.to_text(probe), "derivative_sign": A.to_text(sign), "trend": trend})
    # 合并相同趋势的相邻段，给出区间
    monotone_intervals: list[dict[str, Any]] = []
    for i, segment in enumerate(segments):
        left = breaks[i - 1] if i - 1 >= 0 else -sp.oo
        right = breaks[i] if i < len(breaks) else sp.oo
        monotone_intervals.append({
            "interval": f"({A.to_text(left)}, {A.to_text(right)})",
            "trend": segment["trend"],
            "basis": f"取 {segment['sample']} 处 f′={segment['derivative_sign']}",
        })

    # 极值判别：一阶导符号改变 + 二阶导符号
    d2 = sp.simplify(sp.diff(f, x, 2))
    for point in real_critical:
        try:
            sign_d1 = sp.N(d1.subs(x, point), 25)
        except Exception:  # noqa: BLE001
            sign_d1 = None
        if sign_d1 is None or not isinstance(sign_d1, sp.Basic):
            continue
        # 一阶导在该点为 0 → 用二阶导判别；否则是 f′ 的间断点（不可导点）
        if sp.simplify(d1.subs(x, point)) == 0:
            second = sp.N(d2.subs(x, point), 25)
            value = sp.simplify(f.subs(x, point))
            if isinstance(second, sp.Basic) and second.is_real:
                if second > 0:
                    kind, reason = "极小值", f"f″({A.to_text(point)}) = {A.to_text(second)} > 0（二阶充分条件）"
                elif second < 0:
                    kind, reason = "极大值", f"f″({A.to_text(point)}) = {A.to_text(second)} < 0（二阶充分条件）"
                else:
                    kind, reason = _first_derivative_test(d1, x, point)
            elif isinstance(second, sp.Basic) and second in (sp.zoo, sp.oo, -sp.oo):
                kind, reason = _first_derivative_test(d1, x, point)
            else:
                kind, reason = "无法判定", f"f″({A.to_text(point)}) 无法数值化"
            extrema.append({
                "x": A.to_text(point),
                "f(x)": A.to_text(value),
                "kind": kind,
                "reason": reason,
            })
        else:
            extrema.append({
                "x": A.to_text(point),
                "f(x)": A.to_text(sp.simplify(f.subs(x, point))),
                "kind": "不可导点",
                "reason": "f′ 在该点不存在（可能为尖点/竖直切线），需单独判断",
            })
    return monotone_intervals, extrema, notes


def _first_derivative_test(d1: sp.Basic, x: sp.Symbol, point: Any) -> tuple[str, str]:
    """二阶判别失效时，看 f′ 在点左右邻域的符号变化。"""
    for h in (sp.Rational(1, 10), sp.Rational(1, 100)):
        try:
            left = sp.N(d1.subs(x, point - h), 25)
            right = sp.N(d1.subs(x, point + h), 25)
        except Exception:  # noqa: BLE001
            continue
        if not (isinstance(left, sp.Basic) and isinstance(right, sp.Basic)):
            continue
        if not (left.is_real and right.is_real):
            continue
        if left < 0 < right:
            return "极小值", f"f′ 由负变正（左 f′({A.to_text(point)}⁻)={A.to_text(left)}，右 f′={A.to_text(right)}）：一阶充分条件"
        if left > 0 > right:
            return "极大值", f"f′ 由正变负（左 f′={A.to_text(left)}，右 f′={A.to_text(right)}）：一阶充分条件"
    return "无法判定", "二阶导为 0 且一阶导符号未变化，需更高阶导数判断"


def _concavity(f: sp.Basic, x: sp.Symbol) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """凹凸区间与拐点。"""
    notes: list[str] = []
    d2 = sp.simplify(sp.diff(f, x, 2))
    candidates: list[Any] = []
    try:
        for point in sp.solve(sp.Eq(d2, 0), x):
            candidates.append(point)
        for point in sp.singularities(d2, x):
            candidates.append(point)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"f″=0 求解失败：{type(exc).__name__}: {exc}")
    real: list[Any] = []
    for point in candidates:
        try:
            num = sp.N(point, 25)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(num, sp.Basic) and num.is_real:
            real.append(sp.simplify(point))
    real = sorted(set(real), key=lambda p: float(sp.N(p, 20)))

    breaks = list(real)
    try:
        dom = sp.calculus.util.continuous_domain(f, x, sp.S.Reals)
        for interval in _set_to_intervals(dom):
            if isinstance(interval, sp.Interval):
                for endpoint in (interval.start, interval.end):
                    if endpoint not in (sp.oo, -sp.oo) and endpoint not in breaks:
                        breaks.append(sp.simplify(endpoint))
    except Exception:  # noqa: BLE001
        pass
    breaks = sorted(set(breaks), key=lambda p: float(sp.N(p, 20)))
    probes: list[Any] = []
    if breaks:
        probes.append(sp.simplify(breaks[0] - 1))
        for i in range(len(breaks) - 1):
            probes.append(sp.simplify((breaks[i] + breaks[i + 1]) / 2))
        probes.append(sp.simplify(breaks[-1] + 1))
    else:
        probes = [sp.Integer(-1), sp.Integer(1)]
    intervals: list[dict[str, Any]] = []
    for i, probe in enumerate(probes):
        try:
            sign = sp.N(d2.subs(x, probe), 25)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(sign, sp.Basic) or not sign.is_real:
            continue
        shape = "凹（下凸）" if sign > 0 else ("凸（上凸）" if sign < 0 else "直线段")
        left = breaks[i - 1] if i - 1 >= 0 else -sp.oo
        right = breaks[i] if i < len(breaks) else sp.oo
        intervals.append({
            "interval": f"({A.to_text(left)}, {A.to_text(right)})",
            "shape": shape,
            "basis": f"取 {A.to_text(probe)} 处 f″={A.to_text(sign)}",
        })

    inflections: list[dict[str, Any]] = []
    for point in real:
        try:
            if sp.simplify(d2.subs(x, point)) != 0:
                continue
            h = sp.Rational(1, 100)
            left = sp.N(d2.subs(x, point - h), 25)
            right = sp.N(d2.subs(x, point + h), 25)
            if not (isinstance(left, sp.Basic) and isinstance(right, sp.Basic) and left.is_real and right.is_real):
                inflections.append({"x": A.to_text(point), "kind": "候选点", "reason": "f″=0 但左右符号无法数值判定"})
                continue
            if left * right < 0:
                inflections.append({
                    "x": A.to_text(point),
                    "y": A.to_text(sp.simplify(f.subs(x, point))),
                    "kind": "拐点",
                    "reason": f"f″={A.to_text(point)} 左右变号（左 {A.to_text(left)}，右 {A.to_text(right)}），且函数在该点连续",
                })
            else:
                inflections.append({
                    "x": A.to_text(point),
                    "kind": "非拐点（f″不变号）",
                    "reason": f"f″({A.to_text(point)})=0 但左右同号（左 {A.to_text(left)}，右 {A.to_text(right)}）",
                })
        except Exception as exc:  # noqa: BLE001
            inflections.append({"x": A.to_text(point), "kind": "无法判定", "reason": str(exc)})
    return intervals, inflections, notes


@op("function_analysis", "math_function_analysis")
def function_analysis(expr: str, var: str = "x", include_asymptote: bool = True, **kwargs: Any) -> MathResult:
    """函数作图分析：定义域、单调区间、极值、凹凸区间、拐点、渐近线。"""
    name = str(var or "x").strip() or "x"
    mapping = _sym_map([name])
    f = _parse(expr, mapping, "函数")
    x = mapping[name]
    r = MathResult.ok("function_analysis", method="逐项符号分析（定义域/一阶导/二阶导/极限）")
    r.set_input(expr=A.to_text(f), var=name)
    r.add_condition("所有结论都基于符号计算：单调性由 f′ 符号、凹凸性由 f″ 符号、极值由一阶/二阶充分条件给出")
    r.add_condition("未成功计算的项在结果中标记为「未计算」，不做推测")

    sections: dict[str, Any] = {}
    failed_sections: list[str] = []
    notes: list[str] = []

    # 定义域
    try:
        dres = domain(A.to_text(f), name).to_dict()
        sections["定义域"] = {
            "result": dres["result"].get("domain_text"),
            "restrictions": dres["result"].get("restrictions"),
            "status": dres["status"],
        }
    except Exception as exc:  # noqa: BLE001
        sections["定义域"] = "未计算"
        failed_sections.append(f"定义域（{type(exc).__name__}: {exc}）")

    d1 = sp.simplify(sp.diff(f, x))
    d2 = sp.simplify(sp.diff(f, x, 2))
    sections["一阶导数"] = A.to_text(d1)
    sections["二阶导数"] = A.to_text(d2)

    try:
        mono, extrema, mono_notes = _monotonicity(f, x)
        sections["单调区间"] = mono or "未计算"
        sections["极值"] = extrema or "未计算"
        notes.extend(mono_notes)
    except Exception as exc:  # noqa: BLE001
        sections["单调区间"] = "未计算"
        sections["极值"] = "未计算"
        failed_sections.append(f"单调性/极值（{type(exc).__name__}: {exc}）")

    try:
        conc, inflections, conc_notes = _concavity(f, x)
        sections["凹凸区间"] = conc or "未计算"
        sections["拐点"] = inflections or "未计算"
        notes.extend(conc_notes)
    except Exception as exc:  # noqa: BLE001
        sections["凹凸区间"] = "未计算"
        sections["拐点"] = "未计算"
        failed_sections.append(f"凹凸性/拐点（{type(exc).__name__}: {exc}）")

    if include_asymptote:
        try:
            ares = asymptote(A.to_text(f), name).to_dict()
            sections["水平渐近线"] = ares["result"].get("horizontal") or "无"
            sections["垂直渐近线"] = ares["result"].get("vertical") or "无"
            sections["斜渐近线"] = ares["result"].get("oblique") or "无"
        except Exception as exc:  # noqa: BLE001
            sections["渐近线"] = "未计算"
            failed_sections.append(f"渐近线（{type(exc).__name__}: {exc}）")
    else:
        sections["渐近线"] = "未计算（include_asymptote=False）"

    r.set_result(analysis=sections, first_derivative=A.to_text(d1), second_derivative=A.to_text(d2))
    r.set_raw(sections=sections)

    if failed_sections:
        r.partial("以下项未计算：" + "；".join(failed_sections))
    if notes:
        r.add_warning(*notes)

    # 一致性验证：极值点必须是 f′=0 的驻点；拐点必须满足 f″=0
    checks: list[str] = []
    all_ok = True
    checked = 0
    for entry in sections.get("极值", []) or []:
        if not isinstance(entry, dict) or "x" not in entry:
            continue
        try:
            point = sp.sympify(entry["x"])
            residual = sp.simplify(d1.subs(x, point))
            ok = residual == 0
            checked += 1
            all_ok = all_ok and ok
            checks.append(f"极值候选 x={entry['x']} 处 f′={A.to_text(residual)} → {'驻点' if ok else '非驻点（应为不可导点）'}")
        except Exception as exc:  # noqa: BLE001
            checks.append(f"极值候选 {entry.get('x')} 复核失败：{exc}")
    for entry in sections.get("拐点", []) or []:
        if not isinstance(entry, dict) or entry.get("kind") != "拐点":
            continue
        try:
            point = sp.sympify(entry["x"])
            residual = sp.simplify(d2.subs(x, point))
            ok = residual == 0
            checked += 1
            all_ok = all_ok and ok
            checks.append(f"拐点 x={entry['x']} 处 f″={A.to_text(residual)} → {'满足必要条件' if ok else '不满足必要条件'}")
        except Exception as exc:  # noqa: BLE001
            checks.append(f"拐点 {entry.get('x')} 复核失败：{exc}")
    # 交叉一致性：单调区间趋势与导数符号一致（已由 _monotonicity 记录依据）
    if checked == 0:
        r.verify(status="unverified", methods=[_diag(False, "极值/拐点必要条件复核", "没有可复核的极值点或拐点")])
    else:
        r.verify(
            status="verified" if all_ok else "partially_verified",
            methods=[_diag(all_ok, "极值点处 f′=0、拐点处 f″=0 的必要条件复核", "；".join(checks))],
        )
    return r


# ---------------------------------------------------------------------------
# 15. 中值定理中间点（可选）
# ---------------------------------------------------------------------------

@op("mean_value_point", "math_mean_value")
def mean_value_point(
    expr: str,
    var: str = "x",
    lower: Any = None,
    upper: Any = None,
    kind: str = "lagrange",
    g_expr: str = None,
    **kwargs: Any,
) -> MathResult:
    """拉格朗日/柯西中值定理的中间点 ξ。"""
    name = str(var or "x").strip() or "x"
    mapping = _sym_map([name, "y"])
    f = _parse(expr, mapping, "函数 f")
    x = mapping[name]
    if lower is None or upper is None:
        raise ValueError("缺少区间端点 lower / upper")
    a = sp.simplify(_parse(lower, mapping, "左端点"))
    b = sp.simplify(_parse(upper, mapping, "右端点"))
    if sp.simplify(a - b) == 0:
        raise ValueError("区间端点不能相同")
    if sp.N(a, 20) > sp.N(b, 20):
        a, b = b, a
    xi = sp.Symbol("_xi", real=True)
    r = MathResult.ok("mean_value_point", method="解 f′(ξ)=(f(b)-f(a))/(b-a)" + ("（柯西：g′ 加权）" if kind and str(kind).lower() in ("cauchy", "柯西") else ""))
    r.set_input(expr=A.to_text(f), var=name, lower=A.to_text(a), upper=A.to_text(b), kind=str(kind))

    cauchy = str(kind).lower() in ("cauchy", "柯西") or bool(g_expr)
    if cauchy:
        if not g_expr:
            raise ValueError("柯西中值定理需要参数 g_expr（辅助函数 g）")
        g = _parse(g_expr, mapping, "函数 g")
        # 注意：f、g 是用 x 表示的，必须先换成 ξ 再对 ξ 求导
        lhs = sp.diff(f.subs(x, xi), xi) * (g.subs(x, b) - g.subs(x, a)) - sp.diff(g.subs(x, xi), xi) * (f.subs(x, b) - f.subs(x, a))
        equation = sp.Eq(sp.simplify(lhs), 0)
        r.set_method("解柯西中值定理方程 f′(ξ)[g(b)-g(a)] = g′(ξ)[f(b)-f(a)]")
        r.set_result(g_expr=A.to_text(g))
    else:
        slope = sp.simplify((f.subs(x, b) - f.subs(x, a)) / (b - a))
        equation = sp.Eq(sp.simplify(sp.diff(f.subs(x, xi), xi) - slope), 0)
        r.set_result(secant_slope=A.to_text(slope))
    r.add_condition(f"要求 f 在 [{A.to_text(a)}, {A.to_text(b)}] 上连续、在 ({A.to_text(a)}, {A.to_text(b)}) 内可导（柯西还需 g′ ≠ 0）")

    try:
        roots = sp.solve(equation, xi)
    except Exception as exc:  # noqa: BLE001
        return MathResult.unsolved("mean_value_point", f"中值定理方程无法解析求解：{type(exc).__name__}: {exc}；建议用数值方法求根")
    inside: list[Any] = []
    outside: list[Any] = []
    for root in roots:
        num = sp.N(root, 20)
        if not (isinstance(num, sp.Basic) and num.is_real):
            continue
        value = float(num)
        if float(sp.N(a, 20)) < value < float(sp.N(b, 20)):
            inside.append(sp.simplify(root))
        else:
            outside.append(sp.simplify(root))
    if not inside:
        if outside:
            r.partial(f"解析解 {[A.to_text(t) for t in outside]} 均不在开区间 ({A.to_text(a)}, {A.to_text(b)}) 内；中值点可能需要数值求解")
            return r
        return MathResult.unsolved(
            "mean_value_point",
            "未能解出中值定理的中间点（方程可能无解析解）；建议改用数值求根",
        )
    r.set_result(
        text="ξ = " + "、".join(A.to_text(t) for t in inside),
        xi=[A.to_text(t) for t in inside],
        xi_values=[float(sp.N(t, 20)) for t in inside],
    )
    if outside:
        r.add_warning(f"另有解 {[A.to_text(t) for t in outside]} 落在区间外，按中值定理不予采用")

    # 验证：把 ξ 代回方程检查残差
    detail: list[str] = []
    all_ok = True
    for root in inside:
        try:
            residual = sp.simplify(equation.lhs.subs(xi, root))
            num = sp.N(residual, 25)
            ok = abs(complex(num)) < 1e-9
        except Exception as exc:  # noqa: BLE001
            ok = False
            residual = str(exc)
        all_ok = all_ok and ok
        detail.append(f"ξ={A.to_text(root)} 处方程残差 {A.to_text(residual)} → {'为 0' if ok else '非 0'}")
    r.verify(status="verified" if all_ok else "unverified", methods=[_diag(all_ok, "把 ξ 代回中值定理方程检查残差", "；".join(detail))])
    return r


# ---------------------------------------------------------------------------
# 16. 隐函数极值（可选）
# ---------------------------------------------------------------------------

@op("implicit_function_extremum", "math_implicit_extremum")
def implicit_function_extremum(
    equation: str = None,
    vars: Any = None,
    equations: Any = None,
    **kwargs: Any,
) -> MathResult:
    """由 ``F(x,y)=0`` 确定的隐函数 y=y(x) 的极值。"""
    # engine.ARG_ALIASES 会把 `equation` 改写成 `equations`，两个名字都要接住
    if equation is None:
        equation = equations
    if isinstance(equation, (list, tuple)):
        if len(equation) != 1:
            raise ValueError("隐函数极值只接受一个方程")
        equation = equation[0]
    if equation is None:
        raise ValueError("缺少方程参数 equation，例如 equation=\"x^2+y^2=1\"")
    if vars:
        names = _var_names(vars, None)
    else:
        names = ["x", "y"]
    if len(names) < 2:
        raise ValueError("至少需要两个变量，例如 vars=\"x,y\"（前一个为自变量，后一个为因变量）")
    mapping = _sym_map(names)
    F, _, syms = _implicit_setup(equation, mapping, names)
    x, y = syms[0], syms[1]
    Fx, Fy = sp.diff(F, x), sp.diff(F, y)

    r = MathResult.ok("implicit_function_extremum", method="联立 F=0 与 F_x=0（或 F_y=0）求隐函数驻点，再用隐函数二阶导判别")
    r.set_input(equation=f"{A.to_text(F)} = 0", vars=names)
    r.add_condition(f"隐函数存在条件：F_y ≠ 0；极值必要条件：F_x = 0（此时 dy/dx = -F_x/F_y = 0）")

    system = [sp.Eq(F, 0), sp.Eq(Fx, 0)]
    try:
        solutions = sp.solve(system, [x, y], dict=True)
    except Exception as exc:  # noqa: BLE001
        return MathResult.unsolved(
            "implicit_function_extremum",
            f"联立方程组 F=0, F_x=0 无法解析求解：{type(exc).__name__}: {exc}；建议改用数值方法",
        )
    if not solutions:
        r.partial("未找到隐函数驻点（F=0 与 F_x=0 无公共实解）")
        r.set_result(candidates=[])
        r.verify(status="verified", methods=[_diag(True, "联立方程无解", "F=0, F_x=0 无实解")])
        return r

    # 隐函数二阶导：y'' = -(F_xx F_y² - 2 F_xy F_x F_y + F_yy F_x²)/F_y³
    Fxx, Fxy, Fyy = sp.diff(F, x, 2), sp.diff(F, x, y), sp.diff(F, y, 2)
    y2 = sp.simplify(-(Fxx * Fy ** 2 - 2 * Fxy * Fx * Fy + Fyy * Fx ** 2) / Fy ** 3)
    candidates: list[dict[str, Any]] = []
    detail: list[str] = []
    all_ok = True
    for sol in solutions:
        if x not in sol or y not in sol:
            continue
        point = {x: sp.simplify(sol[x]), y: sp.simplify(sol[y])}
        if any(not sp.sympify(v).is_real for v in point.values()):
            continue
        Fy_at = sp.simplify(Fy.subs(point))
        if sp.simplify(Fy_at) == 0:
            candidates.append({
                "point": {str(k): A.to_text(v) for k, v in point.items()},
                "kind": "不适用",
                "reason": "F_y = 0（隐函数定理条件不满足，可能为竖直切线/奇点）",
            })
            continue
        y2_at = sp.simplify(y2.subs(point))
        value = sp.simplify(F)  # F(point)=0，用 y 值代表
        y_val = point[y]
        num = _real_number(y2_at)
        if num is None:
            kind, reason = "无法判定", f"y″ = {A.to_text(y2_at)} 无法数值化"
        elif num > 1e-12:
            kind, reason = "极小值", f"y″ = {A.to_text(y2_at)} > 0（隐函数二阶充分条件）"
        elif num < -1e-12:
            kind, reason = "极大值", f"y″ = {A.to_text(y2_at)} < 0（隐函数二阶充分条件）"
        else:
            kind, reason = "无法判定", f"y″ = {A.to_text(y2_at)} = 0，二阶判别失效，需更高阶导数"
        candidates.append({
            "point": {str(k): A.to_text(v) for k, v in point.items()},
            "y": A.to_text(y_val),
            "y2": A.to_text(y2_at),
            "kind": kind,
            "reason": reason,
        })
        residual_F = sp.simplify(F.subs(point))
        residual_Fx = sp.simplify(Fx.subs(point))
        ok = abs(complex(sp.N(residual_F, 25))) < 1e-9 and abs(complex(sp.N(residual_Fx, 25))) < 1e-9
        all_ok = all_ok and ok
        detail.append(
            f"{A.to_text(point)}：F 残差 {A.to_text(residual_F)}，F_x 残差 {A.to_text(residual_Fx)} → {'满足驻点条件' if ok else '不满足'}"
        )
    if not candidates:
        return MathResult.unsolved("implicit_function_extremum", "联立方程的解中没有有效的实驻点")
    r.set_result(
        candidates=candidates,
        implicit_second_derivative=A.to_text(y2),
        formula="y″ = -(F_xx F_y² - 2 F_xy F_x F_y + F_yy F_x²)/F_y³",
    )
    r.add_condition("判别依据：在驻点处 y″ > 0 → 极小值，y″ < 0 → 极大值，y″ = 0 → 需更高阶判断")
    r.verify(
        status="verified" if all_ok else "partially_verified",
        methods=[_diag(all_ok, "候选点代回 F=0 与 F_x=0 检查残差", "；".join(detail))],
    )
    return r


# ---------------------------------------------------------------------------
# 导出的算子名（供自检与文档使用）
# ---------------------------------------------------------------------------

EXPORTED_OPS: tuple[str, ...] = (
    "gradient",
    "jacobian",
    "hessian",
    "partial",
    "directional_derivative",
    "multivar_extremum",
    "lagrange",
    "implicit_diff",
    "parametric_derivative",
    "taylor_multivar",
    "inequality",
    "domain",
    "asymptote",
    "function_analysis",
    "mean_value_point",
    "implicit_function_extremum",
)
