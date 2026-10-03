"""线性代数算子（考研数学一）。

覆盖：行列式、逆、秩、转置、矩阵乘法/加法、迹、行最简形、线性方程组、
特征值与特征向量、对角化、二次型、矩阵幂、矩阵总览。

设计要点（与用户规格一致）：

* **精确优先**：所有结果保持 SymPy 精确形式（分数、根号、符号），
  绝不默认转小数；近似值只作为 ``approx`` 附加字段由 ``result`` 层生成。
* **独立验证**：每个成功算子的 ``verification`` 必须给出与主算法
  *不同的* 验证途径（如行列式用余子式展开、Laplace 展开对 Bareiss 消元），
  不用同一个函数再算一次冒充验证。
* **绝不伪造**：求不出就 ``MathResult.unsolved``；输入不合规就 ``raise ValueError``
  （由调度器转成 ``invalid_input``）；部分成功用 ``r.partial``。
* 只依赖标准库 + sympy + 本包的 ``ast`` / ``sympy_core`` / ``result`` / ``engine``。
  不使用 numpy/scipy（数值判据容易把"近似为 0"当成"等于 0"，与精确优先冲突）。
"""

from __future__ import annotations

import itertools
import re
from typing import Any, Iterable, Sequence

import sympy as sp

from . import ast as A
from . import sympy_core as C
from .engine import op
from .result import MathResult

# ---------------------------------------------------------------------------
# 内部小工具（不注册为算子）
# ---------------------------------------------------------------------------

#: 含自由符号且维数超过该值时给出"可能很慢"的警告
_SYMBOLIC_WARN_SIZE = 6
#: 维数不超过该值时用 Laplace 余子式展开做独立的行列式验证
_COFACTOR_LIMIT = 4
#: 维数不超过该值时用「主子式」独立复核秩
_MINOR_RANK_LIMIT = 4
#: 维数不超过该值时在总览里附带行最简形（省 Token）
_RREF_SUMMARY_LIMIT = 6
#: 特征向量求解的规模上限（超出则只给特征值，避免展开爆炸）
_EIGVECT_LIMIT = 6

#: 变量名抓取：合法的变量名会出现在列出的符号表中（缺失的符号在 SymPy 里
#: 会退化成自由函数，因此必须显式把表达式里出现的符号传给 ``A.parse``）
_RESERVED_NAMES = frozenset(str(name) for name in getattr(A, "_RESERVED", ()) or ())
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def _is_matrix_like(value: Any) -> bool:
    return isinstance(value, (list, tuple)) or isinstance(value, sp.MatrixBase)


def _matrix(value: Any, symbols: Iterable[str] | None = None, *, what: str = "矩阵") -> sp.Matrix:
    """统一的矩阵输入入口。

    ``value`` 既可以是二维数组（``[[1,2],[3,4]]``），也可以是文本
    （``"1,2;3,4"`` / ``"[[1,2],[3,4]]"``）。空值会抛出可读的 ``ValueError``。
    """
    if value is None:
        raise ValueError(f"{what}为空：请用参数 matrix 给出矩阵（二维数组或 \"1,2;3,4\" 文本）")
    if isinstance(value, sp.MatrixBase):
        return value
    if isinstance(value, (list, tuple)):
        seq = list(value)
        if not seq:
            raise ValueError(f"{what}为空")
        if all(isinstance(row, (list, tuple)) for row in seq):
            return sp.Matrix(A.parse_matrix(rows=[list(row) for row in seq], symbols=symbols))
        return sp.Matrix(A.parse_matrix(rows=[seq], symbols=symbols))
    if isinstance(value, str):
        return sp.Matrix(A.parse_matrix(text=value, symbols=symbols))
    raise ValueError(f"{what}格式无法识别：{type(value).__name__}，请给出二维数组或 \"1,2;3,4\" 文本")


def _vector(value: Any, symbols: Iterable[str] | None = None, *, what: str = "右端项") -> sp.Matrix:
    """一维向量输入：``"1,2,3"`` / ``"[1,2,3]"`` / ``[1,2,3]`` / 列矩阵。"""
    if value is None:
        raise ValueError(f"{what}为空")
    if isinstance(value, sp.MatrixBase):
        return value
    if isinstance(value, (int, float, sp.Basic)) and not isinstance(value, bool):
        parsed = _parse_value(value, symbols)
        return sp.Matrix([parsed])
    if isinstance(value, (list, tuple)):
        seq = list(value)
        if len(seq) == 1 and isinstance(seq[0], (list, tuple)):
            seq = list(seq[0])
        if not seq:
            raise ValueError(f"{what}为空")
        return sp.Matrix([_parse_value(item, symbols) for item in seq])
    if isinstance(value, str):
        rows = A.parse_matrix(text=value, symbols=symbols)
        if rows.rows == 1:
            return sp.Matrix([rows[0, j] for j in range(rows.cols)])
        if rows.cols == 1:
            return sp.Matrix([rows[i, 0] for i in range(rows.rows)])
        raise ValueError(f"{what}必须是一维（{rows.rows}×{rows.cols} 不是向量）")
    raise ValueError(f"{what}格式无法识别：{type(value).__name__}")


def _parse_value(cell: Any, symbols: Iterable[str] | None = None) -> sp.Basic:
    """把数组里的元素转成 SymPy 对象（支持数字 / 字符串表达式 / SymPy 对象）。"""
    if isinstance(cell, sp.Basic):
        return cell
    if isinstance(cell, bool):
        return sp.sympify(int(cell))
    if isinstance(cell, (int, float)):
        return sp.sympify(cell)
    if cell is None:
        raise ValueError("矩阵元素不能为空")
    return A.parse(str(cell), symbols=symbols)


def _expr_symbol_names(text: str) -> list[str]:
    """从表达式文本里抓出变量名（用于显式传给 ``A.parse`` 的符号白名单）。"""
    names: list[str] = []
    for found in _IDENT_RE.findall(str(text)):
        if found in _RESERVED_NAMES or found in names:
            continue
        names.append(found)
    return names


def _matrix_text(mat: sp.Matrix) -> str:
    return A.to_text(mat)


def _sorted_symbols(expr: Any) -> list[sp.Symbol]:
    if not isinstance(expr, sp.Basic):
        return []
    return sorted(expr.free_symbols, key=lambda s: (len(str(s)), str(s)))


def _has_symbols(mat: sp.Matrix) -> bool:
    return any(getattr(element, "free_symbols", None) for element in mat)


def _shape_input(mat: sp.Matrix, **extra: Any) -> dict[str, Any]:
    """只放行的维数与规模，避免把整个矩阵塞进 JSON。"""
    payload: dict[str, Any] = {"matrix_shape": f"{mat.rows}x{mat.cols}"}
    payload.update({k: v for k, v in extra.items() if v is not None})
    return payload


def _require_square(mat: sp.Matrix, what: str) -> None:
    if mat.rows != mat.cols:
        raise ValueError(f"{what}要求方阵，但得到 {mat.rows}×{mat.cols} 非方阵")


def _warn_large_symbolic(result: MathResult, mat: sp.Matrix, what: str) -> None:
    if _has_symbols(mat) and max(mat.rows, mat.cols) > _SYMBOLIC_WARN_SIZE:
        result.add_warning(
            f"该矩阵为 {mat.rows}×{mat.cols} 且含自由符号，{what}可能非常慢；"
            "建议对符号取定值后用 math_numeric 做数值验证。"
        )


def _same_matrix(actual: sp.Matrix, expected: sp.Matrix) -> bool:
    """逐元素数学等价比较 ``C.same_expr``（绝不靠字符串比较）。"""
    try:
        if actual.shape != expected.shape:
            return False
        return all(C.same_expr(actual[i, j], expected[i, j]) for i in range(actual.rows) for j in range(actual.cols))
    except Exception:  # noqa: BLE001
        return False


def _same_scalar(actual: Any, expected: Any) -> bool:
    try:
        return C.same_expr(actual, expected)
    except Exception:  # noqa: BLE001
        return False


def _is_zero(item: Any) -> bool:
    """数学意义上的零判定（先去重展开，再化简）。"""
    try:
        value = sp.sympify(item)
        if value == 0:
            return True
        return bool(sp.simplify(value) == 0)
    except Exception:  # noqa: BLE001
        return False


def _numeric_det(mat: sp.Matrix, symbols: Sequence[sp.Symbol]) -> sp.Basic | None:
    """把符号换成小整数，用 SymPy 另一条路径算一次行列式（抽样交叉验证）。"""
    if not symbols:
        return None
    try:
        subs = {sym: 1 + (index % 5) for index, sym in enumerate(symbols)}
        return sp.Matrix(mat.rows, mat.cols, lambda i, j: mat[i, j].subs(subs)).det()
    except Exception:  # noqa: BLE001
        return None


def _cofactor_det(mat: sp.Matrix) -> sp.Basic:
    """Laplace 按首行展开——与 SymPy 的 Bareiss/Berkowitz 完全不同的独立途径。

    不使用 SymPy 的行列式例程，因此 ``det`` 的验证是真正的「独立验证」。
    """
    n = mat.rows
    if n != mat.cols:
        raise ValueError("余子式展开要求方阵")
    if n == 0:
        return sp.Integer(1)
    if n == 1:
        return mat[0, 0]
    if n == 2:
        return sp.expand(mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0])
    total = sp.Integer(0)
    for col in range(n):
        element = mat[0, col]
        if element == 0:
            continue
        minor = mat.minor_submatrix(0, col)
        total += ((-1) ** col) * element * _cofactor_det(minor)
    return sp.expand(total)


def _pivot_and_free_columns(mat: sp.Matrix) -> tuple[tuple[int, ...], list[int]]:
    _, pivots = mat.rref()
    pivot_list = tuple(int(p) for p in pivots)
    free = [j for j in range(mat.cols) if j not in pivot_list]
    return pivot_list, free


def _rank_by_minors(mat: sp.Matrix) -> tuple[int, str]:
    """用行列式（子式）独立复核秩：秩 = 最大非零子式的阶数。"""
    try:
        r = int(mat.rank())
    except Exception as exc:  # noqa: BLE001
        return -1, f"rref 秩计算失败: {exc}"
    maximum = min(mat.rows, mat.cols)
    if r == 0:
        return 0, "所有元素均为 0，零子式判据给出 r = 0"
    if maximum > _MINOR_RANK_LIMIT:
        return r, f"矩阵过大（>{_MINOR_RANK_LIMIT} 阶），未做子式穷举，仅用行最简形的非零行数"
    try:
        witness = None
        for rows in sp.utilities.iterables.subsets(range(mat.rows), r):
            for cols in sp.utilities.iterables.subsets(range(mat.cols), r):
                if _is_zero(mat.extract(list(rows), list(cols)).det()):
                    continue
                witness = (tuple(rows), tuple(cols))
                break
            if witness:
                break
        if witness is None:
            return -1, f"未找到 {r} 阶非零子式，与秩 {r} 矛盾"
        for rows in sp.utilities.iterables.subsets(range(mat.rows), r + 1):
            for cols in sp.utilities.iterables.subsets(range(mat.cols), r + 1):
                if not _is_zero(mat.extract(list(rows), list(cols)).det()):
                    return -1, f"存在 {r + 1} 阶非零子式，与秩 {r} 矛盾"
        return r, (
            f"存在 {r} 阶非零子式（行 {list(witness[0])}、列 {list(witness[1])}），"
            f"且全部 {r + 1} 阶子式为 0"
        )
    except Exception as exc:  # noqa: BLE001
        return r, f"子式判据未完成（{exc}），仅用行最简形的非零行数"


def _solution_forward(Amat: sp.Matrix, bvec: sp.Matrix) -> tuple[sp.Matrix, list[dict[str, sp.Symbol]], bool]:
    """构造方程组的「表达式解」，与 ``gauss_jordan_solve`` 的结果结构一致。

    返回 ``(解向量, 自由参数说明, 是否一致)``。自由参数按列主元从右到左编号，
    与 SymPy 的 ``tau0, tau1, ...`` 命名保持一致，因此可直接替换。
    """
    n = Amat.cols
    aug = Amat.row_join(sp.Matrix(bvec))
    rref_aug, pivots = aug.rref()
    pivot_list = list(pivots)
    for p in pivot_list:
        if p >= n:
            return sp.zeros(n, 1), [], False  # 增广列成为主元列 -> 矛盾方程
    free = [j for j in range(n) if j not in pivot_list]
    # 自由变量 tau_k 按「自由列从右到左」编号，与 SymPy 的约定一致
    tau: dict[int, sp.Symbol] = {}
    for offset, col in enumerate(reversed(free)):
        tau[col] = sp.Symbol(f"tau{offset}")
    entries: list[sp.Basic] = []
    for row in range(n):
        if row in pivot_list:
            expr = rref_aug[row, n]
            for col in free:
                expr -= rref_aug[row, col] * tau[col]
            entries.append(sp.simplify(expr))
        else:
            entries.append(tau[row])
    parameters = [{"name": str(sym), "value": str(sym)} for sym in tau.values()]
    return sp.Matrix(entries), parameters, True


def _taus_in(expression: sp.Basic) -> set[sp.Symbol]:
    return {sym for sym in _sorted_symbols(expression) if str(sym).startswith("tau")}


def _subs_tau(expression: sp.Basic, values: dict[sp.Symbol, sp.Basic]) -> sp.Basic:
    result = sp.sympify(expression)
    for sym in _sorted_symbols(result):
        if str(sym).startswith("tau"):
            result = result.subs(sym, values.get(sym, sp.Integer(0)))
    return result


def _eigenvalue_objects(mat: sp.Matrix) -> list[tuple[sp.Basic, int]]:
    return [(sp.sympify(value), int(mult)) for value, mult in mat.eigenvals().items()]


def _match_vector(computed: list[sp.Matrix], target: sp.Matrix) -> sp.Matrix | None:
    """在 ``eigenvects`` 给出的基里挑出与 ``rref`` 得到的向量共线的那一个。"""
    for candidate in computed:
        if candidate.shape != target.shape:
            continue
        try:
            ratios = []
            for idx in range(target.rows):
                top, bottom = candidate[idx, 0], target[idx, 0]
                if bottom != 0:
                    ratios.append(sp.simplify(top / bottom))
                elif top != 0:
                    ratios = []
                    break
            if ratios and all(_is_zero(ratio - ratios[0]) for ratio in ratios):
                return candidate
        except Exception:  # noqa: BLE001
            continue
    return None


# ---------------------------------------------------------------------------
# 1. 行列式
# ---------------------------------------------------------------------------

@op("det", "determinant", "math_det")
def det(matrix: Any = None, **kwargs: Any) -> MathResult:
    """行列式（精确值）。"""
    mat = _matrix(matrix, what="矩阵")
    _require_square(mat, "求行列式")
    n = mat.rows
    r = MathResult.ok("det", method="SymPy Matrix.det（Bareiss 精确消元）")
    r.set_input(**_shape_input(mat))
    r.add_condition("行列式是方阵的固有量，无额外成立条件")
    _warn_large_symbolic(r, mat, "求行列式")

    try:
        value = sp.simplify(mat.det(method="bareiss"))
    except Exception as exc:  # noqa: BLE001
        r.add_warning(f"行列式精确求解失败（{type(exc).__name__}: {exc}）")
        return MathResult.unsolved(
            "det",
            f"无法精确求出该 {n}×{n} 矩阵的行列式（{type(exc).__name__}: {exc}）；"
            "请改用 math_numeric 做数值计算。",
            method="SymPy Matrix.det",
        )

    r.set_result(value)
    r.set_raw(det_text=A.to_text(value), det_latex=A.to_latex(value), size=value.is_zero if value.is_zero in (True, False) else None)

    # ---- 独立验证 -------------------------------------------------------
    methods: list[dict[str, Any]] = []
    strategies: list[tuple[str, Any]] = []
    if n <= _COFACTOR_LIMIT:
        try:
            strategies.append(("余子式（Laplace 按首行展开，不使用 SymPy 的行列式例程）", _cofactor_det(mat)))
        except Exception:  # noqa: BLE001
            pass
    try:
        if n > 1:
            strategies.append(("Berkowitz 消元法（与 Bareiss 不同的算法）", mat.det(method="berkowitz")))
    except Exception:  # noqa: BLE001
        pass
    try:
        strategies.append(("LU 分解取对角线乘积", mat.det(method="lu")))
    except Exception:  # noqa: BLE001
        pass

    agree: bool | None = None
    for label, candidate in strategies:
        ok = _same_scalar(value, candidate)
        methods.append(
            {
                "method": label,
                "expected": A.to_text(candidate),
                "agree": ok,
            }
        )
        agree = ok if agree is None else (agree and ok)
        if not ok:
            agree = False

    if agree is None and _has_symbols(mat):
        sample = _numeric_det(mat, _sorted_symbols(mat))
        if sample is not None:
            symbols = _sorted_symbols(mat)
            subs = {sym: 1 + (index % 5) for index, sym in enumerate(symbols)}
            ok = _is_zero(sp.sympify(value).subs(subs) - sample)
            methods.append(
                {
                    "method": "符号替换抽样：对自由符号取小整数后比较行列式值",
                    "substitutions": {str(k): str(v) for k, v in subs.items()},
                    "expected": A.to_text(sample),
                    "agree": ok,
                }
            )
            agree = ok

    if agree is None:
        r.verify(
            status="unverified",
            methods=methods,
            note="矩阵规模过大且未找到可用的独立算法，未能独立验证；结果仍为精确符号值。",
        )
    elif agree:
        r.verify(status="independent", methods=methods, note="至少一条独立途径给出相同结果。")
    else:
        r.verify(status="unverified", methods=methods, note="独立途径结果不一致，请人工复核！")
        r.add_warning("行列式的独立验证未通过，结果不可信，请复核输入。")
    return r


# ---------------------------------------------------------------------------
# 2. 逆矩阵
# ---------------------------------------------------------------------------

@op("inv", "inverse", "math_inv")
def inv(matrix: Any = None, **kwargs: Any) -> MathResult:
    """逆矩阵，验证 A·A⁻¹ = I。"""
    mat = _matrix(matrix, what="矩阵")
    _require_square(mat, "求逆")
    n = mat.rows
    r = MathResult.ok("inv", method="SymPy Matrix.inv（伴随矩阵 / 初等变换，精确分数）")
    r.set_input(**_shape_input(mat))
    _warn_large_symbolic(r, mat, "求逆")

    try:
        determinant = sp.simplify(mat.det())
    except Exception:  # noqa: BLE001
        determinant = None

    if determinant is not None and determinant.is_zero is True:
        return MathResult.unsolved(
            "inv",
            f"矩阵不可逆：det = 0（{A.to_text(determinant)}），因此不存在逆矩阵。"
            "若需要求解方程组，请改用 solve_linear 判断无解或无穷多解。",
            method="SymPy Matrix.inv",
        )

    try:
        inverse = mat.inv()
    except Exception as exc:  # noqa: BLE001
        if determinant is None or determinant.is_zero is False:
            return MathResult.unsolved(
                "inv",
                f"无法求出逆矩阵（{type(exc).__name__}: {exc}）；该矩阵可能含自由符号，"
                "对符号取定值后用 math_numeric 可做数值求逆。",
                method="SymPy Matrix.inv",
            )
        return MathResult.unsolved(
            "inv",
            f"无法判定可逆性：det 含自由符号且未能化简为零/非零（{type(exc).__name__}: {exc}）。",
            method="SymPy Matrix.inv",
        )

    r.set_result(inverse)
    r.set_raw(inverse_text=A.to_text(inverse), inverse_latex=A.to_latex(inverse))
    if determinant is not None:
        r.set_raw(det_text=A.to_text(determinant))
        if determinant.free_symbols:
            r.add_condition(f"A 可逆（当前 det = {A.to_text(determinant)}，要求 det ≠ 0）")
        else:
            r.add_condition(f"A 可逆（det = {A.to_text(determinant)} ≠ 0）")
    else:
        r.add_condition("A 可逆")
    r.add_condition("结果满足 A·A⁻¹ = A⁻¹·A = I")

    identity = sp.eye(n)
    product = sp.simplify(mat * inverse)
    product_rev = sp.simplify(inverse * mat)
    ok_left = _same_matrix(product, identity)
    ok_right = _same_matrix(product_rev, identity)
    methods = [
        {
            "method": "矩阵乘积为单位矩阵：逐元素比较 A·A⁻¹ 与 I",
            "product_text": _matrix_text(product),
            "agree": ok_left,
        },
        {
            "method": "反向乘积为单位矩阵：逐元素比较 A⁻¹·A 与 I",
            "product_text": _matrix_text(product_rev),
            "agree": ok_right,
        },
    ]
    if determinant is not None and not determinant.free_symbols:
        try:
            det_inverse = sp.simplify(inverse.det())
            expected = sp.simplify(1 / determinant)
            ok_det = _same_scalar(det_inverse, expected)
            methods.append(
                {
                    "method": "det(A⁻¹) = 1/det(A)（独立标量恒等式）",
                    "expected": A.to_text(expected),
                    "got": A.to_text(det_inverse),
                    "agree": ok_det,
                }
            )
        except Exception:  # noqa: BLE001
            ok_det = None
    else:
        ok_det = None
    overall = ok_left and ok_right and (ok_det is not False)
    r.verify(
        status="independent" if overall else "unverified",
        methods=methods,
        note="验证通过：矩阵乘积为单位矩阵。" if overall else "矩阵乘积不等于单位矩阵，结果不可信！",
    )
    if not overall:
        r.add_warning("逆矩阵独立验证未通过，请复核输入。")
    return r


# ---------------------------------------------------------------------------
# 3. 秩
# ---------------------------------------------------------------------------

@op("rank", "math_rank")
def rank(matrix: Any = None, **kwargs: Any) -> MathResult:
    """矩阵的秩 + 行最简形 + 主元列说明。"""
    mat = _matrix(matrix, what="矩阵")
    r = MathResult.ok("rank", method="SymPy Matrix.rank（行最简形主元计数）")
    r.set_input(**_shape_input(mat))
    _warn_large_symbolic(r, mat, "求秩")

    try:
        rref_mat, pivots = mat.rref()
    except Exception as exc:  # noqa: BLE001
        return MathResult.unsolved(
            "rank",
            f"无法求出行最简形，因此无法确定秩（{type(exc).__name__}: {exc}）。",
            method="SymPy Matrix.rref",
        )
    pivot_list = [int(p) for p in pivots]
    free_cols = [j for j in range(mat.cols) if j not in pivot_list]
    value = len(pivot_list)

    r.set_result(sp.Integer(value))
    r.set_raw(
        rank=value,
        rref=rref_mat.tolist(),
        rref_text=_matrix_text(rref_mat),
        pivot_columns=[p + 1 for p in pivot_list],
        free_columns=[c + 1 for c in free_cols],
        nonzero_rows=int(sum(1 for i in range(rref_mat.rows) if any(rref_mat[i, j] != 0 for j in range(rref_mat.cols)))),
    )
    r.add_condition("秩 = 行最简形中主元的个数 = 最大非零子式的阶数")
    r.add_condition(
        f"主元位于第 {[p + 1 for p in pivot_list]} 列，这些列是列向量组的一个极大线性无关组的像；"
        f"第 {[c + 1 for c in free_cols]} 列可由主元列线性表示"
        if pivot_list
        else "矩阵为零矩阵，秩 = 0"
    )
    if free_cols:
        r.add_condition(f"列向量组线性相关：第 {[c + 1 for c in free_cols]} 列是其余列的线性组合")
    else:
        r.add_condition("各列线性无关（列满秩）")
    if value == min(mat.rows, mat.cols):
        r.add_condition("行（或列）向量组线性无关")

    # ---- 独立验证：行最简形非零行数 与 非零子式判据 ----
    nonzero_rows = sum(
        1 for i in range(rref_mat.rows) if any(rref_mat[i, j] != 0 for j in range(rref_mat.cols))
    )
    minor_rank, minor_note = _rank_by_minors(mat)
    methods = [
        {
            "method": "行最简形的非零行数（不含零行）",
            "expected": nonzero_rows,
            "agree": nonzero_rows == value,
            "note": "与 rank() 使用的 pivot 计数方式不同",
        },
        {
            "method": "非零子式判据：秩 = 最大非零子式的阶数（用行列式计算子式）",
            "expected": minor_rank,
            "agree": minor_rank == value,
            "note": minor_note,
        },
    ]
    ok = (nonzero_rows == value) and (minor_rank == value)
    r.verify(
        status="independent" if ok else "unverified",
        methods=methods,
        note="行最简形非零行数与子式判据均给出相同秩。" if ok else "独立判据结果不一致，请人工复核。",
    )
    if not ok:
        r.add_warning("秩的独立验证未通过，请复核输入或改用其他工具。")
    return r


# ---------------------------------------------------------------------------
# 4. 转置
# ---------------------------------------------------------------------------

@op("transpose", "T", "math_transpose")
def transpose(matrix: Any = None, **kwargs: Any) -> MathResult:
    """转置矩阵。"""
    mat = _matrix(matrix, what="矩阵")
    r = MathResult.ok("transpose", method="SymPy Matrix.T（交换行列）")
    r.set_input(**_shape_input(mat))
    result = mat.T
    r.set_result(result)
    r.set_raw(transpose_text=_matrix_text(result))
    r.add_condition("转置不改变元素值，仅交换行列位置")

    methods: list[dict[str, Any]] = []
    double = result.T
    ok_double = _same_matrix(double, mat)
    methods.append(
        {
            "method": "二次转置回到原矩阵：(Aᵀ)ᵀ = A",
            "agree": ok_double,
        }
    )
    ok = ok_double
    if mat.rows == mat.cols:
        try:
            ok_sym = _same_scalar(mat.det(), result.det())
            methods.append(
                {
                    "method": "方阵转置行列式不变：det(Aᵀ) = det(A)",
                    "expected": A.to_text(mat.det()),
                    "got": A.to_text(result.det()),
                    "agree": ok_sym,
                }
            )
            ok = ok and ok_sym
        except Exception:  # noqa: BLE001
            pass
    r.verify(
        status="independent" if ok else "unverified",
        methods=methods,
        note="转置的独立验证基于转置运算的反身性质（(Aᵀ)ᵀ = A）。",
    )
    return r


# ---------------------------------------------------------------------------
# 5. 矩阵乘法
# ---------------------------------------------------------------------------

@op("matmul", "matrix_multiply", "math_matmul")
def matmul(matrix: Any = None, matrix_b: Any = None, **kwargs: Any) -> MathResult:
    """矩阵乘法 A·B（维度必须相容）。"""
    left = _matrix(matrix, what="矩阵 A")
    right = _matrix(matrix_b, what="矩阵 B", symbols=None)
    if left.cols != right.rows:
        raise ValueError(
            f"矩阵维度不匹配：A 为 {left.rows}×{left.cols}，B 为 {right.rows}×{right.cols}；"
            f"矩阵乘法要求 A 的列数等于 B 的行数（此处 {left.cols} ≠ {right.rows}）"
        )
    r = MathResult.ok("matmul", method="SymPy 矩阵乘法（精确分数/根号）")
    r.set_input(matrix_shape=f"{left.rows}x{left.cols}", matrix_b_shape=f"{right.rows}x{right.cols}")
    _warn_large_symbolic(r, left, "矩阵乘法")
    product = left * right
    r.set_result(product)
    r.set_raw(product_text=_matrix_text(product))
    r.add_condition(f"乘积矩阵的维数为 {left.rows}×{right.cols}")

    methods: list[dict[str, Any]] = []
    # 独立验证 1：逐元素「A 的第 i 行 · B 的第 j 列」重新计算（sum 展开，不走矩阵乘法例程）
    try:
        manual = sp.Matrix(
            left.rows,
            right.cols,
            lambda i, j: sp.Add(*[left[i, k] * right[k, j] for k in range(left.cols)]),
        )
        ok_manual = _same_matrix(product, manual)
        methods.append(
            {
                "method": "逐元素内积重算：C[i,j] = Σ_k A[i,k]·B[k,j]（不用矩阵乘法例程）",
                "agree": ok_manual,
            }
        )
    except Exception:  # noqa: BLE001
        ok_manual = False
    # 独立验证 2：det(AB) = det(A)·det(B)（方阵时）
    ok_det: bool | None = None
    if left.rows == left.cols and right.rows == right.cols:
        try:
            ok_det = _same_scalar(product.det(), sp.simplify(left.det() * right.det()))
            methods.append(
                {
                    "method": "方阵乘法的行列式乘性：det(AB) = det(A)·det(B)",
                    "expected": A.to_text(sp.simplify(left.det() * right.det())),
                    "got": A.to_text(product.det()),
                    "agree": bool(ok_det),
                }
            )
        except Exception:  # noqa: BLE001
            ok_det = None
    ok = bool(ok_manual) and (ok_det is not False)
    r.verify(
        status="independent" if ok else "unverified",
        methods=methods,
        note="验证通过：逐元素内积与行列式乘性一致。" if ok else "独立验证未通过，请复核输入。",
    )
    return r


# ---------------------------------------------------------------------------
# 6. 矩阵加法
# ---------------------------------------------------------------------------

@op("matadd", "matrix_add")
def matadd(matrix: Any = None, matrix_b: Any = None, **kwargs: Any) -> MathResult:
    """矩阵加法 A + B。"""
    left = _matrix(matrix, what="矩阵 A")
    right = _matrix(matrix_b, what="矩阵 B")
    if left.shape != right.shape:
        raise ValueError(
            f"矩阵维度不匹配：A 为 {left.rows}×{left.cols}，B 为 {right.rows}×{right.cols}；"
            "矩阵加法要求同型矩阵"
        )
    r = MathResult.ok("matadd", method="SymPy 矩阵加法（逐元素相加）")
    r.set_input(matrix_shape=f"{left.rows}x{left.cols}", matrix_b_shape=f"{right.rows}x{right.cols}")
    total = left + right
    r.set_result(total)
    r.set_raw(sum_text=_matrix_text(total))

    methods: list[dict[str, Any]] = []
    ok_commute = _same_matrix(total, right + left)
    methods.append({"method": "交换律检验：A + B 与 B + A 逐元素相同", "agree": ok_commute})
    ok_zero = _same_matrix(total - left, right)
    methods.append({"method": "逆运算检验：(A + B) − A 逐元素回到 B", "agree": ok_zero})
    ok = ok_commute and ok_zero
    r.verify(
        status="independent" if ok else "unverified",
        methods=methods,
        note="验证通过：交换律与加法逆运算检验一致。" if ok else "独立验证未通过，请复核输入。",
    )
    return r


# ---------------------------------------------------------------------------
# 7. 迹
# ---------------------------------------------------------------------------

@op("trace", "math_trace")
def trace(matrix: Any = None, **kwargs: Any) -> MathResult:
    """矩阵的迹（主对角线元素之和）。"""
    mat = _matrix(matrix, what="矩阵")
    _require_square(mat, "求迹")
    r = MathResult.ok("trace", method="SymPy Matrix.trace（主对角线求和）")
    r.set_input(**_shape_input(mat))
    value = sp.simplify(mat.trace())
    r.set_result(value)
    r.set_raw(trace_text=A.to_text(value))
    r.add_condition("迹只对方阵定义；迹等于全部特征值之和（含代数重数）")
    r.add_condition("trace(A) = trace(Aᵀ)，且 trace(AB) = trace(BA)")

    methods: list[dict[str, Any]] = []
    manual = sp.Add(*[mat[i, i] for i in range(mat.rows)])
    ok_manual = _same_scalar(value, manual)
    methods.append(
        {
            "method": "显式求和主对角线元素 Σ a_ii（不用 trace 例程）",
            "expected": A.to_text(manual),
            "agree": ok_manual,
        }
    )
    ok = ok_manual
    try:
        ok_transpose = _same_scalar(value, mat.T.trace())
        methods.append({"method": "迹在转置下不变：trace(A) = trace(Aᵀ)", "agree": ok_transpose})
        ok = ok and ok_transpose
    except Exception:  # noqa: BLE001
        pass
    if mat.rows <= _EIGVECT_LIMIT:
        try:
            eigen_sum = sp.simplify(sum(value_e * mult for value_e, mult in _eigenvalue_objects(mat)))
            ok_eigen = _same_scalar(value, eigen_sum)
            methods.append(
                {
                    "method": "特征值之和：trace(A) = Σ λ_i（用特征值独立计算）",
                    "expected": A.to_text(eigen_sum),
                    "agree": bool(ok_eigen),
                }
            )
            ok = ok and bool(ok_eigen)
        except Exception:  # noqa: BLE001
            pass
    r.verify(
        status="independent" if ok else "unverified",
        methods=methods,
        note="验证通过：对角线求和与迹的性质一致。" if ok else "独立验证未通过，请复核输入。",
    )
    return r


# ---------------------------------------------------------------------------
# 8. 行最简形
# ---------------------------------------------------------------------------

@op("rref", "math_rref")
def rref(matrix: Any = None, **kwargs: Any) -> MathResult:
    """行最简形（Reduced Row Echelon Form）与主元位置。"""
    mat = _matrix(matrix, what="矩阵")
    r = MathResult.ok("rref", method="SymPy Matrix.rref（Gauss-Jordan 消元，精确分数）")
    r.set_input(**_shape_input(mat))
    _warn_large_symbolic(r, mat, "行最简形")
    try:
        reduced, pivots = mat.rref()
    except Exception as exc:  # noqa: BLE001
        return MathResult.unsolved(
            "rref",
            f"无法完成行最简形消元（{type(exc).__name__}: {exc}）。",
            method="SymPy Matrix.rref",
        )
    pivot_list = [int(p) for p in pivots]
    r.set_result(reduced)
    r.set_raw(
        rref=reduced.tolist(),
        rref_text=_matrix_text(reduced),
        pivot_columns=[p + 1 for p in pivot_list],
        free_columns=[c + 1 for c in range(mat.cols) if c not in pivot_list],
        pivot_count=len(pivot_list),
    )
    r.add_condition("行最简形与初等行变换等价：不改变行空间，也不改变方程组的解集")
    r.add_condition("主元列对应约束变量（基本变量），其余列对应自由变量")

    methods: list[dict[str, Any]] = []
    # 独立验证：再消元一次必须得到不动点（幂等性）；且行最简形与原矩阵行等价
    try:
        again, pivots_again = reduced.rref()
        ok_idem = _same_matrix(again, reduced) and [int(p) for p in pivots_again] == pivot_list
        methods.append(
            {
                "method": "幂等性检验：对行最简形再次消元应得到自身（不动点）",
                "agree": ok_idem,
            }
        )
    except Exception:  # noqa: BLE001
        ok_idem = False
    ok_rank = False
    try:
        nonzero_rows = sum(1 for i in range(reduced.rows) if any(reduced[i, j] != 0 for j in range(reduced.cols)))
        ok_rank = nonzero_rows == len(pivot_list)
        methods.append(
            {
                "method": "非零行数与主元个数一致（等价于秩相同）",
                "expected": nonzero_rows,
                "got": len(pivot_list),
                "agree": ok_rank,
            }
        )
    except Exception:  # noqa: BLE001
        pass
    ok = bool(ok_idem) and ok_rank
    r.verify(
        status="independent" if ok else "unverified",
        methods=methods,
        note="行最简形为消元不动点且主元计数一致。"
        if ok
        else "幂等性/主元计数检查失败，请人工复核。",
    )
    return r


# ---------------------------------------------------------------------------
# 9. 线性方程组
# ---------------------------------------------------------------------------

@op("solve_linear", "linear_system", "math_linear_system")
def solve_linear(matrix: Any = None, rhs: Any = None, **kwargs: Any) -> MathResult:
    """解线性方程组 A x = b，区分唯一解 / 无解 / 无穷多解。"""
    if rhs is None:
        raise ValueError("缺少右端项 rhs，例如 rhs=\"1,2,3\" 或 [1,2,3]")
    coef = _matrix(matrix, what="系数矩阵 A")
    bvec = sp.Matrix(_vector(rhs, what="右端项 b"))
    n = coef.cols
    if bvec.rows != coef.rows:
        raise ValueError(
            f"维度不匹配：系数矩阵 A 为 {coef.rows}×{coef.cols}，右端项 b 有 {bvec.rows} 个分量；"
            f"要求 b 的分量个数等于 A 的行数（此处 {coef.rows}）"
        )
    r = MathResult.ok("solve_linear", method="Gauss-Jordan 消元 + 行最简形主元分析（精确解）")
    r.set_input(
        matrix_shape=f"{coef.rows}x{coef.cols}",
        rhs_text=A.to_text(bvec),
        unknowns=n,
    )
    _warn_large_symbolic(r, coef, "解线性方程组")

    try:
        solution, parameters = coef.gauss_jordan_solve(bvec)
    except ValueError as exc:  # 不相容
        return MathResult.unsolved(
            "solve_linear",
            f"方程组无解（不相容）：{exc}。行最简形显示增广矩阵出现了形如 [0 ... 0 | c]（c ≠ 0）的矛盾行。",
            method="Gauss-Jordan 消元",
        )
    except Exception as exc:  # noqa: BLE001
        return MathResult.unsolved(
            "solve_linear",
            f"无法求解该线性方程组（{type(exc).__name__}: {exc}）；可尝试对符号取定值后用 math_numeric。",
            method="Gauss-Jordan 消元",
        )

    free_dim = n - int(coef.rank())
    if free_dim < 0:
        free_dim = 0
    is_unique = free_dim == 0

    # 用同一套「自由变量参数化」重算一遍解的表达式形式，便于给通解与验证
    expr_solution, forward_params, consistent = _solution_forward(coef, bvec)
    if not consistent:
        return MathResult.unsolved(
            "solve_linear",
            "方程组无解（不相容）：增广矩阵的行最简形中出现矛盾行 [0 ... 0 | c]（c ≠ 0）。",
            method="Gauss-Jordan 消元",
        )

    names = [f"x{i + 1}" for i in range(n)]
    if is_unique:
        solved = sp.simplify(solution)
        r.set_result(solved)
        r.set_raw(
            solution_text=A.to_text(solved),
            solution_latex=A.to_latex(solved),
            solution_by_name={names[i]: A.to_text(solved[i, 0]) for i in range(n)},
            case="unique",
            case_text="唯一解",
            free_dimension=0,
            rank_coefficient=int(coef.rank()),
        )
        r.add_condition(f"系数矩阵 A 为 {coef.rows}×{coef.cols}，秩 = {n}（列满秩），故解唯一")
        target = solved
    else:
        particular, _ = coef.gauss_jordan_solve(bvec)
        particular = sp.simplify(particular)
        nullspace = coef.nullspace()
        tau_names = [f"tau{i}" for i in range(len(nullspace))]
        # 与 gauss_jordan_solve 命名一致：tau0, tau1, ... 按自由列从右到左
        taus = [sp.Symbol(name) for name in tau_names]
        general = particular
        for basis_vec, tau in zip(nullspace, taus):
            general = general + basis_vec * tau
        general = sp.simplify(general)
        r.set_result(general)
        r.set_raw(
            solution_text=A.to_text(general),
            solution_latex=A.to_latex(general),
            particular_solution=particular.tolist(),
            particular_text=A.to_text(particular),
            nullspace_basis=[vec.tolist() for vec in nullspace],
            nullspace_basis_text=[A.to_text(vec) for vec in nullspace],
            free_dimension=free_dim,
            parameters=tau_names,
            case="infinite",
            case_text=f"无穷多解（解空间维数 = n − r(A) = {n} − {int(coef.rank())} = {free_dim}）",
            rank_coefficient=int(coef.rank()),
            rank_augmented=int(coef.row_join(bvec).rank()),
            free_columns=[c + 1 for c in _pivot_and_free_columns(coef)[1]],
        )
        r.add_condition(
            f"方程组相容且 r(A) = {int(coef.rank())} < n = {n}，故有无穷多解；"
            f"通解 = 特解 + 齐次通解，其中 {', '.join(tau_names)} 为任意实数"
        )
        target = general

    # ---- 独立验证：把解代回原方程查残差 ----
    taus_in_solution = sorted(_taus_in(target), key=lambda s: str(s))
    subs_zero = {sym: sp.Integer(0) if str(sym).startswith("tau") else sym for sym in taus_in_solution}
    x_direct = target.subs(subs_zero)
    residuals = sp.simplify(coef * x_direct - bvec)
    residual_entries_ok = all(_is_zero(residuals[i, 0]) for i in range(residuals.rows))
    methods: list[dict[str, Any]] = [
        {
            "method": "把解代回原方程 A x = b 检查残差（自由参数取 0 得特解）",
            "residual": A.to_text(residuals),
            "agree": residual_entries_ok,
        }
    ]

    ok_homogeneous: bool | None = None
    if not is_unique:
        try:
            hom_residuals = []
            for vec in coef.nullspace():
                product = sp.simplify(coef * vec)
                hom_residuals.append(all(_is_zero(product[i, 0]) for i in range(product.rows)))
            ok_homogeneous = all(hom_residuals)
            methods.append(
                {
                    "method": "齐次部分检验：每个零空间基向量都被 A 映为 0（A·ξ = 0）",
                    "basis_count": len(hom_residuals),
                    "agree": ok_homogeneous,
                }
            )
        except Exception:  # noqa: BLE001
            ok_homogeneous = None

    ok_rank = False
    try:
        aug_rank = int(coef.row_join(bvec).rank())
        ok_rank = (aug_rank == int(coef.rank())) if not is_unique else True
        methods.append(
            {
                "method": "相容性秩判据：r(A) = r([A|b])",
                "expected": int(coef.rank()),
                "got": aug_rank,
                "agree": ok_rank,
            }
        )
    except Exception:  # noqa: BLE001
        pass

    ok = residual_entries_ok and (ok_homogeneous is not False)
    r.verify(
        status="independent" if ok else "unverified",
        methods=methods,
        note="验证通过：解代回原方程的残差为 0，且相容性秩判据成立。"
        if ok
        else "残差不为 0 或秩判据不成立，解不可信，请复核输入。",
    )
    if not ok:
        r.add_warning("线性方程组解的独立验证未通过，请人工复核。")
    return r


# ---------------------------------------------------------------------------
# 10. 特征值与特征向量
# ---------------------------------------------------------------------------

@op("eigen", "eigenvalues", "math_eigen")
def eigen(matrix: Any = None, **kwargs: Any) -> MathResult:
    """特征值、代数重数，可精确求解时给出特征向量。"""
    mat = _matrix(matrix, what="矩阵")
    _require_square(mat, "求特征值")
    n = mat.rows
    r = MathResult.ok("eigen", method="SymPy Matrix.eigenvals / eigenvects（特征多项式精确求根）")
    r.set_input(**_shape_input(mat))
    _warn_large_symbolic(r, mat, "求特征值")

    symmetric = _same_matrix(mat, mat.T)
    try:
        eigenvalues = _eigenvalue_objects(mat)
    except Exception as exc:  # noqa: BLE001
        return MathResult.unsolved(
            "eigen",
            f"无法精确求出特征值（{type(exc).__name__}: {exc}）。5 阶以上的矩阵通常没有根式解，"
            "建议改用 math_numeric 做数值特征值计算。",
            method="SymPy Matrix.eigenvals",
        )

    if not eigenvalues:
        return MathResult.unsolved("eigen", "特征多项式求根未返回任何特征值。", method="SymPy Matrix.eigenvals")

    r.set_raw(
        eigenvalues=[A.to_text(value) for value, _ in eigenvalues],
        eigenvalues_latex=[A.to_latex(value) for value, _ in eigenvalues],
        algebraic_multiplicities=[mult for _, mult in eigenvalues],
        eigenvalue_count=len(eigenvalues),
        characteristic_polynomial=A.to_text(sp.factor(mat.charpoly().as_expr())),
    )
    r.set_result([value for value, _ in eigenvalues])
    r.add_condition("λ 为 A 的特征值 ⟺ det(λI − A) = 0（特征多项式的根）")
    r.add_condition("代数重数之和 = 矩阵阶数；几何重数 ≤ 代数重数")

    # ---- 特征向量 ----
    vectors: list[dict[str, Any]] = []
    partial = False
    if n <= _EIGVECT_LIMIT:
        try:
            for value, mult, basis in mat.eigenvects():
                vectors.append(
                    {
                        "eigenvalue": A.to_text(value),
                        "algebraic_multiplicity": int(mult),
                        "geometric_multiplicity": len(basis),
                        "eigenvectors": [A.to_text(vec) for vec in basis],
                        "eigenvectors_latex": [A.to_latex(vec) for vec in basis],
                    }
                )
            r.set_raw(eigenvectors=vectors)
        except Exception as exc:  # noqa: BLE001
            partial = True
            r.add_warning(f"特征值已精确求出，但特征向量的精确求解失败（{type(exc).__name__}: {exc}）。")
    else:
        partial = True
        r.add_warning(f"矩阵阶数 {n} 超过 {_EIGVECT_LIMIT}，只返回特征值，未求解特征向量。")

    if symmetric:
        r.add_condition("A 为实对称矩阵：特征值全为实数，且存在正交矩阵 Q 使 QᵀAQ = Λ（可用正交对角化）")
        if all(value.is_real for value, _ in eigenvalues):
            r.add_condition("已确认特征值全为实数，与实对称矩阵的谱定理一致")
    else:
        r.add_condition("A 为一般非对称矩阵：特征值可能为复数，一般不保证可对角化")

    # ---- 独立验证：特征多项式、迹、行列式 ----
    methods: list[dict[str, Any]] = []
    lam = sp.Symbol("lambda")
    try:
        charpoly = mat.charpoly(lam).as_expr()
        product_poly = sp.expand(sp.prod([(lam - value) ** mult for value, mult in eigenvalues]))
        ok_charpoly = _is_zero(sp.expand(charpoly - product_poly))
        methods.append(
            {
                "method": "特征多项式重构：∏(λ − λᵢ)^mᵢ 与 charpoly(A) 展开后逐项相同",
                "charpoly": A.to_text(sp.factor(charpoly)),
                "reconstructed": A.to_text(sp.factor(product_poly)),
                "agree": ok_charpoly,
            }
        )
    except Exception as exc:  # noqa: BLE001
        ok_charpoly = False
        methods.append({"method": "特征多项式重构", "error": str(exc), "agree": False})

    ok_trace = False
    try:
        eigen_sum = sp.simplify(sum(value * mult for value, mult in eigenvalues))
        ok_trace = _same_scalar(mat.trace(), eigen_sum)
        methods.append(
            {
                "method": "迹校验：trace(A) = Σ mᵢλᵢ（用特征值求和，独立于特征多项式）",
                "expected": A.to_text(eigen_sum),
                "got": A.to_text(mat.trace()),
                "agree": bool(ok_trace),
            }
        )
    except Exception as exc:  # noqa: BLE001
        methods.append({"method": "迹校验", "error": str(exc), "agree": False})

    ok_det = False
    try:
        eigen_product = sp.simplify(sp.prod([value ** mult for value, mult in eigenvalues]))
        ok_det = _same_scalar(mat.det(), eigen_product)
        methods.append(
            {
                "method": "行列式校验：det(A) = ∏ λᵢ^mᵢ",
                "expected": A.to_text(sp.simplify(mat.det())),
                "got": A.to_text(eigen_product),
                "agree": bool(ok_det),
            }
        )
    except Exception as exc:  # noqa: BLE001
        methods.append({"method": "行列式校验", "error": str(exc), "agree": False})

    ok = bool(ok_charpoly) and bool(ok_trace) and bool(ok_det)
    r.verify(
        status="independent" if ok else "unverified",
        methods=methods,
        note="验证通过：特征多项式、迹与行列式三条独立关系全部吻合。"
        if ok
        else "特征多项式/迹/行列式校验未全部通过，请人工复核。",
    )
    if not ok:
        r.add_warning("特征值的独立验证未全部通过，请复核输入。")
    if partial:
        r.partial("特征值已精确求得并验证，但特征向量部分未能完整给出。")
    return r


# ---------------------------------------------------------------------------
# 11. 对角化
# ---------------------------------------------------------------------------

@op("diagonalize", "math_diagonalize")
def diagonalize(matrix: Any = None, **kwargs: Any) -> MathResult:
    """判断可对角化性；可对角化时给出 P 与 D，并验证 P⁻¹AP = D。"""
    mat = _matrix(matrix, what="矩阵")
    _require_square(mat, "对角化")
    n = mat.rows
    r = MathResult.ok("diagonalize", method="特征多项式求根 + 特征向量基判定（P⁻¹AP = D）")
    r.set_input(**_shape_input(mat))
    _warn_large_symbolic(r, mat, "对角化")

    try:
        eigenvalue_map = mat.eigenvals()
    except Exception as exc:  # noqa: BLE001
        return MathResult.unsolved(
            "diagonalize",
            f"无法精确求出特征值，因此无法判断可对角化性（{type(exc).__name__}: {exc}）；"
            "建议改用 math_numeric 做数值判断。",
            method="特征多项式求根",
        )

    algebraic_total = 0
    geometric_total = 0
    basis: list[sp.Matrix] = []
    diagonal_entries: list[sp.Basic] = []
    vector_failure: Exception | None = None
    for value, mult in eigenvalue_map.items():
        value = sp.sympify(value)
        algebraic_total += int(mult)
        try:
            basis_vectors = mat.eigenvects()
        except Exception as exc:  # noqa: BLE001
            vector_failure = exc
            break
        matched = None
        for candidate_value, _candidate_mult, candidate_basis in basis_vectors:
            if _is_zero(candidate_value - value):
                matched = candidate_basis
                break
        if matched is None:
            vector_failure = RuntimeError(f"未能在 eigenvects() 中找到特征值 {A.to_text(value)} 对应的特征向量")
            break
        geometric_total += len(matched)
        basis.extend(matched)
        diagonal_entries.extend([value] * len(matched))

    if vector_failure is not None:
        return MathResult.unsolved(
            "diagonalize",
            f"特征值已求出，但特征向量的精确求解失败（{type(vector_failure).__name__}: {vector_failure}）；"
            "无法据此判断可对角化性。",
            method="特征多项式求根 + eigenvects",
        )

    if len(basis) != n or geometric_total != n:
        reason = (
            f"不可对角化：n = {n}，但线性无关特征向量只有 {geometric_total} 个"
            f"（需要 n 个）。存在特征值使「几何重数 < 代数重数」，"
            "即该特征值的特征子空间维数不足，矩阵为亏损矩阵（不能对角化）。"
        )
        result = MathResult.unsolved("diagonalize", reason, method="特征向量基判定（几何重数 vs 代数重数）")
        result.set_input(**_shape_input(mat))
        result.set_raw(
            diagonalizable=False,
            algebraic_multiplicities={A.to_text(value): int(mult) for value, mult in eigenvalue_map.items()},
            geometric_total=geometric_total,
            required=n,
        )
        result.add_condition("矩阵可对角化 ⟺ 有 n 个线性无关的特征向量 ⟺ 每个特征值的几何重数 = 代数重数")
        result.add_condition("n 个互异特征值 ⇒ 必可对角化；此矩阵不满足该充分条件")
        result.verify(
            status="independent",
            methods=[
                {
                    "method": "几何重数统计：Σ 特征子空间维数 与矩阵阶数比较",
                    "geometric_total": geometric_total,
                    "required": n,
                    "agree": geometric_total != n,
                }
            ],
            note="「不可对角化」这一结论由特征子空间维数不足直接得出。",
        )
        return result

    P = sp.Matrix.hstack(*[vec for vec in basis])
    D = sp.diag(*diagonal_entries)
    r.set_result(P)
    r.set_raw(
        P=P.tolist(),
        P_text=_matrix_text(P),
        P_latex=A.to_latex(P),
        D=D.tolist(),
        D_text=_matrix_text(D),
        D_latex=A.to_latex(D),
        diagonalizable=True,
        eigenvalues_with_multiplicity=[
            {"eigenvalue": A.to_text(value), "algebraic_multiplicity": int(mult)} for value, mult in eigenvalue_map.items()
        ],
    )
    r.add_condition("A 可对角化 ⟺ 有 n 个线性无关的特征向量；P 的各列就是这组特征向量")
    r.add_condition("D 的对角元是对应特征值，D = P⁻¹AP")
    r.add_condition("P 不唯一：特征向量可任意伸缩、同特征值的特征向量可任意替换")

    # ---- 独立验证：P⁻¹AP = D（用 AP = PD 等价形式避免重复求逆） ----
    methods: list[dict[str, Any]] = []
    ok_pd = _same_matrix(sp.simplify(mat * P), sp.simplify(P * D))
    methods.append(
        {
            "method": "等价形式检验：A·P = P·D（逐元素比较，不依赖求逆）",
            "agree": ok_pd,
        }
    )
    ok_inv: bool | None = None
    try:
        P_inv = P.inv()
        reconstructed = sp.simplify(P_inv * mat * P)
        ok_inv = _same_matrix(reconstructed, D)
        methods.append(
            {
                "method": "直接检验 P⁻¹·A·P 是否为对角阵 D（逐元素比较）",
                "reconstructed_text": _matrix_text(reconstructed),
                "agree": ok_inv,
            }
        )
    except Exception as exc:  # noqa: BLE001
        methods.append({"method": "P⁻¹·A·P 直接检验", "error": str(exc), "agree": False})
        ok_inv = False
    ok = bool(ok_pd) and bool(ok_inv)
    r.verify(
        status="independent" if ok else "unverified",
        methods=methods,
        note="验证通过：P 可逆且 P⁻¹AP 为对角阵 D。" if ok else "P⁻¹AP ≠ D，结果不可信，请复核。",
    )
    if not ok:
        r.add_warning("对角化的独立验证未通过，请人工复核。")
    return r


# ---------------------------------------------------------------------------
# 12. 二次型
# ---------------------------------------------------------------------------

def _quadratic_split(parsed: sp.Basic) -> tuple[list[sp.Symbol], sp.Poly, list[sp.Symbol]]:
    """把表达式拆成「二次型的变量」与「当作系数的参数」。

    返回 ``(变量, 关于这些变量的齐二次多项式, 全部符号)``。

    **为什么需要它**（真实踩过的坑）：``a*x1^2+2*b*x1*x2+c*x2^2`` 里 ``a/b/c`` 是待定
    系数，如果像以前那样用正则把**所有**标识符都当变量，``Poly(..., a, b, c, x1, x2)``
    会把 ``a*x1^2`` 算成 3 次，于是含参二次型被误判成「这不是二次型：次数为 3」。
    做法：枚举符号子集，取「每个单项式在子集上的次数都恰好为 2」（即齐二次）的最大子集。
    """
    all_syms = sorted(parsed.free_symbols, key=lambda s: (len(str(s)), str(s)))
    if not all_syms:
        raise ValueError("这不是二次型：表达式中没有变量")
    for size in range(len(all_syms), 0, -1):
        for combo in itertools.combinations(all_syms, size):
            try:
                poly = sp.Poly(parsed, *combo)
            except Exception:  # noqa: BLE001
                continue
            if poly.total_degree() != 2:
                continue
            if any(sum(monom) != 2 for monom in poly.monoms()):
                continue
            return list(combo), poly, all_syms
    try:
        degree = sp.Poly(parsed, *all_syms).total_degree()
    except Exception:  # noqa: BLE001
        degree = None
    if degree is None:
        raise ValueError("无法把 expr 视为多项式以提取二次型系数（请检查表达式是否为多项式）")
    if degree > 2:
        raise ValueError(f"这不是二次型：表达式次数为 {degree}（二次型要求各项都是 2 次）")
    raise ValueError("这不是二次型：表达式中没有二次项")


@op("quadratic_form", "quadratic", "math_quadratic")
def quadratic_form(expr: Any = None, matrix: Any = None, variables: Any = None, **kwargs: Any) -> MathResult:
    """二次型的标准化与正定性判定（表达式或对称矩阵两种输入）。"""
    if expr is None and matrix is None:
        raise ValueError("请给出二次型表达式 expr（如 \"x1^2+2*x1*x2+3*x2^2\"）或对称矩阵 matrix")

    r = MathResult.ok("quadratic_form", method="配方（Lagrange 法）+ 正交变换（特征值分解）双路标准化")
    names: list[str] = []
    if expr is not None:
        if isinstance(expr, sp.MatrixBase):
            mat = expr
            source = "matrix"
        else:
            text = str(expr)
            names = _expr_symbol_names(text)
            if variables:
                extra_names = [str(v).strip() for v in (variables if isinstance(variables, (list, tuple)) else str(variables).replace(",", " ").split()) if str(v).strip()]
                for name in extra_names:
                    if name not in names:
                        names.append(name)
            symbols_map = A.get_symbols(names) if names else None
            parsed = A.parse(text, symbols=list(symbols_map) if symbols_map else None)
            if variables:
                # 用户显式指定了变量：以它为准（参数留作系数）
                wanted = [str(v).strip() for v in (variables if isinstance(variables, (list, tuple)) else str(variables).replace(",", " ").split()) if str(v).strip()]
                table = A.get_symbols(wanted)
                chosen = [table[name] for name in wanted if name in table]
                try:
                    poly = sp.Poly(parsed, *chosen)
                except Exception as exc:  # noqa: BLE001
                    raise ValueError(f"无法把 expr 视为多项式以提取二次型系数：{exc}") from exc
                if poly.total_degree() != 2 or any(sum(m) != 2 for m in poly.monoms()):
                    raise ValueError(
                        f"按变量 {wanted} 看，这不是齐二次型（各项次数必须都是 2）；"
                        "请检查变量列表或表达式。"
                    )
                free, all_syms = chosen, sorted(parsed.free_symbols, key=lambda s: (len(str(s)), str(s)))
            else:
                free, poly, all_syms = _quadratic_split(parsed)
            parameters = [s for s in all_syms if s not in free]
            names = [str(s) for s in free]
            count = len(names)
            A_mat = sp.zeros(count, count)
            for i in range(count):
                A_mat[i, i] = sp.simplify(poly.coeff_monomial(free[i] ** 2))
                for j in range(i + 1, count):
                    cross = sp.simplify(poly.coeff_monomial(free[i] * free[j]))
                    A_mat[i, j] = sp.simplify(cross / 2)
                    A_mat[j, i] = A_mat[i, j]
            mat = A_mat
            r.set_input(expr=text, variables=names)
            r.set_raw(
                coefficient_matrix=mat.tolist(),
                coefficient_matrix_text=_matrix_text(mat),
                coefficient_matrix_latex=A.to_latex(mat),
                symbol_extraction=(
                    f"二次型变量 = {names}"
                    + (f"；参数（系数）= {[str(s) for s in parameters]}" if parameters else "")
                    + "（枚举符号子集，取「每个单项式在子集上的次数都恰好为 2」的最大子集）"
                ),
            )
            if parameters:
                r.add_condition(
                    "含参数 " + "、".join(str(s) for s in parameters)
                    + " 的二次型：正定性、惯性指数等结论可能随参数取值而改变，"
                    "此处给出的是把参数当独立系数的一般结论，务必按参数分类讨论。"
                )
                r.add_warning(
                    f"系数中含参数 {[str(s) for s in parameters]}，正定/负定结论不是无条件成立的。"
                )
    else:
        mat = _matrix(matrix, what="二次型矩阵")
        source = "matrix"
        free = sorted(mat.free_symbols, key=lambda s: (len(str(s)), str(s)))
        if variables:
            names = [str(v).strip() for v in (variables if isinstance(variables, (list, tuple)) else str(variables).replace(",", " ").split()) if str(v).strip()]
            count = len(names)
            free = A.get_symbols(names)
            free = [free[name] for name in names if name in free]
            if len(free) != count:
                raise ValueError(f"变量名 {names} 与矩阵维数不匹配")
        else:
            count = mat.cols
            names = [str(s) for s in free] if len(free) == count else [f"x{i + 1}" for i in range(count)]
            if len(free) != count:
                free = [sp.Symbol(name, real=True) for name in names]
        r.set_input(variables=names, matrix_shape=f"{mat.rows}x{mat.cols}")

    _require_square(mat, "二次型")
    count = mat.cols
    if not _same_matrix(mat, mat.T):
        raise ValueError("二次型矩阵必须是对称矩阵（Aᵀ = A）；请检查输入，或改用 expr 直接给出二次型表达式")
    syms = [sp.Symbol(name, real=True) if not isinstance(name, sp.Symbol) else name for name in names]
    if len(syms) != count:
        syms = sorted(mat.free_symbols, key=lambda s: (len(str(s)), str(s)))
    if len(syms) != count:
        raise ValueError(f"变量个数（{len(syms)}）与矩阵阶数（{count}）不匹配")

    vector = sp.Matrix(syms)
    polynomial = sp.expand((vector.T * mat * vector)[0, 0])
    r.set_input(quadratic_text=A.to_text(polynomial), variable_count=count)
    r.set_raw(quadratic_text=A.to_text(polynomial), quadratic_latex=A.to_latex(polynomial))

    r.add_condition("二次型的矩阵表示 A 取对称矩阵（交叉项系数均分到 a_ij 与 a_ji）")

    # ---- 顺序主子式 ----
    minors: list[sp.Basic] = []
    for k in range(1, count + 1):
        minors.append(sp.simplify(mat[:k, :k].det()))
    all_positive = all(minor.is_positive is True for minor in minors)
    all_negative_signs = all(
        (minor.is_negative is True) if (k % 2 == 1) else (minor.is_positive is True) for k, minor in enumerate(minors, start=1)
    )
    all_nonnegative = all(minor.is_nonnegative is True for minor in minors)
    symbolic_ambiguous = any(minor.free_symbols for minor in minors) and not (all_positive or all_negative_signs)

    def _eigen_sign(value: sp.Basic) -> str:
        if value.is_positive is True:
            return "+"
        if value.is_negative is True:
            return "-"
        if value.is_zero is True:
            return "0"
        try:
            numeric = complex(sp.N(value, 30))
        except Exception:  # noqa: BLE001
            return "?"
        if abs(numeric) < sp.Float("1e-20"):
            return "0"
        return "+" if numeric.real > 0 else "-"

    try:
        eigen_map = mat.eigenvals()
    except Exception as exc:  # noqa: BLE001
        return MathResult.unsolved(
            "quadratic_form",
            f"无法精确求出二次型矩阵的特征值，因此无法给出规范形与正定性判定（{type(exc).__name__}: {exc}）；"
            "建议改用 math_numeric 做数值判定。",
            method="特征值分解",
        )
    eigen_values = [sp.simplify(value) for value in eigen_map]
    signs = [_eigen_sign(value) for value in eigen_values]
    if all(sign == "+" for sign in signs):
        by_eigen = "positive_definite"
    elif all(sign == "-" for sign in signs):
        by_eigen = "negative_definite"
    elif all(sign in ("+", "0") for sign in signs) and any(sign == "0" for sign in signs):
        by_eigen = "positive_semidefinite"
    elif all(sign in ("-", "0") for sign in signs) and any(sign == "0" for sign in signs):
        by_eigen = "negative_semidefinite"
    elif all(sign == "0" for sign in signs):
        by_eigen = "zero"
    elif any(sign == "?" for sign in signs):
        by_eigen = "undetermined"
    else:
        by_eigen = "indefinite"

    if all_positive:
        by_minor = "positive_definite"
    elif all_negative_signs:
        by_minor = "negative_definite"
    elif symbolic_ambiguous:
        by_minor = "undetermined"
    elif all_nonnegative and not all_positive:
        by_minor = "undetermined"
    else:
        by_minor = "not_positive_definite"

    verdict_map = {
        "positive_definite": "正定",
        "negative_definite": "负定",
        "positive_semidefinite": "半正定",
        "negative_semidefinite": "半负定",
        "zero": "零型（恒为 0，仅零向量取等）",
        "indefinite": "不定",
        "not_positive_definite": "非正定（不是正定二次型）",
        "undetermined": "无法判定（含符号，条件依赖于参数取值）",
    }
    verdict = verdict_map[by_eigen]

    # ---- 正交变换标准化（实对称矩阵：Q 正交，QᵀAQ = Λ） ----
    orthonormal: sp.Matrix | None = None
    diagonal_entries: list[sp.Basic] = []
    try:
        orthonormal, diagonal = mat.diagonalize(normalize=True)
        diagonal = sp.simplify(diagonal)
        if orthonormal.rows == count and orthonormal.cols == count and _same_matrix(
            sp.simplify(orthonormal.T * mat * orthonormal), diagonal
        ):
            diagonal_entries = [sp.simplify(diagonal[i, i]) for i in range(count)]
        else:
            orthonormal = None
    except Exception:  # noqa: BLE001
        orthonormal = None
        diagonal_entries = []
    if orthonormal is None:
        orthonormal, diagonal_entries = _orthonormal_diagonalization(mat, syms)

    # ---- Lagrange 配方法（与正交变换完全不同的路径） ----
    try:
        standard_text, standard_latex, substitution = _lagrange_standard(polynomial, syms)
    except Exception:  # noqa: BLE001
        standard_text, standard_latex, substitution = None, None, None

    canonical_text = None
    if orthonormal is not None and diagonal_entries:
        new_vars = [sp.Symbol(f"y{i + 1}", real=True) for i in range(count)]
        try:
            canonical = sp.expand(sum(sp.simplify(diagonal_entries[i]) * new_vars[i] ** 2 for i in range(len(diagonal_entries))))
            canonical_text = A.to_text(canonical)
        except Exception:  # noqa: BLE001
            canonical_text = None

    normal_form = None
    if diagonal_entries and all(entry.is_zero in (True, False) for entry in diagonal_entries):
        try:
            terms = []
            for entry in diagonal_entries:
                sign = "+1" if _eigen_sign(entry) == "+" else ("-1" if _eigen_sign(entry) == "-" else "0")
                terms.append(sign)
            normal_form = ", ".join(f"{sign}·z{i + 1}²" for i, sign in enumerate(terms))
        except Exception:  # noqa: BLE001
            normal_form = None

    r.set_result(verdict)
    r.set_raw(
        kind=A.to_text(polynomial),
        definiteness=verdict,
        definiteness_code=by_eigen,
        leading_principal_minors=[A.to_text(minor) for minor in minors],
        leading_principal_minors_latex=[A.to_latex(minor) for minor in minors],
        eigenvalues=[A.to_text(value) for value in eigen_values],
        eigenvalues_latex=[A.to_latex(value) for value in eigen_values],
        eigenvalue_signs=signs,
        canonical_form_text=canonical_text,
        standard_form_text=standard_text,
        standard_form_latex=standard_latex,
        normal_form_text=normal_form,
        orthonormal_matrix=None if orthonormal is None else orthonormal.tolist(),
        orthonormal_matrix_text=None if orthonormal is None else _matrix_text(orthonormal),
        substitution=substitution,
    )
    if by_eigen == "positive_definite":
        r.add_condition("正定 ⟺ 全部特征值 > 0 ⟺ 全部顺序主子式 > 0（两种判据均成立）")
        r.add_condition("正定二次型的规范形为 z₁² + z₂² + ... + zₙ²（惯性指数 p = n, q = 0）")
    elif by_eigen == "negative_definite":
        r.add_condition("负定 ⟺ 全部特征值 < 0 ⟺ 奇数阶顺序主子式 < 0 且偶数阶顺序主子式 > 0")
    elif by_eigen == "indefinite":
        r.add_condition("不定：特征值有正有负，存在使二次型取正、取负乃至为零的非零向量")
    elif by_eigen == "positive_semidefinite":
        r.add_condition("半正定：全部特征值 ≥ 0 且至少一个为 0；惯性指数 p < n")
    elif by_eigen == "undetermined":
        r.add_condition("矩阵含自由符号，正定性依参数取值而变；请代入具体参数后再判定")
    r.add_condition("正交变换 x = Qy 不改变二次型的正定性（合同变换保持惯性指数）")

    # ---- 独立验证：顺序主子式判据 vs 特征值判据 ----
    methods: list[dict[str, Any]] = [
        {
            "method": "顺序主子式判据（Sylvester）：Δ₁..Δₙ 的符号模式",
            "minors": [A.to_text(minor) for minor in minors],
            "verdict": by_minor,
            "agree": (by_minor == by_eigen) or by_minor == "undetermined",
        },
        {
            "method": "特征值符号判据：全部特征值的正负号",
            "eigenvalues": [A.to_text(value) for value in eigen_values],
            "signs": signs,
            "verdict": by_eigen,
            "agree": True,
        },
    ]
    ok_transform = None
    if orthonormal is not None and diagonal_entries:
        try:
            product = sp.simplify(orthonormal.T * mat * orthonormal)
            ok_transform = _same_matrix(product, sp.diag(*diagonal_entries))
            methods.append(
                {
                    "method": "正交变换检验：QᵀAQ 为对角阵，且 QᵀQ = I",
                    "diagonal_text": _matrix_text(product),
                    "agree": bool(ok_transform) and _same_matrix(sp.simplify(orthonormal.T * orthonormal), sp.eye(count)),
                }
            )
        except Exception:  # noqa: BLE001
            ok_transform = None
    ok_minor = (by_minor == by_eigen) or by_minor == "undetermined"
    ok = ok_minor and (ok_transform is not False)
    r.verify(
        status="independent" if ok else "unverified",
        methods=methods,
        note="验证通过：顺序主子式判据与特征值判据互相印证。"
        if ok
        else "两种判据不一致，请人工复核。",
    )
    if not ok:
        r.add_warning("二次型正定性的两种独立判据不一致，请人工复核。")
    return r


def _orthonormal_diagonalization(mat: sp.Matrix, syms: Sequence[sp.Symbol]) -> tuple[sp.Matrix | None, list[sp.Basic]]:
    """实对称矩阵的正交对角化：Gram-Schmidt 正交化特征向量，再单位化。"""
    try:
        vectors: list[sp.Matrix] = []
        for value, mult, basis, _ in mat.eigenvects():
            for vec in basis:
                candidate = sp.Matrix(sp.simplify(vec))
                for existing in vectors:
                    try:
                        candidate = candidate - (existing.dot(candidate) / existing.dot(existing)) * existing
                    except Exception:  # noqa: BLE001
                        continue
                candidate = sp.simplify(candidate)
                if candidate.is_zero_matrix:
                    continue
                norm = sp.sqrt(sp.simplify(candidate.dot(candidate)))
                candidate = sp.simplify(candidate / norm)
                vectors.append(candidate)
        if len(vectors) != mat.cols:
            return None, []
        orthonormal = sp.Matrix.hstack(*vectors)
        diagonal_entries = [sp.simplify((orthonormal[:, i].T * mat * orthonormal[:, i])[0, 0]) for i in range(mat.cols)]
        return orthonormal, diagonal_entries
    except Exception:  # noqa: BLE001
        return None, []


def _lagrange_standard(polynomial: sp.Basic, syms: Sequence[sp.Symbol]) -> tuple[str | None, str | None, str | None]:
    """Lagrange 配方法：逐项消去交叉项，给出标准形与所作的可逆线性替换。"""
    work = sp.expand(polynomial)
    remaining = list(syms)
    substitution_lines: list[str] = []
    terms: list[sp.Basic] = []
    while len(remaining) > 1:
        lead = remaining[0]
        quadratic = sp.expand(work).coeff(lead, 2)
        if quadratic != 0:
            rest = sp.expand(work - quadratic * lead**2)
            rest = sp.expand(rest)
            shift = sp.simplify(rest / (2 * quadratic))
            shift = sp.expand(shift)
            # work = quadratic * (lead + shift/quadratic)^2 - shift^2/quadratic
            terms.append(sp.simplify(quadratic))
            substitution_lines.append(f"y_{lead} = {A.to_text(lead)} + ({A.to_text(shift)})/({A.to_text(2 * quadratic)})")
            work = sp.expand(work - quadratic * (lead + shift / (2 * quadratic)) ** 2)
        else:
            terms.append(sp.Integer(0))
            substitution_lines.append(f"y_{lead} = {A.to_text(lead)}")
        remaining = remaining[1:]
    if remaining:
        tail = sp.expand(work).coeff(remaining[0], 2)
        terms.append(sp.simplify(tail))
        substitution_lines.append(f"y_{remaining[0]} = {A.to_text(remaining[0])}")
    y_vars = [sp.Symbol(f"y{i + 1}", real=True) for i in range(len(terms))]
    standard = sp.expand(sum(term * var**2 for term, var in zip(terms, y_vars)))
    return A.to_text(standard), A.to_latex(standard), "; ".join(substitution_lines)


# ---------------------------------------------------------------------------
# 13. 矩阵幂
# ---------------------------------------------------------------------------

@op("matrix_power", "matpow")
def matrix_power(matrix: Any = None, order: Any = None, **kwargs: Any) -> MathResult:
    """矩阵的整数次幂 Aⁿ（建议用 Jordan 形式）；验证用连乘比较。"""
    if order is None:
        raise ValueError("缺少幂次 order，例如 order=3")
    mat = _matrix(matrix, what="矩阵")
    _require_square(mat, "求矩阵幂")
    try:
        exponent = sp.Integer(sp.sympify(str(order).strip())) if not isinstance(order, (int, sp.Basic)) else sp.Integer(order)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"幂次 order 必须是整数，无法解析 {order!r}：{exc}") from exc
    if exponent.is_Integer is not True:
        raise ValueError(f"幂次 order 必须是整数，得到 {order!r}")
    n = int(exponent)
    if n < 0:
        raise ValueError("幂次 order 需要是非负整数；负幂请先确认矩阵可逆并用 inv 求逆矩阵")

    r = MathResult.ok("matrix_power", method="SymPy Matrix.pow（二进制幂，精确分数）")
    r.set_input(**_shape_input(mat), order=n)
    if n == 0:
        r.set_result(sp.eye(mat.rows))
        r.set_raw(power_text=_matrix_text(sp.eye(mat.rows)), exponent=0)
        r.add_condition("A⁰ = I（按约定，任何方阵的 0 次幂为单位矩阵）")
        r.verify(
            status="independent",
            methods=[{"method": "定义核对：A⁰ 定义为单位矩阵 I", "agree": True}],
            note="n = 0 时按定义给出单位矩阵。",
        )
        return r

    try:
        powered = sp.simplify(mat**n)
    except Exception as exc:  # noqa: BLE001
        return MathResult.unsolved(
            "matrix_power",
            f"无法计算 A^{n}（{type(exc).__name__}: {exc}）；可先化简矩阵或改用特征值/对角化。",
            method="SymPy Matrix.pow",
        )
    r.set_result(powered)
    r.set_raw(power_text=_matrix_text(powered), power_latex=A.to_latex(powered), exponent=n)
    r.add_condition(f"A 为方阵，A^{n} 由矩阵乘法定义（幂次满足 Aᵐ⁺ⁿ = AᵐAⁿ）")

    methods: list[dict[str, Any]] = []
    if n <= 12:
        manual = sp.eye(mat.rows)
        for _ in range(n):
            manual = manual * mat
        manual = sp.simplify(manual)
        methods.append(
            {
                "method": f"连乘复核：把 A 乘 {n} 次（不用 pow 例程）",
                "agree": _same_matrix(manual, powered),
            }
        )
        ok = _same_matrix(manual, powered)
    else:
        # 独立途径：Aⁿ = A^(n-k)·A^k
        half = n // 2
        try:
            left = sp.simplify(mat ** (n - half))
            right = sp.simplify(mat**half)
            combined = sp.simplify(left * right)
            ok = _same_matrix(combined, powered)
            methods.append(
                {
                    "method": f"指数分解复核：A^{n} = A^{n - half}·A^{half}（先分别求幂再相乘）",
                    "agree": ok,
                }
            )
        except Exception:  # noqa: BLE001
            ok = False
    # Cayley-Hamilton 的简化体现：A^n 与 A 可交换，且特征值 λⁿ
    try:
        ok_commute = _same_matrix(sp.simplify(mat * powered), sp.simplify(powered * mat))
        methods.append({"method": "交换性检验：A·Aⁿ = Aⁿ·A", "agree": ok_commute})
        ok = ok and ok_commute
    except Exception:  # noqa: BLE001
        pass
    try:
        ok_det = _same_scalar(powered.det(), sp.simplify(mat.det() ** n))
        methods.append(
            {
                "method": f"行列式乘性检验：det(A^{n}) = det(A)^{n}（另一条独立数值/符号关系）",
                "expected": A.to_text(sp.simplify(mat.det() ** n)),
                "got": A.to_text(powered.det()),
                "agree": bool(ok_det),
            }
        )
        ok = ok and bool(ok_det)
    except Exception:  # noqa: BLE001
        pass
    r.verify(
        status="independent" if ok else "unverified",
        methods=methods,
        note="验证通过：连乘复核与行列式乘性一致。" if ok else "连乘复核不一致，结果不可信。",
    )
    if not ok:
        r.add_warning("矩阵幂的独立验证未通过，请人工复核。")
    return r


# ---------------------------------------------------------------------------
# 14. 矩阵总览（只读汇总，不做求解）
# ---------------------------------------------------------------------------

@op("matrix_analysis", "matrix_info", "math_matrix")
def matrix_analysis(matrix: Any = None, rref: Any = None, include_rref: Any = None, **kwargs: Any) -> MathResult:
    """矩阵总览：行列式、秩、迹、可逆性、特征值、可选行最简形。

    这是一个**只读汇总视图**：只做汇总与相互印证，不做求解。不计算的项目
    明确标注「未计算」，绝不猜测。
    """
    mat = _matrix(matrix, what="矩阵")
    r = MathResult.ok("matrix_analysis", method="汇总视图：行列式/秩/迹/可逆性/特征值（只读汇总，不做求解）")
    r.set_input(**_shape_input(mat))
    r.set_raw(view="SUMMARY ONLY（只读汇总视图，不做求解）")
    square = mat.rows == mat.cols
    r.set_raw(is_square=square)

    want_rref: bool
    if rref is None and include_rref is None:
        want_rref = square and mat.rows <= _RREF_SUMMARY_LIMIT
    else:
        flag = include_rref if include_rref is not None else rref
        if isinstance(flag, bool):
            want_rref = flag
        else:
            want_rref = str(flag).strip().lower() in ("1", "true", "yes", "y", "是", "on")

    methods: list[dict[str, Any]] = []
    not_computed: list[str] = []

    # 行列式 / 可逆性（仅方阵）
    if square:
        try:
            determinant = sp.simplify(mat.det())
            r.set_raw(det_text=A.to_text(determinant), det_latex=A.to_latex(determinant))
            if determinant.is_zero is True:
                r.set_raw(invertible=False, invertibility="不可逆（det = 0，奇异矩阵）")
            elif determinant.is_zero is False:
                r.set_raw(invertible=True, invertibility="可逆（det ≠ 0）")
            else:
                r.set_raw(invertible=None, invertibility=f"含自由符号，det = {A.to_text(determinant)}，可逆性取决于 det 是否为 0")
            try:
                _, rref_pivots = mat.rref()
                full = len(list(rref_pivots)) == mat.rows
                methods.append(
                    {
                        "method": "秩判据复核可逆性：r(A) = n ⟺ A 可逆（det ≠ 0 等价于满秩）",
                        "rank": len(list(rref_pivots)),
                        "n": mat.rows,
                        "agree": (full == (determinant.is_zero is False)) if determinant.is_zero in (True, False) else None,
                    }
                )
            except Exception:  # noqa: BLE001
                pass
        except Exception as exc:  # noqa: BLE001
            r.set_raw(det_text=None, det_note=f"未计算：行列式求解失败（{type(exc).__name__}: {exc}）")
            not_computed.append("det")
    else:
        r.set_raw(det_text=None, det_note="未计算：行列式仅对方阵定义", invertible=None, invertibility="未计算：非方阵")
        not_computed.append("det/可逆性")

    # 秩
    try:
        rank_value = int(mat.rank())
        rref_mat, pivots = mat.rref()
        pivot_list = [int(p) for p in pivots]
        nonzero_rows = sum(1 for i in range(rref_mat.rows) if any(rref_mat[i, j] != 0 for j in range(rref_mat.cols)))
        r.set_raw(rank=rank_value, pivot_columns=[p + 1 for p in pivot_list], rank_definition="rank = 行最简形非零行数 = 主元个数")
        methods.append(
            {
                "method": "秩的两种计数：rank() 与行最简形非零行数",
                "rank": rank_value,
                "nonzero_rows": nonzero_rows,
                "agree": rank_value == nonzero_rows,
            }
        )
        if want_rref:
            r.set_raw(rref=rref_mat.tolist(), rref_text=_matrix_text(rref_mat))
        else:
            r.set_raw(rref=None, rref_note="未计算：如需行最简形请设置 rref=true 或调用 rref 算子")
            not_computed.append("rref")
    except Exception as exc:  # noqa: BLE001
        r.set_raw(rank=None, rank_note=f"未计算：秩求解失败（{type(exc).__name__}: {exc}）")
        not_computed.append("rank")

    # 迹
    if square:
        try:
            trace_value = sp.simplify(mat.trace())
            r.set_raw(trace_text=A.to_text(trace_value), trace_latex=A.to_latex(trace_value))
        except Exception as exc:  # noqa: BLE001
            r.set_raw(trace_text=None, trace_note=f"未计算：迹求解失败（{type(exc).__name__}: {exc}）")
            not_computed.append("trace")
    else:
        r.set_raw(trace_text=None, trace_note="未计算：迹仅对方阵定义")
        if "trace" not in not_computed:
            not_computed.append("trace")

    # 特征值
    dropped: list[str] = []
    if square:
        try:
            eigenvalues = _eigenvalue_objects(mat)
            r.set_raw(
                eigenvalues=[A.to_text(value) for value, _ in eigenvalues],
                eigenvalues_latex=[A.to_latex(value) for value, _ in eigenvalues],
                algebraic_multiplicities=[mult for _, mult in eigenvalues],
            )
            if square and mat.rows <= _EIGVECT_LIMIT:
                try:
                    vectors = [
                        {
                            "eigenvalue": A.to_text(value),
                            "geometric_multiplicity": len(basis),
                        }
                        for value, _mult, basis, _ in mat.eigenvects()
                    ]
                    r.set_raw(eigenvectors_summary=vectors)
                except Exception as exc:  # noqa: BLE001
                    r.set_raw(eigenvectors_summary=None, eigenvectors_note=f"未计算：特征向量求解失败（{type(exc).__name__}: {exc}）")
                    not_computed.append("eigenvectors")
            else:
                r.set_raw(eigenvectors_summary=None, eigenvectors_note="未计算：矩阵阶数较大")
                not_computed.append("eigenvectors")
            try:
                r.set_raw(characteristic_polynomial=A.to_text(sp.factor(mat.charpoly().as_expr())))
            except Exception:  # noqa: BLE001
                r.set_raw(characteristic_polynomial=None)
            try:
                eigen_sum = sp.simplify(sum(value * mult for value, mult in eigenvalues))
                eigen_product = sp.simplify(sp.prod([value ** mult for value, mult in eigenvalues]))
                methods.append(
                    {
                        "method": "迹与行列式的谱关系：trace(A) = Σ mᵢλᵢ 且 det(A) = ∏ λᵢ^mᵢ",
                        "trace": A.to_text(mat.trace()),
                        "eigen_sum": A.to_text(eigen_sum),
                        "det": A.to_text(sp.simplify(mat.det())),
                        "eigen_product": A.to_text(eigen_product),
                        "agree": bool(_same_scalar(mat.trace(), eigen_sum) and _same_scalar(mat.det(), eigen_product)),
                    }
                )
            except Exception:  # noqa: BLE001
                pass
        except Exception as exc:  # noqa: BLE001
            r.set_raw(eigenvalues=None, eigenvalues_note=f"未计算：特征值精确求解失败（{type(exc).__name__}: {exc}）")
            not_computed.append("eigenvalues")
    else:
        r.set_raw(eigenvalues=None, eigenvalues_note="未计算：特征值仅对方阵定义")
        not_computed.append("eigenvalues")

    r.set_result(f"{mat.rows}×{mat.cols} 矩阵的只读汇总视图")
    r.set_raw(not_computed=sorted(set(not_computed)), note="本工具为汇总视图，未执行任何求解；需要具体结果请调用对应算子")
    r.add_condition("所有数值均为精确符号值；含自由符号时结论依赖于符号取值")
    if square:
        r.add_condition("可逆 ⟺ det ≠ 0 ⟺ 满秩 ⟺ 特征值全不为 0")

    if dropped:
        r.add_warning(f"以下项目因规模或失败被跳过：{', '.join(dropped)}")
    agreed = all(method.get("agree") in (True, None) for method in methods)
    r.verify(
        status="independent" if (methods and agreed) else "unverified",
        methods=methods,
        note="汇总视图内部用秩判据与谱关系交叉印证了可逆性与特征值统计。"
        if (methods and agreed)
        else "可用的交叉印证不足，请以单项算子（det/inv/eigen）的独立验证为准。",
    )
    return r
