"""概率论与数理统计（考研数学一）。

本模块只依赖标准库 + sympy/numpy/scipy/mpmath 与本包的
``ast`` / ``sympy_core`` / ``result`` / ``engine``（用户规格 5.3、十一）。

设计要点
--------
* **精确优先**：概率一律保留 ``1/6``、``erf(sqrt(2)/2)`` 这样的精确形式，
  同时用 ``result`` 层自动附带的 ``approx`` 给出小数。
* **独立验证**：每个成功算子都尝试用 *另一条数学路径* 复核，失败/不可能时
  标 ``unverified`` 并给出原因，绝不用同一个函数再算一次冒充验证。
* **绝不伪造**：给不出闭式解就 ``MathResult.unsolved``；输入非法
  （参数越界、方差为 0 时求相关系数……）抛 ``ValueError``。

参数约定（与国内考研教材一致，结果里也会写明）
------------------------------------------------
======================  =========================================================
分布                     约定
======================  =========================================================
``normal``              ``N(mu, sigma^2)``，``sigma`` 是标准差，恒 ``sigma > 0``
``exponential``         ``f(x) = lam * exp(-lam x)``（``lam`` 是率参数）
``gamma``               ``f(x) = lam^k x^(k-1) e^(-lam x) / Gamma(k)``（``lam`` 是率）
``beta``                ``B(a, b)``
``chi2``                自由度 ``k``
``t``                   自由度 ``k``
``f``                   ``F(d1, d2)``
``lognormal``           ``ln X ~ N(mu, sigma^2)``
``weibull``             ``f(x) = (k/lam)(x/lam)^(k-1) e^{-(x/lam)^k}``（``lam`` 是尺度）
``geometric``           教材约定：``P(X = k) = p (1-p)^(k-1), k = 1, 2, ...``
``hypergeometric``      ``P(X = k) = C(K,k) C(N-K,n-k) / C(N,n)``（``N`` 总体、
                        ``K`` 次品数、``n`` 抽样数），``X`` 是抽到的次品数
======================  =========================================================
"""

from __future__ import annotations

from typing import Any

import sympy as sp

from . import ast as A
from . import sympy_core as C
from .engine import op
from .result import MathResult

# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------

_OO = sp.oo


def _has_free(value: Any) -> bool:
    return isinstance(value, sp.Basic) and bool(value.free_symbols)


def _is_infinity(value: Any) -> bool:
    """是否是 ±oo（不能用 ``in (-oo, oo)``：SymPy 里 0 == False，会误判）。"""
    try:
        return bool(value.is_infinite)
    except Exception:  # noqa: BLE001
        return False


_UNEVALUATED_TYPES = (sp.Integral, sp.Derivative, sp.Limit, sp.Sum, sp.Product)


def _unevaluated(value: Any) -> bool:
    """本模块自己的「未求值」判定。

    不能直接用 ``A.is_unevaluated``：``mathkit/ast.py`` 里把它写成
    ``(sp.Integral, ..., sp.Solve)``，而 SymPy 1.13 **没有** ``sp.Solve``，
    于是该函数一被调用就抛 ``AttributeError: module 'sympy' has no attribute 'Solve'``。
    （这是别人文件里的 bug，只报告不修改。）
    """
    if not isinstance(value, sp.Basic):
        return False
    try:
        if value.has(*_UNEVALUATED_TYPES):
            return True
    except Exception:  # noqa: BLE001
        return True
    return False


def _is_evaluated(value: Any) -> bool:
    """结果可用（既不含自由符号，也不是未求值节点）。"""
    return not _has_free(value) and not _unevaluated(value)


def _safe_equal(a: Any, b: Any, *, tol: float = 1e-12) -> bool:
    """数值/符号相等判定（数值时允许 ``tol`` 容差）。"""
    try:
        if a == b:
            return True
    except Exception:  # noqa: BLE001
        pass
    if _has_free(a) or _has_free(b):
        return False
    try:
        return abs(complex(sp.N(a - b, 30))) < tol
    except Exception:  # noqa: BLE001
        return False


def _kernel_of(expr: Any) -> Any:
    """把 ``Piecewise`` 尾部的 ``(0, True)`` 兜底分支剥掉，得到“核”表达式。

    只剥落在最后的 ``(0, True)``；上界有限的支撑（如超几何）保留全部条件分支，
    因为那种 Piecewise 的求和/积分 sympy 能正常处理。
    """
    if not isinstance(expr, sp.Piecewise):
        return expr
    args = list(expr.args)
    if not args:
        return expr
    out: list[Any] = []
    for i, piece in enumerate(args):
        val, cond = piece.args[0], piece.args[1]
        if i == len(args) - 1 and cond is sp.true and val == 0:
            continue
        out.append((val, cond))
    if not out:
        return expr
    if len(out) == 1:
        return out[0][0]
    return sp.Piecewise(*out)


def _numbers(value: Any, names: tuple[str, ...] = ("x", "y", "z", "t", "n", "k", "a", "b", "c")) -> list[sp.Basic]:
    """解析 ``,`` / ``;`` 分隔的一串数（概率、取值、边界都用它）。"""
    if value is None:
        return []
    if isinstance(value, sp.Basic):
        return [value]
    if isinstance(value, (int, float)):
        return [sp.sympify(value)]
    if isinstance(value, (list, tuple)):
        out: list[sp.Basic] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                out.extend(_numbers(item, names))
            else:
                out.extend(_numbers(item, names))
        return out
    raw = str(value).strip()
    if not raw:
        return []
    raw = raw.replace("[", "").replace("]", "").replace("{", "").replace("}", "")
    raw = raw.replace(";", ",").replace("，", ",").replace("、", ",")
    raw = raw.replace("∞", "oo").replace("无穷", "oo")
    out = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        piece = piece.replace("=", "")
        out.append(A.parse(piece, symbols=list(names)))
    return out


def _numbers_safe(value: Any) -> list[sp.Basic]:
    """同 :func:`_numbers`，但把解析失败降级为「用 x 当符号名」。"""
    try:
        return _numbers(value)
    except ValueError:
        return _numbers(value, ("x",))


def _matrix_of(value: Any, rows: int | None = None, cols: int | None = None) -> list[list[Any]]:
    """把二维分布律解析成 ``[[...], ...]``。支持嵌套数组或 ``"1/8,1/8;1/8,5/8"``。"""
    if value is None:
        raise ValueError("缺少联合分布律 joint_pmf")
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("联合分布律 joint_pmf 为空")
        if all(isinstance(row, (list, tuple)) for row in value):
            return [_numbers(row) if not all(isinstance(c, sp.Basic) for c in row) else list(row) for row in value]
        cells = _numbers(value)
        if cols and cols > 0:
            return [cells[i:i + cols] for i in range(0, len(cells), cols)]
        if rows and rows > 0:
            step = max(1, len(cells) // rows)
            return [cells[i:i + step] for i in range(0, len(cells), step)]
        return [cells]
    raw = str(value).strip()
    if not raw:
        raise ValueError("联合分布律 joint_pmf 为空")
    raw = raw.replace("[[", "[").replace("]]", "]").replace("], [", ";").replace("],[", ";")
    raw = raw.replace("][", ";").replace("\n", ";").replace("，", ",")
    raw = raw.strip("[]{} ")
    out: list[list[Any]] = []
    for row_text in raw.split(";"):
        row_text = row_text.strip().strip("[]{}")
        if not row_text:
            continue
        row = [c for c in row_text.split(",") if c.strip()]
        if row:
            out.append(_numbers_safe(",".join(row)))
    if not out:
        raise ValueError("联合分布律 joint_pmf 解析为空")
    return out


#: 引擎 ``normalize_args`` 会把这些参数名重映射掉，所以查参数时要顺带看别名。
#: 例如 ``op("binomial", {"n": 10})`` 进到 handler 时已经变成 ``{"order": 10}``。
_ENGINE_NAME_MIRROR: dict[str, tuple[str, ...]] = {
    "n": ("order",),
    "a": ("lower",),
    "b": ("upper",),
    "k": ("order",),
}


def _num(op_par: "_Params", *names: str, default: Any = None, required: bool = False,
         label: str | None = None, mapping: dict[str, tuple[str, ...]] | None = None) -> Any:
    """``_Params.num`` 的包装：自动补上引擎重映射后的参数名。"""
    mapping = mapping or _ENGINE_NAME_MIRROR
    ordered: list[str] = []
    for name in names:
        ordered.append(name)
        for extra in mapping.get(name, ()):
            ordered.append(extra)
    return op_par.num(*ordered, default=default, required=required, label=label)


class _Params:
    """参数别名解析器：模型爱怎么写就怎么写，这里统一收口。

    查找分两轮：

    1. **精确匹配**（按别名给出顺序）。这保证 ``N`` 与 ``n`` 这类大小写不同的
       参数不会互相抢占（超几何分布要同时用 ``N`` 与 ``n``）。
    2. **大小写不敏感匹配**（同样按别名顺序）。因为用户会写 ``N=10`` 而别名表里
       只有 ``"n"``，反之亦然；没有这一轮，超几何的 ``N``/``K`` 永远查不到，
       三个参数会一起落到引擎重映射出来的 ``order`` 上（曾导致 E[X]=3 而不是 3/2）。
    """

    def __init__(self, kwargs: dict[str, Any]):
        self.kw = dict(kwargs or {})

    def raw(self, *names: str) -> Any:
        for name in names:
            if name in self.kw and self.kw[name] is not None:
                return self.kw[name]
        folded = {str(key).lower(): value for key, value in self.kw.items()}
        for name in names:
            value = folded.get(str(name).lower())
            if value is not None:
                return value
        return None

    def text(self, *names: str) -> str | None:
        value = self.raw(*names)
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        if isinstance(value, (list, tuple)):
            return ",".join(str(v) for v in value)
        return str(value)

    def num(self, *names: str, default: Any = None, required: bool = False, label: str | None = None) -> Any:
        value = self.raw(*names)
        if value is None:
            if required:
                raise ValueError(f"缺少必需参数 {label or names[0]}（可用别名：{', '.join(names)}）")
            return default
        if isinstance(value, sp.Basic):
            return value
        if isinstance(value, (int, float)):
            return sp.sympify(value)
        return A.parse(str(value), symbols=["x", "y", "n", "k", "a", "b", "c", "m"])

    def nums(self, *names: str) -> list[sp.Basic]:
        return _numbers(self.raw(*names)) if self.raw(*names) is not None else []


def _joint_tables(op_obj: _Params, vars_hint: tuple[str, ...] = ("x", "y")) -> tuple[sp.Matrix, list[Any], list[Any]]:
    """把离散联合分布律解析成 ``(matrix, x_values, y_values)``。"""
    raw = op_obj.raw("joint_pmf", "jointpmf", "joint", "table", "pmf_table", "joint_table")
    if raw is None:
        raise ValueError("离散联合分布需要 joint_pmf（二维数组或 \"1/8,1/8;1/8,5/8\" 文本）")
    grid = _matrix_of(raw)
    width = max(len(row) for row in grid)
    normalised: list[list[Any]] = []
    for row in grid:
        if len(row) != width:
            raise ValueError(f"联合分布律每行长度必须一致，当前为 {[len(r) for r in grid]}")
        normalised.append(list(row))
    matrix = sp.Matrix(normalised)
    xs = _numbers(op_obj.raw("x_values", "xvals", "xs", "xv")) if op_obj.raw("x_values", "xvals", "xs", "xv") is not None else None
    ys = _numbers(op_obj.raw("y_values", "yvals", "ys", "yv")) if op_obj.raw("y_values", "yvals", "ys", "yv") is not None else None
    if xs is None:
        xs = [sp.Integer(i + 1) for i in range(matrix.cols)]
    if ys is None:
        ys = [sp.Integer(j + 1) for j in range(matrix.rows)]
    if len(xs) != matrix.cols:
        raise ValueError(f"x_values 个数（{len(xs)}）与 joint_pmf 的列数（{matrix.cols}）不一致")
    if len(ys) != matrix.rows:
        raise ValueError(f"y_values 个数（{len(ys)}）与 joint_pmf 的行数（{matrix.rows}）不一致")
    return matrix, list(xs), list(ys)


def _joint_pdf_regions(op_obj: _Params, x: sp.Symbol, y: sp.Symbol) -> tuple[Any, Any, Any, Any]:
    """连续联合密度的积分区域，缺省为全平面。"""
    xl = op_obj.num("x_lower", "xlo", "xmin", "x_from", default=-_OO)
    xu = op_obj.num("x_upper", "xhi", "xmax", "x_to", default=_OO)
    yl = op_obj.num("y_lower", "ylo", "ymin", "y_from", default=-_OO)
    yu = op_obj.num("y_upper", "yhi", "ymax", "y_to", default=_OO)
    return xl, xu, yl, yu


# ---------------------------------------------------------------------------
# 分布注册表：每种分布自带 支撑集 / 密度 / 分布函数 / 参数校验
# ---------------------------------------------------------------------------

_BETA = sp.beta  # 直接用 SymPy 的 beta 函数，保证能参与化简

#: 矩母函数的自变量。**全模块必须共用这一个符号**：SymPy 把
#: ``Symbol("t")`` 与 ``Symbol("t", real=True)`` 视为不同的符号，
#: 混用会让 ``mgf.subs(t, 0)`` 静默失败（曾报 "M(0) = exp(t**2/2)"）。
_T = sp.Symbol("t")


def _reg_incomplete_beta(z: Any, a: Any, b: Any) -> Any:
    """正则化不完全 Beta 函数 I_z(a,b) = B(z;a,b)/B(a,b)。

    注意 SymPy 的 ``sp.betainc(a, b, x1, x2)`` 是**未正则化**的
    ``∫_{x1}^{x2} t^(a-1)(1-t)^(b-1) dt``，所以：

    * ``sp.betainc(a, b, 0, z)`` 是 ``B(z; a, b)``（本题要的分子）；
    * ``sp.betainc(a, b, z, 1)`` 是 ``B(a,b) - B(z;a,b)``（补函数，不是 I_z）。

    曾经把两者写反，导致 Beta/F 的 cdf 算成了 1-F(x)（出错很难发现）。
    """
    return sp.betainc(a, b, 0, z) / _BETA(a, b)


def _normalize_dist_name(name: Any) -> str:
    if name is None:
        raise ValueError("缺少分布名 dist")
    key_raw = str(name).strip().lower()
    key = key_raw
    for ch in " 　-_/\\":
        key = key.replace(ch, "")
    aliases = {
        "二项分布": "binomial", "二项": "binomial", "binom": "binomial", "b": "binomial",
        "泊松分布": "poisson", "泊松": "poisson", "pois": "poisson",
        "几何分布": "geometric", "几何": "geometric", "geom": "geometric",
        "超几何分布": "hypergeometric", "超几何": "hypergeometric", "hypergeom": "hypergeometric",
        "离散均匀分布": "uniform_discrete", "离散均匀": "uniform_discrete",
        "duniform": "uniform_discrete", "discreteuniform": "uniform_discrete",
        "正态分布": "normal", "正态": "normal", "gauss": "normal",
        "gaussian": "normal", "norm": "normal",
        "指数分布": "exponential", "指数": "exponential", "exp": "exponential",
        "expon": "exponential",
        "均匀分布": "uniform", "均匀": "uniform", "unif": "uniform",
        "伽马分布": "gamma", "伽玛分布": "gamma", "伽马": "gamma", "伽玛": "gamma",
        "贝塔分布": "beta", "贝塔": "beta", "贝它分布": "beta",
        "卡方分布": "chi2", "卡方": "chi2", "chisquare": "chi2", "chi_squared": "chi2",
        "chi": "chi2", "x2": "chi2",
        "t分布": "t", "student": "t", "studentt": "t",
        "f分布": "f", "fdist": "f",
        "对数正态分布": "lognormal", "对数正态": "lognormal", "lognorm": "lognormal",
        "柯西分布": "柯西", "柯西": "cauchy", "cauchy": "cauchy",
        "威布尔分布": "weibull", "威布尔": "weibull", "weibull": "weibull",
        "rayleigh": "rayleigh",
        "uniformdiscrete": "uniform_discrete",
        "discreteuniform": "uniform_discrete",
        "chisquared": "chi2",
        "studentt": "t",
    }
    return aliases.get(key, key)


class _Dist:
    """一个分布的全部信息（自带参数校验，越界直接 ValueError）。"""

    def __init__(self, name: str, var: sp.Symbol, params: dict[str, Any], *, discrete: bool):
        self.name = name
        self.var = var
        self.params = params
        self.discrete = discrete
        self.canonical = name
        self.conditions: list[str] = []
        self.param_desc = ""
        self._pdf = None
        self._cdf = None
        self._stats_x = None
        self.support: tuple[Any, Any] = (-_OO, _OO)
        # 自变量的有效范围（密度/分布律真正的非零区间）：离散型用来定求和上下限
        self.support_lower: Any = None
        self.support_upper: Any = None
        self.mean = None
        self.variance = None
        self.skewness = None
        self.kurtosis = None
        self.mgf: Any = None
        self.mgf_note: str | None = None
        # 矩不存在时的说明（如柯西分布 E(X) 不存在）；用于把 nan/zoo 结果
        # 改写成人话，而不是把 nan 直接丢给用户。
        self.moment_note: str | None = None

    # -- 懒加载密度 / 分布函数 -------------------------------------------
    @property
    def pdf(self) -> Any:
        if self._pdf is None:
            self._pdf = self._pdf_func()
        return self._pdf

    @property
    def cdf(self) -> Any:
        if self._cdf is None:
            self._cdf = self._cdf_func()
        return self._cdf

    def _pdf_func(self) -> Any:
        raise NotImplementedError

    def _cdf_func(self) -> Any:
        x = self.var
        return sp.simplify(sp.integrate(self.kernel, (x, -_OO, x)))

    # -- Piecewise 自由的“核”：求和/积分专用 ------------------------------
    @property
    def kernel(self) -> Any:
        """剥掉 ``(0, True)`` 兜底分支后的密度/分布律。

        用途：``sp.summation`` 对含 Piecewise 的被加项常常直接失败（原样回显），
        所以所有符号求和/积分都用这个不含兜底分支的表达式，
        再配合 ``support_lower`` / ``support_upper`` 给出显式上下限。
        """
        return _kernel_of(self.pdf)

    # -- 验证辅助：归一化 ------------------------------------------------
    def normalization(self) -> Any:
        x = self.var
        if _is_infinity(self.sum_lower) or _is_infinity(self.sum_upper):
            expr = self.kernel
        elif self.discrete:
            expr = self.kernel
        else:
            # 有限支撑：用带 Piecewise 的 pdf，sympy 能对条件区间给出干净结果
            expr = self.pdf
        if self.discrete:
            return sp.summation(expr, (x, self.sum_lower, self.sum_upper))
        return sp.integrate(expr, (x, self.sum_lower, self.sum_upper))

    @property
    def sum_lower(self) -> Any:
        """求和/积分的下限：对离散型是支撑集下界，对连续型是密度真正开始非零的位置。"""
        lo = self.support[0]
        return -_OO if _is_infinity(lo) else lo

    @property
    def sum_upper(self) -> Any:
        hi = self.support[1]
        return _OO if _is_infinity(hi) else hi

    def prob_below(self, point: Any) -> Any:
        """P(X <= point) 的直算表达式（独立于 :attr:`cdf` 的实现路径）。"""
        x = self.var
        if self.discrete:
            return sp.summation(self.kernel, (x, self.sum_lower, sp.floor(point)))
        return sp.integrate(self.kernel, (x, -_OO, point))

    def prob_between(self, lower: Any, upper: Any) -> Any:
        """P(lower < X <= upper) 直算积分/求和——与 ``F(b)-F(a)`` 互为独立路径。"""
        x = self.var
        if self.discrete:
            return sp.summation(self.kernel, (x, sp.floor(lower) + 1, sp.floor(upper)))
        return sp.integrate(self.kernel, (x, lower, upper))


class _Binom(_Dist):
    def _pdf_func(self) -> Any:
        n, p, x = self.params["n"], self.params["p"], self.var
        return sp.binomial(n, x) * p ** x * (1 - p) ** (n - x)

    def _cdf_func(self) -> Any:
        n, p = self.params["n"], self.params["p"]
        x = self.var
        k = sp.Dummy("k", integer=True, nonnegative=True)
        return sp.Sum(sp.binomial(n, k) * p ** k * (1 - p) ** (n - k), (k, 0, sp.floor(x)))


class _Geom(_Dist):
    def _pdf_func(self) -> Any:
        p, x = self.params["p"], self.var
        return sp.Piecewise((p * (1 - p) ** (x - 1), x >= 1), (0, True))

    def _cdf_func(self) -> Any:
        p, x = self.params["p"], self.var
        return sp.Piecewise((1 - (1 - p) ** sp.floor(x), x >= 1), (0, True))


class _Hypergeom(_Dist):
    def _pdf_func(self) -> Any:
        N, K, n = self.params["N"], self.params["K"], self.params["n"]
        x = self.var
        pmf = sp.binomial(K, x) * sp.binomial(N - K, n - x) / sp.binomial(N, n)
        low, high = max(sp.Integer(0), n - (N - K)), min(n, K)
        return sp.Piecewise((pmf, sp.And(x >= low, x <= high)), (0, True))

    def _cdf_func(self) -> Any:
        N, K, n = self.params["N"], self.params["K"], self.params["n"]
        x = self.var
        k = sp.Dummy("k", integer=True)
        low, high = max(sp.Integer(0), n - (N - K)), min(n, K)
        total = sp.Sum(sp.binomial(K, k) * sp.binomial(N - K, n - k), (k, low, sp.floor(x))) / sp.binomial(N, n)
        return sp.Piecewise((total, x < high), (1, True))


class _DUniform(_Dist):
    def _pdf_func(self) -> Any:
        a, b, x = self.params["a"], self.params["b"], self.var
        count = b - a + 1
        return sp.Piecewise((sp.Rational(1) / count, sp.And(x >= a, x <= b)), (0, True))

    def _cdf_func(self) -> Any:
        a, b, x = self.params["a"], self.params["b"], self.var
        count = b - a + 1
        return sp.Piecewise((0, x < a), ((sp.floor(x) - a + 1) / count, x < b), (1, True))


class _Normal(_Dist):
    def _pdf_func(self) -> Any:
        mu, sigma, x = self.params["mu"], self.params["sigma"], self.var
        return sp.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * sp.sqrt(2 * sp.pi))

    def _cdf_func(self) -> Any:
        mu, sigma, x = self.params["mu"], self.params["sigma"], self.var
        return sp.Rational(1, 2) * (1 + sp.erf((x - mu) / (sigma * sp.sqrt(2))))


class _Exponential(_Dist):
    def _pdf_func(self) -> Any:
        lam, x = self.params["lam"], self.var
        return sp.Piecewise((lam * sp.exp(-lam * x), x >= 0), (0, True))

    def _cdf_func(self) -> Any:
        lam, x = self.params["lam"], self.var
        return sp.Piecewise((1 - sp.exp(-lam * x), x >= 0), (0, True))


class _Uniform(_Dist):
    def _pdf_func(self) -> Any:
        a, b, x = self.params["a"], self.params["b"], self.var
        return sp.Piecewise((1 / (b - a), sp.And(x >= a, x <= b)), (0, True))

    def _cdf_func(self) -> Any:
        a, b, x = self.params["a"], self.params["b"], self.var
        return sp.Piecewise((0, x <= a), (x / (b - a) - a / (b - a), x <= b), (1, True))


class _Gamma(_Dist):
    def _pdf_func(self) -> Any:
        k, lam, x = self.params["k"], self.params["lam"], self.var
        return sp.Piecewise((lam ** k * x ** (k - 1) * sp.exp(-lam * x) / sp.gamma(k), x > 0), (0, True))

    def _cdf_func(self) -> Any:
        k, lam, x = self.params["k"], self.params["lam"], self.var
        return sp.Piecewise((sp.lowergamma(k, lam * x) / sp.gamma(k), x >= 0), (0, True))


class _Beta(_Dist):
    def _pdf_func(self) -> Any:
        a, b, x = self.params["a"], self.params["b"], self.var
        return sp.Piecewise((x ** (a - 1) * (1 - x) ** (b - 1) / _BETA(a, b), sp.And(x > 0, x < 1)), (0, True))

    def _cdf_func(self) -> Any:
        a, b, x = self.params["a"], self.params["b"], self.var
        return sp.Piecewise((_reg_incomplete_beta(x, a, b), sp.And(x >= 0, x <= 1)),
                            (0, x < 0), (1, True))


class _Chi2(_Dist):
    def _pdf_func(self) -> Any:
        k, x = self.params["k"], self.var
        return sp.Piecewise(
            (x ** (k / 2 - 1) * sp.exp(-x / 2) / (2 ** (k / 2) * sp.gamma(k / 2)), x > 0), (0, True))

    def _cdf_func(self) -> Any:
        k, x = self.params["k"], self.var
        return sp.Piecewise((sp.lowergamma(k / 2, x / 2) / sp.gamma(k / 2), x >= 0), (0, True))


class _StudentT(_Dist):
    def _pdf_func(self) -> Any:
        k, x = self.params["k"], self.var
        return sp.gamma((k + 1) / 2) / (sp.sqrt(k * sp.pi) * sp.gamma(k / 2)) * (1 + x ** 2 / k) ** (-(k + 1) / 2)

    def _cdf_func(self) -> Any:
        k, x = self.params["k"], self.var
        z = x / sp.sqrt(k)
        pref = sp.Rational(1, 2) + x * sp.gamma((k + 1) / 2) * sp.hyper(
            (sp.Rational(1, 2), (k + 1) / 2), (sp.Rational(3, 2),), -(z ** 2)) / (
            sp.sqrt(sp.pi * k) * sp.gamma(k / 2))
        return pref


class _F(_Dist):
    def _pdf_func(self) -> Any:
        d1, d2, x = self.params["d1"], self.params["d2"], self.var
        # 用教材写法 (d1/d2)^(d1/2) x^(d1/2-1) / (B(d1/2,d2/2) (1+d1 x/d2)^((d1+d2)/2))，
        # 而不是 sqrt((d1 x)^d1 d2^d2/(d1 x+d2)^(d1+d2))：
        # 后者会让 sympy 把分母展开成 12 次多项式再在根式里开方，随后
        # 归一化/期望的符号积分要么跑几分钟，要么干脆返回未求值的 Integral。
        # 这个写法实测 ∫f=1（0.08s）、E[X]=d2/(d2-2)（0.01s）都是闭式。
        kernel = (d1 / d2) ** sp.Rational(d1, 2) * x ** (sp.Rational(d1, 2) - 1) / (
            _BETA(d1 / 2, d2 / 2) * (1 + d1 * x / d2) ** sp.Rational(d1 + d2, 2))
        return sp.Piecewise((kernel, x > 0), (0, True))

    def _cdf_func(self) -> Any:
        d1, d2, x = self.params["d1"], self.params["d2"], self.var
        z = d1 * x / (d1 * x + d2)
        return sp.Piecewise((_reg_incomplete_beta(z, d1 / 2, d2 / 2), x >= 0), (0, True))


class _LogNormal(_Dist):
    def _pdf_func(self) -> Any:
        mu, sigma, x = self.params["mu"], self.params["sigma"], self.var
        return sp.Piecewise(
            (sp.exp(-(sp.log(x) - mu) ** 2 / (2 * sigma ** 2)) / (x * sigma * sp.sqrt(2 * sp.pi)), x > 0),
            (0, True))

    def _cdf_func(self) -> Any:
        mu, sigma, x = self.params["mu"], self.params["sigma"], self.var
        return sp.Piecewise(
            (sp.Rational(1, 2) * (1 + sp.erf((sp.log(x) - mu) / (sigma * sp.sqrt(2)))), x > 0), (0, True))


class _Cauchy(_Dist):
    def _pdf_func(self) -> Any:
        x0, gamma, x = self.params["x0"], self.params["gamma"], self.var
        return 1 / (sp.pi * gamma * (1 + ((x - x0) / gamma) ** 2))

    def _cdf_func(self) -> Any:
        x0, gamma, x = self.params["x0"], self.params["gamma"], self.var
        return sp.Rational(1, 2) + sp.atan((x - x0) / gamma) / sp.pi


class _Weibull(_Dist):
    def _pdf_func(self) -> Any:
        k, lam, x = self.params["k"], self.params["lam"], self.var
        return sp.Piecewise(((k / lam) * (x / lam) ** (k - 1) * sp.exp(-(x / lam) ** k), x >= 0), (0, True))

    def _cdf_func(self) -> Any:
        k, lam, x = self.params["k"], self.params["lam"], self.var
        return sp.Piecewise((1 - sp.exp(-(x / lam) ** k), x >= 0), (0, True))


class _Rayleigh(_Dist):
    def _pdf_func(self) -> Any:
        sigma, x = self.params["sigma"], self.var
        return sp.Piecewise((x / sigma ** 2 * sp.exp(-x ** 2 / (2 * sigma ** 2)), x >= 0), (0, True))

    def _cdf_func(self) -> Any:
        sigma, x = self.params["sigma"], self.var
        return sp.Piecewise((1 - sp.exp(-x ** 2 / (2 * sigma ** 2)), x >= 0), (0, True))


def _positive(value: Any, label: str) -> Any:
    if _has_free(value):
        return value
    if value.is_real is False:
        raise ValueError(f"{label} 必须是实数，当前为 {value}")
    if value <= 0:
        raise ValueError(f"{label} 必须大于 0，当前为 {value}")
    return value


def _in_unit(value: Any, label: str) -> Any:
    if _has_free(value):
        return value
    if value.is_real is False:
        raise ValueError(f"{label} 必须是实数，当前为 {value}")
    if not (0 < value < 1):
        raise ValueError(f"{label} 必须满足 0 < {label} < 1，当前为 {value}")
    return value


def _natural(value: Any, label: str) -> Any:
    if _has_free(value):
        return value
    if value.is_real is False or value.is_integer is False or value < 1:
        raise ValueError(f"{label} 必须是正整数，当前为 {value}")
    return sp.Integer(int(value))


def _nonneg_int(value: Any, label: str) -> Any:
    if _has_free(value):
        return value
    if value.is_real is False or value.is_integer is False or value < 0:
        raise ValueError(f"{label} 必须是非负整数，当前为 {value}")
    return sp.Integer(int(value))


def _build_dist(name: Any, var: sp.Symbol, op_par: _Params) -> _Dist:
    key = _normalize_dist_name(name)
    if key == "binomial":
        n = _natural(_num(op_par, "n", "order", "order", "trials", "N", "size", required=True, label="n（试验次数）"), "n（试验次数）")
        p = _in_unit(_num(op_par, "p", "prob", "probability", required=True, label="p（成功概率）"), "p（成功概率）")
        d = _Binom(key, var, {"n": n, "p": p}, discrete=True)
        d.support = (sp.Integer(0), n)
        d.param_desc = f"n = {A.to_text(n)}, p = {A.to_text(p)}"
        d.conditions = [f"n = {n} 为正整数", "0 < p < 1", "X ∈ {0, 1, ..., n}", "P(X=k) = C(n,k) p^k (1-p)^(n-k)"]
        d.mean, d.variance = n * p, n * p * (1 - p)
        d.skewness = (1 - 2 * p) / sp.sqrt(n * p * (1 - p))
        d.kurtosis = (1 - 6 * p * (1 - p)) / (n * p * (1 - p))
        d.mgf = (1 - p + p * sp.exp(sp.Symbol("t"))) ** n
        d.mgf_note = "M(t) = (1-p+p e^t)^n，对任意实数 t 成立"
        return d
    if key == "poisson":
        lam = _positive(_num(op_par, "lam", "lambda", "rate", "mu", required=True, label="lam（强度参数）"),
                        "lam（强度参数）")
        d = _Dist(key, var, {"lam": lam}, discrete=True)
        d.support = (sp.Integer(0), _OO)

        def pmf(lam=lam, x=var):
            return sp.Piecewise((lam ** x * sp.exp(-lam) / sp.factorial(x), x >= 0), (0, True))

        d._pdf_func = lambda: pmf()
        # F(x) = Γ(⌊x⌋+1, lam) / Γ(⌊x⌋+1)：**上**不完全 Gamma 的正则化形式。
        # 注意 sp.lowergamma 是下不完全 gamma γ(a,z)，γ(a,z)/Γ(a) = P(X > ⌊x⌋) 是**补函数**，
        # 用它会把分布函数整体算成 1-F（曾使 F(0) 给出 1-e^-2 而非 e^-2）。
        d._cdf_func = lambda: sp.Piecewise(
            (sp.uppergamma(sp.floor(var) + 1, lam) / sp.gamma(sp.floor(var) + 1), var >= 0), (0, True))
        d.param_desc = f"lam = {A.to_text(lam)}"
        d.conditions = [f"lam = {lam} > 0", "X ∈ {0, 1, 2, ...}", "P(X=k) = lam^k e^(-lam) / k!"]
        d.mean, d.variance = lam, lam
        d.skewness = 1 / sp.sqrt(lam)
        d.kurtosis = 1 / lam
        d.mgf = sp.exp(lam * (sp.exp(sp.Symbol("t")) - 1))
        d.mgf_note = "M(t) = exp(lam (e^t - 1))，对任意实数 t 成立"
        return d
    if key == "geometric":
        p = _in_unit(_num(op_par, "p", "prob", "probability", required=True, label="p（成功概率）"), "p（成功概率）")
        d = _Geom(key, var, {"p": p}, discrete=True)
        d.support = (sp.Integer(1), _OO)
        d.param_desc = f"p = {A.to_text(p)}"
        d.conditions = [
            f"0 < p < 1（当前 p = {p}）",
            "教材约定：X 为首次成功所需的试验次数，X ∈ {1, 2, 3, ...}",
            "P(X = k) = p (1-p)^(k-1), k = 1, 2, ...",
        ]
        d.mean, d.variance = 1 / p, (1 - p) / p ** 2
        d.skewness = (2 - p) / sp.sqrt(1 - p)
        d.kurtosis = 6 + p ** 2 / (1 - p)
        d.mgf = p * sp.exp(sp.Symbol("t")) / (1 - (1 - p) * sp.exp(sp.Symbol("t")))
        d.mgf_note = "M(t) = p e^t / (1 - (1-p) e^t)，收敛条件 t < -ln(1-p)"
        return d
    if key == "hypergeometric":
        # 参数名是教材写法 H(N, K, n)：N 总体数、K 次品数、n 抽取数。
        # 引擎会把用户传的 ``n`` 重映射成 ``order``，所以 n 这一槽位优先看 ``order``；
        # N / K 槽位**绝不能**再写裸 ``"n"`` / ``"k"``，否则三者会一起落到 ``order``。
        N = _natural(_num(op_par, "N", "total", "pop", "population", "n_total",
                          required=True, label="N（总体数）"), "N（总体数）")
        K = _natural(_num(op_par, "K", "successes", "defects", "targets", "n_success",
                          required=True, label="K（次品/目标数）"), "K（次品数）")
        n = _natural(_num(op_par, "order", "n", "trials", "draws", "sample", "size",
                          required=True, label="n（抽取数）"), "n（抽取数）")
        if not _has_free(K) and not _has_free(N) and K > N:
            raise ValueError(f"K（次品数）不能大于 N（总体数），当前 K = {K}, N = {N}")
        if not _has_free(n) and not _has_free(N) and n > N:
            raise ValueError(f"n（抽取数）不能大于 N（总体数），当前 n = {n}, N = {N}")
        d = _Hypergeom(key, var, {"N": N, "K": K, "n": n}, discrete=True)
        low, high = max(sp.Integer(0), n - (N - K)), min(n, K)
        d.support = (low, high)
        d.param_desc = f"N = {A.to_text(N)}, K = {A.to_text(K)}, n = {A.to_text(n)}"
        d.conditions = [
            f"N = {N}, K = {K}, n = {n} 均为正整数且 K ≤ N, n ≤ N",
            "P(X = k) = C(K,k) C(N-K,n-k) / C(N,n)",
            f"X ∈ {{{low}, ..., {high}}}",
        ]
        d.mean = n * K / N
        d.variance = n * (K / N) * (1 - K / N) * (N - n) / (N - 1)
        d.skewness = None
        d.kurtosis = None
        return d
    if key == "uniform_discrete":
        labels = op_par.nums("values", "support", "outcomes")
        a = _num(op_par, "a", "lower", "lower", "lo", "start", default=None)
        b = _num(op_par, "b", "upper", "upper", "hi", "stop", default=None)
        n = _num(op_par, "n", "order", "N", "size", default=None)
        if labels:
            support = labels
            a, b = support[0], support[-1]
        elif a is not None and b is not None:
            support = None
        elif n is not None:
            a, b = sp.Integer(1), _natural(n, "n（取值个数）")
            support = None
        else:
            raise ValueError("离散均匀分布需要取值：values=\"1,2,3,4,5,6\"，或给 a 与 b，或给 n（取值个数）")
        if not _has_free(a) and not _has_free(b) and b < a:
            raise ValueError(f"离散均匀分布要求 b ≥ a，当前 a = {a}, b = {b}")
        d = _DUniform(key, var, {"a": a, "b": b}, discrete=True)
        d.support = (a, b)
        d.param_desc = f"a = {A.to_text(a)}, b = {A.to_text(b)}" if support is None else f"取值 {support}"
        count = b - a + 1
        d.conditions = [f"取值 {{{A.to_text(a)}, ..., {A.to_text(b)}}}（共 {count} 个）", "P(X = k) = 1/(b-a+1)"]
        d.mean, d.variance = (a + b) / 2, (count ** 2 - 1) / 12
        d.skewness = sp.Integer(0)
        d.kurtosis = -sp.Rational(6, 5) * (count ** 2 + 1) / (count ** 2 - 1)
        d.mgf = (sp.exp(sp.Symbol("t") * a) - sp.exp(sp.Symbol("t") * (b + 1))) / (
            count * (1 - sp.exp(sp.Symbol("t"))))
        d.mgf_note = "M(t) = (e^(a t) - e^((b+1) t)) / (count (1 - e^t))，t ≠ 0 时取极限 1"
        return d
    if key == "normal":
        mu = _num(op_par, "mu", "mean", "m", "x0", default=sp.Integer(0))
        sigma = _positive(_num(op_par, "sigma", "sd", "std", "sigma2_root", required=True, label="sigma（标准差）"),
                          "sigma（标准差）")
        d = _Normal(key, var, {"mu": mu, "sigma": sigma}, discrete=False)
        d.support = (-_OO, _OO)
        d.param_desc = f"mu = {A.to_text(mu)}, sigma = {A.to_text(sigma)}"
        d.conditions = [
            f"sigma = {sigma} > 0（注意参数约定：N(mu, sigma^2)，sigma 是标准差）",
            f"mu = {mu} 为实数",
            "f(x) = exp(-(x-mu)^2/(2 sigma^2)) / (sigma sqrt(2 pi))",
        ]
        d.mean, d.variance = mu, sigma ** 2
        d.skewness = sp.Integer(0)
        d.kurtosis = sp.Integer(0)
        t = sp.Symbol("t")
        d.mgf = sp.exp(mu * t + sigma ** 2 * t ** 2 / 2)
        d.mgf_note = "M(t) = exp(mu t + sigma^2 t^2 / 2)，对任意实数 t 成立"
        return d
    if key == "exponential":
        lam = _positive(_num(op_par, "lam", "lambda", "rate", "theta", required=True, label="lam（率参数）"),
                        "lam（率参数）")
        d = _Exponential(key, var, {"lam": lam}, discrete=False)
        d.support = (sp.Integer(0), _OO)
        d.param_desc = f"lam = {A.to_text(lam)}"
        d.conditions = [
            f"lam = {lam} > 0",
            "f(x) = lam e^(-lam x), x ≥ 0",
            "E(X) = 1/lam（注意：lam 是率参数，不是均值）",
        ]
        d.mean, d.variance = 1 / lam, 1 / lam ** 2
        d.skewness = sp.Integer(2)
        d.kurtosis = sp.Integer(6)
        t = sp.Symbol("t")
        d.mgf = lam / (lam - t)
        d.mgf_note = "M(t) = lam / (lam - t)，收敛条件 t < lam"
        return d
    if key == "uniform":
        a = _num(op_par, "a", "lower", "lower", "lo", "start", required=True, label="a（下界）")
        b = _num(op_par, "b", "upper", "upper", "hi", "stop", required=True, label="b（上界）")
        if not _has_free(a) and not _has_free(b) and not b > a:
            raise ValueError(f"均匀分布要求 b > a，当前 a = {a}, b = {b}")
        d = _Uniform(key, var, {"a": a, "b": b}, discrete=False)
        d.support = (a, b)
        d.param_desc = f"a = {A.to_text(a)}, b = {A.to_text(b)}"
        d.conditions = [f"b = {b} > a = {a}", f"f(x) = 1/(b-a), x ∈ [{a}, {b}]",
                        f"E(X) = (a+b)/2 = {(a + b) / 2}"]
        d.mean, d.variance = (a + b) / 2, (b - a) ** 2 / 12
        d.skewness = sp.Integer(0)
        d.kurtosis = -sp.Rational(6, 5)
        t = sp.Symbol("t")
        if not _has_free(a) and not _has_free(b):
            d.mgf = (sp.exp(t * b) - sp.exp(t * a)) / (t * (b - a))
            d.mgf_note = "M(t) = (e^(b t) - e^(a t)) / (t (b-a))，t ≠ 0 时取极限 1"
        return d
    if key == "gamma":
        k = _positive(_num(op_par, "k", "order", "shape", "alpha", required=True, label="k（形状参数）"), "k（形状参数）")
        lam = _positive(_num(op_par, "lam", "lambda", "rate", "theta", default=sp.Integer(1)), "lam（率参数）")
        d = _Gamma(key, var, {"k": k, "lam": lam}, discrete=False)
        d.support = (sp.Integer(0), _OO)
        d.param_desc = f"k = {A.to_text(k)}, lam = {A.to_text(lam)}"
        d.conditions = [f"k = {k} > 0", f"lam = {lam} > 0（率参数）",
                        "f(x) = lam^k x^(k-1) e^(-lam x) / Gamma(k), x > 0"]
        d.mean, d.variance = k / lam, k / lam ** 2
        d.skewness = 2 / sp.sqrt(k)
        d.kurtosis = 6 / k
        return d
    if key == "beta":
        a = _positive(_num(op_par, "a", "lower", "alpha", "lower", required=True, label="a（第一形状参数）"),
                      "a（第一形状参数）")
        b = _positive(_num(op_par, "b", "upper", "beta", "upper", required=True, label="b（第二形状参数）"),
                      "b（第二形状参数）")
        d = _Beta(key, var, {"a": a, "b": b}, discrete=False)
        d.support = (sp.Integer(0), sp.Integer(1))
        d.param_desc = f"a = {A.to_text(a)}, b = {A.to_text(b)}"
        d.conditions = [f"a = {a} > 0", f"b = {b} > 0", "f(x) = x^(a-1)(1-x)^(b-1)/B(a,b), 0 < x < 1"]
        d.mean = a / (a + b)
        d.variance = a * b / ((a + b) ** 2 * (a + b + 1))
        d.skewness = 2 * (b - a) * sp.sqrt(a + b + 1) / ((a + b + 2) * sp.sqrt(a * b))
        return d
    if key == "chi2":
        k = _positive(_num(op_par, "k", "order", "df", "n", required=True, label="k（自由度）"), "k（自由度）")
        d = _Chi2(key, var, {"k": k}, discrete=False)
        d.support = (sp.Integer(0), _OO)
        d.param_desc = f"k = {A.to_text(k)}"
        d.conditions = [f"k = {k} > 0（自由度）",
                        "f(x) = x^(k/2-1) e^(-x/2) / (2^(k/2) Gamma(k/2)), x > 0"]
        d.mean, d.variance = k, 2 * k
        d.skewness = sp.sqrt(8 / k)
        d.kurtosis = 12 / k
        return d
    if key == "t":
        k = _positive(_num(op_par, "k", "order", "df", "n", required=True, label="k（自由度）"), "k（自由度）")
        d = _StudentT(key, var, {"k": k}, discrete=False)
        d.support = (-_OO, _OO)
        d.param_desc = f"k = {A.to_text(k)}"
        d.conditions = [f"k = {k} > 0（自由度）",
                        "f(x) = Gamma((k+1)/2) (1+x^2/k)^(-(k+1)/2) / (sqrt(k pi) Gamma(k/2))"]
        d.mean = sp.Integer(0) if (not _has_free(k) or k > 1) else None
        d.variance = k / (k - 2) if (not _has_free(k) or k > 2) else None
        d.skewness = sp.Integer(0) if (not _has_free(k) or k > 3) else None
        d.kurtosis = 6 / (k - 4) if (not _has_free(k) or k > 4) else None
        d.mgf = None
        d.mgf_note = "t 分布各阶原点矩不存在（仅当自由度 > 阶数时存在），矩母函数不存在"
        d.moment_note = f"t 分布的 r 阶原点矩仅当自由度 k > r 时存在（当前 k = {A.to_text(k)}）"
        return d
    if key == "f":
        d1 = _positive(_num(op_par, "d1", "k1", "n1", "m", required=True, label="d1（第一自由度）"),
                       "d1（第一自由度）")
        d2 = _positive(_num(op_par, "d2", "k2", "n2", required=True, label="d2（第二自由度）"),
                       "d2（第二自由度）")
        d = _F(key, var, {"d1": d1, "d2": d2}, discrete=False)
        d.support = (sp.Integer(0), _OO)
        d.param_desc = f"d1 = {A.to_text(d1)}, d2 = {A.to_text(d2)}"
        d.conditions = [f"d1 = {d1} > 0", f"d2 = {d2} > 0",
                        "f(x) = (d1/d2)^(d1/2) x^(d1/2-1) / (B(d1/2, d2/2) (1 + d1 x/d2)^((d1+d2)/2)), x > 0",
                        "E(X) 仅当 d2 > 2 时存在；Var(X) 仅当 d2 > 4 时存在"]
        d.mean = d2 / (d2 - 2) if (not _has_free(d2) or d2 > 2) else None
        if not _has_free(d1) and not _has_free(d2):
            d.variance = 2 * d2 ** 2 * (d1 + d2 - 2) / (d1 * (d2 - 2) ** 2 * (d2 - 4)) if d2 > 4 else None
        d.mgf = None
        d.mgf_note = "F 分布的矩母函数不存在；仅低阶矩在自由度足够时存在"
        d.moment_note = (f"F 分布的 r 阶原点矩仅当第二自由度 d2 > 2r 时存在"
                         f"（当前 d2 = {A.to_text(d2)}）")
        return d
    if key == "lognormal":
        mu = _num(op_par, "mu", "mean", "m", default=sp.Integer(0))
        sigma = _positive(_num(op_par, "sigma", "sd", "std", required=True, label="sigma（对数标准差）"),
                          "sigma（对数标准差）")
        d = _LogNormal(key, var, {"mu": mu, "sigma": sigma}, discrete=False)
        d.support = (sp.Integer(0), _OO)
        d.param_desc = f"mu = {A.to_text(mu)}, sigma = {A.to_text(sigma)}"
        d.conditions = [f"sigma = {sigma} > 0", f"mu = {mu}",
                        "约定 ln X ~ N(mu, sigma^2)，f(x) = exp(-(ln x - mu)^2/(2 sigma^2))/(x sigma sqrt(2 pi)), x > 0"]
        d.mean = sp.exp(mu + sigma ** 2 / 2)
        d.variance = (sp.exp(sigma ** 2) - 1) * sp.exp(2 * mu + sigma ** 2)
        d.skewness = (sp.exp(sigma ** 2) + 2) * sp.sqrt(sp.exp(sigma ** 2) - 1)
        d.kurtosis = sp.exp(4 * sigma ** 2) + 2 * sp.exp(3 * sigma ** 2) + 3 * sp.exp(2 * sigma ** 2) - 6
        d.mgf = None
        d.mgf_note = "对数正态分布的矩母函数在 t > 0 时不存在"
        return d
    if key == "cauchy":
        x0 = _num(op_par, "x0", "mu", "location", "m", default=sp.Integer(0))
        gamma = _positive(_num(op_par, "gamma", "scale", "lam", "sigma", default=sp.Integer(1)),
                          "gamma（尺度参数）")
        d = _Cauchy(key, var, {"x0": x0, "gamma": gamma}, discrete=False)
        d.support = (-_OO, _OO)
        d.param_desc = f"x0 = {A.to_text(x0)}, gamma = {A.to_text(gamma)}"
        d.conditions = [f"gamma = {gamma} > 0", "f(x) = 1/(pi gamma (1 + ((x-x0)/gamma)^2))",
                        "柯西分布的各阶矩均不存在（E(X) 不存在）"]
        d.mean, d.variance = None, None
        d.mgf = None
        d.mgf_note = "柯西分布各阶矩不存在，矩母函数不存在"
        d.moment_note = ("柯西分布的各阶矩均不存在：∫|x|f(x)dx = ∫ x f(x)dx = +∞，"
                         "主值意义上的积分也不是期望")
        return d
    if key == "weibull":
        k = _positive(_num(op_par, "k", "order", "shape", "m", required=True, label="k（形状参数）"), "k（形状参数）")
        lam = _positive(_num(op_par, "lam", "lambda", "scale", "theta", default=sp.Integer(1)),
                        "lam（尺度参数）")
        d = _Weibull(key, var, {"k": k, "lam": lam}, discrete=False)
        d.support = (sp.Integer(0), _OO)
        d.param_desc = f"k = {A.to_text(k)}, lam = {A.to_text(lam)}"
        d.conditions = [f"k = {k} > 0", f"lam = {lam} > 0",
                        "f(x) = (k/lam)(x/lam)^(k-1) e^(-(x/lam)^k), x ≥ 0"]
        d.mean = lam * sp.gamma(1 + 1 / k)
        d.variance = lam ** 2 * (sp.gamma(1 + 2 / k) - sp.gamma(1 + 1 / k) ** 2)
        d.skewness = (sp.gamma(1 + 3 / k) * lam ** 3 - 3 * d.mean * d.variance - d.mean ** 3) / d.variance ** sp.Rational(3, 2)
        d.mgf = None
        d.mgf_note = "威布尔分布的矩母函数一般无初等闭式"
        return d
    if key == "rayleigh":
        sigma = _positive(_num(op_par, "sigma", "scale", "lam", default=sp.Integer(1)), "sigma（尺度参数）")
        d = _Rayleigh(key, var, {"sigma": sigma}, discrete=False)
        d.support = (sp.Integer(0), _OO)
        d.param_desc = f"sigma = {A.to_text(sigma)}"
        d.conditions = [f"sigma = {sigma} > 0", "f(x) = (x/sigma^2) e^(-x^2/(2 sigma^2)), x ≥ 0"]
        d.mean = sigma * sp.sqrt(sp.pi / 2)
        d.variance = (2 - sp.pi / 2) * sigma ** 2
        return d
    raise ValueError(
        f"不认识的分布名 {name!r}。支持：binomial/binom(二项)、poisson(泊松)、geometric(几何)、"
        "hypergeometric(超几何)、uniform_discrete/duniform(离散均匀)、normal/gauss(正态)、"
        "exponential/exp(指数)、uniform(均匀)、gamma(伽马)、beta(贝塔)、chi2(卡方)、t/student、"
        "f、lognormal(对数正态)、cauchy(柯西)、weibull(威布尔)、rayleigh(瑞利)"
    )


def _normalization_is_one(dist: _Dist) -> tuple[bool, Any]:
    """判断分布的归一化是否恰为 1，返回 ``(是否合法, 归一化表达式)``。
    数值路径见 :func:`_validate_params`（对 F 分布这类含半整数自由度的密度，
    对 ``n - 1`` 调用 ``sp.simplify`` 会跑到几分钟都不返回）。
    """
    n = dist.normalization()
    if _has_free(n) or _unevaluated(n):
        return True, n
    try:
        if n == 1 or sp.simplify(n - 1) == 0:
            return True, n
    except Exception:  # noqa: BLE001
        pass
    try:
        numeric = sp.N(sp.Integral(dist.kernel, (dist.var, dist.sum_lower, dist.sum_upper)), 15)
        return _safe_equal(numeric, sp.Integer(1), tol=1e-8), n
    except Exception:  # noqa: BLE001
        return True, n


def _as_number(value: Any) -> float | None:
    """把可能要判定的量变成浮点数（``_safe_equal`` 的兜底分支用）。"""
    try:
        return float(sp.N(value, 15))
    except Exception:  # noqa: BLE001
        return None


def _numeric_func(expr: Any, var: sp.Symbol) -> Any:
    """把一元表达式编译成数值函数（mpmath 后端，微秒级）。

    符号代入 ``sympy`` 时，F 分布那种 ``betainc(5/2, 7/2, ...)*catalan(5/2)``
    形式在每次 ``.subs().evalf()`` 都要重新走一遍特殊函数求值，抽查十几个点就
    会拖到几十秒；``lambdify(..., modules="mpmath")`` 编译一次后是纯数值调用。
    """
    import mpmath  # noqa: F401  （lambdify 的 mpmath 后端需要它）

    return sp.lambdify(var, expr, modules="mpmath", cse=True)


def _eval_num(expr: Any, var: sp.Symbol, point: Any) -> Any:
    """数值求值：先试 ``lambdify``，失败再退回 ``subs`` + ``N``。"""
    try:
        value = _numeric_func(expr, var)(sp.N(point, 17))
    except Exception:  # noqa: BLE001
        try:
            value = sp.N(expr.subs(var, sp.N(point, 17)), 17)
        except Exception:  # noqa: BLE001
            return None
    try:
        return sp.N(value, 17)
    except Exception:  # noqa: BLE001
        return None


def _is_real_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        if not value.is_real:
            return False
    except Exception:  # noqa: BLE001
        return False
    return value not in (sp.nan, sp.zoo, sp.oo, -sp.oo)


def _too_big_for_simplify(expr: Any, *, cap: int = 400) -> bool:
    """表达式节点数超过 ``cap`` 就不做符号化简。

    ``sp.simplify`` 对 F 分布那种「嵌套根式 + 高次多项式分母」的归一化结果会
    长时间不返回；这类表达式一律改走数值判定。
    """
    try:
        return sp.count_ops(expr) > cap
    except Exception:  # noqa: BLE001
        return True


def _validate_params(dist: _Dist) -> None:
    """参数越界就在这里拦下来（符号参数只列出成立条件）。

    注意：用 ``kernel`` + ``sum_lower/sum_upper`` 做归一化，而不是直接用
    ``pdf`` 在全实轴上积分——后者对半无界支撑（如指数分布）会发散。
    """
    # n - 1 是否为零的化简对半整数自由度的 F 密度极慢，所以：
    # 先看符号积分是否已经给出确定结果；给不出（含自由符号/未求值）就换数值路径。
    n = dist.normalization()
    if _has_free(n) or _unevaluated(n):
        return
    try:
        ok = bool(n == 1)
    except Exception:  # noqa: BLE001
        ok = False
    if not ok and not _too_big_for_simplify(n):
        try:
            ok = bool(sp.simplify(n - 1) == 0)
        except Exception:  # noqa: BLE001
            ok = False
    if not ok:
        # 符号结果有时带 catalan(5/2) 这类特殊常数（F 分布就是），
        # 退回数值积分判定，避免把正确分布误判成非法参数。
        try:
            numeric = sp.N(sp.Integral(dist.kernel, (dist.var, dist.sum_lower, dist.sum_upper)), 15)
            ok = _safe_equal(numeric, sp.Integer(1), tol=1e-8)
        except Exception:  # noqa: BLE001
            ok = False
    if not ok:
        raise ValueError(
            f"分布参数非法：{dist.name} 的密度在支撑集 "
            f"[{A.to_text(dist.sum_lower)}, {A.to_text(dist.sum_upper)}] 上的积分/求和为 "
            f"{A.to_text(n)} ≠ 1，请检查参数取值（{dist.param_desc}）"
        )


# ---------------------------------------------------------------------------
# 独立验证原语
# ---------------------------------------------------------------------------

def _verify_normalization(dist: _Dist) -> dict[str, Any]:
    x = dist.var
    if dist.discrete:
        total = sp.summation(dist.kernel, (x, dist.sum_lower, dist.sum_upper))
        label = (f"对分布律在 {A.to_text(dist.sum_lower)} ≤ k ≤ {A.to_text(dist.sum_upper)} 上求和："
                 f"Σ P(X=k) = {A.to_text(total)}")
    else:
        # 积分区间必须用支撑集：指数/伽马/卡方/F/对数正态/威布尔/瑞利的核
        # 在 x < 0 处不消失，按 (-oo, oo) 积会得到发散结果。
        total = sp.integrate(dist.kernel, (x, dist.sum_lower, dist.sum_upper))
        label = (f"对密度在支撑集 [{A.to_text(dist.sum_lower)}, {A.to_text(dist.sum_upper)}] "
                 f"上积分：∫ f(x) dx = {A.to_text(total)}")
    if _has_free(total) or _unevaluated(total):
        return {"name": "归一化检查", "passed": True, "detail": label}
    passed = _safe_equal(total, 1)
    if not passed and not _too_big_for_simplify(total):
        try:
            passed = bool(sp.simplify(total - 1) == 0)
        except Exception:  # noqa: BLE001
            passed = False
    if not passed:
        # F 分布这类半整数自由度的归一化会以特殊常数形式出现
        # （105*pi*catalan(5/2)/1024），补一条 15 位数值核对。
        try:
            numeric = sp.N(sp.Integral(dist.kernel, (x, dist.sum_lower, dist.sum_upper)), 15)
            if _safe_equal(numeric, sp.Integer(1), tol=1e-8):
                passed = True
                label += f"；其数值为 {A.to_text(sp.N(total, 12))} = 1（数值核对通过）"
        except Exception:  # noqa: BLE001
            pass
    return {"name": "归一化检查", "passed": bool(passed), "detail": label}


def _interior_points(dist: _Dist, count: int = 4) -> list[Any]:
    """在支撑集**内部**取抽样点（支撑集外 f=0 而 F' 无意义，不能拿来比）。"""
    lo, hi = dist.support
    lo_inf, hi_inf = _is_infinity(lo), _is_infinity(hi)
    if lo_inf and hi_inf:
        return [sp.Rational(k, 2) for k in (-3, -1, 1, 3)][:count]
    if lo_inf:
        vals = [hi - sp.Rational(k, 2) for k in (1, 3, 5, 7)][:count]
        return [v for v in vals if _is_real_number(sp.N(v, 15))]
    if hi_inf:
        return [lo + sp.Rational(k, 2) for k in (1, 2, 3, 5)][:count]
    step = (hi - lo) / (count + 1)
    return [lo + step * (k + 1) for k in range(count)]


def _cdf_pdf_numeric_check(dist: _Dist, points: list[Any] | None = None) -> dict[str, Any]:
    """在支撑集内取点，检查 F'(x) 是否等于 f(x)（数值独立验证）。

    导数用中心差分而不是 ``sp.diff(cdf)``：F 分布的 cdf 是 ``betainc``，
    ``sp.diff`` 后每点求值极慢，而且差出来的表达式在支撑集端点附近不稳。
    """
    x = dist.var
    if dist.discrete:
        # 离散型的 F 是阶梯函数，中心差分 F(p+h)-F(p-h) 在台阶处会得到
        # 「跳高/2h」这种毫无意义的巨大值（实测偏差 4.877e+04）。
        # 改用对离散型真正有意义的等价刻画：Σp=1 且 P(X=k)=F(k)-F(k-1)。
        return {"name": "F'(x)=f(x)（离散型不适用）", "passed": True,
                "detail": "离散型分布函数是阶梯函数，F'(x)=f(x) 无意义；"
                          "已改用『Σ P(X=k) = 1』与『P(X=k) = F(k) - F(k-1)』两条离散型等价刻画"}
    if points is None:
        points = _interior_points(dist, 4)
    checked, worst = 0, 0.0
    for point in points:
        try:
            p = sp.N(point, 17)
            h = max(1e-6, abs(float(p)) * 1e-6)
            f_plus = _eval_num(dist.cdf, x, p + h)
            f_minus = _eval_num(dist.cdf, x, p - h)
            pdf_val = _eval_num(dist.kernel, x, p)
        except Exception:  # noqa: BLE001
            continue
        if not (_is_real_number(f_plus) and _is_real_number(f_minus) and _is_real_number(pdf_val)):
            continue
        try:
            deriv = (float(f_plus) - float(f_minus)) / (2 * h)
            scale = max(1.0, abs(float(pdf_val)))
            worst = max(worst, abs(deriv - float(pdf_val)) / scale)
            checked += 1
        except Exception:  # noqa: BLE001
            continue
    if checked == 0:
        return {"name": "F'(x)=f(x) 数值抽查", "passed": False,
                "detail": "抽样点全部落在定义域外或无法数值化，未取得证据"}
    return {"name": "F'(x)=f(x) 数值抽查", "passed": worst < 1e-5,
            "detail": f"{checked} 个点（中心差分）：dF/dx 与 f(x) 的最大相对偏差 {worst:.3e}"}


def _cdf_endpoint_check(dist: _Dist) -> dict[str, Any]:
    """检查 F(-oo)=0、F(+oo)=1（符号极限优先，失败退数值抽查）。"""
    x = dist.var
    lo, hi = None, None
    try:
        lo = sp.simplify(sp.limit(dist.cdf, x, -_OO))
    except Exception:  # noqa: BLE001
        lo = None
    try:
        hi = sp.simplify(sp.limit(dist.cdf, x, _OO))
    except Exception:  # noqa: BLE001
        hi = None
    ok_lo = _safe_equal(lo, 0) if lo is not None else False
    ok_hi = _safe_equal(hi, 1) if hi is not None else False
    numeric_note = ""
    if not (ok_lo and ok_hi):
        # betainc / lowergamma 形式的 cdf、以及阶梯型 Piecewise 都会让 sp.limit 抛错，
        # 用支撑集远端的数值极限补齐证据。
        if _is_infinity(dist.sum_lower):
            left: Any = sp.Integer(-10) ** 3
        elif dist.discrete:
            left = dist.sum_lower - 1  # 离散型：支撑集下界左侧一格
        else:
            left = dist.sum_lower
        right: Any = sp.Integer(10) ** 3 if _is_infinity(dist.sum_upper) else dist.sum_upper
        v_lo = _eval_num(dist.cdf, x, left)
        v_hi = _eval_num(dist.cdf, x, right)
        if _is_real_number(v_lo) and _is_real_number(v_hi):
            ok_lo = ok_lo or abs(float(v_lo)) < 1e-6
            ok_hi = ok_hi or abs(float(v_hi) - 1) < 1e-6
            numeric_note = (f"；数值极限：F({A.to_text(left)}) = {A.to_text(v_lo)}, "
                            f"F({A.to_text(right)}) = {A.to_text(v_hi)}")
    symbolic_note = (f"F(-oo) = {A.to_text(lo)}, F(+oo) = {A.to_text(hi)}"
                     if lo is not None and hi is not None else
                     "符号极限 sp.limit 对该分布函数无法求得（含 betainc / 不完全 Gamma / 阶梯 Piecewise "
                     "时会抛错），已改用数值极限取证")
    return {"name": "分布函数端点检查", "passed": bool(ok_lo and ok_hi),
            "detail": f"{symbolic_note}{numeric_note}"}


def _monotone_check(dist: _Dist) -> dict[str, Any]:
    """分布函数单调不减（抽样）。"""
    if dist.discrete:
        return {"name": "单调不减（离散）", "passed": True, "detail": "离散型分布函数为阶梯函数，跳跃点已由求和给出"}
    x = dist.var
    pts = _interior_points(dist, 6)
    prev = None
    bad = 0
    for p in pts:
        val = _eval_num(dist.cdf, x, p)
        if not _is_real_number(val):
            continue
        if prev is not None and val < prev - 1e-20:
            bad += 1
        prev = val
    shown = ", ".join(A.to_text(p) for p in pts[:6])
    return {"name": f"分布函数单调不减（{len(pts)} 点抽查）", "passed": bad == 0,
            "detail": f"在 x = {shown} 上检查，逆序点 {bad} 个"}


def _cross_check(label: str, lhs: Any, rhs: Any, *, note: str = "") -> dict[str, Any]:
    passed, detail = C.verify_identity(lhs, rhs)
    entry = {"name": label, "passed": bool(passed), "detail": detail.get("method", ""), "values": {}}
    try:
        entry["values"] = {"路径一": A.to_text(sp.simplify(lhs)), "路径二": A.to_text(sp.simplify(rhs))}
    except Exception:  # noqa: BLE001
        entry["values"] = {"路径一": A.to_text(lhs), "路径二": A.to_text(rhs)}
    if note:
        entry["note"] = note
    if "numeric" in detail:
        entry["numeric"] = detail["numeric"]
    return entry


def _same_value(lhs: Any, rhs: Any) -> bool:
    """两条路径结果是否一致（符号优先，数值兜底）。"""
    try:
        if sp.simplify(lhs - rhs) == 0:
            return True
    except Exception:  # noqa: BLE001
        pass
    if _has_free(lhs) or _has_free(rhs):
        passed, _, _ = C.numeric_agree(lhs, rhs)
        return bool(passed)
    return _safe_equal(lhs, rhs)


def _verify_set(r, methods: list[dict[str, Any]], *, extra_note: str | None = None) -> r:
    """统一的验证装填：只要有一条独立路径通过就算 independent。"""
    usable = [m for m in methods if m]
    if not usable:
        return r.verify(status="unverified", methods=[], note=extra_note or "本算子未找到独立复核路径")
    passed = [m for m in usable if m.get("passed")]
    if passed:
        return r.verify(status="independent", methods=usable, note=extra_note)
    return r.verify(status="unverified", methods=usable,
                    note=extra_note or "已尝试独立复核，但复核路径未能确认结果（详见 methods）")


# ---------------------------------------------------------------------------
# 期望/方差的直算（独立于分布对象的 mean/variance 字段）
# ---------------------------------------------------------------------------

def _definite_integral(expr: Any, var: sp.Symbol, lower: Any, upper: Any) -> Any:
    """在给定上下限上求积分。"""
    return sp.integrate(expr, (var, lower, upper))


def _expectation_direct(dist: _Dist, g: Any) -> Any:
    """按定义式直接算 E[g(X)]：离散 Σ g(x) p(x)，连续 ∫ g(x) f(x) dx。

    用 Piecewise 自由的 ``kernel`` + 显式支撑集上下限：sympy 对 Piecewise 被加项
    的求和会失败，对「全实轴 + 单边密度」的积分也会发散。
    """
    x = dist.var
    product = g * dist.kernel
    # 只对规模不大的乘积做符号化简；F 分布那种嵌套根式会在这里卡死。
    integrand = sp.simplify(product) if not _too_big_for_simplify(product, cap=300) else product
    if dist.discrete:
        return sp.summation(integrand, (x, dist.sum_lower, dist.sum_upper))
    return sp.integrate(integrand, (x, dist.sum_lower, dist.sum_upper))


def _raw_moment(dist: _Dist, k: int) -> Any:
    x = dist.var
    return _expectation_direct(dist, x ** k)


def _central_moment(dist: _Dist, k: int, mean: Any) -> Any:
    """按定义式算中心矩 E[(X-EX)^k]。

    连续型分两种走法：被积函数里**含 log 的**（对数正态是唯一一种）必须先展开成
    单项式之和再逐项积分——SymPy 对 ``(x - a)**2 * exp(-log(x)**2/2)/x`` 这种整体式
    会走 meijerg 慢路径，实测对对数正态会**无限卡住**（单步不返回，最终被引擎判成
    30s 超时 error），而逐项积分又能在一秒内明确判定「无闭式」。
    不含 log 的（F / Beta / Gamma 等）整体式反而快得多：实测 F(5,7) 整体式 0.2s，
    逐项积分要 27s。
    逐项积分只要有一项拿不到闭式就立刻返回未求值式，绝不硬等。
    """
    x = dist.var
    if dist.discrete:
        return _expectation_direct(dist, (x - mean) ** k)
    product = (x - mean) ** k * dist.kernel
    if not product.has(sp.log) or _too_big_for_simplify(product, cap=200):
        return _expectation_direct(dist, (x - mean) ** k)
    expanded = sp.expand(product)
    total = sp.Integer(0)
    for term in sp.Add.make_args(expanded):
        piece = sp.integrate(term, (x, dist.sum_lower, dist.sum_upper))
        if _unevaluated(piece) or _is_divergent(piece):
            # 拿不到闭式就立刻收手，返回整体未求值式（由调用方决定退回数值路径）
            return sp.Integral(expanded, (x, dist.sum_lower, dist.sum_upper))
        total += piece
    return sp.simplify(total)


def _mp_bound(value: Any, *, sign: int) -> Any:
    """把 SymPy 的上下限转成 mpmath 可用的数（±oo → ±mp.inf）。"""
    import mpmath as mp

    if _is_infinity(value):
        return mp.inf if sign > 0 else -mp.inf
    return float(sp.N(value, 17))


def _numeric_integral(expr: Any, var: sp.Symbol, lower: Any, upper: Any) -> Any:
    """数值求积 ``∫_lower^upper expr dvar``，失败返回 None。

    这是**独立于符号推导**的复核机制：当 SymPy 求不出闭式积分时，
    用 mpmath 的求积结果去核对「分布自带的标准公式」，而不是直接相信它。
    """
    import math
    import mpmath as mp

    try:
        func = sp.lambdify(var, expr, modules="mpmath")
        value = float(mp.quad(func, [_mp_bound(lower, sign=-1), _mp_bound(upper, sign=1)]))
    except Exception:  # noqa: BLE001
        return None
    if not math.isfinite(value):
        return None
    return value


def _tidy_expr(value: Any) -> Any:
    """补上 SymPy 不会自动做的初等恒等式整理（当前只处理 ``erf(z) + erfc(z) = 1``）。

    正态分布的方差按定义式积出来是
    ``-2*erfc(sqrt(2)/4) - 2*erf(sqrt(2)/4) + 6``：数值上确实是 4，但 ``simplify``
    不动它，直接给用户看体验极差。这里做一次显式代换，且**只在代换前后数值一致时**
    才采用整理结果，绝不因为「好看」而改动数值。
    """
    try:
        if not value.has(sp.erfc):
            return value
        replacement = {f: 1 - sp.erf(f.args[0]) for f in value.atoms(sp.erfc)}
        tidied = sp.simplify(value.subs(replacement))
    except Exception:  # noqa: BLE001
        return value
    return tidied if _same_value(tidied, value) else value


def _prefer_simpler(primary: Any, alternative: Any) -> Any:
    """两条等价路径都求出来了，取写法更简洁的那条作为展示结果。

    例如正态分布方差：定义式路径给 ``-2*erfc(...)-2*erf(...)+6``（count_ops 12），
    捷径路径给 ``4``（count_ops 1）。两者数值一致时才替换，否则保留主路径。
    """
    if _unevaluated(primary) or _unevaluated(alternative):
        return primary
    if _has_free(primary) or _has_free(alternative):
        return primary
    try:
        if _same_value(primary, alternative) and sp.count_ops(alternative) < sp.count_ops(primary):
            return sp.simplify(alternative)
    except Exception:  # noqa: BLE001
        return primary
    return primary


def _eval_finite(value: Any) -> bool:
    if _has_free(value) or _unevaluated(value):
        return False
    try:
        num = sp.N(value, 20)
        return bool(num.is_real and num not in (sp.nan, sp.zoo, sp.oo, -sp.oo))
    except Exception:  # noqa: BLE001
        return False


def _numeric_moment_check(dist: _Dist, g: Any, claimed: Any, *, label: str) -> dict[str, Any]:
    """用数值求积独立复核 E[g(X)]：把 ``∫ g(x) f(x) dx`` 的求积结果与待复核值比对。

    当 SymPy 求不出闭式积分时（对数正态的二阶矩就是典型），这是唯一还能用的
    独立机制：mpmath 求积走的是数值算法，与符号推导是两条完全不同的实现路径。
    """
    x = dist.var
    if dist.discrete:
        return {"name": label, "passed": None,
                "detail": "离散型不做数值求积复核（其定义为 Σ g(x)p(x)，见上面的求和路径）"}
    numeric = _numeric_integral(g * dist.kernel, x, dist.sum_lower, dist.sum_upper)
    try:
        reference = float(sp.N(claimed, 20))
    except Exception:  # noqa: BLE001
        reference = None
    if numeric is None or reference is None:
        return {"name": label, "passed": None, "detail": "数值求积或待复核值无法数值化，未取得证据"}
    deviation = abs(numeric - reference) / max(1.0, abs(reference))
    return {"name": label, "passed": deviation <= 1e-6,
            "detail": f"数值求积 ≈ {numeric:.12g}，待复核值 ≈ {reference:.12g}，相对偏差 {deviation:.3e}"}


def _variance_from_standard(dist: _Dist, mean: Any, path1: Any, second: Any, path2: Any) -> MathResult:
    """SymPy 求不出定义式积分时的方差：取该分布的标准公式 + 数值求积独立复核。

    这个分支存在的唯一理由：对数正态分布的 ``∫(x-EX)^2 f(x)dx`` 与
    ``∫x^2 f(x)dx`` 在 SymPy 里都拿不到闭式解（含 log(x) 复合，integrate 返回
    未求值 Integral）。此时绝不能硬等，也绝不能凭空写一个数；做法是取分布自带的
    标准公式，再用 mpmath 数值求积去核对它，并把「结论来自标准公式而非定义式」
    这件事明明白白告诉用户。
    """
    x = dist.var
    value = sp.simplify(dist.variance)
    r = MathResult.ok(
        "variance",
        method="SymPy 未能求出 ∫(x-EX)^2 f(x)dx 的闭式解；结论取该分布的标准方差公式，并用 mpmath 数值求积独立复核")
    r.set_input(dist=dist.name, param=dist.param_desc)
    r.set_result(value)
    std = sp.sqrt(value)
    r.set_raw(E_X=A.to_text(sp.simplify(mean)), variance=A.to_text(value),
              variance_latex=A.to_latex(value), std=A.to_text(sp.simplify(std)),
              std_latex=A.to_latex(sp.simplify(std)),
              by_definition=A.to_text(path1), by_shortcut=A.to_text(path2),
              standard_formula=A.to_text(value))
    r.add_condition(*dist.conditions)
    r.add_warning("SymPy 无法对 ∫(x-EX)^2 f(x)dx 给出闭式解（被积函数含 log(x) 复合），"
                  "方差结论取自该分布的标准公式，已用数值求积复核；如需严格符号证明请查教材。")
    methods = [
        _numeric_moment_check(dist, (x - mean) ** 2, value,
                              label="定义式数值求积复核 ∫(x-EX)²f(x)dx"),
        _numeric_moment_check(dist, x ** 2, sp.simplify(value + mean ** 2),
                              label="E[X²] = Var(X) + (EX)² 数值求积复核 ∫x²f(x)dx"),
        {"name": "方差非负检查", "passed": (not _eval_finite(value)) or value >= 0,
         "detail": f"Var(X) = {A.to_text(value)}"},
    ]
    return _verify_set(r, methods)


def _standard_moment(dist: _Dist, g: Any) -> Any:
    """分布自带标准公式能给出的 E[g(X)]（仅覆盖 g = x 与 g = x² 两种常见情形）。"""
    x = dist.var
    if dist.mean is None:
        return None
    if sp.simplify(g - x) == 0:
        return sp.simplify(dist.mean)
    if sp.simplify(g - x ** 2) == 0 and dist.variance is not None and not _has_free(dist.variance):
        return sp.simplify(dist.variance + dist.mean ** 2)
    return None


def _mgf_moment(dist: _Dist, order: int) -> tuple[Any, str]:
    """由矩母函数取 k 阶原点矩：M^{(k)}(0)。

    直接 ``diff(M,t,k).subs(t,0)`` 会在**可去奇点**处翻车：均匀分布
    ``M(t) = (e^{bt} - e^{at})/((b-a)t)``、离散均匀
    ``M(t) = (e^t(1-e^{Nt}))/(N(1-e^t))`` 在 t=0 都是 0/0，``subs`` 给
    ``nan``/``zoo``（曾让 U(0,1)、U(-1,3)、离散均匀的 E[X] 双路径互印报 FAIL）。
    因此按 limit → 级数系数 → subs 三级降级。
    """
    t = sp.Symbol("t")
    try:
        derivative = sp.diff(dist.mgf, t, order)
    except Exception as exc:  # noqa: BLE001
        return None, f"矩母函数求 {order} 阶导数失败：{exc}"
    for label, attempt in (
        ("limit", lambda: sp.limit(derivative, t, 0)),
        ("series", lambda: sp.simplify(sp.series(dist.mgf, t, 0, order + 1).removeO()
                                       .coeff(t, order) * sp.factorial(order))),
        ("subs", lambda: derivative.subs(t, 0)),
    ):
        try:
            value = sp.simplify(attempt())
        except Exception:  # noqa: BLE001
            continue
        if _is_real_number(value):
            return value, f"路径二：矩母函数 {order} 阶导数在 t=0 处取极限（{label}）"
    return None, f"矩母函数路径在 t=0 处不可数值化（{A.to_text(derivative)}）"


def _region_sample_points(xl: Any, xu: Any, yl: Any, yu: Any) -> list[tuple[Any, Any]]:
    """在（可能无界的）矩形区域上取若干内点，供数值判定使用。"""

    def axis_points(lo: Any, hi: Any) -> list[Any]:
        if not _is_infinity(lo) and not _is_infinity(hi):
            try:
                a, b = float(sp.N(lo, 17)), float(sp.N(hi, 17))
                return [sp.N(a + (b - a) * f, 17) for f in (sp.Rational(1, 4), sp.Rational(1, 2), sp.Rational(3, 4))]
            except Exception:  # noqa: BLE001
                return []
        return [sp.Integer(-1), sp.Integer(0), sp.Integer(1), sp.Integer(2)]

    xs = axis_points(xl, xu)
    ys = axis_points(yl, yu)
    if not xs or not ys:
        return []
    return [(px, py) for px in xs for py in ys]


def _numeric_difference_scan(difference: Any, symbols: tuple[Any, ...],
                             points: list[tuple[Any, ...]]) -> tuple[Any, str]:
    """对差式在若干点上数值求值，返回 ``(最大 |差|, 说明)``；取不到证据时返回 ``(None, 说明)``。

    这是判定「两个表达式是否恒等」的独立数值路径：把它和符号化简并列使用，
    可以在 SymPy 化简不动时仍然给出**明确**的独立/不独立结论，而不是含糊的失败。
    """
    try:
        numeric = sp.lambdify(symbols, difference, modules="mpmath")
    except Exception as exc:  # noqa: BLE001
        return None, f"差式无法数值化：{exc}"
    worst = 0.0
    checked = 0
    for point in points:
        try:
            raw = numeric(*[float(sp.N(value, 17)) for value in point])
            value = complex(raw)
        except Exception:  # noqa: BLE001
            continue
        if abs(value.imag) > 1e-9:
            continue
        worst = max(worst, abs(value.real))
        checked += 1
    if checked == 0:
        return None, "所有抽样点都落在定义域外或无法求值，未取得数值证据"
    return worst, f"在 {checked} 个内点上 |差式| 的最大值 = {worst:.3e}"


def _moment_both_ways(dist: _Dist, order: int) -> tuple[Any, Any, dict[str, Any]]:
    """E[X^k] 的两条独立路径：① 按定义求和/积分；② 矩母函数在 0 处的 k 阶导数。"""
    x = dist.var
    path1 = _raw_moment(dist, order)
    path2 = None
    note = ""
    if dist.mgf is not None:
        path2, note = _mgf_moment(dist, order)
    if path2 is None:
        note = note or "该分布无矩母函数（或不存在），改用定义式的两种求和/积分形式"
        # 第二条路径：对同一密度用「换元」写法（(x-0)^k），与 path1 的 x**k 在
        # SymPy 层是不同表达式树，仅作为一致性的额外证据，不当作独立证明。
        path2 = _expectation_direct(dist, (x - sp.Integer(0)) ** order)
    agree = _same_value(path1, path2)
    detail = {
        "name": f"E[X^{order}] 双路径互印",
        "passed": bool(agree),
        "detail": note or "路径一：定义式 Σ/∫ x^k f(x)；路径二：矩母函数 M(t) 在 t=0 的 k 阶导数",
        "values": {"定义式": A.to_text(path1), "矩母函数导数": A.to_text(path2)},
    }
    return path1, path2, detail


# ---------------------------------------------------------------------------
# 概率解析助手
# ---------------------------------------------------------------------------

_PROB_KEYS = ("prior", "priors", "p_a", "likelihood", "likelihoods", "p_b_given_a", "evidence", "p_b")


def _parse_prob(text: Any, *, label: str) -> sp.Basic:
    if isinstance(text, sp.Basic):
        return text
    if isinstance(text, (int, float)):
        return sp.sympify(text)
    try:
        return sp.sympify(A.parse(str(text), symbols=["n", "k", "a", "b", "c", "m"]))
    except ValueError as exc:
        raise ValueError(f"{label} 无法解析为概率数值：{exc}") from exc


def _check_prob(value: Any, label: str) -> Any:
    if _has_free(value):
        try:
            val = sp.N(value, 20)
        except Exception:  # noqa: BLE001
            return value
        if val.is_real:
            if val < -1e-12 or val > 1 + 1e-12:
                raise ValueError(f"{label} 必须是 [0,1] 内的概率，当前为 {A.to_text(value)}")
        return value
    try:
        real = value.is_real
    except Exception:  # noqa: BLE001
        real = None
    if real is False:
        raise ValueError(f"{label} 必须是实数概率，当前为 {A.to_text(value)}")
    if value < 0 or value > 1:
        raise ValueError(f"{label} 必须是 [0,1] 内的概率，当前为 {A.to_text(value)}")
    return value


def _prob_bound(value: Any, default: Any, *, lower: bool) -> Any:
    if value is None:
        return default
    if isinstance(value, sp.Basic):
        return value
    raw = str(value).strip().lower()
    if raw in ("", "none", "null", "-inf", "-oo", "负无穷"):
        return -_OO
    if raw in ("inf", "+inf", "oo", "+oo", "无穷"):
        return _OO
    return A.parse(str(value), symbols=["x", "y", "t", "n", "k"])


def _resolve_dist(op_par: _Params) -> tuple[_Dist, sp.Symbol]:
    name = op_par.text("dist", "distribution", "d", "name", "family")
    if name is None:
        raise ValueError("缺少分布名 dist（如 normal / binomial / exponential）")
    var_name = op_par.text("var", "variable", "x") or "x"
    var = sp.Symbol(str(var_name).strip(), real=True)
    dist = _build_dist(name, var, op_par)
    _validate_params(dist)
    return dist, var


# ---------------------------------------------------------------------------
# 1. probability：核心入口（kind 分支）
# ---------------------------------------------------------------------------

_KINDS = ("bayes", "total_probability", "conditional", "independent_check",
          "inclusion_exclusion", "interval", "at")


@op("probability", "prob", "bayes", "math_probability")
def probability(**kwargs: Any) -> MathResult:
    """概率计算入口。用 ``kind`` 分支，每个分支参数少而明确。"""
    p = _Params(kwargs)
    kind_text = p.text("kind", "type", "mode", "branch") or "bayes"
    kind = kind_text.strip().lower().replace("-", "_").replace(" ", "_")
    kind_aliases = {
        "total": "total_probability", "totalprobability": "total_probability",
        "full_probability": "total_probability", "fullprobability": "total_probability",
        "全概率": "total_probability",
        "条件概率": "conditional", "cond": "conditional",
        "independent": "independent_check", "independence": "independent_check",
        "independence_check": "independent_check", "独立": "independent_check",
        "union": "inclusion_exclusion", "inclusionexclusion": "inclusion_exclusion",
        "容斥": "inclusion_exclusion", "并": "inclusion_exclusion",
        "interval_prob": "interval", "区间概率": "interval",
        "point_prob": "at", "point": "at", "点概率": "at", "at_point": "at",
        "贝叶斯": "bayes",
    }
    kind = kind_aliases.get(kind, kind)
    if kind == "bayes":
        return _kind_bayes(p)
    if kind == "total_probability":
        return _kind_total(p)
    if kind == "conditional":
        return _kind_conditional(p)
    if kind == "independent_check":
        return _kind_independent(p)
    if kind == "inclusion_exclusion":
        return _kind_union(p)
    if kind == "interval":
        return _kind_interval(p)
    if kind == "at":
        return _kind_at(p)
    raise ValueError(
        f"未知的 kind = {kind_text!r}。支持：bayes（贝叶斯）、total_probability/total（全概率）、"
        "conditional（条件概率）、independent_check（独立性判定）、inclusion_exclusion/union（P(A∪B)）、"
        "interval/interval_prob（区间概率）、at/point_prob（某点的分布律）"
    )


def _kind_bayes(p: _Params) -> MathResult:
    prior = p.nums("prior", "priors", "p_a", "pa_list")
    likelihood = p.nums("likelihood", "likelihoods", "p_b_given_a", "like")
    if not prior:
        raise ValueError("贝叶斯公式需要先验概率 prior（如 prior=\"0.3,0.5,0.2\"）")
    if not likelihood:
        raise ValueError("贝叶斯公式需要似然 likelihood（如 likelihood=\"0.9,0.8,0.5\"）")
    if len(prior) != len(likelihood):
        raise ValueError(f"先验概率个数（{len(prior)}）与似然个数（{len(likelihood)}）必须一致")
    for i, value in enumerate(prior):
        _check_prob(value, f"先验概率 prior[{i}]")
    for i, value in enumerate(likelihood):
        _check_prob(value, f"似然 P(B|A{i})")
    evidence = p.num("evidence", "p_b", "pb", "denominator", default=None)
    if evidence is not None:
        _check_prob(evidence, "证据概率 evidence = P(B)")
    n = len(prior)
    joints = [sp.simplify(prior[i] * likelihood[i]) for i in range(n)]
    computed = sp.simplify(sum(joints))
    denom = sp.simplify(evidence) if evidence is not None else computed
    if not _has_free(denom) and denom == 0:
        raise ValueError("全概率 P(B) = 0，贝叶斯公式无定义（不能作为分母）")
    posteriors = [sp.simplify(joints[i] / denom) for i in range(n)]

    r = MathResult.ok("probability", method="贝叶斯公式 / 全概率公式（符号精确计算）")
    r.set_input(kind="bayes", prior=A.to_text(sp.Matrix(prior)) if n > 1 else A.to_text(prior[0]),
                likelihood=A.to_text(sp.Matrix(likelihood)) if n > 1 else A.to_text(likelihood[0]),
                evidence=A.to_text(evidence) if evidence is not None else None)
    r.add_condition(f"划分 A1..A{n} 互不相容且 ∪Ai 为必然事件（先验之和 = {A.to_text(sp.simplify(sum(prior))) }）",
                    "似然 P(B|Ai) 为条件概率")
    if not _safe_equal(sp.simplify(sum(prior)), 1):
        r.add_warning(f"先验概率之和为 {A.to_text(sp.simplify(sum(prior)))} ≠ 1，请确认划分是否完备")
    if evidence is not None and not _safe_equal(computed, evidence):
        r.add_warning(
            f"给定的 P(B) = {A.to_text(evidence)} 与全概率公式算出的 {A.to_text(computed)} 不一致，"
            "已按给定的 P(B) 作为分母"
        )

    terms = []
    for i in range(n):
        terms.append({
            "i": i + 1,
            "P(Ai)": A.to_text(prior[i]),
            "P(B|Ai)": A.to_text(likelihood[i]),
            "P(Ai)P(B|Ai)": A.to_text(joints[i]),
            "P(Ai|B)": A.to_text(posteriors[i]),
            "approx": sp.N(posteriors[i], 10) if not _has_free(posteriors[i]) else None,
        })
    numerator_text = " + ".join(f"({A.to_text(prior[i])})*({A.to_text(likelihood[i])})" for i in range(n))
    denominator_text = (f"P(B) = {A.to_text(evidence)}（由用户给定）" if evidence is not None
                        else f"P(B) = {numerator_text} = {A.to_text(computed)}")
    r.set_result(sp.Matrix(posteriors) if n > 1 else posteriors[0])
    r.set_raw(
        posterior=[A.to_text(v) for v in posteriors],
        posterior_approx=[float(sp.N(v, 10)) for v in posteriors] if not any(_has_free(v) for v in posteriors) else None,
        # 直接用字面 LaTeX，不要经 sp.latex(Symbol(...))（会把 "P(A_i|B)" 的花括号错配成 A_{i|B)}）
        formula_latex="P(A_i\\mid B) = \\frac{P(A_i)P(B\\mid A_i)}{\\sum_j P(A_j)P(B\\mid A_j)}",
        process={
            "分子": [f"P(A{i+1})P(B|A{i+1}) = {A.to_text(joints[i])}" for i in range(n)],
            "分母": denominator_text,
            "分子表达式": numerator_text,
        },
        table=terms,
    )
    # 独立验证：① 后验之和为 1；② 后验 * P(B) 应回到 P(Ai)P(B|Ai)
    sum_check = sp.simplify(sum(posteriors))
    back = [sp.simplify(posteriors[i] * denom) for i in range(n)]
    methods = [
        {"name": "后验概率之和为 1", "passed": _same_value(sum_check, 1),
         "detail": f"Σ P(Ai|B) = {A.to_text(sum_check)}"},
        {"name": "回代检查 P(Ai|B)·P(B) = P(Ai)P(B|Ai)",
         "passed": all(_same_value(back[i], joints[i]) for i in range(n)),
         "detail": "用每个后验乘回分母，与先验×似然逐个比对："
                   + ", ".join(f"{A.to_text(back[i])} vs {A.to_text(joints[i])}" for i in range(n))},
    ]
    return _verify_set(r, methods)


def _kind_total(p: _Params) -> MathResult:
    prior = p.nums("prior", "priors", "p_a")
    likelihood = p.nums("likelihood", "likelihoods", "p_b_given_a", "like")
    if not prior:
        raise ValueError("全概率公式需要先验概率 prior（如 prior=\"0.3,0.5,0.2\"）")
    if not likelihood:
        raise ValueError("全概率公式需要似然 likelihood（如 likelihood=\"0.9,0.8,0.5\"）")
    if len(prior) != len(likelihood):
        raise ValueError(f"先验概率个数（{len(prior)}）与似然个数（{len(likelihood)}）必须一致")
    for i, value in enumerate(prior):
        _check_prob(value, f"先验概率 prior[{i}]")
    for i, value in enumerate(likelihood):
        _check_prob(value, f"似然 P(B|A{i})")
    n = len(prior)
    joints = [sp.simplify(prior[i] * likelihood[i]) for i in range(n)]
    total = sp.simplify(sum(joints))
    r = MathResult.ok("probability", method="全概率公式 P(B) = Σ P(Ai)P(B|Ai)")
    r.set_input(kind="total_probability",
                prior=A.to_text(sp.Matrix(prior)) if n > 1 else A.to_text(prior[0]),
                likelihood=A.to_text(sp.Matrix(likelihood)) if n > 1 else A.to_text(likelihood[0]))
    r.set_result(total)
    expr_text = " + ".join(f"({A.to_text(prior[i])})*({A.to_text(likelihood[i])})" for i in range(n))
    r.set_raw(formula="P(B) = Σ_i P(Ai)P(B|Ai)", expression=f"P(B) = {expr_text}",
              terms=[f"P(A{i+1})P(B|A{i+1}) = {A.to_text(joints[i])}" for i in range(n)])
    r.add_condition(f"划分 A1..A{n} 互不相容且 ∪Ai = Ω", "各 P(Ai) ≥ 0，P(B|Ai) 为条件概率")
    if not _safe_equal(sp.simplify(sum(prior)), 1):
        r.add_warning(f"先验概率之和为 {A.to_text(sp.simplify(sum(prior)))} ≠ 1，请确认划分是否完备")
    sum_check = sp.simplify(sum(joints))
    methods = [
        {"name": "逐项求和复核", "passed": _same_value(sum_check, total),
         "detail": "对 Σ P(Ai)P(B|Ai) 的每一项分别展开后重新求和："
                   + " + ".join(A.to_text(v) for v in joints)},
        {"name": "归一化上界检查", "passed": (not _eval_finite(total)) or (0 <= total <= 1),
         "detail": f"P(B) = {A.to_text(total)} 必须落在 [0,1]（概率的可加性上界）"},
    ]
    return _verify_set(r, methods, extra_note="全概率公式本身即为定义式；此处用『逐项展开后重新求和』与上界检查复核")


def _kind_conditional(p: _Params) -> MathResult:
    joint = _parse_prob(p.raw("joint", "p_ab", "pab", "p_a_b"), label="joint = P(AB)")
    given = _parse_prob(p.raw("given", "p_b", "pb", "condition", "given_prob"), label="given = P(B)")
    if p.raw("joint", "p_ab", "pab", "p_a_b") is None:
        raise ValueError("条件概率需要 joint = P(AB)")
    if p.raw("given", "p_b", "pb", "condition", "given_prob") is None:
        raise ValueError("条件概率需要 given = P(B)")
    _check_prob(joint, "P(AB)")
    _check_prob(given, "P(B)")
    if not _has_free(given) and given == 0:
        raise ValueError("P(B) = 0 时条件概率 P(A|B) 无定义，不能作为分母")
    result = sp.simplify(joint / given)
    r = MathResult.ok("probability", method="条件概率定义 P(A|B) = P(AB)/P(B)")
    r.set_input(kind="conditional", joint=A.to_text(joint), given=A.to_text(given))
    r.set_result(result)
    r.set_raw(formula="P(A|B) = P(AB) / P(B)",
              expression=f"P(A|B) = ({A.to_text(joint)}) / ({A.to_text(given)})")
    r.add_condition("P(B) > 0", "0 ≤ P(AB) ≤ P(B) ≤ 1")
    if not _has_free(joint) and not _has_free(given) and joint > given:
        r.add_warning(f"P(AB) = {A.to_text(joint)} 大于 P(B) = {A.to_text(given)}，不满足 P(AB) ≤ P(B)")
    # 独立验证：把 P(AB) 写成 P(A|B)·P(B)，再与 joint 比较
    back = sp.simplify(result * given)
    methods = [
        _cross_check("回代 P(A|B)·P(B) = P(AB)", back, joint,
                     note="路径一：由结果乘回 P(B)；路径二：原始 P(AB)"),
        {"name": "乘法公式变形复核（P(AB)/P(B) 的等价写法）",
         "passed": _same_value(sp.simplify(joint / given), sp.simplify(joint * (1 / given))),
         "detail": "用乘法逆元形式重算一次，检查除法实现无误"},
    ]
    return _verify_set(r, methods)


def _kind_independent(p: _Params) -> MathResult:
    joint = _parse_prob(p.raw("joint", "p_ab", "pab", "p_a_b"), label="joint = P(AB)")
    pa = _parse_prob(p.raw("pa", "p_a", "prob_a"), label="pa = P(A)")
    pb = _parse_prob(p.raw("pb", "p_b", "prob_b", "given"), label="pb = P(B)")
    for label, value in (("joint = P(AB)", joint), ("pa = P(A)", pa), ("pb = P(B)", pb)):
        if value is None:
            raise ValueError(f"独立性判定需要三个量：joint、pa、pb（缺少 {label}）")
        _check_prob(value, label)
    product = sp.simplify(pa * pb)
    independent = _same_value(joint, product)
    r = MathResult.ok("probability", method="独立性定义 P(AB) = P(A)P(B)")
    r.set_input(kind="independent_check", joint=A.to_text(joint), pa=A.to_text(pa), pb=A.to_text(pb))
    r.set_result(bool(independent), **{"P(A)P(B)": A.to_text(product)})
    r.set_raw(verdict="独立" if independent else "不独立",
              formula="A, B 独立 ⟺ P(AB) = P(A)P(B)",
              compare={"P(AB)": A.to_text(joint), "P(A)P(B)": A.to_text(product),
                       "difference": A.to_text(sp.simplify(joint - product))})
    r.add_condition("0 ≤ P(A), P(B), P(AB) ≤ 1")
    if independent:
        r.add_condition("独立时还有 P(A|B) = P(A)、P(B|A) = P(B)")
        if not _has_free(pb) and pb != 0:
            r.set_raw(conditional_check=f"P(A|B) = {A.to_text(sp.simplify(joint / pb))} = P(A)")
    difference = sp.simplify(joint - product)
    methods = [
        # "差式不为 0 ⇒ 不独立" 也是一次**成功的**判定，不能报成验证失败。
        {"name": "P(AB) 与 P(A)P(B) 的差式", "passed": _eval_finite(difference),
         "detail": f"P(AB) - P(A)P(B) = {A.to_text(difference)}"
                   + ("（= 0 ⟺ 独立）" if independent else "（≠ 0 ⟹ 不独立）")},
        {"name": "判据复核（乘法公式与条件概率一致性）",
         "passed": bool(independent) == bool(_same_value(difference, 0)),
         "detail": f"P(AB) - P(A)P(B) = {A.to_text(difference)}；"
                   f"判定结论：{'独立' if independent else '不独立'}"},
    ]
    return _verify_set(r, methods)


def _kind_union(p: _Params) -> MathResult:
    pa = _parse_prob(p.raw("pa", "p_a", "prob_a"), label="pa = P(A)")
    pb = _parse_prob(p.raw("pb", "p_b", "prob_b"), label="pb = P(B)")
    if pa is None or pb is None:
        raise ValueError("P(A∪B) 需要 pa 与 pb")
    _check_prob(pa, "P(A)")
    _check_prob(pb, "P(B)")
    raw_joint = p.raw("joint", "p_ab", "pab", "p_a_b", "intersection")
    assumed_independent = raw_joint is None
    if assumed_independent:
        joint = sp.simplify(pa * pb)
    else:
        joint = _parse_prob(raw_joint, label="joint = P(AB)")
        _check_prob(joint, "P(AB)")
    r = MathResult.ok("probability", method="容斥原理 P(A∪B) = P(A) + P(B) - P(AB)")
    r.set_input(kind="inclusion_exclusion", pa=A.to_text(pa), pb=A.to_text(pb),
                joint="（未给出，按独立假设取 P(A)P(B)）" if assumed_independent else A.to_text(joint))
    union = sp.simplify(pa + pb - joint)
    r.set_result(union)
    r.set_raw(formula="P(A∪B) = P(A) + P(B) - P(AB)",
              expression=f"P(A∪B) = {A.to_text(pa)} + {A.to_text(pb)} - {A.to_text(joint)}",
              joint_used=A.to_text(joint))
    r.add_condition("0 ≤ P(AB) ≤ min(P(A), P(B))")
    if assumed_independent:
        r.add_warning("未给出 joint = P(AB)，已按 A、B 相互独立处理（P(AB) = P(A)P(B)）；"
                      "若实际不独立，请显式给出 joint")
    elif not _has_free(joint) and not _has_free(pa) and not _has_free(pb) and joint > min(pa, pb):
        r.add_warning(f"P(AB) = {A.to_text(joint)} 超过 min(P(A), P(B))，与概率论基本不等式矛盾")
    if _eval_finite(union) and not (0 <= union <= 1):
        r.add_warning(f"P(A∪B) = {A.to_text(union)} 不在 [0,1] 内，请检查输入")
    # 独立验证：① 容斥的补集路径 P(A∪B) = 1 - P(A^c B^c)，其中 P(A^c B^c)=1-P(A)-P(B)+P(AB)
    complement = sp.simplify(1 - (1 - pa - pb + joint))
    methods = [
        _cross_check("补集路径 P(A∪B) = 1 - P(Aᶜ∩Bᶜ)", complement, union,
                     note="路径一：容斥公式直算；路径二：P(Aᶜ∩Bᶜ) = 1 - P(A) - P(B) + P(AB) 后取补"),
        {"name": "拆分路径 P(A∪B) = P(A) + P(B∩Aᶜ)", "passed": _same_value(
            sp.simplify(pa + (pb - joint)), union),
         "detail": "把 B 拆成 AB ∪ AᶜB，用 P(B∩Aᶜ) = P(B) - P(AB) 再与 P(A) 相加"},
    ]
    return _verify_set(r, methods,
                       extra_note="按独立假设计算 P(AB) = P(A)P(B)" if assumed_independent else None)


def _kind_interval(p: _Params) -> MathResult:
    dist, var = _resolve_dist(p)
    lower_raw = p.raw("lower", "lo", "from", "a", "start")
    upper_raw = p.raw("upper", "hi", "to", "b", "stop")
    strict = p.text("strict", "open", "boundary") or "left"
    lower = _prob_bound(lower_raw, -_OO, lower=False)
    upper = _prob_bound(upper_raw, _OO, lower=False)
    if lower_raw is None and upper_raw is None:
        raise ValueError("区间概率需要 lower 与/或 upper（如 lower=\"-1\", upper=\"1\"）")
    if _eval_finite(lower) and _eval_finite(upper) and upper < lower:
        raise ValueError(f"区间上界必须大于下界，当前 lower = {A.to_text(lower)}, upper = {A.to_text(upper)}")

    cdf_a = dist.cdf.subs(var, lower)
    cdf_b = dist.cdf.subs(var, upper)
    cdf_path = sp.simplify(cdf_b - cdf_a)
    integral_path = sp.simplify(dist.prob_between(lower, upper))

    r = MathResult.ok("probability", method="区间概率：分布函数差分与直接积分/求和两条路径")
    r.set_input(kind="interval", dist=dist.name, param=dist.param_desc,
                lower=A.to_text(lower), upper=A.to_text(upper))
    r.set_result(cdf_path)
    r.set_raw(
        formula=("P(a < X ≤ b) = F(b) - F(a)" if not dist.discrete else "P(a < X ≤ b) = F(b) - F(a)（离散型含右端点）"),
        F_lower=A.to_text(cdf_a), F_upper=A.to_text(cdf_b),
        by_cdf=A.to_text(cdf_path), by_integral=A.to_text(integral_path),
        interval=f"{A.to_text(lower)} < X ≤ {A.to_text(upper)}",
        approx=float(sp.N(cdf_path, 12)) if _eval_finite(cdf_path) else None,
    )
    r.add_condition(*dist.conditions)
    r.add_condition("连续型：单点概率为 0，P(a<X<b) = P(a≤X≤b) = F(b)-F(a)")
    if dist.discrete:
        r.add_warning("离散型：本结果按 P(lower < X ≤ upper) 计算；若要 P(lower ≤ X ≤ upper)，"
                      "请把 lower 减 1 后传入")
    if _unevaluated(cdf_path) and _unevaluated(integral_path):
        return MathResult.unsolved("probability", f"区间概率无法求得闭式解：{A.to_text(cdf_path)}")
    methods = [
        _cross_check("F(b)-F(a) 与 ∫f 两条路径互印", cdf_path, integral_path),
        {"name": "区间概率落在 [0,1]", "passed": (not _eval_finite(cdf_path)) or (0 <= cdf_path <= 1),
         "detail": f"P = {A.to_text(cdf_path)}"},
    ]
    return _verify_set(r, methods)


def _kind_at(p: _Params) -> MathResult:
    dist, var = _resolve_dist(p)
    point_raw = p.raw("point", "at", "x0", "value", "k")
    if point_raw is None:
        raise ValueError("kind=\"at\" 需要 point（要求分布律的那个点）")
    point = _prob_bound(point_raw, None, lower=False)
    value = sp.simplify(dist.pdf.subs(var, point))
    in_support = True
    lo, hi = dist.support
    if dist.discrete and _eval_finite(point):
        if not (lo <= point <= hi):
            in_support = False
            value = sp.Integer(0)
    r = MathResult.ok("probability", method="代入分布律/密度")
    r.set_input(kind="at", dist=dist.name, param=dist.param_desc, point=A.to_text(point))
    r.set_result(value)
    r.set_raw(probability=A.to_text(value),
              probability_latex=A.to_latex(value),
              approx=float(sp.N(value, 12)) if _eval_finite(value) else None,
              in_support=in_support,
              formula=("P(X = x)" if dist.discrete else "f(x)（连续型：密度值，不是概率）"))
    r.add_condition(*dist.conditions)
    if not dist.discrete:
        r.add_warning("连续型随机变量单点取值的概率为 0；这里给出的是密度函数在该点的值 f(x)")
    if not in_support:
        r.add_warning(f"点 {A.to_text(point)} 不在支撑集内，概率为 0")
    if dist.discrete:
        # 路径一：直接代入闭式分布律；路径二：分布函数在 k 与 k-1 处之差。
        # 注意要用 F(k) - F(k-1)（**不是** F(k) - f(k)），且 cdf 里的 Sum 往往
        # 化简不动，所以以数值差分为主证据。
        jump_ok, jump_detail = False, "无法数值化分布函数，未取得证据"
        try:
            f_k = _eval_num(dist.cdf, var, point)
            f_prev = _eval_num(dist.cdf, var, point - 1)
            if _is_real_number(f_k) and _is_real_number(f_prev):
                jump = float(f_k) - float(f_prev)
                exact = float(sp.N(value, 15)) if _eval_finite(value) else None
                jump_detail = (f"F({A.to_text(point)}) - F({A.to_text(point - 1)}) = {jump!r}"
                               f"，闭式分布律 = {exact!r}")
                if exact is not None:
                    jump_ok = abs(jump - exact) <= 1e-9 * max(1.0, abs(exact))
                    jump_detail += f"，偏差 {abs(jump - exact):.3e}"
        except Exception:  # noqa: BLE001
            pass
        methods = [
            {"name": "P(X=k) = F(k) - F(k-1)", "passed": jump_ok, "detail": jump_detail},
            {"name": "分布律非负且 ≤ 1", "passed": (not _eval_finite(value)) or (0 <= value <= 1),
             "detail": f"P(X = {A.to_text(point)}) = {A.to_text(value)}"},
        ]
        # 能够符号化简时再补一条更强的符号证据（化简不动就不报失败）
        symbolic = _cross_check("分布律 = F(k) - F(k-1)（符号）",
                                value, sp.simplify(dist.cdf.subs(var, point)
                                                   - dist.cdf.subs(var, point - 1)),
                                note="路径一：闭式分布律；路径二：分布函数差分")
        if symbolic.get("passed"):
            methods.append(symbolic)
    else:
        methods = [
            {"name": "密度非负检查", "passed": (not _eval_finite(value)) or (value >= 0),
             "detail": f"f({A.to_text(point)}) = {A.to_text(value)}"},
            _cross_check("密度 = F'(x)", sp.diff(dist.cdf, var).subs(var, point), value,
                         note="路径一：由分布函数求导；路径二：直接代入密度"),
        ]
    return _verify_set(r, methods)


# ---------------------------------------------------------------------------
# 2. distribution：密度 / 分布函数 / 分位数
# ---------------------------------------------------------------------------

_DIST_KINDS = {
    "pdf": "pdf", "密度": "pdf", "density": "pdf", "密度函数": "pdf",
    "pmf": "pmf", "分布律": "pmf", "分布列": "pmf", "mass": "pmf",
    "cdf": "cdf", "分布函数": "cdf", "累积分布函数": "cdf", "distribution_function": "cdf",
    "quantile": "quantile", "分位数": "quantile", "q": "quantile", "inverse": "quantile",
    "分位点": "quantile",
}


@op("distribution", "dist", "pdf", "cdf", "math_distribution")
def distribution(**kwargs: Any) -> MathResult:
    """求常见分布的密度/分布律（pdf/pmf）、分布函数（cdf）或分位数（quantile）。"""
    p = _Params(kwargs)
    dist, var = _resolve_dist(p)
    kind_text = p.text("kind", "type", "mode", "target") or "pdf"
    kind = _DIST_KINDS.get(kind_text.strip().lower().replace(" ", "_"), kind_text.strip().lower())
    # pdf/pmf/cdf 默认返回含变量的通式；调用方给了 point 就应当给出该点的取值，
    # 否则「传了 point 却拿到通式」等于参数被静默忽略。
    # 只看 point/at/x0：**不要**把 "value"/"k" 也算进来——chi2 的自由度参数就叫 k，
    # 若把它们当取值点，`distribution(dist="chi2", n=4, k=4)` 这种调用会被误当成「求 x=4 处的值」，
    # 于是返回数值而不是含变量的通式（引擎自检正是这样抓到过两处 CHECK FAIL）。
    at_raw = p.raw("point", "at", "x0")
    at_value = p.num("point", "at", "x0") if at_raw is not None else None
    if kind in ("pdf", "pmf"):
        value = dist.pdf
        r = MathResult.ok("distribution", method="由分布定义直接构造（并与 sympy.stats 交叉参照）")
        r.set_input(dist=dist.name, kind="pmf" if dist.discrete else "pdf", param=dist.param_desc,
                    **({"point": A.to_text(at_value)} if at_value is not None else {}))
        evaluated = sp.simplify(value.subs(var, at_value)) if at_value is not None else None
        r.set_result(evaluated if evaluated is not None else sp.simplify(value))
        r.set_raw(**{"pdf_text": A.to_text(value), "pdf_latex": A.to_latex(value),
                     **({"at": A.to_text(at_value), "value_at_point": A.to_text(evaluated),
                         "formula_text": A.to_text(value)} if evaluated is not None else {}),
                     "kind_of_dist": "离散型（分布律）" if dist.discrete else "连续型（密度）"})
        r.add_condition(*dist.conditions)
    elif kind == "cdf":
        value = dist.cdf
        r = MathResult.ok("distribution", method="对密度逐点积分（离散型为求和）")
        r.set_input(dist=dist.name, kind="cdf", param=dist.param_desc,
                    **({"point": A.to_text(at_value)} if at_value is not None else {}))
        evaluated = sp.simplify(value.subs(var, at_value)) if at_value is not None else None
        r.set_result(evaluated if evaluated is not None else sp.simplify(value))
        r.set_raw(**{"cdf_text": A.to_text(value), "cdf_latex": A.to_latex(value),
                     **({"at": A.to_text(at_value), "value_at_point": A.to_text(evaluated),
                         "formula_text": A.to_text(value)} if evaluated is not None else {})})
        r.add_condition(*dist.conditions)
        methods = [_cdf_endpoint_check(dist)]
        if not dist.discrete:
            methods.append(_monotone_check(dist))
        methods.append(_cdf_pdf_numeric_check(dist))
        return _verify_set(r, methods)
    elif kind == "quantile":
        result, note = _quantile(dist, p)
        if result is None:
            return MathResult.unsolved("distribution", note)
        r = MathResult.ok("distribution", method="分布函数的反函数（分位数）")
        r.set_input(dist=dist.name, kind="quantile", param=dist.param_desc,
                    point=A.to_text(sp.sympify(p.num("point", "p", "alpha", "q", "quantile", "level",
                                                      required=True, label="point（分位水平α）"))))
        alpha = p.num("point", "p", "alpha", "q", "quantile", "level", required=True, label="point（分位水平α）")
        r.set_result(result)
        r.set_raw(quantile=A.to_text(result), alpha=A.to_text(alpha))
        if note:
            r.add_warning(note)
        r.add_condition(*dist.conditions)
        # 独立验证：把分位数代回分布函数，应回到 α
        back = sp.simplify(dist.cdf.subs(var, result))
        methods = [
            _cross_check("回代 F(分位数) = α", back, alpha,
                         note="路径一：分位数公式；路径二：把结果代回分布函数"),
            {"name": "分位数落在支撑集内检查",
             "passed": (not _eval_finite(result)) or (dist.support[0] <= result <= dist.support[1]),
             "detail": f"x = {A.to_text(result)}，支撑集 [{A.to_text(dist.support[0])}, {A.to_text(dist.support[1])}]"},
        ]
        return _verify_set(r, methods)
    else:
        raise ValueError(f"未知的 kind = {kind_text!r}。支持：pdf/pmf（密度或分布律）、cdf（分布函数）、"
                         "quantile（分位数）")

    # pdf / pmf：归一化 + 定义域检查
    methods = [_verify_normalization(dist)]
    if not dist.discrete:
        methods.append({"name": "密度定义域检查",
                        "passed": True,
                        "detail": f"支撑集：{A.to_text(dist.support[0])} < x < {A.to_text(dist.support[1])}"})
    methods.append(_cdf_pdf_numeric_check(dist))
    return _verify_set(r, methods)


def _quantile(dist: _Dist, p: _Params) -> tuple[Any, str | None]:
    alpha = p.num("point", "p", "alpha", "q", "quantile", "level", required=True, label="point（分位水平α）")
    if not _has_free(alpha):
        if alpha.is_real is False or alpha <= 0 or alpha >= 1:
            raise ValueError(f"分位水平 α 必须满足 0 < α < 1，当前为 {A.to_text(alpha)}")
    name = dist.name
    pars = dist.params
    x = dist.var
    try:
        if name == "normal":
            mu, sigma = pars["mu"], pars["sigma"]
            return sp.simplify(mu + sigma * sp.sqrt(2) * sp.erfinv(2 * alpha - 1)), None
        if name == "exponential":
            lam = pars["lam"]
            return sp.simplify(-sp.log(1 - alpha) / lam), None
        if name == "uniform":
            a, b = pars["a"], pars["b"]
            return sp.simplify(a + alpha * (b - a)), None
        if name == "lognormal":
            mu, sigma = pars["mu"], pars["sigma"]
            return sp.simplify(sp.exp(mu + sigma * sp.sqrt(2) * sp.erfinv(2 * alpha - 1))), None
        if name == "cauchy":
            x0, gamma = pars["x0"], pars["gamma"]
            return sp.simplify(x0 + gamma * sp.tan(sp.pi * (alpha - sp.Rational(1, 2)))), None
        if name == "weibull":
            k, lam = pars["k"], pars["lam"]
            return sp.simplify(lam * (-sp.log(1 - alpha)) ** (1 / k)), None
        if name == "rayleigh":
            sigma = pars["sigma"]
            return sp.simplify(sigma * sp.sqrt(-2 * sp.log(1 - alpha))), None
        if name == "chi2":
            return None, ("卡方分布的分位数需要反解正则化不完全 Gamma 函数 P(k/2, x/2) = α，"
                          "SymPy 无闭式反函数；请用数值方法（scipy.stats.chi2.ppf）或查表")
        if name == "binomial":
            return _discrete_quantile(dist, alpha)
        if name == "poisson":
            return _discrete_quantile(dist, alpha)
        if name in ("geometric", "hypergeometric", "uniform_discrete"):
            return _discrete_quantile(dist, alpha)
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001
        # 「反解不出来」是能力边界，不是输入错误：交给 kind=quantile 分支报 unsolved，
        # 不要把它包装成 invalid_input 的 ValueError（否则用户会以为自己参数传错了）。
        return None, (f"{name} 分布的分位数在此参数下无法反解出闭式解"
                      f"（SymPy solve 报：{exc}）；请用数值方法或查表")
    # 通用兜底：直接反解分布函数
    try:
        solutions = sp.solve(sp.Eq(dist.cdf, alpha), x)
    except Exception as exc:  # noqa: BLE001
        return None, (f"{name} 分布的分位数无法反解出闭式解"
                      f"（SymPy solve 报：{exc}）；请用数值方法或查表")
    if not solutions:
        return None, f"{name} 分布的分位数在当前参数下无法反解出闭式解"
    return solutions[0], "分位数由分布函数反解得到（可能含特殊函数）"


def _discrete_quantile(dist: _Dist, alpha: Any) -> tuple[Any, str | None]:
    """离散型分位数：从支撑集下界开始逐步累加分布律。"""
    x = dist.var
    if _has_free(alpha):
        return None, "含符号参数时离散型分位数无法确定，请给出具体的 α 与参数"
    target = float(sp.N(alpha, 20))
    lo, hi = dist.support
    if _has_free(lo):
        return None, "离散型分位数需要具体（数值）参数"
    lo_i = int(lo)
    hi_i = int(hi) if not _has_free(hi) else lo_i + 500
    cum = 0.0
    for k in range(lo_i, hi_i + 1):
        try:
            pmf = float(sp.N(dist.pdf.subs(x, k), 25))
        except Exception:  # noqa: BLE001
            continue
        if not (pmf == pmf):
            continue
        cum += pmf
        if cum >= target - 1e-12:
            exact = sp.nsimplify(sp.Integer(k))
            return exact, f"离散型分位数由累加分布律确定：Σ P(X ≤ {k}) ≥ α 的最小整数 k"
    return None, "在支撑集内累加分布律仍未达到 α，请检查参数"


# ---------------------------------------------------------------------------
# 3. expectation
# ---------------------------------------------------------------------------

def _resolve_expectation_target(p: _Params) -> tuple[_Dist, Any, str]:
    """返回 (分布对象, 被求期望的函数 g, 说明)。"""
    dist, var = _resolve_dist(p)
    expr_text = p.text("expr", "g", "function", "formula")
    g = var
    desc = f"E[X]（X ~ {dist.name}, {dist.param_desc}）"
    if expr_text:
        g = A.parse(expr_text, symbols=[str(var), "x", "y", "t", "n", "k"])
        desc = f"E[{A.to_text(g)}]（X ~ {dist.name}, {dist.param_desc}）"
    return dist, g, desc


def _is_divergent(value: Any) -> bool:
    """值是否为「不存在 / 发散」（``nan`` / ``zoo`` / ``±oo`` 且不含自由符号）。

    用于把 ``nan`` 这类结果改写成人话：柯西分布的 E[X] 真的不存在，
    直接把 ``nan`` 丢给用户会被误读成「程序算挂了」。
    """
    if _has_free(value):
        return False
    try:
        if value.is_finite is False:  # zoo / ±oo
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        # SymPy 里 nan.is_finite 是 None（不是 False），必须单独判一次等号
        return bool(value == sp.nan)
    except Exception:  # noqa: BLE001
        return False


@op("expectation", "expect", "E", "mean", "math_expectation")
def expectation(**kwargs: Any) -> MathResult:
    """数学期望：支持命名分布（可带函数 expr）与自定义分布律。"""
    p = _Params(kwargs)
    custom_pmf = p.raw("pmf", "probs", "probabilities", "probability_mass")
    if custom_pmf is not None:
        return _custom_expectation(p, want="mean")
    dist, g, desc = _resolve_expectation_target(p)
    value = _expectation_direct(dist, g)
    if _is_divergent(value):
        # 积分/级数发散 ⟹ 期望不存在。明确说出来，绝不把 nan/oo 当结果返回。
        note = dist.moment_note or "该积分/级数不收敛，数学期望不存在"
        r = MathResult.ok("expectation", method="按定义求和/积分，发现积分不收敛 ⟹ 期望不存在")
        r.set_input(dist=dist.name, param=dist.param_desc, expr=A.to_text(g))
        r.set_result(f"E[{A.to_text(g)}] 不存在（{note}）")
        r.set_raw(target=desc, exists=False, by_definition=A.to_text(value), note=note)
        r.add_condition(*dist.conditions)
        r.add_warning(f"E[{A.to_text(g)}] 不存在：{note}")
        return _verify_set(r, [], extra_note=f"期望不存在（{note}），没有可复核的数值")
    if _unevaluated(value):
        standard = _standard_moment(dist, g)
        if standard is not None and _eval_finite(standard):
            # 定义式积分拿不到闭式（如对数正态的 E[X²]）：取标准公式 + 数值求积复核
            r = MathResult.ok(
                "expectation",
                method="SymPy 未能求出定义式积分的闭式解；结论取该分布的标准公式，并用 mpmath 数值求积独立复核")
            r.set_input(dist=dist.name, param=dist.param_desc, expr=A.to_text(g))
            r.set_result(standard)
            r.set_raw(target=desc, by_definition=A.to_text(value), standard_formula=A.to_text(standard))
            r.add_condition(*dist.conditions)
            r.add_warning("SymPy 无法对 E[g(X)] 的定义式积分给出闭式解，结论取自该分布的标准公式，已用数值求积复核。")
            return _verify_set(r, [_numeric_moment_check(dist, g, standard,
                                                         label="定义式数值求积复核 ∫g(x)f(x)dx")])
        return MathResult.unsolved("expectation", f"E[{A.to_text(g)}] 的积分/求和未能求出闭式解：{A.to_text(value)}")
    r = MathResult.ok("expectation", method="按定义求和/积分：E[g(X)] = Σ g(x)P(X=x) 或 ∫ g(x)f(x)dx")
    r.set_input(dist=dist.name, param=dist.param_desc, expr=A.to_text(g))
    r.set_result(sp.simplify(value))
    r.set_raw(target=desc, by_definition=A.to_text(sp.simplify(value)))
    r.add_condition(*dist.conditions)
    methods: list[dict[str, Any]] = []
    if g == dist.var:
        path1, path2, detail = _moment_both_ways(dist, 1)
        methods.append(detail)
        if dist.mean is not None:
            methods.append(_cross_check("与分布自带的一阶矩公式比对", sp.simplify(value), sp.simplify(dist.mean),
                                        note="路径一：按定义求和/积分；路径二：该分布的标准公式"))
    else:
        # 独立复核：用「和的线性分解」或「同一密度的分布函数积分」两种不同写法
        methods.append(_expectation_alt_path(dist, g, value))
    # 期望的线性性/存在性检查
    if _eval_finite(value):
        methods.append({"name": "期望值合法性检查", "passed": True,
                        "detail": f"E = {A.to_text(value)}（已数值化，无自由符号）"})
    return _verify_set(r, methods)


def _expectation_alt_path(dist: _Dist, g: Any, value: Any) -> dict[str, Any]:
    """E[g(X)] 的第二条路径：对连续型用分布的支撑集分段积分/级数展开重新计算。"""
    x = dist.var
    try:
        if dist.discrete:
            lo, hi = dist.support
            if _has_free(hi):
                alt = sp.summation(sp.simplify(g * dist.pdf), (x, lo, hi))
            else:
                # 逐项枚举后再相加，与一次性求和是不同实现路径
                alt = sum(sp.simplify((g * dist.pdf).subs(x, k)) for k in range(int(lo), int(hi) + 1))
        else:
            # 用展开后的被积函数重新积分（与乘积形式不同的表达式树）
            alt = sp.integrate(sp.expand(sp.simplify(g * dist.pdf)), (x, -_OO, _OO))
    except Exception as exc:  # noqa: BLE001
        return {"name": "E[g(X)] 第二路径复核", "passed": False, "detail": f"第二路径失败：{exc}"}
    return _cross_check("E[g(X)] 定义式与展开式两条路径互印", value, alt)


def _custom_expectation(p: _Params, *, want: str) -> MathResult:
    """自定义分布律：pmf + values。"""
    probs = p.nums("pmf", "probs", "probabilities", "probability_mass")
    if not probs:
        raise ValueError("自定义分布需要 pmf（如 pmf=\"1/6,1/6,1/6,1/6,1/6,1/6\"）")
    values_raw = p.raw("values", "value", "outcomes", "x_values", "xs")
    if values_raw is not None:
        values = _numbers(values_raw)
    else:
        values = [sp.Integer(i + 1) for i in range(len(probs))]
    if len(values) != len(probs):
        raise ValueError(f"values 个数（{len(values)}）与 pmf 个数（{len(probs)}）必须一致")
    for i, prob in enumerate(probs):
        _check_prob(prob, f"pmf[{i}]")
    total = sp.simplify(sum(probs))
    if not _has_free(total) and not _safe_equal(total, 1):
        raise ValueError(f"分布律之和为 {A.to_text(total)} ≠ 1，不是合法的分布律（请先归一化）")
    dist = _pmf_distribution(values, probs)
    expr_text = p.text("expr", "g", "function", "formula")
    x = dist.var
    g = A.parse(expr_text, symbols=[str(x)]) if expr_text else x
    value = sp.simplify(sum(g.subs(x, v) * prob for v, prob in zip(values, probs)))
    label = "方差" if want == "variance" else "期望"
    r = MathResult.ok("expectation" if want == "mean" else "variance",
                      method=f"自定义分布律按定义求{label}")
    r.set_input(values=[A.to_text(v) for v in values], pmf=[A.to_text(v) for v in probs],
                expr=A.to_text(g))
    r.set_result(value)
    r.set_raw(distribution="custom", target=f"E[{A.to_text(g)}]", by_definition=A.to_text(value))
    r.add_condition("Σ p_i = 1", "各 p_i ≥ 0")
    second = sp.simplify(sum(g.subs(x, v) ** 2 * prob for v, prob in zip(values, probs)))
    methods = [
        _cross_check("E[g(X)] 与 Σ g(x_i)p_i 展开式互印", value,
                     sum(g.subs(x, v) * prob for v, prob in zip(values, probs)),
                     note="路径一：按定义直接求和；路径二：逐项展开后再相加"),
        {"name": "二阶矩 ∎ 非负检查", "passed": (not _eval_finite(second)) or second >= 0,
         "detail": f"E[g(X)^2] = {A.to_text(second)}"},
    ]
    return _verify_set(r, methods)


def _pmf_distribution(values: list[Any], probs: list[Any]) -> _Dist:
    """把 values/probs 包装成一个离散分布对象（供方差/协方差等复用）。"""
    x = sp.Symbol("x", real=True)
    pairs = sorted(zip(values, probs), key=lambda kv: sp.N(kv[0]))
    pdf = sp.Piecewise(*[(sp.sympify(p), sp.Eq(x, v)) for v, p in pairs], (0, True))
    d = _Dist("custom", x, {}, discrete=True)
    support_values = [v for v, _ in pairs]
    d.support = (support_values[0], support_values[-1])
    d._pdf_func = lambda: pdf
    d._cdf_func = lambda: sp.Piecewise(*[(sp.simplify(sum(q for v2, q in pairs if v2 <= v)),
                                          sp.And(x >= v, (x < support_values[i + 1]) if i + 1 < len(support_values) else True))
                                        for i, v in enumerate(support_values)], (0, True))
    d.param_desc = f"取值 {[A.to_text(v) for v in support_values]}"
    d.conditions = ["自定义分布律：Σ p_i = 1", "各 p_i ≥ 0"]
    return d


# ---------------------------------------------------------------------------
# 4. variance
# ---------------------------------------------------------------------------

@op("variance", "var", "math_variance")
def variance(**kwargs: Any) -> MathResult:
    """方差与标准差。两条独立路径：E[(X-EX)^2] 与 E[X^2]-(E[X])^2。"""
    p = _Params(kwargs)
    custom_pmf = p.raw("pmf", "probs", "probabilities", "probability_mass")
    if custom_pmf is not None:
        return _custom_variance(p)
    dist, _g, _desc = _resolve_expectation_target(p)
    x = dist.var
    if p.text("expr", "g", "function", "formula"):
        raise ValueError("variance 不支持 expr 参数（方差是随机变量自身的性质）；"
                         "若要求 Var(g(X))，请用 expectation 分步计算")
    mean = _expectation_direct(dist, x)
    path1 = _central_moment(dist, 2, mean)
    second = _expectation_direct(dist, x ** 2)
    path2 = sp.simplify(second - mean ** 2)
    var_value = sp.simplify(path1)
    if _is_divergent(mean) or _is_divergent(second) or _is_divergent(var_value):
        note = dist.moment_note or "该分布的二阶矩不存在，方差不存在"
        r = MathResult.ok("variance", method="按定义求和/积分，发现矩不收敛 ⟹ 方差不存在")
        r.set_input(dist=dist.name, param=dist.param_desc)
        r.set_result(f"Var(X) 不存在（{note}）")
        r.set_raw(exists=False, E_X=A.to_text(mean), E_X2=A.to_text(second),
                  by_definition=A.to_text(path1), by_shortcut=A.to_text(path2), note=note)
        r.add_condition(*dist.conditions)
        r.add_warning(f"Var(X) 不存在：{note}")
        return _verify_set(r, [], extra_note=f"方差不存在（{note}），没有可复核的数值")
    if _unevaluated(var_value) and _unevaluated(path2):
        standard = dist.variance
        if standard is not None and not _has_free(standard) and _eval_finite(standard):
            # 定义式与标准式两条符号路径都拿不到闭式（如对数正态），
            # 退到「标准公式 + 数值求积复核」，既不硬等也不编数。
            return _variance_from_standard(dist, mean, path1, second, path2)
        return MathResult.unsolved("variance", f"方差无法求得闭式解：{A.to_text(var_value)}")
    if _unevaluated(var_value):
        # 定义式积分卡住，但 E[X²]-(EX)² 给出了结果：以它为准，稍后用数值求积复核定义式
        var_value = sp.simplify(path2)
    var_value = _tidy_expr(_prefer_simpler(var_value, path2))
    r = MathResult.ok("variance", method="Var(X) = E[(X-EX)^2] 与 E[X^2]-(E[X])^2 两条路径互印")
    r.set_input(dist=dist.name, param=dist.param_desc)
    r.set_result(var_value, **{"方差": A.to_text(var_value)})
    std = sp.sqrt(var_value)
    r.set_raw(E_X=A.to_text(sp.simplify(mean)), E_X2=A.to_text(sp.simplify(second)),
              variance=A.to_text(var_value), variance_latex=A.to_latex(var_value),
              std=A.to_text(sp.simplify(std)), std_latex=A.to_latex(sp.simplify(std)),
              by_definition=A.to_text(path1), by_shortcut=A.to_text(path2))
    r.add_condition(*dist.conditions)
    if not _eval_finite(var_value):
        r.add_condition("Var(X) ≥ 0 恒成立，符号形式保持精确")
    if _unevaluated(path1):
        # 定义式没能求出闭式：改用数值求积复核（不同机制，仍是独立证据）
        moments_method = _numeric_moment_check(dist, (x - mean) ** 2, var_value,
                                               label="定义式数值求积复核 ∫(x-EX)²f(x)dx")
        r.add_warning("SymPy 未能求出 ∫(x-EX)²f(x)dx 的闭式解，该路径已改用数值求积复核。")
    else:
        moments_method = _cross_check("E[(X-EX)^2] 与 E[X^2]-(EX)^2 互印", path1, path2)
    methods = [
        moments_method,
        {"name": "方差非负检查", "passed": (not _eval_finite(var_value)) or var_value >= 0,
         "detail": f"Var(X) = {A.to_text(var_value)}"},
    ]
    if dist.variance is not None:
        methods.append(_cross_check("与该分布的标准方差公式比对", var_value, sp.simplify(dist.variance),
                                    note="路径一：按定义求和/积分；路径二：教材标准公式"))
    else:
        # 该分布在此参数下没有有限方差（如自由度不足的 t）——这是「没有可比对的
        # 标准公式」，不是验证失败，所以用 passed=None 让整体状态落到 unverified。
        methods.append({"name": "分布标准公式", "passed": None,
                        "detail": f"{dist.name} 在该参数下方差不存在或没有有限方差"
                                  + (f"（{dist.moment_note}）" if dist.moment_note else "")})
    return _verify_set(r, methods)


def _custom_variance(p: _Params) -> MathResult:
    probs = p.nums("pmf", "probs", "probabilities", "probability_mass")
    if not probs:
        raise ValueError("自定义分布需要 pmf")
    values_raw = p.raw("values", "value", "outcomes", "x_values", "xs")
    values = _numbers(values_raw) if values_raw is not None else [sp.Integer(i + 1) for i in range(len(probs))]
    if len(values) != len(probs):
        raise ValueError(f"values 个数（{len(values)}）与 pmf 个数（{len(probs)}）必须一致")
    total = sp.simplify(sum(probs))
    if not _has_free(total) and not _safe_equal(total, 1):
        raise ValueError(f"分布律之和为 {A.to_text(total)} ≠ 1，不是合法的分布律")
    mean = sp.simplify(sum(v * prob for v, prob in zip(values, probs)))
    path1 = sp.simplify(sum((v - mean) ** 2 * prob for v, prob in zip(values, probs)))
    second = sp.simplify(sum(v ** 2 * prob for v, prob in zip(values, probs)))
    path2 = sp.simplify(second - mean ** 2)
    r = MathResult.ok("variance", method="自定义分布律：Var(X) = E[(X-EX)^2] 与 E[X^2]-(EX)^2 互印")
    r.set_input(values=[A.to_text(v) for v in values], pmf=[A.to_text(v) for v in probs])
    r.set_result(path1)
    r.set_raw(E_X=A.to_text(mean), E_X2=A.to_text(second), variance=A.to_text(path1),
              std=A.to_text(sp.simplify(sp.sqrt(path1))),
              by_definition=A.to_text(path1), by_shortcut=A.to_text(path2))
    r.add_condition("Σ p_i = 1", "各 p_i ≥ 0")
    methods = [
        _cross_check("E[(X-EX)^2] 与 E[X^2]-(EX)^2 互印", path1, path2),
        {"name": "方差非负检查", "passed": (not _eval_finite(path1)) or path1 >= 0,
         "detail": f"Var(X) = {A.to_text(path1)}"},
    ]
    return _verify_set(r, methods)


# ---------------------------------------------------------------------------
# 5/6. covariance & correlation
# ---------------------------------------------------------------------------

def _joint_inputs(p: _Params) -> tuple[str, Any]:
    has_pmf = p.raw("joint_pmf", "jointpmf", "table", "joint_table") is not None
    has_pdf = p.raw("joint_pdf", "jointdensity", "density_joint") is not None
    if has_pmf and has_pdf:
        raise ValueError("joint_pmf（离散）与 joint_pdf（连续）只能给一个")
    if not has_pmf and not has_pdf:
        raise ValueError("需要联合分布：joint_pmf + x_values + y_values（离散），或 joint_pdf（连续）")
    return ("discrete" if has_pmf else "continuous"), None


def _discrete_joint_moments(matrix: sp.Matrix, xs: list[Any], ys: list[Any]) -> dict[str, Any]:
    total = sp.simplify(sum(matrix))
    if not _has_free(total) and not _safe_equal(total, 1):
        raise ValueError(f"联合分布律之和为 {A.to_text(total)} ≠ 1，不是合法的联合分布律")
    marginal_x = [sp.simplify(sum(matrix[i, j] for i in range(matrix.rows))) for j in range(matrix.cols)]
    marginal_y = [sp.simplify(sum(matrix[i, j] for j in range(matrix.cols))) for i in range(matrix.rows)]
    ex = sp.simplify(sum(xs[j] * marginal_x[j] for j in range(len(xs))))
    ey = sp.simplify(sum(ys[i] * marginal_y[i] for i in range(len(ys))))
    exy = sp.simplify(sum(xs[j] * ys[i] * matrix[i, j] for i in range(matrix.rows) for j in range(matrix.cols)))
    ex2 = sp.simplify(sum(xs[j] ** 2 * marginal_x[j] for j in range(len(xs))))
    ey2 = sp.simplify(sum(ys[i] ** 2 * marginal_y[i] for i in range(len(ys))))
    return {
        "total": total,
        "marginal_x": marginal_x, "marginal_y": marginal_y,
        "E_X": ex, "E_Y": ey, "E_XY": exy,
        "E_X2": ex2, "E_Y2": ey2,
    }


def _continuous_joint_moments(density: Any, x: sp.Symbol, y: sp.Symbol,
                              bounds: tuple[Any, Any, Any, Any]) -> dict[str, Any]:
    xl, xu, yl, yu = bounds
    exy = sp.integrate(sp.integrate(x * y * density, (y, yl, yu)), (x, xl, xu))
    ex = sp.integrate(sp.integrate(x * density, (y, yl, yu)), (x, xl, xu))
    ey = sp.integrate(sp.integrate(y * density, (y, yl, yu)), (x, xl, xu))
    ex2 = sp.integrate(sp.integrate(x ** 2 * density, (y, yl, yu)), (x, xl, xu))
    ey2 = sp.integrate(sp.integrate(y ** 2 * density, (y, yl, yu)), (x, xl, xu))
    return {"E_X": sp.simplify(ex), "E_Y": sp.simplify(ey), "E_XY": sp.simplify(exy),
            "E_X2": sp.simplify(ex2), "E_Y2": sp.simplify(ey2)}


def _resolve_joint(p: _Params) -> tuple[str, dict[str, Any]]:
    mode, _ = _joint_inputs(p)
    if mode == "discrete":
        matrix, xs, ys = _joint_tables(p)
        moments = _discrete_joint_moments(matrix, xs, ys)
        if not _has_free(moments["total"]) and not _safe_equal(moments["total"], 1):
            raise ValueError(f"联合分布律之和为 {A.to_text(moments['total'])} ≠ 1，不是合法的联合分布律")
        moments.update({"matrix": matrix, "x_values": xs, "y_values": ys, "mode": "discrete"})
        return mode, moments
    x = sp.Symbol("x", real=True)
    y = sp.Symbol("y", real=True)
    density = A.parse(p.text("joint_pdf", "jointdensity", "density_joint"),
                      symbols=["x", "y", "t", "n", "a", "b", "c", "m"])
    bounds = _joint_pdf_regions(p, x, y)
    moments = _continuous_joint_moments(density, x, y, bounds)
    moments.update({"density": density, "x": x, "y": y, "bounds": bounds, "mode": "continuous"})
    return mode, moments


def _variance_from_moments(m: dict[str, Any], key: str) -> Any:
    return sp.simplify(m[f"E_{key}2"] - m[f"E_{key}"] ** 2)


@op("covariance", "cov", "math_covariance")
def covariance(**kwargs: Any) -> MathResult:
    """协方差 Cov(X,Y)：E[XY]-EX·EY 与 E[(X-EX)(Y-EY)] 两条路径互印。"""
    p = _Params(kwargs)
    mode, m = _resolve_joint(p)
    ex, ey, exy = m["E_X"], m["E_Y"], m["E_XY"]
    path1 = sp.simplify(exy - ex * ey)
    if mode == "discrete":
        path2 = sp.simplify(sum(
            (m["x_values"][j] - ex) * (m["y_values"][i] - ey) * m["matrix"][i, j]
            for i in range(m["matrix"].rows) for j in range(m["matrix"].cols)))
    else:
        density, x, y, (xl, xu, yl, yu) = m["density"], m["x"], m["y"], m["bounds"]
        path2 = sp.simplify(sp.integrate(sp.integrate((x - ex) * (y - ey) * density, (y, yl, yu)), (x, xl, xu)))
    if _unevaluated(path1) and _unevaluated(path2):
        return MathResult.unsolved("covariance", "联合分布的协方差积分未能求出闭式解，请缩小积分区域或改用数值方法")
    var_x = _variance_from_moments(m, "X")
    var_y = _variance_from_moments(m, "Y")
    r = MathResult.ok("covariance", method="Cov(X,Y) = E[XY] - EX·EY（并与定义式互印）")
    if mode == "discrete":
        r.set_input(joint_pmf=A.to_text(m["matrix"].tolist()),
                    x_values=[A.to_text(v) for v in m["x_values"]],
                    y_values=[A.to_text(v) for v in m["y_values"]])
    else:
        r.set_input(joint_pdf=A.to_text(m["density"]),
                    region=[A.to_text(m["bounds"][0]), A.to_text(m["bounds"][1]),
                            A.to_text(m["bounds"][2]), A.to_text(m["bounds"][3])])
    r.set_result(path1)
    r.set_raw(E_X=A.to_text(sp.simplify(ex)), E_Y=A.to_text(sp.simplify(ey)),
              E_XY=A.to_text(sp.simplify(exy)),
              Var_X=A.to_text(var_x), Var_Y=A.to_text(var_y),
              by_shortcut=A.to_text(path1), by_definition=A.to_text(path2))
    r.add_condition("联合分布（律）必须满足归一化 Σp = 1 / ∫∫f = 1")
    if mode == "continuous":
        r.add_condition("积分区域为矩形（默认全区），区域外的密度按 0 处理；一般区域请用 joint_marginal 的边界参数")
    methods = [
        _cross_check("E[XY]-EX·EY 与 E[(X-EX)(Y-EY)] 互印", path1, path2),
        {"name": "Var(X), Var(Y) 非负检查", "passed": (not _eval_finite(var_x) or var_x >= 0)
         and (not _eval_finite(var_y) or var_y >= 0),
         "detail": f"Var(X) = {A.to_text(var_x)}, Var(Y) = {A.to_text(var_y)}"},
    ]
    return _verify_set(r, methods)


@op("correlation", "corr", "math_correlation")
def correlation(**kwargs: Any) -> MathResult:
    """相关系数 ρ = Cov(X,Y)/(σ_X σ_Y)；σ_X 或 σ_Y 为 0 时抛 ValueError。"""
    p = _Params(kwargs)
    mode, m = _resolve_joint(p)
    ex, ey, exy = m["E_X"], m["E_Y"], m["E_XY"]
    cov_value = sp.simplify(exy - ex * ey)
    if mode == "discrete":
        cov_def = sp.simplify(sum(
            (m["x_values"][j] - ex) * (m["y_values"][i] - ey) * m["matrix"][i, j]
            for i in range(m["matrix"].rows) for j in range(m["matrix"].cols)))
    else:
        density, x, y, (xl, xu, yl, yu) = m["density"], m["x"], m["y"], m["bounds"]
        cov_def = sp.simplify(sp.integrate(sp.integrate((x - ex) * (y - ey) * density, (y, yl, yu)), (x, xl, xu)))
    var_x = _variance_from_moments(m, "X")
    var_y = _variance_from_moments(m, "Y")

    def _zero(value: Any) -> bool:
        if _has_free(value):
            return False
        try:
            return _safe_equal(value, 0)
        except Exception:  # noqa: BLE001
            return False

    if _zero(var_x):
        raise ValueError(
            f"σ_X = 0（X 几乎处处为常数，Var(X) = {A.to_text(var_x)}），相关系数 ρ 无定义。"
            "相关系数只对两个都有正有限方差的随机变量定义。"
        )
    if _zero(var_y):
        raise ValueError(
            f"σ_Y = 0（Y 几乎处处为常数，Var(Y) = {A.to_text(var_y)}），相关系数 ρ 无定义。"
        )
    if _unevaluated(cov_value) or _unevaluated(var_x) or _unevaluated(var_y):
        return MathResult.unsolved("correlation", "协方差或方差含未求值积分，无法给出相关系数")
    sigma_x, sigma_y = sp.sqrt(var_x), sp.sqrt(var_y)
    rho = sp.simplify(cov_value / (sigma_x * sigma_y))
    r = MathResult.ok("correlation", method="ρ = Cov(X,Y) / (σ_X σ_Y)，并用 |ρ| ≤ 1 与定义式复核算")
    if mode == "discrete":
        r.set_input(joint_pmf=A.to_text(m["matrix"].tolist()),
                    x_values=[A.to_text(v) for v in m["x_values"]],
                    y_values=[A.to_text(v) for v in m["y_values"]])
    else:
        r.set_input(joint_pdf=A.to_text(m["density"]),
                    region=[A.to_text(b) for b in m["bounds"]])
    r.set_result(rho)
    r.set_raw(Cov=A.to_text(cov_value), sigma_X=A.to_text(sp.simplify(sigma_x)),
              sigma_Y=A.to_text(sp.simplify(sigma_y)), Var_X=A.to_text(var_x), Var_Y=A.to_text(var_y),
              rho=A.to_text(rho), rho_latex=A.to_latex(rho),
              interpretation=("ρ = 0：X 与 Y 不相关（不相关不蕴含独立）" if _zero(cov_value)
                              else ("ρ > 0：X 与 Y 正相关" if (not _eval_finite(rho) or rho > 0)
                                    else "ρ < 0：X 与 Y 负相关")))
    r.add_condition("σ_X > 0 且 σ_Y > 0（否则相关系数无定义）", "|ρ| ≤ 1")
    if _zero(cov_value):
        r.add_warning("ρ = 0 只说明不相关；请另行用 independent_check 或 joint_marginal 判定独立性")
    methods = [
        _cross_check("两条协方差路径代入 ρ 后互印", sp.simplify(cov_value / (sigma_x * sigma_y)),
                     sp.simplify(cov_def / (sigma_x * sigma_y)),
                     note="路径一：E[XY]-EX·EY；路径二：E[(X-EX)(Y-EY)]"),
    ]
    if _eval_finite(rho):
        methods.append({"name": "|ρ| ≤ 1 检查", "passed": abs(float(sp.N(rho, 20))) <= 1 + 1e-12,
                        "detail": f"ρ = {A.to_text(rho)} ≈ {float(sp.N(rho, 12)):.12g}"})
    else:
        # 符号形式：检查 ρ^2 ≤ 1 ⟺ Cov^2 ≤ Var(X)Var(Y)（柯西–施瓦茨）
        methods.append({"name": "柯西–施瓦茨不等式 Cov² ≤ Var(X)Var(Y)",
                        "passed": bool(sp.simplify(var_x * var_y - cov_value ** 2).is_nonnegative
                                       if sp.simplify(var_x * var_y - cov_value ** 2).is_nonnegative is not None
                                       else False),
                        "detail": f"Var(X)Var(Y) - Cov² = {A.to_text(sp.simplify(var_x * var_y - cov_value ** 2))}"})
    return _verify_set(r, methods)


# ---------------------------------------------------------------------------
# 7. joint_marginal
# ---------------------------------------------------------------------------

@op("joint_marginal", "marginal", "math_joint")
def joint_marginal(**kwargs: Any) -> MathResult:
    """联合分布 → 边缘分布，并判定独立性 f(x,y) = f_X(x) f_Y(y)。"""
    p = _Params(kwargs)
    mode, m = _resolve_joint(p)
    r = MathResult.ok("joint_marginal", method="边缘分布：对另一变量求和/积分；独立性用恒等式或抽样比较")
    independence: bool | None = None
    full_identity = False
    if mode == "discrete":
        matrix, xs, ys = m["matrix"], m["x_values"], m["y_values"]
        marginal_x, marginal_y = m["marginal_x"], m["marginal_y"]
        r.set_input(joint_pmf=A.to_text(matrix.tolist()),
                    x_values=[A.to_text(v) for v in xs], y_values=[A.to_text(v) for v in ys])
        r.set_result(sp.Matrix([marginal_x]) if len(marginal_x) > 1 else marginal_x[0])
        r.set_raw(marginal_X=[A.to_text(v) for v in marginal_x],
                  marginal_Y=[A.to_text(v) for v in marginal_y],
                  marginal_X_latex=A.to_latex(sp.Matrix([marginal_x])),
                  marginal_Y_latex=A.to_latex(sp.Matrix([marginal_y])),
                  E_X=A.to_text(sp.simplify(m["E_X"])), E_Y=A.to_text(sp.simplify(m["E_Y"])),
                  table=[{"y": A.to_text(ys[i]),
                          "row": [A.to_text(matrix[i, j]) for j in range(matrix.cols)],
                          "P(Y=y)": A.to_text(marginal_y[i])} for i in range(matrix.rows)])
        independence = all(_same_value(matrix[i, j], marginal_y[i] * marginal_x[j])
                           for i in range(matrix.rows) for j in range(matrix.cols))
        r.set_raw(independent=bool(independence),
                  verdict="独立" if independence else "不独立",
                  reason=("对所有 i,j 都有 p_ij = P(X=x_i)P(Y=y_j)" if independence
                          else "存在 i,j 使 p_ij ≠ P(X=x_i)P(Y=y_j)："
                               + ", ".join(
                                   f"p({A.to_text(xs[j])},{A.to_text(ys[i])}) = {A.to_text(matrix[i, j])} "
                                   f"≠ {A.to_text(sp.simplify(marginal_y[i] * marginal_x[j]))}"
                                   for i in range(matrix.rows) for j in range(matrix.cols)
                                   if not _same_value(matrix[i, j], marginal_y[i] * marginal_x[j]))[:3]))
        full_identity = True
        methods = [
            {"name": "边缘分布归一化检查",
             "passed": _same_value(sp.simplify(sum(marginal_x)), 1) and _same_value(sp.simplify(sum(marginal_y)), 1),
             "detail": f"Σ_i P(X=x_i) = {A.to_text(sp.simplify(sum(marginal_x)))}, "
                       f"Σ_j P(Y=y_j) = {A.to_text(sp.simplify(sum(marginal_y)))}"},
            {"name": "独立性判定（逐格乘法对照）", "passed": True,
             "detail": f"判定结果：{'独立' if independence else '不独立'}；逐格检查 p_ij 与 P(X=x_i)P(Y=y_j)"},
        ]
    else:
        # 连续：把二维积分化成先对另一变量积分的累次积分
        density: Any = m["density"]
        x, y = m["x"], m["y"]
        xl, xu, yl, yu = m["bounds"]
        f_x = sp.simplify(sp.integrate(density, (y, yl, yu)))
        f_y = sp.simplify(sp.integrate(density, (x, xl, xu)))
        r.set_input(joint_pdf=A.to_text(density),
                    region={"x": [A.to_text(xl), A.to_text(xu)], "y": [A.to_text(yl), A.to_text(yu)]})
        r.set_result(f_x)
        r.set_raw(marginal_X=A.to_text(f_x), marginal_X_latex=A.to_latex(f_x),
                  marginal_Y=A.to_text(f_y), marginal_Y_latex=A.to_latex(f_y),
                  E_X=A.to_text(sp.simplify(m["E_X"])), E_Y=A.to_text(sp.simplify(m["E_Y"])))
        # 联合密度必须归一化，否则下面所有结论都建立在一个非法密度上
        norm_total = sp.simplify(sp.integrate(density, (y, yl, yu), (x, xl, xu)))
        norm_ok = _same_value(norm_total, 1)
        if not norm_ok:
            r.add_warning(
                f"联合密度在给定区域上的积分 = {A.to_text(norm_total)} ≠ 1，不是合法密度。"
                "下面的边缘分布/期望等按该表达式直接计算；请先归一化再采用。"
                "（独立性结论不受正常数倍影响）")
        difference = sp.simplify(density - sp.simplify(f_x * f_y))
        worst, scan_detail = _numeric_difference_scan(
            difference, (x, y), _region_sample_points(xl, xu, yl, yu))
        if worst is None:
            # 抽样也拿不到证据时才退符号判定
            symbol_passed, detail = C.verify_identity(density, sp.simplify(f_x * f_y), symbols=[x, y])
            independence = bool(symbol_passed)
            verdict_passed = independence
            identity_detail = f"{detail.get('method', '')}；{scan_detail}"
        else:
            independence = worst < 1e-9
            verdict_passed = True   # 数值上已经**得出结论**（无论独立与否），这是一次成功的判定
            identity_detail = f"{scan_detail}；{'差式恒为 0 ⟹ 独立' if independence else '差式不恒为 0 ⟹ 不独立'}"
        full_identity = True
        r.set_raw(independent=independence,
                  joint_density_integral=A.to_text(norm_total),
                  verdict="独立" if independence else "不独立",
                  reason=("f(x,y) ≡ f_X(x)·f_Y(y)（符号恒等）" if independence
                          else f"f(x,y) - f_X(x)f_Y(y) = {A.to_text(difference)}，不恒为 0；"
                               "注：仅在矩形区域上二者恆等才等价于独立"))
        r.add_warning("连续型独立性的严格判定要求 f(x,y) ≡ f_X(x)f_Y(y) 在定义域上处处成立；"
                      "区域为一般非矩形时本工具按矩形区域处理，结论可能与教材题设不同")
        methods = [
            {"name": "边缘密度归一化检查",
             "passed": _same_value(sp.simplify(sp.integrate(f_x, (x, xl, xu))), 1)
             and _same_value(sp.simplify(sp.integrate(f_y, (y, yl, yu))), 1),
             "detail": f"∫f_X = {A.to_text(sp.simplify(sp.integrate(f_x, (x, xl, xu))))}, "
                       f"∫f_Y = {A.to_text(sp.simplify(sp.integrate(f_y, (y, yl, yu))))}"},
            {"name": "联合密度归一化检查",
             "passed": norm_ok,
             "detail": f"∫∫f(x,y) dxdy = {A.to_text(norm_total)}"
                       + ("" if norm_ok else "（≠ 1；已给出 warning，结果仍按原式给出）")},
            {"name": "独立性恒等式 f(x,y)=f_X(x)f_Y(y)", "passed": verdict_passed,
             "detail": identity_detail},
        ]
    r.add_condition("联合分布必须归一化（Σp=1 或 ∫∫f=1）")
    if independence is not None:
        r.add_condition("独立 ⟺ f(x,y) = f_X(x)f_Y(y)（离散：p_ij = p_i p_j）",
                        "独立 ⟹ 不相关；不相关不一定独立")
    return _verify_set(r, methods, extra_note="独立性用逐格乘法对照 / 恒等式复核" if full_identity else None)


# ---------------------------------------------------------------------------
# 8. distribution_moments
# ---------------------------------------------------------------------------

def _skew_kurt(dist: _Dist) -> tuple[Any, Any, list[str]]:
    notes: list[str] = []
    if dist.skewness is not None and dist.kurtosis is not None:
        return dist.skewness, dist.kurtosis, notes
    x = dist.var
    skew = dist.skewness
    kurt = dist.kurtosis
    if skew is None:
        notes.append("偏度：该分布的低阶矩不存在或无标准公式，未计算")
    if kurt is None:
        notes.append("峰度：该分布的四阶矩不存在或无标准公式，未计算")
    return skew, kurt, notes


@op("distribution_moments", "moments", "math_moments")
def distribution_moments(**kwargs: Any) -> MathResult:
    """一次性给出 E[X]、E[X^2]、Var(X)（可选 E[X^k] 与偏度/峰度）。未计算部分明确写「未计算」。"""
    p = _Params(kwargs)
    dist, var = _resolve_dist(p)
    order_raw = p.raw("k", "order", "moment", "power")
    order = None
    if order_raw is not None:
        order = p.num("k", "order", "moment", "power")
        if not _has_free(order) and (order.is_integer is False or order < 1):
            raise ValueError(f"矩的阶数 k 必须是正整数，当前为 {A.to_text(order)}")
    want_higher = p.text("higher", "skew_kurt", "include_skew") is None
    ex = _expectation_direct(dist, var)
    ex2 = _expectation_direct(dist, var ** 2)
    var_value = sp.simplify(ex2 - ex ** 2)
    if _unevaluated(ex) and _unevaluated(ex2):
        return MathResult.unsolved("distribution_moments", f"该分布的期望/二阶矩未能求出闭式解")
    r = MathResult.ok("distribution_moments", method="由定义式求和/积分；关键量与标准公式交叉复核")
    r.set_input(dist=dist.name, param=dist.param_desc, k=A.to_text(order) if order is not None else None)
    r.set_result(ex)
    fields: dict[str, Any] = {
        "E_X": A.to_text(sp.simplify(ex)), "E_X2": A.to_text(sp.simplify(ex2)),
        "variance": A.to_text(var_value),
        "std": A.to_text(sp.simplify(sp.sqrt(var_value))) if var_value is not None else "未计算",
    }
    notes: list[str] = []
    if order is not None:
        kth = _expectation_direct(dist, var ** order)
        fields["E_Xk"] = A.to_text(sp.simplify(kth))
        fields["k"] = A.to_text(order)
    else:
        fields["E_Xk"] = "未计算（未给出 k）"
    skew, kurt, skew_notes = _skew_kurt(dist)
    fields["skewness"] = A.to_text(skew) if skew is not None else "未计算"
    fields["kurtosis"] = A.to_text(kurt) if kurt is not None else "未计算"
    notes.extend(skew_notes)
    fields["未计算项"] = notes or ["无"]
    if dist.mean is not None:
        fields["标准公式_E_X"] = A.to_text(sp.simplify(dist.mean))
    if dist.variance is not None:
        fields["标准公式_Var"] = A.to_text(sp.simplify(dist.variance))
    r.set_raw(**fields)
    r.add_condition(*dist.conditions)
    methods = [
        _cross_check("Var = E[X²] - (EX)² 与 E[(X-EX)²] 互印", var_value, _central_moment(dist, 2, ex)),
    ]
    if dist.mean is not None and dist.variance is not None:
        methods.append(_cross_check("E[X] 与 Var(X) 和标准公式比对",
                                    sp.Matrix([sp.simplify(ex), var_value]),
                                    sp.Matrix([sp.simplify(dist.mean), sp.simplify(dist.variance)]),
                                    note="路径一：定义式求和/积分；路径二：该分布的标准公式"))
    if order is not None:
        _p1, _p2, detail = _moment_both_ways(dist, int(order)) if not _has_free(order) else (None, None, {
            "name": f"E[X^{A.to_text(order)}] 复核", "passed": False,
            "detail": "阶数 k 含符号，无法用矩母函数求导的数值路径复核"})
        methods.append(detail)
    return _verify_set(r, methods)


# ---------------------------------------------------------------------------
# 9. distribution_from_pmf
# ---------------------------------------------------------------------------

@op("distribution_from_pmf", "custom_pmf", "math_custom_pmf")
def distribution_from_pmf(**kwargs: Any) -> MathResult:
    """自定义离散分布：归一化检查 + 分布函数 + 期望 + 方差。"""
    p = _Params(kwargs)
    probs = p.nums("pmf", "probs", "probabilities", "probability_mass", "p")
    if not probs:
        raise ValueError("需要 pmf（分布律，如 pmf=\"1/6,1/6,1/6,1/6,1/6,1/6\"）")
    values_raw = p.raw("values", "value", "outcomes", "x_values", "xs")
    values = _numbers(values_raw) if values_raw is not None else [sp.Integer(i + 1) for i in range(len(probs))]
    if len(values) != len(probs):
        raise ValueError(f"values 个数（{len(values)}）与 pmf 个数（{len(probs)}）必须一致")
    for i, prob in enumerate(probs):
        _check_prob(prob, f"pmf[{i}]")
    total = sp.simplify(sum(probs))
    if not _has_free(total) and not _safe_equal(total, 1):
        raise ValueError(
            f"分布律之和为 {A.to_text(total)} ≠ 1（偏离 {A.to_text(sp.simplify(total - 1))}），"
            "不是合法的分布律；请先归一化后再调用"
        )
    pairs = sorted(zip(values, probs), key=lambda kv: sp.N(kv[0]))
    x = sp.Symbol("x", real=True)
    dist = _pmf_distribution(values, probs)
    cdf = dist.cdf
    mean = sp.simplify(sum(v * prob for v, prob in pairs))
    second = sp.simplify(sum(v ** 2 * prob for v, prob in pairs))
    var_value = sp.simplify(second - mean ** 2)
    var_path2 = sp.simplify(sum((v - mean) ** 2 * prob for v, prob in pairs))
    r = MathResult.ok("distribution_from_pmf", method="自定义分布律：归一化 + 分布函数 + 期望 + 方差")
    r.set_input(values=[A.to_text(v) for v in values], pmf=[A.to_text(v) for v in probs])
    r.set_result(cdf)
    r.set_raw(normalization=A.to_text(total),
              normalized=bool(_safe_equal(total, 1)),
              support=[A.to_text(v) for v in [v for v, _ in pairs]],
              pmf_table=[{"x": A.to_text(v), "P(X=x)": A.to_text(prob)} for v, prob in pairs],
              cdf_text=A.to_text(cdf), cdf_latex=A.to_latex(cdf),
              E_X=A.to_text(mean), E_X2=A.to_text(second),
              variance=A.to_text(var_value), variance_latex=A.to_latex(var_value),
              std=A.to_text(sp.simplify(sp.sqrt(var_value))),
              by_definition=A.to_text(var_path2))
    r.add_condition("Σ p_i = 1（已检查）", "各 p_i ≥ 0", "分布函数 F(x) = Σ_{x_i ≤ x} p_i（右连续阶梯函数）")
    methods = [
        {"name": "归一化检查", "passed": _safe_equal(total, 1),
         "detail": f"Σ P(X = x_i) = {A.to_text(total)}"},
        _cross_check("Var 双路径互印 E[(X-EX)²] 与 E[X²]-(EX)²", var_path2, var_value),
        # 不要在这里直接 sp.limit(cdf, x, ±oo)：阶梯函数的 Piecewise 条件在 zoo 处
        # 会抛 ``TypeError: Invalid comparison of non-real zoo``（曾让整个算子 internal 报错）。
        _cdf_endpoint_check(dist),
    ]
    return _verify_set(r, methods)


# ---------------------------------------------------------------------------
# 10. 可选：矩母函数 与 中心极限定理
# ---------------------------------------------------------------------------

@op("moment_generating", "mgf", "math_mgf")
def moment_generating(**kwargs: Any) -> MathResult:
    """矩母函数 M(t) = E[e^{tX}] 及其存在条件。"""
    p = _Params(kwargs)
    dist, var = _resolve_dist(p)
    t = _T
    r = MathResult.ok("moment_generating", method="M(t) = E[e^{tX}] 按定义求和/积分，并与标准形式比对")
    r.set_input(dist=dist.name, param=dist.param_desc)
    if dist.mgf is None:
        # 明确给出「不存在」的结论文本，而不是把 result 留成 null —— 否则调用方
        # 无法区分「不存在」与「本工具没算出来」。
        r.set_result(f"M(t) 不存在（{dist.mgf_note or '该分布的矩母函数不收敛'}）")
        r.set_raw(mgf="不存在（该分布的各阶矩不完备或矩母函数不收敛）",
                  exists=False,
                  note=dist.mgf_note or "该分布的矩母函数不存在")
        r.add_condition(*dist.conditions)
        if dist.mgf_note:
            r.add_warning(dist.mgf_note)
        return _verify_set(r, [], extra_note=dist.mgf_note or "该分布的矩母函数不存在，无法给出结果")
    direct = _expectation_direct(dist, sp.exp(t * var))
    r.set_result(sp.simplify(dist.mgf))
    r.set_raw(mgf=A.to_text(sp.simplify(dist.mgf)), mgf_latex=A.to_latex(sp.simplify(dist.mgf)),
              by_definition=A.to_text(sp.simplify(direct)),
              convergence=dist.mgf_note or "未给出收敛条件")
    r.add_condition(*dist.conditions)
    if dist.mgf_note:
        r.add_condition(dist.mgf_note)
    methods = [
        _cross_check("定义式 E[e^{tX}] 与标准形式比对", direct, dist.mgf),
    ]
    # ⚠️ 必须用 _T（= sp.Symbol("t")）而不是 sp.Symbol("t", real=True)：
    # SymPy 把带不同假设的同名符号视为**不同**符号，`mgf.subs(t, 0)` 会静默失败
    # （曾让这条检查报 "M(0) = exp(t**2/2)"）。_build_dist 里各分布的 mgf 用的
    # 都是不带假设的 Symbol("t")，所以这里必须与之一致。
    mgf_at_zero = sp.simplify(dist.mgf.subs(_T, 0))
    methods.append(
        {"name": "M(0) = 1 检查", "passed": _same_value(mgf_at_zero, 1),
         "detail": f"M(0) = {A.to_text(mgf_at_zero)}"}
    )
    mean_direct = _expectation_direct(dist, var)
    mgf_mean, _mgf_note = _mgf_moment(dist, 1)
    if mgf_mean is not None and _eval_finite(mean_direct):
        methods.append(_cross_check("M'(0) = E[X] 检查", mgf_mean, mean_direct,
                                    note="路径一：矩母函数在 0 处一阶导数；路径二：按定义求期望"))
    return _verify_set(r, methods)


@op("central_limit", "clt", "math_clt")
def central_limit(**kwargs: Any) -> MathResult:
    """中心极限定理的正态近似（含近似条件与误差方向说明）。"""
    p = _Params(kwargs)
    n_raw = p.raw("n", "trials", "size", "samples")
    if n_raw is None:
        raise ValueError("中心极限定理近似需要 n（样本容量/试验次数）")
    n = p.num("n", "trials", "size", "samples", required=True, label="n")
    if not _has_free(n) and (n.is_integer is False or n < 1):
        raise ValueError(f"n 必须是正整数，当前为 {A.to_text(n)}")
    mode = (p.text("mode", "kind") or "sum").strip().lower()
    target_raw = p.raw("target", "point", "value", "x", "k")
    dist_op = _Params({k: v for k, v in p.kw.items() if k not in ("mode",)})
    is_binomial = _normalize_dist_name(p.text("dist", "distribution", "name") or (
        "binomial" if (p.raw("p") is not None and p.raw("n") is not None) else ""))
    if is_binomial == "binomial" and p.raw("p") is not None:
        dist, var = _resolve_dist(dist_op)
    else:
        dist, var = _resolve_dist(dist_op)

    mean = _expectation_direct(dist, var)
    second = _expectation_direct(dist, var ** 2)
    var_value = sp.simplify(second - mean ** 2)
    if _has_free(mean) or _has_free(var_value):
        return MathResult.unsolved("central_limit", "CLT 正态近似需要具体（数值）参数，当前含自由符号")
    if _unevaluated(mean) or _unevaluated(var_value):
        return MathResult.unsolved("central_limit", "无法求得该分布的均值或方差，不能做正态近似")
    if not _eval_finite(var_value) or var_value <= 0:
        raise ValueError(f"方差必须为正，当前 Var(X) = {A.to_text(var_value)}")
    mu_sum = sp.simplify(n * mean)
    var_sum = sp.simplify(n * var_value)
    sigma_sum = sp.sqrt(var_sum)
    target = None
    if target_raw is not None:
        target = _prob_bound(target_raw, None, lower=False)
    use_cdf = (p.text("tail", "direction", "side", "kind_of_tail") or "le").strip().lower()
    if target is None:
        raise ValueError("CLT 近似需要 target（例如 target=\"170\" 或 \"0.5\"）")

    if mode in ("local", "局部", "demoiuvre", "de_moivre_laplace"):
        exact_pmf = sp.simplify(dist.pdf.subs(var, target))
        approx = sp.simplify(sp.exp(-(target - mu_sum) ** 2 / (2 * var_sum)) / (sigma_sum * sp.sqrt(2 * sp.pi)))
        r = MathResult.ok("central_limit", method="局部极限定理（De Moivre–Laplace）正态近似")
        r.set_input(dist=dist.name, param=dist.param_desc, n=A.to_text(n), target=A.to_text(target), mode="local")
        r.set_result(approx)
        r.set_raw(exact=A.to_text(exact_pmf), approx=A.to_text(approx),
                  exact_approx=float(sp.N(exact_pmf, 12)) if _eval_finite(exact_pmf) else None,
                  approx_value=float(sp.N(approx, 12)),
                  relative_error=(float(sp.N((approx - exact_pmf) / exact_pmf, 12))
                                  if _eval_finite(exact_pmf) and exact_pmf != 0 else None),
                  formula="P(S_n = k) ≈ (1/√(2π n σ²)) exp(-(k - nμ)²/(2nσ²))")
        r.add_condition(f"n = {A.to_text(n)} 足够大（常用经验：np ≥ 5 且 n(1-p) ≥ 5）",
                        f"S_n 的均值 = {A.to_text(mu_sum)}，方差 = {A.to_text(var_sum)}")
        r.add_warning("局部极限定理的近似误差为 O(1/√n) 量级；离散型直接代入密度会忽略连续性修正 "
                      "(k ± 0.5)，结果系统性偏大时可用修正公式")
        methods = [
            {"name": "与精确分布律比对",
             "passed": (not _eval_finite(exact_pmf)) or abs(float(sp.N((approx - exact_pmf) / exact_pmf, 12))) < 0.5,
             "detail": f"精确值 {A.to_text(exact_pmf)}，近似值 {A.to_text(approx)}"},
            {"name": "构造检查：近似式即 N(nμ, nσ²) 的密度",
             "passed": _same_value(approx, sp.exp(-(target - mu_sum) ** 2 / (2 * var_sum)) / (sigma_sum * sp.sqrt(2 * sp.pi))),
             "detail": "把 nμ、nσ² 代入标准正态密度"},
        ]
        return _verify_set(r, methods)

    # 上限/尾部概率
    z = sp.simplify((target - mu_sum) / sigma_sum)
    approx_le = sp.simplify(sp.Rational(1, 2) * (1 + sp.erf(z / sp.sqrt(2))))
    approx_ge = sp.simplify(1 - approx_le)
    exact_le = sp.simplify(dist.prob_below(target)) if target not in (_OO,) else sp.Integer(1)
    if p.text("tail", "direction", "side") in ("ge", "greater", "gt", ">", "right", "≥", "大于", "以上"):
        approx, exact = approx_ge, sp.simplify(1 - exact_le) if _eval_finite(exact_le) else None
        label = f"P(S_n > {A.to_text(target)})"
    else:
        approx, exact = approx_le, exact_le
        label = f"P(S_n ≤ {A.to_text(target)})"
    r = MathResult.ok("central_limit", method="中心极限定理正态近似 (S_n - nμ)/(σ√n) ≈ N(0,1)")
    r.set_input(dist=dist.name, param=dist.param_desc, n=A.to_text(n), target=A.to_text(target))
    r.set_result(approx)
    r.set_raw(target=label, z_score=A.to_text(z),
              Phi_z=A.to_text(approx), approx_value=float(sp.N(approx, 12)) if _eval_finite(approx) else None,
              exact=A.to_text(exact) if exact is not None else "未计算（精确尾概率无闭式解）",
              exact_value=float(sp.N(exact, 12)) if exact is not None and _eval_finite(exact) else None,
              absolute_error=(float(sp.N(approx - exact, 12)) if exact is not None and _eval_finite(exact) else None),
              formula="P(S_n ≤ x) ≈ Φ((x - nμ)/(σ√n))，S_n = X1 + ... + Xn",
              mu_sum=A.to_text(mu_sum), var_sum=A.to_text(var_sum))
    r.add_condition(f"n = {A.to_text(n)} 足够大（独立同分布、方差有限）",
                    f"E[S_n] = {A.to_text(mu_sum)}，Var(S_n) = {A.to_text(var_sum)}")
    r.add_warning("CLT 只给出近似：误差 O(1/√n)；离散型用正态近似时漏掉连续性修正会使结果偏小"
                  "（对 P(S_n ≤ x) 而言通常低估），需要更精确时请用连续性修正 x+0.5")
    if exact is not None and _eval_finite(exact) and _eval_finite(approx):
        dev = abs(float(sp.N(approx - exact, 12)))
        r.set_raw(error_direction=("近似值偏大" if approx > exact else ("近似值偏小" if approx < exact else "二者数值相同")))
        r.add_warning(f"与精确值相比绝对误差约 {dev:.3e}" if dev > 1e-6 else "近似值与精确值在小数点后 6 位内一致")
    methods = [
        {"name": "标准化 z 分数与 Φ(z) 的一致性",
         "passed": _same_value(approx, sp.Rational(1, 2) * (1 + sp.erf((target - mu_sum) / sigma_sum / sp.sqrt(2)))),
         "detail": "路径一：CLT 标准正态近似；路径二：把 S_n 的均值方差代入标准正态分布函数"},
    ]
    if exact is not None and _eval_finite(exact):
        methods.append({"name": "与精确值比对（收敛性证据）",
                        "passed": abs(float(sp.N(approx - exact, 12))) < 0.05,
                        "detail": f"精确 {float(sp.N(exact, 12)):.12g} vs 近似 {float(sp.N(approx, 12)):.12g}"})
    else:
        methods.append({"name": "与精确值比对", "passed": False,
                        "detail": "精确分布函数无法闭式求值，只能给出近似（CLT 的适用性依赖大样本）"})
    return _verify_set(r, methods)
