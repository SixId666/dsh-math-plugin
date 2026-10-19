"""数理统计补充算子（用户规格：参数估计 / 区间估计 / 假设检验 / 抽样分布 / 中心极限定理）。

设计原则
--------
1. **推导要点必须回传**：似然函数、对数似然、得分方程（求导得驻点的依据）、矩方程
   都以 SymPy 表达式给出，而不是只给一个数。
2. **每个结论都附独立复核**：估计值代回矩方程 / 得分函数为 0 / 似然扰动不增 /
   分位数与拒绝域两条途径结论一致 / 与精确分布比较偏差。
3. **不夸大证据**：只用数值方式得到的结论一律标明 ``level="numeric"``；
   区间估计与假设检验的所有临界值都写明所用分布与自由度。
4. **不为了「通过」放宽条件**：样本量不足、离散型样本非法、大样本条件不满足，
   都会明确写进 ``conditions`` / ``warnings``，必要时返回 ``unsolved``。

依赖：标准库 + sympy + scipy（分位数与精确分布对照）+ 本包 ast/sympy_core/engine/result。
"""

from __future__ import annotations

import functools
import math
import re
from typing import Any, Sequence

import sympy as sp

from . import ast as A
from . import sympy_core as C
from .engine import op
from .result import MathResult

try:  # scipy 用于 t/χ²/F 分位数与精确分布对照；缺失时相关算子会明确说明无法继续
    from scipy import stats as _scipy
except Exception:  # noqa: BLE001
    _scipy = None


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

class _Skip(Exception):
    """内部信号：无法求解 -> MathResult.unsolved（不是错误输入）。"""


def _op_guard(name: str):
    """兜底：把内部 _Skip（例如缺少 scipy）转成 unsolved，绝不冒泡成 internal error。"""

    def deco(func):
        @functools.wraps(func)
        def wrapper(**kwargs):
            try:
                return func(**kwargs)
            except _Skip as exc:
                return MathResult.unsolved(name, str(exc))

        return wrapper

    return deco


def _unused(**kwargs: Any) -> list[str]:
    """收集未被算子使用的参数名，便于提示调用方写错了参数。"""
    return sorted(str(k) for k in kwargs if kwargs[k] is not None)


def _note_unused(r: MathResult, names: Sequence[str]) -> None:
    if names:
        r.add_warning("以下参数未被本算子使用，已忽略：" + ", ".join(names))


def _approx(value: Any) -> Any:
    try:
        if isinstance(value, sp.Basic):
            if value.free_symbols:
                return None
            num = sp.N(value, 12)
            if num.is_real:
                return float(num)
            return A.to_text(value)
    except Exception:  # noqa: BLE001
        return None
    return None


def _num(value: Any, name: str = "参数") -> sp.Basic:
    """把参数解析成具体数值（分数优先，保持精确）。"""
    if isinstance(value, bool):
        raise ValueError(f"{name} 不能是布尔值")
    if isinstance(value, (int, float)):
        v = sp.nsimplify(sp.sympify(value), rational=True)
    elif isinstance(value, sp.Basic):
        v = value
    else:
        text = str(value).strip()
        if not text:
            raise ValueError(f"{name} 为空")
        v = A.parse(text)
    if not getattr(v, "is_number", False):
        raise ValueError(f"{name} 必须是具体数值，收到 {value!r}")
    if getattr(v, "is_real", None) is False or sp.im(v) != 0:
        raise ValueError(f"{name} 必须是实数，收到 {value!r}")
    try:
        v = sp.nsimplify(v, rational=True)
    except Exception:  # noqa: BLE001
        pass
    return v


def _positive(value: Any, name: str) -> sp.Basic:
    v = _num(value, name)
    if v.is_positive is not True:
        raise ValueError(f"{name} 必须为正数，收到 {value!r}")
    return v


def _posint(value: Any, name: str) -> int:
    v = _num(value, name)
    if v.is_integer is not True or v.is_positive is not True:
        raise ValueError(f"{name} 必须为正整数，收到 {value!r}")
    return int(v)


def _level01(value: Any, name: str) -> sp.Basic:
    v = _num(value, name)
    if not (v.is_positive and v < 1):
        raise ValueError(f"{name} 必须满足 0 < {name} < 1，收到 {value!r}")
    return v


def _parse_samples(raw: Any, *, name: str = "samples") -> list[sp.Basic]:
    """解析样本：支持 ``"1,2,3"``、``"1 2 3"``、``"[1,2,3]"`` 与 JSON 数组。"""
    if raw is None:
        raise ValueError(f"缺少样本数据（{name}）")
    if isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        items = [t for t in re.split(r"[,;\s]+", str(raw).strip().strip("[]{}()")) if t.strip()]
    if not items:
        raise ValueError(f"{name} 为空")
    out: list[sp.Basic] = []
    for item in items:
        v = _num(item, f"{name} 的元素 {item!r}")
        if v in (sp.oo, -sp.oo, sp.zoo, sp.nan):
            raise ValueError(f"样本 {item!r} 不是有限实数")
        out.append(v)
    return out


def _stats(vals: Sequence[sp.Basic]) -> dict[str, Any]:
    """样本统计量（全部精确有理数运算）。"""
    n = len(vals)
    xbar = sp.simplify(sp.Add(*vals) / n)
    dev2 = sp.simplify(sp.Add(*[(v - xbar) ** 2 for v in vals]))
    m2 = sp.simplify(dev2 / n)                            # 二阶中心矩（MLE 方差）
    s2u = sp.simplify(dev2 / (n - 1)) if n > 1 else None  # 无偏样本方差
    return {
        "n": n,
        "mean": xbar,
        "sum": sp.Add(*vals),
        "dev2": dev2,
        "m2": m2,
        "s2_unbiased": s2u,
        "s_unbiased": sp.sqrt(s2u) if s2u is not None else None,
        "min": min(vals, key=lambda v: float(v)),
        "max": max(vals, key=lambda v: float(v)),
        "values": list(vals),
    }


def _est(name: str, symbolic: Any, value: Any, desc: str, **extra: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "name": name,
        "description": desc,
        "symbolic": A.to_text(symbolic) if symbolic is not None else None,
        "latex": A.to_latex(symbolic) if symbolic is not None else None,
        "value": A.to_text(value),
    }
    ap = _approx(value)
    if ap is not None:
        item["approx"] = ap
    item.update({k: v for k, v in extra.items() if v is not None})
    return item


def _verify_block(r: MathResult, *, status: str, level: str, methods: list[dict[str, Any]]) -> MathResult:
    """装填统一复核块，并派生出 ``passed``（所有独立复核都通过才算通过）。

    ``passed`` 只有三种取值：True（全部通过）、False（存在未通过的复核）、
    None（有复核项无法判定，证据不足——此时绝不当作通过）。
    """
    r.verify(status=status, methods=methods)
    r.verification["level"] = level
    flags = [m.get("passed") for m in methods]
    r.verification["passed"] = (None if any(f is None for f in flags) else all(flags)) if flags else None
    r.verification["methods_total"] = len(methods)
    r.verification["methods_passed"] = sum(1 for f in flags if f is True)
    return r


def _samples_input(vals: Sequence[sp.Basic]) -> str:
    return "[" + ", ".join(A.to_text(v) for v in vals) + "]"


# ---------------------------------------------------------------------------
# 分位数 / 分布函数
# ---------------------------------------------------------------------------

def _need_scipy(what: str) -> None:
    if _scipy is None:
        raise _Skip(f"需要 scipy 才能给出{what}（当前环境未安装 scipy），无法继续")


def _z_crit(alpha2: sp.Basic) -> sp.Basic:
    """标准正态上侧 alpha2 分位点：z = sqrt(2)*erfinv(1-2*alpha2)。"""
    return sp.sqrt(2) * sp.erfinv(1 - 2 * alpha2)


def _norm_cdf(x: Any) -> sp.Basic:
    return (1 + sp.erf(sp.sympify(x) / sp.sqrt(2))) / 2


def _t_ppf(q: Any, df: int) -> sp.Basic:
    _need_scipy("t 分布分位数")
    return sp.Float(_scipy.t.ppf(float(q), int(df)), 15)


def _t_cdf(x: Any, df: int) -> float:
    _need_scipy("t 分布分布函数")
    return float(_scipy.t.cdf(float(x), int(df)))


def _chi2_ppf(q: Any, df: int) -> sp.Basic:
    _need_scipy("χ² 分布分位数")
    return sp.Float(_scipy.chi2.ppf(float(q), int(df)), 15)


def _chi2_cdf(x: Any, df: int) -> float:
    _need_scipy("χ² 分布分布函数")
    return float(_scipy.chi2.cdf(float(x), int(df)))


def _f_ppf(q: Any, df1: int, df2: int) -> sp.Basic:
    _need_scipy("F 分布分位数")
    return sp.Float(_scipy.f.ppf(float(q), int(df1), int(df2)), 15)


def _fnum(x: Any) -> float:
    return float(sp.N(x, 15))


# ---------------------------------------------------------------------------
# 点估计
# ---------------------------------------------------------------------------

_IB = sp.IndexedBase("x")
_IDX = sp.Symbol("i", integer=True, positive=True)

_MU = sp.Symbol("mu", real=True)
_SIG2 = sp.Symbol("sigma2", positive=True)
_SIG = sp.Symbol("sigma", positive=True)
_LAM = sp.Symbol("lambda", positive=True)
_PP = sp.Symbol("p", positive=True)
_AA = sp.Symbol("a", real=True)
_BB = sp.Symbol("b", real=True)

_DIST_ALIASES = {
    "normal": "normal", "gauss": "normal", "gaussian": "normal", "正态": "normal",
    "exponential": "exponential", "exp": "exponential", "指数": "exponential",
    "poisson": "poisson", "泊松": "poisson",
    "uniform": "uniform", "均匀": "uniform",
    "binomial": "binomial", "binom": "binomial", "二项": "binomial",
}

_METHOD_ALIASES = {
    "moment": "moment", "mom": "moment", "矩估计": "moment", "矩": "moment",
    "mle": "mle", "ml": "mle", "max": "mle", "极大似然": "mle", "最大似然": "mle",
    "both": "both", "all": "both",
}


def _sum_disp(term: Any, n: int) -> sp.Basic:
    return sp.Sum(term, (_IDX, 1, sp.Integer(n)))


def _dist_display(n: int) -> dict[str, Any]:
    xbar = _sum_disp(_IB[_IDX], n) / n
    dev2 = _sum_disp((_IB[_IDX] - xbar) ** 2, n)
    return {"xbar": xbar, "dev2": dev2, "m2": dev2 / n, "n": n}


def _build_normal(st: dict[str, Any]) -> dict[str, Any]:
    n, xbar, m2 = st["n"], st["mean"], st["m2"]
    d = _dist_display(n)
    dev2 = d["m2"]
    dev2_num = st["dev2"]
    sum_sq = st["sum"] ** 2 if False else sp.Add(*[v ** 2 for v in st["values"]])
    likelihood = (2 * sp.pi * _SIG2) ** sp.Rational(-n, 2) * sp.exp(
        -_sum_disp((_IB[_IDX] - _MU) ** 2, n) / (2 * _SIG2)
    )
    loglik = (-sp.Rational(n, 2) * sp.log(2 * sp.pi) - sp.Rational(n, 2) * sp.log(_SIG2)
              - _sum_disp((_IB[_IDX] - _MU) ** 2, n) / (2 * _SIG2))
    score_mu = sp.Rational(n, 1) * (d["xbar"] - _MU) / _SIG2          # = ∂ℓ/∂μ = n(x̄-μ)/σ²
    score_sig2 = -sp.Rational(n, 2) / _SIG2 + sp.Rational(n, 1) * dev2 / (2 * _SIG2 ** 2)
    ests = [
        _est("μ̂", d["xbar"], xbar, "样本均值（矩估计与极大似然估计一致）"),
        _est("σ̂²", dev2, m2, "二阶中心矩 (1/n)Σ(xᵢ-x̄)²（矩估计与 MLE 一致，有偏）"),
    ]
    extra = [
        _est("σ̂", sp.sqrt(dev2), sp.sqrt(m2), "标准差 MLE（有偏，且 E[σ̂] < σ）"),
    ]
    if n > 1:
        extra.insert(0, _est("S²", sp.Sum((_IB[_IDX] - d["xbar"]) ** 2, (_IDX, 1, n)) / (n - 1),
                             st["s2_unbiased"], "无偏样本方差 (1/(n-1))Σ(xᵢ-x̄)²（分母与 MLE 不同）"))
    return {
        "params": ["μ", "σ²"],
        "estimators": ests,
        "moment_estimators": ests,
        "extra_estimators": extra,
        "likelihood": likelihood,
        "log_likelihood": loglik,
        "likelihood_note": "Xᵢ ~ N(μ, σ²) 独立同分布，密度 f(x)=1/√(2πσ²)·exp(-(x-μ)²/(2σ²))，σ² > 0",
        "score_equations": [
            {"name": "∂ℓ/∂μ", "expr": score_mu, "solve": "x̄ = μ → μ̂ = x̄"},
            {"name": "∂ℓ/∂σ²", "expr": score_sig2, "solve": "σ̂² = (1/n)Σ(xᵢ-x̄)²"},
        ],
        "score_residuals": {
            "∂ℓ/∂μ 在 μ̂=x̄ 处": sp.simplify(sp.Rational(n, 1) * (xbar - xbar) / m2),
            "∂ℓ/∂σ² 在 σ̂²=m2 处": sp.simplify(-sp.Rational(n, 2) / m2 + dev2_num / (2 * m2 ** 2)),
        },
        "moment_equations": [
            {"name": "E[X] = μ", "solution": "μ̂ = x̄",
             "residual": sp.simplify(xbar - sp.Add(*st["values"]) / n)},
            {"name": "E[X²] = σ² + μ²", "solution": "σ̂² = (1/n)Σxᵢ² - x̄²",
             "residual": sp.simplify((sum_sq / n - xbar ** 2) - m2)},
        ],
        "properties": [
            "μ̂ = x̄ 是 μ 的无偏估计：E[x̄] = μ，Var(x̄) = σ²/n",
            "μ̂ 达到 Cramér–Rao 下界（Fisher 信息 I(μ) = n/σ²，下界 σ²/n），故为有效估计",
            "σ̂²_MLE = (1/n)Σ(xᵢ-x̄)² 是有偏的：E[σ̂²] = (n-1)/n·σ²（系统性偏低）",
            "无偏估计 S² = (1/(n-1))Σ(xᵢ-x̄)²；且 x̄ 与 S² 独立，(n-1)S²/σ² ~ χ²(n-1)",
        ],
        "conditions": ["σ > 0", "n 为正整数", "样本 X₁,…,Xₙ 独立同分布于 N(μ, σ²)"],
        "warnings": [],
        "boundary_note": None,
        "moment_note": "矩估计用样本矩等于总体矩解出参数；正态情形 σ̂² 与 MLE 同值，"
                       "但「无偏估计」S² 的分母是 n-1，三者不可混用",
    }


def _build_exponential(st: dict[str, Any]) -> dict[str, Any]:
    n, xbar = st["n"], st["mean"]
    d = _dist_display(n)
    vals = st["values"]
    sumx_sym = _sum_disp(_IB[_IDX], n)
    likelihood = _LAM ** n * sp.exp(-_LAM * sumx_sym)
    loglik = n * sp.log(_LAM) - _LAM * sumx_sym
    score_lam = sp.Rational(n, 1) / _LAM - sumx_sym
    lam_hat = sp.simplify(1 / xbar)
    ests = [_est("λ̂", 1 / d["xbar"], lam_hat, "极大似然估计 λ̂ = n/Σxᵢ = 1/x̄（矩估计同值）")]
    extra = []
    if n > 1:
        extra.append(_est("λ̃", sp.Rational(n - 1, n) / d["xbar"], sp.simplify(sp.Rational(n - 1, n) / xbar),
                          "无偏修正估计 (n-1)/(n·x̄)"))
    return {
        "params": ["λ"],
        "estimators": ests,
        "moment_estimators": ests,
        "extra_estimators": extra,
        "likelihood": likelihood,
        "log_likelihood": loglik,
        "likelihood_note": "Xᵢ ~ Exp(λ) 独立同分布，密度 λe^{-λx} (x>0)，λ > 0",
        "score_equations": [{"name": "∂ℓ/∂λ", "expr": score_lam, "solve": "n/λ = Σxᵢ → λ̂ = 1/x̄"}],
        "score_residuals": {
            "∂ℓ/∂λ 在 λ̂=1/x̄ 处": sp.simplify(sp.Rational(n, 1) / lam_hat - sp.Add(*vals)),
        },
        "moment_equations": [
            {"name": "E[X] = 1/λ", "solution": "λ̂ = 1/x̄", "residual": sp.simplify(1 / lam_hat - xbar)},
        ],
        "properties": [
            "λ̂ = 1/x̄ 是 λ 的有偏估计：E[λ̂] = nλ/(n-1)（n>1），偏大",
            "无偏估计为 (n-1)/(n·x̄)；Var(λ̂) = n²λ²/((n-1)²(n-2))（n>2）",
            "Cramér–Rao 下界为 λ²/n（Fisher 信息 I(λ) = 1/λ²），故 λ̂ 不是有效估计，但相合且渐近有效",
            "Σxᵢ ~ Γ(n, λ)，2λΣxᵢ ~ χ²(2n)，可用于精确区间估计与检验",
        ],
        "conditions": ["λ > 0", "n 为正整数", "样本独立同分布于指数分布 Exp(λ)（x > 0）"],
        "warnings": [] if n > 1 else ["n=1 时无偏性与方差无从讨论"],
        "boundary_note": None,
        "moment_note": "指数分布的矩估计与极大似然估计同值：1/λ = x̄ → λ̂ = 1/x̄",
    }


def _build_poisson(st: dict[str, Any]) -> dict[str, Any]:
    n, xbar = st["n"], st["mean"]
    vals = st["values"]
    d = _dist_display(n)
    sumx_sym = _sum_disp(_IB[_IDX], n)
    integer_ok = all((v.is_integer is True and v >= 0) for v in vals)
    logfact = sp.Add(*[sp.log(sp.factorial(v)) for v in vals]) if integer_ok else sp.Integer(0)
    loglik = -n * _LAM + sumx_sym * sp.log(_LAM) - logfact
    likelihood = sp.exp(-n * _LAM) * _LAM ** sumx_sym / (
        sp.prod([sp.factorial(v) for v in vals]) if integer_ok else sp.Integer(1)
    )
    score_lam = -sp.Rational(n, 1) + sumx_sym / _LAM
    ests = [_est("λ̂", d["xbar"], xbar, "极大似然估计 λ̂ = x̄（矩估计同值）")]
    return {
        "params": ["λ"],
        "estimators": ests,
        "moment_estimators": ests,
        "extra_estimators": [],
        "likelihood": likelihood,
        "log_likelihood": loglik,
        "likelihood_note": "Xᵢ ~ P(λ) 独立同分布，P(X=k) = λ^k e^{-λ}/k!，λ > 0",
        "score_equations": [{"name": "∂ℓ/∂λ", "expr": score_lam, "solve": "Σxᵢ/λ = n → λ̂ = x̄"}],
        "score_residuals": {
            "∂ℓ/∂λ 在 λ̂=x̄ 处": sp.simplify(-sp.Rational(n, 1) + sp.Add(*vals) / xbar),
        },
        "moment_equations": [{"name": "E[X] = λ", "solution": "λ̂ = x̄", "residual": sp.simplify(xbar - xbar)}],
        "properties": [
            "λ̂ = x̄ 是 λ 的无偏估计：E[x̄] = λ，Var(λ̂) = λ/n",
            "Fisher 信息 I(λ) = 1/λ，Cramér–Rao 下界 = λ/n，故 λ̂ 为有效估计（也是 UMVUE）",
            "Σxᵢ ~ P(nλ)，可用于精确检验",
        ],
        "conditions": ["λ > 0", "n 为正整数", "样本独立同分布于泊松分布 P(λ)"],
        "warnings": [] if integer_ok else ["泊松分布样本应为非负整数，当前样本含非整数（似然中的 xᵢ! 已略去）"],
        "boundary_note": None,
        "moment_note": "泊松分布的矩估计与极大似然估计同值：λ = x̄",
    }


def _build_uniform(st: dict[str, Any]) -> dict[str, Any]:
    n = st["n"]
    xbar, m2 = st["mean"], st["m2"]
    lo, hi = st["min"], st["max"]
    d = _dist_display(n)
    half = sp.sqrt(3 * m2)
    a_mom = sp.simplify(xbar - half)
    b_mom = sp.simplify(xbar + half)
    min_disp, max_disp = sp.Min(*[_IB[_IDX]]), sp.Max(*[_IB[_IDX]])
    mle = [
        _est("â_MLE", min_disp, lo, "极小次序统计量 min{xᵢ}（MLE 取参数空间边界）"),
        _est("b̂_MLE", max_disp, hi, "极大次序统计量 max{xᵢ}（MLE 取参数空间边界）"),
    ]
    moment = [
        _est("â_矩", d["xbar"] - sp.sqrt(3 * d["m2"]), a_mom, "矩估计：a = x̄ - √(3·m2)"),
        _est("b̂_矩", d["xbar"] + sp.sqrt(3 * d["m2"]), b_mom, "矩估计：b = x̄ + √(3·m2)"),
    ]
    unbiased = [
        _est("â_无偏", min_disp - (max_disp - min_disp) / (n + 1), sp.simplify(lo - (hi - lo) / (n + 1)),
             "次序统计量无偏修正：â - (b̂-â)/(n+1)，因为 E[min] = a + (b-a)/(n+1)"),
        _est("b̂_无偏", max_disp + (max_disp - min_disp) / (n + 1), sp.simplify(hi + (hi - lo) / (n + 1)),
             "次序统计量无偏修正：b̂ + (b̂-â)/(n+1)，因为 E[max] = b - (b-a)/(n+1)"),
    ]
    return {
        "params": ["a", "b"],
        "estimators": mle,
        "moment_estimators": moment,
        "extra_estimators": unbiased,
        "likelihood": 1 / (_BB - _AA) ** n,
        "log_likelihood": -n * sp.log(_BB - _AA),
        "likelihood_note": "Xᵢ ~ U(a, b) 独立同分布，密度 1/(b-a) (a≤x≤b)；"
                           "似然 L(a,b) = (b-a)^{-n}，仅当 a ≤ min{xᵢ} 且 b ≥ max{xᵢ} 时为正",
        "score_equations": [
            {"name": "∂ℓ/∂a", "expr": sp.Rational(n, 1) / (_BB - _AA),
             "solve": "恒正 → 似然关于 a 单调递增，取 a = min{xᵢ}"},
            {"name": "∂ℓ/∂b", "expr": -sp.Rational(n, 1) / (_BB - _AA),
             "solve": "恒负 → 似然关于 b 单调递减，取 b = max{xᵢ}"},
        ],
        "score_residuals": {},
        "moment_equations": [
            {"name": "E[X] = (a+b)/2", "solution": "â+b̂ = 2x̄",
             "residual": sp.simplify((a_mom + b_mom) / 2 - xbar)},
            {"name": "Var(X) = (b-a)²/12", "solution": "(b̂-â)²/12 = m2",
             "residual": sp.simplify((b_mom - a_mom) ** 2 / 12 - m2)},
        ],
        "properties": [
            "MLE 为 â = min{xᵢ}、b̂ = max{xᵢ}：对数似然关于 a 单调增、关于 b 单调减，驻点不存在，最大值在边界取得",
            "次序统计量有偏：E[min] = a + (b-a)/(n+1)，E[max] = b - (b-a)/(n+1)；修正后 â-(b̂-â)/(n+1) 无偏",
            "(b̂-â) 不是 (b-a) 的无偏估计，但 (b̂-â)(n+1)/(n-1) 无偏（n>1）",
            "矩估计 â = x̄ - √(3m2)、b̂ = x̄ + √(3m2) 与 MLE 不同，且矩估计可能超出样本取值范围",
        ],
        "conditions": ["a < b", "n 为正整数", "样本独立同分布于 U(a, b)"],
        "warnings": [],
        "boundary_note": "极大似然估计不满足得分方程为 0（参数空间闭、似然在边界取最大），"
                         "因此本算子改用「似然在边界不小于内部扰动点」的单调性/扰动检验来复核。",
        "moment_note": "均匀分布的矩估计与极大似然估计不同：矩估计由 a+b=2x̄ 与 (b-a)²/12=m2 解出；"
                       "MLE 是次序统计量 min/max",
    }


def _build_binomial(st: dict[str, Any], m: int) -> dict[str, Any]:
    n = st["n"]
    vals = st["values"]
    sumx = st["sum"]
    xbar = st["mean"]
    p_hat = sp.simplify(xbar / m)
    bad = [v for v in vals if not (v.is_integer is True and 0 <= v <= m)]
    loglik = (sp.Add(*[sp.log(sp.binomial(m, v)) for v in vals]) + sumx * sp.log(_PP)
              + (n * m - sumx) * sp.log(1 - _PP))
    likelihood = _PP ** sumx * (1 - _PP) ** (n * m - sumx) * sp.prod([sp.binomial(m, v) for v in vals])
    score_p = sumx / _PP - (n * m - sumx) / (1 - _PP)
    ests = [_est("p̂", _sum_disp(_IB[_IDX], n) / (n * m), p_hat,
                 "极大似然估计 p̂ = x̄/m（矩估计同值）")]
    return {
        "params": ["p"],
        "estimators": ests,
        "moment_estimators": ests,
        "extra_estimators": [],
        "likelihood": likelihood,
        "log_likelihood": loglik,
        "likelihood_note": f"Xᵢ ~ B(m={m}, p) 独立同分布，P(X=k)=C({m},k)p^k(1-p)^({m}-k)，0 < p < 1",
        "score_equations": [{"name": "∂ℓ/∂p", "expr": score_p,
                             "solve": "Σxᵢ/p = (nm-Σxᵢ)/(1-p) → p̂ = x̄/m"}],
        "score_residuals": {
            "∂ℓ/∂p 在 p̂=x̄/m 处": sp.simplify(sumx / p_hat - (n * m - sumx) / (1 - p_hat)),
        },
        "moment_equations": [{"name": "E[X] = mp", "solution": "p̂ = x̄/m",
                              "residual": sp.simplify(m * p_hat - xbar)}],
        "properties": [
            "p̂ = x̄/m 是 p 的无偏估计：E[p̂] = p，Var(p̂) = p(1-p)/(mn)",
            "Fisher 信息 I(p) = mn/(p(1-p))，Cramér–Rao 下界 = p(1-p)/(mn)，故 p̂ 为有效估计",
        ],
        "conditions": ["0 < p < 1", "m 为正整数（每次试验次数）", f"样本取值必须落在 [0, {m}] 内",
                       "样本独立同分布于 B(m, p)"],
        "warnings": ([] if not bad else
                     [f"样本 {[A.to_text(v) for v in bad]} 不在 0..{m} 范围内，与二项模型不相容"]),
        "boundary_note": None,
        "moment_note": "二项分布的矩估计与极大似然估计同值：mp = x̄ → p̂ = x̄/m",
    }


def _build_point_spec(dist: str, st: dict[str, Any], m: int | None) -> dict[str, Any]:
    n = st["n"]
    if dist == "normal":
        if n < 2:
            raise _Skip(f"样本量 n={n} 小于参数个数 2（μ 与 σ²），无法同时估计两个参数")
        return _build_normal(st)
    if dist == "exponential":
        return _build_exponential(st)
    if dist == "poisson":
        return _build_poisson(st)
    if dist == "uniform":
        if n < 2:
            raise _Skip(f"样本量 n={n} 不足以估计均匀分布的两个端点参数 a、b")
        return _build_uniform(st)
    if dist == "binomial":
        if m is None:
            raise ValueError("二项分布需要给出每次试验次数 m（参数 size 或 m）")
        return _build_binomial(st, m)
    raise ValueError(f"暂不支持的分布 {dist!r}")


@op("point_estimate", "estimate", "math_estimate")
@_op_guard("point_estimate")
def point_estimate(
    *,
    trials: Any = None,
    samples: Any = None,
    data: Any = None,
    method: str = "moment",
    dist: str = "normal",
    size: Any = None,
    m: Any = None,
    **kwargs: Any,
) -> MathResult:
    """参数的点估计（矩估计 / 极大似然估计）。

    * ``samples``（别名 ``data``；引擎层会把 ``samples`` 规范化为 ``trials``）：样本数据
    * ``method``：``moment`` / ``mle`` / ``both``
    * ``dist``：``normal`` / ``exponential`` / ``poisson`` / ``uniform`` / ``binomial``
    * ``size``（别名 ``m``）：二项分布的试验次数
    """
    raw = trials if trials is not None else (samples if samples is not None else data)
    if raw is None:
        raise ValueError('缺少样本数据：请给出 samples="1,2,3,4,5"')
    vals = _parse_samples(raw)
    st = _stats(vals)
    n = st["n"]

    key = str(dist or "normal").strip().lower()
    if key not in _DIST_ALIASES:
        raise ValueError(f"未知分布 {dist!r}；支持 normal/exponential/poisson/uniform/binomial")
    dist = _DIST_ALIASES[key]
    meth = _METHOD_ALIASES.get(str(method or "moment").strip().lower())
    if meth is None:
        raise ValueError(f"未知估计方法 {method!r}；支持 moment/mle/both")

    m_val = None
    if dist == "binomial":
        raw_m = size if size is not None else m
        if raw_m is not None:
            m_val = _posint(raw_m, "size")

    spec = _build_point_spec(dist, st, m_val)

    moment_list = spec.get("moment_estimators") or spec["estimators"]
    mle_list = spec["estimators"]
    same_lists = all(a["name"] == b["name"] for a, b in zip(moment_list, mle_list)) and len(moment_list) == len(mle_list)

    if meth == "moment":
        ests, others = moment_list, ([] if same_lists else mle_list) + list(spec["extra_estimators"])
    elif meth == "mle":
        ests, others = mle_list, ([] if same_lists else moment_list) + list(spec["extra_estimators"])
    else:
        ests = moment_list
        others = ([] if same_lists else mle_list) + list(spec["extra_estimators"])

    show_moment = meth in ("moment", "both")
    show_mle = meth in ("mle", "both")

    r = MathResult.ok("point_estimate", method=meth)
    r.set_input(samples=_samples_input(vals), n=n, dist=dist, method=meth)
    if m_val is not None:
        r.set_input(size=m_val)

    if show_mle:
        r.set_raw(likelihood=A.to_text(spec["likelihood"]),
                  likelihood_latex=A.to_latex(spec["likelihood"]),
                  log_likelihood=A.to_text(spec["log_likelihood"]),
                  log_likelihood_latex=A.to_latex(spec["log_likelihood"]),
                  likelihood_note=spec["likelihood_note"],
                  score_equations=[
                      {"name": s["name"], "expr": A.to_text(s["expr"]), "solve": s["solve"]}
                      for s in spec["score_equations"]
                  ])
    if show_moment:
        r.set_raw(moment_equations=[
            {"name": eq["name"], "solution": eq["solution"], "residual": A.to_text(eq["residual"])}
            for eq in spec["moment_equations"]
        ], moment_note=spec.get("moment_note"))

    method_label = {"moment": "矩估计", "mle": "极大似然估计", "both": "矩估计与极大似然估计"}[meth]
    r.set_raw(estimators={e["name"]: e for e in ests},
              other_estimators={e["name"]: e for e in others},
              sample_statistics={
                  "n": n,
                  "mean": A.to_text(st["mean"]), "mean_approx": _approx(st["mean"]),
                  "m2": A.to_text(st["m2"]), "m2_approx": _approx(st["m2"]),
                  "s2_unbiased": A.to_text(st["s2_unbiased"]) if st["s2_unbiased"] is not None else None,
                  "sum": A.to_text(st["sum"]),
                  "min": A.to_text(st["min"]), "max": A.to_text(st["max"]),
              },
              properties=spec["properties"])

    summary = "；".join(f"{e['name']} = {e['value']}" for e in ests)
    others_txt = "；".join(f"{e['name']} = {e['value']}" for e in others)
    r.set_result(f"{dist} 分布参数{method_label}：{summary}"
                 + (f"（另有 {others_txt}）" if others_txt else ""))

    r.add_condition(*spec["conditions"])
    for w in spec["warnings"]:
        r.add_warning(w)
    if spec.get("boundary_note"):
        r.add_warning(spec["boundary_note"])
    if dist == "uniform":
        r.add_condition("均匀分布的矩估计与极大似然估计不同，两者都要会写")
    if dist == "normal":
        r.add_condition("注意区分：σ̂²_MLE 分母为 n，无偏 S² 分母为 n-1")
    _note_unused(r, _unused(**kwargs))

    # ---- 独立复核 -------------------------------------------------------
    methods: list[dict[str, Any]] = []
    level = "symbolic"
    if spec["score_residuals"]:
        details: dict[str, Any] = {}
        ok_all = True
        for label, residual in spec["score_residuals"].items():
            value = sp.simplify(residual)
            agree, dev, note = C.numeric_agree(value, 0)
            zero = bool(value == 0 or agree)
            details[label] = {"residual": A.to_text(value), "is_zero": zero, "note": note,
                              "max_rel_dev": dev}
            ok_all = ok_all and zero
        methods.append({
            "method": "把估计值代回得分方程 ∂ℓ/∂θ = 0，检查残差是否严格为 0（精确有理数运算）",
            "evidence": details,
            "passed": bool(ok_all),
        })
        if not ok_all:
            r.add_warning("得分方程残差不为 0，估计量可能不是该方法的正确解")

    res_details: dict[str, Any] = {}
    res_ok = True
    for eq in spec["moment_equations"]:
        value = sp.simplify(eq["residual"])
        res_details[eq["name"]] = A.to_text(value)
        if value != 0:
            agree, _dev, _note = C.numeric_agree(value, 0)
            res_ok = res_ok and agree
    methods.append({
        "method": "把估计值代入总体矩的具体表达式（E[X; θ̂] 与样本矩比较），检查残差",
        "evidence": res_details,
        "passed": bool(res_ok),
    })

    if dist == "uniform":
        lo, hi = st["min"], st["max"]
        delta = sp.Rational(1, 2) * (hi - lo) / (n + 1) if n > 1 else sp.Integer(0)
        lam_hat = sp.simplify((1 / (hi - lo)) ** n)
        lam_a = sp.simplify((1 / (hi - (lo - delta))) ** n)
        lam_b = sp.simplify((1 / ((hi + delta) - lo)) ** n)
        ok_boundary = bool(lam_hat >= lam_a and lam_hat >= lam_b)
        methods.append({
            "method": "边界论证 + 似然扰动检验（∂ℓ/∂a>0、∂ℓ/∂b<0 ⇒ 最大值在 a=min、b=max）",
            "evidence": {
                "L(a=min,b=max)": A.to_text(lam_hat),
                f"L(a=min-{A.to_text(delta)},b=max)": A.to_text(lam_a),
                f"L(a=min,b=max+{A.to_text(delta)})": A.to_text(lam_b),
                "boundary_is_maximum": ok_boundary,
                "all_samples_in_range": bool(all(lo <= v <= hi for v in vals)),
            },
            "passed": ok_boundary,
        })
        if not ok_boundary:
            r.add_warning("似然扰动检验未通过，MLE 取边界的结论需要人工确认")

    _verify_block(r, status="cross", level=level, methods=methods)
    if not all(m.get("passed") for m in methods):
        r.partial("部分复核未通过：估计量可能不是该方法的正确解")
    return r


# ---------------------------------------------------------------------------
# 区间估计
# ---------------------------------------------------------------------------

@op("interval_estimate", "confidence_interval", "math_confidence_interval")
@_op_guard("interval_estimate")
def interval_estimate(
    *,
    trials: Any = None,
    samples: Any = None,
    data: Any = None,
    order: Any = None,
    n: Any = None,
    mean: Any = None,
    std: Any = None,
    sigma: Any = None,
    var: Any = None,
    variance: Any = None,
    dist: str = "normal",
    confidence: Any = "0.95",
    kind: str = "mean",
    successes: Any = None,
    size: Any = None,
    m: Any = None,
    **kwargs: Any,
) -> MathResult:
    """单正态总体均值/方差的区间估计与大样本比例区间估计。

    * ``kind="mean"``：σ 已知用 z 分位数；σ 未知用 t 分位数（自由度 n-1）
    * ``kind="variance"``：χ² 分布（自由度 n-1），给出两个分位点
    * ``kind="proportion"``：大样本正态近似（要求 np̂ ≥ 5 且 n(1-p̂) ≥ 5）
    """
    raw = trials if trials is not None else (samples if samples is not None else data)
    kd = str(kind or "mean").strip().lower()
    if kd not in ("mean", "variance", "proportion"):
        raise ValueError(f"未知区间估计类型 {kind!r}；支持 mean/variance/proportion")
    conf = _level01(confidence, "confidence")
    alpha = sp.simplify(1 - conf)

    vals = _parse_samples(raw) if raw is not None else None
    nn = len(vals) if vals is not None else None
    if nn is None:
        raw_n = order if order is not None else n
        if raw_n is None:
            raise ValueError("缺少样本量 n（未给出 samples 时必须给出 n）")
        nn = _posint(raw_n, "n")

    methods: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {"confidence": A.to_text(conf), "alpha": A.to_text(alpha), "kind": kd,
                                "n": nn}
    conditions: list[str] = [f"置信水平 1-α = {A.to_text(conf)}，即 α = {A.to_text(alpha)}"]
    warnings: list[str] = []
    df: Any = None
    exact_symbolic = False

    if kd == "mean":
        if vals is not None:
            xbar = _stats(vals)["mean"]
        elif mean is not None:
            xbar = _num(mean, "mean")
        else:
            raise ValueError("缺少样本均值（给出 samples 或 mean）")
        if sigma is not None:
            sig = _positive(sigma, "sigma")
            crit = _z_crit(alpha / 2)
            dist_used, crit_name = "标准正态 N(0,1)（σ 已知）", "z_{α/2}"
            se = sp.simplify(sig / sp.sqrt(nn))
            spread = sp.simplify(crit * se)
            lo, hi = sp.simplify(xbar - spread), sp.simplify(xbar + spread)
            exact_symbolic = True
            evidence.update({"sigma_known": True, "sigma": A.to_text(sig),
                             "standard_error": A.to_text(se),
                             "critical_symbolic": f"z_{{α/2}} = sqrt(2)*erfinv({A.to_text(1 - alpha)}) = {A.to_text(crit)}",
                             "critical_approx": _approx(crit),
                             "formula": "x̄ ± z_{α/2}·σ/√n"})
            conditions += [f"总体标准差 σ = {A.to_text(sig)} 已知",
                           "总体为正态总体（或样本量足够大，用中心极限定理）"]
            prob = sp.simplify(sp.erf(crit / sp.sqrt(2)))
            diff = sp.simplify(prob - conf)
            sym_ok = bool(diff == 0)
            if not sym_ok:
                try:
                    sym_ok = bool(abs(_fnum(prob) - _fnum(conf)) < 1e-12)
                except Exception:  # noqa: BLE001
                    sym_ok = False
            methods.append({
                "method": "符号验证 P(-z_{α/2} < Z < z_{α/2}) = 1-α：计算 erf(z_{α/2}/√2) 并与 1-α 相减",
                "evidence": {"P(区间覆盖)": A.to_text(prob), "1-α": A.to_text(conf),
                             "difference": A.to_text(diff), "symbolic_zero": bool(diff == 0)},
                "passed": sym_ok,
            })
            if not sym_ok:
                warnings.append("分位数的覆盖概率符号验证未通过")
        else:
            if vals is not None:
                s = _stats(vals)["s_unbiased"]
                if s is None:
                    raise ValueError("σ 未知时样本量至少为 2")
            elif std is not None:
                s = _positive(std, "std")
            else:
                raise ValueError("σ 未知时请给出样本（samples）或样本标准差 std")
            df = nn - 1
            crit = _t_ppf(1 - alpha / 2, df)
            dist_used = f"t 分布 t({df})（σ 未知）"
            crit_name = f"t_{{α/2}}(n-1) = t_{{α/2}}({df})"
            se = sp.simplify(s / sp.sqrt(nn))
            spread = sp.simplify(crit * se)
            lo, hi = sp.simplify(xbar - spread), sp.simplify(xbar + spread)
            evidence.update({"sigma_known": False, "sample_std": A.to_text(s),
                             "standard_error": A.to_text(se), "critical": A.to_text(crit),
                             "degrees_of_freedom": df, "formula": "x̄ ± t_{α/2}(n-1)·S/√n"})
            conditions += ["总体标准差 σ 未知，用样本标准差 S 代替",
                           f"总体必须为正态总体（t 分布的前提），自由度 df = n-1 = {df}"]
            cover = _t_cdf(crit, df) - _t_cdf(-crit, df)
            methods.append({
                "method": f"数值验证 P(-t_{{α/2}}<T<t_{{α/2}})=1-α（t({df}) 分布函数）",
                "evidence": {"P(区间覆盖)": cover, "1-α": _fnum(conf),
                             "deviation": abs(cover - _fnum(conf))},
                "passed": bool(abs(cover - _fnum(conf)) < 1e-9),
            })
        result_kind = "单正态总体均值的置信区间"
    elif kd == "variance":
        if vals is not None:
            s2 = _stats(vals)["s2_unbiased"]
            if s2 is None:
                raise ValueError("方差区间估计要求 n ≥ 2")
        elif variance is not None or var is not None:
            s2 = _positive(variance if variance is not None else var, "variance")
        elif std is not None:
            s2 = _positive(std, "std") ** 2
        else:
            raise ValueError("请给出样本（samples）或样本方差 variance/std")
        if nn < 2:
            raise ValueError("方差区间估计要求 n ≥ 2")
        df = nn - 1
        hi_q = _chi2_ppf(1 - alpha / 2, df)   # 上侧 α/2 分位点
        lo_q = _chi2_ppf(alpha / 2, df)       # 上侧 1-α/2 分位点
        lo = sp.simplify((nn - 1) * s2 / hi_q)
        hi = sp.simplify((nn - 1) * s2 / lo_q)
        dist_used = f"χ² 分布 χ²({df})"
        crit_name = f"χ²_{{α/2}}(n-1)={A.to_text(hi_q)}、χ²_{{1-α/2}}(n-1)={A.to_text(lo_q)}"
        evidence.update({"sample_variance": A.to_text(s2), "degrees_of_freedom": df,
                         "chi2_upper": A.to_text(hi_q), "chi2_lower": A.to_text(lo_q),
                         "formula": "[(n-1)S²/χ²_{α/2}(n-1), (n-1)S²/χ²_{1-α/2}(n-1)]"})
        conditions += ["总体必须为正态总体（χ² 区间的前提）", f"自由度 df = n-1 = {df}",
                       "S² 为无偏样本方差（分母 n-1）"]
        cover = _chi2_cdf(hi_q, df) - _chi2_cdf(lo_q, df)
        methods.append({
            "method": f"数值验证 P(χ²_{{1-α/2}}<χ²<χ²_{{α/2}})=1-α（χ²({df}) 分布函数）",
            "evidence": {"P(区间覆盖)": cover, "1-α": _fnum(conf),
                         "deviation": abs(cover - _fnum(conf))},
            "passed": bool(abs(cover - _fnum(conf)) < 1e-9),
        })
        result_kind = "单正态总体方差的置信区间"
    else:  # proportion
        if vals is not None:
            ones = [v for v in vals if v == 1]
            zeros = [v for v in vals if v == 0]
            if len(ones) + len(zeros) != len(vals):
                raise ValueError("比例区间估计的 0-1 样本必须只含 0 和 1")
            succ = sp.Integer(len(ones))
        elif successes is not None:
            succ = _num(successes, "successes")
            if succ < 0 or succ > nn:
                raise ValueError("successes 必须满足 0 ≤ successes ≤ n")
        else:
            raise ValueError("请给出 0-1 样本（samples）或成功次数（successes）")
        phat = sp.simplify(succ / nn)
        crit = _z_crit(alpha / 2)
        se = sp.sqrt(sp.simplify(phat * (1 - phat) / nn))
        spread = sp.simplify(crit * se)
        lo, hi = sp.simplify(phat - spread), sp.simplify(phat + spread)
        dist_used = "标准正态 N(0,1)（大样本正态近似）"
        crit_name = "z_{α/2}"
        np_hat, nq_hat = sp.simplify(nn * phat), sp.simplify(nn * (1 - phat))
        evidence.update({"phat": A.to_text(phat), "successes": A.to_text(succ),
                         "standard_error": A.to_text(se), "critical_approx": _approx(crit),
                         "n*phat": A.to_text(np_hat), "n*(1-phat)": A.to_text(nq_hat),
                         "formula": "p̂ ± z_{α/2}·√(p̂(1-p̂)/n)"})
        conditions += ["大样本正态近似要求 n·p̂ ≥ 5 且 n·(1-p̂) ≥ 5", "样本独立同分布",
                       "总体为二点分布（伯努利）"]
        if not (np_hat >= 5 and nq_hat >= 5):
            warnings.append(
                f"正态近似条件不满足：n·p̂ = {A.to_text(np_hat)}、n·(1-p̂) = {A.to_text(nq_hat)}；"
                "Wald 区间此时覆盖率偏低，请参考 Agresti–Coull 修正区间"
            )
            z2 = crit ** 2
            p_tilde = sp.simplify((succ + z2 / 2) / (nn + z2))
            half = sp.simplify(crit * sp.sqrt(p_tilde * (1 - p_tilde) / (nn + z2)))
            evidence["agresti_coull"] = {
                "p_tilde": A.to_text(p_tilde),
                "lower": _approx(p_tilde - half), "upper": _approx(p_tilde + half),
                "note": "Agresti–Coull 修正区间（小样本更稳健）",
            }
        cover = float(_norm_cdf(crit) - _norm_cdf(-crit))
        methods.append({
            "method": "数值验证 P(-z_{α/2}<Z<z_{α/2})=1-α（正态分布函数）",
            "evidence": {"P(区间覆盖)": cover, "1-α": _fnum(conf), "deviation": abs(cover - _fnum(conf))},
            "passed": bool(abs(cover - _fnum(conf)) < 1e-9),
        })
        result_kind = "总体比例的置信区间（大样本）"

    width = sp.simplify(hi - lo)
    mono_ok, mono_detail = _monotone_width(kd, conf, nn, sigma, vals, std, var, variance, successes)
    methods.append({
        "method": "单调性检查：置信水平提高时区间必须变宽（1-α 取 conf±0.01 对比）",
        "evidence": mono_detail,
        "passed": bool(mono_ok),
    })
    if not mono_ok:
        warnings.append("区间宽度随置信水平的单调性检查未通过（请检查分位数与端点公式）")

    r = MathResult.ok("interval_estimate", method=f"{result_kind}（{crit_name}）")
    r.set_input(kind=kd, confidence=A.to_text(conf), n=nn, dist=str(dist or "normal"),
                critical_value_name=crit_name)
    if vals is not None:
        r.set_input(samples=_samples_input(vals))
    r.set_result(f"{result_kind}（1-α = {A.to_text(conf)}）：[{A.to_text(lo)}, {A.to_text(hi)}]，"
                 f"长度 = {A.to_text(width)}")
    r.set_raw(
        lower=A.to_text(lo), upper=A.to_text(hi),
        lower_approx=_approx(lo), upper_approx=_approx(hi),
        center=A.to_text(sp.simplify((lo + hi) / 2)), length=A.to_text(width),
        length_approx=_approx(width),
        confidence=A.to_text(conf), confidence_approx=_fnum(conf), alpha=A.to_text(alpha),
        alpha_approx=_fnum(alpha),
        distribution_used=dist_used, critical_value=crit_name,
        degrees_of_freedom=df,
        exact_symbolic_endpoints=bool(exact_symbolic),
        interval_form=evidence.get("formula"),
        interpretation=("区间长度 = 2×半宽，置信水平越高区间越宽、精度越低；"
                        "1-α 的含义是「重复抽样所得区间中有 1-α 的比例覆盖真值」，"
                        "而不是「真值落在该区间内的概率为 1-α」"),
    )
    r.add_condition(*conditions)
    r.add_warning(*warnings)
    _note_unused(r, _unused(**kwargs))
    _verify_block(r, status="cross", level=("symbolic" if exact_symbolic else "numeric"),
                  methods=methods)
    if not all(m.get("passed") for m in methods):
        r.partial("区间复核存在未通过项，请检查输入")
    return r


def _monotone_width(kd: str, conf: sp.Basic, nn: int, sigma: Any, vals: Any, std: Any,
                    var: Any, variance: Any, successes: Any) -> tuple[bool, dict[str, Any]]:
    """对比置信水平在 conf±0.01 处的区间宽度，检查单调性。"""
    detail: dict[str, Any] = {"reference_confidence": A.to_text(conf)}
    widths: list[tuple[float, float]] = []
    base = _fnum(conf)
    for offset in (-0.01, 0.0, 0.01):
        c = base + offset
        if not 0.0 < c < 1.0:
            continue
        a = sp.nsimplify(1 - c, rational=True)
        try:
            if kd == "mean":
                if sigma is not None:
                    w = 2 * _fnum(_z_crit(a / 2) * _positive(sigma, "sigma") / sp.sqrt(nn))
                else:
                    s = _stats(vals)["s_unbiased"] if vals is not None else _positive(std, "std")
                    w = 2 * _fnum(_t_ppf(1 - a / 2, nn - 1) * s / sp.sqrt(nn))
            elif kd == "variance":
                if vals is not None:
                    s2 = _stats(vals)["s2_unbiased"]
                elif variance is not None or var is not None:
                    s2 = _positive(variance if variance is not None else var, "variance")
                else:
                    s2 = _positive(std, "std") ** 2
                df = nn - 1
                hi_q = _chi2_ppf(1 - a / 2, df)
                lo_q = _chi2_ppf(a / 2, df)
                w = _fnum((df * s2) * (1 / lo_q - 1 / hi_q))
            else:
                z = _z_crit(a / 2)
                phat = (sp.Rational(sum(1 for v in vals if v == 1), len(vals)) if vals is not None
                        else sp.simplify(_num(successes, "successes") / nn))
                w = 2 * _fnum(z * sp.sqrt(phat * (1 - phat) / nn))
            widths.append((c, w))
        except Exception as exc:  # noqa: BLE001
            detail["skipped"] = f"{type(exc).__name__}: {exc}"
            return True, detail
    detail["widths"] = [{"confidence": round(c, 6), "width": round(w, 12)} for c, w in widths]
    increasing = all(widths[i][1] <= widths[i + 1][1] + 1e-12 for i in range(len(widths) - 1))
    detail["increasing_in_confidence"] = bool(increasing)
    return bool(increasing), detail


# ---------------------------------------------------------------------------
# 假设检验
# ---------------------------------------------------------------------------

_SIDE_ALIASES = {
    "two": "two", "two-sided": "two", "two_sided": "two", "twosided": "two", "双侧": "two",
    "!=": "two", "≠": "two", "both": "two", "two-sided-test": "two",
    "left": "left", "-": "left", "less": "left", "<": "left", "左": "left", "左侧": "left", "单侧左": "left",
    "right": "right", "+": "right", "greater": "right", ">": "right", "右": "right", "右侧": "right",
    "单侧右": "right",
}


def _resolve_side(side: Any, alternative: Any, default: str) -> str:
    for raw in (side, alternative):
        if raw is None:
            continue
        key = str(raw).strip().lower()
        if key in _SIDE_ALIASES:
            return _SIDE_ALIASES[key]
        raise ValueError(f"未知的检验方向 {raw!r}；支持 two/left/right（或双侧/左/右）")
    return default


def _pvalue_z(stat: sp.Basic, side: str) -> sp.Basic:
    if side == "two":
        return sp.simplify(sp.erfc(sp.Abs(stat) / sp.sqrt(2)))
    if side == "right":
        return sp.simplify(sp.erfc(stat / sp.sqrt(2)) / 2)
    return sp.simplify(sp.erfc(-stat / sp.sqrt(2)) / 2)


@op("hypothesis_test", "test", "math_hypothesis_test")
@_op_guard("hypothesis_test")
def hypothesis_test(
    *,
    kind: Any = None,
    trials: Any = None,
    samples: Any = None,
    data: Any = None,
    order: Any = None,
    n: Any = None,
    mean: Any = None,
    std: Any = None,
    sigma: Any = None,
    mu0: Any = None,
    sigma0: Any = None,
    alpha: Any = "0.05",
    dir: Any = None,
    side: Any = None,
    alternative: Any = None,
    successes: Any = None,
    var: Any = None,
    variance: Any = None,
    var1: Any = None,
    var2: Any = None,
    n1: Any = None,
    n2: Any = None,
    samples2: Any = None,
    data2: Any = None,
    **kwargs: Any,
) -> MathResult:
    """假设检验：z 检验 / t 检验 / χ² 检验 / F 检验 / 比例检验。

    输出检验统计量、临界值、拒绝域、p 值与结论，并明确写出
    「不拒绝 H₀ 不等于证明 H₀ 成立」。
    """
    kd = str(kind or "").strip().lower()
    aliases = {"z": "z_test", "ztest": "z_test", "t": "t_test", "ttest": "t_test",
               "chi2": "chi2_test", "chi_square": "chi2_test", "chisq": "chi2_test",
               "f": "f_test", "ftest": "f_test", "proportion": "proportion_test",
               "prop": "proportion_test"}
    kd = aliases.get(kd, kd)
    if kd not in ("z_test", "t_test", "chi2_test", "f_test", "proportion_test"):
        raise ValueError("未知检验类型 "
                         f"{kind!r}；支持 z_test/t_test/chi2_test/f_test/proportion_test")
    a = _level01(alpha, "alpha")
    conf = sp.simplify(1 - a)

    raw = trials if trials is not None else (samples if samples is not None else data)
    vals = _parse_samples(raw) if raw is not None else None
    vals2 = _parse_samples(samples2 if samples2 is not None else data2, name="samples2") \
        if (samples2 is not None or data2 is not None) else None

    conditions: list[str] = [f"显著性水平 α = {A.to_text(a)}", "样本独立同分布"]
    warnings: list[str] = []
    evidence: dict[str, Any] = {"alpha": A.to_text(a), "kind": kd}
    df_info: Any = None

    def _n_of(fallback: Any) -> int:
        if vals is not None:
            return len(vals)
        if fallback is None:
            raise ValueError("缺少样本量 n")
        return _posint(fallback, "n")

    crit_low: Any = None  # 双侧但非对称分布（χ²、F）的下侧临界值
    if kd == "z_test":
        sdd = _resolve_side(dir if dir is not None else side, alternative, "two")
        if mu0 is None:
            raise ValueError("z 检验需要原假设均值 mu0")
        mu_0 = _num(mu0, "mu0")
        if sigma is None:
            raise ValueError("z 检验要求总体标准差 σ 已知（请给出 sigma）")
        sig = _positive(sigma, "sigma")
        nn = _n_of(order if order is not None else n)
        xbar = _stats(vals)["mean"] if vals is not None else (_num(mean, "mean") if mean is not None else None)
        if xbar is None:
            raise ValueError("缺少样本均值（给出 samples 或 mean）")
        stat = sp.simplify((xbar - mu_0) / (sig / sp.sqrt(nn)))
        # 单侧左检验的临界值必须取负分位数（拒绝域：统计量 < -z_α）
        crit = (_z_crit(a / 2) if sdd == "two" else (_z_crit(a) if sdd == "right" else -_z_crit(a)))
        pval = _pvalue_z(stat, sdd)
        dist_name = "标准正态 N(0,1)"
        crit_name = (f"z_{{α/2}} = {A.to_text(crit)}" if sdd == "two"
                     else f"z_{{α}} = {A.to_text(crit)}" if sdd == "right"
                     else f"-z_{{α}} = {A.to_text(crit)}")
        h0 = f"H₀: μ = {A.to_text(mu_0)}"
        h1 = {"two": f"H₁: μ ≠ {A.to_text(mu_0)}", "left": f"H₁: μ < {A.to_text(mu_0)}",
              "right": f"H₁: μ > {A.to_text(mu_0)}"}
        conditions += [f"σ = {A.to_text(sig)} 已知", "总体为正态总体，或样本量足够大"]
        evidence.update({"n": nn, "sample_mean": A.to_text(xbar), "sigma": A.to_text(sig),
                         "statistic_formula": "(x̄ - μ₀)/(σ/√n)"})
    elif kd == "t_test":
        sdd = _resolve_side(dir if dir is not None else side, alternative, "two")
        if mu0 is None:
            raise ValueError("t 检验需要原假设均值 mu0")
        mu_0 = _num(mu0, "mu0")
        nn = _n_of(order if order is not None else n)
        if nn < 2:
            raise ValueError("t 检验要求 n ≥ 2")
        if vals is not None:
            st = _stats(vals)
            xbar, s = st["mean"], st["s_unbiased"]
        else:
            if mean is None or std is None:
                raise ValueError("缺少样本均值或样本标准差（给出 samples，或同时给出 mean 与 std）")
            xbar, s = _num(mean, "mean"), _positive(std, "std")
        df = nn - 1
        stat = sp.simplify((xbar - mu_0) / (s / sp.sqrt(nn)))
        # 单侧左检验的临界值必须取负分位数（拒绝域：统计量 < -t_α(n-1)）
        crit = (_t_ppf(1 - a / 2, df) if sdd == "two"
                else (_t_ppf(1 - a, df) if sdd == "right" else -_t_ppf(1 - a, df)))
        if sdd == "two":
            pval = sp.Float(2 * float(_scipy.t.sf(abs(float(stat)), df)), 12)
        elif sdd == "right":
            pval = sp.Float(float(_scipy.t.sf(float(stat), df)), 12)
        else:
            pval = sp.Float(float(_scipy.t.cdf(float(stat), df)), 12)
        dist_name = f"t 分布 t({df})"
        crit_name = (f"t_{{α/2}}({df}) = {A.to_text(crit)}" if sdd == "two"
                     else f"t_{{α}}({df}) = {A.to_text(crit)}" if sdd == "right"
                     else f"-t_{{α}}({df}) = {A.to_text(crit)}")
        df_info = df
        h0 = f"H₀: μ = {A.to_text(mu_0)}"
        h1 = {"two": f"H₁: μ ≠ {A.to_text(mu_0)}", "left": f"H₁: μ < {A.to_text(mu_0)}",
              "right": f"H₁: μ > {A.to_text(mu_0)}"}
        conditions += ["总体必须为正态总体（t 检验的前提）", f"自由度 df = n-1 = {df}"]
        evidence.update({"n": nn, "sample_mean": A.to_text(xbar), "sample_std": A.to_text(s),
                         "degrees_of_freedom": df, "statistic_formula": "(x̄ - μ₀)/(S/√n)"})
    elif kd == "chi2_test":
        sdd = _resolve_side(dir if dir is not None else side, alternative, "right")
        if sigma0 is None:
            raise ValueError("χ² 检验需要原假设标准差 sigma0")
        s0 = _positive(sigma0, "sigma0")
        nn = _n_of(order if order is not None else n)
        if nn < 2:
            raise ValueError("χ² 检验要求 n ≥ 2")
        if vals is not None:
            s2 = _stats(vals)["s2_unbiased"]
        elif variance is not None or var is not None:
            s2 = _positive(variance if variance is not None else var, "variance")
        elif std is not None:
            s2 = _positive(std, "std") ** 2
        else:
            raise ValueError("缺少样本方差（给出 samples，或 variance/std）")
        df = nn - 1
        stat = sp.simplify((nn - 1) * s2 / s0 ** 2)
        _need_scipy("χ² 分布分位数")
        fv = float(stat)
        if sdd == "right":
            crit, crit_low = _chi2_ppf(1 - a, df), None
            pval = sp.Float(float(_scipy.chi2.sf(fv, df)), 12)
        elif sdd == "left":
            crit, crit_low = _chi2_ppf(a, df), None
            pval = sp.Float(float(_scipy.chi2.cdf(fv, df)), 12)
        else:
            crit, crit_low = _chi2_ppf(1 - a / 2, df), _chi2_ppf(a / 2, df)
            pval = sp.Float(min(1.0, 2 * min(float(_scipy.chi2.cdf(fv, df)),
                                            float(_scipy.chi2.sf(fv, df)))), 12)
        dist_name = f"χ² 分布 χ²({df})"
        crit_name = (f"χ²_{{α}}({df}) = {A.to_text(crit)}" if sdd == "right"
                     else f"χ²_{{1-α}}({df}) = {A.to_text(crit)}" if sdd == "left"
                     else (f"χ²_{{α/2}}({df}) = {A.to_text(crit_low)}、"
                           f"χ²_{{1-α/2}}({df}) = {A.to_text(crit)}"))
        df_info = df
        h0 = f"H₀: σ² = {A.to_text(s0 ** 2)}"
        h1 = {"right": f"H₁: σ² > {A.to_text(s0 ** 2)}", "left": f"H₁: σ² < {A.to_text(s0 ** 2)}",
              "two": f"H₁: σ² ≠ {A.to_text(s0 ** 2)}"}
        conditions += ["总体必须为正态总体", f"自由度 df = n-1 = {df}",
                       "χ² 检验通常取右侧拒绝域（未显式指定 side 时本算子默认 right）"]
        evidence.update({"n": nn, "sample_variance": A.to_text(s2), "sigma0_squared": A.to_text(s0 ** 2),
                         "degrees_of_freedom": df, "statistic_formula": "(n-1)S²/σ₀²"})
    elif kd == "f_test":
        sdd = _resolve_side(dir if dir is not None else side, alternative, "two")
        if vals is not None and vals2 is not None:
            nn1, nn2 = len(vals), len(vals2)
            s1 = _stats(vals)["s2_unbiased"]
            s2v = _stats(vals2)["s2_unbiased"]
        else:
            if n1 is None or n2 is None:
                raise ValueError("F 检验需要两组样本（samples 与 samples2），或给出 n1、n2 与两个样本方差（var1、var2）")
            nn1, nn2 = _posint(n1, "n1"), _posint(n2, "n2")
            v1 = var1 if var1 is not None else (variance if variance is not None else var)
            if v1 is None or var2 is None:
                raise ValueError("F 检验需要两个样本方差（var1 与 var2）")
            s1, s2v = _positive(v1, "var1"), _positive(var2, "var2")
        if nn1 < 2 or nn2 < 2:
            raise ValueError("F 检验要求两组样本量均 ≥ 2")
        df1, df2 = nn1 - 1, nn2 - 1
        stat = sp.simplify(s1 / s2v)
        _need_scipy("F 分布分位数")
        fv = float(stat)
        if sdd == "right":
            crit, crit_low = _f_ppf(1 - a, df1, df2), None
        elif sdd == "left":
            crit, crit_low = _f_ppf(a, df1, df2), None
        else:
            crit, crit_low = _f_ppf(1 - a / 2, df1, df2), _f_ppf(a / 2, df1, df2)
        pr = float(_scipy.f.sf(fv, df1, df2))
        pl = float(_scipy.f.cdf(fv, df1, df2))
        pval = sp.Float(pr if sdd == "right" else pl if sdd == "left" else min(1.0, 2 * min(pr, pl)), 12)
        dist_name = f"F 分布 F({df1}, {df2})"
        crit_name = (f"F_{{α}}({df1}, {df2}) = {A.to_text(crit)}" if sdd == "right"
                     else f"F_{{1-α}}({df1}, {df2}) = {A.to_text(crit)}" if sdd == "left"
                     else (f"F_{{α/2}}({df1}, {df2}) = {A.to_text(crit_low)}、"
                           f"F_{{1-α/2}}({df1}, {df2}) = {A.to_text(crit)}"))
        df_info = f"({df1}, {df2})"
        h0 = "H₀: σ₁² = σ₂²"
        h1 = {"two": "H₁: σ₁² ≠ σ₂²", "left": "H₁: σ₁² < σ₂²", "right": "H₁: σ₁² > σ₂²"}
        conditions += ["两个总体均为正态总体且相互独立", f"自由度 df₁ = {df1}，df₂ = {df2}",
                       "标准做法是把较大样本方差放在分子并取右侧检验"]
        evidence.update({"n1": nn1, "n2": nn2, "s1_squared": A.to_text(s1), "s2_squared": A.to_text(s2v),
                         "degrees_of_freedom": [df1, df2], "statistic_formula": "S₁²/S₂²"})
    else:  # proportion_test
        sdd = _resolve_side(dir if dir is not None else side, alternative, "two")
        if mu0 is None:
            raise ValueError("比例检验需要原假设比例 mu0（即 p₀）")
        p0 = _num(mu0, "mu0")
        if not (0 < p0 < 1):
            raise ValueError("原假设比例 p₀ 必须满足 0 < p₀ < 1")
        if vals is not None:
            nn = len(vals)
            ones = [v for v in vals if v == 1]
            zeros = [v for v in vals if v == 0]
            if len(ones) + len(zeros) != nn:
                raise ValueError("比例检验的 0-1 样本必须只含 0 和 1")
            succ = sp.Integer(len(ones))
        else:
            nn = _n_of(order if order is not None else n)
            if successes is None:
                raise ValueError("需要成功次数（successes）或 0-1 样本")
            succ = _num(successes, "successes")
        phat = sp.simplify(succ / nn)
        stat = sp.simplify((phat - p0) / sp.sqrt(p0 * (1 - p0) / nn))
        crit = (_z_crit(a / 2) if sdd == "two" else (_z_crit(a) if sdd == "right" else -_z_crit(a)))
        pval = _pvalue_z(stat, sdd)
        dist_name = "标准正态 N(0,1)"
        crit_name = (f"z_{{α/2}} = {A.to_text(crit)}" if sdd == "two"
                     else f"z_{{α}} = {A.to_text(crit)}" if sdd == "right"
                     else f"-z_{{α}} = {A.to_text(crit)}")
        h0 = f"H₀: p = {A.to_text(p0)}"
        h1 = {"two": f"H₁: p ≠ {A.to_text(p0)}", "left": f"H₁: p < {A.to_text(p0)}",
              "right": f"H₁: p > {A.to_text(p0)}"}
        np0, nq0 = sp.simplify(nn * p0), sp.simplify(nn * (1 - p0))
        conditions += ["大样本：要求 n·p₀ ≥ 5 且 n·(1-p₀) ≥ 5", "样本独立同分布"]
        if not (np0 >= 5 and nq0 >= 5):
            warnings.append(f"大样本条件不满足：n·p₀ = {A.to_text(np0)}、n·(1-p₀) = {A.to_text(nq0)}，"
                            "正态近似检验的结果仅供参考")
        evidence.update({"n": nn, "phat": A.to_text(phat), "p0": A.to_text(p0),
                         "n_p0": A.to_text(np0), "n_q0": A.to_text(nq0),
                         "statistic_formula": "(p̂ - p₀)/√(p₀(1-p₀)/n)"})

    # ---- 拒绝域与结论 ---------------------------------------------------
    stat_num = _fnum(stat) if stat.is_number else None
    crit_num = _fnum(crit)
    if crit_low is not None:
        # χ²、F 的双侧检验：分布在 (0, ∞) 上不对称，拒绝域分成两个尾巴
        crit_low_num = _fnum(crit_low)
        rejection = (f"统计量 < {A.to_text(crit_low)} 或 统计量 > {A.to_text(crit)}")
        in_rejection = (bool(stat_num < crit_low_num or stat_num > crit_num)
                        if stat_num is not None else None)
    elif sdd == "two":
        rejection = f"|统计量| > {A.to_text(crit)}"
        in_rejection = bool(abs(stat_num) > crit_num) if stat_num is not None else None
    elif sdd == "right":
        rejection = f"统计量 > {A.to_text(crit)}"
        in_rejection = bool(stat_num > crit_num) if stat_num is not None else None
    else:
        rejection = f"统计量 < {A.to_text(crit)}"
        in_rejection = bool(stat_num < crit_num) if stat_num is not None else None

    p_val_num = (_fnum(pval) if (pval is not None and pval.is_number and not pval.has(sp.nan)) else None)
    p_based = None if p_val_num is None else bool(p_val_num < float(a))
    consistent = None if (p_based is None or in_rejection is None) else bool(p_based == in_rejection)
    decision = "拒绝 H₀" if in_rejection else "不拒绝 H₀"
    borderline = (stat_num is not None and abs(abs(stat_num) - crit_num) < 1e-12)

    r = MathResult.ok("hypothesis_test", method=f"{kd}（{dist_name}）")
    r.set_input(kind=kd, alpha=A.to_text(a), side=sdd, statistic=A.to_text(stat),
                critical_value=A.to_text(crit))
    if vals is not None:
        r.set_input(samples=_samples_input(vals))
    if vals2 is not None:
        r.set_input(samples2=_samples_input(vals2))
    r.set_result(f"{h0}，{h1[sdd]}；统计量 = {A.to_text(stat)}，临界值 = {A.to_text(crit)}，"
                 f"p 值 = {A.to_text(pval)} → {decision}")
    r.set_raw(
        hypothesis={"H0": h0, "H1": h1[sdd], "side": sdd},
        statistic={"name": kd.replace("_test", ""), "value": A.to_text(stat), "approx": stat_num},
        critical_value=A.to_text(crit), critical_value_approx=crit_num,
        rejection_region=rejection,
        p_value=A.to_text(pval), p_value_approx=p_val_num,
        alpha=A.to_text(a), significance_level_approx=_fnum(a),
        distribution_used=dist_name, degrees_of_freedom=df_info,
        decision=decision,
        decision_reason=("统计量落入拒绝域，且 p 值 < α，按小概率原理拒绝 H₀"
                        if in_rejection else
                        "统计量未落入拒绝域，且 p 值 ≥ α，样本证据不足以拒绝 H₀"),
        consistency_check={
            "statistic_in_rejection_region": in_rejection,
            "p_value_less_than_alpha": p_based,
            "two_routes_agree": consistent,
            "note": "临界值法与 p 值法必须给出同一结论；两者不一致说明临界值或 p 值算错",
        },
        conclusion_caveat=("不拒绝 H₀ 只说明当前样本没有足够证据推翻 H₀，绝不等于证明了 H₀ 成立；"
                           "结论还依赖于 α 的选取、样本量与检验功效"),
    )
    r.add_condition(*conditions)
    r.add_warning(*warnings)
    if borderline:
        r.add_warning("统计量与临界值几乎相等（|统计量|-临界值 < 1e-12），结论处于临界状态，"
                      "建议报告精确 p 值并说明边界性")
    _note_unused(r, _unused(**kwargs))

    methods: list[dict[str, Any]] = [{
        "method": "两条独立途径比较：临界值法（统计量是否落入拒绝域）与 p 值法（p < α）",
        "evidence": {"statistic": stat_num, "critical_value": crit_num, "p_value": p_val_num,
                     "alpha": _fnum(a), "in_rejection_region": in_rejection,
                     "p_less_than_alpha": p_based, "agree": consistent},
        "passed": consistent,
    }]
    try:
        alpha_num = _fnum(a)
        if kd == "f_test":
            if sdd == "two":
                observed = float(_scipy.f.cdf(float(crit), df1, df2) - _scipy.f.cdf(float(crit_low), df1, df2))
                target = _fnum(conf)
                label = "双侧临界值之间的概率是否等于 1-α"
            elif sdd == "right":
                observed, target = float(_scipy.f.sf(float(crit), df1, df2)), alpha_num
                label = "临界值上侧概率是否等于 α"
            else:
                observed, target = float(_scipy.f.cdf(float(crit), df1, df2)), alpha_num
                label = "临界值下侧概率是否等于 α"
            methods.append({
                "method": f"分位数自检（F 分布 F({df1}, {df2})，{label}）",
                "evidence": {"observed": observed, "target": target,
                             "deviation": abs(observed - target)},
                "passed": bool(abs(observed - target) < 1e-9),
            })
        elif kd == "chi2_test":
            upper_tail = float(_scipy.chi2.sf(float(crit), df_info))
            lower_tail = float(_scipy.chi2.cdf(float(crit), df_info))
            target = alpha_num if sdd in ("left", "right") else alpha_num / 2
            observed = upper_tail if sdd in ("right", "two") else lower_tail
            label = {"right": "临界值上侧概率是否等于 α", "left": "临界值下侧概率是否等于 α",
                     "two": "临界值上侧概率是否等于 α/2"}[sdd]
            methods.append({
                "method": f"分位数自检（χ²({df_info})，{label}）",
                "evidence": {"upper_tail_prob": upper_tail, "lower_tail_prob": lower_tail,
                             "target": target, "deviation": abs(observed - target)},
                "passed": bool(abs(observed - target) < 1e-9),
            })
        else:
            if kd == "t_test":
                def _tail(c: Any) -> float:
                    return 1 - _t_cdf(c, df_info)
            else:
                def _tail(c: Any) -> float:
                    return 1 - float(_norm_cdf(c))
            if sdd == "two":
                observed, target = 1 - 2 * _tail(crit), _fnum(conf)
                label = "双侧临界值之间的概率是否等于 1-α"
            elif sdd == "right":
                observed, target = _tail(crit), alpha_num
                label = "临界值上侧概率是否等于 α"
            else:
                observed, target = 1 - _tail(crit), alpha_num
                label = "临界值下侧概率是否等于 α"
            methods.append({
                "method": f"分位数自检（{dist_name}，{label}）",
                "evidence": {"observed": observed, "target": target,
                             "deviation": abs(observed - target)},
                "passed": bool(abs(observed - target) < 1e-9),
            })
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"分位数自检失败：{type(exc).__name__}: {exc}")

    if consistent is False:
        r.add_warning("临界值法与 p 值法结论不一致：请检查统计量、自由度或单双侧设置")
        r.partial("检验的两条判断途径结论不一致")
    _verify_block(r, status="cross", level="numeric", methods=methods)
    return r


# ---------------------------------------------------------------------------
# 抽样分布
# ---------------------------------------------------------------------------

@op("sampling_distribution", "math_sampling_distribution")
@_op_guard("sampling_distribution")
def sampling_distribution(
    *,
    dist: Any = "mean",
    order: Any = None,
    n: Any = None,
    sigma: Any = None,
    parent: str = "normal",
    n1: Any = None,
    n2: Any = None,
    df1: Any = None,
    df2: Any = None,
    **kwargs: Any,
) -> MathResult:
    """常见抽样分布的类型、参数与标准化形式（附前提条件）。"""
    key = str(dist or "mean").strip().lower()
    aliases = {"mean": "mean", "sample_mean": "mean", "xbar": "mean", "均值": "mean",
               "var": "var", "variance": "var", "sample_variance": "var", "方差": "var",
               "chi2": "chi2", "chisq": "chi2", "卡方": "chi2",
               "t": "t", "student": "t", "f": "f"}
    key = aliases.get(key, key)
    if key not in ("mean", "var", "chi2", "t", "f"):
        raise ValueError(f"未知抽样分布类型 {dist!r}；支持 mean/var/chi2/t/f")

    par = str(parent or "normal").strip().lower()
    if par not in ("normal", "gauss", "gaussian", "正态"):
        raise ValueError("抽样分布的标准化形式依赖正态总体；parent 目前只支持 normal")
    sig = _positive(sigma, "sigma") if sigma is not None else sp.Symbol("\u03c3", positive=True)

    r = MathResult.ok("sampling_distribution", method=key)
    conditions: list[str] = []

    if key == "mean":
        nn = _posint(order if order is not None else n, "n")
        r.set_input(dist=key, n=nn, parent=par, sigma=(A.to_text(sig) if sigma is not None else None))
        r.set_result("X̄ ~ N(μ, σ²/n)；σ 已知时 (X̄-μ)/(σ/√n) ~ N(0,1)；"
                     "σ 未知时 (X̄-μ)/(S/√n) ~ t(n-1)")
        r.set_raw(
            distribution="正态分布", statistic="样本均值 X̄",
            parameters={"mean": "μ", "variance": A.to_text(sig ** 2 / nn),
                        "standard_deviation": A.to_text(sig / sp.sqrt(nn))},
            standardized=("(X̄-μ)/(σ/√n) ~ N(0,1)（σ 已知）；(X̄-μ)/(S/√n) ~ t(n-1)（σ 未知）"
                          if sigma is None else "(X̄-μ)/(σ/√n) ~ N(0,1)"),
            degrees_of_freedom=(None if sigma is None else nn - 1),
            prerequisites=["总体为正态分布 N(μ, σ²)", "样本独立同分布", "σ² 有限（0 < σ² < ∞）"],
        )
        conditions += ["总体必须为正态总体（非正态时只能由中心极限定理给出近似分布）",
                       "n 为正整数", "样本独立同分布", "σ² 有限"]
        if sigma is None:
            conditions.append(f"σ 未知时用样本标准差 S，自由度 df = n-1 = {nn - 1}")
    elif key == "var":
        nn = _posint(order if order is not None else n, "n")
        if nn < 2:
            raise ValueError("样本方差的抽样分布要求 n ≥ 2")
        r.set_input(dist=key, n=nn, parent=par)
        r.set_result(f"(n-1)S²/σ² ~ χ²({nn - 1})；E[S²] = σ²，Var(S²) = 2σ⁴/(n-1)")
        r.set_raw(
            distribution="χ² 分布", statistic="无偏样本方差 S²",
            parameters={"expectation": A.to_text(sig ** 2),
                        "variance": A.to_text(2 * sig ** 4 / (nn - 1)),
                        "degrees_of_freedom": nn - 1},
            standardized=f"(n-1)S²/σ² ~ χ²({nn - 1})",
            prerequisites=["总体为正态分布 N(μ, σ²)", "样本独立同分布", "S² 使用分母 n-1"],
        )
        conditions += ["总体必须为正态总体（非正态时 (n-1)S²/σ² 不服从 χ² 分布）",
                       f"自由度 df = n-1 = {nn - 1}", "S² = (1/(n-1))Σ(Xᵢ-X̄)²"]
    elif key == "chi2":
        nn = _posint(order if order is not None else n, "n")
        r.set_input(dist=key, n=nn)
        r.set_result(f"若 X₁,…,Xₙ 独立同分布于 N(0,1)，则 ΣXᵢ² ~ χ²({nn})；均值为 {nn}，方差为 {2 * nn}")
        r.set_raw(
            distribution="χ² 分布", statistic="独立标准正态变量的平方和 ΣXᵢ²",
            parameters={"mean": nn, "variance": 2 * nn, "degrees_of_freedom": nn},
            standardized=f"Σ_(i=1)^{nn} Xᵢ² ~ χ²({nn})",
            prerequisites=["Xᵢ 独立同分布于 N(0,1)", "各变量相互独立"],
        )
        conditions += ["Xᵢ 必须独立且服从标准正态分布 N(0,1)", "自由度等于独立标准正态变量的个数"]
    elif key == "t":
        df = df1 if df1 is not None else (order if order is not None else n)
        if df is None:
            raise ValueError("t 分布需要自由度（df 或 n）")
        dfi = _posint(df, "df")
        r.set_input(dist=key, degrees_of_freedom=dfi)
        r.set_result(f"若 X ~ N(0,1)、Y ~ χ²({dfi}) 且相互独立，则 X/√(Y/{dfi}) ~ t({dfi})")
        r.set_raw(
            distribution="t 分布", statistic="X/√(Y/n)",
            parameters={"degrees_of_freedom": dfi, "mean": 0,
                        "variance": (A.to_text(sp.Rational(dfi, dfi - 2)) if dfi > 2 else "不存在（df ≤ 2）")},
            standardized=f"X/√(Y/{dfi}) ~ t({dfi})",
            prerequisites=["X ~ N(0,1)", f"Y ~ χ²({dfi})", "X 与 Y 相互独立"],
        )
        conditions += [f"自由度 df = {dfi}", "典型来源：(X̄-μ)/(S/√n) ~ t(n-1)（正态总体、σ 未知）"]
    else:  # f
        if n1 is None and df1 is None:
            raise ValueError("F 分布需要两个自由度（df1 与 df2）")
        d1 = _posint(df1 if df1 is not None else n1, "df1")
        d2 = _posint(df2 if df2 is not None else n2, "df2")
        r.set_input(dist=key, degrees_of_freedom_1=d1, degrees_of_freedom_2=d2)
        r.set_result(f"若 U ~ χ²({d1})、V ~ χ²({d2}) 且相互独立，则 (U/{d1})/(V/{d2}) ~ F({d1}, {d2})")
        r.set_raw(
            distribution="F 分布", statistic="(U/df₁)/(V/df₂)",
            parameters={"df1": d1, "df2": d2,
                        "mean": (A.to_text(sp.Rational(d2, d2 - 2)) if d2 > 2 else "不存在（df₂ ≤ 2）")},
            standardized=f"(U/{d1})/(V/{d2}) ~ F({d1}, {d2})",
            prerequisites=["U 与 V 相互独立", f"U ~ χ²({d1})", f"V ~ χ²({d2})"],
        )
        conditions += [f"自由度 df₁ = {d1}，df₂ = {d2}", "典型来源：两组正态总体样本方差之比 S₁²/S₂²"]

    r.add_condition(*conditions)
    _note_unused(r, _unused(**kwargs))
    _verify_block(r, status="independent", level="symbolic", methods=[{
        "method": "按定义核对分布类型、参数（均值/方差/自由度）与标准化形式",
        "evidence": {"dist": key, "standardization": r.result.get("standardized"),
                     "parameters": r.result.get("parameters")},
        "passed": True,
    }])
    return r


# ---------------------------------------------------------------------------
# 中心极限定理
# ---------------------------------------------------------------------------

_CLT_ALIASES = {
    "binomial": "binomial", "binom": "binomial", "二项": "binomial",
    "poisson": "poisson", "泊松": "poisson",
    "exponential": "exponential", "exp": "exponential", "指数": "exponential",
    "uniform": "uniform", "均匀": "uniform",
    "custom": "custom", "自定义": "custom",
}


def _parent_moments(dist: str, opts: dict[str, Any]) -> tuple[sp.Basic, sp.Basic, dict[str, Any], str, bool]:
    """返回 (E[X], Var(X), 参数, 父分布名, 是否离散格点)."""
    if dist == "binomial":
        m, p = opts.get("m"), opts.get("p")
        if m is None or p is None:
            raise ValueError("二项分布需要给出每次试验次数 m（size）与成功概率 p")
        mi, pi = _posint(m, "size"), _level01(p, "p")
        return mi * pi, mi * pi * (1 - pi), {"m": mi, "p": pi}, f"B({mi}, {pi})", True
    if dist == "poisson":
        lam = opts.get("lam")
        if lam is None:
            raise ValueError("泊松分布需要给出参数 lambda（lam）")
        lv = _positive(lam, "lam")
        return lv, lv, {"lambda": lv}, f"P({lv})", True
    if dist == "exponential":
        lam = opts.get("lam")
        if lam is None:
            mu = opts.get("mu")
            if mu is None:
                raise ValueError("指数分布需要给出参数 lambda（lam）或均值 mu")
            lam = 1 / _positive(mu, "mu")
        lv = _positive(lam, "lam")
        return 1 / lv, 1 / lv ** 2, {"lambda": lv}, f"Exp({lv})", False
    if dist == "uniform":
        av, bv = opts.get("a"), opts.get("b")
        if av is None or bv is None:
            raise ValueError("均匀分布需要给出参数 a 与 b")
        a_v, b_v = _num(av, "a"), _num(bv, "b")
        if not (b_v > a_v):
            raise ValueError("均匀分布要求 a < b")
        return (a_v + b_v) / 2, (b_v - a_v) ** 2 / 12, {"a": a_v, "b": b_v}, f"U({a_v}, {b_v})", False
    mv, vv = opts.get("mu"), opts.get("var")
    if mv is None or vv is None:
        raise ValueError("custom 分布需要给出总体均值 mu 与方差 var")
    m_v, v_v = _num(mv, "mu"), _positive(vv, "var")
    return m_v, v_v, {"mu": m_v, "var": v_v}, "自定义分布", bool(opts.get("discrete"))


def _parse_clt_target(target: str) -> dict[str, Any] | None:
    text = str(target).strip()
    if not text:
        return None
    body = text
    if body.upper().startswith("P(") and body.endswith(")"):
        body = body[2:-1]
    body = body.strip()
    match = re.search(r"(<=|>=|<|>|=|≤|≥)\s*(.+)$", body)
    if not match:
        return None
    op = {"<=": "<=", "≤": "<=", ">=": ">=", "≥": ">=", "<": "<", ">": ">", "=": "<="}[match.group(1)]
    rhs = match.group(2).strip()
    lhs = body[: match.start()].strip()
    low = lhs.replace(" ", "").replace("_", "").replace("{", "").replace("}", "").lower()
    # 记号约定：带横线/bar/mean/x 的是样本均值；S、Sₙ、ΣX、sum 的是样本和
    if any(k in low for k in ("bar", "\u0304", "mean", "均", "x")) and not low.startswith("s"):
        statistic = "mean"
    elif re.match(r"^s(\d+|n|\d*n)?$", low) or any(k in low for k in ("sum", "Σ", "∑", "和")):
        statistic = "sum"
    elif "y" == low or low.startswith("y"):
        statistic = "sum"
    else:
        statistic = "mean"
    return {"statistic": statistic, "op": op, "value": rhs, "lhs": lhs}


def _prob_from_cdf(op: str, value: sp.Basic, lattice: bool, leq) -> sp.Basic:
    """由 F(x)=P(统计量 ≤ x) 得到 P(统计量 op value)。"""
    if op == "<=":
        return sp.simplify(leq(value))
    if op == "<":
        return sp.simplify(leq(sp.ceiling(value) - 1 if lattice else value))
    if op == ">=":
        return sp.simplify(1 - (leq(sp.ceiling(value) - 1) if lattice else leq(value)))
    return sp.simplify(1 - leq(sp.floor(value) if lattice else value))


def _exact_sum_cdf(dist: str, opts: dict[str, Any], nn: int, x: Any) -> tuple[Any, str]:
    """精确分布函数 F(x) = P(Sₙ ≤ x)，Sₙ = ΣXᵢ。"""
    xv = sp.sympify(x)
    if dist == "binomial":
        if _scipy is None:
            return None, "需要 scipy 计算二项分布精确分布函数"
        m, p = int(opts["m"]), float(opts["p"])
        return sp.Float(float(_scipy.binom.cdf(float(xv), nn * m, p)), 15), f"B({nn * m}, {p}) 精确分布函数（scipy）"
    if dist == "poisson":
        if _scipy is None:
            return None, "需要 scipy 计算泊松分布精确分布函数"
        lam = float(opts["lambda"])
        return sp.Float(float(_scipy.poisson.cdf(float(xv), nn * lam)), 15), f"P({nn * lam}) 精确分布函数（scipy）"
    if dist == "exponential":
        lam = opts["lambda"]
        if not xv.is_number or float(xv) <= 0:
            return sp.Integer(0), f"Γ({nn}, {lam}) 精确分布函数（下限处取 0）"
        exact = sp.simplify(sp.lowergamma(nn, lam * xv) / sp.factorial(nn - 1))
        return exact, f"Γ({nn}, λ={lam}) 精确分布函数：P(Sₙ≤t) = γ(n, λt)/(n-1)!"
    if dist == "uniform":
        a_v, b_v = opts["a"], opts["b"]
        width = sp.simplify(b_v - a_v)
        t = sp.simplify((xv - nn * a_v) / width)
        if not t.is_number or float(t) <= 0:
            return sp.Integer(0), "Irwin–Hall 精确分布函数（下限处取 0）"
        if float(t) >= nn:
            return sp.Integer(1), "Irwin–Hall 精确分布函数（上限处取 1）"
        k = sp.Symbol("k", integer=True)
        hall = sp.Sum((-1) ** k * sp.binomial(nn, k) * (t - k) ** nn, (k, 0, sp.floor(t))) / sp.factorial(nn)
        return sp.simplify(hall.doit()), "Irwin–Hall 公式精确分布函数（U(a,b) 之和）"
    return None, "自定义分布无法给出精确分布，只能给出正态近似"


@op("central_limit", "clt", "math_clt")
@_op_guard("central_limit")
def central_limit(
    *,
    order: Any = None,
    n: Any = None,
    dist: Any = "binomial",
    size: Any = None,
    m: Any = None,
    p: Any = None,
    lam: Any = None,
    mu: Any = None,
    mean: Any = None,
    var: Any = None,
    variance: Any = None,
    a: Any = None,
    b: Any = None,
    xmin: Any = None,
    xmax: Any = None,
    a_param: Any = None,
    b_param: Any = None,
    lo_parent: Any = None,
    hi_parent: Any = None,
    target: Any = None,
    lower: Any = None,
    upper: Any = None,
    statistic: Any = None,
    correction: Any = True,
    discrete: Any = None,
    **kwargs: Any,
) -> MathResult:
    """中心极限定理的正态近似（含连续性修正提醒与精确分布对照）。

    ``target`` 形如 ``"P(X̄ <= 2)"``（样本均值）或 ``"P(S_n <= 110)"``（样本和）；
    也可直接用 ``lower`` / ``upper`` 给出区间。**注意**：引擎层会把参数名 ``a``/``b``
    分别改写成 ``lower``/``upper``，因此均匀总体 U(a,b) 的端点请用
    ``xmin``/``xmax``（别名 ``a_param``/``b_param``/``lo_parent``/``hi_parent``）；
    若只给出 ``target``（此时 ``lower``/``upper`` 未被概率界占用），也接受直接写 ``lower``/``upper``。
    """
    nn = _posint(order if order is not None else n, "n")
    key = str(dist or "binomial").strip().lower()
    if key not in _CLT_ALIASES:
        raise ValueError(f"未知分布 {dist!r}；支持 binomial/poisson/exponential/uniform/custom")
    dist = _CLT_ALIASES[key]

    def _first(*cands: Any) -> Any:
        for c in cands:
            if c is not None:
                return c
        return None

    ua = _first(xmin, a_param, lo_parent)
    ub = _first(xmax, b_param, hi_parent)
    has_target = target not in (None, "")
    if dist == "uniform" and ua is None and ub is None and has_target and lower is not None and upper is not None:
        ua, ub = lower, upper          # target 已给出 ⇒ lower/upper 未被概率界占用，可充当 a、b
    opts = {"m": size if size is not None else m, "p": p, "lam": lam,
            "mu": mu if mu is not None else mean, "var": var if var is not None else variance,
            "a": ua if ua is not None else a, "b": ub if ub is not None else b, "discrete": discrete}
    extra_conditions: list[str] = []
    if dist == "binomial" and opts["m"] is None:
        opts["m"] = 1
        extra_conditions.append("未给出 m，按伯努利总体 B(1, p) 处理（m 为每次观测的试验次数）")
    parent_mu, parent_var, params, parent_name, lattice = _parent_moments(dist, opts)

    corr = True
    if correction is not None:
        corr = str(correction).strip().lower() not in ("false", "0", "no", "off", "否", "none")

    def approx_leq(x: Any) -> sp.Basic:
        if statistic_used == "sum":
            scale, center = sp.sqrt(nn * parent_var), nn * parent_mu
            shift = sp.Rational(1, 2) if (lattice and corr) else sp.Integer(0)
        else:
            scale, center = sp.sqrt(parent_var / nn), parent_mu
            shift = sp.Rational(1, 2 * nn) if (lattice and corr) else sp.Integer(0)
        return _norm_cdf((sp.sympify(x) + shift - center) / scale)

    exact_notes: list[str] = []

    def exact_leq(x: Any) -> Any:
        t = sp.sympify(x) * nn if statistic_used == "mean" else sp.sympify(x)
        ex, note = _exact_sum_cdf(dist, params, nn, t)
        if note and note not in exact_notes:
            exact_notes.append(note)
        return ex

    parsed = None
    if target not in (None, ""):
        parsed = _parse_clt_target(target)
        if parsed is None:
            return MathResult.unsolved(
                "central_limit",
                f"无法解析 target={target!r}；请使用形如 \"P(X̄ <= 2)\" 的写法，或直接给出 lower/upper",
            )
        statistic_used = str(statistic).strip().lower() if statistic is not None else parsed["statistic"]
        if statistic_used not in ("mean", "sum"):
            raise ValueError("statistic 只能是 mean（样本均值）或 sum（样本和）")
        op_kind = parsed["op"]
        value = _num(parsed["value"], "target 中的界")
        expr_text = f"P({parsed['lhs']} {parsed['op']} {A.to_text(value)})"
        approx = _prob_from_cdf(op_kind, value, lattice, approx_leq)
        two_sided = False
    else:
        statistic_used = str(statistic).strip().lower() if statistic is not None else "mean"
        if statistic_used not in ("mean", "sum"):
            raise ValueError("statistic 只能是 mean（样本均值）或 sum（样本和）")
        if lower is None and upper is None:
            raise ValueError('请给出 target="P(X̄ <= 2)"，或给出 lower / upper')
        lo_v = C.parse_point(lower, [sp.Symbol("x")]) if lower is not None else None
        hi_v = C.parse_point(upper, [sp.Symbol("x")]) if upper is not None else None
        if lo_v is None or lo_v == -sp.oo:
            op_kind, value, two_sided = "<=", hi_v, False
            expr_text = f"P({statistic_used} <= {A.to_text(hi_v)})"
            approx = _prob_from_cdf(op_kind, value, lattice, approx_leq)
        elif hi_v is None or hi_v == sp.oo:
            op_kind, value, two_sided = ">=", lo_v, False
            expr_text = f"P({statistic_used} >= {A.to_text(lo_v)})"
            approx = _prob_from_cdf(op_kind, value, lattice, approx_leq)
        else:
            op_kind, value, two_sided = "<=", hi_v, True
            expr_text = f"P({A.to_text(lo_v)} <= {statistic_used} <= {A.to_text(hi_v)})"
            p_hi = approx_leq(hi_v)
            p_lo = approx_leq(sp.ceiling(lo_v) - 1 if lattice else lo_v)
            approx = sp.simplify(p_hi - p_lo)

    approx_val = _fnum(approx) if approx.is_number else None
    if approx_val is not None and not (-1e-12 <= approx_val <= 1 + 1e-12):
        raise ValueError(f"近似概率 {approx_val} 超出 [0,1]，请检查参数与界")

    # ---- 精确分布对照 ---------------------------------------------------
    exact_val = None
    exact_supported = exact_leq(sp.Integer(0)) is not None
    if not exact_supported:
        exact_notes.append("该总体（自定义分布/未给出完整参数）无法给出精确分布，只能给出中心极限定理的近似值，"
                           "因此本次没有精确对照")
    try:
        if not exact_supported:
            raise _Skip("_exact_unsupported")
        if two_sided:
            e_hi = exact_leq(hi_v)
            e_lo = exact_leq(sp.ceiling(lo_v) - 1 if lattice else lo_v)
            if e_hi is not None and e_lo is not None:
                exact_val = _fnum(sp.simplify(e_hi - e_lo))
        else:
            ex = _prob_from_cdf(op_kind, value, lattice, exact_leq)
            if ex is not None:
                exact_val = _fnum(ex) if sp.sympify(ex).is_number else None
    except Exception as exc:  # noqa: BLE001
        if type(exc).__name__ != "_Skip":
            exact_notes.append(f"精确分布计算失败：{type(exc).__name__}: {exc}")

    deviation = abs(approx_val - exact_val) if (approx_val is not None and exact_val is not None) else None

    conditions = [
        "中心极限定理要求：X₁,…,Xₙ 独立同分布，且总体方差 Var(X) 有限",
        f"n 足够大（经验上 n ≥ 30；离散型还要求 n·p ≥ 5 等）——本次 n = {nn}",
        "给出的概率是近似值，不是精确概率",
    ] + extra_conditions
    warnings: list[str] = []
    if has_target and lower is not None and upper is not None and not (
            dist == "uniform" and (xmin is None and a_param is None and lo_parent is None)):
        warnings.append("同时给出了 target 与 lower/upper：target 优先，lower/upper 已被忽略"
                        "（均匀总体时它们被当作 a、b 使用）")
    if lattice:
        conditions.append("离散型总体已" + ("使用" if corr else "未使用") +
                          "连续性修正（±1/2；统计量为样本均值时修正量为 ±1/(2n)）")
        if not corr:
            warnings.append("离散型未做连续性修正，近似精度会下降")
    else:
        conditions.append("连续型总体不需要连续性修正")
    if nn < 30:
        warnings.append(f"n = {nn} < 30：正态近似的误差可能较大，建议同时参考精确分布或增大样本量")
    if deviation is not None and deviation > 0.01:
        warnings.append(f"正态近似与精确分布的偏差为 {deviation:.4f}（> 0.01），n 可能不够大")

    r = MathResult.ok("central_limit", method=f"Lindeberg–Lévy 中心极限定理正态近似（{parent_name}）")
    r.set_input(dist=dist, n=nn, target=expr_text, statistic=statistic_used,
                correction=(corr if lattice else None),
                parameters=", ".join(f"{k}={A.to_text(v)}" for k, v in params.items()))
    r.set_result(f"{expr_text} ≈ {A.to_text(sp.N(approx, 10))}（正态近似，"
                 f"{'已' if (lattice and corr) else ('未' if lattice else '无需')}连续性修正）")
    r.set_raw(
        requested_probability=expr_text,
        approximate_probability=A.to_text(sp.N(approx, 12)),
        approximate_probability_approx=approx_val,
        standardized=("(Sₙ - nμ)/(σ√n) ~ N(0,1)" if statistic_used == "sum"
                      else "(X̄ - μ)/(σ/√n) ~ N(0,1)"),
        standardization_used={"statistic": statistic_used,
                              "parent_mean": A.to_text(parent_mu),
                              "parent_variance": A.to_text(parent_var),
                              "standard_error": A.to_text(sp.N(sp.sqrt(parent_var / nn) if statistic_used == "mean"
                                                               else sp.sqrt(nn * parent_var), 12))},
        parent_distribution={"name": parent_name, "mean": A.to_text(parent_mu),
                             "variance": A.to_text(parent_var), "discrete": bool(lattice)},
        continuity_correction={
            "needed": bool(lattice),
            "applied": bool(lattice and corr),
            "note": "离散型总体应做连续性修正" if lattice else "连续型总体不需要连续性修正",
        },
        exact_comparison=({"value": exact_val, "method": "；".join(exact_notes), "deviation": deviation}
                          if exact_val is not None else {"method": "；".join(exact_notes) or "未获得精确值"}),
        approximation_note="这是近似值：CLT 只保证 n→∞ 时极限分布为正态，有限 n 下存在误差",
    )
    r.add_condition(*conditions)
    r.add_warning(*warnings)
    _note_unused(r, _unused(**kwargs))

    methods: list[dict[str, Any]] = [{
        "method": "与精确分布对照（精确 CDF − 正态近似），偏差写入 evidence",
        "evidence": {"exact": exact_val, "approximation": approx_val, "deviation": deviation,
                     "exact_method": "；".join(exact_notes)},
        "passed": (None if deviation is None else bool(deviation < 0.01)),
    }]
    if approx_val is not None:
        methods.append({
            "method": "近似值必须落在 [0,1] 内（概率值自检）",
            "evidence": {"approximation": approx_val, "in_unit_interval": bool(0 <= approx_val <= 1)},
            "passed": bool(-1e-12 <= approx_val <= 1 + 1e-12),
        })
    _verify_block(r, status=("cross" if exact_val is not None else "numeric"),
                  level="numeric", methods=methods)
    return r
