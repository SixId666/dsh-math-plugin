"""概率统计模块自检（probability.py）。

走的是 Node 侧完全相同的调用路径：``E.execute("<算子名>", {...})``。
覆盖：贝叶斯 / 全概率 / 条件概率 / 独立性 / 容斥 / 常见分布的 pdf·cdf·区间概率 /
期望·方差 / 协方差·相关系数 / 联合边缘与独立性判定 / 自定义分布律 / 矩汇总。

运行：``cd E:\\dsh-math\\engine; python _selftest_probability.py``
"""

import sys
import os

sys.path.insert(0, r"E:\dsh-math\engine")

import math  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import traceback  # noqa: E402

from mathkit import engine as E  # noqa: E402
from mathkit import ops as _ops  # noqa: E402  （必须 import，否则算子未注册）

# ---------------------------------------------------------------------------
# 结果收集与打印
# ---------------------------------------------------------------------------

RESULTS = []


def run(op, args, expect=None, note="", expect_approx=None, tol=1e-6, expect_contains=None):
    """执行一次算子调用并打印。expect 是期望的 result.text 字符串（精确值）。

    expect_contains 用于「结果是一句人话」的场合（例如「E[X] 不存在」），
    这时比对整串文本没有意义，只查关键子串。
    """
    label = note or f"{op} {args}"
    try:
        out = E.execute(op, args)
    except Exception:  # noqa: BLE001  引擎正常情况下不抛异常，这里兜底
        print(f"\n### {label}")
        print("!!! execute 抛异常：")
        traceback.print_exc()
        RESULTS.append({"label": label, "status": "traceback", "ok": False, "approx": None})
        return None

    status = out.get("status")
    result = out.get("result") or {}
    text = result.get("text")
    approx = result.get("approx")
    verification = out.get("verification") or {}
    vstatus = verification.get("status")
    conditions = out.get("conditions") or []
    warnings = out.get("warnings") or []
    error = out.get("error")

    print(f"\n### {label}")
    print(f"    op={out.get('operation')}  status={status}  success={out.get('success')}")
    print(f"    result.text = {text}")
    print(f"    result.approx = {approx}")
    print(f"    conditions = {conditions}")
    if warnings:
        print(f"    warnings = {warnings}")
    print(f"    verification.status = {vstatus}")
    for m in verification.get("methods") or []:
        # MathResult 的约定是 name/detail/passed；个别并行模块（probability_extra）
        # 写成 method/evidence/passed=None，这里做容错，免得打印出 "None: None"。
        label = m.get("name") or m.get("method") or "未命名验证方法"
        if m.get("detail") is not None:
            detail = m.get("detail")
        elif m.get("evidence") is not None:
            detail = m.get("evidence")
        else:
            detail = ""
        flag = m.get("passed")
        mark = "PASS" if flag else ("N/A" if flag is None else "FAIL")
        print(f"        - [{mark}] {label}: {detail}")
    if verification.get("note"):
        print(f"        note: {verification['note']}")
    if error:
        print(f"    error.kind={error.get('kind')}  error.message={error.get('message')}")
    # set_raw(**kwargs) 会把内容并进 to_dict() 的顶层，所以这里打印「非标准顶层键」
    _STANDARD = {"success", "status", "operation", "method", "input", "result",
                 "conditions", "verification", "warnings", "elapsed_ms", "error"}
    extra = {k: v for k, v in out.items() if k not in _STANDARD}
    if extra:
        print(f"    extra(raw) = {extra}")
    # 硬性约定 5：结果必须能直接序列化成 JSON（绝不能把 SymPy 对象塞进 set_input）
    try:
        json.dumps(out, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        print(f"!!! JSON 序列化失败（有非 JSON 对象泄漏进结果）：{exc}")
        RESULTS.append({"label": label, "status": "not_json", "ok": False, "approx": None})
        return out

    check_pass = None
    checks = []

    if expect is not None:
        got = str(text).strip() if text is not None else None
        checks.append((f"result.text == {expect!r}", got == expect, got))

    if expect_contains is not None:
        got = str(text) if text is not None else ""
        checks.append((f"result.text 含 {expect_contains!r}", expect_contains in got, got))

    if expect_approx is not None:
        # result.approx 有时是纯数字，有时是一句摘要（例如 CLT 的
        # "P(X̄ <= 2/5) ≈ 0.9357796380（正态近似，已连续性修正）"），
        # 所以先直接 float()，失败再从字符串里正则抠出最后一个浮点数。
        got_num = None
        for candidate in (approx, text):
            if candidate is None:
                continue
            try:
                got_num = float(candidate)
                break
            except (TypeError, ValueError):
                found = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(candidate))
                if found:
                    try:
                        got_num = float(found[-1])
                        break
                    except ValueError:
                        continue
        if got_num is None:
            checks.append((f"≈ {expect_approx}", False, text))
        else:
            delta = abs(got_num - expect_approx)
            checks.append((f"≈ {expect_approx} (tol {tol:g})", delta <= tol, f"{got_num!r}, |Δ|={delta:.3e}"))

    if status not in ("ok", "partial", "unsolved"):
        checks.append(("status 合法", False, status))

    for name, passed, got in checks:
        print(f"    CHECK {'PASS' if passed else 'FAIL'}: {name} -> {got}")

    ok = all(c[1] for c in checks) if checks else None
    if ok is False:
        print("    ^^^ 该例校验失败")

    RESULTS.append({
        "label": label, "op": out.get("operation"), "status": status,
        "success": out.get("success"), "text": text, "approx": approx,
        "vstatus": vstatus, "ok": ok,
        "n_methods": len(verification.get("methods") or []),
        "n_pass": sum(1 for m in (verification.get("methods") or []) if m.get("passed")),
    })
    return out


def expect_error(op, args, note="", contains=None):
    """期望输入非法 → status=error 且 kind=invalid_input。"""
    label = note or f"{op} {args}（应报错）"
    out = E.execute(op, args)
    status = out.get("status")
    error = out.get("error") or {}
    kind = error.get("kind")
    message = error.get("message") or ""
    passed = status == "error" and kind == "invalid_input" and (contains is None or contains in message)
    print(f"\n### {label}")
    print(f"    status={status}  error.kind={kind}")
    print(f"    error.message={message}")
    print(f"    CHECK {'PASS' if passed else 'FAIL'}: 应报 invalid_input 错误（含 {contains!r}）")
    RESULTS.append({"label": label, "status": status, "ok": passed, "op": op})
    return out


def expect_unsolved(op, args, note="", contains=None):
    """期望「本工具确实求不出来」→ status=unsolved（**不是** error，也不是冒充 ok）。

    用户硬性要求：无法求解必须明确标记，绝不伪造结果。
    """
    label = note or f"{op} {args}（应 unsolved）"
    out = E.execute(op, args)
    status = out.get("status")
    error = out.get("error") or {}
    message = error.get("message") or ""
    passed = status == "unsolved" and bool(message) and (contains is None or contains in message)
    print(f"\n### {label}")
    print(f"    status={status}  success={out.get('success')}")
    print(f"    unsolved 说明 = {message}")
    print(f"    CHECK {'PASS' if passed else 'FAIL'}: 应明确标为 unsolved 且给出原因（含 {contains!r}）")
    RESULTS.append({"label": label, "status": status, "ok": passed, "op": op})
    return out


# ===========================================================================
print("=" * 78)
print("概率统计模块自检  (mathkit.probability)")
print("=" * 78)
print("ops.status() ->", _ops.status())

# --- 1. 贝叶斯（经典三工厂题） ---------------------------------------------
run("probability", {"kind": "bayes", "prior": "0.3,0.5,0.2", "likelihood": "0.9,0.8,0.5"},
    expect_approx=None, note="贝叶斯：三工厂 prior=0.3,0.5,0.2 likelihood=0.9,0.8,0.5")

bayes = E.execute("probability", {"kind": "bayes", "prior": "0.3,0.5,0.2", "likelihood": "0.9,0.8,0.5"})
post = (bayes.get("raw") or bayes.get("extra") or {})
print("\n    贝叶斯后验求和校验：")
for key, value in post.items():
    print(f"        {key} = {value}")

# --- 2. 全概率 -------------------------------------------------------------
run("probability", {"kind": "total_probability", "prior": "0.3,0.5,0.2", "likelihood": "0.9,0.8,0.5"},
    expect_approx=0.3 * 0.9 + 0.5 * 0.8 + 0.2 * 0.5,
    note="全概率 P(B)=Σ P(Ai)P(B|Ai)=0.77")

# --- 3. 条件概率 -----------------------------------------------------------
run("probability", {"kind": "conditional", "joint": "1/6", "given": "1/2"},
    expect="1/3", note="条件概率 P(A|B)=P(AB)/P(B)=(1/6)/(1/2)=1/3")
expect_error("probability", {"kind": "conditional", "joint": "1/6", "given": "0"},
             note="条件概率 given=0 应报错", contains="0")

# --- 4. 独立性判定 ---------------------------------------------------------
run("probability", {"kind": "independent_check", "joint": "1/6", "pa": "1/2", "pb": "1/3"},
    note="独立性判定：P(AB)=1/6=1/2·1/3 → 独立")
run("probability", {"kind": "independent_check", "joint": "1/4", "pa": "1/2", "pb": "1/3"},
    note="独立性判定：P(AB)=1/4≠1/6 → 不独立")

# --- 5. 容斥 P(A∪B) --------------------------------------------------------
run("probability", {"kind": "inclusion_exclusion", "pa": "1/2", "pb": "1/3", "joint": "1/6"},
    expect="2/3", note="容斥 P(A∪B)=1/2+1/3−1/6=2/3")
run("probability", {"kind": "inclusion_exclusion", "pa": "1/2", "pb": "1/3"},
    expect_approx=1 / 2 + 1 / 3 - 1 / 6, note="容斥（未给 joint → 假设独立，结果同为 2/3）")

# --- 6. 标准正态 -----------------------------------------------------------
run("distribution", {"dist": "normal", "mu": "0", "sigma": "1", "kind": "pdf"},
    expect="sqrt(2)*exp(-x**2/2)/(2*sqrt(pi))", note="N(0,1) 密度")
run("distribution", {"dist": "normal", "mu": "0", "sigma": "1", "kind": "cdf"},
    note="N(0,1) 分布函数")
# 分位数：normal/exponential 有闭式（含 erfinv），chi2 与 t 无闭式 → 必须是 unsolved 而不是 error
run("distribution", {"dist": "exponential", "lam": "2", "kind": "quantile", "point": "1/2"},
    expect="log(2)/2", note="Exp(2) 的中位数=ln2/λ=ln2/2（闭式分位数）")
run("distribution", {"dist": "uniform", "a": "0", "b": "1", "kind": "quantile", "point": "1/4"},
    expect="1/4", note="U(0,1) 的 1/4 分位数=1/4（闭式分位数）")
expect_unsolved("distribution", {"dist": "chi2", "k": "4", "kind": "quantile", "point": "0.95"},
                contains="卡方分布的分位数",
                note="χ²(4) 的 0.95 分位数：无闭式 → 应为 unsolved（不是 error）")
expect_unsolved("distribution", {"dist": "t", "k": "5", "kind": "quantile", "point": "0.975"},
                contains="无法反解出闭式解",
                note="t(5) 的 0.975 分位数：solve 解不动 → 应为 unsolved（不是 error）")
run("probability", {"kind": "interval", "dist": "normal", "mu": "0", "sigma": "1",
                    "lower": "-1", "upper": "1"},
    expect="erf(sqrt(2)/2)", note="P(|X|<1) 标准正态（=erf(√2/2)≈0.6827）")
run("probability", {"kind": "interval", "dist": "normal", "mu": "0", "sigma": "1",
                    "lower": "-1", "upper": "1.96"},
    note="P(-1<X<1.96) 精确表达式（含 erf）≈0.8163")
run("probability", {"kind": "interval", "dist": "normal", "mu": "1", "sigma": "2",
                    "lower": "a", "upper": "b"},
    note="N(1,4) 的 P(a<X<b) 符号精确表达式（含 erf）")
run("probability", {"kind": "interval", "dist": "normal", "mu": "0", "sigma": "1",
                    "lower": "-oo", "upper": "0"},
    expect="1/2", note="P(X<0)=1/2（-oo 输入）")

# --- 7. 指数分布 -----------------------------------------------------------
run("distribution", {"dist": "exponential", "lam": "2", "kind": "pdf"},
    expect="Piecewise((2*exp(-2*x), x >= 0), (0, True))", note="Exp(λ=2) 密度")
run("expectation", {"dist": "exponential", "lam": "2"}, expect="1/2", note="E[X]=1/λ=1/2")
run("variance", {"dist": "exponential", "lam": "2"}, expect="1/4", note="Var(X)=1/λ²=1/4")
run("probability", {"kind": "interval", "dist": "exponential", "lam": "2",
                    "lower": "0", "upper": "1"}, expect_approx=1 - math.exp(-2),
    note="Exp(2) 的 P(0<X<1)=1−e^-2≈0.8647")

# --- 8. 二项分布 -----------------------------------------------------------
run("expectation", {"dist": "binomial", "n": "10", "p": "1/3"}, expect="10/3",
    note="B(10,1/3) 的 E[X]=np=10/3")
run("variance", {"dist": "binomial", "n": "10", "p": "1/3"}, expect="20/9",
    note="B(10,1/3) 的 Var=np(1-p)=20/9")
run("probability", {"kind": "at", "dist": "binomial", "n": "10", "p": "1/3", "point": "3"},
    expect="5120/19683", note="B(10,1/3) 的 P(X=3)=C(10,3)(1/3)^3(2/3)^7=15360/59049")
run("distribution", {"dist": "binomial", "n": "10", "p": "1/3", "kind": "pmf"},
    note="B(10,1/3) 分布律（符号 x）")
# 离散型区间概率：本工具按 P(lower < X <= upper) 计算，并应给出「要 P(lower<=X<=upper)
# 就把 lower 减 1」的 warning。
run("probability", {"kind": "interval", "dist": "binomial", "n": "10", "p": "1/3",
                    "lower": "2", "upper": "4"},
    expect_approx=9600 / 19683,
    note="B(10,1/3) 的 P(2<X≤4)=P(X=3)+P(X=4)=5120/19683+4480/19683=9600/19683≈0.4877")

# --- 9. 泊松分布 -----------------------------------------------------------
run("probability", {"kind": "at", "dist": "poisson", "lam": "2", "point": "0"},
    expect="exp(-2)", note="Poisson(2) 的 P(X=0)=e^-2")
run("expectation", {"dist": "poisson", "lam": "2"}, expect="2", note="Poisson(2) E[X]=λ=2")
run("variance", {"dist": "poisson", "lam": "2"}, expect="2", note="Poisson(2) Var=λ=2")
run("probability", {"kind": "at", "dist": "poisson", "lam": "2", "point": "3"},
    expect="4*exp(-2)/3", note="Poisson(2) 的 P(X=3)=8e^-2/6=4e^-2/3")

# --- 10. 均匀分布 ----------------------------------------------------------
run("expectation", {"dist": "uniform", "a": "0", "b": "1"}, expect="1/2",
    note="U(0,1) 的 E[X]=1/2")
run("variance", {"dist": "uniform", "a": "0", "b": "1"}, expect="1/12",
    note="U(0,1) 的 Var=1/12")
run("expectation", {"dist": "uniform", "a": "-1", "b": "3"}, expect="1",
    note="U(-1,3) 的 E[X]=1")
run("variance", {"dist": "uniform", "a": "-1", "b": "3"}, expect="4/3",
    note="U(-1,3) 的 Var=(b-a)²/12=4/3")

# --- 11. 几何分布（教材约定 P(X=k)=p(1-p)^(k-1), k=1,2,...） ----------------
run("expectation", {"dist": "geometric", "p": "1/3"}, expect="3",
    note="Geom(1/3) 的 E[X]=1/p=3（教材约定）")
run("variance", {"dist": "geometric", "p": "1/3"}, expect="6",
    note="Geom(1/3) 的 Var=(1-p)/p²=6")
run("probability", {"kind": "at", "dist": "geometric", "p": "1/3", "point": "1"},
    expect="1/3", note="Geom(1/3) 的 P(X=1)=p=1/3")

# --- 12. 超几何 / 离散均匀 -------------------------------------------------
run("expectation", {"dist": "hypergeometric", "N": "10", "K": "5", "n": "3"}, expect="3/2",
    note="H(N=10,K=5,n=3) 的 E[X]=nK/N=3/2")
run("variance", {"dist": "hypergeometric", "N": "10", "K": "5", "n": "3"},
    note="H(10,5,3) 的 Var")
run("probability", {"kind": "at", "dist": "hypergeometric", "N": "10", "K": "5", "n": "3",
                    "point": "2"}, expect="5/12",
    note="H(N=10,K=5,n=3) 的 P(X=2)=C(5,2)C(5,1)/C(10,3)=5/12")
run("expectation", {"dist": "uniform_discrete", "values": "1,2,3,4,5,6"}, expect="7/2",
    note="离散均匀（骰子）E[X]=7/2")
run("variance", {"dist": "uniform_discrete", "values": "1,2,3,4,5,6"}, expect="35/12",
    note="离散均匀（骰子）Var=35/12")

# --- 13. chi2 / t / f / gamma / beta / lognormal / weibull / cauchy ---------
run("distribution", {"dist": "chi2", "k": "4", "kind": "pdf"},
    expect="Piecewise((x*exp(-x/2)/4, x > 0), (0, True))", note="χ²(4) 密度")
run("expectation", {"dist": "chi2", "k": "4"}, expect="4", note="χ²(4) 的 E[X]=k=4")
run("variance", {"dist": "chi2", "k": "4"}, expect="8", note="χ²(4) 的 Var=2k=8")
run("distribution", {"dist": "t", "k": "5", "kind": "pdf"},
    expect="200*sqrt(5)/(3*pi*(x**2 + 5)**3)", note="t(5) 密度（归一化 ∫f=1）")
run("distribution", {"dist": "t", "k": "5", "kind": "cdf"}, note="t(5) 分布函数")
run("distribution", {"dist": "f", "d1": "5", "d2": "7", "kind": "pdf"},
    note="F(5,7) 密度（归一化 ∫f=1）")
run("expectation", {"dist": "f", "d1": "5", "d2": "7"}, expect_approx=7 / 5, tol=1e-9,
    note="F(5,7) 的 E[X]=d2/(d2-2)=7/5")
run("variance", {"dist": "f", "d1": "5", "d2": "7"}, expect_approx=2.6133333333333333, tol=1e-6,
    note="F(5,7) 的 Var=2d2²(d1+d2-2)/(d1(d2-2)²(d2-4))≈2.61333")
run("distribution", {"dist": "gamma", "k": "2", "lam": "3", "kind": "pdf"}, note="Γ(k=2,λ=3) 密度")
run("expectation", {"dist": "gamma", "k": "2", "lam": "3"}, expect="2/3", note="Γ(2,3) 的 E=k/λ=2/3")
run("variance", {"dist": "gamma", "k": "2", "lam": "3"}, expect="2/9", note="Γ(2,3) 的 Var=k/λ²=2/9")
run("distribution", {"dist": "beta", "a": "2", "b": "3", "kind": "pdf"}, note="Beta(2,3) 密度")
run("expectation", {"dist": "beta", "a": "2", "b": "3"}, expect="2/5",
    note="Beta(2,3) 的 E=a/(a+b)=2/5")
run("variance", {"dist": "beta", "a": "2", "b": "3"}, expect="1/25",
    note="Beta(2,3) 的 Var=ab/((a+b)²(a+b+1))=1/25")
run("distribution", {"dist": "lognormal", "mu": "0", "sigma": "1", "kind": "pdf"},
    note="对数正态(0,1) 密度")
run("expectation", {"dist": "lognormal", "mu": "0", "sigma": "1"}, expect="exp(1/2)",
    note="对数正态(0,1) 的 E=e^(μ+σ²/2)=√e")
run("expectation", {"dist": "lognormal", "mu": "0", "sigma": "1", "expr": "x^2"}, expect="exp(2)",
    note="对数正态(0,1) 的 E[X²]=e^(2μ+2σ²)=e²（SymPy 定义式积分无闭式，取标准公式+数值求积复核）")
run("variance", {"dist": "lognormal", "mu": "0", "sigma": "1"},
    expect_approx=math.e ** 2 - math.e, tol=1e-6,
    note="对数正态(0,1) 的 Var=(e^σ²-1)e^(2μ+σ²)=e²-e≈4.670774（定义式无闭式，数值求积复核）")
run("expectation", {"dist": "weibull", "k": "2", "lam": "1"}, expect="sqrt(pi)/2",
    note="Weibull(k=2,λ=1)（=Rayleigh）的 E=λΓ(1+1/k)=Γ(3/2)=√π/2")
run("distribution", {"dist": "cauchy", "kind": "pdf"}, note="柯西分布密度（期望∉R）")
run("expectation", {"dist": "cauchy"}, expect_contains="不存在",
    note="柯西分布期望不存在 → 必须明确说「不存在」，不能返回 nan/0")
run("variance", {"dist": "cauchy"}, expect_contains="不存在",
    note="柯西分布方差不存在 → 必须明确说「不存在」")
run("variance", {"dist": "t", "k": "1"}, expect_contains="不存在",
    note="t(1)（柯西）方差不存在 → 明确标记")
run("expectation", {"dist": "f", "d1": "2", "d2": "2"}, expect_contains="不存在",
    note="F(2,2) 的 E[X] 不存在（需 d2>2）→ 明确标记")

# --- 14. 期望 / 方差的互印（含 g(X) 与自定义分布律） ------------------------
run("expectation", {"dist": "binomial", "n": "10", "p": "1/3", "expr": "x^2"}, expect="40/3",
    note="B(10,1/3) 的 E[X²]=40/3")
run("expectation", {"dist": "normal", "mu": "1", "sigma": "2"}, expect="1",
    note="N(1,4) 的 E[X]=μ=1")
run("variance", {"dist": "normal", "mu": "1", "sigma": "2"}, expect="4",
    note="N(1,4) 的 Var=σ²=4")
run("variance", {"dist": "poisson", "lam": "3"}, expect="3", note="Poisson(3) 的 Var=3")
run("expectation", {"pmf": "1/6,1/6,1/6,1/6,1/6,1/6", "values": "1,2,3,4,5,6"},
    expect="7/2", note="自定义分布律（骰子）E[X]=7/2")
run("variance", {"pmf": "1/6,1/6,1/6,1/6,1/6,1/6", "values": "1,2,3,4,5,6"},
    expect="35/12", note="自定义分布律（骰子）Var=35/12")
run("expectation", {"pmf": "1/6,1/6,1/6,1/6,1/6,1/6", "values": "1,2,3,4,5,6", "expr": "x^2"},
    expect="91/6", note="自定义分布律（骰子）E[X²]=91/6")

# --- 15. 协方差 / 相关系数（离散联合：两枚独立硬币） -----------------------
run("covariance", {"joint_pmf": "1/4,1/4;1/4,1/4", "x_values": "0,1", "y_values": "0,1"},
    expect="0", note="两枚独立硬币（矩阵行=y、列=x，全部 1/4）Cov=0")
run("correlation", {"joint_pmf": "1/4,1/4;1/4,1/4", "x_values": "0,1", "y_values": "0,1"},
    expect="0", note="两枚独立硬币 ρ=0")

# 不独立（相关）的联合分布律：p00=1/8,p10=1/8,p01=1/8,p11=5/8
# X 边缘 = (1/4, 3/4)；Y 边缘 = (1/4, 3/4)；E[XY]=5/8；EX=EY=3/4；Cov=5/8−9/16=1/16
# Var(X)=Var(Y)=3/4−9/16=3/16；ρ=(1/16)/(3/16)=1/3
run("covariance", {"joint_pmf": "1/8,1/8;1/8,5/8", "x_values": "0,1", "y_values": "0,1"},
    expect="1/16", note="不独立联合分布 Cov=1/16")
run("correlation", {"joint_pmf": "1/8,1/8;1/8,5/8", "x_values": "0,1", "y_values": "0,1"},
    expect="1/3", note="不独立联合分布 ρ=1/3")

# 相关系数无定义：X 恒等于常数（列的 X 边缘为 (1, 0)）
expect_error("correlation", {"joint_pmf": "1/2,0;1/2,0", "x_values": "0,1", "y_values": "0,1"},
             note="σ_X=0 时相关系数无定义，应报错", contains="无定义")
# 相关系数无定义：Y 恒等于常数（行的 Y 边缘为 (1, 0)）
expect_error("correlation", {"joint_pmf": "1,0;0,0", "x_values": "0,1", "y_values": "0,1"},
             note="σ_Y=0 时相关系数无定义，应报错", contains="无定义")

# 连续联合：单位正方形上的均匀分布 f(x,y)=1（独立）
run("covariance", {"joint_pdf": "1", "x_lower": "0", "x_upper": "1",
                   "y_lower": "0", "y_upper": "1"},
    expect="0", note="[0,1]² 上均匀分布（独立）Cov=0")
run("correlation", {"joint_pdf": "1", "x_lower": "0", "x_upper": "1",
                    "y_lower": "0", "y_upper": "1"},
    expect="0", note="[0,1]² 上均匀分布 ρ=0")

# --- 16. 联合与边缘分布 + 独立性判定 ---------------------------------------
run("joint_marginal", {"joint_pmf": "1/4,1/4;1/4,1/4", "x_values": "0,1", "y_values": "0,1"},
    note="独立例子：两枚独立硬币的边缘分布与独立性判定（应判独立）")
out_dep = run("joint_marginal", {"joint_pmf": "1/8,1/8;1/8,5/8", "x_values": "0,1", "y_values": "0,1"},
              note="不独立例子：p00=1/8≠P(X=0)P(Y=0)=1/16（应判不独立）")
dep_raw = (out_dep or {}).get("raw") or (out_dep or {}).get("extra") or {}
print(f"    不独立例子的判定 = {dep_raw.get('verdict')}；理由：{dep_raw.get('reason')}")
run("joint_marginal", {"joint_pdf": "1", "x_lower": "0", "x_upper": "1",
                       "y_lower": "0", "y_upper": "1"},
    note="连续型独立例子：单位正方形均匀分布的边缘密度")
run("joint_marginal", {"joint_pdf": "x + y", "x_lower": "0", "x_upper": "1",
                       "y_lower": "0", "y_upper": "1"},
    note="连续型不独立例子：f(x,y)∝x+y（未归一化，用于检验独立性判定）")

# --- 17. 自定义分布律 distribution_from_pmf（骰子） ------------------------
run("distribution_from_pmf", {"pmf": "1/6,1/6,1/6,1/6,1/6,1/6", "values": "1,2,3,4,5,6"},
    note="骰子：归一化 + 分布函数 + E=7/2 + Var=35/12")
expect_error("distribution_from_pmf", {"pmf": "1/6,1/6,1/6", "values": "1,2,3"},
             note="分布律之和=1/2≠1 应报错", contains="1/2")
run("distribution_from_pmf", {"pmf": "1/4,1/4,1/2", "values": "-1,0,2"},
    note="自定义分布律（含负数取值）：E=3/4")

# --- 18. 矩汇总 distribution_moments ---------------------------------------
run("distribution_moments", {"dist": "binomial", "n": "10", "p": "1/3", "k": "3"},
    note="B(10,1/3) 的汇总视图：E、E[X²]、Var、E[X³]")
run("distribution_moments", {"dist": "normal", "mu": "0", "sigma": "1"},
    note="N(0,1) 的汇总视图（含偏度/峰度）")
run("distribution_moments", {"dist": "poisson", "lam": "2"},
    note="Poisson(2) 的汇总视图")

# --- 19. 可选算子：矩母函数 / 中心极限定理 ---------------------------------
run("moment_generating", {"dist": "normal", "mu": "0", "sigma": "1"},
    note="N(0,1) 的矩母函数 M(t)=e^(t²/2)")
run("moment_generating", {"dist": "binomial", "n": "10", "p": "1/3"},
    note="B(10,1/3) 的矩母函数 M(t)=(1-p+pe^t)^n")
run("moment_generating", {"dist": "cauchy"},
    note="柯西分布矩母函数不存在（应说明原因）")


def _binom_cdf(k, n, p):
    """精确二项分布函数 P(S ≤ k)，纯 Python，用于给 CLT 做独立对照。"""
    total = 0.0
    for i in range(0, int(k) + 1):
        total += math.comb(n, i) * p ** i * (1 - p) ** (n - i)
    return total


# 注意：ops.py 的加载顺序里 mathkit.probability_extra 在我的模块之后，
# 它注册的 central_limit 覆盖了本模块的同名算子，所以这里按"实际生效的那个"的
# target 语法（"P(X̄ <= x)"）来测，才算走通 Node 侧真实路径。
run("central_limit", {"n": "100", "dist": "binomial", "size": "1", "p": "1/3",
                      "target": "P(X̄ <= 0.4)"},
    expect_approx=_binom_cdf(40, 100, 1 / 3), tol=0.01,
    note="CLT：B(1,1/3) 总体、n=100 的 P(X̄≤0.4)=P(S≤40) 正态近似 vs 精确二项值")
run("central_limit", {"n": "100", "dist": "binomial", "size": "1", "p": "1/3",
                      "target": "P(S_n <= 40)", "statistic": "sum"},
    expect_approx=_binom_cdf(40, 100, 1 / 3), tol=0.01,
    note="CLT：同上但用 statistic=sum 写 P(S_n≤40)")
run("central_limit", {"n": "36", "dist": "custom", "mu": "1", "var": "4",
                      "target": "P(X̄ <= 1.5)"},
    expect_approx=0.9331927987311419, tol=1e-6,
    note="CLT：自定义总体 μ=1,σ²=4,n=36 的 P(X̄≤1.5)=Φ(1.5)≈0.9332")

# --- 20. 参数越界 / 未知名称 必须报错 --------------------------------------
expect_error("distribution", {"dist": "normal", "mu": "0", "sigma": "-1", "kind": "pdf"},
             note="σ=-1 越界应报错", contains="-1")
expect_error("distribution", {"dist": "binomial", "n": "10", "p": "2", "kind": "pmf"},
             note="p=2 越界应报错", contains="2")
expect_error("distribution", {"dist": "exponential", "lam": "0", "kind": "pdf"},
             note="λ=0 越界应报错")
expect_error("distribution", {"dist": "不存在的分布", "kind": "pdf"},
             note="未知分布名应报错", contains="不存在的分布")
expect_error("probability", {"kind": "不存在的分支", "pa": "1/2"},
             note="未知 kind 应报错并列出支持值", contains="不存在的分支")
expect_error("probability", {"kind": "bayes", "prior": "0.3,0.5", "likelihood": "0.9"},
             note="prior 与 likelihood 个数不一致应报错")
expect_error("correlation", {"joint_pmf": "1,0;1,0", "x_values": "0,1", "y_values": "0,1"},
             note="联合分布律之和=2 应报错")

# ===========================================================================
print("\n" + "=" * 78)
print("汇总")
print("=" * 78)

n_total = len(RESULTS)
n_ok_status = sum(1 for r in RESULTS if r["status"] == "ok")
n_partial = sum(1 for r in RESULTS if r["status"] == "partial")
n_unsolved = sum(1 for r in RESULTS if r["status"] == "unsolved")
n_error = sum(1 for r in RESULTS if r["status"] == "error")
n_traceback = sum(1 for r in RESULTS if r["status"] == "traceback")
n_failed_check = sum(1 for r in RESULTS if r["ok"] is False)

print(f"总用例数                : {n_total}")
print(f"status=ok               : {n_ok_status}")
print(f"status=partial          : {n_partial}")
print(f"status=unsolved         : {n_unsolved}")
print(f"status=error            : {n_error}")
print(f"status=traceback        : {n_traceback}")
print(f"校验未通过的用例        : {n_failed_check}")

vstats = {}
for r in RESULTS:
    if r.get("vstatus"):
        vstats[r["vstatus"]] = vstats.get(r["vstatus"], 0) + 1
print(f"verification.status 分布: {vstats}")

n_with_methods = sum(1 for r in RESULTS if r.get("n_methods"))
n_all_pass = sum(1 for r in RESULTS if r.get("n_methods") and r["n_pass"] == r["n_methods"])
print(f"带独立验证方法的用例    : {n_with_methods}（其中全部方法通过：{n_all_pass}）")

print("\n--- 逐条状态 ---")
for r in RESULTS:
    flag = "" if r.get("ok") is not False else "   <<< CHECK FAILED"
    print(f"  [{r['status']:>9}] {r.get('op', '?'):<22} {r['label'][:74]}{flag}")

if n_traceback:
    print("\n!!! 存在 traceback")
if n_failed_check:
    print("\n!!! 存在校验未通过的用例")

sys.exit(0)
