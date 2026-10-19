# -*- coding: utf-8 -*-
"""probability_extra.py 自检脚本（统计推断补充：点估计 / 区间估计 / 假设检验 / 抽样分布 / 中心极限定理）。

走的是 Node 侧同一条调用路径：``engine.execute(算子名, 参数)``（参数先经 engine 的别名归一化）。

每条用例都断言：
* status 不是 error（参数越界用例单独标记为「期望报错」，不计入失败）；
* 关键结果字段的值（估计量、区间端点、检验统计量、临界值、p 值、结论）；
* verification.methods 中的**独立复核**逐条通过；
* 反例/一致性检查：临界值法与 p 值法结论必须一致、区间宽度随置信水平单调递增；
* 期望值全部用 scipy 独立重算（不引用被测代码的任何中间结果）。

运行：python _selftest_probability_extra.py
"""
import sys
import os
import json
import math

sys.path.insert(0, r"E:\dsh-math\engine")
from mathkit import engine as E  # noqa: E402
from mathkit import ops as _ops  # noqa: E402,F401  触发 handler 模块注册（worker.py 的真实做法）

from scipy import stats as S  # noqa: E402  独立重算期望值

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

_MISSING = object()
TALLY = {"ok": 0, "unsolved": 0, "error": 0, "expected_error": 0,
         "methods_pass": 0, "methods_false": 0, "methods_none": 0, "case_fail": 0}
OUTS = {}


def _short(obj, limit=700):
    text = json.dumps(obj, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "…(截断)"


def _dig(obj, path, default=_MISSING):
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _verify_of(out):
    """顶层 verification 是规范位置；result.verification 是同一份内容的镜像。"""
    ver = {}
    if isinstance(out.get("verification"), dict):
        ver.update(out["verification"])
    nested = (out.get("result") or {}).get("verification")
    if isinstance(nested, dict):
        for k, v in nested.items():
            ver.setdefault(k, v)
    return ver


def _methods_of(out):
    ms = _verify_of(out).get("methods")
    return ms if isinstance(ms, list) else []


def _derived_passed(out):
    """优先用算子自己写进 verification 的 passed；缺失时才用「所有独立复核是否都通过」派生。"""
    ver = _verify_of(out)
    if isinstance(ver, dict) and ver.get("passed") is not None:
        return bool(ver["passed"])
    vals = [m.get("passed") for m in _methods_of(out) if isinstance(m, dict)]
    if not vals:
        return None
    if any(v is None for v in vals):
        return None
    return all(bool(v) for v in vals)


def _num_of(value):
    try:
        return float(str(value))
    except Exception:  # noqa: BLE001
        return None


def _check_path(out, path, spec):
    problems = []
    val = _dig(out, path, _MISSING)
    if val is _MISSING:
        return [f"结果缺少字段 {path}（实际顶层键：{sorted((out.get('result') or {}))[:12]}）"]
    text = str(val)
    if "eq" in spec and text != str(spec["eq"]):
        problems.append(f"{path} 期望 {spec['eq']!r}，实际 {text!r}")
    if "contains" in spec and spec["contains"] not in text:
        problems.append(f"{path} 应包含 {spec['contains']!r}，实际 {text!r}")
    if "truthy" in spec and not val:
        problems.append(f"{path} 不应为空，实际 {text!r}")
    if "approx" in spec:
        target, tol = spec["approx"]
        got = _num_of(val)
        if got is None:
            problems.append(f"{path} 不是可比较的数值：{text!r}")
        elif abs(got - float(target)) > tol:
            problems.append(f"{path} ≈ {got!r}，期望 {target!r}（容差 {tol}）")
    return problems


def run_case(case):
    title = case["title"]
    op = case.get("op", "point_estimate")
    args = case["args"]
    out = E.execute(op, args)
    OUTS[title] = out
    res = out.get("result") or {}
    ver = _verify_of(out)
    methods = _methods_of(out)
    status = out.get("status")
    passed = res.get("passed", _MISSING)
    if passed is _MISSING:
        passed = _derived_passed(out)

    print("=" * 100)
    print(f"[{title}] {op} " + _short(args, 260))
    print(f"  op: {out.get('operation')} | status: {status} | success: {out.get('success')} | passed: {passed}")
    print(f"  verification.level: {ver.get('level')} | verification.status: {ver.get('status')}")
    print(f"  verification.method: {_short(ver.get('method') or [m.get('method') for m in methods], 500)}")
    print(f"  verification.methods(独立复核): {_short([{'method': m.get('method'), 'passed': m.get('passed')} for m in methods], 800)}")
    print(f"  verification.evidence: {_short(ver.get('evidence') or [m.get('evidence') for m in methods], 700)}")
    print(f"  unhandled_conditions: {_short(ver.get('unhandled_conditions'), 300)}")
    print(f"  cannot_verify_reason: {ver.get('cannot_verify_reason')}")
    print(f"  result(摘要): {_short(res, 600)}")
    print(f"  conditions: {_short(out.get('conditions'), 400)}")
    print(f"  warnings: {_short(out.get('warnings'), 400)}")

    problems = []
    if case.get("expect_error"):
        TALLY["expected_error"] += 1
        if status != "error":
            problems.append(f"期望参数越界报错，实际 status={status}")
        elif (out.get("error") or {}).get("kind") != "invalid_input":
            problems.append(f"期望 error.kind=invalid_input，实际 {(out.get('error') or {}).get('kind')}")
        else:
            print(f"  error(期望): {_short(out.get('error'), 200)}")
    else:
        if status == "error":
            problems.append(f"status=error（不允许）：{_short(out.get('error'), 300)}")
        if case.get("status") and status != case["status"]:
            problems.append(f"status 期望 {case['status']}，实际 {status}")
        if case.get("level_in") is not None and ver.get("level") not in case["level_in"]:
            problems.append(f"verification.level 期望属于 {case['level_in']}，实际 {ver.get('level')}")
        if case.get("min_methods") and len(methods) < case["min_methods"]:
            problems.append(f"独立复核方法数应 ≥ {case['min_methods']}，实际 {len(methods)}")
        if case.get("all_methods_passed") and _derived_passed(out) is not True:
            problems.append(f"并非所有独立复核方法都通过：{_short([{'m': m.get('method'), 'p': m.get('passed')} for m in methods], 400)}")
        for path, spec in (case.get("paths") or {}).items():
            problems += _check_path(out, path, spec)
        blob = json.dumps({"result": res, "verification": ver}, ensure_ascii=False, default=str)
        for token in case.get("contains") or []:
            if token not in blob:
                problems.append(f"结果/复核文本应包含 {token!r}")
        for token in case.get("not_contains") or []:
            if token in blob:
                problems.append(f"结果/复核文本不应包含 {token!r}")
        cond_blob = json.dumps(out.get("conditions") or [], ensure_ascii=False)
        for token in case.get("conditions_contains") or []:
            if token not in cond_blob:
                problems.append(f"conditions 应包含 {token!r}，实际 {cond_blob[:200]}")
        warn_blob = json.dumps(out.get("warnings") or [], ensure_ascii=False)
        for token in case.get("warnings_contains") or []:
            if token not in warn_blob:
                problems.append(f"warnings 应包含 {token!r}，实际 {warn_blob[:200]}")
        unsolved_blob = json.dumps({"result": res, "method": out.get("method"), "error": out.get("error")},
                                   ensure_ascii=False, default=str)
        for token in case.get("unsolved_contains") or []:
            if status != "unsolved":
                problems.append(f"期望 unsolved，实际 {status}")
            elif token not in unsolved_blob:
                problems.append(f"unsolved 说明应包含 {token!r}，实际 {unsolved_blob[:200]}")
        if case.get("custom"):
            problems += list(case["custom"](out, res, ver))

    if status == "ok":
        TALLY["ok"] += 1
    elif status == "unsolved":
        TALLY["unsolved"] += 1
    elif not case.get("expect_error"):
        TALLY["error"] += 1
    if not case.get("expect_error"):
        if passed is True:
            TALLY["methods_pass"] += 1
        elif passed is False:
            TALLY["methods_false"] += 1
        else:
            TALLY["methods_none"] += 1

    if problems:
        TALLY["case_fail"] += 1
        print("  >>> 用例不通过：")
        for p in problems:
            print("      - " + p)
    else:
        print("  >>> 用例通过")
    return problems


# ===========================================================================
# 期望值（全部用 scipy 独立重算）
# ===========================================================================
Z_975 = float(S.norm.ppf(0.975))
Z_95 = float(S.norm.ppf(0.95))                            # 单侧 α=0.05 的上分位数 z_α
SIG_KNOWN_HW = Z_975 * 2.0 / math.sqrt(5)          # σ=2, n=5 时的半宽
VAR_LO = 10.0 / float(S.chi2.ppf(0.975, 4))        # s²=5/2, df=4
VAR_HI = 10.0 / float(S.chi2.ppf(0.025, 4))
PHAT_SE = math.sqrt(0.4 * 0.6 / 100)
T_STAT = (3.0 - 2.0) / (math.sqrt(2.5) / math.sqrt(5))   # 样本 1..5 的 t 统计量
T_CRIT = float(S.t.ppf(0.95, 4))
T_P = float(S.t.sf(T_STAT, 4))
CHI2_STAT = 4 * 2.5
CHI2_CRIT = float(S.chi2.ppf(0.95, 4))
CHI2_P = float(S.chi2.sf(CHI2_STAT, 4))
F_STAT = 4.0 / 1.0
F_HI = float(S.f.ppf(0.975, 9, 9))
F_LO = float(S.f.ppf(0.025, 9, 9))
F_P = 2 * min(float(S.f.sf(F_STAT, 9, 9)), float(S.f.cdf(F_STAT, 9, 9)))
BINOM_EXACT = float(S.binom.cdf(55, 100, 0.5))     # P(ΣXᵢ ≤ 55) = P(X̄ ≤ 0.55)
POIS_EXACT = float(S.poisson.cdf(110, 100))        # ΣXᵢ ~ Poisson(50·2)，P(S ≤ 110)

CASES = [
    # ---------------- point_estimate ----------------
    dict(title="点估计 正态矩估计（μ̂=3, σ̂²=2）", op="point_estimate",
         args=dict(samples="1,2,3,4,5", method="moment", dist="normal"),
         level_in={"symbolic"}, all_methods_passed=True, min_methods=2,
         paths={"result.estimators.μ̂.value": {"eq": "3"},
                "result.estimators.σ̂².value": {"eq": "2"},
                "result.moment_equations": {"truthy": True},
                "result.sample_statistics.s2_unbiased": {"eq": "5/2"}},
         conditions_contains=["n 为正整数"],
         contains=["矩估计"]),
    dict(title="点估计 正态 MLE（似然/对数似然/得分方程 + S² 对比）", op="point_estimate",
         args=dict(samples="1,2,3,4,5", method="mle", dist="normal"),
         level_in={"symbolic"}, all_methods_passed=True, min_methods=2,
         paths={"result.estimators.μ̂.value": {"eq": "3"},
                "result.estimators.σ̂².value": {"eq": "2"},
                "result.other_estimators.S².value": {"eq": "5/2"},
                "result.likelihood": {"contains": "exp"},
                "result.log_likelihood": {"contains": "log"},
                "result.score_equations": {"truthy": True}},
         conditions_contains=["σ > 0", "独立同分布"],
         contains=["极大似然"]),
    dict(title="点估计 正态 both（矩估计与 MLE 都给）", op="point_estimate",
         args=dict(samples="1,2,3,4,5", method="both", dist="normal"),
         level_in={"symbolic"}, all_methods_passed=True,
         paths={"result.likelihood": {"truthy": True}, "result.moment_equations": {"truthy": True}}),
    dict(title="点估计 指数分布 MLE（λ̂=5/17、无偏 λ̃=4/17）", op="point_estimate",
         args=dict(samples="1.5,2.5,3,4.5,5.5", method="mle", dist="exponential"),
         level_in={"symbolic"}, all_methods_passed=True,
         paths={"result.estimators.λ̂.value": {"eq": "5/17"},
                "result.other_estimators.λ̃.value": {"eq": "4/17"}},
         conditions_contains=["λ > 0"]),
    dict(title="点估计 指数分布矩估计（与 MLE 同值）", op="point_estimate",
         args=dict(samples="1.5,2.5,3,4.5,5.5", method="moment", dist="exponential"),
         level_in={"symbolic"}, all_methods_passed=True,
         paths={"result.estimators.λ̂.value": {"eq": "5/17"}}),
    dict(title="点估计 均匀分布 both（MLE 取次序统计量）", op="point_estimate",
         args=dict(samples="1,2,2,3,4,5", method="both", dist="uniform"),
         level_in={"symbolic"}, all_methods_passed=True, min_methods=2,
         paths={"result.other_estimators.â_MLE.value": {"eq": "1"},
                "result.other_estimators.b̂_MLE.value": {"eq": "5"},
                "result.other_estimators.â_无偏.value": {"eq": "3/7"},
                "result.other_estimators.b̂_无偏.value": {"eq": "39/7"},
                "result.estimators.â_矩.value": {"truthy": True}},
         warnings_contains=["边界"],
         contains=["边界论证"]),
    dict(title="点估计 泊松 MLE（λ̂=31/8）", op="point_estimate",
         args=dict(samples="3,1,4,1,5,9,2,6", method="mle", dist="poisson"),
         level_in={"symbolic"}, all_methods_passed=True,
         paths={"result.estimators.λ̂.value": {"eq": "31/8"}}),
    dict(title="点估计 二项分布 MLE（p̂=2/5，m=10）", op="point_estimate",
         args=dict(samples="2,4,5,3,6", method="mle", dist="binomial", size=10),
         level_in={"symbolic"}, all_methods_passed=True,
         paths={"result.estimators.p̂.value": {"eq": "2/5"}}),
    dict(title="点估计 样本量不足 → unsolved（不是 error）", op="point_estimate",
         args=dict(samples="3", method="both", dist="normal"),
         status="unsolved",
         unsolved_contains=["样本"]),
    dict(title="点估计 未知分布 → 期望报错", op="point_estimate",
         args=dict(samples="1,2,3", dist="no_such"), expect_error=True),

    # ---------------- interval_estimate ----------------
    dict(title="区间估计 σ 已知 0.95（z 分位数，端点精确符号式）", op="interval_estimate",
         args=dict(samples="1,2,3,4,5", sigma="2", confidence="0.95", kind="mean"),
         level_in={"symbolic"}, all_methods_passed=True, min_methods=2,
         paths={"result.lower_approx": {"approx": (3 - SIG_KNOWN_HW, 1e-9)},
                "result.upper_approx": {"approx": (3 + SIG_KNOWN_HW, 1e-9)},
                "result.exact_symbolic_endpoints": {"truthy": True},
                "result.distribution_used": {"contains": "正态"},
                "result.critical_value": {"contains": "z"}},
         conditions_contains=["已知"],
         contains=["erfinv"]),
    dict(title="区间估计 σ 已知 0.99（比 0.95 更宽）", op="interval_estimate",
         args=dict(samples="1,2,3,4,5", sigma="2", confidence="0.99", kind="mean"),
         level_in={"symbolic"}, all_methods_passed=True,
         paths={"result.lower_approx": {"approx": (3 - float(S.norm.ppf(0.995)) * 2 / math.sqrt(5), 1e-9)}}),
    dict(title="区间估计 σ 未知 0.95（t(n-1)，自由度必须写明）", op="interval_estimate",
         args=dict(samples="1,2,3,4,5", confidence="0.95", kind="mean"),
         level_in={"numeric", "symbolic"}, all_methods_passed=True,
         paths={"result.degrees_of_freedom": {"eq": 4},
                "result.distribution_used": {"contains": "t"},
                "result.lower_approx": {"approx": (3 - float(S.t.ppf(0.975, 4)) * math.sqrt(2.5) / math.sqrt(5), 1e-6)}},
         conditions_contains=["自由度"]),
    dict(title="区间估计 σ 未知 0.99", op="interval_estimate",
         args=dict(samples="1,2,3,4,5", confidence="0.99", kind="mean"),
         level_in={"numeric", "symbolic"}, all_methods_passed=True,
         paths={"result.degrees_of_freedom": {"eq": 4}}),
    dict(title="区间估计 方差（χ²(n-1)，两个分位点）", op="interval_estimate",
         args=dict(samples="1,2,3,4,5", confidence="0.95", kind="variance"),
         level_in={"numeric", "symbolic"}, all_methods_passed=True,
         paths={"result.lower_approx": {"approx": (VAR_LO, 1e-6)},
                "result.upper_approx": {"approx": (VAR_HI, 1e-6)},
                "result.degrees_of_freedom": {"eq": 4},
                "result.distribution_used": {"contains": "χ"}},
         conditions_contains=["χ²"]),
    dict(title="区间估计 比例（n=100, 成功 40）", op="interval_estimate",
         args=dict(n=100, successes=40, confidence="0.95", kind="proportion"),
         level_in={"numeric", "symbolic"}, all_methods_passed=True,
         paths={"result.lower_approx": {"approx": (0.4 - Z_975 * PHAT_SE, 5e-3)},
                "result.upper_approx": {"approx": (0.4 + Z_975 * PHAT_SE, 5e-3)}},
         conditions_contains=["大样本"]),
    dict(title="区间估计 置信水平越界 → 期望报错", op="interval_estimate",
         args=dict(samples="1,2,3", confidence="1.5"), expect_error=True),

    # ---------------- hypothesis_test ----------------
    dict(title="z 检验 双侧（统计量 1，p≈0.3173，不拒绝 H₀）", op="hypothesis_test",
         args=dict(kind="z_test", n=25, mean="3.2", sigma="1", mu0="3", alpha="0.05", side="two"),
         level_in={None, "symbolic", "numeric"}, all_methods_passed=True, min_methods=2,
         paths={"result.statistic.approx": {"approx": (1.0, 1e-12)},
                "result.critical_value_approx": {"approx": (Z_975, 1e-9)},
                "result.p_value_approx": {"approx": (2 * float(S.norm.sf(1.0)), 1e-9)},
                "result.decision": {"eq": "不拒绝 H₀"},
                "result.consistency_check.two_routes_agree": {"truthy": True},
                "result.conclusion_caveat": {"contains": "不等于证明"}},
         contains=["拒绝域"]),
    dict(title="z 检验 右侧（统计量 2.5 > z_α，拒绝 H₀）", op="hypothesis_test",
         args=dict(kind="z_test", n=25, mean="3.5", sigma="1", mu0="3", alpha="0.05", side="right"),
         all_methods_passed=True,
         paths={"result.statistic.approx": {"approx": (2.5, 1e-12)},
                "result.decision": {"eq": "拒绝 H₀"},
                "result.consistency_check.two_routes_agree": {"truthy": True}},
         warnings_contains=[]),
    dict(title="z 检验 左侧（统计量 -2.5 < -z_α，拒绝 H₀；临界值必须为负）", op="hypothesis_test",
         args=dict(kind="z_test", n=25, mean="2.5", sigma="1", mu0="3", alpha="0.05", side="left"),
         all_methods_passed=True,
         paths={"result.statistic.approx": {"approx": (-2.5, 1e-12)},
                "result.critical_value_approx": {"approx": (-Z_95, 1e-9)},
                "result.p_value_approx": {"approx": (float(S.norm.cdf(-2.5)), 1e-9)},
                "result.decision": {"eq": "拒绝 H₀"},
                "result.consistency_check.two_routes_agree": {"truthy": True}},
         contains=["-1.64485"]),
    dict(title="t 检验 单侧右（samples 1..5, μ0=2）", op="hypothesis_test",
         args=dict(kind="t_test", samples="1,2,3,4,5", mu0="2", alpha="0.05", side="right"),
         all_methods_passed=True,
         paths={"result.statistic.approx": {"approx": (T_STAT, 1e-9)},
                "result.critical_value_approx": {"approx": (T_CRIT, 1e-9)},
                "result.p_value_approx": {"approx": (T_P, 1e-9)},
                "result.degrees_of_freedom": {"eq": 4},
                "result.decision": {"eq": "不拒绝 H₀"},
                "result.consistency_check.two_routes_agree": {"truthy": True}},
         conditions_contains=["正态总体"]),
    dict(title="t 检验 单侧左（同数据另一侧，p≈0.8849）", op="hypothesis_test",
         args=dict(kind="t_test", samples="1,2,3,4,5", mu0="2", alpha="0.05", side="left"),
         all_methods_passed=True,
         paths={"result.p_value_approx": {"approx": (float(S.t.cdf(T_STAT, 4)), 1e-9)},
                "result.decision": {"eq": "不拒绝 H₀"}}),
    dict(title="χ² 检验 右侧（S²=5/2, σ₀=1 → 统计量 10 > 9.4877，拒绝 H₀）", op="hypothesis_test",
         args=dict(kind="chi2_test", samples="1,2,3,4,5", sigma0="1", alpha="0.05", side="right"),
         all_methods_passed=True,
         paths={"result.statistic.approx": {"approx": (CHI2_STAT, 1e-9)},
                "result.critical_value_approx": {"approx": (CHI2_CRIT, 1e-9)},
                "result.p_value_approx": {"approx": (CHI2_P, 1e-9)},
                "result.degrees_of_freedom": {"eq": 4},
                "result.decision": {"eq": "拒绝 H₀"},
                "result.consistency_check.two_routes_agree": {"truthy": True}},
         conditions_contains=["自由度"]),
    dict(title="χ² 检验 双侧（两个临界值，拒绝域分两尾）", op="hypothesis_test",
         args=dict(kind="chi2_test", samples="1,2,3,4,5", sigma0="1", alpha="0.05", side="two"),
         all_methods_passed=True,
         paths={"result.critical_value_approx": {"approx": (float(S.chi2.ppf(0.975, 4)), 1e-9)},
                "result.rejection_region": {"contains": "或"},
                "result.consistency_check.two_routes_agree": {"truthy": True}}),
    dict(title="F 检验 双侧（var1=4, var2=1, n1=n2=10）", op="hypothesis_test",
         args=dict(kind="f_test", n1=10, n2=10, var1="4", var2="1", alpha="0.05", side="two"),
         all_methods_passed=True,
         paths={"result.statistic.approx": {"approx": (F_STAT, 1e-9)},
                "result.critical_value_approx": {"approx": (F_HI, 1e-9)},
                "result.p_value_approx": {"approx": (F_P, 1e-9)},
                "result.decision": {"eq": "不拒绝 H₀"},
                "result.consistency_check.two_routes_agree": {"truthy": True},
                "result.degrees_of_freedom": {"eq": "(9, 9)"}}),
    dict(title="比例检验 双侧（n=100, 成功 40, p₀=0.5，统计量 -2）", op="hypothesis_test",
         args=dict(kind="proportion_test", n=100, successes=40, mu0="0.5", alpha="0.05", side="two"),
         all_methods_passed=True,
         paths={"result.statistic.approx": {"approx": (-2.0, 1e-9)},
                "result.p_value_approx": {"approx": (2 * float(S.norm.sf(2.0)), 1e-9)},
                "result.decision": {"eq": "拒绝 H₀"},
                "result.consistency_check.two_routes_agree": {"truthy": True}},
         conditions_contains=["大样本"]),
    dict(title="比例检验 小样本警告（n=20, 成功 1, p₀=0.2 → n·p₀=4<5）", op="hypothesis_test",
         args=dict(kind="proportion_test", n=20, successes=1, mu0="0.2", alpha="0.05"),
         all_methods_passed=True,
         warnings_contains=["大样本条件不满足"]),
    dict(title="检验 α 越界 → 期望报错", op="hypothesis_test",
         args=dict(kind="z_test", n=25, mean="3.2", sigma="1", mu0="3", alpha="1.5"),
         expect_error=True),
    dict(title="检验 未知类型 → 期望报错", op="hypothesis_test",
         args=dict(kind="no_such_test", samples="1,2,3"), expect_error=True),

    # ---------------- sampling_distribution ----------------
    dict(title="抽样分布 样本均值（σ²/n 与标准化形式）", op="sampling_distribution",
         args=dict(dist="mean", n=10),
         level_in={"symbolic"}, all_methods_passed=True,
         paths={"result.parameters.variance": {"contains": "10"},
                "result.standardized": {"contains": "N(0,1)"}},
         conditions_contains=["正态总体"]),
    dict(title="抽样分布 样本方差（(n-1)S²/σ² ~ χ²(n-1)）", op="sampling_distribution",
         args=dict(dist="var", n=10),
         level_in={"symbolic"}, all_methods_passed=True,
         paths={"result.standardized": {"contains": "χ²(9)"},
                "result.parameters.degrees_of_freedom": {"eq": 9}},
         conditions_contains=["自由度"]),
    dict(title="抽样分布 t 分布（方差 df/(df-2)）", op="sampling_distribution",
         args=dict(dist="t", n=10),
         level_in={"symbolic"}, all_methods_passed=True,
         paths={"result.parameters.variance": {"contains": "5/4"},
                "result.parameters.degrees_of_freedom": {"eq": 10}}),
    dict(title="抽样分布 F 分布（df₁=5, df₂=7）", op="sampling_distribution",
         args=dict(dist="f", df1=5, df2=7),
         level_in={"symbolic"}, all_methods_passed=True,
         paths={"result.parameters.df1": {"eq": 5}, "result.parameters.df2": {"eq": 7}}),
    dict(title="抽样分布 非法 n=1 的方差分布 → 期望报错", op="sampling_distribution",
         args=dict(dist="var", n=1), expect_error=True),

    # ---------------- central_limit ----------------
    dict(title="CLT 二项总体（n=100, m=1, p=1/2, P(X̄≤0.55)）", op="central_limit",
         args=dict(n=100, dist="binomial", size=1, p="1/2", target="P(X̄ <= 0.55)"),
         level_in={"numeric"}, all_methods_passed=True, min_methods=2,
         paths={"result.approximate_probability_approx": {"approx": (0.864333, 1e-4)},
                "result.exact_comparison.value": {"approx": (BINOM_EXACT, 1e-9)},
                "result.exact_comparison.deviation": {"approx": (abs(0.864333 - BINOM_EXACT), 1e-3)},
                "result.continuity_correction.needed": {"truthy": True},
                "result.continuity_correction.applied": {"truthy": True}},
         conditions_contains=["近似", "n 足够大"]),
    dict(title="CLT 泊松总体（n=50, λ=2, P(S≤110)）", op="central_limit",
         args=dict(n=50, dist="poisson", lam="2", target="P(S_n <= 110)"),
         level_in={"numeric"}, all_methods_passed=True,
         paths={"result.approximate_probability_approx": {"approx": (0.853141, 1e-4)},
                "result.exact_comparison.value": {"approx": (POIS_EXACT, 1e-9)},
                "result.standardized": {"contains": "Sₙ"}},
         conditions_contains=["连续性修正"]),
    dict(title="CLT 均匀总体（n=30, U(0,1), P(S≤17)，Irwin–Hall 精确对照）", op="central_limit",
         args=dict(n=30, dist="uniform", xmin="0", xmax="1", target="P(S_n <= 17)"),
         level_in={"numeric"}, all_methods_passed=True,
         paths={"result.parent_distribution.discrete": {"eq": False},
                "result.continuity_correction.applied": {"eq": False},
                "result.exact_comparison.deviation": {"truthy": True}},
         contains=["Irwin"]),
    dict(title="CLT 指数总体（n=30, λ=2, P(S≤20)）", op="central_limit",
         args=dict(n=30, dist="exponential", lam="2", target="P(S_n <= 20)"),
         level_in={"numeric"}, all_methods_passed=True,
         paths={"result.exact_comparison.value": {"approx": (float(S.gamma.cdf(20, 30, scale=0.5)), 1e-9)}}),
    dict(title="CLT 自定义分布（mu/var，无精确对照）", op="central_limit",
         args=dict(n=36, dist="custom", mu="1", var="4", target="P(X̄ <= 1.5)"),
         level_in={"numeric"},
         paths={"result.approximate_probability_approx": {"approx": (float(S.norm.cdf(1.5)), 1e-9)},
                "result.standardization_used.statistic": {"eq": "mean"}},
         contains=["无法给出精确分布"]),
    dict(title="CLT n 太小警告（n=10）", op="central_limit",
         args=dict(n=10, dist="binomial", size=1, p="1/2", target="P(S_n <= 6)"),
         level_in={"numeric"}, warnings_contains=["n = 10 < 30"]),
    dict(title="CLT 未知分布 → 期望报错", op="central_limit",
         args=dict(n=30, dist="no_such", target="P(S_n <= 1)"), expect_error=True),
]


# ===========================================================================
# 交叉用例（需要比较两次调用的结果：单调性、两法一致性）
# ===========================================================================
def cross_checks():
    problems = []
    print("=" * 100)
    print("### 交叉验证：置信区间宽度随置信水平单调递增")
    t95 = "区间估计 σ 已知 0.95（z 分位数，端点精确符号式）"
    t99 = "区间估计 σ 已知 0.99（比 0.95 更宽）"
    l95 = _num_of(_dig(OUTS[t95], "result.length_approx"))
    l99 = _num_of(_dig(OUTS[t99], "result.length_approx"))
    print(f"  0.95 宽度 = {l95} | 0.99 宽度 = {l99}")
    if not (l95 is not None and l99 is not None and l99 > l95):
        problems.append(f"宽度未随置信水平递增：0.95→{l95}, 0.99→{l99}")
    else:
        print("  >>> 单调递增，通过")

    print("### 交叉验证：σ 未知时用 t(4) 而非 z（同一置信水平下 t 区间更宽）")
    t_unknown = _num_of(_dig(OUTS["区间估计 σ 未知 0.95（t(n-1)，自由度必须写明）"], "result.length_approx"))
    print(f"  t 区间宽度 = {t_unknown} | z 区间宽度 = {l95}（σ=2）")
    if not (t_unknown is not None and l95 is not None and t_unknown > l95):
        problems.append("t 区间应比同置信水平的 z 区间宽（t 分位数 > z 分位数、且 S 估计 σ）")
    else:
        print("  >>> t 区间更宽，通过")

    print("### 交叉验证：CLT 二项近似 vs scipy 精确二项分布（独立重算）")
    clt = OUTS["CLT 二项总体（n=100, m=1, p=1/2, P(X̄≤0.55)）"]
    approx = _num_of(_dig(clt, "result.approximate_probability_approx"))
    dev = _num_of(_dig(clt, "result.exact_comparison.deviation"))
    print(f"  近似 = {approx} | scipy 精确 P(ΣXᵢ≤55) = {BINOM_EXACT} | evidence 偏差 = {dev}")
    if not (approx is not None and abs(approx - BINOM_EXACT) < 0.01):
        problems.append(f"CLT 近似与精确值偏差过大：{approx} vs {BINOM_EXACT}")
    elif dev is None or abs(dev - abs(approx - BINOM_EXACT)) > 1e-6:
        problems.append(f"evidence 里的偏差 {dev} 与独立重算 {abs(approx - BINOM_EXACT)} 不一致")
    else:
        print("  >>> 偏差与独立重算一致且 < 0.01，通过")

    print("### 交叉验证：z 检验临界值/p 值与 scipy 独立重算")
    z = OUTS["z 检验 双侧（统计量 1，p≈0.3173，不拒绝 H₀）"]
    crit = _num_of(_dig(z, "result.critical_value_approx"))
    pv = _num_of(_dig(z, "result.p_value_approx"))
    print(f"  临界值 = {crit} (scipy {Z_975}) | p = {pv} (scipy {2 * float(S.norm.sf(1.0))})")
    if not (crit is not None and abs(crit - Z_975) < 1e-9):
        problems.append("z 临界值与 scipy 独立重算不一致")
    if not (pv is not None and abs(pv - 2 * float(S.norm.sf(1.0))) < 1e-9):
        problems.append("z 检验 p 值与 scipy 独立重算不一致")
    if not problems:
        print("  >>> 一致，通过")

    print("### 交叉验证：χ² 检验两条途径（临界值法 / p 值法）结论一致")
    chi = OUTS["χ² 检验 右侧（S²=5/2, σ₀=1 → 统计量 10 > 9.4877，拒绝 H₀）"]
    cons = _dig(chi, "result.consistency_check") or {}
    print(f"  {_short(cons, 300)}")
    if not cons.get("two_routes_agree"):
        problems.append("χ² 检验的临界值法与 p 值法结论不一致")
    else:
        print("  >>> 两法一致，通过")
    return problems


def main():
    print("handler 模块加载情况：" + _short(_ops.status(), 400))
    print("### 自检 1：point_estimate 点估计（矩估计 / 极大似然）")
    fails = []
    for case in CASES:
        if case.get("op", "point_estimate") == "point_estimate":
            fails += run_case(case)
    print("### 自检 2：interval_estimate 区间估计")
    for case in CASES:
        if case.get("op") == "interval_estimate":
            fails += run_case(case)
    print("### 自检 3：hypothesis_test 假设检验")
    for case in CASES:
        if case.get("op") == "hypothesis_test":
            fails += run_case(case)
    print("### 自检 4：sampling_distribution 抽样分布")
    for case in CASES:
        if case.get("op") == "sampling_distribution":
            fails += run_case(case)
    print("### 自检 5：central_limit 中心极限定理")
    for case in CASES:
        if case.get("op") == "central_limit":
            fails += run_case(case)
    cross = cross_checks()
    if cross:
        TALLY["case_fail"] += len(cross)
    fails += cross

    print("=" * 100)
    print("统计：")
    print(f"  用例总数: {len(CASES)}（另有 5 组交叉验证）")
    print(f"  status=ok: {TALLY['ok']} | status=unsolved: {TALLY['unsolved']} | "
          f"status=error: {TALLY['error']}（另有 {TALLY['expected_error']} 条参数越界用例按预期报 invalid_input）")
    print(f"  独立复核全部通过(passed=True): {TALLY['methods_pass']} | "
          f"存在未通过(passed=False): {TALLY['methods_false']} | 判定不了(passed=None): {TALLY['methods_none']}")
    print(f"  断言不通过的用例: {TALLY['case_fail']}")
    if fails:
        print(f"结果：FAILED（{len(fails)} 项）")
        for p in fails:
            print("  - " + p)
        return 1
    print("结果：ALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
