# -*- coding: utf-8 -*-
"""verifier.py 自检脚本（math_verify 统一复核入口 / conditional_check / domain_check）。

走的是 Node 侧同一条调用路径：``engine.execute(算子名, 参数)``。
每条用例都断言：status 不是 error、passed 与预期一致、验证等级符合预期、
反例必须带差异证据（不允许只有 False 没有证据）、证据不足时绝不允许 passed=True。

运行：python _selftest_verifier.py
"""
import sys
import os
import json

sys.path.insert(0, r"E:\dsh-math\engine")
from mathkit import engine as E  # noqa: E402
from mathkit import ops as _ops  # noqa: E402,F401  触发 handler 模块注册（worker.py 的真实做法）

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

_ANY = object()
TALLY = {"ok": 0, "unsolved": 0, "error": 0, "passed_true": 0, "passed_false": 0,
         "passed_none": 0, "case_fail": 0}


def _short(obj, limit=700):
    text = json.dumps(obj, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "…(截断)"


def _verification_of(out):
    """顶层 verification 是规范位置；result.verification 是同一份内容的镜像。"""
    ver = {}
    if isinstance(out.get("verification"), dict):
        ver.update(out["verification"])
    nested = (out.get("result") or {}).get("verification")
    if isinstance(nested, dict):
        for k, v in nested.items():
            ver.setdefault(k, v)
    return ver


def _evidence_of(out):
    res = out.get("result") or {}
    ev = {}
    for src in (_verification_of(out).get("evidence"), res.get("verification", {}).get("evidence")
                if isinstance(res.get("verification"), dict) else None, res.get("evidence")):
        if isinstance(src, dict):
            ev.update(src)
    return ev


def _has_counterexample(out):
    ev = _evidence_of(out)
    if ev.get("counterexample"):
        return True
    num = ev.get("numeric") or {}
    if num.get("agree") is False and num.get("note"):
        return True
    for key in ("deviation", "independent_note", "numeric_approach", "per_condition"):
        v = ev.get(key)
        if v:
            return True
    text = json.dumps(out.get("result") or {}, ensure_ascii=False, default=str)
    return "反例" in text or "不一致" in text or "偏差" in text


def run_case(case):
    title = case["title"]
    op = case.get("op", "verify")
    args = case["args"]
    out = E.execute(op, args)
    res = out.get("result") or {}
    ver = _verification_of(out)
    status = out.get("status")
    passed = res.get("passed")
    print("=" * 100)
    print(f"[{title}] {op} " + _short(args, 300))
    print(f"  status: {status} | success: {out.get('success')} | passed: {passed}")
    print(f"  verification.level: {ver.get('level')}")
    print(f"  verification.method: {ver.get('method')}")
    print(f"  verification.evidence: {_short(ver.get('evidence') or {}, 600)}")
    print(f"  unhandled_conditions: {_short(ver.get('unhandled_conditions'), 400)}")
    print(f"  cannot_verify_reason: {ver.get('cannot_verify_reason')}")
    print(f"  conditions: {_short(out.get('conditions'), 300)}")
    print(f"  warnings: {_short(out.get('warnings'), 300)}")
    if status == "error":
        print(f"  error: {_short(out.get('error'), 300)}")

    problems = []
    if status == "error":
        problems.append(f"status=error（不允许）：{out.get('error')}")
    if case.get("op_status") and status != case["op_status"]:
        problems.append(f"status 期望 {case['op_status']}，实际 {status}")
    if case.get("status_in") and status not in case["status_in"]:
        problems.append(f"status 期望属于 {case['status_in']}，实际 {status}")
    if case.get("passed", _ANY) is not _ANY and passed != case["passed"]:
        problems.append(f"passed 期望 {case['passed']}，实际 {passed}")
    if case.get("level_in") is not None and ver.get("level") not in case["level_in"]:
        problems.append(f"level 期望属于 {case['level_in']}，实际 {ver.get('level')}")
    if case.get("needs_counterexample") and not _has_counterexample(out):
        problems.append("反例缺少差异证据（只有 False，没有反例/差异）")
    if case.get("must_not_pass") and passed is True:
        problems.append("证据不足却返回 passed=True（严重违规）")
    if case.get("unhandled_contains"):
        blob = json.dumps(ver.get("unhandled_conditions") or [], ensure_ascii=False)
        for token in case["unhandled_contains"]:
            if token not in blob:
                problems.append(f"unhandled_conditions 应包含 {token!r}，实际 {blob[:200]}")
    if case.get("cannot_contains"):
        blob = str(ver.get("cannot_verify_reason") or "")
        if case["cannot_contains"] not in blob:
            problems.append(f"cannot_verify_reason 应包含 {case['cannot_contains']!r}，实际 {blob!r}")
    if case.get("evidence_key"):
        ev = _evidence_of(out)
        for key in case["evidence_key"]:
            if key not in ev:
                problems.append(f"evidence 缺少关键量 {key!r}（实际键：{sorted(ev)[:12]}）")
    if case.get("result_text_contains"):
        text = json.dumps(res, ensure_ascii=False, default=str)
        for token in case["result_text_contains"]:
            if token not in text:
                problems.append(f"结果文本应包含 {token!r}")

    TALLY[status if status in ("ok", "unsolved", "error") else "error"] += 1
    if passed is True:
        TALLY["passed_true"] += 1
    elif passed is False:
        TALLY["passed_false"] += 1
    else:
        TALLY["passed_none"] += 1
    if problems:
        TALLY["case_fail"] += 1
        print("  >>> 用例不通过：")
        for p in problems:
            print("      - " + p)
    else:
        print("  >>> 用例通过")
    return problems


CASES = [
    # ---------------- 正例：应当 passed=True ----------------
    dict(title="正例 导数 x^2 → 2x（符号级）", args=dict(expr="x^2", candidate="2*x", kind="derivative"),
         passed=True, level_in={"symbolic"}, evidence_key=["difference"]),
    dict(title="正例 不定积分 x*exp(x^2) → exp(x^2)/2", args=dict(expr="x*exp(x^2)", candidate="exp(x^2)/2", kind="integral"),
         passed=True, level_in={"symbolic"}),
    dict(title="正例 方程解 x^2-3x+2=0 → 2", args=dict(equations="x^2-3*x+2=0", candidate="2", kind="equation_solution"),
         passed=True, level_in={"symbolic"}, evidence_key=["residuals"]),
    dict(title="正例 恒等式 (x^2-1)/(x-1) → x+1（须报告 x≠1）",
         args=dict(expr="(x^2-1)/(x-1)", candidate="x+1", kind="identity"),
         passed=True, level_in={"symbolic"}, unhandled_contains=["x = 1"]),
    dict(title="正例 矩阵求逆", args=dict(matrix="[[1,2],[3,4]]", candidate="[[-2,1],[3/2,-1/2]]", kind="matrix_inverse"),
         passed=True, level_in={"symbolic"}, evidence_key=["difference_MC_minus_I"]),
    dict(title="正例 行列式", args=dict(matrix="[[1,2],[3,4]]", candidate="-2", kind="determinant"),
         passed=True, level_in={"symbolic"}, evidence_key=["independent_determinant"]),
    dict(title="正例 特征值（候选写成逗号列表）", args=dict(matrix="[[2,0],[0,3]]", candidate="2,3", kind="eigen"),
         passed=True, level_in={"symbolic"}),
    dict(title="正例 归一化 exp(-x^2) 的常数 → 1/sqrt(pi)",
         args=dict(expr="exp(-x^2)", candidate="1/sqrt(pi)", kind="normalization"),
         passed=True, level_in={"symbolic"}, evidence_key=["check_expression"]),
    dict(title="正例 极限 sin(x)/x → 1 (x→0)",
         args=dict(expr="sin(x)/x", candidate="1", kind="limit", point="0"),
         passed=True, level_in={"symbolic"}),
    dict(title="正例 定积分 ∫_0^1 x^2 = 1/3",
         args=dict(expr="x^2", candidate="1/3", kind="definite_integral", lower="0", upper="1"),
         passed=True, level_in={"symbolic"}),
    dict(title="正例 化简 x^2+2x+1 → (x+1)^2", args=dict(expr="x^2+2*x+1", candidate="(x+1)^2", kind="simplify"),
         passed=True, level_in={"symbolic"}),
    dict(title="正例 因式分解 x^2-1", args=dict(expr="x^2-1", candidate="(x-1)*(x+1)", kind="factor"),
         passed=True, level_in={"symbolic"}),
    dict(title="正例 展开 (x+1)^2", args=dict(expr="(x+1)^2", candidate="x^2+2*x+1", kind="expand"),
         passed=True, level_in={"symbolic"}),
    dict(title="正例 期望 U(0,1) 的 E[X]=1/2", args=dict(expr="X", candidate="1/2", kind="expectation",
                                                     dist="uniform", lower="0", upper="1"),
         passed=True, level_in={"symbolic"}),
    dict(title="正例 方差 B(10,1/2) 的 Var=5/2", args=dict(expr="X", candidate="5/2", kind="variance",
                                                       dist="binomial", m=10, p="1/2"),
         passed=True, level_in={"symbolic"}),
    dict(title="正例 概率 P(X<=3) B(10,1/2) 的数值（水平只能是 numeric）",
         args=dict(expr="P(X<=3)", candidate="0.171875", kind="probability", dist="binomial", m=10, p="0.5"),
         passed=True, level_in={"symbolic", "numeric"}),
    dict(title="正例 条件被真正使用：sqrt(x^2)→x 且声明 x>0",
         args=dict(expr="sqrt(x^2)", candidate="x", kind="identity", conditions="x > 0"),
         passed=True, level_in={"symbolic"}),
    dict(title="正例/未处理条件 1/x 的原函数 log(x)（须报定义域问题）",
         args=dict(expr="1/x", candidate="log(x)", kind="integral"),
         passed=True, unhandled_contains=["定义域", "log"]),

    # ---------------- 反例：应当 passed=False 且带差异证据 ----------------
    dict(title="反例 导数 x^2 → x", args=dict(expr="x^2", candidate="x", kind="derivative"),
         passed=False, needs_counterexample=True, evidence_key=["difference"]),
    dict(title="反例 恒等式 x^2 → x^3", args=dict(expr="x^2", candidate="x^3", kind="identity"),
         passed=False, needs_counterexample=True),
    dict(title="反例 方程解 x^2-3x+2=0 → 3（注：1 其实是根，规格书写错了）",
         args=dict(equations="x^2-3*x+2=0", candidate="3", kind="equation_solution"),
         passed=False, needs_counterexample=True, evidence_key=["residuals"]),
    dict(title="反例 定积分 ∫_0^1 x^2 ≠ 1/4",
         args=dict(expr="x^2", candidate="1/4", kind="definite_integral", lower="0", upper="1"),
         passed=False, needs_counterexample=True),
    dict(title="反例 归一化 exp(-x^2) 的常数 ≠ 1",
         args=dict(expr="exp(-x^2)", candidate="1", kind="normalization"),
         passed=False, needs_counterexample=True),
    dict(title="反例 矩阵求逆给错", args=dict(matrix="[[1,2],[3,4]]", candidate="[[-2,1],[1,-1/2]]", kind="matrix_inverse"),
         passed=False, needs_counterexample=True, evidence_key=["difference_MC_minus_I"]),
    dict(title="反例 极限 sin(x)/x ≠ 0", args=dict(expr="sin(x)/x", candidate="0", kind="limit", point="0"),
         passed=False, needs_counterexample=True),
    dict(title="反例 概率 P(X<=3)≠0.5", args=dict(expr="P(X<=3)", candidate="0.5", kind="probability",
                                              dist="binomial", m=10, p="0.5"),
         passed=False, needs_counterexample=True, evidence_key=["independent_value"]),
    dict(title="反例 期望 U(0,1) 的 E[X]≠1", args=dict(expr="X", candidate="1", kind="expectation",
                                                     dist="uniform", lower="0", upper="1"),
         passed=False, needs_counterexample=True),
    dict(title="反例 化简/展开 (x+1)^2 ≠ x^2+2x+2", args=dict(expr="(x+1)^2", candidate="x^2+2*x+2", kind="expand"),
         passed=False, needs_counterexample=True),

    # ---------------- 无法验证：必须 unsolved / level=none，绝不 passed=True ----------------
    dict(title="无法验证 复数分支 log(-x) vs log(x)（符号不收敛且取不到实数抽样点）",
         args=dict(expr="log(-x)", candidate="log(x)", kind="identity"),
         passed=None, level_in={"none"}, must_not_pass=True,
         cannot_contains="无法验证", op_status="unsolved"),
    dict(title="无法验证 只给 candidate 没有原始表达式",
         args=dict(candidate="42", kind="general"),
         passed=None, level_in={"none"}, must_not_pass=True, op_status="unsolved",
         cannot_contains="没有给出原始表达式"),
    dict(title="未知 kind 不能报错，要退化为通用复核并说明局限",
         args=dict(expr="x^2", candidate="2*x", kind="no_such_kind"),
         passed=False, status_in={"ok"}, unhandled_contains=["未知的计算类型"]),
    dict(title="无法验证 声明的条件无法纳入证明（统计假设）",
         args=dict(expr="x^2", candidate="2*x", kind="identity", conditions="样本来自正态总体"),
         passed=False, unhandled_contains=["正态"]),
]

COND_CASES = [
    dict(title="条件检查 log(x) 与 x>0 相容", op="conditional_check",
         args=dict(expr="log(x)", var="x", conditions="x > 0"), passed=True),
    dict(title="条件检查 log(-x) 与 x>0 不相容", op="conditional_check",
         args=dict(expr="log(-x)", var="x", conditions="x > 0"), passed=False),
    dict(title="条件检查 1/x 与 x≠0 相容", op="conditional_check",
         args=dict(expr="1/x", var="x", conditions="x != 0"), passed=True),
    dict(title="条件检查 A 可逆 与 det(A)=0 不相容", op="conditional_check",
         args=dict(expr="det(A)", var="A", conditions="A 可逆", matrix="[[1,2],[2,4]]"), passed=False),
    dict(title="条件检查 统计类条件无法检验 → unknown", op="conditional_check",
         args=dict(expr="x^2", var="x", conditions="样本来自正态总体"), passed=None),
    dict(title="条件检查 恒真条件 x^2>=0", op="conditional_check",
         args=dict(expr="x^2 + 1", var="x", conditions="x^2 + 1 > 0"), passed=True),
]

DOMAIN_CASES = [
    dict(title="定义域 1/(x-1) 在 x=1 不合法", op="domain_check",
         args=dict(expr="1/(x-1)", var="x", point="1"), passed=False),
    dict(title="定义域 log(x) 在 x=-1 不合法", op="domain_check",
         args=dict(expr="log(x)", var="x", point="-1"), passed=False),
    dict(title="定义域 tan(x) 在 x=pi/2 不合法", op="domain_check",
         args=dict(expr="tan(x)", var="x", point="pi/2"), passed=False),
    dict(title="定义域 sqrt(x)+log(x-1) 在 x=2 合法", op="domain_check",
         args=dict(expr="sqrt(x)+log(x-1)", var="x", point="2"), passed=True),
    dict(title="定义域 sqrt(x) 在 x=1/2 合法", op="domain_check",
         args=dict(expr="sqrt(x)", var="x", point="1/2"), passed=True),
    dict(title="定义域 不给 point → 只给自然定义域", op="domain_check",
         args=dict(expr="x^2/(x-1)", var="x"), passed=None),
    dict(title="定义域 asin(x) 在 x=2 不合法", op="domain_check",
         args=dict(expr="asin(x)", var="x", point="2"), passed=False),
]


def main():
    print("handler 模块加载情况：" + _short(_ops.status(), 400))
    print("### 自检 1：verify（math_verify 统一复核入口）")
    fails = []
    for case in CASES:
        fails += run_case(case)
    print("### 自检 2：conditional_check")
    for case in COND_CASES:
        fails += run_case(case)
    print("### 自检 3：domain_check")
    for case in DOMAIN_CASES:
        fails += run_case(case)

    total = len(CASES) + len(COND_CASES) + len(DOMAIN_CASES)
    print("=" * 100)
    print("统计：")
    print(f"  用例总数: {total}")
    print(f"  status=ok: {TALLY['ok']} | status=unsolved: {TALLY['unsolved']} | status=error: {TALLY['error']}")
    print(f"  passed=True: {TALLY['passed_true']} | passed=False: {TALLY['passed_false']} | "
          f"passed=None(判定不了): {TALLY['passed_none']}")
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
