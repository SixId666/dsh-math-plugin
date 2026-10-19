# -*- coding: utf-8 -*-
"""mathkit.numeric 自检：走 Node 侧同一条调用路径 engine.execute(op, args)。"""
import sys, os

sys.path.insert(0, r"E:\dsh-math\engine")

from mathkit import engine as E  # noqa: E402
import mathkit.ops  # noqa: E402,F401  —— 与 worker.py 相同：导入汇总入口以触发 @op 注册

PI = 3.14159265358979323846
SQRT2 = "1.41421356237309504880168872420969807856967187537694"
SQRT_PI = "1.77245385090551602729816748334114518279754945612239"
PI2_6 = "1.64493406684822643647241516664602518921894990120680"

RAW_KEYS = (
    "digits", "abs_error", "rel_error", "approximate_value", "exact_form", "is_approximate",
    "is_exact", "error_bound_note", "routes", "cross_route_deviation", "suspected_closed_form",
    "residual", "max_residual", "converged", "candidates", "agreement", "agree", "max_rel_dev",
    "note", "sampling_detail", "samples_used", "samples_checked", "chosen_route", "samples_detail", "sequence", "extrapolated", "convergence_ratio", "tail_estimate",
    "sum_partial", "acceleration", "determinant", "eigenvalues", "inverse", "rank", "condition_number",
    "independent_check",
)


def _fmt(x, limit=200):
    s = repr(x)
    return s if len(s) <= limit else s[:limit] + "..."


def _digits_match(text, ref, need=45):
    """text 里最长的十进制数字串与 ref 的前缀匹配位数。"""
    best = 0
    for tok in (text or "").split():
        tok = tok.strip(",()[]")
        if tok.count(".") != 1:
            continue
        a = tok.split(".")
        if not (a[0].isdigit() or a[0].lstrip("-").isdigit()) or not a[1].isdigit():
            continue
        cand = a[0].lstrip("-") + "." + a[1]
        n = 0
        for i in range(min(len(cand), len(ref))):
            if cand[i] == ref[i]:
                n += 1
            else:
                break
        best = max(best, n)
    return best


def run(label, op, args, check=None):
    data = E.execute(op, args)
    status = data.get("status")
    print("=" * 78)
    print(f"[{label}] op={op}  args={_fmt(args, 160)}")
    print(f"  status={status}  success={data.get('success')}  method={data.get('method')}")
    res = data.get("result") or {}
    for k in ("text", "latex", "approx"):
        if k in res:
            print(f"  result.{k}: {_fmt(res[k], 240)}")
    raw = {k: v for k, v in res.items() if k in RAW_KEYS}
    shown = {}
    for k, v in raw.items():
        if k in ("samples_detail", "sampling_detail") and isinstance(v, list):
            shown[k] = f"<{len(v)} 个抽样点> 前两点={_fmt(v[:2], 240)}"
        elif k in ("routes", "candidates", "sequence", "eigenvalues", "inverse") and isinstance(v, list):
            shown[k] = f"<{len(v)} 项> {_fmt(v[:2], 200)}"
        else:
            shown[k] = v
    if shown:
        print(f"  raw: {_fmt(shown, 700)}")
    if data.get("conditions"):
        print(f"  conditions: {data['conditions']}")
    if data.get("verification"):
        print(f"  verification: {_fmt(data['verification'], 400)}")
    if data.get("warnings"):
        print(f"  warnings: {data['warnings']}")
    if status == "error":
        print(f"  ERROR: {data.get('error')}")
    ok = True
    msg = "（无断言）"
    if check is not None:
        try:
            ok, msg = check(data)
        except Exception as exc:  # noqa: BLE001
            ok, msg = False, f"断言抛异常 {type(exc).__name__}: {exc}"
    print(f"  CHECK {'PASS' if ok else 'FAIL'}: {msg}")
    return data, ok


def ap(data):
    r = data.get("result") or {}
    return r.get("approx", r.get("text"))


def close(got, want, tol=1e-10):
    return abs(float(got) - float(want)) <= tol


def near_str(got, ref, need=45):
    n = _digits_match(got, ref, need)
    return n >= need, n


def case_numeric_pi(d):
    v = ap(d)
    return close(v, PI, 1e-12), f"pi → {v}（偏差 {abs(float(v)-PI):.2e}）"


def case_numeric_sin(d):
    v = ap(d)
    raw = d["result"]
    # sin(pi/6) 会被 SymPy 化简成精确有理数 1/2：此时必须标注 is_exact=True / is_approximate=False
    labeled = (raw.get("is_exact") is True and str(raw.get("exact_form")) == "1/2") or raw.get("is_approximate") is True
    return (close(v, 0.5, 1e-12) and labeled,
            f"sin(pi/6) → {v}，is_approximate={raw.get('is_approximate')} is_exact={raw.get('is_exact')}，exact_form={raw.get('exact_form')}")


def case_numeric_sqrt2_50(d):
    v, raw = ap(d), d["result"]
    n = _digits_match(str(raw.get("approximate_value")) + " " + str(raw.get("text")) + " " + repr(v), SQRT2)
    return (int(raw.get("digits", 0)) == 50 and n >= 45,
            f"digits={raw.get('digits')}，高精度串与 sqrt(2) 前 {n} 位一致（要求≥45）")


def case_int_sinc(d):
    v, raw = ap(d), d["result"]
    return (close(v, PI / 2, 1e-9) and raw.get("abs_error") is not None,
            f"∫sin(x)/x = {v}（pi/2 偏差 {abs(float(v)-PI/2):.2e}），abs_error={raw.get('abs_error')}")


def case_int_gauss(d):
    v = ap(d)
    return close(v, float(SQRT_PI[:20]), 1e-9), f"∫exp(-x²) = {v}（sqrt(pi) 偏差 {abs(float(v)-float(SQRT_PI[:20])):.2e}）"


def case_int_singular(d):
    v, raw = ap(d), d["result"]
    return close(v, 2.0, 1e-9), f"∫1/sqrt(x) = {v}，abs_error={raw.get('abs_error')}"


def case_solve_cubic(d):
    v, raw = ap(d), d["result"]
    exp = 1.32471795724474602596
    res = raw.get("residual")
    return (close(v, exp, 1e-10) and d.get("verification", {}).get("status") == "numeric",
            f"根={v}（偏差 {abs(float(v)-exp):.2e}），残差={res}，verification={d.get('verification', {}).get('status')}")


def case_solve_system(d):
    a = ap(d)
    xs = [float(a["x"]), float(a["y"])] if isinstance(a, dict) else None
    ref = 0.7071067811865476
    raw = d["result"]
    return (xs is not None and all(abs(t - ref) < 1e-9 for t in xs),
            f"解={a}，残差={raw.get('max_residual') or raw.get('residual')}")


def case_opt_1d(d):
    a = ap(d)
    return (abs(float(a["value"]) + 1.0) < 1e-8 and abs(float(a["point"][0]) - 1.0) < 1e-8,
            f"min x²-2x → point={a['point']} value={a['value']}")


def case_opt_2d(d):
    a = ap(d)
    return (abs(float(a["value"])) < 1e-8 and max(abs(t) for t in a["point"]) < 1e-6,
            f"min x²+y² → point={a['point']} value={a['value']}")


def case_deriv(d):
    v, raw = ap(d), d["result"]
    want = 0.54030230586813971740  # cos(1)
    return (close(v, want, 1e-10) and raw.get("abs_error") is not None,
            f"sin'(1)={v}（cos(1) 偏差 {abs(float(v)-want):.2e}），abs_error={raw.get('abs_error')}")


def case_cmp_true(d):
    raw = d["result"]
    return (raw.get("agree") is True and d.get("status") == "ok",
            f"agree={raw.get('agree')} max_rel_dev={raw.get('max_rel_dev')} status={d.get('status')}")


def case_cmp_false(d):
    raw = d["result"]
    return (raw.get("agree") is False, f"agree={raw.get('agree')} max_rel_dev={raw.get('max_rel_dev')}")


def case_limit(d):
    v, raw = ap(d), d["result"]
    return (close(v, 1.0, 1e-8), f"lim sin(x)/x = {v}，收敛比={raw.get('convergence_ratio')}")


def case_sum(d):
    v, raw = ap(d), d["result"]
    n = _digits_match(str(raw.get("approximate_value")) + " " + str(v), PI2_6)
    return (close(v, float(PI2_6[:20]), 1e-10), f"Σ1/n²={v}（pi²/6 偏差 {abs(float(v)-float(PI2_6[:20])):.2e}），与 pi²/6 前 {n} 位一致")


def case_matrix(d):
    raw = d["result"]
    det = raw.get("determinant")
    return (det is not None and close(det, 0.0, 1e-9), f"det={det}（理论 0），rank={raw.get('rank')}，cond={raw.get('condition_number')}")


def case_int_left_inf(d):
    v, raw = ap(d), d["result"]
    return close(v, 1.0, 1e-9), f"∫_{{(-oo,0]}}exp(x) = {v}（真值 1，偏差 {abs(float(v)-1.0):.2e}），abs_error={raw.get('abs_error')}, 路线={raw.get('chosen_route')}"


CASES = [
    ("01 numeric pi", "numeric", {"expr": "pi"}, case_numeric_pi),
    ("02 numeric sin(x)@pi/6", "numeric", {"expr": "sin(x)", "vars": {"x": "pi/6"}}, case_numeric_sin),
    ("03 numeric sqrt(2) digits=50", "numeric", {"expr": "sqrt(2)", "digits": 50}, case_numeric_sqrt2_50),
    ("04 integrate sin(x)/x [0,oo]", "numeric_integrate", {"expr": "sin(x)/x", "var": "x", "lower": "0", "upper": "oo"}, case_int_sinc),
    ("05 integrate exp(-x^2) (-oo,oo)", "numeric_integrate", {"expr": "exp(-x^2)", "var": "x", "lower": "-oo", "upper": "oo"}, case_int_gauss),
    ("06 integrate 1/sqrt(x) [0,1]", "numeric_integrate", {"expr": "1/sqrt(x)", "var": "x", "lower": "0", "upper": "1"}, case_int_singular),
    ("07 solve x^3-x-1=0", "numeric_solve", {"equations": "x^3-x-1=0", "vars": "x", "guesses": {"x": 1}}, case_solve_cubic),
    ("08 solve system", "numeric_solve", {"equations": "x^2+y^2=1; x-y=0", "vars": ["x", "y"], "guesses": {"x": 0.7, "y": 0.7}}, case_solve_system),
    ("09 optimize x^2-2x", "numeric_optimize", {"expr": "x^2-2*x", "vars": "x"}, case_opt_1d),
    ("10 optimize x^2+y^2", "numeric_optimize", {"expr": "x^2+y^2", "vars": ["x", "y"]}, case_opt_2d),
    ("11 derivative sin(x)@1", "numeric_derivative", {"expr": "sin(x)", "var": "x", "point": "1"}, case_deriv),
    ("12 compare (x^2-1)/(x-1) vs x+1", "compare_numeric", {"expr": "(x^2-1)/(x-1)", "candidate": "x+1", "vars": "x"}, case_cmp_true),
    ("13 compare x^2 vs x^3", "compare_numeric", {"expr": "x^2", "candidate": "x^3", "vars": "x"}, case_cmp_false),
    ("14 limit sin(x)/x @0", "numeric_limit", {"expr": "sin(x)/x", "var": "x", "point": "0"}, case_limit),
    ("15 sum 1/n^2 1..oo", "numeric_sum", {"expr": "1/n^2", "var": "n", "lower": "1", "upper": "oo"}, case_sum),
    ("16 matrix det", "numeric_matrix", {"matrix": "1,2,3; 4,5,6; 7,8,9"}, case_matrix),
    ("17 integrate exp(x) (-oo,0]", "numeric_integrate", {"expr": "exp(x)", "var": "x", "lower": "-oo", "upper": "0"}, case_int_left_inf),
]


def main():
    stats = {"ok": 0, "partial": 0, "unsolved": 0, "error": 0}
    passed = failed = 0
    fails = []
    for label, op, args, check in CASES:
        data, ok = run(label, op, args, check)
        st = data.get("status", "error")
        stats[st] = stats.get(st, 0) + 1
        if ok:
            passed += 1
        else:
            failed += 1
            fails.append(label)
        if st == "error":
            fails.append(label + " (status=error)")
    print("=" * 78)
    print(f"status 统计: ok={stats['ok']} partial={stats['partial']} unsolved={stats['unsolved']} error={stats['error']}")
    print(f"断言统计: PASS={passed} FAIL={failed}")
    if fails:
        print("失败项: " + "; ".join(fails))
    print("RESULT: " + ("ALL GREEN" if failed == 0 and stats["error"] == 0 else "NEEDS FIX"))
    return 0 if (failed == 0 and stats["error"] == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
