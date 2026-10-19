"""微积分模块自检脚本：走 ``E.execute``（与 Node 侧完全相同的调用路径）。

运行： ``cd E:\\dsh-math\\engine; python _selftest_calculus.py``
"""

from __future__ import annotations

import json
import sys
import os

sys.path.insert(0, r"E:\dsh-math\engine")

from mathkit import engine as E  # noqa: E402
from mathkit import ops as OPS  # noqa: E402,F401  触发算子注册

CASES: list[tuple[str, dict]] = [
    # 极限
    ("limit", {"expr": "sin(x)/x", "var": "x", "point": "0"}),
    ("limit", {"expr": "1/x", "var": "x", "point": "0", "dir": "+"}),
    ("limit", {"expr": "1/x", "var": "x", "point": "0", "dir": "left"}),
    ("limit", {"expr": "(1+1/n)^n", "var": "n", "point": "oo"}),
    ("limit", {"expr": "1/x", "var": "x", "point": "0"}),
    ("lim", {"expr": "(1-cos(x))/x^2", "var": "x", "point": "0"}),
    # 导数
    ("diff", {"expr": "x^x", "var": "x"}),
    ("differentiate", {"expr": "sin(x)", "var": "x", "order": 2}),
    ("math_derivative", {"expr": "log(x)/x", "var": "x"}),
    # 积分
    ("integrate", {"expr": "x*exp(x^2)", "var": "x"}),
    ("int", {"expr": "1/(x^2+1)", "var": "x"}),
    ("integrate", {"expr": "x^2", "var": "x", "lower": "0", "upper": "1"}),
    ("integrate", {"expr": "exp(-x)", "var": "x", "lower": "0", "upper": "oo"}),
    ("integrate", {"expr": "sin(x)/x", "var": "x", "lower": "0", "upper": "oo"}),
    ("integrate", {"expr": "1/x", "var": "x", "lower": "-1", "upper": "1"}),
    ("integrate", {"expr": "exp(-x^2)", "var": "x", "lower": "-oo", "upper": "oo"}),
    # 级数求和
    ("sum", {"expr": "1/n^2", "var": "n", "lower": "1", "upper": "oo"}),
    ("series_sum", {"expr": "1/2^n", "var": "n", "lower": "0", "upper": "oo"}),
    ("sum", {"expr": "n", "var": "n", "lower": "1", "upper": "oo"}),
    # 方程
    ("solve", {"equations": "x^2 - 3*x + 2 = 0"}),
    ("math_solve", {"equations": ["x + y = 3", "x - y = 1"]}),
    ("solve", {"equations": "x + y = 3; x + y = 4"}),
    ("solve", {"equations": "sin(x) = 0", "vars": "x"}),
    # 微分方程
    ("dsolve", {"equations": "y' - 2*y = 0"}),
    ("ode", {"equations": "diff(y(x),x,x) + y(x) = 0"}),
    ("ode_check", {"equations": "y' - 2*y = 0", "candidate": "y(x) = C1*exp(2*x)"}),
    ("verify_ode", {"equations": "y' - 2*y = 0", "candidate": "y(x) = C1*exp(3*x)"}),
    # 化简 / 展开 / 因式分解
    ("simplify", {"expr": "sin(x)^2 + cos(x)^2"}),
    ("math_simplify", {"expr": "(x^2-1)/(x-1)"}),
    ("factor", {"expr": "x^2 - 1"}),
    ("expand", {"expr": "(x+1)^3"}),
    ("apart", {"expr": "1/(x^2-1)"}),
    # 幂级数
    ("series", {"expr": "sin(x)", "var": "x", "point": "0", "order": 6}),
    ("taylor", {"expr": "exp(x)", "var": "x", "point": "0", "order": 5}),
    ("series", {"expr": "log(x)", "var": "x", "point": "1", "order": 4}),
    # 多元微分
    ("gradient", {"expr": "x^2*y + sin(y)", "vars": "x,y"}),
    ("jacobian", {"exprs": ["x^2*y", "x + y^2"], "vars": "x,y"}),
    ("hessian", {"expr": "x^2*y^3 + x*y", "vars": "x,y"}),
    # 错误输入（应转成 invalid_input，不是 traceback）
    ("diff", {"expr": "x^2", "var": "x", "order": 0}),
    ("integrate", {"expr": "x^2", "var": "x", "lower": "0"}),
    ("integrate", {"expr": "x^2", "var": "x", "lower": "0", "upper": "1", "order": None}),
]


def main() -> int:
    stats = {"ok": 0, "partial": 0, "unsolved": 0, "error": 0}
    problems: list[str] = []
    print("=" * 100)
    print("微积分模块自检：", E.known_operations().__len__(), "个算子名（含别名）")
    print("已注册的微积分算子：", ", ".join(sorted(
        name for name in E.known_operations()
        if E.canonical_op(name) in {
            "limit", "diff", "integrate", "sum", "solve", "dsolve", "simplify",
            "expand", "factor", "together", "series", "ode_check",
            "gradient", "jacobian", "hessian",
        }
    )))
    print("ops 模块加载情况：", json.dumps(OPS.status(), ensure_ascii=False))
    print("=" * 100)

    for name, args in CASES:
        payload = E.execute(name, args)
        status = payload.get("status")
        stats[status] = stats.get(status, 0) + 1
        result = payload.get("result", {})
        text = result.get("text", "")
        conditions = payload.get("conditions", [])
        verification = payload.get("verification", {})
        print("-" * 100)
        print(f"op={name} args={json.dumps(args, ensure_ascii=False)}")
        print(f"  status={status}  success={payload.get('success')}  method={payload.get('method')}")
        print(f"  result.text={text}")
        if result.get("approx") is not None:
            print(f"  result.approx={result['approx']}")
        for key in ("solutions", "solution", "components", "residuals", "side", "left_limit",
                    "right_limit", "op_counts", "chosen", "free_variables", "arbitrary_constants",
                    "remainder", "solution_count", "antiderivative_note"):
            if key in result:
                print(f"  result[{key}]={json.dumps(result[key], ensure_ascii=False)}")
        print(f"  conditions={json.dumps(conditions, ensure_ascii=False)}")
        print(f"  verification.status={verification.get('status')}")
        for method in verification.get("methods", [])[:3]:
            brief = {k: v for k, v in method.items() if k not in ("deviations",)}
            print(f"    - {json.dumps(brief, ensure_ascii=False, default=str)[:600]}")
        if verification.get("note"):
            print(f"    note: {verification['note'][:300]}")
        if payload.get("warnings"):
            print(f"  warnings={json.dumps(payload['warnings'], ensure_ascii=False)[:500]}")
        if payload.get("error"):
            print(f"  error={json.dumps(payload['error'], ensure_ascii=False)[:400]}")

        if status == "error":
            kind = (payload.get("error") or {}).get("kind")
            problems.append(f"{name} {args} -> error/{kind}: {(payload.get('error') or {}).get('message')}")
        if status == "ok" and not result:
            problems.append(f"{name} {args} -> status=ok 但没有结果内容")
        if status == "ok" and verification.get("status") == "unverified":
            problems.append(f"{name} {args} -> 成功但完全未验证")

    print("=" * 100)
    print("统计：", json.dumps(stats, ensure_ascii=False))
    print(f"总用例：{len(CASES)}")
    if problems:
        print("需要处理的问题：")
        for item in problems:
            print("  !", item)
    else:
        print("没有 error 级失败，也没有「成功但未验证」的用例。")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
