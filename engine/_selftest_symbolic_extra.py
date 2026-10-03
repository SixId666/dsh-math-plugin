# -*- coding: utf-8 -*-
"""mathkit.symbolic_extra 自检：走 Node 侧同一入口 mathkit.engine.execute()。"""

import sys
import os

sys.path.insert(0, r"E:\dsh-math\engine")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

from mathkit import engine as E  # noqa: E402
from mathkit import ops as _OPS  # noqa: E402,F401  —— 与 worker.py 一致，触发算子注册

# (标签, 算子名, 参数字典)
CASES = [
    ("梯度", "gradient", {"expr": "x^2*y+y^3", "vars": ["x", "y"]}),
    ("梯度(别名 math_gradient)", "math_gradient", {"expr": "x^2*y+y^3", "vars": ["x", "y"]}),
    ("雅可比", "jacobian", {"exprs": ["x^2+y", "y^2+x"], "vars": ["x", "y"]}),
    ("海塞", "hessian", {"expr": "x^2*y+y^3", "vars": ["x", "y"]}),
    ("偏导(order=2)", "partial", {"expr": "x^2*y", "var": "x", "order": 2}),
    ("偏导(别名 pderiv)", "pderiv", {"expr": "x^2*y+y^3", "var": "y"}),
    ("方向导数", "directional_derivative",
     {"expr": "x^2+y^2", "vars": ["x", "y"], "point": ["1", "1"], "direction": ["1", "0"]}),
    ("方向导数(非单位方向)", "directional_derivative",
     {"expr": "x^2+y^2", "vars": ["x", "y"], "point": ["1", "1"], "direction": ["3", "4"]}),
    ("无条件极值(极小)", "multivar_extremum", {"expr": "x^2+y^2", "vars": ["x", "y"]}),
    ("无条件极值(鞍点)", "multivar_extremum", {"expr": "x^2-y^2", "vars": ["x", "y"]}),
    ("无条件极值(极大)", "multivar_extremum", {"expr": "-(x^2+y^2)", "vars": ["x", "y"]}),
    ("拉格朗日(单约束)", "lagrange", {"expr": "x*y", "constraints": "x+y=2", "vars": ["x", "y"]}),
    ("拉格朗日(圆约束)", "lagrange", {"expr": "x^2+y^2", "constraints": "x+y=1", "vars": ["x", "y"]}),
    ("拉格朗日(双约束)", "lagrange",
     {"expr": "x+y+z", "constraints": ["x^2+y^2=1", "x+y+z=1"], "vars": ["x", "y", "z"]}),
    ("隐函数求导", "implicit_diff", {"equation": "x^2+y^2=1", "vars": ["x", "y"]}),
    ("隐函数求导(二阶)", "implicit_diff", {"equation": "x^2+y^2=1", "vars": ["x", "y"], "order": 2}),
    ("参数方程求导", "parametric_derivative", {"x_expr": "cos(t)", "y_expr": "sin(t)", "param": "t"}),
    ("参数方程二阶导", "parametric_derivative",
     {"x_expr": "cos(t)", "y_expr": "sin(t)", "param": "t", "order": 2}),
    ("多元泰勒", "taylor_multivar",
     {"expr": "sin(x)*cos(y)", "vars": ["x", "y"], "point": ["0", "0"], "order": 3}),
    ("不等式", "inequality", {"expr": "x^2-3*x+2<0", "var": "x"}),
    ("不等式(无解)", "inequality", {"expr": "x^2+1<0", "var": "x"}),
    ("定义域(根式/分式)", "domain", {"expr": "sqrt(x-1)/(x-2)", "var": "x"}),
    ("定义域(对数/根式)", "domain", {"expr": "log(x)/sqrt(1-x^2)", "var": "x"}),
    ("定义域(反正弦)", "domain", {"expr": "arcsin(x-1)/sqrt(x-1)", "var": "x"}),
    ("渐近线", "asymptote", {"expr": "(x^2+1)/x", "var": "x"}),
    ("渐近线(纯水平)", "asymptote", {"expr": "atan(x)", "var": "x"}),
    ("函数分析", "function_analysis", {"expr": "x^3-3*x", "var": "x"}),
    ("函数分析(别名)", "math_function_analysis", {"expr": "(x-1)/(x+1)", "var": "x"}),
    ("中值点", "mean_value_point", {"expr": "x^3", "var": "x", "lower": "0", "upper": "1"}),
    ("柯西中值点", "mean_value_point",
     {"expr": "x^3", "var": "x", "lower": "0", "upper": "1", "kind": "cauchy", "g_expr": "x^2"}),
    ("隐函数极值", "implicit_function_extremum", {"equation": "x^2+y^2=1", "vars": ["x", "y"]}),
]

# 应当被判定为「输入错误」的用例（invalid_input），用来确认参数校验没有被吞掉
NEGATIVE_CASES = [
    ("表达式语法错误", "gradient", {"expr": "x^2+*", "vars": ["x", "y"]}),
    ("定义域含两个变量未指定 var", "domain", {"expr": "x^2+y^2", "var": None}),
    ("不等式缺不等号", "inequality", {"expr": "x^2-1", "var": "x"}),
    ("负阶数偏导", "partial", {"expr": "x^2", "var": "x", "order": 0}),
]


def show(payload):
    result = payload.get("result") or {}
    verification = payload.get("verification") or {}
    text = result.get("text")
    if text is None and "approx" in result:
        text = result["approx"]
    return {
        "status": payload.get("status"),
        "success": payload.get("success"),
        "text": text,
        "result_keys": sorted(result.keys()),
        "conditions": payload.get("conditions"),
        "verification": verification.get("status"),
        "warnings": payload.get("warnings"),
        "error": (payload.get("error") or {}).get("message"),
    }


def main():
    print("=" * 100)
    print("mathkit.symbolic_extra 自检 —— 入口 mathkit.engine.execute()")
    print("=" * 100)

    printed = E.known_operations()
    mine = [
        name for name in printed
        if name in {
            "gradient", "jacobian", "hessian", "partial", "directional_derivative",
            "multivar_extremum", "lagrange", "implicit_diff", "parametric_derivative",
            "taylor_multivar", "inequality", "domain", "asymptote", "function_analysis",
            "mean_value_point", "implicit_function_extremum",
        }
    ]
    print(f"已注册算子总数 {len(printed)}；本模块规范算子 {len(mine)} 个：{', '.join(sorted(mine))}")
    missing = {
        "gradient", "jacobian", "hessian", "partial", "directional_derivative",
        "multivar_extremum", "lagrange", "implicit_diff", "parametric_derivative",
        "taylor_multivar", "inequality", "domain", "asymptote", "function_analysis",
    } - set(mine)
    if missing:
        print(f"!! 缺少规范算子：{sorted(missing)}")
    print()

    tally = {"ok": 0, "partial": 0, "unsolved": 0, "error": 0}
    problems = []

    for label, name, args in CASES:
        payload = E.execute(name, args)
        info = show(payload)
        tally[info["status"]] = tally.get(info["status"], 0) + 1
        print("-" * 100)
        print(f"[{label}] op={name} status={info['status']} success={info['success']} "
              f"verify={info['verification']}")
        print(f"  result.text = {info['text']}")
        print(f"  result.keys = {info['result_keys']}")
        print(f"  conditions  = {info['conditions']}")
        if info["warnings"]:
            print(f"  warnings    = {info['warnings']}")
        if info["error"]:
            print(f"  error       = {info['error']}")
        if info["status"] == "error":
            kind = (payload.get("error") or {}).get("kind")
            problems.append(f"{label}（{name}）status=error kind={kind}: {info['error']}")
        # 关键约束：成功但结果为空 / 结果里带未求值回显
        if info["success"] and not (info["text"] or (payload.get("result") or {})):
            problems.append(f"{label}（{name}）标记成功但 result 为空")

    print()
    print("=" * 100)
    print("输入错误校验（期望 invalid_input，不得是 internal）")
    for label, name, args in NEGATIVE_CASES:
        payload = E.execute(name, args)
        kind = (payload.get("error") or {}).get("kind")
        print(f"  [{label}] op={name} status={payload.get('status')} kind={kind} "
              f"msg={(payload.get('error') or {}).get('message')}")
        if kind != "invalid_input":
            problems.append(f"{label}（{name}）期望 invalid_input，实际 kind={kind}")

    print()
    print("=" * 100)
    print(f"统计：ok={tally['ok']} partial={tally['partial']} "
          f"unsolved={tally['unsolved']} error={tally['error']}  共 {len(CASES)} 例")
    if problems:
        print("存在问题：")
        for item in problems:
            print(f"  - {item}")
    else:
        print("没有问题：无 traceback、无 status=error、无 success 但空结果。")
    print("=" * 100)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
