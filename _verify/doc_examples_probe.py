# -*- coding: utf-8 -*-
"""交付文档《使用示例》里每一个示例的算子调用，真跑一遍并落盘真实输出。"""
import json
import sys

sys.path.insert(0, r"E:\dsh-math\engine")
from mathkit import engine as E, ops  # noqa: F401  ops 导入才会注册算子

CALLS = [
    # A 计算
    ("A1_integrate_def", "integrate", {"expr": "x*sin(x)", "var": "x", "lower": "0", "upper": "pi"}),
    ("A2_eigen", "eigen", {"matrix": "2,1;1,2"}),
    ("A2b_matrix_analysis", "matrix_analysis", {"matrix": "2,1;1,2"}),
    # B 验证
    ("B1_verify_integral_wrong", "verify", {"kind": "integral", "expr": "x*exp(x)", "candidate": "x*exp(x)+exp(x)", "var": "x"}),
    ("B1b_verify_integral_right", "verify", {"kind": "integral", "expr": "x*exp(x)", "candidate": "x*exp(x)-exp(x)", "var": "x"}),
    ("B2_ode_check_wrong", "ode_check", {"equations": ["y' - 2*y = 0"], "candidate": "2*exp(2*x)", "vars": ["x"], "funcs": ["y"]}),
    ("B2b_ode_check_wrong2", "ode_check", {"equations": ["y' - 2*y = 0"], "candidate": "C1*exp(2*x)+x", "vars": ["x"], "funcs": ["y"]}),
    ("B2c_ode_check_right", "ode_check", {"equations": ["y' - 2*y = 0"], "candidate": "C1*exp(2*x)", "vars": ["x"], "funcs": ["y"]}),
    ("B3_verify_derivative_wrong", "verify", {"kind": "derivative", "expr": "x^x", "candidate": "x^x", "var": "x"}),
    ("B4_verify_eq_wrong", "verify", {"kind": "equation_solution", "equations": ["x^2-3*x+2=0"], "candidate": "x=1, x=-2"}),
    ("B5_verify_matrix_inverse_wrong", "verify", {"kind": "matrix_inverse", "matrix": "1,2;3,4", "candidate": "-2,1;3/2,-1/2"}),
    # C 思路
    ("C1_domain", "domain", {"expr": "x^2*log(x)", "var": "x"}),
    ("C2_numeric_sum", "numeric_sum", {"expr": "n/(2^n)", "var": "n", "lower": "1", "upper": "oo"}),
    ("C2b_numeric_sum_partial", "numeric_sum", {"expr": "n/(2^n)", "var": "n", "lower": "1", "upper": "20"}),
    # 跨工具
    ("X1_diff", "diff", {"expr": "log(x+sqrt(1+x^2))", "var": "x", "order": 1}),
    ("X1b_verify_integral", "verify", {"kind": "integral", "expr": "1/sqrt(1+x^2)", "candidate": "log(x+sqrt(1+x^2))", "var": "x"}),
    ("X2_diff", "diff", {"expr": "x^3-3*x", "var": "x", "order": 1}),
    ("X2_solve", "solve", {"equations": ["3*x^2-3=0"], "vars": ["x"]}),
    ("X2b_diff2", "diff", {"expr": "x^3-3*x", "var": "x", "order": 2}),
    ("X2c_analysis", "function_analysis", {"expr": "x^3-3*x", "var": "x"}),
    ("X3_prob_interval", "probability", {"kind": "interval", "dist": "normal", "mu": "0", "sigma": "1", "lower": "-1.96", "upper": "1.96"}),
    ("X3b_distribution_cdf", "distribution", {"dist": "normal", "mu": "0", "sigma": "1", "kind": "cdf", "point": "1.96"}),
    ("X3c_numeric_integrate", "numeric_integrate", {"expr": "exp(-x^2/2)/sqrt(2*pi)", "var": "x", "lower": "-1.96", "upper": "1.96", "digits": 15}),
    ("X4_green", "green", {"p": "2*x*y-x^2", "q": "x+y^2", "x_lower": "-1", "x_upper": "1", "y_lower_expr": "x^2", "y_upper_expr": "1"}),
    ("X4b_curve_bottom", "curve_integral", {"expr": "2*x*y-x^2", "expr2": "x+y^2", "x_expr": "t", "y_expr": "t^2", "var": "t", "lower": "-1", "upper": "1", "kind": "second"}),
    ("X4c_curve_top", "curve_integral", {"expr": "2*x*y-x^2", "expr2": "x+y^2", "x_expr": "t", "y_expr": "1", "var": "t", "lower": "1", "upper": "-1", "kind": "second"}),
    ("X5_verify_eigen", "verify", {"kind": "eigen", "matrix": "2,1;1,2", "candidate": "1,3"}),
    ("X5b_verify_eigen_wrong", "verify", {"kind": "eigen", "matrix": "2,1;1,2", "candidate": "0,4"}),
    ("X6_integrate_erf", "integrate", {"expr": "exp(-x^2)", "var": "x", "lower": "0", "upper": "1"}),
    ("X6b_numeric_recheck", "numeric_integrate", {"expr": "exp(-x^2)", "var": "x", "lower": "0", "upper": "1", "digits": 15}),
    # 追问：只改一个条件
    ("Y1_integrate_halflimit", "integrate", {"expr": "x*sin(x)", "var": "x", "lower": "0", "upper": "pi/2"}),
    ("Y2_matrix_3x3", "matrix_analysis", {"matrix": "2,1,1;1,2,1;1,1,2"}),
    # D 完整解题
    ("D1_dsolve", "dsolve", {"equations": ["diff(y(x),x,2)-3*diff(y(x),x)+2*y(x)=exp(x)"], "vars": ["x"], "funcs": ["y"]}),
    ("D1b_ode_check", "ode_check", {"equations": ["diff(y(x),x,2)-3*diff(y(x),x)+2*y(x)=exp(x)"], "candidate": "C1*exp(x)+C2*exp(2*x)-x*exp(x)", "vars": ["x"], "funcs": ["y"]}),
    ("D2_status", "@cache_stats", {}),
]

out = {}
for label, op, args in CALLS:
    try:
        resp = E.execute(op, args)
    except Exception as exc:  # noqa: BLE001
        resp = {"status": "PYEXC", "error": f"{type(exc).__name__}: {exc}"}
    out[label] = {"op": op, "args": args, "resp": resp}

with open(r"E:\dsh-math\_verify\doc_probe_out.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)

for label, op, _ in CALLS:
    r = out[label]["resp"]
    print(f"=== {label} [{op}] status={r.get('status')} elapsed={r.get('elapsed_ms')}")
