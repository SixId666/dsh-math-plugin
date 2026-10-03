# -*- coding: utf-8 -*-
"""文档《使用示例.md》中出现的每一条引擎复现调用，按文档里写的参数原样再跑一遍并打印关键字段。"""
import sys, json
sys.path.insert(0, r"E:\dsh-math\engine")
from mathkit import engine as E, ops

CASES = [
    ("A1  integrate 定积分", "integrate", {"expr": "x*sin(x)", "var": "x", "lower": "0", "upper": "pi"}),
    ("A2  eigen", "eigen", {"matrix": "2,1;1,2"}),
    ("A2b matrix_analysis", "matrix_analysis", {"matrix": "2,1;1,2"}),
    ("B1  verify integral 错解", "verify", {"kind": "integral", "expr": "x*exp(x)", "candidate": "x*exp(x)+exp(x)", "var": "x"}),
    ("B1b verify integral 正确", "verify", {"kind": "integral", "expr": "x*exp(x)", "candidate": "x*exp(x)-exp(x)", "var": "x"}),
    ("B2  ode_check 错解(撇号写法)", "ode_check", {"equations": ["y' - 2*y = 0"], "candidate": "C1*exp(2*x)+x", "vars": ["x"], "funcs": ["y"]}),
    ("B2c ode_check 正确", "ode_check", {"equations": ["y' - 2*y = 0"], "candidate": "C1*exp(2*x)", "vars": ["x"], "funcs": ["y"]}),
    ("C1  domain", "domain", {"expr": "x^2*log(x)", "var": "x"}),
    ("C2  numeric_sum 无穷", "numeric_sum", {"expr": "n/(2^n)", "var": "n", "lower": "1", "upper": "oo"}),
    ("C2b numeric_sum 部分和", "numeric_sum", {"expr": "n/(2^n)", "var": "n", "lower": "1", "upper": "20"}),
    ("D1  dsolve", "dsolve", {"equations": ["diff(y(x),x,2)-3*diff(y(x),x)+2*y(x)=exp(x)"], "vars": ["x"], "funcs": ["y"]}),
    ("D1b ode_check 候选通解", "ode_check", {"equations": ["diff(y(x),x,2)-3*diff(y(x),x)+2*y(x)=exp(x)"],
                                            "candidate": "C1*exp(x)+C2*exp(2*x)-x*exp(x)", "vars": ["x"], "funcs": ["y"]}),
    ("D2  green", "green", {"p": "2*x*y-x^2", "q": "x+y^2", "x_lower": "-1", "x_upper": "1",
                            "y_lower_expr": "x^2", "y_upper_expr": "1"}),
    ("D2b curve_integral 下段", "curve_integral", {"expr": "2*x*y-x^2", "expr2": "x+y^2", "x_expr": "t", "y_expr": "t^2",
                                                   "var": "t", "lower": "-1", "upper": "1", "kind": "second"}),
    ("D2c curve_integral 上段", "curve_integral", {"expr": "2*x*y-x^2", "expr2": "x+y^2", "x_expr": "t", "y_expr": "1",
                                                   "var": "t", "lower": "1", "upper": "-1", "kind": "second"}),
    ("X1  diff", "diff", {"expr": "log(x+sqrt(1+x^2))", "var": "x", "order": 1}),
    ("X1b verify integral 对解", "verify", {"kind": "integral", "expr": "1/sqrt(1+x^2)", "candidate": "log(x+sqrt(1+x^2))", "var": "x"}),
    ("X2a diff", "diff", {"expr": "x^3-3*x", "var": "x", "order": 1}),
    ("X2b solve", "solve", {"equations": ["3*x^2-3=0"], "vars": ["x"]}),
    ("X2c diff order2", "diff", {"expr": "x^3-3*x", "var": "x", "order": 2}),
    ("X2d function_analysis", "function_analysis", {"expr": "x^3-3*x", "var": "x"}),
    ("X3  probability interval", "probability", {"kind": "interval", "dist": "normal", "mu": "0", "sigma": "1",
                                                 "lower": "-1.96", "upper": "1.96"}),
    ("X3c numeric_integrate", "numeric_integrate", {"expr": "exp(-x^2/2)/sqrt(2*pi)", "var": "x",
                                                    "lower": "-1.96", "upper": "1.96", "digits": 15}),
    ("X6  integrate erf", "integrate", {"expr": "exp(-x^2)", "var": "x", "lower": "0", "upper": "1"}),
    ("X6b numeric_integrate erf", "numeric_integrate", {"expr": "exp(-x^2)", "var": "x", "lower": "0", "upper": "1", "digits": 15}),
    ("Y1  integrate 换上限", "integrate", {"expr": "x*sin(x)", "var": "x", "lower": "0", "upper": "pi/2"}),
    ("Y2  matrix_analysis 3x3", "matrix_analysis", {"matrix": "2,1,1;1,2,1;1,1,2"}),
    ("X5b verify eigen 错候选", "verify", {"kind": "eigen", "matrix": "2,1;1,2", "candidate": "0,4"}),
]

for label, op, args in CASES:
    d = E.execute(op, args)
    r = d.get("result") or {}
    v = d.get("verification") or {}
    print("###", label, "| op=", op, "| status=", d.get("status"), "| v.status=", v.get("status"))
    for k in ("text", "solution", "approx", "domain", "residuals", "eigenvalues", "characteristic_polynomial",
              "derivative_of_candidate", "difference", "counterexample", "passed", "warnings"):
        if k in r:
            print("   r.%s = %s" % (k, json.dumps(r[k], ensure_ascii=False)[:300]))
        if k in v:
            print("   v.%s = %s" % (k, json.dumps(v[k], ensure_ascii=False)[:300]))
    if isinstance(v.get("evidence"), dict):
        ev = v["evidence"]
        for k in ("difference", "derivative_of_candidate", "counterexample", "characteristic_polynomial",
                  "numeric", "numeric_value", "symbolic_value", "max_rel_dev", "independent_eigenvalues"):
            if k in ev:
                print("   ev.%s = %s" % (k, json.dumps(ev[k], ensure_ascii=False)[:300]))
    for k in ("by_cdf", "by_integral", "abs_error", "chosen_route", "suspected_closed_form", "terms_summed",
              "eigenvalues", "algebraic_multiplicities", "det_text", "rank", "trace_text", "value_text",
              "monotonic_intervals", "extrema", "inflection_points", "numeric", "routes"):
        if k in r:
            print("   r2.%s = %s" % (k, json.dumps(r[k], ensure_ascii=False)[:300]))
    if d.get("warnings"):
        print("   warnings =", json.dumps(d["warnings"], ensure_ascii=False)[:300])
    if d.get("error"):
        print("   error =", json.dumps(d["error"], ensure_ascii=False)[:300])
