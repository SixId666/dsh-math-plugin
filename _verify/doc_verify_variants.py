# -*- coding: utf-8 -*-
"""B 模式验证类示例的可用参数组合试探（category=equation_solution / matrix_inverse / determinant）。"""
import json
import sys

sys.path.insert(0, r"E:\dsh-math\engine")
from mathkit import engine as E, ops  # noqa: F401

VARIANTS = [
    ("eq_cand_x_eq_1", {"kind": "equation_solution", "equations": ["x^2-3*x+2=0"], "candidate": "x=1"}),
    ("eq_cand_1", {"kind": "equation_solution", "equations": ["x^2-3*x+2=0"], "candidate": "1"}),
    ("eq_cand_list", {"kind": "equation_solution", "equations": ["x^2-3*x+2=0"], "candidate": "[1, 2]"}),
    ("eq_wrong_one", {"kind": "equation_solution", "equations": ["x^2-3*x+2=0"], "candidate": "-2"}),
    ("eq_two_eqs_wrong", {"kind": "equation_solution", "equations": ["x+y=3", "x-y=1"], "candidate": "x=2"}),
    ("inv_cand_matrix", {"kind": "matrix_inverse", "matrix": "1,2;3,4", "candidate": "Matrix([[-2,1],[3/2,-1/2]])"}),
    ("inv_cand_plain", {"kind": "matrix_inverse", "matrix": "1,2;3,4", "candidate": "-2,1;3/2,-1/2"}),
    ("det_cand", {"kind": "determinant", "matrix": "1,2;3,4", "candidate": "-2"}),
    ("eigen_ok", {"kind": "eigen", "matrix": "2,1;1,2", "candidate": "1,3"}),
    ("eigen_wrong", {"kind": "eigen", "matrix": "2,1;1,2", "candidate": "0,4"}),
    ("normalization_wrong", {"kind": "normalization", "expr": "k*x", "candidate": "1", "lower": "0", "upper": "1"}),
    ("identity_wrong", {"kind": "identity", "expr": "sin(x)^2+cos(x)^2=1", "candidate": "1"}),
    ("limit_wrong", {"kind": "limit", "expr": "sin(x)/x", "candidate": "0", "var": "x", "point": "0"}),
    ("simplify_wrong", {"kind": "simplify", "expr": "(x^2-1)/(x-1)", "candidate": "x"}),
]

out = {}
for label, args in VARIANTS:
    try:
        resp = E.execute("verify", args)
    except Exception as exc:  # noqa: BLE001
        resp = {"status": "PYEXC", "error": f"{type(exc).__name__}: {exc}"}
    out[label] = {"args": args, "resp": resp}
    r = resp
    msg = ""
    if r.get("error"):
        msg = json.dumps(r["error"], ensure_ascii=False)
    else:
        res = r.get("result", {})
        msg = str(res.get("text", ""))[:160] + " || passed=" + str(res.get("passed"))
    print(f"{label}: status={r.get('status')} {msg}")

with open(r"E:\dsh-math\_verify\doc_probe_verify_variants.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
