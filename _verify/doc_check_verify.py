# -*- coding: utf-8 -*-
"""核对《使用示例.md》里 verify 各例的 verification.status 与关键字段。"""
import sys, json
sys.path.insert(0, r"E:\dsh-math\engine")
from mathkit import engine as E, ops

cases = [
    ("B1 wrong integral", {"kind": "integral", "expr": "x*exp(x)", "candidate": "x*exp(x)+exp(x)", "var": "x"}),
    ("B1b right integral", {"kind": "integral", "expr": "x*exp(x)", "candidate": "x*exp(x)-exp(x)", "var": "x"}),
    ("S1 verify", {"kind": "integral", "expr": "1/sqrt(1+x^2)", "candidate": "log(x+sqrt(1+x^2))", "var": "x"}),
    ("S5 eigen verify", {"kind": "eigen", "matrix": "2,1;1,2", "candidate": "1,3"}),
]
for lab, a in cases:
    d = E.execute("verify", a)
    v = d.get("verification") or {}
    print("###", lab, "| status =", d.get("status"), "| verification.status =", v.get("status"))
    print("passed:", (d.get("result") or {}).get("passed"))
    print("evidence keys:", list((v.get("evidence") or {}).keys()))
    print("ev:", json.dumps(v.get("evidence"), ensure_ascii=False)[:500])
    print("v.note:", v.get("note"))
