# -*- coding: utf-8 -*-
"""核对《使用示例.md》追问 2 引用的 matrix_analysis(3x3) 字段。"""
import sys, json
sys.path.insert(0, r"E:\dsh-math\engine")
from mathkit import engine as E, ops

d = E.execute("matrix_analysis", {"matrix": "2,1,1;1,2,1;1,1,2"})
r = d.get("result") or {}
v = d.get("verification") or {}
print("status:", d.get("status"), "| verification.status:", v.get("status"))
for k in ("det_text", "rank", "trace_text", "eigenvalues", "algebraic_multiplicities",
          "characteristic_polynomial", "not_computed", "eigenvectors_note", "text"):
    print(k, "=", json.dumps(r.get(k), ensure_ascii=False))
print("verification.note:", v.get("note"))
print("methods:", json.dumps(v.get("methods"), ensure_ascii=False)[:600])
