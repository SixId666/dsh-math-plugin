# -*- coding: utf-8 -*-
"""《使用示例.md》第 6 节「追问 4」的三条附加调用（只改条件，观察结果如何变）。"""
import sys, json
sys.path.insert(0, r"E:\dsh-math\engine")
from mathkit import engine as E, ops

for lab, op, a in [
    ("Z1 green 负向（orientation=negative）", "green",
     {"p": "2*x*y-x^2", "q": "x+y^2", "x_lower": "-1", "x_upper": "1",
      "y_lower_expr": "x^2", "y_upper_expr": "1", "orientation": "negative"}),
    ("Z2 green 上边界改成 y=2", "green",
     {"p": "2*x*y-x^2", "q": "x+y^2", "x_lower": "-1", "x_upper": "1",
      "y_lower_expr": "x^2", "y_upper_expr": "2"}),
    ("Z3 verify 定积分，候选 0.7468", "verify",
     {"kind": "definite_integral", "expr": "exp(-x^2)", "var": "x", "lower": "0", "upper": "1",
      "candidate": "0.7468"}),
]:
    d = E.execute(op, a)
    v = d.get("verification") or {}
    print("###", lab, "| status =", d.get("status"), "| verification.status =", v.get("status"))
    print("result:", json.dumps(d.get("result"), ensure_ascii=False)[:400])
    print("conditions:", json.dumps(d.get("conditions"), ensure_ascii=False)[:400])
    print("warnings:", json.dumps(d.get("warnings"), ensure_ascii=False)[:300])
    if isinstance(v.get("evidence"), dict):
        print("evidence:", json.dumps(v["evidence"], ensure_ascii=False)[:400])
