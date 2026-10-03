# -*- coding: utf-8 -*-
import sys, json
sys.path.insert(0, r"E:\dsh-math\engine")
from mathkit import engine as E, ops

for lab, op, a in [
    ("C2 numeric_sum", "numeric_sum", {"expr": "n/(2^n)", "var": "n", "lower": "1", "upper": "oo"}),
    ("S3 numeric_integrate", "numeric_integrate", {"expr": "exp(-x^2/2)/sqrt(2*pi)", "var": "x",
                                                   "lower": "-1.96", "upper": "1.96", "digits": 15}),
    ("S4 numeric_integrate", "numeric_integrate", {"expr": "exp(-x^2)", "var": "x",
                                                   "lower": "0", "upper": "1", "digits": 15}),
]:
    d = E.execute(op, a)
    v = d.get("verification") or {}
    print("###", lab, "| v.status =", v.get("status"))
    print("note:", v.get("note"))
    print("methods:", json.dumps(v.get("methods"), ensure_ascii=False)[:700])
    r = d.get("result") or {}
    print("chosen_route:", r.get("chosen_route"))
    print("suspected_closed_form:", r.get("suspected_closed_form"), "| abs_error:", r.get("abs_error"))
