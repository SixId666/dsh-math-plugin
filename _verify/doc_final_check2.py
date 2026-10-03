# -*- coding: utf-8 -*-
import sys, json
sys.path.insert(0, r"E:\dsh-math\engine")
from mathkit import engine as E, ops

for lab, op, a in [
    ("B2", "ode_check", {"equations": ["y' - 2*y = 0"], "candidate": "C1*exp(2*x)+x", "vars": ["x"], "funcs": ["y"]}),
    ("B2c", "ode_check", {"equations": ["y' - 2*y = 0"], "candidate": "C1*exp(2*x)", "vars": ["x"], "funcs": ["y"]}),
    ("S1diff", "diff", {"expr": "log(x+sqrt(1+x^2))", "var": "x", "order": 1}),
    ("D1b", "ode_check", {"equations": ["diff(y(x),x,2)-3*diff(y(x),x)+2*y(x)=exp(x)"],
                          "candidate": "C1*exp(x)+C2*exp(2*x)-x*exp(x)", "vars": ["x"], "funcs": ["y"]}),
    ("D1", "dsolve", {"equations": ["diff(y(x),x,2)-3*diff(y(x),x)+2*y(x)=exp(x)"], "vars": ["x"], "funcs": ["y"]}),
]:
    d = E.execute(op, a)
    print("###", lab)
    print("RES", json.dumps(d.get("result"), ensure_ascii=False)[:400])
    print("VER", json.dumps(d.get("verification"), ensure_ascii=False)[:800])
