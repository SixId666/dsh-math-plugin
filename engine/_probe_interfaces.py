"""接口自检：确认 ast / engine 的实际调用形态（不依赖猜测）。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sympy as sp
from mathkit import ast as A
from mathkit import engine as E
from mathkit import sympy_core as C
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_worker_mod", os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py"))
_wm = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_wm)
Cache = _wm.Cache

print("== parse ==")
for s in ["x^2*sin(x)", "\\frac{1}{x}", "1/(x^2+1)", "lg(x)", "e^{x^2}", "arcsin(x)", "∞"]:
    try:
        e = A.parse(s, symbols=["x"])
        print(f"  {s!r:24} -> {e!r:40} text={A.to_text(e)!r:20} latex={A.to_latex(e)!r}")
    except Exception as exc:
        print(f"  {s!r:24} -> ERROR {type(exc).__name__}: {exc}")

print("== to_text / to_latex on containers ==")
print("  list:", A.to_text([sp.Integer(1), sp.sqrt(2)]), "|", A.to_latex([sp.Integer(1), sp.sqrt(2)]))
print("  matrix:", A.to_text(sp.Matrix([[1, 2], [3, 4]])), "|", A.to_latex(sp.Matrix([[1, 2], [3, 4]])))
print("  dict:", A.to_text({sp.Symbol("x"): sp.Integer(1)}))

print("== canonical_key ==")
print(" ", A.canonical_key(sp.sin(sp.Symbol("x")) ** 2))

print("== parse_all ==")
print(" ", A.parse_all(["x+1", "x-1"], symbols=["x"]))
print("== parse_matrix ==")
print(" ", A.parse_matrix([[1, 2], [3, "x"]], symbols=["x"]))
try:
    print("  text form:", A.parse_matrix(None, text="[[1,2],[3,4]]", symbols=["x"]))
except Exception as exc:
    print("  text form ERROR:", type(exc).__name__, exc)

print("== engine ==")
print("  ops:", E.known_operations()[:5])
print("  resolve alias:", E.resolve("diff") if hasattr(E, "resolve") else "n/a")
print("  execute unknown:", E.execute("nope", {}))
r = E.execute("noop", {"expr": "x"})
print("  execute missing op:", r["status"], r.get("error", {}).get("message"))

print("== result round trip ==")
from mathkit.result import MathResult
m = MathResult.ok("diff", method="sympy diff").set_input(expr="x^2").set_result(sp.Derivative(sp.sin(sp.Symbol("x")), sp.Symbol("x")))
print(" ", m.to_dict())

print("== cache ==")
c = Cache()
c.store("diff", {"expr": "x"}, {"status": "ok", "result": {"text": "1"}})
print(" ", c.get("diff", {"expr": "x"}))
print(" ", c.get("diff", {"expr": "y"}))
