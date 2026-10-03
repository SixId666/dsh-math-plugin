import sys
sys.path.insert(0, r"E:\dsh-math\engine")
import sympy as sp
from mathkit import ast as A

print("== parse_matrix text forms ==")
for t in ["[[1,2],[3,4]]", "1,2;3,4", "1 2; 3 4", "[[1, 2], [3, 4]]", "[[a,b],[c,d]]", "[1,2,3]"]:
    try:
        m = A.parse_matrix(text=t)
        print(f"  {t!r:22} -> shape={m.shape} {A.to_text(m)}")
    except Exception as e:
        print(f"  {t!r:22} -> ERR {type(e).__name__}: {e}")

print("== parse_matrix rows with python numbers ==")
try:
    m = A.parse_matrix([[1,2],[3,4]])
    print("  rows=[[1,2],[3,4]] ->", A.to_text(m))
except Exception as e:
    print("  ERR", e)

print("== x1 parsing ==")
for s in ["x1^2+2*x1*x2+3*x2^2", "x1^2+x2^2-x3^2"]:
    e = A.parse(s, symbols=["x1","x2","x3"])
    print(f"  {s!r} -> {e}")
    print("     free:", sorted([str(z) for z in e.free_symbols]))
    p = sp.Poly(e, sp.Symbol("x1"), sp.Symbol("x2"))
    print("     deg:", p.total_degree())

print("== x1 without whitelist ==")
try:
    e = A.parse("x1^2")
    print("  ->", e, sorted([str(z) for z in e.free_symbols]))
except Exception as ex:
    print("  ERR", ex)

print("== reserved containing x1 ==")
print("  ", "x1" in A._RESERVED, "x1" in A._FUNC_ALIASES)

print("== det methods ==")
M = sp.Matrix([[1,2,3],[4,5,6],[7,8,10]])
print("  default:", M.det(), "berkowitz:", M.det(method="berkowitz"), "bareiss:", M.det(method="bareiss"), "lu:", M.det(method="lu"))
print("== det default on 2x2 ==")
print("  ", sp.Matrix([[1,2],[3,4]]).det(method="bareiss"))

print("== eigenvals / gauss_jordan_solve ==")
print("  ", sp.Matrix([[2,1],[1,2]]).eigenvals())
gj = sp.Matrix([[1,1],[1,-1]]).gauss_jordan_solve(sp.Matrix([3,1]))
print("  gj:", gj)
gj2 = sp.Matrix([[1,1],[1,1]]).gauss_jordan_solve(sp.Matrix([2,2]))
print("  gj singular:", gj2)
print("  rref pivots:", sp.Matrix([[1,2,3],[2,4,6]]).rref())
print("== charpoly shape ==")
print("  ", sp.Matrix([[2,1],[1,2]]).charpoly().as_expr())
print("== symbols/keyword-only check ==")
import inspect
print("  parse_matrix sig:", inspect.signature(A.parse_matrix))
