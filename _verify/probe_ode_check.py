"""探针：ode_check / dsolve 的输入拼写与残差判定（历史 bug：Derivative 项被求成 0）。"""
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"E:\dsh-math\engine")

from mathkit import engine as E  # noqa: E402
from mathkit import ops  # noqa: E402

CASES = [
    ("y' = x*y", "y = C1*exp(x^2/2)"),
    ("y' = x*y", "y(x) = C1*exp(x^2/2)"),
    ("y'(x) = x*y(x)", "y(x) = C1*exp(x^2/2)"),
    ("y' - x*y = 0", "y = C1*exp(x^2/2)"),
    ("y' = 2*x", "y = x^2 + C1"),
    ("y'' + y = 0", "y = C1*cos(x) + C2*sin(x)"),
    ("y' = x*y", "y = C1*exp(x)"),
]

for equations, candidate in CASES:
    out = E.execute("ode_check", {"equations": equations, "candidate": candidate})
    print("=" * 72)
    print(f"eq={equations!r} cand={candidate!r}")
    print("  status:", out.get("status"), "| text:", out.get("text"))
    raw = out.get("raw") or {}
    if raw.get("residuals"):
        print("  residuals:", raw["residuals"])
    if out.get("error"):
        print("  error:", out["error"].get("message"))
    for m in (out.get("verification") or {}).get("methods") or []:
        print("  method:", {k: v for k, v in m.items() if k in ("name", "method", "passed", "agree")})
