"""探测：哪些积分真的求不出解析解（用于专项测试三）。"""
import sys, json
sys.path.insert(0, r"E:\dsh-math\engine")
sys.stdout.reconfigure(encoding="utf-8")
from mathkit import engine as E  # noqa: E402
from mathkit import ops  # noqa: E402,F401  （导入后算子才会注册）

CASES = [
    ("exp(-x^4)", {}),
    ("sqrt(1+x^3)", {}),
    ("exp(x^3)", {}),
    ("sin(x^2)", {}),
    ("1/log(x)", {}),
    ("exp(-x^4)", {"lower": "0", "upper": "1"}),
    ("exp(x^2)", {"lower": "0", "upper": "1"}),
    ("x^x", {}),
    ("sin(sin(x))", {}),
    ("sqrt(1+x^3)", {"lower": "0", "upper": "1"}),
]

for expr, extra in CASES:
    args = {"expr": expr, **extra}
    payload = E.execute("integrate", args, timeout=20)
    status = payload.get("status")
    result = payload.get("result") or {}
    text = (result.get("text") or "")[:90] if isinstance(result, dict) else ""
    err = (payload.get("error") or {}).get("message", "")
    print(f"{expr!r:22} {json.dumps(extra):24} -> {status:9} {text} {err[:70]}")
