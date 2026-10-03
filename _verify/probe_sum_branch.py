"""验证 series_sum 的 Piecewise 主分支提取（幂级数收敛域）。"""
import json
import sys

sys.path.insert(0, r"E:\dsh-math\engine")
sys.stdout.reconfigure(encoding="utf-8")

from mathkit import engine as E  # noqa: E402
from mathkit import ops  # noqa: E402,F401  （必须显式导入才会注册全部算子）

CASES = [
    ("幂级数 Σx^n (n≥0)", {"expr": "x^n", "var": "n", "lower": "0", "upper": "oo"}),
    ("幂级数 Σx^n/n (n≥1)", {"expr": "x^n/n", "var": "n", "lower": "1", "upper": "oo"}),
    ("Σ1/n²", {"expr": "1/n^2", "var": "n", "lower": "1", "upper": "oo"}),
    ("Σ1/n（发散）", {"expr": "1/n", "var": "n", "lower": "1", "upper": "oo"}),
    ("Σ(1/2)^n", {"expr": "(1/2)^n", "var": "n", "lower": "1", "upper": "oo"}),
]

for label, args in CASES:
    payload = E.execute("sum", args)
    result = payload.get("result") or {}
    print(f"[{payload.get('status')}] {label}")
    print(f"    结果: {result.get('text')}")
    for condition in payload.get("conditions") or []:
        if "收敛" in condition:
            print(f"    条件: {condition}")
    for warning in payload.get("warnings") or []:
        print(f"    提醒: {warning[:160]}")
    print(f"    验证: {(payload.get('verification') or {}).get('status')}")
    print(f"    原始 JSON 长度: {len(json.dumps(payload, ensure_ascii=False))}")
