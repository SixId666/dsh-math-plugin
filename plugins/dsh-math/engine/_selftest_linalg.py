"""线性代数模块自检：通过 ``engine.execute``（与 Node 侧完全相同的调用路径）逐条测试。

运行：cd E:\\dsh-math\\engine; python _selftest_linalg.py

判定标准：
* 不允许出现 traceback
* 不允许出现 status=error 且 kind 为 internal / invalid_input（用作反例的 invalid_input 除外）
* 不允许把未求值结果标成 ok
"""

import sys, os

sys.path.insert(0, r"E:\dsh-math\engine")

import sympy as sp  # noqa: E402

from mathkit import engine as E  # noqa: E402
from mathkit import ops as OPS  # noqa: E402

TALLY = {"ok": 0, "partial": 0, "unsolved": 0, "error": 0}
FAILURES = []
INVALID_INPUT_EXPECTED = set()
TRACEBACKS = []


def run(label, op_name, args, *, expect_status=None, expect_error_kind=None, note=""):
    """执行一条用例并打印关键字段。"""
    try:
        payload = E.execute(op_name, args)
    except BaseException as exc:  # noqa: BLE001 - 自检脚本必须捕捉一切
        import traceback

        TRACEBACKS.append((label, traceback.format_exc()))
        print(f"\n!!! [{label}] 抛出异常 {type(exc).__name__}: {exc}")
        FAILURES.append(f"{label}: 抛出 {type(exc).__name__}: {exc}")
        return None

    status = payload.get("status")
    TALLY[status] = TALLY.get(status, 0) + 1
    error = payload.get("error") or {}
    verification = payload.get("verification") or {}
    result = payload.get("result") or {}

    print(f"\n=== [{label}] op={op_name} args={args}")
    print(f"    status={status}  success={payload.get('success')}  elapsed_ms={payload.get('elapsed_ms')}")
    print(f"    result.text={result.get('text')!r}")
    print(f"    result.latex={(result.get('latex') or '')[:160]!r}")
    print(f"    result.approx={result.get('approx')}")
    extra_keys = [k for k in result if k not in ("text", "latex", "approx")]
    for key in extra_keys:
        value = result[key]
        text = repr(value)
        print(f"    result.{key}={text[:220]}{'...' if len(text) > 220 else ''}")
    print(f"    conditions={payload.get('conditions')}")
    print(f"    verification.status={verification.get('status')}  methods={len(verification.get('methods') or [])}")
    for method in verification.get("methods") or []:
        print(f"        - {method.get('method')} -> agree={method.get('agree')}")
    if verification.get("note"):
        print(f"    verification.note={verification['note']}")
    if payload.get("warnings"):
        print(f"    warnings={payload['warnings']}")
    if error:
        print(f"    error={error}")
    if note:
        print(f"    (期望: {note})")

    # ---- 断言 ----
    if status == "error":
        kind = error.get("kind")
        if kind in ("internal", "unknown_operation"):
            FAILURES.append(f"{label}: status=error kind={kind} message={error.get('message')}")
        elif expect_error_kind and kind != expect_error_kind:
            FAILURES.append(f"{label}: 期望 error kind={expect_error_kind}，得到 {kind}")
    if expect_status and status != expect_status:
        FAILURES.append(f"{label}: 期望 status={expect_status}，得到 {status}")
    if expect_error_kind and error.get("kind") != expect_error_kind:
        FAILURES.append(f"{label}: 期望 error kind={expect_error_kind}，得到 {error.get('kind')}")
    if status == "ok" and not result.get("text") and not extra_keys:
        FAILURES.append(f"{label}: status=ok 但 result 为空（疑似未求值却标成功）")
    if status in ("ok", "partial") and verification.get("status") == "unverified" and label not in (
        "det-symbolic-4x4-nosym",
    ):
        # 允许但记录：成功却未验证的项应显式列出
        print("    [提示] 该成功结果没有独立验证")
    return payload


print("=" * 78)
print("注册情况：linalg 已导入 ->", "mathkit.linalg" in OPS.imported, "| 失败模块:", OPS.failed)
print("已知算子总数:", len(E.known_operations()))
lin_ops = [
    "det", "determinant", "math_det", "inv", "inverse", "math_inv", "rank", "math_rank",
    "transpose", "T", "math_transpose", "matmul", "matrix_multiply", "math_matmul",
    "matadd", "matrix_add", "trace", "math_trace", "rref", "math_rref",
    "solve_linear", "linear_system", "math_linear_system", "eigen", "eigenvalues", "math_eigen",
    "diagonalize", "math_diagonalize", "quadratic_form", "quadratic", "math_quadratic",
    "matrix_power", "matpow", "matrix_analysis", "matrix_info", "math_matrix",
]
missing = [name for name in lin_ops if E.resolve(name) is None]
print("缺失的线代算子:", missing or "无")
if missing:
    FAILURES.append(f"以下线代算子未注册: {missing}")
print("=" * 78)

# ---------------------------------------------------------------- det
run("det-2x2-int", "det", {"matrix": [[1, 2], [3, 4]]}, expect_status="ok")
run("det-symbolic-2x2", "det", {"matrix": [["a", "b"], ["c", "d"]]}, expect_status="ok")
run("det-3x3-int", "det", {"matrix": [[1, 2, 3], [4, 5, 6], [7, 8, 10]]}, expect_status="ok")
run("det-text-form", "det", {"matrix": "1,2;3,4"}, expect_status="ok")
run("det-alias-A", "determinant", {"A": "[[2,0],[0,3]]"}, expect_status="ok")
run("det-4x4", "det", {"matrix": [[2, 1, 0, 0], [1, 2, 1, 0], [0, 1, 2, 1], [0, 0, 1, 2]]}, expect_status="ok")
run("det-nonsquare", "det", {"matrix": [[1, 2, 3], [4, 5, 6]]}, expect_error_kind="invalid_input")

# ---------------------------------------------------------------- inv
run("inv-2x2", "inv", {"matrix": [[2, 1], [1, 1]]}, expect_status="ok")
run("inv-singular", "inv", {"matrix": [[1, 2], [2, 4]]}, expect_status="unsolved")
run("inv-symbolic", "inv", {"matrix": [["a", "b"], ["c", "d"]]}, expect_status="ok")
run("inv-nonsquare", "inv", {"matrix": [[1, 2, 3], [4, 5, 6]]}, expect_error_kind="invalid_input")

# ---------------------------------------------------------------- rank
run("rank-dependent-rows", "rank", {"matrix": [[1, 2, 3], [2, 4, 6]]}, expect_status="ok")
run("rank-full", "rank", {"matrix": [[1, 2], [3, 4]]}, expect_status="ok")
run("rank-zero", "rank", {"matrix": [[0, 0], [0, 0]]}, expect_status="ok")
run("rank-4x3", "rank", {"matrix": [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 0, 1]]}, expect_status="ok")

# ---------------------------------------------------------------- transpose / matmul / matadd / trace
run("transpose-2x3", "transpose", {"matrix": [[1, 2, 3], [4, 5, 6]]}, expect_status="ok")
run("transpose-alias-T", "T", {"M": "1,2;3,4"}, expect_status="ok")
run("matmul-ok", "matmul", {"matrix": [[1, 2], [3, 4]], "matrix_b": [[5, 6], [7, 8]]}, expect_status="ok")
run("matmul-dim-mismatch", "matmul", {"matrix": [[1, 2, 3], [4, 5, 6]], "matrix_b": [[1, 2], [3, 4]]}, expect_error_kind="invalid_input")
run("matmul-alias-AB", "matrix_multiply", {"A": "1,2;3,4", "B": "1,0;0,1"}, expect_status="ok")
run("matadd-ok", "matadd", {"matrix": [[1, 2], [3, 4]], "matrix_b": [[5, 6], [7, 8]]}, expect_status="ok")
run("matadd-dim-mismatch", "matrix_add", {"matrix": [[1, 2], [3, 4]], "matrix_b": [[1, 2, 3]]}, expect_error_kind="invalid_input")
run("trace-3x3", "trace", {"matrix": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}, expect_status="ok")
run("trace-nonsquare", "trace", {"matrix": [[1, 2, 3]]}, expect_error_kind="invalid_input")

# ---------------------------------------------------------------- rref
run("rref-basic", "rref", {"matrix": [[1, 2, 3], [2, 4, 6]]}, expect_status="ok")
run("rref-3x3", "rref", {"matrix": [[1, 2, -1, -4], [2, 3, -1, -11], [-2, 0, -3, 22]]}, expect_status="ok")

# ---------------------------------------------------------------- solve_linear
run("solve-unique", "solve_linear", {"matrix": [[1, 1], [1, -1]], "rhs": "3,1"}, expect_status="ok")
run("solve-unique-3x3", "solve_linear", {"matrix": [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], "rhs": "8,-11,-3"}, expect_status="ok")
run("solve-no-solution", "solve_linear", {"matrix": [[1, 1], [2, 2]], "rhs": [1, 3]}, expect_status="unsolved")
run("solve-infinite", "solve_linear", {"matrix": [[1, 1], [2, 2]], "rhs": [1, 2]}, expect_status="ok")
run("solve-underdetermined", "solve_linear", {"matrix": [[1, 2, 3], [4, 5, 6]], "rhs": [7, 8]}, expect_status="ok")
run("solve-rhs-mismatch", "solve_linear", {"matrix": [[1, 2], [3, 4]], "rhs": [1, 2, 3]}, expect_error_kind="invalid_input")
run("solve-homogeneous", "solve_linear", {"matrix": [[1, 2], [2, 4]], "rhs": [0, 0]}, expect_status="ok")

# ---------------------------------------------------------------- eigen
run("eigen-symmetric-2x2", "eigen", {"matrix": [[2, 1], [1, 2]]}, expect_status="ok")
run("eigen-defective", "eigen", {"matrix": [[1, 1], [0, 1]]}, expect_status="ok")
run("eigen-3x3-distinct", "eigen", {"matrix": [[1, 2, 3], [0, 4, 5], [0, 0, 6]]}, expect_status="ok")
run("eigen-rotation-complex", "eigen", {"matrix": [[0, -1], [1, 0]]}, expect_status="ok")
run("eigen-symbolic", "eigen", {"matrix": [["a", 0], [0, "b"]]}, expect_status="ok")

# ---------------------------------------------------------------- diagonalize
run("diag-diagonal", "diagonalize", {"matrix": [[2, 0], [0, 3]]}, expect_status="ok")
run("diag-defective", "diagonalize", {"matrix": [[1, 1], [0, 1]]}, expect_status="unsolved")
run("diag-symmetric", "diagonalize", {"matrix": [[2, 1], [1, 2]]}, expect_status="ok")
run("diag-3x3-distinct", "diagonalize", {"matrix": [[1, 2, 3], [0, 4, 5], [0, 0, 6]]}, expect_status="ok")

# ---------------------------------------------------------------- quadratic_form
run("quad-2var", "quadratic_form", {"expr": "x1^2+2*x1*x2+3*x2^2"}, expect_status="ok")
run("quad-3var-indefinite", "quadratic_form", {"expr": "x1^2+x2^2-x3^2"}, expect_status="ok")
run("quad-matrix", "quadratic_form", {"matrix": [[2, -1], [-1, 2]]}, expect_status="ok")
run("quad-negdef", "quadratic_form", {"expr": "-x1^2-x2^2"}, expect_status="ok")
run("quad-semidef", "quadratic_form", {"expr": "x1^2+2*x1*x2+x2^2"}, expect_status="ok")
run("quad-symbolic", "quadratic_form", {"expr": "a*x1^2+2*b*x1*x2+c*x2^2"}, expect_status="ok")
run("quad-nonsymmetric", "quadratic_form", {"matrix": [[1, 2], [0, 1]]}, expect_error_kind="invalid_input")
run("quad-cubic", "quadratic_form", {"expr": "x1^3+x2^2"}, expect_error_kind="invalid_input")

# ---------------------------------------------------------------- matrix_power
run("matpow-3", "matrix_power", {"matrix": [[1, 1], [0, 1]], "order": 3}, expect_status="ok")
run("matpow-0", "matrix_power", {"matrix": [[1, 2], [3, 4]], "order": 0}, expect_status="ok")
run("matpow-n-alias", "matpow", {"matrix": [[1, 1], [0, 1]], "n": 5}, expect_status="ok")
run("matpow-negative", "matrix_power", {"matrix": [[1, 1], [0, 1]], "order": -1}, expect_error_kind="invalid_input")
run("matpow-13-symbolic-fallback", "matrix_power", {"matrix": [[1, 1], [0, 1]], "order": 13}, expect_status="ok")

# ---------------------------------------------------------------- matrix_analysis
run("summary-2x2", "matrix_analysis", {"matrix": [[2, 1], [1, 2]]}, expect_status="ok")
run("summary-singular", "matrix_analysis", {"matrix": [[1, 2], [2, 4]]}, expect_status="ok")
run("summary-nonsquare", "matrix_analysis", {"matrix": [[1, 2, 3], [4, 5, 6]]}, expect_status="ok")
run("summary-rref-true", "math_matrix", {"matrix": [[1, 2], [3, 4]], "rref": True}, expect_status="ok")

# ---------------------------------------------------------------- 符号慢操作保护
run("det-symbolic-4x4", "det", {"matrix": [["a", "b", 1, 0], ["c", "d", 0, 1], [1, 0, "e", "f"], [0, 1, "g", "h"]]}, expect_status="ok")
run("inv-symbolic-3x3", "inv", {"matrix": [["a", 1, 0], [0, "b", 1], [1, 0, "c"]]}, expect_status="ok")
run("det-7x7-symbolic-warning", "det", {"matrix": [[f"a{i}{j}" for j in range(7)] for i in range(7)]}, expect_status=None)

# ---------------------------------------------------------------- 边界：空输入 / 未知算子
run("det-missing-matrix", "det", {}, expect_error_kind="invalid_input")
run("solve-missing-rhs", "solve_linear", {"matrix": [[1, 2], [3, 4]]}, expect_error_kind="invalid_input")

print("\n" + "=" * 78)
print("统计：", TALLY)
print("失败项：")
if FAILURES:
    for item in FAILURES:
        print("  -", item)
else:
    print("  无")
print("traceback：", len(TRACEBACKS))
for label, tb in TRACEBACKS:
    print(f"  --- {label} ---")
    print(tb)
print("=" * 78)
sys.exit(1 if (FAILURES or TRACEBACKS) else 0)
