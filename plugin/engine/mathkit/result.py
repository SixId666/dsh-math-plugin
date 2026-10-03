"""统一结果封装（用户规格 5.3）。

所有工具返回同一个结构：

    success      是否严格成功（False 表示失败或未求得结果）
    status       ok | partial | unsolved | error
    operation    执行的数学操作
    input        规范化后的输入
    result       计算结果（含 text/latex/近似值）
    conditions   成立条件 / 定义域限制
    verification 验证状态（independent / cross / numeric / unverified）
    method       使用的计算方法
    error        失败原因
    warnings     需要注意的问题
"""

from __future__ import annotations

import time
from typing import Any

OK = "ok"
PARTIAL = "partial"
UNSOLVED = "unsolved"
ERROR = "error"

UNVERIFIED = "unverified"


def _unwrap(value: Any) -> dict[str, Any]:
    """把任意 SymPy 对象 / 数值变成 {text, latex, approx} 三元组。"""
    from . import ast as A

    if value is None:
        return {}
    out: dict[str, Any] = {}
    try:
        out["text"] = A.to_text(value)
        out["latex"] = A.to_latex(value)
    except Exception:  # noqa: BLE001
        out["text"] = str(value)
    approx = _approx(value)
    if approx is not None:
        out["approx"] = approx
    return out


def _approx(value: Any) -> Any:
    import sympy as sp

    try:
        if isinstance(value, sp.MatrixBase):
            if value.free_symbols:
                return None
            return [[_fmt_num(x) for x in row] for row in value.tolist()]
        if isinstance(value, (list, tuple)):
            if any(getattr(v, "free_symbols", set()) for v in value):
                return None
            return [_fmt_num(v) for v in value]
        if isinstance(value, dict):
            return {str(k): _approx(v) for k, v in value.items()}
        if isinstance(value, sp.Basic):
            if value.free_symbols:
                return None
            if value.is_number:
                return _fmt_num(value)
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float, str)):
            return value
    except Exception:  # noqa: BLE001
        return None
    return None


def _fmt_num(value: Any) -> Any:
    """把数值结果转成 JSON 友好的近似值。

    精度取 **17 位有效数字**（double 的完整往返精度）：早先用 ``sp.N(value, 12)``
    只留 12 位，``pi`` 会变成 ``3.1415926535896688`` —— **第 13 位起就是错的**。
    17 位再交给 ``float`` 打印时是 Python 的最短往返表示（``3.141592653589793``），
    既准确又不会出现一串无意义的尾数。需要更高精度的调用方读 raw 里的
    ``approximate_value`` / ``text``（那里是 mpmath 的 ``digits`` 位结果）。
    """
    import sympy as sp

    try:
        if isinstance(value, sp.Basic):
            if not value.is_number:
                return str(value)
            num = sp.N(value, 17)
            if num.is_Integer:
                return int(num)
            if num.is_real:
                return float(num)
            return str(num)
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return value
    except Exception:  # noqa: BLE001
        return str(value)
    return value


def wrap_value(value: Any) -> Any:
    """对外暴露，供各模块把单个结果转成结构化表达。"""
    return _unwrap(value)


def json_safe(value: Any) -> Any:
    """把任意值递归转换成可 JSON 序列化的形式（SymPy 对象转文本，未知对象转 str）。

    各模块用 ``set_raw`` 写入的片段常常夹带原生 SymPy 对象（例如
    ``linalg.quadratic_form`` 会把 ``sp.S.One`` 放进 result），
    到 ``json.dumps`` 时才炸成 ``TypeError: Object of type One is not JSON serializable``，
    让整个工具调用变成内部错误。出口统一消毒，比逐个模块打补丁可靠。
    """
    import sympy as sp

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else str(value)
    if isinstance(value, sp.Basic):
        if isinstance(value, sp.MatrixBase):
            return json_safe(value.tolist())
        try:
            return _unwrap(value).get("text", str(value))
        except Exception:  # noqa: BLE001
            return str(value)
    if isinstance(value, sp.MatrixBase):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    return str(value)


class MathResult:
    """一次计算的结果对象，链式补充信息后序列化为 dict。"""

    def __init__(self, operation: str, *, method: str | None = None):
        self.operation = operation
        self.method = method or operation
        self.status = OK
        self.input: dict[str, Any] = {}
        self.result: dict[str, Any] = {}
        self.conditions: list[str] = []
        self.verification: dict[str, Any] = {"status": UNVERIFIED, "methods": []}
        self.error: dict[str, Any] | None = None
        self.warnings: list[str] = []
        self.extra: dict[str, Any] = {}
        self.t0 = time.perf_counter()

    # -- builders ---------------------------------------------------------
    @classmethod
    def ok(cls, operation: str, *, method: str | None = None) -> "MathResult":
        return cls(operation, method=method)

    @classmethod
    def fail(cls, operation: str, message: str, *, kind: str = "solver", method: str | None = None) -> "MathResult":
        r = cls(operation, method=method)
        r.status = ERROR
        r.error = {"kind": kind, "message": message}
        return r

    @classmethod
    def unsolved(cls, operation: str, message: str, *, method: str | None = None) -> "MathResult":
        """引擎无法求得结果——必须与「失败」和「成功」区分。"""
        r = cls(operation, method=method)
        r.status = UNSOLVED
        r.error = {"kind": "unsolved", "message": message}
        return r

    def set_input(self, **kwargs: Any) -> "MathResult":
        self.input.update({k: v for k, v in kwargs.items() if v is not None})
        return self

    def set_result(self, value: Any = None, **kwargs: Any) -> "MathResult":
        if value is not None:
            self.result.update(_unwrap(value))
        self.result.update({k: v for k, v in kwargs.items() if v is not None})
        return self

    def set_raw(self, **kwargs: Any) -> "MathResult":
        """直接写入已经结构化的结果片段（未经 SymPy 转换）。"""
        for key, value in kwargs.items():
            if value is not None:
                self.result[key] = value
        return self

    def add_condition(self, *conditions: str) -> "MathResult":
        for c in conditions:
            if c and c not in self.conditions:
                self.conditions.append(c)
        return self

    def add_warning(self, *warnings: str) -> "MathResult":
        for w in warnings:
            if w and w not in self.warnings:
                self.warnings.append(w)
        return self

    def set_method(self, method: str) -> "MathResult":
        self.method = method
        return self

    def verify(self, *, status: str, methods: list[dict[str, Any]] | None, note: str | None = None) -> "MathResult":
        self.verification = {
            "status": status,
            "methods": methods or [],
        }
        if note:
            self.verification["note"] = note
        return self

    def partial(self, message: str) -> "MathResult":
        self.status = PARTIAL
        self.add_warning(message)
        return self

    def set_extra(self, **kwargs: Any) -> "MathResult":
        self.extra.update({k: v for k, v in kwargs.items() if v is not None})
        return self

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        self.elapsed_ms = round((time.perf_counter() - self.t0) * 1000, 2)
        payload: dict[str, Any] = {
            "success": self.status == OK,
            "status": self.status,
            "operation": self.operation,
            "method": self.method,
            "input": json_safe(self.input),
            "result": json_safe(self.result),
            "conditions": [str(c) for c in self.conditions],
            "verification": json_safe(self.verification),
            "warnings": [str(w) for w in self.warnings],
            "elapsed_ms": self.elapsed_ms,
        }
        if self.error is not None:
            payload["error"] = json_safe(self.error)
        if self.extra:
            payload["extra"] = json_safe(self.extra)
        return payload
