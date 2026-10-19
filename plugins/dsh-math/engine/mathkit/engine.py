"""操作注册表与调度器。

Python worker 收到 ``{"op": "...", "args": {...}}`` 后交给这里执行。
调度器的职责：

1. 参数归一化（别名、单数/复数、数学等价写法）
2. 超时保护（SymPy 的慢操作不能拖死 worker）
3. 异常 -> 结构化错误（绝不抛出未捕获异常，绝不返回伪结果）
4. 结果缓存与「同一计算不重复执行」的账本（用户规格 6.2 / 6.3 / 10）

各模块通过 :func:`op` 装饰器注册算子。
"""

from __future__ import annotations

import inspect
import os
import signal
import threading
import time
from typing import Any, Callable

from . import result as R

_HANDLERS: dict[str, Callable[..., R.MathResult]] = {}
_ALIASES: dict[str, str] = {}

# 一批输入参数别名：不同工具、不同模型的自然写法都指向同一语义参数
ARG_ALIASES: dict[str, str] = {
    "expr": "expr",
    "expression": "expr",
    "e": "expr",
    "f": "expr",
    "input": "expr",
    "formula": "expr",
    "func": "expr",
    "f_expr": "expr",
    "var": "var",
    "variable": "var",
    "wrt": "var",
    "x": "var",
    "variables": "vars",
    "vars": "vars",
    "symbols": "symbols",
    "point": "point",
    "at": "point",
    "x0": "point",
    "dir": "dir",
    "direction": "dir",
    "side": "dir",
    "lower": "lower",
    "upper": "upper",
    "lo": "lower",
    "hi": "upper",
    "a": "lower",
    "b": "upper",
    "n": "order",
    "order": "order",
    "times": "order",
    "degree": "order",
    "terms": "order",
    "equations": "equations",
    "eqs": "equations",
    "system": "equations",
    "equation": "equations",
    "A": "matrix",
    "B": "matrix_b",
    "M": "matrix",
    "matrix": "matrix",
    "mat": "matrix",
    "matrix1": "matrix",
    "matrix2": "matrix_b",
    "matrix_a": "matrix",
    "lhs": "lhs",
    "rhs": "rhs",
    "target": "target",
    "candidate": "candidate",
    "result_expr": "candidate",
    "answer": "candidate",
    "claim": "claim",
    "claim_type": "kind",
    "type": "kind",
    "op_type": "kind",
    "n_trials": "trials",
    "samples": "trials",
}
# 参数值别名（字符串枚举）
VALUE_ALIASES: dict[str, dict[str, str]] = {
    "dir": {
        "+": "+", "-": "-", "left": "-", "right": "+",
        "左": "-", "右": "+", "左极限": "-", "右极限": "+",
        "从左侧": "-", "从右侧": "+", "小于": "-", "大于": "+",
        "below": "-", "above": "+", "lt": "-", "gt": "+",
    },
}


class OperatorTimeout(Exception):
    pass


def op(name: str, *aliases: str) -> Callable[[Callable[..., R.MathResult]], Callable[..., R.MathResult]]:
    """把一个函数注册为算子。"""

    def deco(func: Callable[..., R.MathResult]) -> Callable[..., R.MathResult]:
        _HANDLERS[name] = func
        for alias in aliases:
            _ALIASES[alias] = name
        return func

    return deco


def register(name: str, func: Callable[..., R.MathResult], *aliases: str) -> None:
    _HANDLERS[name] = func
    for alias in aliases:
        _ALIASES[alias] = name


def known_operations() -> list[str]:
    return sorted(set(_HANDLERS) | set(_ALIASES))


def resolve(name: str) -> Callable[..., R.MathResult] | None:
    if not name:
        return None
    key = str(name).strip()
    if key in _HANDLERS:
        return _HANDLERS[key]
    if key in _ALIASES:
        return _HANDLERS[_ALIASES[key]]
    # 允许 math_ 前缀 / 大小写差异
    stripped = key[5:] if key.startswith("math_") else key
    for candidate in (stripped, stripped.lower()):
        if candidate in _HANDLERS:
            return _HANDLERS[candidate]
        if candidate in _ALIASES:
            return _HANDLERS[_ALIASES[candidate]]
    return None


def canonical_op(name: str) -> str | None:
    func = resolve(name)
    if func is None:
        return None
    for key, value in _HANDLERS.items():
        if value is func:
            return key
    return None


# ---------------------------------------------------------------------------
# 参数归一化
# ---------------------------------------------------------------------------

def normalize_args(args: dict[str, Any] | None) -> dict[str, Any]:
    if not args:
        return {}
    out: dict[str, Any] = {}
    for raw_key, value in args.items():
        if value is None:
            continue
        key = ARG_ALIASES.get(str(raw_key), str(raw_key))
        if key in VALUE_ALIASES and isinstance(value, str):
            value = VALUE_ALIASES[key].get(value.strip().lower(), VALUE_ALIASES[key].get(value.strip(), value))
        out[key] = value
    return out


def _call_with_declared_kwargs(func: Callable[..., R.MathResult], args: dict[str, Any]) -> R.MathResult:
    """按函数签名裁剪参数；多余参数交给 **kwargs，否则忽略但记录。"""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return func(**args)

    accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if accepts_var_kw:
        return func(**args)

    allowed = {
        name
        for name, p in sig.parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    filtered = {k: v for k, v in args.items() if k in allowed}
    dropped = sorted(str(k) for k in args if k not in allowed)
    missing = [
        name
        for name, p in sig.parameters.items()
        if p.default is inspect.Parameter.empty
        and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and name not in filtered
    ]
    if missing:
        raise ValueError(f"缺少必需参数: {', '.join(missing)}")
    result = func(**filtered)
    # 参数名写错时不能表现为「无效果」：如实告诉调用方哪些参数被忽略了。
    if dropped and hasattr(result, "add_warning"):
        result.add_warning(
            "以下参数该算子不接受，已被忽略：" + ", ".join(dropped) + "（请检查参数名是否写错）"
        )
    return result


# ---------------------------------------------------------------------------
# 超时保护
# ---------------------------------------------------------------------------

class _Timeout:
    """在 Unix 上用 SIGALRM；Windows 上用守护线程 + 结果栅栏。

    SymPy 的某些调用无法被强制打断，因此 Windows 分支只保证「不阻塞 worker」：
    watchdog 触发后，调度器立刻向上层返回超时错误，后台线程自然结束。
    """

    def __init__(self, seconds: float):
        self.seconds = float(seconds)

    def run(self, func: Callable[[], Any]) -> Any:
        if self.seconds <= 0:
            return func()
        if hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread():
            def _handler(signum, frame):  # noqa: ANN001, ARG001
                raise OperatorTimeout()

            old = signal.signal(signal.SIGALRM, _handler)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
            try:
                return func()
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old)

        box: dict[str, Any] = {}

        def target() -> None:
            try:
                box["value"] = func()
            except BaseException as exc:  # noqa: BLE001
                box["error"] = exc

        worker = threading.Thread(target=target, daemon=True)
        worker.start()
        worker.join(self.seconds)
        if worker.is_alive():
            raise OperatorTimeout()
        if "error" in box:
            raise box["error"]
        return box.get("value")


# ---------------------------------------------------------------------------
# 调度
# ---------------------------------------------------------------------------

def execute(op_name: str, args: dict[str, Any] | None = None, *, timeout: float | None = None) -> dict[str, Any]:
    """执行一个算子并返回结构化结果字典。永不抛出。"""
    started = time.perf_counter()
    func = resolve(op_name)
    if func is None:
        payload = R.MathResult.fail(
            str(op_name), f"未知操作 {op_name!r}；可用操作：{', '.join(known_operations())}", kind="unknown_operation"
        ).to_dict()
        payload["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return payload

    canonical = canonical_op(op_name) or str(op_name)
    normalized = normalize_args(args)
    limit = 30.0 if timeout is None else float(timeout)

    def _fail(message: str, kind: str) -> dict[str, Any]:
        """失败路径同样要带真实耗时（否则调用方看到的 elapsed_ms 恒为 0）。"""
        payload = R.MathResult.fail(canonical, message, kind=kind).to_dict()
        payload["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return payload

    try:
        outcome = _Timeout(limit).run(lambda: _call_with_declared_kwargs(func, normalized))
    except OperatorTimeout:
        return _fail(
            f"计算超时（>{limit:g}s）。请简化输入、分段计算，或改用数值方法（math_numeric）。",
            "timeout",
        )
    except ValueError as exc:
        return _fail(str(exc), "invalid_input")
    except RecursionError:
        return _fail("计算深度溢出：表达式过于复杂，请先化简或降阶。", "recursion")
    except MemoryError:
        return _fail("内存不足：请降低问题规模（如矩阵维数、展开阶数）。", "memory")
    except Exception as exc:  # noqa: BLE001 - 任何未预期异常都必须变成结构化错误
        detail = f"{type(exc).__name__}: {exc}"
        if os.environ.get("DSH_MATH_DEBUG"):
            import traceback

            detail += "\n" + traceback.format_exc()
        return _fail(detail, "internal")

    if not isinstance(outcome, R.MathResult):
        payload = R.MathResult.ok(canonical)
        if outcome is not None:
            payload.set_result(outcome)
        outcome = payload

    payload = outcome.to_dict()
    payload["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return payload
