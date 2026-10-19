"""算子汇总注册入口。

``worker.py`` 只导入本模块，各 handler 模块在此统一导入以触发 ``@op`` 注册。
导入失败不会让 worker 崩溃：缺失模块会被记录并通过 ``@ops`` 暴露，
这样「部分能力不可用」也能被明确告知，而不是静默消失。
"""

from __future__ import annotations

import importlib
from typing import Any

_HANDLER_MODULES = (
    "mathkit.calculus",
    "mathkit.linalg",
    "mathkit.probability",
    "mathkit.numeric",
    "mathkit.symbolic_extra",
    "mathkit.vector_calc",
    "mathkit.probability_extra",
    "mathkit.verifier",
)

imported: list[str] = []
failed: dict[str, str] = {}

for _name in _HANDLER_MODULES:
    try:
        importlib.import_module(_name)
        imported.append(_name)
    except Exception as _exc:  # noqa: BLE001 - 单个模块缺失不应影响整体
        failed[_name] = f"{type(_exc).__name__}: {_exc}"


def status() -> dict[str, Any]:
    """返回模块加载情况，供 ``@ops`` 与诊断使用。"""
    return {"imported": list(imported), "failed": dict(failed)}
