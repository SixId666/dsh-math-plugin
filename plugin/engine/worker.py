"""常驻 Python worker：以 newline-delimited JSON 与 Node 插件通信。

协议（stdin 一行一个 JSON 请求，stdout 一行一个 JSON 响应）：

    请求  {"id": 1, "op": "diff", "args": {...}, "timeout": 20}
    成功  {"id": 1, "ok": true, "data": {...MathResult...}}
    失败  {"id": 1, "ok": false, "data": {...MathResult(status=error)...}}

其他控制操作：

    {"id": 2, "op": "@ping"}
    {"id": 3, "op": "@ops"}
    {"id": 4, "op": "@cache_stats"}
    {"id": 5, "op": "@cache_clear"}
    {"id": 6, "op": "@shutdown"}

**stdout 只承载协议**：所有日志走 stderr，保证 Node 侧解析不会被污染。
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import Any

# 允许 `python worker.py` 直接运行时找到包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# 协议通道必须固定为 UTF-8。
# Windows 下 stdout 接管道时默认编码是 GBK，一旦响应里出现 GBK 无法表示的字符
# （例如 math gradient 的条件文本、Lagrange 乘子里的 ∇ / λ），json.dumps 写 stdout 就会抛
#     UnicodeEncodeError: 'gbk' codec can't encode character '\u2207'
# 并且**直接终止 worker 进程**，Node 侧管道随之断开、整个插件变成哑巴。
# 必须在任何可能产出数学符号的导入之前完成 reconfigure。
# ---------------------------------------------------------------------------
for _stream in (sys.stdout, sys.stderr):
    try:
        if (_stream.encoding or "").lower().replace("-", "") != "utf8":
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001  某些环境（重定向到非文本对象）不支持 reconfigure
        pass
del _stream

from mathkit import engine  # noqa: E402
from mathkit import result as R  # noqa: E402
from mathkit import ops  # noqa: E402,F401  触发算子注册

VERSION = "1.0.0"
CACHE_LIMIT = 512


class Cache:
    """表达式规范化 + 参数条件共同决定键；失败结果也记账（用户规格 6.2 / 6.3）。"""

    def __init__(self, limit: int = CACHE_LIMIT):
        self.limit = limit
        self.success: dict[str, dict[str, Any]] = {}
        self.failures: dict[str, dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        self.sequence: list[str] = []

    @staticmethod
    def key(op_name: str, args: dict[str, Any]) -> str:
        try:
            payload = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:  # noqa: BLE001
            payload = repr(args)
        return f"{op_name}::{payload}"

    def get(self, op_name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        key = self.key(op_name, args)
        if key in self.success:
            self.hits += 1
            hit = dict(self.success[key])
            hit["cached"] = True
            return hit
        self.misses += 1
        return None

    def failed_before(self, op_name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        key = self.key(op_name, args)
        return self.failures.get(key)

    def store(self, op_name: str, args: dict[str, Any], payload: dict[str, Any]) -> None:
        key = self.key(op_name, args)
        if payload.get("status") == R.OK:
            self.success[key] = payload
            self.sequence.append(key)
            while len(self.sequence) > self.limit:
                evicted = self.sequence.pop(0)
                self.success.pop(evicted, None)
        else:
            self.failures[key] = {
                "status": payload.get("status"),
                "error": payload.get("error"),
                "at": time.time(),
            }
            if len(self.failures) > self.limit:
                oldest = sorted(self.failures.items(), key=lambda kv: kv[1].get("at", 0))[: len(self.failures) - self.limit]
                for k, _ in oldest:
                    self.failures.pop(k, None)

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "success_entries": len(self.success),
            "failure_entries": len(self.failures),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }

    def clear(self) -> None:
        self.success.clear()
        self.failures.clear()
        self.sequence.clear()
        self.hits = 0
        self.misses = 0


CACHE = Cache()
_CACHEABLE = True


def log(message: str) -> None:
    print(f"[math-worker] {message}", file=sys.stderr, flush=True)


def _respond(payload: dict[str, Any]) -> None:
    # 兜底：即使某个环境的 stdout 仍不是 UTF-8，也绝不让一次写入失败杀死整个 worker
    # （宁可把无法编码的字符替换掉，也不能让 Node 侧管道断开）。
    try:
        line = json.dumps(payload, ensure_ascii=False, default=str)
        sys.stdout.write(line + "\n")
    except UnicodeEncodeError:
        safe = json.dumps(payload, ensure_ascii=True, default=str)
        sys.stdout.write(safe + "\n")
    except Exception:  # noqa: BLE001
        sys.stdout.write(json.dumps({"id": payload.get("id"), "ok": False, "data": {
            "status": "error",
            "operation": payload.get("op"),
            "error": {"kind": "internal", "message": "响应序列化失败"},
            "result": {},
        }}, ensure_ascii=True) + "\n")
    finally:
        try:
            sys.stdout.flush()
        except Exception:  # noqa: BLE001
            pass


def handle(request: dict[str, Any]) -> dict[str, Any]:
    req_id = request.get("id")
    op_name = str(request.get("op") or "").strip()
    args = request.get("args") or {}
    timeout = request.get("timeout")
    use_cache = bool(request.get("cache", _CACHEABLE))

    if op_name in ("@ping", "ping"):
        return {"id": req_id, "ok": True, "data": {"status": "pong", "version": VERSION, "python": sys.version.split()[0]}}

    if op_name in ("@ops", "ops", "@operations"):
        return {"id": req_id, "ok": True, "data": {"operations": engine.known_operations(), "count": len(engine.known_operations())}}

    if op_name in ("@cache_stats",):
        return {"id": req_id, "ok": True, "data": CACHE.stats()}

    if op_name in ("@cache_clear",):
        CACHE.clear()
        return {"id": req_id, "ok": True, "data": {"status": "cleared"}}

    if op_name in ("@shutdown",):
        return {"id": req_id, "ok": True, "data": {"status": "bye"}}

    if not op_name:
        return {
            "id": req_id,
            "ok": False,
            "data": R.MathResult.fail("", "请求缺少 op 字段", kind="protocol").to_dict(),
        }

    if use_cache:
        cached = CACHE.get(op_name, args)
        if cached is not None:
            return {"id": req_id, "ok": True, "data": cached}

    payload = engine.execute(op_name, args, timeout=timeout)

    if use_cache:
        CACHE.store(op_name, args, payload)

    return {"id": req_id, "ok": payload.get("status") == R.OK, "data": payload}


def main() -> int:
    log(f"math worker {VERSION} ready (python {sys.version.split()[0]}, sympy imported)")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req_id: Any = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("请求必须是 JSON 对象")
            req_id = request.get("id")
            response = handle(request)
        except json.JSONDecodeError as exc:
            response = {
                "id": None,
                "ok": False,
                "data": R.MathResult.fail("", f"JSON 解析失败: {exc}", kind="protocol").to_dict(),
            }
        except Exception as exc:  # noqa: BLE001 - worker 必须存活
            response = {
                "id": req_id,
                "ok": False,
                "data": R.MathResult.fail("", f"{type(exc).__name__}: {exc}", kind="internal").to_dict(),
            }
            log("未捕获异常:\n" + traceback.format_exc())
        _respond(response)
        if str(request.get("op") if isinstance(request, dict) else "") == "@shutdown":
            break
    log("math worker exiting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
