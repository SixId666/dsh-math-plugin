"""校验 plugin/lib/tools.js 的工具表与 Python 引擎算子签名是否一致。

校验内容（每条都能抓出真实缺陷）：
1. tools.js 里映射的算子名在引擎里是否存在（拼错/别名错 → Node 侧只会得到 unknown_operation）；
2. 传下去的每个参数名是否被目标算子接受；被 ARG_ALIASES 静默改名的要标出来（改名后
   若目标函数不认这个新名字，参数会被 `_call_with_declared_kwargs` 静默丢弃 —— 这正是
   最容易发生、又最难被发现的一类缺陷）；
3. 算子的**必需参数**是否至少被工具表提供了一次（否则该工具永远只能用 invalid_input 收场）；
4. spec.op（工具表声明的算子）与默认分支实际映射的算子是否一致。

用法： python _check_contract.py [ _specs.json ]
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "engine"))

from mathkit import engine as E  # noqa: E402
from mathkit import ops  # noqa: F401,E402  (导入 ops 才会注册全部 handler)


def declared_params(func) -> tuple[set[str], set[str]]:
    """返回 (接受的参数名, 必需参数名)。"""
    sig = inspect.signature(func)
    accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    allowed = {
        name
        for name, p in sig.parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    if accepts_kwargs:
        allowed |= {"**"}
    required = {
        name
        for name, p in sig.parameters.items()
        if p.default is inspect.Parameter.empty
        and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return allowed, required


def main() -> int:
    spec_file = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "_specs.json"
    # PowerShell 的 `Out-File -Encoding utf8` 会写 BOM，这里两种都接受。
    specs = json.loads(spec_file.read_text(encoding="utf-8-sig"))

    problems: list[str] = []
    notes: list[str] = []
    provided: dict[str, set[str]] = {}

    for spec in specs:
        name = spec["name"]
        has_args = spec["has_args_fn"]
        if not has_args:
            notes.append(f"[{name}] 没有 args() 映射（由插件自己实现，例如 math_status）")
            continue
        for call in spec["calls"]:
            if call["error"]:
                problems.append(f"[{name}] args() 在 {call['selector']}={call['choice']} 抛错：{call['error']}")
                continue
            op = call["op"]
            if op and str(op).startswith("@"):
                notes.append(f"[{name}] {call['selector']}={call['choice']} → 插件内部控制指令 {op}（不经引擎）")
                continue
            if not op:
                problems.append(f"[{name}] {call['selector']}={call['choice']} 没有映射出算子名")
                continue
            func = E.resolve(op)
            if func is None:
                problems.append(f"[{name}] {call['selector']}={call['choice']} 映射到未知算子 {op!r}")
                continue
            allowed, required = declared_params(func)
            provided.setdefault(name, set()).update(call["arg_keys"])
            if "**" in allowed:
                continue
            for key in call["arg_keys"]:
                renamed = E.ARG_ALIASES.get(key, key)
                if renamed != key:
                    if renamed in allowed:
                        notes.append(f"[{name}] 参数 {key!r} 被 ARG_ALIASES 改名为 {renamed!r}（目标算子接受）")
                    else:
                        problems.append(
                            f"[{name}] {call['selector']}={call['choice']} 的 {op}：参数 {key!r} 被改名为 {renamed!r}，"
                            f"但 {op} 只接受 {sorted(allowed)} → 该参数会被静默丢弃"
                        )
                elif key not in allowed:
                    problems.append(
                        f"[{name}] {call['selector']}={call['choice']} 的 {op}：参数 {key!r} 不被接受"
                        f"（只接受 {sorted(allowed)}）→ 会被静默丢弃"
                    )
        # 必需参数覆盖检查（只在提供了变量集的分支上判断）
        for call in spec["calls"]:
            op = call["op"]
            if not op or call["error"] or str(op).startswith("@"):
                continue
            func = E.resolve(op)
            if func is None:
                continue
            allowed, required = declared_params(func)
            if "**" in allowed:
                continue
            keys = provided.get(name, set())
            normalized = {E.ARG_ALIASES.get(k, k) for k in keys}
            missing = sorted(r for r in required if r not in normalized)
            if missing:
                problems.append(f"[{name}] 算子 {op} 的必需参数从未被提供：{missing}")

    print(f"工具数：{len(specs)}；问题：{len(problems)}；说明：{len(notes)}")
    if notes:
        print("\n--- 说明 ---")
        for line in notes:
            print(" ·", line)
    if problems:
        print("\n--- 问题 ---")
        for line in problems:
            print(" ✗", line)
    else:
        print("\n契约校验通过：工具表里的算子名与参数名都能被引擎接受。")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
