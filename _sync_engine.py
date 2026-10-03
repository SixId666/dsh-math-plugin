"""把开发树 engine/ 同步到可安装包 plugin/engine/。

单一事实来源是 `E:\\dsh-math\\engine\\`（开发与调试都在这里改），
`plugin/` 必须是自包含的 bundle：安装到 DSH profile 后不会再回来读开发树。

同步内容（白名单，其它一律删除 —— 之前的镜像是手工拷贝的，混进了
_probe_*.py / _tmp_*.py / *.log / __pycache__ 和过期版本）：
  worker.py
  mathkit/*.py
  _selftest_*.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "engine"
DST = ROOT / "plugin" / "engine"


def main() -> int:
    if not SRC.is_dir():
        print(f"找不到开发树：{SRC}")
        return 2
    if DST.exists():
        shutil.rmtree(DST)
    (DST / "mathkit").mkdir(parents=True)

    copied: list[str] = []
    shutil.copy2(SRC / "worker.py", DST / "worker.py")
    copied.append("worker.py")
    for path in sorted((SRC / "mathkit").glob("*.py")):
        shutil.copy2(path, DST / "mathkit" / path.name)
        copied.append(f"mathkit/{path.name}")
    for path in sorted(SRC.glob("_selftest_*.py")):
        shutil.copy2(path, DST / path.name)
        copied.append(path.name)

    total = sum(p.stat().st_size for p in DST.rglob("*") if p.is_file())
    print(f"已同步 {len(copied)} 个文件 → {DST}（共 {total / 1024:.0f} KiB）")
    for name in copied:
        print("  ·", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
