"""把 E:\\dsh-math\\plugin 打成可被 plugin_manager 安装的 tarball。

为什么要 tarball：profile 里用目录路径安装会得到 pnpm 的 `link:`（junction），
其真实路径在 profile 之外，DSH 宿主的 linked 解析层可能拒绝加载；tarball 安装会
在 `<profile>/node_modules/dsh-math` 解出**真实目录**，与已稳定工作的
dsh-context / dshmarket / modlens 完全同构。

用法（任意位置）：
    python scripts\\_pack.py                 # 在仓库根执行，生成 _dist/dsh-math-<version>.tgz
"""

from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # 仓库根（脚本位于 <repo>/scripts/）
PLUGIN = ROOT / "plugin"
DIST = ROOT / "_dist"

REQUIRED = ["package.json", "cordis.patch.yml", "icon.svg", "lib/entry.js", "engine/worker.py"]
INCLUDE_DIRS = ["lib", "engine", "locale"]


def main() -> int:
    manifest = json.loads((PLUGIN / "package.json").read_text(encoding="utf-8"))
    name, version = manifest["name"], manifest["version"]

    missing = [rel for rel in REQUIRED if not (PLUGIN / rel).exists()]
    if missing:
        print(f"缺少必需文件：{missing}", file=sys.stderr)
        return 1

    files: list[Path] = [PLUGIN / rel for rel in ["package.json", "cordis.patch.yml", "icon.svg"]]
    for directory in INCLUDE_DIRS:
        base = PLUGIN / directory
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                files.append(path)
    readme = PLUGIN / "README.md"
    if readme.exists():
        files.append(readme)

    DIST.mkdir(parents=True, exist_ok=True)
    target = DIST / f"{name}-{version}.tgz"
    with tarfile.open(target, "w:gz") as archive:
        for path in files:
            rel = path.relative_to(PLUGIN).as_posix()
            info = archive.gettarinfo(str(path), arcname=f"package/{rel}")
            info.uid = info.gid = 0
            info.uname = info.gname = "root"
            info.mtime = 1700000000  # 固定时间戳，保证可复现
            with open(path, "rb") as handle:
                archive.addfile(info, handle)

    size_kb = target.stat().st_size / 1024
    print(f"已生成 {target}（{len(files)} 个文件，{size_kb:.1f} KiB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
