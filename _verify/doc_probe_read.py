# -*- coding: utf-8 -*-
"""把 probe 输出压成可读摘要（字段裁剪 + 截断），供撰写文档时引用真实数据。"""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else r"E:\dsh-math\_verify\doc_probe_out.json"
only = sys.argv[2:] if len(sys.argv) > 2 else None
lim = 1200

data = json.load(open(path, encoding="utf-8"))
for label, item in data.items():
    if only and label not in only:
        continue
    r = item["resp"]
    print(f"### {label}  op={item['op']} status={r.get('status')} method={r.get('method')}")
    print(f"    args={json.dumps(item['args'], ensure_ascii=False)}")
    for key in ("result", "conditions", "verification", "warnings", "error", "extra"):
        if key in r and r[key] not in (None, [], {}):
            txt = json.dumps(r[key], ensure_ascii=False)
            if len(txt) > lim:
                txt = txt[:lim] + f"…[截断 共{len(txt)}字符]"
            print(f"    {key}={txt}")
    print()
