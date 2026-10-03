# 安全政策（Security Policy）

## 支持的版本

| 版本 | 是否支持 |
| --- | --- |
| 1.0.2 | 是 |
| < 1.0.2 | 否 |

## 报告漏洞

**请勿通过公开 Issue 报告安全漏洞。**

请使用以下任一渠道：

1. GitHub 私有漏洞报告：<https://github.com/SixId666/dsh-math-plugin/security/advisories/new>
2. 邮箱：1027573327@qq.com（请在标题注明 `[security] dsh-math`）

我们会在 7 天内确认收到，并在修复后同步披露信息。

## 范围说明

本插件为**本地运行**的 DSH bundle，不提供网络服务、不采集数据。数学引擎通过
JSON lines over stdio 与一个常驻 Python 子进程通信。特别欢迎以下方向的报告：

- `plugin/lib/bridge.js` 的 Python 解释器解析（`resolvePython`）与子进程启动参数是否可被环境变量/配置操纵；
- 引擎 `ast.py` 表达式解析层对恶意构造输入的处理（拒绝服务、无限求值、内存耗尽等）；
- `workerPath` / `python` 配置项被非预期覆盖导致的任意代码执行路径；
- 缓存与失败记账（`engine/worker.py`）中键规范化不足导致的语义混用。

以下内容**不在**漏洞范围内：数学结果错误（请提普通 Issue）、依赖宿主 DSH
自身缺陷的问题、以及需要本机已有恶意代码才能成立的攻击。
