/**
 * DSH 考研数学一智能计算与验证插件 —— Host 插件入口。
 *
 * 设计原则（用户规格）：
 *   「计算交给工具，思考交给模型，验证交给独立机制。」
 *
 * 本插件只做三件事：
 *   1. 把数学计算请求通过常驻 Python 引擎（SymPy/NumPy/SciPy/mpmath）执行
 *   2. 在工具描述里写清「何时该调用、何时不该调用」，避免重复推理与无意义自写程序
 *   3. 对重要结果提供独立验证，并如实报告验证状态
 */

import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { defineTool } from "./define_tool.js";
import { MathLedger, PythonBridge, resolvePython } from "./bridge.js";
import { renderResult } from "./render.js";
import { TOOL_SPECS } from "./tools.js";

const name = "dsh-math";
const inject = ["tools", "subprocess", "systemPrompt"];

/** 工具 section 排在 TOOL_WORKFLOW(2600) 与 TOOL_RALPH(2700) 之间。 */
const MATH_SECTION_ORDER = 2500;

/**
 * 配置归一化（对应 cordis.patch.yml 里那一行的 `config` 字段）。
 *
 * 本插件**不导出 Config schema**：DSH 的 Config 描述依赖 `@deepseek-ai/schemastery`，
 * 而 linked bundle 在本构建里解析不到宿主包（见 define_tool.js 顶部说明）。
 * 因此这里手工归一化并给默认值，配置项与用法不变：
 *   python / workerPath / timeoutMs / cache
 */
const DEFAULT_TIMEOUT_MS = 60000;

function normalizeConfig(raw) {
  const config = raw && typeof raw === "object" ? raw : {};
  return {
    python: typeof config.python === "string" ? config.python.trim() : "",
    workerPath: typeof config.workerPath === "string" ? config.workerPath.trim() : "",
    timeoutMs:
      Number.isFinite(config.timeoutMs) && config.timeoutMs > 0
        ? Number(config.timeoutMs)
        : DEFAULT_TIMEOUT_MS,
    cache: config.cache === undefined ? true : config.cache !== false,
  };
}

const pluginRoot = (() => {
  try {
    return dirname(dirname(fileURLToPath(import.meta.url)));
  } catch {
    return process.cwd();
  }
})();

function defaultWorkerPath() {
  return join(pluginRoot, "engine", "worker.py");
}

function defaultWorkingDirectory() {
  return homedir() || process.cwd();
}

/** 工具 section 的提示词：把调用纪律写进系统提示。 */
const SECTION_TEXT = `# 数学计算工具（dsh-math）

当前环境已装好一套由 SymPy 驱动的数学计算与验证工具：
math_simplify（化简/展开/因式分解/通分/部分分式）、math_solve（方程与方程组）、
math_limit（极限，含单侧）、math_derivative（导数/偏导/梯度/雅可比/海森/隐函数/参数方程）、
math_integrate（不定/定/反常/重积分）、math_series（级数展开与求和）、math_dsolve（微分方程）、
math_matrix（线代：行列式/逆/秩/特征值/对角化/二次型/线性方程组）、
math_probability（概率/分布/期望/方差/协方差/估计/假设检验）、
math_numeric（高精度数值与数值积分/求根/优化/数值比较）、
math_vector_calculus（曲线曲面积分/格林/高斯/斯托克斯）、
math_verify（结果验证）、math_analysis（函数分析：定义域/渐近线/多元极值/拉格朗日/不等式）、
math_status（引擎与缓存状态）。使用纪律：

1. **计算交给工具。** 需要精确的极限、导数、积分、级数、行列式、矩阵逆、特征值、解方程、
   概率与期望时，直接调用对应工具取得精确结果。这些计算手算或自行编写程序都更容易出错。
2. **不要为简单计算自写程序。** 像 ∫₀¹x²dx 这样的积分，直接调用 math_integrate 拿到精确结果，
   不要创建临时 Python 文件、执行、再读输出去绕一圈。
3. **已获得可靠结果就停止。** 同一个表达式不要用等价方法反复计算；结果已经确定且验证通过时，
   不要再尝试其他解法——是否比较多解法由用户的任务要求决定，而不是由工具的可用性决定。
4. **验证交给独立机制。** 重要结果在给出最终答案前，可用 math_verify 复核（它用与计算不同的
   途径检查：求导回去、代回方程、矩阵乘积、归一化等）。工具会如实区分「符号级证明」「仅有数值
   证据」和「未能验证」，不要把数值抽样一致当作严格证明，也不要把未求值表达式当作已算出的结果。
5. **不要跳过数学推理与讲解。** 工具负责算得快而准，不负责替代思路分析。定义域、成立条件、
   参数取值限制、解法选择理由、几何与积分区域判断仍然必须由你分析并讲清。
6. **如实报告。** 工具返回 status 为 unsolved 表示当前方法求不出解析解，error 表示执行失败；
   这两种情况都要向用户说明具体原因和已尝试的方法，不要编造结果，也不要反复用同样参数重试。
7. 追问某一步时只回答该步即可，不需要重讲整题；用户修改条件后要重新检查受影响的结果。`;

function buildTool(spec, manager) {
  // 每个工具都额外提供 retry：默认情况下「完全相同的参数失败过」会被账本拦下，
  // 只有明确声明换了方法（改参数）或传 retry=true 才允许再试一次。
  const parameters = {
    ...(spec.params ?? {}),
    retry: {
      type: "boolean",
      description:
        "是否允许对完全相同的参数再尝试一次（默认不允许）。只有在确有理由（例如引擎刚才崩过、外部条件变了）时才用；改参数或换方法不需要这个开关。",
    },
  };
  return defineTool({
    name: spec.name,
    description: spec.description,
    parameters,
    output: {
      // 注意：DSH 的 value schema 只接受 string/number/integer/boolean/null/array/object/json
      // 或 oneOf —— 写成 { type: "text" } 会在 defineTool 里直接抛
      // `schema.type must be string/number/...`（渲染块用 type:"text" 是另一回事：
      // 那是 render() 返回的 content block，不是 schema）。
      schema: { type: "string" },
      render: (_args, value) => [{ type: "text", text: String(value ?? "") }],
    },
    async execute(args, exec) {
      return manager.execute(spec, args, exec);
    },
    presentCall: (args) => {
      const summary = Object.entries(args ?? {})
        .filter(([, v]) => v !== undefined && v !== null && v !== "")
        .slice(0, 3)
        .map(([k, v]) => `${k}=${Array.isArray(v) ? v.join(",") : String(v)}`)
        .join("  ");
      return {
        card: "generic",
        title: spec.title,
        description: summary || spec.short,
        kind: "math",
        rawInput: args,
        content: [{ type: "text", text: summary || spec.short }],
      };
    },
  });
}

/** 引擎管理器：持有 Python 桥与调用账本。 */
class MathEngine {
  constructor(ctx, config) {
    this.ctx = ctx;
    this.config = config ?? {};
    this.ledger = new MathLedger();
    this.bridge = null;
    this.starting = null;
    this.pythonInfo = null;
    this.logs = [];
    this.tools = new Map();
  }

  log(message) {
    this.logs.push({ at: Date.now(), message: String(message) });
    if (this.logs.length > 50) this.logs.shift();
    try {
      this.ctx.logger?.debug?.(`[dsh-math] ${message}`);
    } catch {
      // 日志失败不影响计算
    }
  }

  async start(signal) {
    if (this.bridge) return this.bridge;
    if (this.starting) return this.starting;
    this.starting = (async () => {
      const subprocess = this.ctx.get("subprocess");
      if (!subprocess) {
        throw new Error("当前 profile 没有 subprocess 服务，数学引擎无法启动。");
      }
      const configured = this.config.python?.trim();
      let python = configured;
      let source = "插件配置";
      if (!configured) {
        const resolved = await resolvePython(subprocess, { signal });
        python = resolved.command;
        source = resolved.source;
      }
      const workerPath = this.config.workerPath?.trim() || defaultWorkerPath();
      this.pythonInfo = { command: python, source, workerPath };
      this.bridge = new PythonBridge({
        subprocess,
        pythonCommand: python,
        workerPath,
        cwd: defaultWorkingDirectory(),
        ledger: this.ledger,
        onLog: (message) => this.log(message),
      });
      await this.bridge.ensureStarted(signal);
      return this.bridge;
    })();
    try {
      return await this.starting;
    } finally {
      this.starting = null;
    }
  }

  async execute(spec, args, exec) {
    const timeoutMs = Number.isFinite(this.config.timeoutMs) && this.config.timeoutMs > 0
      ? this.config.timeoutMs
      : undefined;

    if (spec.name === "math_status") {
      return this.status(spec, args);
    }

    let mapped;
    try {
      mapped = spec.args ? spec.args(args ?? {}) : { op: spec.op, args: args ?? {} };
    } catch (error) {
      return renderResult(
        {
          status: "error",
          operation: spec.op,
          result: {},
          conditions: [],
          verification: { status: "unverified", methods: [] },
          warnings: [],
          error: { kind: "invalid_input", message: `参数组装失败：${error?.message ?? error}` },
        },
        { op: spec.op, args, spec },
      );
    }

    const cacheEnabled = this.config.cache !== false;
    let payload;
    try {
      const bridge = await this.start(exec?.signal);
      payload = await bridge.call(mapped.op, mapped.args, {
        timeoutMs,
        cache: cacheEnabled,
        allowRetry: Boolean(args?.retry),
        signal: exec?.signal,
      });
    } catch (error) {
      payload = {
        success: false,
        status: "error",
        operation: mapped.op,
        method: mapped.op,
        input: mapped.args,
        result: {},
        conditions: [],
        verification: { status: "unverified", methods: [] },
        warnings: [],
        error: { kind: "engine_unavailable", message: String(error?.message ?? error) },
      };
    }
    return renderResult(payload, { op: mapped.op, args: mapped.args, spec });
  }

  async status(spec, args) {
    const action = String(args?.action ?? "status").toLowerCase();
    if (!this.bridge) {
      return [
        `${spec.title} — 引擎尚未启动（首次调用任意数学工具时会自动拉起 Python 引擎）`,
        `Python 解释器: ${this.pythonInfo?.command ?? "未探测"}`,
        `worker: ${this.pythonInfo?.workerPath ?? defaultWorkerPath()}`,
      ].join("\n");
    }
    const lines = [];
    try {
      if (action === "ops") {
        const response = await this.bridge.control("@ops");
        // bridge 的 #onLine 用 `message.data ?? message` 解析响应，所以这里拿到的
        // 已经是 data 本体；下面保留 data 兜底以兼容协议变化。
        const ops = response?.operations ?? response?.data?.operations ?? [];
        lines.push(`${spec.title} — 可用算子 ${ops.length} 个`);
        lines.push(ops.join(", "));
        return lines.join("\n");
      }
      if (action === "cache_clear") {
        await this.bridge.control("@cache_clear");
        this.ledger.clear();
        return `${spec.title} — 缓存已清空（Python 侧与 Node 侧）`;
      }
      const stats = this.ledger.snapshot();
      const response = await this.bridge.control("@cache_stats");
      lines.push(`${spec.title}`);
      lines.push(`Python 解释器: ${this.pythonInfo?.command}（${this.pythonInfo?.source}）`);
      lines.push(`worker: ${this.pythonInfo?.workerPath}`);
      lines.push(
        `Node 侧: 命中 ${stats.hits} / 未命中 ${stats.misses}（命中率 ${(stats.hitRate * 100).toFixed(1)}%），` +
          `已拦截重复失败重试 ${stats.blockedRetries} 次`,
      );
      lines.push(
        `Python 侧: 成功缓存 ${response?.success_entries ?? response?.data?.success_entries ?? "?"} 条，` +
          `失败记录 ${response?.failure_entries ?? response?.data?.failure_entries ?? "?"} 条，` +
          `命中 ${response?.hits ?? response?.data?.hits ?? "?"} 次`,
      );
      if (this.logs.length) {
        lines.push("最近日志:");
        for (const entry of this.logs.slice(-3)) lines.push(`  · ${entry.message}`);
      }
      return lines.join("\n");
    } catch (error) {
      return `${spec.title} — 状态查询失败：${error?.message ?? error}`;
    }
  }

  async dispose() {
    const bridge = this.bridge;
    this.bridge = null;
    if (bridge) await bridge.dispose();
  }
}

function apply(ctx, config) {
  const engine = new MathEngine(ctx, normalizeConfig(config));

  ctx.effect(() => () => {
    // 服务卸载时结束 Python 进程，避免留下孤儿进程
    void engine.dispose();
  });

  for (const spec of TOOL_SPECS) {
    const tool = buildTool(spec, engine);
    engine.tools.set(spec.name, tool);
    const unregister = ctx.tools.register(tool);
    ctx.effect(() => () => unregister());
  }

  ctx.systemPrompt.section({
    name: "dsh-math:tools",
    order: MATH_SECTION_ORDER,
    text: SECTION_TEXT,
  });

  engine.log(`已注册 ${TOOL_SPECS.length} 个数学工具；Python 引擎将在首次调用时启动`);
}

export { apply, inject, name };
