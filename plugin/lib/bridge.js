/**
 * 常驻 Python 计算引擎的 Node 侧桥。
 *
 * 职责：
 *  - 定位可用的 Python 解释器（环境变量 → PATH → 已知安装位置）
 *  - 通过 ctx.subprocess 启动 `worker.py`，以 newline-delimited JSON 通信
 *  - 请求/响应配对、超时、进程崩溃后的自动重建（限量，避免无意义重试）
 *  - 会话级缓存与「同一计算不重复执行」账本（用户规格 6.2 / 6.3 / 10）
 *
 * 设计约束：stdout 只承载协议；worker 的日志全部走 stderr，读取 stderr 仅为诊断。
 */

import { existsSync } from "node:fs";
import { join } from "node:path";
import { createInterface } from "node:readline";

/** worker 启动就绪等待上限（首次启动要 import sympy，需要一点时间）。 */
const READY_TIMEOUT_MS = 30_000;
/** 单次请求默认超时；超过则该算子返回 timeout 错误而不是拖死对话。 */
const DEFAULT_REQUEST_TIMEOUT_MS = 60_000;
/** stderr 只用于诊断，保留尾部若干字节即可。 */
const STDERR_KEEP_BYTES = 64 * 1024;
/** 允许的自动重建次数：防止「引擎反复崩溃 → 反复重启」的无意义循环。 */
const MAX_AUTO_RESTARTS = 2;

/** 已知的 Python 安装位置（最后兜底，避免 PATH 干净时找不到解释器）。 */
const WELL_KNOWN_PYTHON = [
  "E:\\anaconda3\\python.exe",
  "C:\\ProgramData\\Anaconda3\\python.exe",
  "C:\\ProgramData\\miniconda3\\python.exe",
  "C:\\Python313\\python.exe",
  "C:\\Python312\\python.exe",
  "/usr/bin/python3",
  "/usr/local/bin/python3",
];

const PYTHON_ENV_KEYS = ["DSH_MATH_PYTHON", "DSH_PYTHON", "PYTHON", "PYTHON_EXE"];

/**
 * 解析 Python 解释器路径。返回 `{ command, source }`；全部失败时抛错。
 */
export async function resolvePython(subprocess, { env = process.env, signal } = {}) {
  for (const key of PYTHON_ENV_KEYS) {
    const candidate = env?.[key]?.trim();
    if (candidate && existsSync(candidate)) {
      return { command: candidate, source: `环境变量 ${key}` };
    }
  }
  for (const bare of ["python", "python3", "py"]) {
    try {
      const resolved = await subprocess.resolveExecutable(bare, undefined, signal);
      if (resolved) return { command: resolved, source: `PATH 上的 ${bare}` };
    } catch {
      // 继续尝试下一个候选名
    }
  }
  for (const candidate of WELL_KNOWN_PYTHON) {
    if (existsSync(candidate)) {
      return { command: candidate, source: "已知安装位置" };
    }
  }
  throw new Error(
    "找不到可用的 Python 解释器。请安装 Python 3.10+ 并安装 sympy，或用环境变量 DSH_MATH_PYTHON 指定 python.exe 的绝对路径。",
  );
}

/** 会话级缓存 + 调用账本（模型看不到内部日志，只影响返回结果）。 */
export class MathLedger {
  constructor(limit = 256) {
    this.limit = limit;
    this.success = new Map();
    this.failures = new Map();
    this.inflight = new Map();
    this.stats = { hits: 0, misses: 0, calls: 0, blockedRetries: 0 };
  }

  static key(op, args) {
    let payload;
    try {
      payload = JSON.stringify(args, Object.keys(args ?? {}).sort());
    } catch {
      payload = String(args);
    }
    return `${op}::${payload}`;
  }

  get(op, args) {
    const key = MathLedger.key(op, args);
    if (this.success.has(key)) {
      this.stats.hits += 1;
      return { ...this.success.get(key), cached: true, cacheSource: "node" };
    }
    this.stats.misses += 1;
    return undefined;
  }

  store(op, args, value) {
    const key = MathLedger.key(op, args);
    if (value?.status === "ok") {
      this.success.set(key, value);
      while (this.success.size > this.limit) {
        this.success.delete(this.success.keys().next().value);
      }
    } else {
      this.failures.set(key, {
        status: value?.status,
        kind: value?.error?.kind,
        message: value?.error?.message,
        at: Date.now(),
      });
    }
  }

  /** 同参数失败过就拒绝再次尝试——除非调用方明确声明换了方法或参数。 */
  repeatedFailure(op, args, allowRetry) {
    if (allowRetry) return undefined;
    return this.failures.get(MathLedger.key(op, args));
  }

  snapshot() {
    const total = this.stats.hits + this.stats.misses;
    return {
      ...this.stats,
      successEntries: this.success.size,
      failureEntries: this.failures.size,
      hitRate: total ? Number((this.stats.hits / total).toFixed(4)) : 0,
    };
  }

  clear() {
    this.success.clear();
    this.failures.clear();
    this.inflight.clear();
  }
}

/** 与 worker.py 保持一致的 JSON 行协议客户端。 */
export class PythonBridge {
  constructor({ subprocess, pythonCommand, workerPath, cwd, ledger, onLog }) {
    this.subprocess = subprocess;
    this.pythonCommand = pythonCommand;
    this.workerPath = workerPath;
    this.cwd = cwd;
    this.ledger = ledger ?? new MathLedger();
    this.onLog = onLog ?? (() => {});
    this.handle = null;
    this.pending = new Map();
    this.nextId = 1;
    this.ready = null;
    this.restarts = 0;
    this.stderrTail = "";
    this.disposed = false;
  }

  get running() {
    return Boolean(this.handle);
  }

  /** 惰性启动：第一次调用工具时才拉起进程。 */
  async ensureStarted(signal) {
    if (this.disposed) throw new Error("数学引擎已关闭");
    if (this.handle) return this.handle;
    if (this.ready) return this.ready;
    this.ready = this.#start(signal).finally(() => {
      this.ready = null;
    });
    return this.ready;
  }

  async #start(signal) {
    const handle = this.subprocess.spawn({
      argv: [this.pythonCommand, this.workerPath],
      cwd: this.cwd,
      stdio: {
        stdin: "pipe",
        stdout: "pipe",
        stderr: { maxBytes: STDERR_KEEP_BYTES },
      },
      graceMs: 2_000,
      signal,
    });
    this.handle = handle;

    const stdout = handle.stdout;
    const stdin = handle.stdin;
    if (!stdout || !stdin) {
      throw new Error("Python 引擎管道创建失败（stdout/stdin 不可用）");
    }

    const reader = createInterface({ input: stdout, crlfDelay: Infinity });
    reader.on("line", (line) => this.#onLine(line));

    handle.done
      .then(() => this.#onExit("正常退出"))
      .catch((error) => this.#onExit(`异常：${error?.message ?? error}`));

    // 首次 @ping 同时确认协议可用并拿到解释器版本
    const pong = await this.#send({ op: "@ping" }, READY_TIMEOUT_MS);
    this.restarts = 0;
    this.onLog(
      `Python 引擎就绪：${pong?.data?.python ?? "?"} · ${this.pythonCommand}`,
    );
    return handle;
  }

  #onLine(line) {
    const text = String(line).trim();
    if (!text) return;
    let message;
    try {
      message = JSON.parse(text);
    } catch {
      this.onLog(`收到无法解析的输出：${text.slice(0, 200)}`);
      return;
    }
    const id = message?.id;
    const entry = this.pending.get(id);
    if (!entry) return;
    this.pending.delete(id);
    clearTimeout(entry.timer);
    entry.resolve(message.data ?? message);
  }

  #onExit(reason) {
    const wasRunning = Boolean(this.handle);
    const handle = this.handle;
    this.handle = null;
    if (handle) {
      // stderr 用的是 collect 模式（stdio.stderr = {maxBytes}），所以流在
      // handle.collected.stderr 上，而不是 handle.stderr；读取只为诊断，
      // 任何失败都不影响主流程。
      try {
        const dump = handle.collected?.stderr?.readFrom?.(0);
        if (dump?.text) this.stderrTail = String(dump.text).slice(-STDERR_KEEP_BYTES);
      } catch {
        // 忽略诊断读取失败
      }
    }
    for (const [id, entry] of this.pending) {
      clearTimeout(entry.timer);
      this.pending.delete(id);
      entry.reject(new Error(`Python 引擎已退出（${reason}）`));
    }
    if (wasRunning) this.onLog(`Python 引擎退出：${reason}`);
    if (wasRunning && this.stderrTail) {
      const lastLine = this.stderrTail.trim().split(/\r?\n/).filter(Boolean).slice(-1)[0] ?? "";
      if (lastLine) this.onLog(`引擎 stderr 末行：${lastLine.slice(0, 300)}`);
    }
  }

  #send(request, timeoutMs) {
    const id = this.nextId++;
    const stdin = this.handle?.stdin;
    if (!stdin) throw new Error("Python 引擎未运行");
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`超过 ${timeoutMs}ms 未收到引擎响应`));
      }, timeoutMs);
      this.pending.set(id, { resolve, reject, timer });
      stdin.write(`${JSON.stringify({ id, ...request })}\n`);
    });
  }

  /**
   * 执行一个算子。caching 开启时命中 Node 侧缓存直接返回，不再打扰 Python 进程。
   */
  async call(op, args, { timeoutMs = DEFAULT_REQUEST_TIMEOUT_MS, cache = true, allowRetry = false, signal } = {}) {
    if (cache) {
      const hit = this.ledger.get(op, args);
      if (hit) return hit;
      const prior = this.ledger.repeatedFailure(op, args, allowRetry);
      if (prior) {
        this.ledger.stats.blockedRetries += 1;
        return {
          success: false,
          status: "error",
          operation: op,
          method: op,
          input: args,
          result: {},
          conditions: [],
          verification: { status: "unverified", methods: [] },
          warnings: [
            "这一组完全相同的参数此前已经失败过，按防止重复计算的原则不再重复尝试。若确实需要再试，请改变方法或参数。",
          ],
          error: {
            kind: "repeated_failure",
            message: `此前失败原因：${prior.kind ?? "unknown"} · ${prior.message ?? "未记录"}`,
          },
        };
      }
    }

    const key = MathLedger.key(op, args);
    const inflight = this.ledger.inflight.get(key);
    if (inflight) return inflight;

    const task = (async () => {
      for (let attempt = 0; attempt <= MAX_AUTO_RESTARTS; attempt += 1) {
        try {
          await this.ensureStarted(signal);
          const response = await this.#send({ op, args, timeout: timeoutMs / 1000, cache }, timeoutMs);
          return response;
        } catch (error) {
          const crashed = !this.handle;
          if (!crashed || attempt === MAX_AUTO_RESTARTS) {
            return {
              success: false,
              status: "error",
              operation: op,
              method: op,
              input: args,
              result: {},
              conditions: [],
              verification: { status: "unverified", methods: [] },
              warnings: [],
              error: { kind: "engine_unavailable", message: String(error?.message ?? error) },
            };
          }
          this.onLog(`引擎不可用，重建后重试（第 ${attempt + 1} 次）：${error?.message ?? error}`);
        }
      }
      return {
        success: false,
        status: "error",
        operation: op,
        result: {},
        conditions: [],
        verification: { status: "unverified", methods: [] },
        warnings: [],
        error: { kind: "engine_unavailable", message: "引擎多次重建失败" },
      };
    })();

    this.ledger.inflight.set(key, task);
    try {
      const value = await task;
      this.ledger.stats.calls += 1;
      if (cache) this.ledger.store(op, args, value);
      return value;
    } finally {
      this.ledger.inflight.delete(key);
    }
  }

  /** 控制类请求：不做缓存。 */
  async control(op, { timeoutMs = 15_000 } = {}) {
    await this.ensureStarted();
    return this.#send({ op }, timeoutMs);
  }

  async dispose() {
    this.disposed = true;
    const handle = this.handle;
    this.handle = null;
    if (!handle) return;
    try {
      if (handle.stdin) {
        handle.stdin.write(`${JSON.stringify({ id: this.nextId++, op: "@shutdown" })}\n`);
        handle.stdin.end?.();
      }
    } catch {
      // 管道可能已经断了，直接走 terminate
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
    try {
      handle.terminate();
    } catch {
      // 已经退出
    }
    try {
      await handle.waitForExit?.();
    } catch {
      // 忽略退出等待失败
    }
  }
}

export { DEFAULT_REQUEST_TIMEOUT_MS, MAX_AUTO_RESTARTS, WELL_KNOWN_PYTHON };
