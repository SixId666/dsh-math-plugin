/**
 * 共享的「无宿主」插件启动器。
 *
 * 用真实的 Python 引擎跑通「插件 → 工具 → 参数组装 → 引擎 → 渲染」全链路，
 * 不需要 DSH 宿主进程。harness.mjs 与 special_tests.mjs 都用它。
 */

import { spawn } from "node:child_process";
import { pathToFileURL } from "node:url";

const ENTRY = pathToFileURL("E:/dsh-math/plugin/lib/entry.js").href;

function createSubprocessShim() {
  return {
    async resolveExecutable(command) {
      return process.env.DSH_MATH_PYTHON || command;
    },
    spawn(spec) {
      const [program, ...rest] = spec.argv;
      const child = spawn(program, rest, {
        cwd: spec.cwd,
        stdio: ["pipe", "pipe", "pipe"],
        windowsHide: true,
      });
      const stderrChunks = [];
      child.stderr?.on("data", (chunk) => {
        stderrChunks.push(chunk);
        if (stderrChunks.length > 200) stderrChunks.shift();
      });
      let exited = false;
      const exitWaiters = [];
      child.on("exit", () => {
        exited = true;
        for (const waiter of exitWaiters) waiter();
      });
      return {
        stdin: child.stdin,
        stdout: child.stdout,
        stderr: undefined, // collect 模式：stderr 在 collected 上（与 DSH 行为一致）
        control: {},
        collected: {
          stderr: {
            readFrom: () => ({
              text: Buffer.concat(stderrChunks).toString("utf8"),
              nextOffset: 0,
              lossy: false,
            }),
            snapshot: () => ({ bytes: 0, totalBytes: 0 }),
          },
        },
        done: new Promise((resolve, reject) => {
          child.on("exit", (code) => (code === 0 ? resolve({ code }) : reject(new Error(`exit ${code}`))));
          child.on("error", reject);
        }),
        terminate: () => child.kill(),
        terminateForHostExit: () => child.kill(),
        waitForExit: () => (exited ? Promise.resolve() : new Promise((resolve) => exitWaiters.push(resolve))),
      };
    },
  };
}

/** 启动插件：返回可调用工具与清理函数。 */
export async function boot() {
  const plugin = await import(ENTRY);
  const disposers = [];
  const registered = new Map();
  const sections = [];
  const shim = createSubprocessShim();
  const ctx = {
    get: (name) => (name === "subprocess" ? shim : undefined),
    effect: (fn) => {
      disposers.push(fn());
    },
    logger: { debug: () => {}, info: () => {}, warn: () => {} },
    tools: {
      register: (definition) => {
        registered.set(definition.name, definition);
        return () => registered.delete(definition.name);
      },
    },
    systemPrompt: {
      section: (section) => {
        sections.push(section);
      },
    },
  };
  plugin.apply(ctx, {});

  const call = async (name, args) => {
    const definition = registered.get(name);
    if (!definition) throw new Error(`工具未注册：${name}`);
    const value = await definition.execute(args, { signal: undefined });
    return definition.output
      .render(args, value)
      .map((block) => block.text ?? "")
      .join("\n");
  };

  const shutdown = async () => {
    for (const dispose of disposers.reverse()) {
      try {
        if (typeof dispose === "function") await dispose();
      } catch {
        // 忽略清理失败
      }
    }
  };

  return { plugin, call, shutdown, registered, sections };
}

/** 从 math_status 的 cache_stats 文本里取出 Node 侧计数。引擎未启动时视为 0。 */
export function readStats(text) {
  const pick = (pattern) => {
    const matched = pattern.exec(text);
    return matched ? Number(matched[1]) : 0;
  };
  return {
    hits: pick(/命中 (\d+) \//),
    misses: pick(/未命中 (\d+)/),
    blocked: pick(/已拦截重复失败重试 (\d+) 次/),
  };
}

/** 取渲染文本里的「结果:」行（用于比较两次调用是否给出同一个结果，忽略缓存提示行）。 */
export function resultOf(text) {
  const matched = /^结果: (.*)$/m.exec(String(text));
  return matched ? matched[1].trim() : "";
}

export function report(label, ok, detail = "") {
  console.log(`${ok ? "[PASS]" : "[FAIL]"} ${label}${detail ? ` —— ${detail}` : ""}`);
  return { label, ok, detail };
}
