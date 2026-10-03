/**
 * 把 plugin/lib/tools.js 的工具表导出成 JSON，供 Python 侧做「工具→算子」契约校验。
 * 用法（Node 不在 PATH，用 Electron 当 Node）：
 *   $env:ELECTRON_RUN_AS_NODE=1; & "D:\deepseek_harness\DeepSeek Harness.exe" scripts/_dump_specs.mjs > scripts/_specs.json
 *
 * 分支选择参数（mode / operation / kind / action）会按其描述里的 "a | b | c" 逐个取值展开，
 * 这样每个工具的所有分支都会被 Python 侧校验到。
 */
import { TOOL_SPECS } from "../plugin/lib/tools.js";

const SELECTOR_KEYS = ["mode", "operation", "kind", "action"];

/**
 * 「JSON 对象字符串」型参数的哑元。
 *
 * 这类参数（math_probability 的 `params`、math_numeric 的 `values`）在 args() 里会被
 * parseParamObject / JSON.parse 解析，通用哑元 "x" 不是合法 JSON，会让契约校验误报
 * 「args() 抛错：params 不是合法 JSON：x」——那是哑元的问题，不是工具的问题。
 */
const JSON_DUMMIES = {
  params: '{"mu":0,"sigma":1}',
  values: '{"x":1.5}',
};

const dummy = (param, key = "") => {
  // oneOf（例如 AS：矩阵既接受 ['1,2','3,4'] 也接受 '1,2;3,4'）取第一个非 string 分支的形状
  const shape =
    (Array.isArray(param?.oneOf) ? param.oneOf.find((b) => b && b.type && b.type !== "string") : null) ??
    param ??
    {};
  if (shape.type === "number") return 1;
  if (shape.type === "boolean") return true;
  if (shape.type === "array") return ["x"];
  if (JSON_DUMMIES[key]) return JSON_DUMMIES[key];
  if (typeof param?.description === "string" && /JSON/i.test(param.description)) return '{"x":1.5}';
  return "x";
};

/** 从参数描述里抽出可选分支，例如 "a（默认）| b | c。" → ["a","b","c"] */
function branchesOf(param) {
  if (typeof param?.description !== "string") return [];
  return param.description
    .split("|")
    .map((part) => part.trim().split(/[（(\s]/)[0].trim().replace(/[。．.,，;；:：]+$/u, ""))
    .filter((token) => /^[A-Za-z_][A-Za-z0-9_]*$/.test(token));
}

const report = [];
for (const spec of TOOL_SPECS) {
  const params = spec.params ?? {};
  const baseArgs = {};
  for (const [key, param] of Object.entries(params)) baseArgs[key] = dummy(param, key);

  const selector = SELECTOR_KEYS.find((key) => key in params && branchesOf(params[key]).length > 1);
  const choices = selector ? branchesOf(params[selector]) : [undefined];

  const calls = [];
  for (const choice of choices) {
    const args = { ...baseArgs };
    if (selector && choice !== undefined) args[selector] = choice;
    let mapped = null;
    let error = null;
    try {
      mapped = spec.args ? spec.args(args) : null;
    } catch (exc) {
      error = String((exc && exc.message) || exc);
    }
    calls.push({
      selector: selector ?? null,
      choice: choice ?? null,
      op: mapped?.op ?? spec.op ?? null,
      arg_keys: mapped?.args ? Object.keys(mapped.args) : [],
      error,
    });
  }
  report.push({
    name: spec.name,
    title: spec.title ?? null,
    declared_op: spec.op ?? null,
    has_args_fn: Boolean(spec.args),
    params: Object.keys(params),
    calls,
  });
}

process.stdout.write(JSON.stringify(report, null, 1));
