/**
 * 验证 math_matrix / math_verify 的矩阵参数在参数校验层同时接受两种写法：
 *   ['1,2','3,4']（字符串数组，工具层 matrixArg 会拼成 "1,2;3,4"）
 *   '1,2;3,4'    （单个字符串，此前会被 arguments.matrix must be a array 拒掉）
 * 并确认非法形态（数字、数组里混入数字）仍被拒。
 * 用法：$env:ELECTRON_RUN_AS_NODE=1; & "D:\deepseek_harness\DeepSeek Harness.exe" E:\dsh-math\_verify\probe_schema_forms.mjs
 */
import { TOOL_SPECS } from "../plugin/lib/tools.js";
import { compileParameters, validateArguments } from "../plugin/lib/define_tool.js";

const byName = new Map(TOOL_SPECS.map((s) => [s.name, s]));

function tryCall(name, args) {
  const spec = byName.get(name);
  const schema = compileParameters(spec.params);
  // validateArguments 返回违规说明列表（空数组 = 通过），不抛异常；defineTool 里才抛 ToolArgsError
  const violations = validateArguments(schema, args);
  if (violations.length) return { ok: false, detail: "invalid arguments: " + violations.join("; ") };
  try {
    const mapped = spec.args ? spec.args(args) : null;
    return { ok: true, op: mapped?.op ?? null, args: mapped?.args ?? null };
  } catch (exc) {
    return { ok: false, detail: "args(): " + String(exc.message || exc) };
  }
}

const cases = [
  ["math_verify", { kind: "matrix_inverse", matrix: "1,2;3,4", candidate: "-2,1;1.5,-0.5" }, "ok"],
  ["math_verify", { kind: "matrix_inverse", matrix: ["1,2", "3,4"], candidate: "-2,1;1.5,-0.5" }, "ok"],
  ["math_verify", { kind: "matrix_inverse", matrix: ["[1,2]", "[3,4]"], candidate: "[[-2,1],[1.5,-0.5]]" }, "ok"],
  ["math_verify", { kind: "determinant", matrix: "1,2;3,4", candidate: "-2" }, "ok"],
  ["math_verify", { kind: "eigen", matrix: "2,1;1,2", candidate: "1,3" }, "ok"],
  ["math_matrix", { operation: "det", matrix: "1,2;3,4" }, "ok"],
  ["math_matrix", { operation: "inv", matrix: ["1,2", "3,4"] }, "ok"],
  ["math_matrix", { operation: "matmul", matrix: "1,2;3,4", matrix_b: "5,6;7,8" }, "ok"],
  ["math_matrix", { operation: "matmul", matrix: ["1,2", "3,4"], matrix_b: ["5,6", "7,8"] }, "ok"],
  // 非法形态必须仍然被拒
  ["math_matrix", { operation: "det", matrix: 123 }, "reject"],
  ["math_matrix", { operation: "det", matrix: ["1,2", 3] }, "reject"],
];

let bad = 0;
for (const [name, args, expect] of cases) {
  const r = tryCall(name, args);
  const label = `${name}(${JSON.stringify(args).slice(0, 74)})`;
  const good = expect === "ok" ? r.ok : !r.ok;
  if (!good) bad += 1;
  console.log(`${good ? "PASS" : "BAD "} [期望${expect === "ok" ? "接受" : "拒绝"}] ${label}`);
  if (r.ok) console.log(`      → op=${r.op} args=${JSON.stringify(r.args).slice(0, 110)}`);
  else console.log(`      → ${r.detail}`);
}
console.log(bad === 0 ? "\nSCHEMA FORMS OK（两种写法都接受，非法形态仍被拒）" : `\nSCHEMA FORMS: ${bad} 处异常`);
process.exitCode = bad === 0 ? 0 : 1;
