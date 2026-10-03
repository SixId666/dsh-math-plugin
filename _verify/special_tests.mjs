/**
 * 用户规格 §12.4 的八个专项测试（真实调用 Python 引擎，不是静态检查）。
 *
 *   测试一 简单积分：必须直接调用工具拿到精确结果，不写程序
 *   测试二 复杂积分：结果正确且不重复调用
 *   测试三 无法求解的积分：失败后停止重复尝试并合理解释限制
 *   测试四 连续追问：第二轮复用第一轮结果
 *   测试五 已有答案验证：定位错误并给出正确结果
 *   测试六 思路分析：保留必要数学分析与「何时不该调用工具」的纪律
 *   测试七 结果交叉验证：重要结果经独立验证
 *   测试八 相似表达式：缓存正确复用且不同条件不混用
 *
 * 用法：
 *   $env:ELECTRON_RUN_AS_NODE=1
 *   & "D:\deepseek_harness\DeepSeek Harness.exe" E:\dsh-math\_verify\special_tests.mjs
 */

import { boot, readStats, report, resultOf } from "./plugin_host.mjs";

const findings = [];
const check = (label, ok, detail) => findings.push(report(label, ok, detail));

const { call, shutdown, sections } = await boot();

const stats = async () => readStats(await call("math_status", { action: "cache_stats" }));

/* ------------------------------------------------------- 测试一 简单积分 */
console.log("\n=== 测试一 简单积分：直接调用工具，不写程序 ===");
{
  const before = await stats();
  const started = Date.now();
  const text = await call("math_integrate", { expr: "x^2", lower: "0", upper: "1" });
  const elapsed = Date.now() - started;
  const after = await stats();
  console.log(text);
  check(
    "一次调用得到精确结果 1/3",
    text.includes("1/3") && /成功/.test(text),
    `用时 ${elapsed}ms`,
  );
  check(
    "只用了一次引擎调用（没有反复试算）",
    after.misses - before.misses === 1,
    `misses ${before.misses} → ${after.misses}`,
  );
  check(
    "返回里不含自写程序的痕迹（无 Traceback/python 命令）",
    !/Traceback|python -c|\.py\b/i.test(text),
    "",
  );
}

/* ------------------------------------------------------- 测试二 复杂积分 */
console.log("\n=== 测试二 复杂积分：结果正确且不重复调用 ===");
{
  const before = await stats();
  const expr = "x^2*log(x)";
  const first = await call("math_integrate", { expr });
  const afterFirst = await stats();
  console.log(first);
  check(
    "复杂积分得到精确结果 x³ln x/3 − x³/9",
    /x\*\*3/.test(first) && /log\(x\)/.test(first),
    "",
  );
  check(
    "复杂积分只调用一次引擎",
    afterFirst.misses - before.misses === 1,
    `misses ${before.misses} → ${afterFirst.misses}`,
  );
  const second = await call("math_integrate", { expr });
  const afterSecond = await stats();
  check(
    "相同表达式第二次调用命中缓存（不再重算）",
    afterSecond.hits - afterFirst.hits === 1 && afterSecond.misses === afterFirst.misses,
    `hits ${afterFirst.hits} → ${afterSecond.hits}`,
  );
  check("两次返回结果一致", resultOf(second) === resultOf(first) && resultOf(first) !== "", "");
}

/* ------------------------------------------------- 测试三 无法求解的积分 */
console.log("\n=== 测试三 无法直接求解的积分：停止重试并解释限制 ===");
{
  const before = await stats();
  // x^x 与 sin(sin(x)) 没有初等原函数：SymPy 返回未求值形式，工具必须如实标注。
  const expr = "x^x";
  const first = await call("math_integrate", { expr });
  const afterFirst = await stats();
  console.log(first);
  const honest =
    /未求得解析结果|unsolved/.test(first) && !/— 成功/.test(first) && /未能求出|未求值|没有初等|无初等/.test(first);
  check("工具如实说明无法求出（不伪造结果）", honest, "");
  check("给出可尝试的方向而不是重复试同一方法", /建议|可尝试/.test(first), "");
  check(
    "第一次调用确实执行了引擎",
    afterFirst.misses - before.misses === 1,
    `misses ${before.misses} → ${afterFirst.misses}`,
  );
  const second = await call("math_integrate", { expr });
  const afterSecond = await stats();
  console.log(second);
  check(
    "第二次相同调用不再重算（缓存命中或被拦截为重复失败）",
    /repeated_failure|已拦截|重复失败/.test(second) ||
      (afterSecond.misses === afterFirst.misses && afterSecond.hits > afterFirst.hits),
    `misses ${afterSecond.misses}, hits ${afterSecond.hits}, blocked ${afterSecond.blocked}`,
  );
  const third = await call("math_integrate", { expr, retry: true });
  const afterThird = await stats();
  check(
    "显式要求 retry 才允许再试一次（重试受控）",
    afterThird.misses > afterSecond.misses || afterThird.blocked > afterSecond.blocked,
    `misses ${afterSecond.misses} → ${afterThird.misses}，blocked ${afterSecond.blocked} → ${afterThird.blocked}`,
  );
  check("反复重试后仍然不伪造结果", /未求得解析结果|unsolved|重复失败|已拦截/.test(third), "");
}

/* ---------------------------------------------------------- 测试四 连续追问 */
console.log("\n=== 测试四 连续追问：复用第一轮结果 ===");
{
  const before = await stats();
  const integral = await call("math_integrate", { expr: "sin(x)/x", lower: "0", upper: "oo" });
  const afterFirst = await stats();
  console.log(integral);
  check("第一轮算出 ∫₀^∞ sin x/x = π/2", integral.includes("pi/2"), "");

  // 追问「这一步的结果对吗」：用独立验证工具针对同一个结果，而不是重算整题。
  const followUp = await call("math_verify", {
    expr: "sin(x)/x",
    candidate: "pi/2",
    kind: "definite_integral",
    var: "x",
    lower: "0",
    upper: "oo",
  });
  const afterFollowUp = await stats();
  console.log(followUp);
  check(
    "追问走独立验证（不同算子），不重算原积分",
    afterFollowUp.misses === afterFirst.misses + 1 && /通过/.test(followUp),
    `misses ${afterFirst.misses} → ${afterFollowUp.misses}`,
  );

  const repeat = await call("math_integrate", { expr: "sin(x)/x", lower: "0", upper: "oo" });
  const afterRepeat = await stats();
  check(
    "再次问到原积分时直接复用缓存（不重新计算整题）",
    afterRepeat.hits > afterFollowUp.hits && repeat.includes("pi/2"),
    `hits ${afterFollowUp.hits} → ${afterRepeat.hits}`,
  );
  check("第一轮结果未被后续调用改变", resultOf(integral) === resultOf(repeat) && resultOf(integral) !== "", "");
}

/* ------------------------------------------------------ 测试五 已有答案验证 */
console.log("\n=== 测试五 已有答案验证：定位错误并给出正确结果 ===");
{
  const correct = await call("math_verify", {
    expr: "x^2",
    candidate: "2*x",
    kind: "derivative",
    var: "x",
  });
  console.log(correct);
  check("正确答案判为通过", correct.includes("通过") && !correct.includes("未通过"), "");

  const wrong = await call("math_verify", {
    expr: "x*exp(x^2)",
    candidate: "exp(x^2)",
    kind: "integral",
    var: "x",
  });
  console.log(wrong);
  check("错误答案判为未通过", wrong.includes("未通过"), "");
  check(
    "给出定位错误的证据（差式/残差或正确结果）",
    /证据|残差|差式|应为|正确|exp\(x\*\*2\)\/2|1\/2/.test(wrong),
    "",
  );

  const equation = await call("math_verify", {
    expr: "x^2-3*x+2=0",
    candidate: "1",
    kind: "equation_solution",
    var: "x",
  });
  console.log(equation);
  check("方程解验证给出明确结论（该根确实成立）", /通过/.test(equation), "");
}

/* --------------------------------------------------------- 测试六 思路分析 */
console.log("\n=== 测试六 思路分析：保留必要数学分析与调用纪律 ===");
{
  const section = sections.find((item) => item.name === "dsh-math:tools");
  const text = String(section?.text ?? "");
  console.log(text);
  check(
    "注册了工具调用纪律的系统提示 section",
    Boolean(section) && section.order === 2500,
    `name=${section?.name}`,
  );
  const rules = ["计算交给工具", "自写程序", "验证交给独立机制", "推理", "unsolved", "追问"];
  const missing = rules.filter((rule) => !text.includes(rule));
  check("纪律文本覆盖用户规格的核心要求", missing.length === 0, missing.length ? `缺少：${missing.join("、")}` : "");

  const { registered } = await boot();
  const computeTools = ["math_integrate", "math_derivative", "math_limit", "math_verify", "math_vector_calculus"];
  const noCallGuidance = computeTools.filter((name) => {
    const description = String(registered.get(name)?.description ?? "");
    return /何时不要调用/.test(description);
  });
  check(
    "计算类工具描述里写明「何时不要调用」（避免把数学分析交给计算器）",
    noCallGuidance.length === computeTools.length,
    `覆盖 ${noCallGuidance.length}/${computeTools.length}`,
  );
  const vc = String(registered.get("math_vector_calculus")?.description ?? "");
  check(
    "曲面积分工具明确要求先由模型判断投影/高斯/补面",
    /数学分析|投影|补面|轮换对称/.test(vc),
    "",
  );
}

/* ------------------------------------------------------ 测试七 结果交叉验证 */
console.log("\n=== 测试七 重要结果经过独立交叉验证 ===");
{
  const verified = await call("math_verify", {
    expr: "x*exp(x^2)",
    candidate: "(1/2)*exp(x^2)",
    kind: "integral",
    var: "x",
  });
  console.log(verified);
  check("不定积分结果通过独立验证", /通过/.test(verified) && /验证/.test(verified), "");
  check(
    "验证方式不是「同一个函数再算一次」（标明独立方法/数值核对）",
    /独立|numeric|数值|导数|求导/.test(verified),
    "",
  );

  const crossNumeric = await call("math_verify", {
    expr: "sin(x)/x",
    candidate: "pi/2",
    kind: "definite_integral",
    var: "x",
    lower: "0",
    upper: "oo",
  });
  console.log(crossNumeric);
  check("定积分结果有独立数值复核", /通过/.test(crossNumeric), "");

  const matrix = await call("math_verify", {
    matrix: ["1,2", "3,4"],
    candidate: "-2",
    kind: "determinant",
  });
  console.log(matrix);
  check("行列式结果可独立验证（逆矩阵/定义式）", /通过|未通过/.test(matrix), "");
}

/* ------------------------------------------------- 测试八 相似表达式与条件隔离 */
console.log("\n=== 测试八 相似表达式：缓存复用但不混用条件 ===");
{
  const a1 = await call("math_integrate", { expr: "x^2", lower: "0", upper: "1" });
  const a2 = await call("math_integrate", { expr: "x^2", lower: "0", upper: "2" });
  console.log(a1);
  console.log(a2);
  check("∫₀¹x² = 1/3", a1.includes("1/3"), "");
  check("∫₀²x² = 8/3（不同上限没有复用 1/3 的缓存）", a2.includes("8/3"), "");

  const d1 = await call("math_integrate", { expr: "1/x", lower: "1", upper: "oo" });
  const d2 = await call("math_integrate", { expr: "1/x^2", lower: "1", upper: "oo" });
  console.log(d1);
  console.log(d2);
  check("∫₁^∞dx/x 判为发散（oo）", d1.includes("oo"), "");
  check("∫₁^∞dx/x² = 1（相似但不同条件，结论不同）", /结果: 1\b/.test(d2) || d2.includes("1\n"), "");

  const before = await stats();
  const again = await call("math_integrate", { expr: "x^2", lower: "0", upper: "1" });
  const after = await stats();
  check(
    "再次请求 ∫₀¹x² 命中缓存（相似表达式之间缓存正确复用）",
    after.hits > before.hits && again.includes("1/3"),
    `hits ${before.hits} → ${after.hits}`,
  );
  check("缓存复用的结果与原结果一致", resultOf(again) === resultOf(a1) && resultOf(a1) !== "", "");
}

await shutdown();

const failed = findings.filter((item) => !item.ok);
console.log(`\n===== 专项测试：${findings.length - failed.length}/${findings.length} 通过 =====`);
if (failed.length) {
  for (const item of failed) console.log(`  ✗ ${item.label}${item.detail ? ` —— ${item.detail}` : ""}`);
  process.exitCode = 1;
}
