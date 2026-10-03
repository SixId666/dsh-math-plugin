/**
 * dsh-math 绔埌绔獙鏀惰剼鏈紙涓嶄緷璧?DSH 瀹夸富锛夈€? *
 * 鐩殑锛氬湪娌℃湁瀹夸富鐨勬儏鍐典笅锛岀敤鐪熷疄鐨?Python 寮曟搸璺戦€? *   銆屾彃浠跺叆鍙?鈫?14 涓伐鍏锋敞鍐?鈫?鍙傛暟鏍￠獙 鈫?鍙傛暟缁勮 鈫?Python 寮曟搸 鈫?缁撴瀯鍖栨覆鏌撱€? * 鍏ㄩ摼璺紝骞舵鏌ワ細缂撳瓨澶嶇敤銆侀噸澶嶅け璐ユ嫤鎴€侀潪娉曞弬鏁版姤閿欍€佸悇妯″潡浠ｈ〃鎬х粨鏋溿€? *
 * 鐢ㄦ硶锛坣ode 鍦ㄦ湰鏈轰笉鍦?PATH锛岀敤 Electron 鐨?node锛夛細
 *   $env:ELECTRON_RUN_AS_NODE=1
 *   & "D:\deepseek_harness\DeepSeek Harness.exe" E:\dsh-math\_verify\harness.mjs
 */

import { spawn } from "node:child_process";
import { pathToFileURL } from "node:url";

const ENTRY = pathToFileURL("E:/dsh-math/plugin/lib/entry.js").href;

/* ------------------------------------------------------------------ 鍋?subprocess 鏈嶅姟 */
function createSubprocessShim() {
  return {
    async resolveExecutable(command) {
      const configured = process.env.DSH_MATH_PYTHON;
      if (configured) return configured;
      return command;
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
        stderr: undefined, // collect 妯″紡锛歴tderr 鍦?collected 涓婏紙涓?DSH 琛屼负涓€鑷达級
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
        waitForExit: () =>
          exited ? Promise.resolve() : new Promise((resolve) => exitWaiters.push(resolve)),
      };
    },
  };
}

/* ------------------------------------------------------------------ 鍋?ctx */
function createContext(disposers, registered, sections) {
  const shim = createSubprocessShim();
  return {
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
}

/* ------------------------------------------------------------------ 娴嬭瘯椹卞姩 */
const disposers = [];
const registered = new Map();
const sections = [];
const results = [];

let plugin;

async function callTool(name, args) {
  const definition = registered.get(name);
  if (!definition) throw new Error(`宸ュ叿鏈敞鍐岋細${name}`);
  const value = await definition.execute(args, { signal: undefined });
  const blocks = definition.output.render(args, value);
  return blocks.map((block) => block.text ?? "").join("\n");
}

async function check(label, name, args, expectations) {
  const started = Date.now();
  let text;
  let failed = null;
  try {
    text = await callTool(name, args);
  } catch (error) {
    failed = `${error?.name}: ${error?.message}`;
  }
  const elapsed = Date.now() - started;
  const problems = [];
  if (failed) {
    problems.push(`鎶涢敊 ${failed}`);
  } else {
    for (const expected of expectations) {
      if (!text.includes(expected)) problems.push(`缂哄皯 芦${expected}禄`);
    }
  }
  const ok = problems.length === 0;
  results.push({ label, ok, elapsed, problems, text });
  console.log(`${ok ? "[PASS]" : "[FAIL]"} ${label}锛?{elapsed}ms锛塦);
  if (!ok) {
    console.log(`       闂锛?{problems.join("锛?)}`);
    console.log(
      `       瀹為檯杈撳嚭锛歕n${String(text ?? failed)
        .split("\n")
        .map((line) => `         ${line}`)
        .join("\n")}`,
    );
  }
  return text;
}

async function main() {
  plugin = await import(ENTRY);
  console.log(`鎻掍欢鍏ュ彛鍔犺浇鎴愬姛锛?{Object.keys(plugin).join(", ")}`);
  console.log(`inject = ${JSON.stringify(plugin.inject)}`);

  plugin.apply(createContext(disposers, registered, sections), {});
  console.log(`宸叉敞鍐屽伐鍏?${registered.size} 涓細${[...registered.keys()].join(", ")}`);
  console.log(`绯荤粺鎻愮ず section锛?{sections.map((section) => `${section.name}@${section.order}`).join(", ")}`);

  // 鍙傛暟 schema 褰㈡€佹牳鏌ワ紙DSH 鍙帴鍙楀彈闄愬瓙闆嗭級
  for (const [toolName, definition] of registered) {
    if (definition.parameters.type !== "object" || typeof definition.output.render !== "function") {
      throw new Error(`宸ュ叿 ${toolName} 鐨勫畾涔夊舰鎬佷笉瀵筦);
    }
    if (definition.output.schema.type !== "string") throw new Error(`宸ュ叿 ${toolName} 鐨?output schema 涓嶅`);
    for (const [key, node] of Object.entries(definition.parameters.properties)) {
      if (typeof node.type !== "string") throw new Error(`${toolName}.${key} 缂哄皯 type`);
      if (Object.hasOwn(node, "required") && node.required !== undefined) {
        throw new Error(`${toolName}.${key} 缂栬瘧鍚庝粛甯?required 閿甡);
      }
    }
  }
  console.log("鍙傛暟 schema 褰㈡€佹牳鏌ラ€氳繃锛坥bject + 姣忛」鍗曚竴 type锛?);

  console.log("\n--- 鏈惎鍔ㄥ紩鎿庢椂鐨勭姸鎬佹煡璇?---");
  await check("math_status锛堝紩鎿庢湭鍚姩锛?, "math_status", { action: "status" }, ["寮曟搸灏氭湭鍚姩"]);

  console.log("\n--- 楂樻暟锛氭瀬闄?瀵兼暟/绉垎/绾ф暟/寰垎鏂圭▼ ---");
  await check("瀹氱Н鍒?鈭個鹿x虏dx", "math_integrate", { expr: "x^2", lower: "0", upper: "1" }, ["1/3"]);
  const beforeCache = await callTool("math_status", { action: "cache_stats" });
  await check("鍚屼竴绉垎绗簩娆¤皟鐢紙搴斿懡涓紦瀛橈級", "math_integrate", { expr: "x^2", lower: "0", upper: "1" }, ["1/3"]);
  const afterCache = await callTool("math_status", { action: "cache_stats" });
  const hitsOf = (text) => {
    const matched = /鍛戒腑 (\d+)/.exec(text);
    return matched ? Number(matched[1]) : NaN;
  };
  const cacheOk = hitsOf(afterCache) > hitsOf(beforeCache);
  console.log(`${cacheOk ? "[PASS]" : "[FAIL]"} 缂撳瓨鍛戒腑鏁板鍔狅紙${hitsOf(beforeCache)} 鈫?${hitsOf(afterCache)}锛塦);
  if (!cacheOk) {
    results.push({ label: "缂撳瓨鍛戒腑鏁板鍔?, ok: false, problems: ["鍛戒腑鏁版湭澧炲姞"], text: afterCache });
  } else {
    results.push({ label: "缂撳瓨鍛戒腑鏁板鍔?, ok: true, problems: [], text: "" });
  }

  await check("涓嶅畾绉垎 鈭玿路e^{x虏}dx", "math_integrate", { expr: "x*exp(x^2)" }, ["exp(x**2)"]);
  await check("绉垎 鈭玠x/(1+x虏)", "math_integrate", { expr: "1/(1+x^2)" }, ["atan(x)"]);
  await check("鍙嶅父绉垎 鈭個^鈭瀞in(x)/x", "math_integrate", { expr: "sin(x)/x", lower: "0", upper: "oo" }, ["pi/2"]);
  await check("鍙戞暎鍙嶅父绉垎 鈭倎^鈭瀌x/x", "math_integrate", { expr: "1/x", lower: "1", upper: "oo" }, ["oo"]);
  await check("鏋侀檺 lim sin(x)/x", "math_limit", { expr: "sin(x)/x", var: "x", point: "0" }, ["1"]);
  await check("鍗曚晶鏋侀檺 lim (1/x)", "math_limit", { expr: "1/x", var: "x", point: "0", dir: "+" }, ["oo"]);
  await check("瀵兼暟 (sin x)'", "math_derivative", { expr: "sin(x)", var: "x" }, ["cos(x)"]);
  await check("浜岄樁瀵?(x鲁)''", "math_derivative", { expr: "x^3", var: "x", order: 2 }, ["6*x"]);
  await check("鍋忓 鈭?鈭倄 (x虏y+y鲁)", "math_derivative", { expr: "x^2*y+y^3", var: "x", mode: "partial" }, ["2*x*y"]);
  await check("娉板嫆灞曞紑 sin(x) 鍒?x^5", "math_series", { expr: "sin(x)", var: "x", point: "0", order: 5 }, ["x**5"]);
  await check("绾ф暟姹傚拰 危1/n虏", "math_series", { expr: "1/n^2", var: "n", mode: "sum" }, ["pi**2/6"]);
  await check("瑙ｆ柟绋?x虏-3x+2=0", "math_solve", { equations: ["x^2-3*x+2=0"] }, ["1", "2"]);
  await check("寰垎鏂圭▼ y'=x路y", "math_dsolve", { equations: ["y' = x*y"] }, ["exp(x**2/2)"]);

  console.log("\n--- 绾夸唬 ---");
  await check("琛屽垪寮?|1 2;3 4|", "math_matrix", { matrix: ["1,2", "3,4"], operation: "det" }, ["-2"]);
  await check("閫嗙煩闃?2脳2", "math_matrix", { matrix: ["1,2", "3,4"], operation: "inv" }, ["-1"]);
  await check("鐗瑰緛鍊?[[2,0],[0,3]]", "math_matrix", { matrix: ["2,0", "0,3"], operation: "eigen" }, ["2", "3"]);
  await check("绾挎€ф柟绋嬬粍锛堝敮涓€瑙?x=-4, y=9/2锛?, "math_matrix", {
    matrix: ["1,2", "3,4"],
    rhs: "5,6",
    operation: "solve_linear",
  }, ["-4", "9/2"]);

  console.log("\n--- 姒傜巼缁熻 ---");
  await check("鏍囧噯姝ｆ€?P(-1<X<1)", "math_probability", {
    kind: "interval",
    dist: "normal",
    params: '{"mu":0,"sigma":1}',
    lower: "-1",
    upper: "1",
  }, ["erf"]);
  await check("娉婃澗鍒嗗竷 P(X=0)", "math_probability", {
    kind: "at",
    dist: "poisson",
    point: "0",
    lam: "2",
  }, ["exp(-2)"]);
  await check("浜岄」鍒嗗竷鏈熸湜", "math_probability", {
    kind: "expectation",
    dist: "binomial",
    n: "10",
    p: "1/3",
  }, ["10/3"]);

  console.log("\n--- 鏇茬嚎鏇查潰绉垎涓庝笁澶у叕寮?---");
  await check("鏍兼灄鍏紡锛堝崟浣嶅渾锛孭=-y,Q=x锛?, "math_vector_calculus", {
    mode: "green",
    expr: "-y",
    expr2: "x",
    x_lower: "-1",
    x_upper: "1",
    y_lower_expr: "-sqrt(1-x^2)",
    y_upper_expr: "sqrt(1-x^2)",
  }, ["2*pi"]);

  console.log("\n--- 鏁板€间笌楠岃瘉 ---");
  await check("鏁板€兼眰鍊?sqrt(2)", "math_numeric", { expr: "sqrt(2)", mode: "eval" }, ["1.414"]);
  await check("鐙珛楠岃瘉涓嶅畾绉垎缁撴灉", "math_verify", {
    expr: "x*exp(x^2)",
    candidate: "(1/2)*exp(x^2)",
    kind: "integral",
    var: "x",
  }, ["閫氳繃"]);
  await check("楠岃瘉閿欒缁撴灉锛堝簲鍒ゅ畾涓嶉€氳繃锛?, "math_verify", {
    expr: "x*exp(x^2)",
    candidate: "exp(x^2)",
    kind: "integral",
    var: "x",
  }, ["鏈€氳繃"]);
  await check("鍑芥暟瀹氫箟鍩?sqrt(x-1)/(x-2)", "math_analysis", { expr: "sqrt(x-1)/(x-2)", mode: "domain" }, ["x"]);

  console.log("\n--- 寮傚父璺緞锛氶潪娉曞弬鏁?/ 閲嶅澶辫触鎷︽埅 ---");
  let invalidArgs = "锛堟湭瑙﹀彂锛?;
  try {
    await callTool("math_integrate", { lower: "0", upper: "1" });
    invalidArgs = "娌℃湁鎶ラ敊";
  } catch (error) {
    invalidArgs = `${error.name}/${error.code}: ${error.message}`;
  }
  const invalidOk = invalidArgs.includes("INVALID_ARGS");
  console.log(`${invalidOk ? "[PASS]" : "[FAIL]"} 缂哄皯蹇呭～鍙傛暟琚嫤鎴細${invalidArgs}`);
  results.push({ label: "缂哄皯蹇呭～鍙傛暟琚嫤鎴?, ok: invalidOk, problems: [], text: invalidArgs });

  await check("鏃犳硶瑙ｆ瀽鐨勮緭鍏ワ紙绗竴娆★級", "math_simplify", { expr: "@@@ not-an-expression @@@" }, ["鎵ц澶辫触", "invalid_input"]);

  console.log("\n--- 寮曟搸鐘舵€佷笌绠楀瓙娓呭崟 ---");
  const opsText = await callTool("math_status", { action: "ops" });
  const opMatch = /鍙敤绠楀瓙 (\d+) 涓?.exec(opsText);
  console.log(`${opMatch ? "[PASS]" : "[FAIL]"} 绠楀瓙娓呭崟锛?{opMatch ? opMatch[1] : "?"} 涓猔);
  results.push({
    label: "绠楀瓙娓呭崟鍙",
    ok: Boolean(opMatch) && Number(opMatch[1]) > 150,
    problems: opMatch ? [] : ["鏈В鏋愬嚭绠楀瓙鏁伴噺"],
    text: opsText.slice(0, 400),
  });

  // 鍏抽棴锛氳蛋鎻掍欢娉ㄥ唽鐨?disposer锛堜細缁撴潫 Python 杩涚▼锛?  for (const dispose of disposers.reverse()) {
    try {
      if (typeof dispose === "function") await dispose();
    } catch {
      // 蹇界暐娓呯悊澶辫触
    }
  }

  const failed = results.filter((item) => !item.ok);
  console.log(`\n===== 缁撴灉锛?{results.length - failed.length}/${results.length} 閫氳繃 =====`);
  if (failed.length) {
    for (const item of failed) console.log(`  鉁?${item.label} 鈥斺€?${item.problems.join("锛?)}`);
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error(`harness 澶辫触锛?{error?.stack ?? error}`);
  process.exitCode = 1;
});
