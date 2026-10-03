// Final acceptance proof inside the host's own loader, using the **installed** copy of the
// packaged plugin (the 1.0.2 tarball content unpacked under the desktop profile), not the
// source directory. Boots a clone profile with @deepseek-ai/dsh-base + dsh-math, captures the
// tool definitions our plugin registers, and executes real calls — including the cases that
// were fixed late (green non-rectangular verification, matrix_inverse text candidate, and the
// matrix arguments passed as a single string instead of an array of row strings).
import { cp, mkdir, rm, writeFile } from "node:fs/promises";
import { appendFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const OUT = "E:/dsh-math/_verify/scratch_boot_installed_out.txt";
writeFileSync(OUT, "");
const rawLog = console.log.bind(console);
console.log = (...args) => {
  rawLog(...args);
  try { appendFileSync(OUT, args.map((a) => (typeof a === "string" ? a : JSON.stringify(a))).join(" ") + "\n"); } catch {}
};
const rawErr = process.stderr.write.bind(process.stderr);
process.stderr.write = (chunk, ...rest) => {
  try { appendFileSync(OUT, String(chunk)); } catch {}
  return rawErr(chunk, ...rest);
};

const ASAR_DSH = "D:/deepseek_harness/resources/app.asar/dsh";
const app = await import(pathToFileURL(ASAR_DSH + "/node_modules/@deepseek-ai/dsh-app-boot/lib/index.js").href);
const installAnchor = ASAR_DSH + "/node_modules/@deepseek-ai/dsh/package.json";
const CLONE = "E:/dsh-math/_verify/profile-clone-installed";
const SRC = "C:/Users/HP/.dsh/profiles/desktop/node_modules/dsh-math"; // the installed 1.0.2 copy
const HOME = "C:/Users/HP/.dsh";
const PYTHON = "E:\\anaconda3\\python.exe";

await rm(CLONE, { recursive: true, force: true });
await mkdir(join(CLONE, "node_modules"), { recursive: true });
await cp(SRC, join(CLONE, "node_modules", "dsh-math"), { recursive: true });
await writeFile(join(CLONE, "package.json"), JSON.stringify({
  name: "dsh-profile-clone-installed",
  private: true,
  dependencies: { "@deepseek-ai/dsh-base": "0.2.0-rc.2", "dsh-math": "file:" + SRC },
  dsh: { profile: { bundles: ["@deepseek-ai/dsh-base", "dsh-math"] } },
}, null, 2) + "\n");
await writeFile(join(CLONE, "cordis.yml"), "[]\n");
await writeFile(join(CLONE, "cordis.patch.yml"), `- id: dsh-math\n  config:\n    python: ${PYTHON}\n`);

const version = JSON.parse(String(await (await import("node:fs/promises")).readFile(join(SRC, "package.json"), "utf8"))).version;
console.log("installed copy source:", SRC, "version:", version);

const profile = app.loadProfileDirectory("dsh", CLONE, installAnchor);
console.log("layers:", profile.layers.map((l) => l.packageName).join(" | "), "| skipped:", JSON.stringify(profile.skippedBundles));
const resolution = await app.createRuntimeResolution({ installAnchor, profile });

const profileContext = {
  name: "clone-installed", dir: CLONE, patchPath: profile.patchPath, installAnchor,
  home: HOME, overlays: [], telemetryDisabledEnv: undefined,
};
const patches = app.readProfilePatches("dsh", profileContext, profile);
console.log("patch count:", patches.length);

const captured = [];
let ctx;
try {
  ctx = await app.boot("dsh", join(CLONE, "cordis.yml"), patches, async (hostCtx) => {
    globalThis.__bootCtx = hostCtx;
    hostCtx.provide("profileContext", profileContext);
    await hostCtx.plugin(app.PluginPackages, { resolution });
    await hostCtx.plugin({
      name: "capture-probe",
      inject: ["tools"],
      apply(probeCtx) {
        const tools = probeCtx.tools;
        const original = tools.register.bind(tools);
        tools.register = (definition) => {
          captured.push(definition);
          return original(definition);
        };
      },
    });
  });
  console.log("BOOT OK; entries active");
} catch (error) {
  console.log("BOOT ERR", error?.name, String(error?.message).slice(0, 800));
  for (const m of error?.startup?.messages ?? []) console.log("  LOG", m.type, m.name, JSON.stringify(m.args).slice(0, 600));
}

const mathTools = captured.filter((d) => String(d.name).startsWith("math_"));
console.log("registered tools:", captured.length, "| math_* :", mathTools.length);
console.log("math names:", mathTools.map((d) => d.name).join(", "));

const call = async (name, args) => {
  const definition = captured.find((d) => d.name === name);
  if (!definition) return console.log(`-- ${name}: NOT REGISTERED`);
  const started = Date.now();
  try {
    const value = await definition.execute(args, {});
    console.log(`-- ${name} (${Date.now() - started}ms):\n${String(value).slice(0, 1200)}\n`);
  } catch (error) {
    console.log(`-- ${name} THREW:`, String(error?.message).slice(0, 400));
  }
};

await call("math_status", { action: "ops" });
await call("math_integrate", { expr: "x^2", mode: "definite", lower: "0", upper: "1" });
await call("math_vector_calculus", { mode: "green", expr: "2*x*y-x^2", expr2: "x+y^2", x_lower: "-1", x_upper: "1", y_lower_expr: "x^2", y_upper_expr: "1" });
await call("math_verify", { kind: "matrix_inverse", matrix: ["1,2", "3,4"], candidate: "-2,1;1.5,-0.5" });
await call("math_verify", { kind: "matrix_inverse", matrix: ["1,2", "3,4"], candidate: "1,0;0,1" });
// 矩阵参数以「单个字符串」形式传入（1.0.2 起 schema 同时接受字符串与字符串数组；
// 1.0.1 及以前会在这里抛 invalid arguments: arguments.matrix must be a array (got string)）
await call("math_verify", { kind: "matrix_inverse", matrix: "1,2;3,4", candidate: "-2,1;1.5,-0.5" });
await call("math_verify", { kind: "determinant", matrix: "1,2;3,4", candidate: "-2" });
await call("math_matrix", { operation: "matmul", matrix: "1,2;3,4", matrix_b: "5,6;7,8" });
await call("math_matrix", { operation: "quadratic", expr: "x1^2+2*x1*x2+3*x2^2" });

try {
  await globalThis.__bootCtx?.fiber?.dispose?.();
  console.log("disposed");
} catch (error) {
  console.log("dispose error:", String(error?.message).slice(0, 200));
}
process.exit(0);
