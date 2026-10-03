// End-to-end proof inside the host's own loader: boot a clone profile containing
// @deepseek-ai/dsh-base + dsh-math, capture the definitions our plugin registers and
// execute real tool calls through them.
import { cp, mkdir, rm, writeFile } from "node:fs/promises";
import { appendFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const OUT = "E:/dsh-math/_verify/scratch_boot_full_out.txt";
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
const CLONE = "E:/dsh-math/_verify/profile-clone-full";
const SRC = "E:/dsh-math/plugin";
const HOME = "C:/Users/HP/.dsh";
const PYTHON = "E:\\anaconda3\\python.exe";

await rm(CLONE, { recursive: true, force: true });
await mkdir(join(CLONE, "node_modules"), { recursive: true });
await cp(SRC, join(CLONE, "node_modules", "dsh-math"), { recursive: true });
await writeFile(join(CLONE, "package.json"), JSON.stringify({
  name: "dsh-profile-clone-full",
  private: true,
  dependencies: { "@deepseek-ai/dsh-base": "0.2.0-rc.2", "dsh-math": "file:" + SRC },
  dsh: { profile: { bundles: ["@deepseek-ai/dsh-base", "dsh-math"] } },
}, null, 2) + "\n");
await writeFile(join(CLONE, "cordis.yml"), "[]\n");
await writeFile(join(CLONE, "cordis.patch.yml"), `- id: dsh-math\n  config:\n    python: ${PYTHON}\n`);

const profile = app.loadProfileDirectory("dsh", CLONE, installAnchor);
console.log("layers:", profile.layers.map((l) => l.packageName).join(" | "), "| skipped:", JSON.stringify(profile.skippedBundles));
const resolution = await app.createRuntimeResolution({ installAnchor, profile });

const profileContext = {
  name: "clone-full", dir: CLONE, patchPath: profile.patchPath, installAnchor,
  home: HOME, overlays: [], telemetryDisabledEnv: undefined,
};
const patches = app.readProfilePatches("dsh", profileContext, profile);
console.log("patch count:", patches.length, "rows:", JSON.stringify(patches.flatMap((p) => p.insert ?? [])).slice(0, 300));

const captured = [];
let ctx;
try {
  ctx = await app.boot("dsh", join(CLONE, "cordis.yml"), patches, async (hostCtx) => {
    globalThis.__bootCtx = hostCtx;
    hostCtx.provide("profileContext", profileContext);
    await hostCtx.plugin(app.PluginPackages, { resolution });
    // Same mechanism our plugin uses; runs as soon as "tools" exists.
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
        console.log("capture-probe: tools.register patched");
      },
    });
  });
  console.log("BOOT OK; entries active");
} catch (error) {
  console.log("BOOT ERR", error?.name, String(error?.message).slice(0, 800));
  for (const m of error?.startup?.messages ?? []) console.log("  LOG", m.type, m.name, JSON.stringify(m.args).slice(0, 600));
}

console.log("registered tools:", captured.length);
console.log("names:", captured.map((d) => d.name).join(", "));

const call = async (name, args) => {
  const definition = captured.find((d) => d.name === name);
  if (!definition) return console.log(`-- ${name}: NOT REGISTERED`);
  const started = Date.now();
  try {
    const value = await definition.execute(args, {});
    console.log(`-- ${name} (${Date.now() - started}ms):\n${String(value).slice(0, 900)}\n`);
  } catch (error) {
    console.log(`-- ${name} THREW:`, String(error?.message).slice(0, 400));
  }
};

await call("math_status", { action: "ops" });
await call("math_simplify", { expr: "sin(x)^2+cos(x)^2" });
await call("math_integrate", { expr: "x^2", mode: "definite", lower: "0", upper: "1" });

try {
  await globalThis.__bootCtx?.fiber?.dispose?.();
  console.log("disposed");
} catch (error) {
  console.log("dispose error:", String(error?.message).slice(0, 200));
}
process.exit(0);
