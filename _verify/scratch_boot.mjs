// Scratch boot: reproduce the host's own plugin loading against a clone profile that
// contains ONLY dsh-math, so we can see the real import error (or prove the import works).
import { cp, mkdir, rm, writeFile, readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const ASAR_DSH = "D:/deepseek_harness/resources/app.asar/dsh";
const app = await import(pathToFileURL(ASAR_DSH + "/node_modules/@deepseek-ai/dsh-app-boot/lib/index.js").href);
const installAnchor = ASAR_DSH + "/node_modules/@deepseek-ai/dsh/package.json";
const CLONE = "E:/dsh-math/_verify/profile-clone";
const SRC = "E:/dsh-math/plugin";
const HOME = "C:/Users/HP/.dsh";

await rm(CLONE, { recursive: true, force: true });
await mkdir(join(CLONE, "node_modules"), { recursive: true });
await cp(SRC, join(CLONE, "node_modules", "dsh-math"), { recursive: true });
await writeFile(join(CLONE, "package.json"), JSON.stringify({
  name: "dsh-profile-clone",
  private: true,
  dependencies: { "dsh-math": "file:" + SRC },
  dsh: { profile: { bundles: ["dsh-math"] } },
}, null, 2) + "\n");
await writeFile(join(CLONE, "cordis.yml"), "[]\n");

const profile = app.loadProfileDirectory("dsh", CLONE, installAnchor);
console.log("layers:", profile.layers.map((l) => `${l.packageName} -> ${l.packageDir}`).join(" | "));
console.log("skipped:", JSON.stringify(profile.skippedBundles));
console.log("patch rows:", JSON.stringify(profile.layers.flatMap((l) => l.patches)).slice(0, 400));

const resolution = await app.createRuntimeResolution({ installAnchor, profile });
const allEntries = resolution.entries instanceof Map ? [...resolution.entries.values()] : Object.values(resolution.entries);
const lookup = (name) => (resolution.entries instanceof Map ? resolution.entries.get(name) : allEntries.find((e) => e?.name === name));
console.log("resolution entries:", allEntries.length, "of which named:", allEntries.filter((e) => e?.name).length);
console.log("first entries:", JSON.stringify(allEntries.slice(0, 3)));
console.log("localPackageNames:", JSON.stringify([...resolution.localPackageNames].slice(0, 12)));
console.log("dsh-math entry:", JSON.stringify(lookup("dsh-math")));

let ctx;
const profileContext = {
  name: "clone", dir: CLONE, patchPath: profile.patchPath, installAnchor,
  home: HOME, overlays: [], telemetryDisabledEnv: undefined,
};
const patches = app.readProfilePatches("dsh", profileContext, profile);
console.log("patches:", JSON.stringify(patches).slice(0, 400));

const diagFiles = [join(SRC, ".diag.log"), join(SRC, "lib", "diag.log"), join(process.env.TEMP ?? ".", "dsh-math-diag.log")];
for (const f of diagFiles) await rm(f, { force: true });

try {
  ctx = await app.boot("dsh", join(CLONE, "cordis.yml"), patches, async (hostCtx) => {
    globalThis.__bootCtx = hostCtx;
    hostCtx.provide("profileContext", profileContext);
    await hostCtx.plugin(app.PluginPackages, { resolution });
  });
  console.log("BOOT OK");
} catch (error) {
  console.log("BOOT ERR name=", error?.name, "code=", error?.code);
  console.log("BOOT ERR message=", String(error?.message).slice(0, 1200));
  const messages = error?.startup?.messages ?? [];
  console.log("startup.messages:", messages.length);
  for (const m of messages) console.log("  LOG", m.type, m.name, JSON.stringify(m.args).slice(0, 1500));
  try {
    const loader = globalThis.__bootCtx?.get?.("loader");
    if (loader?.entries !== undefined) {
      console.log("loader entries:");
      for (const [key, entry] of loader.entries) {
        console.log("  ", key, "fiber=", entry.fiber === undefined ? "undefined" : `state=${entry.fiber.state}`, "options.name=", entry.options?.name);
      }
    } else {
      console.log("no loader on ctx");
    }
  } catch (inner) {
    console.log("post-mortem failed:", String(inner?.message));
  }
}

for (const f of diagFiles) {
  console.log("diag", f, existsSync(f) ? "-> " + (await readFile(f, "utf8")).trim().replace(/\n/g, " ; ") : "MISSING");
}
process.exit(0);
