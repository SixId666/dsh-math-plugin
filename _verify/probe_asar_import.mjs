// Decide whether we can import the host's own modules straight out of app.asar.
const targets = [
  "file:///D:/deepseek_harness/resources/app.asar/dsh/node_modules/@deepseek-ai/dsh-app-boot/lib/index.js",
  "file:///D:/deepseek_harness/resources/app.asar/dsh/node_modules/@deepseek-ai/cordis-plugin-loader/lib/index.js",
  "file:///D:/deepseek_harness/resources/app.asar/dsh/node_modules/@deepseek-ai/dsh/lib/profile-boot-BZ2ZjNWi.js",
];
for (const url of targets) {
  try {
    const mod = await import(url);
    console.log("IMPORT OK", url.split("/").slice(-3).join("/"), "keys:", Object.keys(mod).length);
  } catch (error) {
    console.log("IMPORT ERR", url.split("/").slice(-3).join("/"), error.code, "|", String(error.message).split("\n")[0]);
  }
}
