/**
 * 插件自带的 defineTool —— 零裸导入版本。
 *
 * 为什么自带：DSH 的 linked bundle（workspace 目录直接 link 进 profile）在本构建里
 * 解析不到 shipped 的 `@deepseek-ai/*` 包（`@deepseek-ai/schemastery`、
 * `@deepseek-ai/dsh-tools`），一旦入口里有这类裸导入，插件就会以
 * `failed to import` 收场。本文件复刻了 `@deepseek-ai/dsh-tools` 的
 * `defineTool` 契约（见 dsh-tools/lib/index.js:838-887 与
 * `ctx.tools.register` 的真实要求，dsh-tools/lib/index.js:2878-2887），
 * 只用标准 JavaScript 实现，不依赖任何宿主包。
 *
 * 复刻的契约：
 *   - parameters 用「作者级」写法（type/description/required/items/enum/oneOf），
 *     编译成 DSH 支持的受限 JSON Schema 子集；
 *   - execute 前先校验参数，违规抛 ToolArgsError（code INVALID_ARGS）；
 *   - output 必须声明 { schema, render }，render 返回 content block 数组；
 *   - timeoutMs 若给出必须是正的有限数。
 *
 * 与宿主实现的唯一有意差异：作者级节点里的 `required: false` 被接受并忽略
 * （宿主会直接抛 `parameters.<name>.required must be true when present`），
 * 这样工具表里可以显式写出「可选」而不会炸掉整个插件。
 */

/** DSH 支持的 value schema 类型。 */
const SCHEMA_TYPES = ["string", "number", "integer", "boolean", "null", "array", "object", "json"];

/** 允许直接透传的注解键。 */
const ANNOTATION_KEYS = ["title", "description", "default", "examples", "deprecated"];

function isRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function fail(message) {
  const error = new Error(message);
  error.name = "ToolSchemaError";
  throw error;
}

/** 作者级节点 → JSON Schema（受限子集）。 */
function compileValue(input, path) {
  if (!isRecord(input)) fail(`${path} must be a value schema object`);
  const node = {};
  for (const key of ANNOTATION_KEYS) {
    if (Object.hasOwn(input, key) && input[key] !== undefined) node[key] = input[key];
  }
  if (Object.hasOwn(input, "enum")) {
    if (!Array.isArray(input.enum) || input.enum.length === 0) fail(`${path}.enum must be a non-empty array`);
    node.enum = input.enum;
  }
  if (Object.hasOwn(input, "const")) node.const = input.const;
  if (Object.hasOwn(input, "oneOf")) {
    if (Object.hasOwn(input, "type")) fail(`${path}: oneOf cannot be combined with type`);
    if (!Array.isArray(input.oneOf) || input.oneOf.length < 2) fail(`${path}.oneOf needs at least 2 branches`);
    node.oneOf = input.oneOf.map((branch, index) => compileValue(branch, `${path}.oneOf[${index}]`));
    return node;
  }
  const type = input.type;
  if (typeof type !== "string" || !SCHEMA_TYPES.includes(type)) {
    fail(`${path}.type must be one of ${SCHEMA_TYPES.join("/")}`);
  }
  node.type = type;
  if (type === "array") {
    if (!Object.hasOwn(input, "items")) fail(`${path}.items is required when type is array`);
    node.items = compileValue(input.items, `${path}.items`);
  }
  if (type === "object") {
    if (!Object.hasOwn(input, "properties")) fail(`${path}.properties is required when type is object`);
    node.properties = compilePropertyMap(input.properties, `${path}.properties`);
    if (Object.hasOwn(input, "additionalProperties")) {
      if (typeof input.additionalProperties !== "boolean") fail(`${path}.additionalProperties must be a boolean`);
      node.additionalProperties = input.additionalProperties;
    }
    if (input.required !== undefined && !Array.isArray(input.required)) fail(`${path}.required must be an array`);
    if (Array.isArray(input.required) && input.required.length) node.required = input.required;
  }
  return node;
}

/** 作者级参数表 → { type:"object", properties, required? } */
function compilePropertyMap(spec, path) {
  if (!isRecord(spec)) fail(`${path} must be an object of value schemas`);
  const properties = {};
  const required = [];
  for (const [key, node] of Object.entries(spec)) {
    if (!isRecord(node)) fail(`${path}.${key} must be a value schema object`);
    if (Object.hasOwn(node, "required")) {
      if (typeof node.required !== "boolean") fail(`${path}.${key}.required must be a boolean`);
      if (node.required === true) required.push(key);
    }
    properties[key] = compileValue(node, `${path}.${key}`);
  }
  const compiled = { type: "object", properties };
  if (required.length) compiled.required = required;
  return compiled;
}

/** 工具参数 JSON Schema。 */
export function compileParameters(spec) {
  return compilePropertyMap(spec ?? {}, "parameters");
}

function kindOf(value) {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  if (typeof value === "number") return Number.isInteger(value) ? "integer" : "number";
  return typeof value;
}

function sameValue(a, b) {
  if (Array.isArray(a) && Array.isArray(b)) {
    return a.length === b.length && a.every((item, index) => sameValue(item, b[index]));
  }
  if (isRecord(a) && isRecord(b)) {
    const keys = Object.keys(a);
    return keys.length === Object.keys(b).length && keys.every((key) => sameValue(a[key], b[key]));
  }
  return a === b;
}

function validateValue(schema, value, path, violations) {
  if (schema.oneOf) {
    const ok = schema.oneOf.some((branch) => {
      const local = [];
      validateValue(branch, value, path, local);
      return local.length === 0;
    });
    if (!ok) violations.push(`${path} must match one of the allowed schemas`);
    return;
  }
  if (Object.hasOwn(schema, "const") && !sameValue(schema.const, value)) {
    violations.push(`${path} must equal ${JSON.stringify(schema.const)}`);
    return;
  }
  if (schema.enum && !schema.enum.some((item) => sameValue(item, value))) {
    violations.push(`${path} must be one of ${schema.enum.map((item) => JSON.stringify(item)).join(", ")}`);
    return;
  }
  const type = schema.type;
  if (type === "json" || type === undefined) return;
  const actual = kindOf(value);
  const matches =
    type === actual ||
    (type === "number" && actual === "integer") ||
    (type === "object" && actual === "object");
  if (!matches) {
    violations.push(`${path} must be a ${type} (got ${actual})`);
    return;
  }
  if (type === "array" && schema.items) {
    value.forEach((item, index) => validateValue(schema.items, item, `${path}[${index}]`, violations));
    return;
  }
  if (type === "object") {
    const properties = schema.properties ?? {};
    for (const key of schema.required ?? []) {
      if (!Object.hasOwn(value, key)) violations.push(`${path}.${key} is required`);
    }
    for (const [key, sub] of Object.entries(properties)) {
      if (Object.hasOwn(value, key)) validateValue(sub, value[key], `${path}.${key}`, violations);
    }
    if (schema.additionalProperties === false) {
      for (const key of Object.keys(value)) {
        if (!Object.hasOwn(properties, key)) violations.push(`${path}.${key} is not an allowed property`);
      }
    }
  }
}

/** 参数校验，返回违规说明列表（空数组表示通过）。 */
export function validateArguments(schema, args) {
  const violations = [];
  validateValue(schema, args ?? {}, "arguments", violations);
  return violations;
}

/** 参数不合法时抛出的错误（与宿主 ToolArgsError 同形）。 */
export class ToolArgsError extends Error {
  constructor(tool, violations) {
    super(`invalid arguments: ${violations.join("; ")}`);
    this.name = "ToolArgsError";
    this.code = "INVALID_ARGS";
    this.tool = tool;
    this.violations = violations;
  }
}

/** 与 @deepseek-ai/dsh-tools 的 defineTool 同形的工具定义。 */
export function defineTool(options) {
  const name = options?.name;
  if (typeof name !== "string" || !name.trim()) fail("defineTool: name must be a non-empty string");
  if (name === "run_code") fail('defineTool: tool name "run_code" is reserved');
  const output = options?.output;
  if (!isRecord(output) || typeof output.render !== "function") {
    fail(`tool "${name}" must declare output { schema, render, presentationMeta? }`);
  }
  if (output.presentationMeta !== undefined && typeof output.presentationMeta !== "function") {
    fail(`tool "${name}" output.presentationMeta must be a function`);
  }
  if (options.timeoutMs !== undefined) {
    if (!(Number.isFinite(options.timeoutMs) && options.timeoutMs > 0)) {
      fail(`DefineTool(${name}): timeoutMs must be a positive finite number`);
    }
  }
  const parameters = compileParameters(options.parameters);
  const schema = compileValue(output.schema ?? { type: "json" }, `output.schema of "${name}"`);
  const userExecute = options.execute;
  if (typeof userExecute !== "function") fail(`DefineTool(${name}): execute must be a function`);

  const definition = {
    name,
    description: String(options.description ?? ""),
    parameters,
    output: {
      schema,
      render: output.render,
      ...(output.presentationMeta ? { presentationMeta: output.presentationMeta } : {}),
    },
    async execute(args, exec) {
      const violations = validateArguments(parameters, args);
      if (violations.length) throw new ToolArgsError(name, violations);
      return userExecute(args, exec);
    },
  };
  if (options.deferLoading !== undefined) definition.deferLoading = options.deferLoading;
  if (options.timeoutMs !== undefined) definition.timeoutMs = options.timeoutMs;
  if (typeof options.projectContent === "function") definition.projectContent = options.projectContent;
  if (typeof options.finalizeContent === "function") definition.finalizeContent = options.finalizeContent;
  if (typeof options.presentCall === "function") {
    definition.presentCall = (args) =>
      validateArguments(parameters, args).length ? void 0 : options.presentCall(args);
  }
  if (typeof options.presentResult === "function") definition.presentResult = options.presentResult;
  if (options.isConcurrencySafe !== undefined) definition.isConcurrencySafe = options.isConcurrencySafe;
  return definition;
}

export default defineTool;
