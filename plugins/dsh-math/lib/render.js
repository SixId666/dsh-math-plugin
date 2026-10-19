/**
 * 把引擎返回的结构化结果渲染成模型可读的紧凑文本。
 *
 * Token 约束（用户规格十）：只给必要信息——结果、成立条件、验证状态、错误与警告；
 * 不回传内部计算日志。验证证据只在「未通过」或「未验证」时才展开，
 * 通过时只给一行方法说明。
 */

const STATUS_LABEL = {
  ok: "成功",
  partial: "部分成功",
  unsolved: "未求得解析结果",
  error: "执行失败",
};

const VERIFY_LABEL = {
  symbolic: "符号级独立核验通过",
  numeric: "仅数值抽样一致（数值证据，不构成符号证明）",
  none: "未能验证",
  unverified: "未验证",
  independent: "独立机制核验通过",
  cross: "交叉验证通过",
  verified: "验证通过",
  partially_verified: "部分方法通过（其余方法未通过，详见方法明细）",
  "symbolic+numeric": "符号级独立核验 + 数值复核通过",
  "independent+numeric": "独立机制核验 + 数值复核通过",
};

/** passed === false 时必须换用「未通过」的措辞，不能让标签继续说「通过」。 */
const VERIFY_LABEL_FAIL = {
  symbolic: "符号级独立核验未通过",
  numeric: "数值抽样不一致（仅有数值证据，且不一致）",
  independent: "独立机制核验未通过",
  cross: "交叉验证未通过",
  none: "未能验证",
  unverified: "未验证",
  verified: "验证未通过",
  partially_verified: "验证未通过（存在未通过的方法，详见方法明细）",
  "symbolic+numeric": "符号级核验未通过",
  "independent+numeric": "独立核验未通过",
};

/** result 里这些键已由验证段落渲染，重复打印会浪费 token。 */
const RESULT_SKIP = new Set([
  "text",
  "latex",
  "approx",
  "passed",
  "verification",
  "methods",
  "method",
  "evidence",
  "unhandled_conditions",
  "cannot_verify_reason",
  "grade_note",
  "level",
]);

/** 验证证据摘要：只保留能说明结论的字段，不整段回传内部结构。 */
function summarizeEvidence(evidence, limit = 180) {
  if (!isPlainObject(evidence)) return undefined;
  const parts = [];
  const clip = (value) => {
    const text = typeof value === "string" ? value : JSON.stringify(value);
    return text.length > limit ? `${text.slice(0, limit)}…` : text;
  };
  for (const key of [
    "difference",
    "counterexample",
    "expected",
    "derivative_of_candidate",
    "closed_form",
    "residuals",
    "numeric",
    "finite_difference",
    "check_derivative_agree",
    "check_antiderivative_agree",
    "check_solution_agree",
  ]) {
    const value = evidence[key];
    if (value === undefined || value === null) continue;
    if (isPlainObject(value)) {
      const inner = [];
      if (value.agree !== undefined) inner.push(value.agree ? "一致" : "不一致");
      if (value.max_rel_dev !== undefined) inner.push(`最大相对偏差 ${value.max_rel_dev}`);
      if (value.note) inner.push(clip(value.note));
      if (value.counterexample) inner.push(`反例 ${JSON.stringify(value.counterexample)}`);
      parts.push(`${LABELS[key] ?? key}: ${inner.join("，") || clip(value)}`);
    } else if (Array.isArray(value)) {
      parts.push(`${LABELS[key] ?? key}: ${value.map((item) => clip(item)).join(", ")}`);
    } else {
      parts.push(`${LABELS[key] ?? key}: ${clip(value)}`);
    }
  }
  return parts.length ? parts.join("；") : undefined;
}

function isPlainObject(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function pushWrapped(lines, label, value) {
  if (value === undefined || value === null || value === "") return;
  if (Array.isArray(value)) {
    if (!value.length) return;
    lines.push(`${label}: ${value.map((v) => (isPlainObject(v) ? v.text ?? JSON.stringify(v) : String(v))).join("；")}`);
    return;
  }
  if (isPlainObject(value)) {
    const text = value.text ?? value.latex;
    if (text !== undefined) {
      const approx = value.approx !== undefined ? `   ≈ ${formatApprox(value.approx)}` : "";
      lines.push(`${label}: ${text}${approx}`);
      return;
    }
    lines.push(`${label}: ${JSON.stringify(value)}`);
    return;
  }
  lines.push(`${label}: ${String(value)}`);
}

function formatApprox(value) {
  if (Array.isArray(value)) {
    if (value.length && Array.isArray(value[0])) return `[${value.map((row) => `[${row.join(", ")}]`).join(", ")}]`;
    return `[${value.join(", ")}]`;
  }
  if (isPlainObject(value)) return JSON.stringify(value);
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : String(value);
  return String(value);
}

/** 主渲染函数：payload 是引擎返回的 MathResult 字典。 */
export function renderResult(payload, { op, args, spec } = {}) {
  const lines = [];
  const title = spec?.title ? `【${spec.title}】` : "";
  const operation = payload?.operation ?? op ?? "";

  if (payload?.status === "ok" || payload?.status === "partial") {
    lines.push(`${title}${operation} — ${STATUS_LABEL[payload.status] ?? payload.status}`);
  } else {
    lines.push(`${title}${operation} — ${STATUS_LABEL[payload?.status] ?? payload?.status ?? "未知状态"}`);
  }

  const result = payload?.result;
  if (isPlainObject(result) && Object.keys(result).length) {
    // 标量/表达式结果
    if (result.text !== undefined || result.latex !== undefined) {
      const shown = result.text ?? result.latex;
      const sameAsApprox = result.approx !== undefined && String(formatApprox(result.approx)) === String(shown);
      const approx = result.approx !== undefined && !sameAsApprox ? `   ≈ ${formatApprox(result.approx)}` : "";
      lines.push(`结果: ${shown}${approx}`);
    }
    for (const [key, value] of Object.entries(result)) {
      if (RESULT_SKIP.has(key)) continue;
      pushWrapped(lines, LABELS[key] ?? key, value);
    }
  } else if (result !== undefined && !isPlainObject(result)) {
    pushWrapped(lines, "结果", result);
  }

  if (isPlainObject(payload?.result) && payload.result.method !== undefined && !isPlainObject(payload.result.method)) {
    lines.push(`方法: ${String(payload.result.method)}`);
  }

  if (payload?.conditions?.length) {
    lines.push(`成立条件: ${payload.conditions.join("；")}`);
  }

  const verification = payload?.verification;
  if (isPlainObject(verification)) {
    const level = verification.level ?? verification.status;
    const passed = verification.passed;
    const label = ((passed === false ? VERIFY_LABEL_FAIL : VERIFY_LABEL)[level] ?? level ?? "未验证");
    if (passed === true) {
      lines.push(`验证: 通过 — ${label}${verification.method ? `（${verification.method}）` : ""}`);
    } else if (passed === false) {
      lines.push(`验证: 未通过 — ${label}${verification.method ? `（${verification.method}）` : ""}`);
      const summary = summarizeEvidence(verification.evidence);
      if (summary) lines.push(`关键证据: ${summary}`);
      if (verification.methods?.length) {
        for (const method of verification.methods) {
          const name = method?.name ?? method?.method ?? "未命名验证方法";
          const detail = summarizeEvidence(method?.evidence) ?? method?.note ?? method?.detail;
          const mark = method?.passed === true ? "通过" : method?.passed === false ? "未通过" : "无结论";
          lines.push(`  · [${mark}] ${name}${detail !== undefined ? `：${String(detail)}` : ""}`);
        }
      }
    } else if (level && level !== "none" && level !== "unverified") {
      lines.push(`验证: ${label}${verification.method ? `（${verification.method}）` : ""}`);
    } else {
      const reason = verification.note ?? payload?.error?.message ?? "工具未提供独立验证";
      lines.push(`验证: 未验证 — ${reason}`);
    }
    if (verification.unhandled_conditions?.length) {
      lines.push(`未纳入验证的条件: ${verification.unhandled_conditions.join("；")}`);
    }
    if (verification.cannot_verify_reason) {
      lines.push(`无法验证的原因: ${verification.cannot_verify_reason}`);
    }
  }

  if (payload?.error && payload.status !== "ok") {
    const kind = payload.error.kind ? `[${payload.error.kind}] ` : "";
    lines.push(`错误: ${kind}${payload.error.message ?? "未提供原因"}`);
  }

  if (payload?.warnings?.length) {
    lines.push(`注意: ${payload.warnings.join("；")}`);
  }

  const extra = payload?.extra;
  if (isPlainObject(extra) && Object.keys(extra).length) {
    if (extra.methods_tried) lines.push(`已尝试的方法: ${[].concat(extra.methods_tried).join(" → ")}`);
    if (extra.hint) lines.push(`建议: ${extra.hint}`);
    if (extra.suggestion) lines.push(`建议: ${extra.suggestion}`);
    if (extra.cached !== undefined) lines.push(`缓存: ${extra.cached ? "命中（同一表达式与条件此前已算过）" : "未命中"}`);
  }
  if (payload?.cached) lines.push("缓存: 命中（完全相同的表达式与条件，直接复用此前结果，未重复计算）");

  return lines.join("\n");
}

const LABELS = {
  solutions: "解",
  solution: "解",
  points: "驻点",
  eigenvalues: "特征值",
  eigenvectors: "特征向量",
  determin: "行列式",
  determinant: "行列式",
  inverse: "逆矩阵",
  rank: "秩",
  rref: "行最简形",
  transpose: "转置",
  product: "乘积",
  sum: "和",
  trace: "迹",
  diagonal: "对角阵",
  P: "变换矩阵 P",
  D: "对角阵 D",
  standard_form: "标准形",
  canonical_form: "规范形",
  definiteness: "正定性",
  principal_minors: "顺序主子式",
  gradient: "梯度",
  hessian: "海塞矩阵",
  jacobian: "雅可比矩阵",
  series: "展开式",
  expansion: "展开式",
  remainder: "余项",
  general_solution: "通解",
  particular_solution: "特解",
  pdf: "概率密度",
  pmf: "分布律",
  cdf: "分布函数",
  probability: "概率",
  posterior: "后验概率",
  prior: "先验概率",
  expectation: "期望",
  variance: "方差",
  std: "标准差",
  covariance: "协方差",
  correlation: "相关系数",
  marginal: "边缘分布",
  independent: "是否独立",
  value: "取值",
  approx: "近似值",
  exact: "精确值",
  intervals: "解集区间",
  set: "集合表示",
  domain: "定义域",
  restrictions: "限制来源",
  monotonic: "单调区间",
  extrema: "极值",
  concavity: "凹凸区间",
  inflection: "拐点",
  asymptotes: "渐近线",
  derivative: "导数",
  integral: "积分",
  limit: "极限",
  candidates: "候选点",
  multipliers: "拉格朗日乘子",
  estimate: "估计值",
  interval: "置信区间",
  statistic: "检验统计量",
  critical_value: "临界值",
  p_value: "p 值",
  decision: "结论",
  rejection_region: "拒绝域",
  difference: "差式",
  counterexample: "反例",
  expected: "应为",
  derivative_of_candidate: "候选结果的导数",
  closed_form: "闭式",
  residuals: "残差",
  numeric: "数值抽样",
  finite_difference: "中心差分",
  check_derivative_agree: "符号求导复核",
  check_antiderivative_agree: "原函数求导复核",
  check_solution_agree: "代回检验",
  candidate_substitution: "候选解代入",
  independent_solution_set: "独立求得的解集",
  domain_caveats: "定义域说明",
};

/** 校验失败的提示（模型把参数写错时给出可直接修正的说明）。 */
export function renderArgError(error) {
  return `参数校验失败：${error?.message ?? error}`;
}

export { LABELS, STATUS_LABEL, VERIFY_LABEL };
