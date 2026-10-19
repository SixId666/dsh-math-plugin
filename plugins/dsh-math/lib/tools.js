/**
 * 数学工具表：每个工具对应 Python 引擎里的一个算子。
 *
 * 工具描述里写清「何时调用 / 何时不该调用」（用户规格 6.1），这是防止
 * 模型陷入重复推理与无意义自写程序的主要手段。
 */

/**
 * 参数简写：s=string, n=number, b=boolean, a=array。
 *
 * 注意：`required` 只在为 true 时写出。DSH 的作者级 schema 方言里
 * `required: false` 会被判为非法（`parameters.<name>.required must be true when present`），
 * 所以「可选」的表达方式是不写这个键。
 */
const requiredFlag = (required) => (required ? { required: true } : {});
const S = (description, required = true) => ({ type: "string", description, ...requiredFlag(required) });
const N = (description, required = false) => ({ type: "number", description, ...requiredFlag(required) });
const B = (description, required = false) => ({ type: "boolean", description, ...requiredFlag(required) });
const A = (description, required = false) => ({
  type: "array",
  items: { type: "string" },
  description,
  ...requiredFlag(required),
});
/**
 * 字符串数组 **或** 单个字符串。
 *
 * 矩阵/向量类参数在引擎侧既接受 `['1,2','3,4']`（工具层用 matrixArg 拼成 `"1,2;3,4"`），
 * 也直接接受 `'1,2;3,4'`；如果 schema 只声明 array，模型按描述里的字符串写法调用会在
 * 参数校验处被拒（`arguments.matrix must be a array (got string)`），而这与工具描述自相矛盾。
 */
const AS = (description, required = false) => ({
  oneOf: [{ type: "array", items: { type: "string" } }, { type: "string" }],
  description,
  ...requiredFlag(required),
});

/** 把 JSON 对象字符串（例如 '{"mu":0,"sigma":1}'）解析成对象；非法输入给出可读错误。 */
const parseParamObject = (value, label = "params") => {
  if (value === undefined || value === null || value === "") return {};
  if (typeof value === "object" && !Array.isArray(value)) return value;
  if (typeof value !== "string") {
    throw new Error(`${label} 必须是 JSON 对象或对象字符串，例如 '{"mu":0,"sigma":1}'`);
  }
  let parsed;
  try {
    parsed = JSON.parse(value);
  } catch {
    throw new Error(`${label} 不是合法 JSON：${value}（示例 '{"mu":0,"sigma":1}'）`);
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`${label} 必须是 JSON 对象，例如 '{"mu":0,"sigma":1}'`);
  }
  return parsed;
};

/**
 * 分布参数（params）解开成具名参数。
 *
 * 引擎的分布参数是具名 kwargs（mu/sigma/n/p/lam…），早期实现把 params 当字符串
 * 原样透传，导致「缺少必需参数 sigma」。显式给出的具名参数优先于 params 里的同名键。
 */
const withParams = (args) => {
  const { params, ...rest } = args ?? {};
  return { ...parseParamObject(params), ...rest };
};

/**
 * 矩阵/向量参数统一成引擎认识的形式。
 *
 * 引擎的 `_matrix` 把「一维字符串数组」当成**单行**处理（`['1,2','3,4']` 会解析成
 * 1×2 而报「非方阵」），所以这里把它拼成引擎明确支持的文本形式 `"1,2;3,4"`。
 */
const matrixArg = (value) => {
  if (value === undefined || value === null || typeof value === "string") return value;
  if (!Array.isArray(value) || value.length === 0) return value;
  if (value.every((row) => Array.isArray(row))) return value;
  return value.map((row) => String(row)).join(";");
};

const MSG_WHEN_NOT =
  "不需要调用的情况：可以口算或直接套用基础公式的简单算式；题目考查的是证明思路、定理适用条件或解题策略而不是具体计算；几何关系、积分区域、法向量方向这类必须靠数学分析判断的问题。";

/** 全部数学工具定义。 */
export const TOOL_SPECS = [
  // ---------------------------------------------------------------- 化简与恒等变形
  {
    name: "math_simplify",
    title: "化简与恒等变形",
    short: "化简表达式、展开、因式分解、通分、部分分式",
    description:
      "对符号表达式做恒等变形：simplify 化简、expand 展开、factor 因式分解、together 通分、apart 部分分式分解。\n" +
      "何时调用：表达式冗长需要整理；分式需要拆成部分分式以便积分；验证两个表达式是否恒等前先各自化简。\n" +
      "何时不要调用：只是抄写或代入数值。" +
      MSG_WHEN_NOT,
    params: {
      expr: S("待处理的数学表达式。支持 LaTeX 与自然写法，例如 '1/(x^2-1)'、'\\frac{x^2-1}{x-1}'、'sin(x)^2+cos(x)^2'。"),
      mode: S("变形方式：simplify（默认）| expand | factor | together | apart。", false),
      var: S("部分分式分解（mode=apart）的分解变量，例如 'x'。留空时由 SymPy 自行选择。", false),
    },
    op: "simplify",
    args: (a) => {
      const mode = (a.mode ?? "simplify").toLowerCase();
      if (mode === "simplify") return { op: "simplify", args: { expr: a.expr } };
      // together 不接受 var（sp.together 的第二个位置参数是 deep，不是变量），
      // 传下去会被引擎静默丢弃，所以这里只在 apart 分支带上 var。
      if (mode === "together") return { op: "together", args: { expr: a.expr } };
      if (mode === "apart") return { op: "apart", args: { expr: a.expr, var: a.var } };
      return { op: mode, args: { expr: a.expr } };
    },
  },
  {
    name: "math_solve",
    title: "解方程与方程组",
    short: "求方程、方程组的精确解（含无解与无穷多解判断）",
    description:
      "求解方程或方程组，返回精确解集（不默认转小数）。\n" +
      "何时调用：需要方程的准确解；解含参数需讨论；需要判断无解或无穷多解。\n" +
      "何时不要调用：需要的是解的几何意义或解的结构分析而非具体解；" +
      MSG_WHEN_NOT,
    params: {
      equations: A("一个或多个方程。可含等号，例如 ['x^2-3*x+2=0'] 或 ['x+y=3','x-y=1']；也可写成不含等号的表达式，此时默认等于 0。"),
      vars: A("未知数列表，例如 ['x', 'y']。留空时自动识别。"),
    },
    op: "solve",
    args: (a) => ({ op: "solve", args: { equations: a.equations, vars: a.vars } }),
  },

  // ---------------------------------------------------------------- 极限
  {
    name: "math_limit",
    title: "极限",
    short: "求极限，支持双侧与单侧、含极限点与符号参数",
    description:
      "计算函数极限，支持左右极限、无穷远点、单侧极限。\n" +
      "何时调用：需要准确的极限值；幂指函数、未定式需要精确判定；需要确认左右极限是否相等。\n" +
      "何时不要调用：需要的是判断极限存在性的论证思路或夹逼构造。" +
      MSG_WHEN_NOT,
    params: {
      expr: S("函数表达式，例如 'sin(x)/x'、'(1+1/n)^n'、'x*log(x)'。"),
      var: S("自变量，例如 'x' 或 'n'。留空时自动识别唯一自由变量。", false),
      point: S("趋近点，例如 '0'、'1'、'oo'、'-oo'、'无穷'。留空时按 '0' 处理（引擎与工具层的默认值一致）。", false),
      dir: S("趋近方向：'+'（右极限）、'-'（左极限），留空为双侧。", false),
    },
    op: "limit",
    args: (a) => ({ op: "limit", args: { expr: a.expr, var: a.var, point: a.point ?? "0", dir: a.dir } }),
  },

  // ---------------------------------------------------------------- 导数
  {
    name: "math_derivative",
    title: "导数与微分",
    short: "求导数（含高阶）、偏导数、隐函数与参数方程求导",
    description:
      "求导：一元导数（任意阶）、偏导数、方向导数、隐函数求导、参数方程求导、梯度、雅可比、海塞矩阵。\n" +
      "何时调用：求导结果需要精确表达式；高阶导、复合结构、参数方程容易出错；需要梯度和海塞矩阵用于极值判定。\n" +
      "何时不要调用：只需要说明求导法则（链式法则/隐函数定理）本身的思路。\n" +
      "工具内部会对结果做独立复核（再求导或数值抽样核对）。",
    params: {
      expr: S("函数表达式，例如 'x^x'、'ln(x+sqrt(1+x^2))'、'x^2*y+y^3'。"),
      var: S("求导变量，例如 'x'。留空时自动识别。", false),
      order: N("阶数，默认 1。例如 2 表示二阶导。"),
      variables: A("多元情形下的变量列表，如 ['x','y']（用于 gradient / jacobian / hessian / directional_derivative）。"),
      mode: S(
        "求导类型：derivative（默认，一元导数）| partial（偏导）| gradient | jacobian | hessian | directional | implicit | parametric。",
        false,
      ),
      point: A("求值点或方向向量（directional 模式的方向），如 ['1','1']。"),
      direction: A("方向导数的方向向量，如 ['1','1']。"),
      equation: S("implicit 模式的隐式方程，如 'x^2+y^2=1'。（仅 mode=implicit 需要）", false),
      x_expr: S("parametric 模式的 x(t)，如 'cos(t)'。（仅 mode=parametric 需要）", false),
      y_expr: S("parametric 模式的 y(t)，如 'sin(t)'。（仅 mode=parametric 需要）", false),
      param: S("parametric 模式的参数名，默认 't'。", false),
    },
    op: "diff",
    args: (a) => {
      const mode = (a.mode ?? "derivative").toLowerCase();
      if (mode === "partial") return { op: "partial", args: { expr: a.expr, var: a.var, order: a.order ?? 1 } };
      if (mode === "gradient") return { op: "gradient", args: { expr: a.expr, vars: a.variables } };
      if (mode === "jacobian") return { op: "jacobian", args: { exprs: a.expr, vars: a.variables } };
      if (mode === "hessian") return { op: "hessian", args: { expr: a.expr, vars: a.variables } };
      if (mode === "directional")
        return {
          op: "directional_derivative",
          args: { expr: a.expr, vars: a.variables, point: a.point, direction: a.direction ?? a.point },
        };
      if (mode === "implicit") return { op: "implicit_diff", args: { equation: a.equation, vars: a.variables, order: a.order ?? 1 } };
      if (mode === "parametric")
        return {
          op: "parametric_derivative",
          args: { x_expr: a.x_expr, y_expr: a.y_expr, param: a.param ?? "t", order: a.order ?? 1 },
        };
      return { op: "diff", args: { expr: a.expr, var: a.var, order: a.order ?? 1 } };
    },
  },

  // ---------------------------------------------------------------- 积分
  {
    name: "math_integrate",
    title: "积分",
    short: "不定积分与定积分（含反常积分），带独立复核",
    description:
      "计算不定积分与定积分。给出上下限时为定积分，可含无穷端点（反常积分）。\n" +
      "何时调用：任何需要精确积分结果的情形；被积函数含根式、有理分式、三角结构时尤其容易手算出错。\n" +
      "重要：像 ∫₀¹x²dx 这样的积分也应直接调用本工具取得精确结果，" +
      "不要为了它去写 Python 程序、建临时文件再执行——那就是被明确禁止的无必要程序编写。\n" +
      "何时不要调用：需要说明积分方法与区域划分思路；被积函数含抽象函数 f(x) 且无具体形式。" +
      MSG_WHEN_NOT,
    params: {
      expr: S("被积函数，例如 'x*exp(x^2)'、'1/(1+x^2)'、'sin(x)/x'。"),
      var: S("积分变量，默认 'x'。", false),
      lower: S("积分下限，例如 '0'、'-oo'、'无穷'。留空表示不定积分。", false),
      upper: S("积分上限，例如 '1'、'oo'、'无穷'。", false),
    },
    op: "integrate",
    args: (a) => ({ op: "integrate", args: { expr: a.expr, var: a.var, lower: a.lower, upper: a.upper } }),
  },

  // ---------------------------------------------------------------- 级数
  {
    name: "math_series",
    title: "级数与泰勒展开",
    short: "泰勒/麦克劳林展开、级数求和、敛散性相关的精确计算",
    description:
      "泰勒展开（可指定展开点与阶数）、数项级数求和、幂级数相关计算。\n" +
      "何时调用：需要函数的泰勒/麦克劳林展开式；需要数项级数的精确和；用泰勒展开求极限时先取得展开式。\n" +
      "何时不要调用：需要判断敛散性的论证（比较判别法、比值判别法的适用性分析）。\n" +
      "注意：泰勒展开是局部近似而不是恒等式，工具会在返回里注明这一点。",
    params: {
      expr: S("函数或被求和的通项，例如 'sin(x)'、'1/n^2'、'x^n/n'。"),
      var: S("变量或求和指标，例如 'x'、'n'。", false),
      point: S("展开点（仅 mode=taylor 使用），例如 '0' 表示麦克劳林展开。", false),
      order: N("展开阶数或求和阶数，默认 6。"),
      mode: S("taylor（默认，泰勒展开）| sum（级数求和）。", false),
      lower: S("求和下限（mode=sum），默认 '1'。", false),
      upper: S("求和上限（mode=sum），默认 'oo'。", false),
    },
    op: "series",
    args: (a) => {
      const mode = (a.mode ?? "taylor").toLowerCase();
      if (mode === "sum")
        return { op: "sum", args: { expr: a.expr, var: a.var, lower: a.lower ?? "1", upper: a.upper ?? "oo" } };
      return { op: "series", args: { expr: a.expr, var: a.var, point: a.point ?? "0", order: a.order ?? 6 } };
    },
  },

  // ---------------------------------------------------------------- 微分方程
  {
    name: "math_dsolve",
    title: "微分方程",
    short: "求解常微分方程并独立验证解",
    description:
      "求解常微分方程（可分离变量、一阶线性、齐次、二阶及高阶常系数等），可用 mode=check 独立验证给定的解。\n" +
      "何时调用：需要微分方程的通解/特解；需要把某个候选解代回方程验证。\n" +
      "导数写法：'y\\'' 或 'diff(y(x),x)' 都可以，例如 \"y' - 2*y = 0\"、\"diff(y(x),x,2) + y = 0\"。\n" +
      "何时不要调用：需要说明方程类型识别过程与求解方法的适用条件。",
    params: {
      equations: A("一个或多个微分方程，例如 [\"y' - 2*y = 0\"]、['diff(y(x),x,2)+y=0']。含初值条件时一同给出。"),
      vars: A("自变量列表，默认 ['x']。"),
      funcs: A("未知函数列表，例如 ['y']。"),
      candidate: S("mode=check 时待验证的解表达式。", false),
      mode: S("solve（默认，求解）| check（验证给定的解）。", false),
    },
    op: "dsolve",
    args: (a) => {
      const mode = (a.mode ?? "solve").toLowerCase();
      if (mode === "check")
        return { op: "ode_check", args: { equations: a.equations, candidate: a.candidate, vars: a.vars, funcs: a.funcs } };
      return { op: "dsolve", args: { equations: a.equations, vars: a.vars, funcs: a.funcs } };
    },
  },

  // ---------------------------------------------------------------- 线性代数
  {
    name: "math_matrix",
    title: "矩阵与线性代数",
    short: "行列式、逆、秩、特征值、对角化、线性方程组、二次型",
    description:
      "线性代数计算：行列式、逆矩阵、秩与行最简形、转置、矩阵乘法、迹、特征值与特征向量、" +
      "相似对角化、线性方程组（区分唯一解/无解/无穷多解）、二次型与正定性、矩阵幂。\n" +
      "何时调用：需要精确的行列式/逆/秩/特征值；需要判断向量组的线性相关性；需要二次型的标准形与正定性判定。\n" +
      "何时不要调用：需要的是线性相关性的证明或秩的推理而不是数值结果。\n" +
      "工具内部会做独立复核（逆矩阵验乘积为单位矩阵、特征值验特征方程、解代回原方程组等）。",
    params: {
      matrix: AS(
        "矩阵元素，按行给出，例如 ['1,2','3,4'] 或 ['[1,2]','[3,4]']；也接受单个字符串 '1,2;3,4'。",
      ),
      matrix_b: AS("第二个矩阵（乘法/加法）或右端项。"),
      expr: S("二次型表达式，例如 'x1^2+2*x1*x2+3*x2^2'（operation=quadratic 时使用）。", false),
      variables: A("二次型的变量名（operation=quadratic 时使用），例如 ['x1','x2']；也接受 'x1,x2'。留空时从 expr 自动识别。"),
      operation: S(
        "运算：analysis（默认，总览）| det | inv | rank | transpose | matmul | matadd | trace | rref | solve_linear | eigen | diagonalize | quadratic | power。",
        false,
      ),
      rhs: S("线性方程组右端项，例如 '1,2,3'（operation=solve_linear）。", false),
      order: N("矩阵幂的指数或特征值的重数参数。"),
    },
    op: "matrix_analysis",
    args: (a) => {
      const op = (a.operation ?? "analysis").toLowerCase();
      const matrix = matrixArg(a.matrix);
      const matrixB = matrixArg(a.matrix_b);
      const rhs = matrixArg(a.rhs ?? a.matrix_b);
      if (op === "power") return { op: "matrix_power", args: { matrix, order: a.order ?? 2 } };
      if (op === "quadratic")
        return { op: "quadratic_form", args: { expr: a.expr, matrix, variables: a.variables } };
      if (op === "solve_linear") return { op: "solve_linear", args: { matrix, rhs } };
      if (op === "transpose") return { op: "transpose", args: { matrix } };
      if (op === "matmul") return { op: "matmul", args: { matrix, matrix_b: matrixB } };
      if (op === "matadd") return { op: "matadd", args: { matrix, matrix_b: matrixB } };
      return { op, args: { matrix } };
    },
  },

  // ---------------------------------------------------------------- 概率统计
  {
    name: "math_probability",
    title: "概率统计",
    short: "分布、期望方差、协方差相关系数、贝叶斯、参数估计、假设检验",
    description:
      "概率统计计算：常见分布的概率密度/分布函数/分位数，期望、方差、协方差、相关系数，" +
      "联合分布与边缘分布、独立性判定，条件概率、全概率公式、贝叶斯公式，" +
      "参数估计（矩估计与极大似然）、区间估计、假设检验、中心极限定理近似。\n" +
      "何时调用：需要精确概率值（保留分数形式）；需要分布的期望方差；需要判定独立性；需要估计量或检验结论。\n" +
      "何时不要调用：需要的是概率模型的建立过程或检验方法选择的论证。\n" +
      "工具内部会做独立复核（密度归一化、方差的两条计算路径互印、检验统计量与拒绝域一致性等）。",
    params: {
      kind: S(
        "计算类型：bayes | total_probability | conditional | independent_check | inclusion_exclusion | interval | at | distribution | expectation | variance | covariance | correlation | joint_marginal | moments | custom_pmf | estimate | interval_estimate | hypothesis_test | sampling | clt。",
        false,
      ),
      dist: S("分布名称，例如 'normal'、'binomial'、'poisson'、'exponential'、'uniform'、'chi2'、't'、'f'。", false),
      prior: S("贝叶斯/全概率的先验概率，逗号分隔，例如 '0.3,0.5,0.2'。", false),
      likelihood: S("似然 P(B|A_i)，逗号分隔，例如 '0.9,0.8,0.5'。", false),
      joint: S("联合概率 P(AB)，或联合分布的说明。", false),
      pa: S("P(A) 的数值或表达式，例如 '1/3'。", false),
      pb: S("P(B) 的数值或表达式。", false),
      given: S("条件概率中的 P(B)。", false),
      expr: S("随机变量的函数，例如 'x^2'；或自定义分布律。", false),
      var: S("变量名，默认 'x'。", false),
      lower: S("区间下限。", false),
      upper: S("区间上限。", false),
      point: S("计算分布函数/分布律的取值点。", false),
      params: S("分布参数的 JSON 对象，例如 '{\"mu\":0,\"sigma\":1}' 或 '{\"n\":10,\"p\":\"1/3\"}'。", false),
      p: S("二项/几何分布的成功概率 p。", false),
      n: S("试验次数或样本量 n。", false),
      lam: S("泊松/指数分布的参数 λ。", false),
      mu: S("正态分布的均值 μ。", false),
      sigma: S("正态分布的标准差 σ。", false),
      samples: S("样本数据，逗号分隔，例如 '1.2,3.4,2.1'。", false),
      method: S("估计方法：moment（矩估计）| mle（极大似然）| both。", false),
      confidence: S("置信水平，例如 '0.95'。", false),
      alpha: S("显著性水平，例如 '0.05'。", false),
      side: S("检验类型：two（双侧）| left（左侧）| right（右侧）。", false),
      mu0: S("假设检验的原假设均值 μ₀。", false),
    },
    op: "probability",
    args: (a) => {
      const p = withParams(a);
      const kind = (p.kind ?? "distribution").toLowerCase();
      if (kind === "expectation") return { op: "expectation", args: { ...p } };
      if (kind === "variance") return { op: "variance", args: { ...p } };
      if (kind === "covariance") return { op: "covariance", args: { ...p } };
      if (kind === "correlation") return { op: "correlation", args: { ...p } };
      if (kind === "joint_marginal") return { op: "joint_marginal", args: { ...p } };
      if (kind === "moments") return { op: "distribution_moments", args: { ...p } };
      if (kind === "custom_pmf") return { op: "distribution_from_pmf", args: { ...p } };
      if (kind === "estimate") return { op: "point_estimate", args: { ...p } };
      if (kind === "interval_estimate") return { op: "interval_estimate", args: { ...p } };
      if (kind === "hypothesis_test") return { op: "hypothesis_test", args: { ...p } };
      if (kind === "sampling") return { op: "sampling_distribution", args: { ...p } };
      if (kind === "clt") return { op: "central_limit", args: { ...p } };
      if (kind === "distribution" || kind === "pdf" || kind === "cdf" || kind === "pmf")
        return { op: "distribution", args: { ...p, kind } };
      // bayes / total_probability / conditional / independent_check / inclusion_exclusion / interval / at
      return { op: "probability", args: { ...p, kind } };
    },
  },

  // ---------------------------------------------------------------- 数值计算
  {
    name: "math_numeric",
    title: "高精度数值计算",
    short: "高精度数值求值、数值积分、数值求根/优化、以及为符号结果做独立数值验证",
    description:
      "数值计算：表达式高精度求值、数值积分（含反常积分与瑕积分）、数值求根、数值极值、" +
      "数值极限、数值级数求和、数值导数。\n" +
      "何时调用：需要高精度近似值；符号方法求不出解析解时做数值兜底；" +
      "**对重要的符号结果做独立数值交叉验证**（这是规格要求的独立验证机制之一）。\n" +
      "何时不要调用：能求得精确解析结果时不要用数值替代；用户明确要求精确解时不要给近似值。\n" +
      "注意：数值结果是近似，不构成数学证明，工具会明确标注这一点。",
    params: {
      expr: S("表达式，例如 'sqrt(2)'、'exp(-x^2)'、'sin(x)/x'。"),
      mode: S(
        "数值计算类型：eval（默认，求值）| integrate | solve | optimize | derivative | limit | sum | compare。",
        false,
      ),
      var: S("变量，默认 'x'。", false),
      vars: A("多变量列表。"),
      values: S("代入值（JSON 对象字符串），例如 '{\"x\":1.5}' 或 '{\"x\":\"pi/4\"}'。", false),
      lower: S("积分下限。", false),
      upper: S("积分上限。", false),
      point: S("求导/求极限的点。", false),
      candidate: S("mode=compare 时待比较的另一个表达式。", false),
      digits: N("有效位数，默认 15（范围 1–200）。"),
      order: N("阶数。"),
      samples: N("抽样点个数。"),
    },
    op: "numeric",
    args: (a) => {
      const mode = (a.mode ?? "eval").toLowerCase();
      let values = a.values;
      if (typeof values === "string" && values.trim()) {
        try {
          values = JSON.parse(values);
        } catch {
          values = undefined;
        }
      }
      if (mode === "integrate")
        return { op: "numeric_integrate", args: { expr: a.expr, var: a.var, lower: a.lower, upper: a.upper, digits: a.digits } };
      if (mode === "solve") return { op: "numeric_solve", args: { equations: a.expr, vars: a.vars, guesses: values } };
      if (mode === "optimize") return { op: "numeric_optimize", args: { expr: a.expr, vars: a.vars } };
      if (mode === "derivative")
        return { op: "numeric_derivative", args: { expr: a.expr, var: a.var, point: a.point, order: a.order ?? 1, digits: a.digits } };
      if (mode === "limit")
        return { op: "numeric_limit", args: { expr: a.expr, var: a.var, point: a.point, digits: a.digits } };
      if (mode === "sum")
        return { op: "numeric_sum", args: { expr: a.expr, var: a.var, lower: a.lower, upper: a.upper, digits: a.digits } };
      if (mode === "compare")
        return { op: "compare_numeric", args: { expr: a.expr, candidate: a.candidate, vars: a.vars, samples: a.samples } };
      return { op: "numeric", args: { expr: a.expr, vars: values, digits: a.digits } };
    },
  },

  // ---------------------------------------------------------------- 曲线/曲面积分与三大公式
  {
    name: "math_vector_calculus",
    title: "曲线曲面积分与三大公式",
    short: "第一/二类曲线积分、第一/二类曲面积分、格林公式、高斯公式、斯托克斯公式",
    description:
      "曲线积分与曲面积分计算：第一类（弧长/面积）与第二类（坐标/通量）积分，以及格林、高斯、斯托克斯三大公式。\n" +
      "何时调用：曲线或曲面**已经参数化**、积分区域**已经确定**，需要把积分精确算出来时。" +
      "考研中格林/高斯/斯托克斯公式的选择（曲面是否封闭、投影区域是否简单、被积函数是否轮换对称、补面还是直接投影）" +
      "属于数学分析判断，必须先由你判断并说明理由，再用本工具计算已选定的那个积分。\n" +
      "何时不要调用：还没决定用投影法还是高斯公式、需要讲清为什么补面或为什么要轮换对称——" +
      "这些必须保留在推理里，不能交给计算器代替。" +
      "工具内部会对结果做独立复核（同参数化的高精度数值积分、交换积分次序、按定义重算旋度/散度）。",
    params: {
      mode: S(
        "计算类型：curve（第一/二类曲线积分，默认）| surface（第一/二类曲面积分）| green（格林公式）| gauss（高斯公式）| stokes（斯托克斯公式）。",
        false,
      ),
      kind: S(
        "积分类型（curve / surface 模式）：first（默认，第一类弧长/面积）| second（第二类坐标/通量）。第二类积分与曲线方向、曲面定向有关，反向取负号。",
        false,
      ),
      expr: S("被积函数或向量场 P 分量。curve 模式：被积函数或 P；surface 模式：被积函数或 P；green：P；gauss/stokes：P。"),
      expr2: S("Q 分量（第二类积分、green、gauss、stokes 需要）。", false),
      expr3: S("R 分量（空间曲线第二类积分、gauss、stokes 需要；gauss/stokes 也可用 r 参数）。", false),
      r: S("R 分量（gauss/stokes 的第三个分量，等价于 expr3）。避开 r_component 是因为它与参数名 r_component 重名。", false),
      x_expr: S("参数方程 x(t) 或 x(u,v)，例如 'cos(t)'、'u*cos(v)'。（curve / surface / stokes 需要；green、gauss 不需要）", false),
      y_expr: S("参数方程 y(t) 或 y(u,v)，例如 'sin(t)'、'u*sin(v)'。", false),
      z_expr: S("参数方程 z(t) 或 z(u,v)，例如 'cos(u)'；平面区域可写 '0'。", false),
      var: S("曲线参数名，默认 't'。", false),
      lower: S("曲线参数的积分下限，默认 '0'。（仅 curve 模式）", false),
      upper: S("曲线参数的积分上限，默认 '2*pi'。（仅 curve 模式）", false),
      u_var: S("曲面参数名 u，默认 'u'。", false),
      v_var: S("曲面参数名 v，默认 'v'。", false),
      u_lower: S("曲面参数 u 的下限，默认 '0'。", false),
      u_upper: S("曲面参数 u 的上限，默认 'pi'。", false),
      v_lower: S("曲面参数 v 的下限，默认 '0'。", false),
      v_upper: S("曲面参数 v 的上限，默认 '2*pi'。", false),
      x_lower: S("区域 x 的下限（green / gauss），例如 '-1'。", false),
      x_upper: S("区域 x 的上限（green / gauss），例如 '1'。", false),
      y_lower_expr: S("区域 y 的下边界（green / gauss），可含 x，例如 '-sqrt(1-x^2)'。", false),
      y_upper_expr: S("区域 y 的上边界（green / gauss），可含 x，例如 'sqrt(1-x^2)'。", false),
      z_lower_expr: S("区域 z 的下边界（gauss），可含 x,y，例如 '-sqrt(1-x^2-y^2)'。", false),
      z_upper_expr: S("区域 z 的上边界（gauss），可含 x,y，例如 'sqrt(1-x^2-y^2)'。", false),
      y_lower: S("区域 y 的常数下界（green / gauss 矩形区域）。", false),
      y_upper: S("区域 y 的常数上界（green / gauss 矩形区域）。", false),
      z_lower: S("区域 z 的常数下界（gauss 长方体区域）。", false),
      z_upper: S("区域 z 的常数上界（gauss 长方体区域）。", false),
      orientation: S("曲线定向（green）：positive（默认，逆时针/正向）| negative（顺时针/负向，结果取负）。", false),
    },
    op: "curve_integral",
    args: (a) => {
      const mode = (a.mode ?? "curve").toLowerCase();
      const third = a.expr3 ?? a.r;
      if (mode === "green")
        return {
          op: "green",
          args: {
            p: a.expr,
            q: a.expr2,
            x_lower: a.x_lower,
            x_upper: a.x_upper,
            y_lower: a.y_lower,
            y_upper: a.y_upper,
            y_lower_expr: a.y_lower_expr,
            y_upper_expr: a.y_upper_expr,
            orientation: a.orientation,
          },
        };
      if (mode === "gauss")
        return {
          op: "gauss",
          args: {
            p: a.expr,
            q: a.expr2,
            r_component: third,
            x_lower: a.x_lower,
            x_upper: a.x_upper,
            y_lower: a.y_lower,
            y_upper: a.y_upper,
            z_lower: a.z_lower,
            z_upper: a.z_upper,
            y_lower_expr: a.y_lower_expr,
            y_upper_expr: a.y_upper_expr,
            z_lower_expr: a.z_lower_expr,
            z_upper_expr: a.z_upper_expr,
          },
        };
      if (mode === "stokes")
        return {
          op: "stokes",
          args: {
            p: a.expr,
            q: a.expr2,
            r_component: third,
            x_expr: a.x_expr,
            y_expr: a.y_expr,
            z_expr: a.z_expr,
            u_var: a.u_var,
            v_var: a.v_var,
            u_lower: a.u_lower,
            u_upper: a.u_upper,
            v_lower: a.v_lower,
            v_upper: a.v_upper,
          },
        };
      if (mode === "surface")
        return {
          op: "surface_integral",
          args: {
            expr: a.expr,
            expr2: a.expr2,
            expr3: third,
            x_expr: a.x_expr,
            y_expr: a.y_expr,
            z_expr: a.z_expr,
            u_var: a.u_var,
            v_var: a.v_var,
            u_lower: a.u_lower,
            u_upper: a.u_upper,
            v_lower: a.v_lower,
            v_upper: a.v_upper,
            kind: a.kind,
          },
        };
      return {
        op: "curve_integral",
        args: {
          expr: a.expr,
          expr2: a.expr2,
          expr3: third,
          x_expr: a.x_expr,
          y_expr: a.y_expr,
          z_expr: a.z_expr,
          var: a.var,
          lower: a.lower,
          upper: a.upper,
          kind: a.kind,
        },
      };
    },
  },

  // ---------------------------------------------------------------- 验证
  {
    name: "math_verify",
    title: "独立验证结果",
    short: "对已有结果做独立复核：判断对错、定位错误、给出无法验证的原因",
    description:
      "独立验证机制：把待验证结果与原始表达式/方程放在一起复核，输出是否通过、用了什么验证方式、" +
      "关键证据（残差、差式化简结果、抽样偏差）、以及**未能纳入验证的条件**。\n" +
      "何时调用：学生给出答案需要判对错；自己算出一个结果需要确认；怀疑某一步推导有误；" +
      "重要结论在给出最终答案前需要独立复核。\n" +
      "何时不要调用：结果尚不存在时（应先计算）；纯粹要求讲解思路时。\n" +
      "硬性保证：证据不足时不会返回通过；无法验证时会明确说明原因，绝不为了给出「通过」而放宽数学条件。",
    params: {
      expr: S(
        "原始表达式 / 被积函数 / 原方程，视 kind 而定。例如 kind=derivative 时填 'x^2'。" +
          "矩阵类验证（matrix_inverse/determinant/eigen）不需要 expr，给出 matrix 即可。",
        false,
      ),
      candidate: S("待验证的结果，例如 '2*x'（矩阵类验证时填数值或矩阵）。"),
      kind: S(
        "验证类型：general（默认）| derivative | integral（不定积分）| definite_integral | limit | equation_solution | identity | simplify | factor | expand | matrix_inverse | determinant | eigen | probability | expectation | variance | normalization。",
        false,
      ),
      var: S("变量，例如 'x'。", false),
      vars: A("变量列表。", false),
      equations: A("原方程（kind=equation_solution 或方程组时使用）。", false),
      point: S("求值点或极限点。", false),
      lower: S("定积分下限。", false),
      upper: S("定积分上限。", false),
      matrix: AS("原矩阵（矩阵类验证），例如 ['1,2','3,4'] 或 '1,2;3,4'。"),
      matrix_b: AS("第二个矩阵，例如 ['1,0','0,1'] 或 '1,0;0,1'。"),
      conditions: S("用户声明的附加条件与定义域，例如 'x>0'。工具会说明这些条件是否真正被纳入验证。", false),
    },
    op: "verify",
    args: (a) => ({
      op: "verify",
      args: { ...a, matrix: matrixArg(a.matrix), matrix_b: matrixArg(a.matrix_b) },
    }),
  },

  // ---------------------------------------------------------------- 定义域与函数分析
  {
    name: "math_analysis",
    title: "函数分析与定义域",
    short: "定义域、渐近线、单调性凹凸性极值拐点、不等式解集、多元极值与条件极值",
    description:
      "数学分析类计算：自然定义域、水平/垂直/斜渐近线、单调区间、极值、凹凸区间、拐点、" +
      "不等式解集、多元函数无条件极值（海塞矩阵判别）、拉格朗日乘数法条件极值、" +
      "多元泰勒展开、中值定理中间点。\n" +
      "何时调用：需要准确的单调区间/极值/拐点结果；需要定义域与其来源（分母、根式、对数、反三角）；" +
      "需要条件极值的驻点与乘子。\n" +
      "何时不要调用：需要作图思路或极值判别法的原理说明。" +
      MSG_WHEN_NOT,
    params: {
      expr: S("函数表达式，例如 'x^3-3*x'、'sqrt(x-1)/(x-2)'、'(x^2+1)/x'。"),
      var: S("自变量，默认 'x'。", false),
      variables: A("多元情形变量列表，例如 ['x','y']。"),
      mode: S(
        "分析类型：all（默认，完整函数分析）| domain | asymptote | extremum | lagrange | taylor | inequality | mean_value。",
        false,
      ),
      constraints: A("约束条件（mode=lagrange），例如 ['x+y=1']。"),
      point: A("展开点（mode=taylor）。"),
      order: N("展开阶数。"),
      inequality: S("不等式（mode=inequality），例如 'x^2-3*x+2<0'。（仅 mode=inequality 需要）", false),
    },
    op: "function_analysis",
    args: (a) => {
      const mode = (a.mode ?? "all").toLowerCase();
      if (mode === "domain") return { op: "domain", args: { expr: a.expr, var: a.var } };
      if (mode === "asymptote") return { op: "asymptote", args: { expr: a.expr, var: a.var ?? "x" } };
      if (mode === "extremum") return { op: "multivar_extremum", args: { expr: a.expr, vars: a.variables } };
      if (mode === "lagrange")
        return { op: "lagrange", args: { expr: a.expr, constraints: a.constraints, vars: a.variables } };
      if (mode === "taylor")
        return { op: "taylor_multivar", args: { expr: a.expr, vars: a.variables, point: a.point, order: a.order ?? 2 } };
      if (mode === "inequality")
        return { op: "inequality", args: { expr: a.inequality ?? a.expr, var: a.var } };
      if (mode === "mean_value") return { op: "mean_value_point", args: { expr: a.expr, var: a.var } };
      return { op: "function_analysis", args: { expr: a.expr, var: a.var ?? "x" } };
    },
  },

  // ---------------------------------------------------------------- 引擎状态
  {
    name: "math_status",
    title: "数学引擎状态",
    short: "查看计算引擎状态、缓存命中与可用算子",
    description:
      "查看数学计算引擎的运行状态：可用算子清单、缓存命中率、已记录的成功与失败计算。" +
      "用于诊断「为什么某个计算没有执行」或确认缓存是否生效。日常解题不需要调用。",
    params: {
      action: S("status（默认）| ops（算子清单）| cache_stats | cache_clear。", false),
    },
    op: "@cache_stats",
    args: (a) => ({ op: "@cache_stats", args: {}, control: (a.action ?? "status").toLowerCase() }),
    control: true,
  },
];

export { MSG_WHEN_NOT };
