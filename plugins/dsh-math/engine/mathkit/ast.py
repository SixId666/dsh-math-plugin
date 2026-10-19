"""表达式解析、变量提取与 LaTeX 输出。

这是整个引擎的「可信底座」：所有模块都通过这里把用户字符串变成 SymPy 对象，
从而保证解析行为、错误信息、LaTeX 渲染风格完全一致。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, Sequence

import sympy as sp
from sympy.parsing.sympy_parser import (
    _token_splittable,
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------

_TRANSFORMS = standard_transformations + (convert_xor, implicit_multiplication_application)

# 常见函数的中文/别名写法 -> sympy 名称
_FUNC_ALIASES = {
    "lg": "log",
    "ln": "log",
    "arcsin": "asin",
    "arccos": "acos",
    "arctan": "atan",
    "arccot": "acot",
    "arcsec": "asec",
    "arccsc": "acsc",
    "tg": "tan",
    "ctg": "cot",
    "sh": "sinh",
    "ch": "cosh",
    "th": "tanh",
    "lim": "limit",
}

# 允许出现的保留名（不当作自由变量）
#
# 除常量与初等函数外，**必须**把 SymPy 的运算函数名也列进来（diff/integrate/limit/solve/…）：
# get_symbols 会跳过 _RESERVED 里的名字，正是这一步保证「用户不写变量名时不会凭空造出
# 一个叫 diff 的 Symbol」。一旦 Symbol('diff') 进了 local_dict，它会盖住 global_dict 里的
# sp.diff，`diff(y(x), x)` 就被解析成 Symbol('diff')*Symbol('y')*… → TypeError:
# can't multiply sequence by non-int of type 'Symbol'（真实踩过的坑）。
_RESERVED = {
    "pi", "E", "I", "oo", "zoo", "nan", "true", "false",
    "sin", "cos", "tan", "cot", "sec", "csc",
    "asin", "acos", "atan", "acot", "asec", "acsc",
    "sinh", "cosh", "tanh", "coth", "sech", "csch",
    "asinh", "acosh", "atanh", "acoth",
    "exp", "log", "sqrt", "cbrt", "Abs", "abs", "sign",
    "floor", "ceiling", "factorial", "gamma", "beta", "max", "min",
    "erf", "erfc", "erfi", "Max", "Min", "Rational", "re", "im", "conjugate",
    "Piecewise", "DiracDelta", "Heaviside", "Sum", "Product", "Integral",
    "Derivative", "Limit", "root", "binomial", "polygamma", "zeta",
    "airyai", "besselj", "bessely", "besseli", "besselk",
    "diff", "integrate", "limit", "solve", "dsolve", "simplify", "expand",
    "factor", "series", "subs", "Eq", "Ne", "Ge", "Le", "Gt", "Lt",
    "Matrix", "det", "inv", "transpose",
    "Gamma", "Beta", "Zeta", "digamma", "Ei", "Si", "Ci", "erfinv",
    "LambertW", "harmonic",
}

# SymPy 顶层**没有**导出的大写函数名（sp.Gamma / sp.Beta / sp.Zeta 都不存在，
# 只有小写的 sp.gamma / sp.beta / sp.zeta），需要手工补进命名空间。
# ``abs`` / ``max`` / ``min`` 是 Python 内建名，``sp.abs`` / ``sp.max`` / ``sp.min`` 也**不存在**
# （只有 sp.Abs / sp.Max / sp.Min）。不显式补进来的话，``abs(x)`` 会被
# implicit_multiplication 的 split_symbols 拆成 ``a*b*s*x`` —— 一个**静默的错误答案**
# （绝对值在考研题里极常见：|x|、|x-1|、分段函数的绝对值定义）。
_EXTRA_NAMESPACE = {
    "Gamma": sp.gamma, "Beta": sp.beta, "Zeta": sp.zeta,
    "abs": sp.Abs, "max": sp.Max, "min": sp.Min,
}

_ABS_PAIR_RE = re.compile(r"\|([^|]+)\|")

# 与希腊字母同名的 SymPy 函数。考研里 `beta`/`gamma`/`zeta` 更常是**变量**（β、γ、ζ），
# 而不是 Beta/Gamma/Zeta 特殊函数，所以：不写括号时一律当变量，写了 `beta(...)`
# 之类才当函数用（见 _text_symbols / _called_like_function）。
_FUNCTION_NAME_COLLISIONS = {
    "beta", "gamma", "zeta", "Gamma", "Beta", "Zeta",
    "digamma", "harmonic",
}

# 默认取实数的常用符号名（与 default_symbols 对齐）
_REAL_NAMES = ("x", "t", "y", "z", "u", "v", "s")

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def _called_like_function(text: str, end: int) -> bool:
    """``end`` 处之后的第一个非空字符是左括号吗（即这个名字被当函数调用）？"""
    rest = text[end:].lstrip()
    return rest.startswith("(")


def _is_atomic_name(name: str) -> bool:
    """这个名字应当作「一个不可再分的符号」而不是若干符号的连写吗？

    背景（真实踩过的坑）：``implicit_multiplication_application`` 会把符号名拆开
    （``xyz`` → ``x*y*z``），实现方式是**逐字符**看能否在命名空间里找到，找不到的
    字符一律变成单字母 Symbol、数字变成 Number。于是：

    * ``C1`` 被拆成 ``C*1`` → 只剩 ``C``（积分常数被静默吃掉）
    * ``x1`` 被拆成 ``x*1`` → 只剩 ``x``
    * ``a1 + 2*a2`` 被算成 ``5*a``

    规则：

    * 含数字或下划线（``C1``、``x_1``、``a12``）→ 整名
    * 希腊字母名（``alpha``、``theta``、``Gamma``…）→ 整名
      （sympy 的 ``_token_splittable`` 只认**小写**希腊名，大写要自己补）
    * 含非字母字符 → 整名（保守处理）
    * 其余纯字母名（``xy``、``ab``）→ 允许拆分，保持数学直觉
    """
    if any(ch.isdigit() or ch == "_" for ch in name):
        return True
    if not name.isalpha():
        return True
    if not _token_splittable(name):
        return True
    try:
        unicodedata.lookup("GREEK CAPITAL LETTER " + name)
    except KeyError:
        return False
    return True


def _text_symbols(text: str, existing: dict[str, Any]) -> dict[str, sp.Symbol]:
    """为文本里出现的「整名符号」补 Symbol，塞进 parse_expr 的 local_dict。

    只有进了 ``local_dict`` 的名字才不会被隐式拆分吃掉：sympy 的 ``_split_symbols``
    只处理 ``Symbol(...)``/``Function(...)`` 调用里的名字，而 ``auto_symbol`` 对
    ``local_dict`` 里已有的名字是原样输出、不包 ``Symbol(...)``。

    这里还负责把「与函数同名的希腊字母」在**未被调用**时改回变量：
    ``alpha + beta`` 里的 ``beta`` 必须是变量 β，而 ``beta(2,3)`` 才是 Beta 函数。
    """
    out: dict[str, sp.Symbol] = {}
    for match in _IDENT_RE.finditer(text):
        name = match.group(0)
        if name in out:
            continue
        # 被当函数调用（后面紧跟括号）的名字**不能**塞进 local_dict：一旦进了，
        # auto_symbol 会把它当已知符号原样输出，implicit_application 就不再把它包成
        # Function，于是 ``f(x)`` 被静默解析成 ``f*x``（错的答案，不是报错）。
        if _called_like_function(text, match.end()):
            continue
        if name in _FUNCTION_NAME_COLLISIONS:
            out[name] = sp.Symbol(name)
            continue
        if name in existing or name in _RESERVED:
            continue
        if not _is_atomic_name(name):
            continue
        out[name] = sp.Symbol(name, real=True) if name in _REAL_NAMES else sp.Symbol(name)
    return out


def _split_top_level_operator(text: str, operators: Sequence[str]) -> tuple[str, str, str] | None:
    """按**括号深度 0** 找到最外层的第一个关系运算符，返回 ``(左, 运算符, 右)``。"""
    depth = 0
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif depth == 0:
            for operator in operators:
                if text.startswith(operator, index):
                    left = text[:index].strip()
                    right = text[index + len(operator):].strip()
                    if left and right:
                        return left, operator, right
        index += 1
    return None


_RELATION_BUILDERS = {"==": "Eq", "!=": "Ne", ">=": "Ge", "<=": "Le", "=": "Eq"}


def _rewrite_relations(text: str) -> str:
    """把 ``x == 2`` / ``a != b`` / ``x^2-3*x+2=0`` 改写成 ``Eq(...)`` / ``Ne(...)``。

    为什么必须改写：``parse_expr(..., evaluate=True)`` 下的 ``==``/``!=`` 是 **Python**
    的比较，走的是结构相等判断，于是 ``A.parse("x == 2")`` 会静默返回 ``False``、
    ``A.parse("x != 0")`` 返回 ``True`` —— 一个不会报错的错误答案。裸 ``=`` 更是直接
    语法错误。``>=``/``<=`` 也一并改写：sympy 对能自行判定的式子（如 ``x^2+1 >= 1``）
    会直接坍缩成 ``True``，丢掉原式。
    """
    found = _split_top_level_operator(text, ("==", "!=", ">=", "<=", "="))
    if found is None:
        return text
    left, operator, right = found
    builder = _RELATION_BUILDERS[operator]
    return f"{builder}({_rewrite_relations(left)}, {_rewrite_relations(right)})"


def _unknown_functions(text: str, existing: dict[str, Any],
                       default_symbols: Sequence[str]) -> dict[str, Any]:
    """把 ``f(x)`` / ``g(x,y)`` / ``u(x,t)`` 里的未知名字注册成 SymPy 的未定义函数。

    为什么必须显式做：``implicit_multiplication`` 会把 ``f(x)`` 当成 ``f * x`` ——
    一个**静默的错误答案**（例如 ``integrate f(x) dx`` 会算成 ``f*x**2/2``），
    而 ``f(x,y)`` 直接报 ``can't multiply sequence by non-int of type 'Symbol'``。
    考研题里抽象函数 ``f(x)``、``g(x,y)``、``u(x,t)`` 很常见，必须按函数解析。

    只对「不像变量名」的字母名生效：``default_symbols`` 里的名字（``a``、``b``、``c``…）
    后面跟括号仍按乘法处理（``a(x+1)`` = ``a*(x+1)``），避免把系数当成函数。
    """
    out: dict[str, Any] = {}
    for match in _IDENT_RE.finditer(text):
        name = match.group(0)
        if name in out or name in existing or name in _RESERVED:
            continue
        if name in _FUNCTION_NAME_COLLISIONS:
            continue
        if not name.isalpha() or name in default_symbols:
            continue
        if not _called_like_function(text, match.end()):
            continue
        out[name] = sp.Function(name)
    return out


def normalize(text: str) -> str:
    if text is None:
        raise ValueError("表达式为空")
    s = str(text).strip()
    if not s:
        raise ValueError("表达式为空")
    # 去掉包裹的 $...$ 或 $$...$$
    s = re.sub(r"^\$\$?(.*?)\$\$?$", r"\1", s, flags=re.S)
    # LaTeX 常用记号
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    s = s.replace("\\pi", "pi").replace("\\infty", "oo").replace("\\infty", "oo")
    s = s.replace("\\ln", "log").replace("\\log", "log").replace("\\exp", "exp")
    s = s.replace("\\sin", "sin").replace("\\cos", "cos").replace("\\tan", "tan")
    s = s.replace("\\cot", "cot").replace("\\sec", "sec").replace("\\csc", "csc")
    s = s.replace("\\arcsin", "asin").replace("\\arccos", "acos")
    s = s.replace("\\arctan", "atan")
    s = s.replace("\\sqrt", "sqrt")
    s = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\dfrac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = s.replace("\\,", "").replace("\\;", "").replace("\\!", "").replace("\\ ", " ")
    # 幂与下标
    s = s.replace("^", "**") if "**" not in s else s
    s = s.replace("{", "(").replace("}", ")")
    # 中文标点
    s = s.replace("（", "(").replace("）", ")").replace("，", ",")
    # 绝对值竖线：|x| → Abs(x)、|x-1|+1 → Abs(x-1)+1
    # （sympy 不认竖线，原样送进去是语法错误；不处理等于直接放弃绝对值这一类题）
    if "|" in s:
        s = _ABS_PAIR_RE.sub(r"Abs(\1)", s)
    # 函数别名（整词替换）
    for alias, target in _FUNC_ALIASES.items():
        s = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(alias)}(?![A-Za-z0-9_])", target, s)
    # 一元负号规范化由 implicit_multiplication 处理
    return s


# 只允许白名单内的名字进入 eval 命名空间
def _build_namespace(symbols: dict[str, sp.Basic]) -> dict[str, Any]:
    ns: dict[str, Any] = {}
    for key in _RESERVED:
        obj = getattr(sp, key, None)
        if obj is None:
            continue
        # sympy 里 abs 是内置名；只接受 sympy 对象
        if isinstance(obj, (sp.Basic, sp.Function, type)) or callable(obj):
            ns[key] = obj
    ns.update(_EXTRA_NAMESPACE)
    ns.update(symbols)
    return ns


def get_symbols(names: Iterable[str] | None = None, *, default: Sequence[str] = ()) -> dict[str, sp.Symbol]:
    """构造符号表。未指定时给出考研数学常用符号。"""
    resolved: list[str] = []
    if names:
        for raw in names:
            raw = str(raw).strip()
            if raw:
                resolved.append(raw)
    elif default:
        resolved.extend(default)
    out: dict[str, sp.Symbol] = {}
    for name in resolved:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
            raise ValueError(f"非法变量名: {name!r}")
        if name in _RESERVED and name not in _FUNCTION_NAME_COLLISIONS:
            continue
        out[name] = sp.Symbol(name, real=True) if name in _REAL_NAMES else sp.Symbol(name)
    return out


def make_symbols(names: Iterable[str] | None = None, default: Sequence[str] = ()) -> dict[str, sp.Symbol]:
    """同 get_symbols，但只返回符号对象本身（不含保留名）。"""
    return get_symbols(names, default=default)


def free_symbols(expr: sp.Basic) -> list[str]:
    """返回自由变量名（字典序，排除常量）。"""
    return sorted({str(s) for s in expr.free_symbols}, key=lambda n: (len(n), n))


def parse(
    text: str,
    symbols: Iterable[str] | None = None,
    *,
    default_symbols: Sequence[str] = ("x", "y", "z", "t", "n", "a", "b", "c", "k"),
    extra: dict[str, Any] | None = None,
) -> sp.Basic:
    """把字符串解析为 SymPy 表达式。

    会抛出 ValueError（可读原因），由上层转成结构化的错误结果。
    """
    normalized = normalize(text)
    normalized = _rewrite_relations(normalized)
    sym_map = get_symbols(symbols, default=default_symbols)
    ns = _build_namespace(sym_map)
    # 先把文本里的「整名符号」（C1、x_1、alpha…）锁进 local_dict，避免被
    # implicit_multiplication_application 的符号拆分吃掉（见 _is_atomic_name）。
    ns.update(_text_symbols(normalized, ns))
    # 再把 f(x)/g(x,y) 这类未知函数注册成 Function，否则会被当成 f*x。
    ns.update(_unknown_functions(normalized, ns, default_symbols))
    if extra:
        ns.update(extra)
    try:
        # global_dict 必须是独立的 sympy 命名空间：parse_expr 在 auto_symbol 阶段
        # 会往 local_dict 写入 Symbol('Integer') 之类的临时名，若 local/global 是
        # 同一个 dict，就会污染 Integer/Symbol 等构造名，导致 NameError。
        expr = parse_expr(
            normalized,
            local_dict=dict(ns),
            global_dict=dict(sp.__dict__),
            transformations=_TRANSFORMS,
            evaluate=True,
        )
    except SyntaxError as exc:  # 语法层面的错误
        raise ValueError(f"表达式语法错误（已规范化为 {normalized!r}）: {exc.msg}") from exc
    except Exception as exc:  # noqa: BLE001 - 解析失败原因必须回传模型
        raise ValueError(f"表达式无法解析（已规范化为 {normalized!r}）: {exc}") from exc
    if expr is None:
        raise ValueError("表达式解析结果为空")
    if not isinstance(expr, sp.Basic):
        # parse_expr 可能返回 python 数值
        expr = sp.sympify(expr)
    return expr


def parse_all(
    texts: Iterable[str],
    symbols: Iterable[str] | None = None,
    *,
    default_symbols: Sequence[str] = ("x", "y", "z", "t", "n", "a", "b", "c", "k"),
    extra: dict[str, Any] | None = None,
) -> list[sp.Basic]:
    return [
        parse(t, symbols, default_symbols=default_symbols, extra=extra)
        for t in texts
    ]


def parse_matrix(
    rows: Sequence[Sequence[str]] | None = None,
    *,
    text: str | None = None,
    symbols: Iterable[str] | None = None,
    default_symbols: Sequence[str] = ("x", "y", "z", "t", "n", "a", "b", "c", "k"),
) -> sp.Matrix:
    """支持两种输入：二维字符串数组，或 ``1,2;3,4`` / ``[[1,2],[3,4]]`` 文本。"""
    if rows is not None:
        data = [
            [parse(cell, symbols, default_symbols=default_symbols) for cell in row]
            for row in rows
        ]
        return sp.Matrix(data)
    if text is None:
        raise ValueError("矩阵为空")
    raw = str(text).strip()
    if not raw:
        raise ValueError("矩阵为空")
    raw = raw.strip("[]{} ")
    # 统一分隔符
    raw = raw.replace("],[", ";").replace("][", ";").replace("], [", ";")
    rows_text = [r for r in re.split(r"[;\n]|\\\\(?=\s)", raw) if r.strip()]
    data = []
    for row in rows_text:
        cells = [c for c in re.split(r"[, ]+", row.strip().strip("[]{}")) if c.strip()]
        if not cells:
            continue
        data.append([parse(c, symbols, default_symbols=default_symbols) for c in cells])
    if not data:
        raise ValueError("矩阵为空")
    return sp.Matrix(data)


# ---------------------------------------------------------------------------
# LaTeX 输出
# ---------------------------------------------------------------------------

def to_latex(expr: Any) -> str:
    if expr is None:
        return ""
    try:
        if isinstance(expr, sp.MatrixBase):
            return sp.latex(expr)
        return sp.latex(sp.sympify(expr))
    except Exception:  # noqa: BLE001
        return str(expr)


def to_text(expr: Any) -> str:
    """稳定的纯文本形式，用于缓存键与模型复核。"""
    if expr is None:
        return ""
    try:
        if isinstance(expr, sp.MatrixBase):
            return str(expr.tolist())
        return sp.sstr(sp.sympify(expr))
    except Exception:  # noqa: BLE001
        return str(expr)


def canonical_key(expr: Any) -> str:
    """表达式的规范化形式：用于缓存/查重，同值不同形应得到同一 key。"""
    if expr is None:
        return ""
    try:
        e = sp.sympify(expr)
    except Exception:  # noqa: BLE001
        return str(expr)
    if isinstance(e, sp.MatrixBase):
        return "[" + ",".join(canonical_key(x) for x in e) + "]"
    try:
        e = sp.expand(sp.simplify(e))
    except Exception:  # noqa: BLE001
        pass
    try:
        e = sp.nsimplify(e)
    except Exception:  # noqa: BLE001
        pass
    return sp.srepr(e)


def is_unevaluated(expr: Any) -> bool:
    """判断结果是否是「未求值」—— 例如 integrate 原样返回。

    这类结果不能被当作计算成功（用户规格 5.3 明确要求）。
    """
    if expr is None:
        return True
    e = expr
    # 注意：**不要**把 sp.Solve 放进来 —— SymPy 1.13.3 顶层没有 Solve 这个名字
    # （sp.Solve 抛 AttributeError: module 'sympy' has no attribute 'Solve'），
    # 会让本函数对任何输入都崩掉。solve() 的"未求值"结果是普通的 Eq/list，不靠类型识别。
    unevaluated = (sp.Integral, sp.Derivative, sp.Limit, sp.Sum, sp.Product)
    if isinstance(e, unevaluated):
        return True
    if isinstance(e, sp.Basic):
        for node in sp.preorder_traversal(e):
            if isinstance(node, unevaluated):
                return True
    return False
