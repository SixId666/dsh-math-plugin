"""规格 12.4 八个专项测试 —— 重点检验「异常推理与重复计算」。

这八个测试检验的不是数学正确性（那在 _test_acceptance.py 里），而是**行为**：

  测试一 简单积分：是否直接调用工具而不是自写 Python 程序
  测试二 复杂积分：结果是否正确、是否重复调用
  测试三 无法求解的积分：工具失败后是否停止重复尝试并合理解释限制
  测试四 连续追问：第二轮能否复用第一轮结果
  测试五 已有答案验证：能否定位推导错误而不是重新无目的求解
  测试六 思路分析：是否保留必要数学分析而不是全交给计算器
  测试七 结果交叉验证：重要结果是否真正经过独立验证
  测试八 相似表达式：缓存能否正确复用，同时避免不同条件混用

其中「是否自写程序」「是否保留数学分析」「是否定位错误」属于 DSH 侧的行为，
无法由 Python 单测断言，因此这些测试以**结构化证据**的形式输出，供人工与
_test_special_node.mjs（走真实插件调用路径）共同判定。
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine"))

from mathkit import engine as E  # noqa: E402
from mathkit import ops as _ops  # noqa: E402,F401  触发全部算子注册（与 worker.py 一致）
from worker import Cache  # noqa: E402  (worker.py 自行插入 sys.path；Cache 定义在 worker.py 里)
import sympy as sp  # noqa: E402  仅用于测试里做「候选原函数是否正确」的独立比对

print(f"已注册算子 {len(E.known_operations())} 个；未加载模块：{_ops.failed or '无'}")

RESULTS = []


def record(test_no, name, passed, evidence, extra=None):
    entry = {"test": test_no, "name": name, "passed": bool(passed), "evidence": evidence}
    if extra:
        entry["extra"] = extra
    RESULTS.append(entry)
    mark = "PASS" if passed else "FAIL"
    print(f"[{mark}] 测试{test_no} {name}")
    print(f"       {evidence}")
    return entry


# --------------------------------------------------------------------------------------
# 测试一：简单积分必须直接得出精确结果（对应规格 6.4）
# --------------------------------------------------------------------------------------
def test1_simple_integral():
    payload = E.execute("integrate", {"expr": "x^2", "var": "x", "lower": "0", "upper": "1"})
    text = json.dumps(payload.get("result", {}), ensure_ascii=False)
    ok = payload.get("status") == "ok" and "1/3" in text.replace(" ", "")
    methods = payload.get("verification", {}).get("methods") or []
    record(
        1,
        "简单积分 ∫₀¹x²dx 直接返回精确结果 1/3",
        ok,
        f"status={payload.get('status')} result={text[:120]} 验证方法数={len(methods)}",
        {"elapsed_ms": payload.get("elapsed_ms")},
    )


# --------------------------------------------------------------------------------------
# 测试二：复杂积分——结果正确且不重复调用
# --------------------------------------------------------------------------------------
def test2_complex_integral():
    # 需要分部积分的经典题
    payload = E.execute("integrate", {"expr": "x*exp(x)", "var": "x"})
    text = json.dumps(payload.get("result", {}), ensure_ascii=False).replace(" ", "")
    # 正确性用数学比对，而不是匹配某一种书写形式：
    # 引擎返回 (x-1)*exp(x)，与教科书写法 x*e^x - e^x 恒等，字符串比对会误判。
    got_text = payload.get("result", {}).get("text", "")
    try:
        _x = sp.Symbol("x")
        correct = (payload.get("status") == "ok"
                   and got_text != ""
                   and sp.simplify(sp.sympify(got_text) - (_x - 1) * sp.exp(_x)) == 0)
    except Exception:  # noqa: BLE001
        correct = False

    # 同一请求走 worker 的 Cache，必须命中而不重复计算
    cache = Cache()
    args = {"expr": "x*exp(x)", "var": "x"}
    key = cache.key("integrate", args)
    first = cache.get("integrate", args)
    cache.store("integrate", args, payload)
    second = cache.get("integrate", args)
    no_repeat = first is None and second is not None and second.get("status") == "ok"

    record(
        2,
        "复杂积分 ∫xe^x dx 结果正确且缓存可复用（不重复计算）",
        correct and no_repeat,
        f"结果正确={correct} 缓存首次未命中/二次命中={first is None and second is not None} "
        f"result={text[:120]}",
        {"cache_key": key},
    )


# --------------------------------------------------------------------------------------
# 测试三：无法求解的积分——必须失败得明确，且不无限重试
# --------------------------------------------------------------------------------------
def test3_unsolvable_integral():
    # ∫ x^x dx 没有初等原函数（∫e^{x²}dx 有 erfi 闭式，不能用来测「无法求解」）
    payload = E.execute("integrate", {"expr": "x^x", "var": "x"})
    status = payload.get("status")
    result = payload.get("result", {})
    text = json.dumps(result, ensure_ascii=False)

    # 不允许把未求值的 Integral 当作成功结果（规格 5.3）
    misreported = status == "ok" and "Integral(" in text
    honest = status in ("unsolved", "partial", "error")
    has_reason = bool(payload.get("error", {}).get("message")) or bool(result.get("note")) or bool(payload.get("warnings"))

    # 失败必须被记账，避免同参数重复尝试
    cache = Cache()
    args = {"expr": "x^x", "var": "x"}
    cache.store("integrate", args, payload)
    recorded = cache.failed_before("integrate", args) is not None

    record(
        3,
        "无法求得初等原函数的积分必须如实报告且不重复尝试",
        honest and not misreported and has_reason and recorded,
        f"status={status} 误报成功={misreported} 有失败原因={has_reason} 失败已记账={recorded}",
        {"error": payload.get("error"), "warnings": payload.get("warnings")},
    )


# --------------------------------------------------------------------------------------
# 测试四：连续追问——第二轮必须能复用第一轮结果
# --------------------------------------------------------------------------------------
def test4_followup_reuse():
    # 第一轮：求导
    first = E.execute("diff", {"expr": "x^3-3*x", "var": "x"})
    first_text = json.dumps(first.get("result", {}), ensure_ascii=False).replace(" ", "")
    first_ok = first.get("status") == "ok" and "3*x**2-3" in first_text

    # 第二轮：追问极值点——复用第一轮导函数，不重新对原函数求导
    second = E.execute("solve", {"equation": first.get("result", {}).get("text", "3*x**2-3"), "var": "x"})
    second_text = json.dumps(second.get("result", {}), ensure_ascii=False).replace(" ", "")
    second_ok = second.get("status") in ("ok", "partial") and "1" in second_text

    # 缓存层面：第一轮结果已存入，第二轮若问同一表达式应命中
    cache = Cache()
    cache.store("diff", {"expr": "x^3-3*x", "var": "x"}, first)
    reused = cache.get("diff", {"expr": "x^3-3*x", "var": "x"}) is not None

    record(
        4,
        "连续追问可复用前一轮结果（导函数 → 极值点）",
        first_ok and second_ok and reused,
        f"第一轮导函数={first_text[:60]} 第二轮解={second_text[:80]} 缓存复用={reused}",
    )


# --------------------------------------------------------------------------------------
# 测试五：已有答案验证——必须能判定对错并定位错误步骤
# --------------------------------------------------------------------------------------
def test5_verify_given_answer():
    # 正确推导：应判通过
    good = E.execute(
        "verify",
        {"kind": "derivative", "input": "x^2*sin(x)", "var": "x",
         "candidate": "2*x*sin(x)+x^2*cos(x)"},
    )
    good_passed = good.get("result", {}).get("passed") is True

    # 漏掉乘积法则第二项：必须判否，且给出证据
    bad = E.execute(
        "verify",
        {"kind": "derivative", "input": "x^2*sin(x)", "var": "x", "candidate": "2*x*sin(x)"},
    )
    bad_result = bad.get("result", {})
    bad_failed = bad_result.get("passed") is False
    evidence = bad_result.get("evidence") or bad.get("verification", {}).get("evidence")

    # 积分验证：错误的原函数必须被发现
    bad_int = E.execute(
        "verify",
        {"kind": "antiderivative", "input": "x*exp(x)", "var": "x", "candidate": "exp(x)"},
    )
    bad_int_failed = bad_int.get("result", {}).get("passed") is False

    record(
        5,
        "验证模式能定位错误步骤（漏乘积法则项 / 错误原函数均被判否）",
        good_passed and bad_failed and bad_int_failed and bool(evidence),
        f"正确判定通过={good_passed} 漏项判定失败={bad_failed} 错误原函数判否={bad_int_failed} "
        f"证据={json.dumps(evidence, ensure_ascii=False, default=str)[:160]}",
    )


# --------------------------------------------------------------------------------------
# 测试六：思路分析——工具应提供分析所需的客观要素，而不是替模型下结论
# --------------------------------------------------------------------------------------
def test6_analysis_mode_support():
    # 曲面积分/格林公式类问题需要「封闭性、投影区域、对称性」等分析要素。
    # 工具应给出定义域、单调性、极值、渐近线等客观信息，供模型据以分析。
    analysis = E.execute("function_analysis", {"expr": "x^3-3*x", "var": "x"})
    result = analysis.get("result", {})
    has_structured = analysis.get("status") == "ok" and len(result) >= 3

    domain = E.execute("domain", {"expr": "sqrt(x-1)/(x-2)", "var": "x"})
    domain_result = domain.get("result", {})
    domain_ok = domain.get("status") in ("ok", "partial") and bool(
        domain_result.get("domain") or domain_result.get("conditions") or domain_result.get("restrictions")
    )

    # 四类任务模式的区分依赖 status/method/verification 字段齐全
    fields = set(result) | set(analysis.get("verification", {}))
    structured = {"domain", "monotonic", "extrema"} & set(map(str.lower, map(str, result)))

    record(
        6,
        "思路分析模式有客观要素支撑（函数分析 + 定义域结构化输出）",
        has_structured and domain_ok,
        f"函数分析字段={sorted(map(str, result))[:8]} 定义域输出={json.dumps(domain_result, ensure_ascii=False, default=str)[:140]}",
        {"structured_keys": sorted(structured)},
    )


# --------------------------------------------------------------------------------------
# 测试七：结果交叉验证——重要结果必须真正经过独立验证
# --------------------------------------------------------------------------------------
def test7_cross_verification():
    # 定积分：符号结果应能被独立数值方法复核
    payload = E.execute("integrate", {"expr": "sin(x)", "var": "x", "lower": "0", "upper": "pi"})
    verification = payload.get("verification", {}) or {}
    methods = verification.get("methods") or []
    level = verification.get("level") or verification.get("status")
    verified = bool(methods) or level in ("symbolic", "numeric", "independent", "cross")

    # 矩阵求逆：应有乘积为单位阵的独立核验
    inv = E.execute("inv", {"matrix": [["1", "2"], ["3", "4"]]})
    inv_verification = inv.get("verification", {}) or {}
    inv_methods = inv_verification.get("methods") or []
    inv_level = inv_verification.get("level") or inv_verification.get("status")
    inv_verified = bool(inv_methods) or inv_level in ("symbolic", "numeric", "independent", "cross")

    # 只有数值证据时必须如实标注，不允许冒充严格证明
    honest_label = level in ("symbolic", "numeric", "independent", "cross", "none", None)

    record(
        7,
        "重要结果经过独立验证且验证级别诚实标注",
        verified and inv_verified and honest_label,
        f"定积分验证级别={level} 方法数={len(methods)}；求逆验证级别={inv_level} 方法数={len(inv_methods)}",
        {"integrate_verification": verification, "inverse_verification": inv_verification},
    )


# --------------------------------------------------------------------------------------
# 测试八：相似表达式——缓存正确复用，且不同条件绝不混用
# --------------------------------------------------------------------------------------
def test8_similar_expressions_cache():
    cache = Cache()

    base_args = {"expr": "x^2", "var": "x"}
    base = E.execute("integrate", base_args)
    cache.store("integrate", base_args, base)

    # 同一表达式同条件：应命中
    hit = cache.get("integrate", base_args) is not None

    # 相似但条件不同：不定积分 vs 定积分，绝不能混用
    definite_args = {"expr": "x^2", "var": "x", "lower": "0", "upper": "1"}
    no_false_hit = cache.get("integrate", definite_args) is None

    # 换变量：不能命中
    other_var = cache.get("integrate", {"expr": "x^2", "var": "t"}) is None

    # 不同方向/条件的极限不能混用
    left = {"expr": "1/x", "var": "x", "point": "0", "dir": "-"}
    right = {"expr": "1/x", "var": "x", "point": "0", "dir": "+"}
    cache.store("limit", left, E.execute("limit", left))
    dir_no_mix = cache.get("limit", right) is None

    # 精度不同不能混用
    cache.store("numeric", {"expr": "pi", "precision": 20}, E.execute("numeric", {"expr": "pi", "precision": 20}))
    precision_no_mix = cache.get("numeric", {"expr": "pi", "precision": 50}) is None

    record(
        8,
        "相似表达式缓存：同条件命中，不同条件（定/不定、变量、方向、精度）不混用",
        hit and no_false_hit and other_var and dir_no_mix and precision_no_mix,
        f"同条件命中={hit} 定积分不误命中={no_false_hit} 换变量不误命中={other_var} "
        f"极限方向不混用={dir_no_mix} 精度不混用={precision_no_mix}",
    )


TESTS = [
    test1_simple_integral,
    test2_complex_integral,
    test3_unsolvable_integral,
    test4_followup_reuse,
    test5_verify_given_answer,
    test6_analysis_mode_support,
    test7_cross_verification,
    test8_similar_expressions_cache,
]


def main():
    for fn in TESTS:
        started = time.perf_counter()
        try:
            fn()
        except Exception as error:  # noqa: BLE001
            import traceback

            record(0, fn.__name__, False, f"测试执行异常：{type(error).__name__}: {error}")
            traceback.print_exc()
        finally:
            print(f"       （耗时 {time.perf_counter() - started:.2f}s）")

    passed = sum(1 for r in RESULTS if r["passed"])
    print(f"\n八个专项：通过 {passed} / {len(RESULTS)}")
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_special_result.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(RESULTS, handle, ensure_ascii=False, indent=2, default=str)
    print(f"详细结果已写入 {out_path}")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
