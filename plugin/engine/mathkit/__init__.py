r"""mathkit —— DSH 考研数学计算引擎。

模块划分：

* :mod:`mathkit.ast`          表达式规范化与解析（$\LaTeX$ 与自然写法兼容）
* :mod:`mathkit.result`       统一结构化返回格式（success/status/conditions/verification/...）
* :mod:`mathkit.engine`       算子注册表、参数别名归一、超时与异常隔离
* :mod:`mathkit.sympy_core`   严谨性原语（等价判定、独立复核、抽样验算）
* :mod:`mathkit.calculus`     极限、导数、积分、级数、微分方程
* :mod:`mathkit.linalg`       行列式、矩阵、特征值、二次型
* :mod:`mathkit.probability`  概率分布、期望方差、参数估计
* :mod:`mathkit.numeric`      高精度数值计算与数值验证
* :mod:`mathkit.verifier`     math_verify 统一复核入口
* :mod:`mathkit.ops`          汇总导入，触发全部算子注册

所有算子只依赖标准库、sympy/numpy/scipy/mpmath 与本包的 ast/result/sympy_core。
"""

__all__ = ["ast", "result", "engine", "sympy_core", "ops"]
__version__ = "1.0.0"
