# -*- coding: utf-8 -*-
"""服务端参数网格寻优：AI 只给搜索范围，本模块负责实际调参。

设计原则（与产品分工一致）：

- AI 负责「写底层引擎 + 决定调参范围」：因子权重的上下限/步长，或自定义
  代码引擎的参数范围；
- 服务端负责「真正把参数试出来」：在范围内生成网格，逐个组合回测评估，
  保留目标指标最优的一组；
- 本模块只做搜索编排与数值计算，不依赖 LLM、FastAPI、数据库；回测评估由
  调用方注入 ``evaluate``，因此既可以在 service 里接真实回测，也可以用
  合成评估函数做快速单测。

搜索算法：全组合网格 + 预算内确定性分层采样（不再静默砍轴/砍候选）。

1. 按每个轴的 min/max/step 生成该轴的全部候选值，不做逐轴候选数硬裁剪；
2. ``total_combinations`` = 各轴候选数的笛卡尔积；
3. 总组合数不超过 ``max_trials`` 时**全量评估**所有组合；
4. 超过预算时用固定随机种子做确定性分层采样：每个轴的候选值在所有采样点中
   分层出现，保证每个轴都被搜索到，并总是包含「当前参数（基线）」一组，
   最多评估 ``max_trials`` 组。

事件里带 ``total_combinations`` / ``planned_trials`` / ``sampled`` 字段，
调用方（tuning 卡片）可以如实展示「全量评估」还是「预算内采样」。
"""
from __future__ import annotations

import itertools
import math
import os
import random
import time
from dataclasses import dataclass
from collections.abc import Callable, Iterator, Mapping
from typing import Any

from .factors import FACTOR_NAMES, MAX_WEIGHT

__all__ = [
    "OBJECTIVES",
    "OBJECTIVE_LABELS",
    "TuningError",
    "Axis",
    "MIN_TRIALS",
    "MAX_TRIALS",
    "DEFAULT_MAX_TRIALS",
    "DEFAULT_TUNING_TRIALS",
    "max_trials_limit",
    "normalize_weight_space",
    "normalize_param_space",
    "score_metrics",
    "run_search",
]

OBJECTIVES = ("sharpe", "cagr", "mdd", "calmar", "sortino", "ir", "win_rate")

OBJECTIVE_LABELS = {
    "sharpe": "夏普比率",
    "cagr": "年化收益",
    "mdd": "最大回撤",
    "calmar": "卡玛比率",
    "sortino": "索提诺比率",
    "ir": "信息比率",
    "win_rate": "月胜率",
}

# 排序方向：sharpe / cagr / calmar / sortino / ir / win_rate 越大越好；
# mdd 是负数，越接近 0 越好，因此统一用「分数越大越好」比较即可。


def _env_positive_int(name: str, default: int) -> int:
    """读取正整数环境变量；缺省/非法/<=0 时退回默认值。"""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return int(default)
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return int(default)
    return value if value >= 1 else int(default)


# 单次寻优的评估次数上限（保护回测时长），可用环境变量 TUNING_MAX_TRIALS 覆盖
MIN_TRIALS = 1
DEFAULT_MAX_TRIALS = 1000
MAX_TRIALS = _env_positive_int("TUNING_MAX_TRIALS", DEFAULT_MAX_TRIALS)
# 调用方未给 max_trials（None/省略）时的默认评估预算（与调参方案默认值一致）
DEFAULT_TUNING_TRIALS = 32
# 采样固定种子：同样的输入必须得到同样的候选组合，结果可复现、可解释
_SAMPLING_SEED = 20240601
# 单轴候选值数量保护：超过则明确报错（绝不静默裁剪轴或候选）
_MAX_VALUES_PER_AXIS = 20000
# 浮点网格去重精度
_EPS = 1e-9
# 年化交易日数（与回测引擎一致）
_TRADING_DAYS = 252
# 无下行波动时索提诺比率的有限大值（数学上是无穷大，这里用于排序）
_SORTINO_CAP = 1e6


def max_trials_limit() -> int:
    """当前生效的评估次数上限（每次调用读取环境变量，便于运维调整）。"""
    return _env_positive_int("TUNING_MAX_TRIALS", MAX_TRIALS)


class TuningError(ValueError):
    """搜索范围非法或所有参数组合评估失败。"""


@dataclass(frozen=True)
class Axis:
    """一个待搜索参数轴。"""

    key: str
    label: str
    kind: str  # "weight" | "param"
    integer: bool
    minimum: float
    maximum: float
    step: float
    candidates: tuple[float, ...]

    def public(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "min": _round_number(self.minimum),
            "max": _round_number(self.maximum),
            "step": _round_number(self.step),
        }


# ---------------------------------------------------------------------------
# 校验与网格生成
# ---------------------------------------------------------------------------


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TuningError(f"{name} 必须是数值，当前为 {value!r}") from exc
    if not math.isfinite(number):
        raise TuningError(f"{name} 必须是有限数值")
    return number


def _round_number(value: float) -> float | int:
    if float(value).is_integer():
        return int(value)
    return round(float(value), 6)


def _same_value(left: Any, right: Any, integer: bool) -> bool:
    try:
        if integer:
            return int(round(float(left))) == int(round(float(right)))
        return abs(float(left) - float(right)) <= 1e-9
    except (TypeError, ValueError):
        return False


def _grid_values(minimum: float, maximum: float, step: float, integer: bool) -> list[float]:
    """在 [minimum, maximum] 内按 step 生成全部网格值；端点必须可用。"""
    if integer:
        minimum_i = int(round(minimum))
        maximum_i = int(round(maximum))
        step_i = max(1, int(round(step)))
        values = list(range(minimum_i, maximum_i + 1, step_i))
        if not values or values[-1] != maximum_i:
            values.append(maximum_i)
        return [float(value) for value in values]

    count = int(math.floor((maximum - minimum) / step + 1e-9)) + 1
    count = max(2, count)
    values = [round(minimum + index * step, 10) for index in range(count)]
    if values[-1] < maximum - _EPS:
        values.append(round(maximum, 10))
    # 去重并去掉越界值
    unique: list[float] = []
    for value in values:
        if value < minimum - _EPS or value > maximum + _EPS:
            continue
        if not unique or abs(value - unique[-1]) > _EPS:
            unique.append(value)
    if len(unique) == 1:
        unique.append(unique[0])
    return unique


def normalize_weight_space(
    space: Mapping[str, Any],
    *,
    min_axes: int = 1,
) -> dict[str, dict[str, float]]:
    """校验权重搜索范围，返回保持 AI 优先级顺序的规范化字典。

    ``min_axes`` 默认 1：允许只搜索 1 个因子（不再强制至少 3 个）。
    """
    if not isinstance(space, Mapping) or not space:
        raise TuningError("search_space 必须是非空对象（因子名 -> {min,max,step}）")
    normalized: dict[str, dict[str, float]] = {}
    for raw_key, raw_spec in space.items():
        key = str(raw_key).strip()
        if key not in FACTOR_NAMES:
            raise TuningError(
                f"search_space 含未知因子 {key!r}；只允许 {', '.join(FACTOR_NAMES)}"
            )
        if not isinstance(raw_spec, Mapping):
            raise TuningError(f"因子 {key} 的搜索范围必须是对象")
        minimum = _finite_number(raw_spec.get("min"), f"{key}.min")
        maximum = _finite_number(raw_spec.get("max"), f"{key}.max")
        step = _finite_number(raw_spec.get("step"), f"{key}.step")
        if minimum < 0:
            raise TuningError(f"因子 {key} 的搜索下限不能为负")
        if maximum > MAX_WEIGHT + _EPS:
            raise TuningError(f"因子 {key} 的搜索上限不能超过 {MAX_WEIGHT}")
        if maximum - minimum < step - _EPS:
            raise TuningError(
                f"因子 {key} 的范围 [{minimum}, {maximum}] 不足一个步长 {step}，至少需要两个候选值"
            )
        if step <= 0:
            raise TuningError(f"因子 {key} 的步长必须大于 0")
        normalized[key] = {"min": minimum, "max": maximum, "step": step}
    if len(normalized) < int(min_axes):
        raise TuningError(
            f"search_space 至少需要 {min_axes} 个因子，当前只有 {len(normalized)} 个"
        )
    return normalized


def normalize_param_space(space: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """校验自定义引擎的参数搜索范围（>=1 个参数，不设个数上限）。"""
    if not isinstance(space, Mapping) or not space:
        raise TuningError("param_space 必须是非空对象（参数名 -> {min,max,step,type}）")
    normalized: dict[str, dict[str, Any]] = {}
    for raw_key, raw_spec in space.items():
        key = str(raw_key).strip()
        if not key or not key.replace("_", "a").isalnum() or key[0].isdigit():
            raise TuningError(f"参数名 {key!r} 不是合法 Python 标识符")
        if not isinstance(raw_spec, Mapping):
            raise TuningError(f"参数 {key} 的搜索范围必须是对象")
        minimum = _finite_number(raw_spec.get("min"), f"{key}.min")
        maximum = _finite_number(raw_spec.get("max"), f"{key}.max")
        step = _finite_number(raw_spec.get("step"), f"{key}.step")
        kind = str(raw_spec.get("type") or "float").strip().lower()
        if kind not in ("int", "float"):
            raise TuningError(f"参数 {key} 的 type 只能是 int 或 float")
        if step <= 0:
            raise TuningError(f"参数 {key} 的步长必须大于 0")
        if maximum - minimum < step - _EPS:
            raise TuningError(
                f"参数 {key} 的范围 [{minimum}, {maximum}] 不足一个步长 {step}，至少需要两个候选值"
            )
        if kind == "int" and (int(round(step)) < 1 or int(round(minimum)) != minimum
                              or int(round(maximum)) != maximum):
            raise TuningError(f"参数 {key} 是整数参数，min/max/step 必须是整数")
        default = raw_spec.get("default")
        default_value = _finite_number(default, f"{key}.default") if default is not None else (
            (minimum + maximum) / 2.0
        )
        if kind == "int":
            default_value = float(int(round(default_value)))
        normalized[key] = {
            "min": minimum,
            "max": maximum,
            "step": step,
            "type": kind,
            "default": default_value,
        }
    return normalized


def _build_axes(
    space: Mapping[str, Mapping[str, Any]],
    *,
    kind: str,
    labels: Mapping[str, str] | None,
) -> list[Axis]:
    """把规范化范围展开成轴（每轴保留 min/max/step 生成的全部候选值）。"""
    axes: list[Axis] = []
    for key, spec in space.items():
        integer = kind == "param" and str(spec.get("type")) == "int"
        minimum = float(spec["min"])
        maximum = float(spec["max"])
        step = float(spec["step"])
        values = _grid_values(minimum, maximum, step, integer)
        if len(values) > _MAX_VALUES_PER_AXIS:
            raise TuningError(
                f"参数 {key} 的候选值过多（{len(values)} 个，单轴上限 {_MAX_VALUES_PER_AXIS}）；"
                "请增大 step 或缩小搜索范围"
            )
        label = str((labels or {}).get(key) or key)
        axes.append(Axis(key=key, label=label, kind=kind, integer=integer,
                         minimum=minimum, maximum=maximum, step=step,
                         candidates=tuple(values)))
    return axes


# ---------------------------------------------------------------------------
# 组合规划：全量网格 / 预算内确定性分层采样
# ---------------------------------------------------------------------------


def _typed_value(axis: Axis, candidate: float) -> float | int:
    return int(round(candidate)) if axis.integer else float(candidate)


def _base_combination(axes: list[Axis], base: Mapping[str, Any]) -> dict[str, Any]:
    """当前参数组合（基线）；缺失时用该轴第一个候选值兜底。"""
    combo: dict[str, Any] = {}
    for axis in axes:
        raw = base.get(axis.key) if isinstance(base, Mapping) else None
        try:
            value = float(raw) if raw is not None else float(axis.candidates[0])
        except (TypeError, ValueError):
            value = float(axis.candidates[0])
        combo[axis.key] = _typed_value(axis, value)
    return combo


def _combo_key(axes: list[Axis], combo: Mapping[str, Any]) -> tuple[Any, ...]:
    parts: list[Any] = []
    for axis in axes:
        value = float(combo.get(axis.key, 0.0))
        parts.append(int(round(value)) if axis.integer else round(value, 9))
    return tuple(parts)


def _combo_equals(axes: list[Axis], left: Mapping[str, Any],
                  right: Mapping[str, Any]) -> bool:
    return all(
        _same_value(left.get(axis.key), right.get(axis.key), axis.integer)
        for axis in axes
    )


def _sampled_combinations(
    axes: list[Axis],
    *,
    limit: int,
    seed: int = _SAMPLING_SEED,
) -> list[dict[str, Any]]:
    """确定性分层采样：每轴候选值在采样点中分层出现，保证不丢轴。

    - 固定种子 => 同样输入得到同样候选，结果可复现；
    - 每个轴独立打乱顺序后按分层位置取值，轴与轴之间互不相同、覆盖分散；
    - ``limit <= 该轴候选数`` 时各轴取互不相同的档位；``limit`` 更大时按轮
      重复整层并每轮重新打乱，保证每个候选值至少出现一次；
    - 分层序列若意外产生重复组合，用固定种子的随机组合补齐（仍可复现）。
    """
    if limit <= 0 or not axes:
        return []
    rng = random.Random(seed)
    candidates_per_axis: list[list[float]] = [list(axis.candidates) for axis in axes]
    picks_per_axis: list[list[int]] = []
    for candidates in candidates_per_axis:
        count = len(candidates)
        picks: list[int] = []
        while len(picks) < limit:
            order = list(range(count))
            rng.shuffle(order)
            if limit <= count:
                # 预算比档位少：均匀抽取互不相同的档位
                picks.extend(order[(index * count) // limit]
                             for index in range(limit))
            else:
                # 预算比档位多：整轮覆盖全部档位，下一轮重新打乱以去相关
                picks.extend(order)
        picks_per_axis.append(picks[:limit])

    combos: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    def add(combo: dict[str, Any]) -> bool:
        key = _combo_key(axes, combo)
        if key in seen:
            return False
        seen.add(key)
        combos.append(combo)
        return True

    for index in range(limit):
        combo = {
            axis.key: _typed_value(axis, candidates[picks[index]])
            for axis, candidates, picks in zip(axes, candidates_per_axis, picks_per_axis)
        }
        add(combo)

    attempts = 0
    max_attempts = limit * 8 + 16
    while len(combos) < limit and attempts < max_attempts:
        attempts += 1
        combo = {
            axis.key: _typed_value(axis, rng.choice(candidates))
            for axis, candidates in zip(axes, candidates_per_axis)
        }
        add(combo)
    return combos


def _plan_combinations(
    axes: list[Axis],
    base_combo: Mapping[str, Any],
    max_trials: int,
) -> tuple[int, list[dict[str, Any]], bool, int]:
    """规划要评估的组合。

    Returns
    -------
    ``(total_combinations, planned, sampled, planned_trials)``
        ``total_combinations`` 是全组合网格大小（各轴候选数乘积）；
        ``planned`` 是实际要评估的参数组合列表（基线在需要时排在最前）。
    """
    total_combinations = 1
    for axis in axes:
        total_combinations *= len(axis.candidates)

    if total_combinations <= max_trials:
        planned: list[dict[str, Any]] = []
        for values in itertools.product(*(axis.candidates for axis in axes)):
            planned.append({
                axis.key: _typed_value(axis, value)
                for axis, value in zip(axes, values)
            })
        # 全量评估：基线若在网格内提到最前；网格外且预算还够时补一组基线
        for index, combo in enumerate(planned):
            if _combo_equals(axes, combo, base_combo):
                if index:
                    planned.insert(0, planned.pop(index))
                break
        else:
            if len(planned) < max_trials:
                planned.insert(0, dict(base_combo))
        return total_combinations, planned, False, len(planned)

    # 预算不足：固定种子分层采样；基线固定占 1 组，不参与采样
    limit = max(0, max_trials - 1)
    planned = [dict(base_combo)]
    for combo in _sampled_combinations(axes, limit=limit):
        if _combo_equals(axes, combo, base_combo):
            continue
        planned.append(combo)
    planned = planned[:max_trials]
    return total_combinations, planned, True, len(planned)


# ---------------------------------------------------------------------------
# 指标打分
# ---------------------------------------------------------------------------


def _metric_number(value: Any) -> float | None:
    """把指标转成有限 float；缺失/非法返回 None。"""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_metric(metrics: Mapping[str, Any], key: str) -> float:
    number = _metric_number(metrics.get(key))
    return number if number is not None else 0.0


def _nav_values(nav: Any) -> list[float]:
    """从各种净值表示里取出正的净值序列（pandas.Series / list / dict 均可）。"""
    if nav is None:
        return []
    values = nav
    if isinstance(values, Mapping):
        for key in ("nav", "nav_values", "values", "data"):
            if key in values:
                return _nav_values(values[key])
        return []
    to_numpy = getattr(values, "to_numpy", None)
    if callable(to_numpy):
        try:
            values = to_numpy()
        except (TypeError, ValueError, AttributeError):
            return []
    try:
        items = list(values)
    except TypeError:
        return []
    result: list[float] = []
    for item in items:
        if isinstance(item, Mapping):
            picked = None
            for key in ("nav", "nav_value", "value", "close"):
                if key in item:
                    picked = item[key]
                    break
            if picked is None:
                return []
            item = picked
        try:
            number = float(item)
        except (TypeError, ValueError):
            return []
        if not math.isfinite(number) or number <= 0:
            return []
        result.append(number)
    return result


def _derived_calmar(metrics: Mapping[str, Any]) -> float | None:
    """metrics 未直接给 calmar 时，用 年化收益 / |最大回撤| 派生。"""
    mdd = abs(_safe_metric(metrics, "mdd"))
    if mdd <= 1e-12:
        return None
    return _safe_metric(metrics, "cagr") / mdd


def _derived_sortino(metrics: Mapping[str, Any], nav: Any) -> float | None:
    """metrics 未直接给 sortino 时，用净值日收益的下行波动派生。"""
    values = _nav_values(nav)
    if len(values) < 3:
        return None
    returns = [values[index] / values[index - 1] - 1.0
               for index in range(1, len(values))]
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    downside = [min(value, 0.0) for value in returns]
    deviation = math.sqrt(sum(value * value for value in downside) / len(returns))
    if deviation <= 1e-12:
        # 没有下行波动：数学上是无穷大，正收益记为大值参与排序
        return _SORTINO_CAP if mean > 0 else 0.0
    return mean / deviation * math.sqrt(_TRADING_DAYS)


def _objective_value(metrics: Mapping[str, Any], objective: str,
                     nav: Any = None) -> float:
    """目标指标的展示值/排序值（越大越好；mdd 为负数、越接近 0 越好）。"""
    if objective == "calmar":
        value = _metric_number(metrics.get("calmar"))
        if value is not None:
            return value
        derived = _derived_calmar(metrics)
        return derived if derived is not None else 0.0
    if objective == "sortino":
        value = _metric_number(metrics.get("sortino"))
        if value is not None:
            return value
        derived = _derived_sortino(metrics, nav)
        return derived if derived is not None else 0.0
    return _safe_metric(metrics, objective)


def score_metrics(metrics: Mapping[str, Any], objective: str,
                  tolerance: float | None = None,
                  nav: Any = None) -> tuple[float, bool]:
    """把回测指标折算成可比较的分数（越大越好）。

    - sharpe / cagr / calmar / sortino / ir / win_rate：直接取指标，越大越好；
    - calmar 缺失时用 年化收益 / |最大回撤| 派生；sortino 缺失时用净值日收益的
      下行波动派生（不扣无风险利率，仅用于同轮候选之间的相对排序）；
    - mdd：最大回撤是负数，越接近 0 越好，因此取原值比较；
    - 若客户给出最大回撤上限而候选超限，则按超出幅度扣分，避免选出
      收益高但风险失控的参数。
    """
    if objective not in OBJECTIVES:
        raise TuningError(f"未知优化目标 {objective!r}；可选 {', '.join(OBJECTIVES)}")
    score = _objective_value(metrics, objective, nav)
    constraint_ok = True
    if tolerance is not None and objective != "mdd":
        drawdown = abs(_safe_metric(metrics, "mdd"))
        if math.isfinite(drawdown) and drawdown > float(tolerance) + _EPS:
            constraint_ok = False
            # 超过客户最大回撤上限属于一票否决级风险：先大幅扣分，
            # 只有所有候选都超限时才会退回到「相对亏损最小」的方案。
            score -= (drawdown - float(tolerance)) * 20.0 + 1.0
    return score, constraint_ok


# ---------------------------------------------------------------------------
# 搜索主流程
# ---------------------------------------------------------------------------


def run_search(
    evaluate: Callable[[dict[str, Any]], Mapping[str, Any]],
    *,
    space: Mapping[str, Any],
    base_params: Mapping[str, Any],
    objective: str = "sharpe",
    max_trials: int | None = DEFAULT_TUNING_TRIALS,
    tolerance: float | None = None,
    kind: str = "weight",
    labels: Mapping[str, str] | None = None,
    method: str = "",
) -> Iterator[dict[str, Any]]:
    """执行全组合网格搜索（超预算时确定性采样），逐个 ``yield`` 进度事件。

    ``evaluate(params)`` 必须返回 dict，至少包含 ``metrics``；可附带
    ``params``（评估时实际使用的参数，例如归一化后的权重）与 ``nav``。
    评估抛异常的组合会被跳过；全部组合都失败时抛 :class:`TuningError`。

    事件类型：
    - ``{"event": "start", ...}``：搜索开始，含范围、目标、总组合数与计划评估数；
    - ``{"event": "trial", ...}``：完成一次评估，含当前/最优指标与进度；
    - ``{"event": "done", ...}``：搜索结束，含最优参数、指标、耗时与预算信息。

    ``start`` / ``done`` 都带 ``total_combinations``（全组合网格大小）、
    ``planned_trials``（本次计划评估数）、``sampled``（是否采样）与兼容字段
    ``total``（= planned_trials）。
    """
    if kind not in ("weight", "param"):
        raise TuningError(f"未知搜索类型 {kind!r}")
    if objective not in OBJECTIVES:
        raise TuningError(f"未知优化目标 {objective!r}")
    requested_trials = DEFAULT_TUNING_TRIALS if max_trials is None else max_trials
    try:
        max_trials_i = int(requested_trials)
    except (TypeError, ValueError) as exc:
        raise TuningError("max_trials 必须是整数") from exc
    max_trials_i = max(MIN_TRIALS, min(max_trials_limit(), max_trials_i))

    if kind == "weight":
        normalized_space = normalize_weight_space(space)
        base = {name: float((base_params or {}).get(name, 0.0)) for name in FACTOR_NAMES}
        default_method = "全组合网格搜索"
    else:
        normalized_space = normalize_param_space(space)
        base = {}
        for key, spec in normalized_space.items():
            value = float(spec.get("default", (spec["min"] + spec["max"]) / 2.0))
            base[key] = int(round(value)) if spec.get("type") == "int" else value
        default_method = "参数全组合网格搜索"

    axes = _build_axes(normalized_space, kind=kind, labels=labels)
    if not axes:
        raise TuningError("搜索范围为空，无法寻优")
    base_combo = _base_combination(axes, base)
    total_combinations, planned, sampled, planned_trials = _plan_combinations(
        axes, base_combo, max_trials_i)

    started = time.time()
    best: dict[str, Any] | None = None
    trials: list[dict[str, Any]] = []
    failures = 0
    last_error = ""
    evaluated = 0

    def make_best(params: Mapping[str, Any], metrics: Mapping[str, Any],
                  score: float, constraint_ok: bool, trial: int = 0,
                  extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "trial": trial,
            "score": round(float(score), 6),
            "params": dict(params),
            "metrics": dict(metrics),
            "constraint_ok": bool(constraint_ok),
        }
        if extra and extra.get("nav") is not None:
            payload["nav"] = extra["nav"]
        return payload

    def evaluate_one(trial_params: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
        nonlocal failures, last_error
        try:
            raw = evaluate(dict(trial_params))
        except Exception as exc:  # noqa: BLE001 - 单个组合失败不应中断整轮寻优
            failures += 1
            last_error = f"{type(exc).__name__}: {exc}"
            return None, last_error
        if not isinstance(raw, Mapping):
            failures += 1
            last_error = "评估函数必须返回 dict"
            return None, last_error
        metrics = raw.get("metrics")
        if not isinstance(metrics, Mapping):
            failures += 1
            last_error = "评估结果缺少 metrics"
            return None, last_error
        effective = raw.get("params") if isinstance(raw.get("params"), Mapping) else trial_params
        nav = raw.get("nav")
        score, constraint_ok = score_metrics(metrics, objective, tolerance, nav=nav)
        best = make_best(effective, metrics, score, constraint_ok,
                         extra={"nav": nav})
        best["metric_value"] = _objective_value(metrics, objective, nav=nav)
        return best, ""

    yield {
        "event": "start",
        "method": method or default_method,
        "objective": objective,
        "objective_label": OBJECTIVE_LABELS.get(objective, objective),
        "space": [axis.public() for axis in axes],
        "total": planned_trials,
        "total_combinations": total_combinations,
        "planned_trials": planned_trials,
        "sampled": sampled,
        "note": ("AI 只给定搜索范围，服务端按全组合网格逐个回测评估；"
                 "超过预算时用固定种子做确定性分层采样，不会丢弃参数轴。"),
    }

    for index, combo in enumerate(planned, start=1):
        result, error = evaluate_one(combo)
        evaluated = index
        is_baseline = _combo_equals(axes, combo, base_combo)
        label = ("当前参数（基线）" if is_baseline
                 else _combo_label(axes, combo, base_combo))
        changed_axes = [axis.key for axis in axes
                        if not _same_value(combo.get(axis.key),
                                           base_combo.get(axis.key), axis.integer)]
        progress = round(index / planned_trials, 4) if planned_trials else 0.0
        if result is None:
            yield {
                "event": "trial",
                "trial": index,
                "total": planned_trials,
                "total_combinations": total_combinations,
                "planned_trials": planned_trials,
                "sampled": sampled,
                "axis": ",".join(changed_axes[:3]),
                "label": label,
                "error": error,
                "best": _public_best(best),
                "progress": progress,
            }
            continue
        result["trial"] = index
        trials.append(result)
        improved = best is None or float(result["score"]) > float(best["score"]) + _EPS
        if improved:
            best = result
        yield {
            "event": "trial",
            "trial": index,
            "total": planned_trials,
            "total_combinations": total_combinations,
            "planned_trials": planned_trials,
            "sampled": sampled,
            "axis": ",".join(changed_axes[:3]),
            "label": label,
            "params": result["params"],
            "metrics": result["metrics"],
            "score": result["score"],
            "metric_value": result.get("metric_value"),
            "improved": improved,
            "best": _public_best(best),
            "progress": progress,
        }

    if best is None:
        detail = f"（最后一次错误：{last_error or '未知错误'}）" if last_error else ""
        raise TuningError(f"所有参数组合回测都失败了{detail}")

    yield {
        "event": "done",
        "method": method or default_method,
        "objective": objective,
        "objective_label": OBJECTIVE_LABELS.get(objective, objective),
        "space": [axis.public() for axis in axes],
        "total": planned_trials,
        "total_combinations": total_combinations,
        "planned_trials": planned_trials,
        "sampled": sampled,
        "evaluated": evaluated,
        "failures": failures,
        "elapsed": round(time.time() - started, 3),
        "best": best,
        "trials": trials,
        "note": ("寻优结果来自历史回测，最优参数可能过拟合；候选仍需人工确认后才会启用。"
                 + (f"本次组合数 {total_combinations}，超过预算已按固定种子采样。"
                    if sampled else "本次已在预算内全量评估。")),
    }


def _combo_label(axes: list[Axis], combo: Mapping[str, Any],
                 base_combo: Mapping[str, Any]) -> str:
    """用「相对基线改动的轴」描述一组候选参数。"""
    parts: list[str] = []
    for axis in axes:
        if _same_value(combo.get(axis.key), base_combo.get(axis.key), axis.integer):
            continue
        value = combo.get(axis.key)
        if axis.integer:
            parts.append(f"{axis.label} = {int(round(float(value)))}")
        else:
            parts.append(f"{axis.label} = {float(value):.4g}")
    if not parts:
        return "当前参数（基线）"
    if len(parts) > 3:
        return "、".join(parts[:3]) + f" 等 {len(parts)} 项"
    return "、".join(parts)


def _public_best(best: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not best:
        return None
    return {
        "trial": best.get("trial"),
        "score": best.get("score"),
        "metric_value": best.get("metric_value"),
        "params": best.get("params"),
        "metrics": best.get("metrics"),
        "constraint_ok": best.get("constraint_ok", True),
    }
