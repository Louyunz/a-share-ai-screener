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

搜索算法：逐因子轮转网格搜索（coordinate grid search）。10 个权重因子的
全组合网格会爆炸，本模块把网格降维为：

1. 先评估当前参数（基线）；
2. 按 AI 给出的范围顺序（视为优先级）逐个参数轴轮转；
3. 每个轴只取该轴网格中的若干候选值，其余参数固定在当前最优；
4. 评估并保留全局最优，继续下一个轴；总评估次数受 ``max_trials`` 硬约束。

因此每一轮最多只改一个参数，计算量可控、结果可解释、进度可流式展示。
"""
from __future__ import annotations

import math
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
    "normalize_weight_space",
    "normalize_param_space",
    "run_search",
]

OBJECTIVES = ("sharpe", "cagr", "mdd")

OBJECTIVE_LABELS = {
    "sharpe": "夏普比率",
    "cagr": "年化收益",
    "mdd": "最大回撤",
}

# 单次寻优的评估次数上限（保护演示时长）
MIN_TRIALS = 4
MAX_TRIALS = 64
# 单个参数轴最多取多少个网格候选值
MAX_CANDIDATES_PER_AXIS = 5
# 浮点网格去重精度
_EPS = 1e-9


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
    """在 [minimum, maximum] 内按 step 生成网格；端点必须可用。"""
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


def _select_candidates(values: list[float], base_value: float | None, limit: int) -> tuple[float, ...]:
    """从网格中挑 limit 个候选：保留两端，尽量包含当前基准值。"""
    if not values:
        return ()
    limit = max(2, min(int(limit), MAX_CANDIDATES_PER_AXIS))

    if base_value is not None and values[0] - _EPS <= base_value <= values[-1] + _EPS:
        # 把当前基准值本身也作为候选，保证寻优可以「原地不动」，且进度可预估
        if not any(abs(value - base_value) <= _EPS for value in values):
            values = sorted(values + [float(base_value)])
    if len(values) <= limit:
        return tuple(values)

    picked: set[int] = {0, len(values) - 1}
    if base_value is not None:
        base_index = min(range(len(values)), key=lambda index: abs(values[index] - base_value))
        picked.add(base_index)

    remaining = limit - len(picked)
    if remaining > 0:
        interior = [index for index in range(1, len(values) - 1) if index not in picked]
        if interior:
            if remaining == 1:
                chosen = [interior[len(interior) // 2]]
            else:
                chosen = []
                for slot in range(remaining):
                    position = round(slot * (len(interior) - 1) / max(1, remaining - 1))
                    chosen.append(interior[position])
            picked.update(chosen)

    ordered = sorted(picked)
    # 如果 picked 数量仍超过 limit（基准值插入导致），按两端+基准优先级截断
    if len(ordered) > limit:
        must = {0, len(values) - 1}
        if base_value is not None and len(values) > 2:
            must.add(min(range(len(values)), key=lambda index: abs(values[index] - base_value)))
        optional = [index for index in ordered if index not in must]
        keep = list(sorted(must)) + optional[: max(0, limit - len(must))]
        ordered = sorted(keep)
    return tuple(values[index] for index in ordered)


def normalize_weight_space(
    space: Mapping[str, Any],
    *,
    min_axes: int = 3,
) -> dict[str, dict[str, float]]:
    """校验权重搜索范围，返回保持 AI 优先级顺序的规范化字典。"""
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
    """校验自定义引擎的参数搜索范围。"""
    if not isinstance(space, Mapping) or not space:
        raise TuningError("param_space 必须是非空对象（参数名 -> {min,max,step,type}）")
    if len(space) > 4:
        raise TuningError("param_space 最多支持 4 个参数（演示场景下保证寻优时长可控）")
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
    base_params: Mapping[str, Any],
    per_axis: int,
    labels: Mapping[str, str] | None,
) -> list[Axis]:
    axes: list[Axis] = []
    for key, spec in space.items():
        integer = kind == "param" and str(spec.get("type")) == "int"
        minimum = float(spec["min"])
        maximum = float(spec["max"])
        step = float(spec["step"])
        values = _grid_values(minimum, maximum, step, integer)
        raw_base = base_params.get(key) if isinstance(base_params, Mapping) else None
        try:
            base_value = float(raw_base) if raw_base is not None else None
        except (TypeError, ValueError):
            base_value = None
        if base_value is not None and not (minimum - _EPS <= base_value <= maximum + _EPS):
            base_value = None
        candidates = _select_candidates(values, base_value, per_axis)
        label = str((labels or {}).get(key) or key)
        axes.append(Axis(key=key, label=label, kind=kind, integer=integer,
                         minimum=minimum, maximum=maximum, step=step,
                         candidates=candidates))
    return axes


def _axis_budget(axis_count: int, max_trials: int) -> tuple[int, list[int]]:
    """根据总评估预算，决定每个轴取几个候选、保留几个轴。

    返回 ``(per_axis, kept_axis_indexes)``；总评估次数 <= max_trials。
    """
    axis_count = max(1, axis_count)
    budget = max(1, int(max_trials) - 1)  # 减去基线
    per_axis = max(2, budget // axis_count + 1)
    per_axis = min(MAX_CANDIDATES_PER_AXIS, per_axis)
    # 每个轴多一个候选就多 axis_count 次评估，超预算时减少候选数
    while per_axis > 2 and (per_axis - 1) * axis_count > budget:
        per_axis -= 1
    max_axes = max(1, budget // max(1, per_axis - 1))
    kept = list(range(min(axis_count, max_axes)))
    return per_axis, kept


# ---------------------------------------------------------------------------
# 指标打分
# ---------------------------------------------------------------------------


def _safe_metric(metrics: Mapping[str, Any], key: str) -> float:
    try:
        value = float(metrics.get(key))
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def score_metrics(metrics: Mapping[str, Any], objective: str,
                  tolerance: float | None = None) -> tuple[float, bool]:
    """把回测指标折算成可比较的分数（越大越好）。

    - cagr / sharpe：直接取指标；
    - mdd：最大回撤是负数，越接近 0 越好，因此取原值比较；
    - 若客户给出最大回撤上限而候选超限，则按超出幅度扣分，避免选出
      收益高但风险失控的参数。
    """
    if objective not in OBJECTIVES:
        raise TuningError(f"未知优化目标 {objective!r}；可选 {', '.join(OBJECTIVES)}")
    score = _safe_metric(metrics, objective)
    constraint_ok = True
    if tolerance is not None and objective != "mdd":
        try:
            drawdown = abs(float(metrics.get("mdd")))
        except (TypeError, ValueError):
            drawdown = 0.0
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
    max_trials: int = 28,
    tolerance: float | None = None,
    kind: str = "weight",
    labels: Mapping[str, str] | None = None,
    method: str = "",
) -> Iterator[dict[str, Any]]:
    """执行逐轴网格搜索，逐个 ``yield`` 进度事件。

    ``evaluate(params)`` 必须返回 dict，至少包含 ``metrics``；可附带
    ``params``（评估时实际使用的参数，例如归一化后的权重）与 ``nav``。
    评估抛异常的组合会被跳过；全部组合都失败时抛 :class:`TuningError`。

    事件类型：
    - ``{"event": "start", ...}``：搜索开始，含范围、目标、总评估次数；
    - ``{"event": "trial", ...}``：完成一次评估，含当前/最优指标与进度；
    - ``{"event": "done", ...}``：搜索结束，含最优参数、指标、耗时。
    """
    if kind not in ("weight", "param"):
        raise TuningError(f"未知搜索类型 {kind!r}")
    if objective not in OBJECTIVES:
        raise TuningError(f"未知优化目标 {objective!r}")
    try:
        max_trials_i = int(max_trials)
    except (TypeError, ValueError) as exc:
        raise TuningError("max_trials 必须是整数") from exc
    max_trials_i = max(MIN_TRIALS, min(MAX_TRIALS, max_trials_i))

    if kind == "weight":
        normalized_space = normalize_weight_space(space)
        base = {name: float((base_params or {}).get(name, 0.0)) for name in FACTOR_NAMES}
        default_method = "逐因子网格搜索"
    else:
        normalized_space = normalize_param_space(space)
        base = {}
        for key, spec in normalized_space.items():
            value = float(spec.get("default", (spec["min"] + spec["max"]) / 2.0))
            base[key] = int(round(value)) if spec.get("type") == "int" else value
        default_method = "参数网格搜索"

    per_axis, kept_indexes = _axis_budget(len(normalized_space), max_trials_i)
    kept_space = {key: normalized_space[key] for key in list(normalized_space)[: len(kept_indexes)]}
    axes = _build_axes(kept_space, kind=kind, base_params=base,
                       per_axis=per_axis, labels=labels)
    if not axes:
        raise TuningError("搜索范围为空，无法寻优")

    def planned_total(axis_list: list[Axis]) -> int:
        """总评估数 = 基线 + 每个轴除「当前值」以外的候选数。"""
        count = 1
        for axis in axis_list:
            skip = any(_same_value(value, base.get(axis.key), axis.integer)
                       for value in axis.candidates)
            count += len(axis.candidates) - (1 if skip else 0)
        return count

    # 若当前值不在 AI 给定范围内，预估会多出几次评估；硬性裁剪到预算内
    while planned_total(axes) > max_trials_i:
        trimmable = [axis for axis in axes if len(axis.candidates) > 2]
        if trimmable:
            target = max(trimmable, key=lambda axis: len(axis.candidates))
            index = axes.index(target)
            middle = len(target.candidates) // 2
            candidates = tuple(value for position, value in enumerate(target.candidates)
                               if position != middle)
            axes[index] = Axis(key=target.key, label=target.label, kind=target.kind,
                               integer=target.integer, minimum=target.minimum,
                               maximum=target.maximum, step=target.step,
                               candidates=candidates)
        elif len(axes) > 1:
            axes.pop()
        else:
            break
    total = planned_total(axes)

    started = time.time()
    best: dict[str, Any] | None = None
    trials: list[dict[str, Any]] = []
    failures = 0
    last_error = ""

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
        score, constraint_ok = score_metrics(metrics, objective, tolerance)
        best = make_best(effective, metrics, score, constraint_ok,
                         extra={"nav": raw.get("nav")})
        best["metric_value"] = _safe_metric(metrics, objective)
        return best, ""

    yield {
        "event": "start",
        "method": method or default_method,
        "objective": objective,
        "objective_label": OBJECTIVE_LABELS.get(objective, objective),
        "space": [axis.public() for axis in axes],
        "total": total,
        "note": "AI 只给定搜索范围，具体参数由服务端网格寻优逐个回测选出。",
    }

    baseline, error = evaluate_one(base)
    trial_index = 0
    if baseline is not None:
        trial_index = 1
        baseline["trial"] = 1
        best = baseline
        trials.append(baseline)
        yield {
            "event": "trial",
            "trial": 1,
            "total": total,
            "axis": "",
            "label": "当前参数（基线）",
            "params": baseline["params"],
            "metrics": baseline["metrics"],
            "score": baseline["score"],
            "metric_value": baseline.get("metric_value"),
            "best": _public_best(best),
            "progress": round(1 / total, 4),
        }
    elif last_error:
        error = last_error

    for axis in axes:
        if best is None:
            break
        current_params = dict(best["params"])
        for candidate in axis.candidates:
            if trial_index >= total:
                break
            if _same_value(candidate, current_params.get(axis.key), axis.integer):
                continue
            trial_params = dict(current_params)
            if axis.integer:
                trial_params[axis.key] = int(round(candidate))
            else:
                trial_params[axis.key] = float(candidate)
            result, error = evaluate_one(trial_params)
            trial_index += 1
            if result is None:
                yield {
                    "event": "trial",
                    "trial": trial_index,
                    "total": total,
                    "axis": axis.key,
                    "label": _candidate_label(axis, candidate),
                    "error": error,
                    "best": _public_best(best),
                    "progress": round(trial_index / total, 4),
                }
                continue
            result["trial"] = trial_index
            trials.append(result)
            improved = best is None or float(result["score"]) > float(best["score"]) + _EPS
            if improved:
                best = result
            yield {
                "event": "trial",
                "trial": trial_index,
                "total": total,
                "axis": axis.key,
                "label": _candidate_label(axis, candidate),
                "params": result["params"],
                "metrics": result["metrics"],
                "score": result["score"],
                "metric_value": result.get("metric_value"),
                "improved": improved,
                "best": _public_best(best),
                "progress": round(trial_index / total, 4),
            }

    if best is None:
        detail = f"（最后一次错误：{error or last_error}）" if (error or last_error) else ""
        raise TuningError(f"所有参数组合回测都失败了{detail}")

    yield {
        "event": "done",
        "method": method or default_method,
        "objective": objective,
        "objective_label": OBJECTIVE_LABELS.get(objective, objective),
        "space": [axis.public() for axis in axes],
        "total": total,
        "evaluated": trial_index,
        "failures": failures,
        "elapsed": round(time.time() - started, 3),
        "best": best,
        "trials": trials,
        "note": "寻优结果来自历史回测，最优参数可能过拟合；候选仍需人工确认后才会启用。",
    }


def _candidate_label(axis: Axis, candidate: float) -> str:
    if axis.integer:
        return f"{axis.label} = {int(round(candidate))}"
    return f"{axis.label} = {float(candidate):.4g}"


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
