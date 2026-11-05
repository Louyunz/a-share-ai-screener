"""因子权重随机搜索（tuner）。

对 10 项因子权重做带约束的随机采样：
- 每项 ∈ [0, 0.4]，总和 = 1；
- 目标 J = sharpe - 0.5 * |mdd| + 0.2 * ir；
- 每个 trial 同时计算全区间、IS（前 70%）、OOS（后 30%）指标；
- 防过拟合门槛：候选的 IS CAGR >= 0.8 * 基准 IS CAGR；
- 每完成一组通过 ``progress_cb`` 回调一次（字段严格按契约）。

本模块只依赖 numpy / pandas / 标准库与同目录 factors / selection 模块。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from .factors import DEFAULT_WEIGHTS, FACTOR_NAMES, MAX_WEIGHT, validate_weights
from .selection import MIN_BACKTEST_DAYS, run_backtest

# 因子计算所需的预热交易日数（与 factors.compute_factor_scores 的默认 min_history 一致）
WARMUP_DAYS = 120
# IS / OOS 切分比例
IS_RATIO = 0.7
# tuner 至少需要覆盖的评估区间长度
MIN_EVAL_DAYS = 80
# 采样接受上限（防止极端随机数导致死循环）
_MAX_SAMPLE_ATTEMPTS = 100_000
# 新最优必须有可感知的提升，避免浮点噪声反复触发 is_best
_IMPROVE_EPS = 1e-12

__all__ = ["run_tuning", "WARMUP_DAYS", "IS_RATIO", "MIN_EVAL_DAYS"]


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _get(data, name: str):
    """从 DataBundle 对象或 dict 中读取字段（duck typing）。"""
    if isinstance(data, Mapping):
        if name not in data:
            raise ValueError(f"data 缺少字段: {name}")
        value = data[name]
    else:
        try:
            value = getattr(data, name)
        except AttributeError as exc:
            raise ValueError(f"data 缺少字段: {name}") from exc
    if value is None:
        raise ValueError(f"data.{name} 不能为 None")
    return value


def _positive_int(value, name: str) -> int:
    """校验正整数参数。"""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} 必须是正整数")
    number = int(value)
    if number < 1:
        raise ValueError(f"{name} 必须 >= 1")
    return number


def _timestamp(value, index: pd.DatetimeIndex) -> pd.Timestamp:
    """解析时间并与行情索引时区对齐。"""
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"非法日期: {value!r}") from exc
    if stamp.tzinfo is not None and index.tz is None:
        stamp = stamp.tz_localize(None)
    elif stamp.tzinfo is None and index.tz is not None:
        stamp = stamp.tz_localize(index.tz)
    return stamp


def _trial_metrics(metrics: dict) -> dict[str, float]:
    """按契约裁剪 trial 指标字段（cagr/sharpe/mdd/ir）。"""
    return {
        "cagr": float(metrics["cagr"]),
        "sharpe": float(metrics["sharpe"]),
        "mdd": float(metrics["mdd"]),
        "ir": float(metrics["ir"]),
    }


def _objective(metrics: dict) -> float:
    """目标函数 J = sharpe - 0.5*|mdd| + 0.2*ir（契约公式）。"""
    return float(
        float(metrics["sharpe"])
        - 0.5 * abs(float(metrics["mdd"]))
        + 0.2 * float(metrics["ir"])
    )


def _sample_weights(rng: np.random.Generator) -> dict[str, float]:
    """在单纯形上均匀采样，并拒绝超出 [0, 0.4] 的样本。

    Dirichlet(1,...,1) 等价于归一化的均匀随机向量；被接受后总和严格为 1。
    """
    for _ in range(_MAX_SAMPLE_ATTEMPTS):
        weights = rng.dirichlet(np.ones(len(FACTOR_NAMES)), size=1)[0]
        if np.all(weights <= MAX_WEIGHT + 1e-12):
            weights = np.minimum(weights, MAX_WEIGHT)
            weights = weights / weights.sum()
            return {name: float(value) for name, value in zip(FACTOR_NAMES, weights)}
    raise RuntimeError("权重随机采样失败：无法在约束内找到可行解")


def _resolve_dates(prices: pd.DataFrame, start, end) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """解析回测评估区间，默认从预热 120 日后的第一个交易日开始。"""
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("prices 为空，无法调参")
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(prices.index))
    except (TypeError, ValueError) as exc:
        raise ValueError("prices 的 index 必须是可解析的日期") from exc
    if dates.has_duplicates:
        dates = dates[~dates.duplicated(keep="last")]
    dates = dates.sort_values()

    if start is None:
        if len(dates) < WARMUP_DAYS + MIN_EVAL_DAYS:
            raise ValueError(
                f"数据只有 {len(dates)} 个交易日，调参至少需要 "
                f"{WARMUP_DAYS} 日预热 + {MIN_EVAL_DAYS} 日评估"
            )
        start_stamp = dates[WARMUP_DAYS]
    else:
        start_stamp = _timestamp(start, dates)

    end_stamp = dates[-1] if end is None else _timestamp(end, dates)
    if start_stamp > end_stamp:
        raise ValueError(f"调参起点 {start_stamp} 晚于终点 {end_stamp}")

    window = dates[(dates >= start_stamp) & (dates <= end_stamp)]
    if len(window) < MIN_EVAL_DAYS:
        raise ValueError(
            f"调参评估区间只有 {len(window)} 个交易日，至少需要 {MIN_EVAL_DAYS} 个"
        )
    return dates, window


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def run_tuning(
    data,
    base_weights: dict | None = None,
    trials: int = 60,
    top_n: int = 10,
    start=None,
    end=None,
    seed: int = 42,
    progress_cb: Callable[[dict], None] | None = None,
) -> dict:
    """随机搜索因子权重，返回最优权重与全部 trial 结果。

    Parameters
    ----------
    data:
        DataBundle 或 dict。
    base_weights:
        基准权重，默认 ``DEFAULT_WEIGHTS``（balanced）。
    trials:
        随机采样组数。
    top_n:
        每期持股数，透传给 ``run_backtest``。
    start, end:
        评估区间；``start=None`` 时从预热 120 日后的首个交易日起。
    seed:
        随机种子，保证可复现。
    progress_cb:
        每完成一组回调一次，参数为契约规定的 trial dict。

    Returns
    -------
    dict
        ``{"best_weights", "base_metrics", "best_metrics", "objective", "trials"}``。

    Notes
    -----
    契约未规定目标函数使用哪个区间的指标，这里按字面取 ``start..end`` 全区间
    回测指标计算 J；同时每个 trial 都额外计算 IS/OOS 指标，IS CAGR 作为防过拟合门槛。
    """
    trials = _positive_int(trials, "trials")
    top_n = _positive_int(top_n, "top_n")
    if progress_cb is not None and not callable(progress_cb):
        raise ValueError("progress_cb 必须是可调用对象")

    if base_weights is None:
        base_weights = dict(DEFAULT_WEIGHTS)
    validate_weights(base_weights)
    base_weights = {name: float(base_weights[name]) for name in FACTOR_NAMES}

    prices = _get(data, "prices")
    _, window_dates = _resolve_dates(prices, start, end)

    split = int(len(window_dates) * IS_RATIO)
    if split < MIN_BACKTEST_DAYS or len(window_dates) - split < MIN_BACKTEST_DAYS:
        raise ValueError(
            "IS/OOS 切分后区间过短："
            f"IS={split} 日，OOS={len(window_dates) - split} 日，"
            f"各至少需要 {MIN_BACKTEST_DAYS} 日"
        )
    window_start = window_dates[0]
    is_end = window_dates[split - 1]
    oos_start = window_dates[split]
    window_end = window_dates[-1]

    # ---- 基准表现（全区间 + IS，用于 stability gate）----
    base_backtest = run_backtest(
        data, base_weights, start=window_start, end=window_end, top_n=top_n
    )
    base_is_backtest = run_backtest(
        data, base_weights, start=window_start, end=is_end, top_n=top_n
    )
    base_metrics = dict(base_backtest["metrics"])
    base_objective = _objective(base_metrics)
    base_is_cagr = float(base_is_backtest["metrics"]["cagr"])

    best_weights = dict(base_weights)
    best_metrics = dict(base_metrics)
    best_objective = base_objective
    records: list[dict] = []

    rng = np.random.default_rng(seed)

    for trial_index in range(trials):
        weights = _sample_weights(rng)

        # 全区间指标：用于目标函数与前端展示
        backtest = run_backtest(
            data, weights, start=window_start, end=window_end, top_n=top_n
        )
        metrics = _trial_metrics(backtest["metrics"])
        objective = _objective(metrics)

        # 防过拟合：同时计算 IS / OOS 指标
        is_backtest = run_backtest(
            data, weights, start=window_start, end=is_end, top_n=top_n
        )
        oos_backtest = run_backtest(
            data, weights, start=oos_start, end=window_end, top_n=top_n
        )
        is_cagr = float(is_backtest["metrics"]["cagr"])
        oos_cagr = float(oos_backtest["metrics"]["cagr"])
        if not (math.isfinite(is_cagr) and math.isfinite(oos_cagr)):
            raise ValueError("调参过程中 IS/OOS CAGR 出现非有限值")
        stable = is_cagr >= 0.8 * base_is_cagr

        is_best = bool(stable and objective > best_objective + _IMPROVE_EPS)
        if is_best:
            best_weights = dict(weights)
            best_metrics = dict(backtest["metrics"])
            best_objective = objective

        trial = {
            "trial_index": int(trial_index),
            "weights": dict(weights),
            "metrics": dict(metrics),
            "objective": float(objective),
            "is_best": bool(is_best),
        }
        records.append(trial)
        if progress_cb is not None:
            progress_cb(
                {
                    "trial_index": int(trial_index),
                    "weights": dict(weights),
                    "metrics": dict(metrics),
                    "objective": float(objective),
                    "is_best": bool(is_best),
                }
            )

    return {
        "best_weights": dict(best_weights),
        "base_metrics": dict(base_metrics),
        "best_metrics": dict(best_metrics),
        "objective": float(best_objective),
        "trials": records,
    }
