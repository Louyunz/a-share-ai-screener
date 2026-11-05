"""A 股 10 因子计算、去极值、标准化、行业中性化与权重注册表。

本模块是纯函数库，只依赖 numpy / pandas / 标准库，禁止依赖后端其他模块。

处理流程（横截面，每个交易日独立计算）：
1. 从行情与财务数据计算 10 个原始因子 ``raw_<name>``；
2. 对每个因子做 1%/99% 去极值；
3. 按行业分组做 z-score；行业样本 < 3 时退化为全市场 z-score；
4. 乘以因子方向 ``FACTOR_DIRECTION``，保证数值越大越好（``z_<name>``）；
5. 用权重加权求和得到 ``factor_score``；缺失因子的 z 一律置 0。

说明：``raw_<name>`` 返回的是未去极值的原始因子值（含缺失 NaN），
便于前端展示“因子贡献”时追溯；``z_<name>`` 才是最终参与打分的值。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 常量定义（契约冻结，禁止随意改动名称/方向）
# ---------------------------------------------------------------------------

FACTOR_NAMES = [
    "ep",
    "bp",
    "roe",
    "ocf_to_profit",
    "revenue_growth",
    "momentum_60",
    "reversal_5",
    "volatility_60",
    "liquidity_20",
    "size",
]

FACTOR_DIRECTION = {
    "ep": 1,
    "bp": 1,
    "roe": 1,
    "ocf_to_profit": 1,
    "revenue_growth": 1,
    "momentum_60": 1,
    "reversal_5": -1,
    "volatility_60": -1,
    "liquidity_20": -1,
    "size": -1,
}

# 单因子权重上限（tuner 采样与权重校验共用）
MAX_WEIGHT = 0.4
# 权重求和误差容忍度
WEIGHT_TOL = 1e-6
# 年化交易日数
TRADING_DAYS = 252
# 波动率计算最少需要的日收益样本数
MIN_VOL_OBS = 20

RISK_WEIGHTS: dict[str, dict[str, float]] = {
    "conservative": {
        "ep": 0.12,
        "bp": 0.10,
        "roe": 0.16,
        "ocf_to_profit": 0.12,
        "revenue_growth": 0.06,
        "momentum_60": 0.06,
        "reversal_5": 0.06,
        "volatility_60": 0.16,
        "liquidity_20": 0.10,
        "size": 0.06,
    },
    "balanced": {
        "ep": 0.12,
        "bp": 0.10,
        "roe": 0.12,
        "ocf_to_profit": 0.10,
        "revenue_growth": 0.10,
        "momentum_60": 0.10,
        "reversal_5": 0.08,
        "volatility_60": 0.10,
        "liquidity_20": 0.10,
        "size": 0.08,
    },
    "aggressive": {
        "ep": 0.08,
        "bp": 0.06,
        "roe": 0.08,
        "ocf_to_profit": 0.06,
        "revenue_growth": 0.16,
        "momentum_60": 0.18,
        "reversal_5": 0.12,
        "volatility_60": 0.06,
        "liquidity_20": 0.10,
        "size": 0.10,
    },
}

DEFAULT_WEIGHTS: dict[str, float] = dict(RISK_WEIGHTS["balanced"])

# 用户可选的权重预设（含三档风险偏好 + 常见风格）
WEIGHT_PRESETS: dict[str, dict[str, float]] = {
    "conservative": dict(RISK_WEIGHTS["conservative"]),
    "balanced": dict(RISK_WEIGHTS["balanced"]),
    "aggressive": dict(RISK_WEIGHTS["aggressive"]),
    "value": {
        "ep": 0.22,
        "bp": 0.18,
        "roe": 0.10,
        "ocf_to_profit": 0.08,
        "revenue_growth": 0.04,
        "momentum_60": 0.04,
        "reversal_5": 0.08,
        "volatility_60": 0.10,
        "liquidity_20": 0.10,
        "size": 0.06,
    },
    "growth": {
        "ep": 0.06,
        "bp": 0.04,
        "roe": 0.10,
        "ocf_to_profit": 0.06,
        "revenue_growth": 0.24,
        "momentum_60": 0.20,
        "reversal_5": 0.06,
        "volatility_60": 0.06,
        "liquidity_20": 0.08,
        "size": 0.10,
    },
    "quality": {
        "ep": 0.10,
        "bp": 0.08,
        "roe": 0.24,
        "ocf_to_profit": 0.20,
        "revenue_growth": 0.08,
        "momentum_60": 0.06,
        "reversal_5": 0.04,
        "volatility_60": 0.10,
        "liquidity_20": 0.06,
        "size": 0.04,
    },
}

__all__ = [
    "FACTOR_NAMES",
    "FACTOR_DIRECTION",
    "RISK_WEIGHTS",
    "DEFAULT_WEIGHTS",
    "WEIGHT_PRESETS",
    "MAX_WEIGHT",
    "TRADING_DAYS",
    "weights_for",
    "validate_weights",
    "register_preset",
    "compute_factor_scores",
]


# ---------------------------------------------------------------------------
# 权重注册表
# ---------------------------------------------------------------------------


def _copy_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """按标准因子顺序复制一份权重，返回普通 float dict。"""
    return {name: float(weights[name]) for name in FACTOR_NAMES}


def weights_for(risk_level: str) -> dict[str, float]:
    """按风险等级取权重；未知等级回退到 ``balanced``（契约要求）。"""
    key = str(risk_level).strip().lower() if risk_level is not None else ""
    selected = RISK_WEIGHTS.get(key, RISK_WEIGHTS["balanced"])
    return dict(selected)


def validate_weights(weights: dict) -> None:
    """校验 10 项因子权重。

    要求：10 项齐全、每项为有限数值且 ∈ [0, 0.4]、总和 = 1 ± 1e-6。
    不满足时抛 ``ValueError``，错误信息指出具体原因。
    """
    if not isinstance(weights, Mapping):
        raise ValueError("weights 必须是 dict（因子名 -> 权重）")

    missing = [name for name in FACTOR_NAMES if name not in weights]
    if missing:
        raise ValueError(f"权重缺少因子: {', '.join(missing)}")

    values: list[float] = []
    for name in FACTOR_NAMES:
        try:
            value = float(weights[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"因子 {name} 的权重不是数值: {weights[name]!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"因子 {name} 的权重不是有限数值: {value!r}")
        if value < -WEIGHT_TOL:
            raise ValueError(f"因子 {name} 的权重不能为负: {value}")
        if value > MAX_WEIGHT + WEIGHT_TOL:
            raise ValueError(f"因子 {name} 的权重超过上限 {MAX_WEIGHT}: {value}")
        values.append(value)

    total = math.fsum(values)
    if abs(total - 1.0) > WEIGHT_TOL:
        raise ValueError(f"因子权重之和必须为 1（当前 {total:.10f}）")


def register_preset(name: str, weights: dict) -> None:
    """注册（或覆盖）一个权重预设；先校验权重再写入注册表。"""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("预设名称必须是非空字符串")
    validate_weights(weights)
    WEIGHT_PRESETS[name.strip()] = _copy_weights(weights)


# ---------------------------------------------------------------------------
# 数据清洗辅助
# ---------------------------------------------------------------------------


def _clean_frame(frame, label: str) -> pd.DataFrame:
    """校验宽表并统一为按日期升序、索引唯一的 DataFrame。"""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{label} 必须是 pandas.DataFrame")
    if frame.empty:
        raise ValueError(f"{label} 为空，无法计算因子")
    if frame.columns.has_duplicates:
        raise ValueError(f"{label} 存在重复的股票代码列")

    out = frame.copy()
    try:
        out.index = pd.DatetimeIndex(pd.to_datetime(out.index))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 的 index 必须是可解析的日期") from exc
    if out.index.has_duplicates:
        out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    return out


def _clean_cross_section(frame, label: str) -> pd.DataFrame:
    """校验横截面表（index=代码），并去除重复代码。"""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{label} 必须是 pandas.DataFrame")
    if frame.empty:
        raise ValueError(f"{label} 为空，无法计算因子")
    out = frame.copy()
    if out.index.has_duplicates:
        out = out[~out.index.duplicated(keep="last")]
    return out


def _asof_cutoff(asof, index: pd.DatetimeIndex) -> Optional[pd.Timestamp]:
    """把 ``asof`` 解析为与行情索引时区兼容的 Timestamp。"""
    if asof is None:
        return None
    try:
        cutoff = pd.Timestamp(asof)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"asof 不是合法日期: {asof!r}") from exc
    if cutoff.tzinfo is not None and index.tz is None:
        cutoff = cutoff.tz_localize(None)
    elif cutoff.tzinfo is None and index.tz is not None:
        cutoff = cutoff.tz_localize(index.tz)
    return cutoff


def _industry_series(industry, codes: pd.Index) -> pd.Series:
    """从 DataFrame/Series 中取出行业列并 reindex 到股票代码。"""
    if isinstance(industry, pd.DataFrame):
        if "industry" not in industry.columns:
            raise ValueError("industry 必须包含 'industry' 列")
        series = industry["industry"]
    elif isinstance(industry, pd.Series):
        series = industry
    else:
        raise ValueError("industry 必须是 DataFrame 或 Series")

    series = series.copy()
    if series.index.has_duplicates:
        series = series[~series.index.duplicated(keep="last")]
    series = series.reindex(codes).astype("object")
    series = series.where(series.notna(), "未知")
    series = series.map(lambda value: str(value).strip() if str(value).strip() else "未知")
    return series


# ---------------------------------------------------------------------------
# 因子横截面处理
# ---------------------------------------------------------------------------


def _winsorize(series: pd.Series) -> pd.Series:
    """1%/99% 分位去极值；有效样本不足 3 个时原样返回。"""
    values = pd.to_numeric(series, errors="coerce").astype(float)
    valid = values.dropna()
    if len(valid) < 3:
        return values
    low, high = np.nanpercentile(valid.to_numpy(), [1.0, 99.0])
    if not (np.isfinite(low) and np.isfinite(high)) or high < low:
        return values
    return values.clip(lower=low, upper=high)


def _zscore(series: pd.Series) -> pd.Series:
    """横截面 z-score（总体标准差）；常量因子全部记 0，缺失记 0。"""
    values = pd.to_numeric(series, errors="coerce").astype(float)
    mean = values.mean(skipna=True)
    std = values.std(ddof=0, skipna=True)
    if pd.isna(mean) or pd.isna(std) or float(std) <= 1e-12:
        return pd.Series(0.0, index=values.index, dtype=float)
    z = (values - float(mean)) / float(std)
    return z.fillna(0.0)


def _industry_neutral_z(series: pd.Series, industry: pd.Series) -> pd.Series:
    """行业中性化：行业内 z-score；行业样本 < 3 时用全市场 z。"""
    market_z = _zscore(series)
    result = pd.Series(np.nan, index=series.index, dtype=float)

    for _, idx in series.groupby(industry, sort=False).groups.items():
        group_values = series.loc[idx]
        valid_count = int(group_values.notna().sum())
        if valid_count >= 3:
            mean = group_values.mean(skipna=True)
            std = group_values.std(ddof=0, skipna=True)
            if pd.isna(mean) or pd.isna(std) or float(std) <= 1e-12:
                # 行业内因子完全一致：不给任何股票额外加分
                result.loc[idx] = 0.0
            else:
                result.loc[idx] = (group_values - float(mean)) / float(std)
        else:
            # 小行业样本不足，整组退化为全市场 z（契约要求）
            result.loc[idx] = market_z.loc[idx]

    # 原始因子缺失的股票 z 记 0（契约要求），而不是做插补
    return result.fillna(0.0)


def _trailing_return(prices: pd.DataFrame, periods: int) -> pd.Series:
    """近 ``periods`` 个交易日收益（用最后一行与之前第 periods 行比较）。"""
    if len(prices) <= periods:
        return pd.Series(np.nan, index=prices.columns, dtype=float)
    latest = prices.iloc[-1]
    prior = prices.iloc[-1 - periods]
    with np.errstate(divide="ignore", invalid="ignore"):
        return latest / prior - 1.0


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------


def compute_factor_scores(
    prices: pd.DataFrame,
    close_raw: pd.DataFrame,
    amount: pd.DataFrame,
    fundamentals: pd.DataFrame,
    industry: pd.DataFrame,
    weights: Optional[dict] = None,
    asof=None,
    min_history: int = 120,
) -> pd.DataFrame:
    """计算截至 ``asof``（含）的横截面因子得分。

    Parameters
    ----------
    prices:
        前复权收盘价宽表，index=交易日，columns=股票代码。
    close_raw:
        未复权收盘价宽表（本函数不使用，保留签名以对齐数据契约）。
    amount:
        成交额宽表，用于 ``liquidity_20``。
    fundamentals:
        index=代码，至少包含 ep/bp/roe/ocf_to_profit/revenue_growth/float_mv。
    industry:
        DataFrame（含 ``industry`` 列）或 Series，index=代码。
    weights:
        10 因子权重；``None`` 时使用 ``DEFAULT_WEIGHTS``。
    asof:
        截止日（含）；``None`` 表示使用行情最后一天。
    min_history:
        有效行情天数下限，不足的股票剔除。

    Returns
    -------
    pd.DataFrame
        index=代码，按 ``factor_score`` 降序；列包含
        ``raw_<name>``、``z_<name>``（行业中性化 + 方向对齐后）与 ``factor_score``。

    Raises
    ------
    ValueError
        行情为空、截止日前无数据、整体长度低于 ``min_history``、
        财务/行业字段缺失或权重非法。
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    validate_weights(weights)

    if not isinstance(min_history, (int, np.integer)) or isinstance(min_history, bool):
        raise ValueError("min_history 必须是正整数")
    if int(min_history) < 1:
        raise ValueError("min_history 必须 >= 1")

    prices = _clean_frame(prices, "prices")
    if len(prices) < int(min_history):
        raise ValueError(
            f"行情长度 {len(prices)} 不足 min_history={min_history}，无法计算因子"
        )

    cutoff = _asof_cutoff(asof, prices.index)
    if cutoff is not None:
        prices = prices.loc[prices.index <= cutoff]
        if prices.empty:
            raise ValueError(f"asof={asof!r} 之前没有任何行情数据")

    # 对齐成交额与不复权价（仅用 amount；close_raw 保留校验以尽早暴露坏数据）
    amount = _clean_frame(amount, "amount").reindex(
        index=prices.index, columns=prices.columns
    )
    close_raw = _clean_frame(close_raw, "close_raw")
    if close_raw.empty:
        raise ValueError("close_raw 为空")

    # 有效行情天数不足的股票剔除
    history_count = prices.notna().sum(axis=0)
    eligible = history_count[history_count >= int(min_history)].index
    if len(eligible) == 0:
        raise ValueError(
            f"没有股票满足 min_history={min_history} 个有效行情日，请检查数据或 asof"
        )
    prices = prices.loc[:, eligible]

    # 财务横截面
    fundamentals = _clean_cross_section(fundamentals, "fundamentals").reindex(eligible)
    required_fundamental = [
        "ep",
        "bp",
        "roe",
        "ocf_to_profit",
        "revenue_growth",
        "float_mv",
    ]
    missing_cols = [col for col in required_fundamental if col not in fundamentals.columns]
    if missing_cols:
        raise ValueError(f"fundamentals 缺少列: {', '.join(missing_cols)}")

    def fundamental_col(name: str) -> pd.Series:
        return pd.to_numeric(fundamentals[name], errors="coerce").astype(float)

    industry_series = _industry_series(industry, eligible)

    # ---- 1) 原始因子 ----
    raw = pd.DataFrame(index=eligible, dtype=float)

    for name in ("ep", "bp", "roe", "ocf_to_profit", "revenue_growth"):
        raw[name] = fundamental_col(name)

    raw["momentum_60"] = _trailing_return(prices, 60)
    raw["reversal_5"] = _trailing_return(prices, 5)

    # 近 60 日年化波动率：需要足够多的有效日收益
    vol_window = prices.iloc[-61:]
    vol_returns = vol_window.pct_change(fill_method=None).iloc[1:]
    vol = vol_returns.std(ddof=1, skipna=True) * math.sqrt(TRADING_DAYS)
    valid_vol_obs = vol_returns.notna().sum(axis=0)
    raw["volatility_60"] = vol.where(valid_vol_obs >= MIN_VOL_OBS)

    # 近 20 日日均成交额取对数（0/NaN 视为无成交，用有效正值求均值）
    amount_window = amount.iloc[-20:]
    positive_amount = amount_window.where(amount_window > 0)
    avg_amount = positive_amount.mean(skipna=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw["liquidity_20"] = np.log(avg_amount.where(avg_amount > 0))

    # 流通市值取对数
    float_mv = fundamental_col("float_mv")
    with np.errstate(divide="ignore", invalid="ignore"):
        raw["size"] = np.log(float_mv.where(float_mv > 0))

    # ---- 2) 去极值 + 行业中性化 + 方向对齐 ----
    z_aligned = pd.DataFrame(index=eligible, dtype=float)
    for name in FACTOR_NAMES:
        winsorized = _winsorize(raw[name])
        neutral = _industry_neutral_z(winsorized, industry_series)
        z_aligned[name] = neutral * float(FACTOR_DIRECTION[name])

    # ---- 3) 合成因子得分 ----
    factor_score = pd.Series(0.0, index=eligible, dtype=float)
    for name in FACTOR_NAMES:
        factor_score += float(weights[name]) * z_aligned[name]

    result = pd.DataFrame(index=eligible)
    for name in FACTOR_NAMES:
        result[f"raw_{name}"] = raw[name].astype(float)
        result[f"z_{name}"] = z_aligned[name].astype(float)
    result["factor_score"] = factor_score.astype(float)

    result.index.name = "code"
    result = result.sort_values("factor_score", ascending=False, kind="stable")
    return result
