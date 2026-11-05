"""选股筛选、组合构建与 A 股规则回测。

核心函数：
- ``screen``：按因子权重对股票池打分并取 Top N；
- ``build_portfolio``：把 Top N 转成等权或风险平价组合；
- ``run_backtest``：按 A 股 T+1、停牌、涨跌停、双边成本规则回测。

本模块只依赖 numpy / pandas / 标准库与同目录 factors 模块。
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .factors import (
    DEFAULT_WEIGHTS,
    FACTOR_NAMES,
    TRADING_DAYS,
    compute_factor_scores,
    validate_weights,
)

# 回测最少需要的交易日数（再短指标没有统计意义）
MIN_BACKTEST_DAYS = 20
# A 股主板涨跌停判定阈值（契约：±9.8%）
LIMIT_UP = 0.098
LIMIT_DOWN = -0.098
# 订单最小金额占净值比例，避免频繁微调
MIN_TRADE_RATIO = 1e-4
# 风险平价波动率窗口
VOL_WINDOW = 60

__all__ = [
    "screen",
    "build_portfolio",
    "run_backtest",
    "MIN_BACKTEST_DAYS",
]


# ---------------------------------------------------------------------------
# 通用辅助
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


def _finite_number(value, name: str, *, minimum: float | None = None) -> float:
    """校验有限浮点参数。"""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数值") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数值")
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} 必须 >= {minimum}")
    return number


def _to_timestamp(value, index: pd.DatetimeIndex) -> pd.Timestamp:
    """解析时间并与索引时区对齐。"""
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"非法日期: {value!r}") from exc
    if stamp.tzinfo is not None and index.tz is None:
        stamp = stamp.tz_localize(None)
    elif stamp.tzinfo is None and index.tz is not None:
        stamp = stamp.tz_localize(index.tz)
    return stamp


def _industry_lookup(industry, codes: pd.Index) -> pd.Series:
    """取行业映射并补齐缺失行业为“未知”。"""
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
    return series.map(lambda value: str(value).strip() if str(value).strip() else "未知")


def _normalize_exclude(exclude_codes) -> set[str]:
    """把排除列表统一成去空格的字符串集合。"""
    if exclude_codes is None:
        return set()
    if isinstance(exclude_codes, str):
        raw = exclude_codes.split(",")
    else:
        try:
            raw = list(exclude_codes)
        except TypeError as exc:
            raise ValueError("exclude_codes 必须是代码可迭代对象") from exc
    return {str(code).strip() for code in raw if str(code).strip()}


def _clean_date_frame(frame, label: str) -> pd.DataFrame:
    """统一回测宽表：日期索引升序去重、列名唯一。"""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{label} 必须是 pandas.DataFrame")
    if frame.empty:
        raise ValueError(f"{label} 为空，无法回测")
    if frame.columns.has_duplicates:
        raise ValueError(f"{label} 存在重复的股票代码列")

    result = frame.copy()
    try:
        result.index = pd.DatetimeIndex(pd.to_datetime(result.index))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 的 index 必须是可解析的日期") from exc
    if result.index.has_duplicates:
        result = result[~result.index.duplicated(keep="last")]
    return result.sort_index()


def _clean_benchmark(benchmark, index: pd.DatetimeIndex) -> pd.Series:
    """把基准收盘价对齐到行情索引，前向填充后必须无缺失。"""
    if isinstance(benchmark, pd.DataFrame):
        if benchmark.shape[1] != 1:
            raise ValueError("benchmark 必须是单列 Series")
        benchmark = benchmark.iloc[:, 0]
    if not isinstance(benchmark, pd.Series):
        raise ValueError("benchmark 必须是 pandas.Series")
    if benchmark.empty:
        raise ValueError("benchmark 为空，无法回测")

    series = benchmark.copy()
    try:
        series.index = pd.DatetimeIndex(pd.to_datetime(series.index))
    except (TypeError, ValueError) as exc:
        raise ValueError("benchmark 的 index 必须是可解析的日期") from exc
    if series.index.has_duplicates:
        series = series[~series.index.duplicated(keep="last")]
    series = series.sort_index().reindex(index)
    series = series.ffill().bfill()
    if series.isna().any():
        raise ValueError("benchmark 与行情日期无法对齐")
    return series.astype(float)


def _finite(value) -> bool:
    """判断标量是否有限数值。"""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# 筛选
# ---------------------------------------------------------------------------


def screen(
    data,
    weights: dict,
    top_n: int = 10,
    asof=None,
    exclude_codes=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按因子权重筛选 Top N 股票。

    Parameters
    ----------
    data:
        DataBundle 或 dict，需包含 prices/close_raw/amount/fundamentals/industry。
    weights:
        10 因子权重。
    top_n:
        选取数量，必须为正整数。
    asof:
        截止日（含）；None 表示最新交易日。
    exclude_codes:
        需排除的代码集合，也接受逗号分隔字符串（画像里的“排除行业/股票”）。

    Returns
    -------
    (top_df, scores_df)
        ``top_df`` 列固定为 code、score、industry、z_ep...z_size；
        ``scores_df`` 为全市场因子明细（index=code），供图表/调试使用。
    """
    top_n = _positive_int(top_n, "top_n")

    prices = _get(data, "prices")
    close_raw = _get(data, "close_raw")
    amount = _get(data, "amount")
    fundamentals = _get(data, "fundamentals")
    industry = _get(data, "industry")

    scores = compute_factor_scores(
        prices=prices,
        close_raw=close_raw,
        amount=amount,
        fundamentals=fundamentals,
        industry=industry,
        weights=weights,
        asof=asof,
    )

    excluded = _normalize_exclude(exclude_codes)
    if excluded:
        keep_mask = ~scores.index.map(lambda code: str(code).strip()).isin(excluded)
        scores = scores.loc[keep_mask]
    if scores.empty:
        raise ValueError("排除后没有可用股票，无法筛选")

    selected = scores.head(top_n)
    industry_series = _industry_lookup(industry, scores.index)

    top_df = pd.DataFrame(
        {
            "code": selected.index,
            "score": selected["factor_score"].to_numpy(dtype=float),
            "industry": industry_series.reindex(selected.index).to_numpy(),
        }
    )
    for name in FACTOR_NAMES:
        column = f"z_{name}"
        top_df[column] = selected[column].to_numpy(dtype=float)
    top_df.index = pd.RangeIndex(len(top_df))

    return top_df, scores.copy()


# ---------------------------------------------------------------------------
# 组合构建
# ---------------------------------------------------------------------------


def _trailing_volatility(prices: pd.DataFrame, codes: pd.Index, window: int = VOL_WINDOW) -> pd.Series:
    """计算代码截至最新日期的年化波动率（窗口内日收益标准差 * sqrt(252)）。"""
    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("prices 为空，无法计算风险平价权重")
    if len(prices) < window + 1:
        raise ValueError(
            f"风险平价需要至少 {window + 1} 个交易日行情，当前只有 {len(prices)} 个"
        )
    tail = prices.iloc[-(window + 1) :]
    returns = tail.pct_change(fill_method=None).iloc[1:]
    volatility = returns.std(ddof=1, skipna=True) * math.sqrt(TRADING_DAYS)
    return volatility.reindex(codes).astype(float)


def build_portfolio(
    top_df: pd.DataFrame,
    prices: pd.DataFrame,
    method: str = "equal_weight",
    capital: float = 1_000_000.0,
) -> pd.DataFrame:
    """把选股结果转成目标组合。

    Parameters
    ----------
    top_df:
        ``screen`` 返回的 Top 表，至少含 ``code`` 列。
    prices:
        行情宽表，用于风险平价的 60 日波动率。
    method:
        ``equal_weight`` 或 ``risk_parity``。
    capital:
        组合资金（元）。

    Returns
    -------
    pd.DataFrame
        列：code、weight_pct（合计 100）、amount（合计 capital）、method。
    """
    if not isinstance(top_df, pd.DataFrame) or top_df.empty:
        raise ValueError("top_df 为空，无法构建组合")
    if "code" not in top_df.columns:
        raise ValueError("top_df 缺少 'code' 列")
    if method not in ("equal_weight", "risk_parity"):
        raise ValueError(f"不支持的组合方法: {method!r}")
    capital = _finite_number(capital, "capital", minimum=0.0)
    if capital <= 0:
        raise ValueError("capital 必须大于 0")

    codes = list(top_df["code"])
    if not codes:
        raise ValueError("top_df 没有股票代码")
    if len(set(map(str, codes))) != len(codes):
        raise ValueError("top_df 存在重复股票代码")

    weights = np.full(len(codes), 1.0 / len(codes), dtype=float)

    if method == "risk_parity":
        volatility = _trailing_volatility(prices, pd.Index(codes))
        with np.errstate(divide="ignore", invalid="ignore"):
            inverse = pd.Series(
                np.where(volatility.to_numpy() > 0, 1.0 / volatility.to_numpy(), np.nan),
                index=volatility.index,
                dtype=float,
            )
        valid = inverse.dropna()
        if valid.empty:
            raise ValueError("风险平价无法计算任何标的的 60 日波动率")
        # 个别停牌/新上市标的历史不足时，用有效标的中位数倒数波动填充
        inverse = inverse.fillna(float(valid.median()))
        total = float(inverse.sum())
        if not math.isfinite(total) or total <= 0:
            raise ValueError("风险平价权重计算失败")
        weights = inverse.to_numpy(dtype=float) / total

    weight_pct = weights * 100.0
    amount = weights * capital
    return pd.DataFrame(
        {
            "code": codes,
            "weight_pct": weight_pct,
            "amount": amount,
            "method": method,
        }
    )


# ---------------------------------------------------------------------------
# 回测
# ---------------------------------------------------------------------------


def _compute_metrics(
    nav: pd.Series,
    benchmark_nav: pd.Series,
    risk_free_rate: float,
) -> dict[str, float]:
    """根据每日净值计算 CAGR / Sharpe / MDD / IR / 月胜率。"""
    returns = nav.pct_change(fill_method=None).iloc[1:].dropna()
    if len(returns) < 1:
        raise ValueError("净值序列过短，无法计算回测指标")

    daily_rf = risk_free_rate / TRADING_DAYS
    std = float(returns.std(ddof=1))
    mean = float(returns.mean())
    sharpe = (mean - daily_rf) / std * math.sqrt(TRADING_DAYS) if std > 1e-12 else 0.0

    growth = float(nav.iloc[-1] / nav.iloc[0])
    if growth <= 0:
        raise ValueError("组合净值出现非正值，无法计算 CAGR")
    cagr = growth ** (TRADING_DAYS / len(returns)) - 1.0

    drawdown = nav / nav.cummax() - 1.0
    mdd = float(drawdown.min())

    benchmark_returns = benchmark_nav.pct_change(fill_method=None).iloc[1:]
    active = (returns - benchmark_returns.reindex(returns.index)).dropna()
    active_std = float(active.std(ddof=1)) if len(active) > 1 else 0.0
    ir = (
        float(active.mean()) / active_std * math.sqrt(TRADING_DAYS)
        if active_std > 1e-12
        else 0.0
    )

    monthly_last = nav.resample("ME").last()
    monthly_returns = monthly_last.pct_change(fill_method=None)
    monthly_returns.iloc[0] = monthly_last.iloc[0] / nav.iloc[0] - 1.0
    monthly_valid = monthly_returns.dropna()
    win_rate = float((monthly_valid > 0).mean()) if len(monthly_valid) else 0.0

    def finite(value: float) -> float:
        return float(value) if math.isfinite(float(value)) else 0.0

    return {
        "cagr": finite(cagr),
        "sharpe": finite(sharpe),
        "mdd": finite(mdd),
        "ir": finite(ir),
        "win_rate": finite(win_rate),
    }


def run_backtest(
    data,
    weights: dict,
    start,
    end,
    top_n: int = 10,
    rebalance_days: int = 20,
    cost_buy: float = 0.0003,
    cost_sell: float = 0.0013,
    risk_free_rate: float = 0.02,
) -> dict:
    """按 A 股交易规则回测因子选股策略。

    规则：
    - 调仓日 t 收盘出信号，t+1 收盘成交；若停牌/无成交则逐日顺延；
    - 买入时 ``close_raw`` 日涨幅 >= 9.8% 跳过，卖出时跌幅 <= -9.8% 跳过；
    - 买入收 ``cost_buy``，卖出收 ``cost_sell``（含印花税），按成交额计；
    - 每日净值用前复权收盘价盯市，停牌股票沿用最近有效收盘价。

    Parameters
    ----------
    data:
        DataBundle 或 dict（需要 prices/close_raw/amount/fundamentals/industry/benchmark）。
    weights:
        10 因子权重。
    start, end:
        回测区间（含端点）；None 表示使用数据首/尾日期。
    top_n, rebalance_days:
        每期持股数与调仓间隔（交易日）。
    cost_buy, cost_sell:
        买入/卖出成本率（卖出含印花税）。
    risk_free_rate:
        年化无风险利率，用于 Sharpe。

    Returns
    -------
    dict
        ``{"nav": Series, "benchmark_nav": Series,
        "metrics": {cagr, sharpe, mdd, ir, win_rate}, "rebalances": int}``。
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    validate_weights(weights)

    top_n = _positive_int(top_n, "top_n")
    rebalance_days = _positive_int(rebalance_days, "rebalance_days")
    cost_buy = _finite_number(cost_buy, "cost_buy", minimum=0.0)
    cost_sell = _finite_number(cost_sell, "cost_sell", minimum=0.0)
    risk_free_rate = _finite_number(risk_free_rate, "risk_free_rate")
    if cost_buy >= 1 or cost_sell >= 1:
        raise ValueError("交易成本率必须小于 1")

    prices = _clean_date_frame(_get(data, "prices"), "prices")
    close_raw = _clean_date_frame(_get(data, "close_raw"), "close_raw").reindex(
        index=prices.index, columns=prices.columns
    )
    amount = _clean_date_frame(_get(data, "amount"), "amount").reindex(
        index=prices.index, columns=prices.columns
    )
    benchmark = _clean_benchmark(_get(data, "benchmark"), prices.index)

    fundamentals = _get(data, "fundamentals")
    industry = _get(data, "industry")

    all_dates = prices.index
    start_stamp = all_dates[0] if start is None else _to_timestamp(start, all_dates)
    end_stamp = all_dates[-1] if end is None else _to_timestamp(end, all_dates)
    if start_stamp > end_stamp:
        raise ValueError(f"回测起点 {start_stamp} 晚于终点 {end_stamp}")

    window_positions = np.flatnonzero(
        (all_dates >= start_stamp) & (all_dates <= end_stamp)
    )
    if len(window_positions) < MIN_BACKTEST_DAYS:
        raise ValueError(
            f"回测区间只有 {len(window_positions)} 个交易日，"
            f"至少需要 {MIN_BACKTEST_DAYS} 个"
        )

    first_pos = int(window_positions[0])
    last_pos = int(window_positions[-1])
    window_dates = all_dates[window_positions]

    # 预计算：涨跌停判定收益、盯市价格（停牌沿用最近价）
    close_raw_returns = close_raw.pct_change(fill_method=None)
    valuation_prices = prices.ffill()

    benchmark_window = benchmark.iloc[window_positions]
    if not _finite(benchmark_window.iloc[0]) or float(benchmark_window.iloc[0]) <= 0:
        raise ValueError("benchmark 首个价格必须为正数")
    benchmark_nav = benchmark_window / float(benchmark_window.iloc[0])

    # 组合状态
    cash = 1.0
    holdings: dict[str, float] = {}
    pending: list[dict] = []
    executed_signals: set[int] = set()
    nav_values = np.full(len(window_positions), np.nan, dtype=float)

    signal_positions = set(range(first_pos, last_pos, rebalance_days))

    def signal_price(pos: int, code: str) -> float:
        """信号日盯市价（停牌用最近有效价）。"""
        try:
            value = valuation_prices.iloc[pos][code]
        except KeyError:
            return float("nan")
        return float(value) if _finite(value) else float("nan")

    def execution_price(pos: int, code: str) -> float:
        """成交日真实收盘价；停牌日价格为 NaN，用于触发顺延。"""
        try:
            value = prices.iloc[pos][code]
        except KeyError:
            return float("nan")
        return float(value) if _finite(value) else float("nan")

    def current_equity(pos: int) -> float:
        """按 pos 日收盘价计算组合总资产。"""
        total = cash
        for code, shares in holdings.items():
            price = signal_price(pos, code)
            if _finite(price) and price > 0:
                total += shares * price
        return float(total)

    def execute_order(order: dict, pos: int) -> str:
        """尝试成交，返回 filled / postponed / skipped。"""
        nonlocal cash
        code = order["code"]
        price = execution_price(pos, code)
        try:
            amount_value = float(amount.iloc[pos][code])
        except KeyError:
            amount_value = float("nan")
        # 停牌 / 无成交：顺延
        if not (_finite(price) and price > 0 and _finite(amount_value) and amount_value > 0):
            return "postponed"

        ret = close_raw_returns.iloc[pos].get(code, float("nan"))
        ret = float(ret) if _finite(ret) else float("nan")

        if order["side"] == "buy":
            if _finite(ret) and ret >= LIMIT_UP:
                return "skipped"
            affordable = cash / (1.0 + cost_buy)
            value = min(float(order["value"]), affordable)
            if value <= 1e-10:
                return "skipped"
            shares = value / price
            cost = shares * price * (1.0 + cost_buy)
            if cost > cash and cost - cash < 1e-9:
                cost = cash
            cash -= cost
            if cash < 0 and cash > -1e-9:
                cash = 0.0
            holdings[code] = holdings.get(code, 0.0) + shares
            return "filled"

        # 卖出
        if _finite(ret) and ret <= LIMIT_DOWN:
            return "skipped"
        held = holdings.get(code, 0.0)
        if held <= 0:
            return "skipped"
        if order.get("all"):
            shares = held
        else:
            shares = min(held, float(order["value"]) / price)
        if shares <= 1e-10:
            return "skipped"
        cash += shares * price * (1.0 - cost_sell)
        remaining = held - shares
        if remaining <= 1e-10:
            holdings.pop(code, None)
        else:
            holdings[code] = remaining
        return "filled"

    def build_orders(pos: int) -> list[dict]:
        """在信号日生成下一交易日开始的订单（先卖后买）。"""
        scores = compute_factor_scores(
            prices=prices,
            close_raw=close_raw,
            amount=amount,
            fundamentals=fundamentals,
            industry=industry,
            weights=weights,
            asof=all_dates[pos],
        )
        top_codes = list(scores.head(top_n).index)
        if not top_codes:
            raise ValueError("调仓日没有可用股票，无法生成组合")
        weight = 1.0 / len(top_codes)
        equity = current_equity(pos)
        if not math.isfinite(equity) or equity <= 0:
            raise ValueError("组合净值非正，无法继续回测")
        min_trade = max(equity * MIN_TRADE_RATIO, 1e-9)

        target_value = {code: equity * weight for code in top_codes}
        orders: list[dict] = []

        # 先卖：清仓不在目标内的，减仓超出目标权重的
        for code, shares in list(holdings.items()):
            if shares <= 0:
                continue
            price = signal_price(pos, code)
            current = shares * price if _finite(price) and price > 0 else 0.0
            if code not in target_value:
                if current > min_trade:
                    orders.append(
                        {
                            "code": code,
                            "side": "sell",
                            "value": current,
                            "all": True,
                            "signal_pos": pos,
                            "attempt_pos": pos + 1,
                        }
                    )
            else:
                delta = current - target_value[code]
                if delta > min_trade:
                    orders.append(
                        {
                            "code": code,
                            "side": "sell",
                            "value": delta,
                            "all": False,
                            "signal_pos": pos,
                            "attempt_pos": pos + 1,
                        }
                    )

        # 再买：补足目标权重
        for code, target in target_value.items():
            shares = holdings.get(code, 0.0)
            price = signal_price(pos, code)
            current = shares * price if _finite(price) and price > 0 else 0.0
            delta = target - current
            if delta > min_trade:
                orders.append(
                    {
                        "code": code,
                        "side": "buy",
                        "value": delta,
                        "all": False,
                        "signal_pos": pos,
                        "attempt_pos": pos + 1,
                    }
                )
        return orders

    # ---- 逐日循环 ----
    for offset, pos in enumerate(window_positions):
        pos = int(pos)
        if pending:
            # 先卖后买，先到的批次先执行
            pending.sort(key=lambda item: 0 if item["side"] == "sell" else 1)
            remaining_orders: list[dict] = []
            for order in pending:
                if order["attempt_pos"] != pos:
                    remaining_orders.append(order)
                    continue
                status = execute_order(order, pos)
                if status == "filled":
                    executed_signals.add(order["signal_pos"])
                elif status == "postponed":
                    order["attempt_pos"] = pos + 1
                    if order["attempt_pos"] <= last_pos:
                        remaining_orders.append(order)
                # skipped：本批次放弃（涨跌停规则）
            pending = remaining_orders

        nav_values[offset] = current_equity(pos)

        # 调仓日 t 收盘出信号，订单最早 t+1 成交；新的调仓会作废旧挂单
        if pos in signal_positions and pos < last_pos:
            pending = build_orders(pos)

    nav = pd.Series(nav_values, index=window_dates, name="nav")
    if not np.isfinite(nav.to_numpy()).all() or (nav <= 0).any():
        raise ValueError("回测净值出现非正/非有限值")

    metrics = _compute_metrics(nav, benchmark_nav, risk_free_rate)
    return {
        "nav": nav,
        "benchmark_nav": pd.Series(
            benchmark_nav.to_numpy(dtype=float), index=window_dates, name="benchmark_nav"
        ),
        "metrics": metrics,
        "rebalances": int(len(executed_signals)),
    }
