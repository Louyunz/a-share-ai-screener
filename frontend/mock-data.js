/* =========================================================================
 * DEV-ONLY · frontend/mock-data.js
 * 仅当页面 URL 带 ?mock=1 时由 index.html 加载，用于无后端时预览完整 UI。
 * 严禁在生产/默认访问路径下启用；这里的数值全部是演示数据，不是真实行情。
 * ========================================================================= */
(function (global) {
  'use strict';

  var FACTOR_KEYS = ['ep', 'bp', 'roe', 'ocf_to_profit', 'revenue_growth', 'momentum_60', 'reversal_5', 'volatility_60', 'liquidity_20', 'size'];
  var FACTOR_LABELS = {
    ep: '价值 EP',
    bp: '账面价值',
    roe: 'ROE 质量',
    ocf_to_profit: '现金流质量',
    revenue_growth: '营收成长',
    momentum_60: '60 日动量',
    reversal_5: '5 日反转',
    volatility_60: '低波动',
    liquidity_20: '流动性',
    size: '小市值'
  };

  var baseWeights = {
    ep: 0.12, bp: 0.08, roe: 0.15, ocf_to_profit: 0.10, revenue_growth: 0.12,
    momentum_60: 0.12, reversal_5: 0.05, volatility_60: 0.12, liquidity_20: 0.08, size: 0.06
  };
  var bestWeights = {
    ep: 0.18, bp: 0.05, roe: 0.17, ocf_to_profit: 0.09, revenue_growth: 0.13,
    momentum_60: 0.12, reversal_5: 0.03, volatility_60: 0.12, liquidity_20: 0.06, size: 0.05
  };

  function pad(n) { return n < 10 ? '0' + n : String(n); }

  function monthLabels(count, startYear) {
    var out = [];
    var cursor = new Date(Date.UTC(startYear || 2021, 0, 1));
    for (var i = 0; i < count; i += 1) {
      out.push(cursor.getUTCFullYear() + '-' + pad(cursor.getUTCMonth() + 1));
      cursor.setUTCMonth(cursor.getUTCMonth() + 1);
    }
    return out;
  }

  // 确定性“净值曲线”：趋势 + 多频波动，仅用于视觉演示。
  function makeNav(count, endValue, seed, volatility) {
    var values = [1];
    for (var i = 1; i < count; i += 1) {
      var progress = i / (count - 1);
      var wave = Math.sin(i * 0.73 + seed) * volatility
        + Math.sin(i * 0.29 + seed * 1.7) * volatility * 0.75
        + Math.sin(i * 1.37 + seed * 0.6) * volatility * 0.28;
      var trend = Math.pow(endValue, progress);
      values.push(Number((trend * (1 + wave)).toFixed(4)));
    }
    values[0] = 1;
    values[values.length - 1] = endValue;
    return values;
  }

  function makeRng(seed) {
    var state = seed >>> 0;
    return function () {
      state += 0x6D2B79F5;
      var t = state;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function normalizedWeights(values) {
    var total = values.reduce(function (sum, value) { return sum + value; }, 0);
    var output = {};
    var running = 0;
    FACTOR_KEYS.forEach(function (key, index) {
      var value = index === FACTOR_KEYS.length - 1
        ? Number((1 - running).toFixed(4))
        : Number((values[index] / total * 0.9998).toFixed(4));
      output[key] = value;
      running += value;
    });
    output[FACTOR_KEYS[FACTOR_KEYS.length - 1]] = Number((1 - (running - output[FACTOR_KEYS[FACTOR_KEYS.length - 1]])).toFixed(4));
    return output;
  }

  function objective(metrics) {
    return Number((metrics.sharpe - 0.5 * Math.abs(metrics.mdd) + 0.2 * metrics.ir).toFixed(4));
  }

  var dates = monthLabels(60, 2021);
  var strategyNav = makeNav(60, 1.63, 2.7, 0.055);
  var benchmarkNav = makeNav(60, 1.31, 7.1, 0.028);
  var candidateNav = makeNav(60, 1.93, 4.4, 0.048);

  var baseMetrics = {
    cagr: 0.104,
    sharpe: 0.89,
    mdd: -0.205,
    ir: 0.29,
    win_rate: 0.536
  };
  var bestMetrics = {
    cagr: 0.142,
    sharpe: 1.21,
    mdd: -0.168,
    ir: 0.42,
    win_rate: 0.583
  };

  var stockRows = [
    ['600519', '贵州茅台', '食品饮料', 2.43, [1.82, 1.35, 1.24, 1.61, 1.42, 1.05, 0.32, 0.88, 1.18, 0.72]],
    ['300750', '宁德时代', '电力设备', 2.31, [0.92, 0.74, 1.68, 1.12, 1.86, 1.64, 0.16, 0.62, 1.48, 1.02]],
    ['600036', '招商银行', '银行', 2.18, [1.42, 1.58, 1.12, 1.28, 0.84, 0.61, 0.44, 1.42, 1.34, 0.38]],
    ['601318', '中国平安', '非银金融', 2.05, [1.12, 1.24, 0.92, 1.08, 0.72, 0.86, 0.58, 1.16, 0.92, 0.44]],
    ['000858', '五粮液', '食品饮料', 1.97, [1.38, 1.21, 1.34, 1.18, 1.04, 0.52, 0.28, 0.96, 0.86, 0.52]],
    ['002594', '比亚迪', '汽车', 1.86, [0.86, 0.64, 1.22, 0.94, 1.74, 1.68, 0.12, 0.44, 1.52, 1.24]],
    ['600900', '长江电力', '公用事业', 1.78, [1.64, 1.08, 0.88, 1.44, 0.48, 0.34, 0.42, 1.62, 1.28, 0.40]],
    ['601899', '紫金矿业', '有色金属', 1.65, [1.28, 1.48, 0.72, 0.88, 0.86, 1.12, 0.36, 0.84, 0.76, 0.62]],
    ['002415', '海康威视', '电子', 1.54, [0.94, 0.82, 1.02, 0.72, 1.28, 1.34, 0.24, 0.56, 1.02, 0.88]],
    ['600276', '恒瑞医药', '医药生物', 1.42, [1.22, 0.94, 0.86, 1.10, 1.06, 0.74, 0.34, 0.72, 0.68, 0.58]]
  ];

  var top = stockRows.map(function (row) {
    var stock = {
      code: row[0],
      name: row[1],
      industry: row[2],
      score: row[3]
    };
    FACTOR_KEYS.forEach(function (key, index) {
      stock['z_' + key] = row[4][index];
    });
    return stock;
  });

  var portfolioWeights = [12.8, 11.6, 10.9, 10.2, 9.8, 9.4, 9.1, 8.8, 8.6, 8.8];
  var portfolio = top.map(function (stock, index) {
    var weight = portfolioWeights[index];
    return {
      code: stock.code,
      weight_pct: weight,
      amount: Math.round(weight / 100 * 1000000),
      method: 'risk_parity'
    };
  });
  // 四舍五入后补齐金额到 100 万（仅演示快照）。
  var amountSum = portfolio.reduce(function (sum, item) { return sum + item.amount; }, 0);
  portfolio[0].amount += 1000000 - amountSum;

  var selectionResult = {
    as_of: '2025-12-31',
    top_n: 10,
    top: top,
    portfolio: portfolio,
    method: 'risk_parity',
    metrics: baseMetrics,
    nav: {
      dates: dates,
      strategy: strategyNav,
      benchmark: benchmarkNav,
      benchmark_name: '沪深 300'
    }
  };

  var interpretationText = [
    '### 本轮选股在回答什么',
    '系统在沪深 300 + 中证 500 股票池内，用 10 个方向对齐后的因子做行业中性化打分，并叠加停牌、涨跌停、流动性与交易成本约束。本次组合共 **10 只**，采用风险平价分配权重，当前回测区间为 2021-01 至 2025-12。',
    '### 组合结构：质量与成长略占上风',
    '- 权重最高的三只标的为贵州茅台（12.8%）、宁德时代（11.6%）与招商银行（10.9%），风险平价使单票权重集中在 8.6%–12.8%。',
    '- 贵州茅台在价值 EP、盈利质量与动量因子上同时为正暴露；宁德时代主要贡献营收成长与 60 日动量；招商银行则在账面价值与低波动因子上得分较高。',
    '- 食品饮料行业出现两只标的，虽然单票权重受限，但行业集中度仍需在组合层面持续观察。',
    '### 回测事实与需要警惕的部分',
    '当前基线组合的 **Sharpe 为 0.89**，年化收益 **10.4%**，最大回撤 **−20.5%**，信息比率 **0.29**。历史最大回撤说明这套多因子组合并非低波资产，在市场风格快速切换时可能出现较长时间的相对落后。',
    '- **因子拥挤**：价值与质量因子同时暴露较高，若市场转向高波动成长风格，组合可能跑输基准。',
    '- **行业暴露**：食品饮料占比偏高，需关注单一行业景气度变化对组合波动的放大。',
    '- **样本局限**：回测使用历史缓存数据，无法覆盖未来制度、流动性与宏观环境变化，历史表现不代表未来收益。',
    '### 接下来可以跟踪',
    '建议在每个调仓周期检查因子 IC 稳定性、行业权重偏离以及组合相对沪深 300 的信息比率变化；如果最大回撤继续扩大，应优先复核低波动与动量因子的权重假设。',
    '> 以上内容基于本次选股与回测快照生成，仅供研究演示，不构成投资建议。'
  ].join('\n');

  var messages = [
    {
      role: 'assistant',
      content: '你好，我是智投罗盘的 AI 画像分析师。为了把选股因子调到更适合你的位置，我想先了解你的风险偏好、投资期限和可接受的最大回撤。你可以直接用一句话描述。',
      created_at: '2025-12-31T09:30:00+08:00'
    },
    {
      role: 'user',
      content: '我偏向稳健，能接受 10% 左右回撤，打算持有 3 年以上，主要目标是长期资产增值。',
      created_at: '2025-12-31T09:31:12+08:00'
    },
    {
      role: 'assistant',
      content: '已记录：**平衡型风险偏好、3 年以上期限、约 10% 的最大回撤容忍**。我再补充两个可选信息，用于降低组合与你的偏好冲突。\n\n- 有没有希望排除的行业？\n- 大致的可投资金规模是多少？',
      created_at: '2025-12-31T09:32:03+08:00'
    },
    {
      role: 'user',
      content: '不投房地产和煤炭，资金大概 100 万，股票基金经验 5 年。',
      created_at: '2025-12-31T09:33:18+08:00'
    },
    {
      role: 'assistant',
      content: '画像已完整：**平衡型 · 3 年以上 · 10% 回撤 · 长期增值 · 排除房地产/煤炭 · 100 万**。右侧画像卡已生成，确认后即可进入 10 因子选股。',
      created_at: '2025-12-31T09:33:49+08:00'
    }
  ];

  // ---- 模拟 24 组调参 trial（与 /tune/events 的 trial 字段一致） ----
  var rng = makeRng(20251231);
  var trials = [];
  var insertBestAt = 18;
  for (var i = 0; i < 24; i += 1) {
    var weights;
    var metrics;
    var isBest = false;
    if (i === insertBestAt) {
      weights = bestWeights;
      metrics = bestMetrics;
      isBest = true;
    } else {
      var raw = FACTOR_KEYS.map(function () { return 0.02 + rng(); });
      weights = normalizedWeights(raw);
      metrics = {
        cagr: Number((0.07 + rng() * 0.062).toFixed(4)),
        sharpe: Number((0.70 + rng() * 0.38).toFixed(4)),
        mdd: Number((-0.135 - rng() * 0.092).toFixed(4)),
        ir: Number((0.14 + rng() * 0.18).toFixed(4))
      };
    }
    trials.push({
      trial_index: i + 1,
      weights: weights,
      metrics: metrics,
      objective: objective(metrics),
      is_best: isBest
    });
  }
  // 让演示里的最优 trial 明确是 is_best 且为最高目标值。
  trials[insertBestAt].is_best = true;

  var versions = [
    {
      id: 'v1',
      weights: baseWeights,
      metrics: { cagr: 0.086, sharpe: 0.74, mdd: -0.224, ir: 0.21 },
      status: 'archived',
      created_at: '2025-11-18T10:20:00+08:00',
      parent_id: null
    },
    {
      id: 'v2',
      weights: baseWeights,
      metrics: baseMetrics,
      status: 'active',
      created_at: '2025-12-12T15:06:00+08:00',
      parent_id: 'v1'
    }
  ];

  var candidateVersion = {
    id: 'v3',
    weights: bestWeights,
    metrics: bestMetrics,
    status: 'candidate',
    created_at: '2025-12-31T10:22:00+08:00',
    parent_id: 'v2'
  };

  var tuning = {
    job_id: 'mock-tune-job-01',
    trials_total: 24,
    base_weights: baseWeights,
    best_weights: bestWeights,
    base_metrics: baseMetrics,
    best_metrics: bestMetrics,
    trials: trials,
    candidate_version_id: 'v3',
    candidate_version: candidateVersion,
    compare: {
      dates: dates,
      base_nav: strategyNav,
      candidate_nav: candidateNav,
      benchmark_nav: benchmarkNav,
      base_metrics: baseMetrics,
      candidate_metrics: bestMetrics
    }
  };

  global.MockData = {
    dev_only: true,
    session_id: 'mock-session-a1b2c3',
    created_at: '2025-12-31T09:30:00+08:00',
    profile: {
      slots: {
        risk_level: 'balanced',
        horizon: '3 年以上',
        drawdown_tolerance: '约 10%',
        goal: '长期资产增值',
        experience: '5 年股票基金经验',
        excluded_industries: '房地产, 煤炭',
        capital: '100 万元'
      },
      risk_level: 'balanced',
      confirmed: true,
      complete: true
    },
    messages: messages,
    selectionResult: selectionResult,
    interpretation: interpretationText,
    versions: versions,
    active_id: 'v2',
    tuning: tuning,
    factors: FACTOR_KEYS,
    factorLabels: FACTOR_LABELS,
    health: {
      ok: true,
      source: 'sample',
      as_of: '2025-12-31',
      llm: 'ok'
    },
    // 发送消息后的模拟回复；不访问真实 LLM。
    chatReply: {
      content: '收到，我会在已有画像上继续校准。你补充的信息已经记录，当前画像保持**平衡型 / 3 年以上 / 约 10% 回撤**。\n\n如果确认没有问题，可以点击右侧「确认画像」并开始选股；后续也可以在「AI 解读」里继续追问本次结果。',
      slots: {
        risk_level: 'balanced',
        horizon: '3 年以上',
        drawdown_tolerance: '约 10%',
        goal: '长期资产增值',
        experience: '5 年股票基金经验',
        excluded_industries: '房地产, 煤炭',
        capital: '100 万元'
      },
      profile_ready: true
    },
    followupReply: '从本次结果看，组合的主要波动来源是食品饮料行业的集中暴露与价值/质量因子的同步回撤。你可以重点观察两项事实：一是组合相对沪深 300 的信息比率是否持续为正；二是下一次调仓时低波动因子是否仍能提供缓冲。\n\n需要说明的是，这只是基于既有回测快照的解释，不构成对未来收益的判断。'
  };
})(window);
