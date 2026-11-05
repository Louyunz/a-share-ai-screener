/* =========================================================================
 * 智投罗盘 · 前端状态与 API 编排
 * 零构建、零外部依赖；SSE 对话使用 fetch + ReadableStream，进度使用 EventSource。
 * 契约：docs/CONTRACT.md 第 3 / 4 节。
 * ========================================================================= */
(function () {
  'use strict';

  /* ----------------------------------------------------------- 常量与工具 */
  var API_BASE = '/api';
  var SESSION_KEY = 'session_id';
  var DEV_MOCK = !!window.__MOCK_MODE__;
  var VIEWS = ['chat', 'selection', 'interpretation', 'tune'];
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
  var FACTOR_SHORT = {
    ep: '价值', bp: '账面', roe: '质量', ocf_to_profit: '现金', revenue_growth: '成长',
    momentum_60: '动量', reversal_5: '反转', volatility_60: '低波', liquidity_20: '流动', size: '规模'
  };
  var RISK_LABELS = { conservative: '保守型', balanced: '平衡型', aggressive: '积极型' };
  var SLOT_KEYS = ['risk_level', 'horizon', 'drawdown_tolerance', 'goal', 'experience', 'excluded_industries', 'capital'];
  var REQUIRED_SLOTS = ['risk_level', 'horizon', 'drawdown_tolerance'];
  var SLOT_DEFS = [
    { key: 'risk_level', label: '风险偏好', required: true, formatter: riskLabel },
    { key: 'horizon', label: '投资期限', required: true },
    { key: 'drawdown_tolerance', label: '回撤容忍', required: true },
    { key: 'goal', label: '投资目标' },
    { key: 'experience', label: '投资经验' },
    { key: 'excluded_industries', label: '排除行业' },
    { key: 'capital', label: '可投资金' }
  ];
  var METHOD_LABELS = { equal_weight: '等权', risk_parity: '风险平价' };
  var ALLOCATION_COLORS = ['#22D3EE', '#34D399', '#F59E0B', '#60A5FA', '#A78BFA', '#FB7185', '#2DD4BF', '#FBBF24', '#38BDF8', '#818CF8', '#F472B6', '#4ADE80'];
  var MOCK_SELECTION_TIMERS = [];
  var MOCK_TUNE_TIMERS = [];
  var toastTimer = null;
  var tuneElapsedTimer = null;
  var tuneRenderHandle = null;
  var interpretationTyper = null;

  function $(id) { return document.getElementById(id); }
  function all(selector, root) { return Array.prototype.slice.call((root || document).querySelectorAll(selector)); }
  function setText(node, value) { if (node) node.textContent = value == null ? '' : String(value); }
  function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
  function deepClone(value) {
    try { return JSON.parse(JSON.stringify(value)); } catch (err) { return value; }
  }
  function sleep(ms) { return new Promise(function (resolve) { setTimeout(resolve, ms); }); }
  function esc(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, function (char) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char];
    });
  }
  function toNum(value) {
    if (typeof value === 'number') return Number.isFinite(value) ? value : null;
    if (typeof value === 'string') {
      var raw = value.trim().replace(/,/g, '');
      if (!raw) return null;
      var isPercent = raw.charAt(raw.length - 1) === '%';
      if (isPercent) raw = raw.slice(0, -1);
      var parsed = Number(raw);
      if (!Number.isFinite(parsed)) return null;
      return isPercent ? parsed / 100 : parsed;
    }
    return null;
  }
  function firstNum() {
    for (var i = 0; i < arguments.length; i += 1) {
      var value = toNum(arguments[i]);
      if (value !== null) return value;
    }
    return null;
  }
  function firstText() {
    for (var i = 0; i < arguments.length; i += 1) {
      if (arguments[i] !== null && arguments[i] !== undefined && String(arguments[i]).trim() !== '') return String(arguments[i]);
    }
    return '';
  }
  function fmtNum(value, digits) {
    var number = toNum(value);
    return number === null ? '—' : number.toFixed(digits === undefined ? 2 : digits);
  }
  function fmtPct(value, digits, signed) {
    var number = toNum(value);
    if (number === null) return '—';
    var text = Math.abs(number * 100).toFixed(digits === undefined ? 1 : digits) + '%';
    if (number < 0) return '−' + text;
    if (signed && number > 0) return '+' + text;
    return text;
  }
  function fmtPp(value, digits) {
    var number = toNum(value);
    if (number === null) return '—';
    var text = Math.abs(number * 100).toFixed(digits === undefined ? 1 : digits) + 'pp';
    if (number < 0) return '−' + text;
    if (number > 0) return '+' + text;
    return text;
  }
  function fmtMoney(value) {
    var number = toNum(value);
    if (number === null) return '—';
    try {
      return new Intl.NumberFormat('zh-CN', { style: 'currency', currency: 'CNY', maximumFractionDigits: 0 }).format(number);
    } catch (err) {
      return '¥' + Math.round(number).toLocaleString('zh-CN');
    }
  }
  function riskLabel(value) {
    var key = String(value || '').toLowerCase();
    return RISK_LABELS[key] || (value || '未填写');
  }
  function methodLabel(value) {
    return METHOD_LABELS[value] || value || '—';
  }
  function normalizeCode(code) {
    var raw = String(code == null ? '' : code).trim();
    if (/^\d{1,6}$/.test(raw)) return raw.padStart(6, '0');
    return raw;
  }
  function timeText(value) {
    if (!value) return '';
    var date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      var match = String(value).match(/\d{2}:\d{2}/);
      return match ? match[0] : '';
    }
    return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', hour12: false });
  }
  function shortDate(value) {
    if (!value) return '—';
    return String(value).replace(/-/g, '.').slice(0, 10);
  }
  function formatElapsed(ms) {
    var total = Math.max(0, Math.floor(ms / 1000));
    var minutes = Math.floor(total / 60);
    var seconds = total % 60;
    return String(minutes).padStart(2, '0') + ':' + String(seconds).padStart(2, '0');
  }
  function nowIso() { return new Date().toISOString(); }
  function mockChunkText(text, size) {
    var chars = Array.from(String(text || ''));
    var chunks = [];
    for (var i = 0; i < chars.length; i += size) chunks.push(chars.slice(i, i + size).join(''));
    return chunks;
  }
  function clearTimers(list) {
    while (list.length) clearTimeout(list.pop());
  }
  function setHidden(node, hidden) { if (node) node.hidden = !!hidden; }
  function scrollChat(force) {
    var box = $('chat-messages');
    if (!box) return;
    var nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 180;
    if (force || nearBottom) box.scrollTop = box.scrollHeight;
  }
  function autoGrow(textarea) {
    if (!textarea) return;
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 140) + 'px';
  }
  function chartWidth(node, fallback) {
    if (!node) return fallback || 360;
    var width = node.clientWidth || (node.parentElement && node.parentElement.clientWidth) || 0;
    return Math.round(Math.max(250, width - 24) || fallback || 360);
  }
  function chartEmpty(node, message) {
    if (!node) return;
    node.innerHTML = '';
    var box = document.createElement('div');
    box.className = 'chart-empty small';
    box.textContent = message || '暂无图表数据';
    node.appendChild(box);
  }

  /* ------------------------------------------------------------- 全局状态 */
  var state = {
    mock: DEV_MOCK,
    sessionId: '',
    booted: false,
    activeView: 'chat',
    health: null,
    profile: emptyProfile(),
    profileEditing: false,
    profileBusy: false,
    messages: [],
    chatBusy: false,
    selection: {
      status: 'idle',
      result: null,
      taskId: '',
      progress: 0,
      stage: '',
      message: '',
      error: '',
      startedAt: 0,
      finishedAt: 0,
      source: null,
      runSeq: 0
    },
    interpretation: {
      text: '',
      status: 'idle',
      busy: false,
      busyKind: '',
      followups: [],
      typer: null
    },
    tune: {
      status: 'idle',
      jobId: '',
      total: 0,
      trials: [],
      baseWeights: null,
      bestWeights: null,
      baseMetrics: null,
      bestMetrics: null,
      compare: null,
      candidateVersionId: '',
      bestTrialIndex: -1,
      startedAt: 0,
      finishedAt: 0,
      elapsedMs: 0,
      error: '',
      source: null,
      runSeq: 0,
      renderQueued: false,
      weightChartRendered: false,
      actionBusy: false,
      currentStep: 0
    },
    versions: [],
    activeVersionId: '',
    versionsLoaded: false,
    versionsLoading: false,
    versionError: '',
    retryAction: null
  };

  function emptyProfile() {
    return { slots: {}, confirmed: false, complete: false, risk_level: 'balanced' };
  }

  /* ------------------------------------------------------------- 错误与提示 */
  function AppError(message, status) {
    this.name = 'AppError';
    this.message = message;
    this.status = status || 0;
  }
  AppError.prototype = Object.create(Error.prototype);
  AppError.prototype.constructor = AppError;

  function extractApiError(payload, status) {
    if (payload && typeof payload === 'object') {
      if (typeof payload.detail === 'string' && payload.detail.trim()) return payload.detail;
      if (Array.isArray(payload.detail) && payload.detail.length) {
        return payload.detail.map(function (item) { return item && (item.msg || item.message || String(item)); }).join('；');
      }
      if (typeof payload.message === 'string' && payload.message.trim()) return payload.message;
      if (typeof payload.error === 'string' && payload.error.trim()) return payload.error;
    }
    if (typeof payload === 'string' && payload.trim()) return payload.trim().slice(0, 300);
    return '请求失败（HTTP ' + status + '）';
  }

  function friendlyError(err) {
    if (!err) return '未知错误，请重试。';
    if (err.status === 404) return '请求的资源不存在或会话已失效，请刷新后重试。';
    var message = err.message || String(err);
    if (/Failed to fetch|NetworkError|Load failed|fetch failed|Network request failed/i.test(message)) {
      return '无法连接后端服务，请确认 dev_server / FastAPI 已启动，然后重试。';
    }
    return message;
  }

  function showError(message, retryAction) {
    var banner = $('error-banner');
    if (!banner) return;
    setText($('error-message'), message || '操作失败，请重试。');
    state.retryAction = typeof retryAction === 'function' ? retryAction : null;
    var retry = $('error-retry');
    retry.hidden = !state.retryAction;
    banner.hidden = false;
    banner.style.animation = 'none';
    window.requestAnimationFrame(function () { banner.style.animation = ''; });
  }

  function hideError() {
    var banner = $('error-banner');
    if (banner) banner.hidden = true;
    state.retryAction = null;
  }

  function showToast(message) {
    var node = $('toast');
    if (!node) return;
    clearTimeout(toastTimer);
    node.textContent = message;
    node.hidden = false;
    toastTimer = setTimeout(function () { node.hidden = true; }, 3600);
  }

  /* ---------------------------------------------------------------- API 层 */
  async function request(path, options) {
    var opts = Object.assign({ method: 'GET' }, options || {});
    var headers = Object.assign({ Accept: 'application/json' }, opts.headers || {});
    if (opts.body !== undefined && opts.body !== null && typeof opts.body !== 'string') {
      headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(opts.body);
    }
    opts.headers = headers;
    var response;
    try {
      response = await fetch(API_BASE + path, opts);
    } catch (err) {
      throw new AppError('无法连接后端服务，请确认服务已启动。', 0);
    }
    var rawText = await response.text();
    var payload = null;
    if (rawText) {
      try { payload = JSON.parse(rawText); } catch (err) { payload = rawText; }
    }
    if (!response.ok) throw new AppError(extractApiError(payload, response.status), response.status);
    return payload;
  }

  async function createSession() {
    var payload = await request('/sessions', { method: 'POST' });
    if (!payload || !payload.session_id) throw new AppError('创建会话失败：响应缺少 session_id。', 0);
    return String(payload.session_id);
  }

  function readStoredSession() {
    try { return localStorage.getItem(SESSION_KEY) || ''; } catch (err) { return ''; }
  }
  function storeSession(sessionId) {
    try { localStorage.setItem(SESSION_KEY, sessionId); } catch (err) { /* 隐私模式不阻塞使用 */ }
  }
  function clearStoredSession() {
    try { localStorage.removeItem(SESSION_KEY); } catch (err) { /* ignore */ }
  }
  function sessionPath(suffix) {
    return '/sessions/' + encodeURIComponent(state.sessionId) + suffix;
  }

  /* ------------------------------------------------------------ 数据归一化 */
  function normalizeSlots(raw) {
    var slots = {};
    if (!raw) return slots;
    if (Array.isArray(raw)) {
      raw.forEach(function (item) {
        if (item && item.key) slots[item.key] = item.value == null ? '' : String(item.value);
      });
      return slots;
    }
    if (typeof raw !== 'object') return slots;
    SLOT_KEYS.forEach(function (key) {
      var value = raw[key];
      if (value === null || value === undefined) return;
      if (Array.isArray(value)) value = value.join('、');
      slots[key] = String(value).trim();
    });
    // 仅在明确缺失 source 字段时保留额外字段，契约之外不主动展示。
    return slots;
  }

  function completeFromSlots(slots) {
    return REQUIRED_SLOTS.every(function (key) { return !!(slots && slots[key]); });
  }

  function normalizeProfile(payload) {
    var source = payload || {};
    var slots = normalizeSlots(source.slots || source);
    var complete = typeof source.complete === 'boolean' ? source.complete : completeFromSlots(slots);
    return {
      slots: slots,
      risk_level: source.risk_level || slots.risk_level || 'balanced',
      confirmed: !!source.confirmed,
      complete: complete
    };
  }

  function slotsChanged(oldSlots, newSlots) {
    return SLOT_KEYS.some(function (key) { return String((oldSlots || {})[key] || '') !== String((newSlots || {})[key] || ''); });
  }

  function normalizeWeights(raw) {
    if (!raw) return null;
    var output = {};
    var found = false;
    FACTOR_KEYS.forEach(function (key, index) {
      var value = null;
      if (Array.isArray(raw)) value = toNum(raw[index]);
      else if (raw[key] !== undefined) value = toNum(raw[key]);
      else if (raw.weights && raw.weights[key] !== undefined) value = toNum(raw.weights[key]);
      output[key] = value;
      if (value !== null) found = true;
    });
    return found ? output : null;
  }

  function normalizeMetrics(raw) {
    var source = raw || {};
    var cagr = firstNum(source.cagr, source.annual_return, source.annualized_return, source.return);
    var sharpe = firstNum(source.sharpe, source.sharpe_ratio);
    var mdd = firstNum(source.mdd, source.max_drawdown);
    var ir = firstNum(source.ir, source.information_ratio);
    var winRate = firstNum(source.win_rate, source.monthly_win_rate);
    return {
      cagr: cagr,
      sharpe: sharpe,
      mdd: mdd === null ? null : -Math.abs(mdd),
      ir: ir,
      win_rate: winRate
    };
  }

  function normalizeStock(stock) {
    if (typeof stock === 'string') stock = { code: stock };
    stock = stock || {};
    var factors = {};
    FACTOR_KEYS.forEach(function (key) {
      var value = stock['z_' + key];
      if (value === undefined && stock.z && typeof stock.z === 'object') value = stock.z[key];
      if (value === undefined && Array.isArray(stock.z_values)) value = stock.z_values[FACTOR_KEYS.indexOf(key)];
      factors[key] = toNum(value);
    });
    var rawCode = stock.code || stock.ts_code || stock.symbol || '';
    return {
      code: normalizeCode(rawCode),
      name: stock.name || stock.stock_name || '',
      industry: stock.industry || stock.industry_name || '—',
      score: firstNum(stock.score, stock.factor_score, stock.total_score, stock.composite_score),
      factors: factors,
      raw: stock
    };
  }

  function normalizeSelectionPayload(raw, fallbackAsOf) {
    var payload = raw && raw.result ? raw.result : (raw || {});
    var topRaw = payload.top || payload.top_list || payload.items || [];
    var top = Array.isArray(topRaw) ? topRaw.map(normalizeStock) : [];
    var portfolioRaw = payload.portfolio || payload.portfolios || [];
    var portfolio = Array.isArray(portfolioRaw) ? portfolioRaw.map(function (item) {
      var originalWeight = toNum(item.weight_pct);
      if (originalWeight === null) {
        var ratio = toNum(item.weight);
        if (ratio !== null) originalWeight = ratio <= 1 ? ratio * 100 : ratio;
      }
      return {
        code: normalizeCode(item.code || item.symbol || ''),
        weight_pct: originalWeight,
        amount: firstNum(item.amount, item.market_value, item.value),
        method: item.method || payload.method || ''
      };
    }) : [];
    var rawMetrics = payload.metrics || (payload.backtest && payload.backtest.metrics) || {};
    var metrics = normalizeMetrics(rawMetrics);
    var navRaw = payload.nav || (payload.backtest && payload.backtest.nav) || null;
    var nav = normalizeSelectionNav(navRaw);
    var embedded = firstText(
      payload.interpretation, payload.ai_interpretation, payload.ai_commentary,
      payload.commentary, payload.summary, raw && raw.interpretation, raw && raw.summary
    );
    return {
      top: top,
      portfolio: portfolio,
      metrics: metrics,
      nav: nav,
      method: firstText(payload.method, portfolio[0] && portfolio[0].method),
      top_n: firstNum(payload.top_n) || top.length,
      as_of: firstText(payload.as_of, payload.asof, fallbackAsOf, state.health && state.health.as_of),
      benchmark_name: firstText(nav && nav.benchmark_name, payload.benchmark_name, '沪深 300'),
      interpretation: embedded,
      raw: payload
    };
  }

  function axisSeries(raw) {
    if (raw === null || raw === undefined) return null;
    if (Array.isArray(raw)) {
      if (!raw.length) return null;
      if (typeof raw[0] !== 'object') {
        var plain = raw.map(function (value) { return toNum(value); }).filter(function (value) { return value !== null; });
        return plain.length ? { labels: [], values: plain } : null;
      }
      var recordLabels = [];
      var recordValues = [];
      raw.forEach(function (entry, index) {
        var value = firstNum(entry.nav, entry.value, entry.y, entry.strategy, entry.current, entry.candidate);
        if (value === null) return;
        recordValues.push(value);
        recordLabels.push(firstText(entry.date, entry.label, entry.x, String(index)));
      });
      return recordValues.length ? { labels: recordLabels, values: recordValues } : null;
    }
    if (typeof raw === 'object') {
      if (Array.isArray(raw.values)) {
        return {
          labels: Array.isArray(raw.labels) ? raw.labels.slice() : (Array.isArray(raw.dates) ? raw.dates.slice() : []),
          values: raw.values.map(function (value) { return toNum(value); }).filter(function (value) { return value !== null; })
        };
      }
      var keys = Object.keys(raw).filter(function (key) { return toNum(raw[key]) !== null; }).sort();
      if (!keys.length) return null;
      return {
        labels: keys.slice(),
        values: keys.map(function (key) { return toNum(raw[key]); })
      };
    }
    return null;
  }

  function normalizeSelectionNav(raw) {
    if (!raw) return null;
    if (!Array.isArray(raw) && raw.nav && typeof raw.nav === 'object' && !Array.isArray(raw.nav) &&
      (raw.nav.dates || raw.nav.labels || raw.nav.strategy || raw.nav.values)) {
      raw = raw.nav;
    }
    var labels = [];
    var strategyRaw;
    var benchmarkRaw;
    if (Array.isArray(raw)) {
      strategyRaw = raw;
      labels = raw.map(function (entry, index) {
        return (entry && typeof entry === 'object') ? firstText(entry.date, entry.label, String(index)) : '';
      });
    } else {
      labels = (raw.labels || raw.dates || raw.x || raw.index || []).slice();
      var series = raw.series || raw.lines;
      if (Array.isArray(series) && series.length) {
        strategyRaw = series[0] && (series[0].values || series[0].data);
        benchmarkRaw = series[1] && (series[1].values || series[1].data);
      } else {
        strategyRaw = raw.strategy !== undefined ? raw.strategy
          : raw.nav !== undefined ? raw.nav
            : raw.portfolio_nav !== undefined ? raw.portfolio_nav
              : raw.net_value !== undefined ? raw.net_value
                : raw.current !== undefined ? raw.current
                  : raw.candidate !== undefined ? raw.candidate : raw.values;
        benchmarkRaw = raw.benchmark !== undefined ? raw.benchmark
          : raw.benchmark_nav !== undefined ? raw.benchmark_nav
            : raw.bench !== undefined ? raw.bench : raw.hs300;
      }
    }
    var strategy = axisSeries(strategyRaw);
    var benchmark = axisSeries(benchmarkRaw);
    if (!strategy || strategy.values.length < 2) return null;
    if (!labels.length) labels = strategy.labels || (benchmark && benchmark.labels) || [];
    if (!labels.length) labels = strategy.values.map(function (_, index) { return 'T' + (index + 1); });
    var benchmarkValues = benchmark ? benchmark.values : [];
    var size = strategy.values.length;
    return {
      labels: labels.slice(0, size),
      strategy: strategy.values.slice(0, size),
      benchmark: benchmarkValues.slice(0, size),
      benchmark_name: firstText(benchmark && benchmark.name, '沪深 300')
    };
  }

  function normalizeCompareNav(compare) {
    if (!compare) return null;
    var source = compare.nav && typeof compare.nav === 'object' ? compare.nav : compare;
    if (!Array.isArray(source) && source.nav && typeof source.nav === 'object' && !Array.isArray(source.nav) &&
      (source.nav.dates || source.nav.labels || source.nav.base_nav || source.nav.candidate_nav)) {
      source = source.nav;
    }
    var labels = (source.dates || source.labels || compare.dates || compare.labels || []).slice();
    var baseRaw = source.base_nav !== undefined ? source.base_nav
      : source.current_nav !== undefined ? source.current_nav
        : source.current !== undefined ? source.current
          : source.base !== undefined ? source.base
            : compare.base_nav !== undefined ? compare.base_nav
              : compare.current_nav;
    var candidateRaw = source.candidate_nav !== undefined ? source.candidate_nav
      : source.best_nav !== undefined ? source.best_nav
        : source.candidate !== undefined ? source.candidate
          : source.best !== undefined ? source.best
            : compare.candidate_nav !== undefined ? compare.candidate_nav
              : compare.best_nav;
    if (!labels.length && Array.isArray(source.series)) {
      labels = source.series[0] && (source.series[0].labels || []);
    }
    var base = axisSeries(baseRaw);
    var candidate = axisSeries(candidateRaw);
    if (!base && !candidate) return null;
    var benchmark = axisSeries(source.benchmark !== undefined ? source.benchmark : source.benchmark_nav);
    if (!labels.length) labels = (base && base.labels) || (candidate && candidate.labels) || [];
    var lengths = [];
    if (base) lengths.push(base.values.length);
    if (candidate) lengths.push(candidate.values.length);
    if (!lengths.length) return null;
    var size = Math.min.apply(null, lengths);
    if (!labels.length) labels = Array.from({ length: size }, function (_, index) { return 'T' + (index + 1); });
    return {
      labels: labels.slice(0, size),
      base: base ? base.values.slice(0, size) : null,
      candidate: candidate ? candidate.values.slice(0, size) : null,
      benchmark: benchmark ? benchmark.values.slice(0, size) : null
    };
  }

  /* ------------------------------------------------------------- 视图切换 */
  function getInitialView() {
    var raw = (window.location.hash || '').replace(/^#/, '');
    return VIEWS.indexOf(raw) >= 0 ? raw : 'chat';
  }

  function switchView(view, options) {
    var opts = options || {};
    if (VIEWS.indexOf(view) < 0) view = 'chat';
    state.activeView = view;
    all('.nav-item').forEach(function (button) {
      var active = button.getAttribute('data-view') === view;
      button.classList.toggle('is-active', active);
      if (active) button.setAttribute('aria-current', 'page');
      else button.removeAttribute('aria-current');
    });
    all('.view[data-view-panel]').forEach(function (panel) {
      var active = panel.getAttribute('data-view-panel') === view;
      panel.hidden = !active;
      panel.classList.toggle('is-active', active);
    });
    if (!opts.keepHash) {
      try { history.replaceState(null, '', '#' + view); } catch (err) { /* file:// 下可能受限 */ }
    }
    if (!opts.initial) window.scrollTo(0, 0);
    if (view === 'selection') renderSelectionView();
    if (view === 'interpretation') renderInterpretation();
    if (view === 'tune') {
      renderTuneView();
      if (state.booted && !state.mock) ensureVersions();
    }
    if (view === 'chat') {
      if (!state.messages.length && !state.chatBusy) renderChatEmpty();
      scrollChat(false);
    }
  }

  /* --------------------------------------------------------------- 启动引导 */
  function init() {
    bindEvents();
    setHealth('connecting', '连接中');
    updateSessionLabel();
    if (DEV_MOCK) {
      var flag = $('mock-flag');
      if (flag) flag.hidden = false;
    }
    renderProfile();
    renderSelectionView();
    renderInterpretation();
    renderTuneView();
    switchView(getInitialView(), { initial: true, keepHash: false });
    if (DEV_MOCK) bootMock();
    else bootLive();
  }

  async function bootLive() {
    var sessionId = readStoredSession();
    try {
      if (!sessionId) {
        sessionId = await createSession();
        storeSession(sessionId);
      }
      state.sessionId = sessionId;
      updateSessionLabel();
      var results = await Promise.all([
        request(sessionPath('/messages')),
        request(sessionPath('/profile'))
      ]);
      state.messages = Array.isArray(results[0] && results[0].messages) ? results[0].messages : [];
      state.profile = normalizeProfile(results[1] || {});
      state.booted = true;
      renderMessages();
      renderProfile();
      renderSelectionView();
      renderInterpretation();
      renderTuneView();
      if (state.activeView === 'tune') ensureVersions();
      loadHealth();
    } catch (err) {
      if (err.status === 404 && readStoredSession()) {
        clearStoredSession();
        state.sessionId = '';
        updateSessionLabel();
        return bootLive();
      }
      state.booted = false;
      state.sessionId = sessionId || '';
      updateSessionLabel();
      renderChatEmpty();
      renderProfile();
      renderSelectionView();
      showError('初始化失败：' + friendlyError(err), function () { hideError(); bootLive(); });
    }
  }

  /* DEV-ONLY：?mock=1 时装载本地 mock-data.js，不访问任何网络接口。 */
  function bootMock() {
    if (!window.MockData) {
      showError('Mock 数据未加载：请确认 frontend/mock-data.js 与 index.html 位于同一目录，且以 ?mock=1 打开。', function () { window.location.reload(); });
      renderChatEmpty();
      return;
    }
    var data = window.MockData;
    state.mock = true;
    state.sessionId = data.session_id || 'mock-session';
    state.profile = normalizeProfile(data.profile || {});
    state.messages = deepClone(data.messages || []);
    state.selection.result = normalizeSelectionPayload(data.selectionResult || {}, '2025-12-31');
    state.selection.status = 'complete';
    state.selection.runSeq += 1;
    state.interpretation.text = data.interpretation || '';
    state.interpretation.status = data.interpretation ? 'ready' : 'idle';
    state.versions = deepClone(data.versions || []);
    state.activeVersionId = data.active_id || ((state.versions.find(function (version) { return version.status === 'active'; }) || {}).id || '');
    state.versionsLoaded = true;
    state.booted = true;
    setHealth('warn', 'Mock 预览');
    var healthPill = $('health-pill');
    if (healthPill) healthPill.title = '开发预览：所有数据来自 frontend/mock-data.js';
    updateSessionLabel();
    renderMessages();
    renderProfile();
    renderSelectionView();
    renderInterpretation();
    renderTuneView();
    showToast('开发预览模式：当前页面不连接后端，展示的是 mock 数据。');
  }

  function loadHealth() {
    if (state.mock) return;
    fetch(API_BASE + '/health', { headers: { Accept: 'application/json' } }).then(function (response) {
      if (!response.ok) throw new Error('health ' + response.status);
      return response.json();
    }).then(function (payload) {
      state.health = payload || {};
      var llmOk = !payload || !payload.llm || payload.llm === 'ok';
      setHealth(llmOk ? 'ok' : 'warn', llmOk ? '服务在线' : 'LLM 异常');
      var pill = $('health-pill');
      if (pill) pill.title = '数据源：' + (payload.source || '—') + ' · 截至 ' + (payload.as_of || '—') + ' · LLM：' + (payload.llm || '—');
    }).catch(function () {
      setHealth('error', '服务离线');
    });
  }

  function setHealth(kind, text) {
    var pill = $('health-pill');
    if (!pill) return;
    pill.classList.remove('is-ok', 'is-warn', 'is-error');
    if (kind) pill.classList.add('is-' + kind);
    setText($('health-text'), text || '连接中');
  }

  function updateSessionLabel() {
    var node = $('session-label');
    if (!node) return;
    if (!state.sessionId) {
      node.textContent = 'SESSION —';
      return;
    }
    var suffix = state.sessionId.length > 8 ? state.sessionId.slice(-8) : state.sessionId;
    node.textContent = (state.mock ? 'MOCK · ' : 'SESSION · ') + suffix.toUpperCase();
    node.title = state.sessionId;
  }

  /* ------------------------------------------------------------ Markdown 安全渲染 */
  function inlineMarkdown(text) {
    var output = esc(text);
    output = output.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    output = output.replace(/`([^`]+)`/g, '<code>$1</code>');
    return output;
  }

  function renderMarkdown(input) {
    var lines = String(input == null ? '' : input).replace(/\r\n?/g, '\n').split('\n');
    var html = [];
    var paragraph = [];
    var list = null;
    var quote = [];

    function flushParagraph() {
      if (!paragraph.length) return;
      html.push('<p>' + paragraph.map(inlineMarkdown).join('<br>') + '</p>');
      paragraph = [];
    }
    function flushList() {
      if (!list) return;
      html.push('<' + list.type + '>' + list.items.map(function (item) { return '<li>' + inlineMarkdown(item) + '</li>'; }).join('') + '</' + list.type + '>');
      list = null;
    }
    function flushQuote() {
      if (!quote.length) return;
      html.push('<blockquote>' + quote.map(inlineMarkdown).join('<br>') + '</blockquote>');
      quote = [];
    }
    function flushAll() {
      flushParagraph(); flushList(); flushQuote();
    }

    lines.forEach(function (line) {
      var trimmed = line.trim();
      if (!trimmed) { flushAll(); return; }
      var heading = trimmed.match(/^#{1,4}\s+(.+)$/);
      if (heading) {
        flushAll();
        html.push('<h3>' + inlineMarkdown(heading[1]) + '</h3>');
        return;
      }
      if (/^>\s?/.test(trimmed)) {
        flushParagraph(); flushList();
        quote.push(trimmed.replace(/^>\s?/, ''));
        return;
      }
      var unordered = trimmed.match(/^[-*+]\s+(.+)$/);
      var ordered = trimmed.match(/^\d+[.)]\s+(.+)$/);
      if (unordered || ordered) {
        flushParagraph(); flushQuote();
        var type = unordered ? 'ul' : 'ol';
        var content = (unordered || ordered)[1];
        if (!list || list.type !== type) {
          flushList();
          list = { type: type, items: [] };
        }
        list.items.push(content);
        return;
      }
      flushList(); flushQuote();
      paragraph.push(trimmed);
    });
    flushAll();
    return html.join('');
  }

  /* --------------------------------------------------------------- 聊天视图 */
  function renderChatEmpty() {
    var box = $('chat-messages');
    if (!box) return;
    box.innerHTML =
      '<div class="chat-empty">' +
        '<div class="chat-empty-orbit" aria-hidden="true"><svg class="icon"><use href="#i-spark"></use></svg></div>' +
        '<h3>从一个投资目标开始</h3>' +
        '<p>不需要填冗长表单。用一句话描述你的风险偏好、期限与目标，AI 会逐轮追问并抽取可确认的画像槽位。</p>' +
        '<div class="empty-tags"><span>风险偏好</span><span>投资期限</span><span>回撤容忍</span><span>排除行业</span></div>' +
      '</div>';
    var quick = $('quick-prompts');
    if (quick) quick.hidden = false;
  }

  function renderMessages() {
    var box = $('chat-messages');
    if (!box) return;
    box.innerHTML = '';
    if (!state.messages.length) {
      renderChatEmpty();
      return;
    }
    var quick = $('quick-prompts');
    if (quick) quick.hidden = true;
    state.messages.forEach(function (message) {
      box.appendChild(createMessageElement(message, false).root);
    });
    scrollChat(true);
  }

  function createMessageElement(message, streaming) {
    var role = message.role === 'user' ? 'user' : 'assistant';
    var root = document.createElement('div');
    root.className = 'message message-' + role;
    var avatar = document.createElement('div');
    avatar.className = 'msg-avatar' + (role === 'user' ? ' user-avatar' : '');
    if (role === 'assistant') {
      avatar.innerHTML = '<svg class="icon"><use href="#i-spark"></use></svg>';
    } else {
      avatar.textContent = '你';
    }
    var main = document.createElement('div');
    main.className = 'msg-main';
    var meta = document.createElement('div');
    meta.className = 'msg-meta';
    var name = document.createElement('b');
    name.textContent = role === 'assistant' ? 'AI 画像分析师' : '你';
    var time = document.createElement('time');
    time.textContent = timeText(message.created_at) || (streaming ? '正在回复' : '');
    meta.appendChild(name);
    if (time.textContent) meta.appendChild(time);
    var bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    var textNode = document.createElement('div');
    textNode.className = 'message-text';
    if (!streaming) {
      if (role === 'assistant') textNode.innerHTML = renderMarkdown(message.content || '');
      else textNode.textContent = message.content || '';
    }
    bubble.appendChild(textNode);
    var dots = null;
    var caret = null;
    if (streaming) {
      dots = document.createElement('span');
      dots.className = 'typing-dots';
      dots.innerHTML = '<i></i><i></i><i></i>';
      bubble.appendChild(dots);
      caret = document.createElement('span');
      caret.className = 'typing-caret';
      caret.hidden = true;
      bubble.appendChild(caret);
    }
    main.appendChild(meta);
    main.appendChild(bubble);
    root.appendChild(avatar);
    root.appendChild(main);
    return { root: root, meta: meta, bubble: bubble, textEl: textNode, dots: dots, caret: caret, role: role };
  }

  function appendChatMessage(message) {
    var box = $('chat-messages');
    if (!box) return;
    if (!state.messages.length || box.querySelector('.chat-empty')) box.innerHTML = '';
    var quick = $('quick-prompts');
    if (quick) quick.hidden = true;
    box.appendChild(createMessageElement(message, false).root);
    scrollChat(true);
  }

  function appendStreamingChatBubble() {
    var box = $('chat-messages');
    if (!box) return null;
    if (box.querySelector('.chat-empty')) box.innerHTML = '';
    var quick = $('quick-prompts');
    if (quick) quick.hidden = true;
    var info = createMessageElement({ role: 'assistant', content: '', created_at: nowIso() }, true);
    box.appendChild(info.root);
    scrollChat(true);
    return info;
  }

  function setChatBusy(busy) {
    state.chatBusy = busy;
    var button = $('chat-send');
    var panel = document.querySelector('.chat-panel');
    if (button) {
      button.disabled = busy;
      var span = button.querySelector('span');
      if (span) span.textContent = busy ? '生成中' : '发送';
    }
    if (panel) panel.classList.toggle('is-streaming', busy);
  }

  function finalizeStreamBubble(info, text, markdown) {
    if (!info) return;
    if (info.dots) info.dots.remove();
    if (info.caret) info.caret.remove();
    if (markdown) info.textEl.innerHTML = renderMarkdown(text);
    else info.textEl.textContent = text;
  }

  function createTypewriter(node, onProgress) {
    var buffer = '';
    var text = '';
    var timer = null;
    var closing = false;
    var finishResolvers = [];
    var finished = false;

    function resolveIfDone() {
      if (!closing || buffer.length || timer || finished) return;
      finished = true;
      finishResolvers.splice(0).forEach(function (resolve) { resolve(text); });
    }
    function tick() {
      timer = null;
      if (!buffer.length) {
        resolveIfDone();
        return;
      }
      var take = buffer.length > 200 ? 5 : (buffer.length > 60 ? 3 : 2);
      text += buffer.slice(0, take);
      buffer = buffer.slice(take);
      node.textContent = text;
      if (typeof onProgress === 'function') onProgress();
      timer = setTimeout(tick, 15);
    }
    function start() {
      if (timer || !buffer.length) return;
      timer = setTimeout(tick, 15);
    }
    return {
      push: function (chunk) {
        if (!chunk) return;
        buffer += String(chunk);
        start();
      },
      finish: function () {
        closing = true;
        if (!buffer.length && !timer) {
          finished = true;
          return Promise.resolve(text);
        }
        return new Promise(function (resolve) { finishResolvers.push(resolve); });
      },
      abort: function () {
        if (timer) clearTimeout(timer);
        timer = null;
        if (buffer) {
          text += buffer;
          buffer = '';
          node.textContent = text;
        }
        closing = true;
        finished = true;
        finishResolvers.splice(0).forEach(function (resolve) { resolve(text); });
      },
      getText: function () { return text + buffer; }
    };
  }

  function handleSlotsUpdate(rawSlots) {
    if (!rawSlots || typeof rawSlots !== 'object') return;
    var incoming = normalizeSlots(rawSlots.slots || rawSlots);
    if (!Object.keys(incoming).length) return;
    var before = state.profile.slots || {};
    var merged = Object.assign({}, before, incoming);
    var changed = slotsChanged(before, merged);
    if (!changed) return;
    var wasConfirmed = state.profile.confirmed;
    state.profile.slots = merged;
    state.profile.complete = completeFromSlots(merged);
    state.profile.risk_level = merged.risk_level || state.profile.risk_level || 'balanced';
    if (wasConfirmed) state.profile.confirmed = false;
    renderProfile();
    if (wasConfirmed) showToast('画像槽位已更新，请重新确认后开始选股。');
  }

  function handleProfileReady(rawProfile) {
    if (!rawProfile || typeof rawProfile !== 'object') return;
    var incoming = rawProfile.profile || rawProfile;
    if (!incoming.slots && !SLOT_KEYS.some(function (key) { return incoming[key] !== undefined; })) return;
    var normalized = normalizeProfile(incoming);
    var mergedSlots = Object.assign({}, state.profile.slots, normalized.slots);
    var changed = slotsChanged(state.profile.slots, mergedSlots);
    var wasComplete = state.profile.complete;
    state.profile.slots = mergedSlots;
    state.profile.complete = typeof normalized.complete === 'boolean' ? normalized.complete : completeFromSlots(mergedSlots);
    if (!state.profile.complete) state.profile.complete = completeFromSlots(mergedSlots);
    state.profile.risk_level = normalized.risk_level || mergedSlots.risk_level || state.profile.risk_level || 'balanced';
    if (Object.prototype.hasOwnProperty.call(incoming, 'confirmed')) {
      state.profile.confirmed = !!incoming.confirmed;
    } else if (changed) {
      state.profile.confirmed = false;
    }
    renderProfile();
    if (!wasComplete && state.profile.complete) showToast('画像核心槽位已完整，可确认后开始选股。');
  }

  async function handleChatSubmit() {
    if (state.chatBusy) return;
    var input = $('chat-input');
    var content = (input && input.value || '').trim();
    if (!content) return;
    if (!state.sessionId) {
      showError('会话尚未就绪，请稍后重试。', function () { hideError(); bootLive(); });
      return;
    }
    input.value = '';
    autoGrow(input);
    await sendChatContent(content, { retry: false });
  }

  async function sendChatContent(content, options) {
    var opts = options || {};
    if (state.chatBusy) return;
    hideError();
    var userMessage = { role: 'user', content: content, created_at: nowIso() };
    if (!opts.retry) {
      state.messages.push(userMessage);
      appendChatMessage(userMessage);
    }
    setChatBusy(true);
    var info = appendStreamingChatBubble();
    var typer = createTypewriter(info.textEl, function () { scrollChat(false); });
    var collected = '';
    var firstToken = false;
    var handlers = {
      onToken: function (token) {
        if (!firstToken) {
          firstToken = true;
          if (info.dots) info.dots.remove();
          if (info.caret) info.caret.hidden = false;
        }
        collected += token;
        typer.push(token);
      },
      onSlots: handleSlotsUpdate,
      onProfileReady: handleProfileReady
    };
    try {
      await runConversationStream(content, 'chat', handlers);
      var finalText = await typer.finish();
      if (info.caret) info.caret.remove();
      if (!finalText.trim()) throw new AppError('AI 返回了空内容，请重试。', 0);
      finalizeStreamBubble(info, finalText, true);
      state.messages.push({ role: 'assistant', content: finalText, created_at: nowIso() });
      if (info.meta) {
        var oldTime = info.meta.querySelector('time');
        if (oldTime) oldTime.textContent = timeText(nowIso());
      }
      scrollChat(true);
    } catch (err) {
      typer.abort();
      if (info.caret) info.caret.remove();
      if (info.dots) info.dots.remove();
      var message = friendlyError(err);
      showError('AI 对话失败：' + message, function () {
        hideError();
        sendChatContent(content, { retry: true });
      });
    } finally {
      setChatBusy(false);
    }
  }

  async function runConversationStream(content, kind, handlers) {
    if (state.mock) return mockConversationStream(content, kind, handlers);
    return postMessageSse(content, handlers);
  }

  async function postMessageSse(content, handlers) {
    var response;
    try {
      response = await fetch(API_BASE + sessionPath('/messages'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
        body: JSON.stringify({ content: content })
      });
    } catch (err) {
      throw new AppError('无法连接后端服务，请确认服务已启动。', 0);
    }
    if (!response.ok) {
      var rawError = await response.text();
      var payload = null;
      try { payload = JSON.parse(rawError); } catch (err) { payload = rawError; }
      throw new AppError(extractApiError(payload, response.status), response.status);
    }
    if (!response.body) throw new AppError('浏览器不支持流式响应。', 0);

    var reader = response.body.getReader();
    var decoder = new TextDecoder('utf-8');
    var buffer = '';
    var eventCount = 0;
    var parseBlock = function (block) {
      if (!block || !block.trim()) return;
      var lines = block.split('\n');
      var eventName = 'message';
      var dataLines = [];
      lines.forEach(function (line) {
        if (!line || line.charAt(0) === ':') return;
        var colon = line.indexOf(':');
        var field = colon === -1 ? line : line.slice(0, colon);
        var value = colon === -1 ? '' : line.slice(colon + 1).replace(/^ /, '');
        if (field === 'event') eventName = value || 'message';
        if (field === 'data') dataLines.push(value);
      });
      if (!dataLines.length) return;
      var dataText = dataLines.join('\n');
      var payload = dataText;
      try { payload = JSON.parse(dataText); } catch (err) { payload = dataText; }
      eventCount += 1;
      if (eventName === 'token') {
        var token = '';
        if (typeof payload === 'string') token = payload;
        else if (payload && typeof payload === 'object') token = firstText(payload.text, payload.token, payload.delta, payload.content);
        if (token && handlers.onToken) handlers.onToken(token);
      } else if (eventName === 'slots') {
        var slots = payload && payload.slots ? payload.slots : payload;
        if (handlers.onSlots) handlers.onSlots(slots);
      } else if (eventName === 'profile_ready') {
        var profile = payload && payload.profile ? payload.profile : payload;
        if (handlers.onProfileReady) handlers.onProfileReady(profile);
      } else if (eventName === 'error') {
        var message = extractApiError(payload, 0);
        var error = new AppError(message || 'AI 对话流返回错误。', 0);
        error.fromSse = true;
        throw error;
      }
    };

    try {
      while (true) {
        var read = await reader.read();
        if (read.done) break;
        buffer += decoder.decode(read.value, { stream: true });
        buffer = buffer.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
        var boundary = buffer.indexOf('\n\n');
        while (boundary >= 0) {
          parseBlock(buffer.slice(0, boundary));
          buffer = buffer.slice(boundary + 2);
          boundary = buffer.indexOf('\n\n');
        }
      }
      buffer += decoder.decode();
      buffer = buffer.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
      if (buffer.trim()) parseBlock(buffer);
      if (!eventCount) throw new AppError('AI 返回了空响应，请重试。', 0);
    } finally {
      try { reader.releaseLock(); } catch (err) { /* ignore */ }
    }
  }

  /* ------------------------------------------------------------------ 画像 */
  function renderProfile() {
    var slotsBox = $('profile-slots');
    if (!slotsBox) return;
    var slots = state.profile.slots || {};
    var filledRequired = REQUIRED_SLOTS.filter(function (key) { return !!slots[key]; }).length;
    var progress = $('profile-progress-bar');
    if (progress) progress.style.width = String((filledRequired / REQUIRED_SLOTS.length) * 100) + '%';
    setText($('profile-required-count'), filledRequired + ' / ' + REQUIRED_SLOTS.length);

    slotsBox.innerHTML = '';
    SLOT_DEFS.forEach(function (def) {
      var raw = slots[def.key];
      var value = raw ? (def.formatter ? def.formatter(raw) : String(raw)) : '未填写';
      var row = document.createElement('div');
      row.className = 'slot-item' + (raw ? ' is-filled' : '');
      var dot = document.createElement('span');
      dot.className = 'slot-dot';
      var label = document.createElement('div');
      label.className = 'slot-label';
      label.textContent = def.label;
      if (def.required) {
        var required = document.createElement('em');
        required.textContent = '必填';
        label.appendChild(required);
      }
      var valueNode = document.createElement('div');
      valueNode.className = 'slot-value' + (raw ? '' : ' is-empty');
      valueNode.textContent = value;
      row.appendChild(dot);
      row.appendChild(label);
      row.appendChild(valueNode);
      slotsBox.appendChild(row);
    });

    var status = $('profile-status');
    var confirm = $('profile-confirm');
    var start = $('start-selection');
    var hint = $('profile-hint');
    var startHint = $('selection-start-hint');
    var confirmed = state.profile.confirmed;
    var complete = state.profile.complete;
    if (status) {
      status.classList.remove('confirmed', 'ready');
      if (confirmed) {
        status.textContent = '已确认';
        status.classList.add('confirmed');
      } else if (complete) {
        status.textContent = '待确认';
        status.classList.add('ready');
      } else {
        status.textContent = '待补全';
      }
    }
    if (confirm) {
      confirm.disabled = state.profileBusy || !complete || confirmed;
      confirm.textContent = confirmed ? '已确认' : (state.profileBusy ? '保存中…' : '确认画像');
    }
    if (start) {
      start.disabled = state.profileBusy || !complete || !confirmed || !state.booted || state.selection.status === 'running' || state.selection.status === 'starting';
      var span = start.querySelector('.btn-label');
      if (span) span.textContent = state.selection.status === 'running' ? '选股进行中…' : '开始选股';
    }
    if (hint) {
      if (confirmed) hint.textContent = '画像已确认，可开始 10 因子选股；也可以在「AI 解读」中继续追问。';
      else if (complete) hint.textContent = '核心槽位已完整，请确认后启用选股。';
      else hint.textContent = '补全风险偏好、投资期限、回撤容忍后可确认。';
    }
    if (startHint) {
      if (!state.booted) startHint.textContent = '会话初始化中';
      else if (!complete) startHint.textContent = '等待补全核心槽位';
      else if (!confirmed) startHint.textContent = '等待画像确认';
      else if (state.selection.status === 'running' || state.selection.status === 'starting') startHint.textContent = '选股任务进行中';
      else startHint.textContent = '画像已确认，可选股';
    }
    if ($('profile-save')) $('profile-save').disabled = state.profileBusy;
    if ($('profile-cancel')) $('profile-cancel').disabled = state.profileBusy;
    if ($('profile-edit')) $('profile-edit').disabled = state.profileBusy;
    var editor = $('profile-editor');
    if (editor) {
      editor.hidden = !state.profileEditing;
    }
  }

  function openProfileEditor() {
    if (state.profileBusy) return;
    var slots = state.profile.slots || {};
    SLOT_KEYS.forEach(function (key) {
      var input = $('slot-' + key);
      if (input) input.value = slots[key] || '';
    });
    state.profileEditing = true;
    renderProfile();
    var first = $('slot-risk_level');
    if (first) first.focus();
  }

  function closeProfileEditor() {
    state.profileEditing = false;
    renderProfile();
  }

  function collectEditorSlots() {
    var slots = {};
    SLOT_KEYS.forEach(function (key) {
      var input = $('slot-' + key);
      var value = input && input.value ? input.value.trim() : '';
      if (value) slots[key] = value;
    });
    return slots;
  }

  async function persistProfile(slots, confirmed) {
    if (state.profileBusy) return;
    if (!state.sessionId) {
      showError('会话尚未就绪，请稍后重试。', function () { hideError(); bootLive(); });
      return;
    }
    if (!completeFromSlots(slots)) {
      showToast('请先补全风险偏好、投资期限和回撤容忍。');
      return;
    }
    state.profileBusy = true;
    renderProfile();
    hideError();
    try {
      var payload;
      if (state.mock) {
        await sleep(320);
        payload = normalizeProfile({ slots: slots, confirmed: confirmed, complete: true, risk_level: slots.risk_level });
      } else {
        payload = await request(sessionPath('/profile'), {
          method: 'PUT',
          body: { slots: slots, confirmed: confirmed }
        });
      }
      state.profile = normalizeProfile(payload);
      state.profileEditing = false;
      renderProfile();
      if (state.profile.confirmed) showToast('画像已确认，可以开始选股了。');
      else showToast('画像已保存，尚待确认。');
    } catch (err) {
      showError('画像保存失败：' + friendlyError(err), function () { hideError(); persistProfile(slots, confirmed); });
    } finally {
      state.profileBusy = false;
      renderProfile();
    }
  }

  function handleConfirmProfile() {
    if (state.profileBusy) return;
    if (!state.profile.complete) {
      openProfileEditor();
      showToast('请先补全核心画像槽位。');
      return;
    }
    persistProfile(state.profile.slots, true);
  }

  function handleSaveProfile() {
    var slots = collectEditorSlots();
    var missing = REQUIRED_SLOTS.filter(function (key) { return !slots[key]; });
    if (missing.length) {
      showToast('请补全：' + missing.map(function (key) { return (SLOT_DEFS.find(function (item) { return item.key === key; }) || {}).label || key; }).join('、'));
      return;
    }
    persistProfile(slots, true);
  }

  /* ---------------------------------------------------------------- 选股结果 */
  function renderSelectionView() {
    var selection = state.selection;
    var running = selection.status === 'starting' || selection.status === 'running';
    var hasResult = selection.status === 'complete' && !!selection.result;
    var runButton = $('run-selection');
    var emptyButton = $('selection-empty-run');
    var select = $('selection-top-n');
    setHidden($('selection-progress'), !running);
    setHidden($('selection-skeleton'), !running);
    setHidden($('selection-content'), !hasResult);
    setHidden($('selection-empty'), running || hasResult);
    if (runButton) {
      runButton.disabled = running || !state.booted;
      var runSpan = runButton.querySelector('span');
      if (runSpan) runSpan.textContent = running ? '选股中…' : '重新选股';
    }
    if (emptyButton) {
      emptyButton.disabled = running || !state.booted || !state.profile.confirmed;
      var emptySpan = emptyButton.querySelector('span');
      if (emptySpan) emptySpan.textContent = selection.status === 'error' ? '重试选股' : '开始选股';
    }
    if (select) select.disabled = running;
    if (running) {
      var stage = selection.stage || (selection.status === 'starting' ? '任务创建中' : '因子计算');
      var message = selection.message || (selection.status === 'starting' ? '正在提交选股请求…' : '正在执行 10 因子评分与回测…');
      var percent = clamp(Number(selection.progress) || 0, 0, 100);
      setText($('selection-stage'), mapStage(stage));
      setText($('selection-progress-message'), message);
      setText($('selection-progress-percent'), Math.round(percent) + '%');
      var fill = $('selection-progress-fill');
      if (fill) fill.style.width = percent + '%';
    }
    if (!hasResult) {
      renderSelectionEmptyState();
      return;
    }
    renderSelectionData(selection.result);
  }

  function mapStage(stage) {
    var map = {
      loading_data: '读取行情数据',
      data: '读取行情数据',
      scoring: '因子评分',
      score: '因子评分',
      portfolio: '构建组合',
      construction: '构建组合',
      backtest: '历史回测',
      started: '任务已启动',
      running: '运行中'
    };
    return map[stage] || stage || '运行中';
  }

  function renderSelectionEmptyState() {
    var selection = state.selection;
    var title = $('selection-empty-title');
    var text = $('selection-empty-text');
    var actions = $('selection-go-chat');
    var runButton = $('selection-empty-run');
    if (selection.status === 'error') {
      setText(title, '本次选股未完成');
      setText(text, selection.error || '选股任务执行失败，请重试。当前页面不会展示任何模拟或降级结果。');
      if (actions) actions.hidden = false;
      if (runButton) runButton.hidden = false;
      return;
    }
    if (!state.booted) {
      setText(title, '正在初始化选股服务');
      setText(text, '正在连接后端并恢复会话状态，请稍候。');
      return;
    }
    if (!state.profile.confirmed) {
      setText(title, '先确认投资画像');
      setText(text, state.profile.complete ? '画像核心槽位已完整，请在对话页确认画像后再启动选股。' : '补全风险偏好、投资期限与回撤容忍，AI 才能按画像适配因子权重。');
      if (actions) actions.hidden = false;
      if (runButton) runButton.hidden = true;
      return;
    }
    setText(title, '尚未生成选股结果');
    setText(text, '投资画像已确认。点击开始选股，系统将执行 10 因子行业中性化打分、组合构建与 A 股规则回测。');
    if (actions) actions.hidden = true;
    if (runButton) {
      runButton.hidden = false;
      runButton.disabled = false;
    }
  }

  function renderSelectionData(result) {
    var metrics = result.metrics || {};
    setText($('selection-asof'), '数据截至 ' + (result.as_of || '—'));
    setText($('selection-meta'), 'A 股多因子 · 行业中性化 · ' + methodLabel(result.method) + ' · ' + (result.top && result.top.length || 0) + ' 只');
    setText($('selection-count'), 'Top ' + (result.top_n || result.top && result.top.length || 0));
    renderSelectionMetrics(metrics);
    renderSelectionTable(result.top || [], result.portfolio || []);
    renderSelectionFactorChart(result.top || []);
    renderPortfolio(result.portfolio || [], result.top || [], result.method);
    renderSelectionNav(result.nav, result.benchmark_name);
  }

  function renderSelectionMetrics(metrics) {
    var box = $('selection-metrics');
    if (!box) return;
    var items = [
      {
        label: '年化收益', en: 'CAGR', value: fmtPct(metrics.cagr, 1),
        tone: toNum(metrics.cagr) !== null && metrics.cagr < 0 ? 'negative' : 'positive',
        note: '回测区间年化收益'
      },
      {
        label: 'Sharpe', en: 'RISK ADJ', value: fmtNum(metrics.sharpe, 2),
        tone: 'plain', note: '风险调整后收益'
      },
      {
        label: '最大回撤', en: 'MAX DD', value: fmtPct(metrics.mdd, 1),
        tone: 'negative', note: '峰谷最大跌幅'
      },
      {
        label: '信息比率', en: 'IR', value: fmtNum(metrics.ir, 2),
        tone: 'plain', note: '相对沪深 300'
      },
      {
        label: '月胜率', en: 'WIN RATE', value: fmtPct(metrics.win_rate, 1),
        tone: 'plain', note: '月度正收益占比'
      }
    ];
    box.innerHTML = items.map(function (item) {
      var valueClass = item.tone === 'positive' ? ' positive' : (item.tone === 'negative' ? ' negative' : '');
      return '<div class="metric-card">' +
        '<div class="metric-top"><span>' + esc(item.label) + '</span><em>' + esc(item.en) + '</em></div>' +
        '<div class="metric-value' + valueClass + '">' + esc(item.value) + '</div>' +
        '<div class="metric-note">' + esc(item.note) + '</div>' +
      '</div>';
    }).join('');
  }

  function renderSelectionTable(top, portfolio) {
    var body = $('selection-table-body');
    if (!body) return;
    if (!top.length) {
      body.innerHTML = '<tr><td colspan="5" class="table-empty-cell">暂无可展示的入选股票。</td></tr>';
      return;
    }
    var scores = top.map(function (stock) { return toNum(stock.score); }).filter(function (value) { return value !== null; });
    var maxScore = scores.length ? Math.max.apply(null, scores) : null;
    var minScore = scores.length ? Math.min.apply(null, scores) : null;
    var portfolioMap = {};
    (portfolio || []).forEach(function (item) { portfolioMap[item.code] = item; });
    body.innerHTML = top.map(function (stock, index) {
      var factors = FACTOR_KEYS.map(function (key) {
        return { key: key, label: FACTOR_LABELS[key], value: stock.factors[key] };
      }).filter(function (item) { return item.value !== null; }).sort(function (a, b) { return b.value - a.value; }).slice(0, 2);
      var score = toNum(stock.score);
      var scoreRatio = 50;
      if (score !== null && maxScore !== null) {
        scoreRatio = maxScore === minScore ? 60 : 20 + ((score - minScore) / (maxScore - minScore)) * 80;
      }
      var rankClass = index < 3 ? ' rank-' + (index + 1) : '';
      var displayName = stock.name || stock.code || '—';
      var subCode = stock.name ? stock.code : '股票代码';
      return '<tr>' +
        '<td><span class="rank-badge' + rankClass + '">' + String(index + 1).padStart(2, '0') + '</span></td>' +
        '<td><div class="stock-id"><strong>' + esc(displayName) + '</strong><span>' + esc(subCode) + '</span></div></td>' +
        '<td><span class="industry-tag">' + esc(stock.industry || '—') + '</span></td>' +
        '<td><div class="score-cell"><span class="score-num">' + esc(fmtNum(score, 2)) + '</span><span class="score-track"><i style="width:' + clamp(scoreRatio, 0, 100).toFixed(1) + '%"></i></span></div></td>' +
        '<td><div class="factor-tags">' + (factors.length ? factors.map(function (factor) {
          return '<span class="factor-tag" title="' + esc(factor.label) + ' z=' + esc(fmtNum(factor.value, 2)) + '">' + esc(factor.label) + '</span>';
        }).join('') : '<span class="factor-tag">—</span>') + '</div></td>' +
      '</tr>';
    }).join('');
  }

  function renderSelectionFactorChart(top) {
    var container = $('selection-factor-chart');
    if (!container) return;
    var sums = {};
    var counts = {};
    FACTOR_KEYS.forEach(function (key) { sums[key] = 0; counts[key] = 0; });
    (top || []).forEach(function (stock) {
      FACTOR_KEYS.forEach(function (key) {
        var value = stock.factors && stock.factors[key];
        if (value !== null && value !== undefined && Number.isFinite(Number(value))) {
          sums[key] += Number(value);
          counts[key] += 1;
        }
      });
    });
    var items = FACTOR_KEYS.filter(function (key) { return counts[key] > 0; }).map(function (key) {
      return { label: FACTOR_LABELS[key], current: sums[key] / counts[key] };
    });
    if (!items.length) {
      chartEmpty(container, '等待因子暴露数据');
      return;
    }
    if (!window.Charts) {
      chartEmpty(container, '图表库未加载');
      return;
    }
    Charts.bar(container, { items: items }, {
      width: chartWidth(container, 340),
      rowHeight: 27,
      labelWidth: 75,
      showValues: true,
      valueFormatter: function (value) { return (value > 0 ? '+' : '') + fmtNum(value, 2) + 'z'; },
      ariaLabel: '入选组合因子平均暴露'
    });
  }

  function renderPortfolio(portfolio, top, method) {
    var container = $('portfolio-list');
    var methodNode = $('portfolio-method');
    if (!container) return;
    if (methodNode) methodNode.textContent = methodLabel(method || (portfolio[0] && portfolio[0].method));
    if (!portfolio.length) {
      container.innerHTML = '<p class="muted-small">等待组合权重数据。</p>';
      setText($('portfolio-total'), '—');
      setText($('portfolio-count'), '—');
      var ring = $('portfolio-ring');
      if (ring) {
        ring.style.setProperty('--p', '0');
        ring.innerHTML = '<span>—</span>';
      }
      return;
    }
    var nameMap = {};
    (top || []).forEach(function (stock) { nameMap[stock.code] = stock.name; });
    var weights = portfolio.map(function (item) { return toNum(item.weight_pct); }).filter(function (value) { return value !== null; });
    var totalWeight = weights.reduce(function (sum, value) { return sum + value; }, 0);
    var maxWeight = weights.length ? Math.max.apply(null, weights) : 1;
    var totalAmount = portfolio.some(function (item) { return toNum(item.amount) !== null; })
      ? portfolio.reduce(function (sum, item) { return sum + (toNum(item.amount) || 0); }, 0)
      : null;
    var ring = $('portfolio-ring');
    if (ring) {
      ring.style.setProperty('--p', String(clamp(totalWeight, 0, 100)));
      ring.innerHTML = '<span>' + esc(Math.round(totalWeight)) + '<small>%</small></span>';
    }
    setText($('portfolio-total'), totalAmount === null ? '—' : fmtMoney(totalAmount));
    setText($('portfolio-count'), portfolio.length + ' 只标的 · ' + (totalWeight >= 99.9 ? '满仓' : fmtNum(totalWeight, 1) + '% 配置'));
    var visible = portfolio.slice(0, 12);
    container.innerHTML = visible.map(function (item, index) {
      var weight = toNum(item.weight_pct);
      var width = weight === null ? 0 : clamp((weight / (maxWeight || 1)) * 100, 1, 100);
      var color = ALLOCATION_COLORS[index % ALLOCATION_COLORS.length];
      var name = nameMap[item.code] || item.code || '—';
      return '<div class="allocation-row">' +
        '<div class="allocation-name"><i style="background:' + color + '"></i><span>' + esc(name) + '</span><em>' + esc(item.code) + '</em></div>' +
        '<div class="allocation-pct">' + esc(weight === null ? '—' : (weight.toFixed(1) + '%')) + '</div>' +
        '<div class="allocation-track"><i style="width:' + width.toFixed(1) + '%;background:' + color + '"></i></div>' +
      '</div>';
    }).join('');
  }

  function renderSelectionNav(nav, benchmarkName) {
    var container = $('selection-nav-chart');
    if (!container) return;
    var caption = $('selection-nav-caption');
    if (!nav || !nav.strategy || nav.strategy.length < 2) {
      chartEmpty(container, '暂无净值数据（后端未返回 nav）');
      if (caption) caption.textContent = '当前结果没有可用于绘制的净值序列，不使用模拟数据替代。';
      return;
    }
    if (!window.Charts) {
      chartEmpty(container, '图表库未加载');
      return;
    }
    var series = [{
      name: '策略净值',
      values: nav.strategy,
      color: Charts.palette.cyan,
      fill: true
    }];
    if (nav.benchmark && nav.benchmark.length) {
      series.push({
        name: benchmarkName || nav.benchmark_name || '沪深 300',
        values: nav.benchmark,
        color: Charts.palette.slate,
        dashed: true,
        fill: false
      });
    }
    Charts.line(container, { labels: nav.labels, series: series }, {
      width: chartWidth(container, 700),
      height: 238,
      yFormatter: function (value) { return Number(value).toFixed(2); },
      xFormatter: function (value) { return String(value).slice(0, 7).replace('-', '.'); },
      ariaLabel: '策略净值与沪深 300 对比'
    });
    if (caption) {
      var first = nav.labels[0] || '—';
      var last = nav.labels[nav.labels.length - 1] || '—';
      caption.textContent = first + ' 至 ' + last + ' · T+1 成交 · 买卖双边成本 · 停牌与涨跌停约束已计入';
    }
  }

  async function startSelection() {
    if (!state.booted || !state.sessionId) {
      showError('会话尚未就绪，暂时无法启动选股。', function () { hideError(); bootLive(); });
      return;
    }
    if (!state.profile.complete || !state.profile.confirmed) {
      showToast('请先在「AI 对话画像」确认投资画像。');
      switchView('chat');
      openProfileEditor();
      return;
    }
    if (state.selection.status === 'starting' || state.selection.status === 'running') return;
    closeSelectionSource();
    clearTimers(MOCK_SELECTION_TIMERS);
    var topN = Number(($('selection-top-n') || {}).value) || 10;
    var runSeq = state.selection.runSeq + 1;
    state.selection = {
      status: 'starting',
      result: null,
      taskId: '',
      progress: 2,
      stage: '任务创建中',
      message: '正在提交选股请求…',
      error: '',
      startedAt: Date.now(),
      finishedAt: 0,
      source: null,
      runSeq: runSeq
    };
    resetInterpretation('');
    hideError();
    switchView('selection');
    renderSelectionView();
    try {
      if (state.mock) {
        mockStartSelection(topN, runSeq);
        return;
      }
      var payload = await request(sessionPath('/selection'), {
        method: 'POST',
        body: { top_n: String(topN) }
      });
      if (runSeq !== state.selection.runSeq) return;
      if (!payload || !payload.task_id) throw new AppError('选股任务创建失败：响应缺少 task_id。', 0);
      state.selection.taskId = String(payload.task_id);
      state.selection.status = 'running';
      state.selection.progress = 5;
      state.selection.stage = '任务已启动';
      state.selection.message = '正在读取因子与行情数据…';
      renderSelectionView();
      openSelectionEvents(runSeq);
    } catch (err) {
      handleSelectionFailure(err, runSeq);
    }
  }

  function openSelectionEvents(runSeq) {
    var url = API_BASE + sessionPath('/selection/events') + '?task_id=' + encodeURIComponent(state.selection.taskId);
    var source;
    try { source = new EventSource(url); }
    catch (err) { handleSelectionFailure(new AppError('浏览器无法建立 SSE 连接。', 0), runSeq); return; }
    state.selection.source = source;
    var settled = false;
    function isCurrent() { return runSeq === state.selection.runSeq; }
    function fail(message) {
      if (settled || !isCurrent()) return;
      settled = true;
      source.close();
      handleSelectionFailure(new AppError(message || '选股实时连接中断。', 0), runSeq);
    }
    source.addEventListener('started', function (event) {
      if (!isCurrent()) return;
      state.selection.status = 'running';
      state.selection.message = '任务已启动，正在计算因子…';
      state.selection.progress = Math.max(state.selection.progress, 5);
      parseEventData(event.data, {});
      renderSelectionView();
    });
    source.addEventListener('progress', function (event) {
      if (!isCurrent()) return;
      var payload = parseEventData(event.data, {});
      state.selection.status = 'running';
      state.selection.progress = clamp(Number(payload.percent) || state.selection.progress, 0, 100);
      state.selection.stage = payload.stage || state.selection.stage;
      state.selection.message = payload.message || state.selection.message;
      renderSelectionView();
    });
    source.addEventListener('result', function (event) {
      if (settled || !isCurrent()) return;
      var payload;
      try { payload = parseEventData(event.data, null); }
      catch (err) { fail('选股结果解析失败。'); return; }
      settled = true;
      source.close();
      handleSelectionResult(payload, runSeq);
    });
    source.addEventListener('error', function (event) {
      if (settled || !isCurrent()) return;
      if (event && event.data) {
        var payload = parseEventData(event.data, {});
        fail(extractApiError(payload, 0) || '选股任务返回错误。');
      } else {
        fail('选股事件连接中断，请重试。');
      }
    });
  }

  function closeSelectionSource() {
    if (state.selection.source) {
      try { state.selection.source.close(); } catch (err) { /* ignore */ }
      state.selection.source = null;
    }
  }

  function handleSelectionResult(payload, runSeq) {
    if (runSeq !== state.selection.runSeq) return;
    closeSelectionSource();
    var result;
    try {
      result = normalizeSelectionPayload(payload, state.health && state.health.as_of);
    } catch (err) {
      handleSelectionFailure(new AppError('选股结果格式异常，无法展示。', 0), runSeq);
      return;
    }
    state.selection.status = 'complete';
    state.selection.result = result;
    state.selection.progress = 100;
    state.selection.finishedAt = Date.now();
    state.selection.error = '';
    resetInterpretation(result.interpretation || '');
    hideError();
    renderSelectionView();
    renderProfile();
    if (state.activeView === 'selection') window.requestAnimationFrame(function () {
      var title = $('selection-title');
      if (title) title.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    showToast('选股完成：已生成 Top ' + (result.top_n || result.top.length) + ' 组合。');
  }

  function handleSelectionFailure(err, runSeq) {
    if (runSeq !== state.selection.runSeq) return;
    closeSelectionSource();
    state.selection.status = 'error';
    state.selection.error = friendlyError(err);
    state.selection.progress = 0;
    state.selection.finishedAt = Date.now();
    renderSelectionView();
    showError('选股失败：' + state.selection.error, function () {
      hideError();
      startSelection();
    });
  }

  function parseEventData(raw, fallback) {
    if (raw === undefined || raw === null || raw === '') return fallback;
    if (typeof raw === 'object') return raw;
    try { return JSON.parse(raw); } catch (err) { return raw; }
  }

  /* ---------------------------------------------------------------- AI 解读 */
  function resetInterpretation(text) {
    state.interpretation = {
      text: text || '',
      status: text ? 'ready' : 'idle',
      busy: false,
      busyKind: '',
      followups: [],
      typer: null
    };
    interpretationTyper = null;
    renderInterpretation();
  }

  function renderInterpretation() {
    if (!window.Charts) { /* charts not used here; 保持容错 */ }
    var result = state.selection.result;
    var interp = state.interpretation;
    var statusNode = $('interpret-status');
    var contextBox = $('interpret-context');
    var body = $('interpret-body');
    var thread = $('followup-thread');
    var input = $('followup-input');
    var send = $('followup-send');
    var followupHeadNote = document.querySelector('.followup-head span');
    if (!body) return;

    if (statusNode) {
      statusNode.classList.remove('ready', 'confirmed');
      if (interp.busy) {
        statusNode.textContent = interp.busyKind === 'main' ? '生成中' : '追问生成中';
        statusNode.classList.add('ready');
      } else if (interp.status === 'error') {
        statusNode.textContent = '生成中断';
      } else if (interp.text) {
        statusNode.textContent = '已生成';
        statusNode.classList.add('ready');
      } else if (result) {
        statusNode.textContent = '待生成';
      } else {
        statusNode.textContent = '暂无数据';
      }
    }

    if (contextBox) {
      if (result) {
        var chips = [
          { label: 'TOP', value: String(result.top_n || result.top.length) },
          { label: '风险', value: riskLabel(state.profile.risk_level || (state.profile.slots || {}).risk_level) },
          { label: '方法', value: methodLabel(result.method) },
          { label: 'AS OF', value: result.as_of || '—' }
        ];
        if (toNum(result.metrics.sharpe) !== null) chips.push({ label: 'SHARPE', value: fmtNum(result.metrics.sharpe, 2) });
        contextBox.innerHTML = chips.map(function (chip) {
          return '<span class="context-chip"><i class="ctx-dot"></i>' + esc(chip.label) + ' <b>' + esc(chip.value) + '</b></span>';
        }).join('');
      } else {
        contextBox.innerHTML = '<span class="context-chip"><i class="ctx-dot"></i>等待选股结果</span>';
      }
    }

    if (!interp.busy || interp.busyKind !== 'main') {
      if (interp.text) {
        body.innerHTML = renderMarkdown(interp.text);
        body.classList.remove('is-streaming');
      } else if (result) {
        body.innerHTML =
          '<div class="empty-inline">' +
            '<div class="empty-icon"><svg class="icon"><use href="#i-doc"></use></svg></div>' +
            '<h3>' + (interp.status === 'error' ? '解读生成中断' : '让 AI 解读本次选股') + '</h3>' +
            '<p>将基于本次 Top N、因子暴露与回测指标生成解读。内容只使用系统事实，不预测收益，不构成投资建议。</p>' +
            '<button class="btn btn-primary" type="button" data-action="generate-interpretation">' + (interp.status === 'error' ? '重新生成解读' : '让 AI 解读本次结果') + ' <svg class="icon"><use href="#i-arrow"></use></svg></button>' +
          '</div>';
      } else {
        body.innerHTML =
          '<div class="empty-inline">' +
            '<div class="empty-icon"><svg class="icon"><use href="#i-doc"></use></svg></div>' +
            '<h3>完成选股后生成解读</h3>' +
            '<p>AI 将结合选股结果、回测指标与因子暴露给出解读，不编造未获得的行情事实。</p>' +
            '<button class="btn btn-outline" type="button" data-action="go-selection">前往选股</button>' +
          '</div>';
      }
    }

    if (thread && (!interp.busy || interp.busyKind !== 'followup')) {
      renderFollowupThread();
    }
    var canFollow = !!result && !interp.busy;
    if (input) {
      input.disabled = !canFollow;
      input.placeholder = result ? '继续追问本次选股，例如：最大回撤的主要风险来源是什么？' : '完成选股后可继续追问';
    }
    if (send) send.disabled = !canFollow;
    if (followupHeadNote) followupHeadNote.textContent = result ? '复用同一会话消息接口' : '选股后可追问';
    renderInterpretSnapshot(result);
  }

  function renderFollowupThread() {
    var thread = $('followup-thread');
    if (!thread) return;
    thread.innerHTML = '';
    state.interpretation.followups.forEach(function (item) {
      thread.appendChild(createFollowupElement(item, false).root);
    });
  }

  function createFollowupElement(message, streaming) {
    var root = document.createElement('div');
    root.className = 'followup-msg' + (message.role === 'user' ? ' user' : '');
    var meta = document.createElement('div');
    meta.className = 'fu-meta';
    meta.textContent = (message.role === 'user' ? '你' : 'AI 研究员') + (message.created_at ? ' · ' + timeText(message.created_at) : '');
    var bubble = document.createElement('div');
    bubble.className = 'fu-bubble';
    var textNode = document.createElement('div');
    textNode.className = 'message-text';
    if (!streaming) {
      if (message.role === 'assistant') textNode.innerHTML = renderMarkdown(message.content || '');
      else textNode.textContent = message.content || '';
    }
    bubble.appendChild(textNode);
    var dots = null;
    var caret = null;
    if (streaming) {
      dots = document.createElement('span');
      dots.className = 'typing-dots';
      dots.innerHTML = '<i></i><i></i><i></i>';
      bubble.appendChild(dots);
      caret = document.createElement('span');
      caret.className = 'typing-caret';
      caret.hidden = true;
      bubble.appendChild(caret);
    }
    root.appendChild(meta);
    root.appendChild(bubble);
    return { root: root, textEl: textNode, bubble: bubble, dots: dots, caret: caret, meta: meta };
  }

  function renderInterpretSnapshot(result) {
    var box = $('interpret-snapshot');
    if (!box) return;
    if (!result) {
      box.innerHTML = '<p class="muted-small">暂无已完成的选股任务。</p>';
      return;
    }
    var metrics = result.metrics || {};
    var top = (result.top || []).slice(0, 3);
    var rows = [
      ['年化收益', fmtPct(metrics.cagr, 1)],
      ['Sharpe', fmtNum(metrics.sharpe, 2)],
      ['最大回撤', fmtPct(metrics.mdd, 1)],
      ['信息比率', fmtNum(metrics.ir, 2)]
    ];
    box.innerHTML =
      '<div class="snapshot-top">' +
        rows.map(function (row) { return '<div class="snapshot-row"><span>' + esc(row[0]) + '</span><b>' + esc(row[1]) + '</b></div>'; }).join('') +
      '</div>' +
      top.map(function (stock, index) {
        return '<div class="snapshot-stock"><i>' + String(index + 1).padStart(2, '0') + '</i><span class="snap-name">' + esc(stock.name || stock.code) + '</span><span class="snap-code">' + esc(stock.code) + '</span></div>';
      }).join('');
  }

  function buildInterpretationPrompt(result) {
    var topCodes = (result.top || []).slice(0, 10).map(function (stock) { return stock.code + (stock.name ? '(' + stock.name + ')' : ''); }).join('、');
    var metrics = result.metrics || {};
    return '请基于本次已完成的选股与回测结果，生成一份面向个人投资者的中文解读。' +
      '事实：截至 ' + (result.as_of || '未知') + '，Top ' + (result.top_n || result.top.length) + '；组合方法 ' + methodLabel(result.method) + '；' +
      '标的 ' + (topCodes || '无') + '；年化收益 ' + fmtPct(metrics.cagr, 1) + '，Sharpe ' + fmtNum(metrics.sharpe, 2) +
      '，最大回撤 ' + fmtPct(metrics.mdd, 1) + '，信息比率 ' + fmtNum(metrics.ir, 2) + '，月胜率 ' + fmtPct(metrics.win_rate, 1) + '。' +
      '请使用三个小节：选股逻辑、主要风险、建议跟踪项。只依据以上事实，不预测收益、不编造数据，并附一句风险提示，总字数不超过 600 字。';
  }

  async function generateInterpretation() {
    var result = state.selection.result;
    if (!result || state.interpretation.busy) {
      if (!result) showToast('请先完成选股。');
      return;
    }
    hideError();
    state.interpretation.busy = true;
    state.interpretation.busyKind = 'main';
    state.interpretation.status = 'generating';
    var body = $('interpret-body');
    if (!body) return;
    body.classList.add('is-streaming');
    body.innerHTML = '<div class="stream-target"></div><span class="typing-caret"></span>';
    var textNode = body.querySelector('.stream-target');
    var caret = body.querySelector('.typing-caret');
    var typer = createTypewriter(textNode, function () {});
    interpretationTyper = typer;
    state.interpretation.typer = typer;
    var status = $('interpret-status');
    if (status) { status.textContent = '生成中'; status.classList.add('ready'); }
    var input = $('followup-input');
    var send = $('followup-send');
    if (input) input.disabled = true;
    if (send) send.disabled = true;
    var collected = '';
    var firstToken = false;
    var handlers = {
      onToken: function (token) {
        if (!firstToken) { firstToken = true; if (caret) caret.hidden = false; }
        collected += token;
        typer.push(token);
      },
      onSlots: handleSlotsUpdate,
      onProfileReady: handleProfileReady
    };
    try {
      await runConversationStream(buildInterpretationPrompt(result), 'interpretation', handlers);
      var finalText = await typer.finish();
      if (!finalText.trim()) throw new AppError('AI 返回了空解读，请重试。', 0);
      state.interpretation.text = finalText;
      state.interpretation.status = 'ready';
      state.interpretation.busy = false;
      state.interpretation.busyKind = '';
      state.interpretation.typer = null;
      interpretationTyper = null;
      renderInterpretation();
      showToast('AI 解读已生成。');
    } catch (err) {
      typer.abort();
      state.interpretation.busy = false;
      state.interpretation.busyKind = '';
      state.interpretation.typer = null;
      interpretationTyper = null;
      state.interpretation.text = collected.trim();
      state.interpretation.status = 'error';
      renderInterpretation();
      showError('AI 解读生成失败：' + friendlyError(err), function () { hideError(); generateInterpretation(); });
    }
  }

  async function handleFollowupSubmit() {
    var input = $('followup-input');
    var content = (input && input.value || '').trim();
    if (!content || state.interpretation.busy) return;
    if (!state.selection.result) {
      showToast('请先完成选股。');
      return;
    }
    input.value = '';
    autoGrow(input);
    hideError();
    var userMessage = { role: 'user', content: content, created_at: nowIso() };
    state.interpretation.followups.push(userMessage);
    var thread = $('followup-thread');
    if (thread) thread.appendChild(createFollowupElement(userMessage, false).root);
    state.interpretation.busy = true;
    state.interpretation.busyKind = 'followup';
    var send = $('followup-send');
    if (send) send.disabled = true;
    if (input) input.disabled = true;
    var status = $('interpret-status');
    if (status) { status.textContent = '追问生成中'; status.classList.add('ready'); }
    var info = createFollowupElement({ role: 'assistant', content: '', created_at: nowIso() }, true);
    if (thread) thread.appendChild(info.root);
    var typer = createTypewriter(info.textEl, function () {
      if (thread) thread.scrollTop = thread.scrollHeight;
    });
    interpretationTyper = typer;
    state.interpretation.typer = typer;
    var collected = '';
    var firstToken = false;
    try {
      await runConversationStream(content, 'followup', {
        onToken: function (token) {
          if (!firstToken) {
            firstToken = true;
            if (info.dots) info.dots.remove();
            if (info.caret) info.caret.hidden = false;
          }
          collected += token;
          typer.push(token);
        },
        onSlots: handleSlotsUpdate,
        onProfileReady: handleProfileReady
      });
      var finalText = await typer.finish();
      if (!finalText.trim()) throw new AppError('AI 返回了空内容，请重试。', 0);
      finalizeStreamBubble(info, finalText, true);
      state.interpretation.followups.push({ role: 'assistant', content: finalText, created_at: nowIso() });
      state.interpretation.busy = false;
      state.interpretation.busyKind = '';
      state.interpretation.typer = null;
      interpretationTyper = null;
      renderInterpretation();
    } catch (err) {
      typer.abort();
      if (info.dots) info.dots.remove();
      if (info.caret) info.caret.remove();
      state.interpretation.busy = false;
      state.interpretation.busyKind = '';
      state.interpretation.typer = null;
      interpretationTyper = null;
      renderInterpretation();
      showError('追问失败：' + friendlyError(err), function () {
        hideError();
        sendFollowupRetry(content);
      });
    }
  }

  async function sendFollowupRetry(content) {
    if (state.interpretation.busy) return;
    hideError();
    state.interpretation.busy = true;
    state.interpretation.busyKind = 'followup';
    var thread = $('followup-thread');
    var info = createFollowupElement({ role: 'assistant', content: '', created_at: nowIso() }, true);
    if (thread) thread.appendChild(info.root);
    var typer = createTypewriter(info.textEl);
    interpretationTyper = typer;
    state.interpretation.typer = typer;
    var collected = '';
    try {
      await runConversationStream(content, 'followup', {
        onToken: function (token) { collected += token; typer.push(token); },
        onSlots: handleSlotsUpdate,
        onProfileReady: handleProfileReady
      });
      var finalText = await typer.finish();
      if (!finalText.trim()) throw new AppError('AI 返回了空内容，请重试。', 0);
      finalizeStreamBubble(info, finalText, true);
      state.interpretation.followups.push({ role: 'assistant', content: finalText, created_at: nowIso() });
      state.interpretation.busy = false;
      state.interpretation.busyKind = '';
      state.interpretation.typer = null;
      interpretationTyper = null;
      renderInterpretation();
    } catch (err) {
      typer.abort();
      state.interpretation.busy = false;
      state.interpretation.busyKind = '';
      state.interpretation.typer = null;
      interpretationTyper = null;
      renderInterpretation();
      showError('追问失败：' + friendlyError(err), function () { hideError(); sendFollowupRetry(content); });
    }
  }

  /* ---------------------------------------------------------------- 调参台 */
  function renderTuneView() {
    var tune = state.tune;
    var running = tune.status === 'starting' || tune.status === 'running';
    var done = tune.status === 'done';
    var hasData = tune.status !== 'idle' && (tune.status !== 'starting');
    var grid = $('tune-grid');
    if (!grid) return;
    grid.classList.toggle('is-loading', !hasData && !(tune.total > 0 || tune.trials.length));

    var titleState = $('tune-title-state');
    if (titleState) {
      if (running) titleState.textContent = '· 模型自动优化中';
      else if (done) titleState.textContent = '· 候选版本待启用';
      else if (tune.status === 'error') titleState.textContent = '· 调参中断';
      else titleState.textContent = '· 等待启动';
    }
    var pill = $('tune-status-pill');
    if (pill) {
      pill.classList.remove('is-running', 'is-done', 'is-error');
      if (running) pill.classList.add('is-running');
      else if (done) pill.classList.add('is-done');
      else if (tune.status === 'error') pill.classList.add('is-error');
    }
    var statusText = $('tune-status-text');
    if (statusText) {
      statusText.textContent = running ? (tune.trials.length ? '回测评估中' : '启动中') :
        done ? '候选就绪' : tune.status === 'error' ? '已中断' : '等待启动';
    }
    var runTag = $('tune-run-tag');
    if (runTag) runTag.textContent = running ? 'LIVE' : done ? 'DONE' : tune.status === 'error' ? 'ERROR' : 'IDLE';
    var startButton = $('start-tune');
    if (startButton) {
      startButton.disabled = running;
      var label = startButton.querySelector('span');
      if (label) label.textContent = running ? '调参中…' : done || tune.status === 'error' ? '重新调参' : '启动 AI 调参';
    }
    var trialSelect = $('tune-trial-select');
    if (trialSelect) trialSelect.disabled = running;
    setText($('tune-elapsed'), formatElapsed(tune.elapsedMs || (tune.startedAt && running ? Date.now() - tune.startedAt : 0)));

    var progress = tune.total > 0 ? clamp(tune.trials.length / tune.total * 100, 0, 100) : 0;
    if (done) progress = 100;
    if (window.Charts) {
      var gaugeNode = $('tune-gauge');
      if (gaugeNode) Charts.gauge(gaugeNode, { value: progress, max: 100, label: '进度' }, { size: 62, thickness: 4, ariaLabel: '调参进度' });
    }

    var compareLabel = $('tune-compare-label');
    if (compareLabel) {
      var activeLabel = versionLabel(state.activeVersionId) || '当前';
      var candidateLabel = versionLabel(tune.candidateVersionId) || (tune.bestWeights ? '候选' : '候选参数');
      if (tune.baseMetrics || tune.baseWeights) {
        compareLabel.textContent = '当前 ' + activeLabel + '  vs  ' + candidateLabel;
        if (tune.total) compareLabel.textContent += ' · ' + (done ? '完整回测' : tune.trials.length + ' / ' + tune.total + ' 组');
      } else {
        compareLabel.textContent = '等待基线权重载入';
      }
    }
    var candidateBadge = $('candidate-badge');
    if (candidateBadge) {
      candidateBadge.classList.toggle('is-on', !!tune.bestWeights);
      var versionSuffix = tune.candidateVersionId ? ' ' + esc(versionLabel(tune.candidateVersionId)) : '';
      candidateBadge.innerHTML = '<i></i>候选' + versionSuffix;
    }
    var metricLabel = $('tune-metric-label');
    if (metricLabel) {
      metricLabel.textContent = tune.baseMetrics ? ('候选 vs 当前 · ' + (done ? '回测完成' : (tune.total ? tune.trials.length + ' / ' + tune.total + ' 组随机搜索' : '载入中'))) : '候选 vs 当前 · 等待回测';
    }
    var bestTag = $('tune-best-tag');
    if (bestTag) bestTag.textContent = done ? 'BEST' : (tune.bestWeights ? 'LIVE' : '—');

    renderTuneTimeline();
    renderTuneCharts();
    renderTuneMetrics();
    renderVersionTimeline();
    renderTuneActions();
    if (grid.classList.contains('is-loading')) return;
    renderTuneNav();
    renderTuneVerdict();
  }

  function versionLabel(id) {
    if (!id) return '';
    return String(id);
  }

  function renderTuneTimeline() {
    var box = $('tune-timeline');
    if (!box) return;
    var tune = state.tune;
    var running = tune.status === 'starting' || tune.status === 'running';
    var done = tune.status === 'done';
    var complete = tune.trials.length;
    var total = tune.total || 0;
    var currentStep = 0;
    if (tune.status === 'starting') currentStep = 1;
    else if (running) currentStep = complete === 0 ? 2 : (complete >= total ? 4 : 3);
    else if (done) currentStep = 5;
    var baseSharpe = tune.baseMetrics ? fmtNum(tune.baseMetrics.sharpe, 2) : '—';
    var bestSharpe = tune.bestMetrics ? fmtNum(tune.bestMetrics.sharpe, 2) : '—';
    var descriptions = [
      tune.baseMetrics ? '读取当前权重与基线指标 · Sharpe ' + baseSharpe : '读取线上权重与基线指标',
      total ? '在权重和 = 100%、单项 ≤ 40% 约束内生成 ' + total + ' 组候选' : '在约束空间内生成随机候选权重',
      total ? '已完成 ' + complete + ' / ' + total + ' 组回测' + (running ? ' · 已用 ' + formatElapsed(Date.now() - (tune.startedAt || Date.now())) : '') : '逐组执行 A 股规则回测',
      tune.bestMetrics ? '对比新旧指标 · 候选 Sharpe ' + bestSharpe : '对比新旧指标，确认样本内 / 样本外稳健性',
      tune.candidateVersionId ? '候选 ' + versionLabel(tune.candidateVersionId) + ' 已就绪，等待人工启用' : '候选完成后等待人工启用'
    ];
    var labels = ['诊断基线', '生成候选参数', '候选回测', '对比新旧指标', '建议启用候选'];
    var icons = ['01', '02', '03', '04', '05'];
    box.innerHTML = labels.map(function (label, index) {
      var step = index + 1;
      var stepState;
      if (!running && !done && tune.status !== 'error') stepState = 'waiting';
      else if (done) stepState = step < 5 ? 'done' : 'active';
      else if (tune.status === 'error') stepState = step < Math.max(1, currentStep) ? 'done' : (step === Math.max(1, currentStep) ? 'active' : 'waiting');
      else stepState = step < currentStep ? 'done' : (step === currentStep ? 'active' : 'waiting');
      var statusLabel = stepState === 'done' ? (step === 5 ? '待启用' : '已完成') : stepState === 'active' ? '进行中' : '等待';
      if (done && step === 5) statusLabel = '待启用';
      return '<div class="timeline-step is-' + stepState + '"' + (stepState === 'active' ? ' aria-current="step"' : '') + '>' +
        '<span class="step-index">' + (stepState === 'done' ? '<svg class="icon" style="width:11px;height:11px"><use href="#i-check"></use></svg>' : icons[index]) + '</span>' +
        '<div class="step-body">' +
          '<div class="step-top"><h3>' + esc(label) + '</h3><span class="step-status">' + statusLabel + '</span></div>' +
          '<p>' + esc(descriptions[index]) + '</p>' +
        '</div>' +
        '<span class="step-pulse"></span>' +
      '</div>';
    }).join('');
    var note = $('tune-agent-note');
    if (note) {
      var text;
      if (tune.status === 'error') text = tune.error || '调参任务中断，请检查服务后重试。';
      else if (done) text = '本次共评估 ' + (tune.total || complete) + ' 组候选，最优目标值 ' + (tune.bestTrialIndex >= 0 ? fmtNum(tune.trials[tune.bestTrialIndex] && tune.trials[tune.bestTrialIndex].objective, 3) : fmtNum(tune.bestMetrics && tune.bestMetrics.sharpe, 2)) + '。版本变更仍需人工确认。';
      else if (running) text = complete ? '正在评估第 ' + Math.min(complete + 1, total || complete + 1) + ' 组候选；当前最优结构将实时映射到左侧权重与散点。' : '正在读取基线并生成第一批候选参数…';
      else text = '等待启动。AI 将在合法权重空间内生成候选并逐轮回测。';
      note.innerHTML = '<span class="note-mark"><svg class="icon"><use href="#i-spark"></use></svg></span><p>' + esc(text) + '</p>';
    }
  }

  function renderTuneCharts() {
    var tune = state.tune;
    var weightsBox = $('tune-weight-chart');
    var radarBox = $('tune-radar-chart');
    var scatterBox = $('tune-scatter-chart');
    if (!window.Charts) return;
    var hasBase = !!tune.baseWeights;
    var hasBest = !!tune.bestWeights;
    if (weightsBox) {
      if (hasBase || hasBest) {
        var items = FACTOR_KEYS.map(function (key) {
          return {
            label: FACTOR_LABELS[key],
            current: hasBase ? tune.baseWeights[key] : null,
            candidate: hasBest ? tune.bestWeights[key] : null
          };
        });
        Charts.bar(weightsBox, { items: items }, {
          width: chartWidth(weightsBox, 540),
          rowHeight: 27,
          labelWidth: 86,
          valueFormatter: function (value) { return fmtPct(value, 1); },
          animate: !tune.weightChartRendered,
          ariaLabel: '当前与候选因子权重对比'
        });
        tune.weightChartRendered = true;
      } else {
        chartEmpty(weightsBox, '等待基线权重载入');
      }
    }
    if (radarBox) {
      if (hasBase || hasBest) {
        Charts.radar(radarBox, {
          axes: FACTOR_KEYS.map(function (key) {
            return {
              label: FACTOR_SHORT[key],
              current: hasBase ? tune.baseWeights[key] : null,
              candidate: hasBest ? tune.bestWeights[key] : null
            };
          })
        }, {
          width: chartWidth(radarBox, 300),
          height: 236,
          ariaLabel: '因子权重雷达'
        });
      } else {
        chartEmpty(radarBox, '等待权重数据');
      }
    }
    if (scatterBox) {
      if (tune.trials.length) {
        var points = tune.trials.map(function (trial) {
          return {
            x: trial.metrics.sharpe,
            y: trial.objective,
            value: trial.objective,
            label: trial.trial_index,
            isBest: trial.is_best
          };
        });
        var highlight = tune.trials.findIndex(function (trial) { return trial.is_best; });
        if (highlight < 0) {
          var bestIndex = -1;
          var bestValue = -Infinity;
          tune.trials.forEach(function (trial, index) {
            if (trial.objective !== null && trial.objective > bestValue) { bestValue = trial.objective; bestIndex = index; }
          });
          highlight = bestIndex;
        }
        Charts.scatter(scatterBox, { points: points }, {
          width: chartWidth(scatterBox, 350),
          height: 236,
          highlightIndex: highlight,
          xLabel: 'Sharpe',
          yLabel: '目标 J',
          ariaLabel: '候选参数散点'
        });
        setText($('tune-scatter-count'), tune.trials.length + ' 组候选 · 按目标值着色');
      } else {
        chartEmpty(scatterBox, '等待候选参数');
        setText($('tune-scatter-count'), '0 组候选 · 按目标值着色');
      }
    }
  }

  function renderTuneMetrics() {
    var box = $('tune-metrics');
    if (!box) return;
    var tune = state.tune;
    if (!tune.baseMetrics && !tune.bestMetrics) {
      box.innerHTML = '<div class="chart-empty small">等待回测指标</div>';
      return;
    }
    var defs = [
      { key: 'sharpe', label: 'Sharpe 比率', short: 'Sharpe', digits: 2, percent: false },
      { key: 'cagr', label: '年化收益', short: 'CAGR', digits: 1, percent: true },
      { key: 'mdd', label: '最大回撤', short: 'MDD', digits: 1, percent: true, inverse: true },
      { key: 'ir', label: '信息比率', short: 'IR', digits: 2, percent: false }
    ];
    box.innerHTML = defs.map(function (def) {
      var base = tune.baseMetrics ? toNum(tune.baseMetrics[def.key]) : null;
      var best = tune.bestMetrics ? toNum(tune.bestMetrics[def.key]) : null;
      var displayValue = best !== null ? best : base;
      var improved = null;
      if (base !== null && best !== null) {
        improved = def.inverse ? Math.abs(best) < Math.abs(base) : best > base;
      }
      var mainText = displayValue === null ? '—' : (def.percent ? fmtPct(displayValue, def.digits) : fmtNum(displayValue, def.digits));
      var baseText = base === null ? '—' : (def.percent ? fmtPct(base, def.digits) : fmtNum(base, def.digits));
      var deltaText = '—';
      var deltaClass = '';
      if (base !== null && best !== null) {
        if (def.percent) {
          deltaText = fmtPp(best - base, def.digits);
        } else {
          var diff = best - base;
          deltaText = (diff > 0 ? '+' : diff < 0 ? '−' : '') + Math.abs(diff).toFixed(def.digits);
        }
        if (improved === true) deltaClass = 'good';
        else if (improved === false) deltaClass = 'bad';
      }
      return '<div class="tune-metric">' +
        '<div class="metric-label"><span>' + esc(def.short) + ' · ' + esc(def.label) + '</span><i class="metric-dot ' + (improved === true ? 'good' : improved === false ? 'bad' : '') + '"></i></div>' +
        '<div class="metric-big">' + esc(mainText) + '</div>' +
        '<div class="metric-sub"><span>基线 ' + esc(baseText) + '</span><span class="delta ' + (deltaClass === 'bad' ? 'bad' : '') + '">' + esc(deltaText) + '</span></div>' +
      '</div>';
    }).join('');
  }

  function renderTuneNav() {
    var container = $('tune-nav-chart');
    var empty = $('tune-nav-empty');
    if (!container) return;
    var nav = normalizeCompareNav(state.tune.compare);
    if (!nav || (!nav.base && !nav.candidate)) {
      container.innerHTML = '';
      if (empty) {
        empty.hidden = false;
        empty.textContent = state.tune.status === 'running' ? '候选回测完成后显示净值对比' : '本次结果未返回调参净值曲线';
      }
      return;
    }
    if (empty) empty.hidden = true;
    var series = [];
    if (nav.candidate) series.push({ name: '候选', values: nav.candidate, color: Charts.palette.cyan, fill: true });
    if (nav.base) series.push({ name: '当前', values: nav.base, color: Charts.palette.slate, dashed: true, fill: false });
    Charts.line(container, { labels: nav.labels, series: series }, {
      width: chartWidth(container, 390),
      height: 158,
      yFormatter: function (value) { return Number(value).toFixed(2); },
      xFormatter: function (value) { return String(value).slice(0, 4); },
      ariaLabel: '调参候选与当前净值对比'
    });
  }

  function renderTuneVerdict() {
    var node = $('tune-verdict');
    if (!node) return;
    var tune = state.tune;
    if (tune.status === 'done') {
      var baseSharpe = tune.baseMetrics && toNum(tune.baseMetrics.sharpe);
      var bestSharpe = tune.bestMetrics && toNum(tune.bestMetrics.sharpe);
      var delta = (baseSharpe !== null && baseSharpe !== undefined && bestSharpe !== null && bestSharpe !== undefined) ? bestSharpe - baseSharpe : null;
      node.textContent = delta === null
        ? '候选版本已完成回测，等待人工复核与启用。'
        : '候选相较当前 Sharpe ' + (delta >= 0 ? '提升 ' : '下降 ') + Math.abs(delta).toFixed(2) + '，已完成回测，等待人工启用。';
    } else if (tune.status === 'running') {
      node.textContent = '已完成 ' + tune.trials.length + ' / ' + (tune.total || '—') + ' 组候选。仅当最优目标值更新时，指标卡与权重才会切换。';
    } else if (tune.status === 'error') {
      node.textContent = '调参流已中断，当前展示的是已收到的真实事件，不补充模拟数据。';
    } else {
      node.textContent = '启动后由后端 tune SSE 实时驱动，不使用前端伪造进度。';
    }
  }

  function renderVersionTimeline() {
    var box = $('version-timeline');
    if (!box) return;
    var versions = (state.versions || []).slice().sort(function (a, b) {
      return String(a.created_at || '').localeCompare(String(b.created_at || ''));
    });
    if (!versions.length) {
      box.innerHTML = '<div class="version-empty">' + (state.versionsLoading ? '正在读取版本记录…' : '暂无可用版本记录') + '</div>';
      setText($('version-active-label'), '—');
      return;
    }
    var display = versions.slice(-4);
    var active = versions.find(function (version) { return version.id === state.activeVersionId; }) ||
      versions.find(function (version) { return version.status === 'active'; });
    setText($('version-active-label'), active ? '线上 · ' + active.id : '—');
    box.innerHTML = display.map(function (version) {
      var isActive = version.id === state.activeVersionId || version.status === 'active';
      var isCandidate = !isActive && (version.id === state.tune.candidateVersionId || version.status === 'candidate');
      var cls = isActive ? 'is-active' : isCandidate ? 'is-candidate' : 'is-history';
      var sub = isActive ? '当前' : isCandidate ? '候选' : '历史';
      return '<div class="version-node ' + cls + '" title="' + esc(version.id + ' · ' + (version.status || 'history')) + '">' +
        '<span class="version-dot"></span>' +
        '<span class="version-label">' + esc(version.id) + '</span>' +
        '<span class="version-sub">' + sub + '</span>' +
      '</div>';
    }).join('');
  }

  function renderTuneActions() {
    var tune = state.tune;
    var done = tune.status === 'done';
    var activate = $('tune-activate');
    var rollback = $('tune-rollback');
    var hint = $('tune-action-hint');
    var activeIsCandidate = done && tune.candidateVersionId && state.activeVersionId === tune.candidateVersionId;
    if (activate) {
      activate.disabled = !done || !tune.candidateVersionId || activeIsCandidate || tune.actionBusy;
      activate.innerHTML = '<svg class="icon"><use href="#i-' + (activeIsCandidate ? 'check' : 'check') + '"></use></svg>' +
        (activeIsCandidate ? '候选已启用' : '启用候选版本');
    }
    if (rollback) {
      rollback.disabled = !done || !state.activeVersionId || tune.actionBusy;
    }
    if (hint) {
      if (tune.status === 'running' || tune.status === 'starting') hint.textContent = '调参进行中 · 完成后可人工启用候选版本';
      else if (activeIsCandidate) hint.textContent = '候选 ' + versionLabel(tune.candidateVersionId) + ' 已启用 · 可按需回滚到上一版本';
      else if (done && tune.candidateVersionId) hint.textContent = '候选 ' + versionLabel(tune.candidateVersionId) + ' 已通过回测 · 所有变更需人工确认';
      else if (tune.status === 'error') hint.textContent = '调参中断 · 可重新启动或稍后重试';
      else hint.textContent = '所有变更需人工确认 · 历史版本可回滚';
    }
  }

  function updateTuneElapsed() {
    var tune = state.tune;
    if (tune.startedAt && (tune.status === 'running' || tune.status === 'starting')) {
      tune.elapsedMs = Date.now() - tune.startedAt;
      setText($('tune-elapsed'), formatElapsed(tune.elapsedMs));
    }
  }

  function startTuneClock() {
    stopTuneClock();
    tuneElapsedTimer = setInterval(updateTuneElapsed, 1000);
  }

  function stopTuneClock() {
    if (tuneElapsedTimer) clearInterval(tuneElapsedTimer);
    tuneElapsedTimer = null;
    if (state.tune.startedAt) {
      state.tune.elapsedMs = state.tune.status === 'running' || state.tune.status === 'starting'
        ? Date.now() - state.tune.startedAt
        : state.tune.elapsedMs || (Date.now() - state.tune.startedAt);
    }
  }

  function scheduleTuneRender() {
    if (state.tune.renderQueued) return;
    state.tune.renderQueued = true;
    if (tuneRenderHandle) cancelAnimationFrame(tuneRenderHandle);
    tuneRenderHandle = requestAnimationFrame(function () {
      tuneRenderHandle = null;
      state.tune.renderQueued = false;
      if (state.activeView === 'tune') renderTuneView();
    });
  }

  async function startTune() {
    if (!state.booted || !state.sessionId) {
      showError('会话尚未就绪，暂时无法启动调参。', function () { hideError(); bootLive(); });
      return;
    }
    if (state.tune.status === 'starting' || state.tune.status === 'running') return;
    hideError();
    closeTuneSource();
    clearTimers(MOCK_TUNE_TIMERS);
    stopTuneClock();
    var requested = Number(($('tune-trial-select') || {}).value) || 24;
    var runSeq = state.tune.runSeq + 1;
    state.tune = {
      status: 'starting',
      jobId: '',
      total: 0,
      trials: [],
      baseWeights: null,
      bestWeights: null,
      baseMetrics: null,
      bestMetrics: null,
      compare: null,
      candidateVersionId: '',
      bestTrialIndex: -1,
      startedAt: Date.now(),
      finishedAt: 0,
      elapsedMs: 0,
      error: '',
      source: null,
      runSeq: runSeq,
      renderQueued: false,
      weightChartRendered: false,
      actionBusy: false,
      currentStep: 1
    };
    if (state.activeView !== 'tune') switchView('tune');
    renderTuneView();
    startTuneClock();
    try {
      if (state.mock) {
        mockStartTune(requested, runSeq);
        return;
      }
      var payload = await request(sessionPath('/tune'), {
        method: 'POST',
        body: { trials: requested }
      });
      if (runSeq !== state.tune.runSeq) return;
      if (!payload || !payload.job_id) throw new AppError('调参任务创建失败：响应缺少 job_id。', 0);
      state.tune.jobId = String(payload.job_id);
      state.tune.status = 'running';
      renderTuneView();
      openTuneEvents(runSeq);
    } catch (err) {
      handleTuneFailure(err, runSeq);
    }
  }

  function openTuneEvents(runSeq) {
    var url = API_BASE + sessionPath('/tune/events') + '?job_id=' + encodeURIComponent(state.tune.jobId);
    var source;
    try { source = new EventSource(url); }
    catch (err) { handleTuneFailure(new AppError('浏览器无法建立调参 SSE 连接。', 0), runSeq); return; }
    state.tune.source = source;
    var settled = false;
    function isCurrent() { return runSeq === state.tune.runSeq; }
    function fail(message) {
      if (settled || !isCurrent()) return;
      settled = true;
      source.close();
      handleTuneFailure(new AppError(message || '调参事件连接中断。', 0), runSeq);
    }
    source.addEventListener('tuning_started', function (event) {
      if (!isCurrent()) return;
      var payload = parseEventData(event.data, {});
      handleTuneStarted(payload, runSeq);
    });
    source.addEventListener('trial', function (event) {
      if (!isCurrent()) return;
      var payload;
      try { payload = parseEventData(event.data, null); }
      catch (err) { fail('调参 trial 数据解析失败。'); return; }
      handleTuneTrial(payload, runSeq);
    });
    source.addEventListener('best_update', function (event) {
      if (!isCurrent()) return;
      var payload = parseEventData(event.data, {});
      handleBestUpdate(payload, runSeq);
    });
    source.addEventListener('done', function (event) {
      if (settled || !isCurrent()) return;
      var payload = parseEventData(event.data, {});
      settled = true;
      source.close();
      handleTuneDone(payload, runSeq);
    });
    source.addEventListener('error', function (event) {
      if (settled || !isCurrent()) return;
      if (event && event.data) {
        var payload = parseEventData(event.data, {});
        fail(extractApiError(payload, 0) || '调参流返回错误。');
      } else {
        fail('调参事件连接中断，请重试。');
      }
    });
  }

  function closeTuneSource() {
    if (state.tune.source) {
      try { state.tune.source.close(); } catch (err) { /* ignore */ }
      state.tune.source = null;
    }
  }

  function normalizeTrial(raw) {
    var source = raw || {};
    var metrics = normalizeMetrics(source.metrics || {});
    var objective = firstNum(source.objective);
    if (objective === null && metrics.sharpe !== null) {
      objective = metrics.sharpe - 0.5 * Math.abs(metrics.mdd || 0) + 0.2 * (metrics.ir || 0);
    }
    return {
      trial_index: firstNum(source.trial_index, source.index) || 0,
      weights: normalizeWeights(source.weights),
      metrics: metrics,
      objective: objective,
      is_best: !!source.is_best
    };
  }

  function handleTuneStarted(payload, runSeq) {
    if (runSeq !== state.tune.runSeq) return;
    var tune = state.tune;
    tune.status = 'running';
    tune.jobId = firstText(payload.job_id, tune.jobId);
    tune.total = firstNum(payload.trials_total, payload.total, tune.total) || 0;
    tune.baseWeights = normalizeWeights(payload.base_weights) || tune.baseWeights;
    tune.baseMetrics = Object.keys(payload.base_metrics || {}).length ? normalizeMetrics(payload.base_metrics) : tune.baseMetrics;
    tune.startedAt = tune.startedAt || Date.now();
    tune.weightChartRendered = false;
    var grid = $('tune-grid');
    if (grid) grid.classList.remove('is-loading');
    startTuneClock();
    renderTuneView();
    scheduleTuneRender();
  }

  function handleTuneTrial(payload, runSeq) {
    if (runSeq !== state.tune.runSeq) return;
    var trial = normalizeTrial(payload);
    if (!trial.trial_index) trial.trial_index = state.tune.trials.length + 1;
    state.tune.trials.push(trial);
    if (trial.is_best) {
      if (trial.weights) state.tune.bestWeights = trial.weights;
      state.tune.bestMetrics = trial.metrics;
      state.tune.bestTrialIndex = state.tune.trials.length - 1;
    }
    if (!state.tune.total) state.tune.total = state.tune.trials.length;
    state.tune.currentStep = state.tune.trials.length >= state.tune.total ? 4 : 3;
    updateTuneElapsed();
    scheduleTuneRender();
  }

  function handleBestUpdate(payload, runSeq) {
    if (runSeq !== state.tune.runSeq) return;
    var weights = normalizeWeights(payload.best_weights);
    if (weights) state.tune.bestWeights = weights;
    if (payload.best_metrics && Object.keys(payload.best_metrics).length) {
      state.tune.bestMetrics = normalizeMetrics(payload.best_metrics);
    }
    var bestIndex = state.tune.trials.findIndex(function (trial) { return trial.is_best; });
    if (bestIndex >= 0) state.tune.bestTrialIndex = bestIndex;
    scheduleTuneRender();
  }

  function applyBestFromTrials() {
    var tune = state.tune;
    var bestIndex = -1;
    var bestObjective = -Infinity;
    tune.trials.forEach(function (trial, index) {
      if (trial.is_best) { bestIndex = index; return; }
      if (trial.objective !== null && trial.objective > bestObjective) {
        bestObjective = trial.objective;
      }
    });
    if (bestIndex < 0) {
      tune.trials.forEach(function (trial, index) {
        if (trial.objective !== null && trial.objective > bestObjective) {
          bestObjective = trial.objective;
          bestIndex = index;
        }
      });
    }
    if (bestIndex >= 0) {
      tune.bestTrialIndex = bestIndex;
      if (!tune.bestWeights && tune.trials[bestIndex].weights) tune.bestWeights = tune.trials[bestIndex].weights;
      if (!tune.bestMetrics) tune.bestMetrics = tune.trials[bestIndex].metrics;
    }
  }

  function handleTuneDone(payload, runSeq) {
    if (runSeq !== state.tune.runSeq) return;
    closeTuneSource();
    stopTuneClock();
    var tune = state.tune;
    tune.status = 'done';
    tune.finishedAt = Date.now();
    tune.elapsedMs = tune.startedAt ? tune.finishedAt - tune.startedAt : tune.elapsedMs;
    tune.candidateVersionId = firstText(payload.candidate_version_id, payload.version_id, tune.candidateVersionId);
    tune.compare = payload.compare || tune.compare || null;
    // 部分实现可能只在 done 中回传最终权重/指标，做兼容读取但不伪造。
    if (payload.best_weights) tune.bestWeights = normalizeWeights(payload.best_weights) || tune.bestWeights;
    if (payload.best_metrics && Object.keys(payload.best_metrics).length) tune.bestMetrics = normalizeMetrics(payload.best_metrics);
    applyBestFromTrials();
    if (!tune.total) tune.total = tune.trials.length;
    if (!tune.bestWeights) {
      var compare = payload.compare || {};
      tune.bestWeights = normalizeWeights(compare.best_weights || compare.candidate_weights) || tune.bestWeights;
    }
    if (!tune.bestMetrics) {
      var compareMetrics = (payload.compare || {}).candidate_metrics || (payload.compare || {}).best_metrics;
      if (compareMetrics) tune.bestMetrics = normalizeMetrics(compareMetrics);
    }
    upsertCandidateVersion();
    var grid = $('tune-grid');
    if (grid) grid.classList.remove('is-loading');
    renderTuneView();
    ensureVersions(true);
    showToast('AI 调参完成，候选版本等待人工启用。');
  }

  function handleTuneFailure(err, runSeq) {
    if (runSeq !== state.tune.runSeq) return;
    closeTuneSource();
    stopTuneClock();
    state.tune.status = 'error';
    state.tune.error = friendlyError(err);
    state.tune.elapsedMs = state.tune.startedAt ? Date.now() - state.tune.startedAt : state.tune.elapsedMs;
    renderTuneView();
    showError('调参失败：' + state.tune.error, function () {
      hideError();
      startTune();
    });
  }

  function upsertCandidateVersion() {
    var tune = state.tune;
    if (!tune.candidateVersionId) return;
    var existingIndex = state.versions.findIndex(function (version) { return version.id === tune.candidateVersionId; });
    var candidate = {
      id: tune.candidateVersionId,
      weights: tune.bestWeights,
      metrics: tune.bestMetrics,
      status: 'candidate',
      created_at: nowIso(),
      parent_id: state.activeVersionId || null
    };
    if (existingIndex >= 0) {
      candidate = Object.assign({}, state.versions[existingIndex], candidate);
      candidate.status = state.versions[existingIndex].status === 'active' ? 'active' : 'candidate';
      state.versions[existingIndex] = candidate;
    } else {
      state.versions.push(candidate);
    }
    renderVersionTimeline();
  }

  async function ensureVersions(force) {
    if (state.mock) {
      state.versionsLoaded = true;
      renderVersionTimeline();
      renderTuneActions();
      return;
    }
    if (!state.booted) return;
    if (state.versionsLoading) return;
    if (state.versionsLoaded && !force) return;
    state.versionsLoading = true;
    state.versionError = '';
    try {
      var payload = await request('/versions');
      var versions = payload && Array.isArray(payload.versions) ? payload.versions : [];
      state.versions = versions.map(function (version) {
        return {
          id: String(version.id),
          weights: normalizeWeights(version.weights),
          metrics: normalizeMetrics(version.metrics || {}),
          status: version.status || 'archived',
          created_at: version.created_at || '',
          parent_id: version.parent_id || null
        };
      });
      state.activeVersionId = firstText(payload && payload.active_id, state.activeVersionId);
      if (!state.activeVersionId) {
        var active = state.versions.find(function (version) { return version.status === 'active'; });
        if (active) state.activeVersionId = active.id;
      }
      state.versionsLoaded = true;
      renderVersionTimeline();
      renderTuneActions();
      if (state.activeView === 'tune') renderTuneView();
    } catch (err) {
      state.versionError = friendlyError(err);
      renderVersionTimeline();
      showError('版本记录加载失败：' + state.versionError, function () { hideError(); ensureVersions(true); });
    } finally {
      state.versionsLoading = false;
    }
  }

  async function performVersionAction(action) {
    if (state.tune.actionBusy) return;
    var target;
    if (action === 'activate') {
      target = state.tune.candidateVersionId;
      if (!target) { showToast('候选版本尚未就绪。'); return; }
    } else {
      target = state.activeVersionId;
      if (!target) { showToast('当前没有可回滚的线上版本。'); return; }
    }
    state.tune.actionBusy = true;
    renderTuneActions();
    hideError();
    try {
      var payload;
      if (state.mock) {
        payload = await mockVersionAction(action, target);
      } else {
        payload = await request('/versions/' + encodeURIComponent(target) + '/' + (action === 'activate' ? 'activate' : 'rollback'), {
          method: 'POST',
          body: { reason: action === 'activate' ? '用户在 AI 调参台人工启用候选版本' : '用户在 AI 调参台人工回滚当前线上版本' }
        });
      }
      if (payload && payload.active_id) state.activeVersionId = String(payload.active_id);
      if (state.mock) {
        renderVersionTimeline();
        renderTuneActions();
      } else {
        state.versionsLoaded = false;
        await ensureVersions(true);
      }
      showToast(action === 'activate' ? '候选版本已启用。' : '已回滚到上一版本。');
    } catch (err) {
      showError((action === 'activate' ? '启用失败：' : '回滚失败：') + friendlyError(err), function () {
        hideError();
        performVersionAction(action);
      });
    } finally {
      state.tune.actionBusy = false;
      renderTuneActions();
    }
  }

  /* =========================================================================
   * DEV-ONLY：Mock 适配器
   * 仅在 ?mock=1 且 mock-data.js 已加载时被调用；生产路径不会执行。
   * ========================================================================= */
  function mockStartSelection(topN, runSeq) {
    clearTimers(MOCK_SELECTION_TIMERS);
    var result = deepClone(window.MockData.selectionResult || {});
    var top = (result.top || []).slice(0, Math.min(topN, (result.top || []).length));
    var portfolio = (result.portfolio || []).slice(0, top.length);
    var selectedWeights = portfolio.map(function (item) { return toNum(item.weight_pct) || 0; });
    var weightSum = selectedWeights.reduce(function (sum, value) { return sum + value; }, 0) || 1;
    portfolio = portfolio.map(function (item, index) {
      var weight = Number((selectedWeights[index] / weightSum * 100).toFixed(1));
      return Object.assign({}, item, { weight_pct: weight, amount: Math.round(weight / 100 * 1000000) });
    });
    result.top = top;
    result.portfolio = portfolio;
    result.top_n = top.length;
    result.nav = window.MockData.selectionResult.nav;
    var staged = [
      { stage: 'loading_data', percent: 18, message: '读取示例行情与基本面缓存…' },
      { stage: 'scoring', percent: 46, message: '计算 10 因子并进行行业中性化…' },
      { stage: 'portfolio', percent: 71, message: '按风险平价构建组合…' },
      { stage: 'backtest', percent: 88, message: '执行 T+1、涨跌停与成本约束回测…' }
    ];
    function later(ms, fn) {
      MOCK_SELECTION_TIMERS.push(setTimeout(function () {
        if (runSeq !== state.selection.runSeq) return;
        fn();
      }, ms));
    }
    later(220, function () {
      state.selection.status = 'running';
      state.selection.progress = 8;
      state.selection.stage = 'started';
      state.selection.message = '示例选股任务已启动…';
      renderSelectionView();
    });
    staged.forEach(function (item, index) {
      later(500 + index * 330, function () {
        state.selection.progress = item.percent;
        state.selection.stage = item.stage;
        state.selection.message = item.message;
        renderSelectionView();
      });
    });
    later(2050, function () {
      handleSelectionResult(result, runSeq);
    });
  }

  function mockStartTune(requested, runSeq) {
    clearTimers(MOCK_TUNE_TIMERS);
    var data = window.MockData.tuning;
    var all = data.trials.slice();
    var count = Math.max(3, Math.min(requested, all.length));
    var selected = all.slice();
    if (count < all.length) {
      var best = all.find(function (trial) { return trial.is_best; });
      selected = all.slice(0, count - 1);
      if (best && selected.indexOf(best) < 0) selected.push(best);
      selected.sort(function (a, b) { return a.trial_index - b.trial_index; });
    }
    var total = selected.length;
    function later(ms, fn) {
      MOCK_TUNE_TIMERS.push(setTimeout(function () {
        if (runSeq !== state.tune.runSeq) return;
        fn();
      }, ms));
    }
    later(320, function () {
      handleTuneStarted({
        job_id: data.job_id,
        trials_total: total,
        base_weights: data.base_weights,
        base_metrics: data.base_metrics
      }, runSeq);
    });
    selected.forEach(function (trial, index) {
      later(650 + index * 125, function () {
        handleTuneTrial(trial, runSeq);
        if (trial.is_best) {
          later(160, function () {
            handleBestUpdate({ best_weights: data.best_weights, best_metrics: data.best_metrics }, runSeq);
          });
        }
      });
    });
    later(850 + total * 125, function () {
      handleTuneDone({
        candidate_version_id: data.candidate_version_id,
        compare: data.compare
      }, runSeq);
    });
  }

  async function mockConversationStream(content, kind, handlers) {
    await sleep(260);
    var data = window.MockData;
    var text = kind === 'interpretation' ? data.interpretation : (kind === 'followup' ? data.followupReply : data.chatReply.content);
    if (kind === 'chat' && data.chatReply) {
      await sleep(120);
      if (handlers.onSlots) handlers.onSlots(data.chatReply.slots);
      if (handlers.onProfileReady) handlers.onProfileReady({ slots: data.chatReply.slots, confirmed: state.profile.confirmed, complete: true });
    }
    var chunks = mockChunkText(text, 5);
    for (var i = 0; i < chunks.length; i += 1) {
      await sleep(18);
      if (handlers.onToken) handlers.onToken(chunks[i]);
    }
  }

  async function mockVersionAction(action, target) {
    await sleep(320);
    if (action === 'activate') {
      var previousActive = state.activeVersionId;
      state.versions.forEach(function (version) {
        if (version.id === previousActive) version.status = 'archived';
        if (version.id === target) version.status = 'active';
      });
      state.activeVersionId = target;
      return { active_id: target };
    }
    var current = state.versions.find(function (version) { return version.id === target; });
    var parentId = current && current.parent_id;
    var parent = parentId ? state.versions.find(function (version) { return version.id === parentId; }) : null;
    if (!parent) {
      var sorted = state.versions.slice().sort(function (a, b) { return String(a.created_at || '').localeCompare(String(b.created_at || '')); });
      var idx = sorted.findIndex(function (version) { return version.id === target; });
      parent = sorted[Math.max(0, idx - 1)];
    }
    if (!parent) return { active_id: target };
    state.versions.forEach(function (version) {
      if (version.id === target) version.status = 'archived';
      if (version.id === parent.id) version.status = 'active';
    });
    state.activeVersionId = parent.id;
    return { active_id: parent.id };
  }

  /* ------------------------------------------------------------- 事件绑定 */
  function bindEvents() {
    all('.nav-item').forEach(function (button) {
      button.addEventListener('click', function () { switchView(button.getAttribute('data-view')); });
    });
    all('[data-nav]').forEach(function (link) {
      link.addEventListener('click', function (event) {
        event.preventDefault();
        switchView(link.getAttribute('data-nav') || 'chat');
      });
    });
    window.addEventListener('hashchange', function () {
      var view = (window.location.hash || '').replace(/^#/, '');
      if (VIEWS.indexOf(view) >= 0 && view !== state.activeView) switchView(view, { keepHash: true });
    });
    window.addEventListener('online', function () {
      if (!state.mock) loadHealth();
    });

    var errorRetry = $('error-retry');
    if (errorRetry) errorRetry.addEventListener('click', function () {
      var action = state.retryAction;
      hideError();
      if (action) action();
    });
    var errorClose = $('error-close');
    if (errorClose) errorClose.addEventListener('click', hideError);

    var chatForm = $('chat-form');
    if (chatForm) chatForm.addEventListener('submit', function (event) {
      event.preventDefault();
      handleChatSubmit();
    });
    var chatInput = $('chat-input');
    if (chatInput) {
      chatInput.addEventListener('input', function () { autoGrow(chatInput); });
      chatInput.addEventListener('keydown', function (event) {
        if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
          event.preventDefault();
          chatForm.requestSubmit();
        }
      });
    }
    var quick = $('quick-prompts');
    if (quick) quick.addEventListener('click', function (event) {
      var chip = event.target.closest('.quick-chip');
      if (!chip || state.chatBusy) return;
      var prompt = chip.getAttribute('data-prompt') || chip.textContent.trim();
      if (chatInput) {
        chatInput.value = prompt;
        autoGrow(chatInput);
        handleChatSubmit();
      }
    });

    var editButton = $('profile-edit');
    if (editButton) editButton.addEventListener('click', function () {
      if (state.profileEditing) closeProfileEditor();
      else openProfileEditor();
    });
    var cancel = $('profile-cancel');
    if (cancel) cancel.addEventListener('click', closeProfileEditor);
    var save = $('profile-save');
    if (save) save.addEventListener('click', handleSaveProfile);
    var confirm = $('profile-confirm');
    if (confirm) confirm.addEventListener('click', handleConfirmProfile);
    var editor = $('profile-editor');
    if (editor) editor.addEventListener('submit', function (event) { event.preventDefault(); handleSaveProfile(); });
    var start = $('start-selection');
    if (start) start.addEventListener('click', startSelection);

    var runSelection = $('run-selection');
    if (runSelection) runSelection.addEventListener('click', startSelection);
    var emptyRun = $('selection-empty-run');
    if (emptyRun) emptyRun.addEventListener('click', startSelection);
    var goChat = $('selection-go-chat');
    if (goChat) goChat.addEventListener('click', function () { switchView('chat'); });

    var interpretBody = $('interpret-body');
    if (interpretBody) interpretBody.addEventListener('click', function (event) {
      var generate = event.target.closest('[data-action="generate-interpretation"], #generate-interpretation');
      if (generate) { generateInterpretation(); return; }
      var go = event.target.closest('[data-action="go-selection"]');
      if (go) switchView('selection');
    });
    var followupForm = $('followup-form');
    if (followupForm) followupForm.addEventListener('submit', function (event) {
      event.preventDefault();
      handleFollowupSubmit();
    });
    var followupInput = $('followup-input');
    if (followupInput) {
      followupInput.addEventListener('input', function () { autoGrow(followupInput); });
      followupInput.addEventListener('keydown', function (event) {
        if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
          event.preventDefault();
          followupForm.requestSubmit();
        }
      });
    }

    var startTuneButton = $('start-tune');
    if (startTuneButton) startTuneButton.addEventListener('click', startTune);
    var activate = $('tune-activate');
    if (activate) activate.addEventListener('click', function () { performVersionAction('activate'); });
    var rollback = $('tune-rollback');
    if (rollback) rollback.addEventListener('click', function () { performVersionAction('rollback'); });
  }

  /* --------------------------------------------------------------- 启动 */
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
