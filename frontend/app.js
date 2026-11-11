/* =========================================================================
 * 智投罗盘 · AI 投资顾问（单对话页 + AI 工作台）
 * 全部操作通过聊天完成：SSE 流式回复 + 富卡片按 id upsert + 右侧工作台自动更新
 * ========================================================================= */
(function () {
  'use strict';

  /* ============================== 常量 ============================== */
  var SESSION_KEY = 'zhitou.screener.session';
  var METRIC_META = {
    cagr: { label: '年化收益', note: '历史回测：平均每年涨多少', type: 'pct', tone: 'up' },
    sharpe: { label: '夏普比率', note: '每承担一份波动获得的收益', type: 'num', tone: 'info' },
    mdd: { label: '最大回撤', note: '历史回测：从高点最大亏损幅度', type: 'pct', tone: 'down' },
    ir: { label: '信息比率', note: '相对基准的超额收益稳定性', type: 'num', tone: 'neutral' },
    win_rate: { label: '胜率', note: '持有期间取得正收益的占比', type: 'pct', tone: 'info' }
  };
  var METRIC_ORDER = ['cagr', 'sharpe', 'mdd', 'ir', 'win_rate'];

  var FACTOR_LABELS = {
    ep: '估值便宜',
    bp: '账面便宜',
    roe: '盈利能力强',
    ocf_to_profit: '现金流实在',
    revenue_growth: '营收在增长',
    momentum_60: '趋势走强',
    reversal_5: '短期反转',
    volatility_60: '低波动',
    liquidity_20: '流动性',
    size: '小市值'
  };

  var FACTOR_SHORT = {
    ep: '估值',
    bp: '账面',
    roe: '盈利',
    ocf_to_profit: '现金流',
    revenue_growth: '成长',
    momentum_60: '动量',
    reversal_5: '反转',
    volatility_60: '低波',
    liquidity_20: '流动',
    size: '市值'
  };

  var SLOT_LABELS = {
    goal: '投资目标',
    horizon: '投资期限',
    capital: '大致投入',
    liquidity_need: '多久要用钱',
    loss_reaction: '亏钱时的反应',
    drawdown_tolerance: '能接受亏多少',
    income_stability: '收入稳定程度',
    experience: '投资经验',
    preferred_industries: '偏好行业',
    excluded_industries: '不碰行业',
    trade_frequency: '调整频率',
    concerns: '最担心什么'
  };

  var STEP_KIND_LABELS = {
    analysis: '分析',
    data: '数据',
    plan: '方案',
    code: '改代码',
    check: '检查',
    backtest: '回测',
    compare: '对比',
    tune: '寻优',
    wait: '等待确认',
    version: '版本',
    selection: '选股',
    profile: '画像',
    risk: '风险',
    work: '执行'
  };

  var STARTERS = [
    { label: '如何开始投资？', message: '如何开始投资？' },
    { label: '稳健一点，能接受 10% 左右回撤', message: '我想稳健一点，能接受 10% 左右回撤，计划投资 3 年以上。' },
    { label: '我比较积极，想抓成长机会', message: '我比较积极，能接受 20% 左右的回撤，希望抓住成长股的长期机会。' },
    { label: '不想碰的行业怎么排除？', message: '我不想碰白酒和房地产行业，选股时帮我排除掉。' }
  ];

  /* ============================== 状态 ============================== */
  var state = {
    sessionId: '',
    streaming: false,
    task: null,
    tuningStatus: '',
    activeTurn: null,
    cardNodes: new Map(),
    profile: null,
    activeModel: null,
    versions: [],
    pendingProposal: null,
    pendingCandidates: [],
    pendingSignature: '',
    pendingDismissed: '',
    lastSelection: null,
    lastUserMessage: '',
    lastMessageId: null,
    thinkingBudget: 'off',
    reactMaxSteps: 6,
    stateError: null,
    following: true,
    pinnedOptimization: null,
    authRole: '',
    authUsername: '',
    resetUserId: null
  };

  var dom = {};
  var magiBlinkTimers = [];
  var magiReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  var composing = false;
  var taskPollTimer = null;
  var taskPollGeneration = 0;
  var historyRows = [];
  var activeSetting = null;

  /* ============================== 工具函数 ============================== */
  function $(selector, root) {
    return (root || document).querySelector(selector);
  }

  function esc(value) {
    return String(value === null || value === undefined ? '' : value).replace(/[&<>"']/g, function (ch) {
      if (ch === '&') return '&amp;';
      if (ch === '<') return '&lt;';
      if (ch === '>') return '&gt;';
      if (ch === '"') return '&quot;';
      return '&#39;';
    });
  }

  function icon(name, className) {
    return '<svg class="icon' + (className ? ' ' + className : '') + '" aria-hidden="true"><use href="#i-' + name + '"></use></svg>';
  }

  function num(value) {
    if (value === null || value === undefined || value === '') return null;
    var n = typeof value === 'number' ? value : parseFloat(value);
    return Number.isFinite(n) ? n : null;
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function fmtNum(value, digits) {
    var n = num(value);
    return n === null ? '—' : n.toFixed(digits === undefined ? 2 : digits);
  }

  function fmtPct(value, digits) {
    var n = num(value);
    return n === null ? '—' : (n * 100).toFixed(digits === undefined ? 1 : digits) + '%';
  }

  function metricMeta(key) {
    return METRIC_META[key] || { label: String(key), note: '', type: 'num', tone: 'cyan' };
  }

  function formatMetric(key, value) {
    var meta = metricMeta(key);
    return meta.type === 'pct' ? fmtPct(value, 1) : fmtNum(value, 2);
  }

  function truncate(text, max) {
    var chars = Array.from(String(text === null || text === undefined ? '' : text));
    return chars.length > max ? chars.slice(0, max - 1).join('') + '…' : chars.join('');
  }

  function shortId(id) {
    var text = String(id || '');
    return text.length > 10 ? text.slice(0, 8) : text;
  }

  function shortDate(value) {
    return value ? String(value).slice(0, 10) : '';
  }

  function shortDateLabel(label) {
    var text = String(label === null || label === undefined ? '' : label);
    var iso = text.match(/^(\d{4})-(\d{2})-(\d{2})/);
    if (iso) return iso[1].slice(2) + '/' + iso[2];
    return text.length >= 10 ? text.slice(5, 10) : text;
  }

  function pad2(value) {
    return value < 10 ? '0' + value : String(value);
  }

  function nowClock() {
    var date = new Date();
    return pad2(date.getHours()) + ':' + pad2(date.getMinutes()) + ':' + pad2(date.getSeconds());
  }

  function formatClock(value) {
    if (!value) return nowClock();
    var text = String(value);
    var iso = text.match(/T(\d{2}:\d{2})/);
    if (iso) return iso[1];
    var plain = text.match(/\b(\d{2}:\d{2})\b/);
    if (plain) return plain[1];
    return '';
  }

  /* ============================== DOM 引用 ============================== */
  function cacheDom() {
    dom.stream = $('#chat-stream');
    dom.scroll = $('#chat-scroll');
    dom.composer = $('#composer');
    dom.input = $('#composer-input');
    dom.thinkingBudget = $('#thinking-budget');
    dom.reactMaxSteps = $('#react-max-steps');
    dom.thinkingToggle = $('#thinking-toggle');
    dom.reactToggle = $('#react-toggle');
    dom.thinkingPanel = $('#thinking-panel');
    dom.reactPanel = $('#react-panel');
    dom.sendBtn = $('#send-btn');
    dom.healthPill = $('#health-pill');
    dom.healthDot = $('#health-dot');
    dom.healthText = $('#health-text');
    dom.sessionChip = $('#session-chip');
    dom.wbToggle = $('#wb-toggle');
    dom.wbToggleLabel = $('#wb-toggle-label');
    dom.workbench = $('#workbench');
    dom.wbAlert = $('#wb-alert');
    dom.profile = $('#wb-profile');
    dom.model = $('#wb-model');
    dom.result = $('#wb-result');
    dom.chatModelChip = $('#chat-model-chip');
    dom.chatSessionName = $('#chat-session-name');
    dom.chatRunStatus = $('#chat-run-status');
    dom.footerMeta = $('#footer-meta');
    dom.newChat = $('#new-chat');
    dom.sidebarToggle = $('#sidebar-toggle');
    dom.contextToggle = $('#context-toggle');
    dom.historyBtn = $('#history-btn');
    dom.clearChat = $('#clear-chat');
    dom.clearInput = $('#clear-input');
    dom.stopBtn = $('#stop-btn');
    dom.regenerateBtn = $('#regenerate-btn');
    dom.attachBtn = $('#attach-btn');
    dom.attachInput = $('#attach-input');
    dom.sidebar = $('#sidebar');
    dom.helpBtn = $('#help-btn');
    dom.logoutBtn = $('#logout-btn');
    dom.settingsBtn = $('#settings-btn');
    dom.infoOverlay = $('#info-overlay');
    dom.infoTitle = $('#info-title');
    dom.infoKicker = $('#info-kicker');
    dom.infoSubtitle = $('#info-subtitle');
    dom.infoBody = $('#info-body');
    dom.infoClose = $('#info-close');
    dom.historyOverlay = $('#history-overlay');
    dom.historyList = $('#history-list');
    dom.historyClose = $('#history-close');
    dom.navUsers = $('#nav-users');
    dom.usersOverlay = $('#users-overlay');
    dom.usersList = $('#users-list');
    dom.usersClose = $('#users-close');
    dom.userCreateForm = $('#user-create-form');
    dom.userCreateName = $('#user-create-name');
    dom.userCreatePassword = $('#user-create-password');
    dom.userCreateRole = $('#user-create-role');
    dom.userCreateSubmit = $('#user-create-submit');
    dom.userCreateMsg = $('#user-create-msg');
    dom.userResetForm = $('#user-reset-form');
    dom.userResetLabel = $('#user-reset-label');
    dom.userResetPassword = $('#user-reset-password');
    dom.userResetSubmit = $('#user-reset-submit');
    dom.userResetMsg = $('#user-reset-msg');
    dom.userResetCancel = $('#user-reset-cancel');
    dom.syncNow = $('#sync-now');
    dom.syncStatus = $('#sync-status');
  }

  /* ============================== 网络请求 ============================== */
  function redirectToLogin() {
    if (window.location.pathname !== '/login') {
      window.location.replace('/login?expired=1');
    }
  }

  async function api(path, options) {
    var opts = options || {};
    var response = await fetch(path, {
      method: opts.method || 'GET',
      credentials: 'same-origin',
      headers: Object.assign({ Accept: 'application/json', 'Content-Type': 'application/json' }, opts.headers || {}),
      body: opts.body === undefined ? undefined : JSON.stringify(opts.body)
    });
    var raw = await response.text();
    var data = null;
    if (raw) {
      try {
        data = JSON.parse(raw);
      } catch (err) {
        data = null;
      }
    }
    if (!response.ok) {
      if (response.status === 401) redirectToLogin();
      var detail = data && (data.detail || data.message || data.error);
      if (Array.isArray(detail)) {
        detail = detail.map(function (item) {
          return item && (item.msg || item.message) ? (item.msg || item.message) : '';
        }).filter(Boolean).join('；');
      }
      var error = new Error(detail ? String(detail) : '请求失败（HTTP ' + response.status + '）');
      error.status = response.status;
      throw error;
    }
    return data === null ? {} : data;
  }

  function delay(ms) {
    return new Promise(function (resolve) { setTimeout(resolve, ms); });
  }

  async function apiGet(path, attempts) {
    var max = Math.max(1, attempts || 2);
    var lastError = null;
    for (var attempt = 0; attempt < max; attempt += 1) {
      try {
        return await api(path);
      } catch (error) {
        lastError = error;
        var retryable = !error.status || error.status >= 500;
        if (!retryable || attempt === max - 1) break;
        await delay(350 * (attempt + 1));
      }
    }
    throw lastError || new Error('请求失败');
  }

  async function checkHealth() {
    try {
      var data = await api('/api/health');
      var ok = data.ok !== false;
      var title = [];
      if (data.source) title.push('数据源 ' + data.source);
      if (data.as_of) title.push('截至 ' + data.as_of);
      if (data.universe) title.push('股票池 ' + data.universe + ' 只');
      if (data.llm) title.push('LLM ' + data.llm);
      setHealth(ok ? 'ok' : 'warn', ok ? '服务正常' : '服务异常', title.join(' · '));
      if (ok && data.as_of) {
        dom.footerMeta.textContent = '数据截至 ' + data.as_of + (data.source ? ' · ' + data.source : '');
      }
      return data;
    } catch (error) {
      setHealth('error', '服务离线', error.message || '后端服务不可用');
      dom.footerMeta.textContent = '演示原型 · 后端未响应';
      return null;
    }
  }

  function setHealth(level, text, title) {
    dom.healthDot.className = 'health-dot is-' + level;
    dom.healthText.textContent = text;
    dom.healthPill.title = title || '后端与数据服务状态';
    var failure = $('#nerv-service-error');
    failure.hidden = level !== 'error';
    failure.textContent = level === 'error' ? 'WARNING / ' + (title || text) : '';
  }

  async function fetchMessages(sessionId) {
    try {
      var data = await apiGet('/api/sessions/' + encodeURIComponent(sessionId) + '/messages', 2);
      return { ok: true, messages: Array.isArray(data.messages) ? data.messages : [] };
    } catch (error) {
      return { ok: false, notFound: error.status === 404, error: error };
    }
  }

  async function loadState() {
    try {
      var data = await apiGet('/api/sessions/' + encodeURIComponent(state.sessionId) + '/state', 2);
      applySessionState(data);
      state.stateError = null;
      setWorkbenchAlert('');
    } catch (error) {
      state.stateError = error.message || '会话状态加载失败';
      setWorkbenchAlert('会话状态加载失败：' + state.stateError);
    }
  }

  function pendingCandidateSignature(candidates) {
    return (candidates || []).map(function (item) {
      return String(item.index) + ':' + String(item.status || 'pending');
    }).join('|');
  }

  /* 把 /state（或启用按钮返回值里的 state）应用到工作台状态。 */
  function applySessionState(data) {
    if (!data || typeof data !== 'object') return;
    if (data.task !== undefined) state.task = data.task;
    if (data.profile !== undefined) state.profile = normalizeProfile(data.profile);
    if (data.active_model !== undefined) state.activeModel = normalizeModel(data.active_model);
    if (data.versions !== undefined) state.versions = normalizeVersions(data.versions);
    if (data.pending_proposal !== undefined) state.pendingProposal = data.pending_proposal || null;
    if (data.last_selection !== undefined) state.lastSelection = normalizeSelection(data.last_selection);
    var candidates = Array.isArray(data.pending_candidates) ? data.pending_candidates : [];
    var signature = pendingCandidateSignature(candidates);
    if (signature !== state.pendingSignature) {
      state.pendingSignature = signature;
      state.pendingDismissed = '';
    }
    state.pendingCandidates = candidates;
  }

  function stopTaskPolling() {
    taskPollGeneration += 1;
    if (taskPollTimer !== null) window.clearTimeout(taskPollTimer);
    taskPollTimer = null;
  }

  function resumeTaskPolling(force) {
    stopTaskPolling();
    updateComposerState();
    if (state.streaming || ((!state.task || !state.task.running) && !force)) return;
    var sessionId = state.sessionId;
    var generation = taskPollGeneration;
    setChatRunStatus(state.task && state.task.status === 'cancelling' ? '正在停止…' : '后台执行中…', 'processing');
    async function poll() {
      if (generation !== taskPollGeneration || sessionId !== state.sessionId || state.streaming) return;
      try {
        var data = await apiGet('/api/sessions/' + encodeURIComponent(sessionId) + '/state', 1);
        var fetched = await fetchMessages(sessionId);
        if (generation !== taskPollGeneration || sessionId !== state.sessionId || state.streaming) return;
        var previousProfile = JSON.stringify(state.profile);
        var previousModel = workbenchModelSnapshot();
        var previousResult = JSON.stringify(state.lastSelection);
        applySessionState(data);
        if (fetched.ok) {
          syncHistory(fetched.messages);
          setWorkbenchAlert('');
        } else {
          setWorkbenchAlert('历史消息暂时无法同步，正在重试。');
        }
        if (JSON.stringify(state.profile) !== previousProfile) renderProfileSection();
        if (workbenchModelSnapshot() !== previousModel) {
          renderModelSection();
          renderChatModelChip();
        }
        if (JSON.stringify(state.lastSelection) !== previousResult) renderResultSection();
        updateComposerState();
        if ((!state.task || !state.task.running) && fetched.ok) {
          var status = state.task && state.task.status;
          if (status === 'error') setChatRunStatus('执行失败', 'error');
          else if (status === 'completed') setChatRunStatus('执行完成', 'completed');
          else setChatRunStatus(status === 'cancelled' ? '已停止' : '就绪');
          if (state.task && state.task.error) setWorkbenchAlert(state.task.error);
          return;
        }
        setChatRunStatus(state.task && state.task.status === 'cancelling' ? '正在停止…' : '后台执行中…', 'processing');
      } catch (error) {
        if (generation !== taskPollGeneration || sessionId !== state.sessionId) return;
        setWorkbenchAlert('任务状态暂时无法同步，正在重试：' + error.message);
      }
      taskPollTimer = window.setTimeout(poll, 1000);
    }
    taskPollTimer = window.setTimeout(poll, 0);
  }

  async function loadVersions() {
    try {
      var path = state.sessionId
        ? '/api/versions?session=' + encodeURIComponent(state.sessionId)
        : '/api/versions';
      var data = await apiGet(path, 2);
      var list = data && (data.versions || data.items) ? (data.versions || data.items) : data;
      var versions = normalizeVersions(list);
      // 会话状态接口已返回当前会话版本链，这里只作为兜底，避免把其他会话的版本显示进来
      if (!state.versions.length && versions.length) state.versions = versions;
      if (!state.activeModel) {
        var active = findActiveVersion(versions, data ? data.active_id : '');
        if (active) state.activeModel = normalizeModel(active);
      }
    } catch (error) {
      /* 版本清单不可用不影响对话主流程 */
    }
  }

  function findActiveVersion(versions, activeId) {
    var list = versions || [];
    for (var i = 0; i < list.length; i += 1) {
      if (list[i].status === 'active') return list[i];
    }
    if (activeId) {
      for (var j = 0; j < list.length; j += 1) {
        if (list[j].id === activeId) return list[j];
      }
    }
    return null;
  }

  function readSavedSession() {
    try {
      return window.localStorage.getItem(SESSION_KEY) || '';
    } catch (err) {
      return '';
    }
  }

  function saveSession(sessionId) {
    try {
      window.localStorage.setItem(SESSION_KEY, sessionId);
    } catch (err) {
      /* 本地存储不可用时仅影响刷新恢复 */
    }
  }

  var THINKING_BUDGET_KEY = 'zhitou.thinkingBudget.v2';
  var REACT_STEPS_KEY = 'zhitou.reactMaxSteps';
  var THINKING_LEVELS = [
    { value: 'off', label: '关闭', description: '快速直接作答，不限制自主执行轮数。' },
    { value: 'low', label: '低', description: '以较少的思考开销处理常规分析，兼顾速度。' },
    { value: 'high', label: '高', description: '更充分地分析复杂问题，响应可能需要更长时间。' },
    { value: 'max', label: '最大', description: '适合较复杂的建模任务，通常需要更多思考时间。' }
  ];

  function updateSettingSlider(slider, index, count) {
    var visual = slider.parentElement;
    visual.style.setProperty('--setting-progress', String(index / (count - 1)));
    visual.querySelectorAll('[data-position]').forEach(function (dot) {
      dot.classList.toggle('is-passed', Number(dot.dataset.position) <= index);
    });
  }

  function syncExecutionSettings() {
    var index = THINKING_LEVELS.findIndex(function (level) { return level.value === state.thinkingBudget; });
    if (index < 0) index = 0;
    var level = THINKING_LEVELS[index];
    var rounds = state.reactMaxSteps;
    var roundsLabel = rounds === null ? '无上限' : rounds + ' 轮';
    dom.thinkingBudget.value = String(index);
    dom.reactMaxSteps.value = String(rounds === null ? 11 : rounds);
    dom.thinkingBudget.setAttribute('aria-valuetext', level.label);
    dom.reactMaxSteps.setAttribute('aria-valuetext', roundsLabel);
    updateSettingSlider(dom.thinkingBudget, index, 4);
    updateSettingSlider(dom.reactMaxSteps, rounds === null ? 10 : rounds - 1, 11);
    $('#thinking-summary').textContent = level.label;
    $('#thinking-value').textContent = level.label;
    $('#thinking-description').textContent = level.description;
    $('#react-summary').textContent = roundsLabel;
    $('#react-value').textContent = roundsLabel;
    $('#react-description').textContent = rounds === null ? '不预设轮数上限，任务完成或手动停止时结束。' : '每条消息最多执行 ' + rounds + ' 轮，'
      + (rounds <= 3 ? '适合简短分析和单步任务。' : rounds <= 6 ? '适合连续分析与模型优化。' : '为复杂任务提供更多自主执行空间。');
    dom.thinkingToggle.setAttribute('aria-label', '模型思考强度：' + level.label);
    dom.reactToggle.setAttribute('aria-label', '自主执行上限：' + roundsLabel);
    dom.thinkingPanel.querySelectorAll('[data-thinking-level]').forEach(function (tick) {
      tick.classList.toggle('is-selected', tick.dataset.thinkingLevel === level.value);
    });
    dom.reactPanel.querySelectorAll('[data-react-level]').forEach(function (tick) {
      tick.classList.toggle('is-selected', tick.dataset.reactLevel === 'unlimited' ? rounds === null : Number(tick.dataset.reactLevel) === rounds);
    });
  }

  function closeSettingPopover(restoreFocus) {
    if (!activeSetting) return;
    var trigger = activeSetting.trigger;
    activeSetting.panel.hidden = true;
    trigger.setAttribute('aria-expanded', 'false');
    activeSetting = null;
    if (restoreFocus) trigger.focus({ preventScroll: true });
  }

  function positionSettingPopover() {
    if (!activeSetting) return;
    var anchor = activeSetting.trigger.getBoundingClientRect();
    var panel = activeSetting.panel;
    var rect = panel.getBoundingClientRect();
    var viewport = window.visualViewport;
    var leftEdge = viewport ? viewport.offsetLeft : 0;
    var topEdge = viewport ? viewport.offsetTop : 0;
    var width = viewport ? viewport.width : window.innerWidth;
    var height = viewport ? viewport.height : window.innerHeight;
    var left = Math.min(Math.max(leftEdge + 12, anchor.right - rect.width), leftEdge + width - rect.width - 12);
    var top = anchor.top - rect.height - 10;
    if (top < topEdge + 12) top = anchor.bottom + 10;
    top = Math.max(topEdge + 12, Math.min(top, topEdge + height - rect.height - 12));
    panel.style.left = left + 'px';
    panel.style.top = top + 'px';
  }

  function toggleSettingPopover(kind) {
    var trigger = kind === 'thinking' ? dom.thinkingToggle : dom.reactToggle;
    if (activeSetting && activeSetting.trigger === trigger) {
      closeSettingPopover(false);
      return;
    }
    closeSettingPopover(false);
    activeSetting = { trigger: trigger,
      panel: kind === 'thinking' ? dom.thinkingPanel : dom.reactPanel,
      slider: kind === 'thinking' ? dom.thinkingBudget : dom.reactMaxSteps };
    activeSetting.panel.hidden = false;
    trigger.setAttribute('aria-expanded', 'true');
    positionSettingPopover();
    activeSetting.slider.focus({ preventScroll: true });
  }

  function readThinkingBudget() {
    try {
      var value = window.localStorage.getItem(THINKING_BUDGET_KEY) || 'off';
      return ['off', 'low', 'high', 'max'].indexOf(value) >= 0 ? value : 'off';
    } catch (err) {
      return 'off';
    }
  }

  function saveThinkingBudget(value) {
    try {
      window.localStorage.setItem(THINKING_BUDGET_KEY, value);
    } catch (err) {
      /* 忽略本地存储异常 */
    }
  }

  /* ============================== 数据归一化 ============================== */
  function normalizeProfile(raw) {
    if (!raw || typeof raw !== 'object') return null;
    var items = [];
    if (Array.isArray(raw.items)) {
      var hasFilledFlag = raw.items.some(function (item) {
        return item && item.filled !== undefined;
      });
      raw.items.forEach(function (item) {
        if (!item || item.label === undefined || item.value === undefined || item.value === null) return;
        if (hasFilledFlag && item.filled === false) return;
        if (String(item.value).trim() === '') return;
        items.push({ label: String(item.label), value: String(item.value) });
      });
    } else if (raw.slots && typeof raw.slots === 'object') {
      Object.keys(SLOT_LABELS).forEach(function (key) {
        var value = raw.slots[key];
        if (value !== undefined && value !== null && String(value).trim() !== '') {
          items.push({ label: SLOT_LABELS[key], value: String(value) });
        }
      });
    }
    var completeness = num(raw.completeness);
    if (completeness === null && items.length) {
      completeness = items.length / Object.keys(SLOT_LABELS).length;
    }
    if (completeness !== null) completeness = clamp(completeness, 0, 1);
    var riskLevel = raw.risk_level || (raw.slots && raw.slots.risk_level) || '';
    return {
      summary: raw.summary ? String(raw.summary) : '',
      items: items,
      completeness: completeness,
      missing: Array.isArray(raw.missing) ? raw.missing.map(String) : [],
      confirmed: !!raw.confirmed,
      riskLevel: riskLevel ? String(riskLevel) : ''
    };
  }

  function buildProfileSummary(profile) {
    if (!profile || !profile.items.length) return '';
    if (profile.summary) return profile.summary;
    var count = profile.items.length;
    if (profile.confirmed) return '已确认的画像包含 ' + count + ' 项信息，AI 会按它来选股和解释结果。';
    return 'AI 已记录 ' + count + ' 项画像信息，继续补充后确认，就能开始选股。';
  }

  function normalizeModel(raw) {
    if (!raw || typeof raw !== 'object') return null;
    return {
      id: raw.id ? String(raw.id) : (raw.version_id ? String(raw.version_id) : ''),
      name: raw.name ? String(raw.name) : (raw.model_name ? String(raw.model_name) : '未命名模型'),
      description: raw.description ? String(raw.description) : '',
      metrics: raw.metrics && typeof raw.metrics === 'object' ? raw.metrics : null,
      status: raw.status ? String(raw.status) : 'active'
    };
  }

  function normalizeVersions(raw) {
    var list = Array.isArray(raw) ? raw : [];
    return list.filter(function (item) {
      return item && typeof item === 'object' && (item.id || item.name);
    }).map(function (item) {
      return {
        id: item.id ? String(item.id) : '',
        name: item.name ? String(item.name) : '未命名版本',
        status: item.status ? String(item.status) : '',
        created_at: item.created_at ? String(item.created_at) : '',
        metrics: item.metrics && typeof item.metrics === 'object' ? item.metrics : null
      };
    });
  }

  function normalizeSelection(raw) {
    if (!raw || typeof raw !== 'object') return null;
    var metrics = raw.metrics && typeof raw.metrics === 'object' ? raw.metrics : {};
    var hasMetrics = METRIC_ORDER.some(function (key) { return num(metrics[key]) !== null; });
    return {
      as_of: raw.as_of ? String(raw.as_of) : '',
      model: raw.model && typeof raw.model === 'object' ? raw.model : null,
      top: Array.isArray(raw.top) ? raw.top.filter(function (item) { return item && typeof item === 'object'; }) : [],
      metrics: metrics,
      metric_notes: raw.metric_notes && typeof raw.metric_notes === 'object' ? raw.metric_notes : {},
      portfolio: Array.isArray(raw.portfolio) ? raw.portfolio : [],
      factor_contributions: Array.isArray(raw.factor_contributions) ? raw.factor_contributions : [],
      nav: Array.isArray(raw.nav) ? raw.nav : [],
      benchmark_nav: Array.isArray(raw.benchmark_nav) ? raw.benchmark_nav : [],
      explanation: raw.explanation ? String(raw.explanation) : '',
      hasMetrics: hasMetrics
    };
  }

  /* ============================== 聊天区渲染 ============================== */
  function clearChat() {
    historyRows = [];
    state.tuningStatus = '';
    dom.stream.innerHTML = '';
    dom.stream.classList.remove('is-welcome');
    state.cardNodes.clear();
    state.activeTurn = null;
    state.following = true;
  }

  function appendWelcome() {
    var wrap = document.createElement('div');
    wrap.className = 'msg msg-ai msg-welcome';
    wrap.innerHTML = '<div class="msg-avatar">' + icon('spark') + '</div>'
      + '<div class="msg-main"><div class="msg-meta"><strong>AI 投资顾问</strong><time>刚刚</time></div>'
      + '<div class="card card-welcome"><div class="card-body">'
      + '<h2 class="welcome-title">你好，我是你的 AI 投资顾问</h2>'
      + '<p class="welcome-text">先用几句话说说你的投资想法就行——打算投多久、能接受亏多少、有没有不想碰的行业。我会边聊边把投资画像整理给你看，确认后直接帮你选股、解释结果，也能和你一起优化模型。</p>'
      + '<p class="welcome-text">每个新对话默认从均衡多因子模型开始；模型调整只影响当前会话，新对话会回到默认模型。</p>'
      + '<p class="welcome-text">我可以：建立投资者画像、筛选 A 股并回测、解读结果、优化模型（更换基础模型 / 调整因子权重 / 由 AI 重写数学打分代码），重要变更由你审批，版本可回滚。</p>'
      + '<div class="starter-list">' + STARTERS.map(function (item) {
        return '<button class="starter" type="button" data-action="quick" data-message="' + esc(item.message) + '">' + esc(item.label) + '</button>';
      }).join('') + '</div>'
      + '</div></div></div>';
    dom.stream.classList.add('is-welcome');
    dom.stream.appendChild(wrap);
  }

  function appendTextMessage(role, content, createdAt) {
    var wrap = document.createElement('div');
    var isUser = role === 'user';
    wrap.className = 'msg ' + (isUser ? 'msg-user' : 'msg-ai');
    dom.stream.classList.remove('is-welcome');
    var meta = '<div class="msg-meta"><strong>' + (isUser ? '你' : 'AI 投资顾问') + '</strong><time>' + esc(formatClock(createdAt)) + '</time></div>';
    var bubble = '<div class="bubble">' + esc(content) + '</div>';
    wrap.innerHTML = isUser
      ? '<div class="msg-main">' + meta + bubble + '</div>'
      : '<div class="msg-avatar">' + icon('spark') + '</div><div class="msg-main">' + meta + bubble + '</div>';
    dom.stream.appendChild(wrap);
    return wrap;
  }

  function syncHistoryCards(row, blocks) {
    var cards = row.node.querySelector('.history-cards');
    var visible = (blocks || []).filter(function (block) {
      return block && block.id && block.type && block.type !== 'profile';
    });
    if (!visible.length) {
      if (cards) cards.remove();
      row.cards.clear();
      return;
    }
    if (!cards) {
      cards = document.createElement('div');
      cards.className = 'cards history-cards';
      row.node.querySelector('.msg-main').appendChild(cards);
    }
    var next = new Map();
    visible.forEach(function (block, index) {
      var entry = row.cards.get(block.id);
      var snapshot = JSON.stringify(block);
      if (!entry) {
        var slot = document.createElement('div');
        slot.className = 'card-slot';
        slot.setAttribute('data-card-id', 'history-' + block.id);
        entry = { node: slot, snapshot: '' };
      }
      if (cards.children[index] !== entry.node) {
        cards.insertBefore(entry.node, cards.children[index] || null);
      }
      if (entry.snapshot !== snapshot) {
        var details = Array.from(entry.node.querySelectorAll('details'), function (item) {
          return item.open;
        });
        var code = entry.node.querySelector('.code-box');
        var codeTop = code && code.scrollTop;
        entry.node.innerHTML = renderCard(block);
        entry.node.querySelectorAll('details').forEach(function (item, i) {
          if (i < details.length) item.open = details[i];
        });
        code = entry.node.querySelector('.code-box');
        if (code && codeTop !== null) code.scrollTop = codeTop;
        mountCardCharts(entry.node, block);
        entry.snapshot = snapshot;
      }
      next.set(block.id, entry);
    });
    row.cards.forEach(function (entry, id) {
      if (!next.has(id)) entry.node.remove();
    });
    row.cards = next;
  }

  function visibleHistory(messages) {
    return messages.filter(function (message) {
      return message && (String(message.content || '').trim() || (message.blocks && message.blocks.length));
    });
  }

  function renderHistory(messages) {
    visibleHistory(messages).forEach(function (message) {
      var role = message.role === 'user' ? 'user' : 'assistant';
      var content = String(message.content || '');
      var node = appendTextMessage(role, content, message.created_at);
      node.classList.add('msg-history');
      var row = { node: node, role: role, createdAt: message.created_at, content: content, cards: new Map() };
      historyRows.push(row);
      if (role === 'assistant') syncHistoryCards(row, message.blocks);
      else state.lastUserMessage = content;
    });
  }

  function restoreTuningStatus(messages) {
    var latest = messages[messages.length - 1];
    var blocks = latest && latest.role === 'assistant' ? latest.blocks || [] : [];
    var tuning = blocks.filter(function (block) { return block && block.type === 'tuning'; }).pop();
    state.tuningStatus = tuning ? tuning.status : '';
  }

  function syncHistory(messages) {
    var visible = visibleHistory(messages);
    restoreTuningStatus(visible);
    var follow = state.following;
    var scrollTop = dom.scroll.scrollTop;
    var previousHeight = dom.scroll.scrollHeight;
    var tracked = new Set(historyRows.map(function (row) { return row.node; }));
    if (visible.length) {
      Array.from(dom.stream.children).forEach(function (node) {
        if (!tracked.has(node)) node.remove();
      });
      state.cardNodes.clear();
      state.activeTurn = null;
    }
    var matching = 0;
    while (matching < historyRows.length && matching < visible.length) {
      var row = historyRows[matching];
      var message = visible[matching];
      var role = message.role === 'user' ? 'user' : 'assistant';
      if (row.role !== role || row.createdAt !== message.created_at) break;
      var content = String(message.content || '');
      if (row.content !== content) {
        row.node.querySelector('.bubble').textContent = content;
        row.content = content;
      }
      if (role === 'assistant') syncHistoryCards(row, message.blocks);
      else state.lastUserMessage = content;
      matching += 1;
    }
    historyRows.splice(matching).forEach(function (row) { row.node.remove(); });
    renderHistory(visible.slice(matching));
    state.following = follow;
    if (follow && previousHeight !== dom.scroll.scrollHeight) scrollToBottom(true);
    else if (!follow && dom.scroll.scrollTop !== scrollTop) dom.scroll.scrollTop = scrollTop;
  }

  function beginAssistantTurn() {
    var wrap = document.createElement('div');
    wrap.className = 'msg msg-ai is-streaming';
    dom.stream.classList.remove('is-welcome');
    wrap.innerHTML = '<div class="msg-avatar">' + icon('spark') + '</div>'
      + '<div class="msg-main"><div class="msg-meta"><strong>AI 投资顾问</strong><time>' + esc(nowClock()) + '</time></div>'
      + '<details class="thinking-box" hidden><summary>AI 思考过程</summary><div class="thinking-text"></div></details>'
      + '<div class="bubble"><span class="typing"><i></i><i></i><i></i></span></div>'
      + '<div class="cards"></div></div>';
    dom.stream.appendChild(wrap);
    var turn = {
      node: wrap,
      bubble: wrap.querySelector('.bubble'),
      cards: wrap.querySelector('.cards'),
      thinkingBox: wrap.querySelector('.thinking-box'),
      thinkingText: wrap.querySelector('.thinking-text'),
      thinking: '',
      text: '',
      cardCount: 0,
      failed: false
    };
    state.activeTurn = turn;
    state.following = true;
    state.pinnedOptimization = null;
    scrollToBottom(true);
    return turn;
  }

  function ensureBubble(turn) {
    if (turn.bubble && turn.bubble.isConnected) return turn.bubble;
    var bubble = document.createElement('div');
    bubble.className = 'bubble';
    turn.cards.parentNode.insertBefore(bubble, turn.cards);
    turn.bubble = bubble;
    return bubble;
  }

  function appendToken(turn, text) {
    if (!turn || !text) return;
    var bubble = ensureBubble(turn);
    turn.text += String(text);
    if (bubble) bubble.textContent = turn.text;
    if (state.following) scrollToBottom(true);
  }

  function appendThinking(turn, text) {
    if (!turn || !text || !turn.thinkingBox || !turn.thinkingText) return;
    turn.thinkingBox.hidden = false;
    turn.thinkingBox.open = true;
    turn.thinking = (turn.thinking || '') + String(text);
    turn.thinkingText.textContent = turn.thinking;
    turn.thinkingText.scrollTop = turn.thinkingText.scrollHeight;
    var summary = turn.thinkingBox.querySelector('summary');
    if (summary) summary.textContent = 'AI 思考中…';
    if (state.following) scrollToBottom(true);
  }

  function flushTurnText(turn) {
    if (turn && turn.bubble && turn.bubble.isConnected) {
      turn.bubble.textContent = turn.text;
    }
  }

  function finalizeTurn(turn) {
    if (!turn) return;
    turn.node.classList.remove('is-streaming');
    if (turn.thinking && turn.thinkingBox) {
      var summary = turn.thinkingBox.querySelector('summary');
      if (summary) summary.textContent = 'AI 思考过程（已完成）';
    }
    if (!turn.text) {
      if (turn.bubble && turn.bubble.isConnected) turn.bubble.remove();
    }
    if (state.activeTurn === turn) state.activeTurn = null;
  }

  function errorCardHtml(title, message, action) {
    var actionHtml = '';
    if (action === 'retry-init') {
      actionHtml = '<button class="btn btn-sm" type="button" data-action="retry-init">' + icon('refresh') + '重新连接</button>';
    } else if (action === 'retry') {
      actionHtml = '<button class="btn btn-sm" type="button" data-action="retry">' + icon('refresh') + '重试</button>';
    }
    return '<article class="card card-error"><span class="error-mark">!</span><div class="error-copy"><strong>' + esc(title) + '</strong><p>' + esc(message) + '</p></div>' + actionHtml + '</article>';
  }

  function showTurnError(turn, title, message, retryEnabled) {
    if (typeof console !== 'undefined' && console.error) {
      console.error('[chat error]', title, message, 'session=' + (state.sessionId || 'none'));
    }
    if (!turn) {
      appendSystemError(title, message, retryEnabled === false ? null : 'retry');
      return;
    }
    turn.failed = true;
    var slot = document.createElement('div');
    slot.className = 'card-slot';
    slot.innerHTML = errorCardHtml(title, message, retryEnabled === false ? null : 'retry');
    turn.cards.appendChild(slot);
    turn.cardCount += 1;
    scrollToBottom(true);
  }

  function appendSystemError(title, message, action) {
    var wrap = document.createElement('div');
    wrap.className = 'msg msg-ai';
    wrap.innerHTML = '<div class="msg-avatar">' + icon('alert') + '</div>'
      + '<div class="msg-main"><div class="cards">' + errorCardHtml(title, message, action || 'retry-init') + '</div></div>';
    dom.stream.appendChild(wrap);
    scrollToBottom(true);
  }

  function scrollToBottom(force) {
    if (!dom.scroll) return;
    if (force || isNearBottom()) dom.scroll.scrollTop = dom.scroll.scrollHeight;
  }

  var cardRevealTimer = null;

  function scheduleCardReveal(node) {
    // 新卡片先让用户看清楚，不立刻把页面拽到底部；稍后平滑聚焦。
    node.__revealAfter = Date.now() + 1000;
    if (cardRevealTimer) clearTimeout(cardRevealTimer);
    cardRevealTimer = setTimeout(function () {
      cardRevealTimer = null;
      if (!node.isConnected || !state.following) return;
      node.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 1000);
  }

  function isCardHeld(node) {
    return !!node && Date.now() < (node.__revealAfter || 0);
  }

  function isNearBottom() {
    if (!dom.scroll) return true;
    return dom.scroll.scrollHeight - dom.scroll.scrollTop - dom.scroll.clientHeight < 160;
  }

  /* ============================== 卡片模板 ============================== */
  function renderCard(block) {
    switch (block.type) {
      case 'profile':
        return profileCardHtml(normalizeProfile(block));
      case 'selection':
        return selectionCardHtml(normalizeSelection(block));
      case 'work':
        return workCardHtml(block);
      case 'comparison':
        return comparisonCardHtml(block);
      case 'approval':
        return approvalCardHtml(block);
      case 'version':
        return versionCardHtml(block);
      case 'tuning':
        return tuningCardHtml(block);
      case 'review':
        return reviewCardHtml(block);
      case 'program':
        return programCardHtml(block);
      default:
        return genericCardHtml(block);
    }
  }

  function genericCardHtml(block) {
    return '<article class="card"><header class="card-head"><span class="card-icon tone-slate">' + icon('layers') + '</span>'
      + '<div class="card-title"><h3>' + esc(block.title || block.type || 'AI 内容') + '</h3></div></header>'
      + '<div class="card-body"><p class="wb-empty">收到一条暂不支持展示的内容。</p></div></article>';
  }

  /* ---------------------------------------------------------- 画像卡 */
  function profileCardHtml(profile) {
    if (!profile) return '';
    var pct = profile.completeness === null ? null : Math.round(profile.completeness * 100);
    var summary = profile.summary || buildProfileSummary(profile);
    var itemsHtml = profile.items.length
      ? '<div class="profile-items">' + profile.items.map(function (item) {
        return '<div class="profile-item"><span class="profile-item-label">' + esc(item.label) + '</span><span class="profile-item-value">' + esc(item.value) + '</span></div>';
      }).join('') + '</div>'
      : '<p class="wb-empty">AI 还在了解你的情况，继续聊聊吧。</p>';
    var missingHtml = profile.missing.length
      ? '<p class="missing-line">还差：<b>' + profile.missing.map(function (item) { return esc(item); }).join('、') + '</b></p>'
      : '';
    var progressHtml = pct === null ? '' : '<div class="completeness"><div class="completeness-head"><span>资料完整度</span><b>' + pct + '%</b></div>'
      + '<div class="progress-line"><i style="width:' + pct + '%"></i></div>' + missingHtml + '</div>';
    return '<article class="card card-profile">'
      + '<header class="card-head">'
      + '<div class="card-title"><h3>你的投资画像</h3></div>'
      + '<span class="status-pill ' + (profile.confirmed ? 'is-ok' : 'is-warn') + '">' + (profile.confirmed ? '已确认' : '待确认') + '</span></header>'
      + '<div class="card-body">'
      + (summary ? '<p class="card-summary">' + esc(summary) + '</p>' : '')
      + itemsHtml + progressHtml
      + '</div></article>';
  }

  /* ---------------------------------------------------------- 选股卡 */
  function metricCardsHtml(selection) {
    var keys = METRIC_ORDER.filter(function (key) { return num(selection.metrics[key]) !== null; });
    if (!keys.length) {
      keys = Object.keys(selection.metrics).filter(function (key) { return num(selection.metrics[key]) !== null; }).slice(0, 5);
    }
    if (!keys.length) return '';
    return '<div class="metric-grid">' + keys.map(function (key) {
      var meta = metricMeta(key);
      return '<div class="metric-card tone-' + meta.tone + '">'
        + '<span class="metric-label">' + esc(meta.label) + '</span>'
        + '<span class="metric-value">' + esc(formatMetric(key, selection.metrics[key])) + '</span>'
        + '</div>';
    }).join('') + '</div>'
      + '<p class="metric-disclaimer">以上为历史回测数据' + (selection.as_of ? '（截至 ' + esc(selection.as_of) + '）' : '') + '，不代表未来实际表现，不构成投资建议。</p>';
  }

  function stockRowsHtml(top) {
    return top.map(function (stock, index) {
      var reasons = Array.isArray(stock.reasons) ? stock.reasons.filter(function (reason) { return reason; }) : [];
      var reasonHtml = reasons.length
        ? '<div class="reason-chips">' + reasons.slice(0, 3).map(function (reason) {
          return '<span class="reason-chip">' + esc(reason) + '</span>';
        }).join('') + '</div>'
        : '—';
      return '<tr>'
        + '<td class="stock-rank">' + (index + 1) + '</td>'
        + '<td class="stock-code">' + esc(stock.code || '—') + '</td>'
        + '<td class="industry-tag">' + esc(stock.industry || '—') + '</td>'
        + '<td class="stock-score">' + (num(stock.score) === null ? '—' : fmtNum(stock.score, 2)) + '</td>'
        + '<td>' + reasonHtml + '</td>'
        + '</tr>';
    }).join('');
  }

  function factorItems(selection) {
    if (Array.isArray(selection.factor_contributions) && selection.factor_contributions.length) {
      return selection.factor_contributions.map(function (item) {
        if (!item || typeof item !== 'object') return null;
        var value = num(item.value !== undefined ? item.value : item.current);
        if (value === null) return null;
        var name = item.name ? String(item.name) : '';
        var fullLabel = item.label ? String(item.label) : '';
        return {
          key: name,
          label: FACTOR_SHORT[name] || fullLabel || name || '因子',
          title: fullLabel || FACTOR_LABELS[name] || name,
          current: value
        };
      }).filter(Boolean).slice(0, 10);
    }
    var sources = [];
    selection.top.forEach(function (stock) {
      if (stock && stock.contributions && typeof stock.contributions === 'object') sources.push(stock.contributions);
    });
    if (!sources.length) {
      selection.top.forEach(function (stock) {
        if (stock && stock.factors && typeof stock.factors === 'object') sources.push(stock.factors);
      });
    }
    if (!sources.length) return [];
    var sums = {};
    var counts = {};
    sources.forEach(function (source) {
      Object.keys(source).forEach(function (key) {
        var value = num(source[key]);
        if (value === null) return;
        sums[key] = (sums[key] || 0) + value;
        counts[key] = (counts[key] || 0) + 1;
      });
    });
    return Object.keys(sums).map(function (key) {
      return {
        key: key,
        label: FACTOR_SHORT[key] || FACTOR_LABELS[key] || key,
        title: FACTOR_LABELS[key] || key,
        current: sums[key] / counts[key]
      };
    }).sort(function (a, b) {
      return Math.abs(b.current) - Math.abs(a.current);
    }).slice(0, 10);
  }

  function portfolioHtml(selection) {
    var rows = selection.portfolio.map(function (item) {
      if (!item || !item.code) return null;
      var weight = num(item.weight_pct);
      if (weight === null) {
        var rawWeight = num(item.weight);
        weight = rawWeight === null ? null : rawWeight * 100;
      }
      if (weight === null) return null;
      return { code: String(item.code), weight: weight };
    }).filter(Boolean).slice(0, 12);
    if (!rows.length) return '';
    var max = Math.max.apply(null, rows.map(function (row) { return row.weight; }).concat([0.0001]));
    var barRows = rows.map(function (row) {
      var width = clamp((row.weight / max) * 100, 0, 100);
      return '<div class="weight-row"><span class="weight-code">' + esc(row.code) + '</span>'
        + '<span class="weight-track"><i class="weight-fill" style="width:' + width.toFixed(1) + '%"></i></span>'
        + '<span class="weight-pct">' + fmtNum(row.weight, 1) + '%</span></div>';
    }).join('');
    return '<div class="sub-block"><div class="sub-title"><h4>组合权重</h4><span class="sub-meta">共 ' + selection.portfolio.length + ' 只</span></div>'
      + '<div class="weight-list">' + barRows + '</div></div>';
  }

  function buildNavData(selection) {
    var nav = (selection.nav || []).filter(function (point) {
      return point && point.date !== undefined && num(point.value) !== null;
    });
    if (nav.length < 2) return null;
    var labels = nav.map(function (point) { return String(point.date); });
    var series = [{
      name: '策略净值',
      values: nav.map(function (point) { return num(point.value); }),
      color: '#2563EB'
    }];
    var benchmark = (selection.benchmark_nav || []).filter(function (point) {
      return point && point.date !== undefined && num(point.value) !== null;
    });
    if (benchmark.length) {
      var byDate = {};
      benchmark.forEach(function (point) { byDate[String(point.date)] = num(point.value); });
      var byDateValues = labels.map(function (label) {
        return byDate[label] !== undefined ? byDate[label] : null;
      });
      if (byDateValues.some(function (value) { return value !== null; })) {
        series.push({ name: '基准', values: byDateValues, color: '#94A3B8', dashed: true });
      } else if (benchmark.length >= nav.length) {
        series.push({
          name: '基准',
          values: benchmark.slice(0, nav.length).map(function (point) { return num(point.value); }),
          color: '#94A3B8',
          dashed: true
        });
      }
    }
    return { labels: labels, series: series };
  }

  function selectionCardHtml(selection) {
    if (!selection) return '';
    var meta = [];
    meta.push('<span class="meta-tag is-cyan">' + icon('clock') + (selection.as_of ? esc(selection.as_of) : '最新数据') + '</span>');
    if (selection.model && selection.model.name) {
      meta.push('<span class="meta-tag">模型 · ' + esc(selection.model.name) + '</span>');
    }
    var modelDesc = selection.model && selection.model.description
      ? '<p class="card-summary">' + esc(selection.model.description) + '</p>'
      : '';
    var metricsHtml = metricCardsHtml(selection);
    var shownTop = selection.top.slice(0, 12);
    var tableHtml = shownTop.length
      ? '<div class="sub-block"><div class="sub-title"><h4>Top 精选股票</h4><span class="sub-meta">'
        + (selection.top.length > shownTop.length ? '展示前 ' + shownTop.length + ' 只 / 共 ' + selection.top.length + ' 只' : '按综合得分排序')
        + '</span></div><div class="table-scroll"><table class="stock-table"><thead><tr><th>#</th><th>代码</th><th>行业</th><th>综合得分</th><th>推荐理由</th></tr></thead>'
        + '<tbody>' + stockRowsHtml(shownTop) + '</tbody></table></div></div>'
      : '';
    var factors = factorItems(selection);
    var factorsHtml = factors.length
      ? '<div class="sub-block"><div class="sub-title"><h4>因子贡献</h4><span class="sub-meta">组合平均暴露</span></div>'
        + '<div class="chart-box" data-chart="factors"></div>'
      : '';
    var weightsHtml = portfolioHtml(selection);
    var navData = buildNavData(selection);
    var navHtml = navData
      ? '<div class="sub-block"><div class="sub-title"><h4>策略净值 vs 基准</h4><span class="sub-meta">起点归一为 1.00</span></div>'
        + '<div class="chart-box" data-chart="nav"></div>'
      : '';
    var explanationHtml = selection.explanation
      ? '<div class="ai-note"><span class="ai-note-label">' + icon('spark') + 'AI 解读</span><p>' + esc(selection.explanation) + '</p></div>'
      : '';
    var splitHtml = factorsHtml || weightsHtml
      ? '<div class="selection-split">' + factorsHtml + weightsHtml + '</div>'
      : '';
    return '<article class="card card-selection">'
      + '<header class="card-head">'
      + '<div class="card-title"><h3>本次选股结果</h3></div>'
      + '<span class="status-pill is-run">已完成</span></header>'
      + '<div class="card-body">'
      + '<div class="selection-meta-line">' + meta.join('') + '</div>'
      + modelDesc + metricsHtml + tableHtml + splitHtml + navHtml + explanationHtml
      + '</div></article>';
  }

  /* ---------------------------------------------------------- AI 工作过程卡 */
  function stepHtml(step, index) {
    var status = step.status === 'done' ? 'done'
      : step.status === 'running' ? 'running'
        : step.status === 'error' ? 'error' : 'pending';
    var iconHtml = status === 'done' ? icon('check')
      : status === 'running' ? '<i class="spinner"></i>'
        : status === 'error' ? icon('alert')
          : '<i class="step-dot"></i>';
    var kind = step.kind ? (STEP_KIND_LABELS[step.kind] || String(step.kind)) : '';
    return '<li class="step is-' + status + '">'
      + '<span class="step-icon">' + iconHtml + '</span>'
      + '<div class="step-body"><div class="step-title"><span>' + esc(step.title || ('第 ' + (index + 1) + ' 步')) + '</span>'
      + (kind ? '<span class="step-kind">' + esc(kind) + '</span>' : '')
      + '<span class="nerv-step-state" data-nerv-only>'
      + {done: '已完成', running: '执行中', error: '未通过', pending: '待执行'}[status] + '</span></div>'
      + (step.detail ? '<p class="step-detail">' + esc(step.detail) + '</p>' : '')
      + '</div></li>';
  }

  function comparisonItems(comparison) {
    if (!comparison || typeof comparison !== 'object') return [];
    var current = comparison.current && typeof comparison.current === 'object' ? comparison.current : {};
    var candidate = comparison.candidate && typeof comparison.candidate === 'object' ? comparison.candidate : {};
    var keys = [];
    METRIC_ORDER.forEach(function (key) {
      if (num(current[key]) !== null || num(candidate[key]) !== null) keys.push(key);
    });
    Object.keys(current).concat(Object.keys(candidate)).forEach(function (key) {
      if (keys.indexOf(key) !== -1) return;
      if (num(current[key]) !== null || num(candidate[key]) !== null) keys.push(key);
    });
    return keys.map(function (key) {
      return {
        key: key,
        label: metricMeta(key).label,
        current: num(current[key]),
        candidate: num(candidate[key])
      };
    });
  }

  function comparisonHtml(comparison) {
    var items = comparisonItems(comparison);
    if (!items.length) return '';
    var rows = items.map(function (item) {
      var current = item.key === 'mdd' && item.current !== null ? Math.abs(item.current) : item.current;
      var candidate = item.key === 'mdd' && item.candidate !== null ? Math.abs(item.candidate) : item.candidate;
      var delta = '—';
      if (current !== null && candidate !== null) {
        var pct = metricMeta(item.key).type === 'pct';
        var change = candidate - current;
        delta = (change > 0 ? '+' : '') + fmtNum(change * (pct ? 100 : 1), pct ? 1 : 2) + (pct ? ' pp' : '');
      }
      return '<tr><th scope="row">' + esc(item.label) + '</th><td>' + esc(formatMetric(item.key, current))
        + '</td><td>' + esc(formatMetric(item.key, candidate)) + '</td><td>' + esc(delta) + '</td></tr>';
    }).join('');
    return '<div class="sub-block"><div class="sub-title"><h4>效果对比</h4><span class="sub-meta">当前 → 候选</span></div>'
      + '<div class="chart-box" data-chart="compare"></div>'
      + '<div class="table-scroll" data-nerv-only><table class="nerv-matrix" aria-label="历史回测指标对比">'
      + '<thead><tr><th scope="col">指标</th><th scope="col">基准</th><th scope="col">候选</th><th scope="col">变化</th></tr></thead>'
      + '<tbody>' + rows + '</tbody></table></div>'
      + '</div></div>';
  }

  function operationIdHtml(block) {
    return block.id ? '<span class="nerv-operation" data-nerv-only>' + esc(String(block.id).toUpperCase()) + '</span>' : '';
  }

  var CODE_TOKEN_RE = /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|\b(def|class|return|import|from|as|if|elif|else|for|while|in|not|and|or|is|None|True|False|lambda|with|try|except|finally|raise|yield|pass|break|continue|global|nonlocal|assert|del|async|await)\b|\b(\d+(?:\.\d+)?)\b|([A-Za-z_][A-Za-z0-9_]*)(?=\s*\()/g;

  function highlightCodeText(text) {
    var source = String(text || '');
    var output = '';
    var cursor = 0;
    CODE_TOKEN_RE.lastIndex = 0;
    var match;
    while ((match = CODE_TOKEN_RE.exec(source)) !== null) {
      output += esc(source.slice(cursor, match.index));
      var cls = match[1] ? 'tok-comment'
        : match[2] ? 'tok-string'
          : match[3] ? 'tok-keyword'
            : match[4] ? 'tok-number'
              : 'tok-func';
      output += '<span class="' + cls + '">' + esc(match[0]) + '</span>';
      cursor = match.index + match[0].length;
      if (match[0].length === 0) CODE_TOKEN_RE.lastIndex += 1;
    }
    return output + esc(source.slice(cursor));
  }

  function codeBoxHtml(change) {
    if (!change || !change.diff) return '';
    var lines = String(change.diff).split('\n');
    var isPython = change.kind === 'code' || /\.py\b/.test(String(change.file || ''));
    var rows = lines.map(function (line, index) {
      var cls = 'code-line';
      if (line.indexOf('+++') === 0 || line.indexOf('---') === 0) cls += ' is-file';
      else if (line.indexOf('@@') === 0) cls += ' is-hunk';
      else if (line.charAt(0) === '+') cls += ' is-add';
      else if (line.charAt(0) === '-') cls += ' is-del';
      var textHtml = isPython ? highlightCodeText(line) : esc(line);
      return '<div class="' + cls + '"><span class="code-ln">' + (index + 1) + '</span><span class="code-text">' + textHtml + '</span></div>';
    }).join('');
    var added = Number(change.added) || 0;
    var removed = Number(change.removed) || 0;
    var kind = change.kind === 'code' ? '代码改动'
      : change.kind === 'model_type' ? '模型类型 / 权重改动' : '因子权重改动';
    return '<div class="code-change">'
      + '<div class="code-head"><span class="code-head-left"><span class="code-kind">' + esc(kind) + '</span>'
      + '<span class="code-file">' + esc(change.file || 'models/strategy.py') + '</span></span>'
      + '<span class="code-stats">+' + added + ' / -' + removed + '</span></div>'
      + '<pre class="code-box" aria-label="AI 正在修改的模型内容">' + rows + '</pre>'
      + '</div>';
  }

  function proposalHtml(proposal, showCodeDiff) {
    if (!proposal || typeof proposal !== 'object') return '';
    var approachLabels = { model_type: '切换内置模型', weights: '调整因子权重', code: '修改模型代码' };
    var summary = proposal.summary ? '<p class="proposal-summary">' + esc(proposal.summary) + '</p>' : '';
    var tags = [];
    if (proposal.approach) tags.push('<span class="tag is-amber">' + esc(approachLabels[proposal.approach] || proposal.approach) + '</span>');
    if (proposal.model_name) tags.push('<span class="tag">模型 · ' + esc(proposal.model_name) + '</span>');
    var diff = (showCodeDiff !== false && proposal.diff)
      ? '<details class="diff"><summary>展开查看代码改动</summary><pre>' + esc(proposal.diff) + '</pre></details>'
      : '';
    return '<div class="proposal">' + summary
      + (tags.length ? '<div class="proposal-meta">' + tags.join('') + '</div>' : '')
      + diff  + '</div>';
  }

  function workCardHtml(block) {
    var steps = Array.isArray(block.steps) ? block.steps.filter(function (step) {
      return step && typeof step === 'object';
    }) : [];
    var status = block.status === 'done' ? 'done' : block.status === 'error' ? 'error' : 'running';
    var statusLabel = status === 'done' ? '已完成' : status === 'error' ? '失败' : '运行中';
    var statusClass = status === 'done' ? 'is-ok' : status === 'error' ? 'is-error' : 'is-run';
    var statusPill = '<span class="status-pill ' + statusClass + '">'
      + (status === 'running' ? '<i class="spinner"></i>' : '') + statusLabel + '</span>';
    var doneCount = steps.filter(function (step) { return step.status === 'done'; }).length;
    var progressPct = steps.length ? Math.round((doneCount / steps.length) * 100) : 0;
    var progressHtml = steps.length
      ? '<div class="step-progress"><div class="progress-line"><i style="width:' + progressPct + '%"></i></div><span>' + doneCount + ' / ' + steps.length + ' 步</span></div>'
      : '';
    var stepsHtml = '<ol class="steps">' + steps.map(stepHtml).join('') + '</ol>';
    return '<article class="card card-work is-' + status + '">'
      + '<header class="card-head">'
      + '<div class="card-title">' + operationIdHtml(block) + '<h3>' + esc(block.title || 'AI 正在工作') + '</h3></div>'
      + statusPill + '</header>'
      + '<div class="card-body">' + progressHtml + stepsHtml
      + codeBoxHtml(block.code_change)
      + comparisonHtml(block.comparison) + proposalHtml(block.proposal, !block.code_change)
      + '</div></article>';
  }

  function comparisonCardHtml(block) {
    var body = comparisonHtml(block.comparison || {});
    if (!body) return '';
    return '<article class="card card-work is-done">'
      + '<header class="card-head"><div class="card-title">' + operationIdHtml(block) + '<h3>' + esc(block.title || '效果对比')
      + '</h3></div><span class="status-pill is-ok">寻优结果</span></header>'
      + '<div class="card-body">' + body
      + (block.note ? '<p class="tuning-note">' + esc(block.note) + '</p>' : '')
      + '</div></article>';
  }

  /* ---------------------------------------------------------- 审批卡 */
  function reviewCardHtml(block) {
    var issues = Array.isArray(block.issues) ? block.issues : [];
    var suggestions = Array.isArray(block.suggestions) ? block.suggestions : [];
    var failed = block.status === 'failed';
    var risk = block.risk_level ? String(block.risk_level) : 'medium';
    return '<article class="card card-review"' + (risk === 'high' ? ' data-risk="high"' : '') + '>'
      + '<header class="card-head"><div class="card-title"><h3>'
      + esc(block.title || 'AI 程序二次审核') + '</h3></div>'
      + '<span class="status-pill ' + (failed ? 'is-warn' : 'is-ok') + '">'
      + (failed ? '未通过' : '已通过') + '</span></header>'
      + '<div class="card-body">'
      + (risk === 'high' ? '<p class="nerv-risk-heading" data-nerv-only>WARNING <span>程序审核高风险</span></p>' : '')
      + (block.summary ? '<p class="card-summary">' + esc(block.summary) + '</p>' : '')
      + (issues.length
        ? '<ul class="review-list">' + issues.map(function (item) {
          return '<li>' + esc(item) + '</li>';
        }).join('') + '</ul>'
        : '')
      + (suggestions.length
        ? '<p class="review-note">建议：' + esc(suggestions.slice(0, 3).join('；')) + '</p>'
        : '')
      + '<p class="review-note">风险等级：' + esc(risk) + '</p>'
      + '</div></article>';
  }

  /* ---------------------------------------------------------- AI 编写程序（实时） */
  function programCardHtml(block) {
    var contents = block.files && typeof block.files === 'object' ? block.files : {};
    var paths = Object.keys(contents);
    var active = block.active && contents[block.active] !== undefined
      ? block.active
      : (paths.length ? paths[0] : '');
    var code = active ? String(contents[active] || '') : '';
    var generated = block.status === 'generated';
    var tabs = paths.map(function (path) {
      return '<span class="program-file' + (path === active ? ' is-active' : '') + '">'
        + esc(path) + '</span>';
    }).join('');
    return '<article class="card card-program">'
      + '<header class="card-head">'
      + '<div class="card-title"><h3>AI ' + (generated ? '已编写模型程序' : '正在编写模型程序') + '</h3></div>'
      + '<span class="status-pill ' + (generated ? 'is-ok' : 'is-warn') + '">'
      + (generated ? '已生成 · 待安全审核' : '编写中') + '</span></header>'
      + '<div class="card-body">'
      + (tabs ? '<div class="program-files">' + tabs + '</div>' : '')
      + '<pre class="code-box program-code">' + esc(code) + '</pre>'
      + '<p class="wb-note">代码生成过程实时展示；生成完成后进入 LLM 安全审核与服务端校验，不会自动启用。</p>'
      + '</div></article>';
  }

  function approvalCardHtml(block) {
    var actions = Array.isArray(block.actions) ? block.actions : [];
    var candidate = Number(block.candidate) || 0;
    var statusText = '等待确认' + (candidate ? ' · 方案 ' + candidate : '');
    var buttons = actions.map(function (action, index) {
      if (!action || !action.message) return '';
      return '<button class="btn ' + (index === 0 ? 'btn-primary' : 'btn-ghost') + '" type="button" data-action="quick" data-message="' + esc(action.message) + '">'
        + esc(action.label || action.message) + '</button>';
    }).join('');
    var riskHtml = block.risk
      ? '<div class="approval-risk">'  + '<span>' + esc(block.risk) + '</span></div>'
      : '';
    return '<article class="card card-approval"'
      + (candidate ? ' data-candidate="' + esc(candidate) + '"' : '') + '>'
      + '<header class="card-head">'
      + '<div class="card-title"><h3>' + esc(block.title || '需要用户确认') + '</h3></div>'
      + '<span class="status-pill is-warn">' + esc(statusText) + '</span></header>'
      + '<div class="card-body">'
      + (block.summary ? '<p class="card-summary">' + esc(block.summary) + '</p>' : '')
      + riskHtml
      + '<div class="approval-actions">' + buttons + '</div>'
      + '</div></article>';
  }

  /* ---------------------------------------------------------- 版本卡 */
  function metricChipsHtml(metrics) {
    if (!metrics) return '';
    var chips = [];
    METRIC_ORDER.forEach(function (key) {
      var value = num(metrics[key]);
      if (value === null) return;
      chips.push('<span class="version-metric">' + metricMeta(key).label + ' ' + formatMetric(key, value) + '</span>');
    });
    if (!chips.length) {
      Object.keys(metrics).slice(0, 4).forEach(function (key) {
        var value = num(metrics[key]);
        if (value !== null) chips.push('<span class="version-metric">' + esc(key) + ' ' + fmtNum(value, 2) + '</span>');
      });
    }
    return chips.join('');
  }

  function versionStatusText(status) {
    if (status === 'active') return '启用中';
    if (status === 'candidate') return '候选';
    if (status === 'archived' || status === 'rolled_back') return '已归档';
    return status ? String(status) : '未知';
  }

  function versionStatusClass(status) {
    if (status === 'active') return 'is-ok';
    if (status === 'candidate') return 'is-warn';
    return '';
  }

  function versionCardHtml(block) {
    var active = block.active && typeof block.active === 'object' ? block.active : null;
    var versions = normalizeVersions(block.versions);
    var activeHtml = '';
    if (active) {
      var chips = metricChipsHtml(active.metrics);
      activeHtml = '<div class="version-active"><div class="version-active-main"><span>当前启用版本</span>'
        + '<strong>' + esc(active.name || '未命名版本') + '</strong>'
        + (active.id ? '<code>' + esc(active.id) + '</code>' : '') + '</div>'
        + (chips ? '<div class="version-metrics">' + chips + '</div>' : '')
        + '<span class="status-pill is-ok">启用中</span></div>';
    }
    var listHtml = versions.length
      ? '<ul class="version-list">' + versions.map(function (version) {
        var dotClass = version.status === 'active' ? ' is-active' : version.status === 'candidate' ? ' is-candidate' : '';
        return '<li class="version-row"><span class="version-dot' + dotClass + '"></span>'
          + '<span class="version-row-name">' + esc(version.name) + '</span>'
          + '<span class="status-pill ' + versionStatusClass(version.status) + '">' + esc(versionStatusText(version.status)) + '</span>'
          + (version.created_at ? '<time>' + esc(shortDate(version.created_at)) + '</time>' : '')
          + '</li>';
      }).join('') + '</ul>'
      : '';
    return '<article class="card card-version">'
      + '<header class="card-head">'
      + '<div class="card-title"><h3>模型版本已更新</h3></div>'
      + '<span class="status-pill is-ok">已生效</span></header>'
      + '<div class="card-body">' + activeHtml + listHtml + '</div></article>';
  }

  /* ---------------------------------------------------------- 网格寻优卡 */
  function tuningObjectiveLabel(block) {
    if (block && block.objective_label) return String(block.objective_label);
    var objective = block && block.objective ? String(block.objective) : '';
    if (objective === 'sharpe') return '夏普比率';
    if (objective === 'cagr') return '年化收益';
    if (objective === 'mdd') return '最大回撤';
    return '综合得分';
  }

  function formatTuningScore(block, value) {
    var score = num(value);
    if (score === null) return '—';
    var objective = block && block.objective ? String(block.objective) : '';
    if (objective === 'cagr' || objective === 'mdd') return fmtPct(score, 1);
    return fmtNum(score, 2);
  }

  function fmtParamValue(value) {
    var numberValue = num(value);
    if (numberValue === null) {
      return value === undefined || value === null || value === '' ? '—' : String(value);
    }
    if (Math.abs(numberValue - Math.round(numberValue)) < 1e-9) return String(Math.round(numberValue));
    return String(Number(numberValue.toFixed(4)));
  }

  function fmtElapsed(seconds) {
    var value = num(seconds);
    if (value === null || value < 0) return '';
    if (value < 60) return '用时 ' + fmtNum(value, 1) + ' 秒';
    var minutes = Math.floor(value / 60);
    var rest = Math.round(value % 60);
    return '用时 ' + minutes + ' 分 ' + rest + ' 秒';
  }

  function tuningProgress(block) {
    var payload = block && block.progress && typeof block.progress === 'object' ? block.progress : {};
    var trial = num(payload.trial);
    if (trial === null) trial = num(payload.evaluated);
    if (trial === null && block && block.current && typeof block.current === 'object') {
      trial = num(block.current.trial);
    }
    var total = num(payload.total);
    var percent = num(payload.percent);
    if (percent !== null && percent > 0 && percent <= 1) percent = percent * 100;
    if (percent === null) {
      if (trial !== null && total !== null && total > 0) {
        percent = (trial / total) * 100;
      } else {
        percent = block && block.status === 'done' ? 100 : 0;
      }
    }
    percent = clamp(percent, 0, 100);
    var evaluated;
    if (trial !== null || total !== null) {
      evaluated = '已评估 ' + (trial === null ? '—' : trial) + ' / ' + (total === null ? '—' : total) + ' 组';
    } else {
      evaluated = block && block.status === 'done' ? '评估完成' : '正在启动…';
    }
    return {
      trial: trial,
      total: total,
      percent: percent,
      elapsed: num(payload.elapsed),
      evaluated: evaluated
    };
  }

  function tuningHistoryData(block) {
    var history = Array.isArray(block && block.history) ? block.history : [];
    var labels = [];
    var values = [];
    history.forEach(function (point, index) {
      if (!point || typeof point !== 'object') return;
      var score = num(point.score);
      if (score === null) return;
      var trial = num(point.trial);
      labels.push(trial === null ? String(index + 1) : String(trial));
      values.push(score);
    });
    if (values.length > 60) {
      labels = labels.slice(values.length - 60);
      values = values.slice(values.length - 60);
    }
    return { labels: labels, values: values };
  }

  function tuningParamsData(block, best) {
    var space = block && Array.isArray(block.space) ? block.space : [];
    var labels = {};
    space.forEach(function (axis) {
      if (axis && typeof axis === 'object' && axis.key) {
        labels[String(axis.key)] = axis.label
          ? String(axis.label)
          : (FACTOR_SHORT[axis.key] || FACTOR_LABELS[axis.key] || String(axis.key));
      }
    });
    var weights = best && best.weights && typeof best.weights === 'object' ? best.weights : null;
    var params = best && best.params && typeof best.params === 'object' ? best.params : null;
    var source = null;
    var weightMode = false;
    if (weights && Object.keys(weights).length) {
      source = weights;
      weightMode = true;
    } else if (params && Object.keys(params).length) {
      source = params;
      var factorKeys = Object.keys(FACTOR_LABELS);
      var keys = Object.keys(params);
      var factorMatches = 0;
      keys.forEach(function (key) {
        if (factorKeys.indexOf(key) !== -1) factorMatches += 1;
      });
      var method = block && block.method ? String(block.method) : '';
      weightMode = method.indexOf('参数') === -1 && factorMatches === keys.length;
    }
    if (!source) return { type: '', entries: [] };
    var entries = Object.keys(source).map(function (key) {
      var value = num(source[key]);
      if (value === null) return null;
      return {
        key: key,
        label: weightMode ? (FACTOR_SHORT[key] || FACTOR_LABELS[key] || String(key)) : (labels[key] || String(key)),
        value: value
      };
    }).filter(function (entry) {
      return !!entry;
    });
    if (weightMode) {
      entries.sort(function (left, right) {
        return right.value - left.value;
      });
    }
    return { type: weightMode ? 'weight' : 'param', entries: entries.slice(0, 5) };
  }

  function tuningMetricsHtml(best) {
    if (!best || typeof best !== 'object') return '';
    var metrics = best.metrics && typeof best.metrics === 'object' ? best.metrics : null;
    if (!metrics) return '';
    var keys = ['cagr', 'sharpe', 'mdd'];
    var cells = keys.map(function (key) {
      var value = num(metrics[key]);
      if (value === null) return '';
      return '<div class="tuning-metric"><span>' + esc(metricMeta(key).label) + '</span>'
        + '<strong>' + esc(formatMetric(key, value)) + '</strong></div>';
    }).filter(Boolean).join('');
    if (!cells) return '';
    return '<div class="tuning-metrics">' + cells + '</div>';
  }

  function tuningParamsHtml(block, best) {
    var data = tuningParamsData(block, best);
    if (!data.entries.length) return '';
    if (data.type === 'weight') {
      var max = 0;
      data.entries.forEach(function (entry) {
        if (entry.value > max) max = entry.value;
      });
      if (!(max > 0)) max = 1;
      var rows = data.entries.map(function (entry) {
        var width = clamp((entry.value / max) * 100, 0, 100);
        return '<div class="weight-row"><span class="weight-code">' + esc(entry.label) + '</span>'
          + '<span class="weight-track"><i class="weight-fill" style="width:' + width.toFixed(1) + '%"></i></span>'
          + '<span class="weight-pct">' + fmtNum(entry.value, 3) + '</span></div>';
      }).join('');
      return '<div class="sub-block"><div class="sub-title"><h4>当前最优参数</h4>'
        + '<span class="sub-meta">Top ' + data.entries.length + ' 因子权重</span></div>'
        + '<div class="weight-list">' + rows + '</div></div>';
    }
    var paramRows = data.entries.map(function (entry) {
      return '<div class="tuning-param-row"><span>' + esc(entry.label) + '</span>'
        + '<b>' + esc(fmtParamValue(entry.value)) + '</b></div>';
    }).join('');
    return '<div class="sub-block"><div class="sub-title"><h4>当前最优参数</h4>'
      + '<span class="sub-meta">' + data.entries.length + ' 项参数</span></div>'
      + '<div class="tuning-param-list">' + paramRows + '</div></div>';
  }

  function tuningSpaceHtml(block) {
    var space = block && Array.isArray(block.space) ? block.space.filter(function (axis) {
      return axis && typeof axis === 'object';
    }) : [];
    if (!space.length) return '';
    var method = block.method ? String(block.method) : '';
    var unit = method.indexOf('参数') !== -1 ? '参数' : '因子';
    var rows = space.map(function (axis) {
      var label = axis.label
        ? String(axis.label)
        : (FACTOR_LABELS[axis.key] || FACTOR_SHORT[axis.key] || String(axis.key || '参数'));
      return '<tr><td>' + esc(label) + '</td>'
        + '<td>[' + esc(fmtParamValue(axis.min)) + ', ' + esc(fmtParamValue(axis.max)) + ']</td>'
        + '<td>' + esc(fmtParamValue(axis.step)) + '</td></tr>';
    }).join('');
    return '<details class="tuning-space"><summary>搜索范围（' + space.length + ' 个' + unit + '）</summary>'
      + '<div class="tuning-space-body"><table class="tuning-space-table">'
      + '<thead><tr><th>参数 / 因子</th><th>搜索范围</th><th>步长</th></tr></thead>'
      + '<tbody>' + rows + '</tbody></table></div></details>';
  }

  function tuningCardHtml(block) {
    var payload = block && typeof block === 'object' ? block : {};
    var status = payload.status === 'done' ? 'done' : payload.status === 'error' ? 'error' : 'running';
    var statusLabel = status === 'done' ? '已完成' : status === 'error' ? '失败' : '寻优中';
    var statusClass = status === 'done' ? 'is-ok' : status === 'error' ? 'is-error' : 'is-run';
    var statusPill = '<span class="status-pill ' + statusClass + '">'
      + (status === 'running' ? '<i class="spinner"></i>' : '') + statusLabel + '</span>';
    var progress = tuningProgress(payload);
    var elapsedText = progress.elapsed !== null && progress.elapsed > 0 ? fmtElapsed(progress.elapsed) : '';
    var percentText = Math.round(progress.percent) + '%';
    var progressHtml = '<div class="tuning-progress">'
      + '<div class="tuning-progress-head"><span>' + esc(progress.evaluated) + '</span>'
      + (elapsedText ? '<span class="tuning-elapsed">' + esc(elapsedText) + '</span>' : '')
      + '<b class="tuning-percent">' + percentText + '</b></div>'
      + '<div class="progress-line"><i style="width:' + progress.percent.toFixed(1) + '%"></i></div>'
      + '</div>';

    var objectiveLabel = tuningObjectiveLabel(payload);
    var best = payload.best && typeof payload.best === 'object' ? payload.best : null;
    var bestHasData = !!(best && (num(best.trial) !== null || num(best.score) !== null
      || (best.metrics && typeof best.metrics === 'object' && Object.keys(best.metrics).length)
      || (best.params && typeof best.params === 'object' && Object.keys(best.params).length)
      || (best.weights && typeof best.weights === 'object' && Object.keys(best.weights).length)));
    var current = payload.current && typeof payload.current === 'object' ? payload.current : null;
    var bestMeta = [];
    if (bestHasData) {
      var bestTrial = num(best.trial);
      if (bestTrial !== null) bestMeta.push('第 ' + bestTrial + ' 组');
      bestMeta.push(objectiveLabel + ' ' + formatTuningScore(payload, best.score));
    }

    var tags = [];
    tags.push('<span class="tag">方法 · ' + esc(payload.method ? String(payload.method) : '网格搜索') + '</span>');
    tags.push('<span class="tag">目标 · ' + esc(objectiveLabel) + '</span>');
    if (status === 'running' && current) {
      var currentTrial = num(current.trial);
      var currentLabel = current.label
        ? String(current.label)
        : (currentTrial !== null ? '第 ' + currentTrial + ' 组' : '');
      if (currentLabel) tags.push('<span class="tag is-live">正在评估 · ' + esc(currentLabel) + '</span>');
    }
    var tagsHtml = '<div class="tuning-tags">' + tags.join('') + '</div>';

    var metricsHtml = bestHasData ? tuningMetricsHtml(best) : '';
    var bestHtml = '';
    if (bestHasData) {
      if (metricsHtml) {
        bestHtml += '<div class="sub-block"><div class="sub-title"><h4>当前最优</h4>'
          + '<span class="sub-meta">' + esc(bestMeta.join(' · ')) + '</span></div>' + metricsHtml + '</div>';
      }
      bestHtml += tuningParamsHtml(payload, best);
    } else {
      bestHtml = '<p class="tuning-wait">'
        + (status === 'error' ? '这次寻优没有产出可用结果。' : '等待第一组评估结果…') + '</p>';
    }

    var historyData = tuningHistoryData(payload);
    var chartHtml = '<div class="sub-block"><div class="sub-title"><h4>得分走势</h4>'
      + '<span class="sub-meta">共 ' + historyData.values.length + ' 个评估点</span></div>'
      + (historyData.values.length >= 2
        ? '<div class="chart-box" data-chart="tuning-score"></div>'
        : '<div class="chart-empty">' + esc(historyData.values.length
          ? '正在累计评估结果，至少 2 组数据后显示走势。'
          : '等待第一组评估结果…') + '</div>')
      + '</div>';

    var spaceHtml = tuningSpaceHtml(payload);
    var note = payload.note
      ? String(payload.note)
      : '网格寻优基于历史回测，最优参数可能存在过拟合；候选仍需确认后才会启用。';

    return '<article class="card card-tuning is-' + status + '">'
      + '<header class="card-head">'
      + '<div class="card-title"><h3>' + esc(payload.title || '参数网格寻优') + '</h3></div>'
      + statusPill + '</header>'
      + '<div class="card-body">'
      + progressHtml + tagsHtml + bestHtml + spaceHtml + chartHtml
      + '<p class="tuning-note">' + esc(note) + '</p>'
      + '</div></article>';
  }

  /* ============================== 卡片挂载与图表 ============================== */
  function removeProgramCards() {
    state.cardNodes.forEach(function (node, key) {
      if (node && node.isConnected && node.getAttribute('data-card-type') === 'program') {
        node.remove();
        state.cardNodes.delete(key);
      }
    });
  }

  function upsertCard(block) {
    if (!block || !block.id || !block.type) return;
    if (block.type === 'tuning') setTuningStatus(block.status);
    var turn = state.activeTurn;
    if (!turn) turn = beginAssistantTurn();
    var key = String(block.id);
    var node = state.cardNodes.get(key);
    if (node && !node.isConnected) node = null;
    var isNew = !node;
    if (node && turn.node && node.closest('.msg') !== turn.node) {
      turn.cards.appendChild(node);
      turn.cardCount += 1;
    }
    if (!node) {
      node = document.createElement('div');
      node.className = 'card-slot';
      node.setAttribute('data-card-id', key);
      state.cardNodes.set(key, node);
      turn.cards.appendChild(node);
      turn.cardCount += 1;
    }
    node.setAttribute('data-card-type', block.type);
    var html = renderCard(block);
    if (!html) return;
    node.innerHTML = html;
    mountCardCharts(node, block);
    syncFromBlock(block);
    var codeBox = node.querySelector('.code-box');
    if (codeBox) codeBox.scrollTop = codeBox.scrollHeight;
    var hasCode = block.type === 'work'
      && String((block.code_change || {}).diff || '').length > 0;
    if (hasCode) {
      state.pinnedOptimization = node;
      removeProgramCards();
    }
    if (block.status === 'done' || block.status === 'error') {
      if (block.type === 'tuning' || state.pinnedOptimization === node) {
        state.pinnedOptimization = null;
      }
    }
    if (isNew) {
      if (!(state.pinnedOptimization && block.type === 'tuning')) {
        scheduleCardReveal(node);
      }
    } else if (!state.pinnedOptimization && state.following && !isCardHeld(node)) {
      scrollToBottom(true);
    }
  }

  function mountCardCharts(node, block) {
    if (typeof window.Charts === 'undefined') return;
    if (block.type === 'selection') {
      var selection = normalizeSelection(block);
      if (!selection) return;
      var navHost = node.querySelector('[data-chart="nav"]');
      var navData = buildNavData(selection);
      if (navHost && navData) {
        window.Charts.line(navHost, navData, {
          width: 640,
          height: 250,
          yFormatter: function (value) { return fmtNum(value, 2); },
          xFormatter: shortDateLabel,
          ariaLabel: '策略与基准净值曲线'
        });
      }
      var factorHost = node.querySelector('[data-chart="factors"]');
      var factors = factorItems(selection);
      if (factorHost && factors.length) {
        window.Charts.bar(factorHost, { items: factors }, {
          width: 620,
          rowHeight: 27,
          labelWidth: 68,
          valueFormatter: function (value) { return fmtNum(value, 2); },
          ariaLabel: '因子贡献条形图'
        });
      }
    }
    if (block.type === 'work' || block.type === 'comparison') {
      var compareHost = node.querySelector('[data-chart="compare"]');
      var items = comparisonItems(block.comparison);
      if (compareHost && items.length) {
        window.Charts.bar(compareHost, { items: items }, {
          width: 620,
          rowHeight: 34,
          valueFormatter: function (value, item) {
            return formatMetric(item.key, value);
          },
          ariaLabel: '候选与当前模型指标对比'
        });
      }
    }
    if (block.type === 'tuning') {
      var tuningHost = node.querySelector('[data-chart="tuning-score"]');
      if (tuningHost) {
        var tuningData = tuningHistoryData(block);
        if (tuningData.values.length >= 2) {
          window.Charts.line(tuningHost, {
            labels: tuningData.labels,
            series: [{
              name: tuningObjectiveLabel(block) + '得分',
              values: tuningData.values
            }]
          }, {
            width: 660,
            height: 190,
            yFormatter: function (value) { return fmtNum(value, 2); },
            xFormatter: function (label) { return '#' + label; },
            ariaLabel: '网格寻优得分走势',
            emptyText: '暂无可绘制的寻优数据'
          });
        } else {
          tuningHost.innerHTML = '<div class="chart-empty">正在累计评估结果，至少 2 组数据后显示走势。</div>';
        }
      }
    }
  }

  function syncFromBlock(block) {
    switch (block.type) {
      case 'profile':
        state.profile = normalizeProfile(block);
        renderProfileSection();
        break;
      case 'selection':
        state.lastSelection = normalizeSelection(block);
        renderResultSection();
        break;
      case 'approval':
        break;
      case 'version': {
        if (block.active && typeof block.active === 'object') state.activeModel = normalizeModel(block.active);
        var parsedVersions = normalizeVersions(block.versions);
        if (parsedVersions.length) state.versions = parsedVersions;
        renderModelSection();
        renderChatModelChip();
        break;
      }
      default:
        break;
    }
  }

  /* ============================== 工作台 ============================== */
  function workbenchModelSnapshot() {
    return JSON.stringify([state.activeModel, state.versions, state.pendingCandidates,
      state.pendingProposal, state.pendingDismissed,
      state.task && state.task.running ? state.task.status : null]);
  }

  function renderAllWorkbench() {
    renderProfileSection();
    renderModelSection();
    renderResultSection();
    renderChatModelChip();
  }

  function setWorkbenchAlert(message) {
    if (!dom.wbAlert) return;
    if (!message) {
      dom.wbAlert.hidden = true;
      dom.wbAlert.textContent = '';
      return;
    }
    dom.wbAlert.hidden = false;
    dom.wbAlert.textContent = message;
  }

  function renderProfileSection() {
    var profile = state.profile;
    var head = '<header class="wb-head"><h2>用户画像</h2>'
      + '<span class="status-pill ' + (profile && profile.confirmed ? 'is-ok' : 'is-warn') + '">'
      + (profile ? (profile.confirmed ? '已确认' : '完善中') : '未开始') + '</span></header>';
    if (!profile || (!profile.items.length && !profile.summary)) {
      dom.profile.innerHTML = head + '<p class="wb-empty">从左侧聊天开始，AI 会逐步了解你的投资目标、期限和风险承受力。</p>';
      return;
    }
    var pct = profile.completeness === null ? 0 : Math.round(profile.completeness * 100);
    var summary = profile.summary || buildProfileSummary(profile);
    var items = profile.items.slice(0, 6).map(function (item) {
      return '<li class="wb-item"><span>' + esc(item.label) + '</span><b>' + esc(item.value) + '</b></li>';
    }).join('');
    dom.profile.innerHTML = head
      + '<div class="wb-profile-top"><div class="wb-ring" data-chart="ring"></div>'
      + '<div class="wb-profile-copy"><span class="wb-confirmed' + (profile.confirmed ? '' : ' is-pending') + '">'
      + (profile.confirmed ? '已确认，可用于选股' : '等待用户确认') + '</span>'
      + '<p class="wb-summary">' + esc(summary) + '</p></div></div>'
      + (items ? '<ul class="wb-items">' + items + '</ul>' : '');
    var ring = dom.profile.querySelector('[data-chart="ring"]');
    if (ring && typeof window.Charts !== 'undefined') {
      window.Charts.gauge(ring, { value: pct, max: 100, label: '完整度' }, { size: 66, thickness: 5, ariaLabel: '画像完整度' });
    }
  }

  function candidateMetricsHtml(metrics) {
    var chips = ['sharpe', 'mdd'].map(function (key) {
      var value = num(metrics ? metrics[key] : null);
      return value === null ? ''
        : '<span class="version-metric">' + metricMeta(key).label + ' ' + formatMetric(key, value) + '</span>';
    }).join('');
    return chips ? '<div class="wb-candidate-metrics">' + chips + '</div>' : '';
  }

  function setTuningStatus(status) {
    if (status === state.tuningStatus) return;
    state.tuningStatus = status;
    if (document.documentElement.dataset.theme === 'nerv') renderMagi();
  }

  function magiTuningState() {
    if (!state.tuningStatus) return '';
    var taskStatus = state.task && state.task.status;
    if (taskStatus === 'cancelling') return 'cancelling';
    if (taskStatus === 'cancelled') return 'cancelled';
    if (taskStatus === 'error' || state.tuningStatus === 'error') return 'error';
    return state.tuningStatus === 'running' && (state.streaming || (state.task && state.task.running))
      ? 'running' : '';
  }

  function stopMagiBlink() {
    magiBlinkTimers.forEach(function (timer) { window.clearTimeout(timer); });
    magiBlinkTimers = [];
  }

  function syncMagiBlink(panel) {
    if (panel.dataset.run !== 'running' || magiReducedMotion.matches) {
      stopMagiBlink();
      return;
    }
    if (magiBlinkTimers.length) return;
    panel.querySelectorAll('.nerv-core').forEach(function (core, index) {
      core.toggleAttribute('data-blink-warn', Math.random() < 0.5);
      function schedule() {
        magiBlinkTimers[index] = window.setTimeout(function () {
          core.toggleAttribute('data-blink-warn');
          schedule();
        }, 90 + Math.random() * 120);
      }
      schedule();
    });
  }

  function renderMagi() {
    var panel = $('#nerv-magi');
    var run = magiTuningState();
    panel.dataset.run = run;
    var runLabel = {running: '寻优中', cancelling: '正在停止', cancelled: '已停止', error: '执行失败'}[run];
    var pending = state.pendingProposal;
    var source = pending || state.activeModel;
    var metrics = source && source.metrics || {};
    var baseline = pending && pending.base_metrics || {};
    var cores = [
      { name: 'MELCHIOR', role: '策略收益', key: 'cagr', better: 1 },
      { name: 'BALTHASAR', role: '风险控制', key: 'mdd', better: -1 },
      { name: 'CASPER', role: '稳健性验证', key: 'win_rate', better: 0 }
    ].map(function (core) {
      var current = num(metrics[core.key]);
      var previous = num(baseline[core.key]);
      var comparable = core.better && current !== null && previous !== null;
      var delta = comparable ? core.better * (core.key === 'mdd' ? Math.abs(current) - Math.abs(previous) : current - previous) : null;
      var verdict = current === null ? '无结果' : !core.better ? '历史回测'
        : delta === null ? '已测量' : Math.abs(delta) < 1e-9 ? '持平' : delta > 0 ? '改善' : '走弱';
      return {
        name: core.name, role: core.role, verdict: verdict,
        comparable: comparable,
        tone: current === null ? 'idle' : !core.better || delta < -1e-9 ? 'warn' : 'ok',
        metric: current === null ? '—' : metricMeta(core.key).label + ' '
          + formatMetric(core.key, core.key === 'mdd' ? Math.abs(current) : current)
      };
    });
    var summary = '历史回测 · 样本外未验证';
    if (cores.every(function (core) { return core.tone === 'idle'; })) summary = '暂无回测结果';
    else if (cores[0].comparable && cores[1].comparable) {
      summary = '年化收益' + cores[0].verdict + ' · 回撤表现' + cores[1].verdict + ' · 样本外未验证';
    }
    var coreOutlines = [
      '1,1 99,1 99,72 83,99 17,99 1,72',
      '1,1 78,1 99,42 99,99 1,99',
      '22,1 99,1 99,99 1,99 1,42'
    ];
    var html = '<header class="wb-head"><h2 id="nerv-magi-title">MAGI <small>分析矩阵</small></h2></header>'
      + '<div class="nerv-cores">' + cores.map(function (core, index) {
        return '<div class="nerv-core" data-tone="' + core.tone + '">'
          + '<svg class="nerv-core-shape" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true" focusable="false">'
          + '<polygon points="' + coreOutlines[index] + '" vector-effect="non-scaling-stroke"></polygon></svg>'
          + '<strong>' + core.name + '</strong><span>'
          + (runLabel || core.verdict) + '</span></div>';
      }).join('') + '<svg class="nerv-core-links" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true" focusable="false">'
      + '<polygon points="50,25 25,75 75,75" fill="none" stroke="currentColor" stroke-width="3" vector-effect="non-scaling-stroke"></polygon></svg></div>'
      + '<table class="nerv-matrix"><thead><tr><th scope="col">职责</th><th scope="col">分析结果</th></tr></thead><tbody>'
      + cores.map(function (core) {
        return '<tr><th scope="row">' + core.role + '</th><td>' + esc(core.metric) + '</td></tr>';
      }).join('') + '</tbody></table>'
      + '<p class="nerv-magi-summary">' + (source && (source.model_name || source.name)
        ? esc(source.model_name || source.name) + ' · ' : '')
      + summary + '</p>';
    if (panel.dataset.snapshot !== html) {
      stopMagiBlink();
      panel.innerHTML = html;
      panel.dataset.snapshot = html;
    }
    syncMagiBlink(panel);
  }

  function renderModelSection() {
    if (document.documentElement.dataset.theme === 'nerv') renderMagi();
    var head = '<header class="wb-head"><h2>当前模型与版本</h2></header>';
    var html = head;
    if (state.task && state.task.running) {
      html += '<p class="wb-session-note">'
        + (state.task.status === 'cancelling' ? '正在停止任务，当前步骤结束后生效。' : '模型任务正在执行，完成后候选方案会自动显示。刷新页面不会停止任务。')
        + '</p>';
    }
    var model = state.activeModel;
    if (model) {
      var chips = metricChipsHtml(model.metrics);
      html += '<div class="wb-model-top"><strong>' + esc(model.name) + '</strong><span class="status-pill is-ok">运行中</span></div>'
        + (model.description ? '<p class="wb-model-desc">' + esc(model.description) + '</p>' : '')
        + (chips ? '<div class="wb-chips">' + chips + '</div>' : '')
        + (model.id || model.name ? '<p class="wb-session-note">模型调整仅对当前会话生效；新建对话将使用默认均衡多因子模型。</p>' : '')
        + '<p class="wb-note">年化收益、夏普比率、最大回撤等均为历史回测数据，不代表未来实际表现，不构成投资建议。</p>';
    } else {
      html += '<p class="wb-empty">还没有启用的自定义模型。跟 AI 说帮我优化模型，它会先给你方案，再等你确认。</p>';
    }
    var versions = state.versions.slice(0, 8);
    if (versions.length) {
      html += '<ul class="wb-version-list">' + versions.map(function (version) {
        var active = version.status === 'active';
        var dotClass = active ? ' is-active' : version.status === 'candidate' ? ' is-candidate' : '';
        return '<li class="wb-version' + (active ? ' is-active' : '') + '">'
          + '<span class="version-dot' + dotClass + '"></span>'
          + '<span class="wb-version-name">' + esc(version.name) + '</span>'
          + '<span class="status-pill ' + versionStatusClass(version.status) + '">' + esc(versionStatusText(version.status)) + '</span>'
          + (active || !version.id ? ''
            : '<button class="btn btn-sm btn-ghost" type="button" data-action="activate-version" data-version-id="' + esc(version.id) + '">启用</button>')
          + (version.created_at ? '<time>' + esc(shortDate(version.created_at)) + '</time>' : '')
          + '</li>';
      }).join('') + '</ul>';
    }
    var pending = (state.pendingCandidates || []).filter(function (item) {
      return item && item.status !== 'applied';
    });
    if (pending.length && state.pendingDismissed !== state.pendingSignature) {
      html += '<div class="wb-pending-box">'
        + '<div class="wb-pending-head"><strong>有 ' + pending.length + ' 个待确认的方案</strong>'
        + '<button class="btn btn-sm btn-ghost" type="button" data-action="dismiss-pending">确定</button></div>'
        + '<p class="wb-pending-note">可以直接点“启用”使用某个方案，也可以在聊天里继续确认；AI 不会自动启用。</p>'
        + '<ul class="wb-candidate-list">'
        + pending.map(function (item) {
          var disabled = item.available === false;
          var summary = String(item.summary || '');
          if (summary.length > 72) summary = summary.slice(0, 72) + '…';
          return '<li class="wb-candidate">'
            + '<div class="wb-candidate-main"><b>方案 ' + esc(item.index) + '：' + esc(item.name || '候选方案') + '</b>'
            + (summary ? '<span class="wb-candidate-summary">' + esc(summary) + '</span>' : '')
            + candidateMetricsHtml(item.metrics || {})
            + '</div>'
            + '<button class="btn btn-sm ' + (disabled ? 'btn-ghost' : 'btn-primary') + '" type="button"'
            + ' data-action="apply-candidate" data-index="' + esc(item.index) + '"'
            + (disabled ? ' disabled title="旧会话记录缺少完整模型数据，请重新发起优化"' : '')
            + '>' + (disabled ? '不可启用' : '启用') + '</button>'
            + '</li>';
        }).join('')
        + '</ul></div>';
    } else if (state.pendingProposal && !pending.length) {
      // 兼容旧版 state：没有候选列表时回退到聊天确认入口
      var pendingCount = Number(state.pendingProposal.candidate_count) || 1;
      html += '<div class="wb-pending-box"><strong>有 ' + pendingCount + ' 个待确认的方案</strong>'
        + '<br>AI 会在聊天里等你确认，不会自动启用。'
        + '<br><button class="btn btn-sm" type="button" data-action="quick" data-message="展开说说待审批方案的细节">展开说说方案</button></div>';
    }
    dom.model.innerHTML = html;
  }

  function renderResultSection() {
    var head = '<header class="wb-head"><h2>最近选股结果</h2></header>';
    var selection = state.lastSelection;
    if (!selection || (!selection.top.length && !selection.hasMetrics)) {
      dom.result.innerHTML = head + '<p class="wb-empty">完成一次选股后，这里会显示最近的结果摘要和净值曲线。</p>';
      return;
    }
    var meta = '<div class="wb-result-meta"><span>' + esc(selection.as_of || '最新数据') + '</span>'
      + '<span>' + esc(selection.model && selection.model.name ? selection.model.name : 'AI 选股模型') + '</span></div>';
    var chips = '<div class="wb-chips">' + ['cagr', 'sharpe', 'mdd'].map(function (key) {
      var value = num(selection.metrics[key]);
      return value === null ? '' : '<span class="version-metric">' + metricMeta(key).label + ' ' + formatMetric(key, value) + '</span>';
    }).join('') + '</div>';
    var stocks = selection.top.slice(0, 5).map(function (stock) {
      var score = num(stock.score);
      return '<span class="wb-stock-chip">' + esc(stock.code || '—') + (score === null ? '' : '<b>' + fmtNum(score, 2) + '</b>') + '</span>';
    }).join('');
    var navData = buildNavData(selection);
    dom.result.innerHTML = head + meta + chips
      + (stocks ? '<div class="wb-stock-chips">' + stocks + '</div>' : '')
      + (navData ? '<div class="wb-mini-chart" data-chart="wb-nav"></div>' : '');
    var navHost = dom.result.querySelector('[data-chart="wb-nav"]');
    if (navHost && navData && typeof window.Charts !== 'undefined') {
      window.Charts.line(navHost, navData, {
        width: 340,
        height: 118,
        yFormatter: function (value) { return fmtNum(value, 2); },
        xFormatter: shortDateLabel,
        showEnds: false,
        ariaLabel: '最近选股净值曲线'
      });
    }
  }

  function renderChatModelChip() {
    if (!dom.chatModelChip) return;
    dom.chatModelChip.textContent = state.activeModel && state.activeModel.name
      ? '模型 · ' + state.activeModel.name
      : 'AI · 助手';
  }

  /* 工作台按钮：直接启用某个待确认候选方案 / 某个历史模型版本 */
  function markCandidateCardsApplied(index) {
    if (!index || !dom.stream) return;
    var selector = '.card-approval[data-candidate="' + String(index) + '"]';
    Array.prototype.forEach.call(dom.stream.querySelectorAll(selector), function (card) {
      card.classList.add('is-answered');
      var pill = card.querySelector('.card-head .status-pill');
      if (pill) {
        pill.className = 'status-pill is-ok';
        pill.textContent = '已启用';
      }
      Array.prototype.forEach.call(card.querySelectorAll('button'), function (item) {
        item.disabled = true;
      });
    });
  }

  async function applyCandidate(index, button) {
    if (!index || !state.sessionId) return;
    if (button) {
      button.disabled = true;
      button.textContent = '启用中…';
    }
    try {
      var data = await api(
        '/api/sessions/' + encodeURIComponent(state.sessionId) + '/candidates/'
          + encodeURIComponent(index) + '/apply',
        { method: 'POST', body: {} });
      if (data && data.state) applySessionState(data.state);
      renderAllWorkbench();
      markCandidateCardsApplied(index);
      setWorkbenchAlert(data && data.note ? data.note : '');
    } catch (error) {
      setWorkbenchAlert('启用失败：' + (error && error.message ? error.message : '请稍后重试'));
      if (button) {
        button.disabled = false;
        button.textContent = '启用';
      }
    }
  }

  async function activateVersion(versionId, button) {
    if (!versionId || !state.sessionId) return;
    if (button) {
      button.disabled = true;
      button.textContent = '启用中…';
    }
    try {
      var data = await api(
        '/api/sessions/' + encodeURIComponent(state.sessionId) + '/versions/'
          + encodeURIComponent(versionId) + '/activate',
        { method: 'POST', body: {} });
      if (data && data.state) applySessionState(data.state);
      renderAllWorkbench();
      setWorkbenchAlert(data && data.note ? data.note : '');
    } catch (error) {
      setWorkbenchAlert('启用失败：' + (error && error.message ? error.message : '请稍后重试'));
      if (button) {
        button.disabled = false;
        button.textContent = '启用';
      }
    }
  }

  function setChatRunStatus(text, status) {
    if (!dom.chatRunStatus) return;
    dom.chatRunStatus.textContent = text;
    dom.chatRunStatus.dataset.runState = status || 'normal';
  }

  function updateChatSessionLabel() {
    if (!dom.chatSessionName) return;
    var label = '';
    if (state.lastUserMessage) {
      label = String(state.lastUserMessage).replace(/\s+/g, ' ').trim();
      if (label.length > 28) label = label.slice(0, 28) + '…';
    }
    if (!label) label = state.sessionId ? '会话 ' + shortId(state.sessionId) : '新会话';
    dom.chatSessionName.textContent = label;
  }

  function updateSessionChip() {
    if (!dom.sessionChip) return;
    dom.sessionChip.textContent = state.sessionId ? '会话 ' + shortId(state.sessionId) : '会话 —';
    dom.sessionChip.title = state.sessionId ? '会话 ID：' + state.sessionId : '浏览器本地会话';
    updateChatSessionLabel();
  }

  /* ============================== SSE 发送与解析 ============================== */
  function isActionMessage(content) {
    var text = String(content || '').trim();
    if (!text) return false;
    return /^(确认|同意|启用|开始选股|开始筛选|继续选股|先不启用|先不调整|暂不选股|回滚)/.test(text)
      || /(同意启用|同意分析|确认画像|回滚到|先不启用|先不调整|暂不选股|开始选股)/.test(text);
  }

  function updateComposerState() {
    var busy = state.streaming || !!(state.task && state.task.running);
    dom.stream.dataset.busy = String(busy);
    if (document.documentElement.dataset.theme === 'nerv') renderMagi();
    dom.sendBtn.disabled = busy;
    if (dom.stopBtn) dom.stopBtn.hidden = !busy;
    if (dom.regenerateBtn) {
      var actionMessage = isActionMessage(state.lastUserMessage);
      dom.regenerateBtn.disabled = busy || !state.lastUserMessage || actionMessage;
      dom.regenerateBtn.title = actionMessage
        ? '确认 / 启用类消息不能直接重新生成，请重新描述需求'
        : '重新生成回答';
    }
    if (dom.clearInput) dom.clearInput.disabled = busy;
    dom.input.placeholder = busy
      ? 'AI 正在回复，可点击停止生成…'
      : '输入投资问题或分析需求，例如：我应该如何开始投资';
  }

  async function sendMessage(rawContent) {
    var content = String(rawContent === null || rawContent === undefined ? '' : rawContent).trim();
    if (!content) {
      dom.input.focus();
      return;
    }
    if (state.streaming || (state.task && state.task.running)) {
      dom.input.focus();
      return;
    }
    if (!state.sessionId) {
      appendSystemError('会话未就绪', '还没有建立会话，请点击重新连接。', 'retry-init');
      return;
    }
    state.task = null;
    state.tuningStatus = '';
    state.streaming = true;
    closeSettingPopover(false);
    stopTaskPolling();
    setChatRunStatus('生成中…', 'processing');
    updateComposerState();
    dom.input.value = '';
    resizeInput();
    state.following = true;
    appendTextMessage('user', content, null);
    state.lastUserMessage = content;
    updateChatSessionLabel();
    var turn = beginAssistantTurn();
    var failed = false;
    var stopped = false;
    var controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
    state.abortController = controller;
    try {
      var response = await fetch('/api/sessions/' + encodeURIComponent(state.sessionId) + '/chat', {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
        body: JSON.stringify({ content: content, thinking_budget: state.thinkingBudget || 'off', react_max_steps: state.reactMaxSteps }),
        signal: controller ? controller.signal : undefined
      });
      if (response.status === 401) {
        redirectToLogin();
        throw new Error('登录已过期，请重新登录。');
      }
      if (!response.ok) {
        var detail = await readErrorDetail(response);
        throw new Error(detail || '对话请求失败（HTTP ' + response.status + '）');
      }
      if (!response.body) throw new Error('当前浏览器不支持流式响应');
      await readEventStream(response.body, turn, function (payload) {
        failed = true;
        if (state.tuningStatus) setTuningStatus('error');
        showTurnError(turn, 'AI 执行出错', payload && payload.message ? String(payload.message) : '处理失败，请重试。', payload && payload.retryable !== false);
      });
      if (!failed && !turn.text && turn.cardCount === 0) {
        failed = true;
        showTurnError(turn, '没有收到回复', 'AI 没有返回内容，请重试。', true);
      }
    } catch (error) {
      failed = true;
      if (error && error.name === 'AbortError') {
        failed = false;
        stopped = true;
        appendTextMessage('ai', '已断开当前回复显示，任务状态将在下方同步。', null);
      } else {
        showTurnError(turn, '网络异常', error && error.message ? error.message : '请检查后端服务后重试。', true);
      }
    } finally {
      state.abortController = null;
      finalizeTurn(turn);
      state.streaming = false;
      if (stopped) setChatRunStatus('已停止');
      else if (failed) setChatRunStatus('处理出错', 'error');
      else setChatRunStatus('执行完成', 'completed');
      updateComposerState();
      await loadState();
      renderAllWorkbench();
      resumeTaskPolling(stopped || failed);
      checkHealth();
      dom.input.focus();
    }
  }

  async function readErrorDetail(response) {
    try {
      var text = await response.text();
      if (!text) return '';
      try {
        var data = JSON.parse(text);
        var detail = data.detail || data.message || data.error;
        if (detail) return String(detail);
        return text.slice(0, 300);
      } catch (err) {
        return text.slice(0, 300);
      }
    } catch (err) {
      return '';
    }
  }

  async function readEventStream(body, turn, onErrorEvent) {
    var reader = body.getReader();
    var decoder = new TextDecoder('utf-8');
    var buffer = '';
    try {
      for (;;) {
        var chunk = await reader.read();
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, { stream: true });
        var boundary = buffer.match(/\r?\n\r?\n/);
        while (boundary) {
          var raw = buffer.slice(0, boundary.index);
          buffer = buffer.slice(boundary.index + boundary[0].length);
          handleEventBlock(raw, turn, onErrorEvent);
          boundary = buffer.match(/\r?\n\r?\n/);
        }
      }
      buffer += decoder.decode();
      if (buffer.trim()) handleEventBlock(buffer, turn, onErrorEvent);
    } finally {
      try {
        reader.releaseLock();
      } catch (err) {
        /* 忽略释放异常 */
      }
    }
  }

  function handleEventBlock(raw, turn, onErrorEvent) {
    var eventName = 'message';
    var dataLines = [];
    String(raw === null || raw === undefined ? '' : raw).split(/\r?\n/).forEach(function (line) {
      if (!line || line.charAt(0) === ':') return;
      var colon = line.indexOf(':');
      var field = colon === -1 ? line : line.slice(0, colon);
      var value = colon === -1 ? '' : line.slice(colon + 1);
      if (value.charAt(0) === ' ') value = value.slice(1);
      if (field === 'event') eventName = value;
      else if (field === 'data') dataLines.push(value);
    });
    if (!dataLines.length) return;
    var payload;
    try {
      payload = JSON.parse(dataLines.join('\n'));
    } catch (err) {
      return;
    }
    if (!payload || typeof payload !== 'object') payload = { value: payload };
    switch (eventName) {
      case 'token':
        appendToken(turn, typeof payload.text === 'string' ? payload.text : '');
        break;
      case 'thinking':
        appendThinking(turn, typeof payload.text === 'string' ? payload.text : '');
        break;
      case 'ui':
        try {
          if (payload.block && typeof payload.block === 'object') upsertCard(payload.block);
        } catch (err) {
          // 单张卡片渲染失败不能中断后续 token 的流式输出
          if (window.console && console.error) console.error('card render failed', err);
        }
        break;
      case 'done':
        if (payload.message_id) state.lastMessageId = payload.message_id;
        if (state.activeTurn === turn) state.activeTurn = null;
        break;
      case 'error':
        if (typeof onErrorEvent === 'function') onErrorEvent(payload);
        break;
      default:
        break;
    }
  }

  /* ============================== 初始化 ============================== */
  async function bootstrap() {
    stopTaskPolling();
    state.task = null;
    state.thinkingBudget = readThinkingBudget();
    try {
      var storedSteps = window.localStorage.getItem(REACT_STEPS_KEY);
      var savedSteps = Number(storedSteps);
      state.reactMaxSteps = storedSteps === 'unlimited' ? null : Number.isInteger(savedSteps) && savedSteps >= 1 && savedSteps <= 10 ? savedSteps : 6;
    } catch (err) { state.reactMaxSteps = 6; }
    syncExecutionSettings();
    clearChat();
    dom.stream.innerHTML = '<div class="chat-loading"><span class="spinner"></span>正在连接 AI 助手…</div>';
    setHealth('loading', '连接中', '正在检查后端服务');
    await checkHealth();

    var sessionId = readSavedSession();
    var history = null;
    if (sessionId) {
      var fetched = await fetchMessages(sessionId);
      if (fetched.ok) {
        history = fetched.messages;
      } else if (fetched.notFound) {
        sessionId = '';
      }
    }

    if (!sessionId) {
      try {
        var created = await api('/api/sessions', { method: 'POST' });
        sessionId = created && created.session_id ? String(created.session_id) : '';
        if (!sessionId) throw new Error('后端未返回会话 ID');
        history = [];
      } catch (error) {
        clearChat();
        appendSystemError('无法建立会话', error.message || '后端服务不可用，请确认服务已启动。', 'retry-init');
        setHealth('error', '服务离线', error.message || '无法创建会话');
        return;
      }
    }

    state.sessionId = sessionId;
    saveSession(sessionId);
    updateSessionChip();
    clearChat();
    if (history && history.length) {
      renderHistory(history);
      restoreTuningStatus(visibleHistory(history));
    } else {
      appendWelcome();
    }
    updateChatSessionLabel();
    setChatRunStatus('就绪');
    updateComposerState();
    setActiveNav('assistant');
    closeInfo();
    if (history === null) {
      appendSystemError('历史消息加载失败', '聊天记录暂时没有拉到，你可以继续对话，或点一下重新连接。', 'retry-init');
    }

    await loadState();
    await loadVersions();
    if (history && history.length && (state.profile || state.lastSelection || state.activeModel)) {
    }
    renderAllWorkbench();
    state.following = true;
    scrollToBottom(true);
    resumeTaskPolling();
    dom.input.focus();
  }

  function resizeInput() {
    if (!dom.input) return;
    dom.input.style.height = 'auto';
    dom.input.style.height = Math.min(dom.input.scrollHeight, 148) + 'px';
  }

  function formatHistoryTime(value) {
    var text = String(value || '');
    var match = text.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}:\d{2})/);
    return match ? (match[2] + '-' + match[3] + ' ' + match[4]) : (text.slice(0, 16) || '');
  }

  async function fetchSessions() {
    var data = await apiGet('/api/sessions', 2);
    return Array.isArray(data.sessions) ? data.sessions : [];
  }

  function renderHistoryList(sessions) {
    if (!dom.historyList) return;
    if (!sessions.length) {
      dom.historyList.innerHTML = '<p class="history-empty">还没有历史对话。开始一段新对话后会自动归档在这里。</p>';
      return;
    }
    dom.historyList.innerHTML = sessions.map(function (item) {
      var active = item.id === state.sessionId;
      var meta = [formatHistoryTime(item.updated_at), (item.message_count || 0) + ' 条消息'];
      if (item.profile_confirmed) meta.push('画像已确认');
      var summary = item.profile_summary ? String(item.profile_summary) : '';
      return '<button type="button" class="history-item' + (active ? ' is-current' : '') + '" data-session-id="' + esc(item.id) + '">'
        + '<span class="history-item-main"><strong>' + esc(item.title || '新对话') + '</strong>'
        + '<span class="history-item-meta">' + esc(meta.join(' · ')) + '</span>'
        + (summary ? '<span class="history-item-summary">' + esc(summary) + '</span>' : '')
        + '</span><span class="history-item-state">' + (active ? '当前' : '打开') + '</span>'
        + '</button>';
    }).join('');
  }

  async function openHistory() {
    if (!dom.historyOverlay || !dom.historyList) return;
    if (dom.infoOverlay && !dom.infoOverlay.hidden) closeInfo();
    setActiveNav('history');
    dom.historyOverlay.hidden = false;
    document.body.classList.add('history-open');
    dom.historyList.innerHTML = '<p class="history-empty">正在读取历史对话…</p>';
    try {
      renderHistoryList(await fetchSessions());
    } catch (error) {
      dom.historyList.innerHTML = '<p class="history-empty">历史对话读取失败：' + esc(error.message || '请稍后重试') + '</p>';
    }
  }

  function closeHistory() {
    if (!dom.historyOverlay) return;
    dom.historyOverlay.hidden = true;
    document.body.classList.remove('history-open');
    if (dom.infoOverlay && dom.infoOverlay.hidden && (!dom.usersOverlay || dom.usersOverlay.hidden)) setActiveNav('assistant');
  }

  /* ============================== 用户管理 ============================== */
  function showUserMsg(element, text, ok) {
    if (!element) return;
    element.textContent = text || '';
    element.hidden = !text;
    element.classList.toggle('is-ok', !!ok);
  }

  function renderUsers(users) {
    if (!dom.usersList) return;
    if (!users.length) {
      dom.usersList.innerHTML = '<p class="user-empty">还没有用户账号。</p>';
      return;
    }
    dom.usersList.innerHTML = users.map(function (user) {
      var current = user.username === state.authUsername;
      var actions = current
        ? '<span class="user-row-meta">当前账号</span>'
        : '<button class="btn btn-ghost" type="button" data-user-action="toggle" data-active="' + (user.active ? '1' : '0') + '">'
          + (user.active ? '停用' : '启用') + '</button>'
          + '<button class="btn btn-ghost" type="button" data-user-action="reset">重置密码</button>'
          + '<button class="btn btn-ghost" type="button" data-user-action="delete">删除</button>';
      return '<div class="user-row" data-user-id="' + esc(String(user.id)) + '" data-username="' + esc(user.username) + '">'
        + '<div class="user-row-main">'
        + '<span class="user-row-name">' + esc(user.username) + '</span>'
        + '<span class="tag">' + (user.role === 'admin' ? '管理员' : '普通用户') + '</span>'
        + '<span class="status-pill ' + (user.active ? 'is-ok' : 'is-warn') + '">'
        + (user.active ? '已启用' : '已停用') + '</span>'
        + (current ? '<span class="tag">当前账号</span>' : '')
        + '<span class="user-row-meta">创建 ' + esc(formatHistoryTime(user.created_at))
        + ' · 最近登录 ' + esc(user.last_login_at || '从未登录') + '</span>'
        + '</div><div class="user-row-actions">' + actions + '</div></div>';
    }).join('');
  }

  async function refreshUsers() {
    var data = await api('/api/admin/users');
    renderUsers(data.users || []);
  }

  async function openUsers() {
    if (state.authRole !== 'admin' || !dom.usersOverlay) return;
    if (dom.historyOverlay && !dom.historyOverlay.hidden) closeHistory();
    if (dom.infoOverlay && !dom.infoOverlay.hidden) closeInfo();
    setActiveNav('users');
    dom.usersOverlay.hidden = false;
    document.body.classList.add('history-open');
    showUserMsg(dom.userCreateMsg, '');
    showUserMsg(dom.userResetMsg, '');
    if (dom.userResetForm) dom.userResetForm.hidden = true;
    dom.usersList.innerHTML = '<p class="user-empty">正在读取用户…</p>';
    try {
      await refreshUsers();
    } catch (error) {
      dom.usersList.innerHTML = '<p class="user-empty">用户列表读取失败：' + esc(error.message || '请稍后重试') + '</p>';
    }
    await loadSyncStatus();
  }

  function closeUsers() {
    if (!dom.usersOverlay) return;
    dom.usersOverlay.hidden = true;
    if (syncPollTimer) {
      clearTimeout(syncPollTimer);
      syncPollTimer = null;
    }
    document.body.classList.remove('history-open');
    if (dom.historyOverlay && dom.historyOverlay.hidden
        && (!dom.infoOverlay || dom.infoOverlay.hidden)) {
      setActiveNav('assistant');
    }
  }

  var syncPollTimer = null;

  function renderSyncStatus(data) {
    if (!dom.syncStatus) return;
    var parts = [];
    if (data.running || data.status === 'running') parts.push('正在同步行情数据…');
    if (data.status === 'disabled') parts.push(data.message || '自动同步已关闭');
    if (data.status === 'success') parts.push('上次同步成功');
    if (data.status === 'error') parts.push('上次同步失败：' + (data.error || '未知错误'));
    if (data.last_success) parts.push('同步时间：' + data.last_success.replace('T', ' ').slice(0, 19));
    if (data.as_of) parts.push('数据截至：' + data.as_of);
    if (data.universe) parts.push('股票数：' + data.universe);
    if (data.next_run && data.status !== 'running') parts.push('下次自动同步：' + data.next_run.replace('T', ' '));
    dom.syncStatus.textContent = parts.join(' · ') || '暂无同步记录。';
    dom.syncStatus.classList.toggle('is-error', data.status === 'error');
    dom.syncStatus.classList.toggle('is-ok', data.status === 'success');
  }

  async function loadSyncStatus() {
    if (!dom.syncStatus) return null;
    try {
      var data = await api('/api/admin/data/sync');
      renderSyncStatus(data);
      if (data.running || data.status === 'running') {
        if (syncPollTimer) clearTimeout(syncPollTimer);
        syncPollTimer = setTimeout(function () {
          syncPollTimer = null;
          loadSyncStatus();
        }, 2500);
      }
      return data;
    } catch (error) {
      dom.syncStatus.textContent = '同步状态读取失败：' + (error.message || '请稍后重试');
      dom.syncStatus.classList.add('is-error');
      return null;
    }
  }

  async function triggerSync() {
    if (!dom.syncNow) return;
    dom.syncNow.disabled = true;
    try {
      await api('/api/admin/data/sync', { method: 'POST' });
      await loadSyncStatus();
    } catch (error) {
      dom.syncStatus.textContent = '同步启动失败：' + (error.message || '请稍后重试');
      dom.syncStatus.classList.add('is-error');
    } finally {
      dom.syncNow.disabled = false;
    }
  }

  async function createUser(event) {
    event.preventDefault();
    showUserMsg(dom.userCreateMsg, '');
    var username = dom.userCreateName.value.trim();
    var password = dom.userCreatePassword.value;
    var role = dom.userCreateRole.value;
    if (!username || !password) {
      showUserMsg(dom.userCreateMsg, '请填写用户名和初始密码。');
      return;
    }
    dom.userCreateSubmit.disabled = true;
    try {
      await api('/api/admin/users', { method: 'POST', body: {
        username: username, password: password, role: role
      }});
      dom.userCreateForm.reset();
      showUserMsg(dom.userCreateMsg, '用户已创建。', true);
      await refreshUsers();
    } catch (error) {
      showUserMsg(dom.userCreateMsg, error.message || '创建失败。');
    } finally {
      dom.userCreateSubmit.disabled = false;
    }
  }

  function openResetForm(userId, username) {
    state.resetUserId = userId;
    dom.userResetLabel.textContent = '为用户「' + username + '」重置密码；保存后该用户需要重新登录。';
    dom.userResetForm.hidden = false;
    dom.userResetPassword.value = '';
    showUserMsg(dom.userResetMsg, '');
    dom.userResetPassword.focus();
  }

  async function resetUserPassword(event) {
    event.preventDefault();
    var password = dom.userResetPassword.value;
    if (!password) {
      showUserMsg(dom.userResetMsg, '请输入新密码。');
      return;
    }
    try {
      await api('/api/admin/users/' + encodeURIComponent(String(state.resetUserId)), {
        method: 'PATCH', body: { password: password }
      });
      dom.userResetForm.hidden = true;
      showUserMsg(dom.userCreateMsg, '密码已重置。', true);
      await refreshUsers();
    } catch (error) {
      showUserMsg(dom.userResetMsg, error.message || '重置失败。');
    }
  }

  async function handleUserAction(event) {
    var button = event.target.closest ? event.target.closest('[data-user-action]') : null;
    if (!button) return;
    var row = button.closest('[data-user-id]');
    if (!row) return;
    var userId = row.getAttribute('data-user-id');
    var username = row.getAttribute('data-username') || '';
    var action = button.getAttribute('data-user-action');
    if (action === 'reset') {
      openResetForm(userId, username);
      return;
    }
    if (action === 'toggle') {
      var active = button.getAttribute('data-active') === '1';
      if (!window.confirm((active ? '停用' : '启用') + '用户「' + username + '」？')) return;
      try {
        await api('/api/admin/users/' + encodeURIComponent(userId), {
          method: 'PATCH', body: { active: !active }
        });
        await refreshUsers();
      } catch (error) {
        showUserMsg(dom.userCreateMsg, error.message || '操作失败。');
      }
      return;
    }
    if (action === 'delete') {
      if (!window.confirm('删除用户「' + username + '」？该操作不可恢复。')) return;
      try {
        await api('/api/admin/users/' + encodeURIComponent(userId), { method: 'DELETE' });
        await refreshUsers();
      } catch (error) {
        showUserMsg(dom.userCreateMsg, error.message || '删除失败。');
      }
    }
  }

  async function activateSession(sessionId) {
    if (!sessionId) return false;
    if (state.streaming) {
      dom.historyList.insertAdjacentHTML('afterbegin', '<p class="history-empty">当前回复还没结束，请稍后再切换会话。</p>');
      return false;
    }
    closeHistory();
    stopTaskPolling();
    state.task = null;
    state.sessionId = sessionId;
    saveSession(sessionId);
    updateSessionChip();
    clearChat();
    state.profile = null;
    state.activeModel = null;
    state.versions = [];
    state.pendingProposal = null;
    state.lastSelection = null;
    state.lastUserMessage = '';
    state.lastMessageId = null;
    state.stateError = null;
    state.following = true;
    var fetched = await fetchMessages(sessionId);
    var history = fetched.ok ? (fetched.messages || []) : [];
    if (history.length) {
      renderHistory(history);
      restoreTuningStatus(visibleHistory(history));
    }
    else appendWelcome();
    updateChatSessionLabel();
    setChatRunStatus('就绪');
    updateComposerState();
    setActiveNav('assistant');
    closeInfo();
    await loadState();
    await loadVersions();
    renderAllWorkbench();
    scrollToBottom(true);
    resumeTaskPolling();
    dom.input.focus();
    return true;
  }

  async function startNewChat() {
    if (state.streaming) return;
    closeHistory();
    try {
      var created = await api('/api/sessions', { method: 'POST' });
      var sessionId = created && created.session_id ? String(created.session_id) : '';
      if (!sessionId) throw new Error('后端未返回会话 ID');
      await activateSession(sessionId);
      dom.input.focus();
    } catch (error) {
      appendSystemError('无法开启新对话', error.message || '请确认后端服务已启动。', 'retry-init');
    }
  }

  function toggleWorkbench() {
    var wide = window.matchMedia('(min-width: 1200px)').matches;
    if (wide) {
      document.body.classList.toggle('wb-collapsed');
    } else {
      document.body.classList.toggle('wb-open');
    }
    syncWorkbenchToggle();
  }

  function syncWorkbenchToggle() {
    var wide = window.matchMedia('(min-width: 1200px)').matches;
    var expanded = wide ? !document.body.classList.contains('wb-collapsed') : document.body.classList.contains('wb-open');
    dom.wbToggle.setAttribute('aria-expanded', expanded ? 'true' : 'false');
    dom.wbToggleLabel.textContent = expanded ? '收起右侧面板' : '展开右侧面板';
    dom.wbToggle.title = expanded ? '收起右侧面板' : '展开右侧面板';
  }

  /* ============================== 导航与详情 ============================== */
  function setActiveNav(key) {
    var items = document.querySelectorAll('.nav-item[data-nav]');
    Array.prototype.forEach.call(items, function (item) {
      item.classList.toggle('is-active', item.getAttribute('data-nav') === key);
    });
  }

  function closeInfo() {
    if (!dom.infoOverlay) return;
    dom.infoOverlay.hidden = true;
    document.body.classList.remove('info-open');
    if (!dom.historyOverlay || dom.historyOverlay.hidden) setActiveNav('assistant');
  }

  function openInfo(kind) {
    if (!dom.infoOverlay) return;
    var meta = {
      portfolio: { kicker: 'PORTFOLIO', title: '投资组合', subtitle: '最近一次选股生成的组合、权重与回测摘要。' },
      strategy: { kicker: 'STRATEGY', title: '策略分析', subtitle: '历史回测指标与因子暴露，说明当前策略风格。' },
      model: { kicker: 'MODEL', title: '模型管理', subtitle: '当前启用模型的版本、状态与历史回测指标。' },
      settings: { kicker: 'SETTINGS', title: '系统设置', subtitle: '演示版运行环境与数据说明。' },
      help: {  title: '帮助中心', subtitle: '如何用自然语言完成投资画像、选股与模型优化。' }
    }[kind] || { kicker: 'DETAILS', title: '详情', subtitle: '' };
    dom.infoKicker.textContent = meta.kicker;
    dom.infoTitle.textContent = meta.title;
    dom.infoSubtitle.textContent = meta.subtitle;
    dom.infoBody.innerHTML = infoHtml(kind);
    dom.infoOverlay.hidden = false;
    document.body.classList.add('info-open');
    if (kind === 'settings') setActiveNav('settings');
  }

  function infoNote(text) {
    return '<p class="info-note">' + esc(text) + '</p>';
  }

  function infoMetricCells(metrics) {
    var keys = METRIC_ORDER.filter(function (key) { return metrics && num(metrics[key]) !== null; });
    if (!keys.length) return '';
    return '<div class="info-grid">' + keys.map(function (key) {
      return '<div class="info-cell"><span>' + esc(metricMeta(key).label) + '</span><strong>' + esc(formatMetric(key, metrics[key])) + '</strong></div>';
    }).join('') + '</div>';
  }

  function infoHtml(kind) {
    var selection = state.lastSelection;
    if (kind === 'portfolio') {
      if (!selection || (!selection.top.length && !selection.hasMetrics)) {
        return '<p>还没有生成投资组合。回到AI 投资顾问对话，先确认投资画像，再让 AI 帮你选股。</p>';
      }
      var top = selection.top.slice(0, 8);
      var table = top.length
        ? '<table class="info-table"><thead><tr><th>#</th><th>代码</th><th>行业</th><th>综合得分</th></tr></thead><tbody>'
          + top.map(function (stock, index) {
            return '<tr><td>' + (index + 1) + '</td><td>' + esc(stock.code || '—') + '</td><td>' + esc(stock.industry || '—') + '</td><td>' + (num(stock.score) === null ? '—' : fmtNum(stock.score, 2)) + '</td></tr>';
          }).join('') + '</tbody></table>'
        : '<p>组合暂无持仓明细。</p>';
      return '<h3>组合摘要</h3>' + infoMetricCells(selection.metrics) + table
        + infoNote('组合数据来自历史回测（' + (selection.as_of || '最新数据') + '），不代表未来实际表现，不构成投资建议。');
    }
    if (kind === 'strategy') {
      if (!selection || !selection.hasMetrics) {
        return '<p>还没有策略分析结果。完成一次选股后，这里会显示回测指标与因子暴露。</p>';
      }
      var factors = factorItems(selection);
      var factorTable = factors.length
        ? '<table class="info-table"><thead><tr><th>因子</th><th>组合平均暴露</th></tr></thead><tbody>'
          + factors.map(function (item) {
            return '<tr><td>' + esc(item.title || item.label) + '</td><td>' + fmtNum(item.current, 3) + '</td></tr>';
          }).join('') + '</tbody></table>'
        : '<p>暂无因子暴露数据。</p>';
      return '<h3>回测指标</h3>' + infoMetricCells(selection.metrics)
        + '<h3>因子暴露</h3>' + factorTable
        + infoNote('以上指标均为历史回测统计，回测区间以数据服务为准；历史业绩不代表未来表现。');
    }
    if (kind === 'model') {
      var model = state.activeModel;
      var chips = model ? metricChipsHtml(model.metrics) : '';
      var versions = state.versions.slice(0, 8);
      var versionTable = versions.length
        ? '<table class="info-table"><thead><tr><th>版本</th><th>状态</th><th>更新时间</th></tr></thead><tbody>'
          + versions.map(function (version) {
            return '<tr><td>' + esc(version.name) + '</td><td>' + esc(versionStatusText(version.status)) + '</td><td>' + esc(version.created_at ? shortDate(version.created_at) : '—') + '</td></tr>';
          }).join('') + '</tbody></table>'
        : '<p>还没有历史模型版本。</p>';
      return (model ? '<h3>当前模型：' + esc(model.name) + '</h3>' + (chips ? '<div class="info-grid">' + chips + '</div>' : '') : '<h3>当前模型</h3><p>还没有启用自定义模型，可以让 AI 先给出优化方案。</p>')
        + '<h3>模型版本</h3>' + versionTable
        + infoNote('模型指标来自历史回测，不代表未来实际表现，也不构成投资建议。启用或回滚模型前请阅读风险提示并确认。');
    }
    if (kind === 'settings') {
      var health = dom.healthText ? dom.healthText.textContent : '—';
      return '<h3>运行环境</h3><div class="info-grid">'
        + '<div class="info-cell"><span>数据服务</span><strong>' + esc(health) + '</strong></div>'
        + '<div class="info-cell"><span>数据模式</span><strong>本地缓存 / 示例行情</strong></div>'
        + '<div class="info-cell"><span>账户模式</span><strong>比赛演示 · 本地会话</strong></div>'
        + '<div class="info-cell"><span>回测成本</span><strong>含交易成本与涨跌停约束</strong></div>'
        + '</div><h3>风险与说明</h3><ul>'
        + '<li>本系统为比赛演示原型，不接入真实资金账户，不执行真实交易。</li>'
        + '<li>收益、回撤、夏普比率等指标均为历史回测结果，不代表未来实际表现。</li>'
        + '<li>页面内容不构成投资建议；市场有风险，投资需谨慎。</li>'
        + '</ul>' + infoNote('如需切换真实行情或模型服务，请在服务端配置中修改，界面不会直接调用选股或模型接口。');
    }
    return '<h3>如何开始</h3><ul>'
      + '<li>在输入框用自然语言描述投资目标、期限和可接受回撤，AI 会逐步建立投资者画像。</li>'
      + '<li>画像确认后，可以让 AI 选股、解读结果或优化模型；重要操作会先请您审批。</li>'
      + '<li>右侧投资上下文会同步显示画像、当前模型和最近结果。</li>'
      + '<li>顶栏可以打开历史记录，切换回之前的会话继续分析。</li>'
  }

  /* ============================== 事件绑定 ============================== */
  function handleActionClick(event) {
    var button = event.target.closest ? event.target.closest('[data-action]') : null;
    if (!button || button.disabled) return;
    var action = button.getAttribute('data-action');
    if (action === 'dismiss-pending') {
      state.pendingDismissed = state.pendingSignature;
      renderModelSection();
      return;
    }
    if (action === 'apply-candidate') {
      applyCandidate(Number(button.getAttribute('data-index')), button);
      return;
    }
    if (action === 'activate-version') {
      activateVersion(button.getAttribute('data-version-id'), button);
      return;
    }
    if (action === 'quick') {
      var message = button.getAttribute('data-message');
      var approval = button.closest('.card-approval');
      if (approval) {
        approval.classList.add('is-answered');
        var pill = approval.querySelector('.card-head .status-pill');
        if (pill) {
          pill.className = 'status-pill is-ok is-replied';
          pill.textContent = '已回复';
        }
        Array.prototype.forEach.call(approval.querySelectorAll('button'), function (item) {
          item.disabled = true;
        });
      }
      if (message) sendMessage(message);
      return;
    }
    if (action === 'retry') {
      if (state.lastUserMessage) sendMessage(state.lastUserMessage);
      else bootstrap();
      return;
    }
    if (action === 'retry-init') {
      bootstrap();
    }
  }

  function bindEvents() {
    dom.composer.addEventListener('submit', function (event) {
      event.preventDefault();
      sendMessage(dom.input.value);
    });

    dom.composer.addEventListener('click', function (event) {
      var action = event.target.closest ? event.target.closest('[data-action]') : null;
      if (action) handleActionClick(event);
    });

    dom.input.addEventListener('keydown', function (event) {
      if (event.key !== 'Enter' || event.shiftKey || event.isComposing || composing) return;
      event.preventDefault();
      sendMessage(dom.input.value);
    });

    dom.input.addEventListener('compositionstart', function () {
      composing = true;
    });
    dom.input.addEventListener('compositionend', function () {
      composing = false;
    });
    dom.input.addEventListener('input', resizeInput);

    dom.thinkingToggle.addEventListener('click', function () { toggleSettingPopover('thinking'); });
    dom.reactToggle.addEventListener('click', function () { toggleSettingPopover('react'); });
    if (dom.reactMaxSteps) dom.reactMaxSteps.addEventListener('input', function () {
      var value = Number(dom.reactMaxSteps.value);
      state.reactMaxSteps = value === 11 ? null : Number.isInteger(value) && value >= 1 && value <= 10 ? value : 6;
      try { window.localStorage.setItem(REACT_STEPS_KEY, state.reactMaxSteps === null ? 'unlimited' : String(state.reactMaxSteps)); } catch (err) { /* optional storage */ }
      syncExecutionSettings();
    });
    if (dom.thinkingBudget) dom.thinkingBudget.addEventListener('input', function () {
      var level = THINKING_LEVELS[Number(dom.thinkingBudget.value)] || THINKING_LEVELS[0];
      state.thinkingBudget = level.value;
      saveThinkingBudget(state.thinkingBudget);
      syncExecutionSettings();
    });
    document.addEventListener('click', function (event) {
      if (activeSetting && !activeSetting.panel.contains(event.target) && !activeSetting.trigger.contains(event.target)) closeSettingPopover(false);
    });
    document.addEventListener('focusin', function (event) {
      if (activeSetting && !activeSetting.panel.contains(event.target) && !activeSetting.trigger.contains(event.target)) closeSettingPopover(false);
    });
    window.addEventListener('resize', positionSettingPopover);
    window.addEventListener('scroll', positionSettingPopover, true);
    if (window.visualViewport) {
      window.visualViewport.addEventListener('resize', positionSettingPopover);
      window.visualViewport.addEventListener('scroll', positionSettingPopover);
    }

    dom.scroll.addEventListener('scroll', function () {
      state.following = isNearBottom();
    }, { passive: true });

    dom.wbToggle.addEventListener('click', toggleWorkbench);
    dom.stream.addEventListener('click', handleActionClick);
    dom.workbench.addEventListener('click', handleActionClick);

    if (dom.sidebarToggle) dom.sidebarToggle.addEventListener('click', function () {
      document.body.classList.toggle('nav-open');
    });
    if (dom.contextToggle) dom.contextToggle.addEventListener('click', toggleWorkbench);
    if (dom.newChat) dom.newChat.addEventListener('click', startNewChat);
    if (dom.historyBtn) dom.historyBtn.addEventListener('click', openHistory);
    if (dom.clearChat) dom.clearChat.addEventListener('click', clearChatView);
    if (dom.clearInput) dom.clearInput.addEventListener('click', function () {
      dom.input.value = '';
      resizeInput();
      dom.input.focus();
    });
    if (dom.stopBtn) dom.stopBtn.addEventListener('click', async function () {
      try {
        var result = await api('/api/sessions/' + encodeURIComponent(state.sessionId) + '/task/cancel', { method: 'POST', body: {} });
        state.task = result.task || null;
        if (state.abortController) state.abortController.abort();
        else resumeTaskPolling();
        renderAllWorkbench();
      } catch (error) { setWorkbenchAlert('停止任务失败：' + error.message); }
    });
    if (dom.regenerateBtn) dom.regenerateBtn.addEventListener('click', function () {
      if (state.streaming) return;
      if (!state.lastUserMessage) {
        dom.input.focus();
        return;
      }
      if (isActionMessage(state.lastUserMessage)) {
        appendTextMessage('ai', '这是确认 / 启用类消息，不能直接重新生成，以免重复执行业务动作。请重新描述你的需求，或直接输入新的问题。', null);
        return;
      }
      sendMessage(state.lastUserMessage);
    });

    if (dom.attachBtn) dom.attachBtn.addEventListener('click', function () { dom.attachInput.click(); });
    if (dom.attachInput) {
      dom.attachInput.addEventListener('change', function () {
        var file = this.files && this.files[0] ? this.files[0] : null;
        this.value = '';
        if (!file) return;
        if (file.size > 200 * 1024) {
          appendSystemError('附件过大', '演示版只支持 200KB 以内的文本资料（txt / md / csv / json）。', null);
          return;
        }
        var reader = new FileReader();
        reader.onload = function () {
          dom.input.value = '【附件：' + file.name + '】\n' + String(reader.result || '').slice(0, 20000) + '\n\n';
          resizeInput();
          dom.input.focus();
        };
        reader.onerror = function () {
          appendSystemError('附件读取失败', '无法读取这个文件，请换一个文本文件再试。', null);
        };
        reader.readAsText(file);
      });
    }

    if (dom.helpBtn) dom.helpBtn.addEventListener('click', function () { openInfo('help'); });
    if (dom.logoutBtn) dom.logoutBtn.addEventListener('click', async function () {
      dom.logoutBtn.disabled = true;
      try {
        await fetch('/api/auth/logout', {
          method: 'POST',
          credentials: 'same-origin',
          headers: { Accept: 'application/json' }
        });
      } catch (error) {
        // 即使网络失败也离开页面，避免继续使用旧会话
      } finally {
        window.location.replace('/login');
      }
    });
    if (dom.settingsBtn) dom.settingsBtn.addEventListener('click', function () { openInfo('settings'); });
    if (dom.infoClose) dom.infoClose.addEventListener('click', closeInfo);
    if (dom.infoOverlay) {
      dom.infoOverlay.addEventListener('click', function (event) {
        if (event.target === dom.infoOverlay) closeInfo();
      });
    }

    if (dom.sidebar) {
      dom.sidebar.addEventListener('click', function (event) {
        var item = event.target.closest ? event.target.closest('[data-nav]') : null;
        if (!item) return;
        var key = item.getAttribute('data-nav');
        document.body.classList.remove('nav-open');
        setActiveNav(key);
        if (key === 'assistant') {
          closeHistory();
          closeInfo();
          dom.input.focus();
          return;
        }
        if (key === 'history') {
          openHistory();
          return;
        }
        if (key === 'users') {
          openUsers();
          return;
        }
        closeHistory();
        openInfo(key);
      });
    }

    if (dom.historyClose) dom.historyClose.addEventListener('click', closeHistory);
    if (dom.historyOverlay) {
      dom.historyOverlay.addEventListener('click', function (event) {
        if (event.target === dom.historyOverlay) closeHistory();
      });
    }
    if (dom.usersClose) dom.usersClose.addEventListener('click', closeUsers);
    if (dom.usersOverlay) {
      dom.usersOverlay.addEventListener('click', function (event) {
        if (event.target === dom.usersOverlay) closeUsers();
      });
    }
    if (dom.userCreateForm) dom.userCreateForm.addEventListener('submit', createUser);
    if (dom.usersList) dom.usersList.addEventListener('click', handleUserAction);
    if (dom.userResetForm) dom.userResetForm.addEventListener('submit', resetUserPassword);
    if (dom.userResetCancel) dom.userResetCancel.addEventListener('click', function () {
      dom.userResetForm.hidden = true;
      state.resetUserId = null;
    });
    if (dom.syncNow) dom.syncNow.addEventListener('click', triggerSync);
    if (dom.historyList) {
      dom.historyList.addEventListener('click', function (event) {
        var item = event.target.closest ? event.target.closest('[data-session-id]') : null;
        if (!item || item.disabled) return;
        var sessionId = item.getAttribute('data-session-id');
        if (sessionId && sessionId !== state.sessionId) activateSession(sessionId);
        else closeHistory();
      });
    }
    document.addEventListener('keydown', function (event) {
      if (event.key !== 'Escape') return;
      if (activeSetting) {
        event.preventDefault();
        closeSettingPopover(true);
        return;
      }
      if (dom.infoOverlay && !dom.infoOverlay.hidden) closeInfo();
      else closeHistory();
    });

    var media = window.matchMedia('(min-width: 1200px)');
    var onViewportChange = function () {
      syncWorkbenchToggle();
    };
    if (media.addEventListener) media.addEventListener('change', onViewportChange);
    else if (media.addListener) media.addListener(onViewportChange);
  }

  function clearChatView() {
    if (state.streaming) return;
    clearChat();
    appendWelcome();
    setChatRunStatus('新会话');
    updateChatSessionLabel();
    dom.input.focus();
  }

  async function loadIdentity() {
    try {
      var data = await api('/api/auth/status');
      state.authUsername = data.username || '';
      state.authRole = data.role || '';
      var name = document.querySelector('.sidebar-user strong');
      var note = document.querySelector('.sidebar-user small');
      if (name && data.username) name.textContent = data.username;
      if (note) note.textContent = (data.role === 'admin' ? '管理员' : '普通用户') + ' · 本地数据';
      if (dom.navUsers) dom.navUsers.hidden = data.role !== 'admin';
    } catch (error) {
      // 登录态失效会由 api() 统一跳转登录页
    }
  }

  /* ============================== 启动 ============================== */
  function bindNervShortcut() {
    var sequence = 'NERV';
    var matched = 0;
    var root = document.documentElement;
    try {
      if (window.sessionStorage.getItem('nerv-theme') === 'enabled') root.dataset.theme = 'nerv';
    } catch (error) { /* optional storage */ }
    document.addEventListener('keydown', function (event) {
      if (event.repeat || event.key === 'Shift') return;
      if (!event.shiftKey || event.altKey || event.ctrlKey || event.metaKey || event.isComposing
          || event.target.closest?.('input, textarea, select, [contenteditable]')) {
        matched = 0;
        return;
      }
      var key = event.key.toUpperCase();
      matched = key === sequence[matched] ? matched + 1 : Number(key === sequence[0]);
      if (matched !== sequence.length) return;
      matched = 0;
      if (root.dataset.theme === 'nerv') {
        delete root.dataset.theme;
        stopMagiBlink();
      } else {
        root.dataset.theme = 'nerv';
        renderMagi();
      }
      try {
        if (root.dataset.theme === 'nerv') window.sessionStorage.setItem('nerv-theme', 'enabled');
        else window.sessionStorage.removeItem('nerv-theme');
      } catch (error) { /* optional storage */ }
    });
    document.addEventListener('keyup', function (event) {
      if (!event.shiftKey) matched = 0;
    });
    window.addEventListener('blur', function () { matched = 0; });
  }

  function boot() {
    cacheDom();
    bindEvents();
    bindNervShortcut();
    magiReducedMotion.addEventListener('change', function () {
      if (document.documentElement.dataset.theme === 'nerv') syncMagiBlink($('#nerv-magi'));
    });
    syncWorkbenchToggle();
    renderAllWorkbench();
    loadIdentity();
    bootstrap();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
