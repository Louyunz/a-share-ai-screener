/* =========================================================================
 * 智投罗盘 · AI 投资顾问（单对话页 + AI 工作台）
 * 全部操作通过聊天完成：SSE 流式回复 + 富卡片按 id upsert + 右侧工作台自动更新
 * ========================================================================= */
(function () {
  'use strict';

  /* ============================== 常量 ============================== */
  var SESSION_KEY = 'zhitou.screener.session';
  var MAX_ACTIVITY = 12;

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
    wait: '等待确认',
    version: '版本',
    selection: '选股',
    profile: '画像',
    risk: '风险',
    work: '执行'
  };

  var ACTIVITY_ICONS = {
    analysis: 'pulse',
    data: 'chart',
    plan: 'spark',
    code: 'code',
    check: 'check',
    backtest: 'chart',
    compare: 'chart',
    wait: 'shield',
    version: 'layers',
    selection: 'target',
    profile: 'user',
    alert: 'alert',
    ai: 'spark',
    work: 'spark'
  };

  var STARTERS = [
    { label: '稳健一点，能接受 10% 左右回撤', message: '我想稳健一点，能接受 10% 左右回撤，计划投资 3 年以上。' },
    { label: '帮我看看现在适合什么模型', message: '帮我看看现在适合什么模型' },
    { label: '我比较积极，想抓成长机会', message: '我比较积极，能接受 20% 左右的回撤，希望抓住成长股的长期机会。' },
    { label: '不想碰的行业怎么排除？', message: '我不想碰白酒和房地产行业，选股时帮我排除掉。' }
  ];

  /* ============================== 状态 ============================== */
  var state = {
    sessionId: '',
    streaming: false,
    activeTurn: null,
    cardNodes: new Map(),
    profile: null,
    activeModel: null,
    versions: [],
    pendingProposal: null,
    lastSelection: null,
    lastUserMessage: '',
    lastMessageId: null,
    activity: [],
    stateError: null,
    following: true
  };

  var dom = {};
  var composing = false;

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
    dom.activity = $('#wb-activity');
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
  }

  /* ============================== 网络请求 ============================== */
  async function api(path, options) {
    var opts = options || {};
    var response = await fetch(path, {
      method: opts.method || 'GET',
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
  }

  async function fetchMessages(sessionId) {
    try {
      var data = await api('/api/sessions/' + encodeURIComponent(sessionId) + '/messages');
      return { ok: true, messages: Array.isArray(data.messages) ? data.messages : [] };
    } catch (error) {
      return { ok: false, notFound: error.status === 404, error: error };
    }
  }

  async function loadState() {
    try {
      var data = await api('/api/sessions/' + encodeURIComponent(state.sessionId) + '/state');
      state.profile = normalizeProfile(data.profile);
      state.activeModel = normalizeModel(data.active_model);
      state.versions = normalizeVersions(data.versions);
      state.pendingProposal = data.pending_proposal || null;
      state.lastSelection = normalizeSelection(data.last_selection);
      state.stateError = null;
      setWorkbenchAlert('');
    } catch (error) {
      state.stateError = error.message || '会话状态加载失败';
      setWorkbenchAlert('会话状态加载失败：' + state.stateError);
    }
  }

  async function loadVersions() {
    try {
      var data = await api('/api/versions');
      var list = data && (data.versions || data.items) ? (data.versions || data.items) : data;
      var versions = normalizeVersions(list);
      if (versions.length) state.versions = versions;
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
      id: raw.id ? String(raw.id) : '',
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

  function renderHistoryBlocks(wrap, blocks) {
    if (!wrap || !Array.isArray(blocks) || !blocks.length) return;
    var main = wrap.querySelector('.msg-main');
    if (!main) return;
    var cards = document.createElement('div');
    cards.className = 'cards history-cards';
    blocks.forEach(function (block) {
      if (!block || !block.id || !block.type) return;
      if (block.type === 'profile') return;  // 当前画像由右侧工作台常驻展示
      var slot = document.createElement('div');
      slot.className = 'card-slot';
      slot.setAttribute('data-card-id', 'history-' + block.id);
      slot.innerHTML = renderCard(block);
      cards.appendChild(slot);
      mountCardCharts(slot, block);
    });
    if (cards.children.length) main.appendChild(cards);
  }

  function renderHistory(messages) {
    messages.forEach(function (message) {
      if (!message) return;
      var role = message.role === 'user' ? 'user' : 'assistant';
      var content = message.content === null || message.content === undefined ? '' : String(message.content);
      if (!content.trim() && !(message.blocks && message.blocks.length)) return;
      var wrap = appendTextMessage(role, content, message.created_at);
      if (role === 'assistant') renderHistoryBlocks(wrap, message.blocks);
      if (role === 'user') state.lastUserMessage = content;
    });
  }

  function beginAssistantTurn() {
    var wrap = document.createElement('div');
    wrap.className = 'msg msg-ai is-streaming';
    dom.stream.classList.remove('is-welcome');
    wrap.innerHTML = '<div class="msg-avatar">' + icon('spark') + '</div>'
      + '<div class="msg-main"><div class="msg-meta"><strong>AI 投资顾问</strong><time>' + esc(nowClock()) + '</time></div>'
      + '<div class="bubble"><span class="typing"><i></i><i></i><i></i></span></div>'
      + '<div class="cards"></div></div>';
    dom.stream.appendChild(wrap);
    var turn = {
      node: wrap,
      bubble: wrap.querySelector('.bubble'),
      cards: wrap.querySelector('.cards'),
      text: '',
      cardCount: 0,
      failed: false
    };
    state.activeTurn = turn;
    state.following = true;
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
    bubble.textContent = turn.text;
    if (state.following) scrollToBottom(true);
  }

  function finalizeTurn(turn) {
    if (!turn) return;
    turn.node.classList.remove('is-streaming');
    if (!turn.text && turn.bubble && turn.bubble.isConnected) turn.bubble.remove();
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
    pushActivity('error-' + nowClock() + '-' + Math.random().toString(36).slice(2, 6), title + '：' + message, 'error', 'alert');
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
      case 'approval':
        return approvalCardHtml(block);
      case 'version':
        return versionCardHtml(block);
      default:
        return genericCardHtml(block);
    }
  }

  function genericCardHtml(block) {
    return '<article class="card"><header class="card-head"><span class="card-icon tone-slate">' + icon('layers') + '</span>'
      + '<div class="card-title"><span class="card-kicker">AI BLOCK</span><h3>' + esc(block.title || block.type || 'AI 内容') + '</h3></div></header>'
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
      + '<header class="card-head"><span class="card-icon tone-blue">' + icon('user') + '</span>'
      + '<div class="card-title"><span class="card-kicker">INVESTOR PROFILE</span><h3>你的投资画像</h3></div>'
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
      var note = selection.metric_notes && selection.metric_notes[key] ? String(selection.metric_notes[key]) : meta.note;
      return '<div class="metric-card tone-' + meta.tone + '">'
        + '<span class="metric-label">' + esc(meta.label) + '</span>'
        + '<span class="metric-value">' + esc(formatMetric(key, selection.metrics[key])) + '</span>'
        + (note ? '<span class="metric-note">' + esc(note) + '</span>' : '')
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
        + '<p class="chart-note">条越长表示组合在这个因子上越占优（已完成行业中性化）。</p></div>'
      : '';
    var weightsHtml = portfolioHtml(selection);
    var navData = buildNavData(selection);
    var navHtml = navData
      ? '<div class="sub-block"><div class="sub-title"><h4>策略净值 vs 基准</h4><span class="sub-meta">起点归一为 1.00</span></div>'
        + '<div class="chart-box" data-chart="nav"></div>'
        + '<p class="chart-note">蓝色为策略净值，灰色虚线为基准；回测已计入交易成本与涨跌停约束。</p></div>'
      : '';
    var explanationHtml = selection.explanation
      ? '<div class="ai-note"><span class="ai-note-label">' + icon('spark') + 'AI 解读</span><p>' + esc(selection.explanation) + '</p></div>'
      : '';
    var splitHtml = factorsHtml || weightsHtml
      ? '<div class="selection-split">' + factorsHtml + weightsHtml + '</div>'
      : '';
    return '<article class="card card-selection">'
      + '<header class="card-head"><span class="card-icon tone-blue">' + icon('target') + '</span>'
      + '<div class="card-title"><span class="card-kicker">AI SELECTION</span><h3>本次选股结果</h3></div>'
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
      + (kind ? '<span class="step-kind">' + esc(kind) + '</span>' : '') + '</div>'
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
    return '<div class="sub-block"><div class="sub-title"><h4>效果对比</h4><span class="sub-meta">当前 → 候选</span></div>'
      + '<div class="chart-box" data-chart="compare"></div>'
      + '<p class="chart-note">灰色为当前模型，青色为候选模型；回撤越接近 0 越好，其余指标越大越好。</p></div>';
  }

  function codeBoxHtml(change) {
    if (!change || !change.diff) return '';
    var lines = String(change.diff).split('\n');
    var rows = lines.map(function (line, index) {
      var cls = 'code-line';
      if (line.indexOf('+++') === 0 || line.indexOf('---') === 0) cls += ' is-file';
      else if (line.indexOf('@@') === 0) cls += ' is-hunk';
      else if (line.charAt(0) === '+') cls += ' is-add';
      else if (line.charAt(0) === '-') cls += ' is-del';
      return '<div class="' + cls + '"><span class="code-ln">' + (index + 1) + '</span><span class="code-text">' + esc(line) + '</span></div>';
    }).join('');
    var added = Number(change.added) || 0;
    var removed = Number(change.removed) || 0;
    var kind = change.kind === 'code' ? '代码改动'
      : change.kind === 'model_type' ? '模型类型改动' : '因子权重改动';
    return '<div class="code-change">'
      + '<div class="code-head"><span class="code-head-left"><span class="code-kind">' + esc(kind) + '</span>'
      + '<span class="code-file">' + esc(change.file || 'models/strategy.py') + '</span></span>'
      + '<span class="code-stats">+' + added + ' / -' + removed + '</span></div>'
      + '<pre class="code-box" aria-label="AI 正在修改的模型内容">' + rows + '</pre></div>';
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
    var risks = Array.isArray(proposal.risks) && proposal.risks.length
      ? '<div class="risk-list"><h5>风险提示</h5><ul>' + proposal.risks.map(function (risk) {
        return '<li>' + esc(risk) + '</li>';
      }).join('') + '</ul></div>'
      : '';
    return '<div class="proposal">' + summary
      + (tags.length ? '<div class="proposal-meta">' + tags.join('') + '</div>' : '')
      + diff + risks + '</div>';
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
    return '<article class="card card-work">'
      + '<header class="card-head"><span class="card-icon tone-amber">' + icon('spark') + '</span>'
      + '<div class="card-title"><span class="card-kicker">AI WORK LOG</span><h3>' + esc(block.title || 'AI 正在工作') + '</h3></div>'
      + statusPill + '</header>'
      + '<div class="card-body">' + progressHtml + stepsHtml
      + codeBoxHtml(block.code_change)
      + comparisonHtml(block.comparison) + proposalHtml(block.proposal, !block.code_change)
      + '</div></article>';
  }

  /* ---------------------------------------------------------- 审批卡 */
  function approvalCardHtml(block) {
    var actions = Array.isArray(block.actions) ? block.actions : [];
    var buttons = actions.map(function (action, index) {
      if (!action || !action.message) return '';
      return '<button class="btn ' + (index === 0 ? 'btn-primary' : 'btn-ghost') + '" type="button" data-action="quick" data-message="' + esc(action.message) + '">'
        + esc(action.label || action.message) + '</button>';
    }).join('');
    var riskHtml = block.risk
      ? '<div class="approval-risk">' + icon('shield') + '<span>' + esc(block.risk) + '</span></div>'
      : '';
    return '<article class="card card-approval">'
      + '<header class="card-head"><span class="card-icon tone-amber">' + icon('shield') + '</span>'
      + '<div class="card-title"><span class="card-kicker">NEEDS YOUR APPROVAL</span><h3>' + esc(block.title || '需要你确认') + '</h3></div>'
      + '<span class="status-pill is-warn">等待确认</span></header>'
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
      + '<header class="card-head"><span class="card-icon tone-blue">' + icon('layers') + '</span>'
      + '<div class="card-title"><span class="card-kicker">MODEL VERSION</span><h3>模型版本已更新</h3></div>'
      + '<span class="status-pill is-ok">已生效</span></header>'
      + '<div class="card-body">' + activeHtml + listHtml + '</div></article>';
  }

  /* ============================== 卡片挂载与图表 ============================== */
  function upsertCard(block) {
    if (!block || !block.id || !block.type) return;
    var turn = state.activeTurn;
    if (!turn) turn = beginAssistantTurn();
    var key = String(block.id);
    var node = state.cardNodes.get(key);
    if (node && !node.isConnected) node = null;
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
    var html = renderCard(block);
    if (!html) return;
    node.innerHTML = html;
    mountCardCharts(node, block);
    syncFromBlock(block);
    if (state.following) scrollToBottom(true);
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
    if (block.type === 'work') {
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
  }

  function syncFromBlock(block) {
    switch (block.type) {
      case 'profile':
        state.profile = normalizeProfile(block);
        renderProfileSection();
        pushActivity('profile-' + block.id, state.profile && state.profile.confirmed ? '已确认你的投资画像' : '更新了你的投资画像', 'done', 'profile');
        break;
      case 'selection':
        state.lastSelection = normalizeSelection(block);
        renderResultSection();
        pushActivity('selection-' + block.id,
          '完成选股' + (state.lastSelection && state.lastSelection.top.length ? ' · Top ' + state.lastSelection.top.length : ''),
          'done', 'selection');
        break;
      case 'work':
        syncWorkActivity(block);
        break;
      case 'approval':
        pushActivity('approval-' + block.id, '等待你确认：' + (block.title || '待审批方案'), 'waiting', 'wait');
        break;
      case 'version': {
        if (block.active && typeof block.active === 'object') state.activeModel = normalizeModel(block.active);
        var parsedVersions = normalizeVersions(block.versions);
        if (parsedVersions.length) state.versions = parsedVersions;
        renderModelSection();
        renderChatModelChip();
        pushActivity('version-' + block.id, '模型版本已更新' + (block.active && block.active.name ? ' · ' + block.active.name : ''), 'done', 'version');
        break;
      }
      default:
        break;
    }
  }

  function syncWorkActivity(block) {
    var steps = Array.isArray(block.steps) ? block.steps : [];
    steps.forEach(function (step, index) {
      if (!step) return;
      var status = step.status === 'done' ? 'done'
        : step.status === 'running' ? 'running'
          : step.status === 'error' ? 'error' : 'pending';
      var detail = step.detail ? ' · ' + truncate(step.detail, 44) : '';
      pushActivity('work:' + block.id + ':' + index, (step.title || ('第 ' + (index + 1) + ' 步')) + detail, status, step.kind || 'work');
    });
    if (block.status === 'done') {
      pushActivity('work:' + block.id + ':overall', (block.title || 'AI 工作') + ' · 已完成', 'done', 'check');
    } else if (block.status === 'error') {
      pushActivity('work:' + block.id + ':overall', (block.title || 'AI 工作') + ' · 失败', 'error', 'alert');
    } else if (!steps.some(function (step) { return step && step.status === 'running'; })) {
      pushActivity('work:' + block.id + ':overall', block.title || 'AI 正在工作', 'running', 'ai');
    }
  }

  /* ============================== 工作台 ============================== */
  function renderAllWorkbench() {
    renderProfileSection();
    renderActivitySection();
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
    var head = '<header class="wb-head"><span class="wb-kicker">PROFILE</span><h2>当前画像</h2>'
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
      + (profile.confirmed ? '已确认，可用于选股' : '等待你确认') + '</span>'
      + '<p class="wb-summary">' + esc(summary) + '</p></div></div>'
      + (items ? '<ul class="wb-items">' + items + '</ul>' : '');
    var ring = dom.profile.querySelector('[data-chart="ring"]');
    if (ring && typeof window.Charts !== 'undefined') {
      window.Charts.gauge(ring, { value: pct, max: 100, label: '完整度' }, { size: 66, thickness: 5, ariaLabel: '画像完整度' });
    }
  }

  function activityIcon(kind, status) {
    if (status === 'running') return '<i class="spinner"></i>';
    if (status === 'done') return icon('check');
    if (status === 'error') return icon('alert');
    if (status === 'waiting') return icon('shield');
    return icon(ACTIVITY_ICONS[kind] || 'spark');
  }

  function renderActivitySection() {
    var head = '<header class="wb-head"><span class="wb-kicker">ACTIVITY</span><h2>AI 活动时间线</h2></header>';
    if (!state.activity.length) {
      dom.activity.innerHTML = head + '<p class="wb-empty">AI 的每一步动作都会实时显示在这里。</p>';
      return;
    }
    var rows = state.activity.map(function (item) {
      return '<li class="act is-' + esc(item.status) + '"><span class="act-icon">' + activityIcon(item.kind, item.status) + '</span>'
        + '<div class="act-body"><p class="act-text">' + esc(item.text) + '</p><time class="act-time">' + esc(item.time) + '</time></div></li>';
    }).join('');
    dom.activity.innerHTML = head + '<ul class="activity-list">' + rows + '</ul>';
  }

  function pushActivity(key, text, status, kind) {
    var existing = null;
    for (var i = 0; i < state.activity.length; i += 1) {
      if (state.activity[i].key === key) {
        existing = state.activity[i];
        break;
      }
    }
    if (existing) {
      existing.text = text;
      existing.status = status;
      existing.kind = kind || existing.kind;
      existing.time = nowClock();
    } else {
      state.activity.unshift({
        key: key,
        text: text,
        status: status,
        kind: kind || 'ai',
        time: nowClock()
      });
      if (state.activity.length > MAX_ACTIVITY) state.activity.length = MAX_ACTIVITY;
    }
    renderActivitySection();
  }

  function renderModelSection() {
    var head = '<header class="wb-head"><span class="wb-kicker">MODEL</span><h2>当前模型与版本</h2></header>';
    var html = head;
    var model = state.activeModel;
    if (model) {
      var chips = metricChipsHtml(model.metrics);
      html += '<div class="wb-model-top"><strong>' + esc(model.name) + '</strong><span class="status-pill is-ok">运行中</span></div>'
        + (model.description ? '<p class="wb-model-desc">' + esc(model.description) + '</p>' : '')
        + (chips ? '<div class="wb-chips">' + chips + '</div>' : '')
        + '<p class="wb-note">年化收益、夏普比率、最大回撤等均为历史回测数据，不代表未来实际表现，不构成投资建议。</p>';
    } else {
      html += '<p class="wb-empty">还没有启用的自定义模型。跟 AI 说「帮我优化模型」，它会先给你方案，再等你确认。</p>';
    }
    var versions = state.versions.slice(0, 6);
    if (versions.length) {
      html += '<ul class="wb-version-list">' + versions.map(function (version) {
        var dotClass = version.status === 'active' ? ' is-active' : version.status === 'candidate' ? ' is-candidate' : '';
        return '<li class="wb-version"><span class="version-dot' + dotClass + '"></span>'
          + '<span class="wb-version-name">' + esc(version.name) + '</span>'
          + '<span class="status-pill ' + versionStatusClass(version.status) + '">' + esc(versionStatusText(version.status)) + '</span>'
          + (version.created_at ? '<time>' + esc(shortDate(version.created_at)) + '</time>' : '')
          + '</li>';
      }).join('') + '</ul>';
    }
    if (state.pendingProposal) {
      html += '<div class="wb-pending-box"><strong>有一个待确认的改进方案</strong><br>AI 会在聊天里等你确认，不会自动启用。'
        + '<br><button class="btn btn-sm" type="button" data-action="quick" data-message="展开说说待审批方案的细节">展开说说方案</button></div>';
    }
    dom.model.innerHTML = html;
  }

  function renderResultSection() {
    var head = '<header class="wb-head"><span class="wb-kicker">LAST SELECTION</span><h2>最近选股结果</h2></header>';
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

  function setChatRunStatus(text) {
    if (dom.chatRunStatus) dom.chatRunStatus.textContent = text;
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
  function updateComposerState() {
    var busy = state.streaming;
    dom.sendBtn.disabled = busy;
    if (dom.stopBtn) dom.stopBtn.hidden = !busy;
    if (dom.regenerateBtn) dom.regenerateBtn.disabled = busy || !state.lastUserMessage;
    if (dom.clearInput) dom.clearInput.disabled = busy;
    dom.input.placeholder = busy
      ? 'AI 正在回复，可点击停止生成…'
      : '输入投资问题或分析需求，例如：帮我看看现在适合什么模型……';
  }

  async function sendMessage(rawContent) {
    var content = String(rawContent === null || rawContent === undefined ? '' : rawContent).trim();
    if (!content) {
      dom.input.focus();
      return;
    }
    if (state.streaming) {
      dom.input.focus();
      return;
    }
    if (!state.sessionId) {
      appendSystemError('会话未就绪', '还没有建立会话，请点击重新连接。', 'retry-init');
      return;
    }
    state.streaming = true;
    setChatRunStatus('生成中…');
    updateComposerState();
    dom.input.value = '';
    resizeInput();
    state.following = true;
    appendTextMessage('user', content, null);
    state.lastUserMessage = content;
    updateChatSessionLabel();
    var turn = beginAssistantTurn();
    pushActivity('turn', 'AI 正在处理你的请求…', 'running', 'ai');
    var failed = false;
    var stopped = false;
    var controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
    state.abortController = controller;
    try {
      var response = await fetch('/api/sessions/' + encodeURIComponent(state.sessionId) + '/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
        body: JSON.stringify({ content: content }),
        signal: controller ? controller.signal : undefined
      });
      if (!response.ok) {
        var detail = await readErrorDetail(response);
        throw new Error(detail || '对话请求失败（HTTP ' + response.status + '）');
      }
      if (!response.body) throw new Error('当前浏览器不支持流式响应');
      await readEventStream(response.body, turn, function (payload) {
        failed = true;
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
        appendTextMessage('ai', '已停止生成。你可以修改问题后重新发送，或点击输入区的「重新生成」。', null);
        pushActivity('turn', '已停止生成', 'error', 'alert');
      } else {
        showTurnError(turn, '网络异常', error && error.message ? error.message : '请检查后端服务后重试。', true);
      }
    } finally {
      state.abortController = null;
      finalizeTurn(turn);
      state.streaming = false;
      setChatRunStatus(stopped ? '已停止' : failed ? '处理出错' : '就绪');
      updateComposerState();
      if (!stopped) {
        pushActivity('turn', failed ? '本次处理未完成' : '已完成回复', failed ? 'error' : 'done', failed ? 'alert' : 'check');
      }
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
      case 'ui':
        if (payload.block && typeof payload.block === 'object') upsertCard(payload.block);
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
    state.activity = [];
    if (history && history.length) {
      renderHistory(history);
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
      pushActivity('restore', '已恢复上次会话的画像、模型与工作台状态', 'done', 'check');
    }
    renderAllWorkbench();
    state.following = true;
    scrollToBottom(true);
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
    var data = await api('/api/sessions');
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
    if (dom.infoOverlay && dom.infoOverlay.hidden) setActiveNav('assistant');
  }

  async function activateSession(sessionId) {
    if (!sessionId) return false;
    if (state.streaming) {
      dom.historyList.insertAdjacentHTML('afterbegin', '<p class="history-empty">当前回复还没结束，请稍后再切换会话。</p>');
      return false;
    }
    closeHistory();
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
    state.activity = [];
    state.stateError = null;
    state.following = true;
    var fetched = await fetchMessages(sessionId);
    var history = fetched.ok ? (fetched.messages || []) : [];
    if (history.length) renderHistory(history);
    else appendWelcome();
    updateChatSessionLabel();
    setChatRunStatus('就绪');
    updateComposerState();
    setActiveNav('assistant');
    closeInfo();
    await loadState();
    await loadVersions();
    renderAllWorkbench();
    if (history.length) pushActivity('switch', '已切换到一个历史对话', 'done', 'history');
    scrollToBottom(true);
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
      help: { kicker: 'HELP', title: '帮助中心', subtitle: '如何用自然语言完成投资画像、选股与模型优化。' }
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
        return '<p>还没有生成投资组合。回到「AI 投资顾问」对话，先确认投资画像，再让 AI 帮你选股。</p>';
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
      + '<li>右侧「投资上下文」会同步显示画像、AI 活动、当前模型和最近结果。</li>'
      + '<li>顶栏可以打开历史记录，切换回之前的会话继续分析。</li>'
      + '</ul>' + infoNote('AI 不会重复询问已确认的画像字段；所有收益与回撤指标均标注为历史回测数据。');
  }

  /* ============================== 事件绑定 ============================== */
  function handleActionClick(event) {
    var button = event.target.closest ? event.target.closest('[data-action]') : null;
    if (!button || button.disabled) return;
    var action = button.getAttribute('data-action');
    if (action === 'quick') {
      var message = button.getAttribute('data-message');
      var approval = button.closest('.card-approval');
      if (approval) {
        approval.classList.add('is-answered');
        var pill = approval.querySelector('.card-head .status-pill');
        if (pill) {
          pill.className = 'status-pill is-ok';
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
    if (dom.stopBtn) dom.stopBtn.addEventListener('click', function () {
      if (state.abortController) state.abortController.abort();
    });
    if (dom.regenerateBtn) dom.regenerateBtn.addEventListener('click', function () {
      if (!state.streaming && state.lastUserMessage) sendMessage(state.lastUserMessage);
      else dom.input.focus();
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

  /* ============================== 启动 ============================== */
  function boot() {
    cacheDom();
    bindEvents();
    syncWorkbenchToggle();
    renderAllWorkbench();
    bootstrap();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
