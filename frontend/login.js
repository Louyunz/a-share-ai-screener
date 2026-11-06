(function () {
  'use strict';

  var els = {
    message: document.getElementById('auth-message'),
    hint: document.getElementById('auth-hint'),
    loginForm: document.getElementById('login-form'),
    setupForm: document.getElementById('setup-form'),
    loginUser: document.getElementById('login-username'),
    loginPass: document.getElementById('login-password'),
    loginSubmit: document.getElementById('login-submit'),
    setupUser: document.getElementById('setup-username'),
    setupPass: document.getElementById('setup-password'),
    setupConfirm: document.getElementById('setup-confirm'),
    setupSubmit: document.getElementById('setup-submit')
  };

  function showMessage(text, ok) {
    if (!els.message) return;
    els.message.textContent = text || '';
    els.message.hidden = !text;
    els.message.classList.toggle('is-ok', !!ok);
  }

  function setBusy(button, busy, label) {
    if (!button) return;
    button.disabled = !!busy;
    if (busy) {
      button.dataset.label = button.dataset.label || button.textContent;
      button.textContent = label || '请稍候…';
    } else if (button.dataset.label) {
      button.textContent = button.dataset.label;
    }
  }

  async function post(path, body) {
    var response = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(body)
    });
    var raw = await response.text();
    var data = null;
    try { data = raw ? JSON.parse(raw) : null; } catch (err) { data = null; }
    if (!response.ok) {
      var error = new Error((data && (data.error || data.message)) || ('请求失败（HTTP ' + response.status + '）'));
      error.status = response.status;
      throw error;
    }
    return data || {};
  }

  async function loadStatus() {
    var response = await fetch('/api/auth/status', { headers: { Accept: 'application/json' } });
    if (!response.ok) throw new Error('无法获取登录状态（HTTP ' + response.status + '）');
    return response.json();
  }

  function enterApp() {
    window.location.replace('/');
  }

  function bindLogin() {
    els.loginForm.addEventListener('submit', async function (event) {
      event.preventDefault();
      showMessage('');
      setBusy(els.loginSubmit, true, '登录中…');
      try {
        await post('/api/auth/login', {
          username: els.loginUser.value.trim(),
          password: els.loginPass.value
        });
        enterApp();
      } catch (error) {
        showMessage(error.message || '登录失败，请重试。');
        els.loginPass.select();
      } finally {
        setBusy(els.loginSubmit, false);
      }
    });
  }

  function bindSetup() {
    els.setupForm.addEventListener('submit', async function (event) {
      event.preventDefault();
      showMessage('');
      var username = els.setupUser.value.trim();
      var password = els.setupPass.value;
      if (password !== els.setupConfirm.value) {
        showMessage('两次输入的密码不一致。');
        els.setupConfirm.select();
        return;
      }
      if (password.length < 10) {
        showMessage('密码至少需要 10 位。');
        els.setupPass.select();
        return;
      }
      setBusy(els.setupSubmit, true, '创建中…');
      try {
        await post('/api/auth/setup', { username: username, password: password });
        enterApp();
      } catch (error) {
        showMessage(error.message || '创建管理员账号失败。');
      } finally {
        setBusy(els.setupSubmit, false);
      }
    });
  }

  async function main() {
    var params = new URLSearchParams(window.location.search || '');
    if (params.get('expired')) showMessage('登录已过期，请重新登录。');
    try {
      var status = await loadStatus();
      if (status.authenticated) {
        enterApp();
        return;
      }
      if (status.setup_required && status.setup_allowed) {
        els.setupForm.hidden = false;
        bindSetup();
        els.setupUser.focus();
        return;
      }
      if (status.setup_required && !status.setup_allowed) {
        els.hint.hidden = false;
        els.hint.textContent = '管理员账号尚未初始化。请先在服务器本机打开本页面完成初始化，或通过 AUTH_USERNAME / AUTH_PASSWORD_HASH 配置账号。';
        return;
      }
      els.loginForm.hidden = false;
      bindLogin();
      els.loginUser.focus();
    } catch (error) {
      showMessage(error.message || '无法连接到服务器。');
    }
  }

  main();
})();
