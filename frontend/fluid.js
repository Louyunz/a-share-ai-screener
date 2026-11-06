/* =========================================================================
 * 动态流体背景：浅色柔和色团缓慢流动；AI 工作时自动加速
 * 使用 Canvas 2D + CSS blur，无第三方依赖
 * ========================================================================= */
(function (global) {
  'use strict';

  var canvas = null;
  var ctx = null;
  var width = 0;
  var height = 0;
  var dpr = 1;
  var running = false;
  var lastTime = 0;
  var energy = 1;          // 当前速度倍率（平滑过渡）
  var targetEnergy = 1;
  var pointer = { x: 0.5, y: 0.5, active: false };
  var reduceMotion = false;

  var COLORS = [
    { color: [142, 180, 255], alpha: 0.64 },  // 淡蓝
    { color: [196, 176, 255], alpha: 0.56 },  // 淡紫
    { color: [255, 198, 180], alpha: 0.50 },  // 淡橙
    { color: [170, 226, 238], alpha: 0.46 },  // 淡青
    { color: [255, 214, 232], alpha: 0.42 }   // 淡粉
  ];

  var blobs = [];

  function makeBlobs() {
    var count = width < 720 ? 4 : 6;
    blobs = [];
    for (var i = 0; i < count; i += 1) {
      var palette = COLORS[i % COLORS.length];
      blobs.push({
        x: Math.random(),
        y: Math.random(),
        radius: 0.28 + Math.random() * 0.24,
        vx: (Math.random() - 0.5) * 0.0018,
        vy: (Math.random() - 0.5) * 0.0018,
        phase: Math.random() * Math.PI * 2,
        palette: palette
      });
    }
  }

  function resize() {
    dpr = Math.min(global.devicePixelRatio || 1, 2);
    width = global.innerWidth;
    height = global.innerHeight;
    if (canvas) {
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = width + 'px';
      canvas.style.height = height + 'px';
    }
    if (!blobs.length) makeBlobs();
  }

  function draw(time, dt) {
    if (!ctx) return;
    var speed = reduceMotion ? 0.25 : energy;
    var minSide = Math.min(width, height);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.globalCompositeOperation = 'source-over';

    for (var i = 0; i < blobs.length; i += 1) {
      var blob = blobs[i];
      blob.phase += dt * 0.00016 * speed;
      blob.x += blob.vx * speed * dt * 0.06;
      blob.y += blob.vy * speed * dt * 0.06;
      if (blob.x < -0.25) blob.x = 1.25;
      if (blob.x > 1.25) blob.x = -0.25;
      if (blob.y < -0.25) blob.y = 1.25;
      if (blob.y > 1.25) blob.y = -0.25;

      var driftX = Math.sin(blob.phase) * 0.045;
      var driftY = Math.cos(blob.phase * 0.8) * 0.045;
      var px = blob.x + driftX;
      var py = blob.y + driftY;
      if (pointer.active && energy > 1.2) {
        px += (pointer.x - px) * 0.035;
        py += (pointer.y - py) * 0.035;
      }
      var cx = px * width;
      var cy = py * height;
      var radius = blob.radius * minSide * (1 + Math.sin(blob.phase * 0.7) * 0.10);
      var gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
      var rgb = blob.palette.color;
      gradient.addColorStop(0, 'rgba(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ',' + blob.palette.alpha * (0.82 + energy * 0.08) + ')');
      gradient.addColorStop(0.55, 'rgba(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ',' + blob.palette.alpha * 0.35 + ')');
      gradient.addColorStop(1, 'rgba(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ',0)');
      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function loop(time) {
    if (!running) return;
    var dt = lastTime ? Math.min(time - lastTime, 64) : 16;
    lastTime = time;
    var target = targetEnergy;
    energy += (target - energy) * Math.min(1, dt / 420);
    draw(time, dt);
    global.requestAnimationFrame(loop);
  }

  function readWorkingState() {
    targetEnergy = document.body.classList.contains('ai-working') ? 3.6 : 1;
  }

  function start() {
    if (running || !canvas) return;
    running = true;
    lastTime = 0;
    global.requestAnimationFrame(loop);
  }

  function init() {
    reduceMotion = !!(global.matchMedia && global.matchMedia('(prefers-reduced-motion: reduce)').matches);
    canvas = document.getElementById('fluid-canvas');
    if (!canvas) return;
    ctx = canvas.getContext('2d');
    resize();
    makeBlobs();
    readWorkingState();
    var observer = new MutationObserver(readWorkingState);
    observer.observe(document.body, { attributes: true, attributeFilter: ['class'] });
    global.addEventListener('resize', resize, { passive: true });
    document.addEventListener('pointermove', function (event) {
      pointer.x = event.clientX / Math.max(1, width);
      pointer.y = event.clientY / Math.max(1, height);
      pointer.active = true;
    }, { passive: true });
    start();
  }

  global.FluidBackground = {
    start: start,
    refresh: readWorkingState,
    getEnergy: function () { return energy; },
    setEnergy: function (value) {
      targetEnergy = Math.max(0.6, Math.min(3.2, Number(value) || 1));
    }
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})(window);
