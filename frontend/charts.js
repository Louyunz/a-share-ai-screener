/* =========================================================================
 * 智投罗盘 · 纯 SVG 图表库（零依赖）
 * 暴露全局 Charts：bar / line / gauge
 * 所有图表接收容器元素（或选择器）与数据对象，重绘并返回生成的 SVG。
 * ========================================================================= */
(function (global) {
  'use strict';

  var NS = 'http://www.w3.org/2000/svg';
  var uid = 0;
  var COLORS = {
    blue: '#0071E3',
    cyan: '#0A84FF',
    orange: '#C77700',
    green: '#1F8F63',
    red: '#C9333D',
    violet: '#6E56CF',
    slate: '#8E8E93',
    text: '#6E6E73',
    muted: '#8E8E93',
    grid: 'rgba(0, 0, 0, 0.06)',
    axis: 'rgba(0, 0, 0, 0.10)'
  };

  /* ------------------------------------------------------------- 基础工具 */
  function resolve(target) {
    return typeof target === 'string' ? document.querySelector(target) : target;
  }

  function svgEl(tag, attrs) {
    var node = document.createElementNS(NS, tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (key) {
        if (attrs[key] !== null && attrs[key] !== undefined) node.setAttribute(key, attrs[key]);
      });
    }
    return node;
  }

  function textEl(x, y, content, attrs) {
    var node = svgEl('text', Object.assign({ x: x, y: y }, attrs || {}));
    node.textContent = content === null || content === undefined ? '' : String(content);
    return node;
  }

  function num(value) {
    if (value === null || value === undefined || value === '') return null;
    var n = typeof value === 'number' ? value : parseFloat(value);
    return Number.isFinite(n) ? n : null;
  }

  function firstNum() {
    for (var i = 0; i < arguments.length; i += 1) {
      var n = num(arguments[i]);
      if (n !== null) return n;
    }
    return null;
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function fmt(value, digits) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—';
    return Number(value).toFixed(digits === undefined ? 2 : digits);
  }

  function shortText(value, max) {
    var chars = Array.from(String(value === null || value === undefined ? '' : value));
    return chars.length > max ? chars.slice(0, max - 1).join('') + '…' : chars.join('');
  }

  function nextId(prefix) {
    uid += 1;
    return prefix + '-' + uid;
  }

  function root(container, width, height, label) {
    container.innerHTML = '';
    container.classList.add('chart-anchor');
    var svg = svgEl('svg', {
      viewBox: '0 0 ' + width + ' ' + height,
      width: '100%',
      role: 'img',
      'aria-label': label || '数据图表',
      preserveAspectRatio: 'xMidYMid meet',
      class: 'chart-svg'
    });
    container.appendChild(svg);
    return svg;
  }

  function empty(container, message) {
    container.innerHTML = '';
    container.classList.add('chart-anchor');
    var node = document.createElement('div');
    node.className = 'chart-empty';
    node.textContent = message || '暂无图表数据';
    container.appendChild(node);
    return null;
  }

  function tooltipFor(container) {
    var tip = container.querySelector('.chart-tooltip');
    if (!tip) {
      tip = document.createElement('div');
      tip.className = 'chart-tooltip';
      tip.hidden = true;
      container.appendChild(tip);
    }
    return tip;
  }

  function fillTooltip(tip, title, rows) {
    tip.textContent = '';
    var head = document.createElement('div');
    head.className = 'tt-title';
    head.textContent = title;
    tip.appendChild(head);
    (rows || []).forEach(function (row) {
      var line = document.createElement('div');
      line.className = 'tt-row';
      if (row.color) {
        var dot = document.createElement('i');
        dot.className = 'tt-dot';
        dot.style.background = row.color;
        line.appendChild(dot);
      }
      var label = document.createElement('span');
      label.textContent = row.label;
      var value = document.createElement('b');
      value.style.color = row.color || '#1D1D1F';
      value.textContent = row.value;
      line.appendChild(label);
      line.appendChild(value);
      tip.appendChild(line);
    });
    var width = tip.parentElement ? tip.parentElement.clientWidth : 240;
    tip.style.left = '50%';
    tip.style.top = '8px';
    tip.hidden = false;
    tip.style.transform = 'translateX(-50%)';
    return width;
  }

  /* ------------------------------------------------------- 水平对比条 bar */
  /**
   * 单序列：{ items: [{ label, current }] }
   * 双序列：{ items: [{ label, current, candidate, key }] }
   * options: { width, rowHeight, labelWidth, valueFormatter(value, item),
   *            currentLabel, candidateLabel, ariaLabel, emptyText }
   */
  function bar(target, data, options) {
    var container = resolve(target);
    if (!container) return null;
    var opts = options || {};
    var input = data || {};
    var rawItems = Array.isArray(input) ? input : (input.items || input.rows || []);
    var items = rawItems.map(function (item, index) {
      if (item === null || item === undefined) return null;
      return {
        key: item.key || '',
        label: String(item.label !== undefined ? item.label
          : item.name !== undefined ? item.name
            : item.factor !== undefined ? item.factor
              : '项 ' + (index + 1)),
        current: firstNum(item.current, item.base, item.current_value, item.value),
        candidate: firstNum(item.candidate, item.best, item.candidate_value, item.target)
      };
    }).filter(function (item) {
      return item && (item.current !== null || item.candidate !== null);
    });

    if (!items.length) return empty(container, opts.emptyText || '暂无对比数据');

    var hasCandidate = items.some(function (item) { return item.candidate !== null; });
    var formatter = typeof opts.valueFormatter === 'function'
      ? opts.valueFormatter
      : function (value) { return fmt(value, 2); };
    var width = opts.width || 620;
    var rowHeight = opts.rowHeight || 34;
    var showValues = opts.showValues !== false;
    var labelWidth = opts.labelWidth || Math.min(118, Math.max(70, width * 0.17));
    var valueWidth = showValues ? Math.min(150, Math.max(92, width * 0.22)) : 10;
    var plotX = labelWidth;
    var plotW = Math.max(70, width - labelWidth - valueWidth);
    var top = 6;
    var height = opts.height || top + items.length * rowHeight + 4;
    var svg = root(container, width, height, opts.ariaLabel || '水平对比条形图');

    var allValues = [];
    items.forEach(function (item) {
      if (item.current !== null) allValues.push(item.current);
      if (item.candidate !== null) allValues.push(item.candidate);
    });
    var min = Math.min.apply(null, allValues.concat([0]));
    var max = Math.max.apply(null, allValues.concat([0]));
    if (max === min) max = min + 1;
    var span = max - min;
    var zeroX = plotX + ((0 - min) / span) * plotW;
    var plotRight = plotX + plotW;

    var defs = svgEl('defs');
    var gradId = nextId('bar-grad');
    var grad = svgEl('linearGradient', { id: gradId, x1: '0', y1: '0', x2: '1', y2: '0' });
    grad.appendChild(svgEl('stop', { offset: '0%', 'stop-color': '#0071E3' }));
    grad.appendChild(svgEl('stop', { offset: '100%', 'stop-color': '#5AC8FA' }));
    defs.appendChild(grad);
    svg.appendChild(defs);

    items.forEach(function (item, index) {
      var y = top + index * rowHeight;
      var centerY = y + rowHeight / 2;
      var group = svgEl('g', {});

      group.appendChild(textEl(plotX - 10, centerY + 3, shortText(item.label, Math.max(4, Math.floor(labelWidth / 10))), {
        'text-anchor': 'end',
        class: 'chart-label'
      }));

      var specs = [];
      if (hasCandidate) {
        specs.push({ key: 'current', value: item.current, y: centerY - 7.5, h: 5, fill: '#C7C7CC', opacity: 0.7 });
        specs.push({ key: 'candidate', value: item.candidate, y: centerY + 2.5, h: 6, fill: 'url(#' + gradId + ')', opacity: 1 });
      } else {
        specs.push({ key: 'current', value: item.current, y: centerY - 3.5, h: 7, fill: 'url(#' + gradId + ')', opacity: 1 });
      }

      specs.forEach(function (spec) {
        var value = spec.value === null ? 0 : spec.value;
        var endX = plotX + ((value - min) / span) * plotW;
        var x = Math.min(zeroX, endX);
        var barW = Math.abs(endX - zeroX);
        group.appendChild(svgEl('rect', {
          x: plotX, y: spec.y, width: plotW, height: spec.h, rx: spec.h / 2,
          fill: 'rgba(0, 0, 0, 0.04)'
        }));
        if (barW > 0.2) {
          var rect = svgEl('rect', {
            x: x,
            y: spec.y,
            width: Math.max(2, barW),
            height: spec.h,
            rx: spec.h / 2,
            fill: spec.fill,
            opacity: spec.opacity,
            class: opts.animate === false ? '' : 'chart-bar'
          });
          if (opts.animate !== false) rect.style.animationDelay = (index * 30) + 'ms';
          group.appendChild(rect);
        }
      });

      if (showValues) {
        var valueNode;
        if (hasCandidate) {
          valueNode = textEl(plotRight + valueWidth - 4, centerY + 3, '', { 'text-anchor': 'end', class: 'chart-mono' });
          var currentSpan = svgEl('tspan', { fill: '#6E6E73' });
          currentSpan.textContent = item.current === null ? '—' : formatter(item.current, item);
          valueNode.appendChild(currentSpan);
          var arrow = svgEl('tspan', { fill: '#1D1D1F', dx: '3' });
          arrow.textContent = '→';
          valueNode.appendChild(arrow);
          var candidateSpan = svgEl('tspan', { fill: '#0071E3', dx: '3' });
          candidateSpan.textContent = item.candidate === null ? '—' : formatter(item.candidate, item);
          valueNode.appendChild(candidateSpan);
        } else {
          valueNode = textEl(plotRight + valueWidth - 4, centerY + 3, formatter(item.current, item), {
            'text-anchor': 'end',
            class: 'chart-mono'
          });
        }
        group.appendChild(valueNode);
      }

      var title = svgEl('title');
      title.textContent = (item.title || item.label) + '：' + (item.current === null ? '—' : formatter(item.current, item)) +
        (hasCandidate ? ' → ' + (item.candidate === null ? '—' : formatter(item.candidate, item)) : '');
      group.appendChild(title);
      svg.appendChild(group);
    });

    return svg;
  }

  /* -------------------------------------------------------------- 折线 line */
  function smoothPath(points) {
    if (!points.length) return '';
    if (points.length === 1) return 'M' + points[0].x + ',' + points[0].y;
    if (points.length === 2) return 'M' + points[0].x + ',' + points[0].y + ' L' + points[1].x + ',' + points[1].y;
    var d = 'M' + points[0].x.toFixed(2) + ',' + points[0].y.toFixed(2);
    for (var i = 0; i < points.length - 1; i += 1) {
      var p0 = points[i - 1] || points[i];
      var p1 = points[i];
      var p2 = points[i + 1];
      var p3 = points[i + 2] || p2;
      var cp1x = p1.x + (p2.x - p0.x) / 6;
      var cp1y = p1.y + (p2.y - p0.y) / 6;
      var cp2x = p2.x - (p3.x - p1.x) / 6;
      var cp2y = p2.y - (p3.y - p1.y) / 6;
      d += ' C' + cp1x.toFixed(2) + ',' + cp1y.toFixed(2) + ' ' + cp2x.toFixed(2) + ',' + cp2y.toFixed(2) +
        ' ' + p2.x.toFixed(2) + ',' + p2.y.toFixed(2);
    }
    return d;
  }

  /**
   * data: { labels: [...], series: [{ name, values, color, dashed }] }
   * options: { width, height, yFormatter, xFormatter, ariaLabel, showArea,
   *            showEnds, emptyText }
   */
  function line(target, data, options) {
    var container = resolve(target);
    if (!container) return null;
    var opts = options || {};
    var input = data || {};
    var labels = (input.labels || input.dates || input.x || []).slice();
    var rawSeries = input.series || input.lines || [];
    if (!rawSeries.length && input.values) {
      rawSeries = [{ name: input.name || '序列', values: input.values }];
    }
    var series = rawSeries.map(function (item, index) {
      return {
        name: String(item.name || item.label || '序列 ' + (index + 1)),
        values: (item.values || item.data || []).map(function (value) { return num(value); }),
        color: item.color || (index === 0 ? COLORS.cyan : COLORS.slate),
        dashed: !!item.dashed
      };
    }).filter(function (item) {
      return item.values.some(function (value) { return value !== null; });
    });

    if (!series.length) return empty(container, opts.emptyText || '暂无曲线数据');

    var count = Math.max.apply(null, series.map(function (item) { return item.values.length; }));
    if (count < 2) return empty(container, opts.emptyText || '数据点不足');
    series.forEach(function (item) {
      while (item.values.length < count) item.values.push(null);
    });
    while (labels.length < count) labels.push('T' + (labels.length + 1));

    var width = opts.width || 640;
    var height = opts.height || 240;
    var margin = { top: 18, right: 22, bottom: 26, left: 44 };
    var plotW = Math.max(24, width - margin.left - margin.right);
    var plotH = Math.max(24, height - margin.top - margin.bottom);

    var allValues = [];
    series.forEach(function (item) {
      item.values.forEach(function (value) { if (value !== null) allValues.push(value); });
    });
    var rawMin = Math.min.apply(null, allValues);
    var rawMax = Math.max.apply(null, allValues);
    if (rawMin === rawMax) {
      rawMin -= Math.abs(rawMin) * 0.02 || 0.1;
      rawMax += Math.abs(rawMax) * 0.02 || 0.1;
    }
    var padding = (rawMax - rawMin) * 0.12;
    var min = rawMin - padding;
    var max = rawMax + padding;

    var yFormatter = typeof opts.yFormatter === 'function'
      ? opts.yFormatter
      : function (value) { return fmt(value, 2); };
    var xFormatter = typeof opts.xFormatter === 'function'
      ? opts.xFormatter
      : function (value) { return shortText(value, 7); };
    var xStep = plotW / (count - 1);
    var yOf = function (value) { return margin.top + plotH - ((value - min) / (max - min)) * plotH; };
    var xOf = function (index) { return margin.left + index * xStep; };

    var svg = root(container, width, height, opts.ariaLabel || '折线图');
    var defs = svgEl('defs');
    var areaId = nextId('area-grad');
    var areaGrad = svgEl('linearGradient', { id: areaId, x1: '0', y1: '0', x2: '0', y2: '1' });
    areaGrad.appendChild(svgEl('stop', { offset: '0%', 'stop-color': 'rgba(0, 113, 227, 0.16)' }));
    areaGrad.appendChild(svgEl('stop', { offset: '70%', 'stop-color': 'rgba(0, 113, 227, 0.04)' }));
    areaGrad.appendChild(svgEl('stop', { offset: '100%', 'stop-color': 'rgba(0, 113, 227, 0)' }));
    defs.appendChild(areaGrad);
    svg.appendChild(defs);

    var yTicks = 4;
    for (var tick = 0; tick <= yTicks; tick += 1) {
      var ratio = tick / yTicks;
      var y = margin.top + plotH * ratio;
      var value = max - (max - min) * ratio;
      svg.appendChild(svgEl('line', {
        x1: margin.left, y1: y, x2: width - margin.right, y2: y,
        class: 'chart-grid'
      }));
      svg.appendChild(textEl(margin.left - 7, y + 3, yFormatter(value), {
        'text-anchor': 'end', class: 'chart-mono'
      }));
    }
    svg.appendChild(svgEl('line', {
      x1: margin.left, y1: margin.top + plotH, x2: width - margin.right, y2: margin.top + plotH,
      class: 'chart-axis'
    }));

    var tickIndexes = count <= 5
      ? labels.map(function (_, index) { return index; })
      : [0, Math.round((count - 1) / 3), Math.round((count - 1) * 2 / 3), count - 1];
    tickIndexes.forEach(function (index, position) {
      var anchor = position === 0 ? 'start' : (position === tickIndexes.length - 1 ? 'end' : 'middle');
      svg.appendChild(textEl(xOf(index), height - 7, xFormatter(labels[index]), {
        'text-anchor': anchor, class: 'chart-label'
      }));
    });

    var primaryPoints = [];
    series.forEach(function (item, seriesIndex) {
      var points = [];
      item.values.forEach(function (value, index) {
        if (value !== null) points.push({ x: xOf(index), y: yOf(value), index: index, value: value });
      });
      if (!points.length) return;
      if (seriesIndex === 0 && opts.showArea !== false) {
        var first = points[0];
        var last = points[points.length - 1];
        var area = smoothPath(points) +
          ' L' + last.x.toFixed(2) + ',' + (margin.top + plotH).toFixed(2) +
          ' L' + first.x.toFixed(2) + ',' + (margin.top + plotH).toFixed(2) + ' Z';
        svg.appendChild(svgEl('path', { d: area, fill: 'url(#' + areaId + ')', stroke: 'none' }));
      }
      svg.appendChild(svgEl('path', {
        d: smoothPath(points),
        fill: 'none',
        stroke: item.color,
        'stroke-width': seriesIndex === 0 ? 1.9 : 1.25,
        'stroke-linecap': 'round',
        'stroke-linejoin': 'round',
        'stroke-dasharray': item.dashed ? '4 4' : null,
        opacity: seriesIndex === 0 ? 1 : 0.72
      }));
      if (seriesIndex === 0) primaryPoints = points;

      if (opts.showEnds !== false) {
        var endPoint = points[points.length - 1];
        svg.appendChild(svgEl('circle', {
          cx: endPoint.x, cy: endPoint.y, r: seriesIndex === 0 ? 3 : 2.4,
          fill: item.color, stroke: '#FFFFFF', 'stroke-width': 1.5
        }));
        var labelY = endPoint.y - 9;
        if (labelY < 12) labelY = endPoint.y + 15;
        svg.appendChild(textEl(endPoint.x - 3, labelY, yFormatter(endPoint.value), {
          'text-anchor': 'end',
          fill: seriesIndex === 0 ? '#0071E3' : '#8E8E93',
          class: 'chart-mono'
        }));
      }
    });

    if (!primaryPoints.length) return empty(container, opts.emptyText || '暂无曲线数据');

    var tip = tooltipFor(container);
    var hoverGroup = svgEl('g', { opacity: '0' });
    var hoverLine = svgEl('line', {
      x1: 0, y1: margin.top, x2: 0, y2: margin.top + plotH,
      stroke: 'rgba(0, 0, 0, 0.12)', 'stroke-width': 0.8, 'stroke-dasharray': '2 3'
    });
    hoverGroup.appendChild(hoverLine);
    var hoverDots = series.map(function (item) {
      var dot = svgEl('circle', {
        cx: 0, cy: 0, r: 3, fill: item.color, stroke: '#FFFFFF', 'stroke-width': 1.2
      });
      hoverGroup.appendChild(dot);
      return dot;
    });
    svg.appendChild(hoverGroup);

    var hit = svgEl('rect', {
      x: margin.left, y: margin.top, width: plotW, height: plotH,
      fill: 'transparent', class: 'chart-hit'
    });
    svg.appendChild(hit);
    hit.addEventListener('mousemove', function (event) {
      var rect = svg.getBoundingClientRect();
      var containerRect = container.getBoundingClientRect();
      var viewX = ((event.clientX - rect.left) / rect.width) * width;
      var index = clamp(Math.round((viewX - margin.left) / xStep), 0, count - 1);
      hoverGroup.setAttribute('opacity', '1');
      hoverLine.setAttribute('x1', xOf(index));
      hoverLine.setAttribute('x2', xOf(index));
      var rows = [];
      series.forEach(function (item, seriesIndex) {
        var pointValue = item.values[index];
        if (pointValue === null || pointValue === undefined) {
          hoverDots[seriesIndex].setAttribute('opacity', '0');
          return;
        }
        hoverDots[seriesIndex].setAttribute('opacity', '1');
        hoverDots[seriesIndex].setAttribute('cx', xOf(index));
        hoverDots[seriesIndex].setAttribute('cy', yOf(pointValue));
        rows.push({ label: item.name, value: yFormatter(pointValue), color: item.color });
      });
      var cssX = ((xOf(index) / width) * rect.width) + (rect.left - containerRect.left);
      fillTooltip(tip, String(labels[index]), rows);
      var tipWidth = tip.offsetWidth || 160;
      tip.style.left = clamp(cssX, tipWidth / 2 + 6, Math.max(tipWidth / 2 + 6, containerRect.width - tipWidth / 2 - 6)) + 'px';
    });
    hit.addEventListener('mouseleave', function () {
      hoverGroup.setAttribute('opacity', '0');
      tip.hidden = true;
    });

    return svg;
  }

  /* ------------------------------------------------------------- 环形 gauge */
  /**
   * data: { value, max, label }
   * options: { size, thickness, formatter(percent, value, max), ariaLabel }
   */
  function gauge(target, data, options) {
    var container = resolve(target);
    if (!container) return null;
    var opts = options || {};
    var input = typeof data === 'number' ? { value: data } : (data || {});
    var max = firstNum(input.max, opts.max, 100);
    if (max === null || max <= 0) max = 100;
    var value = clamp(firstNum(input.value, opts.value, 0), 0, max);
    var percent = (value / max) * 100;
    var size = opts.size || 64;
    var thickness = opts.thickness || 5;
    var radius = (size - thickness) / 2 - 3;
    var circumference = 2 * Math.PI * radius;
    var label = input.label || opts.label || '完整度';
    var display = typeof opts.formatter === 'function'
      ? opts.formatter(percent, value, max)
      : Math.round(percent) + '%';

    var svg = root(container, size, size, opts.ariaLabel || '环形进度图');
    var defs = svgEl('defs');
    var gradId = nextId('gauge-grad');
    var grad = svgEl('linearGradient', { id: gradId, x1: '0', y1: '0', x2: '1', y2: '1' });
    grad.appendChild(svgEl('stop', { offset: '0%', 'stop-color': '#0071E3' }));
    grad.appendChild(svgEl('stop', { offset: '100%', 'stop-color': '#0A84FF' }));
    defs.appendChild(grad);
    svg.appendChild(defs);

    var center = size / 2;
    svg.appendChild(svgEl('circle', {
      cx: center, cy: center, r: radius,
      fill: 'none', stroke: 'rgba(0, 0, 0, 0.10)', 'stroke-width': thickness
    }));
    var progress = svgEl('circle', {
      cx: center, cy: center, r: radius,
      fill: 'none', stroke: 'url(#' + gradId + ')', 'stroke-width': thickness,
      'stroke-linecap': 'round',
      'stroke-dasharray': circumference,
      'stroke-dashoffset': circumference,
      transform: 'rotate(-90 ' + center + ' ' + center + ')'
    });
    progress.style.transition = 'stroke-dashoffset 0.7s cubic-bezier(0.22, 1, 0.36, 1)';
    svg.appendChild(progress);

    var valueText = textEl(center, center - 1, display, {
      'text-anchor': 'middle',
      'dominant-baseline': 'middle',
      fill: '#F5F5F7',
      class: 'gauge-value'
    });
    valueText.style.fontSize = (size * 0.2) + 'px';
    svg.appendChild(valueText);
    var labelText = textEl(center, center + size * 0.2, label, {
      'text-anchor': 'middle',
      fill: '#8E8E93',
      class: 'gauge-label'
    });
    labelText.style.fontSize = Math.max(7, size * 0.105) + 'px';
    svg.appendChild(labelText);

    var title = svgEl('title');
    title.textContent = label + ' ' + Math.round(percent) + '%';
    svg.appendChild(title);

    global.requestAnimationFrame(function () {
      global.requestAnimationFrame(function () {
        progress.setAttribute('stroke-dashoffset', String(circumference * (1 - percent / 100)));
      });
    });
    return svg;
  }

  global.Charts = {
    bar: bar,
    line: line,
    gauge: gauge,
    palette: COLORS
  };
})(window);
