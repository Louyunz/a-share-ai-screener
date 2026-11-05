/* =========================================================================
 * 智投罗盘 · 纯 SVG 图表库（零依赖）
 * 暴露全局 Charts：bar / radar / line / scatter / gauge
 * 所有图表接收容器元素（或选择器）与数据对象，重绘并返回生成的 SVG。
 * ========================================================================= */
(function (global) {
  'use strict';

  var NS = 'http://www.w3.org/2000/svg';
  var uid = 0;
  var COLORS = {
    cyan: '#22D3EE',
    cyanDeep: '#0891B2',
    orange: '#F59E0B',
    green: '#34D399',
    red: '#FB7185',
    slate: '#64748B',
    text: '#A7B6CA',
    muted: '#6E829B',
    faint: '#4F637D',
    grid: 'rgba(109, 139, 174, 0.13)',
    axis: 'rgba(109, 139, 174, 0.22)'
  };

  /* ------------------------------------------------------------- 基础工具 */
  function resolve(target) {
    var node = typeof target === 'string' ? document.querySelector(target) : target;
    if (!node) throw new Error('Charts: 找不到图表容器');
    return node;
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
    node.textContent = content == null ? '' : String(content);
    return node;
  }

  function num(value, fallback) {
    var n = typeof value === 'number' ? value : parseFloat(value);
    return Number.isFinite(n) ? n : (fallback === undefined ? null : fallback);
  }

  function firstNum() {
    for (var i = 0; i < arguments.length; i += 1) {
      var n = num(arguments[i], null);
      if (n !== null) return n;
    }
    return null;
  }

  function pick() {
    for (var i = 0; i < arguments.length; i += 1) {
      if (arguments[i] !== null && arguments[i] !== undefined && arguments[i] !== '') return arguments[i];
    }
    return '';
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function fmt(value, digits) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—';
    return Number(value).toFixed(digits === undefined ? 2 : digits);
  }

  function shortText(value, max) {
    var chars = Array.from(String(value == null ? '' : value));
    return chars.length > max ? chars.slice(0, max - 1).join('') + '…' : chars.join('');
  }

  function nextId(prefix) {
    uid += 1;
    return prefix + '-' + uid;
  }

  function setup(container, ariaLabel) {
    container.innerHTML = '';
    container.classList.add('chart-anchor');
    var empty = null;
    return {
      empty: function (message) {
        empty = document.createElement('div');
        empty.className = 'chart-empty small';
        empty.textContent = message || '暂无图表数据';
        container.appendChild(empty);
        return empty;
      },
      svg: function (width, height, label) {
        var svg = svgEl('svg', {
          viewBox: '0 0 ' + width + ' ' + height,
          width: '100%',
          role: 'img',
          'aria-label': label || ariaLabel || '数据图表',
          preserveAspectRatio: 'xMidYMid meet',
          class: 'chart-svg'
        });
        container.appendChild(svg);
        return svg;
      }
    };
  }

  function makeTooltip(container) {
    var tip = document.createElement('div');
    tip.className = 'chart-tooltip';
    tip.hidden = true;
    container.appendChild(tip);
    return tip;
  }

  function showTooltip(container, tip, title, rows, left) {
    tip.textContent = '';
    var titleNode = document.createElement('div');
    titleNode.style.marginBottom = '3px';
    var titleStrong = document.createElement('b');
    titleStrong.textContent = title;
    titleNode.appendChild(titleStrong);
    tip.appendChild(titleNode);
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
      label.style.color = '#7E93AA';
      label.textContent = row.label;
      var value = document.createElement('span');
      value.style.color = row.color || '#D8E8F8';
      value.style.fontFamily = "'Cascadia Code', Consolas, monospace";
      value.style.marginLeft = 'auto';
      value.textContent = row.value;
      line.appendChild(label);
      line.appendChild(value);
      tip.appendChild(line);
    });
    var width = container.clientWidth || 240;
    tip.style.left = clamp(left, 72, Math.max(73, width - 72)) + 'px';
    tip.style.top = '9px';
    tip.style.transform = 'translate(-50%, 0)';
    tip.hidden = false;
  }

  function hideTooltip(tip) {
    if (tip) tip.hidden = true;
  }

  /* ------------------------------------------------------- 水平对比条 bar */
  /**
   * data: { items: [{ label, current, candidate }] }
   * options: { valueFormatter, currentLabel, candidateLabel, rowHeight, labelWidth,
   *            width, showValues, animate }
   */
  function bar(target, data, options) {
    var container = resolve(target);
    var opts = options || {};
    var input = data || {};
    var rawItems = Array.isArray(input) ? input : (input.items || input.rows || []);
    if (!rawItems.length) {
      setup(container).empty(opts.emptyText || '暂无因子数据');
      return null;
    }

    var items = rawItems.map(function (item, index) {
      return {
        label: pick(item.label, item.name, item.factor, item.key, '因子 ' + (index + 1)),
        current: firstNum(item.current, item.base, item.current_value, item.baseValue, item.value),
        candidate: firstNum(item.candidate, item.best, item.candidate_value, item.candidateValue, item.target),
        note: item.note || ''
      };
    });
    var hasCandidate = items.some(function (item) { return item.candidate !== null; });

    var formatter = typeof opts.valueFormatter === 'function'
      ? opts.valueFormatter
      : function (value) { return fmt(value, 2); };
    var width = opts.width || 560;
    var rowHeight = opts.rowHeight || (items.length > 8 ? 27 : 33);
    var showValues = opts.showValues !== false;
    var labelWidth = opts.labelWidth || Math.min(86, Math.max(62, width * 0.155));
    var valueWidth = showValues ? Math.min(96, Math.max(74, width * 0.16)) : 8;
    var top = 8;
    var height = opts.height || (rowHeight * items.length + top + 8);
    var plotX = labelWidth;
    var plotW = Math.max(60, width - labelWidth - valueWidth);
    var graph = setup(container, '因子权重对比图');
    var svg = graph.svg(width, height, opts.ariaLabel || '水平条形对比图');

    var allValues = [];
    items.forEach(function (item) {
      if (item.current !== null) allValues.push(item.current);
      if (item.candidate !== null) allValues.push(item.candidate);
    });
    var min = Math.min.apply(null, allValues.concat([0]));
    var max = Math.max.apply(null, allValues.concat([0]));
    if (max === min) max = min + 1;
    var hasNegative = min < 0;
    var domainMin = hasNegative ? min : 0;
    var domainMax = max > 0 ? max : 0.0001;
    var domainSpan = domainMax - domainMin || 1;
    var zeroX = plotX + ((0 - domainMin) / domainSpan) * plotW;

    var defs = svgEl('defs');
    var gradId = nextId('bar-grad');
    var grad = svgEl('linearGradient', { id: gradId, x1: '0', y1: '0', x2: '1', y2: '0' });
    grad.appendChild(svgEl('stop', { offset: '0%', 'stop-color': '#1CBBD6' }));
    grad.appendChild(svgEl('stop', { offset: '100%', 'stop-color': '#67EEFF' }));
    defs.appendChild(grad);
    svg.appendChild(defs);

    items.forEach(function (item, index) {
      var plotRight = plotX + plotW;
      var y = top + index * rowHeight;
      var centerY = y + rowHeight / 2;
      var group = svgEl('g', {});
      var label = textEl(plotX - 9, centerY + 3, shortText(item.label, Math.max(4, Math.floor(labelWidth / 9))), {
        'text-anchor': 'end',
        class: 'chart-label'
      });
      group.appendChild(label);

      var baseBarH = 4.5;
      var candBarH = 5.5;
      var single = !hasCandidate;
      var bars = [];
      if (!single) {
        bars.push({ key: 'current', value: item.current, y: centerY - 7, h: baseBarH, fill: '#5C718B', opacity: 0.65 });
        bars.push({ key: 'candidate', value: item.candidate, y: centerY + 2, h: candBarH, fill: 'url(#' + gradId + ')', opacity: 1 });
      } else {
        bars.push({
          key: 'current',
          value: item.current,
          y: centerY - 3.5,
          h: 7,
          fill: opts.singleAsCurrent ? '#5C718B' : 'url(#' + gradId + ')',
          opacity: opts.singleAsCurrent ? 0.62 : 1
        });
      }
      bars.forEach(function (barSpec) {
        var value = barSpec.value === null ? 0 : barSpec.value;
        var endX = plotX + ((value - domainMin) / domainSpan) * plotW;
        var x = Math.min(zeroX, endX);
        var barW = Math.abs(endX - zeroX);
        group.appendChild(svgEl('rect', {
          x: plotX, y: barSpec.y, width: plotW, height: barSpec.h, rx: barSpec.h / 2,
          fill: 'rgba(91, 116, 146, 0.085)'
        }));
        if (barW > 0.2) {
          var rect = svgEl('rect', {
            x: x,
            y: barSpec.y,
            width: Math.max(2, barW),
            height: barSpec.h,
            rx: barSpec.h / 2,
            fill: barSpec.fill,
            opacity: barSpec.opacity,
            class: opts.animate === false ? '' : 'chart-bar'
          });
          if (opts.animate !== false) rect.style.animationDelay = (index * 28) + 'ms';
          group.appendChild(rect);
        }
      });

      if (showValues) {
        var currentText = item.current === null ? '—' : formatter(item.current);
        var valueNode;
        if (!single) {
          valueNode = textEl(plotRight + valueWidth - 4, centerY + 3, '', { 'text-anchor': 'end', class: 'chart-mono' });
          valueNode.style.fontSize = '8.5px';
          var t1 = svgEl('tspan', { fill: '#6E829B' });
          t1.textContent = currentText;
          valueNode.appendChild(t1);
          var arrow = svgEl('tspan', { fill: '#4C6078', dx: '3', dy: '0' });
          arrow.textContent = '→';
          valueNode.appendChild(arrow);
          var t2 = svgEl('tspan', { fill: '#A9F2FC', dx: '3', dy: '0' });
          t2.textContent = item.candidate === null ? '—' : formatter(item.candidate);
          valueNode.appendChild(t2);
        } else {
          valueNode = textEl(plotRight + valueWidth - 4, centerY + 3, currentText, { 'text-anchor': 'end', class: 'chart-mono' });
          valueNode.style.fontSize = '8.5px';
        }
        group.appendChild(valueNode);
      }

      var title = svgEl('title');
      title.textContent = item.label + '：当前 ' + (item.current === null ? '—' : formatter(item.current)) +
        (hasCandidate ? '，候选 ' + (item.candidate === null ? '—' : formatter(item.candidate)) : '');
      group.appendChild(title);
      svg.appendChild(group);
    });

    return svg;
  }

  /* ------------------------------------------------------------- 雷达 radar */
  /**
   * data: { axes: [{ label, current, candidate }] }
   * options: { max, size, ariaLabel }
   */
  function radar(target, data, options) {
    var container = resolve(target);
    var opts = options || {};
    var input = data || {};
    var axes = input.axes || [];
    if (!axes.length && input.labels && input.series) {
      axes = input.labels.map(function (label, index) {
        return {
          label: label,
          current: input.series[0] ? input.series[0].values[index] : null,
          candidate: input.series[1] ? input.series[1].values[index] : null
        };
      });
    }
    axes = axes.map(function (axis, index) {
      return {
        label: pick(axis.label, axis.name, axis.factor, '因子 ' + (index + 1)),
        current: firstNum(axis.current, axis.base, axis.currentValue, axis.value),
        candidate: firstNum(axis.candidate, axis.best, axis.candidateValue, axis.target)
      };
    }).filter(function (axis) {
      return axis.current !== null || axis.candidate !== null;
    });
    if (axes.length < 3) {
      setup(container).empty(opts.emptyText || '等待权重数据');
      return null;
    }

    var hasCandidate = axes.some(function (axis) { return axis.candidate !== null; });
    var allValues = [];
    axes.forEach(function (axis) {
      if (axis.current !== null) allValues.push(Math.abs(axis.current));
      if (axis.candidate !== null) allValues.push(Math.abs(axis.candidate));
    });
    var dataMax = Math.max.apply(null, allValues.concat([0.0001]));
    var max = opts.max || Math.ceil(dataMax * 1.12 * 100) / 100;
    if (!Number.isFinite(max) || max <= 0) max = 1;

    var width = opts.width || 370;
    var height = opts.height || 258;
    var cx = width / 2;
    var cy = height / 2 + 2;
    var radius = opts.radius || Math.min(width, height) * 0.33;
    var graph = setup(container, '因子权重雷达图');
    var svg = graph.svg(width, height, opts.ariaLabel || '当前与候选因子权重雷达图');
    var levels = 4;

    function point(index, ratio) {
      var angle = (-Math.PI / 2) + (index / axes.length) * Math.PI * 2;
      return {
        x: cx + Math.cos(angle) * radius * ratio,
        y: cy + Math.sin(angle) * radius * ratio,
        angle: angle
      };
    }

    function polygon(values) {
      return values.map(function (value, index) {
        var p = point(index, clamp(Math.abs(value) / max, 0, 1));
        return p.x.toFixed(2) + ',' + p.y.toFixed(2);
      }).join(' ');
    }

    for (var level = 1; level <= levels; level += 1) {
      var gridValues = axes.map(function () { return max * (level / levels); });
      svg.appendChild(svgEl('polygon', {
        points: polygon(gridValues),
        fill: level === levels ? 'rgba(34, 211, 238, 0.018)' : 'none',
        stroke: 'rgba(109, 139, 174, 0.16)',
        'stroke-width': level === levels ? 1 : 0.7
      }));
    }
    axes.forEach(function (axis, index) {
      var end = point(index, 1);
      svg.appendChild(svgEl('line', {
        x1: cx, y1: cy, x2: end.x, y2: end.y,
        stroke: 'rgba(109, 139, 174, 0.17)', 'stroke-width': 0.8
      }));
    });
    svg.appendChild(svgEl('circle', { cx: cx, cy: cy, r: 1.5, fill: 'rgba(109, 139, 174, 0.4)' }));

    function addSeries(key, fill, stroke, dashed, dots) {
      var hasValues = axes.some(function (axis) { return axis[key] !== null; });
      if (!hasValues) return;
      var values = axes.map(function (axis) { return axis[key] === null ? 0 : axis[key]; });
      svg.appendChild(svgEl('polygon', {
        points: polygon(values),
        fill: fill,
        stroke: stroke,
        'stroke-width': key === 'candidate' ? 1.6 : 1.1,
        'stroke-dasharray': dashed ? '3 3' : null,
        'stroke-linejoin': 'round'
      }));
      if (dots) {
        axes.forEach(function (axis, index) {
          if (axis[key] === null) return;
          var p = point(index, clamp(Math.abs(axis[key]) / max, 0, 1));
          var circle = svgEl('circle', {
            cx: p.x, cy: p.y, r: key === 'candidate' ? 2.5 : 1.8,
            fill: key === 'candidate' ? COLORS.cyan : '#71839B',
            stroke: '#152135', 'stroke-width': 1
          });
          var title = svgEl('title');
          title.textContent = axis.label + '：' + fmt(axis[key], 3);
          circle.appendChild(title);
          svg.appendChild(circle);
        });
      }
    }

    addSeries('current', 'rgba(100, 116, 139, 0.07)', 'rgba(157, 176, 198, 0.55)', true, false);
    if (hasCandidate) addSeries('candidate', 'rgba(34, 211, 238, 0.105)', COLORS.cyan, false, true);

    axes.forEach(function (axis, index) {
      var p = point(index, 1.17);
      var cos = Math.cos(p.angle);
      var anchor = cos > 0.22 ? 'start' : (cos < -0.22 ? 'end' : 'middle');
      var display = shortText(axis.label, 8);
      var label = textEl(p.x, p.y + 3, display, {
        'text-anchor': anchor,
        class: 'chart-label',
        'dominant-baseline': 'middle'
      });
      var title = svgEl('title');
      title.textContent = axis.label + '：当前 ' + (axis.current === null ? '—' : fmt(axis.current, 3)) +
        (hasCandidate ? '，候选 ' + (axis.candidate === null ? '—' : fmt(axis.candidate, 3)) : '');
      label.appendChild(title);
      svg.appendChild(label);
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
      d += ' C' + cp1x.toFixed(2) + ',' + cp1y.toFixed(2) + ' ' + cp2x.toFixed(2) + ',' + cp2y.toFixed(2) + ' ' + p2.x.toFixed(2) + ',' + p2.y.toFixed(2);
    }
    return d;
  }

  /**
   * data: { labels: [...], series: [{ name, values, color, dashed, fill }] }
   * options: { height, yFormatter, xFormatter, ariaLabel, showArea, showEnds }
   */
  function line(target, data, options) {
    var container = resolve(target);
    var opts = options || {};
    var input = data || {};
    var labels = input.labels || input.dates || input.x || [];
    var rawSeries = input.series || input.lines || [];
    if (!rawSeries.length && input.values) {
      rawSeries = [{ name: input.name || '序列', values: input.values }];
    }
    var series = rawSeries.map(function (item, index) {
      return {
        name: pick(item.name, item.label, '序列 ' + (index + 1)),
        values: (item.values || item.data || []).map(function (value) { return num(value, null); }),
        color: item.color || (index === 0 ? COLORS.cyan : (index === 1 ? COLORS.slate : COLORS.orange)),
        dashed: !!item.dashed,
        fill: item.fill
      };
    }).filter(function (item) { return item.values.some(function (value) { return value !== null; }); });

    if (!series.length) {
      setup(container).empty(opts.emptyText || '暂无净值数据');
      return null;
    }

    var count = Math.max.apply(null, series.map(function (item) { return item.values.length; }));
    if (count < 2) {
      setup(container).empty(opts.emptyText || '数据点不足');
      return null;
    }
    series.forEach(function (item) {
      while (item.values.length < count) item.values.push(null);
    });
    while (labels.length < count) labels.push('T' + (labels.length + 1));

    var width = opts.width || 620;
    var height = opts.height || 250;
    var margin = { top: 20, right: 26, bottom: 28, left: 45 };
    var plotW = Math.max(20, width - margin.left - margin.right);
    var plotH = Math.max(20, height - margin.top - margin.bottom);
    var allValues = [];
    series.forEach(function (item) {
      item.values.forEach(function (value) { if (value !== null) allValues.push(value); });
    });
    var rawMin = Math.min.apply(null, allValues);
    var rawMax = Math.max.apply(null, allValues);
    if (opts.minY !== undefined) rawMin = Math.min(rawMin, opts.minY);
    if (rawMin === rawMax) {
      rawMin -= Math.abs(rawMin) * 0.02 || 0.1;
      rawMax += Math.abs(rawMax) * 0.02 || 0.1;
    }
    var span = rawMax - rawMin;
    var min = opts.yMin !== undefined ? opts.yMin : rawMin - span * 0.12;
    var max = opts.yMax !== undefined ? opts.yMax : rawMax + span * 0.12;
    if (min === max) { min -= 1; max += 1; }

    var formatter = typeof opts.yFormatter === 'function'
      ? opts.yFormatter
      : function (value) { return fmt(value, 2); };
    var xFormatter = typeof opts.xFormatter === 'function'
      ? opts.xFormatter
      : function (value) { return shortText(value, 7); };
    var xStep = plotW / (count - 1);
    var yOf = function (value) { return margin.top + plotH - ((value - min) / (max - min)) * plotH; };
    var xOf = function (index) { return margin.left + index * xStep; };

    var graph = setup(container, '净值对比折线图');
    var svg = graph.svg(width, height, opts.ariaLabel || '净值对比折线图');
    var defs = svgEl('defs');
    var areaId = nextId('area-grad');
    var areaGrad = svgEl('linearGradient', { id: areaId, x1: '0', y1: '0', x2: '0', y2: '1' });
    areaGrad.appendChild(svgEl('stop', { offset: '0%', 'stop-color': 'rgba(34, 211, 238, 0.24)' }));
    areaGrad.appendChild(svgEl('stop', { offset: '65%', 'stop-color': 'rgba(34, 211, 238, 0.075)' }));
    areaGrad.appendChild(svgEl('stop', { offset: '100%', 'stop-color': 'rgba(34, 211, 238, 0)' }));
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
      svg.appendChild(textEl(margin.left - 7, y + 3, formatter(value), {
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
    tickIndexes.forEach(function (index, i) {
      var anchor = i === 0 ? 'start' : (i === tickIndexes.length - 1 ? 'end' : 'middle');
      svg.appendChild(textEl(xOf(index), height - 7, xFormatter(labels[index]), {
        'text-anchor': anchor, class: 'chart-label'
      }));
    });

    var primaryPoints = [];
    series.forEach(function (item, sIndex) {
      var points = [];
      item.values.forEach(function (value, index) {
        if (value !== null) points.push({ x: xOf(index), y: yOf(value), index: index, value: value });
      });
      if (!points.length) return;
      if (sIndex === 0 && points.length && opts.showArea !== false && item.fill !== false) {
        var first = points[0];
        var last = points[points.length - 1];
        var area = smoothPath(points) +
          ' L' + last.x.toFixed(2) + ',' + (margin.top + plotH).toFixed(2) +
          ' L' + first.x.toFixed(2) + ',' + (margin.top + plotH).toFixed(2) + ' Z';
        svg.appendChild(svgEl('path', { d: area, fill: 'url(#' + areaId + ')', stroke: 'none' }));
      }
      var path = svgEl('path', {
        d: smoothPath(points),
        fill: 'none',
        stroke: item.color,
        'stroke-width': sIndex === 0 ? 1.9 : 1.25,
        'stroke-linecap': 'round',
        'stroke-linejoin': 'round',
        'stroke-dasharray': item.dashed ? '4 4' : null,
        opacity: sIndex === 0 ? 1 : 0.72
      });
      svg.appendChild(path);
      if (sIndex === 0) primaryPoints = points;

      var lastPoint = points[points.length - 1];
      if (opts.showEnds !== false) {
        var endGroup = svgEl('g', {});
        endGroup.appendChild(svgEl('circle', {
          cx: lastPoint.x, cy: lastPoint.y, r: sIndex === 0 ? 3 : 2.4,
          fill: item.color, stroke: '#101B2D', 'stroke-width': 1.5
        }));
        var labelY = lastPoint.y - 9;
        if (labelY < 11) labelY = lastPoint.y + 15;
        var endLabel = textEl(lastPoint.x - 3, labelY, formatter(lastPoint.value), {
          'text-anchor': 'end',
          fill: sIndex === 0 ? '#A9F2FC' : '#8698AE',
          class: 'chart-mono'
        });
        endGroup.appendChild(endLabel);
        svg.appendChild(endGroup);
      }
    });

    if (!primaryPoints.length) {
      container.innerHTML = '';
      setup(container).empty(opts.emptyText || '暂无净值数据');
      return null;
    }

    var tooltip = makeTooltip(container);
    var hoverGroup = svgEl('g', { opacity: '0' });
    var hoverLine = svgEl('line', {
      x1: 0, y1: margin.top, x2: 0, y2: margin.top + plotH,
      stroke: 'rgba(160, 196, 225, 0.35)', 'stroke-width': 0.8, 'stroke-dasharray': '2 3'
    });
    hoverGroup.appendChild(hoverLine);
    var hoverDots = series.map(function (item) {
      var dot = svgEl('circle', {
        cx: 0, cy: 0, r: 3, fill: item.color, stroke: '#101B2D', 'stroke-width': 1.2
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
      var viewX = ((event.clientX - rect.left) / rect.width) * width;
      var index = clamp(Math.round((viewX - margin.left) / xStep), 0, count - 1);
      hoverGroup.setAttribute('opacity', '1');
      hoverLine.setAttribute('x1', xOf(index));
      hoverLine.setAttribute('x2', xOf(index));
      var rows = [];
      series.forEach(function (item, sIndex) {
        var value = item.values[index];
        if (value === null) {
          hoverDots[sIndex].setAttribute('opacity', '0');
          return;
        }
        hoverDots[sIndex].setAttribute('opacity', '1');
        hoverDots[sIndex].setAttribute('cx', xOf(index));
        hoverDots[sIndex].setAttribute('cy', yOf(value));
        rows.push({ label: item.name, value: formatter(value), color: item.color });
      });
      var cssX = ((xOf(index) / width) * rect.width) + (rect.left - container.getBoundingClientRect().left);
      showTooltip(container, tooltip, String(labels[index]), rows, cssX);
    });
    hit.addEventListener('mouseleave', function () {
      hoverGroup.setAttribute('opacity', '0');
      hideTooltip(tooltip);
    });

    return svg;
  }

  /* ------------------------------------------------------------- 散点 scatter */
  function hexToRgb(hex) {
    var raw = hex.replace('#', '');
    return {
      r: parseInt(raw.slice(0, 2), 16),
      g: parseInt(raw.slice(2, 4), 16),
      b: parseInt(raw.slice(4, 6), 16)
    };
  }

  function mixColor(a, b, t) {
    var from = hexToRgb(a);
    var to = hexToRgb(b);
    var r = Math.round(from.r + (to.r - from.r) * t);
    var g = Math.round(from.g + (to.g - from.g) * t);
    var bl = Math.round(from.b + (to.b - from.b) * t);
    return 'rgb(' + r + ',' + g + ',' + bl + ')';
  }

  /**
   * data: { points: [{ x, y, value, label, isBest }] }
   * options: { xLabel, yLabel, highlightIndex, valueFormatter, width, height }
   */
  function scatter(target, data, options) {
    var container = resolve(target);
    var opts = options || {};
    var input = data || {};
    var rawPoints = input.points || input.items || [];
    var points = rawPoints.map(function (point, index) {
      return {
        x: firstNum(point.x, point.sharpe, point.sharpe_ratio, point.sharpeRatio, point.return, point.cagr),
        y: firstNum(point.y, point.objective, point.score, point.target, point.value),
        value: firstNum(point.value, point.objective, point.score, point.y, point.target),
        label: pick(point.label, point.name, point.trial_index, '候选 ' + (index + 1)),
        isBest: !!point.isBest || !!point.is_best
      };
    }).filter(function (point) {
      return point.x !== null && point.y !== null;
    });

    if (!points.length) {
      setup(container).empty(opts.emptyText || '等待候选参数');
      return null;
    }

    var width = opts.width || 390;
    var height = opts.height || 246;
    var margin = { top: 22, right: 14, bottom: 30, left: 33 };
    var plotW = width - margin.left - margin.right;
    var plotH = height - margin.top - margin.bottom;
    var xs = points.map(function (point) { return point.x; });
    var ys = points.map(function (point) { return point.y; });
    var xMin = Math.min.apply(null, xs);
    var xMax = Math.max.apply(null, xs);
    var yMin = Math.min.apply(null, ys);
    var yMax = Math.max.apply(null, ys);
    if (xMin === xMax) { xMin -= 0.1; xMax += 0.1; }
    if (yMin === yMax) { yMin -= 0.1; yMax += 0.1; }
    var xPad = (xMax - xMin) * 0.12;
    var yPad = (yMax - yMin) * 0.16;
    xMin -= xPad; xMax += xPad; yMin -= yPad; yMax += yPad;

    var values = points.map(function (point) { return point.value === null ? point.y : point.value; });
    var valueMin = Math.min.apply(null, values);
    var valueMax = Math.max.apply(null, values);
    var highlightIndex = opts.highlightIndex;
    if (highlightIndex === undefined || highlightIndex === null) {
      highlightIndex = points.findIndex(function (point) { return point.isBest; });
      if (highlightIndex < 0) {
        var best = -Infinity;
        points.forEach(function (point, index) {
          var v = point.value === null ? point.y : point.value;
          if (v > best) { best = v; highlightIndex = index; }
        });
      }
    }

    var formatter = typeof opts.valueFormatter === 'function'
      ? opts.valueFormatter
      : function (value) { return fmt(value, 3); };
    var xOf = function (value) { return margin.left + ((value - xMin) / (xMax - xMin)) * plotW; };
    var yOf = function (value) { return margin.top + plotH - ((value - yMin) / (yMax - yMin)) * plotH; };

    var graph = setup(container, '候选参数散点图');
    var svg = graph.svg(width, height, opts.ariaLabel || '候选参数散点图');

    for (var gx = 0; gx <= 4; gx += 1) {
      var xLine = margin.left + (plotW / 4) * gx;
      svg.appendChild(svgEl('line', {
        x1: xLine, y1: margin.top, x2: xLine, y2: margin.top + plotH,
        class: 'chart-grid'
      }));
    }
    for (var gy = 0; gy <= 3; gy += 1) {
      var yLine = margin.top + (plotH / 3) * gy;
      svg.appendChild(svgEl('line', {
        x1: margin.left, y1: yLine, x2: margin.left + plotW, y2: yLine,
        class: 'chart-grid'
      }));
    }
    svg.appendChild(svgEl('line', {
      x1: margin.left, y1: margin.top + plotH, x2: margin.left + plotW, y2: margin.top + plotH,
      class: 'chart-axis'
    }));
    svg.appendChild(svgEl('line', {
      x1: margin.left, y1: margin.top, x2: margin.left, y2: margin.top + plotH,
      class: 'chart-axis'
    }));

    var xTicks = 3;
    for (var xi = 0; xi <= xTicks; xi += 1) {
      var xv = xMin + ((xMax - xMin) / xTicks) * xi;
      svg.appendChild(textEl(xOf(xv), height - 9, fmt(xv, 1), {
        'text-anchor': xi === 0 ? 'start' : (xi === xTicks ? 'end' : 'middle'), class: 'chart-mono'
      }));
    }
    var yTicks = 3;
    for (var yi = 0; yi <= yTicks; yi += 1) {
      var yv = yMax - ((yMax - yMin) / yTicks) * yi;
      svg.appendChild(textEl(margin.left - 6, yOf(yv) + 3, fmt(yv, 2), {
        'text-anchor': 'end', class: 'chart-mono'
      }));
    }
    svg.appendChild(textEl(margin.left, 11, opts.yLabel || '目标函数 J', {
      class: 'chart-footnote', 'text-anchor': 'start'
    }));
    svg.appendChild(textEl(width - margin.right, height - 20, opts.xLabel || 'Sharpe', {
      class: 'chart-footnote', 'text-anchor': 'end'
    }));

    points.forEach(function (point, index) {
      var value = point.value === null ? point.y : point.value;
      var ratio = valueMax === valueMin ? 1 : clamp((value - valueMin) / (valueMax - valueMin), 0, 1);
      var color = mixColor('#3C5878', '#F59E0B', ratio);
      var isHighlight = index === highlightIndex;
      var cx = xOf(point.x);
      var cy = yOf(point.y);
      var group = svgEl('g', {});
      if (isHighlight) {
        group.appendChild(svgEl('circle', {
          cx: cx, cy: cy, r: 8,
          fill: 'none', stroke: 'rgba(34, 211, 238, 0.4)', 'stroke-width': 1,
          class: 'pulse-ring'
        }));
        group.appendChild(svgEl('circle', {
          cx: cx, cy: cy, r: 5.2,
          fill: 'rgba(34, 211, 238, 0.12)', stroke: COLORS.cyan, 'stroke-width': 1.2
        }));
      } else {
        group.appendChild(svgEl('circle', {
          cx: cx, cy: cy, r: 3.1,
          fill: color, opacity: 0.65 + ratio * 0.3,
          stroke: 'rgba(11, 19, 32, 0.5)', 'stroke-width': 0.7
        }));
      }
      var title = svgEl('title');
      title.textContent = '候选 #' + point.label + ' · Sharpe ' + fmt(point.x, 3) +
        ' · 目标 ' + fmt(point.y, 3);
      group.appendChild(title);
      svg.appendChild(group);
    });

    // 右上角色值图例
    var legendGradId = nextId('scatter-grad');
    var defs = svgEl('defs');
    var legendGrad = svgEl('linearGradient', { id: legendGradId, x1: '0', y1: '0', x2: '1', y2: '0' });
    legendGrad.appendChild(svgEl('stop', { offset: '0%', 'stop-color': '#3C5878' }));
    legendGrad.appendChild(svgEl('stop', { offset: '100%', 'stop-color': '#F59E0B' }));
    defs.appendChild(legendGrad);
    svg.insertBefore(defs, svg.firstChild);
    svg.appendChild(textEl(width - margin.right - 54, 11, '目标值', {
      class: 'chart-footnote', 'text-anchor': 'start'
    }));
    svg.appendChild(svgEl('rect', {
      x: width - margin.right - 22, y: 7, width: 22, height: 3, rx: 1.5,
      fill: 'url(#' + legendGradId + ')'
    }));

    // 高亮最优候选的文字标注（避免左边缘溢出）
    var bestPoint = points[highlightIndex];
    if (bestPoint) {
      var bestX = xOf(bestPoint.x);
      var bestY = yOf(bestPoint.y);
      var anchor = bestX > width - 80 ? 'end' : 'start';
      svg.appendChild(textEl(bestX + (anchor === 'end' ? -9 : 9), bestY - 7, '最优候选', {
        'text-anchor': anchor, fill: '#8FEFFC', class: 'chart-mono'
      }));
    }

    return svg;
  }

  /* ------------------------------------------------------------- 环形 gauge */
  /**
   * data: { value, max, label }
   * options: { size, thickness, ariaLabel, formatter }
   */
  function gauge(target, data, options) {
    var container = resolve(target);
    var opts = options || {};
    var input = (typeof data === 'number') ? { value: data } : (data || {});
    var max = num(input.max, num(opts.max, 100)) || 100;
    var value = clamp(num(input.value, 0) || 0, 0, max);
    var percent = max > 0 ? (value / max) * 100 : 0;
    var size = opts.size || 62;
    var thickness = opts.thickness || 4;
    var radius = (size - thickness) / 2 - 3;
    var circumference = 2 * Math.PI * radius;
    var displayValue = typeof opts.formatter === 'function'
      ? opts.formatter(percent, value, max)
      : Math.round(percent) + '%';
    var stateRef = container.__chartGauge;
    if (stateRef && stateRef.svg && container.contains(stateRef.svg) && stateRef.size === size) {
      stateRef.progress.setAttribute('stroke-dashoffset', String(circumference * (1 - percent / 100)));
      stateRef.valueText.textContent = displayValue;
      stateRef.label.textContent = input.label || opts.label || '进度';
      stateRef.title.textContent = (input.label || '进度') + ' ' + Math.round(percent) + '%';
      return stateRef.svg;
    }
    var graph = setup(container, '进度环形图');
    var svg = graph.svg(size, size, opts.ariaLabel || '调参进度');
    var defs = svgEl('defs');
    var gradId = nextId('gauge-grad');
    var grad = svgEl('linearGradient', { id: gradId, x1: '0', y1: '0', x2: '1', y2: '1' });
    grad.appendChild(svgEl('stop', { offset: '0%', 'stop-color': '#22D3EE' }));
    grad.appendChild(svgEl('stop', { offset: '100%', 'stop-color': '#67E8F9' }));
    defs.appendChild(grad);
    svg.appendChild(defs);

    var center = size / 2;
    svg.appendChild(svgEl('circle', {
      cx: center, cy: center, r: radius,
      fill: 'none', stroke: 'rgba(79, 106, 137, 0.28)', 'stroke-width': thickness
    }));
    var progress = svgEl('circle', {
      cx: center, cy: center, r: radius,
      fill: 'none', stroke: 'url(#' + gradId + ')', 'stroke-width': thickness,
      'stroke-linecap': 'round',
      'stroke-dasharray': circumference,
      'stroke-dashoffset': circumference,
      transform: 'rotate(-90 ' + center + ' ' + center + ')'
    });
    progress.style.transition = 'stroke-dashoffset 0.65s cubic-bezier(0.22, 1, 0.36, 1)';
    svg.appendChild(progress);

    var valueText = textEl(center, center + 1, displayValue, {
      'text-anchor': 'middle', 'dominant-baseline': 'middle',
      fill: '#ECFBFF', class: 'gauge-value'
    });
    valueText.style.fontFamily = "'Cascadia Code', Consolas, monospace";
    valueText.style.fontSize = (size * 0.205) + 'px';
    valueText.style.fontWeight = '560';
    svg.appendChild(valueText);
    var label = textEl(center, center + size * 0.22, input.label || opts.label || '进度', {
      'text-anchor': 'middle', fill: '#7288A0'
    });
    label.style.fontSize = Math.max(6.5, size * 0.11) + 'px';
    label.style.fontFamily = "'Cascadia Code', 'PingFang SC', sans-serif";
    svg.appendChild(label);
    var title = svgEl('title');
    title.textContent = (input.label || '进度') + ' ' + Math.round(percent) + '%';
    svg.appendChild(title);
    container.__chartGauge = {
      svg: svg,
      progress: progress,
      valueText: valueText,
      label: label,
      title: title,
      size: size
    };

    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        progress.setAttribute('stroke-dashoffset', String(circumference * (1 - percent / 100)));
      });
    });
    return svg;
  }

  global.Charts = {
    bar: bar,
    radar: radar,
    line: line,
    scatter: scatter,
    gauge: gauge,
    palette: COLORS
  };
})(window);
