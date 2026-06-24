const TRACE_URL = './assets/traces/demo_case5.json.gz';
const TRACE_TITLE = 'OpGuard — demo_case5 (DeepSpeed)';
const LOSS_RUNS = [
  {
    key: 'baseline',
    label: 'baseline (correct)',
    url: './assets/data/run_20251001_12292e6b.csv',
  },
  {
    key: 'buggy',
    label: 'buggy',
    url: './assets/data/run_20251001_9e609e46.csv',
  },
];
const LOSS_STEP_MIN = 3050;
const LOSS_STEP_MAX = 3450;
const LOSS_NOTE_STEPS = [3080, 3081];

const iframe = document.getElementById('perfetto');
const statusEl = document.getElementById('trace-status');
const overlayEl = document.getElementById('trace-overlay');
const progressEl = document.getElementById('trace-progress');
const lossTooltip = document.getElementById('loss-tooltip');
let lossData = [];

function setStatus(text, state) {
  const label = statusEl.querySelector('span:last-child');
  if (label) label.textContent = text;
  statusEl.className = 'trace-panel__status';
  if (state) statusEl.classList.add(`trace-panel__status--${state}`);
}

function setProgress(pct) {
  progressEl.style.width = `${Math.min(100, pct)}%`;
}

function waitForPerfettoReady() {
  return new Promise((resolve) => {
    const interval = setInterval(() => {
      iframe.contentWindow.postMessage('PING', '*');
    }, 100);

    function onMessage(evt) {
      if (evt.source === iframe.contentWindow && evt.data === 'PONG') {
        clearInterval(interval);
        window.removeEventListener('message', onMessage);
        resolve();
      }
    }

    window.addEventListener('message', onMessage);
  });
}

async function decompressGzip(arrayBuffer) {
  const ds = new DecompressionStream('gzip');
  const blob = new Blob([arrayBuffer]);
  const decompressed = blob.stream().pipeThrough(ds);
  return new Response(decompressed).arrayBuffer();
}

async function loadTrace() {
  try {
    setStatus('Fetching trace file…');
    setProgress(10);

    const response = await fetch(TRACE_URL);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const total = Number(response.headers.get('content-length')) || 0;
    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      if (total) setProgress(10 + (received / total) * 40);
    }

    const compressed = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      compressed.set(chunk, offset);
      offset += chunk.length;
    }

    setStatus('Decompressing trace…');
    setProgress(55);
    const buffer = await decompressGzip(compressed.buffer);

    setStatus('Connecting to Perfetto UI…');
    setProgress(75);
    await waitForPerfettoReady();

    setStatus('Rendering trace…');
    setProgress(90);
    iframe.contentWindow.postMessage(
      {
        perfetto: {
          buffer,
          title: TRACE_TITLE,
          fileName: 'demo_case5.json',
        },
      },
      '*',
    );

    setProgress(100);
    setStatus('Trace loaded', 'ready');
    overlayEl.classList.add('trace-panel__overlay--hidden');
  } catch (err) {
    console.error('Failed to load trace:', err);
    setStatus(`Failed to load trace: ${err.message}`, 'error');
    overlayEl.querySelector('p').textContent =
      'Could not load trace. Serve this page over HTTP (not file://).';
  }
}

function parseRunCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines.shift().split(',').map((header) => header.trim());
  const stepIndex = headers.findIndex((header) => header.toLowerCase().includes('step'));
  const lossIndex = headers.findIndex((header) => header.toLowerCase().includes('loss'));

  if (stepIndex === -1 || lossIndex === -1) {
    throw new Error('CSV is missing step or loss columns');
  }

  const byStep = new Map();
  for (const line of lines) {
    if (!line) continue;
    const cols = line.split(',');
    const step = Number(cols[stepIndex]);
    const loss = Number(cols[lossIndex]);
    if (!Number.isFinite(step) || !Number.isFinite(loss)) continue;
    if (step < LOSS_STEP_MIN || step > LOSS_STEP_MAX) continue;
    byStep.set(step, loss);
  }

  return byStep;
}

async function loadLossData() {
  const [baselineText, buggyText] = await Promise.all(
    LOSS_RUNS.map((run) => fetch(run.url).then((response) => {
      if (!response.ok) throw new Error(`Failed to load ${run.url}: ${response.status}`);
      return response.text();
    })),
  );

  const baselineByStep = parseRunCsv(baselineText);
  const buggyByStep = parseRunCsv(buggyText);
  return [...baselineByStep.keys()]
    .filter((step) => buggyByStep.has(step))
    .sort((a, b) => a - b)
    .map((step) => {
      const baseline = baselineByStep.get(step);
      const buggy = buggyByStep.get(step);
      return {
        step,
        baseline,
        buggy,
        delta: buggy - baseline,
      };
    });
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function chartTheme() {
  return {
    bg: cssVar('--bg-surface-alt'),
    border: cssVar('--border'),
    text: cssVar('--text-primary'),
    muted: cssVar('--text-muted'),
    secondary: cssVar('--text-secondary'),
    surface: cssVar('--bg-surface'),
    grid: cssVar('--border'),
    baseline: '#2ca02c',
    buggy: '#d62728',
    delta: '#d62728',
    zero: cssVar('--text-primary'),
  };
}

function fmtLoss(value) {
  return value.toFixed(6);
}

function fmtDelta(value) {
  if (Math.abs(value) < 1e-12) return '0';
  if (Math.abs(value) < 1e-4) return value.toExponential(5);
  return value.toFixed(6);
}

function pathFor(data, xScale, yScale, key) {
  return data.map((point, index) => {
    const cmd = index === 0 ? 'M' : 'L';
    return `${cmd}${xScale(point.step).toFixed(2)},${yScale(point[key]).toFixed(2)}`;
  }).join(' ');
}

function buildTicks(min, max, count) {
  if (count <= 1) return [min];
  const step = (max - min) / (count - 1);
  return Array.from({ length: count }, (_, index) => min + step * index);
}

function chartSize(svg, fallbackHeight) {
  const rect = svg.getBoundingClientRect();
  return {
    width: Math.max(320, rect.width || 800),
    height: Math.max(120, rect.height || fallbackHeight),
  };
}

function showLossTooltip(event, point) {
  lossTooltip.hidden = false;
  lossTooltip.innerHTML = `
    <div class="loss-tooltip__step">Step ${point.step}</div>
    <div class="loss-tooltip__row">
      <span class="loss-tooltip__label">baseline</span>
      <span class="loss-tooltip__value">${fmtLoss(point.baseline)}</span>
    </div>
    <div class="loss-tooltip__row">
      <span class="loss-tooltip__label">buggy</span>
      <span class="loss-tooltip__value">${fmtLoss(point.buggy)}</span>
    </div>
    <div class="loss-tooltip__row">
      <span class="loss-tooltip__label">Δ buggy-baseline</span>
      <span class="loss-tooltip__value">${fmtDelta(point.delta)}</span>
    </div>
  `;

  const padding = 18;
  const maxLeft = window.innerWidth - lossTooltip.offsetWidth - padding;
  const maxTop = window.innerHeight - lossTooltip.offsetHeight - padding;
  lossTooltip.style.left = `${Math.min(event.clientX + 16, maxLeft)}px`;
  lossTooltip.style.top = `${Math.min(event.clientY + 16, maxTop)}px`;
}

function hideLossTooltip() {
  lossTooltip.hidden = true;
}

function installHover(svg, data, scales, mode) {
  const hoverGroup = svg.querySelector('.chart-hover');
  const dataByStep = new Map(data.map((point) => [point.step, point]));

  svg.onpointermove = (event) => {
    const rect = svg.getBoundingClientRect();
    const px = event.clientX - rect.left;
    const step = Math.round(scales.xInvert(px));
    const point = dataByStep.get(Math.max(LOSS_STEP_MIN, Math.min(LOSS_STEP_MAX, step)));
    if (!point) return;

    const x = scales.x(point.step);
    const markerY = mode === 'loss'
      ? [scales.y(point.buggy), scales.y(point.baseline)]
      : [scales.y(point.delta)];

    hoverGroup.innerHTML = `
      <line x1="${x}" y1="${scales.plotTop}" x2="${x}" y2="${scales.plotBottom}" stroke="${scales.theme.secondary}" stroke-width="1.2" stroke-dasharray="4 4"/>
      ${markerY.map((y, index) => `
        <circle cx="${x}" cy="${y}" r="${mode === 'loss' ? 4.5 : 5}" fill="${mode === 'loss' ? (index === 0 ? scales.theme.buggy : scales.theme.baseline) : scales.theme.delta}" stroke="${scales.theme.surface}" stroke-width="2"/>
      `).join('')}
    `;
    showLossTooltip(event, point);
  };

  svg.onpointerleave = () => {
    hoverGroup.innerHTML = '';
    hideLossTooltip();
  };
}

function drawLossChart(svg, data) {
  const theme = chartTheme();
  const { width, height } = chartSize(svg, 280);
  const margin = { top: 26, right: 22, bottom: 36, left: 62 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const allLoss = data.flatMap((point) => [point.baseline, point.buggy]);
  const minY = Math.min(...allLoss);
  const maxY = Math.max(...allLoss);
  const padY = Math.max((maxY - minY) * 0.12, 0.001);
  const yMin = minY - padY;
  const yMax = maxY + padY;
  const xScale = (step) => margin.left + ((step - LOSS_STEP_MIN) / (LOSS_STEP_MAX - LOSS_STEP_MIN)) * plotWidth;
  const yScale = (value) => margin.top + ((yMax - value) / (yMax - yMin)) * plotHeight;
  const xInvert = (px) => LOSS_STEP_MIN + ((px - margin.left) / plotWidth) * (LOSS_STEP_MAX - LOSS_STEP_MIN);
  const yTicks = buildTicks(yMin, yMax, 5);
  const xTicks = [3050, 3150, 3250, 3350, 3450];

  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.innerHTML = `
    <rect width="${width}" height="${height}" rx="14" fill="transparent"/>
    <text x="${margin.left}" y="18" fill="${theme.text}" font-size="14" font-weight="700">Training Loss</text>
    ${yTicks.map((tick) => `
      <line x1="${margin.left}" y1="${yScale(tick)}" x2="${width - margin.right}" y2="${yScale(tick)}" stroke="${theme.grid}" stroke-width="1" stroke-dasharray="3 4"/>
      <text x="${margin.left - 10}" y="${yScale(tick) + 4}" text-anchor="end" fill="${theme.muted}" font-size="11">${tick.toFixed(2)}</text>
    `).join('')}
    ${xTicks.map((tick) => `
      <line x1="${xScale(tick)}" y1="${margin.top}" x2="${xScale(tick)}" y2="${height - margin.bottom}" stroke="${theme.grid}" stroke-width="0.8" opacity="0.5"/>
      <text x="${xScale(tick)}" y="${height - 12}" text-anchor="middle" fill="${theme.muted}" font-size="11">${tick}</text>
    `).join('')}
    ${LOSS_NOTE_STEPS.map((step) => `
      <line x1="${xScale(step)}" y1="${margin.top}" x2="${xScale(step)}" y2="${height - margin.bottom}" stroke="${theme.secondary}" stroke-width="1" stroke-dasharray="2 4"/>
    `).join('')}
    <path d="${pathFor(data, xScale, yScale, 'buggy')}" fill="none" stroke="${theme.buggy}" stroke-width="1.8" opacity="0.75"/>
    <path d="${pathFor(data, xScale, yScale, 'baseline')}" fill="none" stroke="${theme.baseline}" stroke-width="2.3" opacity="0.95"/>
    ${LOSS_NOTE_STEPS.map((step, index) => {
      const point = data.find((item) => item.step === step);
      if (!point) return '';
      const labelY = index === 0 ? margin.top + 22 : margin.top + 52;
      const labelX = Math.min(xScale(step) + 8, width - 116);
      const bg = index === 0 ? 'rgba(44, 160, 44, 0.12)' : 'rgba(214, 39, 40, 0.12)';
      const stroke = index === 0 ? theme.baseline : theme.buggy;
      return `
        <rect x="${labelX}" y="${labelY - 15}" width="96" height="28" rx="6" fill="${bg}" stroke="${stroke}" stroke-width="1"/>
        <text x="${labelX + 8}" y="${labelY + 4}" fill="${theme.text}" font-size="11" font-weight="700">step ${step}</text>
      `;
    }).join('')}
    <text x="18" y="${height / 2}" transform="rotate(-90 18 ${height / 2})" text-anchor="middle" fill="${theme.text}" font-size="13" font-weight="700">Training Loss</text>
    <g class="chart-hover"></g>
  `;

  installHover(svg, data, {
    x: xScale,
    y: yScale,
    xInvert,
    plotTop: margin.top,
    plotBottom: height - margin.bottom,
    theme,
  }, 'loss');
}

function drawDiffChart(svg, data, compact = false) {
  const theme = chartTheme();
  const { width, height } = chartSize(svg, compact ? 108 : 240);
  const margin = compact
    ? { top: 16, right: 22, bottom: 28, left: 62 }
    : { top: 26, right: 22, bottom: 36, left: 62 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const maxAbs = Math.max(...data.map((point) => Math.abs(point.delta)), 1e-8) * 1.2;
  const yMin = -maxAbs;
  const yMax = maxAbs;
  const xScale = (step) => margin.left + ((step - LOSS_STEP_MIN) / (LOSS_STEP_MAX - LOSS_STEP_MIN)) * plotWidth;
  const yScale = (value) => margin.top + ((yMax - value) / (yMax - yMin)) * plotHeight;
  const xInvert = (px) => LOSS_STEP_MIN + ((px - margin.left) / plotWidth) * (LOSS_STEP_MAX - LOSS_STEP_MIN);
  const yTicks = compact ? [-maxAbs, 0, maxAbs] : buildTicks(yMin, yMax, 5);
  const xTicks = [3050, 3150, 3250, 3350, 3450];

  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.innerHTML = `
    <rect width="${width}" height="${height}" rx="14" fill="transparent"/>
    ${compact ? '' : `<text x="${margin.left}" y="18" fill="${theme.text}" font-size="14" font-weight="700">Δ Loss</text>`}
    ${yTicks.map((tick) => `
      <line x1="${margin.left}" y1="${yScale(tick)}" x2="${width - margin.right}" y2="${yScale(tick)}" stroke="${theme.grid}" stroke-width="1" stroke-dasharray="${tick === 0 ? '0' : '3 4'}" opacity="${tick === 0 ? '0.75' : '0.75'}"/>
      <text x="${margin.left - 10}" y="${yScale(tick) + 4}" text-anchor="end" fill="${theme.muted}" font-size="11">${fmtDelta(tick)}</text>
    `).join('')}
    ${xTicks.map((tick) => `
      <line x1="${xScale(tick)}" y1="${margin.top}" x2="${xScale(tick)}" y2="${height - margin.bottom}" stroke="${theme.grid}" stroke-width="0.8" opacity="0.45"/>
      <text x="${xScale(tick)}" y="${height - 10}" text-anchor="middle" fill="${theme.muted}" font-size="11">${tick}</text>
    `).join('')}
    ${LOSS_NOTE_STEPS.map((step) => `
      <line x1="${xScale(step)}" y1="${margin.top}" x2="${xScale(step)}" y2="${height - margin.bottom}" stroke="${theme.secondary}" stroke-width="1" stroke-dasharray="2 4"/>
    `).join('')}
    <path d="${pathFor(data, xScale, yScale, 'delta')}" fill="none" stroke="${theme.delta}" stroke-width="${compact ? 1.6 : 2.1}" opacity="0.88"/>
    ${LOSS_NOTE_STEPS.map((step, index) => {
      const point = data.find((item) => item.step === step);
      if (!point) return '';
      const x = xScale(step);
      const y = yScale(point.delta);
      const labelX = Math.min(x + 10, width - 128);
      const labelY = index === 0 ? y - 24 : y + (compact ? 34 : 46);
      if (compact && index === 1) {
        return `<circle cx="${x}" cy="${y}" r="4" fill="${theme.delta}" stroke="${theme.surface}" stroke-width="2"/>`;
      }
      return `
        <circle cx="${x}" cy="${y}" r="4" fill="${theme.delta}" stroke="${theme.surface}" stroke-width="2"/>
        <line x1="${x}" y1="${y}" x2="${labelX}" y2="${labelY}" stroke="${theme.delta}" stroke-width="1"/>
        <rect x="${labelX}" y="${labelY - 18}" width="118" height="36" rx="6" fill="${theme.surface}" stroke="${theme.delta}" stroke-width="1"/>
        <text x="${labelX + 7}" y="${labelY - 3}" fill="${theme.text}" font-size="11" font-weight="700">step ${step}</text>
        <text x="${labelX + 7}" y="${labelY + 12}" fill="${theme.text}" font-size="11">Δ = ${fmtDelta(point.delta)}</text>
      `;
    }).join('')}
    <text x="18" y="${height / 2}" transform="rotate(-90 18 ${height / 2})" text-anchor="middle" fill="${theme.text}" font-size="13" font-weight="700">Δ</text>
    <text x="${width / 2}" y="${height - 2}" text-anchor="middle" fill="${theme.text}" font-size="12" font-weight="700">Step</text>
    <g class="chart-hover"></g>
  `;

  installHover(svg, data, {
    x: xScale,
    y: yScale,
    xInvert,
    plotTop: margin.top,
    plotBottom: height - margin.bottom,
    theme,
  }, 'diff');
}

function renderLossCharts() {
  if (!lossData.length) return;
  const lossChart = document.getElementById('loss-chart');
  const diffMiniChart = document.getElementById('loss-diff-mini-chart');
  if (!lossChart || !diffMiniChart) return;
  drawLossChart(lossChart, lossData);
  drawDiffChart(diffMiniChart, lossData, true);
}

async function initLossCharts() {
  try {
    lossData = await loadLossData();
    renderLossCharts();
  } catch (err) {
    console.error('Failed to render loss charts:', err);
    const container = document.querySelector('.loss-visuals');
    if (container) {
      container.insertAdjacentHTML(
        'afterbegin',
        `<div class="trace-panel__status trace-panel__status--error">Failed to load loss CSV: ${err.message}</div>`,
      );
    }
  }
}

let resizeTimer = null;
window.addEventListener('resize', () => {
  window.clearTimeout(resizeTimer);
  resizeTimer = window.setTimeout(() => {
    renderLossCharts();
    updateDiagramGuides();
  }, 120);
}, { passive: true });

function updateRootCauseGuides() {
  const diagram = document.querySelector('.root-cause-diagram');
  const svg = document.getElementById('root-cause-guides');
  const alignedMarker = document.querySelector('.op-lane__marker--aligned');
  const divergesMarker = document.querySelector('.op-lane__marker--diverges');
  const zoomBody = document.querySelector('.op-zoom__body');
  if (!diagram || !svg || !alignedMarker || !divergesMarker || !zoomBody) return;

  const diagramRect = diagram.getBoundingClientRect();
  const width = diagramRect.width;
  const height = diagramRect.height;
  if (!width || !height) return;

  const point = (el, anchor) => {
    const rect = el.getBoundingClientRect();
    const x = rect.left - diagramRect.left;
    const y = rect.top - diagramRect.top;
    if (anchor === 'marker-bottom') {
      return { x: x + rect.width / 2, y: y + rect.height };
    }
    if (anchor === 'zoom-top') {
      return { x, y };
    }
    return { x: x + rect.width, y };
  };

  const alignedStart = point(alignedMarker, 'marker-bottom');
  const divergesStart = point(divergesMarker, 'marker-bottom');
  const zoomTop = point(zoomBody, 'zoom-top').y;
  const alignedEnd = { x: alignedStart.x, y: zoomTop };
  const divergesEnd = { x: divergesStart.x, y: zoomTop };

  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('width', width);
  svg.setAttribute('height', height);

  const alignedLine = svg.querySelector('.root-cause-guides__line--aligned');
  const divergesLine = svg.querySelector('.root-cause-guides__line--diverges');
  alignedLine.setAttribute('x1', alignedStart.x);
  alignedLine.setAttribute('y1', alignedStart.y);
  alignedLine.setAttribute('x2', alignedEnd.x);
  alignedLine.setAttribute('y2', alignedEnd.y);
  divergesLine.setAttribute('x1', divergesStart.x);
  divergesLine.setAttribute('y1', divergesStart.y);
  divergesLine.setAttribute('x2', divergesEnd.x);
  divergesLine.setAttribute('y2', divergesEnd.y);
}

function updateWhatIsOpGuides() {
  const diagram = document.getElementById('what-is-op-diagram');
  const svg = document.getElementById('what-is-op-guides');
  const opDetail = document.querySelector('.op-detail');
  if (!diagram || !svg || !opDetail) return;

  const diagramRect = diagram.getBoundingClientRect();
  const width = diagramRect.width;
  const height = diagramRect.height;
  if (!width || !height) return;

  const detailRect = opDetail.getBoundingClientRect();
  const anchors = diagram.querySelectorAll('[data-op-anchor]');
  const detailLeft = detailRect.left - diagramRect.left;
  const detailTop = detailRect.top - diagramRect.top;
  const detailBottom = detailTop + detailRect.height;
  const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('width', width);
  svg.setAttribute('height', height);
  svg.innerHTML = `
    <defs>
      <marker id="what-is-op-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
        <polygon points="0 0, 8 4, 0 8" fill="#2563eb"/>
      </marker>
    </defs>
  `;

  anchors.forEach((node) => {
    const rect = node.getBoundingClientRect();
    const x1 = rect.right - diagramRect.left;
    const y1 = rect.top - diagramRect.top + rect.height / 2;
    const x2 = detailLeft - 1;
    const y2 = clamp(y1, detailTop + 18, detailBottom - 18);
    const midX = x1 + (x2 - x1) * 0.55;

    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', `M ${x1} ${y1} C ${midX} ${y1}, ${midX} ${y2}, ${x2} ${y2}`);
    path.setAttribute('marker-end', 'url(#what-is-op-arrow)');
    svg.appendChild(path);
  });
}

function updateOpDetailConnectors() {
  const detail = document.getElementById('op-detail');
  const svg = document.getElementById('op-detail-connectors');
  const leftTensor = document.querySelector('.op-matrices--stack');
  const rightTensor = document.querySelector('.op-matrices--single .op-matrix');
  const opBox = document.querySelector('.op-detail__op');
  if (!detail || !svg || !leftTensor || !rightTensor || !opBox) return;

  const detailRect = detail.getBoundingClientRect();
  const width = detailRect.width;
  const height = detailRect.height;
  if (!width || !height) return;

  const localRect = (el) => {
    const rect = el.getBoundingClientRect();
    return {
      left: rect.left - detailRect.left,
      top: rect.top - detailRect.top,
      width: rect.width,
      height: rect.height,
    };
  };

  const point = (rect, anchor) => {
    if (anchor === 'bottom-center') return { x: rect.left + rect.width / 2, y: rect.top + rect.height };
    if (anchor === 'top-left-third') return { x: rect.left + rect.width * 0.36, y: rect.top };
    if (anchor === 'top-right-third') return { x: rect.left + rect.width * 0.64, y: rect.top };
    return { x: rect.left + rect.width / 2, y: rect.top };
  };

  const left = point(localRect(leftTensor), 'bottom-center');
  const right = point(localRect(rightTensor), 'bottom-center');
  const opRect = localRect(opBox);
  const leftTarget = point(opRect, 'top-left-third');
  const rightTarget = point(opRect, 'top-right-third');
  const pathToOp = (start, end) => {
    const midY = start.y + (end.y - start.y) * 0.58;
    return `M ${start.x} ${start.y} C ${start.x} ${midY}, ${end.x} ${midY}, ${end.x} ${end.y - 4}`;
  };

  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('width', width);
  svg.setAttribute('height', height);
  svg.innerHTML = `
    <defs>
      <marker id="op-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
        <polygon points="0 0, 8 4, 0 8" fill="#2563eb"/>
      </marker>
    </defs>
    <path d="${pathToOp(left, leftTarget)}" marker-end="url(#op-arrow)" />
    <path d="${pathToOp(right, rightTarget)}" marker-end="url(#op-arrow)" />
  `;
}

function updateDiagramGuides() {
  updateRootCauseGuides();
  updateWhatIsOpGuides();
  updateOpDetailConnectors();
  updateWorkflowDiagramLinks();
  updateBitwiseAlignmentLink();
}

function getWorkflowPoint(element, canvas, anchor) {
  const rect = element.getBoundingClientRect();
  const canvasRect = canvas.getBoundingClientRect();
  const left = rect.left - canvasRect.left;
  const top = rect.top - canvasRect.top;

  const points = {
    left: { x: left, y: top + rect.height / 2 },
    right: { x: left + rect.width, y: top + rect.height / 2 },
    top: { x: left + rect.width / 2, y: top },
    bottom: { x: left + rect.width / 2, y: top + rect.height },
    center: { x: left + rect.width / 2, y: top + rect.height / 2 },
  };

  return points[anchor] || points.center;
}

function linePath(start, end) {
  return `M ${start.x} ${start.y} L ${end.x} ${end.y}`;
}

function elbowPath(start, end, viaX = (start.x + end.x) / 2) {
  return `M ${start.x} ${start.y} L ${viaX} ${start.y} L ${viaX} ${end.y} L ${end.x} ${end.y}`;
}

function updateWorkflowDiagramLinks() {
  const canvas = document.querySelector('.workflow-diagram__canvas');
  const svg = document.getElementById('workflow-diagram-links');
  if (!canvas || !svg) return;

  const select = (selector) => canvas.querySelector(selector);
  const elements = {
    buggy: select('.workflow-run--buggy'),
    reference: select('.workflow-run--ref'),
    preflightPanel: select('.workflow-panel--preflight'),
    run5: select('.workflow-tag--run5'),
    profile: select('.workflow-tag--profile'),
    determinism: select('.workflow-control--determinism'),
    replayPanel: select('.workflow-panel--replay'),
    runtime: select('.workflow-tag--runtime'),
    signature: select('.workflow-output--signature'),
    diagnosisPanel: select('.workflow-panel--diagnosis'),
    align: select('.workflow-control--align'),
    ui: select('.workflow-tag--ui'),
  };

  if (Object.values(elements).some((element) => !element)) return;

  const canvasRect = canvas.getBoundingClientRect();
  svg.setAttribute('viewBox', `0 0 ${canvasRect.width} ${canvasRect.height}`);
  svg.setAttribute('width', canvasRect.width);
  svg.setAttribute('height', canvasRect.height);

  const p = (key, anchor) => getWorkflowPoint(elements[key], canvas, anchor);
  const preflightLeft = p('preflightPanel', 'left').x;
  const run5Right = p('run5', 'right');
  const determinismLeft = p('determinism', 'left');
  const replayLeft = p('replayPanel', 'left').x;
  const signatureRight = p('signature', 'right');
  const alignLeft = p('align', 'left');

  const paths = [
    { d: linePath(p('buggy', 'right'), { x: preflightLeft, y: p('buggy', 'right').y }) },
    { d: linePath(p('reference', 'right'), { x: preflightLeft, y: p('reference', 'right').y }) },
    { d: elbowPath(p('profile', 'right'), { x: replayLeft, y: p('replayPanel', 'center').y }, replayLeft - 24) },
    { d: linePath(p('determinism', 'bottom'), p('replayPanel', 'top')), dashed: true },
    { d: elbowPath(determinismLeft, run5Right, determinismLeft.x - 42), dashed: true },
    { d: linePath(p('runtime', 'bottom'), p('signature', 'top')) },
    { d: elbowPath(signatureRight, alignLeft, signatureRight.x + 62) },
    { d: linePath(p('diagnosisPanel', 'bottom'), p('ui', 'top')) },
  ];

  svg.innerHTML = `
    <defs>
      <marker id="workflow-arrow" viewBox="0 0 10 10" markerWidth="8" markerHeight="8" refX="9" refY="5" orient="auto">
        <path d="M 0 0 L 10 5 L 0 10 z" />
      </marker>
    </defs>
    ${paths.map(({ d, dashed }) => `<path class="workflow-link${dashed ? ' workflow-link--dashed' : ''}" d="${d}" />`).join('')}
  `;
}

let activeBitwiseLink = null;

function getBitwiseCellPair(sourceCell) {
  const diagram = sourceCell.closest('.bitwise-diagram');
  const sourceTensor = sourceCell.closest('.bitwise-tensor');
  if (!diagram || !sourceTensor || !sourceCell.closest('.bitwise-track--buggy')) return null;

  const buggyTensors = Array.from(diagram.querySelectorAll('.bitwise-track--buggy .bitwise-tensor'));
  const referenceTensors = Array.from(diagram.querySelectorAll('.bitwise-track--ref .bitwise-tensor'));
  const tensorIndex = buggyTensors.indexOf(sourceTensor);
  if (tensorIndex < 0) return null;

  const sourceCells = Array.from(sourceTensor.querySelectorAll('.bitwise-tensor__cell'));
  const cellIndex = sourceCells.indexOf(sourceCell);
  const targetTensor = referenceTensors[tensorIndex];
  const targetCell = targetTensor?.querySelectorAll('.bitwise-tensor__cell')[cellIndex];
  if (!targetCell) return null;

  return { diagram, sourceCell, targetCell };
}

function getElementCenter(element, container) {
  const rect = element.getBoundingClientRect();
  const containerRect = container.getBoundingClientRect();

  return {
    x: rect.left - containerRect.left + rect.width / 2,
    y: rect.top - containerRect.top + rect.height / 2,
  };
}

function clearBitwiseLinkState(diagram) {
  diagram.querySelectorAll(
    '.bitwise-tensor__cell--link-source, .bitwise-tensor__cell--link-target, .bitwise-tensor__cell--link-match, .bitwise-tensor__cell--link-mismatch',
  ).forEach((cell) => {
    cell.classList.remove(
      'bitwise-tensor__cell--link-source',
      'bitwise-tensor__cell--link-target',
      'bitwise-tensor__cell--link-match',
      'bitwise-tensor__cell--link-mismatch',
    );
  });

  const svg = diagram.querySelector('.bitwise-link-layer');
  if (svg) svg.innerHTML = '';
}

function drawBitwiseAlignmentLink(sourceCell, targetCell, isMatch) {
  const flow = sourceCell.closest('.bitwise-flow');
  const svg = flow?.querySelector('.bitwise-link-layer');
  if (!flow || !svg) return;

  const width = flow.scrollWidth;
  const height = flow.scrollHeight;
  const source = getElementCenter(sourceCell, flow);
  const target = getElementCenter(targetCell, flow);
  const midY = source.y + (target.y - source.y) * 0.5;
  const pathClass = isMatch ? 'bitwise-link--match' : 'bitwise-link--mismatch';

  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('width', width);
  svg.setAttribute('height', height);
  svg.innerHTML = `<path class="${pathClass}" d="M ${source.x} ${source.y} C ${source.x} ${midY}, ${target.x} ${midY}, ${target.x} ${target.y}" />`;
}

function updateBitwiseAlignmentLink() {
  if (!activeBitwiseLink) return;

  const { sourceCell, targetCell, isMatch } = activeBitwiseLink;
  if (!document.contains(sourceCell) || !document.contains(targetCell)) {
    activeBitwiseLink = null;
    return;
  }

  drawBitwiseAlignmentLink(sourceCell, targetCell, isMatch);
}

function selectBitwiseAlignmentCell(sourceCell) {
  const pair = getBitwiseCellPair(sourceCell);
  if (!pair) return;

  const { diagram, targetCell } = pair;
  const isMatch = sourceCell.textContent.trim() === targetCell.textContent.trim();
  const stateClass = isMatch ? 'bitwise-tensor__cell--link-match' : 'bitwise-tensor__cell--link-mismatch';

  clearBitwiseLinkState(diagram);
  sourceCell.classList.add('bitwise-tensor__cell--link-source', stateClass);
  targetCell.classList.add('bitwise-tensor__cell--link-target', stateClass);
  activeBitwiseLink = { sourceCell, targetCell, isMatch };
  drawBitwiseAlignmentLink(sourceCell, targetCell, isMatch);
}

function initBitwiseAlignmentLinks() {
  const buggyCells = document.querySelectorAll('.bitwise-track--buggy .bitwise-tensor__cell');

  buggyCells.forEach((cell) => {
    cell.setAttribute('aria-label', `Link buggy value ${cell.textContent.trim()} to the matching reference tensor position`);

    cell.addEventListener('click', () => {
      const selection = window.getSelection();
      if (selection && !selection.isCollapsed) return;
      selectBitwiseAlignmentCell(cell);
    });

    cell.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        selectBitwiseAlignmentCell(cell);
      }
    });
  });
}

function initTensorCellSelection() {
  const cells = document.querySelectorAll('.op-matrix__cell:not(.bitwise-tensor__cell)');
  cells.forEach((cell) => {
    cell.setAttribute('aria-selected', 'false');

    cell.addEventListener('click', () => {
      const selection = window.getSelection();
      if (selection && !selection.isCollapsed) return;
      cell.classList.toggle('op-matrix__cell--selected');
      cell.setAttribute('aria-selected', cell.classList.contains('op-matrix__cell--selected') ? 'true' : 'false');
    });

    cell.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        cell.click();
      }
    });
  });
}

// Nav scroll effect
const nav = document.getElementById('nav');
window.addEventListener('scroll', () => {
  nav.classList.toggle('nav--scrolled', window.scrollY > 40);
}, { passive: true });

// Theme toggle
const themeToggle = document.getElementById('theme-toggle');
const root = document.documentElement;

function setTheme(theme) {
  root.setAttribute('data-theme', theme);
  localStorage.setItem('opguard-theme', theme);
  renderLossCharts();
}

themeToggle.addEventListener('click', () => {
  const next = root.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
  setTheme(next);
});

window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', (e) => {
  if (!localStorage.getItem('opguard-theme')) {
    setTheme(e.matches ? 'light' : 'dark');
  }
});

initLossCharts();
loadTrace();
updateDiagramGuides();
initTensorCellSelection();
initBitwiseAlignmentLinks();
window.addEventListener('load', updateDiagramGuides);
