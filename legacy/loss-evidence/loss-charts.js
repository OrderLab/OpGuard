const LOSS_RUNS = [
  {
    key: 'baseline',
    label: 'baseline (correct)',
    url: './data/run_20251001_12292e6b.csv',
  },
  {
    key: 'buggy',
    label: 'buggy',
    url: './data/run_20251001_9e609e46.csv',
  },
];
const LOSS_STEP_MIN = 3050;
const LOSS_STEP_MAX = 3450;
const LOSS_NOTE_STEPS = [3080, 3081];

// Standalone excerpt — restore into js/app.js when bringing Loss Evidence back.
const lossTooltip = document.getElementById('loss-tooltip');
let lossData = [];

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

