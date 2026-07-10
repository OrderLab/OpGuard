const TRACE_URL = './assets/traces/demo_case5.json.gz?v=track-names';
const TRACE_TITLE = 'OpGuard — demo_case5 (open-source issue)';
const TRACE_SOURCE_URL = './assets/traces/demo_case5_source.json';
const TRACE_DIFF_URL = './assets/traces/demo_case5_diff.json';
// Bottom-most call-stack frame at the first-diff pivot (Chrome JSON µs).
const CALLSTACK_SOURCE_SLICE = {
  tsUs: 128647000,
  get timeSec() {
    return this.tsUs / 1e6;
  },
};

const iframe = document.getElementById('perfetto');
const statusEl = document.getElementById('trace-status');
const overlayEl = document.getElementById('trace-overlay');
const progressEl = document.getElementById('trace-progress');

let cachedTraceBuffer = null;
let cachedTourSource = null;
let cachedTourDiff = null;
let traceReady = false;
let pendingOpenTour = false;

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
          keepApiOpen: true,
        },
      },
      '*',
    );

    cachedTraceBuffer = buffer;

    setProgress(100);
    setStatus('Trace loaded', 'ready');
    overlayEl.classList.add('trace-panel__overlay--hidden');
    onTraceReady();
  } catch (err) {
    console.error('Failed to load trace:', err);
    setStatus(`Failed to load trace: ${err.message}`, 'error');
    overlayEl.querySelector('p').textContent =
      'Could not load trace. Serve this page over HTTP (not file://).';
  }
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

const TRACE_TOUR_STORAGE_KEY = 'opguard-trace-tour-seen-v18';
const TRACE_TOUR_STEPS = [
  {
    target: 'viewer',
    placement: 'dock-bottom',
    spotlight: 'full',
    title: 'Welcome to the trace viewer',
    body: 'This is OpGuard’s Global Alignment Trace for an open-source issue. Two runs are lined up on one timeline: the suspect run (top) and the reference run (bottom), so you can compare them side by side.',
  },
  {
    target: 'viewer',
    placement: 'dock-right',
    spotlight: 'sidebar',
    interactive: true,
    title: 'Expand the suspect run',
    body: 'Click the arrow next to suspect run in the left sidebar (the highlighted area is clickable). Expand it to reveal op tracks and alignment slices — then press Next.',
  },
  {
    target: 'viewer',
    placement: 'dock-bottom',
    spotlight: 'pivot',
    interactive: true,
    title: 'Click the first-mismatch pivot',
    body: 'On the Alignment Status track, click the blue pivot marker (▲). That is the first mismatch between the suspect and reference runs — the clean pivot for debugging. Then press Next.',
  },
  {
    target: 'viewer',
    placement: 'dock-top',
    spotlight: 'following-flows',
    interactive: true,
    title: 'Open the divergent op',
    body: 'In Current Selection → Following Flows, click the linked op (e.g. _linalg.linalg_vector_norm#…). The tensor-level diff details live on that op — not on the pivot marker itself. Then press Next.',
  },
  {
    target: 'viewer',
    placement: 'dock-top',
    spotlight: 'diff-panel',
    interactive: true,
    autoReveal: 'diff-field',
    title: 'Inspect the diff field',
    body: 'Here is the scrolled-to diff for this op: fields = outputs, and the summary shows which tensor mismatched (xor_signature … vs …). Then press Next.',
  },
  {
    target: 'viewer',
    placement: 'dock-top',
    spotlight: 'details-chrome',
    interactive: true,
    title: 'Hide the details panel',
    body: 'In the top-right corner of Current Selection, click the downward chevron (v) to collapse the panel and free the timeline. Then press Next.',
  },
  {
    target: 'viewer',
    placement: 'dock-left',
    spotlight: 'source-panel',
    interactive: true,
    autoReveal: 'callstack-source',
    title: 'Source from the call stack',
    body: 'In the real workflow you’d click the bottom-most Call Stack frame under the pivot, then scroll down in Current Selection to open source. Here we jump straight to that end state: norm · functional.py:1629 — the call site of the first mismatched op.',
  },
];

let traceTourIndex = 0;
let traceTourOpen = false;

function getTraceTourTarget(step) {
  const panel = document.getElementById('trace-panel');
  if (!panel) return null;
  if (step.target === 'panel') return panel;
  return panel.querySelector(`[data-trace-tour="${step.target}"]`);
}

function getTraceTourSpotlightRect(target, step) {
  const panel = document.getElementById('trace-panel');
  if (!panel || !target) return null;

  const panelRect = panel.getBoundingClientRect();
  const rect = target.getBoundingClientRect();
  const pad = 6;

  let top = rect.top - panelRect.top - pad;
  let left = rect.left - panelRect.left - pad;
  let width = rect.width + pad * 2;
  let height = rect.height + pad * 2;

  if (step.target === 'viewer' && step.spotlight) {
    const viewerRect = target.getBoundingClientRect();
    const relTop = viewerRect.top - panelRect.top;
    const relLeft = viewerRect.left - panelRect.left;
    const h = viewerRect.height;
    const w = viewerRect.width;
    const sidebarW = Math.min(Math.max(w * 0.185, 172), 228);
    const searchH = h * 0.082;
    const overviewTop = searchH;
    const overviewH = h * 0.118;
    const tracksTop = overviewTop + overviewH + h * 0.008;
    const tracksHExpanded = h * 0.335;
    const tracksHCollapsed = h - tracksTop - h * 0.085;
    const detailsTop = tracksTop + tracksHExpanded;
    const detailsH = h - detailsTop - 2;

    if (step.spotlight === 'overview') {
      top = relTop + overviewTop;
      left = relLeft + 4;
      width = w - 8;
      height = overviewH;
    } else if (step.spotlight === 'sidebar') {
      // Cover the process tree so the expand arrow is easy to hit.
      top = relTop + tracksTop;
      left = relLeft + 2;
      width = sidebarW;
      height = Math.min(tracksHCollapsed * 0.42, h * 0.28);
    } else if (step.spotlight === 'tracks') {
      top = relTop + tracksTop;
      left = relLeft + sidebarW + 4;
      width = w - sidebarW - 8;
      height = Math.min(tracksHCollapsed * 0.48, h * 0.28);
    } else if (step.spotlight === 'pivot') {
      // first_diff_global sits ~80% along the demo timeline on Alignment Status.
      const trackLeft = relLeft + sidebarW + 4;
      const trackWidth = w - sidebarW - 8;
      const pivotFrac = 0.78;
      const boxW = Math.min(Math.max(trackWidth * 0.22, 160), 280);
      top = relTop + tracksTop + h * 0.055;
      left = trackLeft + trackWidth * pivotFrac - boxW * 0.35;
      width = boxW;
      height = Math.min(h * 0.2, 120);
    } else if (step.spotlight === 'details') {
      top = relTop + detailsTop;
      left = relLeft + 2;
      width = w - 4;
      height = detailsH;
    } else if (step.spotlight === 'following-flows') {
      // Right half of Current Selection — Following Flows table.
      top = relTop + detailsTop + detailsH * 0.08;
      left = relLeft + w * 0.42;
      width = w * 0.56;
      height = Math.min(detailsH * 0.42, h * 0.22);
    } else if (step.spotlight === 'diff-field') {
      // Lower-right args / diff tree in Current Selection.
      top = relTop + detailsTop + detailsH * 0.28;
      left = relLeft + w * 0.36;
      width = w * 0.62;
      height = Math.min(detailsH * 0.55, h * 0.28);
    } else if (step.spotlight === 'diff-panel') {
      top = relTop + h * 0.55;
      left = relLeft + 10;
      width = w - 20;
      height = h * 0.42;
    } else if (step.spotlight === 'details-chrome') {
      // Collapse chevron sits on the Current Selection tab bar, far top-right.
      const boxW = 64;
      const boxH = 52;
      const drawerTop = relTop + h * 0.455;
      top = drawerTop;
      left = relLeft + w - boxW - 6;
      width = boxW;
      height = boxH;
    } else if (step.spotlight === 'callstack') {
      // After details are hidden: suspect Call Stack is the upper track group, near the pivot.
      const trackLeft = relLeft + sidebarW + 4;
      const trackWidth = w - sidebarW - 8;
      const pivotFrac = 0.78;
      const boxW = Math.min(Math.max(trackWidth * 0.2, 150), 220);
      top = relTop + tracksTop + 6;
      left = trackLeft + trackWidth * pivotFrac - boxW * 0.45;
      width = boxW;
      height = Math.min(h * 0.16, 120);
    } else if (step.spotlight === 'source-panel') {
      top = relTop + h * 0.55;
      left = relLeft + 10;
      width = w - 20;
      height = h * 0.42;
    } else if (step.spotlight === 'full') {
      top = relTop + 2;
      left = relLeft + 4;
      width = w - 8;
      height = h - 4;
    }
  }

  return { top, left, width, height };
}

function positionTraceTourCard(card, panelRect, spotlightRect, placement) {
  const margin = 16;
  const cardWidth = card.offsetWidth;
  const cardHeight = card.offsetHeight;

  card.style.right = 'auto';
  card.style.bottom = 'auto';
  card.style.width = '';

  if (placement === 'dock-bottom') {
    card.style.left = `${margin}px`;
    card.style.top = `${Math.max(margin, panelRect.height - cardHeight - margin)}px`;
    card.style.width = `${Math.min(360, panelRect.width - margin * 2)}px`;
    return;
  }

  if (placement === 'dock-top') {
    card.style.left = `${margin}px`;
    card.style.top = `${margin + 58}px`;
    card.style.width = `${Math.min(360, panelRect.width - margin * 2)}px`;
    return;
  }

  if (placement === 'dock-right') {
    // Keep the left sidebar free so the user can click expand arrows.
    const width = Math.min(360, panelRect.width - margin * 2);
    card.style.width = `${width}px`;
    card.style.left = `${Math.max(margin, panelRect.width - width - margin)}px`;
    card.style.top = `${Math.max(margin, panelRect.height - cardHeight - margin)}px`;
    return;
  }

  if (placement === 'dock-left') {
    // Keep the pivot / call-stack region on the right free to click.
    const width = Math.min(320, panelRect.width - margin * 2);
    card.style.width = `${width}px`;
    card.style.left = `${margin}px`;
    card.style.top = `${Math.max(margin + 58, (panelRect.height - cardHeight) / 2)}px`;
    return;
  }

  if (placement === 'bottom') {
    card.style.left = `${Math.min(
      Math.max(margin, spotlightRect.left),
      panelRect.width - cardWidth - margin,
    )}px`;
    card.style.top = `${Math.min(
      spotlightRect.top + spotlightRect.height + margin,
      panelRect.height - cardHeight - margin,
    )}px`;
    return;
  }

  if (placement === 'top') {
    card.style.left = `${Math.min(
      Math.max(margin, spotlightRect.left),
      panelRect.width - cardWidth - margin,
    )}px`;
    card.style.top = `${Math.max(margin, spotlightRect.top - cardHeight - margin)}px`;
    return;
  }

  if (placement === 'center') {
    card.style.left = `${Math.min(
      Math.max(margin, spotlightRect.left + (spotlightRect.width - cardWidth) / 2),
      panelRect.width - cardWidth - margin,
    )}px`;
    card.style.top = `${Math.min(
      spotlightRect.top + spotlightRect.height + margin,
      panelRect.height - cardHeight - margin,
    )}px`;
    return;
  }

  card.style.left = `${Math.min(
    Math.max(margin, spotlightRect.left + spotlightRect.width + margin),
    panelRect.width - cardWidth - margin,
  )}px`;
  card.style.top = `${Math.min(
    Math.max(margin, spotlightRect.top),
    panelRect.height - cardHeight - margin,
  )}px`;
}

function zoomToCallstackSlice() {
  if (!iframe?.contentWindow) return;
  const t = CALLSTACK_SOURCE_SLICE.timeSec;
  iframe.contentWindow.postMessage(
    {
      perfetto: {
        timeStart: t - 0.45,
        timeEnd: t + 0.45,
        viewPercentage: 0.4,
      },
    },
    '*',
  );
}

async function loadTourSource() {
  if (cachedTourSource) return cachedTourSource;
  const response = await fetch(TRACE_SOURCE_URL);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  cachedTourSource = await response.json();
  return cachedTourSource;
}

async function loadTourDiff() {
  if (cachedTourDiff) return cachedTourDiff;
  const response = await fetch(TRACE_DIFF_URL);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  cachedTourDiff = await response.json();
  return cachedTourDiff;
}

function hideTourOverlayPanels() {
  ['trace-source', 'trace-diff'].forEach((id) => {
    const panel = document.getElementById(id);
    if (!panel) return;
    panel.hidden = true;
    panel.setAttribute('aria-hidden', 'true');
  });
}

function hideTourSourcePanel() {
  hideTourOverlayPanels();
}

function renderTourSourcePanel(source) {
  hideTourOverlayPanels();
  const panel = document.getElementById('trace-source');
  const title = document.getElementById('trace-source-title');
  const path = document.getElementById('trace-source-path');
  const code = document.getElementById('trace-source-code');
  if (!panel || !title || !path || !code) return;

  title.textContent = source.title;
  path.textContent = source.original_path || `${source.file}:${source.line}`;
  code.replaceChildren();

  let targetEl = null;
  String(source.code || '')
    .split('\n')
    .forEach((line) => {
      const row = document.createElement('span');
      row.className = 'trace-source__line';
      const isTarget = line.includes('>>>') && line.includes('TARGET LINE');
      if (isTarget) {
        row.classList.add('is-target');
        row.textContent = line
          .replace('>>> ', '')
          .replace(' <<< TARGET LINE', '');
        targetEl = row;
      } else {
        row.textContent = line;
      }
      code.appendChild(row);
    });

  panel.hidden = false;
  panel.setAttribute('aria-hidden', 'false');
  window.requestAnimationFrame(() => {
    targetEl?.scrollIntoView({ block: 'center', behavior: 'smooth' });
  });
}

function renderTourDiffPanel(diff) {
  hideTourOverlayPanels();
  const panel = document.getElementById('trace-diff');
  const title = document.getElementById('trace-diff-title');
  const path = document.getElementById('trace-diff-path');
  const body = document.getElementById('trace-diff-body');
  if (!panel || !title || !path || !body) return;

  title.textContent = diff.title;
  path.textContent = diff.fn || diff.search || '';
  body.replaceChildren();

  const fields = (diff.diff && diff.diff.fields) || [];
  const summary = (diff.diff && diff.diff.summary) || '';
  const output = diff.io && diff.io.outputs && diff.io.outputs[0];

  const rows = [
    ['fields', fields.join(', ') || '—'],
    ['status', diff.status || '—'],
    ['dtype', output?.dtype || '—'],
    ['device', output?.device || '—'],
    ['shape', output ? JSON.stringify(output.shape) : '—'],
  ];

  rows.forEach(([key, val]) => {
    const row = document.createElement('div');
    row.className = 'trace-diff__row';
    const k = document.createElement('div');
    k.className = 'trace-diff__key';
    k.textContent = key;
    const v = document.createElement('div');
    v.className = 'trace-diff__val';
    v.textContent = val;
    row.append(k, v);
    body.appendChild(row);
  });

  if (summary) {
    const summaryEl = document.createElement('div');
    summaryEl.className = 'trace-diff__summary';
    const parts = String(summary).split(' vs ');
    if (parts.length === 2) {
      const left = document.createElement('strong');
      left.textContent = parts[0];
      const right = document.createElement('strong');
      right.textContent = parts[1];
      summaryEl.append(left, document.createTextNode(' vs '), right);
    } else {
      summaryEl.textContent = summary;
    }
    body.appendChild(summaryEl);
  }

  panel.hidden = false;
  panel.setAttribute('aria-hidden', 'false');
  window.requestAnimationFrame(() => {
    body.querySelector('.trace-diff__summary')?.scrollIntoView({
      block: 'nearest',
      behavior: 'smooth',
    });
  });
}

async function revealTourCallstackSourceSlice() {
  zoomToCallstackSlice();
  const source = await loadTourSource();
  renderTourSourcePanel(source);
}

async function revealTourDiffField() {
  const diff = await loadTourDiff();
  renderTourDiffPanel(diff);
}

function refitTourStepChrome(index, target, step, panel, spotlight, card) {
  if (!traceTourOpen || traceTourIndex !== index) return;
  const panelRect = panel.getBoundingClientRect();
  const spotlightRect = getTraceTourSpotlightRect(target, step);
  if (!spotlightRect) return;
  spotlight.style.top = `${spotlightRect.top}px`;
  spotlight.style.left = `${spotlightRect.left}px`;
  spotlight.style.width = `${spotlightRect.width}px`;
  spotlight.style.height = `${spotlightRect.height}px`;
  positionTraceTourCard(card, panelRect, spotlightRect, step.placement);
}

function renderTraceTourStep(index) {
  const tour = document.getElementById('trace-tour');
  const panel = document.getElementById('trace-panel');
  const spotlight = document.getElementById('trace-tour-spotlight');
  const card = document.getElementById('trace-tour-card');
  const progress = document.getElementById('trace-tour-progress');
  const title = document.getElementById('trace-tour-title');
  const body = document.getElementById('trace-tour-body');
  const backBtn = document.getElementById('trace-tour-back');
  const nextBtn = document.getElementById('trace-tour-next');
  if (!tour || !panel || !spotlight || !card || !progress || !title || !body || !backBtn || !nextBtn) return;

  panel.querySelectorAll('[data-trace-tour].trace-tour-target').forEach((el) => {
    el.classList.remove('trace-tour-target');
  });

  const step = TRACE_TOUR_STEPS[index];
  const target = getTraceTourTarget(step);
  if (!target) return;

  target.classList.add('trace-tour-target');
  traceTourIndex = index;

  progress.textContent = `${index + 1} / ${TRACE_TOUR_STEPS.length}`;
  title.textContent = step.title;
  body.textContent = step.body;
  backBtn.disabled = index === 0;
  nextBtn.textContent =
    index === TRACE_TOUR_STEPS.length - 1
      ? 'Start exploring'
      : step.nextLabel || 'Next';

  tour.classList.toggle('trace-tour--interactive', Boolean(step.interactive));
  card.setAttribute('aria-modal', step.interactive ? 'false' : 'true');

  if (step.autoReveal === 'diff-field') {
    body.textContent = 'Scrolling to the diff field…';
    revealTourDiffField()
      .then(() => {
        if (traceTourOpen && traceTourIndex === index) {
          body.textContent = step.body;
          refitTourStepChrome(index, target, step, panel, spotlight, card);
        }
      })
      .catch((err) => {
        console.error('Failed to open tour diff panel:', err);
        if (traceTourOpen && traceTourIndex === index) {
          body.textContent =
            'Could not open the diff panel. In Current Selection, scroll to args → diff manually.';
        }
      });
  } else if (step.autoReveal === 'callstack-source') {
    body.textContent = 'Opening source for norm · functional.py:1629…';
    revealTourCallstackSourceSlice()
      .then(() => {
        if (traceTourOpen && traceTourIndex === index) {
          body.textContent = step.body;
          refitTourStepChrome(index, target, step, panel, spotlight, card);
        }
      })
      .catch((err) => {
        console.error('Failed to open tour source panel:', err);
        if (traceTourOpen && traceTourIndex === index) {
          body.textContent =
            'Could not load the source panel. You can still inspect Call Stack frames in the timeline.';
        }
      });
  } else {
    hideTourOverlayPanels();
  }

  const panelRect = panel.getBoundingClientRect();
  const spotlightRect = getTraceTourSpotlightRect(target, step);
  if (!spotlightRect) return;

  spotlight.style.top = `${spotlightRect.top}px`;
  spotlight.style.left = `${spotlightRect.left}px`;
  spotlight.style.width = `${spotlightRect.width}px`;
  spotlight.style.height = `${spotlightRect.height}px`;

  window.requestAnimationFrame(() => {
    positionTraceTourCard(card, panelRect, spotlightRect, step.placement);
  });
}

function openTraceTour(startIndex = 0) {
  const tour = document.getElementById('trace-tour');
  const panel = document.getElementById('trace-panel');
  if (!tour) return;

  traceTourOpen = true;
  document.body.classList.add('trace-tour-open');
  panel?.classList.add('trace-tour-active');
  tour.hidden = false;
  tour.setAttribute('aria-hidden', 'false');
  renderTraceTourStep(startIndex);
}

function closeTraceTour(markSeen = true) {
  const tour = document.getElementById('trace-tour');
  const panel = document.getElementById('trace-panel');
  if (!tour) return;

  traceTourOpen = false;
  document.body.classList.remove('trace-tour-open');
  panel?.classList.remove('trace-tour-active');
  tour.hidden = true;
  tour.setAttribute('aria-hidden', 'true');
  panel?.querySelectorAll('[data-trace-tour].trace-tour-target').forEach((el) => {
    el.classList.remove('trace-tour-target');
  });
  hideTourOverlayPanels();

  if (markSeen) {
    localStorage.setItem(TRACE_TOUR_STORAGE_KEY, '1');
  }
}

function onTraceReady() {
  const launchBtn = document.getElementById('trace-tour-launch');
  if (launchBtn) launchBtn.hidden = false;
  traceReady = true;

  if (pendingOpenTour) {
    pendingOpenTour = false;
    window.setTimeout(() => openTraceTour(0), 400);
  }
}

function scrollTracePanelToCenter() {
  const panel = document.getElementById('trace-panel');
  if (!panel) {
    document.getElementById('trace')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    return;
  }

  const rect = panel.getBoundingClientRect();
  const targetTop = window.scrollY + rect.top - (window.innerHeight - rect.height) / 2;
  window.scrollTo({ top: Math.max(0, targetTop), behavior: 'smooth' });
}

function launchDemoTour(event) {
  if (event) event.preventDefault();

  scrollTracePanelToCenter();

  if (traceReady) {
    window.setTimeout(() => openTraceTour(0), 450);
    return;
  }

  pendingOpenTour = true;
}

function initTraceTour() {
  const tour = document.getElementById('trace-tour');
  const backBtn = document.getElementById('trace-tour-back');
  const nextBtn = document.getElementById('trace-tour-next');
  const skipBtn = document.getElementById('trace-tour-skip');
  const launchBtn = document.getElementById('trace-tour-launch');
  if (!tour || !backBtn || !nextBtn || !skipBtn || !launchBtn) return;

  backBtn.addEventListener('click', () => {
    if (traceTourIndex > 0) renderTraceTourStep(traceTourIndex - 1);
  });

  nextBtn.addEventListener('click', () => {
    if (traceTourIndex < TRACE_TOUR_STEPS.length - 1) {
      renderTraceTourStep(traceTourIndex + 1);
      return;
    }
    closeTraceTour(true);
  });

  skipBtn.addEventListener('click', () => closeTraceTour(true));
  launchBtn.addEventListener('click', () => openTraceTour(0));

  document.querySelectorAll('[data-open-tour]').forEach((el) => {
    el.addEventListener('click', launchDemoTour);
  });

  window.addEventListener('keydown', (event) => {
    if (!traceTourOpen) return;
    if (event.key === 'Escape') closeTraceTour(true);
  });

  window.addEventListener('resize', () => {
    if (traceTourOpen) renderTraceTourStep(traceTourIndex);
  });
}

function updateWfAlignLinks() {
  const root = document.getElementById('workflow-deck');
  const diagram = document.getElementById('wf-diagram');
  const svg = document.getElementById('wf-align');
  if (!root || !diagram || !svg) return;

  if (Number(root.dataset.step) < 4) {
    svg.innerHTML = '';
    return;
  }

  const width = diagram.clientWidth;
  const height = Math.max(svg.clientHeight || 56, 56);
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('width', String(width));
  svg.setAttribute('height', String(height));

  const svgRect = svg.getBoundingClientRect();
  const colors = {
    v: '#ca8a04',
    k: '#0284c7',
    attn: '#15803d',
  };

  const markers = Object.entries(colors).map(([key, color]) => `
    <marker id="wf-arrow-start-${key}" viewBox="0 0 10 10" refX="2" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 10 0 L 0 5 L 10 10 z" fill="${color}" />
    </marker>
    <marker id="wf-arrow-end-${key}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="${color}" />
    </marker>
  `).join('');

  const paths = ['v', 'k', 'attn'].map((key) => {
    const top = diagram.querySelector(`.wf-fps--suspect .wf-fp[data-fp="${key}"]`);
    const bottom = diagram.querySelector(`.wf-fps--ref .wf-fp[data-fp="${key}"]`);
    if (!top || !bottom) return '';

    const topRect = top.getBoundingClientRect();
    const bottomRect = bottom.getBoundingClientRect();
    const x1 = topRect.left + topRect.width / 2 - svgRect.left;
    const x2 = bottomRect.left + bottomRect.width / 2 - svgRect.left;
    const y1 = 6;
    const y2 = height - 6;
    const midY = height * 0.5;
    return `<path stroke="${colors[key]}" marker-start="url(#wf-arrow-start-${key})" marker-end="url(#wf-arrow-end-${key})" d="M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}" />`;
  }).join('');

  svg.innerHTML = `<defs>${markers}</defs>${paths}`;
}

function initBitwiseAlignment() {
  const root = document.getElementById('ba-deck');
  if (!root) return;

  const dots = Array.from(root.querySelectorAll('.ba-deck__dot'));
  const headings = Array.from(root.querySelectorAll('.ba-heading'));
  const prevBtn = document.getElementById('ba-prev');
  const nextBtn = document.getElementById('ba-next');
  const playBtn = document.getElementById('ba-play');
  const total = 3;
  let step = 1;
  let timer = null;

  function setStep(next) {
    step = ((next - 1 + total) % total) + 1;
    root.dataset.step = String(step);

    dots.forEach((dot) => {
      dot.classList.toggle('is-active', Number(dot.dataset.step) === step);
    });

    headings.forEach((heading) => {
      heading.classList.toggle('is-on', Number(heading.dataset.step) === step);
    });
  }

  function stopPlay() {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
    playBtn?.setAttribute('aria-pressed', 'false');
    if (playBtn) playBtn.textContent = 'Play';
  }

  function startPlay() {
    stopPlay();
    playBtn?.setAttribute('aria-pressed', 'true');
    if (playBtn) playBtn.textContent = 'Pause';
    timer = setInterval(() => setStep(step + 1), 2800);
  }

  prevBtn?.addEventListener('click', () => {
    stopPlay();
    setStep(step - 1);
  });

  nextBtn?.addEventListener('click', () => {
    stopPlay();
    setStep(step + 1);
  });

  playBtn?.addEventListener('click', () => {
    if (timer) stopPlay();
    else startPlay();
  });

  dots.forEach((dot) => {
    dot.addEventListener('click', () => {
      stopPlay();
      setStep(Number(dot.dataset.step));
    });
  });

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting && !timer) startPlay();
        if (!entry.isIntersecting) stopPlay();
      });
    },
    { threshold: 0.4 },
  );
  observer.observe(root);

  setStep(1);
}

function initHowItWorks() {
  const root = document.getElementById('workflow-deck');
  if (!root) return;

  const dots = Array.from(root.querySelectorAll('.wf__dot'));
  const headings = Array.from(root.querySelectorAll('.wf-heading'));
  const prevBtn = document.getElementById('workflow-prev');
  const nextBtn = document.getElementById('workflow-next');
  const playBtn = document.getElementById('workflow-play');
  const total = 4;
  let step = 1;
  let timer = null;

  function setStep(next) {
    step = ((next - 1 + total) % total) + 1;
    root.dataset.step = String(step);

    dots.forEach((dot) => {
      dot.classList.toggle('is-active', Number(dot.dataset.step) === step);
    });

    headings.forEach((heading) => {
      heading.classList.toggle('is-on', Number(heading.dataset.step) === step);
    });

    setTimeout(updateWfAlignLinks, step === 4 ? 80 : 0);
  }

  function stopPlay() {
    if (timer) {
      clearInterval(timer);
      timer = null;
    }
    playBtn?.setAttribute('aria-pressed', 'false');
    if (playBtn) playBtn.textContent = 'Play';
  }

  function startPlay() {
    stopPlay();
    playBtn?.setAttribute('aria-pressed', 'true');
    if (playBtn) playBtn.textContent = 'Pause';
    timer = setInterval(() => setStep(step + 1), 2600);
  }

  prevBtn?.addEventListener('click', () => {
    stopPlay();
    setStep(step - 1);
  });

  nextBtn?.addEventListener('click', () => {
    stopPlay();
    setStep(step + 1);
  });

  playBtn?.addEventListener('click', () => {
    if (timer) stopPlay();
    else startPlay();
  });

  dots.forEach((dot) => {
    dot.addEventListener('click', () => {
      stopPlay();
      setStep(Number(dot.dataset.step));
    });
  });

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting && !timer) startPlay();
        if (!entry.isIntersecting) stopPlay();
      });
    },
    { threshold: 0.4 },
  );
  observer.observe(root);

  window.addEventListener('resize', updateWfAlignLinks);
  setStep(1);
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

loadTrace();
initTensorCellSelection();
initTraceTour();
initHowItWorks();
initBitwiseAlignment();
initCiteCopy();

function initCiteCopy() {
  const btn = document.getElementById('cite-copy');
  const code = document.getElementById('cite-bibtex');
  if (!btn || !code) return;

  btn.addEventListener('click', async () => {
    const text = code.textContent.trim();
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const range = document.createRange();
      range.selectNodeContents(code);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      document.execCommand('copy');
      selection.removeAllRanges();
    }
    btn.textContent = 'Copied';
    btn.classList.add('is-copied');
    window.setTimeout(() => {
      btn.textContent = 'Copy';
      btn.classList.remove('is-copied');
    }, 1600);
  });
}