#!/usr/bin/env python3
import argparse
import json
from collections import Counter
import plotly.graph_objects as go

def build_icicle(data):
    prefix_counts = Counter()
    last_frame_fullpath = {}

    for info in data.values():
        count = info.get("count", 0)
        for cs in info.get("callstacks", []):
            path = []
            for filename, lineno, funcname in cs:
                label = f"{funcname} ({filename.split('/')[-1]}:{lineno})"
                path.append(label)
                full_path = "/".join(path)
                prefix_counts[full_path] += count
                last_frame_fullpath[full_path] = filename

    ids, labels, parents, values, last_frame_paths = [], [], [], [], []
    for fullpath, val in prefix_counts.items():
        ids.append(fullpath)
        labels.append(fullpath.split("/")[-1])
        parent = fullpath.rsplit("/", 1)[0] if "/" in fullpath else ""
        parents.append(parent)
        values.append(val)
        last_frame_paths.append(last_frame_fullpath.get(fullpath, ""))

    return ids, labels, parents, values, last_frame_paths


def create_click_interactive_icicle(data, output_file="interactive_callsites_click.html", mode="empty"):
    ids, labels, parents, values, last_frame_paths = build_icicle(data)
    # Build a lightweight metadata map for export
    leaf_meta = {}
    for k, v in data.items():
        meta = {}
        if 'ops' in v: meta['ops'] = v['ops']
        if 'callstacks' in v: meta['callstacks'] = v['callstacks']
        if 'stream_counts' in v: meta['stream_counts'] = v['stream_counts']
        if meta:
            leaf_meta[k] = meta

    leaf_meta_json = json.dumps(leaf_meta)


    unique_files = sorted({p for p in last_frame_paths if p})
    def extract_func(full_id):
        last_label = full_id.split("/")[-1]
        return last_label.split(" (")[0]
    unique_functions = sorted({extract_func(i) for i in ids if extract_func(i)})

    fig = go.Figure()
    icicle_trace = go.Icicle(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        customdata=last_frame_paths,
        branchvalues="total",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>",
        name="Call Stack",
        pathbar=dict(visible=True)
    )
    fig.add_trace(icicle_trace)
    fig.update_layout(
        title="Click Interactive Callsite Forest - Click to Select",
        margin=dict(t=50, l=25, r=25, b=25),
        height=850,
        showlegend=False,
        dragmode="zoom"
    )

    files_json = json.dumps(unique_files)
    functions_json = json.dumps(unique_functions)
    leaf_meta_json = json.dumps(leaf_meta)

    # NOTE: Use an f-string; double {{ }} for JS braces.
    custom_js = f"""
    <script>
    // ---- STATE ----
    const initialMode = "{mode}";
    const LEAF_META = /* injected from Python */ {leaf_meta_json};

    let selectedPaths = new Set();
    let history = [];
    let isSelectionMode = false;

    let ORIGINAL = null;
    let ORIGINAL_RIGHT_MARGIN = null;
    let ORIGINAL_TOP_MARGIN = null;

    // ---- LAYOUT HELPERS (search / right panel margins) ----
    function adjustPlotMarginForSearch() {{
        const gd = document.querySelector('.plotly-graph-div');
        if (!gd || !gd.layout) return;
        const panel = document.getElementById('search-controls');
        if (ORIGINAL_TOP_MARGIN === null) {{
            ORIGINAL_TOP_MARGIN = (gd.layout.margin && gd.layout.margin.t) ? gd.layout.margin.t : 50;
        }}
        let targetTop = ORIGINAL_TOP_MARGIN;
        if (panel && panel.style.display !== 'none') {{
            const rect = panel.getBoundingClientRect();
            targetTop = Math.max(ORIGINAL_TOP_MARGIN, Math.ceil(rect.height) + 12);
        }}
        Plotly.relayout(gd, {{ 'margin.t': targetTop }});
    }}

    function normalizeStr(s) {{
        return (s || '').trim().toLowerCase();
    }}
    function getTrace() {{
        const gd = document.querySelector('.plotly-graph-div');
        return (gd && gd.data && gd.data[0]) ? gd.data[0] : null;
    }}

    function makeSearchCollapsibleAndResizable() {{
        const panel = document.getElementById('search-controls');
        if (!panel) return;

        panel.style.position = 'fixed';
        panel.style.top = '8px';
        panel.style.left = '50%';
        panel.style.transform = 'translateX(-50%)';
        panel.style.width = 'min(1100px, 92vw)';
        panel.style.zIndex = '1100';
        panel.style.background = 'white';
        panel.style.border = '1px solid #ccc';
        panel.style.borderRadius = '8px';
        panel.style.boxShadow = '0 2px 10px rgba(0,0,0,0.1)';
        panel.style.padding = '8px';
        panel.style.resize = 'vertical';
        panel.style.overflow = 'auto';

        const header = document.createElement('div');
        header.style.cssText = 'display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;';
        header.innerHTML = `
            <div style="font-weight:600;">Search</div>
            <div style="display:flex; gap:6px;">
            <button id="search-toggle" style="padding:4px 8px;">Hide</button>
            <button id="search-close-btn" style="padding:4px 8px;">Close</button>
            </div>
        `;
        panel.insertBefore(header, panel.firstChild);

        const body = document.createElement('div');
        while (header.nextSibling) body.appendChild(header.nextSibling);
        panel.appendChild(body);

        const toggleBtn = header.querySelector('#search-toggle');
        let collapsed = false;
        toggleBtn.onclick = () => {{
            collapsed = !collapsed;
            body.style.display = collapsed ? 'none' : 'block';
            toggleBtn.textContent = collapsed ? 'Show' : 'Hide';
            adjustPlotMarginForSearch();
        }};

        header.querySelector('#search-close-btn').onclick = () => {{
            panel.style.display = 'none';
            adjustPlotMarginForSearch();
        }};

        if ('ResizeObserver' in window) {{
            const ro = new ResizeObserver(() => adjustPlotMarginForSearch());
            ro.observe(panel);
        }} else {{
            panel.addEventListener('mouseup', adjustPlotMarginForSearch);
        }};
        adjustPlotMarginForSearch();
    }}

    function adjustPlotMarginForPanel() {{
        const gd = document.querySelector('.plotly-graph-div');
        if (!gd || !gd.layout) return;
        const panel = document.getElementById('selection-controls');
        if (ORIGINAL_RIGHT_MARGIN === null) {{
            ORIGINAL_RIGHT_MARGIN = (gd.layout.margin && gd.layout.margin.r) ? gd.layout.margin.r : 25;
        }}
        let targetRight = ORIGINAL_RIGHT_MARGIN;
        if (panel && panel.style.display !== 'none') {{
            const rect = panel.getBoundingClientRect();
            targetRight = Math.max(ORIGINAL_RIGHT_MARGIN, Math.ceil(rect.width) + 24);
        }}
        Plotly.relayout(gd, {{ 'margin.r': targetRight }});
    }}

    function makePanelCollapsibleAndResizable() {{
        const panel = document.getElementById('selection-controls');
        if (!panel) return;

        const header = document.createElement('div');
        header.style.cssText = 'display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;';
        header.innerHTML = `
            <div style="font-weight:600;">Selection</div>
            <div style="display:flex; gap:6px;">
            <button id="panel-toggle" style="padding:4px 8px;">Hide</button>
            </div>
        `;
        panel.insertBefore(header, panel.firstChild);

        panel.style.resize = 'horizontal';
        panel.style.overflow = 'auto';
        panel.style.maxWidth = '50vw';
        panel.style.minWidth = '240px';

        const body = document.createElement('div');
        while (header.nextSibling) body.appendChild(header.nextSibling);
        panel.appendChild(body);

        const toggleBtn = header.querySelector('#panel-toggle');
        let collapsed = false;
        toggleBtn.onclick = () => {{
            collapsed = !collapsed;
            body.style.display = collapsed ? 'none' : 'block';
            toggleBtn.textContent = collapsed ? 'Show' : 'Hide';
            adjustPlotMarginForPanel();
        }};

        if ('ResizeObserver' in window) {{
            const ro = new ResizeObserver(() => {{ adjustPlotMarginForPanel() }});
            ro.observe(panel);
        }} else {{
            panel.addEventListener('mouseup', adjustPlotMarginForPanel);
        }}
        adjustPlotMarginForPanel();
    }}

    // ---- ROOT CAPTURE / RESET ----
    function captureOriginals() {{
        const plotDiv = document.querySelector('.plotly-graph-div');
        if (!plotDiv || !plotDiv.data || !plotDiv.data[0]) return;
        const t = plotDiv.data[0];
        ORIGINAL = {{
            ids: (t.ids || []).slice(),
            labels: (t.labels || []).slice(),
            parents: (t.parents || []).slice(),
            values: (t.values || []).slice(),
            customdata: (t.customdata || []).slice(),
        }};
    }}

    function resetToOriginalRoot() {{
        const gd = document.querySelector('.plotly-graph-div');
        if (!gd || !gd.data || !gd.data[0] || !ORIGINAL) return;
        Plotly.restyle(gd, {{
            ids: [ORIGINAL.ids],
            labels: [ORIGINAL.labels],
            parents: [ORIGINAL.parents],
            values: [ORIGINAL.values],
            customdata: [ORIGINAL.customdata],
            root: [""],
            maxdepth: [null],
        }}, [0]);
    }}

    // ---- SELECTION HELPERS ----
    function pushHistory() {{
        history.push(new Set(selectedPaths));
        if (history.length > 50) history.shift();
    }}

    function selectAllLeaves() {{
        const plotDiv = document.querySelector('.plotly-graph-div');
        if (!plotDiv || !plotDiv.data || !plotDiv.data[0]) return;
        const trace = plotDiv.data[0];
        const ids = trace.ids || [];
        const parents = trace.parents || [];
        ids.forEach(id => {{
            if (!parents.includes(id)) {{
                selectedPaths.add(id);
            }}
        }});
    }}

    function updateNodeColors() {{
        const plotDiv = document.querySelector('.plotly-graph-div');
        if (!plotDiv || !plotDiv.data || !plotDiv.data[0]) return;
        const trace = plotDiv.data[0];
        const ids = trace.ids || [];
        const colors = ids.map(id => selectedPaths.has(id) ? '#4CAF50' : '#1f77b4');
        Plotly.restyle(plotDiv, {{ 'marker.colors': [colors] }}, [0]);
    }}

    function updateSelectionDisplay() {{
        const container = document.getElementById('selection-display');
        if (!container) return;
        container.innerHTML = '<h3>Selected Layers:</h3>';
        if (selectedPaths.size === 0) {{
            container.innerHTML += '<p>No layers selected. Click nodes to select.</p>';
        }} else {{
            const ul = document.createElement('ul');
            selectedPaths.forEach(p => {{
                const short = p.split('/').pop();
                const li = document.createElement('li');
                li.textContent = short;
                ul.appendChild(li);
            }});
            container.appendChild(ul);
            const summary = document.createElement('div');
            summary.textContent = `Total: ${{selectedPaths.size}} selected`;
            summary.style.marginTop = '6px';
            container.appendChild(summary);
        }}
        updateNodeColors();
    }}

    function toggleSelectionMode() {{
        const plotDiv = document.querySelector('.plotly-graph-div');
        isSelectionMode = !isSelectionMode;
        const btn = document.getElementById('selection-mode-btn');
        if (isSelectionMode) {{
            if (btn) {{ btn.textContent = 'Exit Selection Mode'; btn.style.background = '#f44336'; }}
            if (plotDiv) Plotly.relayout(plotDiv, {{ dragmode: false }});
        }} else {{
            if (btn) {{ btn.textContent = 'Enter Selection Mode'; btn.style.background = '#4CAF50'; }}
            if (plotDiv) Plotly.relayout(plotDiv, {{ dragmode: 'zoom' }});
        }}
        updateSelectionDisplay();
        adjustPlotMarginForPanel();
    }}

    function onIcicleClick(evt) {{
        if (evt && evt.points && evt.points.length) {{
            const point = evt.points[0];
            const path = point.id || point.label;
            if (isSelectionMode && path) {{
                pushHistory();
                if (selectedPaths.has(path)) selectedPaths.delete(path);
                else selectedPaths.add(path);
                updateSelectionDisplay();
                return false; // suppress drilldown
            }}
        }}
    }}

    // helper: isLeaf in current (full) dataset
    function isLeafId(id, allIds) {{
    const prefix = id + '/';
    for (let i = 0; i < allIds.length; i++) {{
        if (allIds[i].startsWith(prefix)) return false;
    }}
    return true;
    }}

    // gather leaves under a node (including itself if already a leaf)
    function leafDescendants(rootId, allIds) {{
    const leaves = [];
    const prefix = rootId + '/';
    for (let i = 0; i < allIds.length; i++) {{
        const id = allIds[i];
        if (id === rootId || id.startsWith(prefix)) {{
        // is this id a leaf?
        if (isLeafId(id, allIds)) leaves.push(id);
        }}
    }}
    return leaves;
    }}

    // parse helper (unchanged)
    function parseIdToKey(id, trace) {{
    const lastLabel = id.split('/').pop();
    const m = lastLabel.match(/^(.+) \\(([^:]+):(\\d+)\\)$/);
    const func = m ? m[1] : lastLabel;
    const filename = m ? m[2] : '';
    const lineno = m ? parseInt(m[3]) : null;
    const idx = trace.ids.indexOf(id);
    const fullfile = trace.customdata[idx] || "";
    const key = fullfile + ':' + lineno + ' in ' + func;
    return {{ key, idx, fullfile, lineno, func }};
    }}

    function exportSelection() {{
    const gd = document.querySelector('.plotly-graph-div');
    if (!gd || !gd.data || !gd.data[0]) return;
    const t = gd.data[0];

    // IMPORTANT: use ORIGINAL.ids so we can see the full tree (not a zoomed subtree)
    const allIds = ORIGINAL && ORIGINAL.ids ? ORIGINAL.ids : (t.ids || []);

    const exportData = {{}};

    // expand each selected id to its leaf descendants
    const expandedLeafIds = new Set();
    selectedPaths.forEach(id => {{
        leafDescendants(id, allIds).forEach(leaf => expandedLeafIds.add(leaf));
    }});

    // nothing selected? bail politely
    if (expandedLeafIds.size === 0) return;

    expandedLeafIds.forEach(id => {{
        const {{ key, idx, fullfile, lineno, func }} = parseIdToKey(id, t);
        const count = (t.values && idx >= 0) ? t.values[idx] : 0;

        // pull original metadata if present
        const extra = LEAF_META[key] || {{}};

        // if we injected original callstacks into LEAF_META, use them; otherwise synthesize a 1-frame stack
        const callstacks = extra.callstacks
        ? extra.callstacks
        : [[[ fullfile, lineno, func ]]];

        exportData[key] = {{
        count,
        callstacks,
        ...(extra.ops ? {{ ops: extra.ops }} : {{}}),
        ...(extra.streams ? {{ streams: extra.streams }} : {{}}),
        ...(extra.stream_counts ? {{ stream_counts: extra.stream_counts }} : {{}}),
        }};
    }});

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {{ type: 'application/json' }});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'selected_wrapping_layers.json';
    a.click();
    URL.revokeObjectURL(a.href);
    }}


    function clearSelection() {{
        pushHistory();
        selectedPaths.clear();
        updateSelectionDisplay();
    }}

    function revertSelection() {{
        if (history.length === 0) return;
        selectedPaths = history.pop();
        updateSelectionDisplay();
    }}

    // ---- WILDCARD MATCHING (fixed) ----
    // Turn a shell-like wildcard pattern (* ?) into a JS RegExp.
    // Example: "*.py" -> /^.*\\.py$/   "*backward*" -> /^.*backward.*$/
    function wildcardToRegex(pat, insensitive) {{
        // Escape regex specials
        const escaped = pat.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&');
        // Convert wildcards
        const regexBody = '^' + escaped.replace(/\\\\\\*/g, '.*').replace(/\\\\\\?/g, '.') + '$';
        return new RegExp(regexBody, insensitive ? 'i' : '');
    }}

    // Extract clean function name from an id's last label: "func (file.py:123)" -> "func"
    function funcNameFromId(id) {{
        const last = id.split('/').pop() || id;
        const m = last.match(/^(.+) \\([^:]+:\\d+\\)$/);
        return m ? m[1] : last;
    }}

    // ---- FILE/FUNC PATTERN OPERATIONS (using fixed wildcardToRegex) ----
    function addByFilePattern() {{
        const raw = document.getElementById('file-pattern').value.trim();
        if (!raw) return;
        pushHistory();
        const insensitive = !document.getElementById('search-case').checked;
        const re = wildcardToRegex(raw, insensitive);

        const plotDiv = document.querySelector('.plotly-graph-div');
        if (!plotDiv || !plotDiv.data || !plotDiv.data[0]) return;
        const t = plotDiv.data[0];

        (t.ids || []).forEach((id, idx) => {{
            const fullfile = (t.customdata && t.customdata[idx]) || '';
            if (re.test(fullfile)) selectedPaths.add(id);
        }});
        updateSelectionDisplay();
    }}

    function removeByFilePattern() {{
        const raw = document.getElementById('file-pattern').value.trim();
        if (!raw) return;
        pushHistory();
        const insensitive = !document.getElementById('search-case').checked;
        const re = wildcardToRegex(raw, insensitive);

        const plotDiv = document.querySelector('.plotly-graph-div');
        if (!plotDiv || !plotDiv.data || !plotDiv.data[0]) return;
        const t = plotDiv.data[0];

        (t.ids || []).forEach((id, idx) => {{
            const fullfile = (t.customdata && t.customdata[idx]) || '';
            if (re.test(fullfile)) selectedPaths.delete(id);
        }});
        updateSelectionDisplay();
    }}

    function addByFunctionPattern() {{
        const raw = document.getElementById('function-pattern').value.trim();
        if (!raw) return;
        pushHistory();
        const insensitive = !document.getElementById('search-case').checked;
        const re = wildcardToRegex(raw, insensitive);

        const plotDiv = document.querySelector('.plotly-graph-div');
        if (!plotDiv || !plotDiv.data || !plotDiv.data[0]) return;
        const t = plotDiv.data[0];

        (t.ids || []).forEach(id => {{
            const funcName = funcNameFromId(id);
            if (re.test(funcName)) selectedPaths.add(id);
        }});
        updateSelectionDisplay();
    }}

    function removeByFunctionPattern() {{
        const raw = document.getElementById('function-pattern').value.trim();
        if (!raw) return;
        pushHistory();
        const insensitive = !document.getElementById('search-case').checked;
        const re = wildcardToRegex(raw, insensitive);

        const plotDiv = document.querySelector('.plotly-graph-div');
        if (!plotDiv || !plotDiv.data || !plotDiv.data[0]) return;
        const t = plotDiv.data[0];

        (t.ids || []).forEach(id => {{
            const funcName = funcNameFromId(id);
            if (re.test(funcName)) selectedPaths.delete(id);
        }});
        updateSelectionDisplay();
    }}
    function addByFileDropdown() {{
        const sel = document.getElementById('file-dropdown');
        if (!sel) return;
        const chosen = Array.from(sel.selectedOptions)
            .filter(o => !o.disabled)
            .map(o => normalizeStr(o.value));
        if (!chosen.length) return;

        pushHistory();
        const t = getTrace(); if (!t) return;

        (t.ids || []).forEach((id, idx) => {{
            const fullfile = normalizeStr((t.customdata && t.customdata[idx]) || '');
            if (chosen.includes(fullfile)) selectedPaths.add(id);
        }});
        updateSelectionDisplay();
    }}

    function removeByFileDropdown() {{
        const sel = document.getElementById('file-dropdown');
        if (!sel) return;
        const chosen = Array.from(sel.selectedOptions)
            .filter(o => !o.disabled)
            .map(o => normalizeStr(o.value));
        if (!chosen.length) return;

        pushHistory();
        const t = getTrace(); if (!t) return;

        (t.ids || []).forEach((id, idx) => {{
            const fullfile = normalizeStr((t.customdata && t.customdata[idx]) || '');
            if (chosen.includes(fullfile)) selectedPaths.delete(id);
        }});
        updateSelectionDisplay();
    }}

    function addByFunctionDropdown() {{
        const sel = document.getElementById('function-dropdown');
        if (!sel) return;
        const chosen = Array.from(sel.selectedOptions)
            .filter(o => !o.disabled)
            .map(o => normalizeStr(o.value));
        if (!chosen.length) return;

        pushHistory();
        const t = getTrace(); if (!t) return;

        (t.ids || []).forEach(id => {{
            const funcName = normalizeStr(funcNameFromId(id));
            if (chosen.includes(funcName)) selectedPaths.add(id);
        }});
        updateSelectionDisplay();
    }}

    function removeByFunctionDropdown() {{
        const sel = document.getElementById('function-dropdown');
        if (!sel) return;
        const chosen = Array.from(sel.selectedOptions)
            .filter(o => !o.disabled)
            .map(o => normalizeStr(o.value));
        if (!chosen.length) return;

        pushHistory();
        const t = getTrace(); if (!t) return;

        (t.ids || []).forEach(id => {{
            const funcName = normalizeStr(funcNameFromId(id));
            if (chosen.includes(funcName)) selectedPaths.delete(id);
        }});
        updateSelectionDisplay();
    }}


    // ---- SEARCH (unchanged UX, uses root reset) ----
    let searchOpen = false;
    let searchMatches = [];
    let searchIndex = -1;
    let caseSensitive = false;

    function openSearch() {{
        ensureSearchUI();
        const panel = document.getElementById('search-controls');
        const input = document.getElementById('search-input');
        const status = document.getElementById('search-status');
        searchOpen = true;
        panel.style.display = 'block';
        adjustPlotMarginForSearch();

        status.textContent = '';
        input.focus();
        input.select();
        runSearch();
    }}

    function closeSearch() {{
        searchOpen = false;
        const panel = document.getElementById('search-controls');
        if (panel) panel.style.display = 'none';
        adjustPlotMarginForSearch();
    }}

    function resetRoot() {{ resetToOriginalRoot(); }}

    function ensureSearchUI() {{
        if (document.getElementById('search-controls')) return;

        const wrap = document.createElement('div');
        wrap.id = 'search-controls';
        wrap.style.cssText = `
            position: fixed; z-index: 1100;
            background: white; border: 1px solid #ccc; border-radius: 6px;
            padding: 8px; display: none; box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            font-family: system-ui,-apple-system,BlinkMacSystemFont,sans-serif;
        `;
        wrap.innerHTML = `
          <div style="display:flex; align-items:center; gap:6px;">
            <input id="search-input" type="text" placeholder="Search function or file..."
                   style="width:300px; padding:6px 8px;" />
            <label style="font-size:12px; display:flex; align-items:center; gap:4px;">
                <input id="search-case" type="checkbox" />
                Case sensitive
            </label>
            <button id="search-prev" title="Shift+Enter" style="padding:6px 8px;">Prev</button>
            <button id="search-next" title="Enter" style="padding:6px 8px;">Next</button>
            <button id="search-reset" style="padding:6px 8px;">Reset root</button>
            <button id="search-close" style="padding:6px 8px; background:#f44336; color:#fff;">Close</button>
          </div>
          <div id="search-status" style="margin-top:4px; font-size:12px; color:#555;"></div>
        `;
        document.body.appendChild(wrap);
        makeSearchCollapsibleAndResizable();

        document.getElementById('search-next').addEventListener('click', () => jumpToMatch(+1));
        document.getElementById('search-prev').addEventListener('click', () => jumpToMatch(-1));
        document.getElementById('search-close').addEventListener('click', closeSearch);
        document.getElementById('search-reset').addEventListener('click', resetRoot);
        document.getElementById('search-input').addEventListener('input', runSearch);
        document.getElementById('search-case').addEventListener('change', () => {{
            caseSensitive = document.getElementById('search-case').checked;
            runSearch();
        }});
    }}

    function runSearch() {{
        const plotDiv = document.querySelector('.plotly-graph-div');
        if (!plotDiv || !plotDiv.data || !plotDiv.data[0]) return;

        const t = plotDiv.data[0];
        const ids = t.ids || [];
        const labels = t.labels || [];
        const files = t.customdata || [];
        const qRaw = (document.getElementById('search-input').value || '');
        const q = caseSensitive ? qRaw : qRaw.toLowerCase();

        searchMatches = [];
        searchIndex = -1;

        for (let i = 0; i < ids.length; i++) {{
            const label = labels[i] || '';
            const file = files[i] || '';
            const hayFunc = caseSensitive ? label : label.toLowerCase();
            const hayFile = caseSensitive ? file : file.toLowerCase();
            const hayId   = caseSensitive ? ids[i] : (ids[i] || '').toLowerCase();
            if (!q) continue;
            if (hayFunc.includes(q) || hayFile.includes(q) || hayId.includes(q)) {{
                searchMatches.push(i);
            }}
        }}

        const status = document.getElementById('search-status');
        if (!q) {{
            status.textContent = 'Type to search function name, id, or source file.';
            return;
        }}
        if (searchMatches.length === 0) {{
            status.textContent = `No results for "${{qRaw}}".`;
            return;
        }}
        status.textContent = `Found ${{searchMatches.length}} result(s) for "${{qRaw}}". Press Enter to jump.`;
        jumpToMatch(+1);
    }}

    function jumpToMatch(direction) {{
        if (searchMatches.length === 0) return;

        if (searchIndex === -1) {{
            searchIndex = (direction > 0) ? 0 : searchMatches.length - 1;
        }} else {{
            searchIndex = (searchIndex + direction + searchMatches.length) % searchMatches.length;
        }}

        const gd = document.querySelector('.plotly-graph-div');
        if (!gd || !gd.data || !gd.data[0]) return;
        const t = gd.data[0];

        const idx = searchMatches[searchIndex];
        const id  = t.ids[idx];

        Plotly.restyle(gd, {{ level: [id], maxdepth: [null] }}, [0]).then(() => {{
            return Plotly.update(gd, {{}}, {{ transition: {{ duration: 350, easing: 'cubic-in-out' }}}});
        }});

        const status = document.getElementById('search-status');
        const label  = (t.labels && t.labels[idx]) || id;
        const file   = (t.customdata && t.customdata[idx]) || '';
        status.textContent = `Result ${{searchIndex + 1}} of ${{searchMatches.length}}: ${{label}} — ${{file}}`;
    }}

    // Intercept Cmd/Ctrl+F to open custom search
    window.addEventListener('keydown', (e) => {{
        const isMac = navigator.platform.toUpperCase().includes('MAC');
        const modKey = isMac ? e.metaKey : e.ctrlKey;
        if (modKey && e.key.toLowerCase() === 'f') {{
            e.preventDefault();
            openSearch();
        }} else if (searchOpen && e.key === 'Enter') {{
            e.preventDefault();
            if (e.shiftKey) jumpToMatch(-1);
            else jumpToMatch(+1);
        }} else if (searchOpen && e.key === 'Escape') {{
            closeSearch();
        }}
    }});

    // ---- BOOTSTRAP ----
    document.addEventListener('DOMContentLoaded', () => {{
        const attach = () => {{
            const plotDiv = document.querySelector('.plotly-graph-div');
            if (plotDiv && plotDiv.data && plotDiv.data.length) {{
                plotDiv.on('plotly_icicleclick', onIcicleClick);
                captureOriginals();

                if (initialMode === "default") {{
                    pushHistory();
                    selectAllLeaves();
                }}

                if (!document.getElementById('selection-controls')) {{
                    const controls = document.createElement('div');
                    controls.id = 'selection-controls';
                    controls.style.cssText = `
                        position: fixed;
                        top: 10px;
                        right: 10px;
                        background: white;
                        padding: 15px;
                        border: 1px solid #ccc;
                        border-radius: 5px;
                        z-index: 1000;
                        max-width: 560px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                        font-family: system-ui,-apple-system,BlinkMacSystemFont,sans-serif;
                    `;
                    const fileList = {json.dumps(unique_files)};
                    const functionList = {json.dumps(unique_functions)};
                    const fileOptions = fileList
                        .map(f => '<option value="' + f + '">' + f + '</option>')
                        .join('');
                    const functionOptions = functionList
                        .map(fn => '<option value="' + fn + '">' + fn + '</option>')
                        .join('');

                    controls.innerHTML = `
                        <h4 style="margin:5px 0;">Click Selection</h4>
                        <div style="margin:8px 0;">
                            <button id="selection-mode-btn" onclick="toggleSelectionMode()" style="width:100%; padding:8px; background:#4CAF50; color:white; border:none; border-radius:3px;">
                                Enter Selection Mode
                            </button>
                        </div>
                        <div style="font-size:12px; color:#555; margin-bottom:10px;">
                            • Toggle individual nodes<br>
                            • Add / Remove by file/function pattern or dropdown<br>
                            • Revert rolls back last mutation
                        </div>

                        <div style="margin:5px 0; border-top:1px solid #eee; padding-top:8px;">
                            <div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:6px;">
                                <div style="flex:1; min-width:200px;">
                                    <label style="display:block;">File Pattern:</label>
                                    <input type="text" id="file-pattern" placeholder="*module*.py" style="width:100%; box-sizing:border-box;">
                                    <div style="margin-top:4px;">
                                        <button onclick="addByFilePattern()" style="padding:4px 8px;">Add</button>
                                        <button onclick="removeByFilePattern()" style="padding:4px 8px;">Remove</button>
                                    </div>
                                </div>
                                <div style="flex:1; min-width:200px;">
                                    <label style="display:block;">Function Pattern:</label>
                                    <input type="text" id="function-pattern" placeholder="*forward*" style="width:100%; box-sizing:border-box;">
                                    <div style="margin-top:4px;">
                                        <button onclick="addByFunctionPattern()" style="padding:4px 8px;">Add</button>
                                        <button onclick="removeByFunctionPattern()" style="padding:4px 8px;">Remove</button>
                                    </div>
                                </div>
                            </div>
                            <div style="margin-bottom:6px;">
                                <div style="margin-bottom:8px;">
                                    <label style="display:block;">File Dropdown:</label>
                                    <select id="file-dropdown" multiple size="4" style="width:100%; box-sizing:border-box;">
                                        <option disabled>Hold Ctrl/Cmd to multi-select</option>
                                    </select>
                                    <div style="margin-top:4px;">
                                        <button onclick="addByFileDropdown()" style="padding:4px 8px;">Add</button>
                                        <button onclick="removeByFileDropdown()" style="padding:4px 8px;">Remove</button>
                                    </div>
                                </div>
                                <div style="margin-bottom:8px;">
                                    <label style="display:block;">Function Dropdown:</label>
                                    <select id="function-dropdown" multiple size="4" style="width:100%; box-sizing:border-box;">
                                        <option disabled>Hold Ctrl/Cmd to multi-select</option>
                                    </select>
                                    <div style="margin-top:4px;">
                                        <button onclick="addByFunctionDropdown()" style="padding:4px 8px;">Add</button>
                                        <button onclick="removeByFunctionDropdown()" style="padding:4px 8px;">Remove</button>
                                    </div>
                                </div>
                            </div>
                            <div style="margin-bottom:6px;">
                                <button onclick="revertSelection()" style="padding:6px 12px; background:#795548; color:white; border:none; border-radius:3px;">Revert</button>
                                <button onclick="clearSelection()" style="padding:6px 12px; margin-left:6px; background:#f44336; color:white; border:none; border-radius:3px;">Clear All</button>
                                <button onclick="exportSelection()" style="padding:6px 12px; margin-left:6px;">Export</button>
                            </div>
                        </div>
                        <div id="selection-display" style="margin-top:10px; max-height:220px; overflow:auto; border-top:1px solid #eee; padding-top:8px;"></div>
                    `;
                    document.body.appendChild(controls);
                    document.getElementById('file-dropdown').insertAdjacentHTML('beforeend', fileOptions);
                    document.getElementById('function-dropdown').insertAdjacentHTML('beforeend', functionOptions);
                    updateSelectionDisplay();
                    makePanelCollapsibleAndResizable();
                    adjustPlotMarginForPanel();
                    makeSearchCollapsibleAndResizable();
                    adjustPlotMarginForSearch();
                }}

                ensureSearchUI();
            }} else {{
                setTimeout(attach, 100);
            }}
        }};
        attach();
    }});
    </script>
    """

    fig.write_html(output_file, include_plotlyjs="cdn")
    with open(output_file, "r", encoding="utf-8") as f:
        html_content = f.read()
    html_content = html_content.replace("</body>", custom_js + "\n</body>")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Click interactive visualization saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a click interactive callsite icicle chart using Plotly's click events.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input JSON file (callsite dump).")
    parser.add_argument("-o", "--output", default="interactive_callsites_click.html",
                        help="Path for the output HTML file.")
    parser.add_argument("-m", "--mode", choices=["empty","default"], default="default",
                        help="Initial selection: 'empty' (none) or 'default' (all leaf nodes).")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    create_click_interactive_icicle(data, args.output, mode=args.mode)

if __name__ == "__main__":
    main()
