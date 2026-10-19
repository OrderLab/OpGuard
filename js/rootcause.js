(() => {
  const module = document.querySelector('#rootcause');
  if (!module) return;
  const local = ['localhost', '127.0.0.1'].includes(location.hostname);
  const endpoint = local ? '/api/rootcause' : (window.OPGUARD_DEMO_API || '');
  function appendCitations(container, text) {
    for (const part of text.split(/(run[12]\/deepspeed\/runtime\/zero\/stage_1_and_2\.py:\d+)/g)) {
      const match = /^(run[12])\/deepspeed\/runtime\/zero\/stage_1_and_2\.py:(\d+)$/.exec(part);
      const line = match && Number(match[2]);
      if (match && line >= 1 && line <= (match[1] === 'run1' ? 2565 : 2567)) {
        const link = document.createElement('a');
        link.href = `demo/sources/${match[1]}-stage_1_and_2.html#L${line}`;
        link.target = '_blank'; link.rel = 'noopener'; link.textContent = part;
        container.append(link);
      } else container.append(document.createTextNode(part));
    }
  }
  function renderAnswer(container, text) {
    container.replaceChildren();
    // Small, safe inline formatter: model text never enters innerHTML.
    for (const part of text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g)) {
      if (part.startsWith('**') && part.endsWith('**')) {
        const strong = document.createElement('strong');
        appendCitations(strong, part.slice(2, -2)); container.append(strong);
      } else if (part.startsWith('`') && part.endsWith('`')) {
        const code = document.createElement('code');
        appendCitations(code, part.slice(1, -1)); container.append(code);
      } else appendCitations(container, part);
    }
  }
  for (const button of module.querySelectorAll('[data-rootcause]')) {
    const result = document.getElementById(`${button.dataset.rootcause}-answer`);
    let retryAt = 0;
    button.addEventListener('click', async () => {
      result.hidden = false;
      if (retryAt > Date.now()) {
        result.textContent = `Please try again in ${Math.ceil((retryAt - Date.now()) / 1000)} seconds.`;
        return;
      }
      if (!endpoint) { result.textContent = 'Live analysis is not connected on this deployment yet.'; return; }
      button.disabled = true;
      button.textContent = 'Analyzing…';
      result.textContent = 'Reading the evidence…';
      result.setAttribute('aria-busy', 'true');
      try {
        const response = await fetch(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ arm: button.dataset.rootcause }), signal: AbortSignal.timeout(75000) });
        let data;
        if (response.headers.get('content-type')?.includes('application/x-ndjson')) {
          const reader = response.body.getReader(); const decoder = new TextDecoder(); let pending = '';
          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            pending += decoder.decode(value, { stream: true });
            let end;
            while ((end = pending.indexOf('\n')) >= 0) {
              const update = JSON.parse(pending.slice(0, end)); pending = pending.slice(end + 1);
              if (update.progress) result.textContent = update.progress;
              else data = update;
            }
          }
          if (!data) throw new Error('Incomplete response');
        } else data = await response.json();
        if (!response.ok || data.error) {
          if ((response.status === 429 || data.status === 429) && data.retryAfter) retryAt = Date.now() + data.retryAfter * 1000;
          result.textContent = (data.error || 'Analysis failed. Please try again.') + (data.retryAfter ? ` Retry in ${data.retryAfter} seconds.` : '');
        } else {
          renderAnswer(result, data.answer);
        }
      } catch { result.textContent = 'Could not reach the model. Please try again.'; }
      finally {
        button.disabled = false;
        button.textContent = 'Get rootcause';
        result.setAttribute('aria-busy', 'false');
      }
    });
  }
  // README links open the same page and run only the explicitly selected arm.
  const requestedArm = new URLSearchParams(location.search).get('analyze');
  if (['trace', 'opguard'].includes(requestedArm)) {
    module.querySelector(`[data-rootcause="${requestedArm}"]`).click();
  }
})();
