(() => {
  const section = document.getElementById('get-started');
  if (!section) return;
  const status = section.querySelector('[data-copy-status]');
  const tabs = Array.from(section.querySelectorAll('[role="tab"]'));
  const panels = tabs.map(tab => document.getElementById(tab.getAttribute('aria-controls')));
  const stepStatus = document.getElementById('quickstart-step-status');
  const deck = document.getElementById('quickstart-deck');
  const playButton = document.getElementById('quickstart-play');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let step = 0;
  let playing = !reducedMotion.matches;
  let inView = false;
  let timer;

  function scheduleStep() {
    window.clearTimeout(timer);
    if (!playing || !inView || document.hidden) return;
    timer = window.setTimeout(() => {
      showStep(step + 1, { announce: false });
      scheduleStep();
    }, 8000);
  }

  function setPlaying(value) {
    playing = value;
    playButton.textContent = playing ? 'Pause' : 'Play';
    playButton.setAttribute('aria-label', `${playing ? 'Pause' : 'Play'} automatic walkthrough`);
    playButton.setAttribute('aria-pressed', String(playing));
    if (playing) stepStatus.textContent = '';
    scheduleStep();
  }

  function chooseStep(next, options) {
    setPlaying(false);
    showStep(next, options);
  }

  function showStep(next, { focus = false, announce = true } = {}) {
    step = (next + tabs.length) % tabs.length;
    tabs.forEach((tab, index) => {
      const active = index === step;
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
      panels[index].hidden = !active;
    });
    status.textContent = '';
    if (announce) stepStatus.textContent = `Step ${step + 1} of ${tabs.length}: ${tabs[step].textContent.trim()}`;
    if (focus) tabs[step].focus();
  }

  if (tabs.length && panels.every(Boolean)) {
    document.getElementById('quickstart-prev').addEventListener('click', () => chooseStep(step - 1));
    document.getElementById('quickstart-next').addEventListener('click', () => chooseStep(step + 1));
    playButton.addEventListener('click', () => setPlaying(!playing));
    // Keep the current step while someone reads, selects text or copies a command.
    for (const eventName of ['focusin', 'pointerdown']) {
      deck.addEventListener(eventName, event => {
        if (!playButton.contains(event.target)) setPlaying(false);
      });
    }
    tabs.forEach((tab, index) => {
      tab.addEventListener('click', () => chooseStep(index));
      tab.addEventListener('keydown', event => {
        const next = { ArrowLeft: index - 1, ArrowRight: index + 1, Home: 0, End: tabs.length - 1 }[event.key];
        if (next === undefined) return;
        event.preventDefault();
        chooseStep(next, { focus: true });
      });
    });
    showStep(0, { announce: false });
    setPlaying(playing);
    new IntersectionObserver(([entry]) => {
      inView = entry.isIntersecting && entry.intersectionRatio >= 0.25;
      scheduleStep();
    }, { threshold: 0.25 }).observe(deck);
    document.addEventListener('visibilitychange', scheduleStep);
    reducedMotion.addEventListener('change', event => {
      if (event.matches) setPlaying(false);
    });
  }

  for (const button of section.querySelectorAll('[data-copy-code]')) {
    const label = button.textContent;
    let resetTimer;
    button.addEventListener('click', async () => {
      const code = document.getElementById(button.dataset.copyCode);
      if (!code) return;
      try {
        await navigator.clipboard.writeText(code.textContent.trim());
        button.textContent = 'Copied';
        status.textContent = 'Copied to clipboard.';
        window.clearTimeout(resetTimer);
        resetTimer = window.setTimeout(() => { button.textContent = label; }, 1800);
      } catch {
        const range = document.createRange();
        range.selectNodeContents(code);
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        status.textContent = 'Text selected. Use your keyboard to copy it.';
      }
    });
  }
})();
