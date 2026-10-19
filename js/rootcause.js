(() => {
  const module = document.querySelector('#rootcause');
  if (!module) return;
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
    // Small, safe inline formatter: answer text never enters innerHTML.
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
  const bank = window.OPGUARD_DEMO_ANSWERS;
  for (const button of module.querySelectorAll('[data-rootcause]')) {
    const result = document.getElementById(`${button.dataset.rootcause}-answer`);
    let previousId;
    button.addEventListener('click', () => {
      result.hidden = false;
      const answers = bank?.arms?.[button.dataset.rootcause];
      if (!Array.isArray(answers) || !answers.length) {
        result.textContent = 'The example answers could not be loaded. Please reload the page.';
        return;
      }
      const alternatives = answers.filter(answer => answer.id !== previousId);
      const choices = alternatives.length ? alternatives : answers;
      const answer = choices[Math.floor(Math.random() * choices.length)];
      previousId = answer.id;
      result.dataset.answerId = answer.id;
      renderAnswer(result, answer.answer);
      const notice = document.createElement('p');
      notice.className = 'rootcause__answer-label';
      notice.textContent = `${bank.notice} · ${answers.indexOf(answer) + 1}/${answers.length}`;
      result.prepend(notice);
      button.textContent = 'Show another example';
      button.setAttribute('aria-label', `Show another ${button.dataset.rootcause === 'opguard' ? 'OpGuard' : 'standard-log'} example diagnosis`);
    });
  }
  // README links open the same page and show only the explicitly selected example.
  const requestedArm = new URLSearchParams(location.search).get('analyze');
  if (['trace', 'opguard'].includes(requestedArm)) {
    module.querySelector(`[data-rootcause="${requestedArm}"]`)?.click();
  }
})();
