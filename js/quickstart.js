(() => {
  const section = document.getElementById('get-started');
  if (!section) return;
  const status = section.querySelector('[role="status"]');
  for (const button of section.querySelectorAll('[data-copy-code]')) {
    button.addEventListener('click', async () => {
      const code = document.getElementById(button.dataset.copyCode);
      if (!code) return;
      try {
        await navigator.clipboard.writeText(code.textContent.trim());
        button.textContent = 'Copied';
        status.textContent = 'Copied to clipboard.';
        window.setTimeout(() => { button.textContent = 'Copy'; }, 1800);
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
