# OpGuard website

This branch contains the static website and live demo for https://orderlab.io/OpGuard/.

GitHub Pages should publish the root of `webpage`. Python packages, agent harness, and user documentation are maintained on [`main`](https://github.com/OrderLab/OpGuard/tree/main).

Edit `index.html`, `css/styles.css`, and `js/app.js` here. Keep `assets/traces/` URLs stable: the agent harness downloads these public demo artifacts.

Preview locally with `python3 -m http.server 8000` from this directory.

For the Groq-backed root-cause comparison, use `node demo/server.mjs /path/to/.dev.vars`
instead. See [demo setup and evidence provenance](demo/README.md). The public API URL is configured in `js/demo-config.js`; publishing this branch
activates the live buttons on the website.
