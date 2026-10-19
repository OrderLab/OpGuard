# Root-cause comparison

Two fixed evidence inputs are sent independently to Groq GPT-OSS 20B. The browser
sends only an arm identifier. The backend supplies the prompt, evidence, model,
and token limit. The reference answer is separate HTML and never sent to Groq.

## Local preview

Requires Node 24 or newer. The runtime needs no npm dependencies; rebuilding chunks uses gpt-tokenizer.

```sh
node demo/server.mjs /absolute/path/to/private/.dev.vars
```

The private file contains `GROQ_API_KEY=...`; keep it outside the public directory
and restrict permissions to `600`. The server binds to `127.0.0.1:8937`, serves an
explicit allowlist of public paths, and persists quota reservations in ignored
`.demo-state/quota.sqlite`. It does not log keys, prompts, or model answers.

```sh
python3 demo/build_packets.py
node demo/build_chunks.mjs
node --test demo/core.test.mjs
```

## Public deployment

Deployment URL: `https://opguard-rootcause.opguard-website.workers.dev/api/rootcause`.
The browser configuration points to this API; localhost previews continue to use
the local backend. Publishing the website branch is a separate step.

The existing website stays on GitHub Pages. Deploy only the API to Cloudflare:

1. Run `pnpm install --frozen-lockfile`, then `pnpm exec wrangler login --device --scopes account:read user:read workers_scripts:write` and authorize the intended Cloudflare account. Device authorization avoids the two-minute localhost callback timeout and requests only account reads and Worker deployment access.
2. Run `pnpm exec wrangler secret put GROQ_API_KEY` and enter the key privately.
3. Run `pnpm run deploy` using the included `wrangler.jsonc`.
4. Put the returned Worker URL plus `/api/rootcause` in `js/demo-config.js`.
5. Publish `webpage`, verify both buttons, then activate the main README links.

The SQLite Durable Object keeps one atomic shared quota across Worker instances.
The defaults allow 7,800 reserved tokens/minute, 180,000/day, 100 requests/day
globally, and 24/day per hashed IP (all rolling windows). Successful calls settle
to provider-reported usage; failures conservatively keep the reservation. No
result caching or paid fallback. The log reader waits on short quota limits,
without retrying completed model calls. The frontend displays errors and a
retry delay without silently replacing the live answer.

These limits protect this demo, not other applications using the same Groq
organization. Check your actual account limits before public release. The local
SQLite adapter was tested. Wrangler 4.151.1 dry-run bundling passed and resolved
the Durable Object binding. Public deployment checks passed: OPTIONS 204, rejected
origin 403, arbitrary prompt 400, and both evidence arms 200 with actual Groq
completions. Recheck with `node demo/check-deployment.mjs <api-url>` (uses nine
model calls). Use `pnpm exec wrangler deployments list` to inspect the active version.

## Evidence provenance and reading budget

Inputs come from the path-normalized main snapshot `146d6bb523cf`,
harness/examples/demo_case5/paired-capture-20261019.
The four complete console streams and OpGuard summaries share the same capture IDs.
`demo/evidence/manifest.json` pins the imported files; build_packets.py verifies their hashes.
No diagnostic source excerpt, patch, or reference answer is added to either packet.
The left arm retains all log records in order within each of the four streams.
Host-specific paths are normalized to `/opt/opguard`; checksums and event references
identify the normalized export. Numerical values and capture IDs are unchanged.
Separate stdout/stderr cannot be interleaved retrospectively. Expand shows the entire
packet; it is not just the short page preview.

The 55,469 normalized log bytes become eight sequential chunks of at most 2,500 o200k_base
tokens. The build asserts that concatenation reproduces the exact packet. Chunk counts
are pinned at build time; the Worker does not bundle a tokenizer. Each request reserves
chunk tokens plus a conservative byte bound for prompt/previous notes, 3,072 completion
tokens and 512 framing/instruction tokens, under the 7,800 rolling-minute cap.
Successful requests settle to actual Groq usage. Intermediate notes are limited to
1,000 bytes; oversized model notes are marked and shortened; incomplete results produce an error, never a substitute answer.

The left arm streams progress and waits for short quota windows (up to fifteen minutes).
It makes eight real calls, passing previous model notes onward; summaries can lose details.
The right arm makes one call with selected first-observed boundaries on both ranks.
Both use GPT-OSS 20B and the same diagnostic system prompt; their reading strategies differ.
Every upstream call counts against the shared 100/day and per-IP 24/day limits.
Cancellation stops further requests; failures do not produce a cached or fabricated answer.

The right arm retains the 33 unaligned operations before each run2 boundary and the
unverified correspondence limitations. These are first observed captured callsites,
not whole-graph first differences or evidence of exact stream ordering/buffer reuse.
No guarantee is made that ordinary logs cannot suggest the cause, or that OpGuard
alone proves it. The reference cause remains a separate disclosure below the panels.

## Fixed cost comparison

As of 2026-10-19, Groq lists GPT-OSS 20B at $0.075/M input and $0.30/M
output tokens: https://console.groq.com/docs/model/openai/gpt-oss-20b.
The UI shows fixed successful-pass budget caps, not measured charges or a guarantee
of diagnostic accuracy. Raw evidence is 18,640 tokens versus 1,455 (~12.8x);
sequential log reading uses eight calls versus one. Conservative input reservations
(including prompt/framing and bounded model notes) total 33,416 versus 2,427;
maximum output is 24,576 versus 3,072. List-price caps compute to $0.009879
and $0.001103625, rounded upward to $0.010 and $0.0012. Failed/repeated
analyses and evidence collection/hosting are excluded; free-tier usage is not billed.
