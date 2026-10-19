# Root-cause comparison

Two fixed evidence inputs are sent independently to Groq GPT-OSS 20B. Each arm
uses one request containing its complete input, with the same diagnostic prompt,
model, reasoning setting, and 3,072-token completion cap. The browser sends only
an arm identifier; it cannot choose prompts, models, URLs, or token limits.
The reference answer is separate HTML and never sent to Groq.

## Local preview and verification

Requires Node 24 or newer. The local server uses built-in SQLite. Install the
pinned development dependencies with `pnpm install --frozen-lockfile`, then run:

```sh
node demo/server.mjs /absolute/path/to/private/.dev.vars
pnpm test
pnpm run check:deploy
```

The private file contains `GROQ_API_KEY=...`; keep it outside the public directory
and restrict permissions to `600`. The server binds to `127.0.0.1:8937`, serves an
explicit allowlist of public paths, and persists quota reservations in ignored
`.demo-state/quota.sqlite`. It does not log keys, prompts, or model answers.

Tests verify exact full-evidence delivery, spend accounting, rate windows, failures,
and actual SQLite Durable Object concurrency and persistence across runtime restarts.
The integration test starts an isolated local Cloudflare runtime; it never calls Groq.
`python3 demo/build_packets.py` rebuilds the evidence packets from pinned artifacts.

## Paid deployment

API: `https://opguard-rootcause.opguard-website.workers.dev/api/rootcause`.
The browser configuration points to this API; localhost uses the local backend.
The Groq organization must have Developer pay-as-you-go enabled. Its confirmed
GPT-OSS 20B limit on 2026-10-19 is 250,000 tokens/minute and 1,000 requests/minute.
The previous free plan cannot accept the full log packet in one request.

The website stays on GitHub Pages (`webpage` branch). Deploy only the API to
Cloudflare with `pnpm run deploy`. The existing Worker holds `GROQ_API_KEY` in its
secret store; do not put credentials in frontend assets or committed files.
For initial authentication, `pnpm exec wrangler login --device --scopes account:read user:read workers_scripts:write`
authorizes the intended account. Provision a missing secret with
`pnpm exec wrangler secret put GROQ_API_KEY` and enter it privately.

After deployment, `node demo/check-deployment.mjs <api-url>` checks CORS,
request validation, and both live arms (two real model calls, subject to the same
budget). `pnpm exec wrangler deployments list` shows the active version.
Publish `webpage` separately to update the public labels and browser timeout.

## $2 per rolling 24 hours

One persistent SQLite Durable Object (`groq-demo-v1`) is shared by all Worker
instances and regions. Before any Groq call, an atomic transaction reserves the
maximum possible inference cost. A request that would exceed **$2 during any
rolling 24-hour window** receives HTTP 429 without calling Groq.

Accounting uses integer nano-USD at the 2026-10-19 GPT-OSS 20B prices:
$0.075/M input tokens and $0.30/M completion tokens, including reasoning.
Input reservation uses the UTF-8 byte count of the fixed prompt and entire packet
plus 1,024 tokens of framing headroom; output reserves all 3,072 allowed tokens.
Cached inputs are conservatively charged at the full input rate.

Successful calls settle to provider-reported input and completion usage. Missing
or invalid usage, timeouts, upstream 429s and other failures retain the full
reservation. No automatic upstream retries are performed. A successful charge
expires 24 hours after settlement; uncertain calls retain reservations for 24
hours plus the 60-second call deadline. Existing pre-upgrade entries remain in
the same ledger and are conservatively valued at the higher output token price.
Quota storage failures block the request rather than bypassing the budget.

Additional rolling limits: 240,000 reserved tokens/minute, 120 requests/minute
globally, 6 requests/minute per hashed IP, and 200 requests/24 hours per hashed IP.
The IP digest remains stable across deployments with the same Groq secret. The
$2 limit covers this demo's Groq inference at the pinned rates, not other apps
using the organization or hosting/tax charges. Price/model changes require
reviewing the accounting constants. No provider monthly cap replaces this local
per-call gate; Groq's monthly spend reporting can lag by 10–15 minutes.

## Evidence provenance

Inputs come from main commit d10c4a0, harness/examples/demo_case5/paired-capture-20261019.
The four complete console streams and OpGuard summaries share the same capture IDs.
`demo/evidence/manifest.json` pins the imported files; build_packets.py verifies hashes.
No diagnostic source excerpt, patch, or reference answer is added to either packet.
The left arm retains every original log byte, in order within each of four streams.
Separate stdout/stderr cannot be interleaved retrospectively. Expand shows the entire
packet; the page preview is only a few lines. Neither arm uses intermediate summaries.

The right arm retains the 33 unaligned operations before each run2 boundary and
unverified correspondence limitations. These are first observed captured callsites,
not whole-graph first differences or proof of stream ordering/buffer reuse.
Ordinary logs may suggest the cause; OpGuard alone does not prove it. Live model
answers can be wrong. The reference cause remains a separate disclosure.

## Cost labels

Raw evidence is 18,640 tokens versus 1,455 (~12.8x), measured with o200k_base.
Both arms use one request. The conservative per-call reservations are 60,444
versus 11,238 total tokens (UTF-8 input bound, framing, and maximum output).
Their maximum reserved inference costs are $0.0052245 and $0.00153405, rounded
up in the UI to $0.0053 and $0.0016. These are conservative budget caps, not
measured charges or diagnostic-accuracy guarantees. Actual charges are normally
lower. Failed/repeated analyses and evidence collection/hosting are excluded
from the displayed per-analysis caps. Visitors are not charged for the demo.
