# Root-cause comparison

Two fixed evidence inputs are sent independently to Groq GPT-OSS 20B. Each arm
uses one request containing its complete input, with the same diagnostic prompt,
model, reasoning setting, and 3,072-token completion cap. The browser sends only
an arm identifier; it cannot choose prompts, models, URLs, or token limits.
The reference answer is separate HTML and never sent to Groq.
Both arms receive the recorded environment versions (including PyTorch,
DeepSpeed, Accelerate, NumPy, CUDA and driver). The standard-log arm receives
no captured source or preselected suspect code. Only the OpGuard arm receives
the complete captured module referenced by its observed events, with line numbers.
Neither arm has a source-search tool: this is an evidence-input comparison using
single model calls, not a benchmark of an agent searching dependency repositories.
Dependency commits are unknown, and no commit or Megatron dependency is invented.

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

Tests verify evidence isolation, captured-source hashes and line mappings, clickable
citations, exact full-evidence delivery, spend accounting, rate windows, failures,
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
plus 1,024 tokens of framing headroom, capped at the model's 131,072-token input
context bound; output conservatively reserves all 3,072 allowed tokens on top.
Tests also tokenize both complete packets to verify that input plus output fits.
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

Inputs come from main commit 146d6bb523cf8364bbf2788cb319aef1a46f966b,
harness/examples/demo_case5/paired-capture-20261019.
The four complete console streams and OpGuard summaries share the same capture IDs.
`demo/evidence/manifest.json` pins the imported files; build_packets.py verifies hashes.
No launch command, patch, reference answer, or statement that only one configuration
differs is added to either packet. Run labels are normalized to run1/run2 in the
derived inputs, preserving all original log records and their order. Configuration
values printed by the application remain in the logs.
Separate stdout/stderr cannot be interleaved retrospectively. Expand shows the entire
packet; the page preview is only a few lines. Neither arm uses intermediate summaries.

`demo/evidence/source-manifest.json` pins two exact `sourceFiles` entries extracted
from the rank-0 trace. The complete module is supplied only to the OpGuard arm;
this selection comes from its observed callsites, not the known root cause.
Source line numbers include instrumentation, so they differ from upstream package
line numbers and the original operator tags. For example, rank 0's tag 1688 maps
to captured lines 1734/1736, and rank 1's tag 1699 to 1745/1747. Source hashes verify
the original captured snapshots; public viewers and numbered model inputs normalize
run labels without changing line counts. The viewers support `#L<number>` links.
The API attaches all four observed source locations to OpGuard answers, so the UI
can link to the trace's recorded callsites even when a model omits full citations.
These locations are labeled as captured evidence, independently of the hypothesis.

The right arm retains the 33 unaligned operations before each run2 boundary and
unverified correspondence limitations. These are first observed captured callsites,
not whole-graph first differences or proof of stream ordering/buffer reuse.
Ordinary logs may suggest the cause; OpGuard alone does not prove it. Live model
answers can be wrong. The reference cause remains a separate disclosure.

## Cost labels

Complete evidence inputs are 18,689 tokens for logs + versions and 70,291 for
OpGuard observations + captured source + versions, measured with o200k_base.
The shared system prompt adds 210 tokens. Both arms use one request. The
conservative per-call reservations are 61,566 versus 134,144 total tokens
(input bound, framing, and maximum output).
Their maximum reserved inference costs are $0.00530865 and $0.010752, rounded
up in the UI to $0.0054 and $0.0108. These are conservative budget caps, not
measured charges or diagnostic-accuracy guarantees. Actual charges are normally
lower. Failed/repeated analyses and evidence collection/hosting are excluded
from the displayed per-analysis caps. Visitors are not charged for the demo.
