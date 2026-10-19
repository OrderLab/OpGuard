# Reviewed root-cause examples

The public demo serves three reviewed examples per arm. Clicking **Get rootcause**
selects an answer locally; later clicks randomly choose a different answer from
the previous one. The browser makes no API or model request. This also works when
opening `index.html` directly from disk.

These are curated explanations of the fixed capture, not verbatim model outputs
or an accuracy benchmark. Prior model trials sometimes named the buffer-reuse
mechanism but mixed configuration branches and source line numbers. Those errors
were corrected during source review; the website does not present the edited
answers as fresh autonomous diagnoses.

`answer-bank.json` is the editable source of truth. `build-cached-answers.mjs`
checks source hashes and every declared code anchor, then generates:

- `../js/rootcause-answers.js`: the browser's bundled data.
- `cached-answers.js`: the compatibility API's matching data.

The standard-log examples describe the recorded numerical/configuration
observations and their limits. The OpGuard examples use those logs, the trace,
and the complete captured source. They distinguish the norm-append symptom from
buffer reuse during the same backward pass. The copy, index change, reduction
use, one-way wait, and later epilogue synchronization have checked source links.
Source reasoning does not itself prove the GPU completion order in a new run.

## Build and test

```sh
node demo/build-cached-answers.mjs
node --test demo/*.test.mjs
node demo/server.mjs
```

Tests cover offline clicks, random variants without immediate repeats, deep
links, safe rendering, source anchors, generated-data consistency, cached API
responses with no quota/model access, and the existing paid-mode budget. The
SQLite Durable Object integration test proves cached responses remain available
with an exhausted ledger and do not reset or release that ledger.

`build_packets.py` still builds the original fixed live-experiment inputs. These
legacy packets are separate from the curated answer bank and are not read by the
public browser when displaying cached examples. “Expand” opens recorded evidence;
source links open the full captured files, not merely the small legacy excerpts.

## Deployment

`wrangler.jsonc` sets `DEMO_MODE=cached`. The Worker returns cached examples for
older clients still posting `{ "arm": "trace" }` or `{ "arm": "opguard" }`.
Each response explicitly identifies itself as a reviewed example, reports
`mode: "cached"` and `inferenceCalls: 0`, and makes no Groq call. It does not need
an API key or access the quota Durable Object in cached mode.

```sh
pnpm exec wrangler deploy --dry-run
pnpm exec wrangler deploy
node demo/check-deployment.mjs https://opguard-rootcause.opguard-website.workers.dev/api/rootcause
```

The deployment check expects cached responses; it must not be used as a paid
model benchmark. The website is published from the `webpage` branch.

## Retained live experiment path

Paid calls require an explicit server-side `DEMO_MODE=live`. Missing or unknown
mode values default to cached responses. There is no browser switch that enables
paid inference. The shared generic prompt in `diagnostic-guide.js` describes
OpGuard fields and source investigation, but it does not include the case answer.
The live path remains experimental; cached example quality is not evidence of
its diagnostic accuracy.

The original `groq-demo-v1` Durable Object and its persisted entries are retained.
Before each paid request, an atomic transaction reserves its worst-case input
and output cost. Requests that would exceed **$2 in a rolling 24 hours** are
rejected before contacting Groq. Failed or unaccounted calls keep their full
reservations. Successful calls settle input plus completion/reasoning usage at
full rates, with no cache discounts. Existing reservations with other model
prices keep their recorded rates during settlement.

The retained GPT-OSS 20B path uses 3,072 output tokens and prices of $0.075/M input
and $0.30/M output, as verified on 2026-10-19. Reservations use a conservative
UTF-8 byte input bound plus framing, capped at the model context limit. Other
limits remain 240,000 tokens/minute, 120 requests/minute globally, and 6/minute
and 200/day per IP. Storage failures fail closed. The cap applies to this demo's
Groq inference, not hosting or unrelated account usage.

Secrets remain in the Worker's secret store. Never reset quota state to make a
test run fit the budget. No top-up or budget increase is required for cached
examples.
