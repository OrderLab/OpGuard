# README demo

Two panels: **Execution trace** and **OpGuard output**, each with a short excerpt, **Expand ↗**, and **Get rootcause**. A collapsed **Actual root cause** sits below. No case selector or introductory case narrative.

## Implementation

Website and backend source are on [`webpage`](https://github.com/OrderLab/OpGuard/tree/webpage). `demo/server.mjs` runs the website and Groq endpoint locally; `demo/worker.js` is the Cloudflare deployment adapter. Credentials are read only by the backend. The local private configuration is `demo/.dev.vars` in the main checkout and is ignored by git.

Both arms use Groq `openai/gpt-oss-20b`, the same neutral prompt, temperature 0, low reasoning effort, and a 1,536-token completion ceiling. Requests have no shared conversation, retrieval, tools, or reference answer. This is a single model analysis, not an autonomous agent benchmark. Each click makes a fresh inference request; responses are not curated or cached.

The backend accepts only `{ "arm": "trace" }` or `{ "arm": "opguard" }`, never arbitrary prompts. Persistent atomic reservations enforce 7,800 tokens per rolling minute, 180,000 per rolling day, 100 requests/day globally and 12 requests/day per IP. Input UTF-8 bytes plus chat overhead and max output reserve a conservative token budget before inference; successful usage releases unused reservation. Provider limits can still be lower or shared with other apps. Failed requests retain their reservation. No paid fallback.

## Evidence and limits

`demo/build_packets.py` verifies the public export SHA-256 and reproducibly creates both complete model inputs. Expand opens the exact packet passed to Groq. The original full viewer remains at `#trace`.

Both packets share the same embedded DeepSpeed source excerpts, with original line numbers, explicit omissions, and copyright/license notice. The trace arm selects event metadata from records 18997, 19005, 19321, 19327. The OpGuard arm adds the recorded norm-output difference and both selected events' tensor fingerprints. Neither receives `debug.py`, proposed patches, issue links, or the known answer.

The packets are selected with hindsight for a compact demonstration. The execution-trace arm is a metadata projection of an OpGuard export, not an independently captured baseline. Export timestamps/durations are presentation fields, not GPU timings. Counterpart selection does not prove dynamic alignment. Equal XOR signatures do not establish tensor equality. The data can localize a reported difference but cannot prove a stream race; both arms may propose the same hypothesis or remain inconclusive. Do not describe this as a blind evaluation or promise a particular model answer.

The separately displayed reference is the published gradient-buffer reuse race. Its provenance and offline investigation limits remain in [the case walkthrough](../harness/examples/demo_case5/README.md).

## Publication

The Cloudflare Worker is deployed at `https://opguard-rootcause.opguard-website.workers.dev/api/rootcause`, with the Groq key stored as a Worker secret. Public checks passed: CORS preflight 204, disallowed origin 403, custom prompt rejection 400, and both fixed evidence arms returned 200 with real Groq answers. `js/demo-config.js` now points to this endpoint. The README controls link to `?analyze=trace#rootcause` and `?analyze=opguard#rootcause`, which start only the selected analysis when the page opens. GitHub Pages cannot keep a server-side API key. See the website branch's `demo/README.md` for deployment steps.

Provider documentation checked October 19, 2026: [Groq limits](https://console.groq.com/docs/rate-limits), [reasoning parameters](https://console.groq.com/docs/reasoning).

## Local validation, October 19, 2026

Eight backend tests passed, covering input isolation, request validation, token reservations, rolling caps, provider errors, and incomplete responses. HTTP checks confirmed both input files are accessible while credentials, backend source and quota storage return 404. Both buttons returned real Groq completions in browser testing.

The first live trial misinterpreted fingerprints. After clarifying evidence semantics in the shared prompt and adding the same norm-call source lines to both packets, both arms were rerun. The OpGuard response localized the norm mismatch and proposed a gradient-reduction race while acknowledging missing stream-order evidence. It still incorrectly called the mismatch the only differing operation and treated counter 2144 as a line number. The trace response also proposed a race but made unsupported timing/rank claims. These results do not establish a reliable root-cause accuracy advantage. Both trials are retained locally in the website checkout's ignored `.demo-state/` directory.
