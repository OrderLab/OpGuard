# Live diagnosis demo

Two panels: Standard logs and OpGuard output, short previews, Expand and Get rootcause.
The reference cause sits in a collapsed disclosure below. GitHub Pages publishes webpage;
the API is the existing Cloudflare Worker with a server-side Groq secret.

Both inputs now use the same-launch demo5 paired-capture-20261019 artifacts (d10c4a0).
The standard-log packet preserves all four raw streams, ordered within each stream.
The OpGuard packet selects first observed captured-callsite differences on both ranks,
retaining unaligned counts and correspondence limitations. No selected source hint is added.
The historical four-event baseline has been replaced.

Groq GPT-OSS 20B reads complete logs in eight sequential parts with rolling model notes
and streams progress while waiting for free quota. OpGuard uses one call. Both share
the diagnostic prompt, but this is not an equal-call-budget benchmark. Summaries may
lose detail. Expand links to the full evidence packet; previews are only a few lines.
Backend limits and build/verification instructions live in webpage/demo/README.md.

The known race is a reference explanation, not proven by scalar fingerprints. First
observed differences are not a verified whole-graph first difference. Model responses
are live and can be wrong; ordinary logs may also suggest the cause.
