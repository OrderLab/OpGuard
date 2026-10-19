# Alignment maps v1

Automatic callsite/sequence pairs are candidates. An alignment map records an
agent's review or declares a different tensor boundary across implementations.
File paths, caller stacks, function names and ranks may differ. Evidence must
justify the chosen logical domains; output agreement alone does not do so.

Apply a map to a **flat** Perfetto export with unique operation record IDs:

```bash
python -m opguard.trace_export.perfetto.alignment_map \
  --artifact /runs/original.json.gz \
  --alignment-map /investigation/alignment-map.json \
  --out /investigation/reviewed.json.gz
```

The equivalent backend action is `python -m opguard.tools align --request request.json`:

```json
{
  "protocol_version": 1,
  "artifact": "/runs/original.json.gz",
  "alignment_map": "/investigation/alignment-map.json",
  "output": "/investigation/reviewed.json.gz"
}
```

The output must not exist. The response includes its SHA-256, the source/map
SHA-256, and `verdict: not_evaluated`. No workload is launched. The original
export is unchanged; rejected and displaced observations remain in the new view
under `original_comparison`. Operations, their IO and source/stack content stay
present. Visual status lanes, alignment flows and per-run/rank first-observed
difference markers are rebuilt. Presentation order is not GPU causal order.

## Document and selectors

```json
{
  "version": 1,
  "artifact_sha256": "FULL_SHA256_OF_ORIGINAL_EXPORT",
  "reviews": [
    {
      "left_ref": "FULL_SHA256_OF_ORIGINAL_EXPORT:EVENT_INDEX_A",
      "right_ref": "FULL_SHA256_OF_ORIGINAL_EXPORT:EVENT_INDEX_B",
      "left_tensor": ["outputs", 0],
      "right_tensor": ["inputs", 1],
      "decision": "comparable",
      "reason": "These slots represent the same logical boundary under the recorded mapping.",
      "evidence": [
        {"basis": "source_inference", "reference": "SOURCE_SHA256:path:function:line; explanation of boundary"}
      ]
    }
  ]
}
```

`event_ref` is the source artifact hash plus the zero-based `traceEvents` index,
as returned by harness queries. `left_tensor` and `right_tensor` select entries
of the event's **exported** `args.io.inputs` or `args.io.outputs` arrays. These
are flattened tensor summaries, not positional Python arguments or full tensors.
For example, `['inputs', 1]` means the second captured tensor input. Inspect the
event first. A new pair requires both selectors. It compares only those selected
summaries; unrelated operator names, scalar arguments and other outputs are not
part of that comparison. All unselected observations remain in the artifact.

Use null for both selectors to review an existing whole-event candidate. The
decisions are:

- `comparable`: retain the original candidate comparison, or compare the
  explicitly selected slots. This is an agent assertion with evidence, not an
  independently established equivalence proof.
- `not_comparable`: reject this direct comparison and remove its active flow.
  Use null selectors; this judgment addresses the proposed relation as a whole.
- `undetermined`: keep the relation unresolved, also with null selectors.

All decisions require a reason and nonempty evidence references. Evidence bases
are `observed`, `source_inference`, `intervention_tested`, or `externally_reported`.
References are retained as provenance, not executed or automatically verified.
Optional `id` and `created_at` fields identify harness review-history entries.

History is ordered. The last entry for an event pair is active; earlier entries
remain in the map. A view supports one active comparable relation per operation.
Conflicting accepted mappings fail instead of choosing one. A new mapping can
displace automatic peers; those displaced operations become undetermined.
Rejected alternative pairs do not remove unrelated accepted mappings.

## Evidence and limits

`correspondence.validation` distinguishes `unreviewed` from `agent_reviewed`.
Neither claims an independently verified root cause. `ok` means recorded-field
agreement, including possible fingerprint collisions; `tensor_equality` remains
`not_established`. A true numerical difference unrelated to the final symptom
should remain comparable; refute the causal hypothesis in the investigation
record rather than rejecting a valid comparison.

There are no automatic layout conversions, rank regrouping, dtype casts, or
one-to-many operation matching. Use a directly comparable boundary or leave the
relation unresolved. Multiple internal operations can be investigated without
forcing a pair for each one. No output-distance scoring or repair pass is added.

Maps address every operation in the export, including operations originally
marked `no_corresponding_trace_file`. Cross-rank mapping is possible only for
ranks already present in the same artifact; this does not change the capture
backend's matching-rank launch contract.

Maps are limited to 8 MiB; exports to 256 MiB encoded and decoded. Stale hashes,
invalid slots, same-run pairs and conflicting mappings fail before writing an
output. A new view has new event references. Continue reviewing the original
artifact and render again to reuse its history, or start a separate history for
the new revision. Never carry old references silently into a new capture.
