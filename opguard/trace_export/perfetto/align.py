"""Conservative, monotone call correspondence, independent of output values."""
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class Alignment:
    pairs: list[tuple[int, int, str]]
    left_unresolved: dict[int, str]
    right_unresolved: dict[int, str]


def callsite_key(record):
    fn, file = record.get("function"), record.get("fname")
    meta = record.get("meta") or {}
    line = record.get("line_num") or meta.get("lineno")
    if not fn or not file or not line:
        return None
    context = {key: value for key, value in meta.items() if key != "lineno"}
    # Called within a corresponding trace file. Checkout roots may differ; the
    # filename, function, line, rank and captured step/context must agree.
    return (fn, Path(file).name, line, record.get("gpu_rank"),
            json.dumps(context, sort_keys=True, separators=(",", ":")))


def align_records(left, right):
    """Pair equal callsite sequences by order; otherwise require unique anchors.

    This assumes corresponding execution order within a callsite/context, not
    equal tensor values. Input summaries can anchor unequal sequences only when
    unique on both sides and monotone. They remain lossy observations. Output
    values, output shapes and global counter distances never affect pairing.
    Unresolved relationships stay explicit; no score, DP or repair pass follows.
    """
    a, b = [callsite_key(x) for x in left], [callsite_key(x) for x in right]
    result = Alignment([], {}, {})

    def add(i, j, basis):
        result.pairs.append((i, j, basis))

    reason = "ambiguous_call_sequence"
    if a == b:
        for i, key in enumerate(a):
            if key is not None:
                add(i, i, "callsite_sequence_order")
    else:
        def anchors_for(records, keys):
            # Keep complete captured input representations, including scalar
            # strings. Truncating scalars could create false unique anchors.
            return [(key, json.dumps(record["inputs"], sort_keys=True, separators=(",", ":")))
                    if key is not None and record.get("inputs") else None
                    for key, record in zip(keys, records)]

        ak, bk = anchors_for(left, a), anchors_for(right, b)
        ca, cb = Counter(ak), Counter(bk)
        positions = {key: j for j, key in enumerate(bk) if key is not None and cb[key] == 1}
        anchors = [(i, positions[key]) for i, key in enumerate(ak)
                   if key is not None and ca[key] == 1 and key in positions]
        if any(y[1] <= x[1] for x, y in zip(anchors, anchors[1:])):
            reason = "conflicting_input_anchors"
        else:
            start_a = start_b = 0
            for i, j in anchors + [(len(a), len(b))]:
                if a[start_a:i] == b[start_b:j]:
                    for x in range(start_a, i):
                        if a[x] is not None:
                            add(x, start_b + x - start_a, "bounded_callsite_sequence_order")
                if i < len(a):
                    add(i, j, "unique_input_anchor")
                start_a, start_b = i + 1, j + 1

    used_a, used_b = {i for i, _, _ in result.pairs}, {j for _, j, _ in result.pairs}
    def unresolved(keys, other, used):
        other_keys = set(other) - {None}
        return {i: ("missing_callsite_identity" if key is None else
                    "no_matching_callsite" if key not in other_keys else reason)
                for i, key in enumerate(keys) if i not in used}

    result.left_unresolved = unresolved(a, b, used_a)
    result.right_unresolved = unresolved(b, a, used_b)
    return result
