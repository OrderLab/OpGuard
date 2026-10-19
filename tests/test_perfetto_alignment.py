import gzip
import json

import pytest

from opguard.trace_export.perfetto.align import align_records
from opguard.trace_export.perfetto.build import build_perfetto
from opguard.trace_export.perfetto.scoring import compare_simple


def tensor(shape, signature, device="cuda:0"):
    numel = 1
    for size in shape:
        numel *= size
    return {"shape": shape, "stats": {
        "dtype": "torch.float32", "device": device,
        "numel": numel, "xor_signature": signature,
    }}


def call(value=1, output=0):
    return {"function": "<native.multi_output>", "fname": "/checkout/model.py",
            "line_num": 26, "gpu_rank": 0,
            "inputs": {"args": [value], "kwargs": {}}, "outputs": [output]}


def multi_output_call(context, signature):
    record = call()
    record["inputs"]["args"] = [tensor([1, 1, 768], 11), tensor([1], 0), tensor([1], 0, "cpu")]
    record["outputs"] = [tensor([1, 1, 256], signature),
                         tensor([1, 4, context, 64], signature + 1),
                         tensor([1, 4, context, 64], signature + 2)]
    return record


def pairs(result):
    return [(i, j) for i, j, _ in result.pairs]


def test_multi_output_divergence_does_not_prevent_correspondence():
    left = [multi_output_call(1, 100) for _ in range(3)]
    right = [multi_output_call(context, 200 + i * 10) for i, context in enumerate([8, 9, 10])]
    result = align_records(left, right)
    assert pairs(result) == [(0, 0), (1, 1), (2, 2)]
    assert not result.left_unresolved and not result.right_unresolved
    assert all("outputs" in compare_simple(left[i], right[j], False)[0] for i, j, _ in result.pairs)


def test_input_divergence_and_swapped_outputs_do_not_reorder_calls():
    left, right = [call(1, 1), call(2, 2)], [call(3, 2), call(4, 1)]
    assert pairs(align_records(left, right)) == [(0, 0), (1, 1)]


def test_unique_inputs_anchor_an_extra_call():
    result = align_records([call(1), call(2)], [call(1), call(9), call(2)])
    assert pairs(result) == [(0, 0), (1, 2)]
    assert set(result.right_unresolved) == {1}
    assert {basis for _, _, basis in result.pairs} == {"unique_input_anchor"}


def test_repeated_unequal_sequences_remain_undetermined():
    result = align_records([call(1)] * 2, [call(1)] * 3)
    assert not result.pairs
    assert set(result.left_unresolved.values()) == {"ambiguous_call_sequence"}
    assert set(result.right_unresolved) == {0, 1, 2}


def test_crossing_anchors_are_not_selected_arbitrarily():
    result = align_records([call(1), call(2)], [call(2), call(1), call(3)])
    assert not result.pairs
    assert set(result.left_unresolved.values()) == {"conflicting_input_anchors"}


def test_missing_inputs_cannot_anchor_an_unequal_sequence():
    empty = call()
    del empty["inputs"]
    assert not align_records([empty], [empty, empty]).pairs


def test_scalar_strings_are_not_truncated_into_false_anchors():
    prefix = "same-prefix-" * 4
    result = align_records([call(prefix + "a"), call(2)],
                           [call(prefix + "b"), call(2), call(2)])
    assert not result.pairs


@pytest.mark.parametrize("field,value", [
    ("function", "<unrelated>"), ("line_num", 90), ("fname", "/other.py"),
    ("gpu_rank", 1), ("meta", {"global_step": 2}),
])
def test_incompatible_call_context_is_not_forced(field, value):
    left, right = call(), call()
    right[field] = value
    assert not align_records([left], [right]).pairs


def test_missing_identity_is_not_a_pair():
    result = align_records([{}], [{}])
    assert not result.pairs
    assert result.left_unresolved == {0: "missing_callsite_identity"}


def export(tmp_path, left, right):
    for side, records in (("a", left), ("b", right)):
        path = tmp_path / side
        path.mkdir()
        records = [dict(record, counter=i + 1) for i, record in enumerate(records)]
        (path / "calls.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
        (path / "metadata.json").write_text('{"source_files": {}}')
    target = tmp_path / "trace.json.gz"
    build_perfetto(str(tmp_path / "a"), str(tmp_path / "b"), str(target))
    with gzip.open(target, "rt") as stream:
        return json.load(stream)


def operations(doc):
    return [e for e in doc["traceEvents"] if e.get("cat") == "opguard" and e.get("ph") == "X"]


def test_export_keeps_swapped_outputs_as_differences_and_links_exact_records(tmp_path):
    doc = export(tmp_path, [call(1, 1), call(1, 2)], [call(1, 2), call(1, 1)])
    ops = operations(doc)
    assert len(ops) == 4
    by_id = {e["args"]["record_id"]: e for e in ops}
    for event in ops:
        args = event["args"]
        assert args["status"] == "diff"
        assert args["correspondence"]["uses_output_values"] is False
        assert args["correspondence"]["validation"] == "unreviewed"
        assert args["tensor_equality"] == "not_established"
        peer = by_id[args["correspondence"]["peer_record_id"]]["args"]
        assert peer["counter"] == args["counter"]
        assert peer["correspondence"]["peer_record_id"] == args["record_id"]
    # Every visual flow endpoint must lie inside the intended operation slice.
    flows = [e for e in doc["traceEvents"] if e.get("cat") == "opguard:align"]
    assert len(flows) == 4
    grouped = {}
    for endpoint in flows:
        owner = [e for e in ops if e["pid"] == endpoint["pid"] and e["tid"] == endpoint["tid"]
                 and e["ts"] <= endpoint["ts"] < e["ts"] + e["dur"]]
        assert len(owner) == 1
        grouped.setdefault(endpoint["id"], []).append(owner[0]["args"])
    for a, b in grouped.values():
        assert a["correspondence"]["peer_record_id"] == b["record_id"]


def test_export_unknown_relationships_are_neither_diffs_nor_insertions(tmp_path):
    doc = export(tmp_path, [call(1)] * 2, [call(1)] * 3)
    ops = operations(doc)
    assert len(ops) == 5  # Sidecar metadata is not an operation.
    assert all(e["args"]["status"] == "unaligned" for e in ops)
    assert all(e["args"]["correspondence"]["state"] == "undetermined" for e in ops)
    assert all("diff" not in e["args"] for e in ops)
    assert not any(e.get("cat") == "opguard:align" or e.get("args", {}).get("first_diff")
                   for e in doc["traceEvents"])
