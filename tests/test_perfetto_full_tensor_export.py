"""Perfetto export must tolerate raw tensors from OpGuard's full dump mode.

TraceBuffer.write_tensor stores dump_tensors=True payloads as plain tensors under
<rank>/ops/full/. These files must be pruned before operator records are loaded,
while ordinary events and metadata keep their existing handling.
"""
import gzip
import json
from pathlib import Path

import torch

from opguard.trace_export.perfetto.build import build_perfetto


def call(output):
    return {"function": "aten::mm", "fname": "/checkout/model.py", "line_num": 12,
            "gpu_rank": 0, "counter": 1,
            "inputs": {"args": [1], "kwargs": {}}, "outputs": [output]}


def rank_dir(root, output):
    """Mimic one rank directory produced with dump_tensors=True."""
    events = root / "ops" / "aten__mm"
    events.mkdir(parents=True)
    (events / "aten__mm_line12.jsonl").write_text(json.dumps(call(output)) + "\n")
    full = root / "ops" / "full" / "aten::mm" / "gsNA_mbNA_layerNA"
    full.mkdir(parents=True)
    torch.save(torch.ones(2, 2), full / "model.py_line12_input_args_0.pt")
    torch.save(torch.full((2, 2), float(output)), full / "model.py_line12_output_0.pt")
    (root / "metadata.json").write_text('{"source_files": {}}')
    return root


def export(tmp_path, left_output, right_output):
    left = rank_dir(tmp_path / "a", left_output)
    right = rank_dir(tmp_path / "b", right_output)
    target = tmp_path / "trace.json.gz"
    build_perfetto(str(left), str(right), str(target))
    with gzip.open(target, "rt") as stream:
        doc = json.load(stream)
    return [e for e in doc["traceEvents"] if e.get("cat") == "opguard" and e.get("ph") == "X"]


def test_full_dump_tensors_are_not_operator_records(tmp_path):
    ops = export(tmp_path, 0, 0)
    assert len(ops) == 2
    ids = {e["args"]["record_id"] for e in ops}
    for event in ops:
        assert event["args"]["correspondence"]["peer_record_id"] in ids


def test_full_dump_tensors_do_not_hide_a_real_difference(tmp_path):
    ops = export(tmp_path, 0, 1)
    assert len(ops) == 2
    assert {e["args"]["status"] for e in ops} == {"diff"}


def test_full_dump_tensors_never_call_torch_load(tmp_path, monkeypatch):
    def unexpected_load(*args, **kwargs):
        raise AssertionError("Full tensor dumps must not reach torch.load")

    monkeypatch.setattr(torch, "load", unexpected_load)
    assert len(export(tmp_path, 0, 0)) == 2


def test_legacy_pt_operator_records_are_still_loaded(tmp_path, monkeypatch):
    left = rank_dir(tmp_path / "a", 0)
    right = rank_dir(tmp_path / "b", 0)
    legacy_paths = {root / "ops" / "aten__mm" / "legacy.pt" for root in (left, right)}
    for path in legacy_paths:
        torch.save(dict(call(0), counter=2), path)
    original_load = torch.load
    loaded = []

    def checked_load(path, *args, **kwargs):
        path = Path(path)
        assert "full" not in path.parts
        loaded.append(path)
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    target = tmp_path / "trace.json.gz"
    build_perfetto(str(left), str(right), str(target))
    with gzip.open(target, "rt") as stream:
        operations = [e for e in json.load(stream)["traceEvents"]
                      if e.get("cat") == "opguard" and e.get("ph") == "X"]
    assert len(operations) == 4
    assert set(loaded) == legacy_paths
