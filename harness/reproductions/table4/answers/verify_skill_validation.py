"""Check saved controlled experiments; this is not a harness tensor-query API."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def bitwise(left, right):
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and torch.equal(
            left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)
        )
    )


def difference(left, right):
    return {
        "bitwise_equal": bitwise(left, right),
        "num_different": int((left != right).sum()),
        "max_abs": float((left - right).abs().max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("validation_dir", type=Path)
    args = parser.parse_args()
    root = args.validation_dir.resolve()
    manifest = json.loads((root / "evidence-manifest.json").read_text())

    def read(key):
        record = manifest[key]
        path = root / record["archive"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == record["sha256"], (key, "snapshot changed")
        return torch.load(path, map_location="cpu", weights_only=True)

    b22 = read("22-control-baseline")
    c22 = read("22-control-candidate")
    f22a = read("22-reset-baseline")
    f22b = read("22-reset-candidate")
    assert all(bitwise(b22["input"], x["input"]) for x in (c22, f22a, f22b))
    assert not bitwise(b22["outputs"][0], c22["outputs"][0])
    assert all(not bitwise(c22["outputs"][0], x) for x in c22["outputs"][1:])
    assert all(len(x["outputs"]) == 3 for x in (b22, c22, f22a, f22b))
    assert all(bitwise(b22["outputs"][0], x) for run in (b22, f22a, f22b) for x in run["outputs"])
    case22 = {
        "state_hypothesis": "supported_for_tested_native_operator",
        "same_saved_input_bytes": True,
        "baseline_vs_candidate_first_output": difference(b22["outputs"][0], c22["outputs"][0]),
        "candidate_repeated_input": difference(c22["outputs"][0], c22["outputs"][1]),
        "reset_outputs_bitwise_equal_across_three_calls_and_runs": True,
    }

    b28 = read("28-state-baseline")
    c28 = read("28-state-candidate")
    generate = read("28-previous-generate")
    direct = read("28-previous-direct")
    init_b = read("28-initial-baseline")
    init_c = read("28-initial-candidate")
    assert init_b.keys() == init_c.keys()
    for group in ("inputs", "state_dict"):
        assert init_b[group].keys() == init_c[group].keys()
        assert all(bitwise(value, init_c[group][key]) for key, value in init_b[group].items())
    assert b28["get_rope_index_calls"] == 1
    assert c28["get_rope_index_calls"] == 0
    assert b28["forward_states"] == [{
        "cache_position": None, "past_key_values_is_none": True, "rope_deltas_is_none": True,
    }]
    assert c28["forward_states"] == [{
        "cache_position": None, "past_key_values_is_none": True, "rope_deltas_is_none": False,
    }]
    assert len(b28["rope_results"]) == 1 and not c28["rope_results"]
    assert bitwise(b28["rope_results"][0], b28["positions"])
    for key in ("positions", "logits"):
        assert not bitwise(b28[key], c28[key])
        assert bitwise(b28[key], generate[key])
        assert bitwise(c28[key], direct[key])
    case28 = {
        "state_hypothesis": "supported_for_tested_pinned_model",
        "all_four_inputs_and_34_state_dict_tensors_bitwise_equal": True,
        "state_dict_tensor_count": len(init_b["state_dict"]),
        "get_rope_index_calls": {"cleared": 1, "retained": 0},
        "cleared_vs_retained": {key: difference(b28[key], c28[key]) for key in ("positions", "logits")},
        "cleared_matches_generate_reference_bitwise": True,
        "retained_reproduces_original_direct_call_bitwise": True,
    }
    result = {
        "snapshot_hashes_verified": True,
        "method": "Explicit workload snapshots and hooks, read locally; not a harness raw-tensor API",
        "case22": case22,
        "case28": case28,
    }
    (root / "verified-results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
