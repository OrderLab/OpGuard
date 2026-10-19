"""Validate saved independent observations, never infer success from a trace file."""
import argparse
import json
from pathlib import Path
import torch


def different(a, b):
    if a.shape != b.shape:
        return {"same_shape": False, "baseline_shape": list(a.shape), "candidate_shape": list(b.shape)}
    return {"same_shape": True, "equal": torch.equal(a, b),
            "different_elements": (a != b).sum().item(),
            "max_abs": (a-b).abs().max().item()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    def read(case, name):
        return torch.load(root / f"case{case}" / name, weights_only=True)
    results = []
    for case, version, stage, issue in ((21, "0.9.5", 2, "EleutherAI/gpt-neox/issues/1004"),
                                      (25, "0.14.2", 1, "deepspeedai/DeepSpeed/issues/5545")):
        ranks = []
        for rank in (0, 1):
            base = read(case, f"raw-stress-baseline/rank{rank}.pt")
            bad = read(case, f"raw-stress/rank{rank}.pt")
            fixed = read(case, f"raw-stress-fixed/rank{rank}.pt")
            assert not torch.equal(base["loss"], bad["loss"])
            assert torch.equal(base["loss"], fixed["loss"])
            assert all(torch.equal(x, y) for x, y in zip(base["weights"], fixed["weights"]))
            ranks.append({"rank": rank, "loss": different(base["loss"], bad["loss"]),
                          "first_step_grad_norm": [base["grad_norm"][0], bad["grad_norm"][0]],
                          "different_weight_elements": sum((x!=y).sum().item() for x,y in zip(base["weights"], bad["weights"])),
                          "fixed_loss_and_weights_bitwise_equal": True})
        results.append({"case": case, "status": "reproduced_reduced_upstream",
                        "source": f"DeepSpeed v{version}", "issue": "https://github.com/"+issue,
                        "scope": f"2-GPU ZeRO-{stage}; weighted embedding workload; fixed loss scale 128; reduction stream delay 50000000 cycles. Original unweighted #5545 also trips CSAN without delay.",
                        "ranks": ranks})
    bad, good = read(22, "raw-candidate.pt"), read(22, "raw-baseline.pt")
    assert not torch.equal(bad["outputs"][0], bad["outputs"][1])
    assert torch.equal(good["outputs"][0], good["outputs"][1])
    results.append({"case": 22, "status": "reproduced_reduced_upstream",
                    "issue": "https://github.com/deepspeedai/DeepSpeed/issues/3404",
                    "scope": "Compiled unmodified v0.9.1 CUDA softmax_context; one token after seven-token prompt; reset_cache control.",
                    "repeat": different(*bad["outputs"][:2]), "reset_repeat_equal": True})
    left, right = read(23, "baseline/observed.pt"), read(23, "candidate/observed.pt")
    assert (left["vocab"].numel(), right["vocab"].numel()) == (50257, 50258)
    assert not torch.equal(left["loss"], right["loss"])
    results.append({"case": 23, "status": "reproduced_reduced_upstream",
                    "issue": "https://github.com/NVIDIA/Megatron-LM/issues/1754",
                    "scope": "Exact pinned padding helper, logical TP 1 vs 2, full unsharded zero logits; no full training claim.",
                    "padded_vocab": [50257, 50258], "loss": [left["loss"].item(), right["loss"].item()]})
    ranks = []
    for rank in range(4):
        result = json.loads((root/f"case24/raw-topk3/rank{rank}.json").read_text())
        exact = json.loads((root/f"case24/raw-exact/rank{rank}.json").read_text())
        assert result["forward_equal"] and result["output_grad_equal"] and not result["input_grad_equal"]
        assert exact["input_grad_equal"]
        result["exact_integer_gradient_control_equal"] = True
        ranks.append(result)
    results.append({"case": 24, "status": "reproduced_mechanism_config_variant",
                    "issue": "https://github.com/NVIDIA/Megatron-LM/issues/1327",
                    "scope": "Actual pinned dispatchers on 4 GPUs, expert 7 empty, top-k=3; original top-k=2 reduced test did not diverge on GB200. This is not a replay of the original V100 training run.", "ranks": ranks})
    bad, good = read(26, "long-candidate.pt"), read(26, "long-baseline.pt")
    assert torch.equal(bad["inputs"][0], bad["inputs"][1])
    assert not torch.equal(bad["outputs"][0], bad["outputs"][1])
    assert torch.equal(good["outputs"][0], good["outputs"][1])
    results.append({"case": 26, "status": "reproduced_candidate_issue_mapping_unconfirmed",
                    "issue": "https://github.com/deepspeedai/DeepSpeed/issues/2243",
                    "scope": "Unmodified v0.7.0 QKV/softmax CUDA ops, seq=192 exceeds hardcoded cache stride 128. Upstream maintainer's capacity 256 control removes nondeterminism. Paper has no issue ID; mapping is inferred, not established.",
                    "inputs_repeat_equal": True, "repeat": different(*bad["outputs"][:2]), "capacity_control_repeat_equal": True})
    left, right = read(27, "baseline/observed.pt"), read(27, "candidate/observed.pt")
    assert not torch.equal(left, right)
    results.append({"case": 27, "status": "reproduced_reduced_upstream",
                    "issue": "https://github.com/huggingface/transformers/issues/41190",
                    "scope": "Exact SmolVLMVisionEmbeddings 4.54.1 vs 4.55.1; random shared weights, 27x27 grid.",
                    "embedding": different(left, right)})
    left, right = read(28, "baseline/observed.pt"), read(28, "candidate/observed.pt")
    assert not torch.equal(left["positions"], right["positions"])
    assert not torch.equal(left["logits"], right["logits"])
    results.append({"case": 28, "status": "reproduced_reduced_upstream",
                    "issue": "https://github.com/huggingface/transformers/issues/36009",
                    "scope": "Pinned whole Qwen2-VL modeling file; tiny random model; generate prelude followed by generate vs direct forward.",
                    "positions": different(left["positions"], right["positions"]),
                    "logits": different(left["logits"], right["logits"])})
    report = {"environment": {"python": __import__("sys").version, "torch": torch.__version__,
                              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()},
              "cases": sorted(results, key=lambda x: x["case"]),
              "all_original_paper_cases_fully_reproduced": False,
              "limitations": ["Case 24 uses top-k=3; original top-k=2 remains untriggered.",
                              "Case 26 exact paper-to-issue mapping is not published and remains unconfirmed.",
                              "All workloads are reduced; historical x86/CUDA environments are not recreated on aarch64/GB200."]}
    (root/"reproduction-results.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"verified_case_entries": len(results), "report": str(root/"reproduction-results.json"),
                      "all_original_paper_cases_fully_reproduced": False}))


if __name__ == "__main__":
    main()
