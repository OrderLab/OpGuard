"""Actual Qwen2-VL forward/generate, with a tiny randomly initialized model.

modeling_qwen2_vl.py is copied verbatim from the issue's pinned commit. No model
downloads are needed: the cache/position bug does not depend on learned weights.
"""
import argparse
import importlib.util
from pathlib import Path
import sys

import torch
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["generate", "call"], required=True)
    parser.add_argument("--clear-rope", action="store_true")
    args = parser.parse_args()
    name = "transformers.models.qwen2_vl.modeling_qwen2_vl"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("modeling_qwen2_vl.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    torch.manual_seed(1234)
    config = Qwen2VLConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
        vision_config={"depth": 1, "embed_dim": 32, "hidden_size": 64,
                       "num_heads": 4, "mlp_ratio": 2, "patch_size": 2,
                       "temporal_patch_size": 1, "spatial_merge_size": 2},
        image_token_id=120, video_token_id=119, vision_start_token_id=121,
        vision_end_token_id=122, bos_token_id=1, eos_token_id=2, pad_token_id=0,
        attn_implementation="eager",
    )
    model = module.Qwen2VLForConditionalGeneration(config).cuda().eval()
    inputs = {
        "input_ids": torch.tensor([[11, 121, 120, 120, 120, 120, 122, 12]], device="cuda"),
        "attention_mask": torch.ones(1, 8, dtype=torch.long, device="cuda"),
        "pixel_values": torch.randn(16, 12, device="cuda"),
        "image_grid_thw": torch.tensor([[1, 4, 4]], device="cuda"),
    }
    torch.save({"inputs": inputs, "state_dict": model.state_dict()}, "initial.pt")
    positions = []
    rope_calls = []
    forward_states = []
    original_get_rope_index = model.get_rope_index
    def counted_rope_index(*call_args, **kwargs):
        result = original_get_rope_index(*call_args, **kwargs)
        rope_calls.append(result[0].detach().cpu())
        return result
    model.get_rope_index = counted_rope_index
    def record_forward_state(_module, _args, kwargs):
        cache_position = kwargs.get("cache_position")
        forward_states.append({
            "cache_position": None if cache_position is None else cache_position.detach().cpu(),
            "past_key_values_is_none": kwargs.get("past_key_values") is None,
            "rope_deltas_is_none": model.rope_deltas is None,
        })
    model.register_forward_pre_hook(record_forward_state, with_kwargs=True)
    def record_positions(_module, _args, kwargs):
        positions.append(kwargs["position_ids"].detach().cpu())
    model.model.register_forward_pre_hook(record_positions, with_kwargs=True)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1, do_sample=False)
        positions.clear()
        rope_calls.clear()
        forward_states.clear()
        if args.clear_rope:
            model.rope_deltas = None
        if args.mode == "generate":
            output = model.generate(**inputs, max_new_tokens=1, do_sample=False,
                                    output_logits=True, return_dict_in_generate=True).logits[0]
        else:
            output = model(**inputs).logits[:, -1, :]
    torch.save({"logits": output.cpu(), "positions": positions[-1],
                "rope_deltas": model.rope_deltas.cpu(),
                "get_rope_index_calls": len(rope_calls), "rope_results": rope_calls,
                "forward_states": forward_states}, "observed.pt")
    print("get_rope_index_calls", len(rope_calls), "forward_states", forward_states)
    print("positions", positions[-1].tolist(), "logits", output[0, :8].tolist())


if __name__ == "__main__":
    main()
