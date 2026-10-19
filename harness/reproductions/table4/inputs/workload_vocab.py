"""Compute a language-model loss using the configured vocabulary dimension."""
import argparse
import json
from types import SimpleNamespace
import torch
from tokenizer import _vocab_size_with_padding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, required=True)
    args = parser.parse_args()
    torch.manual_seed(1234)
    config = SimpleNamespace(make_vocab_size_divisible_by=1,
                             tensor_model_parallel_size=args.tp, rank=0)
    padded = _vocab_size_with_padding(50257, config)
    vocab = torch.arange(padded, device="cuda")
    logits = torch.zeros((1, padded), device="cuda")
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([42], device="cuda"))
    torch.save({"vocab": vocab.cpu(), "logits": logits.cpu(), "loss": loss.cpu()}, "observed.pt")
    print(json.dumps({"padded_vocab_size": padded, "loss": loss.item()}))


if __name__ == "__main__":
    main()
