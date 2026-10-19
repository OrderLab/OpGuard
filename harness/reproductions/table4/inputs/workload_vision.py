"""Evaluate vision embeddings on a shared synthetic image."""
import json
from types import SimpleNamespace
import torch
from modeling_smolvlm import SmolVLMVisionEmbeddings


def main():
    torch.manual_seed(1234)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = SimpleNamespace(hidden_size=32, image_size=384, patch_size=14, num_channels=3)
    model = SmolVLMVisionEmbeddings(config).cuda().eval()
    pixels = torch.randn(1, 3, 378, 378, device="cuda")
    mask = torch.ones((1, 27, 27), dtype=torch.bool, device="cuda")
    with torch.no_grad():
        output = model(pixels, mask)
    torch.save(output.cpu(), "observed.pt")
    print(json.dumps({"shape": list(output.shape), "sample": output[0, -1, :10].tolist()}))


if __name__ == "__main__":
    main()
