import torch
import torch.nn as nn

class LayerWithBackwardHook(nn.Module):
    def __init__(self, in_features, out_features, layer_num):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.layer_num = layer_num

    def forward(self, x):
        out = self.linear(x)

        # Register a backward hook on this layer's output
        out.register_hook(
            lambda grad: print(f"[BACKWARD HOOK] layer_num={self.layer_num}, grad norm={grad.norm().item():.4f}") or grad
        )

        return out

# Define a multi-layer model: 3 layers
class MultiLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = LayerWithBackwardHook(8, 8, layer_num=1)
        self.layer2 = LayerWithBackwardHook(8, 8, layer_num=2)
        self.layer3 = LayerWithBackwardHook(8, 8, layer_num=3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# Instantiate model
model = MultiLayerNet()

# Create input requiring gradients
input_tensor = torch.randn(4, 8, requires_grad=True)

# Forward pass
output = model(input_tensor)

# Simulate loss (simple sum over output)
loss = output.sum()

# Backward pass triggers hooks
loss.backward()
