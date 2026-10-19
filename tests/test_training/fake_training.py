import torch


def rmsnorm_backward(grad, invvar, input):
    grad = grad * invvar
    grad = grad / torch.sqrt(torch.mean(grad * grad, dim=-1, keepdim=True))
    return grad, grad, grad


class FakeModel:

    def forward(self, which_ln):
        grad_ln1 = torch.ones(3, 3)
        grad_ln2 = torch.zeros(3, 3)
        invvar = torch.ones(3, 3)
        input = torch.randn(3, 3)
        if which_ln == 'ln1':
            return rmsnorm_backward(grad_ln1, invvar, input)
        else:
            return rmsnorm_backward(grad_ln2, invvar, input)


if __name__ == '__main__':
    model = FakeModel()
    for epoch in range(2):
        for iteration in range(3):
            out = model.forward('ln1' if iteration % 2 == 0 else 'ln2')
            print(
                f'[epoch={epoch} iter={iteration}] traced output shape: {out[1].shape}, sum: {out[1].sum().item():.3f}'
                )
