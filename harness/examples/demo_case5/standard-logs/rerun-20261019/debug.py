import argparse
from this import d
import deepspeed.runtime.zero.stage_1_and_2
import torch
import torch.nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.moe.utils import is_moe_param

def patch_deepspeed():
    def backward(self, loss, retain_graph=False):
        """
        :attr:`backward` performs the following steps:

        1. fp32_loss = loss.float()
        2. scaled_loss = fp32_loss*loss_scale
        3. scaled_loss.backward(), which accumulates scaled gradients into the ``.grad`` attributes of the model's fp16 leaves
        """
        self.micro_step_id += 1

        if self.contiguous_gradients:
            self.ipg_buffer = []
            self.ipg_events = []  # 添加的代码，yihang: 缓冲区 event
            buf_0 = torch.empty(int(self.reduce_bucket_size),
                                dtype=self.dtype,
                                device=get_accelerator().current_device_name())
            self.ipg_buffer.append(buf_0)
            self.ipg_events.append(None)  # 添加的代码，yihang: 缓冲区 event

            # Use double buffers to avoid data access conflict when overlap_comm is enabled.
            if self.overlap_comm:
                buf_1 = torch.empty(int(self.reduce_bucket_size),
                                    dtype=self.dtype,
                                    device=get_accelerator().current_device_name())
                self.ipg_buffer.append(buf_1)
                self.ipg_events.append(None)   # 添加的代码，yihang: 缓冲区 event

            self.ipg_index = 0

        if self.custom_loss_scaler:
            scaled_loss = self.external_loss_scale * loss
            scaled_loss.backward()
        else:
            self.loss_scaler.backward(loss.float(), retain_graph=retain_graph)

        # Only for Stage 1, Mode 2
        if self.use_grad_accum_attribute:
            self.fill_grad_accum_attribute()

    def reduce_independent_p_g_buckets_and_remove_grads(self, param, i):

        grad_reduc = self.get_gradient_for_reduction(param)
        if self.elements_in_ipg_bucket + param.numel() > self.reduce_bucket_size:
            self.report_ipg_memory_usage("In ipg_remove_grads before reduce_ipg_grads", param.numel())
            self.reduce_ipg_grads()

            # 添加的代码，yihang: 缓冲区 event
            if self.contiguous_gradients and self.overlap_comm and not get_accelerator().is_synchronized_device():
                with get_accelerator().stream(self.reduction_stream):
                    current_event = torch.cuda.Event() # yihang: 在 reduction_stream 上创建一个事件
                    current_event.record(self.reduction_stream)

                    self.ipg_events[self.ipg_index] = current_event

            if self.contiguous_gradients and self.overlap_comm:
                # Swap ipg_index between 0 and 1
                self.ipg_index = 1 - self.ipg_index

            # 添加的代码，yihang: 缓冲区 event
            prev_event = self.ipg_events[self.ipg_index]
            if prev_event is not None:
                prev_event.wait()

            self.report_ipg_memory_usage("In ipg_remove_grads after reduce_ipg_grads", param.numel())

        param_id = self.get_param_id(param)
        assert self.params_already_reduced[param_id] == False, \
            f"The parameter {param_id} has already been reduced. \
            Gradient computed twice for this partition. \
            Multiple gradient reduction is currently not supported"

        if self.contiguous_gradients:
            if param.numel() > self.reduce_bucket_size:
                self.extra_large_param_to_reduce = param
            else:
                # keeping the gradients contiguous to prevent memory fragmentation, and avoid flattening
                new_grad_tensor = self.ipg_buffer[self.ipg_index].narrow(0, self.elements_in_ipg_bucket, param.numel())
                new_grad_tensor.copy_(grad_reduc.view(-1))
                grad_reduc.data = new_grad_tensor.data.view_as(grad_reduc)

        self.elements_in_ipg_bucket += param.numel()

        assert grad_reduc is not None, f"rank {dist.get_rank()} - Invalid to reduce Param {param_id} with None gradient"

        self.grads_in_ipg_bucket.append(grad_reduc)
        self.params_in_ipg_bucket.append((i, param, param_id))

        #make sure the average tensor function knows how to average the gradients
        if is_moe_param(param):
            self.ipg_bucket_has_moe_params = True

        self.report_ipg_memory_usage("End ipg_remove_grads", 0)

    deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer.backward = backward
    deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer.reduce_independent_p_g_buckets_and_remove_grads = reduce_independent_p_g_buckets_and_remove_grads

class DummyModule(torch.nn.Module):
    def __init__(self, n_layer=16, hidden_size=1024, vocab_size=1024) -> None:
        super(DummyModule, self).__init__()

        self._vocab_size = vocab_size

        self.embs = torch.nn.ModuleList([
            torch.nn.Embedding(vocab_size, hidden_size) for _ in range(n_layer)
        ])

    def init_weights(self):
        for emb in self.embs:
            emb.weight.data.normal_(mean=0.0, std=0.0002)
            if emb.padding_idx is not None:
                emb.weight.data[emb.padding_idx].zero_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        idx = torch.range(0, self._vocab_size - 1, device=input.device, dtype=torch.long)

        hidden_states = input
        for emb in self.embs:
            hidden_states = hidden_states + emb(idx)

        return hidden_states.mean()

def train():
    torch.use_deterministic_algorithms(True)

    accelerator = Accelerator(project_dir="./outputs")

    device = accelerator.device
    dtype = torch.float16

    set_seed(42 + accelerator.process_index)

    model = DummyModule()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model, optimizer = accelerator.prepare(model, optimizer)

    for global_step in range(20):
        input = torch.rand(1, dtype=dtype, device=device)
        label = torch.rand(1, dtype=dtype, device=device)

        optimizer.zero_grad()
        output = model(input)
        
        loss = F.mse_loss(output.float(), label.float())

        accelerator.backward(loss)
        optimizer.step()

        global_grad_norm = -100.0
        if hasattr(optimizer.optimizer, '_global_grad_norm'):
            global_grad_norm = optimizer.optimizer._global_grad_norm

        if accelerator.process_index == 0:
            print(f'rank [{accelerator.process_index}] global_step [{global_step}] loss [{loss.item():.9}] grad_norm [{global_grad_norm:.9}]')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_deepspeed", action='store_true', help="patch deepspeed")
    args = parser.parse_args()

    if args.patch_deepspeed:
        patch_deepspeed()

    train()

if __name__ == '__main__':
    main()
