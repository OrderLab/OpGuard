import hashlib, json, os
from pathlib import Path
import torch
from opguard.tracer.tensor_dump import summarize_tensor, finalize_summary

ROOT=Path('/work/diagnostics')
ROUND=json.loads((ROOT/'current.json').read_text())['round']
SELECT={1,2,32,33,64}
def digest_rows(t):
    return [hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest()[:16] for x in t]
def record(site, step, **data):
    rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else int(os.getenv('RANK',0))
    p=ROOT/f'r{ROUND}-{site}-rank{rank}.jsonl'
    with p.open('a') as f:
        f.write(json.dumps(dict(site=site,step=int(step),rank=rank,**data),separators=(',',':'))+'\n')
def summary(t):
    return finalize_summary(summarize_tensor(t.detach().contiguous(),ignore_cpu=False))
def a_batches(trainer, batches, denominator):
    step=trainer.state.global_step+1
    if step in SELECT:
        record('a.batch',step,denominator=denominator,rows=[digest_rows(b['input_ids']) for b in batches],
               targets=[int(b['labels'][...,1:].ne(-100).sum()) for b in batches],
               inputs=[summary(b['input_ids']) for b in batches])
def a_loss(trainer, inputs, outputs, loss, denominator):
    step=trainer.state.global_step+1
    train=trainer.model.training
    if step in SELECT or not train:
        record('a.loss',step,train=train,rows=digest_rows(inputs['input_ids']),
               loss=float(loss.detach()),denominator=denominator,
               targets=int(inputs['labels'][...,1:].ne(-100).sum()),
               accelerator_processes=trainer.accelerator.num_processes,
               accumulation=trainer.accelerator.gradient_accumulation_steps)
def a_grad(trainer, model, grad_norm):
    step=trainer.state.global_step+1
    if step in SELECT:
        record('a.grad',step,norm=float(grad_norm),
               predivide=getattr(model,'_gradient_predivide_factor',None),
               postdivide=getattr(model,'_gradient_postdivide_factor',None),
               world_size=getattr(model,'world_size',None),
               lr=[g['lr'] for g in trainer.optimizer.param_groups])
def b_batch(tokens,loss_mask):
    from megatron.training import get_args
    from megatron.core import mpu
    step=get_args().curr_iteration+1
    if mpu.get_tensor_model_parallel_rank()==0 and (step in SELECT or not torch.is_grad_enabled()):
        record('b.batch',step,train=torch.is_grad_enabled(),rows=digest_rows(tokens),
               targets=int(loss_mask.sum()),inputs=summary(tokens) if torch.is_grad_enabled() else None)
def b_loss(loss, output_tensor):
    from megatron.training import get_args
    from megatron.core import mpu
    step=get_args().curr_iteration+1
    if mpu.get_tensor_model_parallel_rank()==0 and (step in SELECT or not torch.is_grad_enabled()):
        record('b.loss',step,train=torch.is_grad_enabled(),nll_sum=float(loss[0].detach()),targets=int(loss[1]),
               output=summary(output_tensor) if torch.is_grad_enabled() else None)
def b_tokens(num_tokens):
    from megatron.training import get_args
    from megatron.core import mpu
    step=get_args().curr_iteration+1
    if mpu.get_tensor_model_parallel_rank()==0 and step in SELECT:
        record('b.tokens',step,global_tokens=int(num_tokens))
