# test_meta_vars.py

import pytest
from opguard.tracer.meta_vars import meta_vars, update_meta

# Simulate 4 ranks
def setup_module(module):
    # Resize for test
    meta_vars.clear()
    meta_vars.extend([{} for _ in range(4)])

def test_update_meta_per_rank():
    # Simulate update on rank 0
    rank = 0
    update_meta_for_rank(rank, global_step=42, model_name="Mixtral")
    
    assert meta_vars[0]["global_step"] == 42
    assert meta_vars[0]["model_name"] == "Mixtral"
    
    # Simulate update on rank 1
    rank = 1
    update_meta_for_rank(rank, global_step=999, model_name="LLaMA")
    
    assert meta_vars[1]["global_step"] == 999
    assert meta_vars[1]["model_name"] == "LLaMA"
    
    # Ensure isolation between ranks
    assert meta_vars[0]["global_step"] != meta_vars[1]["global_step"]

def update_meta_for_rank(rank, **kwargs):
    # Override rank within update_meta
    import torch.distributed as dist
    dist.get_rank = lambda: rank
    dist.is_initialized = lambda: True
    
    update_meta(**kwargs)
