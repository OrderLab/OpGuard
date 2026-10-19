# test_trace_with_meta.py

import os
import json
import torch
import tempfile
import shutil
import pytest

from opguard.runtime_tracer import (
    flush_trace_buffer,
)
from opguard._running import get_trace_output_dir
from opguard.runtime_tracer import save_tensor_trace
from opguard.tracer.meta_vars import update_meta, ensure_meta_vars_initialized

import torch.distributed as dist
from unittest.mock import patch


# Patchable distributed rank getter
def mock_dist_rank(rank):
    import torch.distributed as dist
    dist.is_initialized = lambda: True
    dist.get_rank = lambda: rank

@patch("torch.distributed.is_initialized", return_value=True)
@patch("torch.distributed.get_world_size", return_value=4)
def test_trace_with_meta_vars(mock_is_initialized, mock_get_world_size):
    # Setup temp trace dir
    temp_prefix = "_test_meta_trace"
    os.environ["OPGUARD_TRACE_OUTPUT_BASE"] = tempfile.mkdtemp(prefix=temp_prefix)

    # Simulate rank 2
    rank = 2
    mock_dist_rank(rank)
    ensure_meta_vars_initialized()

    # Set meta values for rank 2
    update_meta(global_step=101, model_name="Mixtral", run_tag="test-trace")

    # Trigger a trace
    x = torch.tensor([1.0, 2.0])
    y = x + 5
    save_tensor_trace(
        tag="meta-test",
        args=(x,),
        outputs=y,
        metadata={"test": "manual_meta"},
        tag_prefix=temp_prefix,
        dump_mode="summary"
    )

    # Make sure the trace is flushed to disk
    flush_trace_buffer(force_dump=True)

    trace_dir = get_trace_output_dir(temp_prefix)
    pt_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(trace_dir)
        for f in files if f.endswith(".jsonl")
    ]

    assert len(pt_files) > 0, "No .jsonl trace files found"

    with open(pt_files[0], "r") as f:
        trace = json.load(f)

    # Verify the meta keys are present
    assert trace["meta"]["global_step"] == 101
    assert trace["meta"]["model_name"] == "Mixtral"
    assert trace["meta"]["run_tag"] == "test-trace"
    assert trace["meta"]["test"] == "manual_meta"
    assert trace["gpu_rank"] == 2

    print(f"[PASS] Trace file saved at: {pt_files[0]}")

    # Cleanup
    shutil.rmtree(trace_dir, ignore_errors=True)
    os.environ.pop("OPGUARD_TRACE_OUTPUT_BASE", None)
