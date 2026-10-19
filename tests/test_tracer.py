import unittest
import torch
import os
import json
import tempfile
from opguard.runtime_tracer import (
    scoped_tensor_tracer,
    _trace_counter,
    flush_trace_buffer,
)
from opguard._running import get_trace_output_dir

class TestTensorTracer(unittest.TestCase):
    def setUp(self):
        import opguard.runtime_tracer as tracer
        _trace_counter["count"] = 0
        self.temp_dir = tempfile.TemporaryDirectory()
        # Set the environment variable directly
        os.environ["OPGUARD_TRACE_OUTPUT_BASE"] = self.temp_dir.name
        print("Set OPGUARD_TRACE_OUTPUT_BASE environment variable to ", self.temp_dir.name)
        self.tag_prefix = "_testtrace"
    
    def tearDown(self):
        self.temp_dir.cleanup()
        import opguard.runtime_tracer as tracer
        import opguard.tracer.trace_buffer as trace_buffer
        trace_buffer._trace_output_cache = {}
        # Set the environment variable directly
        os.environ["OPGUARD_TRACE_OUTPUT_BASE"] = "."
        print("Set OPGUARD_TRACE_OUTPUT_BASE environment variable to None")
        # close all files
        trace_buffer.TRACE_BUFFER.close_all_files()


    def test_trace_summary_mode(self):
        @scoped_tensor_tracer(tag_prefix=self.tag_prefix, dump_tensors=True)
        def dummy_fn(x):
            return x * 2

        x = torch.randn(4, 4)
        dummy_fn(x)

        trace_dir = get_trace_output_dir(self.tag_prefix)
        print(trace_dir)
        json_files = [
            os.path.join(root, f)
            for root, _, files in os.walk(trace_dir)
            for f in files if f.endswith(".jsonl")
        ]
        print(json_files)
        self.assertTrue(len(json_files) == 0)
        
        flush_trace_buffer() # without force_dump, should not dump
        
        json_files = [
            os.path.join(root, f)
            for root, _, files in os.walk(trace_dir)
            for f in files if f.endswith(".jsonl")
        ]
        
        self.assertTrue(len(json_files) == 0)
        
        
        flush_trace_buffer(force_dump=True)
        json_files = [
            os.path.join(root, f)
            for root, _, files in os.walk(trace_dir)
            for f in files if f.endswith(".jsonl")
        ]
        self.assertTrue(len(json_files) > 0)

        with open(json_files[0], "r") as f:
            data = json.load(f)
            self.assertIn("inputs", data)
            self.assertIn("outputs", data)
            self.assertIn("stack_trace", data)
            self.assertTrue(isinstance(data["outputs"], dict))

    def test_trace_summary_mode_2(self):
        from opguard.runtime_tracer import save_tensor_trace

        x = torch.randn(2, 2)
        y = x + 1

        save_tensor_trace(
            tag="manual",
            args=(x,),
            outputs=y,
            metadata={"test": "summary_mode"},
            tag_prefix=self.tag_prefix,
            dump_mode="summary"
        )

        trace_dir = get_trace_output_dir(self.tag_prefix)
        print(trace_dir)
        pt_files = [
            os.path.join(root, f)
            for root, _, files in os.walk(trace_dir)
            for f in files if f.endswith(".jsonl")
        ]
        print(pt_files)
        self.assertTrue(len(pt_files) == 0)
        
        flush_trace_buffer(force_dump=True)
        pt_files = [
            os.path.join(root, f)
            for root, _, files in os.walk(trace_dir)
            for f in files if f.endswith(".jsonl")
        ]
        self.assertTrue(len(pt_files) > 0)


        with open(pt_files[0], "r") as f:
            loaded = json.load(f)
        self.assertIn("inputs", loaded)
        self.assertTrue(isinstance(loaded["outputs"], dict))

    def test_tracer_skipped(self):
        trace_dir = get_trace_output_dir(self.tag_prefix)
        all_files = [
            f
            for root, _, files in os.walk(trace_dir)
            for f in files
        ]
        @scoped_tensor_tracer(tag_prefix=self.tag_prefix, dump_tensors=False)
        def dummy_fn(x):
            return x * 3

        x = torch.tensor([1.0])
        dummy_fn(x)
        all_files_after = [
            f
            for root, _, files in os.walk(trace_dir)
            for f in files
        ]
        changed_files = set(all_files_after) - set(all_files)
        self.assertEqual(len(changed_files), 0)


if __name__ == "__main__":
    unittest.main()
