import os
import shutil
import tempfile
import json
import glob
import unittest

from opguard.tracer.trace_buffer import TRACE_BUFFER
from opguard._running import get_trace_output_dir
from opguard.runtime_tracer import save_tensor_trace, flush_trace_buffer, serialize_tensor_io

class TestTraceBuffer(unittest.TestCase):
    def setUp(self):
        # Create a fresh temp directory for outputs
        self.tmpdir = tempfile.mkdtemp()
        os.environ['OPGUARD_TRACE_OUTPUT_BASE'] = self.tmpdir

        # Reset TraceBuffer internal state
        with TRACE_BUFFER.lock:
            TRACE_BUFFER.buffer.clear()
            TRACE_BUFFER.dump_batch.clear()
            TRACE_BUFFER.full_queue.clear()
            TRACE_BUFFER.total_dump_count = 0
        TRACE_BUFFER.file_handles.clear()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_serialize_tensor_io_summary_only(self):
        # Summary mode should return only lightweight stats
        import torch
        t = torch.tensor([1, 2, 3], dtype=torch.float32)
        result = serialize_tensor_io(t, dump_mode='summary')
        self.assertIsInstance(result, dict)
        self.assertIn('shape', result)
        self.assertIn('stats', result)
        stats = result['stats']
        # Ensure key stats are present
        self.assertIn('xor_signature', stats)
        self.assertIn('dtype', stats)
        self.assertIn('device', stats)
        self.assertIn('numel', stats)

    def test_full_dump_includes_summary_and_raw(self):
        import torch
        t = torch.tensor([[5.0, 6.0]], dtype=torch.float32)
        # Save in full mode: both summary and raw should be enqueued
        save_tensor_trace(
            tag='unittest',
            args=(t,),
            outputs=(t,),
            metadata={},
            tag_prefix='trace',
            dump_mode='full'
        )
        # One summary in buffer, one raw in full_queue
        self.assertEqual(len(TRACE_BUFFER.buffer['unittest']), 1)
        self.assertEqual(len(TRACE_BUFFER.full_queue), 1)

        # Flush to disk
        flush_trace_buffer(force_dump=True)

        # Verify JSON summary file exists
        output_dir = get_trace_output_dir('trace')
        jsonl_files = glob.glob(os.path.join(output_dir, 'gpu_00', '*', '*.jsonl'))
        self.assertTrue(jsonl_files, "Missing summary JSON file")
        with open(jsonl_files[0], 'r') as f:
            data = json.loads(f.readline())
        self.assertEqual(data['dump_mode'], 'full')
        # Summary fields present
        self.assertIsInstance(data['inputs'], list)
        self.assertIn('stats', data['inputs'][0])

        # Verify raw tensor files exist
        print(f"Check output .pt dir: {output_dir}")
        pt_files = glob.glob(os.path.join(output_dir, 'full', 'gpu_00', '*', '*', '*.pt'))
        # [OpGuard] File handle Created at /tmp/tmp3vwzyhf3/gpu_00/<non-function>/<non-function>_line48.jsonl
        # [OpGuard] Saved tensor to /tmp/tmp3vwzyhf3/full/gpu_00/<non-function>/gsNA_mbNA_layerNA/unittest_000001_line48_input_0.pt
        # [OpGuard] Saved tensor to /tmp/tmp3vwzyhf3/full/gpu_00/<non-function>/gsNA_mbNA_layerNA/unittest_000001_line48_output_0.pt
        # [OpGuard] Full tensors written to /tmp/tmp3vwzyhf3/full/gpu_00/<non-function>/gsNA_mbNA_layerNA
        self.assertTrue(pt_files, "Missing .pt raw tensor files")
        loaded = torch.load(pt_files[0])
        self.assertTrue(torch.equal(loaded, t))

if __name__ == '__main__':
    unittest.main()
