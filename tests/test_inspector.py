import unittest
import logging
import json

from typing import Any

# Enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def passes_filter(trace: dict, filter_kv: str) -> bool:
    if not filter_kv:
        return True

    try:
        key_path, expected_val = filter_kv.split("=")
        expected_val = json.loads(expected_val)
        keys = key_path.strip().split(".")

        val = trace
        for k in keys:
            val = val.get(k, None)
            if val is None:
                return False
        return val == expected_val
    except Exception as e:
        logger.warning(f"⚠️ Invalid filter or error evaluating filter '{filter_kv}': {e}")
        return False


class TestTraceFilter(unittest.TestCase):
    def setUp(self):
        self.trace1 = {
            "function": "<foo>",
            "meta": {"global_step": 0, "micro_batch_idx": 4}
        }
        self.trace2 = {
            "function": "<foo>",
            "meta": {"global_step": 1, "micro_batch_idx": 4}
        }
        self.trace3 = {
            "function": "<bar>",
            "meta": {"global_step": 0}
        }
        self.trace4 = {
            "meta": {}
        }

    def test_match_global_step(self):
        self.assertTrue(passes_filter(self.trace1, "meta.global_step=0"))
        self.assertFalse(passes_filter(self.trace2, "meta.global_step=0"))

    def test_missing_key(self):
        self.assertFalse(passes_filter(self.trace4, "meta.global_step=0"))

    def test_match_multiple_trace_types(self):
        traces = [self.trace1, self.trace2, self.trace3]
        filtered = [t for t in traces if passes_filter(t, "meta.global_step=0")]
        self.assertEqual(len(filtered), 2)
        self.assertIn(self.trace1, filtered)
        self.assertIn(self.trace3, filtered)

    def test_match_string_value(self):
        trace = {"meta": {"stage": "train"}}
        self.assertTrue(passes_filter(trace, "meta.stage=\"train\""))
        self.assertFalse(passes_filter(trace, "meta.stage=\"eval\""))

    def test_invalid_filter_string(self):
        with self.assertLogs(logger, level='WARNING') as log:
            self.assertFalse(passes_filter(self.trace1, "meta="))
        with self.assertLogs(logger, level='WARNING') as log:
            self.assertFalse(passes_filter(self.trace1, "meta.global_step"))

if __name__ == "__main__":
    unittest.main()
