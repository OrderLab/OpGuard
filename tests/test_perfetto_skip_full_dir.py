"""Full tensor dumps must be pruned before event files are loaded."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from opguard.trace_export.perfetto.timeline import list_pairs


class SkipFullDirectoryTests(unittest.TestCase):
    def test_prune_full_dumps_preserving_other_pairing(self):
        with tempfile.TemporaryDirectory() as temp:
            for prefix in ("", "exec_trace/gpu_000/"):
                bases = [Path(temp) / str(len(prefix)) / side for side in ("left", "right")]
                common = {prefix + "ops/mm/events.jsonl", prefix + "ops/mm/legacy.pt",
                          "reports/full/metadata.json"}
                for base in bases:
                    for relative in common | {prefix + "ops/full/mm/shared.pt"}:
                        path = base / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.touch()
                for base, side in zip(bases, ("left", "right")):
                    (base / (prefix + f"ops/full/mm/{side}.pt")).touch()
                    (base / (prefix + f"ops/mm/{side}.jsonl")).touch()
                visited = []
                real_walk = os.walk

                def walk(*args, **kwargs):
                    for entry in real_walk(*args, **kwargs):
                        visited.append(Path(entry[0]))
                        yield entry

                with patch("opguard.trace_export.perfetto.timeline.os.walk", walk):
                    pairs, left, right = list_pairs(*bases)
                self.assertEqual({pair.rel for pair in pairs}, common)
                self.assertEqual(left, [(bases[0] / (prefix + "ops/mm/left.jsonl")).resolve()])
                self.assertEqual(right, [(bases[1] / (prefix + "ops/mm/right.jsonl")).resolve()])
                for base in bases:
                    full = base / (prefix + "ops/full")
                    self.assertFalse(any(full == p or full in p.parents for p in visited))


if __name__ == "__main__":
    unittest.main()
