import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from opguard_harness.tools import dispatch
from opguard_harness.trace_reader import Trace, first_diff, inspect_event


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = self.root / 'store'
        self.artifact = self.root / 'trace.json'
        events = []
        for i in range(4):
            events.append({'cat': 'opguard', 'ph': 'X', 'name': 'op', 'pid': i % 2,
                'tid': 'gpu_0', 'ts': i * 100, 'dur': 20, 'args': {
                    'record_id': str(i), 'run': 'A' if i % 2 == 0 else 'B', 'rank': 0,
                    'fn': 'helper', 'file': '/model.py', 'line': 2, 'status': 'diff',
                    'diff': {'fields': ['outputs'], 'summary': f'original {i}'},
                    'correspondence': {'state': 'paired', 'basis': 'callsite_sequence_order', 'peer_record_id': str(i ^ 1)},
                    'io': {'inputs': [], 'outputs': [{'shape': [2], 'dtype': 'torch.float32', 'xor_signature': i}]}}})
        self.artifact.write_text(json.dumps({'traceEvents': events}))
        self.trace = Trace(self.artifact)

    def call(self, tool, **kwargs):
        return dispatch(self.store, tool, {'artifact': str(self.artifact), **kwargs})

    def review(self, a=0, b=1, decision='not_comparable', slots=None):
        return {'left_ref': self.trace.ref(a), 'right_ref': self.trace.ref(b),
                'left_tensor': slots, 'right_tensor': slots, 'decision': decision,
                'reason': 'Logical domains differ under this source snapshot.',
                'evidence': [{'basis': 'source_inference', 'reference': 'source-sha:function:line'}]}

    def queue(self, cursor=0, limit=50, include_reviewed=False):
        return self.call('difference_candidates', cursor=cursor, limit=limit, include_reviewed=include_reviewed)

    def test_reject_and_continue_without_changing_trace(self):
        first = self.queue(limit=1)
        self.assertEqual(first['candidate_count'], 2)
        self.assertFalse(self.store.exists())
        result = self.call('review_pair', review=self.review())
        second = self.queue()
        self.assertEqual(second['candidates'][0]['left']['event_ref'], self.trace.ref(2))
        self.assertEqual(second['unreviewed_count'], 1)
        self.assertEqual(len(self.queue(include_reviewed=True)['candidates']), 2)
        self.assertEqual(Trace(self.artifact).sha, self.trace.sha)
        self.assertEqual(first_diff(str(self.artifact))['boundaries'][0]['first_observed_diff']['event_ref'], self.trace.ref(0))
        self.assertEqual(result['verdict'], 'not_evaluated')
        # Cursor indexes the immutable candidate list, so reviewing does not skip the next pair.
        self.assertEqual(self.queue(cursor=first['next_cursor'])['candidates'][0]['left']['event_ref'], self.trace.ref(2))

    def test_undetermined_is_visible_and_history_is_revisable(self):
        first = self.call('review_pair', review=self.review(decision='undetermined'))
        self.assertEqual(self.queue()['review_counts'], {'undetermined': 1})
        self.call('review_pair', review=self.review(decision='comparable'))
        history = json.loads(Path(first['alignment_map']).read_text())['reviews']
        self.assertEqual([r['decision'] for r in history], ['undetermined', 'comparable'])
        self.call('review_pair', review=self.review(2, 3, 'undetermined'))
        result = self.queue()
        self.assertEqual(result['candidates'], [])
        self.assertEqual(result['review_counts']['undetermined'], 1)
        self.assertEqual(result['verdict'], 'not_evaluated')

    def test_stale_refs_missing_evidence_and_new_unspecified_boundary_fail(self):
        for modify in (lambda r: r.update(left_ref='0' * 64 + ':0'),
                       lambda r: r.update(evidence=[]),
                       lambda r: r.update(right_ref=self.trace.ref(2)),
                       lambda r: r.update(right_ref=self.trace.ref(3), decision='comparable'),
                       lambda r: r.update(transform='transpose')):
            review = self.review()
            modify(review)
            with self.assertRaises(ValueError):
                self.call('review_pair', review=review)
        self.assertFalse(self.store.exists())

    def test_conflicting_comparable_maps_rejected_without_losing_previous_review(self):
        result = self.call('review_pair', review=self.review(decision='comparable'))
        before = Path(result['alignment_map']).read_bytes()
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            self.call('review_pair', review=self.review(0, 3, 'comparable', ['outputs', 0]))
        self.assertEqual(Path(result['alignment_map']).read_bytes(), before)

    def test_reviews_are_bound_to_artifact_and_do_not_suppress_new_capture(self):
        self.call('review_pair', review=self.review())
        data = json.loads(self.artifact.read_text())
        data['new_capture'] = True
        self.artifact.write_text(json.dumps(data))
        self.assertEqual(self.queue()['unreviewed_count'], 2)

    def test_inconsistent_exporter_proposal_can_still_be_reviewed(self):
        data = json.loads(self.artifact.read_text())
        data['traceEvents'][1]['args']['fn'] = 'other_function'
        self.artifact.write_text(json.dumps(data))
        self.trace = Trace(self.artifact)
        first = self.queue(limit=1)['candidates'][0]
        self.assertFalse(first['alignment']['unique'])
        self.assertEqual(first['right']['event_ref'], self.trace.ref(1))
        self.call('review_pair', review=self.review())
        self.assertEqual(self.queue()['unreviewed_count'], 1)

    @unittest.skipUnless(importlib.util.find_spec('opguard'), 'OpGuard backend integration requires its checkout/install')
    def test_real_backend_roundtrip_accepts_cross_framework_boundary(self):
        data = json.loads(self.artifact.read_text())
        data['traceEvents'][3]['args'].update(fn='other.fused', file='/other.py', line=99, rank=3)
        self.artifact.write_text(json.dumps(data))
        self.trace = Trace(self.artifact)
        self.call('review_pair', review=self.review(0, 3, 'comparable', ['outputs', 0]))
        output = self.root / 'reviewed.json.gz'
        result = self.call('render_alignment', output=str(output), backend=[sys.executable, '-m', 'opguard.tools'])
        reviewed = Trace(output)
        by_id = {reviewed.events[i]['args']['record_id']: i for i in reviewed.ops}
        detail = inspect_event(str(output), reviewed.ref(by_id['0']))
        self.assertTrue(detail['alignment']['unique'])
        self.assertEqual(detail['alignment']['peers'][0]['fn'], 'other.fused')
        self.assertEqual(detail['alignment']['peers'][0]['rank'], 3)
        self.assertEqual(detail['event']['comparison_scope'], 'selected_tensor_summary')
        self.assertEqual(detail['alignment']['semantic_validation'], 'not_independently_verified')
        self.assertEqual(result['verdict'], 'not_evaluated')
        self.assertEqual(Trace(self.artifact).sha, self.trace.sha)
        with self.assertRaisesRegex(ValueError, 'fresh'):
            self.call('render_alignment', output=str(output), backend=[sys.executable, '-m', 'opguard.tools'])


if __name__ == '__main__':
    unittest.main()
