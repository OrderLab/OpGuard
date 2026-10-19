import gzip
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from opguard_harness.trace_reader import (Trace, first_diff, inspect_event, compare_events,
                                        source_excerpt, search_source, trace_neighbors, trace_summary, code_context)
from opguard_harness.tools import dispatch


def op(name, run, ts, counter, status='diff', rank=0, pid=1):
    return {'ph': 'X', 'cat': 'opguard', 'name': name, 'ts': ts, 'dur': 10, 'pid': pid, 'tid': 10,
            'args': {'run': run, 'rank': rank, 'status': status, 'counter': counter,
                     'fn': name, 'line': 5, 'file': '/source/model.py', 'sig': name,
                     'io': {'inputs': [{'xor_signature': 0}], 'outputs': [{'xor_signature': 123}]}}}


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'trace.json'

    def write(self, events, sources=None):
        self.path.write_text(json.dumps({'traceEvents': events, 'sourceFiles': sources or {}}))
        return str(self.path)

    def test_first_per_run_rank_not_global_time(self):
        events = [op('add', 'A', 10, 1, 'ok'), op('divide', 'A', 20, 2),
                  op('matmul', 'A', 1, 1, rank=1), op('add', 'B', 100, 1, 'insertion', pid=2)]
        data = first_diff(self.write(events))
        self.assertEqual(len(data['boundaries']), 3)
        self.assertEqual(data['boundaries'][0]['previous_exporter_ok']['name'], 'add')
        self.assertIsNone(data['boundaries'][2]['first_observed_diff'])
        self.assertEqual(data['boundaries'][2]['unaligned_before'], 1)
        self.assertEqual(data['root_cause'], 'not_determined')

    def test_stack_closes_without_category_after_op_at_same_timestamp(self):
        events = [
            {'ph': 'B', 'cat': 'opguard:stack', 'pid': 1, 'tid': 12, 'ts': 0,
             'args': {'run': 'A', 'rank': 0, 'func': 'old', 'line': 1}},
            op('divide', 'A', 10, 1),
            {'ph': 'E', 'pid': 1, 'tid': 12, 'ts': 10},
            {'ph': 'B', 'cat': 'opguard:stack', 'pid': 1, 'tid': 12, 'ts': 10,
             'args': {'run': 'A', 'rank': 0, 'func': 'current', 'line': 8,
                      'function_context': {'target_line': 8}}}]
        trace = Trace(self.write(events))
        context = inspect_event(str(self.path), trace.ref(1))['source_context']
        self.assertEqual([x['function'] for x in context], ['current'])
        self.assertEqual(context[0]['target_line'], 8)

    def flow_events(self):
        return [op('divide', 'A', 10, 1), op('divide', 'B', 30, 99, pid=2),
                {'ph': 's', 'cat': 'opguard:align', 'id': 3, 'ts': 19, 'pid': 1, 'tid': 10},
                {'ph': 'f', 'cat': 'opguard:align', 'id': 3, 'ts': 31, 'pid': 2, 'tid': 10}]

    def test_valid_alignment_is_not_dependency(self):
        trace = Trace(self.write(self.flow_events()))
        result = inspect_event(str(self.path), trace.ref(0))
        self.assertTrue(result['alignment']['unique'])
        self.assertFalse(result['alignment']['is_data_dependency'])
        self.assertEqual(result['input_equality'], 'not_established')
        self.assertEqual(result['peer_evidence'][0]['event']['counter'], 99)

    def declared_pair(self):
        events = [op('divide', 'A', 10, 1), op('divide', 'B', 30, 99, pid=2)]
        for i, event in enumerate(events):
            event['args']['record_id'] = str(i)
            event['args']['correspondence'] = {
                'state': 'paired', 'basis': 'callsite_sequence_order',
                'peer_record_id': str(1-i), 'uses_output_values': False,
            }
        return events

    def test_reciprocal_record_ids_work_without_visual_flows(self):
        trace = Trace(self.write(self.declared_pair()))
        result = inspect_event(str(self.path), trace.ref(0))
        self.assertTrue(result['alignment']['unique'])
        self.assertEqual(result['alignment']['basis'], 'exporter_correspondence')
        self.assertEqual(result['peer_evidence'][0]['event']['counter'], 99)
        self.assertFalse(result['alignment']['is_data_dependency'])
        self.assertEqual(result['input_equality'], 'not_established')

    def test_invalid_explicit_pair_does_not_fall_back_to_visual_guess(self):
        events = self.declared_pair()
        events[1]['args']['correspondence']['peer_record_id'] = 'wrong'
        trace = Trace(self.write(events))
        self.assertFalse(trace.aligned(0)['unique'])
        self.assertTrue(trace.aligned(0)['inconsistent_endpoints'])
        events = self.declared_pair()
        events.append(events[1].copy())
        trace = Trace(self.write(events))
        self.assertFalse(trace.aligned(0)['unique'])
        self.assertEqual(trace.aligned(0)['peers'], [])

    def test_undetermined_stays_unknown_even_with_legacy_flow(self):
        events = self.flow_events()
        events[0]['args'].update(status='unaligned', correspondence={
            'state': 'undetermined', 'reason': 'ambiguous_call_sequence'})
        trace = Trace(self.write(events))
        self.assertEqual(trace.aligned(0)['peers'], [])
        detail = inspect_event(str(self.path), trace.ref(0))
        self.assertEqual(detail['event']['correspondence']['reason'], 'ambiguous_call_sequence')

    def test_unaligned_prefix_is_exposed_before_observed_diff(self):
        events = [op('call', 'A', 0, 1, 'unaligned'), op('save', 'A', 10, 2)]
        events[0]['args']['correspondence'] = {'state': 'undetermined', 'reason': 'no_matching_callsite'}
        boundary = first_diff(self.write(events))['boundaries'][0]
        self.assertEqual(boundary['first_observed_diff']['name'], 'save')
        self.assertEqual(boundary['unaligned_before'], 1)
        self.assertEqual(boundary['first_unaligned']['name'], 'call')

    def test_bad_flow_and_duplicate_ids_do_not_force_pairing(self):
        events = self.flow_events()
        events[1]['args']['fn'] = 'unrelated'
        trace = Trace(self.write(events))
        result = trace.aligned(0)
        self.assertFalse(result['unique'])
        self.assertEqual(result['peers'], [])
        self.assertTrue(result['inconsistent_endpoints'])
        events.append(events[-1].copy())
        trace = Trace(self.write(events))
        self.assertEqual(trace.aligned(0)['ambiguous_flow_ids'], [3])

    def test_fingerprint_collision_is_not_equality_or_error_magnitude(self):
        self.assertEqual(1 ^ 1, 2 ^ 2)
        trace = Trace(self.write([op('exp', 'A', 10, 1), op('exp', 'B', 30, 1, pid=2)]))
        result = compare_events(str(self.path), trace.ref(0), trace.ref(1))
        self.assertEqual(result['input_equality'], 'not_established')
        self.assertEqual(result['numeric_error'], 'not_available')

    def test_stale_reference_rejected(self):
        trace = Trace(self.write([op('divide', 'A', 10, 1)]))
        reference = trace.ref(0)
        self.write([op('multiply', 'A', 10, 1)])
        with self.assertRaisesRegex(ValueError, 'another artifact'):
            inspect_event(str(self.path), reference)

    def test_sources_bounded_and_no_filesystem_fallback(self):
        self.write([], {'/not/a/local/path.py': {'content': 'first\nneedle\nthird'}})
        self.assertEqual(search_source(str(self.path), 'needle')['hits'][0]['line'], 2)
        self.assertEqual(source_excerpt(str(self.path), '/not/a/local/path.py', 2, 3)['lines'][0]['text'], 'needle')
        with self.assertRaises(ValueError):
            source_excerpt(str(self.path), str(self.path), 1, 2)
        with self.assertRaises(ValueError):
            source_excerpt(str(self.path), '/not/a/local/path.py', 1, 101)

    def test_neighbors_and_readers_do_not_create_store(self):
        trace = Trace(self.write([op('divide', 'A', 10, 1)]))
        self.assertFalse(trace_neighbors(str(self.path), trace.ref(0), 1)['is_data_dependency'])
        store = Path(self.temp.name) / 'must-not-exist'
        dispatch(store, 'trace_summary', {'artifact': str(self.path)})
        self.assertFalse(store.exists())
        with self.assertRaises(ValueError):
            trace_neighbors(str(self.path), trace.ref(0), 21)

    def test_code_context_finds_enclosing_method_and_preserves_line_coordinates(self):
        source = 'class Model:\n    def forward(self, x):\n        y = x * 2\n        return y\n'
        self.write([], {'/captured/model.py': {'content': source}})
        context = code_context(str(self.path), '/captured/model.py', 3)
        self.assertEqual(context['function'], 'Model.forward')
        self.assertEqual(context['function_start'], 2)
        self.assertEqual(context['function_end'], 4)
        self.assertEqual(context['omitted_lines'], 0)
        self.assertEqual(context['lines'][1]['text'].strip(), 'y = x * 2')
        self.assertEqual(context['runtime_branch_execution'], 'not_established')
        with self.assertRaises(ValueError):
            code_context(str(self.path), '/captured/model.py', 500)

    def test_long_function_context_reports_omissions(self):
        source = 'def compute():\n' + ''.join('    x = %s\n' % i for i in range(200))
        self.write([], {'/captured/model.py': {'content': source}})
        context = code_context(str(self.path), '/captured/model.py', 150)
        self.assertGreater(context['omitted_lines'], 0)
        self.assertLessEqual(len(context['lines']), 80)
        self.assertIn(150, [item['line'] for item in context['lines']])

    def test_gzip_limit(self):
        path = Path(self.temp.name) / 'trace.json.gz'
        path.write_bytes(gzip.compress(b' ' * 1000))
        with patch('opguard_harness.trace_reader.MAX_BYTES', 200):
            with self.assertRaisesRegex(ValueError, 'decoded'):
                Trace(str(path))

    def test_stacked_export_fails_explicitly(self):
        event = op('divide', 'A', 10, 1)
        event['ph'] = 'B'
        with self.assertRaisesRegex(ValueError, 'Stacked'):
            Trace(self.write([event]))


@unittest.skipUnless(os.getenv('OPGUARD_DEMO_TRACE'), 'Optional pinned public demo artifact not provided')
class DemoAcceptanceTests(unittest.TestCase):
    def test_known_answer_artifact_contract_not_blind_diagnosis(self):
        path = os.environ['OPGUARD_DEMO_TRACE']
        trace = Trace(path)
        self.assertEqual(trace.sha, '817bfa4b756bcac73e3d83892da1fee3a72bd5ab58c0a5b1e0900277dacdae51')
        boundaries = first_diff(path)['boundaries']
        self.assertEqual(len(boundaries), 2)
        for boundary in boundaries:
            self.assertEqual(boundary['first_observed_diff']['counter'], 2144)
            self.assertIn('vector_norm', boundary['first_observed_diff']['fn'])
            self.assertFalse(boundary['alignment']['unique'])
        detail = inspect_event(path, boundaries[0]['first_observed_diff']['event_ref'])
        self.assertEqual(len(detail['source_context']), 10)
        self.assertEqual(detail['source_context'][-1]['target_line'], 1629)
        self.assertEqual(detail['source_context'][-2]['function'], 'get_grad_norm_direct')
        self.assertEqual(detail['input_equality'], 'not_established')
        self.assertEqual(trace_summary(path)['operation_count'], 5318)
        caller = detail['source_context'][-2]
        context = code_context(path, caller['source'], caller['source_line'])
        self.assertEqual(context['function'], 'DeepSpeedZeroOptimizer.get_grad_norm_direct')
        self.assertTrue(any('g.data.double().detach()' in item['text'] for item in context['lines']))
