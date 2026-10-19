import copy
import hashlib
import json

import pytest

from opguard.trace_export.perfetto.alignment_map import apply_reviews, render_alignment, read_export
from opguard.trace_export.perfetto.build import build_perfetto


def operation(run, index, value, peer=None, status='diff'):
    return {'ph': 'X', 'cat': 'opguard', 'name': f'{run}-op-{index}', 'pid': run,
            'tid': 'gpu_0', 'ts': index * 100, 'dur': 20, 'args': {
                'record_id': str(index), 'run': run, 'rank': 0, 'fn': 'helper',
                'file': '/model.py', 'line': 2, 'status': status, 'first_diff': True,
                'io': {'inputs': [], 'outputs': [{'shape': [2], 'dtype': 'torch.float32',
                       'numel': 2, 'device': 'cpu', 'xor_signature': value}]},
                'diff': {'fields': ['outputs'], 'summary': 'original observation'},
                'correspondence': {'state': 'paired' if peer is not None else 'undetermined',
                                   'peer_record_id': str(peer), 'basis': 'callsite_sequence_order'}}}


def source(tmp_path):
    doc = {'traceEvents': [operation('run1', 0, 1, 1), operation('run2', 1, 2, 0),
                           operation('run1', 2, 3, 3), operation('run2', 3, 4, 2)],
           'sourceFiles': {'model.py': {'content': 'preserved source'}}}
    path = tmp_path / 'source.json'
    path.write_text(json.dumps(doc))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return doc, path, sha


def review(sha, a=0, b=1, decision='not_comparable', slots=None):
    return {'left_ref': f'{sha}:{a}', 'right_ref': f'{sha}:{b}',
            'left_tensor': slots, 'right_tensor': slots, 'decision': decision,
            'reason': 'Evidence establishes the logical scopes for this pair.',
            'evidence': [{'basis': 'source_inference', 'reference': 'source snapshot sha256 + function location'}]}


def mapping(sha, *reviews):
    return {'version': 1, 'artifact_sha256': sha, 'reviews': list(reviews)}


def ops(doc):
    return {e['args']['record_id']: e for e in doc['traceEvents']
            if e.get('cat') == 'opguard' and e.get('ph') == 'X'}


def test_rejected_pair_does_not_erase_observations_or_later_differences(tmp_path):
    doc, path, sha = source(tmp_path)
    original = copy.deepcopy(doc)
    result = apply_reviews(doc, sha, mapping(sha, review(sha)))
    records = ops(result)
    assert len(records) == 4 and doc == original
    for key in ('0', '1'):
        args = records[key]['args']
        assert args['status'] == 'unaligned'
        assert 'diff' not in args and 'first_diff' not in args
        assert args['original_comparison']['diff']['summary'] == 'original observation'
        assert args['io'] == original['traceEvents'][int(key)]['args']['io']
    assert all(records[key]['args']['first_diff'] for key in ('2', '3'))
    flows = [e for e in result['traceEvents'] if e.get('cat') == 'opguard:align']
    assert len(flows) == 2
    assert result['sourceFiles'] == original['sourceFiles']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha


def test_repair_pair_displaces_old_peers_and_compares_only_selected_tensor(tmp_path):
    doc, _, sha = source(tmp_path)
    doc['traceEvents'][3]['args'].update(fn='different_framework.fused', file='/other/layers.py', line=900, rank=7)
    doc['traceEvents'][3]['args']['io']['outputs'].append({'shape': [8], 'dtype': 'torch.float64', 'xor_signature': 999})
    change = review(sha, 0, 3, 'comparable', ['outputs', 0])
    result = ops(apply_reviews(doc, sha, mapping(sha, change)))
    assert result['0']['args']['correspondence']['peer_record_id'] == '3'
    assert result['3']['args']['correspondence']['peer_record_id'] == '0'
    assert result['1']['args']['status'] == result['2']['args']['status'] == 'unaligned'
    assert result['0']['args']['diff']['fields'] == ['xor_signature']
    assert result['0']['args']['correspondence']['semantic_equivalence'] == 'asserted_not_independently_verified'
    assert result['0']['args']['comparison_scope'] == 'selected_tensor_summary'


def test_cross_file_events_are_available_to_explicit_map(tmp_path):
    for side, name, fn, value in [('a', 'torch/add_line1.jsonl', 'add', 8),
                                  ('b', 'other/fused_line99.jsonl', 'fused', 9)]:
        path = tmp_path / side / name
        path.parent.mkdir(parents=True)
        record = {'function': fn, 'fname': f'/{side}/model.py', 'line_num': 1 if side == 'a' else 99,
                  'gpu_rank': 0, 'counter': 1, 'inputs': {'args': [], 'kwargs': {}},
                  'outputs': [{'shape': [1], 'stats': {'dtype': 'torch.float32', 'numel': 1,
                               'device': 'cpu', 'xor_signature': value, 'xor_sig_mode': 'bytes'}}]}
        path.write_text(json.dumps(record) + '\n')
    original = tmp_path / 'original.json.gz'
    build_perfetto(str(tmp_path / 'a'), str(tmp_path / 'b'), str(original))
    doc, sha = read_export(original)
    indices = [i for i, e in enumerate(doc['traceEvents']) if e.get('cat') == 'opguard' and e.get('ph') == 'X']
    assert all(doc['traceEvents'][i]['args']['status'] == 'unaligned' for i in indices)
    changed = apply_reviews(doc, sha, mapping(sha, review(sha, *indices, 'comparable', ['outputs', 0])))
    assert all(e['args']['status'] == 'diff' for e in ops(changed).values())
    assert all(e['args']['diff']['fields'] == ['xor_signature'] for e in ops(changed).values())


def test_equal_summary_and_review_do_not_certify_bitwise_equality(tmp_path):
    doc, _, sha = source(tmp_path)
    # Different arrays [1,1] and [2,2] can both have this XOR.
    for event in doc['traceEvents'][:2]:
        event['args']['io']['outputs'][0]['xor_signature'] = 0
    result = ops(apply_reviews(doc, sha, mapping(sha, review(sha, decision='comparable', slots=['outputs', 0]))))
    assert result['0']['args']['status'] == 'ok'
    assert result['0']['args']['tensor_equality'] == 'not_established'


def test_output_to_input_boundary_ignores_unselected_outputs(tmp_path):
    doc, _, sha = source(tmp_path)
    doc['traceEvents'][3]['args']['io']['inputs'] = [
        {'shape': [50], 'dtype': 'torch.int64', 'xor_signature': 100},
        copy.deepcopy(doc['traceEvents'][0]['args']['io']['outputs'][0]),
    ]
    change = review(sha, 0, 3, 'comparable', ['outputs', 0])
    change['right_tensor'] = ['inputs', 1]
    result = ops(apply_reviews(doc, sha, mapping(sha, change)))
    assert result['0']['args']['status'] == 'ok'
    assert result['0']['args']['correspondence']['peer_tensor_slot'] == ['inputs', 1]
    assert result['3']['args']['io']['outputs'][0]['xor_signature'] == 4


def test_latest_review_revises_decision_without_discarding_history(tmp_path):
    doc, _, sha = source(tmp_path)
    changes = mapping(sha, review(sha), review(sha, decision='comparable'))
    result = apply_reviews(doc, sha, changes)
    assert ops(result)['0']['args']['status'] == 'diff'
    assert len(result['alignment_review']['map']['reviews']) == 2


@pytest.mark.parametrize('change', [
    lambda m: m.update(artifact_sha256='0' * 64),
    lambda m: m['reviews'][0].update(left_ref='0' * 64 + ':0'),
    lambda m: m['reviews'][0].update(evidence=[]),
    lambda m: m['reviews'][0].update(transform='transpose_until_equal'),
    lambda m: m['reviews'][0].update(decision='comparable', left_tensor=['outputs', 99], right_tensor=['outputs', 0]),
])
def test_invalid_map_fails_without_creating_output(tmp_path, change):
    _, path, sha = source(tmp_path)
    spec = mapping(sha, review(sha))
    change(spec)
    map_path = tmp_path / 'map.json'
    map_path.write_text(json.dumps(spec))
    output = tmp_path / 'new.json.gz'
    with pytest.raises(ValueError):
        render_alignment(path, map_path, output)
    assert not output.exists()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha


def test_conflicting_accepted_pairs_are_not_chosen_arbitrarily(tmp_path):
    doc, _, sha = source(tmp_path)
    with pytest.raises(ValueError, match='Conflicting'):
        apply_reviews(doc, sha, mapping(sha, review(sha, 0, 1, 'comparable', ['outputs', 0]),
                                       review(sha, 0, 3, 'comparable', ['outputs', 0])))


def test_render_is_new_file_with_parent_and_map_provenance(tmp_path):
    _, path, sha = source(tmp_path)
    map_path = tmp_path / 'map.json'
    map_path.write_text(json.dumps(mapping(sha, review(sha))))
    output = tmp_path / 'reviewed.json.gz'
    result = render_alignment(path, map_path, output)
    doc, new_sha = read_export(output)
    assert result['sha256'] == new_sha and result['source_sha256'] == sha
    assert result['verdict'] == 'not_evaluated'
    assert doc['alignment_review']['map_sha256'] == hashlib.sha256(map_path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        render_alignment(path, map_path, output)
    with pytest.raises(FileExistsError):
        render_alignment(path, map_path, path)
