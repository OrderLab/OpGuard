"""Bounded, read-only queries over OpGuard Perfetto exports.

No bug-specific rules, tensor equality inference, or synthetic producer graph.
"""
import collections
import ast
import gzip
import hashlib
import json
import io
from pathlib import Path
from .fingerprints import regional_comparisons

MAX_BYTES = 256 * 1024 * 1024


class Trace:
    def __init__(self, artifact):
        self.path = Path(artifact).expanduser().resolve()
        with self.path.open('rb') as stream:
            encoded = stream.read(MAX_BYTES + 1)
        if len(encoded) > MAX_BYTES:
            raise ValueError('Trace exceeds the 256 MiB encoded query limit')
        self.sha = hashlib.sha256(encoded).hexdigest()
        stream = gzip.GzipFile(fileobj=io.BytesIO(encoded)) if self.path.suffix == '.gz' else io.BytesIO(encoded)
        with stream:
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError('Trace exceeds the 256 MiB decoded query limit')
        data = json.loads(raw)
        if not isinstance(data, dict) or not isinstance(data.get('traceEvents'), list):
            raise ValueError('Expected an OpGuard Perfetto traceEvents export')
        self.events = data['traceEvents']
        self.alignment_policy = data.get('alignment')
        self.alignment_review = data.get('alignment_review')
        self.sources = data.get('sourceFiles', {})
        if not isinstance(self.sources, dict):
            raise ValueError('sourceFiles must be an object')
        if any(e.get('ph') == 'B' and 'status' in e.get('args', {}) for e in self.events):
            raise ValueError('Stacked operation exports are unsupported; use a flat OpGuard export')
        self.ops = [i for i, e in enumerate(self.events) if e.get('ph') == 'X'
                    and e.get('cat') == 'opguard' and 'status' in e.get('args', {})]
        self.op_indices = set(self.ops)
        self.records = collections.defaultdict(list)
        for i in self.ops:
            record_id = self.events[i]['args'].get('record_id')
            if isinstance(record_id, str):
                self.records[record_id].append(i)

    def identity(self):
        return {'path': str(self.path), 'sha256': self.sha}

    def ref(self, index):
        return f'{self.sha}:{index}'

    def index(self, reference):
        sha, number = reference.rsplit(':', 1)
        if sha != self.sha:
            raise ValueError('Event reference belongs to another artifact revision')
        index = int(number)
        if index not in self.op_indices:
            raise ValueError('Reference is not an operation event')
        return index

    def compact(self, index):
        event = self.events[index]
        args = event['args']
        return {'event_ref': self.ref(index), 'name': event.get('name'),
                **{key: args.get(key) for key in ('run', 'rank', 'counter', 'status', 'fn', 'file', 'line')},
                'export_position': event.get('ts'), 'first_diff_marker': args.get('first_diff', False),
                'correspondence': args.get('correspondence'),
                'correspondence_validation': (args.get('correspondence') or {}).get('validation', 'unreviewed'),
                'comparison_scope': args.get('comparison_scope', 'candidate_recorded_fields'),
                'comparison_evidence': args.get('comparison_evidence', 'recorded_metadata_and_fingerprints'),
                'tensor_equality': 'not_established'}

    def sequence(self, index):
        args = self.events[index]['args']
        return sorted((i for i in self.ops if all(self.events[i]['args'].get(k) == args.get(k)
                                                for k in ('run', 'rank'))),
                      key=lambda i: (self.events[i].get('ts', 0), i))

    def aligned(self, index):
        event = self.events[index]
        if 'correspondence' in event['args']:
            return self.declared_alignment(index)
        def contains(op, endpoint):
            return (op.get('pid') == endpoint.get('pid') and op.get('tid') == endpoint.get('tid')
                    and op.get('ts', 0) <= endpoint.get('ts', -1)
                    < op.get('ts', 0) + op.get('dur', 0))
        flows = collections.defaultdict(list)
        for endpoint in self.events:
            if endpoint.get('cat') == 'opguard:align' and endpoint.get('ph') in ('s', 'f'):
                flows[endpoint.get('id')].append(endpoint)
        peers, ambiguous, inconsistent = [], [], []
        for key, endpoints in flows.items():
            if not any(contains(event, end) for end in endpoints):
                continue
            if len(endpoints) != 2 or {e['ph'] for e in endpoints} != {'s', 'f'}:
                ambiguous.append(key)
                continue
            resolved = [[i for i in self.ops if contains(self.events[i], end)] for end in endpoints]
            if any(len(items) != 1 for items in resolved):
                ambiguous.append(key)
                continue
            pair = [items[0] for items in resolved]
            if index in pair:
                for other in pair:
                    if other == index:
                        continue
                    left, right = event['args'], self.events[other]['args']
                    if not left.get('fn') or left.get('fn') != right.get('fn') or left.get('line') is None or left.get('line') != right.get('line'):
                        inconsistent.append({'flow_id': key, 'candidate': self.compact(other), 'reason': 'flow endpoint operator/location disagrees'})
                    else:
                        peers.append(other)
        peers = sorted(set(peers))
        return {'basis': 'exporter_alignment_flow', 'peers': [self.compact(i) for i in peers],
                'ambiguous_flow_ids': ambiguous,
                'unique': len(peers) == 1 and not ambiguous and not inconsistent,
                'inconsistent_endpoints': inconsistent,
                'is_data_dependency': False, 'semantic_validation': 'not_independently_verified'}

    def declared_alignment(self, index):
        """Validate reciprocal exported IDs; unknown relations never use flow guesses."""
        args = self.events[index]['args']
        claim = args.get('correspondence')
        result = {'basis': 'exporter_correspondence', 'correspondence': claim,
                  'peers': [], 'unique': False, 'ambiguous_flow_ids': [],
                  'inconsistent_endpoints': [], 'is_data_dependency': False,
                  'semantic_validation': 'not_independently_verified'}
        if not isinstance(claim, dict) or claim.get('state') != 'paired':
            return result
        record_id, peer_id = args.get('record_id'), claim.get('peer_record_id')
        if (not isinstance(record_id, str) or not isinstance(peer_id, str)
                or self.records.get(record_id) != [index] or len(self.records.get(peer_id, [])) != 1):
            result['inconsistent_endpoints'] = [{'reason': 'missing_or_duplicate_record_id'}]
            return result
        peer_index = self.records[peer_id][0]
        peer = self.events[peer_index]['args']
        back = peer.get('correspondence') or {}
        if (not isinstance(back, dict) or back.get('state') != 'paired'
                or back.get('peer_record_id') != record_id
                or back.get('basis') != claim.get('basis')
                or args.get('run') == peer.get('run')
                or args.get('status') == 'unaligned' or peer.get('status') == 'unaligned'):
            result['inconsistent_endpoints'] = [{'reason': 'nonreciprocal_or_incompatible_correspondence'}]
            return result
        if claim.get('basis') == 'explicit_alignment_map':
            if (claim.get('validation') != 'agent_reviewed' or back.get('validation') != 'agent_reviewed'
                    or claim.get('tensor_slot') != back.get('peer_tensor_slot')
                    or claim.get('peer_tensor_slot') != back.get('tensor_slot')):
                result['inconsistent_endpoints'] = [{'reason': 'incompatible_explicit_tensor_slots'}]
                return result
        elif (not args.get('fn') or args.get('fn') != peer.get('fn')
                or args.get('line') is None or args.get('line') != peer.get('line')
                or not args.get('file') or not peer.get('file')
                or Path(args['file']).name != Path(peer['file']).name
                or args.get('rank') != peer.get('rank')):
            result['inconsistent_endpoints'] = [{'reason': 'incompatible_automatic_candidate'}]
            return result
        result.update(peers=[self.compact(peer_index)], unique=True)
        return result

    def stack(self, index):
        event = self.events[index]
        stacks = collections.defaultdict(list)
        tracks = {e.get('tid') for e in self.events if e.get('pid') == event.get('pid') and e.get('cat') == 'opguard:stack'}
        candidates = [(i, e) for i, e in enumerate(self.events)
                      if e.get('pid') == event.get('pid') and e.get('ph') in ('B', 'E')
                      and e.get('tid') in tracks and e.get('ts', 0) <= event.get('ts', 0)
                      and (e.get('ph') == 'E' or e.get('cat') == 'opguard:stack')]
        # Process every stack transition at this timestamp, including transitions
        # serialized AFTER the X event. These are presentation contexts, not parents.
        for i, frame in sorted(candidates, key=lambda item: (item[1].get('ts', 0), item[0])):
            key = frame.get('tid')
            if frame['ph'] == 'B':
                stacks[key].append((i, frame))
            elif stacks[key]:
                stacks[key].pop()
        result = []
        for tid, frames in stacks.items():
            for i, frame in frames:
                args = frame.get('args', {})
                if any(args.get(k) != event['args'].get(k) for k in ('run', 'rank')):
                    continue
                context = args.get('function_context', {})
                result.append({'context_ref': self.ref(i), 'track': tid, 'function': args.get('func'),
                               'source': args.get('original_path'), 'source_line': args.get('line'),
                               'target_line': context.get('target_line'), 'depth': args.get('depth')})
        if len(result) > 128:
            raise ValueError('Ambiguous/malformed stack exceeds 128 frames')
        return result


def trace_summary(artifact):
    trace = Trace(artifact)
    counts = collections.Counter((trace.events[i]['args'].get('run'), trace.events[i]['args'].get('rank'),
                                  trace.events[i]['args'].get('status')) for i in trace.ops)
    labels = [{'pid': e.get('pid'), 'label': e.get('args', {}).get('name')} for e in trace.events
              if e.get('ph') == 'M' and e.get('name') == 'process_name']
    review_summary = None
    if trace.alignment_review:
        review_summary = {k: trace.alignment_review.get(k) for k in
                          ('artifact_sha256', 'map_sha256', 'semantic_validation', 'numeric_verdict', 'transforms')}
        review_summary['history_count'] = len(trace.alignment_review.get('map', {}).get('reviews', []))
    return {'artifact': trace.identity(), 'event_count': len(trace.events), 'operation_count': len(trace.ops),
            'status_counts': [{'run': run, 'rank': rank, 'status': status, 'count': count}
                              for (run, rank, status), count in counts.items()],
            'process_labels': labels, 'sources': list(trace.sources), 'alignment_policy': trace.alignment_policy,
            'alignment_review': review_summary,
            'limitations': ['Export positions are presentation order, not a cross-rank GPU happens-before relation.',
                            'Statuses and alignment are exporter annotations, not independently recomputed tensor comparisons.',
                            'Fingerprint agreement does not establish tensor equality; raw tensors are not provided by this adapter.',
                            'Embedded sources are snapshots without a guaranteed per-run revision mapping.',
                            'Unaligned means correspondence is undetermined, not an inserted call or tensor mismatch.']}


def first_diff(artifact):
    trace = Trace(artifact)
    groups = collections.defaultdict(list)
    for i in trace.ops:
        args = trace.events[i]['args']
        groups[(args.get('run'), args.get('rank'))].append(i)
    boundaries = []
    for (run, rank), indices in groups.items():
        indices.sort(key=lambda i: (trace.events[i].get('ts', 0), i))
        first = next((i for i in indices if trace.events[i]['args'].get('status') == 'diff'), None)
        before = indices[:indices.index(first)] if first is not None else indices
        previous_ok = next((i for i in reversed(before) if trace.events[i]['args'].get('status') == 'ok'), None)
        unaligned = [i for i in before if trace.events[i]['args'].get('status') in ('unaligned', 'insertion')]
        boundaries.append({'run': run, 'rank': rank, 'first_observed_diff': trace.compact(first) if first is not None else None,
                           'previous_exporter_ok': trace.compact(previous_ok) if previous_ok is not None else None,
                           'unaligned_before': len(unaligned),
                           'first_unaligned': trace.compact(unaligned[0]) if unaligned else None,
                           'alignment': trace.aligned(first) if first is not None else None})
    return {'artifact': trace.identity(), 'boundaries': boundaries,
            'scope': 'per_run_rank_export_order', 'root_cause': 'not_determined',
            'warning': 'Exporter pairs are candidates or agent assertions, not certified semantic equivalence. No observed diff is not proof of equivalence; previous OK is not necessarily a producer.'}


def inspect_event(artifact, event_ref):
    trace = Trace(artifact)
    index = trace.index(event_ref)
    event = trace.events[index]
    alignment = trace.aligned(index)
    peers = []
    for item in alignment['peers']:
        peer = trace.events[trace.index(item['event_ref'])]
        peers.append({'event': item, 'io': peer['args'].get('io'), 'diff': peer['args'].get('diff'),
                      'regional_comparison': regional_comparisons(event['args'], peer['args'])})
    return {'artifact': trace.identity(), 'event': trace.compact(index),
            'io': event['args'].get('io'), 'diff': event['args'].get('diff'),
            'alignment': alignment, 'peer_evidence': peers, 'source_context': trace.stack(index),
            'alignment_reviews': event['args'].get('alignment_reviews', []),
            'original_comparison': event['args'].get('original_comparison'),
            'parent_event_ref': event['args'].get('parent_event_ref'),
            'evidence_level': 'recorded_metadata_and_fingerprints',
            'input_equality': 'not_established', 'producer_identity': 'not_established',
            'note': 'xor_signature values are fingerprints, not tensor values or numeric error magnitudes.'}


def trace_neighbors(artifact, event_ref, radius):
    if type(radius) is not int or not 0 <= radius <= 20:
        raise ValueError('radius must be an integer from 0 to 20')
    trace = Trace(artifact)
    index = trace.index(event_ref)
    sequence = trace.sequence(index)
    at = sequence.index(index)
    return {'artifact': trace.identity(), 'events': [trace.compact(i) for i in sequence[max(0, at-radius):at+radius+1]],
            'relation': 'same_run_rank_export_neighbors', 'is_data_dependency': False}


def source_excerpt(artifact, source, start_line, end_line):
    if type(start_line) is not int or type(end_line) is not int or not 1 <= start_line <= end_line or end_line-start_line >= 100:
        raise ValueError('Request 1 to 100 source lines with positive one-based line numbers')
    trace = Trace(artifact)
    item = trace.sources.get(source)
    if not isinstance(item, dict) or not isinstance(item.get('content'), str):
        raise ValueError('Source key is absent from embedded sourceFiles; no local filesystem fallback')
    lines = item['content'].splitlines()
    if start_line > len(lines):
        raise ValueError('Source line is beyond end of snapshot')
    return {'artifact': trace.identity(), 'source': source, 'source_sha256': hashlib.sha256(item['content'].encode()).hexdigest(),
            'lines': [{'line': i+1, 'text': lines[i]} for i in range(start_line-1, min(end_line, len(lines)))],
            'revision_scope': 'embedded_snapshot_run_mapping_unknown',
            'note': 'Instrumentation can shift source lines; use source_context target_line, not a guessed fixed offset.'}


def search_source(artifact, query):
    if not isinstance(query, str) or not query or len(query) > 200:
        raise ValueError('query must be a nonempty literal of at most 200 characters')
    trace = Trace(artifact)
    hits = []
    for path, item in trace.sources.items():
        for index, line in enumerate(item.get('content', '').splitlines()):
            if query in line:
                hits.append({'source': path, 'line': index+1, 'text': line[:500]})
                if len(hits) > 50:
                    return {'artifact': trace.identity(), 'hits': hits[:50], 'truncated': True}
    return {'artifact': trace.identity(), 'hits': hits, 'truncated': False}


def compare_events(artifact, left_ref, right_ref):
    trace = Trace(artifact)
    left, right = trace.index(left_ref), trace.index(right_ref)
    a, b = trace.events[left]['args'], trace.events[right]['args']
    return {'artifact': trace.identity(), 'left': trace.compact(left), 'right': trace.compact(right),
            'selection_basis': 'caller_selected_candidates_not_verified_alignment',
            'metadata_agreement': {key: a.get(key) == b.get(key) for key in ('fn', 'line', 'sig', 'rank')},
            'left_io': a.get('io'), 'right_io': b.get('io'),
            'regional_comparison': regional_comparisons(a, b),
            'input_equality': 'not_established', 'numeric_error': 'not_available',
            'note': 'Equal fingerprints may collide; differing fingerprints are not numerical error magnitudes.'}


def code_context(artifact, source, line):
    """Find the enclosing function in the embedded source, not today's checkout."""
    if type(line) is not int or line < 1:
        raise ValueError('line must be a positive integer')
    trace = Trace(artifact)
    item = trace.sources.get(source)
    if not isinstance(item, dict) or not isinstance(item.get('content'), str):
        raise ValueError('Source key is absent from embedded sourceFiles')
    content = item['content']
    lines = content.splitlines()
    if line > len(lines):
        raise ValueError('Line is outside the source snapshot')
    tree = ast.parse(content)
    containers = []
    def visit(node, parents):
        named = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        names = parents + [node.name] if named else parents
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno <= line <= node.end_lineno:
            containers.append((node, '.'.join(names)))
        for child in ast.iter_child_nodes(node):
            visit(child, names)
    visit(tree, [])
    if containers:
        node, name = min(containers, key=lambda pair: pair[0].end_lineno-pair[0].lineno)
        start, end = node.lineno, node.end_lineno
    else:
        name, start, end = None, max(1, line-20), min(len(lines), line+20)
    if end-start < 80:
        selected = list(range(start, end+1))
    else:
        selected = sorted(set(range(start, min(start+10, end+1))) |
                          set(range(max(start, line-30), min(end, line+30)+1)))
    return {'artifact': trace.identity(), 'source': source,
            'source_sha256': hashlib.sha256(content.encode()).hexdigest(),
            'function': name, 'function_start': start, 'function_end': end, 'target_line': line,
            'lines': [{'line': i, 'text': lines[i-1]} for i in selected],
            'omitted_lines': end-start+1-len(selected),
            'basis': 'embedded_source_context', 'runtime_branch_execution': 'not_established'}


READERS = {fn.__name__: fn for fn in (trace_summary, first_diff, inspect_event, trace_neighbors, source_excerpt, search_source, compare_events, code_context)}
