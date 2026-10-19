"""Artifact-scoped agent reviews and a resumable difference queue.

Reviews are assertions with provenance, never automatic root-cause verdicts.
The append-only review history is also an OpGuard alignment-map v1 document.
"""
import collections
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid

from .spec import strings
from .store import atomic_json
from .trace_reader import Trace

DECISIONS = {"comparable", "not_comparable", "undetermined"}
BASES = {"observed", "source_inference", "intervention_tested", "externally_reported"}
MAX_REVIEW_BYTES = 8 * 1024 * 1024


def _key(left, right):
    return tuple(sorted((left, right)))


def _slot(trace, index, slot):
    if (not isinstance(slot, list) or len(slot) != 2 or slot[0] not in ("inputs", "outputs")
            or type(slot[1]) is not int or slot[1] < 0):
        raise ValueError('Tensor slot must be [inputs|outputs, nonnegative index] in exported io')
    values = (trace.events[index]['args'].get('io') or {}).get(slot[0], [])
    if slot[1] >= len(values) or not isinstance(values[slot[1]], dict):
        raise ValueError('Tensor slot is absent from exported io')


def _validate(trace, review):
    required = {'left_ref', 'right_ref', 'left_tensor', 'right_tensor', 'decision', 'reason', 'evidence'}
    if (not isinstance(review, dict) or not required <= review.keys()
            or review.keys() - required - {'id', 'created_at'}):
        raise ValueError('Invalid review fields')
    a, b = trace.index(review['left_ref']), trace.index(review['right_ref'])
    left, right = trace.events[a]['args'], trace.events[b]['args']
    if not left.get('run') or not right.get('run') or left['run'] == right['run']:
        raise ValueError('A review must relate different runs')
    if review['decision'] not in DECISIONS or not isinstance(review['reason'], str) or not review['reason'].strip():
        raise ValueError('A review requires a decision and a reason')
    evidence = review['evidence']
    if not isinstance(evidence, list) or not evidence:
        raise ValueError('A review requires evidence references')
    for item in evidence:
        if (not isinstance(item, dict) or set(item) != {'basis', 'reference'}
                or item['basis'] not in BASES or not isinstance(item['reference'], str)
                or not item['reference'].strip()):
            raise ValueError('Invalid evidence basis/reference')
    lp, rp = review['left_tensor'], review['right_tensor']
    if (lp is None) != (rp is None):
        raise ValueError('Select both tensor slots or neither')
    if lp is not None:
        if review['decision'] != 'comparable':
            raise ValueError('Negative/undetermined reviews address the whole proposed relation; use null slots')
        _slot(trace, a, lp)
        _slot(trace, b, rp)
    elif review['decision'] == 'comparable':
        aligned = trace.aligned(a)
        if not aligned['unique'] or aligned['peers'][0]['event_ref'] != trace.ref(b):
            raise ValueError('New correspondences require explicit tensor slots')


def _latest(trace, history):
    latest = {}
    for review in history:
        _validate(trace, review)
        latest[_key(review['left_ref'], review['right_ref'])] = review
    used = set()
    for review in latest.values():
        if review['decision'] == 'comparable':
            refs = {review['left_ref'], review['right_ref']}
            if used & refs:
                raise ValueError('Conflicting comparable reviews: one relation per operation per view')
            used.update(refs)
    return latest


def _path(root, trace):
    return Path(root).expanduser().resolve() / 'reviews' / (trace.sha + '.json')


def _load(root, trace):
    path = _path(root, trace)
    if not path.exists():
        return {'version': 1, 'artifact_sha256': trace.sha, 'reviews': []}
    with path.open('rb') as stream:
        raw = stream.read(MAX_REVIEW_BYTES + 1)
    if len(raw) > MAX_REVIEW_BYTES:
        raise ValueError('Review history exceeds the 8 MiB limit')
    record = json.loads(raw)
    if (record.get('version') != 1 or record.get('artifact_sha256') != trace.sha
            or not isinstance(record.get('reviews'), list)):
        raise ValueError('Review history belongs to another artifact or version')
    _latest(trace, record['reviews'])
    return record


def review_pair(root, artifact, review):
    trace = Trace(artifact)
    _validate(trace, review)
    path = _path(root, trace)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = _load(root, trace)
        entry = {**review, 'id': uuid.uuid4().hex, 'created_at': datetime.now(timezone.utc).isoformat()}
        record['reviews'].append(entry)
        latest = _latest(trace, record['reviews'])
        if len(json.dumps(record, indent=2, ensure_ascii=False).encode()) > MAX_REVIEW_BYTES:
            raise ValueError('Review history exceeds the 8 MiB limit')
        atomic_json(path, record)
    return {'artifact': trace.identity(), 'review': entry, 'alignment_map': str(path),
            'active_review_count': len(latest), 'history_count': len(record['reviews']),
            'validation': 'agent_assertion_not_independently_verified', 'verdict': 'not_evaluated'}


def difference_candidates(root, artifact, cursor, limit, include_reviewed):
    if type(cursor) is not int or cursor < 0 or type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError('cursor must be nonnegative and limit between 1 and 50')
    if type(include_reviewed) is not bool:
        raise ValueError('include_reviewed must be boolean')
    trace = Trace(artifact)
    latest = _latest(trace, _load(root, trace)['reviews'])
    candidates, seen = [], set()
    for i in sorted(trace.ops, key=lambda i: (str(trace.events[i]['args'].get('run')),
                                             trace.events[i]['args'].get('rank', -1),
                                             trace.events[i].get('ts', 0), i)):
        if trace.events[i]['args']['status'] != 'diff':
            continue
        alignment = trace.aligned(i)
        peer = alignment['peers'][0] if alignment['unique'] else None
        if peer is None:
            # Show a declared, uniquely addressable peer even when its automatic
            # constraints fail, so the agent can reject that proposal. The
            # alignment consistency failure remains visible alongside it.
            claim = trace.events[i]['args'].get('correspondence') or {}
            proposed = trace.records.get(claim.get('peer_record_id'), [])
            if len(proposed) == 1 and trace.events[proposed[0]]['args'].get('run') != trace.events[i]['args'].get('run'):
                peer = trace.compact(proposed[0])
        key = _key(trace.ref(i), peer['event_ref']) if peer else (trace.ref(i),)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({'left': trace.compact(i), 'right': peer, 'alignment': alignment,
                           'diff': trace.events[i]['args'].get('diff'), 'review': latest.get(key)})
    if cursor > len(candidates):
        raise ValueError('cursor is beyond this artifact candidate queue')
    selected, position = [], cursor
    while position < len(candidates) and len(selected) < limit:
        item = candidates[position]
        position += 1
        if include_reviewed or item['review'] is None:
            selected.append(item)
    return {'artifact': trace.identity(), 'candidates': selected,
            'next_cursor': position if position < len(candidates) else None,
            'candidate_count': len(candidates),
            'unreviewed_count': sum(item['review'] is None for item in candidates),
            'review_counts': dict(collections.Counter(r['decision'] for r in latest.values())),
            'alignment_map': str(_path(root, trace)) if latest else None,
            'scope': 'grouped_by_run_rank_export_order_not_global_causality',
            'verdict': 'not_evaluated',
            'note': 'Reviews do not erase observations. Use include_reviewed to revisit them; undetermined is unresolved. New declared pairs appear after render_alignment.'}


def render_alignment(root, artifact, output, backend):
    """Freeze the current map and invoke the existing backend protocol, no workload."""
    strings(backend, 'backend', True)
    trace = Trace(artifact)
    mapping = _load(root, trace)
    if not mapping['reviews']:
        raise ValueError('No reviews for this artifact')
    target = Path(output).expanduser()
    if not target.is_absolute() or target.exists():
        raise ValueError('output must be a fresh absolute trace path')
    directory = Path(root).expanduser().resolve() / 'review-renders' / uuid.uuid4().hex
    directory.mkdir(parents=True)
    map_path = directory / 'alignment-map.json'
    atomic_json(map_path, mapping)
    request = {'protocol_version': 1, 'artifact': str(trace.path),
               'alignment_map': str(map_path), 'output': str(target)}
    atomic_json(directory / 'request.json', request)
    env = {k: v for k, v in os.environ.items() if not k.startswith('OPGUARD_')}
    env['OPGUARD_AUTOSTART'] = '0'
    with (directory / 'response.json').open('w') as out, (directory / 'backend.log').open('w') as err:
        process = subprocess.run([*backend, 'align', '--request', str(directory / 'request.json')],
                                 stdout=out, stderr=err, env=env, timeout=120)
    response = json.loads((directory / 'response.json').read_text())
    if process.returncode or response.get('protocol_version') != 1 or response.get('ok') is not True:
        raise RuntimeError(f'Alignment backend failed; inspect {directory}: {response.get("error")}')
    result = response['result']
    rendered = Trace(target)
    if (Path(result['artifact']).resolve() != target.resolve() or result.get('sha256') != rendered.sha
            or result.get('source_sha256') != trace.sha
            or result.get('map_sha256') != hashlib.sha256(map_path.read_bytes()).hexdigest()
            or result.get('verdict') != 'not_evaluated'):
        raise RuntimeError(f'Alignment backend returned inconsistent provenance; inspect {directory}')
    return {**result, 'render_directory': str(directory),
            'note': 'This is a new artifact revision. Obtain new event_refs; original trace and review history are preserved.'}


REVIEW_TOOLS = {fn.__name__: fn for fn in (review_pair, difference_candidates, render_alignment)}
