"""Provider-neutral function schemas and the matching dispatch boundary."""
from .engine import Harness


def tool(name, description, properties, required):
    return {'name': name, 'description': description,
            'parameters': {'type': 'object', 'properties': properties,
                           'required': required, 'additionalProperties': False}}


ID = {'type': 'string', 'description': 'Experiment ID returned by create_experiment'}
ARGV = {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1}
WORKLOAD = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'cwd': {'type': 'string', 'description': 'Dedicated source checkout, visible to all workers'},
        'script': {'type': 'string', 'description': 'Python script within cwd'},
        'launcher': ARGV,
        'args': {'type': 'array', 'items': {'type': 'string'}},
        'env': {'type': 'object', 'additionalProperties': {'type': 'string'}},
        'ranks': {'type': 'array', 'items': {'type': 'integer', 'minimum': 0}, 'minItems': 1, 'uniqueItems': True},
    },
    'required': ['cwd', 'script', 'launcher'],
}
SPEC = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'baseline': WORKLOAD, 'candidate': WORKLOAD,
        'backend': {**ARGV, 'description': 'Backend argv prefix, e.g. /env/bin/python -m opguard.tools'},
        'local_root': {'type': 'string', 'description': 'Absolute node-local staging directory'},
        'timeout_seconds': {'type': 'number', 'exclusiveMinimum': 0, 'default': 3600},
        'max_stage_attempts': {'type': 'integer', 'minimum': 1, 'maximum': 10, 'default': 2},
        'label': {'type': 'string'}, 'input_id': {'type': 'string'},
    },
    'required': ['baseline', 'candidate', 'backend', 'local_root'],
}
TOOLS = [
    tool('create_experiment', 'Validate a spec and create a persistent experiment. Does not run workloads.', {'spec': SPEC}, ['spec']),
    tool('advance_experiment', 'Execute exactly one pending stage. May launch an expensive workload. Inspect status after return; failures do not imply a comparison verdict.', {'experiment_id': ID}, ['experiment_id']),
    tool('experiment_status', 'Inspect stage progress, attempts, errors and log paths.', {'experiment_id': ID}, ['experiment_id']),
    tool('experiment_results', 'Get comparison artifacts. not_evaluated must never be interpreted as equivalent.', {'experiment_id': ID}, ['experiment_id']),
    tool('refine_experiment', 'Create a new experiment with a revised full spec, preserving the parent evidence. No automatic diagnosis or cache reuse.', {'experiment_id': ID, 'spec': SPEC}, ['experiment_id', 'spec']),
    tool('recover_experiment', 'Reconcile an interrupted controller only when the backend has exited and source restoration is confirmed.', {'experiment_id': ID}, ['experiment_id']),
]


ARTIFACT = {'type': 'string', 'description': 'Local path to an OpGuard Perfetto JSON or JSON.gz export'}
EVENT = {'type': 'string', 'description': 'Artifact-bound event_ref returned by a trace query'}
TOOLS += [
    tool('code_context', 'Read the enclosing function around a source location, with signature and bounded body from the captured snapshot.', {'artifact': ARTIFACT, 'source': {'type': 'string'}, 'line': {'type': 'integer', 'minimum': 1}}, ['artifact', 'source', 'line']),
    tool('compare_events', 'Read selected candidate events and regional fingerprint differences with byte ranges; does not certify alignment, tensor equality or padding noise.', {'artifact': ARTIFACT, 'left_ref': EVENT, 'right_ref': EVENT}, ['artifact', 'left_ref', 'right_ref']),
    tool('trace_summary', 'Inspect recorded runs/ranks, statuses, embedded sources and evidence limitations.', {'artifact': ARTIFACT}, ['artifact']),
    tool('first_diff', 'Find exporter-annotated first differences per run/rank; no root-cause or equality inference.', {'artifact': ARTIFACT}, ['artifact']),
    tool('inspect_event', 'Read IO fingerprints, regional byte ranges/differences, exporter-aligned peer and source context. Equal XOR is not proof of equal tensors.', {'artifact': ARTIFACT, 'event_ref': EVENT}, ['artifact', 'event_ref']),
    tool('trace_neighbors', 'Read bounded same-run/rank presentation neighbors, NOT data dependencies.', {'artifact': ARTIFACT, 'event_ref': EVENT, 'radius': {'type': 'integer', 'minimum': 0, 'maximum': 20}}, ['artifact', 'event_ref', 'radius']),
    tool('source_excerpt', 'Read up to 100 lines from an embedded source snapshot, without accessing the original filesystem path.', {'artifact': ARTIFACT, 'source': {'type': 'string'}, 'start_line': {'type': 'integer', 'minimum': 1}, 'end_line': {'type': 'integer', 'minimum': 1}}, ['artifact', 'source', 'start_line', 'end_line']),
    tool('search_source', 'Literal search across embedded source snapshots, capped at 50 hits.', {'artifact': ARTIFACT, 'query': {'type': 'string'}}, ['artifact', 'query']),
]

TENSOR_SLOT = {'type': ['array', 'null'], 'description': 'Exported io slot ["inputs" or "outputs", integer index]; null for reviewing an existing whole-event relation.',
               'minItems': 2, 'maxItems': 2, 'items': {'type': ['string', 'integer']}}
REVIEW = {'type': 'object', 'additionalProperties': False, 'properties': {
    'left_ref': EVENT, 'right_ref': EVENT, 'left_tensor': TENSOR_SLOT, 'right_tensor': TENSOR_SLOT,
    'decision': {'type': 'string', 'enum': ['comparable', 'not_comparable', 'undetermined']},
    'reason': {'type': 'string', 'minLength': 1},
    'evidence': {'type': 'array', 'minItems': 1, 'items': {'type': 'object', 'additionalProperties': False,
        'properties': {'basis': {'type': 'string', 'enum': ['observed', 'source_inference', 'intervention_tested', 'externally_reported']},
                       'reference': {'type': 'string', 'minLength': 1}}, 'required': ['basis', 'reference']}},
}, 'required': ['left_ref', 'right_ref', 'left_tensor', 'right_tensor', 'decision', 'reason', 'evidence']}
TOOLS += [
    tool('review_pair', 'Append an artifact-scoped comparability judgment with evidence. It is an agent assertion, not a numerical verdict. Use null slots for not_comparable/undetermined; new comparable pairs require both tensor slots.',
         {'artifact': ARTIFACT, 'review': REVIEW}, ['artifact', 'review']),
    tool('difference_candidates', 'Page exporter-labelled differences, deduplicating reciprocal peers. Skip reviewed candidates or revisit them. Empty results and undetermined reviews never establish equivalence.',
         {'artifact': ARTIFACT, 'cursor': {'type': 'integer', 'minimum': 0},
          'limit': {'type': 'integer', 'minimum': 1, 'maximum': 50}, 'include_reviewed': {'type': 'boolean'}},
         ['artifact', 'cursor', 'limit', 'include_reviewed']),
    tool('render_alignment', 'Freeze the review map and ask OpGuard to write a new Perfetto view. No workload launch, implicit transforms or tensor-equality certification. Never overwrites an existing output.',
         {'artifact': ARTIFACT, 'output': {'type': 'string', 'description': 'Fresh absolute .json or .json.gz trace path'},
          'backend': ARGV}, ['artifact', 'output', 'backend']),
]


def dispatch(root, name, arguments):
    schema = next((item for item in TOOLS if item['name'] == name), None)
    if schema is None:
        raise ValueError(f'Unknown tool: {name}')
    if not isinstance(arguments, dict) or set(arguments) != set(schema['parameters']['required']):
        raise ValueError('Tool argument names do not match its schema')
    from .trace_reader import READERS
    if name in READERS:
        return READERS[name](**arguments)
    from .reviews import REVIEW_TOOLS
    if name in REVIEW_TOOLS:
        return REVIEW_TOOLS[name](root, **arguments)
    harness = Harness(root)
    functions = {'create_experiment': harness.create, 'advance_experiment': harness.advance,
                 'experiment_status': harness.status, 'experiment_results': harness.results,
                 'refine_experiment': harness.refine, 'recover_experiment': harness.recover}
    return functions[name](**arguments)
