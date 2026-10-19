"""Interpret recorded regional fingerprints, without inferring value equality."""


def _layout(tensor):
    layout = tensor.get('xor_sig_layout')
    values = tensor.get('xor_signature')
    if tensor.get('xor_sig_mode') != 'bytes' or not isinstance(layout, dict):
        return None
    count, size = layout.get('regions'), layout.get('logical_nbytes')
    if (layout.get('scheme') != 'flat-u32-le-v1' or type(count) is not int
            or not 1 <= count <= 64 or type(size) is not int or size < 0):
        return None
    if count == 1 and type(values) in (int, float):
        values = [values]
    if not isinstance(values, list) or len(values) != count or any(
            type(x) not in (int, float) or not 0 <= x <= 0xffffffff or int(x) != x for x in values):
        return None
    base, rem = divmod((size + 3) // 4, count)
    ranges, start = [], 0
    for i in range(count):
        end = start + base + (i < rem)
        ranges.append([min(start * 4, size), min(end * 4, size)])
        start = end
    if layout.get('byte_ranges') != ranges:
        return None
    return layout, list(map(int, values))


def compare_regions(left, right):
    a, b = _layout(left), _layout(right)
    if a is None or b is None:
        return {'status': 'unavailable', 'reason': 'missing_or_unsupported_content_fingerprint_layout'}
    if a[0] != b[0] or left.get('dtype') != right.get('dtype'):
        return {'status': 'incompatible', 'reason': 'different_fingerprint_layout_or_dtype'}
    regions = [{'region': i, 'byte_range': bounds, 'left': av, 'right': bv,
                'fingerprint_xor': av ^ bv, 'fingerprints_agree': av == bv}
               for i, (bounds, av, bv) in enumerate(zip(a[0]['byte_ranges'], a[1], b[1]))]
    return {'status': 'available', 'scheme': a[0]['scheme'], 'regions': regions,
            'differing_regions': [r['region'] for r in regions if not r['fingerprints_agree']],
            'tensor_equality': 'not_established', 'different_element_count': 'not_available',
            'padding_assessment': 'not_determined'}


def regional_comparisons(left, right):
    """Same-slot proposals, or the explicit selected slots of a reviewed pair."""
    a, b = left.get('io') or {}, right.get('io') or {}
    claim = left.get('correspondence') or {}
    selected = (claim.get('basis') == 'explicit_alignment_map'
                and claim.get('peer_record_id') == right.get('record_id')
                and claim.get('tensor_slot') is not None)
    if selected:
        slots = [(claim['tensor_slot'], claim['peer_tensor_slot'])]
    else:
        slots = [([side, i], [side, i]) for side in ('inputs', 'outputs')
                 for i in range(min(len(a.get(side, [])), len(b.get(side, []))))]
    tensors = []
    for x, y in slots[:64]:
        av, bv = a[x[0]][x[1]], b[y[0]][y[1]]
        tensors.append({'left_tensor': x, 'right_tensor': y, **compare_regions(av, bv)})
    return {'selection_basis': 'reviewed_tensor_slots' if selected else 'same_io_slot_candidates',
            'tensors': tensors, 'truncated': len(slots) > 64,
            'note': 'Region differences localize recorded byte changes, not element counts or padding noise. '
                    'Matching regions may still contain XOR collisions. Validate semantic correspondence first.'}
