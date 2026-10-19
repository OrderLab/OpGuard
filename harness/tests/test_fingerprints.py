import copy
import unittest

from opguard_harness.fingerprints import compare_regions, regional_comparisons


def tensor(values):
    return {'dtype': 'torch.float32', 'xor_sig_mode': 'bytes', 'xor_signature': values,
            'xor_sig_layout': {'scheme': 'flat-u32-le-v1', 'regions': 4, 'logical_nbytes': 64,
                               'byte_ranges': [[0, 16], [16, 32], [32, 48], [48, 64]]}}


class FingerprintTests(unittest.TestCase):
    def test_tail_region_is_observation_not_padding_verdict(self):
        result = compare_regions(tensor([0, 0, 0, 0]), tensor([0, 0, 0, 8]))
        self.assertEqual(result['differing_regions'], [3])
        self.assertEqual(result['regions'][3]['byte_range'], [48, 64])
        self.assertEqual(result['padding_assessment'], 'not_determined')
        self.assertEqual(result['different_element_count'], 'not_available')
        self.assertEqual(result['tensor_equality'], 'not_established')

    def test_agreement_does_not_certify_equality(self):
        result = compare_regions(tensor([0]*4), tensor([0]*4))
        self.assertEqual(result['differing_regions'], [])
        self.assertEqual(result['tensor_equality'], 'not_established')

    def test_legacy_placeholder_or_malformed_layout_is_unknown(self):
        for value in ({'xor_signature': 0}, tensor([0]*3), tensor([0, 0, 0, -1])):
            self.assertEqual(compare_regions(value, tensor([0]*4))['status'], 'unavailable')
        a = tensor([0]*4)
        for key, value in [('scheme', 'rows'), ('byte_ranges', [[0, 17]]*4)]:
            b = copy.deepcopy(a)
            b['xor_sig_layout'][key] = value
            self.assertEqual(compare_regions(a, b)['status'], 'unavailable')
        a['xor_sig_mode'] = 'placeholder-meta'
        self.assertEqual(compare_regions(a, tensor([0]*4))['status'], 'unavailable')

    def test_different_dtype_is_not_compatible_content_comparison(self):
        a, b = tensor([0]*4), tensor([0]*4)
        b['dtype'] = 'torch.int32'
        self.assertEqual(compare_regions(a, b)['status'], 'incompatible')

    def test_explicit_cross_io_slots_override_same_slot_proposals(self):
        left = {'io': {'outputs': [tensor([0]*4)]}, 'correspondence': {
            'basis': 'explicit_alignment_map', 'peer_record_id': 'b',
            'tensor_slot': ['outputs', 0], 'peer_tensor_slot': ['inputs', 1]}}
        right = {'record_id': 'b', 'io': {'inputs': [tensor([0]*4), tensor([0, 0, 0, 1])]}}
        result = regional_comparisons(left, right)
        self.assertEqual(result['selection_basis'], 'reviewed_tensor_slots')
        self.assertEqual(result['tensors'][0]['right_tensor'], ['inputs', 1])
        self.assertEqual(result['tensors'][0]['differing_regions'], [3])
