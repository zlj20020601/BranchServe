import unittest

import torch

from gate2_compare_states import compare


class CompareTests(unittest.TestCase):
    def test_equal(self):
        self.assertTrue(compare(torch.ones(2), torch.ones(2))['byte_equal'])

    def test_difference(self):
        result = compare(torch.tensor([1., 2.]), torch.tensor([1., 3.]))
        self.assertEqual(result['different_elements'], 1)
        self.assertEqual(result['max_abs'], 1.)

    def test_shape_mismatch(self):
        self.assertFalse(compare(torch.ones(2), torch.ones(3))['compatible'])

    def test_signed_zero(self):
        result = compare(torch.tensor([0.]), torch.tensor([-0.]))
        self.assertEqual(result['different_elements'], 0)
        self.assertFalse(result['byte_equal'])

    def test_nonfinite(self):
        result = compare(torch.tensor([float('inf')]), torch.tensor([1.]))
        self.assertEqual(result['nonfinite'], 1)
        self.assertIsNone(result['max_abs'])


if __name__ == '__main__':
    unittest.main()
