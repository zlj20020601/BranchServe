import unittest

import torch

from gate2_operator_probe import tensor_bytes


class TensorBytesTests(unittest.TestCase):
    def test_singleton_last_dimension(self):
        value = torch.arange(8448, dtype=torch.float32).reshape(1, -1).t()
        self.assertNotEqual(value.stride(-1), 1)
        self.assertEqual(tensor_bytes(value), tensor_bytes(value.flatten()))

    def test_bfloat16_and_scalar(self):
        value = torch.tensor(1., dtype=torch.bfloat16)
        self.assertEqual(tensor_bytes(value), tensor_bytes(value.reshape(1)))


if __name__ == '__main__':
    unittest.main()
