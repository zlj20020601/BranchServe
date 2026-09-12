import unittest
from prepare_qmsum import common_length


class PrefixTests(unittest.TestCase):
    def test_common_prefix(self):
        self.assertEqual(common_length([[1, 2, 3], [1, 2, 4], [1, 2]]), 2)
        self.assertEqual(common_length([[1], [2]]), 0)
        self.assertEqual(common_length([[1, 2], [1, 2]]), 2)


if __name__ == '__main__':
    unittest.main()
