import unittest

from gate2_compare_apc import first_difference


class DivergenceTests(unittest.TestCase):
    def test_equal(self):
        self.assertIsNone(first_difference([1, 2], [1, 2]))

    def test_first_token(self):
        self.assertEqual(first_difference([1, 2], [3, 2]), 0)

    def test_second_token(self):
        self.assertEqual(first_difference([1, 2], [1, 3]), 1)


if __name__ == '__main__':
    unittest.main()
