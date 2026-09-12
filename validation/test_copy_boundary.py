import unittest
from lmcache_copy_diagnostic import segments


class BoundaryTests(unittest.TestCase):
    def test_within(self):
        self.assertEqual(list(segments(10, 20, 64)), [(0, 20)])

    def test_crossing(self):
        self.assertEqual(list(segments(60, 12, 64)), [(0, 4), (4, 8)])

    def test_exact_boundary(self):
        self.assertEqual(list(segments(32, 32, 64)), [(0, 32)])
        self.assertEqual(list(segments(64, 64, 64)), [(0, 64)])

    def test_many_boundaries(self):
        self.assertEqual(list(segments(60, 140, 64)), [(0, 4), (4, 64), (68, 64), (132, 8)])

    def test_invalid(self):
        with self.assertRaises(ValueError):
            list(segments(0, 1, 63))
        self.assertEqual(list(segments(0, 0, 64)), [])


if __name__ == '__main__':
    unittest.main()
