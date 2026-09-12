import unittest
from collections import Counter

from gate3_fanout_smoke import bounded_cache_salt, execution_order, pending_workloads


class OrderTests(unittest.TestCase):
    def test_resume_key_distinguishes_pressure_and_round(self):
        from gate3_pressure_scan import measurement_key
        row = dict(repeat=1, arm='pack', background_concurrency=4, context_sha256='a')
        self.assertNotEqual(measurement_key(row), measurement_key(dict(row, repeat=2)))
        self.assertNotEqual(measurement_key(row), measurement_key(dict(row, background_concurrency=8)))

    def test_pressure_workloads_are_length_stratified(self):
        from gate3_pressure_scan import select_workloads
        groups = [dict(shared_tokens=i, context_sha256=str(i)) for i in [5, 1, 9, 3, 7]]
        self.assertEqual([g['shared_tokens'] for g in select_workloads(groups)], [1, 5, 9])

    def test_salt_length_all_strategies(self):
        for name in execution_order(1, False):
            identity = 'long-run-' * 100 + name + '-' + 'a' * 64
            for slot in range(8):
                salt = bounded_cache_salt(identity + f'-background-{slot}')
                self.assertEqual(len(salt.encode('utf-8')), 67)

    def test_salt_preserves_sharing_and_isolation(self):
        self.assertEqual(bounded_cache_salt('parent'), bounded_cache_salt('parent'))
        identities = ['parent', 'parent-background-0', 'parent-background-1', 'other-run']
        self.assertEqual(len({bounded_cache_salt(s) for s in identities}), len(identities))

    def test_all_permutations(self):
        orders = [execution_order(i, True) for i in range(1, 7)]
        self.assertEqual(len({tuple(o) for o in orders}), 6)
        for position in range(3):
            self.assertEqual(set(Counter(o[position] for o in orders).values()), {2})

    def test_fixed_and_cycle(self):
        self.assertEqual(execution_order(1, True), execution_order(7, True))
        self.assertEqual(execution_order(1, False), execution_order(6, False))

    def test_resume_skips_only_matching_measured_group(self):
        groups = [{'context_sha256': 'a'}, {'context_sha256': 'b'}]
        rows = [dict(context_sha256='a', measured=True, repeat=1, arm='pack'),
                dict(context_sha256='b', measured=False, repeat=1, arm='pack')]
        self.assertEqual(pending_workloads(rows, groups, 1, 'pack'), groups[1:])
        self.assertEqual(pending_workloads(rows, groups, 2, 'pack'), groups)
        self.assertEqual(pending_workloads(rows, groups, 1, 'spread_retrieve'), groups)


if __name__ == '__main__':
    unittest.main()
