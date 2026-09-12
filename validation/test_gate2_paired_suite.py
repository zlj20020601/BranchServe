import unittest

from gate2_paired_suite import audit, prompt, summarize


def counters(compute=0, external=0, local=0, requests=0):
    out = {(name, (('source', source),)): value for name, source, value in [
        ('vllm:prompt_tokens_by_source_total', 'local_compute', compute),
        ('vllm:prompt_tokens_by_source_total', 'external_kv_transfer', external),
        ('vllm:prompt_tokens_by_source_total', 'local_cache_hit', local)]}
    for reason, value in [('length', requests), ('error', 0), ('abort', 0)]:
        out[('vllm:request_success_total', (('finished_reason', reason),))] = value
    return out


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.result = {'usage': {'prompt_tokens': 8512}, 'output_sha256': 'same'}

    def test_recompute_and_retrieve(self):
        for mode, end in [('recompute', counters(compute=8512, requests=1)),
                          ('retrieve', counters(compute=64, external=8448, requests=1))]:
            self.assertTrue(audit(counters(), end, mode, self.result, 'same', 4, 4)['valid'])

    def test_local_hit_is_not_external_retrieval(self):
        end = counters(compute=64, local=8448, requests=1)
        self.assertFalse(audit(counters(), end, 'retrieve', self.result, 'same', 4, 4)['valid'])

    def test_concurrent_requests_fail(self):
        end = counters(compute=8512, requests=2)
        self.assertFalse(audit(counters(), end, 'recompute', self.result, 'same', 4, 4)['valid'])
        end = counters(compute=8512, requests=1)
        self.assertFalse(audit(counters(), end, 'recompute', self.result, 'same', 4, 5)['valid'])

    def test_mismatch_and_missing_evidence_fail(self):
        end = counters(compute=8512, requests=1)
        self.assertFalse(audit(counters(), end, 'recompute', self.result, 'different', 4, 4)['valid'])
        self.assertFalse(audit({}, {}, 'recompute', self.result, 'same', 4, 4)['valid'])

    def test_inputs_reproducible_and_independent(self):
        a, b = prompt(10, 248320), prompt(11, 248320)
        self.assertEqual(a, prompt(10, 248320))
        self.assertNotEqual(a[:8448], b[:8448])
        self.assertEqual(len(a), 8512)

    def test_incomplete_pairs_not_reported_complete(self):
        self.assertEqual(summarize([])['n_valid'], 0)
        self.assertEqual(summarize([{'pair': 1, 'mode': 'recompute'}])['n_pairs'], 0)


if __name__ == '__main__':
    unittest.main()
