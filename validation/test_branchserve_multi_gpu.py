"""Pure-logic regression tests for the multi-worker deployment service."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deployment import branchserve_service as service


class MultiWorkerRoutingTests(unittest.TestCase):
    def setUp(self):
        service.configure_workers([
            "http://127.0.0.1:8000",
            "http://127.0.0.1:8001",
            "http://127.0.0.1:8002",
        ])
        service.STATE["threshold"] = 5.0
        service.STORE.update(healthy=True, misses=0, last_transfer=None)
        service.TELEMETRY["pressure"] = [8.0, 4.0, 1.0]
        service.TELEMETRY["ts"] = service.time.time()

    def test_retrieve_chooses_lowest_pressure_remote_worker(self):
        self.assertEqual(service.decide(0, "child")[0], "retrieve")
        self.assertEqual(service.apply_action("retrieve", 0), (2, None))

    def test_target_is_not_hard_coded_to_adjacent_gpu(self):
        self.assertEqual(service.apply_action("retrieve", 1), (2, None))
        self.assertEqual(service.apply_action("recompute", 2), (1, "bs-recompute"))

    def test_equal_pressure_candidates_are_rotated(self):
        service.TELEMETRY["pressure"] = [8.0, 1.0, 1.0]
        self.assertEqual(service.apply_action("retrieve", 0)[0], 1)
        self.assertEqual(service.apply_action("retrieve", 0)[0], 2)

    def test_single_worker_degrades_to_pack(self):
        service.configure_workers(["http://127.0.0.1:8000"])
        service.TELEMETRY["ts"] = service.time.time()
        self.assertEqual(service.decide(0, "child")[0], "pack")
        self.assertEqual(service.apply_action("retrieve", 0), (0, None))


if __name__ == "__main__":
    unittest.main()
