import threading
import unittest
from types import SimpleNamespace

from gate2_native_launcher import make_launcher


class NativeLauncherTests(unittest.TestCase):
    def tuner(self):
        config = SimpleNamespace(kwargs={'XBLOCK': 1, 'R0_BLOCK': 4096}, num_warps=16, num_stages=1)
        return SimpleNamespace(fn=SimpleNamespace(fn=object()), lock=threading.Lock(),
                               launchers=[SimpleNamespace(config=config)],
                               _precompile_config=lambda c: SimpleNamespace(make_launcher=lambda: c))

    def test_only_reduction_block_changes(self):
        tuner = self.tuner()
        result = make_launcher(tuner, 2048)
        self.assertEqual(result.kwargs, {'XBLOCK': 1, 'R0_BLOCK': 2048})
        self.assertEqual((result.num_warps, result.num_stages), (16, 1))
        self.assertEqual(tuner.launchers[0].config.kwargs['R0_BLOCK'], 4096)

    def test_same_config_is_independent(self):
        tuner = self.tuner()
        result = make_launcher(tuner, 4096)
        self.assertIsNot(result, tuner.launchers[0].config)
        self.assertEqual(result.kwargs, tuner.launchers[0].config.kwargs)

    def test_reloads_stripped_function(self):
        tuner = self.tuner()
        tuner.fn.fn = None
        restored = SimpleNamespace(fn=object())
        tuner._reload_kernel = lambda: SimpleNamespace(fn=restored)
        make_launcher(tuner, 2048)
        self.assertIs(tuner.fn, restored)


if __name__ == '__main__':
    unittest.main()
