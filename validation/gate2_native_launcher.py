"""Construct an alternative launcher using Inductor's own compiler entry point."""

import copy


def make_launcher(autotuner, block):
    if autotuner.fn.fn is None:
        autotuner.fn = autotuner._reload_kernel().fn
    config = copy.deepcopy(autotuner.launchers[0].config)
    config.kwargs['R0_BLOCK'] = block
    with autotuner.lock:
        return autotuner._precompile_config(config).make_launcher()
