"""Capture the first real prefill call of the suspect compiled RMSNorm."""

import os
from pathlib import Path

import torch
from torch._inductor.runtime.triton_heuristics import CachingAutotuner


class ProbeWorkerExtension:
    pass


def install():
    out = Path(os.environ['KERNEL_PROBE_OUT'])
    out.mkdir(parents=True, exist_ok=True)
    original = CachingAutotuner.run
    captured = False

    def run(self, *args, **kwargs):
        nonlocal captured
        enabled = (not captured and (out / 'ACTIVE').exists()
                   and 'cx4ejhtmhkieyjtpdsr2oqadmdibuec6knnx37uh242ksttakpl7' in str(self.filename)
                   and not torch.cuda.is_current_stream_capturing())
        if not enabled:
            return original(self, *args, **kwargs)
        torch.cuda.synchronize()
        inputs = [v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v for v in args]
        result = original(self, *args, **kwargs)
        torch.cuda.synchronize()
        outputs = [v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v for v in args]
        config = self.launchers[0].config
        torch.save({'before': inputs, 'after': outputs, 'filename': self.filename,
                    'config': dict(config.kwargs), 'num_warps': config.num_warps,
                    'num_stages': config.num_stages,
                    'arg_names': self.fn.arg_names}, out / 'first_rmsnorm.pt')
        captured = True
        return result

    CachingAutotuner.run = run


if os.environ.get('KERNEL_PROBE_OUT'):
    install()
