"""Process-local intervention on one compiled kernel, leaving cache files intact."""

import json
import os
from pathlib import Path

import torch
from torch._inductor.runtime.triton_heuristics import CachingAutotuner
from gate2_native_launcher import make_launcher


class ProbeWorkerExtension:
    pass


def install():
    out = Path(os.environ['KERNEL_PROBE_OUT'])
    out.mkdir(parents=True, exist_ok=True)
    original = CachingAutotuner.run
    counts = {}
    kernels = {}

    def run(self, *args, **kwargs):
        marker = out / 'MODE.json'
        if ('cx4ejhtmhkieyjtpdsr2oqadmdibuec6knnx37uh242ksttakpl7' not in str(self.filename)
                or not marker.exists() or torch.cuda.is_current_stream_capturing()):
            return original(self, *args, **kwargs)
        setting = json.loads(marker.read_text())
        request = setting['request']
        count = counts.get(request, 0)
        counts[request] = count + 1
        mode = setting['mode']
        change = (mode in ('all2048', 'all4096') or (mode == 'first2048' and count == 0)
                  or (mode == 'firstboundary2048' and count == 2)
                  or (mode == 'boundary2048' and count % 3 == 2))
        if change:
            assert len(args) == 8 and args[6] in (528, 64) and args[7] == 2560
            config = self.launchers[0].config
            assert config.kwargs == {'XBLOCK': 1, 'R0_BLOCK': 4096}
            block = 4096 if mode == 'all4096' else 2048
            key = (id(self), block)
            if key not in kernels:
                kernels[key] = make_launcher(self, block)
            previous = self.launchers
            self.launchers = [kernels[key]]
            try:
                result = original(self, *args, **kwargs)
            finally:
                self.launchers = previous
        else:
            result = original(self, *args, **kwargs)
        with (out / 'interventions.jsonl').open('a') as handle:
            handle.write(json.dumps({'request': request, 'mode': mode, 'call': count,
                                     'tokens': args[6], 'changed': change}) + '\n')
        return result

    CachingAutotuner.run = run


if os.environ.get('KERNEL_PROBE_OUT'):
    install()
