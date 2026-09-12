"""Verify the native launcher with the saved real input before live intervention."""

import importlib.util
import json
from pathlib import Path

import torch
import triton

from gate2_kernel_replay import difference
from gate2_native_launcher import make_launcher


def main():
    root = Path('artifacts/gate2_kernel_capture_20260909/kernel')
    data = torch.load(root / 'first_rmsnorm.pt', map_location='cpu', weights_only=False)
    spec = importlib.util.spec_from_file_location('gate2_native_source', data['filename'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tuner = module.triton_red_fused__to_copy_add_fused_add_rms_norm_5
    tuner.configs = [triton.Config(data['config'], num_warps=data['num_warps'], num_stages=data['num_stages'])]
    tuner.precompile()
    results = {}
    for block in [4096, 2048]:
        tuner.launchers = [make_launcher(tuner, block)]
        args = [v.cuda().clone() if isinstance(v, torch.Tensor) else v for v in data['before']]
        tuner.run(*args, stream=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        results[block] = [args[i].cpu() for i in [4, 5]]
    validation = [difference(results[4096][i], data['after'][4 + i]) for i in range(2)]
    assert all(v['equal'] for v in validation), 'Native launcher failed same-config validation'
    result = {'status': 'complete', 'same_config': validation,
              'different_config': [difference(a, b) for a, b in zip(results[4096], results[2048])]}
    (root / 'native_replay_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
