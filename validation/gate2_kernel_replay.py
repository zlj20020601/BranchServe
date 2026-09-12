"""Replay captured RMSNorm inputs with two explicit reduction configurations."""

import argparse
import importlib.util
import json
from pathlib import Path

import torch


def difference(left, right):
    diff = (left.double() - right.double()).abs()
    return {'equal': torch.equal(left, right), 'different_elements': int((left != right).sum()),
            'elements': left.numel(), 'max_abs': float(diff.max()), 'mean_abs': float(diff.mean())}


def replay(path):
    data = torch.load(path, map_location='cpu', weights_only=False)
    spec = importlib.util.spec_from_file_location('gate2_saved_rmsnorm', data['filename'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = module.triton_red_fused__to_copy_add_fused_add_rms_norm_5.fn
    before = data['before']
    assert len(before) == 8 and before[6] == 528 and before[7] == 2560
    assert data['config']['XBLOCK'] == 1
    results = {}
    for block in [4096, 2048]:
        runs = []
        for _ in range(2):
            args = [v.cuda().clone() if isinstance(v, torch.Tensor) else v for v in before]
            kernel[(528,)](*args, XBLOCK=1, R0_BLOCK=block,
                           num_warps=data['num_warps'], num_stages=data['num_stages'],
                           enable_fp_fusion=True)
            torch.cuda.synchronize()
            runs.append([args[4].cpu(), args[5].cpu()])
        assert all(torch.equal(a, b) for a, b in zip(*runs)), 'Replay not repeatable'
        results[block] = runs[0]
    original_block = data['config']['R0_BLOCK']
    validation = [difference(results[original_block][i], data['after'][4 + i]) for i in range(2)]
    result = {'status': 'complete', 'capture': str(path), 'captured_config': data['config'],
              'num_warps': data['num_warps'], 'num_stages': data['num_stages'],
              'shape': [528, 2560], 'repeats_per_configuration': 2,
              'original_replay_vs_capture': validation,
              'block4096_vs_2048': {name: difference(results[4096][i], results[2048][i])
                                     for i, name in enumerate(['residual', 'normalized'])}}
    assert all(v['equal'] for v in validation), 'Original configuration did not reproduce captured output'
    x0, x1, x2, weight = before[:4]
    residual = x0.float() + (x1.float() + x2.float())
    reference = (residual.double() * torch.rsqrt(residual.double().square().mean(-1, keepdim=True) + 1e-6)
                 * (weight.float() + 1).double()).to(torch.bfloat16)
    result['vs_fp64_reduction_reference'] = {str(block): difference(values[1], reference)
                                           for block, values in results.items()}
    torch.save({'results': results, 'reference': reference}, path.parent / 'replay_outputs.pt')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('capture', type=Path)
    args = parser.parse_args()
    result = replay(args.capture)
    (args.capture.parent / 'replay_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
