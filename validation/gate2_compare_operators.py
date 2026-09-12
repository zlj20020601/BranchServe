"""Find the first differing input/output in paired first-prefill traces."""

import argparse
import json
from pathlib import Path

import torch

from gate2_kernel_replay import difference


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def compare(root):
    left, right = [read(root / mode / 'trace.jsonl') for mode in ['apc', 'retrieve']]
    assert len(left) == len(right)
    result = {'trace_points': len(left), 'differences': [], 'config_differences': []}
    for a, b in zip(left, right):
        assert (a['index'], a['kind'], a['name']) == (b['index'], b['kind'], b['name'])
        details = {'index': a['index'], 'kind': a['kind'], 'name': a['name']}
        fields = [field for field in ['before', 'after'] if a[field] != b[field]]
        ma, mb = a['metadata'], b['metadata']
        config_fields = ['config', 'num_warps', 'num_stages']
        if any(ma.get(k) != mb.get(k) for k in config_fields):
            result['config_differences'].append(dict(details, left={k: ma.get(k) for k in config_fields},
                                                      right={k: mb.get(k) for k in config_fields}))
        assert ma.get('weight') == mb.get('weight'), 'Matrix weights differ'
        if fields:
            result['differences'].append(dict(details, fields=fields))
    if result['differences']:
        first = result['differences'][0]
        index = first['index']
        a, b = [torch.load(root / mode / f'op_{index:03d}.pt', map_location='cpu', weights_only=False)
                for mode in ['apc', 'retrieve']]
        first['tensor_differences'] = {field: {k: difference(a[field][k], b[field][k])
                                              for k in a[field]} for field in ['before', 'after']}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    result = compare(args.root)
    (args.root / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
