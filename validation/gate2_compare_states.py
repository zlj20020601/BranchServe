"""Compare diagnostic snapshots without conflating byte and numeric equality."""

import argparse
import json
from pathlib import Path

import torch


def compare(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return {'compatible': False, 'shapes': [list(a.shape), list(b.shape)], 'dtypes': [str(a.dtype), str(b.dtype)]}
    byte_equal = torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
    af, bf = a.double(), b.double()
    finite = torch.isfinite(af) & torch.isfinite(bf)
    diff = (af[finite] - bf[finite]).abs()
    return {'compatible': True, 'byte_equal': byte_equal, 'dtype': str(a.dtype), 'shape': list(a.shape),
            'different_elements': int((a != b).sum()), 'nonfinite': int((~finite).sum()),
            'max_abs': float(diff.max()) if diff.numel() else None,
            'mean_abs': float(diff.mean()) if diff.numel() else None}


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def analyze(root):
    p, c = root / 'gpu0', root / 'gpu1'
    parent = load(p / 'store_8448.pt')
    child = load(c / 'retrieve_8448.pt')
    report = {'transport': {}, 'core64': {}, 'attention64': {}, 'boundary_to_consumer': {}, 'logits': []}
    stores = [load(path) for path in sorted(p.glob('store_*.pt'), key=lambda x: int(x.stem.split('_')[1]))]
    for name, value in child['layers'].items():
        if 'recurrent' in value:
            report['transport'][name] = {key: compare(parent['layers'][name][key], value[key])
                                         for key in ['conv', 'recurrent']}
        else:
            chunks = [s['layers'][name]['page_bytes'] for s in stores]
            merged = torch.cat(chunks, dim=0)
            report['transport'][name] = {'page_bytes': compare(merged, value['page_bytes'])}
    for path in sorted(p.glob('core64_*.pt')):
        other = c / path.name
        if not other.exists():
            report['core64'][path.stem] = {'missing_child': True}
            continue
        a, b = load(path), load(other)
        report['core64'][path.stem] = {key: compare(a[key], b[key]) for key in
            ['conv_before', 'recurrent_before', 'mixed_qkv', 'b', 'a', 'core_out', 'conv_after', 'recurrent_after']}
        name = path.stem[len('core64_'):].rsplit('_', 1)[0]
        for side, core, boundary in [('parent', a, parent), ('child', b, child)]:
            if name in boundary['layers']:
                report['boundary_to_consumer'][side + ':' + name] = {
                    key: compare(boundary['layers'][name][key], core[key + '_before'])
                    for key in ['conv', 'recurrent']}
    # Legacy captures included one discarded logit row per 528-token prefill chunk.
    parent_offset = 0 if (p / 'logits_protocol.json').exists() else 16
    report['parent_logits_offset'] = parent_offset
    for path in sorted(p.glob('logits_*.pt'), key=lambda x: int(x.stem.split('_')[1])):
        position = int(path.stem.split('_')[1]) - parent_offset
        if position < 0:
            continue
        other = c / f'logits_{position}.pt'
        if not other.exists():
            continue
        a, b = load(path), load(other)
        row = {'step': position, 'logits': compare(a['logits'], b['logits']),
               'hidden': compare(a['hidden'], b['hidden'])}
        for side, data in [('parent', a), ('child', b)]:
            values, ids = data['logits'].float().reshape(-1, data['logits'].shape[-1])[-1].topk(5)
            row[side] = {'ids': ids.tolist(), 'scores': values.tolist()}
        report['logits'].append(row)
    for path in sorted(p.glob('attention64_*.pt')):
        other = c / path.name
        if not other.exists():
            continue
        a, b = load(path), load(other)
        row = {key: compare(a[key], b[key]) for key in ['query', 'key', 'value', 'output']}
        ac = a['cache'].transpose(1, 2).flatten(0, 1)
        bc = b['cache'].transpose(1, 2).flatten(0, 1)
        row['prefix_cache'] = compare(ac[:8448], bc[:8448])
        row['suffix_cache'] = compare(ac[8448:8512], bc[8448:8512])
        row['metadata'] = {side: {key: data[key] for key in ['block_ids', 'shape', 'stride', 'use_cascade']}
                           for side, data in [('parent', a), ('child', b)]}
        report['attention64'][path.stem] = row
    rows = [json.loads(line) for line in (root / 'events.jsonl').read_text().splitlines()]
    report['transfers'] = [r for r in rows if r['event'] == 'transfer']
    if report['transfers']:
        transfer = report['transfers'][0]
        report['first_token_divergence'] = next((i for i, (a, b) in enumerate(zip(
            transfer['parent']['token_ids'], transfer['child']['token_ids'])) if a != b), None)
        alignment = {}
        for side, directory, offset in [('parent', p, parent_offset), ('child', c, 0)]:
            checked = []
            for i, token in enumerate(transfer[side]['token_ids']):
                path = directory / f'logits_{i + offset}.pt'
                if not path.exists():
                    break
                scores = load(path)['logits'].flatten()
                checked.append(bool(scores[token] == scores.max()))
            alignment[side] = {'checked': len(checked), 'all_selected_tokens_maximal': bool(checked) and all(checked)}
        report['logit_alignment'] = alignment
    (root / 'state_comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    transport = [v for layer in report['transport'].values() for v in layer.values()]
    print(json.dumps({'transport_layers': len(report['transport']), 'transport_equal': sum(v.get('byte_equal', False) for v in transport),
                      'transport_total': len(transport), 'core_layers': len(report['core64']), 'logits_steps': len(report['logits']),
                      'first_logits': report['logits'][:1]}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    analyze(parser.parse_args().root)
