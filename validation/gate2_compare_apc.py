"""Compare cold-APC controls and locate the earliest captured difference."""

import argparse
import json
from pathlib import Path
import re

from gate2_compare_states import compare, load


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def first_difference(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


def analyze(root):
    events = read_rows(root / 'events.jsonl')
    controls = [r for r in events if r['event'] == 'apc_control']
    reference = next(r for r in read_rows(root.parent / 'gate2_state_sync8_20260909/events.jsonl')
                     if r['event'] == 'transfer' and r['seed'] == 2609091770)['parent']
    baseline = next(r for r in controls if r['mode'] == 'recompute1024')['result']
    report = {'last_event': events[-1], 'controls': [], 'prefill_differences': [], 'final_prefill': {}, 'logits': []}
    for row in controls:
        value = row['result']
        report['controls'].append({key: row[key] for key in ['mode', 'repeat', 'traced']} | {
            'sources': value['sources'], 'token_ids': value['token_ids'],
            'matches_parent': value['token_ids'] == reference['token_ids'],
            'matches_recompute': value['token_ids'] == baseline['token_ids'],
            'first_divergence_from_recompute': first_difference(value['token_ids'], baseline['token_ids'])})
    aroot, broot = root / 'apc', root / 'recompute528'
    traces = []
    for directory in [aroot, broot]:
        traces.append({(r['step'], int(re.search(r'layers\.(\d+)', r['layer'])[1]), r['kind']): r
                       for r in read_rows(directory / 'prefill_hashes.jsonl')})
    report['trace_counts'] = [len(t) for t in traces]
    report['trace_keys_equal'] = traces[0].keys() == traces[1].keys()
    for key in sorted(traces[0].keys() & traces[1].keys()):
        a, b = traces[0][key], traces[1][key]
        diffs = []
        for section in ['inputs', 'before', 'after']:
            for field in a.get(section, {}).keys() | b.get(section, {}).keys():
                if a.get(section, {}).get(field) != b.get(section, {}).get(field):
                    diffs.append(section + '.' + field)
        if diffs or a['num_tokens'] != b['num_tokens']:
            report['prefill_differences'].append({'step': key[0], 'layer': key[1], 'kind': key[2],
                'num_tokens': [a['num_tokens'], b['num_tokens']], 'fields': sorted(diffs)})
    for path in sorted(aroot.glob('core64_*.pt')):
        if not (broot / path.name).exists():
            continue
        a, b = load(path), load(broot / path.name)
        report['final_prefill'][path.stem] = {key: compare(a[key], b[key]) for key in
            ['conv_before', 'recurrent_before', 'mixed_qkv', 'b', 'a', 'core_out', 'conv_after', 'recurrent_after']}
    selected = {mode: next(r['result']['token_ids'] for r in controls if r['mode'] == mode and r['repeat'] == 1)
                for mode in ['apc', 'recompute528']}
    for i in range(32):
        if not (aroot / f'logits_{i}.pt').exists() or not (broot / f'logits_{i}.pt').exists():
            break
        a, b = load(aroot / f'logits_{i}.pt'), load(broot / f'logits_{i}.pt')
        row = {'step': i, 'logits': compare(a['logits'], b['logits']), 'hidden': compare(a['hidden'], b['hidden'])}
        for name, value in [('apc', a), ('recompute528', b)]:
            scores = value['logits'].float().flatten()
            vals, ids = scores.topk(5)
            row[name] = {'ids': ids.tolist(), 'scores': vals.tolist(), 'selected': selected[name][i],
                         'alignment_ok': bool(scores[selected[name][i]] == scores.max())}
        report['logits'].append(row)
    (root / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    summary = {'last_event': report['last_event'], 'controls': report['controls'], 'trace_counts': report['trace_counts'],
                      'trace_keys_equal': report['trace_keys_equal'],
                      'different_prefill_points': len(report['prefill_differences']),
                      'equal_logits_steps': sum(r['logits']['byte_equal'] for r in report['logits']),
                      'logits_steps': len(report['logits']),
                      'first_internal_difference': report['prefill_differences'][:1],
                      'first_two_logits': report['logits'][:2]}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    analyze(parser.parse_args().root)
