"""Compare the same-GPU cold connector trace with APC-only."""

import argparse
import json
from pathlib import Path
import re

from gate2_compare_apc import first_difference, read_rows
from gate2_compare_states import compare, load


def analyze(root, apc_root):
    events = read_rows(root / 'events.jsonl')
    controls = [r for r in events if r['event'] == 'connector_control']
    apc = next(r['result'] for r in read_rows(apc_root / 'events.jsonl')
               if r['event'] == 'apc_control' and r['mode'] == 'apc' and r['repeat'] == 1)
    parent = next(r['parent'] for r in read_rows(root.parent / 'gate2_state_sync8_20260909/events.jsonl')
                  if r['event'] == 'transfer' and r['seed'] == 2609091770)
    report = {'last_event': events[-1], 'controls': [], 'prefill_differences': [], 'logits': []}
    for row in controls:
        value = row['result']
        report['controls'].append({key: row[key] for key in ['repeat', 'traced', 'stored_tokens', 'store_errors']} | {
            'sources': value['sources'], 'token_ids': value['token_ids'],
            'matches_apc': value['token_ids'] == apc['token_ids'],
            'matches_old_parent': value['token_ids'] == parent['token_ids'],
            'first_divergence_from_apc': first_difference(value['token_ids'], apc['token_ids'])})
    aroot, broot = root / 'connector', apc_root / 'apc'
    traces = [{(r['step'], int(re.search(r'layers\.(\d+)', r['layer'])[1]), r['kind']): r
               for r in read_rows(directory / 'prefill_hashes.jsonl')} for directory in [aroot, broot]]
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
    for i in range(32):
        a, b = load(aroot / f'logits_{i}.pt'), load(broot / f'logits_{i}.pt')
        row = {'step': i, 'logits': compare(a['logits'], b['logits']), 'hidden': compare(a['hidden'], b['hidden'])}
        for name, value, tokens in [('connector', a, controls[0]['result']['token_ids']), ('apc', b, apc['token_ids'])]:
            scores = value['logits'].float().flatten()
            vals, ids = scores.topk(5)
            row[name] = {'ids': ids.tolist(), 'scores': vals.tolist(), 'selected': tokens[i],
                         'alignment_ok': bool(scores[tokens[i]] == scores.max())}
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
    parser.add_argument('apc_root', type=Path)
    args = parser.parse_args()
    analyze(args.root, args.apc_root)
