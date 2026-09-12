"""Produce a compact, auditable state-diagnostic summary."""

import json
from pathlib import Path


ROOT = Path('/root/autodl-tmp/branchserve/artifacts')


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    result = {'runs': {}, 'reference_comparison': []}
    for name in ['gate2_state_20260909', 'gate2_state_repair_20260909',
                 'gate2_state_attention_20260909', 'gate2_state_sync_20260909',
                 'gate2_state_sync8_20260909']:
        root = ROOT / name
        events = rows(root / 'events.jsonl')
        pairs = [r for r in events if r['event'] == 'transfer']
        item = {'last_event': events[-1], 'protocol': next(r for r in events if r['event'] == 'protocol'),
                'pair_count': len(pairs), 'outputs_equal': sum(r['tokens_equal'] for r in pairs),
                'all_sources_ok': all(r['source_ok'] for r in pairs)}
        comparison = root / 'state_comparison.json'
        if comparison.exists():
            data = json.loads(comparison.read_text())
            transport = [v for layer in data['transport'].values() for v in layer.values()]
            item.update(transport_equal=sum(v.get('byte_equal', False) for v in transport),
                        transport_total=len(transport),
                        recurrent_transport_equal=sum(layer['recurrent']['byte_equal'] for layer in data['transport'].values() if 'recurrent' in layer),
                        conv_transport_equal=sum(layer['conv']['byte_equal'] for layer in data['transport'].values() if 'conv' in layer),
                        kv_transport_equal=sum(layer['page_bytes']['byte_equal'] for layer in data['transport'].values() if 'page_bytes' in layer),
                        core_all_equal=all(v['byte_equal'] for layer in data['core64'].values() for v in layer.values()),
                        consumer_boundary_equal=all(v['byte_equal'] for layer in data['boundary_to_consumer'].values() for v in layer.values()),
                        logits_equal=sum(v['logits']['byte_equal'] for v in data['logits']),
                        logits_total=len(data['logits']), first_logits=data['logits'][:1],
                        logit_alignment=data.get('logit_alignment'), first_token_divergence=data.get('first_token_divergence'))
        result['runs'][name] = item
    reference = {}
    for row in rows(ROOT / 'gate2_output_controls_clean_20260909/events.jsonl'):
        if row['event'] == 'recompute_control':
            reference.setdefault(row['seed'], []).append(row['result']['token_ids'])
    for row in rows(ROOT / 'gate2_state_sync8_20260909/events.jsonl'):
        if row['event'] != 'transfer':
            continue
        expected = reference[row['seed']]
        result['reference_comparison'].append({'seed': row['seed'], 'reference_repeats': len(expected),
            'reference_consistent': all(x == expected[0] for x in expected),
            'parent_matches_recompute': row['parent']['token_ids'] == expected[0],
            'retrieve_matches_recompute': row['child']['token_ids'] == expected[0],
            'parent_equals_retrieve': row['tokens_equal'], 'sources': row['child']['sources']})
    result['stream_probe'] = [json.loads(line) for line in (ROOT / 'gate2_stream_probe.log').read_text().splitlines()
                              if line.startswith('{')]
    chunk_path = ROOT / 'gate2_chunk_control_20260909/events.jsonl'
    if chunk_path.exists():
        events = rows(chunk_path)
        controls = [r for r in events if r['event'] == 'chunk_control']
        pair = next(r for r in rows(ROOT / 'gate2_state_sync8_20260909/events.jsonl')
                    if r['event'] == 'transfer' and r['seed'] == 2609091770)
        result['chunk_control'] = {'last_event': events[-1], 'rows': []}
        for row in controls:
            result['chunk_control']['rows'].append({'chunk': row['chunk'], 'repeat': row['repeat'],
                'matches_parent': row['result']['token_ids'] == pair['parent']['token_ids'],
                'matches_old_recompute': row['result']['token_ids'] == reference[2609091770][0],
                'sources': row['result']['sources'], 'token_ids': row['result']['token_ids']})
        small = next((r for r in controls if r['chunk'] == 528), None)
        large = next((r for r in controls if r['chunk'] == 1024), None)
        if small and large:
            divergence = next((i for i, (a, b) in enumerate(zip(small['result']['token_ids'],
                                large['result']['token_ids'])) if a != b), None)
            result['chunk_control']['first_divergence'] = divergence
            if divergence is not None:
                result['chunk_control']['first_divergence_scores'] = {str(r['chunk']): {
                    'token': r['result']['token_ids'][divergence],
                    'top_logprobs': r['result']['logprobs']['top_logprobs'][divergence]}
                    for r in [small, large]}
        if large:
            divergence = next((i for i, (a, b) in enumerate(zip(pair['parent']['token_ids'],
                                large['result']['token_ids'])) if a != b), None)
            result['chunk_control']['parent_vs_recompute_first_divergence'] = divergence
            if divergence is not None:
                result['chunk_control']['parent_vs_recompute_scores'] = {name: {
                    'token': data['token_ids'][divergence],
                    'top_logprobs': data['logprobs']['top_logprobs'][divergence]}
                    for name, data in [('parent', pair['parent']), ('recompute', large['result'])]}
    target = ROOT / 'gate2_state_summary_20260909.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
