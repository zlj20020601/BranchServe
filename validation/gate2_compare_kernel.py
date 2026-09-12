"""Validate intervention protocol and compare with historical cold outputs."""

import argparse
from collections import Counter
import json
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def compare(directory, historical):
    events = rows(directory / 'events.jsonl')
    assert events[-1]['event'] == 'finished' and events[-1]['status'] == 'complete'
    requests = [row for row in events if row['event'] == 'kernel_control']
    assert len(requests) == 8
    calls = rows(directory / 'kernel/interventions.jsonl')
    reference = [r['result']['token_ids'] for r in rows(historical) if r['event'] == 'connector_control']
    assert len(reference) == 2 and reference[0] == reference[1]
    native = requests[0]['result']['token_ids']
    summary = {'status': 'complete', 'requests': 8, 'errors': 0,
               'scope': 'only uncaptured 528-token prefill calls; final64 and decode CUDA graphs unchanged',
               'restored_worker_pid': events[-1]['worker_pid'], 'results': []}
    for row in requests:
        result = row['result']
        assert result['sources'] == {'local_compute': 8512., 'local_cache_hit': 0., 'external_kv_transfer': 0.}
        matching = [c for c in calls if c['request'] == row['repeat']]
        changed = sum(c['changed'] for c in matching)
        assert len(matching) == 384 and all(c['tokens'] == 528 for c in matching)
        assert changed == {'native': 0, 'all4096': 384, 'first2048': 1, 'all2048': 384,
                           'firstboundary2048': 1, 'boundary2048': 128}[row['mode']]
        summary['results'].append({'repeat': row['repeat'], 'mode': row['mode'],
                                   'calls': len(matching), 'direct_launches': changed,
                                   'equals_native': result['token_ids'] == native,
                                   'equals_connector': result['token_ids'] == reference[0],
                                   'token_ids': result['token_ids']})
    modes = ['native', 'all4096', 'first2048', 'all2048']
    if any(r['mode'] == 'boundary2048' for r in requests):
        modes = ['native', 'all4096', 'firstboundary2048', 'boundary2048']
    assert Counter(r['mode'] for r in requests) == {k: 2 for k in modes}
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('historical', type=Path)
    args = parser.parse_args()
    summary = compare(args.directory, args.historical)
    (args.directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    for row in summary['results']:
        print({k: v for k, v in row.items() if k != 'token_ids'})
