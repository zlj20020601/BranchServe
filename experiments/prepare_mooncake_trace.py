"""Freeze original Mooncake rows and disjoint chronological experiment windows."""
import hashlib
import json
from pathlib import Path
import urllib.request

URL = 'https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/toolagent_trace.jsonl'


def main():
    output = Path('artifacts/mooncake_frozen_20260910')
    output.mkdir(exist_ok=False)
    with urllib.request.urlopen(URL, timeout=45) as response:
        raw = response.read()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert len(rows) >= 19000
    assert all(a['timestamp'] <= b['timestamp'] for a, b in zip(rows, rows[1:]))
    for row in rows:
        assert row['input_length'] > 0 and row['output_length'] >= 0
        assert isinstance(row['hash_ids'], list)
    (output / 'toolagent_trace.jsonl').write_bytes(raw)
    splits = dict(smoke=dict(start=0, count=500), calibration=dict(start=500, count=1000))
    for index, start in enumerate([2000, 6000, 10000, 14000, 18000]):
        splits[f'test_{index+1}'] = dict(start=start, count=1000, warmup_start=start-200, warmup_count=200)
    for name, split in splits.items():
        start = split.get('warmup_start', split['start'])
        end = split['start'] + split['count']
        selected = [dict(trace_index=i, phase='warmup' if i < split['start'] else 'measured',
                         **rows[i]) for i in range(start, end)]
        encoded = ''.join(json.dumps(r, separators=(',', ':')) + '\n' for r in selected).encode()
        (output / (name + '.jsonl')).write_bytes(encoded)
        split.update(sha256=hashlib.sha256(encoded).hexdigest(),
                     first_timestamp=selected[0]['timestamp'], last_timestamp=selected[-1]['timestamp'])
    manifest = dict(source=URL, source_sha256=hashlib.sha256(raw).hexdigest(),
        rows=len(rows), block_tokens=512, splits=splits,
        selection='Fixed index windows before strategy outcomes; original order and timestamps retained.',
        status='data_frozen_not_replay_validated',
        caveat='Anonymous timing/length/hash trace, not semantic prompts or a branch DAG; no filtering yet.')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
