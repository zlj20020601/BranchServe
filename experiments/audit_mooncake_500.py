"""Audit the first 500 frozen Mooncake rows before any token reconstruction."""
import json
from pathlib import Path
import statistics


def main():
    root = Path('artifacts/mooncake_frozen_20260910')
    rows = [json.loads(line) for line in (root / 'smoke.jsonl').read_text().splitlines()]
    assert len(rows) == 500
    assert all(r['phase'] == 'measured' for r in rows)
    assert all(a['timestamp'] <= b['timestamp'] for a, b in zip(rows, rows[1:]))
    assert all(r['input_length'] > 0 and r['output_length'] >= 0 for r in rows)
    gaps = [b['timestamp'] - a['timestamp'] for a, b in zip(rows, rows[1:])]
    prefix_counts = []
    for a, b in zip(rows, rows[1:]):
        n = 0
        for x, y in zip(a['hash_ids'], b['hash_ids']):
            if x != y: break
            n += 1
        prefix_counts.append(n * 512)
    report = dict(status='audit_complete', rows=len(rows), first_timestamp=rows[0]['timestamp'],
        last_timestamp=rows[-1]['timestamp'], duration_ms=rows[-1]['timestamp'] - rows[0]['timestamp'],
        interarrival_ms=dict(p50=statistics.median(gaps), p95=sorted(gaps)[int(.95 * len(gaps)) - 1],
                             min=min(gaps), max=max(gaps)),
        input_tokens=dict(mean=statistics.mean(r['input_length'] for r in rows),
                          p50=statistics.median(r['input_length'] for r in rows),
                          p95=sorted(r['input_length'] for r in rows)[474]),
        output_tokens=dict(mean=statistics.mean(r['output_length'] for r in rows),
                           p50=statistics.median(r['output_length'] for r in rows),
                           p95=sorted(r['output_length'] for r in rows)[474]),
        adjacent_shared_prefix_tokens=dict(mean=statistics.mean(prefix_counts),
            nonzero=sum(x > 0 for x in prefix_counts), p95=sorted(prefix_counts)[474]),
        caveat='Anonymous trace audit only; no semantic text or token reconstruction performed.')
    (root / 'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
