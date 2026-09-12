"""Filter frozen Mooncake rows against the active BranchServe window."""
import json
from pathlib import Path
from collections import Counter

WINDOW = 16384

def main():
    root = Path('artifacts/mooncake_frozen_20260910')
    rows = [json.loads(x) for x in (root / 'smoke.jsonl').read_text().splitlines()]
    executable, prefill_only, too_long = [], [], []
    for row in rows:
        total = row['input_length'] + max(1, row['output_length'])
        if row['output_length'] == 0:
            prefill_only.append(row)
        elif total <= WINDOW:
            executable.append(row)
        else:
            too_long.append(row)
    for name, part in [('16k_executable', executable), ('prefill_only', prefill_only), ('longer_than_16k', too_long)]:
        (root / (name + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in part))
    report = dict(status='filtered', source_rows=len(rows), window_tokens=WINDOW,
        rule='output_tokens > 0 and input_tokens + max(1, output_tokens) <= 16384',
        executable_rows=len(executable), prefill_only_rows=len(prefill_only),
        over_window_rows=len(too_long),
        executable_input=dict(min=min(r['input_length'] for r in executable),
            p50=sorted(r['input_length'] for r in executable)[len(executable)//2],
            max=max(r['input_length'] for r in executable)),
        output_distribution=dict(p50=sorted(r['output_length'] for r in executable)[len(executable)//2],
            max=max(r['output_length'] for r in executable)),
        excluded_by_window=dict(prefill_only=len(prefill_only), over_window=len(too_long)),
        caveat='Rows are preserved in separate files; no truncation or synthetic padding.')
    (root / '16k_filter_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
