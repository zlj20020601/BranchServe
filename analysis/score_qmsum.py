"""Offline QMSum ROUGE-L using the rouge package employed by LongBench v1."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys


def score_run(result_path, manifest_path, deps):
    sys.path.insert(0, str(deps))
    from rouge import Rouge
    scorer = Rouge()
    result = json.loads(result_path.read_text())
    raw = manifest_path.read_bytes()
    assert result['protocol']['manifest_sha256'] == hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw)
    refs = {b['id']: b['answers'] for g in manifest['groups'] for b in g['branches']}
    rows, scoring_errors = [], []
    for arm in result['arms']:
        if not arm['measured']:
            continue
        for branch in arm['branches']:
            values = []
            for reference in refs[branch['query_id']]:
                try:
                    values.append(scorer.get_scores([branch['text']], [reference])[0]['rouge-l']['f'])
                except ValueError as error:
                    values.append(0.)
                    scoring_errors.append(dict(query_id=branch['query_id'], error=str(error)))
            rows.append(dict(arm=arm['arm'], repeat=arm['repeat'],
                context_sha256=arm.get('context_sha256'), query_id=branch['query_id'],
                rouge_l=100 * max(values), tokens=branch['usage']['completion_tokens'],
                truncated=branch['finish_reason'] == 'length', empty=not branch['text'].strip()))
    summary = dict(status='complete', inference_status=result['status'],
        scoring='rouge==1.0.1 rouge-l F1; maximum over references; percentage scale',
        caveat='ROUGE-L is lexical overlap, not factual correctness.', scoring_errors=scoring_errors,
        arms={name: dict(responses=len(part), rouge_l_mean=statistics.mean(r['rouge_l'] for r in part),
            completion_tokens=sum(r['tokens'] for r in part), truncated=sum(r['truncated'] for r in part),
            empty=sum(r['empty'] for r in part))
            for name in sorted({r['arm'] for r in rows})
            if (part := [r for r in rows if r['arm'] == name])}, rows=rows)
    output = result_path.parent / 'quality.json'
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k != 'rows'}), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--deps', type=Path, required=True)
    args = parser.parse_args()
    score_run(args.result, args.manifest, args.deps)
