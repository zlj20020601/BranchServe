"""Freeze untruncated QMSum four-query groups using the deployment tokenizer."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


TEMPLATE = ('You are given a meeting transcript and a query containing a question or instruction. '
            'Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\n'
            'Now, answer the query based on the above meeting transcript in one or more sentences.\n\n'
            'Query: {input}\nAnswer:')


def common_length(sequences):
    return next((i for i, values in enumerate(zip(*sequences))
                 if len(set(values)) != 1), min(map(len, sequences)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='/root/autodl-tmp/models/Qwen3.5-4B')
    args = parser.parse_args()
    from transformers import AutoTokenizer
    import transformers
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    source = args.source.read_bytes()
    groups = defaultdict(dict)
    for line in source.decode('utf-8-sig').splitlines():
        row = json.loads(line)
        key = hashlib.sha256(row['context'].encode()).hexdigest()
        groups[key].setdefault(row['input'], row)

    def render(context, query):
        text = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': TEMPLATE.format(context=context, input=query)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
        return ids

    accepted, rejected = [], []
    for key, records in sorted(groups.items()):
        if len(records) < 4:
            rejected.append(dict(context_sha256=key, reason='fewer_than_four_distinct_queries', queries=len(records)))
            continue
        rows = sorted(records.values(), key=lambda row: row['_id'])
        rendered = [(row, render(row['context'], row['input'])) for row in rows]
        fitting = [(row, ids) for row, ids in rendered if len(ids) + 512 <= 16384]
        if len(fitting) < 4:
            rejected.append(dict(context_sha256=key, reason='fewer_than_four_full_prompts_fit_16k',
                                 prompt_lengths=[len(ids) for _, ids in rendered]))
            continue
        selected = fitting[:4]
        parent_ids = render(rows[0]['context'], 'Identify the main subject of this meeting in one sentence.')
        if len(parent_ids) + 32 > 16384:
            rejected.append(dict(context_sha256=key, reason='parent_prompt_exceeds_window'))
            continue
        shared = common_length([parent_ids, *[ids for _, ids in selected]])
        cached = shared // 528 * 528
        if cached == 0:
            rejected.append(dict(context_sha256=key, reason='no_aligned_shared_prefix'))
            continue
        accepted.append(dict(context_sha256=key, context=rows[0]['context'],
            original_query_count=len(rows), fitting_query_count=len(fitting),
            shared_tokens=shared, aligned_shared_tokens=cached, parent_ids=parent_ids,
            branches=[dict(id=row['_id'], query=row['input'], answers=row['answers'],
                           prompt_ids=ids, prompt_tokens=len(ids)) for row, ids in selected]))
    manifest = dict(source='THUDM/LongBench v1 qmsum', source_sha256=hashlib.sha256(source).hexdigest(),
        transformers_version=transformers.__version__, model=args.model,
        tokenizer_class=type(tokenizer).__name__,
        chat_template_sha256=hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
        protocol=dict(fanout=4, max_model_len=16384, max_output_tokens=512, parent_max_tokens=32,
            enable_thinking=False, chunk_size=528, temperature=0, truncation=False,
            selection='context SHA256 order; first four fitting distinct questions by original ID',
            scope='Shared-context multi-query workload; not original agent traces'),
        groups=accepted, rejected=rejected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False)
    print(json.dumps(dict(total_contexts=len(groups), accepted=len(accepted),
        rejected=rejected, accepted_lengths=[dict(hash=g['context_sha256'][:12],
        shared=g['shared_tokens'], aligned=g['aligned_shared_tokens'],
        prompts=[b['prompt_tokens'] for b in g['branches']]) for g in accepted]), indent=2))


if __name__ == '__main__':
    main()
