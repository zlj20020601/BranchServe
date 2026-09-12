"""Create auditable token-level replay inputs from anonymous Mooncake blocks.

This preserves length and equality of shared hash blocks, but is not semantic text.
"""
import hashlib
import json
import random
from pathlib import Path

VOCAB_LOW, VOCAB_HIGH = 1000, 247000
BLOCK = 512


def block_tokens(hash_id, length=BLOCK):
    seed = int(hashlib.sha256(str(hash_id).encode()).hexdigest()[:16], 16)
    token = VOCAB_LOW + seed % (VOCAB_HIGH - VOCAB_LOW)
    return [token] * length


def replay_tokens(row, index):
    ids = []
    for hash_id in row['hash_ids']:
        if len(ids) >= row['input_length']:
            break
        ids.extend(block_tokens(hash_id, min(BLOCK, row['input_length'] - len(ids))))
    rng = random.Random(0xB5A + index)
    while len(ids) < row['input_length']:
        ids.append(rng.randrange(VOCAB_LOW, VOCAB_HIGH))
    return ids


def main():
    root = Path('artifacts/mooncake_frozen_20260910')
    rows = [json.loads(line) for line in (root / 'smoke.jsonl').read_text().splitlines()]
    selected = {json.loads(line)['trace_index'] for line in (root / '16k_executable.jsonl').read_text().splitlines()}
    rows = [row for row in rows if row['trace_index'] in selected]
    out = root / 'smoke_replay.jsonl'
    with out.open('w') as handle:
        for index, row in enumerate(rows):
            tokens = replay_tokens(row, index)
            assert len(tokens) == row['input_length']
            handle.write(json.dumps(dict(trace_index=row['trace_index'],
                timestamp=row['timestamp'], input_length=row['input_length'],
                output_length=row['output_length'], hash_ids=row['hash_ids'],
                prompt_ids=tokens, semantic_text_available=False,
                reconstruction='stable hash-block mapping; deterministic tail')) + '\n')
    manifest = dict(status='replay_inputs_ready', rows=len(rows), block_tokens=BLOCK,
        source='smoke.jsonl', output=str(out), semantic_text_available=False,
        preserves=['input_length', 'output_length', 'hash_id equality'],
        does_not_preserve=['original words', 'original answers', 'original token distribution'],
        use='system timing/cache-path only; do not score answer quality')
    (root / 'replay_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))


if __name__ == '__main__': main()
