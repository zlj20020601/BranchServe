"""Compare compile caches without loading executable pickle payloads."""

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickletools
import re

from torch.utils._appending_byte_serializer import BytesReader


def entries(path):
    reader = BytesReader(path.read_bytes())
    parts = []
    while not reader.is_finished():
        parts.append(reader.read_bytes())
    assert parts[0] == b'CacheCompiledArtifact'
    reader = BytesReader(parts[-1])
    assert reader.read_uint64() == 1
    result = {}
    while not reader.is_finished():
        kind = reader.read_str()
        for _ in range(reader.read_uint64()):
            key = reader.read_str()
            result[kind, key] = reader.read_bytes()
    return result


def source_strings(payload):
    # Inspect pickle opcodes only; never execute cached constructors.
    return [value for _, value, _ in pickletools.genops(payload)
            if isinstance(value, str) and ('def ' in value or 'import ' in value)
            and len(value) > 300]


def normalize_source(source):
    return re.sub(r'/torch_aot_compile/[0-9a-f]{64}/',
                  '/torch_aot_compile/CACHE/', source)


def audit(left, right):
    graph = 'computation_graph.py'
    a = ast.literal_eval((left / 'vllm_compile_cache.py').read_text())
    b = ast.literal_eval((right / 'vllm_compile_cache.py').read_text())
    result = {
        'left': str(left), 'right': str(right),
        'graph_equal': (left / graph).read_bytes() == (right / graph).read_bytes(),
        'graph_sha256': hashlib.sha256((left / graph).read_bytes()).hexdigest(),
        'subgraph_keys_equal': a.keys() == b.keys() and all(
            a[k]['cache_key'] == b[k]['cache_key'] for k in a),
        'subgraphs': [],
    }
    for path in sorted(left.glob('artifact_compile*')):
        ea, eb = entries(path), entries(right / path.name)
        row = {'name': path.name, 'entry_keys_equal': ea.keys() == eb.keys(),
               'entry_counts': dict(Counter(k[0] for k in ea)), 'differences': []}
        for key in sorted(ea.keys() | eb.keys()):
            va, vb = ea.get(key), eb.get(key)
            if va == vb:
                continue
            diff = {'kind': key[0], 'key': key[1]}
            if va is None or vb is None:
                diff['missing_side'] = 'left' if va is None else 'right'
            elif key[0] == 'autotune':
                ja, jb = json.loads(va), json.loads(vb)
                diff['fields'] = {k: [ja.get(k), jb.get(k)]
                                  for k in ja.keys() | jb.keys() if ja.get(k) != jb.get(k)}
            elif key[0] in ('inductor', 'aot_autograd'):
                sa, sb = source_strings(va), source_strings(vb)
                diff.update(source_count_left=len(sa), source_count_right=len(sb),
                            source_strings_equal=sa == sb,
                            source_equal_ignoring_cache_path=(
                                list(map(normalize_source, sa)) == list(map(normalize_source, sb))),
                            source_sha256_left=[hashlib.sha256(s.encode()).hexdigest() for s in sa])
            row['differences'].append(diff)
        result['subgraphs'].append(row)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.left, args.right)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
