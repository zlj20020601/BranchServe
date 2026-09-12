"""Capture the first prefill operator sequence without changing compiled graphs."""

import hashlib
import json
import os
from pathlib import Path

import torch


class ProbeWorkerExtension:
    pass


def tensor_bytes(value):
    # Flatten first: singleton dimensions may retain a non-unit final stride.
    return value.reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()


def install():
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    from torch._inductor.select_algorithm import extern_kernels
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention

    out = Path(os.environ['OPERATOR_PROBE_OUT'])
    out.mkdir(parents=True, exist_ok=True)
    done = False
    sequence = 0

    def active():
        return not done and (out / 'ACTIVE').exists() and not torch.cuda.is_current_stream_capturing()

    def snapshot(values):
        return {k: v.detach().cpu().clone() for k, v in values.items() if isinstance(v, torch.Tensor)}

    def digest(values):
        return {k: {'shape': list(v.shape), 'dtype': str(v.dtype),
                    'sha256': hashlib.sha256(tensor_bytes(v)).hexdigest()}
                for k, v in values.items()}

    def record(kind, name, before, after, metadata=None):
        nonlocal sequence
        row = {'index': sequence, 'kind': kind, 'name': name, 'before': digest(before),
               'after': digest(after), 'metadata': metadata or {}}
        torch.save({'before': before, 'after': after, 'metadata': metadata or {}}, out / f'op_{sequence:03d}.pt')
        with (out / 'trace.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        sequence += 1

    original = CachingAutotuner.run

    def run(self, *args, **kwargs):
        if not active():
            return original(self, *args, **kwargs)
        values = dict(zip(self.fn.arg_names, args))
        before = snapshot({k: v for k, v in values.items() if not k.startswith('out_ptr')})
        result = original(self, *args, **kwargs)
        after = snapshot({k: v for k, v in values.items() if 'out_ptr' in k})
        config = self.launchers[0].config
        record('triton', self.fn.__name__, before, after,
               {'filename': self.filename, 'config': config.kwargs, 'num_warps': config.num_warps,
                'num_stages': config.num_stages, 'scalars': {k: v for k, v in values.items() if isinstance(v, (int, float))}})
        return result

    CachingAutotuner.run = run
    original_mm = extern_kernels.mm

    def mm(left, right, *args, **kwargs):
        if not active():
            return original_mm(left, right, *args, **kwargs)
        before = snapshot({'input': left})
        weight = digest(snapshot({'weight': right}))
        result = original_mm(left, right, *args, **kwargs)
        record('mm', 'mm', before, snapshot({'output': kwargs.get('out', result)}), {'weight': weight})
        return result

    extern_kernels.mm = mm
    original_core = QwenGatedDeltaNetAttention._forward_core

    def core(self, *args, **kwargs):
        if not active():
            return original_core(self, *args, **kwargs)
        values = dict(zip(['mixed_qkv', 'b', 'a', 'core_attn_out'], args))
        values.update(kwargs)
        before = snapshot({k: values[k] for k in ['mixed_qkv', 'b', 'a']})
        result = original_core(self, *args, **kwargs)
        record('gdn', self.prefix, before, snapshot({'output': values['core_attn_out']}))
        return result

    QwenGatedDeltaNetAttention._forward_core = core
    original_attention = FlashAttentionImpl.forward

    def attention(self, layer, query, key, value, *args, **kwargs):
        nonlocal done
        if active():
            record('attention_input', layer.layer_name, snapshot({'query': query, 'key': key, 'value': value}), {})
            done = True
            (out / 'DONE').touch()
        return original_attention(self, layer, query, key, value, *args, **kwargs)

    FlashAttentionImpl.forward = attention


if os.environ.get('OPERATOR_PROBE_OUT'):
    install()
