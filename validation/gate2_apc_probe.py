"""Hash prefill steps while retaining detailed final-prefill snapshots."""

import hashlib
import json
import os
from pathlib import Path

import torch

from gate2_state_probe import ProbeWorkerExtension


def install():
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    out = Path(os.environ['STATE_PROBE_OUT'])
    counts = {}

    def digest(tensor):
        value = tensor.detach().cpu().contiguous()
        return {'shape': list(value.shape), 'dtype': str(value.dtype),
                'sha256': hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()}

    def record(kind, name, data):
        key = kind + ':' + name
        step = counts.get(key, 0)
        counts[key] = step + 1
        with (out / 'prefill_hashes.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(kind=kind, layer=name, step=step, **data)) + '\n')

    original_core = QwenGatedDeltaNetAttention._forward_core

    def core(self, *args, **kwargs):
        context = get_forward_context()
        meta = context.attn_metadata
        meta = meta.get(self.prefix) if isinstance(meta, dict) else None
        if not (out / 'ACTIVE').exists() or meta is None or meta.num_actual_tokens < 2:
            return original_core(self, *args, **kwargs)
        values = dict(zip(['mixed_qkv', 'b', 'a', 'core_attn_out'], args))
        values.update(kwargs)
        idx = meta.non_spec_state_indices_tensor.flatten().long()
        idx = idx[idx >= 0].unique()
        initial = bool(meta.has_initial_state.any()) if meta.has_initial_state is not None else False
        data = {'num_tokens': meta.num_actual_tokens, 'has_initial_state': initial,
                'inputs': {key: digest(values[key]) for key in ['mixed_qkv', 'b', 'a']}}
        if initial:
            data['before'] = {'conv': digest(self.kv_cache[0][idx]), 'recurrent': digest(self.kv_cache[1][idx])}
        result = original_core(self, *args, **kwargs)
        data['after'] = {'conv': digest(self.kv_cache[0][idx]), 'recurrent': digest(self.kv_cache[1][idx]),
                         'output': digest(values['core_attn_out'])}
        record('gdn', self.prefix, data)
        return result

    QwenGatedDeltaNetAttention._forward_core = core
    original_attention = FlashAttentionImpl.forward

    def attention(self, layer, query, key, value, kv_cache, attn_metadata, output,
                  output_scale=None, output_block_scale=None):
        enabled = (out / 'ACTIVE').exists() and attn_metadata is not None and attn_metadata.num_actual_tokens >= 2
        data = None
        if enabled:
            data = {'num_tokens': attn_metadata.num_actual_tokens,
                    'seq_lens': attn_metadata.seq_lens.cpu().tolist(),
                    'inputs': {name: digest(tensor) for name, tensor in [('query', query), ('key', key), ('value', value)]}}
        result = original_attention(self, layer, query, key, value, kv_cache, attn_metadata, output,
                                    output_scale, output_block_scale)
        if data is not None:
            data['after'] = {'output': digest(output)}
            record('attention', layer.layer_name, data)
        return result

    FlashAttentionImpl.forward = attention


if os.environ.get('STATE_PROBE_OUT'):
    install()
