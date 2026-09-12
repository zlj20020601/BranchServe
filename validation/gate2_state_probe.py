"""Process-local worker instrumentation; not a latency benchmark."""

import os
from pathlib import Path
import json

import torch


class ProbeWorkerExtension:
    pass


def install():
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
    from lmcache.integration.vllm.vllm_multi_process_adapter import LMCacheMPWorkerAdapter
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    out = Path(os.environ['STATE_PROBE_OUT'])
    out.mkdir(parents=True, exist_ok=True)
    raw = {}
    pending = {}
    step = {}
    logits_count = 0
    logits_ready = False
    (out / 'logits_protocol.json').write_text(json.dumps({'start': 'after_final_64_token_prefill'}))

    def active():
        return (out / 'ACTIVE').exists()

    def save(name, value):
        torch.save(value, out / (name + '.pt'))

    original_register = LMCacheMPConnector.register_kv_caches

    def register(self, caches):
        raw.update(caches)
        result = original_register(self, caches)
        (out / 'layout.json').write_text(json.dumps({name: [
            {'shape': list(t.shape), 'stride': list(t.stride()), 'dtype': str(t.dtype)}
            for t in (cache if isinstance(cache, list) else [cache])]
            for name, cache in caches.items()}, indent=2))
        return result

    LMCacheMPConnector.register_kv_caches = register

    def snapshot(adapter, op, label):
        torch.cuda.synchronize()
        names = list(adapter.kv_caches)
        result = {'start': op.start, 'end': op.end, 'block_ids': op.block_ids, 'layers': {}}
        for group in adapter.engine_group_infos:
            ids = list(op.block_ids[group.engine_group_id])
            # Recurrent groups restore only the last prefix snapshot.
            if group.sw_size_tokens > 0:
                ids = ids[-1:]
            for index in group.layer_indices:
                name = names[index]
                cache = raw[name]
                data = {'ids': ids, 'group': group.engine_group_id}
                if isinstance(cache, list):
                    data['conv'] = cache[0][ids].detach().cpu()
                    data['recurrent'] = cache[1][ids].detach().cpu()
                else:
                    data['page_bytes'] = adapter.kv_caches[name][ids].detach().cpu().contiguous().view(torch.uint8)
                result['layers'][name] = data
        save(label, result)

    original_store = LMCacheMPWorkerAdapter.submit_store_request

    def store(self, request_id, op, event, cache_salt=''):
        if active() and op.end <= 8448:
            snapshot(self, op, 'store_' + str(op.end))
        return original_store(self, request_id, op, event, cache_salt)

    LMCacheMPWorkerAdapter.submit_store_request = store
    original_retrieve = LMCacheMPWorkerAdapter.submit_retrieve_request

    def retrieve(self, request_id, op, event, cache_salt=''):
        if active():
            pending[request_id] = op
        return original_retrieve(self, request_id, op, event, cache_salt)

    LMCacheMPWorkerAdapter.submit_retrieve_request = retrieve
    original_finished = LMCacheMPWorkerAdapter.get_finished

    def finished(self, ids):
        result = original_finished(self, ids)
        for request_id in result[1] or []:
            op = pending.pop(request_id, None)
            if op is not None:
                snapshot(self, op, 'retrieve_' + str(op.end))
                if os.environ.get('STATE_REPAIR_ATTN') == '1':
                    parent_dir = out.parent / 'gpu0'
                    stores = [torch.load(path, map_location='cpu', weights_only=False) for path in
                              sorted(parent_dir.glob('store_*.pt'), key=lambda p: int(p.stem.split('_')[1]))]
                    names = list(self.kv_caches)
                    for group in self.engine_group_infos:
                        for index in group.layer_indices:
                            name = names[index]
                            if isinstance(raw[name], list):
                                continue
                            cache = self.kv_caches[name]
                            values = torch.cat([s['layers'][name]['page_bytes'] for s in stores])
                            values = values.view(cache.dtype).to(cache.device)
                            ids_gpu = torch.tensor(op.block_ids[group.engine_group_id], device=cache.device)
                            cache.index_copy_(0, ids_gpu, values)
                    torch.cuda.synchronize()
                    snapshot(self, op, 'repaired_' + str(op.end))
        return result

    LMCacheMPWorkerAdapter.get_finished = finished
    original_core = QwenGatedDeltaNetAttention._forward_core

    def core(self, *args, **kwargs):
        nonlocal logits_ready
        context = get_forward_context()
        metadata = context.attn_metadata
        meta = metadata.get(self.prefix) if isinstance(metadata, dict) else None
        enabled = active() and meta is not None and meta.num_actual_tokens == 64
        if enabled:
            logits_ready = True
        if not enabled:
            return original_core(self, *args, **kwargs)
        idx = meta.non_spec_state_indices_tensor.detach().cpu().long().flatten()
        idx = idx[idx >= 0].unique().tolist()
        data = {'indices': idx, 'num_tokens': meta.num_actual_tokens,
                'has_initial_state': meta.has_initial_state.detach().cpu() if meta.has_initial_state is not None else None,
                'conv_before': self.kv_cache[0][idx].detach().cpu(),
                'recurrent_before': self.kv_cache[1][idx].detach().cpu()}
        values = dict(zip(['mixed_qkv', 'b', 'a', 'core_attn_out'], args))
        values.update(kwargs)
        for key in ['mixed_qkv', 'b', 'a']:
            data[key] = values[key].detach().cpu()
        result = original_core(self, *args, **kwargs)
        data['core_out'] = values['core_attn_out'].detach().cpu()
        data['conv_after'] = self.kv_cache[0][idx].detach().cpu()
        data['recurrent_after'] = self.kv_cache[1][idx].detach().cpu()
        count = step.get(self.prefix, 0)
        step[self.prefix] = count + 1
        save('core64_' + self.prefix + '_' + str(count), data)
        return result

    QwenGatedDeltaNetAttention._forward_core = core
    original_logits = Qwen3_5ForCausalLMBase.compute_logits

    def logits(self, hidden_states):
        nonlocal logits_count
        result = original_logits(self, hidden_states)
        if active() and logits_ready and result is not None and logits_count < 32:
            save('logits_' + str(logits_count), {'hidden': hidden_states.detach().cpu(), 'logits': result.detach().cpu()})
            logits_count += 1
        return result

    Qwen3_5ForCausalLMBase.compute_logits = logits
    original_attention = FlashAttentionImpl.forward

    def attention(self, layer, query, key, value, kv_cache, attn_metadata, output,
                  output_scale=None, output_block_scale=None):
        enabled = active() and attn_metadata is not None and attn_metadata.num_actual_tokens == 64
        data = None
        if enabled:
            table = attn_metadata.block_table.detach().cpu()
            lengths = attn_metadata.seq_lens.detach().cpu()
            count = (int(lengths[0]) + kv_cache.shape[2] - 1) // kv_cache.shape[2]
            ids = table[0, :count].long().tolist()
            data = {'query': query.detach().cpu(), 'key': key.detach().cpu(), 'value': value.detach().cpu(),
                    'cache': kv_cache[ids].detach().cpu(), 'block_ids': ids,
                    'seq_lens': lengths, 'query_start_loc': attn_metadata.query_start_loc.detach().cpu(),
                    'shape': list(kv_cache.shape), 'stride': list(kv_cache.stride()),
                    'use_cascade': attn_metadata.use_cascade}
        result = original_attention(self, layer, query, key, value, kv_cache, attn_metadata, output,
                                    output_scale, output_block_scale)
        if data is not None:
            data['output'] = output.detach().cpu()
            save('attention64_' + layer.layer_name, data)
        return result

    FlashAttentionImpl.forward = attention
    print('STATE_PROBE_INSTALLED ' + str(out), flush=True)


if os.environ.get('STATE_PROBE_OUT'):
    install()
