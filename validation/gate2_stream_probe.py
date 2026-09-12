"""Check whether the fallback memcpy honors the current CUDA stream."""

import json

import torch
from lmcache.v1.platform import torch_ops


def probe(gpu, synchronize):
    with torch.cuda.device(gpu):
        src = torch.zeros(1024, device='cuda', dtype=torch.uint8)
        dst = torch.full((1024,), 99, pin_memory=True, dtype=torch.uint8)
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(100_000_000)
            src.fill_(7)
            if synchronize:
                stream.synchronize()
            torch_ops.lmcache_memcpy_async(dst.data_ptr(), src.data_ptr(), 1024,
                                         torch_ops.TransferDirection.D2H, 0, 1 << 26)
        stream.synchronize()
        return {'gpu': gpu, 'wait_current_stream': synchronize,
                'source_values': src.unique().cpu().tolist(), 'copied_values': dst.unique().tolist(),
                'correct': bool((dst == 7).all())}


if __name__ == '__main__':
    for gpu in [0, 1]:
        for synchronize in [False, True]:
            print(json.dumps(probe(gpu, synchronize)), flush=True)
