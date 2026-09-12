"""Check CUDA copies within and across separately registered host regions."""

import ctypes
import json
import mmap

import torch
from lmcache.v1.platform.torch_ops import _get_copy_lib


def main():
    libcudart = _get_copy_lib()
    alignment = 64 * 1024 * 1024
    size = 1024 * 1024
    host = mmap.mmap(-1, 2 * alignment)
    address = ctypes.addressof(ctypes.c_char.from_buffer(host))
    registered = []
    try:
        for offset in [0, alignment]:
            ret = libcudart.cudaHostRegister(ctypes.c_void_p(address + offset),
                                            ctypes.c_size_t(alignment), ctypes.c_uint(2))
            if ret:
                raise RuntimeError(f'cudaHostRegister returned {ret}')
            registered.append(address + offset)
        for gpu in [0, 1]:
            with torch.cuda.device(gpu):
                source = torch.full((size,), 73, dtype=torch.uint8, device=f'cuda:{gpu}')
                torch.cuda.synchronize()
                for label, offset, split in [('within', 0, False),
                    ('cross_unsplit', alignment - size // 2, False),
                    ('cross_split', alignment - size // 2, True)]:
                    ctypes.memset(address + offset, 0, size)
                    blocks = [(0, size)] if not split else [(0, size // 2), (size // 2, size // 2)]
                    errors = []
                    for start, length in blocks:
                        errors.append(int(libcudart.cudaMemcpy(ctypes.c_void_p(address + offset + start),
                            ctypes.c_void_p(source.data_ptr() + start), ctypes.c_size_t(length), ctypes.c_int(4))))
                        libcudart.cudaGetLastError()
                    matches = ctypes.string_at(address + offset, size) == bytes([73]) * size
                    print(json.dumps({'gpu': gpu, 'case': label, 'bytes': size,
                        'host_registration_bytes': alignment, 'return_codes': errors,
                        'bytes_match': matches}), flush=True)
                del source
                torch.cuda.empty_cache()
    finally:
        for ptr in registered:
            libcudart.cudaHostUnregister(ctypes.c_void_p(ptr))
        host.close()


if __name__ == '__main__':
    main()
