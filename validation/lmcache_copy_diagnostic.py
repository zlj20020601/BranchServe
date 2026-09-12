"""Process-local LMCache copy tracing, with optional registration-boundary splitting."""

import argparse
import ctypes
import json
import sys


def segments(offset, size, alignment):
    if size < 0 or alignment <= 0 or alignment & (alignment - 1):
        raise ValueError('Invalid size or alignment')
    done = 0
    while done < size:
        length = min(size - done, alignment - ((offset + done) % alignment))
        yield done, length
        done += length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--copy-mode', choices=['observe', 'split'], required=True)
    parser.add_argument('--synchronize-copy', action='store_true',
                        help='Diagnostic ordering barrier around pointer-mode copies')
    args, server_args = parser.parse_known_args()
    import torch
    from lmcache.v1.platform import torch_ops
    original = torch_ops.lmcache_memcpy_async

    def traced(dest, src, nbytes, direction, host_buffer_offset, host_buffer_alignments):
        if not isinstance(dest, int) or not isinstance(src, int):
            return original(dest, src, nbytes, direction, host_buffer_offset, host_buffer_alignments)
        pieces = list(segments(host_buffer_offset, nbytes, host_buffer_alignments))
        details = {'mode': args.copy_mode, 'src': hex(src), 'dest': hex(dest),
            'nbytes': nbytes, 'direction': int(direction), 'device': torch.cuda.current_device(),
            'host_offset': host_buffer_offset, 'alignment': host_buffer_alignments,
            'crosses_boundary': len(pieces) > 1, 'pieces': pieces}
        if len(pieces) > 1:
            print('COPY_DIAG ' + json.dumps(dict(details, event='crossing')), flush=True)
        try:
            if args.synchronize_copy:
                torch.cuda.current_stream().synchronize()
            if args.copy_mode == 'split' and len(pieces) > 1:
                for offset, length in pieces:
                    original(dest + offset, src + offset, length, direction,
                             host_buffer_offset + offset, host_buffer_alignments)
            else:
                original(dest, src, nbytes, direction, host_buffer_offset, host_buffer_alignments)
            if args.synchronize_copy:
                torch.cuda.synchronize()
        except Exception as error:
            print('COPY_DIAG ' + json.dumps(dict(details, event='error', error=repr(error))), flush=True)
            raise

    torch_ops.lmcache_memcpy_async = traced
    from lmcache.cli.main import main as cli
    sys.argv = ['lmcache', *server_args]
    return cli()


if __name__ == '__main__':
    sys.exit(main())
