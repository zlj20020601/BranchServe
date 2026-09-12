"""Run isolated, counterbalanced Gate 2 trials on the existing two-GPU host."""

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import secrets
import signal
import statistics
import subprocess
import time
import urllib.request

from prometheus_client.parser import text_string_to_metric_families


ROOT = Path('/root/autodl-tmp/branchserve')
ENV = Path('/root/autodl-tmp/conda_envs/branchserve')
MODEL = '/root/autodl-tmp/models/Qwen3.5-4B'
MODEL_ID = 'qwen3.5-4b'
PREFIX = 8448
SUFFIX = 64
OUTPUT = 32
CONNECTOR = {
    'kv_connector': 'LMCacheMPConnector',
    'kv_connector_module_path': 'lmcache.integration.vllm.lmcache_mp_connector',
    'kv_role': 'kv_both',
    'kv_connector_extra_config': {'lmcache.mp.host': 'tcp://localhost', 'lmcache.mp.port': 5555},
}


class Client:
    def __init__(self, port, token=None):
        self.url = f'http://127.0.0.1:{port}'
        self.headers = {'Authorization': f'Bearer {token}'} if token else {}

    def get(self, path):
        request = urllib.request.Request(self.url + path, headers=self.headers)
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.read().decode()

    def metrics(self):
        result = {}
        for family in text_string_to_metric_families(self.get('/metrics')):
            for sample in family.samples:
                if not sample.name.endswith('_created'):
                    result[(sample.name, tuple(sorted(sample.labels.items())))] = sample.value
        return result

    def complete(self, prompt):
        body = {'model': MODEL_ID, 'prompt': prompt, 'temperature': 0,
                'max_tokens': OUTPUT, 'stream': True, 'stream_options': {'include_usage': True}}
        request = urllib.request.Request(self.url + '/v1/completions',
            data=json.dumps(body).encode(), headers=dict(self.headers, **{'Content-Type': 'application/json'}))
        started = time.perf_counter()
        first, finish, usage, text = None, None, None, []
        with urllib.request.urlopen(request, timeout=120) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith('data:'):
                    continue
                value = line[5:].strip()
                if value == '[DONE]':
                    break
                value = json.loads(value)
                if value.get('error'):
                    raise RuntimeError(str(value['error']))
                if value.get('usage'):
                    usage = value['usage']
                for choice in value.get('choices', []):
                    if choice.get('text'):
                        if first is None:
                            first = time.perf_counter() - started
                        text.append(choice['text'])
                    if choice.get('finish_reason'):
                        finish = choice['finish_reason']
        elapsed = time.perf_counter() - started
        if first is None or usage is None or not finish:
            raise RuntimeError('Incomplete streaming response or missing usage')
        return {'ttft_s': first, 'total_s': elapsed, 'finish_reason': finish,
                'usage': usage, 'output_sha256': hashlib.sha256(''.join(text).encode()).hexdigest()}


def metric(values, name, **labels):
    matches = [v for (n, fields), v in values.items()
               if n == name and all(dict(fields).get(k) == val for k, val in labels.items())]
    return sum(matches) if matches else None


def delta(before, after, name, **labels):
    a, b = metric(before, name, **labels), metric(after, name, **labels)
    return None if a is None or b is None else b - a


def audit(before, after, mode, result, parent_hash, parent_before, parent_after):
    sources = {s: delta(before, after, 'vllm:prompt_tokens_by_source_total', source=s)
               for s in ['local_compute', 'local_cache_hit', 'external_kv_transfer']}
    # Some vLLM modes omit the two cache counters entirely. Full recompute and
    # exact request/token accounting still establish that no prefix was reused.
    if mode == 'recompute' and sources['local_compute'] == PREFIX + SUFFIX:
        for source in ['local_cache_hit', 'external_kv_transfer']:
            if sources[source] is None:
                sources[source] = 0.0
    success = delta(before, after, 'vllm:request_success_total')
    errors = delta(before, after, 'vllm:request_success_total', finished_reason='error')
    aborts = delta(before, after, 'vllm:request_success_total', finished_reason='abort')
    checks = {
        'one_request': success == 1,
        'no_server_errors': errors == 0 and aborts == 0,
        'parent_idle': parent_before == parent_after,
        'prompt_count': result['usage']['prompt_tokens'] == PREFIX + SUFFIX,
        'output_matches': result['output_sha256'] == parent_hash,
        'no_local_hit': sources['local_cache_hit'] == 0,
        'kv_source': (sources['local_compute'] == PREFIX + SUFFIX and sources['external_kv_transfer'] == 0)
            if mode == 'recompute' else
            (sources['external_kv_transfer'] == PREFIX and sources['local_compute'] == SUFFIX),
    }
    return {'sources': sources, 'request_delta': success, 'checks': checks, 'valid': all(checks.values())}


def prompt(seed, vocab):
    rng = random.Random(seed)
    return [rng.randrange(1000, vocab - 1000) for _ in range(PREFIX + SUFFIX)]


def summarize(rows):
    pairs = []
    for pair in sorted({r['pair'] for r in rows}):
        arms = {r['mode']: r for r in rows if r['pair'] == pair}
        if set(arms) != {'recompute', 'retrieve'}:
            continue
        a, b = arms['recompute'], arms['retrieve']
        valid = a['valid'] and b['valid'] and a['output_sha256'] == b['output_sha256']
        pairs.append({'pair': pair, 'order': a['order'], 'valid': valid,
            'recompute_ttft_s': a['ttft_s'], 'retrieve_ttft_s': b['ttft_s'],
            'ttft_speedup': a['ttft_s'] / b['ttft_s'],
            'total_speedup': a['total_s'] / b['total_s'],
            'same_output_tokens': a['usage']['completion_tokens'] == b['usage']['completion_tokens']})
    result = {'pairs': pairs, 'n_pairs': len(pairs), 'n_valid': sum(p['valid'] for p in pairs)}
    valid_rows = [r for r in rows if r['pair'] in {p['pair'] for p in pairs if p['valid']}]
    for mode in ['recompute', 'retrieve']:
        subset = [r for r in valid_rows if r['mode'] == mode]
        if subset:
            result[mode] = {key: {'mean': statistics.mean(r[key] for r in subset),
                'median': statistics.median(r[key] for r in subset),
                'min': min(r[key] for r in subset), 'max': max(r[key] for r in subset)}
                for key in ['ttft_s', 'total_s']}
    valid_pairs = [p for p in pairs if p['valid']]
    if valid_pairs:
        rng = random.Random(20260909)
        for key in ['ttft_speedup', 'total_speedup']:
            values = [p[key] for p in valid_pairs]
            boot = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(10000))
            result[key] = {'paired_mean': statistics.mean(values), 'median': statistics.median(values),
                          'bootstrap_mean_ci95': [boot[249], boot[9749]]}
        result['order_ttft_speedup'] = {order: statistics.mean(p['ttft_speedup'] for p in valid_pairs if p['order'] == order)
            for order in sorted({p['order'] for p in valid_pairs})}
    return result


class Suite:
    def __init__(self, args):
        self.args = args
        self.output = ROOT / 'artifacts' / args.run_id
        self.output.mkdir(exist_ok=False)
        self.log = (self.output / 'events.jsonl').open('x', buffering=1)
        self.parent = Client(8000)
        self.token = secrets.token_urlsafe(32)
        self.child = Client(18001, self.token)
        self.proc = None
        self.mode = None
        self.parent_count = None
        self.rows = []
        self.restart = 0
        self.restore_needed = False
        self.original_mode = None
        self.gpu_uuid = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=uuid',
            '--format=csv,noheader'], text=True).strip()

    def record(self, event, **data):
        row = {'event': event, 'time': time.strftime('%Y-%m-%d %H:%M:%S'), **data}
        self.log.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)

    def check_parent(self):
        values = self.parent.metrics()
        count = metric(values, 'vllm:request_success_total')
        if count is None or (self.parent_count is not None and count != self.parent_count):
            raise RuntimeError('Unexpected GPU0 request count: concurrent traffic or worker restart')
        if metric(values, 'vllm:num_requests_running') != 0 or metric(values, 'vllm:num_requests_waiting') != 0:
            raise RuntimeError('GPU0 is not idle')
        self.parent_count = count
        return count

    def prepare(self, value, label):
        self.check_parent()
        store_log = Path(self.args.lmcache_log)
        offset = store_log.stat().st_size
        result = self.parent.complete(value)
        self.parent_count += 1
        self.check_parent()
        import re
        deadline = time.monotonic() + 30
        tokens = 0
        while time.monotonic() < deadline:
            with store_log.open('rb') as handle:
                handle.seek(offset)
                tokens = sum(int(s) for s in re.findall(rb'Stored (\d+) tokens', handle.read()))
            if tokens >= PREFIX:
                break
            time.sleep(0.2)
        if tokens != PREFIX:
            raise RuntimeError(f'{label}: expected {PREFIX} newly stored tokens, got {tokens}')
        self.record('parent_store', label=label, stored_tokens=tokens, **result)
        return result

    def command(self, mode, port):
        command = [str(ENV / 'bin/vllm'), 'serve', MODEL, '--served-model-name', MODEL_ID,
            '--host', '127.0.0.1' if port == 18001 else '0.0.0.0', '--port', str(port),
            '--dtype', 'bfloat16', '--max-model-len', '16384', '--gpu-memory-utilization', '0.88',
            '--max-num-batched-tokens', '1024']
        command += ['--no-enable-prefix-caching'] if mode == 'recompute' else [
            '--enable-prefix-caching', '--kv-transfer-config', json.dumps(CONNECTOR)]
        return command

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=90)
        self.proc = None
        self.mode = None
        self.wait_gpu_free()

    def wait_gpu_free(self):
        deadline = time.monotonic() + 300
        next_record = 0
        while time.monotonic() < deadline:
            apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                            '--format=csv,noheader'], text=True)
            occupants = [int(pid.strip()) for uuid, pid in csv.reader(apps.splitlines())
                         if uuid.strip() == self.gpu_uuid]
            memory = subprocess.check_output(['nvidia-smi', '-i', self.gpu_uuid,
                '--query-gpu=memory.free,memory.total', '--format=csv,noheader,nounits'], text=True)
            free, total = map(float, next(csv.reader(memory.splitlines())))
            if time.monotonic() >= next_record:
                self.record('gpu_free_wait', occupants=occupants, free_mib=free,
                            required_mib=total * 0.88 + 128)
                next_record = time.monotonic() + 10
            if all(pid == self.args.lmcache_pid for pid in occupants):
                # IPC cleanup can lag process exit; match the worker's 0.88 reservation.
                if free >= total * 0.88 + 128:
                    return
            time.sleep(2)
        raise RuntimeError('GPU1 occupied or insufficient free memory for 0.88 reservation')

    def start(self, mode, restore=False):
        if not restore and self.mode == mode:
            return
        self.stop()
        if not restore:
            self.check_parent()
        self.restart += 1
        port = 8001 if restore else 18001
        log_path = self.output / f'worker_{self.restart:02d}_{mode}_{port}.log'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', PATH=str(ENV / 'bin') + ':' + os.environ['PATH'])
        for key in ['LMCACHE_CONFIG_FILE', 'VLLM_KV_TRANSFER_CONFIG', 'VLLM_KV_CONNECTOR', 'VLLM_API_KEY']:
            env.pop(key, None)
        if not restore:
            env['VLLM_API_KEY'] = self.token
        with log_path.open('x') as handle:
            self.proc = subprocess.Popen(self.command(mode, port), env=env, stdout=handle,
                                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        self.record('worker_start', mode=mode, port=port, pid=self.proc.pid, log=str(log_path))
        client = Client(8001) if restore else self.child
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f'Worker exited; inspect {log_path}')
            try:
                client.get('/health')
                models = json.loads(client.get('/v1/models'))
                if MODEL_ID not in [m['id'] for m in models['data']]:
                    raise RuntimeError('Unexpected served model')
                values = client.metrics()
                expected = 'False' if mode == 'recompute' else 'True'
                if metric(values, 'vllm:cache_config_info', enable_prefix_caching=expected) != 1:
                    raise RuntimeError('Prefix-cache runtime configuration mismatch')
                self.mode = mode
                self.record('worker_ready', mode=mode, port=port, pid=self.proc.pid)
                return
            except (OSError, ValueError):
                time.sleep(3)
        raise RuntimeError(f'Worker readiness timeout; inspect {log_path}')

    def measure(self, value, parent_result, mode):
        if self.proc.poll() is not None:
            raise RuntimeError('Owned GPU1 worker exited before measurement')
        parent_before = self.check_parent()
        before = self.child.metrics()
        result = self.child.complete(value)
        after = self.child.metrics()
        parent_after = self.check_parent()
        if self.proc.poll() is not None:
            raise RuntimeError('Owned GPU1 worker exited during measurement')
        result.update(audit(before, after, mode, result, parent_result['output_sha256'], parent_before, parent_after))
        result['worker_pid'] = self.proc.pid
        return result

    def run(self):
        lock = (ROOT / 'artifacts/gate2_exclusive.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original_cmd = Path(f'/proc/{self.args.worker_pid}/cmdline').read_bytes().split(b'\0')
        if b'--port' not in original_cmd or original_cmd[original_cmd.index(b'--port') + 1] != b'8001':
            raise RuntimeError('Original worker PID does not own port 8001')
        if b'--kv-transfer-config' in original_cmd and b'--enable-prefix-caching' in original_cmd:
            self.original_mode = 'retrieve'
        elif b'--kv-transfer-config' not in original_cmd and b'--no-enable-prefix-caching' in original_cmd:
            self.original_mode = 'recompute'
        else:
            raise RuntimeError('Original worker has an unexpected configuration')
        server_cmd = Path(f'/proc/{self.args.lmcache_pid}/cmdline').read_bytes().split(b'\0')
        if b'server' not in server_cmd or b'5555' not in server_cmd or not any(b'lmcache' in s for s in server_cmd):
            raise RuntimeError('Unexpected LMCache server PID')
        original_metrics = Client(8001).metrics()
        if metric(original_metrics, 'vllm:num_requests_running') != 0 or metric(original_metrics, 'vllm:num_requests_waiting') != 0:
            raise RuntimeError('Original GPU1 worker is busy')
        self.check_parent()
        cfg = json.loads((Path(MODEL) / 'config.json').read_text())
        vocab = cfg.get('vocab_size') or cfg['text_config']['vocab_size']
        self.record('protocol', pairs=10, prefix_tokens=PREFIX, suffix_tokens=SUFFIX, max_tokens=OUTPUT,
            temperature=0, chunk_size=528, warmups_per_arm=3, base_seed=self.args.base_seed,
            orders='odd=recompute_first,even=retrieve_first', private_port=18001,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            original_worker_pid=self.args.worker_pid)
        self.restore_needed = True
        status = 'failed'
        try:
            os.kill(self.args.worker_pid, signal.SIGTERM)
            self.wait_gpu_free()
            for pair in range(1, 11):
                seed = self.args.base_seed + pair * 10
                value = prompt(seed, vocab)
                warmups = [prompt(seed + n, vocab) for n in range(1, 4)]
                parents = [self.prepare(v, f'pair{pair}_warmup{n}') for n, v in enumerate(warmups, 1)]
                reference = self.prepare(value, f'pair{pair}_measured')
                order = ['recompute', 'retrieve'] if pair % 2 else ['retrieve', 'recompute']
                for mode in order:
                    self.start(mode)
                    for n, (v, ref) in enumerate(zip(warmups, parents), 1):
                        result = self.measure(v, ref, mode)
                        self.record('warmup', pair=pair, mode=mode, index=n, **result)
                        if not all(ok for k, ok in result['checks'].items() if k != 'output_matches'):
                            raise RuntimeError(f'Invalid warmup path/isolation: pair={pair}, mode={mode}')
                    time.sleep(1)
                    result = self.measure(value, reference, mode)
                    result.update(pair=pair, mode=mode, order=order[0] + '_first', seed=seed,
                        prompt_sha256=hashlib.sha256(json.dumps(value).encode()).hexdigest())
                    self.rows.append(result)
                    self.record('measured', **result)
                    (self.output / 'summary.json').write_text(json.dumps(summarize(self.rows), indent=2))
                    if not all(ok for k, ok in result['checks'].items() if k != 'output_matches'):
                        raise RuntimeError(f'Invalid measurement path/isolation: pair={pair}, mode={mode}')
                self.record('pair_complete', pair=pair)
            status = 'complete'
        except Exception as error:
            self.record('failure', error=repr(error))
            raise
        finally:
            self.record('restoring_public_worker')
            # Restoration must still run if unexpected parent traffic stopped collection.
            self.parent_count = None
            self.start(self.original_mode, restore=True)
            summary = summarize(self.rows)
            summary.update(status=status, restored_worker_pid=self.proc.pid, restored_port=8001,
                           restored_mode=self.original_mode)
            (self.output / 'summary.json').write_text(json.dumps(summary, indent=2))
            self.record('finished', status=status, n_pairs=summary['n_pairs'], n_valid=summary['n_valid'])
            self.log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--worker-pid', type=int, required=True)
    parser.add_argument('--lmcache-pid', type=int, required=True)
    parser.add_argument('--base-seed', type=int, default=2609091600)
    parser.add_argument('--lmcache-log', default=str(ROOT / 'logs/lmcache_server_gate1.log'))
    Suite(parser.parse_args()).run()
