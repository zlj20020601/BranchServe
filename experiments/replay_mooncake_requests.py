"""Replay reconstructed Mooncake requests against one worker for protocol smoke."""
import argparse, concurrent.futures, json, time, urllib.request
from pathlib import Path

MODEL='qwen3.5-4b'

def request(port, row):
    requested_output=row['output_length']
    body=dict(model=MODEL, prompt=row['prompt_ids'], temperature=0, max_tokens=max(1, requested_output),
              cache_salt='mc-replay-' + str(row['hash_ids'][:12]))
    req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/completions', data=json.dumps(body).encode(),
        headers={'Content-Type':'application/json'})
    start=time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as response: data=json.load(response)
    return dict(trace_index=row['trace_index'], requested_output_tokens=requested_output, elapsed_s=time.perf_counter()-start,
                prompt_tokens=data['usage']['prompt_tokens'], completion_tokens=data['usage']['completion_tokens'],
                finish_reason=data['choices'][0]['finish_reason'], zero_output_adaptation=requested_output == 0)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--port',type=int,default=8000); ap.add_argument('--count',type=int,default=20); ap.add_argument('--scale',type=float,default=0); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    rows=[json.loads(x) for x in Path('artifacts/mooncake_frozen_20260910/smoke_replay.jsonl').read_text().splitlines()][:args.count]
    assert rows and all(r['prompt_ids'].__len__()==r['input_length'] for r in rows)
    started=time.perf_counter(); results=[]
    for row in rows:
        if args.scale: time.sleep(max(0,row['timestamp']/1000*args.scale-(time.perf_counter()-started)))
        results.append(request(args.port,row))
    report=dict(status='complete',port=args.port,rows=len(results),results=results,
        prompt_tokens=sum(r['prompt_tokens'] for r in results),completion_tokens=sum(r['completion_tokens'] for r in results),
        truncated=sum(r['finish_reason']=='length' for r in results),semantic_quality='not measured',
        caveat='Reconstructed anonymous Mooncake tokens; system replay only.')
    args.output.write_text(json.dumps(report,indent=2)); print(json.dumps(report))
if __name__=='__main__': main()
