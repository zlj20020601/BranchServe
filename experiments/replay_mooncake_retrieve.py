"""Replay reconstructed Mooncake rows through a connector Parent/Retrieve pair.

The trace has no original text, so this measures cache transfer and latency only.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

from prometheus_client.parser import text_string_to_metric_families

MODEL = "qwen3.5-4b"
SOURCES = ("local_compute", "local_cache_hit", "external_kv_transfer")


def metrics(port):
    request = urllib.request.Request(f"http://127.0.0.1:{port}/metrics")
    with urllib.request.urlopen(request, timeout=15) as response:
        text = response.read().decode()
    values = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name.endswith("_created"):
                continue
            labels = tuple(sorted(sample.labels.items()))
            values[(sample.name, labels)] = sample.value
    return values


def metric(values, name, **labels):
    return sum(value for (sample, fields), value in values.items()
               if sample == name and all(dict(fields).get(k) == v for k, v in labels.items()))


def request(port, row):
    output_tokens = row["output_length"]
    body = {
        "model": MODEL,
        "prompt": row["prompt_ids"],
        "temperature": 0,
        "max_tokens": max(1, output_tokens),
        "cache_salt": "mc-replay-" + str(row["hash_ids"][:12]),
    }
    payload = json.dumps(body).encode()
    http = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    before = metrics(port)
    started = time.perf_counter()
    with urllib.request.urlopen(http, timeout=300) as response:
        data = json.load(response)
    elapsed = time.perf_counter() - started
    after = metrics(port)
    sources = {
        source: metric(after, "vllm:prompt_tokens_by_source_total", source=source)
        - metric(before, "vllm:prompt_tokens_by_source_total", source=source)
        for source in SOURCES
    }
    return {
        "elapsed_s": elapsed,
        "prompt_tokens": data["usage"]["prompt_tokens"],
        "completion_tokens": data["usage"]["completion_tokens"],
        "finish_reason": data["choices"][0]["finish_reason"],
        "sources": sources,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-port", type=int, default=8000)
    parser.add_argument("--retrieve-port", type=int, default=8001)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path("artifacts/mooncake_frozen_20260910")
    rows = [json.loads(line) for line in (root / "smoke_replay.jsonl").read_text().splitlines()]
    rows = rows[: args.count]
    if not rows:
        raise RuntimeError("no replay rows")
    results = []
    started = time.perf_counter()
    for row in rows:
        parent = request(args.parent_port, row)
        # The connector's store is asynchronous; a short gap avoids racing the
        # copy server while retaining the trace's sequential replay semantics.
        time.sleep(0.25)
        retrieve = request(args.retrieve_port, row)
        results.append({
            "trace_index": row["trace_index"],
            "input_length": row["input_length"],
            "output_length": row["output_length"],
            "parent": parent,
            "retrieve": retrieve,
            "retrieve_external_transfer": retrieve["sources"]["external_kv_transfer"],
        })
        print(json.dumps(results[-1]), flush=True)
    report = {
        "status": "complete",
        "rows": len(results),
        "wall_time_s": time.perf_counter() - started,
        "parent_prompt_tokens": sum(r["parent"]["prompt_tokens"] for r in results),
        "retrieve_prompt_tokens": sum(r["retrieve"]["prompt_tokens"] for r in results),
        "retrieve_external_transfer_tokens": sum(r["retrieve_external_transfer"] for r in results),
        "semantic_quality": "not measured",
        "caveat": "Reconstructed anonymous Mooncake tokens; Parent/Retrieve system replay only.",
        "results": results,
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "results"}))


if __name__ == "__main__":
    main()
