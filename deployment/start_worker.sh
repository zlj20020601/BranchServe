#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:?GPU id required}"
PORT="${2:?port required}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
VLLM_ENV="${VLLM_ENV:?VLLM_ENV is required}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

export PATH="$VLLM_ENV/bin:$PATH"
exec env CUDA_VISIBLE_DEVICES="$GPU_ID" "$VLLM_ENV/bin/vllm" serve "$MODEL_PATH" \
  --served-model-name "${SERVED_MODEL_NAME:-qwen3.5-4b}" \
  --host "${HOST:-0.0.0.0}" \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enable-prefix-caching
