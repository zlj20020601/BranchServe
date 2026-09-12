#!/bin/bash
# dynamic p6/p8 重跑:每档前重启 lmcache server 保证 store 健康
set -e
cd /root/autodl-tmp/branchserve
VENV=/root/autodl-tmp/conda_envs/branchserve
export PATH=$VENV/bin:$PATH
for p in 6 8; do
  echo "=== restart server before dynamic p$p $(date +%H:%M:%S) ==="
  pkill -f "b[i]n/lmcache server" || true
  sleep 6
  nohup $VENV/bin/lmcache server --host 127.0.0.1 --port 5555 --chunk-size 528 \
    --separate-object-groups --l1-size-gb 100 --eviction-policy LRU \
    > logs/lmcache_server_dyn_20260912.log 2>&1 &
  sleep 18
  timeout 3 bash -c 'echo > /dev/tcp/127.0.0.1/5555' && echo SERVER_UP || echo SERVER_DOWN
  curl -s -o /dev/null -w "H0=%{http_code} " --max-time 3 http://127.0.0.1:8000/health || true
  curl -s -o /dev/null -w "H1=%{http_code}\n" --max-time 3 http://127.0.0.1:8001/health || true
  $VENV/bin/python multiround_pressure.py --strategy dynamic --pressure $p \
    --label up5-dynamic-p$p --server-log logs/lmcache_server_dyn_20260912.log \
    --out artifacts/unified_v5_dynamic_p${p}_20260912.json 2>&1 | tail -6
done
echo DYN_RERUN_DONE
