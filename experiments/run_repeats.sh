#!/bin/bash
# 定向补重复:pack/retrieve × p0/2/4/6/8 × rep2/rep3(recompute 跳过:全程垫底,不影响 oracle)
# retrieve 每次前重启 server(规避 write-only 残留);pack 对 server 状态免疫
set -e
cd /root/autodl-tmp/branchserve
VENV=/root/autodl-tmp/conda_envs/branchserve
export PATH=$VENV/bin:$PATH
PY=$VENV/bin/python

restart_server() {
  pkill -f "b[i]n/lmcache server" || true
  sleep 6
  nohup $VENV/bin/lmcache server --host 127.0.0.1 --port 5555 --chunk-size 528 \
    --separate-object-groups --l1-size-gb 100 --eviction-policy LRU \
    > logs/lmcache_server_repeats_20260912.log 2>&1 &
  sleep 18
  timeout 3 bash -c 'echo > /dev/tcp/127.0.0.1/5555' >/dev/null 2>&1 && echo SERVER_UP || echo SERVER_SLOW
}

for rep in 2 3; do
  for p in 0 2 4 6 8; do
    echo "=== r${rep} pack p$p $(date +%H:%M:%S) ==="
    $PY multiround_pressure.py --strategy pack --pressure $p \
      --label up7-r${rep}-pack-p$p \
      --server-log logs/lmcache_server_repeats_20260912.log \
      --out artifacts/unified_r${rep}_pack_p${p}_20260912.json 2>&1 | tail -2
  done
done

for rep in 2 3; do
  for p in 0 2 4 6 8; do
    echo "=== r${rep} retrieve p$p $(date +%H:%M:%S) ==="
    restart_server
    $PY multiround_pressure.py --strategy retrieve --pressure $p \
      --label up7-r${rep}-retrieve-p$p \
      --server-log logs/lmcache_server_repeats_20260912.log \
      --out artifacts/unified_r${rep}_retrieve_p${p}_20260912.json 2>&1 | tail -2
  done
done
echo REPEATS_DONE
