#!/usr/bin/env python3
"""隔离诊断 service 遥测链:raw fetch → refresh → pressure_of。"""
import sys
sys.path.insert(0, "/root/autodl-tmp/branchserve")
import branchserve_service as S

r, w, t = S.fetch_worker_metrics("http://127.0.0.1:8000")
print("raw fetch (running,waiting,transfer):", r, w, t)
S.telemetry_refresh(ttl=0)
print("TELEMETRY after refresh:", S.TELEMETRY)
print("pressure_of(0):", S.pressure_of(0))
print("STATE workers:", S.STATE["workers"])
