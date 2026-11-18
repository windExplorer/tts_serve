"""切换后端强制 worker 重载新代码。"""
from __future__ import annotations

import httpx

from _base import BASE

client = httpx.Client(trust_env=False, timeout=240.0)
print("切到 cosyvoice3...", flush=True)
r = client.post(BASE + "/api/switch_backend", json={"backend": "cosyvoice3"})
print(" ", r.status_code, flush=True)
print("切回 qwen3...", flush=True)
r = client.post(BASE + "/api/switch_backend", json={"backend": "qwen3"})
print(" ", r.status_code, r.json().get("backend"), flush=True)
print("RELOAD_DONE")
