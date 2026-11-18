"""后台启动统一网关（端口读 .env），自动拉起默认后端 worker。"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import load_settings

PY = ROOT / ".venv-qwen3" / "Scripts" / "python.exe"
LOG = ROOT / "logs" / "gateway.log"
LOG.parent.mkdir(exist_ok=True)

settings = load_settings()
URL = f"http://{settings.host}:{settings.port}"

env = os.environ.copy()
env["MODEL_BACKEND"] = os.environ.get("MODEL_BACKEND", settings.model_backend)

flags = 0
if sys.platform == "win32":
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

log = open(LOG, "w", encoding="utf-8")
process = subprocess.Popen(
    [str(PY), "-m", "app.serve"],
    cwd=str(ROOT),
    stdout=log,
    stderr=subprocess.STDOUT,
    creationflags=flags,
    env=env,
)
print(f"gateway pid: {process.pid} port: {settings.port}", flush=True)

for _ in range(180):
    try:
        response = httpx.get(URL + "/api/gateway_status", timeout=3.0, trust_env=False)
        if response.status_code == 200 and response.json().get("backend"):
            print("GATEWAY_READY", URL, response.text.strip(), flush=True)
            break
    except Exception:
        pass
    time.sleep(1)
else:
    print("GATEWAY_TIMEOUT, check logs/gateway.log", flush=True)
    sys.exit(1)
