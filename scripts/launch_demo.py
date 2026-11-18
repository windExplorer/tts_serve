"""后台启动 qwen-tts 自带 Gradio demo，并等待其就绪。"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / ".venv-qwen3" / "Scripts" / "qwen-tts-demo.exe"
MODEL = ROOT / "models" / "Qwen3-TTS-12Hz-0.6B-Base"
LOG = ROOT / "logs" / "demo.log"
LOG.parent.mkdir(exist_ok=True)
URL = "http://127.0.0.1:8000"

log = open(LOG, "w", encoding="utf-8")
flags = 0
if sys.platform == "win32":
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

process = subprocess.Popen(
    [
        str(EXE),
        str(MODEL),
        "--device",
        "cuda:0",
        "--dtype",
        "bfloat16",
        "--no-flash-attn",
        "--ip",
        "127.0.0.1",
        "--port",
        "8000",
    ],
    cwd=str(ROOT),
    stdout=log,
    stderr=subprocess.STDOUT,
    creationflags=flags,
)
print("demo pid:", process.pid, flush=True)

for _ in range(120):
    try:
        response = httpx.get(URL, timeout=3.0)
        if response.status_code == 200:
            print("DEMO_READY", URL, flush=True)
            break
    except Exception:
        pass
    time.sleep(1)
else:
    print("DEMO_TIMEOUT, check demo.log", flush=True)
    sys.exit(1)
