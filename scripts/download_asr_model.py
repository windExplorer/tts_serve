"""从 hf-mirror 直接下载 faster-whisper-base 模型文件到本地。"""
from __future__ import annotations

from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "models" / "faster-whisper-base"
BASE_URL = "https://hf-mirror.com/Systran/faster-whisper-base/resolve/main"
FILES = ["config.json", "tokenizer.json", "vocabulary.txt", "model.bin"]

TARGET.mkdir(parents=True, exist_ok=True)
with httpx.Client(timeout=300.0, follow_redirects=True, trust_env=False) as client:
    for name in FILES:
        destination = TARGET / name
        if destination.exists() and destination.stat().st_size > 0:
            print("skip", name, destination.stat().st_size, flush=True)
            continue
        print("downloading", name, flush=True)
        with client.stream("GET", f"{BASE_URL}/{name}") as response:
            response.raise_for_status()
            with destination.open("wb") as handle:
                for chunk in response.iter_bytes(1024 * 1024):
                    handle.write(chunk)
        print("done", name, destination.stat().st_size, flush=True)

print("ASR_MODEL_READY", TARGET)
