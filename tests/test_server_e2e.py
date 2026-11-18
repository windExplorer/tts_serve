"""端到端 HTTP 测试：启动服务、验证契约、合成音频、关闭服务。"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
LOGS = ROOT / "logs"
DATA = ROOT / "data"
LOGS.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

PYTHON = ROOT / ".venv-qwen3" / "Scripts" / "python.exe"
from _base import BASE, _settings

log = open(LOGS / "server.log", "w", encoding="utf-8")
server = subprocess.Popen(
    [str(PYTHON), "-m", "uvicorn", "app.main:app", "--host", _settings.host, "--port", str(_settings.port)],
    cwd=str(ROOT),
    stdout=log,
    stderr=subprocess.STDOUT,
)

try:
    client = httpx.Client(timeout=180.0)
    response = None
    for _ in range(120):
        try:
            response = client.get(BASE + "/")
            if response.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        raise RuntimeError("服务未在 120 秒内就绪，请查看 server.log")

    print("GET / ->", response.json())

    response = client.get(BASE + "/voices")
    voices = response.json()
    print("GET /voices ->", len(voices), "voices; sample keys:", list(voices)[:3])

    ref_text = voices["萌1.wav"]["prompt_text"]
    response = client.post(
        BASE + "/inference_zero_shot",
        data={
            "tts_text": "你好，这是一次测试。今天天气不错，我们出去走走吧。",
            "prompt_wav_path": "萌1.wav",
            "prompt_text": ref_text,
        },
    )
    print(
        "POST /inference_zero_shot ->",
        response.status_code,
        "bytes:",
        len(response.content),
        "content-type:",
        response.headers.get("content-type"),
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/octet-stream")

    sr = client.get(BASE + "/").json()["sample_rate"]
    pcm = np.frombuffer(response.content, dtype="<i2")
    out = DATA / "test_http.wav"
    sf.write(str(out), pcm.astype(np.float32) / 32767.0, sr)
    print("saved", out, "sr:", sr, "samples:", len(pcm))

    response = client.post(BASE + "/inference_instruct2")
    print("POST /inference_instruct2 ->", response.status_code)
    assert response.status_code == 501
    print("E2E_OK")
finally:
    server.terminate()
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server.kill()
    log.close()
