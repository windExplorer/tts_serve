"""验证空闲自动卸载巡检：用短间隔配置单独启动 worker，观察 device 自动变化。"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv-qwen3" / "Scripts" / "python.exe"
URL = "http://127.0.0.1:51999"
LOG = ROOT / "logs" / "verify_idle.log"
LOG.parent.mkdir(exist_ok=True)

env = os.environ.copy()
env.update({
    "MODEL_BACKEND": "qwen3",
    "IDLE_UNLOAD_SEC": "3",     # 3秒空闲即卸载
    "VRAM_CHECK_SEC": "1",      # 每秒巡检
    "LOW_VRAM_MB": "512",       # 门槛=2048MB，正常空闲不会被误判为显存挤压
})

print("启动短间隔 worker...", flush=True)
log = open(LOG, "w", encoding="utf-8")
proc = subprocess.Popen(
    [str(PY), "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "51999"],
    cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env,
)
try:
    for _ in range(120):
        try:
            r = httpx.get(URL + "/api/status", timeout=3, trust_env=False)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    s = httpx.get(URL + "/api/status", timeout=5, trust_env=False).json()
    print(f"  启动后 device={s['device']} idle_unload_sec={s['idle_unload_sec']}", flush=True)
    assert s["device"] == "cuda"

    print("等待 6 秒（空闲应自动卸载）...", flush=True)
    time.sleep(6)
    s = httpx.get(URL + "/api/status", timeout=5, trust_env=False).json()
    print(f"  6秒后 device={s['device']} free={s['free_vram_mb']}MB", flush=True)
    assert s["device"] == "none", f"空闲未自动卸载，device={s['device']}"

    # 触发一次合成，应自动恢复
    r = httpx.post(URL + "/inference_zero_shot", data={
        "tts_text": "测试自动恢复。", "prompt_wav_path": "萌1.wav", "fast_clone": "true",
    }, timeout=180, trust_env=False)
    print(f"  合成 HTTP {r.status_code} {len(r.content)}字节", flush=True)
    assert r.status_code == 200
    s = httpx.get(URL + "/api/status", timeout=5, trust_env=False).json()
    print(f"  合成后 device={s['device']}", flush=True)
    assert s["device"] == "cuda"
finally:
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except Exception:
        proc.kill()
    log.close()

print("IDLE_UNLOAD_OK")
