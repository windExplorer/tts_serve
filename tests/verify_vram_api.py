"""端到端验证显存管理 API：网关 -> worker 的卸载/加载/合成/切换。"""
from __future__ import annotations

import httpx

from _base import BASE

client = httpx.Client(timeout=240.0, trust_env=False)
PAYLOAD = {
    "tts_text": "显存管理测试，今天天气不错。",
    "prompt_wav_path": "萌1.wav",
    "prompt_text": "今天的晚霞超好看的！我买了你最喜欢的焦糖奶茶，我们一起去江边走走好不好？就当是放松一下啦。",
    "fast_clone": "true",
}


def status() -> dict:
    return client.get(BASE + "/api/status").json()


print("1) 初始状态:", status(), flush=True)
assert status()["device"] == "cuda"

print("2) 手动卸载 POST /api/vram/unload...", flush=True)
r = client.post(BASE + "/api/vram/unload")
print("   ", r.status_code, r.json(), flush=True)
assert r.status_code == 200
free_after_unload = r.json()["free_vram_mb"]

print("3) 卸载后合成（应自动恢复 GPU）...", flush=True)
r = client.post(BASE + "/inference_zero_shot", data=PAYLOAD)
print("   HTTP", r.status_code, len(r.content), "字节", flush=True)
assert r.status_code == 200 and len(r.content) > 1000
s = status()
print("   合成后 device:", s["device"], "free:", s["free_vram_mb"], flush=True)
assert s["device"] == "cuda"

print("4) 切到 cosyvoice3...", flush=True)
r = client.post(BASE + "/api/switch_backend", json={"backend": "cosyvoice3"})
print("   ", r.status_code, r.json().get("backend"), flush=True)
assert r.status_code == 200
s = status()
print("   cosy status device:", s["device"], "free:", s["free_vram_mb"], flush=True)

print("5) cosyvoice3 手动卸载...", flush=True)
r = client.post(BASE + "/api/vram/unload")
print("   ", r.status_code, r.json(), flush=True)
assert r.status_code == 200
assert r.json()["free_vram_mb"] > s["free_vram_mb"]  # 显存确实释放了

print("6) 切回 qwen3...", flush=True)
r = client.post(BASE + "/api/switch_backend", json={"backend": "qwen3"})
assert r.status_code == 200
print("   ", r.json().get("backend"), flush=True)

print("VRAM_API_OK")
