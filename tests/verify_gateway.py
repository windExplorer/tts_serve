"""验证统一网关：合成转发、切换后端、WebUI 元素。"""
from __future__ import annotations

import httpx

from _base import BASE

client = httpx.Client(timeout=240.0, trust_env=False)
DATA = {
    "tts_text": "网关转发测试。今天天气不错。",
    "prompt_wav_path": "萌1.wav",
    "prompt_text": "今天的晚霞超好看的！我买了你最喜欢的焦糖奶茶，我们一起去江边走走好不好？就当是放松一下啦。",
}

status = client.get(BASE + "/").json()
print("GET / ->", status)
assert status.get("backend") == "qwen3", status

r = client.post(BASE + "/inference_zero_shot", data=DATA)
print("synth qwen3 ->", r.status_code, len(r.content), "bytes", r.headers.get("X-Generation-Time"), "s")
assert r.status_code == 200 and len(r.content) > 1000

html = client.get(BASE + "/ui").text
for token in ("backendSelect", "switchBackendBtn", "/api/switch_backend", "api/gateway_status"):
    assert token in html, f"UI 缺少 {token}"
print("UI elements OK")

# 切换到 CosyVoice3
r = client.post(BASE + "/api/switch_backend", json={"backend": "cosyvoice3"})
print("switch ->", r.status_code, r.text[:200])
assert r.status_code == 200

status = client.get(BASE + "/api/gateway_status").json()
assert status["backend"] == "cosyvoice3" and not status["switching"], status
print("backend now:", status["backend"], "gpu:", status.get("gpu_name"))

r = client.post(BASE + "/inference_zero_shot", data=DATA)
print("synth cosyvoice3 ->", r.status_code, len(r.content), "bytes", r.headers.get("X-Generation-Time"), "s")
assert r.status_code == 200 and len(r.content) > 1000

# 切回 Qwen3
r = client.post(BASE + "/api/switch_backend", json={"backend": "qwen3"})
assert r.status_code == 200, r.text
print("switch back ->", r.json().get("backend"))

print("GATEWAY_VERIFY_OK")
