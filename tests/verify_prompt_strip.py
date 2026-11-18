"""验证 Qwen3 对 CosyVoice 副文本标记的清洗：带标记的文本不应念出指令前缀。"""
from __future__ import annotations

import httpx
import numpy as np
import soundfile as sf

from _base import BASE

client = httpx.Client(timeout=240.0, trust_env=False)
REF = "萌1.wav"

# 确保当前是 qwen3
st = client.get(BASE + "/api/gateway_status").json()
if st["backend"] != "qwen3":
    r = client.post(BASE + "/api/switch_backend", json={"backend": "qwen3"})
    assert r.status_code == 200, r.text

# 带 CosyVoice 副文本前缀的文本
TAGGED = (
    "You are a helpful assistant. 请非常开心地说一句话。"
    "<|endofprompt|>今天天气真不错，我们一起去公园散步吧。"
)
# 期望只念这部分
CLEAN = "今天天气真不错，我们一起去公园散步吧。"

for label, text in (("带标记", TAGGED), ("纯正文", CLEAN)):
    r = client.post(BASE + "/inference_zero_shot", data={
        "tts_text": text,
        "prompt_wav_path": REF,
        "fast_clone": "true",
    })
    assert r.status_code == 200, (label, r.status_code, r.text[:200])
    pcm = np.frombuffer(r.content, dtype="<i2")
    out = f"data/verify_strip_{label}.wav"
    sf.write(out, pcm.astype(np.float32) / 32767.0, 24000)
    print(f"{label}: {len(r.content)}字节 音频{len(pcm)/24000:.2f}s -> {out}", flush=True)

# 用 ASR 复核带标记的那段，确认没念出 "helpful assistant"
with open("data/verify_strip_带标记.wav", "rb") as f:
    r = client.post(BASE + "/api/asr", files={"file": ("a.wav", f, "audio/wav")}, data={"language": "en"})
assert r.status_code == 200, r.text[:200]
text = r.json()["text"]
print("ASR 识别:", repr(text), flush=True)
lower = text.lower()
assert "helpful" not in lower and "assistant" not in lower, "指令前缀仍被念出！"
print("PROMPT_STRIP_OK")
