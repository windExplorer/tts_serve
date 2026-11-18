"""验证语音转文字接口。"""
from __future__ import annotations

from pathlib import Path

import httpx

from _base import BASE

ROOT = Path(__file__).resolve().parent.parent

with (ROOT / "voices" / "萌1.wav").open("rb") as handle:
    response = httpx.post(
        BASE + "/api/asr",
        data={"language": "zh"},
        files={"file": ("ref.wav", handle, "audio/wav")},
        timeout=600.0,
        trust_env=False,
    )

print("status:", response.status_code)
print("body:", response.text[:500])
assert response.status_code == 200, response.text
text = response.json()["text"]
assert len(text) > 0, "识别结果为空"
print("识别文本:", text)
print("ASR_OK")
