"""验证 /ui 页面可访问，且页面调用了正确的接口。"""
from __future__ import annotations

import httpx

from _base import BASE

client = httpx.Client(timeout=30.0)
response = client.get(BASE + "/ui")
assert response.status_code == 200, response.status_code
html = response.text
for token in ("TTS_SERVE WebUI", "/inference_zero_shot", "/voices", "pcmToWav"):
    assert token in html, f"页面缺少: {token}"
print("GET /ui -> 200, bytes:", len(response.content))

response = client.get(BASE + "/")
print("GET / ->", response.json())
assert response.status_code == 200

response = client.get(BASE + "/voices")
print("GET /voices ->", len(response.json()), "voices")
assert response.status_code == 200
print("UI_VERIFY_OK")
