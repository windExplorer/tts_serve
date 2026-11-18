"""验证极速克隆开关：两个后端分别测 fast_clone 开/关。"""
from __future__ import annotations

import httpx

from _base import BASE

client = httpx.Client(timeout=240.0, trust_env=False)
TEXT = "你好，这是一次极速克隆的测试。今天天气不错，我们出去走走吧。"
REF_TEXT = (
    "今天的晚霞超好看的！我买了你最喜欢的焦糖奶茶，"
    "我们一起去江边走走好不好？就当是放松一下啦。"
)


def synth(backend: str, fast: bool) -> tuple[int, int, str]:
    status = client.get(BASE + "/api/gateway_status").json()
    if status["backend"] != backend:
        r = client.post(BASE + "/api/switch_backend", json={"backend": backend})
        assert r.status_code == 200, r.text
    r = client.post(
        BASE + "/inference_zero_shot",
        data={
            "tts_text": TEXT,
            "prompt_wav_path": "萌1.wav",
            "prompt_text": "" if fast else REF_TEXT,
            "fast_clone": "true" if fast else "false",
        },
    )
    return r.status_code, len(r.content), r.headers.get("X-Generation-Time", "n/a")


for backend in ("qwen3", "cosyvoice3"):
    for fast in (True, False):
        code, size, seconds = synth(backend, fast)
        mode = "极速" if fast else "普通"
        ok = code == 200 and size > 1000
        print(("PASS " if ok else "FAIL ") + f"{backend} {mode}: HTTP {code}, {size}B, {seconds}s", flush=True)
        assert ok, f"{backend} {mode} 失败"

# 切回默认
client.post(BASE + "/api/switch_backend", json={"backend": "qwen3"})
print("FAST_CLONE_VERIFY_OK")
