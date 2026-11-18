"""不加载模型的静态验证：路由契约、配置与音色解析。"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings, load_settings
from app.main import app, resolve_voice

EXPECTED = {
    ("/", "GET"),
    ("/voices", "GET"),
    ("/inference_zero_shot", "POST"),
    ("/inference_instruct2", "POST"),
}

actual = set()
for route in app.routes:
    for method in getattr(route, "methods", set()):
        actual.add((route.path, method))

missing = EXPECTED - actual
assert not missing, f"缺少路由: {missing}"
print("路由契约 OK:", sorted(EXPECTED))

settings = load_settings()
assert settings.model_backend == "qwen3"
assert settings.qwen3_model_dir.name == "Qwen3-TTS-12Hz-0.6B-Base"
print("配置加载 OK: backend=%s voices_dir=%s" % (settings.model_backend, settings.voices_dir))

voices = {
    "音色A": {"prompt_wav": "a.wav", "prompt_text": "这是A的原文"},
    "音色B": {"prompt_wav": "b.wav", "prompt_text": "这是B的原文"},
}

ref, text = resolve_voice(settings, voices, "音色A", "", None)
assert ref.endswith("a.wav") and text == "这是A的原文", (ref, text)

ref, text = resolve_voice(settings, voices, "b.wav", "", None)
assert ref.endswith("b.wav") and text == "这是B的原文", (ref, text)

ref, text = resolve_voice(settings, voices, "音色A", "", "C:/tmp/upload.wav")
assert ref == "C:/tmp/upload.wav" and text == "这是A的原文", (ref, text)

ref, text = resolve_voice(settings, voices, "音色A", "自定义原文", None)
assert text == "自定义原文", (ref, text)

default_settings = Settings(default_voice="音色B", voices_dir=settings.voices_dir)
ref, text = resolve_voice(default_settings, voices, "", "", None)
assert ref.endswith("b.wav") and text == "这是B的原文", (ref, text)

print("音色解析 OK")

try:
    from qwen_tts import Qwen3TTSModel

    print("Qwen3TTSModel.from_pretrained:", inspect.signature(Qwen3TTSModel.from_pretrained))
    print("generate_voice_clone:", inspect.signature(Qwen3TTSModel.generate_voice_clone))
except Exception as exc:
    print("qwen_tts 签名检查跳过:", exc)

print("ALL_OK")
