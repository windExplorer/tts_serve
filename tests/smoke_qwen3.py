"""Qwen3-TTS 裸模型冒烟测试：不经过 HTTP 服务，直接生成一段音频。"""
from __future__ import annotations

import sys
from pathlib import Path

import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

from qwen_tts import Qwen3TTSModel

REF_WAV = ROOT / "voices" / "萌1.wav"
REF_TEXT = (
    "今天的晚霞超好看的！我买了你最喜欢的焦糖奶茶，"
    "我们一起去江边走走好不好？就当是放松一下啦。"
)
TEXT = "你好，这是一次测试。今天天气不错，我们出去走走吧。"

model = Qwen3TTSModel.from_pretrained(
    str(ROOT / "models" / "Qwen3-TTS-12Hz-0.6B-Base"),
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

wavs, sr = model.generate_voice_clone(
    text=TEXT,
    language="Chinese",
    ref_audio=str(REF_WAV),
    ref_text=REF_TEXT,
)

out = DATA / "test_qwen3.wav"
sf.write(str(out), wavs[0], sr)
print(f"OK sample_rate={sr} output={out}")
