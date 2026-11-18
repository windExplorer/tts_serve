"""验证 Qwen3 音频差异来自采样随机性：同一调用连跑两次。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qwen_tts import Qwen3TTSModel

MODEL_DIR = ROOT / "models" / "Qwen3-TTS-12Hz-0.6B-Base"
REF = str(ROOT / "voices" / "萌1.wav")
TEXT = "你好，这是一次测试。今天天气不错，我们出去走走吧。"
REF_TEXT = (
    "今天的晚霞超好看的！我买了你最喜欢的焦糖奶茶，"
    "我们一起去江边走走好不好？就当是放松一下啦。"
)

model = Qwen3TTSModel.from_pretrained(
    str(MODEL_DIR), device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa"
)

print("同一 API 连续两次调用（相同参数）:", flush=True)
wavs = []
for i in range(2):
    t0 = time.perf_counter()
    w, sr = model.generate_voice_clone(
        text=TEXT, language="Chinese", ref_audio=REF, ref_text=REF_TEXT,
        x_vector_only_mode=False,
    )
    wavs.append(w[0])
    print(f"  第{i+1}次: {time.perf_counter()-t0:.1f}s 音频{len(w[0])/sr:.2f}s", flush=True)

same = np.array_equal(wavs[0], wavs[1])
print(f"  两次逐样本相同={same}  shape={wavs[0].shape} vs {wavs[1].shape}", flush=True)
print("结论：若两次就不同，说明是模型采样随机性，与原生/我们封装无关", flush=True)
print("RANDOMNESS_DONE")
