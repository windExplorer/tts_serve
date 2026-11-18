"""对照实验：CosyVoice3 原生调用方式 vs 我们项目的方式，找速度差异根因。

直接用 CosyVoice 的 venv 跑（通过 subprocess），在同一进程加载一次模型，
分别测：
  A. 原生 WebUI 的 3s极速复刻 = inference_zero_shot(tts, prompt_text, wav, stream=False)
  B. 我们的极速模式 = inference_cross_lingual(tts, wav, stream=False)
  C. cross_lingual 变体 = inference_cross_lingual stream=True 收集全部
  D. zero_shot + stream=True 收集全部
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

COSY_ROOT = Path(r"D:\AI\CosyVocie\CosyVoice")
sys.path.insert(0, str(COSY_ROOT))
sys.path.insert(0, str(COSY_ROOT / "third_party" / "Matcha-TTS"))

from cosyvoice.cli.cosyvoice import AutoModel

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = COSY_ROOT / "pretrained_models" / "Fun-CosyVoice3-0.5B-2512"
REF = str(ROOT / "voices" / "萌1.wav")
REF_TEXT = (
    "今天的晚霞超好看的！我买了你最喜欢的焦糖奶茶，"
    "我们一起去江边走走好不好？就当是放松一下啦。"
)
TEXT = "你好，这是一次测试。今天天气不错，我们出去走走吧。"

print("加载模型...", flush=True)
model = AutoModel(model_dir=str(MODEL_DIR))
print("sr =", model.sample_rate, flush=True)


def collect(gen):
    chunks = []
    for item in gen:
        sp = item.get("tts_speech")
        if sp is not None:
            chunks.append(sp)
    return chunks


def timed(name, fn):
    fn()  # 预热
    t0 = time.perf_counter()
    chunks = fn()
    dt = time.perf_counter() - t0
    total = sum(c.shape[1] for c in chunks) / model.sample_rate
    print(f"{name}: {dt:.2f}s  音频{total:.2f}s  段数{len(chunks)}", flush=True)


timed("A 原生3s极速 zero_shot stream=False",
      lambda: collect(model.inference_zero_shot(TEXT, REF_TEXT, REF, stream=False)))
timed("B 我们极速 cross_lingual stream=False",
      lambda: collect(model.inference_cross_lingual(TEXT, REF, stream=False)))
timed("C cross_lingual stream=True",
      lambda: collect(model.inference_cross_lingual(TEXT, REF, stream=True)))
timed("D zero_shot stream=True",
      lambda: collect(model.inference_zero_shot(TEXT, REF_TEXT, REF, stream=True)))

print("COSy_COMPARE_DONE")
