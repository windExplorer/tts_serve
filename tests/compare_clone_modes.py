"""对比 Qwen3 的 ICL 模式 vs x-vector-only（极速克隆）模式的耗时。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qwen_tts import Qwen3TTSModel

MODEL_DIR = ROOT / "models" / "Qwen3-TTS-12Hz-0.6B-Base"
REF = str(ROOT / "voices" / "萌1.wav")
REF_TEXT = (
    "今天的晚霞超好看的！我买了你最喜欢的焦糖奶茶，"
    "我们一起去江边走走好不好？就当是放松一下啦。"
)
TEXT = "你好，这是一次测试。今天天气不错，我们出去走走吧。"

model = Qwen3TTSModel.from_pretrained(
    str(MODEL_DIR), device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa"
)

results = {}
for mode in ("ICL(默认)", "x-vector-only(极速)"):
    xvec = mode.startswith("x-vector")
    # 预热一次排除首次开销
    model.generate_voice_clone(
        text="预热。", language="Chinese", ref_audio=REF,
        ref_text=None if xvec else REF_TEXT, x_vector_only_mode=xvec,
    )
    torch.cuda.synchronize()

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        wavs, sr = model.generate_voice_clone(
            text=TEXT, language="Chinese", ref_audio=REF,
            ref_text=None if xvec else REF_TEXT, x_vector_only_mode=xvec,
        )
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    results[mode] = avg
    print(f"{mode}: 平均 {avg:.2f}s  三次 {[round(t,2) for t in times]}  时长 {len(wavs[0])/sr:.2f}s sr={sr}", flush=True)

print("\n结论：", flush=True)
for k, v in results.items():
    print(f"  {k}: {v:.2f}s", flush=True)
print("QWEN_MODE_COMPARE_OK")
