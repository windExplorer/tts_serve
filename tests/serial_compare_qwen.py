"""对照 Qwen3：原生 generate_voice_clone vs 我们后端 synthesize。

同一进程只加载一次模型，串行调用（不重复加载、不占额外显存），
用与 cosyvoice 对比相同的长文本，对比耗时与音频一致性。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backends.qwen3 import Qwen3Backend
from qwen_tts import Qwen3TTSModel

MODEL_DIR = ROOT / "models" / "Qwen3-TTS-12Hz-0.6B-Base"
REF = str(ROOT / "voices" / "萌1.wav")
REF_TEXT = (
    "今天的晚霞超好看的！我买了你最喜欢的焦糖奶茶，"
    "我们一起去江边走走好不好？就当是放松一下啦。"
)
TEXT = (
    "夜幕降临时，我常在这片花园驻足。每一朵绽放的花都承载着逝者的故事，"
    "像是星辰般闪烁。作为记忆的守护者，我愿静静聆听每一个离别的呢喃，"
    "将它们化作永恒的光芒。风穿过树梢，带来远方的低语，我闭上眼睛，"
    "任由这份温柔将我包围。"
)

print("加载 Qwen3 模型...", flush=True)
model = Qwen3TTSModel.from_pretrained(
    str(MODEL_DIR), device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa"
)

# 复用已加载模型，避免二次加载占显存
backend = Qwen3Backend(model_dir=MODEL_DIR)
backend._model = model
backend._sr = 24000


def native(fast: bool):
    wavs, sr = model.generate_voice_clone(
        text=TEXT, language="Chinese", ref_audio=REF,
        ref_text=None if fast else REF_TEXT, x_vector_only_mode=fast,
    )
    return wavs[0], sr


def ours(fast: bool):
    pcm, sr = backend.synthesize(TEXT, REF, REF_TEXT, "Chinese", fast_clone=fast)
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32767.0, sr


results = {}
for label, fn, fast in (
    ("原生ICL", native, False),
    ("我们ICL", ours, False),
    ("原生极速", native, True),
    ("我们极速", ours, True),
):
    fn(fast)  # 预热
    torch.cuda.synchronize()
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        wav, sr = fn(fast)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    out = ROOT / "data" / f"compare_qwen_{label}.wav"
    sf.write(str(out), wav, sr)
    avg = sum(times) / len(times)
    results[label] = (avg, wav, sr)
    print(f"{label}: {[round(t,2) for t in times]} 平均 {avg:.2f}s "
          f"音频 {len(wav)/sr:.2f}s sr={sr}", flush=True)

# 一致性校验（同名原生/我们 应逐样本相同）
for a, b in (("原生ICL", "我们ICL"), ("原生极速", "我们极速")):
    wa, wb = results[a][1], results[b][1]
    print(f"  {a} vs {b}: 逐样本相同={np.array_equal(wa, wb)} "
          f"最大差={float(np.max(np.abs(wa-wb))):.6f}", flush=True)

print("\n========== Qwen3 汇总 ==========", flush=True)
for label, (avg, wav, sr) in results.items():
    print(f"{label:10s} 平均 {avg:6.2f}s  音频 {len(wav)/sr:5.2f}s", flush=True)
print("QWEN_COMPARE_DONE")
