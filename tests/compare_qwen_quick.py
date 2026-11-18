"""Qwen3 快速对照：原生 vs 我们后端，各 1 次，带逐步进度。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

print("[1/5] import...", flush=True)
from backends.qwen3 import Qwen3Backend
from qwen_tts import Qwen3TTSModel

MODEL_DIR = ROOT / "models" / "Qwen3-TTS-12Hz-0.6B-Base"
REF = str(ROOT / "voices" / "萌1.wav")
TEXT = "你好，这是一次测试。今天天气不错，我们出去走走吧。"
REF_TEXT = (
    "今天的晚霞超好看的！我买了你最喜欢的焦糖奶茶，"
    "我们一起去江边走走好不好？就当是放松一下啦。"
)

print("[2/5] load model (约10秒)...", flush=True)
t0 = time.perf_counter()
model = Qwen3TTSModel.from_pretrained(
    str(MODEL_DIR), device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa"
)
print(f"[2/5] loaded in {time.perf_counter()-t0:.1f}s", flush=True)

backend = Qwen3Backend(model_dir=MODEL_DIR)
backend._model = model
backend._sr = 24000

print("[3/5] native generate (ICL)...", flush=True)
t0 = time.perf_counter()
wavs, sr = model.generate_voice_clone(
    text=TEXT, language="Chinese", ref_audio=REF, ref_text=REF_TEXT,
    x_vector_only_mode=False,
)
native_wav = wavs[0]
print(f"[3/5] native done {time.perf_counter()-t0:.1f}s audio={len(native_wav)/sr:.2f}s", flush=True)

print("[4/5] ours synthesize (ICL)...", flush=True)
t0 = time.perf_counter()
pcm, sr2 = backend.synthesize(TEXT, REF, REF_TEXT, "Chinese", fast_clone=False)
ours_wav = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32767.0
print(f"[4/5] ours done {time.perf_counter()-t0:.1f}s audio={len(ours_wav)/sr2:.2f}s", flush=True)

print("[5/5] compare...", flush=True)
same = np.array_equal(native_wav, ours_wav)
diff = float(np.max(np.abs(native_wav - ours_wav))) if len(native_wav) == len(ours_wav) else -1
print(f"  shapes: native={native_wav.shape} ours={ours_wav.shape}", flush=True)
print(f"  sample-identical={same}  max_diff={diff:.8f}  sr={sr}/{sr2}", flush=True)
sf.write(str(ROOT / "data" / "compare_qwen_native.wav"), native_wav, sr)
sf.write(str(ROOT / "data" / "compare_qwen_ours.wav"), ours_wav, sr2)
print("QWEN_QUICK_DONE")
