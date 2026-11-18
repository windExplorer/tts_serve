"""验证 Qwen3 显存释放与恢复：load -> to_cpu(卸载) -> to_cuda(重载) -> 合成。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backends.qwen3 import Qwen3Backend

MODEL_DIR = ROOT / "models" / "Qwen3-TTS-12Hz-0.6B-Base"
REF = str(ROOT / "voices" / "萌1.wav")


def free_mb() -> float:
    return torch.cuda.mem_get_info()[0] / 1024 / 1024


print("1) 加载 Qwen3...", flush=True)
backend = Qwen3Backend(model_dir=MODEL_DIR, warmup_ref_audio=None)
backend.load()
print(f"   device={backend.device} free={free_mb():.0f}MB", flush=True)
assert backend.device == "cuda"

print("2) to_cpu 释放显存（Qwen3 = 整模型卸载）...", flush=True)
t0 = time.perf_counter()
backend.to_cpu()
print(f"   device={backend.device} model_is_none={backend._model is None} "
      f"耗时{time.perf_counter()-t0:.1f}s free={free_mb():.0f}MB", flush=True)
assert backend.device == "none" and backend._model is None

print("3) to_cuda 重新加载...", flush=True)
t0 = time.perf_counter()
backend.to_cuda()
print(f"   device={backend.device} 耗时{time.perf_counter()-t0:.1f}s free={free_mb():.0f}MB", flush=True)
assert backend.device == "cuda"

print("4) 重载后合成...", flush=True)
pcm, sr = backend.synthesize("你好，这是重载后的测试。", REF, "", "Chinese", fast_clone=True)
print(f"   OK {len(pcm)}字节 sr={sr}", flush=True)
assert len(pcm) > 1000

backend.unload()
print("VRAM_MOVE_OK")
