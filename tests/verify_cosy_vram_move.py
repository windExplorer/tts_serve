"""验证 CosyVoice3 权重 GPU<->CPU 搬运（用 CosyVoice 的 venv 跑）。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backends.cosyvoice3 import CosyVoice3Backend

REF = str(ROOT / "voices" / "萌1.wav")
COSY_ROOT = Path(r"D:\AI\CosyVocie\CosyVoice")


def free_mb() -> float:
    return torch.cuda.mem_get_info()[0] / 1024 / 1024


print("1) 加载 CosyVoice3...", flush=True)
backend = CosyVoice3Backend(root=COSY_ROOT, model_dir="pretrained_models/Fun-CosyVoice3-0.5B-2512")
backend.load()
print(f"   device={backend.device} free={free_mb():.0f}MB", flush=True)
assert backend.device == "cuda"

print("2) to_cpu 权重搬回 CPU...", flush=True)
t0 = time.perf_counter()
backend.to_cpu()
print(f"   device={backend.device} 耗时{time.perf_counter()-t0:.1f}s free={free_mb():.0f}MB", flush=True)
assert backend.device == "cpu"

print("3) CPU 状态下合成...", flush=True)
t0 = time.perf_counter()
pcm, sr = backend.synthesize("你好，这是CPU推理测试。", REF, "", "Chinese", fast_clone=True)
print(f"   OK {len(pcm)}字节 sr={sr} 耗时{time.perf_counter()-t0:.1f}s", flush=True)
assert len(pcm) > 1000

print("4) to_cuda 搬回 GPU...", flush=True)
t0 = time.perf_counter()
backend.to_cuda()
print(f"   device={backend.device} 耗时{time.perf_counter()-t0:.1f}s free={free_mb():.0f}MB", flush=True)
assert backend.device == "cuda"

backend.unload()
print("COSY_VRAM_MOVE_OK")
