from __future__ import annotations

import gc
from abc import ABC, abstractmethod

import numpy as np


END_OF_PROMPT = "<|endofprompt|>"


def strip_cosyvoice_prompt(text: str) -> str:
    """剥掉 CosyVoice3 的副文本指令，只保留要念的正文。

    CosyVoice3 文本格式：`<语气/方言指令><|endofprompt|>要念的正文`
    指令部分（如 "You are a helpful assistant. 请非常开心地说一句话。"）不念。
    对不认识该 token 的模型（如 Qwen3），需要提前剥掉，否则整段被念出来。
    """
    if not text:
        return text
    if END_OF_PROMPT in text:
        # 取最后一个分隔符之后的内容（正文中若真出现该 token，以最后的为准）
        return text.rsplit(END_OF_PROMPT, 1)[1].strip()
    return text


def float_to_pcm16(wav) -> bytes:
    """把归一化浮点音频转成 int16 little-endian PCM。"""
    arr = np.asarray(wav, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767.0).astype("<i2").tobytes()


class BaseBackend(ABC):
    """所有推理后端需实现的统一接口。"""

    name: str = "base"

    @abstractmethod
    def load(self) -> None:
        """加载模型；Qwen3 在此阶段做 warmup 以获取真实 sample_rate。"""

    @abstractmethod
    def synthesize(
        self,
        tts_text: str,
        ref_audio: str,
        ref_text: str,
        language: str = "Chinese",
        **gen_kwargs,
    ) -> tuple[bytes, int]:
        """返回 (int16 little-endian PCM bytes, sample_rate)。"""

    @abstractmethod
    def sample_rate(self) -> int:
        """返回实际采样率，供 GET / 使用。"""

    @abstractmethod
    def list_voices(self) -> dict:
        """返回可用音色映射。"""

    @abstractmethod
    def unload(self) -> None:
        """释放模型与显存，便于 ComfyUI 轮班。"""

    @property
    def device(self) -> str:
        """当前权重所在设备：'cuda' / 'cpu' / 'none'。默认实现供不支持搬运的后端。"""
        return getattr(self, "_device", "cuda")

    @property
    def is_loaded(self) -> bool:
        """模型是否已加载到内存（未加载时 synthesize 无法工作）。"""
        return getattr(self, "_model", None) is not None

    def to_cpu(self) -> None:
        """把权重搬到 CPU 释放显存。不支持的后端可空实现。"""

    def to_cuda(self) -> None:
        """把权重搬回 GPU。不支持的后端可空实现。"""

    def clear_cache(self) -> None:
        """清空推理缓存与 CUDA 缓存。"""
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
