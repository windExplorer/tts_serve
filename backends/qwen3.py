from __future__ import annotations

import gc
import logging
from pathlib import Path

from .base import BaseBackend, float_to_pcm16, strip_cosyvoice_prompt

logger = logging.getLogger(__name__)


class Qwen3Backend(BaseBackend):
    name = "qwen3"

    def __init__(
        self,
        model_dir: Path,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        attn: str = "sdpa",
        fallback_sample_rate: int = 24000,
        warmup_ref_audio: str | None = None,
        warmup_ref_text: str = "",
    ):
        self.model_dir = model_dir
        self._target_device = device
        self.dtype = dtype
        self.attn = attn
        self.fallback_sample_rate = fallback_sample_rate
        self.warmup_ref_audio = warmup_ref_audio
        self.warmup_ref_text = warmup_ref_text
        self._model = None
        self._sr = fallback_sample_rate
        self._device = "none"

    @property
    def device(self) -> str:
        return self._device

    def load(self) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        dtype = getattr(torch, self.dtype, torch.bfloat16)
        logger.info(
            "加载 Qwen3-TTS: %s device=%s dtype=%s attn=%s",
            self.model_dir,
            self._target_device,
            self.dtype,
            self.attn,
        )
        self._model = Qwen3TTSModel.from_pretrained(
            str(self.model_dir),
            device_map=self._target_device,
            dtype=dtype,
            attn_implementation=self.attn,
        )
        self._device = "cuda" if self._target_device.startswith("cuda") else "cpu"
        self._warmup()

    def _warmup(self) -> None:
        if not self.warmup_ref_audio or not Path(self.warmup_ref_audio).exists():
            logger.warning(
                "没有可用的参考音频，跳过 warmup，sample_rate 使用默认值 %d",
                self.fallback_sample_rate,
            )
            return
        try:
            wavs, sr = self._model.generate_voice_clone(
                text="你好。",
                language="Chinese",
                ref_audio=self.warmup_ref_audio,
                ref_text=self.warmup_ref_text or "你好。",
            )
            self._sr = int(sr)
            logger.info("Qwen3 warmup 完成，sample_rate=%d", self._sr)
        except Exception as exc:
            logger.warning(
                "Qwen3 warmup 失败（%s），sample_rate 使用默认值 %d",
                exc,
                self.fallback_sample_rate,
            )

    def synthesize(
        self,
        tts_text: str,
        ref_audio: str,
        ref_text: str,
        language: str = "Chinese",
        fast_clone: bool = True,
        **gen_kwargs,
    ) -> tuple[bytes, int]:
        if self._model is None:
            raise RuntimeError("Qwen3 后端未加载")
        # 剥掉 CosyVoice 副文本指令（<|endofprompt|> 前的语气/方言部分），
        # Qwen3 tokenizer 不认识该 token，否则会整段念出来
        tts_text = strip_cosyvoice_prompt(tts_text)
        if not tts_text.strip():
            tts_text = "。"  # 清洗后为空的兜底，避免模型报错
        ref_text = strip_cosyvoice_prompt(ref_text) if ref_text else ref_text
        # x_vector_only_mode=True：只用说话人 embedding（极速克隆），忽略参考文本
        wavs, sr = self._model.generate_voice_clone(
            text=tts_text,
            language=language,
            ref_audio=ref_audio,
            ref_text=None if fast_clone else ref_text,
            x_vector_only_mode=fast_clone,
            **gen_kwargs,
        )
        return float_to_pcm16(wavs[0]), int(sr)

    def sample_rate(self) -> int:
        return self._sr

    def list_voices(self) -> dict:
        return {}

    def to_cpu(self) -> None:
        """释放显存：Qwen3 内部有硬编码 cuda 张量，无法简单 .to(cpu) 后继续推理，
        因此采用整模型卸载的方式（重新用时需 load()）。"""
        if self._model is None:
            return
        logger.info("Qwen3 卸载模型释放显存")
        self.unload()

    def to_cuda(self) -> None:
        """重新加载模型到 GPU。"""
        if self._model is not None and self._device == "cuda":
            return
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")
        logger.info("Qwen3 重新加载模型到 GPU")
        self.load()

    def clear_cache(self) -> None:
        super().clear_cache()

    def unload(self) -> None:
        self._model = None
        self._device = "none"
        self.clear_cache()
