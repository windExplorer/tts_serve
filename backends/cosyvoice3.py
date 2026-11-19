from __future__ import annotations

import gc
import logging
import sys
from pathlib import Path

from .base import BaseBackend, float_to_pcm16, normalize_tts_text

logger = logging.getLogger(__name__)


class CosyVoice3Backend(BaseBackend):
    name = "cosyvoice3"

    def __init__(
        self,
        root: Path,
        model_dir: str,
        fallback_sample_rate: int = 24000,
    ):
        self.root = root
        self.model_dir = model_dir
        self.fallback_sample_rate = fallback_sample_rate
        self._model = None
        self._sr = fallback_sample_rate
        self._device = "none"

    @property
    def device(self) -> str:
        return self._device

    def _resolve_model_dir(self) -> str:
        candidates = [
            Path(self.model_dir),
            self.root / self.model_dir,
            Path(__file__).resolve().parent.parent
            / "models"
            / "Fun-CosyVoice3-0.5B-2512",
        ]
        for candidate in candidates:
            if candidate.is_dir() and (candidate / "cosyvoice3.yaml").exists():
                logger.info("CosyVoice3 模型目录: %s", candidate)
                return str(candidate)
        logger.warning("未找到 cosyvoice3.yaml，仍尝试使用配置值: %s", self.model_dir)
        return str(self.model_dir)

    def load(self) -> None:
        import os

        root = str(self.root.resolve())
        matcha = str(self.root / "third_party" / "Matcha-TTS")
        for path in (root, matcha):
            if path not in sys.path:
                sys.path.insert(0, path)

        # Windows 下把 torch CUDA 运行库前置，避免系统 CUDA 13 的 DLL 抢先加载
        try:
            import torch

            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            if os.path.isdir(torch_lib) and torch_lib not in os.environ.get("PATH", ""):
                os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
        except ImportError:
            pass

        from cosyvoice.cli.cosyvoice import AutoModel, CosyVoice

        model_dir = self._resolve_model_dir()
        try:
            self._model = AutoModel(model_dir=model_dir)
        except Exception as exc:
            logger.warning("AutoModel 初始化失败（%s），回退 CosyVoice", exc)
            self._model = CosyVoice(model_dir=model_dir)

        self._sr = int(getattr(self._model, "sample_rate", self.fallback_sample_rate))
        import torch

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("CosyVoice3 加载完成，sample_rate=%d, model=%s", self._sr, model_dir)

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
            raise RuntimeError("CosyVoice3 后端未加载")

        import numpy as np

        tts_text = normalize_tts_text(tts_text)
        if not tts_text.strip():
            tts_text = "。"  # 清洗后为空的兜底，避免模型报错

        chunks = []
        # 极速克隆用 cross_lingual：内部只保留说话人 embedding，去掉 prompt_text
        # 参考音频内部自行 load_wav，必须传路径字符串而非 tensor
        infer = (
            self._model.inference_cross_lingual(tts_text, ref_audio, stream=False)
            if fast_clone
            else self._model.inference_zero_shot(tts_text, ref_text, ref_audio, stream=False)
        )
        for item in infer:
            speech = item.get("tts_speech")
            if speech is None:
                continue
            chunks.append(
                np.clip(speech.squeeze().cpu().numpy().astype(np.float32), -1.0, 1.0)
            )

        if not chunks:
            return b"", self._sr
        return float_to_pcm16(np.concatenate(chunks)), self._sr

    def sample_rate(self) -> int:
        return self._sr

    def list_voices(self) -> dict:
        return {}

    def _move_to(self, target: str) -> None:
        if self._model is None:
            return
        import torch

        m = self._model.model
        for sub in ("llm", "flow", "hift"):
            part = getattr(m, sub, None)
            if part is not None:
                part.to(target)
        m.device = torch.device(target)
        self._device = target
        logger.info("CosyVoice3 权重已搬到 %s", target)

    def to_cpu(self) -> None:
        if self._device == "cpu" or self._model is None:
            return
        m = self._model.model
        for key in (
            "flow_cache_dict", "hift_cache_dict", "mel_overlap_dict",
            "tts_speech_token_dict", "llm_end_dict",
        ):
            cache = getattr(m, key, None)
            if cache is not None and hasattr(cache, "clear"):
                cache.clear()
        self._move_to("cpu")
        self.clear_cache()

    def to_cuda(self) -> None:
        if self._device == "cuda" or self._model is None:
            return
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")
        self._move_to("cuda")

    def clear_cache(self) -> None:
        super().clear_cache()

    def unload(self) -> None:
        self._model = None
        self._device = "none"
        self.clear_cache()
