from __future__ import annotations

import logging
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

logger = logging.getLogger(__name__)


class ASREngine:
    """本地 faster-whisper 语音转文字，首次使用时懒加载模型。"""

    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        local = Path(__file__).resolve().parent.parent / "models" / "faster-whisper-base"
        self.model_id = str(local) if (local / "model.bin").exists() else model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        logger.info("加载 ASR 模型: %s device=%s", self.model_id, self.device)
        self._model = WhisperModel(
            self.model_id, device=self.device, compute_type=self.compute_type
        )
        logger.info("ASR 模型就绪")

    @staticmethod
    def _read_audio(path: str | Path, target_sr: int = 16000):
        """用 soundfile/torchaudio 解码，绕过 faster-whisper 与 av 19 的不兼容。"""
        import numpy as np
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        audio = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
        if sr != target_sr:
            import torch
            import torchaudio

            tensor = torchaudio.transforms.Resample(sr, target_sr)(
                torch.from_numpy(audio).unsqueeze(0)
            )
            audio = tensor.squeeze(0).numpy()
        return np.ascontiguousarray(audio, dtype=np.float32)

    def transcribe(self, audio_path: str | Path, language: str = "zh") -> str:
        self.load()
        audio = self._read_audio(audio_path)
        segments, _ = self._model.transcribe(
            audio, language=language, beam_size=5, vad_filter=False
        )
        return "".join(segment.text for segment in segments).strip()

    def unload(self) -> None:
        self._model = None
