from __future__ import annotations

import json
import re
from pathlib import Path

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}


class VoicesStore:
    """voices.json 的读写与音色条目管理；兼容旧的扁平格式。"""

    def __init__(self, voices_dir: Path):
        self.dir = voices_dir
        self.path = voices_dir / "voices.json"
        self.dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        result: dict[str, dict] = {}
        if not isinstance(data, dict):
            return result
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = {
                    "prompt_wav": key, "prompt_text": value,
                    "enabled": True, "language": "Chinese",
                }
            elif isinstance(value, dict):
                item = dict(value)
                item.setdefault("enabled", True)
                item.setdefault("language", "Chinese")
                result[key] = item
        return result

    def save(self, voices: dict[str, dict]) -> None:
        self.path.write_text(
            json.dumps(voices, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def safe_filename(name: str, suffix: str) -> str:
        cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip().strip(".")
        cleaned = cleaned or "voice"
        if suffix and not suffix.startswith("."):
            suffix = "." + suffix
        return cleaned + (suffix or "")

    def unique_filename(self, name: str, suffix: str) -> str:
        filename = self.safe_filename(name, suffix)
        candidate = filename
        index = 1
        while (self.dir / candidate).exists():
            stem = Path(filename).stem
            candidate = f"{stem}_{index}{Path(filename).suffix}"
            index += 1
        return candidate

    def add(
        self, name: str, wav_filename: str, prompt_text: str = "",
        enabled: bool = True, language: str = "Chinese",
    ) -> None:
        voices = self.load()
        if name in voices:
            raise ValueError(f"音色已存在: {name}")
        # 新音色置顶：dict 保持插入顺序，前端列表按此顺序渲染
        voices = {
            name: {
                "prompt_wav": wav_filename,
                "prompt_text": prompt_text,
                "enabled": enabled,
                "language": language or "Chinese",
            },
            **voices,
        }
        self.save(voices)

    def update(self, name: str, changes: dict) -> dict[str, dict]:
        voices = self.load()
        if name not in voices:
            raise KeyError(name)
        new_name = changes.pop("name", None)
        if new_name and new_name != name:
            if new_name in voices:
                raise ValueError(f"音色已存在: {new_name}")
            voices[new_name] = voices.pop(name)
            name = new_name
        voices[name].update(changes)
        self.save(voices)
        return voices

    def delete(self, name: str, remove_file: bool = True) -> dict[str, dict]:
        voices = self.load()
        if name not in voices:
            raise KeyError(name)
        item = voices.pop(name)
        if remove_file:
            wav = self.dir / item.get("prompt_wav", "")
            if wav.is_file():
                wav.unlink()
        self.save(voices)
        return voices
