"""运行时可修改的配置（WebUI 改、立即生效、持久化到 data/runtime_config.json）。

与 .env 的区别：.env 是启动时的初始值，这里是运行中 WebUI 改的值，优先级更高。
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CONFIG_FILE = DATA_DIR / "runtime_config.json"

DEFAULT_CONFIG = {
    "fast_clone_default": True,  # API 调用未传 fast_clone 时的默认值
}

_lock = threading.Lock()
_cache: dict | None = None


def _load() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    config = dict(DEFAULT_CONFIG)
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                config.update({k: v for k, v in data.items() if k in DEFAULT_CONFIG})
        except Exception:
            pass
    _cache = config
    return _cache


def get_config() -> dict:
    with _lock:
        return dict(_load())


def get(key: str, default=None):
    with _lock:
        return _load().get(key, default)


def set_value(key: str, value) -> None:
    if key not in DEFAULT_CONFIG:
        raise KeyError(f"未知配置项: {key}")
    with _lock:
        config = _load()
        config[key] = value
        DATA_DIR.mkdir(exist_ok=True)
        CONFIG_FILE.write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
