from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _resolve(path_or_name: str) -> Path:
    p = Path(path_or_name).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


@dataclass
class Settings:
    model_backend: str = field(
        default_factory=lambda: _env("MODEL_BACKEND", "qwen3").lower()
    )
    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    # 对外端口（浏览器 / 插件访问的统一网关端口）
    port: int = field(default_factory=lambda: int(_env("PORT", "50001")))
    # 内部 worker 端口（网关转发用，对外不可见；改对外端口时无需动这两个）
    qwen_worker_port: int = field(
        default_factory=lambda: int(_env("QWEN_WORKER_PORT", "51001"))
    )
    cosy_worker_port: int = field(
        default_factory=lambda: int(_env("COSY_WORKER_PORT", "51002"))
    )
    language: str = field(default_factory=lambda: _env("LANGUAGE", "zh"))
    voices_dir: Path = field(
        default_factory=lambda: _resolve(_env("VOICES_DIR", "voices"))
    )
    default_voice: str = field(default_factory=lambda: _env("DEFAULT_VOICE", ""))
    sample_rate_fallback: int = field(
        default_factory=lambda: int(_env("SAMPLE_RATE_FALLBACK", "24000"))
    )

    qwen3_model_dir: Path = field(
        default_factory=lambda: _resolve(
            _env("QWEN3_MODEL_DIR", "models/Qwen3-TTS-12Hz-0.6B-Base")
        )
    )
    qwen3_device: str = field(default_factory=lambda: _env("QWEN3_DEVICE", "cuda:0"))
    qwen3_dtype: str = field(default_factory=lambda: _env("QWEN3_DTYPE", "bfloat16"))
    qwen3_attn: str = field(default_factory=lambda: _env("QWEN3_ATTN", "sdpa"))

    cosyvoice_root: Path = field(
        default_factory=lambda: _resolve(
            _env("COSYVOICE_ROOT", "d:/AI/CosyVocie/CosyVoice")
        )
    )
    cosyvoice_model_dir: str = field(
        default_factory=lambda: _env(
            "COSYVOICE_MODEL_DIR", "pretrained_models/Fun-CosyVoice3-0.5B-2512"
        )
    )

    # ── 推理超时 ──
    infer_timeout: float = field(
        default_factory=lambda: float(_env("INFER_TIMEOUT", "180"))
    )
    # ── 显存管理 ──
    # 空闲多少秒后把模型权重搬回 CPU 释放显存；0=关闭自动卸载
    idle_unload_sec: int = field(
        default_factory=lambda: int(_env("IDLE_UNLOAD_SEC", "300"))
    )
    # 空闲显存低于此值(MB)视为被其他程序挤占，触发卸载
    low_vram_mb: int = field(
        default_factory=lambda: int(_env("LOW_VRAM_MB", "1024"))
    )
    # 后台巡检间隔(秒)
    vram_check_sec: float = field(
        default_factory=lambda: float(_env("VRAM_CHECK_SEC", "5"))
    )
    # 权重搬运超时(秒)，防止显存被挤占时 .to(cuda) 卡死
    load_timeout: float = field(
        default_factory=lambda: float(_env("LOAD_TIMEOUT", "60"))
    )
    # 显存不足时禁止 CPU 回退，直接返回 503
    no_cpu_fallback: bool = field(
        default_factory=lambda: _env("NO_CPU_FALLBACK", "0").lower()
        in {"1", "true", "yes", "on"}
    )

    def gpu_need_mb(self) -> float:
        """模型安全驻留 GPU 所需空闲显存(MB)：阈值*3 与 2048 取较大者。"""
        return max(self.low_vram_mb * 3, 2048)


def load_settings() -> Settings:
    settings = Settings()
    if settings.model_backend not in {"qwen3", "cosyvoice3"}:
        raise ValueError(
            f"未知 MODEL_BACKEND: {settings.model_backend!r}，应为 qwen3 或 cosyvoice3"
        )
    ports = [settings.port, settings.qwen_worker_port, settings.cosy_worker_port]
    if len(ports) != len(set(ports)):
        raise ValueError(
            f"端口冲突：PORT/QWEN_WORKER_PORT/COSY_WORKER_PORT 必须互不相同，当前为 {ports}"
        )
    settings.voices_dir.mkdir(parents=True, exist_ok=True)
    return settings
