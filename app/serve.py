"""统一启动入口：按 .env 的配置启动网关（对外端口读 PORT）。

用法：python -m app.serve
"""
from __future__ import annotations

import uvicorn

from app.config import load_settings


def main() -> None:
    settings = load_settings()
    print(
        f"启动 TTS_SERVE 网关: http://{settings.host}:{settings.port}/ui\n"
        f"对外端口 PORT={settings.port}（浏览器/插件访问）\n"
        f"内部 worker: qwen3={settings.qwen_worker_port}, cosyvoice3={settings.cosy_worker_port}\n"
        f"初始后端 MODEL_BACKEND={settings.model_backend}",
        flush=True,
    )
    uvicorn.run(
        "app.gateway:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
