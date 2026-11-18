"""测试脚本共用：按 .env 配置生成服务地址。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import load_settings

_settings = load_settings()
BASE = f"http://{_settings.host}:{_settings.port}"
