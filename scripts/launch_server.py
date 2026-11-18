"""后台启动统一网关（兼容旧脚本名；端口读 .env）。"""
from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 直接复用 launch_gateway 的逻辑（端口全部动态读配置）
runpy.run_path(str(ROOT / "scripts" / "launch_gateway.py"), run_name="__main__")
