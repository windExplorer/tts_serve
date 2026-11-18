"""后台启动统一网关（兼容旧脚本名；端口读 .env）。

切换后端请用 WebUI 下拉框，或启动时设 MODEL_BACKEND=cosyvoice3。
"""
from __future__ import annotations

import os
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["MODEL_BACKEND"] = "cosyvoice3"
runpy.run_path(str(ROOT / "scripts" / "launch_gateway.py"), run_name="__main__")
