"""下载 Fun-CosyVoice3-0.5B-2512 模型到本项目 models 目录（ModelScope 源）。

用法：
    .venv-qwen3\Scripts\python.exe scripts\download_cosyvoice_model.py

可选：指定其它模型仓库
    .venv-qwen3\Scripts\python.exe scripts\download_cosyvoice_model.py --model <repo_id>
"""
from __future__ import annotations

import argparse
from pathlib import Path

from modelscope import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "Fun-CosyVoice/Fun-CosyVoice3-0.5B-2512"

parser = argparse.ArgumentParser(description="下载 CosyVoice3 模型")
parser.add_argument("--model", default=DEFAULT_MODEL, help="ModelScope 模型仓库名")
args = parser.parse_args()

target = ROOT / "models" / "Fun-CosyVoice3-0.5B-2512"
target.mkdir(parents=True, exist_ok=True)

print(f"模型源: {args.model}")
print(f"目标目录: {target}")
print("开始下载（首次约 6GB，请耐心等待）...")
snapshot_download(args.model, local_dir=str(target))
print("DOWNLOAD_DONE", target)
