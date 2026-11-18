@echo off
chcp 65001 >nul
cd /d "%~dp0.."
if not exist ".venv-qwen3\Scripts\python.exe" (
  echo [ERROR] 未找到 .venv-qwen3，请先按 docs/02-部署与操作指南.md 的 3.1 搭建环境。
  pause
  exit /b 1
)
".venv-qwen3\Scripts\modelscope.exe" download --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --local_dir ".\models\Qwen3-TTS-12Hz-0.6B-Base"
pause
