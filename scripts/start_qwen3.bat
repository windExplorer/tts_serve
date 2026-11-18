@echo off
chcp 65001 >nul
cd /d "%~dp0.."
rem 统一入口：端口读 .env 的 PORT，默认后端 qwen3
set "MODEL_BACKEND=qwen3"
".venv-qwen3\Scripts\python.exe" -m app.serve
pause
