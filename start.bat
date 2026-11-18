@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ==============================================
echo  TTS_SERVE  语音克隆服务
echo ==============================================
echo.
echo  端口在 .env 的 PORT 配置（默认 50001）
echo  启动后看下方输出的地址，浏览器打开其中的 /ui
echo  在页面「推理后端」下拉框切换 Qwen3 / CosyVoice3
echo.

if not exist ".venv-qwen3\Scripts\python.exe" (
  echo [错误] 未找到 .venv-qwen3 环境，请先按 docs/02-部署与操作指南.md 搭建。
  pause
  exit /b 1
)

".venv-qwen3\Scripts\python.exe" -m app.serve
pause
