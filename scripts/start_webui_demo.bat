@echo off
chcp 65001 >nul
cd /d "%~dp0.."
echo 启动 qwen-tts 自带 Gradio demo，浏览器访问 http://127.0.0.1:8000
".venv-qwen3\Scripts\qwen-tts-demo.exe" "models\Qwen3-TTS-12Hz-0.6B-Base" --device cuda:0 --dtype bfloat16 --no-flash-attn --ip 127.0.0.1 --port 8000
pause
