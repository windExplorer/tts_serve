"""串行对比：原生 cosyvoice_api.py vs 本项目 cosyvoice3 后端。

绝不同时启动两个服务（显存不够）：逐个启动→生成→保存音频→停止。
两边用相同的长文本、相同音色、相同的 zero_shot 模式。
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LOGS = ROOT / "logs"
DATA.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)

COSY_ROOT = Path(r"D:\AI\CosyVocie\CosyVoice")
PY_COSY = COSY_ROOT / ".venv" / "Scripts" / "python.exe"
TORCH_LIB = COSY_ROOT / ".venv" / "Lib" / "site-packages" / "torch" / "lib"

TEXT = (
    "夜幕降临时，我常在这片花园驻足。每一朵绽放的花都承载着逝者的故事，"
    "像是星辰般闪烁。作为记忆的守护者，我愿静静聆听每一个离别的呢喃，"
    "将它们化作永恒的光芒。风穿过树梢，带来远方的低语，我闭上眼睛，"
    "任由这份温柔将我包围。"
)
REF_TEXT = (
    "今天的晚霞超好看的！我买了你最喜欢的焦糖奶茶，"
    "我们一起去江边走走好不好？就当是放松一下啦。"
)
VOICE = "萌1.wav"


def make_env() -> dict:
    env = os.environ.copy()
    env["PATH"] = str(TORCH_LIB) + os.pathsep + env.get("PATH", "")
    return env


def wait_ready(url: str, proc: subprocess.Popen, name: str, timeout: int = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{name} 进程提前退出，查看日志")
        try:
            r = httpx.get(url + "/", timeout=3, trust_env=False)
            if r.status_code == 200:
                print(f"  {name} 就绪", flush=True)
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"{name} 启动超时")


def stop(proc: subprocess.Popen) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except Exception:
        proc.kill()
    time.sleep(2)  # 等显存释放


def run_one(name: str, args: list[str], cwd: Path, url: str, logfile: Path) -> dict:
    print(f"\n=== 启动 {name} ===", flush=True)
    log = open(logfile, "w", encoding="utf-8")
    proc = subprocess.Popen(
        args, cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, env=make_env()
    )
    try:
        wait_ready(url, proc, name)
        # 预热一次（排除首次模型/缓存开销），两边都预热保证公平
        payload = {"tts_text": "预热。", "prompt_text": REF_TEXT, "prompt_wav_path": VOICE}
        httpx.post(url + "/inference_zero_shot", data=payload, timeout=180, trust_env=False)

        # 正式计时（3 次）
        times, sizes, audios = [], [], []
        payload = {"tts_text": TEXT, "prompt_text": REF_TEXT, "prompt_wav_path": VOICE}
        for i in range(3):
            t0 = time.perf_counter()
            r = httpx.post(url + "/inference_zero_shot", data=payload, timeout=180, trust_env=False)
            times.append(time.perf_counter() - t0)
            sizes.append(len(r.content))
            if r.status_code != 200:
                raise RuntimeError(f"{name} 合成失败 HTTP {r.status_code}: {r.text[:200]}")
            audios.append(r.content)

        # 保存最后一次音频
        out = DATA / f"compare_{name}.wav"
        pcm = np.frombuffer(audios[-1], dtype="<i2")
        sf.write(str(out), pcm.astype(np.float32) / 32767.0, 24000)

        result = {
            "name": name,
            "times": times,
            "avg": sum(times) / len(times),
            "bytes": sizes[-1],
            "audio_sec": len(pcm) / 24000,
            "out": out,
        }
        print(f"  {name}: 耗时 {[round(t,2) for t in times]} 平均 {result['avg']:.2f}s "
              f"音频 {result['audio_sec']:.2f}s {sizes[-1]}字节", flush=True)
        return result
    finally:
        stop(proc)
        log.close()
        print(f"  {name} 已停止，等待显存释放…", flush=True)


results = []

# 1) 原生 API（50000）
results.append(run_one(
    "native",
    [str(PY_COSY), "cosyvoice_api.py", "--model_dir",
     "pretrained_models/Fun-CosyVoice3-0.5B-2512",
     "--voices_dir", str(COSY_ROOT / "cosyvoice_voices"),
     "--port", "50000"],
    COSY_ROOT,
    "http://127.0.0.1:50000",
    LOGS / "compare_native.log",
))

# 2) 我们的后端（51002，独立进程，用 CosyVoice venv + main.py）
env2 = make_env()
env2.update({
    "MODEL_BACKEND": "cosyvoice3",
    "COSYVOICE_ROOT": str(COSY_ROOT),
    "COSYVOICE_MODEL_DIR": "pretrained_models/Fun-CosyVoice3-0.5B-2512",
})
# 用我们先前验证过的启动方式（复用 run_one，但需带 env2）
print("\n=== 启动 ours ===", flush=True)
log = open(LOGS / "compare_ours.log", "w", encoding="utf-8")
proc = subprocess.Popen(
    [str(PY_COSY), "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "51002"],
    cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env2,
)
try:
    wait_ready("http://127.0.0.1:51002", proc, "ours")
    payload = {"tts_text": "预热。", "prompt_text": REF_TEXT, "prompt_wav_path": VOICE, "fast_clone": "false"}
    httpx.post("http://127.0.0.1:51002/inference_zero_shot", data=payload, timeout=180, trust_env=False)
    times, sizes, audios = [], [], []
    payload = {"tts_text": TEXT, "prompt_text": REF_TEXT, "prompt_wav_path": VOICE, "fast_clone": "false"}
    for _ in range(3):
        t0 = time.perf_counter()
        r = httpx.post("http://127.0.0.1:51002/inference_zero_shot", data=payload, timeout=180, trust_env=False)
        times.append(time.perf_counter() - t0)
        sizes.append(len(r.content))
        if r.status_code != 200:
            raise RuntimeError(f"ours 合成失败 HTTP {r.status_code}: {r.text[:200]}")
        audios.append(r.content)
    out = DATA / "compare_ours.wav"
    pcm = np.frombuffer(audios[-1], dtype="<i2")
    sf.write(str(out), pcm.astype(np.float32) / 32767.0, 24000)
    results.append({
        "name": "ours", "times": times, "avg": sum(times) / len(times),
        "bytes": sizes[-1], "audio_sec": len(pcm) / 24000, "out": out,
    })
    print(f"  ours: 耗时 {[round(t,2) for t in times]} 平均 {results[-1]['avg']:.2f}s "
          f"音频 {results[-1]['audio_sec']:.2f}s {sizes[-1]}字节", flush=True)
finally:
    stop(proc)
    log.close()
    print("  ours 已停止", flush=True)

print("\n========== 对比结果 ==========", flush=True)
for r in results:
    print(f"{r['name']:8s} 平均 {r['avg']:6.2f}s  音频 {r['audio_sec']:5.2f}s  "
          f"{r['bytes']}字节  -> {r['out'].name}", flush=True)
print("SERIAL_COMPARE_DONE")
