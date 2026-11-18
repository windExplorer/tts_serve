"""验证 WebUI 新增的全部后端接口。"""
from __future__ import annotations

import functools
import http.server
import socketserver
import threading
from pathlib import Path
from urllib.parse import quote

import httpx

ROOT = Path(__file__).resolve().parent.parent
from _base import BASE

client = httpx.Client(timeout=180.0, trust_env=False)


def check(label, condition, detail=""):
    print(("PASS " if condition else "FAIL ") + label + (" | " + detail if detail else ""))
    assert condition, label


status = (client.get(BASE + "/api/status")).json()
check("GET /api/status", status["backend"] == "qwen3" and status["sample_rate"] == 24000, str(status))

all_voices = (client.get(BASE + "/api/voices")).json()
enabled = (client.get(BASE + "/voices")).json()
check("GET /api/voices 全量", len(all_voices) >= 60, f"{len(all_voices)} voices")
check("GET /voices 仅启用", len(enabled) == len(all_voices), f"{len(enabled)} enabled")

html = client.get(BASE + "/ui").text
check("GET /ui 新版页面", all(token in html for token in ("音色管理", "voiceDrop", "waveCanvas", "api/asr", "生成参数")))

ref_wav = ROOT / "voices" / "萌1.wav"
with ref_wav.open("rb") as handle:
    response = client.post(
        BASE + "/api/voices/upload",
        data={"name": "测试音色_API", "prompt_text": "测试参考文本", "enabled": "true"},
        files={"file": ("test.wav", handle, "audio/wav")},
    )
check("POST /api/voices/upload", response.status_code == 200, response.text)
check("上传后音色存在", "测试音色_API" in (client.get(BASE + "/api/voices")).json())

audio_response = client.get(BASE + "/api/voices/" + quote("测试音色_API") + "/audio")
check("GET /api/voices/{name}/audio", audio_response.status_code == 200, str(len(audio_response.content)))

response = client.put(
    BASE + "/api/voices/" + quote("测试音色_API"),
    json={"enabled": False},
)
check("PUT 禁用音色", response.status_code == 200)
check("禁用后 /voices 不暴露", "测试音色_API" not in (client.get(BASE + "/voices")).json())
check("禁用后 /api/voices 仍可见", (client.get(BASE + "/api/voices")).json()["测试音色_API"]["enabled"] is False)

response = client.put(
    BASE + "/api/voices/" + quote("测试音色_API"),
    json={"name": "测试音色_API2", "prompt_text": "改名后的文本", "enabled": True},
)
check("PUT 改名+改文本+启用", response.status_code == 200, response.text)
updated = (client.get(BASE + "/api/voices")).json()["测试音色_API2"]
check("改名结果正确", updated["prompt_text"] == "改名后的文本" and updated["enabled"] is True)

handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT / "voices"))
httpd = socketserver.TCPServer(("127.0.0.1", 8888), handler)
threading.Thread(target=httpd.serve_forever, daemon=True).start()
try:
    url = "http://127.0.0.1:8888/" + quote("普通1.wav")
    response = client.post(
        BASE + "/api/voices/url",
        data={"name": "测试音色_URL", "url": url, "prompt_text": "URL 下载的参考文本", "enabled": "true"},
    )
    check("POST /api/voices/url", response.status_code == 200, response.text)
    url_voice = (client.get(BASE + "/api/voices")).json()["测试音色_URL"]
    check("URL 下载并重命名为音色名", url_voice["prompt_wav"].startswith("测试音色_URL"), url_voice["prompt_wav"])
finally:
    httpd.shutdown()

response = client.post(
    BASE + "/inference_zero_shot",
    data={
        "tts_text": "你好，这是带参数的测试。",
        "prompt_wav_path": "萌1.wav",
        "prompt_text": "今天的晚霞超好看的！我买了你最喜欢的焦糖奶茶，我们一起去江边走走好不好？就当是放松一下啦。",
        "temperature": "0.8",
        "top_k": "40",
    },
)
check("POST /inference_zero_shot 带参数", response.status_code == 200 and len(response.content) > 1000, str(len(response.content)))
check("响应含生成耗时头", "X-Generation-Time" in response.headers, response.headers.get("X-Generation-Time", ""))
check("响应含采样率头", response.headers.get("X-Sample-Rate") == "24000")

for name in ("测试音色_API2", "测试音色_URL"):
    response = client.delete(BASE + "/api/voices/" + quote(name))
    check("DELETE " + name, response.status_code == 200)
final_voices = (client.get(BASE + "/api/voices")).json()
check("清理后音色数恢复", len(final_voices) == len(all_voices), f"{len(final_voices)} vs {len(all_voices)}")

print("FULL_VERIFY_OK")
