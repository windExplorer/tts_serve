"""验证 WebUI 配置的 API 默认克隆模式：不传 fast_clone 时按全局默认走。"""
from __future__ import annotations

import httpx

from _base import BASE

client = httpx.Client(timeout=240.0, trust_env=False)

# 1) 读当前配置
cfg = client.get(BASE + "/api/config").json()
print("初始配置:", cfg, flush=True)

# 2) 设为普通克隆（极速关闭）
r = client.put(BASE + "/api/config", json={"fast_clone_default": False})
print("设为普通:", r.status_code, r.json(), flush=True)
assert r.status_code == 200
assert r.json()["fast_clone_default"] is False

# 3) 用上传文件（非音色名）且不传 prompt_text -> 普通模式应报 400（上传文件不会自动带出映射文本）
import pathlib
ref_path = pathlib.Path(__file__).resolve().parent.parent / "voices" / "萌1.wav"
with ref_path.open("rb") as f:
    r = client.post(BASE + "/inference_zero_shot", data={
        "tts_text": "默认模式测试。",
        # 不传 prompt_wav_path、不传 prompt_text、不传 fast_clone
    }, files={"prompt_wav": ("ref.wav", f, "audio/wav")})
print("普通默认 + 上传无文本 ->", r.status_code, r.text[:120], flush=True)
assert r.status_code == 400, "普通模式应因缺参考文本返回400"

# 4) 设为极速克隆
r = client.put(BASE + "/api/config", json={"fast_clone_default": True})
assert r.json()["fast_clone_default"] is True
print("设为极速:", r.json(), flush=True)

# 5) 极速默认 + 上传无文本 -> 应成功（极速不需要参考文本）
with ref_path.open("rb") as f:
    r = client.post(BASE + "/inference_zero_shot", data={
        "tts_text": "默认模式测试。",
    }, files={"prompt_wav": ("ref.wav", f, "audio/wav")})
print("极速默认 + 上传无文本 ->", r.status_code, len(r.content), "字节", flush=True)
assert r.status_code == 200 and len(r.content) > 1000

# 6) 显式传 fast_clone=false 覆盖全局极速默认 -> 上传无文本应报400
with ref_path.open("rb") as f:
    r = client.post(BASE + "/inference_zero_shot", data={
        "tts_text": "覆盖测试。",
        "fast_clone": "false",
    }, files={"prompt_wav": ("ref.wav", f, "audio/wav")})
print("显式false覆盖 ->", r.status_code, r.text[:120], flush=True)
assert r.status_code == 400, "显式 false 应覆盖全局极速默认"

print("API_DEFAULT_CLONE_OK")
