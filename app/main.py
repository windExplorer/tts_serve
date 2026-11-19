from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import time  # noqa: F401  用于显存巡检时间戳
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

from .asr import ASREngine
from .config import Settings, load_settings
from . import runtime_config
from .schema import HealthResponse
from .voices_store import AUDIO_SUFFIXES, VoicesStore
from backends.base import BaseBackend
from backends.cosyvoice3 import CosyVoice3Backend
from backends.qwen3 import Qwen3Backend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("tts_serve")

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
GEN_PARAMS = {
    "max_new_tokens": int,
    "temperature": float,
    "top_k": int,
    "top_p": float,
    "repetition_penalty": float,
    "subtalker_top_k": int,
    "subtalker_top_p": float,
    "subtalker_temperature": float,
}


def _resolve_ref_wav(settings: Settings, wav: str) -> str:
    path = Path(wav).expanduser()
    return str(path if path.is_absolute() else settings.voices_dir / wav)


def create_backend(settings: Settings, voices: dict) -> BaseBackend:
    if settings.model_backend == "qwen3":
        warmup_audio, warmup_text = None, ""
        for info in voices.values():
            if not isinstance(info, dict) or info.get("enabled") is False:
                continue
            wav = info.get("prompt_wav", "")
            if not wav:
                continue
            candidate = Path(_resolve_ref_wav(settings, wav))
            if candidate.exists():
                warmup_audio = str(candidate)
                warmup_text = info.get("prompt_text", "")
                break
        return Qwen3Backend(
            model_dir=settings.qwen3_model_dir,
            device=settings.qwen3_device,
            dtype=settings.qwen3_dtype,
            attn=settings.qwen3_attn,
            fallback_sample_rate=settings.sample_rate_fallback,
            warmup_ref_audio=warmup_audio,
            warmup_ref_text=warmup_text,
        )

    return CosyVoice3Backend(
        root=settings.cosyvoice_root,
        model_dir=settings.cosyvoice_model_dir,
        fallback_sample_rate=settings.sample_rate_fallback,
    )


def resolve_voice(
    settings: Settings,
    voices: dict,
    prompt_wav_path: str,
    prompt_text: str,
    uploaded_wav: str | None,
) -> tuple[str | None, str | None]:
    """解析参考音频路径与逐字原文；上传文件优先，其次 prompt_wav_path，最后默认音色。"""
    info = None
    if prompt_wav_path:
        info = voices.get(prompt_wav_path)
        if info is None:
            for value in voices.values():
                if isinstance(value, dict) and value.get("prompt_wav") == prompt_wav_path:
                    info = value
                    break

    if uploaded_wav:
        ref_audio = uploaded_wav
    elif isinstance(info, dict):
        wav = info.get("prompt_wav", "")
        ref_audio = _resolve_ref_wav(settings, wav) if wav else None
    elif prompt_wav_path:
        path = Path(prompt_wav_path).expanduser()
        ref_audio = str(path) if path.is_absolute() else str(settings.voices_dir / prompt_wav_path)
    elif settings.default_voice and isinstance(voices.get(settings.default_voice), dict):
        info = voices[settings.default_voice]
        wav = info.get("prompt_wav", "")
        ref_audio = _resolve_ref_wav(settings, wav) if wav else None
    else:
        ref_audio = None

    if not prompt_text and isinstance(info, dict):
        prompt_text = info.get("prompt_text", "")
    return ref_audio, prompt_text or None


def enabled_voices(voices: dict) -> dict:
    return {name: info for name, info in voices.items() if info.get("enabled", True)}


def parse_gen_params(form: dict) -> dict:
    params = {}
    for key, caster in GEN_PARAMS.items():
        value = form.get(key)
        if value is not None and str(value).strip() != "":
            try:
                params[key] = caster(value)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail=f"参数 {key} 格式不正确")
    return params


# ── 显存管理状态 ──
_vram_state = {
    "last_infer_end": time.time(),
    "device_lock": None,      # lifespan 中初始化 asyncio.Lock
    "idle_task": None,
}


def _free_vram_mb() -> float:
    """当前空闲显存(MB)，无 CUDA 时返回 0。"""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.mem_get_info()[0] / 1024 / 1024
    except Exception:
        pass
    return 0.0


async def ensure_loaded(app: FastAPI) -> bool:
    """推理前确保权重就绪。返回 False 表示应拒绝请求（禁用 CPU 回退且显存不足）。"""
    settings = app.state.settings
    backend = app.state.backend

    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    if not has_cuda:
        return True  # 纯 CPU 模式，正常推理

    need_mb = settings.gpu_need_mb()
    lock = _vram_state["device_lock"]

    if backend.device == "cuda":
        free = _free_vram_mb()
        if free < need_mb:
            # 显存被挤占：先搬回 CPU，避免推理时分配激活值卡死
            logger.warning("显存被挤占 (free %.0fMB < 需 %.0fMB)，权重搬回 CPU", free, need_mb)
            try:
                async with lock:
                    await asyncio.wait_for(
                        asyncio.to_thread(backend.to_cpu), timeout=settings.load_timeout
                    )
            except (asyncio.TimeoutError, RuntimeError) as exc:
                logger.warning("搬回 CPU 失败/超时: %s", exc)
            if settings.no_cpu_fallback:
                return False
        return True

    # 当前在 CPU / 已卸载：确保模型可用
    free = _free_vram_mb()
    # 模型未加载（如 Qwen3 卸载后）必须重载，否则无法合成；
    # 显存够则重载到 GPU，不够则按 CPU 回退策略处理
    if free >= need_mb or not backend.is_loaded:
        try:
            async with lock:
                # 更新活跃时间，避免重载期间被巡检误判为空闲而再次卸载
                _vram_state["last_infer_end"] = time.time()
                await asyncio.wait_for(
                    asyncio.to_thread(backend.to_cuda), timeout=settings.load_timeout
                )
                _vram_state["last_infer_end"] = time.time()
            return True
        except (asyncio.TimeoutError, RuntimeError) as exc:
            logger.warning("载入 GPU 失败/超时 (%s)，回退 CPU 推理", exc)
            if backend.is_loaded:
                return True  # 至少模型在内存，可尝试 CPU 推理
            return False

    if settings.no_cpu_fallback:
        logger.warning("显存不足 (free %.0fMB < 需 %.0fMB)，拒绝请求", free, need_mb)
        return False
    logger.warning("显存不足 (free %.0fMB < 需 %.0fMB)，本次 CPU 推理", free, need_mb)
    return True


async def _idle_unloader(app: FastAPI) -> None:
    """后台巡检：空闲超时 或 显存被挤占时，自动把权重搬回 CPU 释放显存。"""
    settings = app.state.settings
    backend = app.state.backend
    while True:
        await asyncio.sleep(settings.vram_check_sec)
        if backend.device != "cuda":
            continue
        # 推理或权重搬运进行中不打扰（两个锁任一持有即视为忙碌）
        if app.state.lock.locked() or _vram_state["device_lock"].locked():
            continue
        timed_out = (
            settings.idle_unload_sec > 0
            and time.time() - _vram_state["last_infer_end"] >= settings.idle_unload_sec
        )
        free = _free_vram_mb()
        pressured = free > 0 and free < settings.gpu_need_mb()
        if timed_out or pressured:
            reason = "空闲超时" if timed_out else f"显存被挤占 (free {free:.0f}MB)"
            logger.info("%s，自动卸载权重释放显存", reason)
            try:
                async with _vram_state["device_lock"]:
                    await asyncio.wait_for(
                        asyncio.to_thread(backend.to_cpu), timeout=settings.load_timeout
                    )
            except (asyncio.TimeoutError, RuntimeError) as exc:
                logger.warning("自动卸载失败: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    store = VoicesStore(settings.voices_dir)
    voices = store.load()
    backend = create_backend(settings, voices)
    logger.info("加载后端: %s", backend.name)
    backend.load()

    app.state.settings = settings
    app.state.store = store
    app.state.voices = voices
    app.state.backend = backend
    app.state.asr = ASREngine()
    app.state.sample_rate = backend.sample_rate()
    app.state.lock = asyncio.Lock()
    _vram_state["device_lock"] = asyncio.Lock()
    _vram_state["last_infer_end"] = time.time()
    _vram_state["idle_task"] = asyncio.create_task(_idle_unloader(app))
    logger.info(
        "服务就绪，sample_rate=%d，后端=%s，空闲卸载=%ss，显存门槛=%dMB",
        app.state.sample_rate,
        backend.name,
        settings.idle_unload_sec,
        settings.low_vram_mb,
    )
    try:
        yield
    finally:
        if _vram_state["idle_task"] is not None:
            _vram_state["idle_task"].cancel()
            try:
                await _vram_state["idle_task"]
            except asyncio.CancelledError:
                pass
        backend.unload()
        app.state.asr.unload()
        logger.info("后端已卸载")


app = FastAPI(title="TTS_SERVE", version="1.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(sample_rate=app.state.sample_rate)


@app.get("/voices")
async def list_voices() -> dict:
    return enabled_voices(app.state.voices)


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def ui() -> HTMLResponse:
    return HTMLResponse((Path(__file__).parent / "ui.html").read_text(encoding="utf-8"))


@app.post("/inference_zero_shot")
async def inference_zero_shot(
    tts_text: str = Form(...),
    prompt_text: str = Form(""),
    prompt_wav: UploadFile | None = File(None),
    prompt_wav_path: str = Form(""),
    max_new_tokens: str = Form(""),
    temperature: str = Form(""),
    top_k: str = Form(""),
    top_p: str = Form(""),
    repetition_penalty: str = Form(""),
    subtalker_top_k: str = Form(""),
    subtalker_top_p: str = Form(""),
    subtalker_temperature: str = Form(""),
    fast_clone: str = Form(""),
) -> Response:
    if not tts_text.strip():
        raise HTTPException(status_code=400, detail="tts_text 不能为空")
    selected = app.state.voices.get(prompt_wav_path)
    if isinstance(selected, dict) and selected.get("enabled") is False:
        raise HTTPException(status_code=400, detail="该音色已禁用")

    gen_kwargs = parse_gen_params(
        {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "subtalker_top_k": subtalker_top_k,
            "subtalker_top_p": subtalker_top_p,
            "subtalker_temperature": subtalker_temperature,
        }
    )

    tmp_path = None
    if prompt_wav is not None:
        suffix = Path(prompt_wav.filename or "").suffix or ".wav"
        fd, tmp_path = tempfile.mkstemp(prefix="tts_ref_", suffix=suffix)
        with os.fdopen(fd, "wb") as handle:
            shutil.copyfileobj(prompt_wav.file, handle)

    try:
        ref_audio, ref_text = resolve_voice(
            app.state.settings,
            app.state.voices,
            prompt_wav_path or "",
            prompt_text or "",
            tmp_path,
        )
        if not ref_audio or not Path(ref_audio).exists():
            raise HTTPException(
                status_code=400,
                detail="未找到可用的参考音频，请检查 prompt_wav_path / prompt_wav 与 voices.json",
            )
        # 未传 fast_clone 时用 WebUI 配置的全局默认值
        if fast_clone.strip().lower() in {"", "default"}:
            is_fast = bool(runtime_config.get("fast_clone_default", True))
        else:
            is_fast = fast_clone.strip().lower() not in {"false", "0", "no"}
        if not ref_text and not is_fast:
            raise HTTPException(
                status_code=400,
                detail="普通模式需要参考文本 prompt_text；极速克隆模式可留空",
            )

        # 推理前确保权重在 GPU（显存不足时按需搬运，必要时 CPU 回退）
        if not await ensure_loaded(app):
            raise HTTPException(status_code=503, detail="显存不足且已禁用 CPU 回退，请稍后重试")

        started = time.perf_counter()
        async with app.state.lock:
            try:
                pcm, sr = await asyncio.wait_for(
                    asyncio.to_thread(
                        app.state.backend.synthesize,
                        tts_text,
                        ref_audio,
                        ref_text,
                        app.state.settings.language,
                        fast_clone=is_fast,
                        **gen_kwargs,
                    ),
                    timeout=app.state.settings.infer_timeout,
                )
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail=f"推理超时（>{app.state.settings.infer_timeout}s）",
                )
        elapsed = time.perf_counter() - started
        _vram_state["last_infer_end"] = time.time()

        if sr != app.state.sample_rate:
            logger.warning(
                "后端返回 sample_rate=%d 与 GET / 的 %d 不一致",
                sr,
                app.state.sample_rate,
            )
        return Response(
            content=pcm,
            media_type="application/octet-stream",
            headers={
                "X-Generation-Time": f"{elapsed:.3f}",
                "X-Sample-Rate": str(sr),
            },
        )
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


@app.post("/inference_instruct2")
async def inference_instruct2() -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={"detail": "本服务暂未实现 /inference_instruct2"},
    )


# ---------------- 管理 API（供 WebUI 使用） ----------------


@app.get("/api/status")
async def api_status() -> dict:
    import torch

    settings = app.state.settings
    return {
        "backend": app.state.backend.name,
        "sample_rate": app.state.sample_rate,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": app.state.backend.device,
        "free_vram_mb": round(_free_vram_mb()),
        "gpu_need_mb": int(settings.gpu_need_mb()),
        "idle_unload_sec": settings.idle_unload_sec,
        "voices_total": len(app.state.voices),
        "voices_enabled": len(enabled_voices(app.state.voices)),
        "asr_loaded": app.state.asr._model is not None,
    }


@app.post("/api/vram/unload")
async def api_vram_unload() -> dict:
    """手动把模型权重搬回 CPU，立即释放显存（供 ComfyUI 等轮班时调用）。"""
    async with _vram_state["device_lock"]:
        await asyncio.to_thread(app.state.backend.to_cpu)
    return {"ok": True, "device": app.state.backend.device, "free_vram_mb": round(_free_vram_mb())}


@app.post("/api/vram/load")
async def api_vram_load() -> dict:
    """手动把模型权重搬回 GPU。"""
    try:
        async with _vram_state["device_lock"]:
            await asyncio.wait_for(
                asyncio.to_thread(app.state.backend.to_cuda),
                timeout=app.state.settings.load_timeout,
            )
    except (asyncio.TimeoutError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"载入 GPU 失败: {exc}")
    return {"ok": True, "device": app.state.backend.device, "free_vram_mb": round(_free_vram_mb())}


@app.get("/api/config")
async def api_get_config() -> dict:
    """读取运行时配置（WebUI 设置用）。"""
    return runtime_config.get_config()


class ConfigUpdate(BaseModel):
    fast_clone_default: bool | None = None


@app.put("/api/config")
async def api_set_config(payload: ConfigUpdate) -> dict:
    """修改运行时配置（立即生效并持久化）。"""
    if payload.fast_clone_default is not None:
        runtime_config.set_value("fast_clone_default", bool(payload.fast_clone_default))
    return {"ok": True, **runtime_config.get_config()}


@app.get("/api/voices")
async def api_list_voices() -> dict:
    return app.state.voices


@app.post("/api/voices/upload")
async def api_upload_voice(
    name: str = Form(...),
    prompt_text: str = Form(""),
    enabled: str = Form("true"),
    file: UploadFile = File(...),
) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"不支持的音频格式: {suffix}")
    filename = app.state.store.unique_filename(name, suffix)
    target = app.state.settings.voices_dir / filename
    total = 0
    try:
        with target.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=400, detail="文件超过 50MB")
                handle.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    try:
        app.state.store.add(name, filename, prompt_text, enabled.lower() not in {"false", "0", "no"})
    except ValueError as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc))
    app.state.voices = app.state.store.load()
    return {"name": name, "prompt_wav": filename}


@app.post("/api/voices/url")
async def api_add_voice_from_url(
    name: str = Form(...),
    url: str = Form(...),
    prompt_text: str = Form(""),
    enabled: str = Form("true"),
) -> dict:
    from urllib.parse import urlparse

    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        suffix = ".wav"
    filename = app.state.store.unique_filename(name, suffix)
    target = app.state.settings.voices_dir / filename
    total = 0
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=400, detail="文件超过 50MB")
                    with target.open("ab") as handle:
                        handle.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="下载到的文件为空")
    except httpx.HTTPError as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"下载失败: {exc}")
    except Exception:
        target.unlink(missing_ok=True)
        raise
    try:
        app.state.store.add(name, filename, prompt_text, enabled.lower() not in {"false", "0", "no"})
    except ValueError as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc))
    app.state.voices = app.state.store.load()
    return {"name": name, "prompt_wav": filename, "bytes": total}


@app.put("/api/voices/{name}")
async def api_update_voice(name: str, payload: dict) -> dict:
    allowed = {"name", "prompt_text", "enabled", "prompt_wav"}
    changes = {k: v for k, v in payload.items() if k in allowed}
    try:
        voices = app.state.store.update(name, changes)
    except KeyError:
        raise HTTPException(status_code=404, detail="音色不存在")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    app.state.voices = voices
    return {"ok": True}


@app.delete("/api/voices/{name}")
async def api_delete_voice(name: str, remove_file: bool = True) -> dict:
    try:
        voices = app.state.store.delete(name, remove_file=remove_file)
    except KeyError:
        raise HTTPException(status_code=404, detail="音色不存在")
    app.state.voices = voices
    return {"ok": True}


@app.get("/api/voices/{name}/audio")
async def api_voice_audio(name: str) -> FileResponse:
    info = app.state.voices.get(name)
    if not isinstance(info, dict):
        raise HTTPException(status_code=404, detail="音色不存在")
    path = (app.state.settings.voices_dir / info.get("prompt_wav", "")).resolve()
    if not path.is_file() or app.state.settings.voices_dir.resolve() not in path.parents:
        raise HTTPException(status_code=404, detail="参考音频不存在")
    return FileResponse(path)


@app.post("/api/audio/fetch")
async def api_fetch_audio(payload: dict) -> Response:
    """服务端下载远端音频并返回字节，供新增音色第一步试听/画波形（不落库）。

    前端直接 fetch 外链会受 CORS 限制，故由服务端代取。
    """
    from urllib.parse import urlparse

    url = str(payload.get("url", "")).strip()
    if not url:
        raise HTTPException(status_code=400, detail="缺少 url")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="url 必须是 http/https 地址")
    suffix = Path(parsed.path).suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"链接不是支持的音频格式（{', '.join(sorted(AUDIO_SUFFIXES))}）",
        )
    media_types = {
        ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac",
        ".ogg": "audio/ogg", ".m4a": "audio/mp4", ".aac": "audio/aac",
        ".opus": "audio/ogg",
    }
    buf = bytearray()
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(1024 * 1024):
                    buf.extend(chunk)
                    if len(buf) > MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=400, detail="音频超过 50MB")
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=400, detail=f"下载失败: {exc}")
    if not buf:
        raise HTTPException(status_code=400, detail="下载到的音频为空")
    filename = Path(parsed.path).name or f"voice{suffix}"
    return Response(
        content=bytes(buf),
        media_type=media_types.get(suffix, "application/octet-stream"),
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/asr")
async def api_asr(
    file: UploadFile = File(...),
    language: str = Form("zh"),
) -> dict:
    suffix = Path(file.filename or "").suffix.lower() or ".wav"
    if suffix not in AUDIO_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"不支持的音频格式: {suffix}")
    fd, tmp_path = tempfile.mkstemp(prefix="tts_asr_", suffix=suffix)
    with os.fdopen(fd, "wb") as handle:
        shutil.copyfileobj(file.file, handle)
    try:
        text = await asyncio.to_thread(app.state.asr.transcribe, tmp_path, language)
        return {"text": text}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"语音识别失败: {exc}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
