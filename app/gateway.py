"""统一网关：对外一个端口一个 WebUI，内部按后端启停 worker 子进程。

每个模型用各自匹配的 Python 环境跑（numpy/transformers 版本冲突无法同进程共存），
网关只做转发与生命周期管理。切换时卸载旧 worker 释放显存。
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))

from app.asr import ASREngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("tts_gateway")

PY_QWEN = ROOT / ".venv-qwen3" / "Scripts" / "python.exe"
PY_COSY = Path(r"D:\AI\CosyVocie\CosyVoice\.venv\Scripts\python.exe")
TORCH_LIB = Path(r"D:\AI\CosyVocie\CosyVoice\.venv\Lib\site-packages\torch\lib")

def get_workers() -> dict:
    """按当前配置生成 worker 规格（端口可经 .env 覆盖）。"""
    from app.config import load_settings

    settings = load_settings()
    return {
        "qwen3": {
            "python": PY_QWEN,
            "port": settings.qwen_worker_port,
            "env": {"MODEL_BACKEND": "qwen3"},
        },
        "cosyvoice3": {
            "python": PY_COSY,
            "port": settings.cosy_worker_port,
            "env": {
                "MODEL_BACKEND": "cosyvoice3",
                "COSYVOICE_ROOT": str(settings.cosyvoice_root),
                "COSYVOICE_MODEL_DIR": settings.cosyvoice_model_dir,
            },
        },
    }

state = {
    "backend": None,
    "process": None,
    "log_handles": {},
    "asr": None,
    "lock": None,
    "switching": False,
}


def worker_url(backend: str) -> str:
    return f"http://127.0.0.1:{get_workers()[backend]['port']}"


def _start_worker(backend: str):
    spec = get_workers()[backend]
    env = os.environ.copy()
    env.update(spec["env"])
    if backend == "cosyvoice3" and TORCH_LIB.is_dir():
        env["PATH"] = str(TORCH_LIB) + os.pathsep + env.get("PATH", "")

    import subprocess

    log = open(LOGS_DIR / f"worker_{backend}.log", "w", encoding="utf-8")
    state["log_handles"][backend] = log
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0

    proc = subprocess.Popen(
        [str(spec["python"]), "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
         "--port", str(spec["port"])],
        cwd=str(ROOT),
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
        creationflags=flags,
    )
    logger.info("启动 worker %s pid=%s port=%s", backend, proc.pid, spec["port"])
    return proc


def _stop_worker() -> None:
    proc = state["process"]
    if proc is None:
        return
    import subprocess

    try:
        proc.terminate()
        proc.wait(timeout=25)
        logger.info("worker 已停止 pid=%s", proc.pid)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=10)
            logger.warning("worker 强制终止 pid=%s", proc.pid)
        except Exception:
            pass
    state["process"] = None


async def _wait_ready(backend: str, timeout: int = 180) -> dict:
    url = worker_url(backend)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    last_error = "unknown"
    while loop.time() < deadline:
        try:
            async with httpx.AsyncClient(timeout=3.0, trust_env=False) as client:
                r = await client.get(url + "/api/status")
                if r.status_code == 200:
                    data = r.json()
                    if data.get("backend") == backend:
                        return data
                    last_error = f"backend mismatch: {data.get('backend')}"
                else:
                    last_error = f"HTTP {r.status_code}"
        except Exception as exc:
            last_error = str(exc)
        await asyncio.sleep(1)
    raise RuntimeError(f"worker {backend} 启动超时: {last_error}（查看 logs/worker_{backend}.log）")


async def switch_backend(backend: str) -> dict:
    if backend not in get_workers():
        raise HTTPException(status_code=400, detail=f"未知后端: {backend}")
    if state["backend"] == backend and state["process"] and state["process"].poll() is None:
        return await _worker_status(backend)

    async with state["lock"]:
        state["switching"] = True
        try:
            _stop_worker()
            state["backend"] = backend
            state["process"] = _start_worker(backend)
            data = await _wait_ready(backend)
            logger.info("后端切换完成: %s", backend)
            return data
        finally:
            state["switching"] = False


async def _worker_status(backend: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
            r = await client.get(worker_url(backend) + "/api/status")
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return {"backend": backend, "sample_rate": 24000, "status": "worker not ready"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["lock"] = asyncio.Lock()
    state["asr"] = ASREngine()
    initial = os.environ.get("MODEL_BACKEND", "qwen3")
    if initial not in get_workers():
        initial = "qwen3"
    await switch_backend(initial)
    yield
    _stop_worker()
    for handle in state["log_handles"].values():
        try:
            handle.close()
        except Exception:
            pass


app = FastAPI(title="TTS_SERVE Gateway", version="2.0.0", lifespan=lifespan)


class SwitchRequest(BaseModel):
    backend: str


@app.get("/")
async def health():
    backend = state["backend"] or "unknown"
    status = await _worker_status(backend) if backend != "unknown" else {}
    return {
        "status": "ok",
        "backend": backend,
        "model_loaded": status.get("backend") == backend,
        "sample_rate": status.get("sample_rate", 24000),
        "switching": state["switching"],
    }


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def ui() -> HTMLResponse:
    return HTMLResponse((Path(__file__).parent / "ui.html").read_text(encoding="utf-8"))


@app.get("/api/gateway_status")
async def gateway_status() -> dict:
    backend = state["backend"] or "unknown"
    data = await _worker_status(backend) if backend != "unknown" else {}
    data["backend"] = backend
    data["switching"] = state["switching"]
    return data


@app.post("/api/switch_backend")
async def api_switch_backend(payload: SwitchRequest) -> dict:
    data = await switch_backend(payload.backend)
    return {"ok": True, **data}


@app.post("/api/asr")
async def api_asr(file: UploadFile = File(...), language: str = Form("zh")) -> dict:
    suffix = Path(file.filename or "").suffix.lower() or ".wav"
    fd, tmp_path = tempfile.mkstemp(prefix="tts_asr_", suffix=suffix)
    with os.fdopen(fd, "wb") as handle:
        shutil.copyfileobj(file.file, handle)
    try:
        text = await asyncio.to_thread(state["asr"].transcribe, tmp_path, language)
        return {"text": text}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"语音识别失败: {exc}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(path: str, request: Request):
    backend = state["backend"]
    if not backend:
        raise HTTPException(status_code=503, detail="后端未就绪")
    url = f"{worker_url(backend)}/{path}"
    body = await request.body()
    headers = {}
    content_type = request.headers.get("content-type")
    if content_type:
        headers["content-type"] = content_type

    async with httpx.AsyncClient(timeout=240.0, trust_env=False) as client:
        try:
            r = await client.request(
                request.method, url, content=body, headers=headers,
                params=dict(request.query_params),
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"转发到 {backend} 失败: {exc}")

    response_headers = {}
    for key in ("X-Generation-Time", "X-Sample-Rate"):
        if key in r.headers:
            response_headers[key] = r.headers[key]
    return Response(content=r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type"), headers=response_headers)
