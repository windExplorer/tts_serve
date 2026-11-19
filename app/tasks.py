"""合成任务追踪：记录当前任务阶段与最近历史，供 /api/tasks 与 WebUI 面板使用。

推理由全局锁串行，但请求可能在排队；用返回的任务对象定位，避免并发请求互相串台。
"""
from __future__ import annotations

import itertools
import logging
import threading
import time

logger = logging.getLogger("tts_serve.tasks")

MAX_HISTORY = 50
PREVIEW_LEN = 60
TERMINAL = {"done", "failed"}


class TaskTracker:
    def __init__(self, max_history: int = MAX_HISTORY):
        self._ids = itertools.count(1)
        self._history: list[dict] = []
        self._current: dict | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _preview(text: str) -> str:
        text = (text or "").strip().replace("\n", " ")
        return text[:PREVIEW_LEN] + ("…" if len(text) > PREVIEW_LEN else "")

    def start(self, tts_text: str, voice: str, backend: str, fast_clone: bool) -> dict:
        """请求到达即登记为 queued，返回任务对象供后续更新。"""
        task = {
            "id": f"t{next(self._ids)}",
            "stage": "queued",
            "text": self._preview(tts_text),
            "voice": voice or "(未指定)",
            "backend": backend,
            "fast_clone": bool(fast_clone),
            "language": None,
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "elapsed": 0.0,
            "queued_sec": 0.0,
            "error": None,
            "sample_rate": None,
            "duration": None,
        }
        with self._lock:
            self._current = task
        logger.info(
            "[task %s] 排队 | 后端=%s 音色=%s 极速=%s 文本=%s",
            task["id"], backend, task["voice"], "是" if fast_clone else "否", task["text"],
        )
        return task

    def set_stage(self, task: dict, stage: str, **fields) -> None:
        """更新任务阶段及附加字段；进入 loading 时记 started_at。"""
        with self._lock:
            if task["stage"] in TERMINAL:
                return
            task["stage"] = stage
            if stage == "loading" and task["started_at"] is None:
                task["started_at"] = time.time()
            task.update(fields)

    def finish(self, task: dict, ok: bool, error: str | None = None, **fields) -> None:
        """结束任务，写入历史并打印结构化日志。"""
        with self._lock:
            if task["stage"] in TERMINAL:
                return
            now = time.time()
            task["finished_at"] = now
            task["queued_sec"] = round(
                (task["started_at"] or now) - task["created_at"], 3
            )
            if task["started_at"] is not None:
                task["elapsed"] = round(now - task["started_at"], 3)
            task["stage"] = "done" if ok else "failed"
            task["error"] = error
            task.update(fields)
            self._history.insert(0, dict(task))
            del self._history[MAX_HISTORY:]
        verb = "完成" if ok else "失败"
        logger.info(
            "[task %s] %s | 耗时=%.2fs 排队=%.2fs 后端=%s 音色=%s 语种=%s 时长=%s 文本=%s%s",
            task["id"], verb, task["elapsed"], task["queued_sec"], task["backend"],
            task["voice"], task.get("language") or "-",
            (f"{task['duration']:.2f}s" if task.get("duration") else "-"),
            task["text"], (" | 错误=" + error) if error else "",
        )

    def snapshot(self) -> dict:
        """返回可 JSON 序列化的当前任务与历史快照。"""
        now = time.time()
        with self._lock:
            current = dict(self._current) if self._current else None
            history = [dict(t) for t in self._history]
        if current and current["stage"] not in TERMINAL:
            base = current["started_at"] or current["created_at"]
            current["elapsed"] = round(now - base, 3)
        return {"current": current, "history": history, "max_history": MAX_HISTORY}
