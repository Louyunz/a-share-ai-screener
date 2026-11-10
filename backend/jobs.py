"""In-process chat jobs. HTTP clients observe jobs; disconnects do not own them."""
from __future__ import annotations

import copy
import queue
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Iterator


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class JobBusyError(ValueError):
    pass


class JobStream:
    def __init__(self) -> None:
        self.queue: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.detached = False

    def publish(self, item) -> None:
        with self.lock:
            if not self.detached:
                self.queue.put(copy.deepcopy(item))

    def __iter__(self):
        return self

    def __next__(self):
        try:
            item = self.queue.get(timeout=1)
        except queue.Empty:
            return "ping", {}
        if item is None:
            self.close()
            raise StopIteration
        return item

    def close(self) -> None:
        with self.lock:
            self.detached = True
            while True:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
            self.queue.put(None)


class ChatJobs:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.jobs: dict[str, dict[str, Any]] = {}

    def state(self, sid: str) -> dict[str, Any] | None:
        with self.lock:
            job = self.jobs.get(sid)
            if job is None:
                return None
            return {key: job[key] for key in
                    ("id", "status", "running", "started_at", "finished_at", "error")}

    def start(self, sid: str, factory: Callable[[], Iterator], on_finish=None) -> JobStream:
        with self.lock:
            if (self.jobs.get(sid) or {}).get("running"):
                raise JobBusyError("当前会话仍在执行，请等待完成或先停止任务")
            stream = JobStream()
            cancel = threading.Event()
            job = {"id": uuid.uuid4().hex, "status": "running", "running": True,
                   "started_at": _now(), "finished_at": None, "error": "",
                   "cancel": cancel, "stream": stream}
            self.jobs[sid] = job
            thread = threading.Thread(target=self._run, args=(job, factory, on_finish),
                                      name="chat-"+sid, daemon=True)
            job["thread"] = thread
            thread.start()
            return stream

    def _run(self, job: dict[str, Any], factory: Callable[[], Iterator], on_finish) -> None:
        stream = job["stream"]
        iterator = None
        started_iteration = False
        status = "completed"
        try:
            iterator = iter(factory())
            while not job["cancel"].is_set():
                try:
                    started_iteration = True
                    event, data = next(iterator)
                except StopIteration:
                    break
                if job["cancel"].is_set():
                    break
                if event == "error":
                    status = "error"
                    with self.lock:
                        job["error"] = str(data.get("message") or "执行失败")
                stream.publish((event, data))
            if job["cancel"].is_set():
                status = "cancelled"
        except Exception as exc:
            status = "error"
            with self.lock:
                job["error"] = str(exc)
            stream.publish(("error", {"message": str(exc), "retryable": False}))
        finally:
            try:
                if iterator is not None and hasattr(iterator, "close"):
                    iterator.close()
                if on_finish is not None and started_iteration:
                    for item in on_finish(status, job["error"]) or []:
                        stream.publish(item)
            finally:
                with self.lock:
                    job.update(status=status, running=False, finished_at=_now())
                stream.publish(None)

    def cancel(self, sid: str) -> dict[str, Any] | None:
        with self.lock:
            job = self.jobs.get(sid)
            if job is not None and job["running"]:
                job["cancel"].set()
                job["status"] = "cancelling"
            return self.state(sid)

    def shutdown(self) -> None:
        with self.lock:
            jobs = list(self.jobs.values())
            for job in jobs:
                if job["running"]:
                    job["cancel"].set()
        for job in jobs:
            job["thread"].join(timeout=5)
