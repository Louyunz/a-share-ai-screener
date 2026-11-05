# -*- coding: utf-8 -*-
"""事件总线与 SSE 格式。

后端把对话 token、选股进度、调参 trial 统一抽象为事件；每个任务一个 channel，
多个订阅者（浏览器连接）各自持有队列。SSE 采用 `event: <type>\\ndata: <json>\\n\\n`。
"""
from __future__ import annotations

import json
import queue
import threading
from typing import Any


class EventBus:
    def __init__(self) -> None:
        self._channels: dict[str, list[queue.Queue]] = {}
        self._lock = threading.RLock()

    def subscribe(self, channel: str) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._channels.setdefault(channel, []).append(q)
        return q

    def unsubscribe(self, channel: str, q: queue.Queue) -> None:
        with self._lock:
            subscribers = self._channels.get(channel)
            if not subscribers:
                return
            if q in subscribers:
                subscribers.remove(q)
            if not subscribers:
                self._channels.pop(channel, None)

    def publish(self, channel: str, event: str, data: dict[str, Any],
                seq: int | None = None) -> None:
        with self._lock:
            subscribers = list(self._channels.get(channel, []))
        for q in subscribers:
            q.put({"seq": seq or 0, "event": event, "data": data})

    def close(self, channel: str) -> None:
        """向所有订阅者发送结束哨兵（None），并清理通道。"""
        with self._lock:
            subscribers = list(self._channels.pop(channel, []))
        for q in subscribers:
            q.put(None)


def sse_format(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"
