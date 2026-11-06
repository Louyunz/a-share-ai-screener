# -*- coding: utf-8 -*-
"""对话式服务集成测试：一次会话内完成画像→确认→选股→追问→AI优化→启用→回滚。"""
from __future__ import annotations

import unittest

from backend.data.base import save_bundle
from backend.data.sample import generate_bundle
from backend.llm import FakeLLM
from backend.schemas import Plan
from backend.service import AppService
from tests import cleanup, workdir


def _blocks(events: list[tuple[str, dict]]) -> list[dict]:
    return [data["block"] for event, data in events if event == "ui"]


class ChatFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dir = workdir("chatflow")
        save_bundle(generate_bundle(n_stocks=30, n_days=360, seed=7), cls.dir / "a_share")
        cls.service = AppService(data_dir=cls.dir / "a_share", db_path=cls.dir / "app.db",
                                 llm=FakeLLM())
        cls.sid = cls.service.create_session()["session_id"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.service.close()
        cleanup(cls.dir)

    def chat(self, content: str) -> tuple[list[tuple[str, dict]], str]:
        events = list(self.service.chat_stream(self.sid, content))
        text = "".join(data.get("text", "") for event, data in events if event == "token")
        return events, text

    def test_chat_flow(self) -> None:
        # 1) 自然语言开始：AI 只问一个缺失问题
        events, text = self.chat("我想为养老攒点钱")
        self.assertTrue(text)
        self.assertTrue(any(block["type"] == "profile" for block in _blocks(events)))

        # 2) 补齐信息：AI 汇总并请确认
        events, text = self.chat("大概 5 年不用这笔钱，最多能接受亏 15%，真亏了会有点慌但先不卖")
        profile_block = next(block for block in _blocks(events) if block["type"] == "profile")
        self.assertTrue(profile_block["complete"])
        self.assertIn("需要您确认", text)

        # 3) 对话确认后 AI 自己完成确认 + 选股
        events, text = self.chat("可以，就按这个来")
        blocks = _blocks(events)
        selection = next(block for block in blocks if block["type"] == "selection")
        self.assertEqual(len(selection["top"]), 10)
        self.assertIn("sharpe", selection["metrics"])
        self.assertTrue(selection["explanation"])
        active = self.service.store.get_profile(self.sid)
        self.assertTrue(active["confirmed"])
        self.assertIsNotNone(self.service.store.get_state(self.sid).get("last_selection"))

        # 4) 追问：基于已选结果解释
        events, text = self.chat("为什么最大回撤会这么大？")
        self.assertTrue(text)
        self.assertIn("不构成投资建议", text)

        # 5) AI 主动优化：展示工作步骤，生成待审批方案
        events, text = self.chat("帮我优化一下模型")
        blocks = _blocks(events)
        works = [block for block in blocks if block["type"] == "work"]
        approvals = [block for block in blocks if block["type"] == "approval"]
        self.assertTrue(works)
        self.assertGreaterEqual(len(works[-1]["steps"]), 5)
        self.assertEqual(works[-1]["status"], "done")
        code_change = works[-1].get("code_change") or {}
        self.assertTrue(code_change.get("diff"), "AI 改代码时应返回代码 diff 供前端展示")
        self.assertGreater(code_change.get("added", 0), 0)
        # 流式写入：同一 work 卡片应多次推送，code_change.diff 逐行增长
        streamed = [block for block in works if block.get("status") == "running"]
        self.assertGreaterEqual(len(streamed), 3, "代码改动应通过多条 running work 事件流式推送")
        diff_lengths = [len((block.get("code_change") or {}).get("diff") or "") for block in works]
        self.assertGreater(diff_lengths[-1], diff_lengths[0])
        self.assertGreaterEqual(len(set(diff_lengths)), 3, "代码 diff 应逐行增长而不是一次性出现")
        self.assertTrue(approvals)
        self.assertNotIn("需要您确认：是否启用该模型改进方案", text,
                         "服务端不应再追加上一版固定审批兜底话术")
        self.assertIsNotNone(self.service.store.get_state(self.sid).get("pending"))
        history = self.service.messages(self.sid)["messages"]
        persisted = [block for message in history for block in (message.get("blocks") or [])]
        self.assertTrue(any(block.get("type") == "work" and block.get("code_change")
                            for block in persisted), "AI 代码改动应随消息持久化，刷新后可恢复")

        # 6) 聊天内同意 → AI 应用并启用新模型
        events, text = self.chat("同意，启用这个改进")
        versions = [block for block in _blocks(events) if block["type"] == "version"]
        self.assertTrue(versions)
        active = self.service.store.active_version()
        self.assertIsNotNone(active)
        self.assertIsNone(self.service.store.get_state(self.sid).get("pending"))
        self.assertIn("已启用", text)

        # 7) 回滚
        events, text = self.chat("回滚到上一个版本")
        self.assertTrue([block for block in _blocks(events) if block["type"] == "version"])
        rolled = self.service.store.active_version()
        self.assertNotEqual(rolled["id"], active["id"])

        # 8) 客户点名清单外模型：AI 必须写自定义代码实现，不能拒绝或硬套内置模型
        events, text = self.chat("帮我用随机森林模型")
        works = [block for block in _blocks(events) if block["type"] == "work"]
        approvals = [block for block in _blocks(events) if block["type"] == "approval"]
        self.assertTrue(works and approvals)
        self.assertTrue((works[-1].get("code_change") or {}).get("diff"), "点名自定义模型时应生成代码改动")
        self.assertEqual(works[-1]["status"], "done")

        # 9) 没有 pending 时模型仍要求“同意启用”：不能报错，应自动重新分析并生成候选
        self.service._save_runtime(self.sid, pending=None)
        original_plan = self.service.agent.plan
        try:
            self.service.agent.plan = lambda state, history, message: Plan(
                reply="已按您的确认启用模型改进方案：提高盈利质量。",
                actions=[{"name": "apply_pending_change", "args": {}}],
            )
            events, text = self.chat("同意")
        finally:
            self.service.agent.plan = original_plan
        self.assertFalse([data for event, data in events if event == "error"])
        self.assertTrue([block for block in _blocks(events) if block["type"] == "work"])
        self.assertIn("没有等待您确认", text)
        self.assertIsNotNone(self.service.store.get_state(self.sid).get("pending"))

    def test_state_endpoint_payload(self) -> None:
        state = self.service.get_state(self.sid)
        self.assertIn("profile", state)
        self.assertIn("versions", state)
        self.assertIsNotNone(state["active_model"])

    def test_confirm_requires_complete_profile(self) -> None:
        local = workdir("chatflow_guard")
        try:
            save_bundle(generate_bundle(n_stocks=10, n_days=220, seed=3), local / "a_share")
            service = AppService(data_dir=local / "a_share", db_path=local / "app.db",
                                 llm=FakeLLM())
            sid = service.create_session()["session_id"]
            list(service.chat_stream(sid, "可以，就按这个来"))
            profile = service.store.get_profile(sid)
            self.assertIsNone(profile)  # 信息不足时 AI 只追问，不会擅自确认画像
            service.close()
        finally:
            cleanup(local)


if __name__ == "__main__":
    unittest.main()
