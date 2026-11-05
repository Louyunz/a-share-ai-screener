# -*- coding: utf-8 -*-
"""Agent / GuardChain / FakeLLM 测试。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.agent import AgentFacade, AgentError, ComplianceError, GuardChain
from backend.llm import FakeLLM, extract_json
from backend.store import Store


class LlmHelperTest(unittest.TestCase):
    def test_extract_json_with_fence(self) -> None:
        data = extract_json('```json\n{"slots": {"risk_level": "balanced"}}\n```')
        self.assertEqual(data["slots"]["risk_level"], "balanced")

    def test_extract_json_invalid(self) -> None:
        with self.assertRaises(Exception):
            extract_json("这不是 JSON")


class GuardChainTest(unittest.TestCase):
    def test_missing_required(self) -> None:
        guard = GuardChain()
        self.assertEqual(guard.missing_required({"risk_level": "balanced"}),
                         ["horizon", "drawdown_tolerance"])

    def test_compliance_blocks(self) -> None:
        guard = GuardChain()
        with self.assertRaises(ComplianceError):
            guard.check_compliance("这只股票稳赚，建议满仓")
        self.assertTrue(guard.check_compliance("历史模拟不代表未来收益"))


class AgentFacadeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "agent.db")
        self.agent = AgentFacade(FakeLLM(), self.store)
        self.sid = self.store.create_session()

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _collect(self, content: str) -> list[tuple[str, dict]]:
        return list(self.agent.stream_chat(self.sid, content))

    def test_profile_dialogue_complete(self) -> None:
        events = self._collect("我是稳健型，打算持有 3 年，能接受 20% 回撤")
        kinds = [e for e, _ in events]
        self.assertIn("slots", kinds)
        self.assertIn("token", kinds)
        self.assertIn("profile_ready", kinds)
        profile = self.store.get_profile(self.sid)
        self.assertIsNotNone(profile)
        self.assertEqual(profile["slots"]["risk_level"], "balanced")
        self.assertFalse(profile["confirmed"])

    def test_profile_dialogue_incomplete(self) -> None:
        events = self._collect("我想稳一点")
        self.assertIn("slots", [e for e, _ in events])
        self.assertNotIn("profile_ready", [e for e, _ in events])
        self.assertGreater(len(self.store.get_messages(self.sid)), 0)

    def test_confirm_profile_requires_complete(self) -> None:
        with self.assertRaises(AgentError):
            self.agent.confirm_profile(self.sid, {"risk_level": "balanced"})

    def test_explain_compliance(self) -> None:
        text = self.agent.explain({"top": [], "metrics": {"sharpe": 1.0}})
        self.assertIn("不构成投资建议", text)


if __name__ == "__main__":
    unittest.main()
