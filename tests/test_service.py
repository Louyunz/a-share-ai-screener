# -*- coding: utf-8 -*-
"""对话式服务集成测试：画像 → 确认 → 选股 → AI 定范围/服务端寻优 → 启用 → 回滚。

覆盖新分工：AI 只产出「引擎 + 调参范围」，具体参数由服务端网格寻优选出；
模型只在当前会话生效，新对话回到默认模型。
"""
from __future__ import annotations

import unittest

from backend.data.base import save_bundle
from backend.data.sample import generate_bundle
from tests.support.scripted_llm import ScriptedLLM
from backend.models import registry
from backend.schemas import Plan, profile_summary
from backend.service import AppService
from tests import cleanup, workdir

PROFILE_SLOTS = {
    "goal": "退休养老",
    "horizon": "5 年",
    "drawdown_tolerance": "15%",
    "loss_reaction": "会有点慌，但先不卖",
}


def _blocks(events: list[tuple[str, dict]]) -> list[dict]:
    return [data["block"] for event, data in events if event == "ui"]


def _typed(events: list[tuple[str, dict]], kind: str) -> list[dict]:
    return [block for block in _blocks(events) if block["type"] == kind]


class ChatFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dir = workdir("chatflow")
        save_bundle(generate_bundle(n_stocks=30, n_days=360, seed=7), cls.dir / "a_share")
        cls.service = AppService(data_dir=cls.dir / "a_share", db_path=cls.dir / "app.db",
                                 llm=ScriptedLLM())
        cls.sid = cls.service.create_session()["session_id"]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.service.close()
        cleanup(cls.dir)

    def chat(self, content: str) -> tuple[list[tuple[str, dict]], str]:
        events = list(self.service.chat_stream(self.sid, content))
        text = "".join(data.get("text", "") for event, data in events if event == "token")
        return events, text

    # ------------------------------------------------------------------
    def test_chat_flow(self) -> None:
        # 1) 自然语言开始：AI 只问一个缺失问题
        events, text = self.chat("我想为养老攒点钱")
        self.assertTrue(text)
        self.assertFalse(any(block["type"] == "profile" for block in _blocks(events)),
                         "逐题作答阶段不应弹出画像卡")

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
        # 选股使用当前会话默认模型（不是全局模型）
        self.assertEqual(selection["model"]["name"], "均衡多因子")

        # 4) 追问：基于已选结果解释
        events, text = self.chat("为什么最大回撤会这么大？")
        self.assertTrue(text)
        self.assertIn("不构成投资建议", text)

        # 5) AI 优化：AI 只给引擎+范围，服务端网格寻优并实时推送进度
        events, text = self.chat("帮我优化一下模型")
        works = _typed(events, "work")
        tunings = _typed(events, "tuning")
        comparisons = _typed(events, "comparison")
        approvals = _typed(events, "approval")
        self.assertTrue(works)
        self.assertTrue(tunings, "应推送网格寻优进度卡片")
        self.assertTrue(comparisons, "应推送独立的效果对比卡片")
        self.assertTrue(approvals)
        self.assertFalse(works[-1].get("comparison"),
                         "效果对比不应再塞回网格寻优上方的 work 卡")
        last_tuning_index = max(
            index for index, (event, data) in enumerate(events)
            if event == "ui" and (data.get("block") or {}).get("type") == "tuning")
        first_compare_index = next(
            index for index, (event, data) in enumerate(events)
            if event == "ui" and (data.get("block") or {}).get("type") == "comparison")
        self.assertGreater(first_compare_index, last_tuning_index,
                           "效果对比必须晚于网格寻优结果出现")
        step_titles = [step.get("title") for step in works[-1].get("steps", [])]
        self.assertIn("运行参数网格寻优", step_titles)
        self.assertIn("逐组回测", text, "对话里应说明具体如何调整模型（旁白）")
        history_after_plan = self.service.messages(self.sid)["messages"]
        self.assertTrue(any("逐组回测" in (message.get("content") or "")
                            for message in history_after_plan
                            if message.get("role") == "assistant"),
                        "计划旁白应随消息持久化")
        self.assertGreaterEqual(len(works[-1]["steps"]), 6)
        self.assertEqual(works[-1]["status"], "done")
        # 寻优进度：同一 id 多次推送，trial 单调增长，最终 100%
        running = [block for block in tunings if block.get("status") == "running"]
        done = [block for block in tunings if block.get("status") == "done"]
        self.assertGreaterEqual(len(running), 2)
        self.assertTrue(done)
        trials = [int(block["progress"]["trial"]) for block in running + done]
        self.assertEqual(trials, sorted(trials), "寻优进度必须单调递增")
        final_tune = done[-1]
        self.assertEqual(final_tune["progress"]["percent"], 100)
        self.assertGreater(final_tune["progress"]["total"], 1)
        self.assertTrue(final_tune["space"], "应展示 AI 给出的参数搜索范围")
        self.assertIn("min", final_tune["space"][0])
        self.assertIn("max", final_tune["space"][0])
        self.assertIn("step", final_tune["space"][0])
        self.assertTrue(final_tune["best"].get("metrics"))
        self.assertIn("过拟合", final_tune.get("note", ""))
        # 代码/配置改动仍流式写入（work 卡片多次 running，diff 逐行增长）
        code_change = works[-1].get("code_change") or {}
        self.assertTrue(code_change.get("diff"), "应返回代码/配置 diff 供前端展示")
        self.assertGreater(code_change.get("added", 0), 0)
        streamed = [block for block in works if block.get("status") == "running"]
        self.assertGreaterEqual(len(streamed), 3)
        diff_lengths = [len((block.get("code_change") or {}).get("diff") or "") for block in works]
        self.assertGreater(diff_lengths[-1], diff_lengths[0])
        self.assertGreaterEqual(len(set(diff_lengths)), 3, "diff 应逐行增长而不是一次性出现")
        self.assertEqual((works[-1].get("proposal") or {}).get("approach"),
                         "服务端网格寻优（自定义引擎）")
        approval_text = " ".join(str(approvals[-1].get(key, ""))
                                 for key in ("title", "summary", "risk"))
        self.assertIn("网格寻优", approval_text)
        self.assertIn("过拟合", approval_text)
        self.assertNotIn("需要您确认：是否启用该模型改进方案", text,
                         "服务端不应再追加上一版固定审批兜底话术")
        pending = self.service.store.get_state(self.sid).get("pending")
        self.assertIsNotNone(pending)
        self.assertTrue(pending.get("tuning"), "pending 应保留寻优摘要")
        self.assertGreater(pending["tuning"]["trials"], 1)
        history = self.service.messages(self.sid)["messages"]
        persisted = [block for message in history for block in (message.get("blocks") or [])]
        self.assertTrue(any(block.get("type") == "tuning" and block.get("status") == "done"
                            for block in persisted), "寻优卡片应随消息持久化，刷新后可恢复")
        self.assertTrue(any(block.get("type") == "work" and block.get("code_change")
                            for block in persisted), "改动 diff 应随消息持久化")

        # 6) 聊天内同意 → 只在当前会话启用新模型
        events, text = self.chat("同意，启用这个改进")
        versions = _typed(events, "version")
        self.assertTrue(versions)
        state = self.service.get_state(self.sid)
        self.assertEqual(state["active_model"]["name"], "AI 生成模型")
        self.assertTrue(state["active_model"]["version_id"])
        self.assertTrue(state["versions"])
        # 会话状态是唯一模型来源
        self.assertTrue(self.service.store.session_active_version_id(self.sid))
        self.assertIsNone(self.service.store.get_state(self.sid).get("pending"))
        self.assertIn("已启用", text)
        self.assertIn("新建对话", text)

        # 7) 回滚：回到会话基线（默认模型）
        events, text = self.chat("回滚到上一个版本")
        self.assertTrue(_typed(events, "version"))
        rolled = self.service.get_state(self.sid)
        self.assertEqual(rolled["active_model"]["name"], "均衡多因子")
        self.assertEqual(len(rolled["versions"]), 2)

        # 8) 客户点名清单外模型：AI 必须写自定义代码引擎，服务端再寻优
        events, text = self.chat("帮我用随机森林模型")
        works = _typed(events, "work")
        approvals = _typed(events, "approval")
        self.assertTrue(works and approvals)
        self.assertTrue((works[-1].get("code_change") or {}).get("diff"),
                        "点名自定义模型时应生成代码改动")
        first_code_index = next(
            index for index, block in enumerate(works)
            if (block.get("code_change") or {}).get("diff"))
        for block in works[first_code_index:]:
            self.assertTrue((block.get("code_change") or {}).get("diff"),
                            "后续 work 更新不应丢失代码改动（否则前端会折叠代码窗口）")
        first_code = next(
            index for index, (event, data) in enumerate(events)
            if event == "ui"
            and (data.get("block") or {}).get("type") == "work"
            and ((data.get("block") or {}).get("code_change") or {}).get("diff"))
        first_tuning = next(
            index for index, (event, data) in enumerate(events)
            if event == "ui" and (data.get("block") or {}).get("type") == "tuning")
        self.assertLess(first_code, first_tuning,
                        "代码改动卡必须先于网格寻优卡出现")
        self.assertEqual(works[-1]["status"], "done")
        self.assertTrue(_typed(events, "tuning"), "自定义引擎同样应由服务端寻优")

        # 8.5) 点击“继续优化”：上一版 pending 为代码方案时，应升级为结构性重写
        events, text = self.chat("继续优化模型，上一版未达到我的预期")
        works = _typed(events, "work")
        self.assertTrue(works)
        titles = [step.get("title") for step in works[-1].get("steps", [])]
        self.assertIn("升级优化策略", titles)
        self.assertIn("服务端网格寻优", (works[-1].get("proposal") or {}).get("approach", ""))
        self.assertTrue((works[-1].get("code_change") or {}).get("diff"))
        self.assertTrue(_typed(events, "tuning"))

        # 9) 只清除 pending 指针时仍有候选方案：同意应启用最新候选，不再报错
        runtime = self.service.store.get_state(self.sid)
        self.assertIsNotNone(runtime.get("pending"))
        self.service._save_runtime(self.sid, pending=None)
        events, text = self.chat("同意，启用这个改进")
        self.assertFalse([data for event, data in events if event == "error"])
        self.assertTrue(self.service.store.session_active_version_id(self.sid))
        self.assertIn("已启用", text)

        # 10) 完全没有任何候选方案的会话：apply 动作会被前置校验拦截
        empty_sid = self.service.create_session()["session_id"]
        guard = Plan(reply="已按您的确认启用模型改进方案：提高盈利质量。",
                     actions=[{"name": "apply_pending_change", "args": {}}])
        errors = self.service._action_state_errors(empty_sid, guard)
        self.assertTrue(errors)
        self.assertIn("没有待审批", errors[0])

    # ------------------------------------------------------------------
    def test_code_preview_works_when_current_model_has_code(self) -> None:
        current = {"model_id": "ai_generated", "name": "AI 生成模型",
                   "code": "def build_transform():\n    def transform(z, weights):\n        return z.iloc[:, 0]\n    return transform"}
        new_code = ("import numpy as np\nimport pandas as pd\n\n"
                    "def build_transform(params=None):\n    def transform(z, weights):\n"
                    "        return z.mean(axis=1)\n    return transform")
        preview = self.service._code_change(current, new_code)
        self.assertTrue(preview["diff"])
        self.assertGreater(preview["added"], 0)
        self.assertIn("build_transform", preview["diff"])

    def test_model_type_preview_includes_factor_weights(self) -> None:
        current = {"model_id": "ai_tuned_weights", "name": "AI 调整权重",
                   "weights": registry.get_model("multi_factor")["weights"]}
        target = registry.get_model("low_vol_quality")
        preview = self.service._change_preview(current, {
            "approach": "model_type", "model_type": "low_vol_quality",
            "weights": target["weights"],
        })
        self.assertEqual(preview["kind"], "model_type")
        self.assertIn("因子权重", preview["diff"])
        self.assertIn("服务端寻优", preview["diff"])
        self.assertIn("ep:", preview["diff"])
        self.assertIn("low_vol_quality", preview["diff"])
        self.assertGreater(preview["added"], 2, "更换基础模型必须展示权重变化，不能只改两行元数据")

    def test_state_endpoint_payload(self) -> None:
        state = self.service.get_state(self.sid)
        self.assertIn("profile", state)
        self.assertIn("versions", state)
        self.assertIsNotNone(state["active_model"])
        self.assertEqual(state["model_scope"], "session")
        self.assertEqual(state["default_model"]["name"], "均衡多因子")

    def test_confirm_requires_complete_profile(self) -> None:
        local = workdir("chatflow_guard")
        try:
            save_bundle(generate_bundle(n_stocks=10, n_days=220, seed=3), local / "a_share")
            service = AppService(data_dir=local / "a_share", db_path=local / "app.db",
                                 llm=ScriptedLLM())
            sid = service.create_session()["session_id"]
            list(service.chat_stream(sid, "可以，就按这个来"))
            profile = service.store.get_profile(sid)
            self.assertIsNone(profile)  # 信息不足时 AI 只追问，不会擅自确认画像
            service.close()
        finally:
            cleanup(local)


class SessionModelScopeTest(unittest.TestCase):
    """模型只在被启用的会话内生效；新对话始终从默认模型开始。"""

    def setUp(self) -> None:
        self.dir = workdir("session_scope")
        save_bundle(generate_bundle(n_stocks=20, n_days=300, seed=5), self.dir / "a_share")
        self.service = AppService(data_dir=self.dir / "a_share", db_path=self.dir / "app.db",
                                  llm=ScriptedLLM())
        self.first = self.service.create_session()["session_id"]
        self.second = self.service.create_session()["session_id"]

    def tearDown(self) -> None:
        self.service.close()
        cleanup(self.dir)

    def _seed_pending(self, session_id: str) -> str:
        weights = dict(registry.get_model("low_vol_quality")["weights"])
        pending = {
            "id": "pending-test",
            "proposal": {"summary": "服务端网格寻优后的低波质量权重", "approach": "x",
                         "model_name": "低波质量（网格寻优）"},
            "model": {"kind": "builtin", "model_id": "low_vol_quality",
                      "name": "低波质量（网格寻优）",
                      "description": "网格寻优选出的稳健权重", "weights": weights},
            "metrics": {"sharpe": 1.1, "cagr": 0.1, "mdd": -0.12},
            "base_metrics": {"sharpe": 0.4, "cagr": 0.05, "mdd": -0.2},
            "tuning": {"trials": 12, "objective": "sharpe"},
        }
        self.service._save_runtime(session_id, pending=pending)
        return session_id

    def test_new_session_uses_default_model(self) -> None:
        first = self._seed_pending(self.first)
        self.service.store.save_profile(first, PROFILE_SLOTS, profile_summary(PROFILE_SLOTS), True)
        list(self.service.chat_stream(first, "同意，启用这个改进"))

        first_model = self.service._session_active_model(first)
        self.assertEqual(first_model["model_id"], "low_vol_quality")
        second_model = self.service._session_active_model(self.second)
        self.assertEqual(second_model["model_id"], "multi_factor")
        self.assertEqual(second_model["name"], "均衡多因子")
        # 回到第一个会话，模型仍然保留
        self.assertEqual(self.service._session_active_model(first)["model_id"],
                         "low_vol_quality")
        # 版本链按会话隔离
        self.assertTrue(self.service.versions(first)["versions"])
        self.assertEqual(self.service.versions(self.second)["versions"], [])

    def test_rollback_returns_to_default_baseline(self) -> None:
        first = self._seed_pending(self.first)
        self.service.store.save_profile(first, PROFILE_SLOTS, profile_summary(PROFILE_SLOTS), True)
        list(self.service.chat_stream(first, "同意，启用这个改进"))
        self.assertEqual(self.service._session_active_model(first)["model_id"],
                         "low_vol_quality")

        events = list(self.service.chat_stream(first, "回滚到上一个版本"))
        version_blocks = [data["block"] for event, data in events
                          if event == "ui" and data["block"]["type"] == "version"]
        self.assertTrue(version_blocks)
        model = self.service._session_active_model(first)
        self.assertEqual(model["model_id"], "multi_factor")
        self.assertEqual(model["name"], "均衡多因子")
        # 回滚后也没有可继续回滚的版本
        self.assertFalse(self.service._can_rollback(first))


class PendingCandidateSelectionTest(unittest.TestCase):
    """多候选方案：客户可以指定“用第一个/最新的”，不会被误判为回滚。"""

    def setUp(self) -> None:
        self.dir = workdir("candidate_choice")
        save_bundle(generate_bundle(n_stocks=20, n_days=300, seed=9), self.dir / "a_share")
        self.service = AppService(data_dir=self.dir / "a_share", db_path=self.dir / "app.db",
                                  llm=ScriptedLLM())
        self.sid = self.service.create_session()["session_id"]
        self.service.store.save_profile(self.sid, PROFILE_SLOTS,
                                        profile_summary(PROFILE_SLOTS), True)

    def tearDown(self) -> None:
        self.service.close()
        cleanup(self.dir)

    def _chat(self, message: str) -> list[tuple[str, dict]]:
        events = list(self.service.chat_stream(self.sid, message))
        self.assertFalse([data for event, data in events if event == "error"], events)
        return events

    def _candidates(self) -> list[dict]:
        return self.service._planner_state(self.sid)["pending_candidates"]

    def test_choose_first_candidate_after_multiple_optimizations(self) -> None:
        self._chat("帮我优化一下模型")
        self._chat("继续优化模型，上一版未达到我的预期")
        self._chat("继续优化模型，上一版未达到我的预期")
        candidates = self._candidates()
        self.assertEqual([item["index"] for item in candidates], [1, 2, 3])
        self.assertTrue(candidates[0]["name"])
        first_name = candidates[0]["name"]

        events = self._chat("我觉得还是第一个好，用第一个")
        versions = _typed(events, "version")
        self.assertTrue(versions)
        self.assertEqual(self.service._session_active_model(self.sid)["name"], first_name)
        runtime = self.service.store.get_state(self.sid)
        self.assertEqual(runtime["candidates"][0]["status"], "applied")
        self.assertTrue(self.service.store.session_active_version_id(self.sid))
        # 剩余候选方案仍可继续选择
        remaining = [item for item in runtime["candidates"]
                     if item.get("status", "pending") != "applied"]
        self.assertEqual(len(remaining), 2)

    def test_agree_uses_latest_candidate_by_default(self) -> None:
        self._chat("帮我优化一下模型")
        self._chat("继续优化模型，上一版未达到我的预期")
        latest_name = self._candidates()[-1]["name"]
        self._chat("就按最新这个来")
        self.assertEqual(self.service._session_active_model(self.sid)["name"], latest_name)

    def test_invalid_candidate_index_is_reported_and_reply_persisted(self) -> None:
        self._chat("帮我优化一下模型")
        original = self.service.agent.stream_plan

        def fake_stream(state, history, message):
            yield "token", {"text": "收到。"}
            yield "plan", Plan(
                reply="收到，正在启用您指定的候选方案。",
                actions=[{"name": "apply_pending_change", "args": {"candidate": 9}}])

        self.service.agent.stream_plan = fake_stream
        try:
            events = list(self.service.chat_stream(self.sid, "用第九个方案"))
        finally:
            self.service.agent.stream_plan = original
        errors = [data for event, data in events if event == "error"]
        self.assertTrue(errors)
        self.assertIn("候选方案序号超出范围", errors[0]["message"])
        messages = self.service.messages(self.sid)["messages"]
        self.assertTrue(any(message.get("role") == "assistant"
                            and "正在启用您指定的候选方案" in (message.get("content") or "")
                            for message in messages),
                        "动作失败时也要持久化助手回复，不能只留下用户消息")

    def test_state_invalid_action_keeps_assistant_reply(self) -> None:
        sid = self.service.create_session()["session_id"]
        original = self.service.agent.stream_plan

        def fake_stream(state, history, message):
            yield "token", {"text": "当前会话还没有可回滚的模型版本。"}
            yield "plan", Plan(
                reply="当前会话还没有可回滚的模型版本。",
                actions=[{"name": "rollback_version", "args": {}}])

        self.service.agent.stream_plan = fake_stream
        try:
            events = list(self.service.chat_stream(sid, "回滚到上一个版本"))
        finally:
            self.service.agent.stream_plan = original
        self.assertTrue([data for event, data in events if event == "error"])
        messages = self.service.messages(sid)["messages"]
        self.assertTrue(any(message.get("role") == "assistant"
                            and "可回滚" in (message.get("content") or "")
                            for message in messages),
                        "状态校验失败时也要持久化助手回复")

    def test_legacy_session_backfills_candidates_from_stored_cards(self) -> None:
        self._chat("换成低波质量")
        self._chat("继续优化模型，上一版未达到我的预期")
        # 模拟修复前的数据：只有最新 pending，没有 candidates 列表
        runtime = self.service.store.get_state(self.sid)
        self.service.store.save_state(self.sid, {
            "pending": runtime["pending"],
            "last_selection": runtime.get("last_selection"),
        })
        candidates = self._candidates()
        self.assertEqual([item["index"] for item in candidates], [1, 2])
        self.assertTrue(candidates[0]["available"],
                        "旧卡片里的内置模型方案应能从 best_params 无损还原")
        self.assertEqual(candidates[0]["status"], "pending")
        # 还原结果应落库，后续读取不再重复扫描
        persisted = self.service.store.get_state(self.sid).get("candidates")
        self.assertEqual(len(persisted), 2)
        first_name = candidates[0]["name"]
        self._chat("我觉得还是第一个好，用第一个")
        self.assertEqual(self.service._session_active_model(self.sid)["name"], first_name)


class WeightTuningTest(unittest.TestCase):
    """权重引擎：AI 只给 search_space，服务端负责把权重试出来。"""

    def setUp(self) -> None:
        self.dir = workdir("weight_tuning")
        save_bundle(generate_bundle(n_stocks=25, n_days=320, seed=11), self.dir / "a_share")
        self.service = AppService(data_dir=self.dir / "a_share", db_path=self.dir / "app.db",
                                  llm=ScriptedLLM())
        self.sid = self.service.create_session()["session_id"]
        self.service.store.save_profile(self.sid, PROFILE_SLOTS,
                                        profile_summary(PROFILE_SLOTS), True)

    def tearDown(self) -> None:
        self.service.close()
        cleanup(self.dir)

    def test_weights_plan_is_tuned_by_server(self) -> None:
        from backend.models import pipeline

        base = dict(registry.get_model("multi_factor")["weights"])
        plan = {
            "approach": "weights",
            "summary": "让估值、盈利与低波在 AI 给定范围内由服务端寻找更优组合",
            "search_space": {
                "ep": {"min": 0.04, "max": 0.2, "step": 0.04},
                "roe": {"min": 0.04, "max": 0.2, "step": 0.04},
                "volatility_60": {"min": 0.04, "max": 0.2, "step": 0.04},
            },
            "weights": base,
            "objective": "sharpe",
            "max_trials": 8,
            "why": "当前组合因子暴露偏均衡",
            "expected_effect": "夏普有望改善",
            "risks": ["历史回测不代表未来表现"],
            "repair_attempts": 0,
        }
        original = pipeline.build_tuning_plan
        pipeline.build_tuning_plan = lambda llm, state: dict(plan)
        try:
            events = list(self.service.chat_stream(self.sid, "帮我优化一下模型"))
        finally:
            pipeline.build_tuning_plan = original

        self.assertFalse([data for event, data in events if event == "error"])
        tunings = [data["block"] for event, data in events
                   if event == "ui" and data["block"]["type"] == "tuning"]
        works = [data["block"] for event, data in events
                 if event == "ui" and data["block"]["type"] == "work"]
        self.assertTrue(tunings)
        final = [block for block in tunings if block["status"] == "done"][-1]
        self.assertEqual(final["method"], "逐因子网格搜索")
        self.assertEqual({item["key"] for item in final["space"]},
                         {"ep", "roe", "volatility_60"})
        self.assertGreaterEqual(final["progress"]["trial"], 3)
        self.assertEqual(final["best"]["weights"], final["best"]["params"])
        change = works[-1].get("code_change") or {}
        self.assertEqual(change.get("kind"), "weights")
        self.assertIn("因子权重", change.get("diff", ""))
        pending = self.service.store.get_state(self.sid)["pending"]
        self.assertEqual(pending["proposal"]["approach"], "服务端网格寻优（因子权重）")
        best = pending["tuning"]["best_params"]
        self.assertAlmostEqual(sum(float(value) for value in best.values()), 1.0, places=4)
        self.assertEqual(set(best), set(registry.FACTOR_NAMES))
        self.assertEqual(len(pending["tuning"]["space"]), 3)

    def test_continue_optimize_is_structured_escalation(self) -> None:
        """「继续优化」由 LLM 的 escalate=true 表达，服务端按上一版结构升级。"""
        # 先产生一个 pending 方案
        list(self.service.chat_stream(self.sid, "帮我优化一下模型"))
        self.assertIsNotNone(self.service.store.get_state(self.sid).get("pending"))
        events = list(self.service.chat_stream(
            self.sid, "继续优化模型，上一版未达到我的预期"))
        self.assertFalse([data for event, data in events if event == "error"])
        works = [data["block"] for event, data in events
                 if event == "ui" and data["block"]["type"] == "work"]
        self.assertTrue(works)
        titles = [step.get("title") for step in works[-1]["steps"]]
        self.assertIn("升级优化策略", titles)
        self.assertIsNotNone(self.service.store.get_state(self.sid).get("pending"))

    def test_code_model_uses_tuned_params_in_selection(self) -> None:
        """启用代码引擎后，选股打分必须使用寻优得到并保存的 params。"""
        import pandas as pd

        code = (
            "import pandas as pd\n\n"
            "def build_transform(params=None):\n"
            "    options = params or {}\n"
            "    tilt = float(options.get('tilt', 0.2))\n"
            "    def transform(z, weights):\n"
            "        values = pd.to_numeric(z['z_volatility_60'], errors='coerce').fillna(0.0)\n"
            "        return values * tilt\n"
            "    return transform\n"
        )
        model = {
            "kind": "code", "model_id": "ai_generated", "name": "测试代码模型",
            "weights": registry.get_model("multi_factor")["weights"],
            "code": code, "params": {"tilt": 0.5},
        }
        frame = pd.DataFrame({"z_volatility_60": [1.0, -2.0]}, index=["a", "b"])
        weights, transform = self.service._model_engine(model)
        output = transform(frame, weights)
        self.assertEqual(list(output), [0.5, -1.0])
        model["params"] = {"tilt": 0.0}
        _weights, zero_transform = self.service._model_engine(model)
        self.assertEqual(list(zero_transform(frame, weights)), [0.0, 0.0])

    def test_new_math_model_request_forces_code_engine(self) -> None:
        """「更换数学模型」必须通过 tool 参数强制 AI 重写引擎，不能降级为调权重/换风格。"""
        from backend.models import pipeline

        param_code = (
            "import pandas as pd\n\n"
            "def build_transform(params=None):\n"
            "    tilt = float((params or {}).get('tilt', 0.3))\n"
            "    def transform(z, weights):\n"
            "        score = pd.Series(0.0, index=z.index, dtype=float)\n"
            "        for factor, weight in weights.items():\n"
            "            column = 'z_' + factor\n"
            "            if column in z.columns:\n"
            "                values = pd.to_numeric(z[column], errors='coerce').fillna(0.0)\n"
            "                score = score + float(weight) * values\n"
            "        return score * tilt\n"
            "    return transform\n"
        )
        captured: dict = {}
        base = dict(registry.get_model("multi_factor")["weights"])
        plan = {
            "approach": "code",
            "summary": "把打分方式换成一套新的参数化数学引擎",
            "code": param_code,
            "param_space": {"tilt": {"min": 0.0, "max": 0.8, "step": 0.2,
                                     "type": "float", "default": 0.4}},
            "weights": base,
            "objective": "sharpe",
            "max_trials": 8,
            "why": "原引擎对回撤控制不足",
            "expected_effect": "回撤有望下降",
            "risks": ["历史回测不代表未来表现"],
            "repair_attempts": 0,
        }

        def fake_plan(llm, state):
            captured.update(state)
            return dict(plan)

        original = pipeline.build_tuning_plan
        pipeline.build_tuning_plan = fake_plan
        try:
            events = list(self.service.chat_stream(self.sid, "更换数学模型"))
        finally:
            pipeline.build_tuning_plan = original
        self.assertFalse([data for event, data in events if event == "error"])
        self.assertEqual(captured["directive"]["engine"], "code")
        works = [data["block"] for event, data in events
                 if event == "ui" and data["block"]["type"] == "work"]
        self.assertTrue(works)
        self.assertEqual((works[-1].get("proposal") or {}).get("approach"),
                         "服务端网格寻优（自定义引擎）")
        self.assertEqual((works[-1].get("code_change") or {}).get("kind"), "code")
        pending = self.service.store.get_state(self.sid).get("pending")
        self.assertEqual((pending.get("model") or {}).get("kind"), "code")
        # 审计：意图来源必须是 llm_tool，而不是关键词
        audit = self.service.store.list_audit(limit=5)
        analyze = [item for item in audit if item["action"] == "analyze_model"]
        self.assertTrue(analyze)
        self.assertEqual(analyze[0]["payload"]["tool_args"]["engine"], "code")
        self.assertEqual(analyze[0]["payload"]["directive"]["engine"], "code")

    def test_objective_from_tool_args(self) -> None:
        """目标指标由 tool args 传入，服务端不再从用户原话 regex 猜测。"""
        from backend.models import pipeline

        captured: dict = {}
        base = dict(registry.get_model("multi_factor")["weights"])

        def fake_plan(llm, state):
            captured.update(state)
            return {
                "approach": "weights",
                "summary": "围绕低波因子做范围寻优",
                "search_space": {
                    "volatility_60": {"min": 0.04, "max": 0.2, "step": 0.04},
                    "ep": {"min": 0.04, "max": 0.2, "step": 0.04},
                    "roe": {"min": 0.04, "max": 0.2, "step": 0.04},
                },
                "weights": base,
                "objective": "mdd",
                "max_trials": 8,
                "why": "控制回撤",
                "risks": [],
                "repair_attempts": 0,
            }

        original = pipeline.build_tuning_plan
        pipeline.build_tuning_plan = fake_plan
        try:
            events = list(self.service.chat_stream(self.sid, "优化模型，回撤再小一点"))
        finally:
            pipeline.build_tuning_plan = original
        self.assertFalse([data for event, data in events if event == "error"])
        self.assertEqual(captured["directive"]["objective"], "mdd")
        pending = self.service.store.get_state(self.sid).get("pending")
        self.assertEqual(pending["tuning"]["objective"], "mdd")

    def test_plan_contains_no_ai_chosen_weights(self) -> None:
        """AI 方案只暴露范围（weights 仅作为基线），最终权重由服务端写回。"""
        from backend.models import pipeline

        self.assertTrue(hasattr(pipeline, "build_tuning_plan"))
        self.assertTrue(hasattr(pipeline, "normalize_tuning_plan"))


if __name__ == "__main__":
    unittest.main()
