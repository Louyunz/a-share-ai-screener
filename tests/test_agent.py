# -*- coding: utf-8 -*-
"""Agent 规划与合规测试（FakeLLM，不联网）。"""
from __future__ import annotations

import unittest

from backend.agent import (Agent, AgentError, ComplianceError, ensure_compliance,
                          has_mechanical_style, is_explanation_question,
                          strip_confirmation_tail, trim_duplicate_confirmation)
from backend.llm import FakeLLM
from backend.schemas import profile_payload


def _state(slots: dict | None = None, confirmed: bool = False, **extra) -> dict:
    payload = profile_payload(slots or {}, confirmed)
    state = {"profile": payload, "pending": None, "versions": [], "last_selection": None}
    state.update(extra)
    return state



class _RewriteLLM:
    def __init__(self, first_reply: str) -> None:
        self.first_reply = first_reply
        self.text_calls = 0

    def chat_json(self, system, messages):
        return {"reply": self.first_reply, "actions": [], "profile_delta": {}}

    def chat_text(self, system, messages):
        self.text_calls += 1
        return ("刚才问的是您的「亏损反应」，也就是市场下跌、持仓出现浮亏时您通常会怎么处理。"
                "它没有标准答案，用于判断您对波动的真实承受方式。您用自己的话说即可，"
                "继续持有、补仓或减仓都只是常见例子，不必按固定选项回答。")


class ExplanationStyleTest(unittest.TestCase):
    def test_mechanical_reply_is_regenerated(self) -> None:
        reply = ("您指的是刚才那个问题。它用于判断您在市场下跌时的真实反应，直接影响适当性匹配结果。"
                 "请选择一项：一是继续持有等待修复，二是分批补仓摊低成本，三是止损离场。回复对应选项即可。")
        self.assertTrue(has_mechanical_style(reply))
        llm = _RewriteLLM(reply)
        agent = Agent(llm)
        state = {"profile": {"complete": False, "confirmed": False,
                             "missing": ["亏损反应"], "summary": ""},
                 "pending": None, "versions": [], "last_selection": None}
        history = [{"role": "assistant", "content": "真亏了会怎么做？例如继续持有等待修复，或者分批补仓摊低成本。"}]
        plan = agent.plan(state, history, "这些都是什么意思")
        self.assertEqual(llm.text_calls, 1)
        self.assertIn("亏损反应", plan.reply)
        self.assertNotIn("请选择一项", plan.reply)
        self.assertNotIn("回复对应选项即可", plan.reply)
        self.assertNotIn("您指的是刚才那个问题", plan.reply)

    def test_interrogation_style_is_regenerated_without_question(self) -> None:
        reply = ("回撤指的是账户净值从阶段高点回落到低点的幅度。结合您三年后购房的用途，"
                 "请按实际情况描述：持有期间若出现阶段性亏损，您最多能接受本金回落多少？")
        self.assertTrue(has_mechanical_style(reply))
        self.assertTrue(is_explanation_question("什么是回撤"))
        llm = _RewriteLLM(reply)
        agent = Agent(llm)
        state = {"profile": {"complete": False, "confirmed": False,
                             "missing": ["能接受的最大亏损"], "summary": ""},
                 "pending": None, "versions": [], "last_selection": None}
        plan = agent.plan(state, [], "什么是回撤")
        self.assertEqual(llm.text_calls, 1)
        self.assertNotIn("请按实际情况描述", plan.reply)
        self.assertNotIn("您最多能接受", plan.reply)

    def test_normal_reply_is_not_rewritten(self) -> None:
        llm = _RewriteLLM("这笔钱大概多久可以不用？您按实际情况说一个大致年限即可。")
        agent = Agent(llm)
        state = {"profile": {"complete": False, "confirmed": False,
                             "missing": ["投资期限"], "summary": ""},
                 "pending": None, "versions": [], "last_selection": None}
        plan = agent.plan(state, [], "这个是什么意思")
        self.assertEqual(llm.text_calls, 0)
        self.assertTrue(plan.reply)
        self.assertTrue(is_explanation_question("这些都是什么意思"))
        self.assertFalse(is_explanation_question("帮我选股"))

class _ContextLLM:
    def __init__(self) -> None:
        self.payload = None

    def chat_json(self, system, messages):
        import json
        self.payload = json.loads(messages[-1]["content"])
        return {"reply": "已记录，出现浮亏时您倾向于继续持有、等待修复。",
                "actions": [], "profile_delta": {"loss_reaction": "继续持有等待修复"}}


class SemanticContextTest(unittest.TestCase):
    def test_plan_passes_question_and_awaiting_slot_to_llm(self) -> None:
        llm = _ContextLLM()
        state = {"profile": {"complete": False, "confirmed": False,
                             "missing": ["亏钱时的反应"], "summary": ""},
                 "pending": None, "versions": [], "last_selection": None}
        history = [{"role": "assistant", "content": "若持有期间真的出现浮亏，您通常会怎么做，比如继续持有等待回升、分批补仓摊低成本，还是先减仓止损？"}]
        plan = Agent(llm).plan(state, history, "继续持有")
        self.assertEqual(llm.payload.get("awaiting_slot"), "loss_reaction")
        self.assertIn("浮亏", llm.payload.get("last_assistant_question", ""))
        self.assertEqual(plan.profile_delta.get("loss_reaction"), "继续持有等待修复")

class ConfirmationDedupTest(unittest.TestCase):
    def test_duplicate_confirmation_is_removed(self) -> None:
        text = ("已按您的要求完成回测，最大回撤由 19% 收窄至 16%。\n\n"
                "需要您确认：是否启用该模型改进方案？请回复「同意启用」应用该方案。\n\n"
                "需要您确认：是否启用该模型改进方案？请回复「同意启用」应用该方案。")
        cleaned = trim_duplicate_confirmation(text)
        self.assertEqual(cleaned.count("需要您确认"), 1)
        self.assertIn("19%", cleaned)

    def test_strip_confirmation_tail_keeps_facts(self) -> None:
        note = "已生成候选方案，最大回撤 16%。需要您确认：是否启用？"
        self.assertEqual(strip_confirmation_tail(note), "已生成候选方案，最大回撤 16%")


class AgentPlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = Agent(FakeLLM())

    def test_ask_for_missing_slot(self) -> None:
        plan = self.agent.plan(_state({"goal": "养老"}), [], "我想为养老攒点钱")
        self.assertEqual(plan.actions, [])
        self.assertTrue(plan.reply)
        self.assertIn("多久", plan.reply)

    def test_confirm_and_run_after_user_agrees(self) -> None:
        slots = {"goal": "退休养老", "horizon": "5 年", "drawdown_tolerance": "15%",
                 "loss_reaction": "会有点慌，但先不卖"}
        plan = self.agent.plan(_state(slots), [], "可以，按这个来")
        self.assertIn("confirm_profile", plan.action_names())
        self.assertIn("run_selection", plan.action_names())
        self.assertIn("已收到确认", plan.reply)

    def test_profile_confirmation_hint_is_explicit(self) -> None:
        slots = {"goal": "退休养老", "horizon": "5 年", "drawdown_tolerance": "15%",
                 "loss_reaction": "会有点慌，但先不卖"}
        plan = self.agent.plan(_state(slots), [], "今天天气不错")
        self.assertEqual(plan.actions, [])
        self.assertIn("需要您确认", plan.reply)
        self.assertIn("回复「确认」", plan.reply)

    def test_question_defers_actions(self) -> None:
        class QuestionLLM:
            def chat_json(self, system, messages):
                return {"reply": "在开始模型优化前，需要向您确认：您更希望改善回撤还是收益？",
                        "actions": [{"name": "analyze_model"}], "profile_delta": {}}

            def chat_text(self, system, messages):
                return ""

        slots = {"goal": "退休养老", "horizon": "5 年", "drawdown_tolerance": "15%",
                 "loss_reaction": "会有点慌，但先不卖"}
        plan = Agent(QuestionLLM()).plan(_state(slots, confirmed=True), [], "帮我优化模型")
        self.assertEqual(plan.actions, [])
        self.assertEqual(plan.deferred_intent, "analyze_model")
        self.assertIn("需要向您确认", plan.reply)

    def test_pending_intent_executes_after_answer(self) -> None:
        slots = {"goal": "退休养老", "horizon": "5 年", "drawdown_tolerance": "15%",
                 "loss_reaction": "会有点慌，但先不卖"}
        state = _state(slots, confirmed=True, pending_intent="analyze_model")
        plan = self.agent.plan(state, [], "回撤小一点更重要")
        self.assertIn("analyze_model", plan.action_names())

    def test_pending_proposal_does_not_get_standard_confirmation(self) -> None:
        slots = {"goal": "退休养老", "horizon": "5 年", "drawdown_tolerance": "15%",
                 "loss_reaction": "会有点慌，但先不卖"}
        plan = self.agent.plan(_state(slots, confirmed=True,
                                     pending={"summary": "换低波模型"}),
                               [], "今天市场怎么样")
        self.assertEqual(plan.actions, [])
        self.assertNotIn("需要您确认：是否启用该模型改进方案", plan.reply)
        self.assertNotIn("请回复「同意启用」", plan.reply)

    def test_optimize_requires_confirmed_profile(self) -> None:
        slots = {"goal": "退休养老", "horizon": "5 年", "drawdown_tolerance": "15%",
                 "loss_reaction": "会有点慌，但先不卖"}
        plan = self.agent.plan(_state(slots, confirmed=True), [], "帮我优化一下模型")
        self.assertIn("analyze_model", plan.action_names())

    def test_apply_pending_when_agreeing(self) -> None:
        slots = {"goal": "退休养老", "horizon": "5 年", "drawdown_tolerance": "15%",
                 "loss_reaction": "会有点慌，但先不卖"}
        plan = self.agent.plan(_state(slots, confirmed=True,
                                      pending={"summary": "换低波模型"}),
                               [], "同意，启用吧")
        self.assertIn("apply_pending_change", plan.action_names())

    def test_rollback(self) -> None:
        slots = {"goal": "退休养老", "horizon": "5 年", "drawdown_tolerance": "15%",
                 "loss_reaction": "会有点慌，但先不卖"}
        plan = self.agent.plan(_state(slots, confirmed=True,
                                      versions=[{"id": "v1", "name": "旧模型"}]),
                               [], "回滚到上一个版本")
        self.assertIn("rollback_version", plan.action_names())

    def test_profile_delta_extraction(self) -> None:
        plan = self.agent.plan(_state({}), [], "我想 3 年后买房，最多能接受亏 10%")
        self.assertEqual(plan.profile_delta.get("goal"), "攒钱买房")
        self.assertEqual(plan.profile_delta.get("horizon"), "3 年")
        self.assertEqual(plan.profile_delta.get("drawdown_tolerance"), "10%")

    def test_explain_is_plain_and_compliant(self) -> None:
        text = self.agent.explain({"top": [{"code": "600000.SH"}],
                                   "metrics": {"cagr": 0.1, "mdd": -0.2, "sharpe": 1.1}})
        self.assertIn("不构成投资建议", text)
        self.assertIn("结论", text)
        self.assertIn("依据", text)


class ComplianceTest(unittest.TestCase):
    def test_blocks_promises(self) -> None:
        with self.assertRaises(ComplianceError):
            ensure_compliance("这只股票稳赚，放心买")
        self.assertTrue(ensure_compliance("历史回测不代表未来"))

    def test_strips_meta_phrases(self) -> None:
        cleaned = ensure_compliance("先问第一个问题，您的投资期限是多久？")
        self.assertNotIn("先问第一个问题", cleaned)
        self.assertIn("投资期限", cleaned)

    def test_empty_reply(self) -> None:
        with self.assertRaises(AgentError):
            ensure_compliance("   ")


if __name__ == "__main__":
    unittest.main()
