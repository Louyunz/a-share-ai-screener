# -*- coding: utf-8 -*-
"""Agent 规划与合规测试（ScriptedLLM / 脚本化 LLM，不联网）。"""
from __future__ import annotations

import json
import unittest

from backend.agent import Agent, AgentError, ComplianceError, chunk_text, ensure_compliance
from backend.llm import ToolCall, ToolChatResult
from tests.support.scripted_llm import ScriptedLLM
from backend.schemas import profile_payload


def _state(slots: dict | None = None, confirmed: bool = False, **extra) -> dict:
    payload = profile_payload(slots or {}, confirmed)
    state = {"profile": payload, "pending": None, "versions": [], "last_selection": None}
    state.update(extra)
    return state


def _tool_result(reply: str, profile_delta: dict | None = None,
                 calls: list[ToolCall] | None = None) -> ToolChatResult:
    calls = list(calls or [])
    if profile_delta:
        calls.insert(0, ToolCall(id="call_update_profile", name="update_profile",
                                 arguments={"slots": profile_delta}))
    message: dict = {"role": "assistant", "content": reply}
    if calls:
        message["tool_calls"] = [
            {"id": call.id, "type": "function",
             "function": {"name": call.name,
                          "arguments": json.dumps(call.arguments, ensure_ascii=False)}}
            for call in calls
        ]
    return ToolChatResult(content=reply, tool_calls=calls, message=message)


def _stream_result(result: ToolChatResult):
    reply = result.content
    for start in range(0, len(reply), 6):
        yield "content", reply[start:start + 6]
    yield "result", result


class _ContextLLM:
    def __init__(self) -> None:
        self.payload = None

    def chat_json(self, system, messages):
        self.payload = json.loads(messages[-1]["content"])
        return {"reply": "已记录，出现浮亏时您倾向于继续持有、等待修复。",
                "profile_delta": {"loss_reaction": "继续持有等待修复"}}

    def _result(self, messages):
        content = str(messages[-1].get("content") or "")
        try:
            self.payload = json.loads(content)
        except json.JSONDecodeError:
            pass
        return _tool_result("已记录，出现浮亏时您倾向于继续持有、等待修复。",
                            {"loss_reaction": "继续持有等待修复"})

    def chat_tools(self, system, messages, tools, tool_choice="auto"):
        return self._result(messages)

    def chat_tools_stream(self, system, messages, tools, tool_choice="auto"):
        yield from _stream_result(self._result(messages))


class SemanticContextTest(unittest.TestCase):
    def test_plan_uses_update_profile_tool_for_contextual_slot(self) -> None:
        llm = _ContextLLM()
        state = {"profile": {"complete": False, "confirmed": False,
                             "missing": ["亏钱时的反应"], "summary": ""},
                 "pending": None, "versions": [], "last_selection": None}
        history = [{"role": "assistant", "content": "若持有期间真的出现浮亏，您通常会怎么做，比如继续持有等待回升、分批补仓摊低成本，还是先减仓止损？"}]
        plan = Agent(llm).plan(state, history, "继续持有")
        self.assertNotIn("awaiting_slot", llm.payload)
        self.assertNotIn("last_assistant_question", llm.payload)
        self.assertEqual(plan.profile_delta.get("loss_reaction"), "继续持有等待修复")

class AgentPlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = Agent(ScriptedLLM())

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
                        "profile_delta": {}}

            def chat_tools(self, system, messages, tools, tool_choice="auto"):
                call = ToolCall(id="call_1", name="analyze_model",
                                arguments={"engine": "auto", "model_type": "",
                                           "objective": "", "escalate": False})
                return _tool_result(
                    "在开始模型优化前，需要向您确认：您更希望改善回撤还是收益？",
                    calls=[call])

            def chat_tools_stream(self, system, messages, tools, tool_choice="auto"):
                yield from _stream_result(self.chat_tools(system, messages, tools))

            def chat_text(self, system, messages):
                return ""

        slots = {"goal": "退休养老", "horizon": "5 年", "drawdown_tolerance": "15%",
                 "loss_reaction": "会有点慌，但先不卖"}
        plan = Agent(QuestionLLM()).plan(_state(slots, confirmed=True), [], "帮我优化模型")
        self.assertEqual(plan.actions, [])
        self.assertEqual(plan.deferred_intent, "analyze_model")
        self.assertIn("需要向您确认", plan.reply)

    def test_analyze_intent_engine_code(self) -> None:
        slots = {"goal": "退休养老", "horizon": "5 年", "drawdown_tolerance": "15%",
                 "loss_reaction": "会有点慌，但先不卖"}
        plan = self.agent.plan(_state(slots, confirmed=True), [], "更换数学模型")
        action = plan.action("analyze_model")
        self.assertIsNotNone(action)
        self.assertEqual(action["args"]["engine"], "code")

    def test_analyze_intent_model_type(self) -> None:
        slots = {"goal": "退休养老", "horizon": "5 年", "drawdown_tolerance": "15%",
                 "loss_reaction": "会有点慌，但先不卖"}
        plan = self.agent.plan(_state(slots, confirmed=True), [], "换成低波质量")
        action = plan.action("analyze_model")
        self.assertEqual(action["args"]["engine"], "model_type")
        self.assertEqual(action["args"]["model_type"], "low_vol_quality")

    def test_analyze_intent_escalate(self) -> None:
        slots = {"goal": "退休养老", "horizon": "5 年", "drawdown_tolerance": "15%",
                 "loss_reaction": "会有点慌，但先不卖"}
        plan = self.agent.plan(_state(slots, confirmed=True, pending={"summary": "旧方案"}),
                               [], "继续优化模型，上一版未达到我的预期")
        action = plan.action("analyze_model")
        self.assertTrue(action["args"]["escalate"])

    def test_analyze_intent_objective(self) -> None:
        slots = {"goal": "退休养老", "horizon": "5 年", "drawdown_tolerance": "15%",
                 "loss_reaction": "会有点慌，但先不卖"}
        plan = self.agent.plan(_state(slots, confirmed=True), [], "优化模型，回撤再小一点")
        action = plan.action("analyze_model")
        self.assertEqual(action["args"]["objective"], "mdd")

    def test_invalid_tool_args_are_repaired_once(self) -> None:
        slots = {"goal": "退休养老", "horizon": "5 年", "drawdown_tolerance": "15%",
                 "loss_reaction": "会有点慌，但先不卖"}

        class _RepairLLM:
            def __init__(self):
                self.calls = 0

            def _result(self):
                call = ToolCall(
                    id=f"call_{self.calls}",
                    name="analyze_model",
                    arguments={"engine": "magic" if self.calls == 1 else "code"},
                )
                return _tool_result("正在按您的要求优化模型。", calls=[call])

            def chat_tools(self, system, messages, tools, tool_choice="auto"):
                self.calls += 1
                return self._result()

            def chat_tools_stream(self, system, messages, tools, tool_choice="auto"):
                self.calls += 1
                yield from _stream_result(self._result())

            def chat_text(self, system, messages):
                return ""

        llm = _RepairLLM()
        plan = Agent(llm).plan(_state(slots, confirmed=True), [], "更换数学模型")
        self.assertEqual(llm.calls, 2)
        self.assertEqual(plan.action("analyze_model")["args"]["engine"], "code")

    def test_empty_content_raises_error(self) -> None:
        slots = {"goal": "养老", "drawdown_tolerance": "10%", "loss_reaction": "先拿住"}

        class _EmptyContentLLM:
            def chat_tools_stream(self, system, messages, tools, tool_choice="auto"):
                call = ToolCall(id="call_update", name="update_profile",
                                arguments={"slots": {"horizon": "5 年"}})
                yield "result", _tool_result("", calls=[call])

            def chat_tools(self, system, messages, tools, tool_choice="auto"):
                return _tool_result("")

        with self.assertRaises(AgentError) as context:
            list(Agent(_EmptyContentLLM()).stream_plan(
                _state(slots), [], "大概 5 年不用"))
        self.assertEqual(str(context.exception), "模型没有给出回复文本")

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
    def test_chunk_text_splits_stream_output(self) -> None:
        self.assertEqual(list(chunk_text("abcdefgh", 3)), ["abc", "def", "gh"])

    def test_chunk_text_rejects_invalid_size(self) -> None:
        with self.assertRaises(ValueError):
            list(chunk_text("text", 0))

    def test_blocks_promises(self) -> None:
        with self.assertRaises(ComplianceError):
            ensure_compliance("这只股票稳赚，放心买")
        self.assertTrue(ensure_compliance("历史回测不代表未来"))

    def test_blocks_meta_phrases(self) -> None:
        with self.assertRaises(ComplianceError):
            ensure_compliance("先问第一个问题，您的投资期限是多久？")

    def test_empty_reply(self) -> None:
        with self.assertRaises(AgentError):
            ensure_compliance("   ")


class ProfilePayloadTest(unittest.TestCase):
    def test_name_alone_does_not_infer_risk(self) -> None:
        payload = profile_payload({"name": "罗先生"})
        self.assertNotIn("risk_level", payload["slots"])
        self.assertNotIn("风险偏好", payload["summary"])

    def test_risk_level_requires_complete_profile(self) -> None:
        partial = profile_payload({"name": "罗先生", "goal": "养老"})
        self.assertNotIn("risk_level", partial["slots"])
        complete = profile_payload({"name": "罗先生", "goal": "养老", "horizon": "5 年",
                                    "drawdown_tolerance": "10%", "loss_reaction": "先不卖"})
        self.assertIn("risk_level", complete["slots"])
        self.assertIn("风险偏好", complete["summary"])


if __name__ == "__main__":
    unittest.main()
