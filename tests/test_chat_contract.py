# -*- coding: utf-8 -*-
"""意图契约测试：动作 schema、服务端策略、以及“禁止关键词路由”的回归断言。"""
from __future__ import annotations

import unittest
from pathlib import Path

from backend.chat.action_schema import (
    ACTION_NAMES,
    TOOL_DEFINITIONS,
    ToolArgsError,
    validate_tool_call,
)
from backend.chat.directives import (
    EngineDirective,
    build_expectation,
    drawdown_tolerance_ratio,
    resolve_directive,
)

BACKEND = Path(__file__).resolve().parent.parent / "backend"


class ActionSchemaTest(unittest.TestCase):
    def test_tool_definitions_cover_all_actions(self) -> None:
        names = [item["function"]["name"] for item in TOOL_DEFINITIONS]
        self.assertEqual(names, list(ACTION_NAMES))
        analyze = next(item for item in TOOL_DEFINITIONS
                       if item["function"]["name"] == "analyze_model")
        properties = analyze["function"]["parameters"]["properties"]
        self.assertIn("engine", properties)
        self.assertEqual(
            set(properties["engine"]["enum"]), {"auto", "weights", "model_type", "code"})
        self.assertEqual(set(properties["objective"]["enum"]),
                         {"", "sharpe", "cagr", "mdd"})
        self.assertIn("escalate", properties)
        self.assertNotIn("actions", properties)

    def test_valid_analyze_args(self) -> None:
        args = validate_tool_call("analyze_model", {
            "engine": "code", "model_type": "", "objective": "mdd",
            "escalate": False, "reason": "构建新的数学模型",
        })
        self.assertEqual(args["engine"], "code")
        self.assertEqual(args["objective"], "mdd")
        self.assertEqual(args["reason"], "构建新的数学模型")

    def test_invalid_analyze_args(self) -> None:
        with self.assertRaises(ToolArgsError):
            validate_tool_call("analyze_model", {"engine": "magic"})
        with self.assertRaises(ToolArgsError):
            validate_tool_call("analyze_model", {"model_type": "not_a_model"})
        with self.assertRaises(ToolArgsError):
            validate_tool_call("analyze_model", {"objective": "profit"})
        with self.assertRaises(ToolArgsError):
            validate_tool_call("analyze_model", {"reason": "x" * 200})

    def test_unknown_tool_and_wrong_type(self) -> None:
        with self.assertRaises(ToolArgsError):
            validate_tool_call("delete_database", {})
        with self.assertRaises(ToolArgsError):
            validate_tool_call("analyze_model", ["not", "an", "object"])

    def test_empty_args_actions_reject_unknown_fields(self) -> None:
        self.assertEqual(validate_tool_call("run_selection", {}), {})
        with self.assertRaises(ToolArgsError):
            validate_tool_call("confirm_profile", {"force": True})

    def test_apply_pending_change_args(self) -> None:
        self.assertEqual(validate_tool_call("apply_pending_change", {}),
                         {"candidate": 0})
        self.assertEqual(validate_tool_call("apply_pending_change", {"candidate": 2}),
                         {"candidate": 2})
        with self.assertRaises(ToolArgsError):
            validate_tool_call("apply_pending_change", {"candidate": -1})
        with self.assertRaises(ToolArgsError):
            validate_tool_call("apply_pending_change", {"candidate": 99})
        with self.assertRaises(ToolArgsError):
            validate_tool_call("apply_pending_change", {"candidate": "first"})
        with self.assertRaises(ToolArgsError):
            validate_tool_call("apply_pending_change", {"force": True})

    def test_apply_pending_tool_schema_exposes_candidate(self) -> None:
        apply_tool = next(item for item in TOOL_DEFINITIONS
                          if item["function"]["name"] == "apply_pending_change")
        properties = apply_tool["function"]["parameters"]["properties"]
        self.assertIn("candidate", properties)
        self.assertEqual(properties["candidate"].get("type"), "integer")

    def test_explain_results_args(self) -> None:
        args = validate_tool_call("explain_results", {"question": "为什么回撤大"})
        self.assertEqual(args["question"], "为什么回撤大")

    def test_update_profile_args(self) -> None:
        args = validate_tool_call("update_profile", {"slots": {"horizon": "5 年"}})
        self.assertEqual(args, {"slots": {"horizon": "5 年"}})
        with self.assertRaises(ToolArgsError):
            validate_tool_call("update_profile", {"slots": {"unknown": "x"}})


class DirectivePolicyTest(unittest.TestCase):
    def test_explicit_engine(self) -> None:
        self.assertEqual(resolve_directive({"engine": "code"}).engine, "code")
        self.assertEqual(
            resolve_directive({"engine": "model_type", "model_type": "low_vol_quality"}),
            EngineDirective("model_type", "low_vol_quality", "sharpe", False, ""))
        self.assertEqual(resolve_directive({"engine": "weights"}).engine, "weights")

    def test_auto_is_auto_without_escalation(self) -> None:
        directive = resolve_directive({"engine": "auto"})
        self.assertEqual(directive.engine, "auto")
        self.assertFalse(directive.escalate)

    def test_escalation_policy_by_previous_structure(self) -> None:
        weights_prev = resolve_directive({"engine": "auto", "escalate": True},
                                         previous_approach="weights")
        self.assertEqual(weights_prev.engine, "model_type")
        code_prev = resolve_directive({"engine": "auto", "escalate": True},
                                      previous_approach="code")
        self.assertEqual(code_prev.engine, "code")
        model_prev = resolve_directive({"engine": "auto", "escalate": True},
                                       previous_approach="model_type")
        self.assertEqual(model_prev.engine, "code")

    def test_explicit_code_wins_over_escalation(self) -> None:
        directive = resolve_directive(
            {"engine": "code", "escalate": True}, previous_approach="weights")
        self.assertEqual(directive.engine, "code")

    def test_objective_passthrough(self) -> None:
        self.assertEqual(resolve_directive({"objective": "mdd"}).objective, "mdd")
        self.assertEqual(resolve_directive({}).objective, "sharpe")

    def test_drawdown_tolerance_from_profile_slot(self) -> None:
        self.assertAlmostEqual(
            drawdown_tolerance_ratio({"drawdown_tolerance": "15%"}), 0.15)
        self.assertAlmostEqual(
            drawdown_tolerance_ratio({"drawdown_tolerance": "最多能接受亏 10% 左右"}), 0.10)
        self.assertAlmostEqual(
            drawdown_tolerance_ratio({"drawdown_tolerance": "0.2"}), 0.2)
        self.assertIsNone(drawdown_tolerance_ratio({"drawdown_tolerance": "看情况"}))
        self.assertIsNone(drawdown_tolerance_ratio(None))

    def test_build_expectation(self) -> None:
        directive = resolve_directive({"engine": "code", "objective": "cagr"})
        expectation = build_expectation(directive, {"drawdown_tolerance": "15%"})
        self.assertEqual(expectation, {"objective": "cagr", "tolerance": 0.15})


class NoKeywordRoutingTest(unittest.TestCase):
    """设计红线：生产路径不得再出现“从自然语言猜意图”的关键词路由。"""

    FORBIDDEN = (
        "_wants_aggressive_retry",
        "_wants_code_change",
        "_wants_new_engine",
        "_requested_model_type",
        "_requested_custom_model",
        "_target_drawdown",
        "MODEL_REQUEST_ALIASES",
        "CUSTOM_MODEL_KEYWORDS",
    )

    def test_service_and_agent_contain_no_keyword_routing(self) -> None:
        for name in ("service.py", "agent.py"):
            source = (BACKEND / name).read_text(encoding="utf-8")
            for pattern in self.FORBIDDEN:
                self.assertNotIn(pattern, source, f"{name} 仍包含关键词路由：{pattern}")

    def test_service_does_not_route_on_reply_text(self) -> None:
        source = (BACKEND / "service.py").read_text(encoding="utf-8")
        self.assertNotIn('"模型改进方案" in plan.reply', source)
        self.assertNotIn('"同意启用" in plan.reply', source)


if __name__ == "__main__":
    unittest.main()
