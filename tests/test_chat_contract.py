# -*- coding: utf-8 -*-
"""意图契约测试：动作 schema、工具说明渲染、thought 字段、服务端策略，
以及“禁止关键词路由”的回归断言。"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

from backend.chat.action_schema import (
    ACTION_NAMES,
    TOOL_DEFINITIONS,
    TOOL_SPECS,
    ToolArgsError,
    render_tool_docs,
    tool_definitions,
    validate_tool_call,
)
from backend.chat.directives import (
    EngineDirective,
    build_expectation,
    drawdown_tolerance_ratio,
    resolve_directive,
)

BACKEND = Path(__file__).resolve().parent.parent / "backend"

# 每个工具的最小合法业务参数（不含 thought），用于参数化断言。
MINIMAL_BUSINESS_ARGS: dict[str, dict] = {
    "update_profile": {"slots": {"horizon": "5 年"}},
    "confirm_profile": {},
    "run_selection": {},
    "explain_results": {"question": "为什么回撤大"},
    "analyze_model": {"engine": "auto"},
    "write_model_program": {
        "files": [{"path": "main.py",
                   "content": "def build_transform(params=None):\n    return None\n"}],
    },
    "apply_pending_change": {},
    "rollback_version": {},
}


class ActionSchemaTest(unittest.TestCase):
    def test_tool_definitions_cover_all_actions(self) -> None:
        definitions = tool_definitions()
        # TOOL_DEFINITIONS 是同一份元数据的模块级快照，内容必须一致。
        self.assertEqual(definitions, TOOL_DEFINITIONS)
        names = [item["function"]["name"] for item in definitions]
        self.assertEqual(names, list(ACTION_NAMES))
        analyze = next(item for item in definitions
                       if item["function"]["name"] == "analyze_model")
        properties = analyze["function"]["parameters"]["properties"]
        self.assertIn("engine", properties)
        self.assertEqual(
            set(properties["engine"]["enum"]), {"auto", "weights", "model_type"})
        self.assertEqual(properties["objective"].get("type"), "string")
        self.assertIn("escalate", properties)
        self.assertNotIn("actions", properties)

    def test_valid_analyze_args(self) -> None:
        args = validate_tool_call("analyze_model", {
            "engine": "weights", "model_type": "", "objective": "mdd",
            "escalate": False, "reason": "构建新的数学模型",
        })
        self.assertEqual(args["engine"], "weights")
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
            validate_tool_call("analyze_model", {"reason": "x" * 600})

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
        self.assertEqual(validate_tool_call("apply_pending_change", {"candidate": 99}),
                         {"candidate": 99})
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

    def test_write_model_program_tool_schema(self) -> None:
        program_tool = next(item for item in TOOL_DEFINITIONS
                            if item["function"]["name"] == "write_model_program")
        properties = program_tool["function"]["parameters"]["properties"]
        for key in ("files", "entry", "summary", "param_space", "objective",
                    "max_trials", "config", "packages", "escalate", "reason"):
            self.assertIn(key, properties)
        self.assertEqual(properties["files"]["type"], "array")
        args = validate_tool_call("write_model_program", {
            "files": [{"path": "main.py",
                       "content": "def build_transform():\n    return None\n"}],
            "entry": "main.py",
            "summary": "测试程序",
            "param_space": {"tilt": {"min": 0, "max": 1, "step": 0.5}},
            "objective": "mdd",
            "packages": ["scipy"],
            "escalate": True,
        })
        self.assertEqual(args["entry"], "main.py")
        self.assertEqual(args["files"]["main.py"].splitlines()[0],
                         "def build_transform():")
        self.assertEqual(args["packages"], ["scipy"])
        self.assertTrue(args["escalate"])
        with self.assertRaises(ToolArgsError):
            validate_tool_call("write_model_program", {"files": []})
        with self.assertRaises(ToolArgsError):
            validate_tool_call("write_model_program", {
                "files": [{"path": "main.py", "content": "x"}],
                "entry": "main.py",
                "unknown_field": True,
            })

    def test_explain_results_args(self) -> None:
        args = validate_tool_call("explain_results", {"question": "为什么回撤大"})
        self.assertEqual(args["question"], "为什么回撤大")

    def test_update_profile_args(self) -> None:
        args = validate_tool_call("update_profile", {"slots": {"horizon": "5 年"}})
        self.assertEqual(args, {"slots": {"horizon": "5 年"}})
        with self.assertRaises(ToolArgsError):
            validate_tool_call("update_profile", {"slots": {"unknown": "x"}})


class ToolSpecTest(unittest.TestCase):
    """工具描述/用法/参数模型是同一份元数据（单一事实源）。"""

    def test_specs_follow_action_names_order(self) -> None:
        self.assertEqual([spec.name for spec in TOOL_SPECS], list(ACTION_NAMES))

    def test_specs_bind_model_and_docs_together(self) -> None:
        for spec in TOOL_SPECS:
            self.assertTrue(spec.description.strip(), f"{spec.name} 缺少 description")
            self.assertTrue(spec.usage.strip(), f"{spec.name} 缺少 usage")
            # 参数模型与元数据绑定：每个工具模型都带 thought 字段。
            self.assertIn("thought", spec.model.model_fields,
                          f"{spec.name} 的参数模型缺少 thought")

    def test_definitions_are_built_from_specs(self) -> None:
        definitions = tool_definitions()
        self.assertEqual([item["function"]["name"] for item in definitions],
                         [spec.name for spec in TOOL_SPECS])
        for spec, definition in zip(TOOL_SPECS, definitions):
            function = definition["function"]
            # description 复用元数据描述（后面统一补充 thought 用法说明）
            self.assertTrue(function["description"].startswith(spec.description),
                            f"{spec.name} 的 function description 未复用元数据")
            self.assertIn("thought", function["description"],
                          f"{spec.name} 的 description 未说明 thought")
            # parameters 直接来自绑定的 Pydantic 模型，不另外维护 schema 副本
            schema = spec.model.model_json_schema()
            schema.pop("title", None)
            self.assertEqual(function["parameters"], schema)
            self.assertIn("thought", function["parameters"]["properties"])

    def test_tool_definitions_are_rebuilt_consistently(self) -> None:
        self.assertEqual(tool_definitions(), tool_definitions())
        self.assertEqual(tool_definitions(), TOOL_DEFINITIONS)


class ToolDocsRenderTest(unittest.TestCase):
    """render_tool_docs()：固定顺序、内容完整、可重复稳定、不含 schema 全文。"""

    def test_render_is_repeatable(self) -> None:
        self.assertEqual(render_tool_docs(), render_tool_docs())

    def test_render_order_matches_action_names(self) -> None:
        sections = re.findall(r"^### (\d+)\. `([a-z_]+)`$",
                              render_tool_docs(), re.MULTILINE)
        self.assertEqual([name for _, name in sections], list(ACTION_NAMES))
        self.assertEqual([int(index) for index, _ in sections],
                         list(range(1, len(ACTION_NAMES) + 1)))

    def test_render_contains_every_tool_and_usage_fields(self) -> None:
        docs = render_tool_docs()
        for position, name in enumerate(ACTION_NAMES, start=1):
            self.assertIn(f"`{name}`", docs)
            self.assertIn(f"### {position}. `{name}`", docs)
        for label in ("用途", "何时调用", "关键参数", "行为注意", "thought"):
            self.assertIn(label, docs)

    def test_render_does_not_embed_json_schema(self) -> None:
        docs = render_tool_docs()
        self.assertNotIn('"properties"', docs)
        self.assertNotIn('"$defs"', docs)
        self.assertNotIn('"type": "function"', docs)

    def test_tool_docs_injection_into_planner_prompt_is_stable(self) -> None:
        # planner 提示词在固定位置留了 {{TOOL_DOCS}} 占位符，由 agent._planner_system 注入；
        # 同一份元数据两次注入必须得到完全一致的提示词（可缓存、可测试）。
        template = (BACKEND / "prompts" / "planner.txt").read_text(encoding="utf-8")
        docs = render_tool_docs()
        first = template.replace("{{TOOL_DOCS}}", docs)
        second = template.replace("{{TOOL_DOCS}}", render_tool_docs())
        self.assertIn(docs, first)
        self.assertEqual(first, second)
        self.assertNotIn("{{TOOL_DOCS}}", first)


class ThoughtFieldTest(unittest.TestCase):
    """所有动作参数模型都暴露可选 thought；校验后的执行参数里不含 thought。"""

    def test_all_tool_schemas_expose_optional_thought(self) -> None:
        for definition in TOOL_DEFINITIONS:
            name = definition["function"]["name"]
            parameters = definition["function"]["parameters"]
            properties = parameters.get("properties") or {}
            self.assertIn("thought", properties, f"{name} schema 缺少 thought")
            thought = properties["thought"]
            self.assertEqual(thought.get("type"), "string", name)
            self.assertEqual(thought.get("default"), "", name)
            self.assertEqual(thought.get("maxLength"), 2000, name)
            self.assertIn("thought", str(thought.get("description") or ""), name)
            self.assertNotIn("thought", parameters.get("required") or [],
                             f"{name} 的 thought 应为可选")

    def test_validate_never_returns_thought(self) -> None:
        thought = "客户想换模型，我先确认意图，再调用对应工具。"
        for name in ACTION_NAMES:
            business_args = MINIMAL_BUSINESS_ARGS[name]
            base = validate_tool_call(name, dict(business_args))
            with_thought = validate_tool_call(name, {**business_args, "thought": thought})
            self.assertNotIn("thought", with_thought,
                             f"{name} 的 clean 结果不应包含 thought")
            self.assertEqual(with_thought, base,
                             f"{name} 写入 thought 不应影响业务参数")

    def test_thought_does_not_shadow_business_args(self) -> None:
        args = validate_tool_call("analyze_model", {
            "engine": "weights",
            "objective": "mdd",
            "thought": "客户只要求调整权重，目标回撤优先。",
        })
        self.assertEqual(args["engine"], "weights")
        self.assertEqual(args["objective"], "mdd")
        self.assertNotIn("thought", args)

    def test_thought_only_call_for_empty_actions(self) -> None:
        self.assertEqual(
            validate_tool_call("run_selection", {"thought": "画像已确认，开始选股。"}), {})
        self.assertEqual(
            validate_tool_call("confirm_profile", {"thought": "客户确认画像。"}), {})
        self.assertEqual(
            validate_tool_call("rollback_version", {"thought": "客户要求回滚。"}), {})

    def test_thought_length_limit(self) -> None:
        self.assertEqual(validate_tool_call("run_selection", {"thought": "思" * 2000}), {})
        with self.assertRaises(ToolArgsError):
            validate_tool_call("run_selection", {"thought": "思" * 2001})


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
        self.assertEqual(code_prev.engine, "model_type")
        model_prev = resolve_directive({"engine": "auto", "escalate": True},
                                       previous_approach="model_type")
        self.assertEqual(model_prev.engine, "model_type")

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
