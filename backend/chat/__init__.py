# -*- coding: utf-8 -*-
"""对话层：动作（工具）契约与服务端策略。

本包只放两类内容：

- ``action_schema``：Planner 可调用的工具及其参数 JSON Schema（唯一契约）；
- ``directives``：把工具参数解析成确定性的执行指令（纯函数，可单测）。

服务端不允许再从自然语言里推断意图；意图只来自 LLM 的 tool_calls。
"""
