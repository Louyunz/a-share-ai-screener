"""Real provider settings, independent ReAct limits and narration persistence."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.agent import Agent, AgentError
from backend.data.sample import generate_bundle
from backend.llm import LLMError, OpenAICompatClient, ToolChatResult
from backend.schemas import Plan, ValidationError
from backend.service import AppService
from tests import cleanup, workdir
from tests.support.scripted_llm import ScriptedLLM


class ProviderSettingsTest(unittest.TestCase):
    def test_all_completion_paths_send_real_settings(self):
        client = OpenAICompatClient('test', 'https://api.deepseek.com/v1', 'deepseek-chat')
        tools = [{'type': 'function', 'function': {'name': 'test', 'parameters': {}}}]
        captured = []

        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self):
                return json.dumps({'choices': [{'message': {'content': '{"text":"ok"}'}}]}).encode()
            def __iter__(self):
                yield b'data: {"choices":[{"delta":{"reasoning_content":"private reasoning"}}]}\n'
                yield b'data: {"choices":[{"delta":{"content":"ok"}}]}\n'
                yield b'data: [DONE]\n'

        def open_request(request, **kwargs):
            captured.append(json.loads(request.data))
            return Response()

        with patch('backend.llm.urllib.request.urlopen', side_effect=open_request):
            for budget in ('off', 'low', 'high', 'max'):
                configured = client.with_thinking_budget(budget)
                configured.chat_json('system', [])
                configured.chat_text('system', [])
                list(configured.chat_text_stream('system', []))
                configured.chat_tools('system', [], tools)
                events = list(configured.chat_tools_stream('system', [], tools))
                result = events[-1][1]
                self.assertEqual(result.message['reasoning_content'], 'private reasoning')
                self.assertEqual(result.content, 'ok')
                for payload in captured[-5:]:
                    self.assertEqual(payload['thinking']['type'], 'disabled' if budget == 'off' else 'enabled')
                    if budget == 'off':
                        self.assertNotIn('reasoning_effort', payload)
                    else:
                        self.assertEqual(payload['reasoning_effort'], budget)
                        self.assertNotIn('temperature', payload)
        self.assertFalse(hasattr(client, 'thinking_budget'))
        low = client.with_thinking_budget('low')
        high = client.with_thinking_budget('max')
        self.assertEqual(low.thinking_budget, 'low')
        self.assertEqual(high.thinking_budget, 'max')

    def test_unsupported_provider_is_not_silently_ignored(self):
        client = OpenAICompatClient('test', 'https://other.example/v1', 'model')
        with self.assertRaises(LLMError):
            client.with_thinking_budget('high')
        payload = {'temperature': 0.6}
        client.with_thinking_budget('off')._configure_thinking(payload)
        self.assertEqual(payload, {'temperature': 0.6})

    def test_repair_messages_preserve_provider_reasoning(self):
        result = ToolChatResult(content='', tool_calls=[], message={
            'role': 'assistant', 'content': '', 'reasoning_content': 'protocol reasoning'})
        self.assertEqual(Agent._repair_messages('input', result, ['bad'])[1]['reasoning_content'],
                         'protocol reasoning')
        self.assertEqual(Agent._content_repair_messages('input', result)[1]['reasoning_content'],
                         'protocol reasoning')


class ChatSettingsTest(unittest.TestCase):
    def setUp(self):
        self.root = workdir('thinking_settings')
        self.service = AppService(bundle=generate_bundle(n_stocks=8, n_days=160, seed=1),
                                  db_path=self.root/'app.db', llm=ScriptedLLM())
        self.sid = self.service.create_session()['session_id']
        self.service.store.save_state(self.sid, {'last_selection': {'metrics': {}, 'top': []}})
        self.calls = 0

        def stream_plan(*args, **kwargs):
            self.calls += 1
            yield 'plan', Plan(reply='', actions=[{
                'name': 'explain_results', 'args': {'question': str(self.calls)}}])

        def stream_final(*args, **kwargs):
            yield 'token', 'final reply'

        self.service.agent = SimpleNamespace(stream_plan=stream_plan, stream_final=stream_final,
                                             explain=lambda *a: 'tool explanation')

    def tearDown(self):
        self.service.close()
        cleanup(self.root)

    def test_thinking_off_still_allows_three_tool_rounds(self):
        events = list(self.service.chat_stream(self.sid, 'explain', thinking_budget='off', react_max_steps=3))
        self.assertEqual(self.calls, 3)
        self.assertFalse([data for event, data in events if event == 'error'])

    def test_tool_only_reply_survives_history_reload(self):
        events = list(self.service.chat_stream(self.sid, 'explain', thinking_budget='high', react_max_steps=1))
        self.assertEqual(self.calls, 1)
        self.assertIn(('token', {'text': 'tool explanation'}), events)
        # Reopen the database, as a page reload or a process restart would do.
        from backend.store import Store
        store = Store(self.root/'app.db')
        try:
            messages = store.get_messages(self.sid)
        finally:
            store.close()
        self.assertEqual([item['role'] for item in messages], ['user', 'assistant'])
        self.assertEqual(messages[-1]['content'], 'tool explanation')

    def test_refresh_during_tool_reply_keeps_every_visible_chunk(self):
        explanation = 'This explanation has multiple chunks and the next planning step is still pending.'
        self.service.agent.explain = lambda *a: explanation
        stream = self.service.chat_stream(self.sid, 'explain', react_max_steps=3)
        visible = ''
        from backend.store import Store
        reader = Store(self.root/'app.db')
        try:
            for event, data in stream:
                if event != 'token':
                    continue
                visible += data['text']
                messages = reader.get_messages(self.sid)
                self.assertEqual([m['role'] for m in messages], ['user', 'assistant'])
                self.assertEqual(messages[-1]['content'], visible.strip())
                if visible == explanation:
                    break
            # Refresh/abort closes the connection before ReAct reaches its end.
            stream.close()
            self.assertEqual(reader.get_messages(self.sid)[-1]['content'], explanation)
        finally:
            stream.close()
            reader.close()

    def test_stream_checkpoints_update_one_history_message(self):
        list(self.service.chat_stream(self.sid, 'explain', react_max_steps=3))
        messages = self.service.store.get_messages(self.sid)
        self.assertEqual([m['role'] for m in messages], ['user', 'assistant'])
        self.assertIn('final reply', messages[-1]['content'])
        self.assertEqual(messages[-1]['content'].count('tool explanation'), 3)

    def test_program_preview_without_text_is_persisted_before_next_step(self):
        def preview(*args, **kwargs):
            yield 'program', {'id': 'program-preview', 'status': 'writing',
                              'files': {'main.py': 'def build_transform(params=None):\n    pass\n'}}
            yield 'plan', Plan(reply='')
        self.service.agent.stream_plan = preview
        stream = self.service.chat_stream(self.sid, 'write code', react_max_steps=1)
        try:
            event, data = next(stream)
            self.assertEqual(event, 'ui')
            history = self.service.messages(self.sid)['messages']
            self.assertEqual(history[-1]['role'], 'assistant')
            self.assertEqual(history[-1]['content'], '')
            self.assertEqual(history[-1]['blocks'][0]['id'], data['block']['id'])
        finally:
            stream.close()

    def test_tool_narration_is_saved_if_next_plan_fails(self):
        original = self.service.agent.stream_plan
        def stream_plan(*args, **kwargs):
            if self.calls:
                raise AgentError('injected planning failure')
            yield from original(*args, **kwargs)
        self.service.agent.stream_plan = stream_plan
        events = list(self.service.chat_stream(self.sid, 'explain', react_max_steps=3))
        self.assertTrue([data for event, data in events if event == 'error'])
        self.assertEqual(self.service.store.get_messages(self.sid)[-1]['content'], 'tool explanation')

    def test_invalid_limits_and_effort_are_rejected(self):
        for invalid in (0, 11, True, '3', 1.5):
            with self.subTest(limit=invalid), self.assertRaises(ValidationError):
                list(self.service.chat_stream(self.sid, 'explain', react_max_steps=invalid))
        with self.assertRaises(ValidationError):
            list(self.service.chat_stream(self.sid, 'explain', thinking_budget='invalid'))
        self.assertEqual(self.service._normalize_thinking_budget('medium'), 'high')

    def test_settings_are_request_local_and_reach_nested_explanation(self):
        client = OpenAICompatClient('test', 'https://api.deepseek.com', 'deepseek-chat')
        self.service.llm = client
        self.service.agent = Agent(client)
        seen = []
        def stream_plan(agent, *args, **kwargs):
            seen.append(agent.llm.thinking_budget)
            yield 'plan', Plan(reply='', actions=[{'name': 'explain_results', 'args': {}}])
        def post(llm, payload):
            llm._configure_thinking(payload)
            seen.append(payload['reasoning_effort'])
            return {'content': 'saved explanation'}
        with patch.object(Agent, 'stream_plan', stream_plan), patch.object(OpenAICompatClient, '_post', post):
            list(self.service.chat_stream(self.sid, 'explain', thinking_budget='low', react_max_steps=1))
            list(self.service.chat_stream(self.sid, 'explain', thinking_budget='max', react_max_steps=1))
        self.assertEqual(seen, ['low', 'low', 'max', 'max'])
        self.assertIs(self.service.llm, client)
        self.assertIs(self.service.agent.llm, client)
        self.assertFalse(hasattr(client, 'thinking_budget'))


if __name__ == '__main__':
    unittest.main()
