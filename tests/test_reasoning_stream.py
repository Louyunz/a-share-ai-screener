import json
import unittest
from unittest.mock import patch

from backend.agent import Agent
from backend.chat.action_schema import TOOL_DEFINITIONS
from backend.data.sample import generate_bundle
from backend.llm import OpenAICompatClient, ToolCall, ToolChatResult
from backend.service import AppService
from tests import cleanup, workdir


class ReasoningTransportTest(unittest.TestCase):
    def test_tool_reasoning_is_emitted_before_result_and_preserved_in_message(self):
        consumed = []
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __iter__(self):
                for index, delta in enumerate(({'reasoning_content': 'A'},
                                                {'reasoning_content': 'B'}, {'content': 'answer'})):
                    consumed.append(index)
                    yield ('data: '+json.dumps({'choices': [{'delta': delta}]})+'\n').encode()
                yield b'data: [DONE]\n'
        client = OpenAICompatClient('test', 'https://api.deepseek.com', 'model')
        with patch('backend.llm.urllib.request.urlopen', return_value=Response()):
            stream = client.chat_tools_stream('system', [], TOOL_DEFINITIONS)
            self.assertEqual(next(stream), ('reasoning', 'A'))
            self.assertEqual(consumed, [0])
            self.assertEqual(next(stream), ('reasoning', 'B'))
            self.assertEqual(next(stream), ('content', 'answer'))
            result = next(stream)[1]
            self.assertEqual(result.message['reasoning_content'], 'AB')
            self.assertEqual(result.content, 'answer')

    def test_text_events_keep_reasoning_separate_and_off_emits_content_only(self):
        class Response:
            def __init__(self, thinking): self.thinking = thinking
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __iter__(self):
                if self.thinking:
                    yield b'data: {"choices":[{"delta":{"reasoning_content":"R"}}]}\n'
                yield b'data: {"choices":[{"delta":{"content":"answer"}}]}\n'
                yield b'data: [DONE]\n'
        def open_request(request, **kwargs):
            return Response(json.loads(request.data)['thinking']['type'] == 'enabled')
        client = OpenAICompatClient('test', 'https://api.deepseek.com', 'model')
        with patch('backend.llm.urllib.request.urlopen', side_effect=open_request):
            self.assertEqual(list(client.with_thinking_budget('high').chat_events_stream('system', [])),
                             [('reasoning', 'R'), ('content', 'answer')])
            self.assertEqual(list(client.with_thinking_budget('off').chat_events_stream('system', [])),
                             [('content', 'answer')])
            self.assertEqual(list(client.with_thinking_budget('high').chat_text_stream('system', [])),
                             ['answer'])


class ReasoningModel:
    def __init__(self, mode='normal'):
        self.mode = mode
        self.requests = []

    def chat_tools_stream(self, system, messages, tools, tool_choice='auto'):
        self.requests.append(messages)
        first = len(self.requests) == 1
        pieces = ['reason-A', 'reason-B'] if first else ['secondary-reason']
        for piece in pieces:
            yield 'reasoning', piece
        content = '正在解读结果。'
        yield 'content', content
        calls = []
        if self.mode != 'review':
            args = {'unexpected': 1} if self.mode == 'repair' and first else {}
            calls = [ToolCall('call_1', 'explain_results', args)]
        message = {'role': 'assistant', 'content': content,
                   'reasoning_content': ''.join(pieces)}
        if calls:
            message['tool_calls'] = [{'id': c.id, 'type': 'function', 'function': {
                'name': c.name, 'arguments': json.dumps(c.arguments)}} for c in calls]
        yield 'result', ToolChatResult(content, calls, message)

    def chat_events_stream(self, system, messages):
        yield 'reasoning', 'final-reason'
        yield 'content', '最终结果。'

    def chat_text(self, system, messages):
        return '解读正文。'


class AgentReasoningTest(unittest.TestCase):
    def test_primary_reasoning_is_streamed(self):
        events = list(Agent(ReasoningModel()).stream_plan({}, [], 'explain'))
        self.assertEqual(events[:2], [('thinking', {'text': 'reason-A'}),
                                      ('thinking', {'text': 'reason-B'})])

    def test_repair_reasoning_is_streamed_and_prior_reasoning_is_replayed(self):
        model = ReasoningModel('repair')
        events = list(Agent(model).stream_plan({}, [], 'explain'))
        self.assertIn(('thinking', {'text': 'secondary-reason'}), events)
        self.assertEqual(model.requests[1][1]['reasoning_content'], 'reason-Areason-B')
        self.assertEqual(events[-1][1].action('explain_results')['args'], {'question': ''})

    def test_semantic_review_streams_reasoning_without_duplicating_answer(self):
        events = list(Agent(ReasoningModel('review')).stream_plan({}, [], 'explain'))
        self.assertIn(('thinking', {'text': 'secondary-reason'}), events)
        self.assertEqual(events[-1][1].reply, '正在解读结果。')

    def test_final_reasoning_is_streamed(self):
        events = list(Agent(ReasoningModel()).stream_final({}, [], 'explain', []))
        self.assertEqual(events, [('thinking', {'text': 'final-reason'}), ('token', '最终结果。')])

    def test_service_forwards_primary_and_final_reasoning_outside_answer(self):
        root = workdir('reasoning_stream')
        service = AppService(bundle=generate_bundle(n_stocks=8, n_days=160, seed=1),
                             db_path=root/'app.db', llm=ReasoningModel())
        try:
            sid = service.create_session()['session_id']
            service.store.save_state(sid, {'last_selection': {'metrics': {}, 'top': []}})
            events = list(service.chat_stream(sid, 'explain', react_max_steps=2))
            thinking = ''.join(data['text'] for kind, data in events if kind == 'thinking')
            self.assertIn('reason-Areason-B', thinking)
            self.assertIn('final-reason', thinking)
            messages = service.messages(sid)['messages']
            self.assertIn('解读正文', messages[-1]['content'])
            self.assertNotIn('reason-', messages[-1]['content'])
        finally:
            service.close()
            cleanup(root)


if __name__ == '__main__':
    unittest.main()
