"""Refresh must detach an observer without cancelling real model optimization."""
import threading
import time
import unittest

from backend.data.sample import generate_bundle
from backend.schemas import Plan, profile_summary
from backend.service import AppService, ConflictError, NotFoundError
from tests import cleanup, workdir
from tests.support.scripted_llm import ScriptedLLM


class BackgroundChatTest(unittest.TestCase):
    def setUp(self):
        self.root = workdir('background_chat')
        self.service = AppService(bundle=generate_bundle(n_stocks=10, n_days=260, seed=9),
                                  db_path=self.root/'app.db', llm=ScriptedLLM())
        self.sid = self.service.create_session(owner_id=7)['session_id']
        slots = {'goal': '养老', 'horizon': '3年', 'drawdown_tolerance': '15%', 'loss_reaction': '先拿住'}
        self.service.store.save_profile(self.sid, slots, profile_summary(slots), True)
        self.reached = threading.Event()
        self.release = threading.Event()
        original = self.service._run_action
        def action(*args, **kwargs):
            gated = False
            generator = original(*args, **kwargs)
            while True:
                try:
                    event, data = next(generator)
                except StopIteration as stop:
                    return stop.value
                yield event, data
                if event == 'ui' and data['block']['type'] == 'tuning' and not gated:
                    gated = True
                    self.reached.set()
                    if not self.release.wait(10):
                        raise RuntimeError('test gate timed out')
        self.service._run_action = action
        def plan(*args, **kwargs):
            yield 'plan', Plan(reply='', actions=[{'name': 'analyze_model', 'args': {
                'engine': 'weights', 'model_type': '', 'objective': 'sharpe',
                'escalate': False, 'reason': 'refresh regression'}}])
        self.service.agent.stream_plan = plan

    def tearDown(self):
        self.release.set()
        self.service.close()
        cleanup(self.root)

    def start_at_tuning(self):
        stream = self.service.start_chat(self.sid, '优化模型', owner_id=7, react_max_steps=1)
        for event, data in stream:
            if event == 'ui' and data['block']['type'] == 'tuning':
                break
        self.assertTrue(self.reached.wait(2))
        return stream

    def wait_finished(self):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state = self.service.get_state(self.sid, owner_id=7)
            if not state['task']['running']:
                return state
            time.sleep(.02)
        self.fail('background job did not finish')

    def test_refresh_recovers_cards_and_background_produces_candidate(self):
        stream = self.start_at_tuning()
        history = self.service.messages(self.sid, owner_id=7)['messages']
        types = {b['type'] for m in history for b in m.get('blocks', [])}
        self.assertTrue({'work', 'tuning'} <= types)
        self.assertTrue(self.service.get_state(self.sid)['task']['running'])
        # Simulate browser refresh: discard the original SSE connection.
        stream.close()
        self.release.set()
        state = self.wait_finished()
        self.assertEqual(state['task']['status'], 'completed')
        self.assertEqual(len(state['pending_candidates']), 1)
        self.assertTrue(state['pending_candidates'][0]['available'])
        self.assertIsNone(state['active_model']['version_id'])
        history = self.service.messages(self.sid)['messages']
        self.assertEqual([m['role'] for m in history], ['user', 'assistant'])
        types = {b['type'] for m in history for b in m.get('blocks', [])}
        self.assertTrue({'work', 'tuning', 'comparison', 'approval'} <= types)

    def test_same_session_cannot_start_twice_and_owner_is_checked(self):
        stream = self.start_at_tuning()
        with self.assertRaises(ConflictError):
            self.service.start_chat(self.sid, 'again', owner_id=7)
        with self.assertRaises(NotFoundError):
            self.service.start_chat(self.sid, 'again', owner_id=8)
        with self.assertRaises(NotFoundError):
            self.service.cancel_chat(self.sid, owner_id=8)
        stream.close()
        self.release.set()
        self.wait_finished()

    def test_explicit_stop_is_separate_from_disconnect(self):
        stream = self.start_at_tuning()
        result = self.service.cancel_chat(self.sid, owner_id=7)
        self.assertEqual(result['task']['status'], 'cancelling')
        stream.close()
        self.release.set()
        state = self.wait_finished()
        self.assertEqual(state['task']['status'], 'cancelled')
        self.assertFalse(state['pending_candidates'])
        history = self.service.messages(self.sid)['messages']
        self.assertIn('任务已停止', history[-1]['content'])
        self.assertTrue(all(b.get('status') != 'running' for b in history[-1]['blocks']))

    def test_job_error_is_persisted_and_exposed_after_disconnect(self):
        def fail(*args, **kwargs):
            raise RuntimeError('injected failure')
            yield
        self.service.agent.stream_plan = fail
        stream = self.service.start_chat(self.sid, '优化', owner_id=7)
        stream.close()
        state = self.wait_finished()
        self.assertEqual(state['task']['status'], 'error')
        self.assertIn('injected failure', state['task']['error'])
        self.assertIn('任务执行失败', self.service.messages(self.sid)['messages'][-1]['content'])


if __name__ == '__main__':
    unittest.main()
