"""Task evidence must not mistake a fragment, reasoning or a malformed call for success."""
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch


path = Path(__file__).resolve().parents[1] / 'scripts/strata_termination_probe.py'
probe = None
if path.exists():  # Diagnostic scripts, like strata_replay.py, are not shipped in the runtime image.
    spec = importlib.util.spec_from_file_location('termination_probe', path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)


@unittest.skipIf(probe is None, 'standalone termination probe is not shipped in the runtime image')
class TerminationProbeTests(unittest.TestCase):
    def test_reasoning_and_nonempty_fragments_do_not_pass_the_task_check(self):
        for text in ('', 'Writing `` `', '```text\n', 'The answer is 142.', '</think>'):
            with self.subTest(text=text):
                self.assertFalse(probe.assess('closing-tag', {'content': text,
                    'reasoning_content': '</think> explains the boundary. 17 + 25 = 42.'},
                    'stop', True)['review_candidate'])

    def test_task_evidence_also_requires_normal_protocol_completion(self):
        message = {'content': '`</think>` ends thinking. 17 + 25 = 42.'}
        self.assertTrue(probe.assess('closing-tag', message, 'stop', True)['review_candidate'])
        for finish, done in [('length', True), ('stop', False), (None, True)]:
            self.assertFalse(probe.assess('closing-tag', message, finish, done)['review_candidate'])

    def test_only_the_exact_declared_arithmetic_can_be_executed(self):
        call = {'id': 'call-1', 'type': 'function',
            'function': {'name': 'add', 'arguments': '{"a":17,"b":25}'}}
        self.assertEqual(probe.tool_results({'tool_calls': [call]}),
            [{'role': 'tool', 'tool_call_id': 'call-1', 'content': '42'}])
        for args in ('null', '[]', '17', '"x"', '{"a":true,"b":25}',
                '{"a":17,"b":26}', '{"a":17,"b":25,"cmd":"bad"}'):
            with self.assertRaises(ValueError):
                probe.tool_results({'tool_calls': [{**call,
                    'function': {**call['function'], 'arguments': args}}]})
        for calls in ([], [call, call], [{**call, 'function': {'name': 'shell', 'arguments': '{}'}}]):
            with self.assertRaises(ValueError):
                probe.tool_results({'tool_calls': calls})

    def test_stream_preserves_channels_usage_and_fragmented_tool_arguments(self):
        chunks = [
            {'choices': [{'delta': {'reasoning_content': 'private calculation 42'}}]},
            {'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'call-1',
                'function': {'name': 'add', 'arguments': '{"a":17,'}}]}}]},
            {'choices': [{'delta': {'tool_calls': [{'index': 0,
                'function': {'arguments': '"b":25}'}}]}, 'finish_reason': 'tool_calls'}]},
            {'choices': [], 'usage': {'completion_tokens': 21}},
        ]
        response = io.BytesIO((''.join('data: ' + json.dumps(c) + '\r\n\r\n' for c in chunks)
            + 'data: [DONE]\r\n\r\n').encode())
        response.status, response.headers = 200, {}
        with patch.object(probe.urllib.request, 'urlopen', return_value=response) as opened:
            result = probe.request('http://localhost:1', 'synthetic', [], True, 2, tools=True)
        self.assertEqual(result['message']['content'], '')
        self.assertEqual(result['message']['reasoning_content'], 'private calculation 42')
        self.assertEqual(probe.tool_results(result['message'])[0]['content'], '42')
        self.assertTrue(result['done'])
        self.assertEqual(result['usage']['completion_tokens'], 21)
        self.assertNotIn('stop', json.loads(opened.call_args.args[0].data))

    def test_stream_without_done_does_not_count_as_finished(self):
        response = io.BytesIO(b'data: {"choices":[{"delta":{"content":"42"},"finish_reason":"stop"}]}\n\n')
        response.status, response.headers = 200, {}
        with patch.object(probe.urllib.request, 'urlopen', return_value=response):
            result = probe.request('http://localhost:1', 'synthetic', [], True, 2)
        self.assertFalse(probe.assess('control', result['message'], result['finish_reason'],
            result['done'])['review_candidate'])


if __name__ == '__main__':
    unittest.main()
