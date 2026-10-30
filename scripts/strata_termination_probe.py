#!/usr/bin/env python3
"""Bounded synthetic probes of task completion, without server changes or private prompts.

Results contain only the synthetic requests below and their replies. A passing structural
check is a candidate for review, never a claim that an arbitrary task was completed.
"""
import argparse
import datetime
import json
from pathlib import Path
import re
import time
import urllib.request


PROMPTS = {
    'closing-tag': 'Explain the closing thinking tag in backticks, then give the answer to 17 + 25.',
    'turn-marker': 'Qwen 聊天模板里，助手回合的结束标记是什么？请原样写出，例如 "标记"。',
    'control': 'What is 17 + 25? State the answer.',
}
TOOL = {'type': 'function', 'function': {'name': 'add',
    'description': 'Add two integers.', 'parameters': {'type': 'object',
    'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
    'required': ['a', 'b'], 'additionalProperties': False}}}


def assess(case, message, finish, done):
    """Reject empty/fragment success; reason over visible content and validated calls only."""
    content = message.get('content') or ''
    arithmetic = re.search(r'(?<!\d)42(?!\d)', content) is not None
    marker = ('</think>' if case in ('closing-tag', 'follow-up') else '<|im_end|>') in content
    task_evidence = arithmetic if case in ('control', 'tool-answer') else (
        marker and (arithmetic if case in ('closing-tag', 'follow-up') else True))
    return {'visible_characters': len(content),
        'reasoning_characters': len(message.get('reasoning_content') or ''),
        'finish_reason': finish, 'done': done, 'arithmetic_present': arithmetic,
        'marker_present': marker, 'task_evidence': task_evidence,
        'review_candidate': bool(done and finish == 'stop' and task_evidence)}


def request(base, model, messages, stream, timeout, tools=False):
    body = {'model': model, 'messages': messages, 'stream': stream, 'max_tokens': 4096,
        'temperature': 1.0, 'top_p': 0.95, 'top_k': 20,
        'chat_template_kwargs': {'enable_thinking': True}}
    if tools:
        body.update(tools=[TOOL], tool_choice='auto')
    if stream:
        body['stream_options'] = {'include_usage': True}
    req = urllib.request.Request(base.rstrip('/') + '/v1/chat/completions',
        data=json.dumps(body, ensure_ascii=False).encode(), headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    events, calls = [], {}
    message = {'role': 'assistant', 'content': '', 'reasoning_content': ''}
    usage, finish, done = None, None, False
    with urllib.request.urlopen(req, timeout=timeout) as response:
        status = response.status
        generation_id = response.headers.get('x-router-generation-id')
        if stream:
            for raw in response:
                if not raw.startswith(b'data:'):
                    continue
                data = raw[5:].strip()
                if data == b'[DONE]':
                    done = True
                    continue
                event = json.loads(data)
                events.append(event)
                if event.get('error'):
                    raise RuntimeError(json.dumps(event['error']))
                if event.get('usage'):
                    usage = event['usage']
                for choice in event.get('choices', []):
                    finish = choice.get('finish_reason') or finish
                    delta = choice.get('delta', {})
                    for key in ('content', 'reasoning_content'):
                        message[key] += delta.get(key) or ''
                    for call in delta.get('tool_calls') or []:
                        value = calls.setdefault(call.get('index', 0),
                            {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                        if call.get('id'):
                            value['id'] = call['id']
                        for key in ('name', 'arguments'):
                            value['function'][key] += (call.get('function') or {}).get(key) or ''
        else:
            result = json.load(response)
            events.append(result)
            choice = result['choices'][0]
            message, finish, usage = choice['message'], choice['finish_reason'], result.get('usage')
            done = True
    if calls:
        message['tool_calls'] = [calls[i] for i in sorted(calls)]
    return {'request': body, 'http_status': status, 'generation_id': generation_id,
        'seconds': round(time.monotonic() - started, 3), 'message': message,
        'finish_reason': finish, 'done': done, 'usage': usage, 'events': events}


def tool_results(message):
    """Execute only the declared synthetic arithmetic; never execute generated code."""
    calls = message.get('tool_calls') or []
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
        raise ValueError('Expected exactly one add call')
    call = calls[0]
    fn = call.get('function') or {}
    if not isinstance(fn, dict):
        raise ValueError('Call must name the add function')
    args = json.loads(fn.get('arguments') or '{}')
    if (fn.get('name') != 'add' or not isinstance(call.get('id'), str) or not call['id']
            or not isinstance(args, dict) or set(args) != {'a', 'b'}
            or any(type(v) is not int for v in args.values()) or sorted(args.values()) != [17, 25]):
        raise ValueError('Call must add exactly 17 and 25')
    return [{'role': 'tool', 'tool_call_id': call['id'], 'content': str(args['a'] + args['b'])}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rounds', type=int, choices=range(1, 11), default=2)
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    summaries = []

    def run(case, messages, stream, round_number):
        record = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'case': case, 'round': round_number, 'stream': stream}
        try:
            record.update(request(args.url, args.model, messages, stream, args.timeout,
                tools=case in ('tool-call', 'tool-answer', 'follow-up')))
            record['assessment'] = assess(case, record['message'], record['finish_reason'], record['done'])
        except Exception as error:
            record['error'] = str(error)
        name = f'{len(summaries) + 1:03}-{case}-{int(stream)}.json'
        (args.out / name).write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        summary = {k: record[k] for k in ('case', 'round', 'stream', 'assessment', 'error') if k in record}
        summary['file'] = name
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
        return record

    for round_number in range(1, args.rounds + 1):
        for stream in (False, True):
            for case, prompt in PROMPTS.items():
                run(case, [{'role': 'user', 'content': prompt}], stream, round_number)
            messages = [{'role': 'user', 'content': 'Call add to compute 17 + 25, then state the result.'}]
            called = run('tool-call', messages, stream, round_number)
            try:
                results = tool_results(called.get('message') or {})
                if not called['done'] or called['finish_reason'] not in ('tool_calls', 'stop'):
                    raise ValueError('Tool turn did not terminate normally')
            except (ValueError, KeyError) as error:
                summaries[-1]['tool_validation'] = str(error)
                continue
            summaries[-1]['tool_validation'] = 'passed'
            messages += [called['message'], *results]
            answer = run('tool-answer', messages, stream, round_number)
            if not answer.get('assessment', {}).get('review_candidate') or answer['message'].get('tool_calls'):
                continue
            messages += [answer['message'], {'role': 'user', 'content': PROMPTS['closing-tag']}]
            run('follow-up', messages, stream, round_number)
    (args.out / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
    return int(any(s.get('error') or (s['case'] != 'tool-call' and
        not s.get('assessment', {}).get('review_candidate')) or
        (s['case'] == 'tool-call' and s.get('tool_validation') != 'passed') for s in summaries))


if __name__ == '__main__':
    raise SystemExit(main())
