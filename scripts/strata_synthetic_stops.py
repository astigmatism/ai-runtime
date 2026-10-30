#!/usr/bin/env python3
"""Exercise unmodified upstream Service with scripted tokens, never model weights.

Run with upstream's Python environment; --source and --tokenizer name installed
upstream source and a pack's tokenizer. STRATA_DEBUG affects only this process.
This proves parser/retry behavior for known tokens, not what a live model sampled.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(args.source), str(args.source / 'tools')]
    from strata_tokenizer import Tokenizer
    from serve.frontend import ChatTemplate
    from serve.server import MockEngine, Service

    vocab = json.loads((args.tokenizer / 'vocab.json').read_text())
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    types = json.loads((args.tokenizer / 'token_type.json').read_text())
    config = json.loads((args.tokenizer / 'tokenizer.json').read_text())
    tok = Tokenizer(tokens, (args.tokenizer / 'merges.txt').read_text().split('\n'), types,
        config.get('pre') or 'qwen35', config.get('special_ids') or None)
    os.environ['STRATA_DEBUG'] = '1'
    print(json.dumps({'source_server_sha256': hashlib.sha256(
        (args.source / 'serve/server.py').read_bytes()).hexdigest(), 'scripted_not_model_sampled': True,
        'special_ids': {s: tok.encode(s, parse_special=True)
            for s in ('<|im_end|>', '<|endoftext|>', '</think>')}}), flush=True)
    cases = {
        'reasoning_stop_then_answer': ['The tag I would quote is `', '42'],
        'visible_fragment_then_stop': ['</think>\n\nWriting `` `'],
        'closed_thinking_after_quote': ['The closing tag is "</think>'],
        'unsuccessful_continuation': ['The tag I would quote is `', ''],
    }
    for name, scripts in cases.items():
        engine = MockEngine(tok, scripts if len(scripts) > 1 else scripts * 2)
        svc = Service(engine, tok, ChatTemplate(args.source / 'serve/chat_template.jinja'))
        svc.reasoning_close_retry = True
        prompt = tok.encode('<|im_start|>assistant\n<think>\n', parse_special=True)
        print('CASE ' + name, flush=True)
        events = list(svc.run(prompt, True, [], 512, {}, threading.Event()))
        visible = ''.join(event.text for kind, event in events if kind == 'event' and event.kind == 'content')
        reasoning = ''.join(event.text for kind, event in events if kind == 'event' and event.kind == 'reasoning')
        done = next(value for kind, value in events if kind == 'done')
        print(json.dumps({'case': name, 'engine_passes': engine.turns, 'visible': visible,
            'reasoning': reasoning, 'termination': done}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
