# [Engine] 0.2.8: empty stops and unfinished visible fragments with reasoning_close_retry enabled

Prepared for the existing [issue 2236](https://github.com/Niko1221/Strata/issues/2236).
Not yet submitted. Related: [1053](https://github.com/Niko1221/Strata/issues/1053),
[1178](https://github.com/Niko1221/Strata/issues/1178),
[2731](https://github.com/Niko1221/Strata/issues/2731),
[4077](https://github.com/Niko1221/Strata/issues/4077), and closed, unmerged
[PR 3207](https://github.com/Niko1221/Strata/pull/3207).

## What happened

Retested on released Strata **0.2.8 / engine 0.1.54**, with upstream engine
and Python serving source unchanged. `reasoning_close_retry: true` improves
earlier failures, but does not ensure a completed answer. We still receive
reasoning-only replies and unfinished visible fragments with `finish_reason:
"stop"`; streaming replies also deliver `[DONE]`. This is not a transport timeout.

## Environment

- Docker, Ubuntu 26.04; official CUDA13 release digest
  `sha256:27f04a7437c7d2940d067331ead66e3942826c55b57334908c8767625bbdbb04`.
  Deployment adapter supplies paths/configuration and uses upstream's source
  serving interface; it patches neither engine nor serving Python source.
- Terminal source `870aedd9e1ebe1c91f94da93b59af17729f6c5c0`;
  engine BUILD.json: `0.1.54`, source `f-028-rel 8c1aea4bca1b`.
  Installed server SHA256 on both machines:
  `7f731e5f38ff81ae77396fb14d00d82781ae37e33007daf25ebcdca80796f2ab`.
- Qwen3.8-Flash-Next UD-IQ4_XS, four text cards (3090 24GB, 4080 SUPER 16GB,
  4080 16GB, 3080 Ti 12GB), separate 3080 10GB vision card, about 123GiB usable
  host RAM, 262,144 context, vision enabled.
- Independently: IQ3_XXS, 2×2080 Ti 11GB, about 60.7GiB host RAM,
  131,072 context, no vision, conversation cache disabled.
- Both: `reasoning_close_retry: true`, `tool_call_recovery: true`,
  `answer_room: false`, MTP `--spec 4 --spec-min-p 0.5`.

## Smallest reproduction

Send directly to the backend, repeat in both stream modes:

```json
{
  "model": "qwen3.8-flash-next-iq3_xxs",
  "messages": [{"role": "user", "content": "Explain the closing thinking tag in backticks, then give the answer to 17 + 25."}],
  "chat_template_kwargs": {"enable_thinking": true},
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 20,
  "max_tokens": 4096,
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

Also reproduces with the user message:

> Qwen 聊天模板里，助手回合的结束标记是什么？请原样写出，例如 "标记"。

Expected: explain and quote the requested tag and answer **42**, or return a
usable indication that the requested response is incomplete. For the Chinese
case, return the requested marker. A non-empty fragment is not task completion.

## Reproduction status and measurements

Reproduced on the real models. Two rounds × two stream modes × two machines
gave 16 fresh trigger requests. Four returned empty with `stop` (three on the
four-card host, one on the two-card host). Only one of eight turn-marker replies
contained the requested marker. None of eight closing-tag replies contained
both the exact Qwen closing tag and 42. Responses are stochastic.

Two clear IQ3_XXS examples:

- Non-stream: 60 visible characters, beginning `The closing thinking tag would
  be written in backticks as ` and ending with two backticks. `stop`, 112
  completion tokens out of 4096; no answer to the arithmetic.
- Stream: visible content was only an opening code fence labelled `text` and
  a newline (8 characters). `stop`, `[DONE]`, 75 completion tokens out of 4096.

Controls in the same 46-request matrix: 8/8 simple arithmetic requests answered
42. Seven of eight requested addition-tool calls had valid arguments; all seven
result replies answered 42. None of the seven subsequent closing-tag follow-ups
qualified. No service restarted during the tests.

## Stop diagnostics and uncertainty

We have **not captured the original sampled stop IDs for these live failures**.
We do not claim that the model, speculation or engine generated any particular
ID in them. Per-request logprobs omit the terminal stop token.

An isolated, scripted replay using the installed upstream Service and actual
pack tokenizer confirmed the retry boundaries with raw IDs. It loaded no
weights and did not change the serving process. Known sequences showed:

- Reasoning → 248046 → retry → 42: two engine passes, visible 42.
- 248069 → visible `Writing` plus backticks → 248046: one pass, fragment, `stop`.
- Quote immediately before 248069 → 248046: one pass, empty, `stop`.
- Reasoning → 248046 → retry immediately emits 248046: two passes, empty, `stop`.

These scripted IDs demonstrate coverage limits, not the live model's mechanism.
Two additional bounded diagnostic requests used the suggested
`logit_bias: {"248046": -100}` with max_tokens 512: one repeated start markers
and ended `length`, the other still returned an incomplete `stop`. This has not
been applied as a persistent setting or presented as a fix.

## Request

Is there an official supported configuration or released fix that covers both
reasoning-only stops and already-visible fragments without promoting reasoning
to the answer or retrying indefinitely? Is there a supported per-request method
to expose the original stop ID and parser state without enabling prompt dumps
for unrelated production traffic? We can qualify an official fix on both packs
and provide synthetic transcripts. No unmerged/private patch will be deployed.
