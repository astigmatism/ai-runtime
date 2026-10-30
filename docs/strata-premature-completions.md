# Residual premature completions on Strata 0.2.8

The upgrade remains in place. This investigation changes no running service,
live settings, container source or image pin. The reliability incident remains
open: a complete HTTP exchange and `finish_reason: "stop"` do not establish that
the requested work was done.

## Verified release and configuration

AI Runtime revision `785923b27eade8765d8f659148aae77cc8b263d8` integrates
the official Strata 0.2.8 / engine 0.1.54 release. Both running containers use
image `sha256:6d7874f9b112d073ab5a1a9c744e02403a5f41cde22a9efec7da0962fb42bf4c`.
The installed upstream `serve/server.py` on both machines hashes to
`7f731e5f38ff81ae77396fb14d00d82781ae37e33007daf25ebcdca80796f2ab`, matching
the unmodified source at `870aedd9e1ebe1c91f94da93b59af17729f6c5c0`.
The engine's BUILD.json separately reports `0.1.54`, source
`f-028-rel 8c1aea4bca1b`; these are different component identities.

Both configurations still have `reasoning_close_retry: true`,
`tool_call_recovery: true` and `answer_room: false`. Rosalina remains at 262,144
tokens with vision; Daisy remains at 131,072 without vision and with its
conversation cache explicitly disabled. Both use MTP `--spec 4 --spec-min-p 0.5`.
Both containers had zero restarts at the beginning and end of the probes.

## New real-model measurements

`scripts/strata_termination_probe.py` ran two rounds of both streaming and
non-streaming requests directly against each backend. Each ordinary probe uses
thinking, temperature 1.0, top_p 0.95, top_k 20 and max_tokens 4096. No request
adds a stop string or changes a server setting. Each tool sequence permits one
validated addition, its result reply, and one follow-up; it never retries a
failure. Results are synthetic requests and replies, kept outside Git.

| Measurement | Rosalina | Daisy |
| --- | ---: | ---: |
| Requests | 24 | 22 |
| Empty replies without a tool call, all ended `stop` | 3 | 1 |
| Ordinary arithmetic controls answered 42 | 4/4 | 4/4 |
| Requested addition tool executed with validated arguments | 4/4 | 3/4 |
| Replies after those tool results answered 42 | 4/4 | 3/3 |
| Closing-tag prompts containing the requested Qwen tag and 42 | 0/4 | 0/4 |
| Turn-marker prompts containing the requested marker | 1/4 | 0/4 |
| Marker follow-ups after successful tool/result turns qualified | 0/4 | 0/3 |

These checks are deliberately stricter than non-empty output, but are still
structural evidence requiring review. Daisy once answered 42 while naming
`</thinking>` instead of Qwen's `</think>`; that is not an exact-tag pass.
Other replies contained unfinished planning in the visible channel. A literal
tag in a reply alone does not establish a useful or complete explanation.

The clearest new fragments were Daisy's non-streaming reply
`The closing thinking tag would be written in backticks as ` followed by two
backticks (60 visible characters, 112 completion tokens), and its streaming
reply consisting solely of an opening `text` code fence and newline (8 visible
characters, 75 completion tokens). Both reported `stop`, far below 4096 tokens;
the streaming one delivered `[DONE]`.

## What the raw diagnostics establish

`scripts/strata_synthetic_stops.py` ran in a separate Python process using the
installed upstream Service, MockEngine and actual pack tokenizer. It loaded no
model weights, opened no listener, and modified no installed file. Its
`STRATA_DEBUG=1` applied only to that process, whose inputs were four hard-coded
synthetic scripts. It printed raw token IDs and termination diagnostics.

The tokenizer identifies `im_end` as 248046, `endoftext` as 248044, and the closing
thinking tag as 248069. The scripted cases showed:

| Known token sequence | Engine passes | Visible result | Termination |
| --- | ---: | --- | --- |
| Reasoning, stop, successful continuation | 2 | 42 | stop |
| Closing think, `Writing` and backticks, stop | 1 | fragment only | stop |
| Quote immediately before a closing think, stop | 1 | empty | stop |
| Reasoning, stop, continuation immediately stops too | 2 | empty | stop |

These are **scripted tokens, not captured live-model samples**. They establish
the released retry's boundaries and confirm that it is finite. They do not
prove which terminal token the live model generated in any retained failure.

The released Service checks stop IDs before feeding those tokens to OutputParser.
The retry requires the parser to remain in reasoning, no answer to have begun,
no held closing-think tag, no prior close retry, no cancellation or user stop,
and space for the inserted close plus at least one token. A visible fragment
and the held-tag case therefore bypass it. A failed continuation gets no second
retry. Ordinary reasoning is never made into an answer by the probe.

The supported per-request logprobs API omits the terminal stop token. Two extra
512-token diagnostics used the maintainer's suggested bias against 248046:
Rosalina repeated start markers and ended `length`; Daisy still returned an
incomplete `stop`. Neither qualifies as a fix. No bias was saved or applied to
other requests. Those two samples also do not prove their final sampled ID.

## Upstream disposition and handoff

[Issue 2236](https://github.com/Niko1221/Strata/issues/2236) is open and is the
best existing thread for new release evidence. Its maintainer explains the
distinction between literal text and a real end token, suggests the bounded
logit-bias diagnostic, and describes prompt-level spacing as a workaround.
That workaround is not a reliable completion guarantee for arbitrary agent work.
[Issue 4077](https://github.com/Niko1221/Strata/issues/4077) separately tracks
thinking-channel transitions. Keep that distinction in any report.

[PR 3207](https://github.com/Niko1221/Strata/pull/3207) was closed **unmerged**;
its author considered the existing retry sufficient on their measured corpus.
Neither that PR nor a contributor's write-back patch is part of this deployment.
No complete supported fix for these residual cases was found in the reviewed
release, official troubleshooting guidance or the linked upstream discussions.
Keep the current upgrade while requesting an official fix or supported setting.

The report prepared in `strata-upstream-report.md` can be added to the existing
thread after review. It has not been submitted. Attach only synthetic evidence;
never publish production conversation archives or credentials.

The remaining diagnostic gap is the **original real-model terminal ID**.
Capturing that with the released raw-dump facility requires an isolated serving
process, the exact pack/config and exclusive test traffic. The live server's
environment must not be changed to dump everybody's prompts. A separate GPU
qualification window must be authorized before stopping or replacing anything
to make room for such a process. Record raw IDs for both generation passes,
parser/channel outcomes, user stop/cancel state, usage and whether the final
requested answer and subsequent tool continuation succeed. Until then, label
the exact live mechanism unconfirmed.

The separate local LLM Router change preserves a typed generation failure,
record ID, backend identity and available usage/termination metadata through
Responses errors and the generation journal. It keeps the empty-answer guard
and adds no inference retry. This is diagnostic work, not a backend repair;
it needs its own reviewed release and normal Portal update authorization.

Validation: 609 AI Runtime tests passed, including the five new probe checks;
the four upstream ReasoningCloseRetry tests passed against the released source.
The separate router's 38 targeted tests and syntax check passed. Its full suite
passed 344/344 using a temporary GNU-stat-interface adapter for macOS; natively,
343/344 passed and the existing GNU `stat -c %Y` stale-lock test failed. That
platform incompatibility was reproduced independently; updater code is unchanged.
The standalone probe tests also skip cleanly in the runtime-image layout, where
diagnostic scripts are intentionally absent, matching the existing replay tests.

The investigation scripts require no deployment. After authorization, the router
diagnostics can be released through its own **Update and restart** action; verify
both Responses modes, archive IDs, normal tools and released admission. Revert
that source commit on the Mac, test, publish to main and use the same Portal
workflow if diagnostics must be rolled back. Do not move the production checkout
by hand. Keep AI Runtime's deployed revision and both Strata images/configurations
as they stand while upstream work is qualified.

Operational evidence is under the investigator's
`ops/reports/20261029-strata-residual/` outside the source tree. Keep compact
summaries while the incident is open; retain complete synthetic evidence for
180 days or while referenced. Remove temporary source downloads and diagnostic
environments within 14 days of handoff. Keep prior production images and
backups while active/previous/recovery release state refers to them.
