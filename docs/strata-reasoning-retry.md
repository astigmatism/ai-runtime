# Strata release upgrade and reasoning-only stops

The 2026-10-29 investigation found complete backend SSE streams containing only
reasoning, followed by `finish_reason: stop` and `[DONE]`. The router correctly
refused an empty answer. This signature is separate from network interruption.

Strata documents `reasoning_close_retry: true` for this case. It closes thinking
and continues once; it does not promote reasoning to a visible answer. Success
must be measured with real prompts and tool calls, not inferred from HTTP health.

The upgrade improved retained replays but did not close the reliability incident.
See [residual premature completions](strata-premature-completions.md) for fresh
0.2.8 measurements, the retry's measured boundaries, and the prepared upstream
report. Existing upgrades and live policy remain in place.

## Release integration

The new `strata-vision-870aedd9e1eb` and `strata-sm75-870aedd9e1eb` entries use
Strata 0.2.8, engine 0.1.54, from the official CUDA13 image pinned to
`sha256:27f04a7437c7d2940d067331ead66e3942826c55b57334908c8767625bbdbb04`.
That published portable image contains sm75, sm86, sm89 and sm120 code. It can
serve both current machines; it needs their existing NVIDIA 580+ driver.

`runtime/overlay/Dockerfile.release` adds the deployment entrypoint, curl health
check, and compatibility paths. It changes no upstream engine or Python serving
source. The server starts using upstream's documented source interface:
`serve/server.py --engine strata --config FILE`. Existing packs, model files,
GPU placement, vision device, context and cache declarations remain in use.
The upstream Terminal COMMIT is `870aedd9e1ebe1c91f94da93b59af17729f6c5c0`;
engine BUILD.json separately identifies the official component build.

The adapter sets upstream's `answer_room` to false to preserve this runtime's
unrestricted-thinking/output contract. The upstream release otherwise introduces
a default thinking reservation. `STRATA_ANSWER_ROOM=on` explicitly opts into it.

`--reasoning-close-retry on|off` maps to `REASONING_CLOSE_RETRY`. An explicit
setting requires the engine's rebuild receipt to match its pinned image ID.
The entrypoint rejects invalid values before writing the workspace. Both off
and absence remove a stale key, so a profile switch cannot inherit recovery.
Old engine identities and image pins remain registered for rollback.

## Rollout and rollback

Build the release adapter from committed source, record its image identity and
upstream digest, then deploy the controller through Service Portal. This first
release leaves existing machine profiles and inference containers unchanged.
On each machine, back up its private profile definitions and small workspace
run configs. Change the selected machine profile to the matching new engine and
state `--reasoning-close-retry on`; do not commit machine profiles. Apply through
the deployed runtime's validated transition, which drains work and owns recovery.
Verify the live image, engine version, settings, GPU placement, model routes,
visible responses, valid tool arguments and continuation after tool results.

If qualification fails, restore the saved private definitions and run the
validated transition back to the old engine. Keep previous images and workspace
backups until the replacement is accepted; do not manufacture active receipts.
Neither a health check nor a finite soak establishes that every future request
will succeed. Report the number and kinds of requests actually tested.

Retention: old engines/images are retained while referenced by running,
active/previous/recovery release state or offered profiles. Remove unreferenced
candidate tags and task staging within 14 days of acceptance, reference-aware;
keep compact checksums and qualification summaries. Never use blanket pruning.
Production conversations remain private and never enter source or test fixtures.
