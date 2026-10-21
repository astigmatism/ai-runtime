# Moving Rosalina onto the shared router without stopping what is serving

Rosalina has one router, installed beside the models, and every client that talks to a large model on this network talks to it: the DeepSeek Harness on Pauline and on local-ai-voice both resolve `ai-router` to `192.168.1.4`, and ComfyUI's prompt generator on Daisy does the same. The new router at `192.168.1.23` is the machine the runtime now publishes to by push ([federation.md](federation.md)). Moving a model host from one router to the other is not one action, and the reason is that **the runtime is not the risk**. Pointing a runtime at a second router moves catalogs. Moving a client moves a session that is running right now — possibly the one writing this.

So the order below is: move the runtime first, where a mistake is a warning on a page; make the new router able to describe a serving model, which needs one change on the router and one on Rosalina; and move clients last, one at a time, with the old address still working underneath them.

## The rule the order falls out of

**Never let both routers be unable to serve.** The legacy router stays installed, running and fed until the last client has been quiet on the new one for longer than a working day. Everything else — the order of the stages, which step gets a window, which step does not — comes from that one rule, plus the ordinary one that a change which recreates a model is scheduled and a change which does not is not.

## What is true right now, measured

| Fact | Value | Where it comes from |
| --- | --- | --- |
| Rosalina's resident | `flash-next-strata-solo-thinking-recovery-cache`, service IDs `local-active`, `daytime`, `nighttime` | `.state/active.json` |
| Warm boot of that resident | **112.5 s** | `.state/provisioning/flash-next-strata-solo-thinking-recovery-cache.json` |
| A switch drains the legacy router globally | yes, as it has always done | today's behaviour |
| A switch drains the new router | per host (`runtime-drain {host: rosalina}`), leaving other hosts serving | `runtime/controller.py`, `tests/test_publication.py` |
| What a client sends today | `http://ai-router:11434/v1`, `openai-responses`, models `daytime` and `nighttime`, `maxConcurrency: 1`, two retries on `SERVER`/`TRANSPORT` | `deployments/deepseek-harness/config/settings.yaml` |
| How `ai-router` resolves | `extra_hosts` line `"ai-router=192.168.1.4"`, one line per client host | Pauline and local-ai-voice `compose.yaml` |
| What the DSH discovery plugin demands of a router | per-model `metadata.schema_version == 2` and `metadata.complete === true`; `available !== true` marks the resident unavailable | `.dsh-plugins/dsh-router-model-discovery.js` |
| What the new router reports for Rosalina today | `available: false`, `metadata.complete: false`, warnings `["ACTIVE_MODEL_MARKER_UNAVAILABLE"]`, `metadata.schema_version: 2`, capabilities document `schema_version: 2` | `GET http://192.168.1.23:11434/v1/router/capabilities` |

## Two things on the router block the client move, and neither is the runtime

**A pushed catalog cannot be `complete` on the new router, and a client refuses an incomplete one.** The router computes a model's metadata completeness as *no warnings at all*, and it adds `ACTIVE_MODEL_MARKER_UNAVAILABLE` to every catalog whose `loadedFrom` is not `file` — which is every catalog that was pushed. The DSH plugin throws `router metadata is incomplete` on exactly that field and marks the resident `incomplete`, which is one of its blocking statuses. So a client repointed at `192.168.1.23` today has no usable model, and it would have none even with a reachable backend, for a reason that is an artifact of the single-host router it grew out of. The fix belongs to the router: a catalog that arrived by push has no local marker to be missing, so the warning should not be pushed for it. The strictness of `complete` is worth keeping — it is the reason a half-known model is never called usable.

**`ROUTER_HOSTS` still declares `local`, and nothing publishes to it.** That is why `/health` says `degraded` and capabilities says `complete: false` at the document level. It is not a blocker for the DSH client — the plugin only logs a `warnOnce` about the capabilities `schema_version` bump — but a router that reports itself degraded while everything works is a router nobody trusts at 3 a.m. Either publish something for that host or take it out of the list.

Both changes are one deploy of the router, and the deploy is free of client impact for as long as no client points at it.

## The stages

| # | Action | What moves | Client impact | Rollback |
| --- | --- | --- | --- | --- |
| 0 | Update Rosalina to the revision with the publisher, changing no setting | controller image only | none — backends are not touched by a controller restart | stay on the old revision |
| 1 | Add `router_target` and the token; keep `router_state_dir` | publication gains a second sink | none — startup publication pushes, and a push drains nothing | remove the two settings, restart the controller |
| 2 | State `backend_address` | port binding `127.0.0.1:18080` → `192.168.1.4:18080`, and the catalog URL with it | **the window**: about 2 minutes of refused requests on the legacy router while the resident reloads | remove the setting and switch again (another window) |
| 3 | Fix the router's completeness rule for pushed catalogs, drop `local` | router image | none — no client is on it yet | redeploy the previous router revision |
| 4 | Point one client's `ai-router` at `192.168.1.23` | that client's requests | that client's sessions restart; the others are untouched | revert the one line, recreate that container |
| 5 | Retire the legacy router: drop `router_state_dir`, stop the container | nothing a client can see | none, if stage 4 has been quiet for a day | restart the container; it was never uninstalled |

**Stage 0 — the update nobody has to notice.** Rosalina runs `ef93ae2`, which predates the publisher: stating `backend_address` today changes nothing at all on that revision, which is worth knowing before it is called a risk. Updating is the ordinary portal action, and the updater validates the candidate against her *active* configuration before the image is cut over. Her backends are not restarted by an update; only the controller container is replaced, and a controller that restarts does not unload a model.

**Stage 1 — two sinks, deliberately.** Add to `.state/host.json`:

```json
"router_target": { "url": "http://192.168.1.23:11435", "token_env": "CATALOG_PUSH_TOKEN", "admin_token_env": "ADMIN_TOKEN" }
```

and `CATALOG_PUSH_TOKEN=…` and `ADMIN_TOKEN=…` to the deployment's `.env`, leaving `router_state_dir` set. Two secrets because the router separates them: the publisher token writes this machine's catalog and nothing else, and a machine that cannot drain the router cannot switch behind it. Stating `host_id` is part of this stage too — the machine publishes as itself, and the first attempt without it said so by refusing to start the controller (which is also a reminder that a `host.json` edit is only live once the service comes back, so bring the token and the host ID together). The controller service is started with that file as its environment, which matters more than it sounds: the credential reached the updater's one-off actions first and not the long-running publisher, so a machine that looked federated after a manual `publish` would have stopped on its own catch-up loop with nothing in the environment to send. The file sink stays because it is the only sink the legacy router can read and the only one that can be restored from disk; it is also what keeps this stage reversible by deleting two lines. Restarting the controller for the new setting publishes on startup — a push is not a drain, so no request is refused — and the page then shows the revision the router returned. Verify `CATALOG_STALE:rosalina` disappears from the new router's warnings; if it persists, the runtime's retry interval and the router's staleness bound need aligning, and that is a better thing to discover now than during stage 4.

What stage 1 costs is a new way for a *later* switch to fail: a router that refuses the catalog (a name clash, a non-routable backend) fails the switch after the drain, and the runtime restores the previous catalog, reopens the legacy router and reports the refusal. That is minutes, not a night, and it is tested in `tests/test_publication.py`. An unreachable router is different and costs nothing: a warning on the page and a retry every five minutes.

**Stage 2 — the reachability window, and the only one.** Stating `backend_address: "192.168.1.4"` moves the published port and the advertised URL together, which moves `config_sha256`, which means the resident is recreated. Measured warm boot of the configuration serving now is 112.5 seconds; add the drain and the verification and call it two to three minutes during which the legacy router refuses new admissions. The drain waits for in-flight requests rather than cutting them, so a turn already streaming finishes; what does not get served is a request started inside the window, and the DSH retry policy (two retries) will usually not outlast it. So: pick a moment when the page shows nothing active and no session is mid-task, say so out loud, and do it. Take the opportunity in the same window to confirm the thing stage 4 depends on — `GET http://192.168.1.23:11434/v1/router/capabilities` reporting `available: true` for `daytime` and no `ROUTER_CANNOT_REACH_BACKEND:rosalina` warning.

**Stage 3 — make the new router describable.** Land the router's completeness fix and the `ROUTER_HOSTS` cleanup, redeploy, and read back what a *client* would read: the capabilities document, run through the DSH plugin's own `validateRouterMetadata` against a live document rather than a fixture. That script is the gate for stage 4 and it costs nothing to run again later.

**Stage 4 — one client, then the next.** Because every client resolves `ai-router` through an `extra_hosts` line, the repoint does not touch a client's configuration at all: same URL, same API key, same retry policy, same service IDs. Change one line, recreate that one container, and check the client's own view of the router before declaring the client moved. Do the voice box first — a session lost there costs a conversation, and the harness on Pauline is the one holding live agent sessions. **Restarting the harness container interrupts every session inside it, including the agent doing the move**, so stage 4 is done at a moment when this session can be spared, and the check afterwards is that the discovery plugin reports both residents `available`, not that a request happened to succeed.

**Stage 5 — the legacy router goes last, quietly.** Drop `router_state_dir` from `host.json` (a controller restart, no drain — the file it wrote keeps working until it stops being read), stop the legacy router container, and leave the model port where stage 2 put it. Nothing about this stage needs a window, which is the point of doing it last. `runtime retire` is *not* part of it: that takes this machine's catalog off the routers, and this machine is still serving.

## Things that look like risks and are not

A publication failure is not an outage: with a file sink configured, the previous catalog is written back and the switch is restored, so the legacy router is serving the model it was serving before. A router that is down is not a blocked machine: an unreachable `router_target` is a warning and a scheduled retry, and switches keep working. A controller restart is not a model restart. And once clients are on the shared router, a switch on Rosalina stops being a household event at all — the new router drains her host and lets the others answer, which is the whole reason for the move.

## Still open

Daisy is unreachable and needs someone at her console; her deployment is staged and waiting, and until she is serving, `daisy` stays an unpublished host on the new router, which is honest and shows as a warning. The client inventory above is what could be read without a password — anything on the network that resolves `ai-router` or names `192.168.1.4:11434` and was not in a readable file has to be found before stage 5, and `GET /v1/router/capabilities` on the legacy router, watched through a quiet hour, is the way to find it.

