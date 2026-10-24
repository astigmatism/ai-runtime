# Moving Rosalina onto the shared router without stopping what is serving

Rosalina has one router, installed beside the models, and every client that talks to a large model on this network talks to it: the DeepSeek Harness on Pauline and on local-ai-voice both resolve `ai-router` to `192.168.1.4`, and ComfyUI's prompt generator on Daisy does the same. The new router at `192.168.1.23` is the machine the runtime now publishes to by push ([federation.md](federation.md)). Moving a model host from one router to the other is not one action, and the reason is that **the runtime is not the risk**. Pointing a runtime at a second router moves catalogs. Moving a client moves a session that is running right now — possibly the one writing this.

So the order below is: move the runtime first, where a mistake is a warning on a page; make the new router able to describe a serving model, which needs one change on the router and one on Rosalina; and move clients last, one at a time, with the old address still working underneath them.

## The rule the order falls out of

**Never let both routers be unable to serve.** The legacy router stays installed, running and fed until the last client has been quiet on the new one for longer than a working day. Everything else — the order of the stages, which step gets a window, which step does not — comes from that one rule, plus the ordinary one that a change which recreates a model is scheduled and a change which does not is not.

## Where it stands

Measured on 2026-10-21, in the order the stages went:

- Stage 0 is done. Rosalina runs the current revision with the model in service untouched, and her page publishes to the file sink exactly as before.
- Stage 3 is done on the router: a pushed catalog is no longer made incomplete by a marker it never needed, the host list is a line in the router's own installer instead of a shell variable in whoever's session deployed it last, and `local` is off the roster because a router that declares a host it can never publish reports itself degraded forever. The client's gate — `validateRouterMetadata` in the Harness plugin — accepts the new router's document today.
- Stage 1 is done, and the receipt is what closes it: `192.168.1.23` accepted Rosalina's catalog at `20:12:55Z` under revision `836d290`, names `flash-next-strata-solo-thinking-recovery-cache` as her configuration, lists `qwen3.8-flash-next-ud-iq4_xs`, reports her as the default host, holds no drain, and carries no warnings — and her page shows the same, with the file sink beside the HTTP one and the model in service never leaving it.
- Stage 2 turned out to cost nothing and is effectively done. She states no `backend_address`, yet the household router reaches her backend — `available: true` on the client's own document, `reachable: true` in the publication receipt — because the backend's port binding, not the declared address, is what the published URL is built from and verified against. The 112-second window this stage was budgeted for is not owed; what remains of it is a documentation matter, whether to have Rosalina *state* the address she is answering on, so the claim is explicit rather than observed.
- Stage 4 has not been started: no client has been repointed. That is now the only step that costs anything, and it costs one client at a time, one line each, reversible in the same edit. It is also blocked, and the blocker is a router matter rather than a runtime one: an alias a machine publishes as a bare name is rendered with its model's display name, so a client picker lists one model several times under one indistinguishable title, and the Harness boot gate — `verify-resident-client.mjs`, which addresses picker rows strictly — fails on every machine while that is true. It fails against the legacy router identically, so it predates this move; what made it visible is that solo configurations publish `daytime` and `nighttime` as aliases of one model instead of two models with different titles. Handed to the session that owns the router: `/home/astigmatism/docs/handoff-router-alias-display-names-20261021.md`.

Measured again after stage 1, because the plan is worth less than the reading:

| Fact | Value | Where it comes from |
| --- | --- | --- |
| Rosalina's publication | two sinks, `file` and `http://192.168.1.23:11435`, revision `836d290`, accepted `2026-10-21T20:12:55Z`, `router_default: rosalina`, no warnings | `GET http://192.168.1.4:11436/api/status` |
| The household router serves her model to a client | `daytime: available=true`, metadata accepted, non-blocking | `validateRouterMetadata`/`residentStatus` in the Harness plugin, run against `192.168.1.23` |
| The legacy router | untouched, `complete: true`, `daytime` available, no warnings | the same gate against `192.168.1.4` |
| The second machine is published too | `daisy` — installed `flash-next-strata-iq3-xxs-daisy` on 2026-10-22, backend `qwen38-daytime` healthy, catalog `qwen3.8-flash-next-iq3_xxs` accepted, revision `5a1f82bd`, no warnings; the deployment's document now reads `complete: true` with both hosts published | `GET /admin/api/hosts`, `GET /v1/router/capabilities` on `192.168.1.23` |

The row that used to read "`daisy` — powered off, unpublished, `complete: false`" is now closed, and the way it closed is worth recording because the new machine was the first to do two things the runtime had never had to do. `HOST_CATALOG_UNPUBLISHED:daisy` cleared the moment the switch published her catalog to `192.168.1.23`, and the deployment's document read `complete: true` — completeness really is a property of the deployment, so a second machine that *is* there is what makes it true, and the earlier `false` had been honest all along.

Getting the switch to run at all took two source fixes, because Daisy is a machine that only *pushes*: it has no router of its own, no `router_state_dir`, no local `router-token`. The first, landed as `31572ff`, is that a switch asked the *legacy local admin port* which backend kinds the router supports before it would drain — a port that on a push-only host has no file to read, so the pre-drain gate failed closed and refused the very first install even though the router over the road reported `strata` in its supported kinds. It now asks every sink the machine publishes to, healing a file router the way the drain does and tolerating a migration mirror that is merely down; a refusal still fails exactly as before. The second, `5b633c6`, is that `runtime install` — the command the bring-up runbook tells an operator to run — was accepted by the argument parser but missing from the branch that calls the transition, so it fell through every branch and died printing a `result` it never assigned; it now shares that branch with `apply`/`adopt`/`deploy`. Both carry tests. Daisy's running controller is on `31572ff` (the fix the switch itself needed); the CLI fix reaches her at the next controller deploy and needs no window, since it is a dispatch table and not a switch.

## What is true right now, measured

| Fact | Value | Where it comes from |
| --- | --- | --- |
| Rosalina's model in service | `flash-next-strata-solo-thinking-recovery-cache`, service IDs `local-active`, `daytime`, `nighttime` | `.state/active.json` |
| Warm boot of that model | **112.5 s** | `.state/provisioning/flash-next-strata-solo-thinking-recovery-cache.json` |
| A switch drains the legacy router globally | yes, as it has always done | today's behaviour |
| A switch drains the new router | per host (`runtime-drain {host: rosalina}`), leaving other hosts serving | `runtime/controller.py`, `tests/test_publication.py` |
| What a client sends today | `http://ai-router:11434/v1`, `openai-responses`, models `daytime` and `nighttime`, `maxConcurrency: 1`, two retries on `SERVER`/`TRANSPORT` | `deployments/deepseek-harness/config/settings.yaml` |
| How `ai-router` resolves | `extra_hosts` line `"ai-router=192.168.1.4"`, one line per client host | Pauline and local-ai-voice `compose.yaml` |
| What the DSH discovery plugin demands of a router | per-model `metadata.schema_version == 2` and `metadata.complete === true`; `available !== true` marks the model unavailable | `.dsh-plugins/dsh-router-model-discovery.js` |
| What the new router reports for Rosalina today | `available: false`, `metadata.complete: false`, warnings `["ACTIVE_MODEL_MARKER_UNAVAILABLE"]`, `metadata.schema_version: 2`, capabilities document `schema_version: 2` | `GET http://192.168.1.23:11434/v1/router/capabilities` |

## Two things on the router block the client move, and neither is the runtime

*(Both of these were fixed on the router on 2026-10-21; the analysis is kept because it is the reason the router is what it is now, and because the second one came back in a new form.)*

**A pushed catalog cannot be `complete` on the new router, and a client refuses an incomplete one.** The router computes a model's metadata completeness as *no warnings at all*, and it adds `ACTIVE_MODEL_MARKER_UNAVAILABLE` to every catalog whose `loadedFrom` is not `file` — which is every catalog that was pushed. The DSH plugin throws `router metadata is incomplete` on exactly that field and marks the model `incomplete`, which is one of its blocking statuses. So a client repointed at `192.168.1.23` today has no usable model, and it would have none even with a reachable backend, for a reason that is an artifact of the single-host router it grew out of. The fix belongs to the router: a catalog that arrived by push has no local marker to be missing, so the warning should not be pushed for it. The strictness of `complete` is worth keeping — it is the reason a half-known model is never called usable.

**`ROUTER_HOSTS` still declares `local`, and nothing publishes to it.** That is why `/health` says `degraded` and capabilities says `complete: false` at the document level. It is not a blocker for the DSH client — the plugin only logs a `warnOnce` about the capabilities `schema_version` bump — but a router that reports itself degraded while everything works is a router nobody trusts at 3 a.m. Either publish something for that host or take it out of the list.

Both changes are one deploy of the router, and the deploy is free of client impact for as long as no client points at it.

## The stages

| # | Action | What moves | Client impact | Rollback |
| --- | --- | --- | --- | --- |
| 0 | Update Rosalina to the revision with the publisher, changing no setting | controller image only | none — backends are not touched by a controller restart | stay on the old revision |
| 1 | Add `router_target` and the token; keep `router_state_dir` | publication gains a second sink | none — startup publication pushes, and a push drains nothing | remove the two settings, restart the controller |
| 2 | State `backend_address` | port binding `127.0.0.1:18080` → `192.168.1.4:18080`, and the catalog URL with it | **the window**: about 2 minutes of refused requests on the legacy router while the selected configuration reloads, which is what every switch does now | remove the setting and switch again (another window) |
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

**Stage 2 — the reachability window, and the only one.** Stating `backend_address: "192.168.1.4"` moves the published port and the advertised URL together, which moves `config_sha256`, which means every model of the configuration is stopped and loaded again. Measured warm boot of the configuration serving now is 112.5 seconds; add the drain and the verification and call it two to three minutes during which the legacy router refuses new admissions. The drain waits for in-flight requests rather than cutting them, so a turn already streaming finishes; what does not get served is a request started inside the window, and the DSH retry policy (two retries) will usually not outlast it. **The arithmetic, now that a switch replaces a configuration rather than patching it:** the pause is the full load of the configuration being selected — every model it names, brought up one at a time — and not just the load of whichever one model differs. Rosalina serves one model, so any switch from here costs that one load (112.5 s warm) plus drain and verification, and this setting pays it because it changes the rendered launch. Selecting a configuration that names two models costs both, in the order it names them; nothing shared between the two configurations is kept serving, and there is no shorter case to schedule around. So: pick a moment when the page shows nothing active and no session is mid-task, say so out loud, and do it. Take the opportunity in the same window to confirm the thing stage 4 depends on — `GET http://192.168.1.23:11434/v1/router/capabilities` reporting `available: true` for `daytime` and no `ROUTER_CANNOT_REACH_BACKEND:rosalina` warning.

**Stage 3 — make the new router describable.** Land the router's completeness fix and the `ROUTER_HOSTS` cleanup, redeploy, and read back what a *client* would read: the capabilities document, run through the DSH plugin's own `validateRouterMetadata` against a live document rather than a fixture. That check is the gate for stage 4 and it costs nothing to run again later — the plugin is an ES module in the Harness deployment's plugin directory, so it can be imported and called directly against a live router:

```js
import { validateRouterMetadata, residentStatus, isBlockingStatus }
  from '/home/astigmatism/deployments/deepseek-harness/data/dsh/.dsh-plugins/dsh-router-model-discovery.js';
const document = await (await fetch('http://192.168.1.23:11434/v1/router/capabilities')).json();
for (const model of document.models) {
  validateRouterMetadata(model.metadata);                 // throws: the client refuses this model
  const status = residentStatus(document, model.service); // the state the client would show
  console.log(model.service, status.status, isBlockingStatus(status.status));
  // `isBlockingStatus` takes the status string, not the record: handed the record it answers "not
  // blocking" for a model the client will refuse, which is the one mistake worth naming here.
}
```

Run against both routers on 2026-10-21: the legacy router's document is accepted (`daytime` available), and the new router's is accepted too, with `daytime` reading *unavailable* and not blocking — which is the honest answer while its backend is a loopback claim, and what stage 2 turns off. A model that reads `unavailable` is a model the client will not route to, so this check passing is necessary and not sufficient; the sufficient check is after stage 2, and again after each client moves.

**Stage 4 — one client, then the next.** Because every client resolves `ai-router` through an `extra_hosts` line, the repoint does not touch a client's configuration at all: same URL, same API key, same retry policy, same service IDs. Change one line, recreate that one container, and check the client's own view of the router before declaring the client moved. Do the voice box first — a session lost there costs a conversation, and the harness on Pauline is the one holding live agent sessions. **Restarting the harness container interrupts every session inside it, including the agent doing the move**, so stage 4 is done at a moment when this session can be spared, and the check afterwards is that the discovery plugin reports the models it routes to `available`, not that a request happened to succeed.

**Stage 5 — the legacy router goes last, quietly.** Drop `router_state_dir` from `host.json` (a controller restart, no drain — the file it wrote keeps working until it stops being read), stop the legacy router container, and leave the model port where stage 2 put it. Nothing about this stage needs a window, which is the point of doing it last. `runtime retire` is *not* part of it: that takes this machine's catalog off the routers, and this machine is still serving.

## Things that look like risks and are not

A publication failure is not an outage: with a file sink configured, the previous catalog is written back and the switch is restored, so the legacy router is serving the model it was serving before. A router that is down is not a blocked machine: an unreachable `router_target` is a warning and a scheduled retry, and switches keep working. A controller restart is not a model restart. And once clients are on the shared router, a switch on Rosalina stops being a household event at all — the new router drains her host and lets the others answer, which is the whole reason for the move.

## Still open

Daisy is up: her fresh install was brought up through the bootstrap runbook (host foundation, NVIDIA driver and Container Toolkit, Service Portal, Harness, then this AI Runtime profile) and installed `flash-next-strata-iq3-xxs-daisy` on 2026-10-22, publishing `qwen3.8-flash-next-iq3_xxs` to the household router. She pushes only (`router_target`, no `router_state_dir`), which is what surfaced the two source fixes recorded above. What is genuinely left:

- **The voice box — moved (2026-10-22).** Its DSH now resolves `ai-router` to `192.168.1.23` (its `REMOTE_OLLAMA_HOST`), the resident-label plugin is deployed (running image `e011298`), and its boot unit reaches `active (exited)` cleanly: the loop it had was a stale unit definition with a finite `TimeoutStartSec` that SIGTERMed the maintenance worker (exit 143) before the bounded health wait could finish, so the fix was the update's own unit (`TimeoutStartSec=infinity`) plus a host-side `systemctl --user daemon-reload`, not the plugin. It reads the router's `schema_version 2` document through the plugin's legacy path and reports its models available.
- **Stage 5 — executed on Rosalina (2026-10-22).** Her controller was first updated to current main, because once `router_state_dir` leaves `host.json` the only admin door left is the router she is about to stop, and a switch after that has to ask the router it actually publishes to — the sink-aware kind gate. She now has a single `http://192.168.1.23:11435` sink, `ready`/`serving` true, both backends untouched, `.23` roster clean with no warnings. The legacy `11434` router container is **still running** as the last client's backstop: stopping it is the genuinely last action, taken only when nothing resolves `192.168.1.4` any more.
- **Pauline — the last client.** Its `ai-router` is `${REMOTE_OLLAMA_HOST:-192.168.1.4}` in the source template, so the cutover is a deployment `.env` value, `REMOTE_OLLAMA_HOST=192.168.1.23`, and a managed recreate; nothing in `compose.yaml` is hand-edited (the deployment copy is generated from the checkout and would be overwritten). The recreate interrupts every session in the Harness container, including the agent performing it, so it is a scheduled moment and the check afterwards is the discovery plugin reporting its models available — a request that happens to succeed is not the check.
- **The client inventory.** `GET /v1/router/capabilities` on the legacy `192.168.1.4` router, watched through a quiet hour, is how anything not in a readable file gets found before that container is stopped; nothing should name `192.168.1.4:11434` once Pauline is across.

