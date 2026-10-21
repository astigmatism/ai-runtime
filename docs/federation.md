# One router, more than one machine

This is a design, agreed on 2026-10-20 and not built. Nothing in it is true of the code today except the first section, which describes what exists so that the changes have something to be changes from. It exists because a second machine (`flash-next-strata-iq3-xxs-daisy`, [More than one machine](deployment.md#more-than-one-machine)) wants to answer through the router the household already uses rather than through a router of its own, and because the router is moving to its own machine with its own address — a decision that turns four things that are currently invisible into four things that have to be designed.

## The contract as it runs today

One runtime talks to one router, and the whole surface is three admin calls and one file.

- **Publication is a file the router is told to re-read.** The controller writes `active-model.json` into `router_state_dir` — bind-mounted into the controller container by `runtime/update.py`, and into the router container by the stack — and then calls `POST /admin/api/reload-config`. The document is whole: one configuration's models, their aliases, backends, context ceilings and capability profile, plus `offline_services`. Its validity is whole too. A router that rejects any part of a catalog serves 503 for every model route while `/health` stays green, which is why the ceiling history in [deployment.md](deployment.md) insists on router image first, controller second, in that order and never the reverse.
- **Publication is verified, and that is not obvious from the code shape.** Immediately after the reload the controller reads `GET /admin/api/runtime-state` and *requires* the router's model set to equal the document it just wrote — same ids, all healthy, the promised context window, output policy `unrestricted`, no inherited output default. A mismatch raises `Router catalog/readiness verification failed`, the exact previous document is written back and reloaded, and recovery takes over. A switch therefore cannot complete while the router disagrees about what is live.
- **The drain is one flag on the router, owned by whoever set it.** `POST /admin/api/runtime-drain {enabled, reason}` with `reason` set to `local-ai-runtime:<transaction id>`; ownership is recognised by reading that reason back. Everything the router serves — including work that has nothing to do with the runtime doing the switching — pauses for admissions while it is set.
- **Everything is neighbourly.** Backends are published on loopback (`127.0.0.1:18080`), the catalog names them by container DNS (`http://qwen38-daytime:8080`) over the shared `local-ai-ollama_default` Docker network, and the router reads a path on the same filesystem as the controller that wrote it.

Nothing about that is wrong. It is the shape of one machine, and it is why the second machine has been waiting.

## The four things that stop being invisible when the router has its own address

1. **A loopback port is not reachable.** A router on another host cannot open `127.0.0.1:18080`. The publish address becomes a host setting, and because it is in the rendered bundle, changing it recreates the backend — a cold model start, minutes on Rosalina's pack and 9–11 on Daisy's spindle. That is a release with a chosen moment, not a reload.
2. **A container name is not an address.** `backend_url` has to become a LAN URL for a remote router. It stays a *rendered* fact of the host and not a line in a profile: a profile measured a model on hardware, and which NIC a machine answers on is a deployment setting, exactly like a GPU UUID.
3. **A file is not an API.** The controller and the router no longer share a filesystem, so publication has to cross a network — which is the decision this document records.
4. **One drain is too few.** With two runtimes calling one router, Rosalina's switch must not pause Daisy's admissions, and "another operation owns the drain" has to mean *this machine's* drain.

## The decision: routers pull

The new router is configured with the machines it serves, and each machine is one URL: the runtime's own HTTP surface. The router fetches each machine's active catalog; a machine never writes anything to the router.

That is the whole idea, and it is chosen over a publish API for four reasons. The runtime already serves its catalog over HTTP, so this adds an endpoint and not a subsystem. The runtime stays the only writer of what is live on its own machine, which is the invariant the whole controller is built on — a push model would make the router a service that accepts authenticated writes from principals it does not supervise, and the first time a runtime is mid-transaction while a router is restarting, that difference becomes a recovery path nobody exercised. A third machine is one more URL in the router's configuration and no code anywhere. And the verification the runtime already performs survives: it asks the router what it holds, and the router can only hold what it fetched, so the question is the same and only its answer becomes eventually-consistent.

The costs are honest. The router must reach each runtime, so the runtime's Host allowlist (`RUNTIME_ALLOWED_HOSTS`, already a setting) has to name the router. Publication becomes asynchronous, so "published" means "published and seen", which the runtime has to wait for rather than assume. And a router that cannot reach a machine needs a rule, stated below, instead of an accident.

### The endpoint

`GET /api/active-model` on the runtime, returning the document the file used to carry, unchanged in shape, plus three fields a fetcher needs and a file consumer never did: `configuration.runtime_id` (which is `host_id`, and is what makes two machines' documents tellable apart when they answer the same alias), `configuration.published_at` (already in the document), and `configuration.config_sha256` (already the bundle's identity). It is readable without a token, like `/api/status`: it is a read-only view of a trusted LAN, and the CSRF/Host protections that guard the page's writes are what keep it honest. `/api/status` stays the human surface and keeps its token; a router should not depend on the page's document, which carries operation state and a CSRF token it has no use for.

### Staleness, and what a machine that is unreachable means

A pull is a promise about the past, so the router stamps each document with when it fetched it and the runtime stamps it with when it was published. Three rules, all chosen so that traffic degrades instead of stopping:

- A machine that cannot be reached keeps its last document and stops being *fresh*. Its models still appear, with their backend health as the router sees it. Nothing is removed because a control-plane poll failed: a router that drops a machine's models on a network blip has turned a control-plane failure into a data-plane outage, which is the one thing this design must not do.
- A document older than the freshness bound admits no new work for that machine's backends. Existing streams finish. This is where a machine that is genuinely gone converges: it stops taking work, keeps what was in flight, and never routes to a backend that is not in the last document it fetched.
- A router never serves zero models. If every machine is unreachable it keeps every last document, which is the router's own last-known-good and is deliberately symmetric with `runtime restore`.

### Aliases are the fleet's decision, and only the router's

Every configuration on both machines publishes `local-active`. Two machines behind one router cannot both mean it. `daytime` and `nighttime` name Rosalina's residents and `daisy` names that machine's, which was settled in the profiles because those names are facts about a launch; `local-active` is not a fact about a launch, it is a policy about which machine the household is currently using. So the router's per-machine entry lists the aliases it may publish and drops the rest before they reach `/v1/models`. The runtime keeps publishing what it answers — its own page and its own `offline_services` stay honest about the machine — and the router decides what it forwards. When a machine's whole catalog is filtered empty by that policy the router refuses to load that entry at all, because a machine that can say nothing through this router is a misconfiguration and should say so on startup rather than serve nothing quietly.

## What changes, in the order it has to happen

**R1 — the runtime grows the endpoint.** Controller-only: no bundle changes, no digest moves, nothing is recreated on either machine. The file publication path stays exactly as it is, because the router in front of clients is still the one that reads it.

**R2 — the router grows a scoped drain, and stays backward compatible.** `runtime-drain` gains an optional scope; absent it, the drain is global, which is what every runtime today sends. `runtime-state` reports which scopes are draining. Nothing on Rosalina changes behaviour, and the new router can be proven against the current runtime.

**R3 — the runtime speaks to a remote router.** Publish/advertise address of backends and the authority for `backend_url` come from host settings; the drain call carries the machine's scope; publication waits, within a bound, for the router to report the document it published, and reports which routers saw it and which did not. **This is the release that recreates backends**, because a published port address moved; it is scheduled like the window-160K→256K releases were. Until the new router is the client-facing one, `reload-config` keeps meaning "re-read the file", and the runtime's post-publication verification keeps working unchanged.

**R4 — the new machine goes live, pulling.** It fronts Rosalina first. Clients change base URL, and only then does the file path stop being load-bearing. Old router and new run side by side until the new one has carried real traffic — that is the chosen migration, and it is what keeps publication intact for the whole of it, because the runtime never stops writing the document the old router reads until this moment.

**R5 — Daisy joins the pull list.** Her `host.json` names herself (`host_id: daisy`, `gpu_architectures: ["75"]`), her backends publish on her LAN address, and she needs no router of her own and no `local-ai-ollama_default` network — one of the few parts of this plan that is a simplification. Her first switch is verified the same way Rosalina's is: the router either reports her document inside the bound, or the switch fails the way it fails today, with the previous release restored and the drain reopened.

**R6 — the file and the shared network go away.** `router_state_dir` leaves the host settings, `reload-config` becomes "fetch now" or disappears, and the external network stops being a prerequisite. That is also the release that finally decides what a runtime page says about a router it cannot see.

## What this does not settle

One router is not one portal: Daisy still needs a place to run a controller from, and the home policy that forbids permanent checkouts has to answer that for a machine whose only job was supposed to be ComfyUI. Which machine owns `local-active` is a policy the router holds, and *changing* it while traffic is flowing is a switch in its own right, with the same drain rules. The router's context ceiling is now the ceiling of the union of two machines' catalogs, so a widening is again router-first, controller-second. And a shared vision card is a per-machine fact — Daisy has no card to lend and publishes no vision capability, so anything that assumes a household vision endpoint has to be re-read once two machines can answer.
