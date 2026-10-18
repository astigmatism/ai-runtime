# AI Runtime

Versioned startup, launch configuration, and catalog publication for Rosalina's resident model services: normally a coding resident and an everyday resident on the two text GPU pairs, or one solo resident across all four text GPUs. The controller shows GPU assignments, switches the registered configurations, and integrates with Service Portal's **Update and restart** action.

The repository is **`ai-runtime`**, formerly `local-ai-runtime`. The application name remains **AI Runtime**. See [repository rename and compatibility](docs/renaming.md) for existing installations.

```sh
git clone https://github.com/astigmatism/ai-runtime.git
cd ai-runtime
```

This is the model lifecycle controller. Inference requests go through [LLM Router](https://github.com/astigmatism/llm-router); AI Runtime exposes a status and configuration-switching UI/API, with a container CLI for recovery and administration.

## Current configurations

Choosing a configuration sets the coding resident and whether an everyday resident runs beside it. The status page lists both for every configuration.

| Configuration | Residents | Coding resident | Everyday resident | GPUs |
| --- | --- | --- | --- | --- |
| `qwen27b-q8-pair` | 2 | Qwen3.8 27B Q8, tensor-parallel (55,45), MTP3, 160K | Qwen3.8 27B Abliterated Q6_K, tensor-parallel (60,40), MTP3, 96K | Coding: RTX 3090 + RTX 4080 SUPER; Everyday: RTX 4080 + RTX 3080 Ti; RTX 3080 shared for vision |
| `qwen27b-q6k-pair` | 2 | Qwen3.8 27B Q6_K (Unsloth UD-Q6_K_XL), otherwise as above, 160K | Same | Same |
| `flash-next-strata-solo-thinking-recovery-cache` | 1 | Qwen3.8 Flash-Next, 160K, served by Strata `v0.1.44.1` from Unsloth UD-IQ4_XS (4-bit, 94 GB), text and images; `flash-next-strata-solo-thinking-recovery` is the same launch with conversations not parked | None: the coding resident answers `nighttime` from its own backend | All four text GPUs, plus the RTX 3080 for the image encoder |

The everyday resident has run MTP3 since 2026-10-07, using the base 27B MTP head on the shared RTX 3080.
- **Speed:** a 16K planning prompt decoded at 65.7 tokens per second, against 37.8 without a draft.
- **Context:** 96K, because at 128K the speculative buffers did not fit on the RTX 4080.
- **Hosts without a vision GPU:** the everyday resident runs without the draft.
- **Variants:** a paired configuration may name a different everyday variant with `"everyday"`. Switching between variants in the browser recreates only the everyday resident.

See [speculative decoding experiments](docs/deployment.md#speculative-decoding-experiments). See [deployment](docs/deployment.md#speculative-decoding-experiments).

`qwen27b-q8-pair` is the default when a host has no recorded release. The everyday resident's RAM prompt cache is capped at 24 GiB; the coding resident uses the shared 48 GiB. `primary` means the selected configuration; it is not a clock-based schedule.
- **27B configurations:** each was benchmarked on a full Bench Studio coding session (2/2 passed), at about 26.1 (Q8) and 27.0 (Q6_K) steps per second.
- **Flash-Next solo configurations (retired 2026-10-15):** the llama.cpp layout kept the whole model in VRAM on all four text GPUs. On a fixed session replay recorded 2026-10-06, `flash-next-solo-128k` decoded at about 94 tokens per second, and decode slowed as the conversation's context filled: about 56 tokens per second at 64K and 36 at 120K.
- **The Strata resident:** added 2026-10-11 as a trial reached only through its own `trial` service ID, `flash-next-strata-solo-thinking-recovery-cache` serves as an ordinary resident: it publishes `local-active`, `daytime` and `nighttime`, so a client that only ever sends `daytime` — which is what LLM Router's `CLIENT_CONTRACT.md` tells it to do — always has something to send, and `nighttime` is answered from this same backend. It reads pictures as well since 2026-10-16: the image encoder is a second process the engine starts, placed on the RTX 3080 — the shared vision card, which loads no second resident here and so is not shared with anything. It is also the one backend that is not llama.cpp, configured by the environment instead of an argument vector. Measured on 2026-10-14 over 771 served requests, it answers the question it was added for: **decode is flat in context — 112 tokens per second at 4K and 108 at 100–131K, against 97 falling to 35 for the llama.cpp layout on the same cards** — median 110.7, with 10% of long generations below 100 and 94% of prompt time going to re-reading text rather than to generation. Those two sides were not decoded alike: none of those requests named a sampling, and with no sampling block the engine answered greedily while llama.cpp answered at its server default — the model card's set. Every configuration of this engine states that set now; the greedy arm it replaced was retired on 2026-10-18. See [deployment](docs/deployment.md#strata-solo-configuration-flash-next-strata-solo-thinking-recovery).
- **Two configurations, one resident:** `flash-next-strata-solo-thinking-recovery` decodes at the Qwen3.8-Flash-Next card's thinking set, which is what the same request gets from the llama.cpp residents, and turns the engine's tool-call recovery on, so a declared tool's call written beside the template's form is read as a call instead of as prose. `flash-next-strata-solo-thinking-recovery-cache` is the one step on: same image, same weights, same workspace, same window, with the engine's conversation cache turned on and sized. The two differ in a single variable, which `AVariantIsOneLineApartFromItsOriginal` holds them to, and moving between them is a portal act on the configuration, not an edit and a deploy. There were two more steps here until 2026-10-17: this resident decoded greedily, and the card's set was offered beside it as a second configuration so the move was reversible at the portal. Sampling costs a drain in either direction, and on this host's own replay it costs a median 0.4% of decode where upstream reports about 6% for a corpus peaked at one answer; recovery is a parser over the tools a request itself declared and costs nothing on the decode path. The greedy pair was retired the day after the levers were measured — one sampling policy for the whole house, and the reason to keep both arms open had been spent. What the switches buy is conformance to the card, the end of the greedy repetition loop upstream reports, and calls that arrive as calls — not a measured accuracy gain, and not a repair this house needed: scanned for the markup a dropped call leaves in the transcript, 0 of this resident's 16,702 turns that wanted a tool lost one (`scripts/transcript_loops.py --dropped`).
- **The two steps that were measured, and what they are now:** the chain did not stop at conformance. `flash-next-strata-solo-thinking-recovery-full-route` sets the engine's own `STRATA_ROUTE_TAIL_SKIP=0`, which is on by default and skipping 96.1% of the expert routes this box misses: upstream prices the skip at KL 0.003–0.006 and switches it off itself when the expert cache runs near full, and a resident holding a 55 GiB resident budget with 88.3% of prompt tokens already reused is the full-cache side of that trade — the configuration existed to find out whether the skip is costing this box decode instead of saving it, and it was retired the day it was declined — the variable, the measurement and the way back are in [deployment](docs/deployment.md#strata-solo-configuration-flash-next-strata-solo-thinking-recovery). `flash-next-strata-solo-thinking-recovery-cache` turns on the conversation cache the engine leaves off, at a budget sized from this box's own engine log (6.3% of prompt time is a conversation being read again; not the 94% this page used to quote, which is upstream's benchmark) and above a floor that keeps the park clear of the page-locked expert complement. Both have now been measured on this box, and they came back different: the route one is **declined** — the skip demonstrably removes sevenfold the expert traffic over PCIe (at most 284 reads in a cache window against 1,942 and 2,878 with it off) while the speed difference is smaller than the instrument can see, and the KL the flag trades is not something a bench weighs — and the conversation cache is **selected**, at 5.1× off a resumed turn: 76,620 tokens read in 31.6 s becomes 5,363 read in 6.4 s when the conversation returns to a parked checkpoint, for 200–820 ms and a fraction of its 16 GiB budget. `scripts/strata_park_probe.py` is the instrument that sees the second one and `scripts/strata_bench_compare.py` is the one that refuses to read the first: the benchmark cannot find a conversation cache at all, because it never lets its own conversations go cold.
- **Selecting a solo configuration:** it drains both models, stops and removes the everyday resident, and publishes a one-model catalog. What a client can no longer reach is published as `offline_services`; `flash-next-strata-solo-thinking-recovery-cache` publishes none, because its own resident answers `nighttime`. See [deployment](docs/deployment.md#exclusive-solo-configurations) for placement, measurements, and recovery.
- **Consolidation (2026-10-07):** the configurations were renamed and the superseded ones retired. Each kept configuration launches the same backend and model alias as before, so its Bench Studio history continues:

  | Old name | New name |
  | --- | --- |
  | `daytime-27b-tensor-next` | `qwen27b-q8-with-nighttime` |
  | `daytime-27b-q6k-tensor-next` | `qwen27b-q6k-with-nighttime` |
  | `daytime-flash-solo-tuned-mtp3` | `flash-next-solo-128k` |
  | `daytime-flash-solo-tuned-mtp3-160k` | `flash-next-solo-160k` |
- **Renaming and retirement (2026-10-15):** the ids that named a time of day now name what the configuration loads. A renamed id still resolves and launches byte-identical backends; a retired id refuses and names what to select instead.

  | Old name | Now |
  | --- | --- |
  | `qwen27b-q8-with-nighttime` | `qwen27b-q8-pair` |
  | `qwen27b-q6k-with-nighttime` | `qwen27b-q6k-pair` |
  | `flash-next-strata-160k` | `flash-next-strata-solo`, and retired with it on 2026-10-18 |
  | `flash-next-solo-128k`, `flash-next-solo-160k` | retired; `flash-next-strata-solo-thinking-recovery` replaces them |
- **Retired:**
  - `flash-next-strata-solo`, `flash-next-strata-solo-thinking` and `flash-next-strata-solo-thinking-recovery-full-route`: the three Strata variants that were levers rather than residents, retired 2026-10-18 once each had been measured or had spent its purpose — greedy decoding, the card-sampling pair, and the route-tail skip that measured a wash. They refuse and name `flash-next-strata-solo-thinking-recovery`; what they mount is stated there, and releasing it takes the last configuration of the engine going, not one name.
  - `daytime`, `daytime-flash-f16`, and `daytime-flash-next`: two-GPU Flash-Next at about 31 tokens per second.
  - `daytime-27b`: layer-split Q8.
  - `daytime-flash-solo` and `daytime-flash-solo-tuned`: superseded by MTP3.
  - `flash-next-solo-two-requests` (formerly `daytime-flash-solo-tuned-mtp3-2slot`): retired the same day. Two requests at once ran each at about half speed. A solo configuration can still set `"parallel_slots": 2`, and LLM Router admits two requests for such a resident.
  - The `flash-next-mtp` engine they used is no longer referenced.

The initial migration preserves the container names `qwen38-daytime` and `qwen38-nighttime`, the `local-ai-primary` backend Compose project, `local-ai-ollama_default`, loopback inference ports 18080/18081, existing model files, and each service's pinned engine image. Every Qwen3.8-27B backend (the everyday resident and both 27B configurations) uses `836d57176dc699a726c55418e4f96b8ca628e1bf` (engine `qwen38-dual-836d571`), and `flash-next-strata-solo` uses `296b59d2f73c8f04a805df29c2fa97655adace78` (engine `strata-vision-296b59d2f73c`, the same revision with its image encoder compiled in; the text-only engine `strata-296b59d2f73c` stays registered while it is under evaluation). The `qwen38-dual-43fe9c6` engine the retired Flash-Next solo configurations used is retired with them, as are the original `8ea2902` engine and the Flash-Next branch engine `d1a92352c`. The controller uses a separate Compose project, `local-ai-runtime`, and no GPUs.

The September 16 source refresh imports the running Flash-Next configuration exactly, including its 33 model shards, CPU/GPU tensor overrides, lazy lookup-table reads, 2048 microbatch, CPU vision projector, and MTP draft on CUDA0. It preserves the saved 27B profile and keeps Swift retired. This source refresh does not itself deploy or migrate the running host controller.

## Edit, publish, deploy

1. Edit and test on the development Mac.
2. Commit and push `main` to this public repository.
3. Select **Update and restart** beside AI Runtime in Service Portal.

The development checkout is `/Users/astigmatism/Projects/ai-runtime`; Rosalina's `/home/astigmatism/deployments/ai-runtime` is a deployment checkout. Do not edit or commit application source on Rosalina or modify running containers. Leave checkout advancement to the updater so the checkout, image, and active release stay aligned. See [agent instructions](AGENTS.md).

Rosalina reads public GitHub over HTTPS. It requires no GitHub write credentials. The updater fetches only fast-forward changes, builds an image marked with the exact source commit, validates it before draining work, and reconciles only changed backends. The source checkout advances after the candidate controller is healthy. A failed update retains the previous image and configuration, and reports its recovery state. Read [deployment and recovery](docs/deployment.md) before the initial migration.

### Where to change settings

- `config/shared.json`: shared launch arguments, container defaults, output/reasoning policy, network, and named engine identities.
- `config/profiles/*.json`: one definition for each selectable configuration and for the everyday resident. A profile with `"exclusive": true` and `"gpu_group": "all"` runs alone on both text pairs (the `daytime` pair's UUIDs, then the `nighttime` pair's, as CUDA0–CUDA3, with the vision GPU as CUDA4) and requires a configured `vision_gpu_id`; an environment-configured engine reserves the four text GPUs alone. Each selects an `engine` from the shared registry. Change `context_tokens` to update both launch flags, the manifest, and router discovery together. Per-profile `arguments` override shared arguments. `argument_order` preserves the exact invocation and must list each effective argument once; an ordered list for `--override-tensor` expands to repeated flags without losing placement rules. Model, projector, and draft metadata are derived from their actual argument paths and declared artifact mounts. A profile is selectable only when its id is listed in `CONFIGURATIONS` (`runtime/config.py`); the status page, `~/local-ai-config.sh list`, `~/local-ai-config.sh names`, and `--profile` all read that one registry.
- `.state/host.json`: private host deployment settings, GPU UUIDs, model root, and router integration paths. Start with `config/host.example.json` for another machine. The migration imports Rosalina's existing settings automatically.

All profiles support a shared vision GPU through Runtime's [versioned renderer and host configuration](docs/shared-vision.md). The source defines projector offload, CUDA ordering, placement validation, and catalog metadata; the host supplies hardware UUIDs. Rosalina uses the RTX 3080 as CUDA2 for vision in both residents of a paired configuration. CPU vision remains the default when the optional setting is absent. Start with [the shared-vision host example](config/host.shared-vision.example.json) when configuring another such host; its UUIDs are synthetic placeholders, not live hardware.

For example, changing the coding resident's `context_tokens` to `98304` publishes 96K consistently and recreates only that resident after drain. Model files are referenced by path relative to the model root and SHA-256; they are never included in Git or the image. A changed file invalidates the checksum cache even when its size stays the same.

The initial implementation preserves the existing single-slot and unrestricted generation contracts. Context is limited to 262144 tokens, the Qwen3.8 native window, by the current router contract (`ROUTER_CONTEXT_LIMIT`). Routers older than LLM Router `7235329` accept at most 163840 tokens and reject a larger catalog, after which the switch restores the previous configuration. Changes outside those contracts require a coordinated router/runtime release. The page shows the active GPU assignments and switches between registered configurations; it cannot edit configuration. Existing SSH configuration commands remain available.

Integrity checks prove that the launch configuration, engine, and model artifacts agree. The short post-change generation check establishes basic operation. Neither is a new throughput benchmark, full-context qualification, or VRAM soak test. `docs/import-provenance.json` records the original import without fabricating new qualification receipts.

## Status and commands

Open `http://192.168.1.4:11436` for the GPU overview, active model/context details, readiness, request counts, and source-deployment results. Select a configuration and press **Switch to…** to apply it. Existing requests finish first; new requests pause for both models while the everyday resident stays loaded, except that selecting or leaving a solo configuration (which asks for confirmation) stops or restarts the everyday resident. Progress survives page refreshes and disconnections. The GPU rail shows assignments and model-service health, not hardware telemetry. The page polls same-origin status and never receives router credentials or model paths.

| Command | Behavior |
| --- | --- |
| `~/primary status` | Current configuration, revision, and health |
| `~/primary` | Ensure the selected configuration is running |
| `~/local-ai-config.sh apply qwen27b-q6k-pair` | Select a configuration by name; drain first, and stop or start the everyday resident as the configuration requires |
| `~/daytime`, `~/daytime-27b` | Retired configurations; they refuse and list the current names |
| `~/local-ai-config.sh list` | List the supported configurations, generated from the configuration registry |
| `~/local-ai-config.sh gpus` | Host GPU status |
| `docker exec local-ai-runtime python3 -m runtime check` | Exit nonzero unless the deployed runtime and router discovery are ready |
| `docker exec local-ai-runtime python3 -m runtime publish` | Verify and republish the selected catalog |
| `docker exec local-ai-runtime python3 -m runtime restore` | Bring back the release that last served and reopen requests, whether or not a failure left the router drained |
| `docker exec local-ai-runtime python3 -m runtime provision --profile <name>` | Measure a self-provisioning engine's workspace build (`--phase warm` for the boot) on this host, as evidence; the switch to it builds and waits for the engine's answer itself |
| `docker exec local-ai-runtime python3 -m runtime gate --profile <name>` | Read-only: what a switch to this configuration would do (notes) and what would refuse it (reasons), without attempting one; exit code 1 only when it would be refused |
| `scripts/recover-update.sh` | Complete or recover an interrupted portal update |

`nighttime`, `daytime-256`, and `nighttime-256` are home-wrapper names for a resident rather than configurations, and they retain their existing compatibility meaning: ensure the selected configuration. Historical `deploy` and `rollback` commands are retired. Stopping the controller does not unload the model servers. Backend restart policies remain in place. Controller startup is idempotent and rejects stale source revisions or unfinished maintenance rather than overwriting them.

### HTTP interface

- `GET /api/status`: revision, active configuration, configuration registry, GPU assignments, model readiness, request counts, source deployment, latest browser operation, switching availability, the service-restoration contract (`service.serving`/`restorable`/`pending`/`profile`, plus `restoration` with its attempts, reason, and next attempt), restore availability, and a process-scoped CSRF token.
- `POST /api/profile-switch`: JSON `{profile, request_id, expected_profile, expected_revision}`. `request_id` is a lowercase UUID; the expected values come from status. Send the same-origin `Origin` and `X-Runtime-CSRF` headers. Returns `202 {operation_id, operation}` immediately; repeated identical IDs return `200` and the existing receipt. Reusing an ID for different parameters, stale selection, or maintenance conflicts returns `409`. Invalid bodies return `400`, cross-origin/token failures `403`, and non-JSON requests `415`.
- `POST /api/restore-last-known-good`: JSON `{request_id}`, same headers. The one action that brings back the release which last served and reopens requests — the same path the supervision loop and `runtime restore` take, and it is not refused for being mid-transaction or not ready. Only a competing owner of the router or the checkout, or a host with no recorded release, returns `409`.
- `GET /api/operations/<id>`: a sanitized durable receipt, or `404`. `kind` is `switch` or `restore`; status is `running`, `succeeded`, `failed`, `recovered`, or `needs-attention`; live phases are `checking`, `draining`, `loading`, `verifying`, and `restoring`.
- `GET /healthz`: `200 {"ready": true}` only after startup reconciliation and verification and outside an active browser switch; otherwise `503`.
- `GET /`, `/app.js`, `/style.css`: runtime UI. Other mutation endpoints/methods return `405`.

### What the router learns from each publication

The published catalog is the router's source for what clients may use. In addition to each model's context, slots, policy, and capability profile, the catalog includes these fields:

- **`configuration`** (catalog root): `{id, exclusive, runtime_revision, published_at}`. It identifies the selected configuration and the controller revision that published it.
- **`offline_services`** (catalog root): the service IDs this configuration deliberately leaves unanswered, each as `{model, aliases, display_name, role, reason: "exclusive_configuration"}`. The rule is about service IDs, not roles: `flash-next-strata-solo` loads one resident and publishes `[]`, because that resident answers `nighttime` itself. LLM Router answers a request for a listed ID with `503 SERVICE_OFFLINE` instead of `404`. Paired configurations publish `[]`. The placement fact is separate and lives on `/api/status` as `offline_roles`, which names a role this configuration does not load.
- **`gpu_names`** (per entry): text GPU card names, in the same order as `text_gpu_uuids`/`gpu_uuids`.
- **`vision_gpu_name`** (per entry): set when the projector runs on the vision GPU.
- **`capability_profile.nsfw`** (per entry, required): `true` for an abliterated model (refusals removed), `false` otherwise. Today only the everyday resident is `true`. Rendering refuses a profile that does not declare it. LLM Router publishes it as `nsfw`, beside an automatic `capability_score`, so clients can ask for the most capable NSFW model.

LLM Router publishes names and counts, never UUIDs, paths, or backend URLs. It combines them with what each backend process reports (`/slots`, `/props`, `/v1/models`) and its own admission state, and serves the result at `GET /v1/router/capabilities`. Changes are pushed at `GET /v1/router/events`. A switch therefore reaches subscribers as: drain, the new catalog, ready. Routers that predate these fields ignore them. These fields change the rendered catalog, so the first update that includes them drains briefly and republishes. It recreates no backend.

This is a trusted-LAN administration surface without a login: anyone who can reach it can switch configurations. The current bind address is preserved. JSON, same-origin checks, a CSRF token, and a Host allowlist protect browser requests; they are not user authentication. `RUNTIME_ALLOWED_HOSTS` in the existing Git-excluded `.env` accepts comma-separated hostnames/IPs without ports. It defaults to `RUNTIME_BIND_IP` and loopback addresses. Add any hostname used to open the page to this setting; unrecognized hosts return `421`. The direct HTTP listener does not trust forwarded origin headers.

The runtime and updater locks cover admission through completion, including validation and recovery. No switch is queued. Operation receipts live in `.state/operations/` and contain private diagnostic errors; HTTP responses expose only safe status messages. A restarted controller reconciles receipts against the existing runtime transaction and never automatically retries an interrupted switch. Full failure diagnostics remain in the private transaction/operation journals and the CLI. Source-deployment results come from the Portal updater journal and are displayed separately from profile-switch results.

### Service restoration: nothing serving is never the resting state

Every destructive transition has to be survivable without human intervention, and the recovery path has to be exercised by tests against the exact failures the deploy paths can produce. This house is run by agents, and a human with no lever is the failure mode, not the fallback.

The runtime therefore never comes to rest with no backend serving and no automatic action pending. A backend whose `/slots` cannot be fetched is serving nothing, so it is idle rather than busy, and a drain deadline during a restoration means continue rather than abort: the router already rejects everything, so there is nothing left to be polite to. A deployment that replaced a backend and failed has the release that last served restored through the same finalize path a healthy boot uses; if that attempt fails it schedules the next one on a capped backoff, and the supervision loop runs it until the service comes back. The page states the contract (`service`, `restoration`) and offers **Restore last-known-good** as a single action; `docker exec local-ai-runtime python3 -m runtime restore` is the same call. See [service restoration](docs/deployment.md#service-restoration-the-host-never-rests-with-nothing-serving), and `tests/test_restoration.py` for the failures it is tested against.

## Development

Python 3.12 or newer, Git, Docker with Buildx, and Docker Compose are required for the full workflow. The Python implementation uses only the standard library.

```sh
python3 -m unittest discover -s tests -v
sh -n scripts/update-and-restart.sh scripts/recover-update.sh scripts/migrate.sh
docker build --build-arg SOURCE_REVISION="$(git rev-parse HEAD)" -t local/ai-runtime:test .
docker run --rm local/ai-runtime:test python3 -m unittest discover -s tests -v
```

On an ARM development machine, use `docker buildx build --platform linux/amd64 --load` for an image intended for Rosalina. Build from committed source for deployment. An optional GitHub Actions template is provided at `docs/ci-workflow.example.yml`. Enabling it under `.github/workflows/` requires GitHub workflow permission, which the initial publication credential did not have. The deployment updater runs the image test suite before every changed release; no production credentials or GPUs are needed for those tests.

## Ownership and persistent data

For replacement-disk recovery, see [the rebuild runbook](docs/rebuild.md). The host-side `python3 -m runtime.recovery export --destination /private/backup/path` command captures private state, Git history and exact images without stopping generations. Model weights and other services' persistent data need separate backups. `python3 -m runtime.recovery install-wrappers` recreates the home commands without any legacy host application.

The runtime image includes Python, Git, Docker CLI, Buildx, and Compose. Service Portal runs the updater as the checkout owner, with the Docker socket group. The controller mounts only its state, model root (read-only), router catalog directory, and Docker socket. It does not mount the host home or manage systemd during routine updates.

`.state/` contains the private router credential, host settings, selected release, verified artifact receipts, generated Compose files, operation journals, prior release source, and migration backups. It is excluded from Git and the Docker build context. The pinned inference images are existing local artifacts; ordinary updates never pull/rebuild those engines or download model weights. Retain both for recovery. Driver installation and model provisioning remain host responsibilities.

The router owns inference APIs, request queues, client policy, and conversation archives. Runtime owns launch definitions and catalog publication. After migration, router-only deployment uses `router-maintenance-begin`, `publish`, and `router-maintenance-end` to preserve this boundary. An interrupted router deployment keeps its reservation until the router is repaired and the end command verifies readiness.
