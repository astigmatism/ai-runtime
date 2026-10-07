# AI Runtime

Versioned startup, launch configuration, and catalog publication for Rosalina's resident llama.cpp services: normally a Daytime and a Nighttime backend, or one exclusive Daytime backend on all four text GPUs. The controller shows GPU assignments, switches the registered Daytime profiles, and integrates with Service Portal's **Update and restart** action.

The repository is **`ai-runtime`**, formerly `local-ai-runtime`. The application name remains **AI Runtime**. See [repository rename and compatibility](docs/renaming.md) for existing installations.

```sh
git clone https://github.com/astigmatism/ai-runtime.git
cd ai-runtime
```

This is the model lifecycle controller. Inference requests go through [LLM Router](https://github.com/astigmatism/llm-router); AI Runtime exposes a status and profile-switching UI/API, with a container CLI for recovery and administration.

## Current configurations

Choosing a configuration sets the Daytime model and whether Nighttime runs beside it. The status page lists both for every configuration.

| Configuration | Daytime model | Nighttime | GPUs |
| --- | --- | --- | --- |
| `qwen27b-q8-with-nighttime` | Qwen3.8 27B Q8, tensor-parallel (55,45), MTP3, 160K | Qwen3.8 27B Abliterated Q6_K, 128K | Daytime: RTX 3090 + RTX 4080 SUPER; Nighttime: RTX 4080 + RTX 3080 Ti; RTX 3080 shared for vision |
| `qwen27b-q6k-with-nighttime` | Qwen3.8 27B Unsloth UD-Q6_K_XL, otherwise as above, 160K | Same Nighttime | Same |
| `flash-next-solo-128k` | Qwen3.8 Flash-Next (AtomicChat AD-4.27, MTP3), 128K: the fastest configuration | Off | All four text GPUs; RTX 3080 for vision and the MTP draft |
| `flash-next-solo-160k` | Same Flash-Next with a 160K window and q8_0 K/V | Off | Same |
| `flash-next-solo-two-requests` | Same Flash-Next serving two requests at once, 128K each (q8_0 K/V, 512 microbatch) | Off | Same |

`qwen27b-q8-with-nighttime` is the default when a host has no recorded release. Nighttime's RAM prompt cache is capped at 24 GiB; Daytime uses the shared 48 GiB. `primary` means the selected configuration; it is not a clock-based schedule.
- **27B configurations:** each was benchmarked on a full Bench Studio coding session (2/2 passed), at about 26.1 (Q8) and 27.0 (Q6_K) steps per second.
- **Flash-Next solo configurations:** the whole model stays in VRAM on all four text GPUs. On a fixed session replay, `flash-next-solo-128k` decodes at about 94 tokens per second. With two concurrent requests, each runs at about half speed. Decode slows as a conversation's context fills, in every configuration: about 56 tokens per second at 64K and 36 at 120K.
- **Selecting a solo configuration:** it drains both models, stops and removes Nighttime, and publishes a one-model catalog. Requests for Nighttime fail until a configuration with Nighttime is selected again. See [deployment](docs/deployment.md#exclusive-flash-next-solo-profiles) for placement, measurements, and recovery.
- **Consolidation (2026-10-07):** the configurations were renamed and the superseded ones retired. Each kept configuration launches the same backend and model alias as before, so its Bench Studio history continues:

  | Old name | New name |
  | --- | --- |
  | `daytime-27b-tensor-next` | `qwen27b-q8-with-nighttime` |
  | `daytime-27b-q6k-tensor-next` | `qwen27b-q6k-with-nighttime` |
  | `daytime-flash-solo-tuned-mtp3` | `flash-next-solo-128k` |
  | `daytime-flash-solo-tuned-mtp3-160k` | `flash-next-solo-160k` |
  | `daytime-flash-solo-tuned-mtp3-2slot` | `flash-next-solo-two-requests` |
- **Retired:**
  - `daytime`, `daytime-flash-f16`, and `daytime-flash-next`: two-GPU Flash-Next at about 31 tokens per second.
  - `daytime-27b`: layer-split Q8.
  - `daytime-flash-solo` and `daytime-flash-solo-tuned`: superseded by MTP3.
  - The `flash-next-mtp` engine they used is no longer referenced.

The initial migration preserves `qwen38-daytime`, `qwen38-nighttime`, the `local-ai-primary` backend Compose project, `local-ai-ollama_default`, loopback inference ports 18080/18081, existing model files, and each service's pinned llama.cpp image. Every Qwen3.8-27B backend (Nighttime and both 27B configurations) uses `836d57176dc699a726c55418e4f96b8ca628e1bf` (engine `qwen38-dual-836d571`). The Flash-Next solo configurations use `43fe9c64281ef735046adc025e9e7559a1f659a5` (engine `qwen38-dual-43fe9c6`). The original `8ea2902` engine and the Flash-Next branch engine `d1a92352c` are retired. The controller uses a separate Compose project, `local-ai-runtime`, and no GPUs.

The September 16 source refresh imports the running Flash-Next configuration exactly, including its 33 model shards, CPU/GPU tensor overrides, lazy lookup-table reads, 2048 microbatch, CPU vision projector, and MTP draft on CUDA0. It preserves the saved 27B profile and keeps Swift retired. This source refresh does not itself deploy or migrate the running host controller.

## Edit, publish, deploy

1. Edit and test on the development Mac.
2. Commit and push `main` to this public repository.
3. Select **Update and restart** beside AI Runtime in Service Portal.

The development checkout is `/Users/astigmatism/Projects/ai-runtime`; Rosalina's `/home/astigmatism/deployments/ai-runtime` is a deployment checkout. Do not edit or commit application source on Rosalina or modify running containers. Leave checkout advancement to the updater so the checkout, image, and active release stay aligned. See [agent instructions](AGENTS.md).

Rosalina reads public GitHub over HTTPS. It requires no GitHub write credentials. The updater fetches only fast-forward changes, builds an image marked with the exact source commit, validates it before draining work, and reconciles only changed backends. The source checkout advances after the candidate controller is healthy. A failed update retains the previous image and configuration, and reports its recovery state. Read [deployment and recovery](docs/deployment.md) before the initial migration.

### Where to change settings

- `config/shared.json`: shared launch arguments, container defaults, output/reasoning policy, network, and named engine identities.
- `config/profiles/*.json`: one definition for each selectable configuration and Nighttime. A profile with `"exclusive": true` and `"gpu_group": "all"` runs alone on both text pairs (Daytime then Nighttime UUIDs as CUDA0–CUDA3, vision GPU as CUDA4) and requires a configured `vision_gpu_id`. Each selects an `engine` from the shared registry. Change `context_tokens` to update both launch flags, the manifest, and router discovery together. Per-profile `arguments` override shared arguments. `argument_order` preserves the exact invocation and must list each effective argument once; an ordered list for `--override-tensor` expands to repeated flags without losing placement rules. Model, projector, and draft metadata are derived from their actual argument paths and declared artifact mounts. A profile is selectable only when its id is listed in `DAYTIME_PROFILES` (`runtime/config.py`); the status page, `~/local-ai-config.sh list`, `~/local-ai-config.sh names`, and `--profile` all read that one registry.
- `.state/host.json`: private host deployment settings, GPU UUIDs, model root, and router integration paths. Start with `config/host.example.json` for another machine. The migration imports Rosalina's existing settings automatically.

All profiles support a shared vision GPU through Runtime's [versioned renderer and host configuration](docs/shared-vision.md). The source defines projector offload, CUDA ordering, placement validation, and catalog metadata; the host supplies hardware UUIDs. Rosalina uses the RTX 3080 as CUDA2 for vision in both backends. CPU vision remains the default when the optional setting is absent. Start with [the shared-vision host example](config/host.shared-vision.example.json) when configuring another such host; its UUIDs are synthetic placeholders, not live hardware.

For example, changing Daytime's `context_tokens` to `98304` publishes 96K consistently and recreates only Daytime after drain. Model files are referenced by path relative to the model root and SHA-256; they are never included in Git or the image. A changed file invalidates the checksum cache even when its size stays the same.

The initial implementation preserves the existing single-slot and unrestricted generation contracts. Context is limited to 262144 tokens, the Qwen3.8 native window, by the current router contract (`ROUTER_CONTEXT_LIMIT`). Routers older than LLM Router `7235329` accept at most 163840 tokens and reject a larger catalog, after which the switch restores the previous profile. Changes outside those contracts require a coordinated router/runtime release. The page shows the active GPU assignments and switches between registered Daytime profiles; it cannot edit configuration. Existing SSH profile commands remain available.

Integrity checks prove that the launch configuration, engine, and model artifacts agree. The short post-change generation check establishes basic operation. Neither is a new throughput benchmark, full-context qualification, or VRAM soak test. `docs/import-provenance.json` records the original import without fabricating new qualification receipts.

## Status and commands

Open `http://192.168.1.4:11436` for the GPU overview, active model/context details, readiness, request counts, and source-deployment results. Select a Daytime profile and press **Switch to…** to apply it. Existing requests finish first; new requests pause for both models while Nighttime stays loaded, except that selecting or leaving an exclusive profile (which asks for confirmation) stops or restarts Nighttime. Progress survives page refreshes and disconnections. The GPU rail shows assignments and model-service health, not hardware telemetry. The page polls same-origin status and never receives router credentials or model paths.

| Command | Behavior |
| --- | --- |
| `~/primary status` | Current pair, revision, and health |
| `~/primary` | Ensure the selected pair is running |
| `~/daytime` | Select Flash-Next at 128K; drain first and preserve Nighttime |
| `~/local-ai-config.sh apply daytime-flash-f16` | Select the experimental F16 KV Flash-Next profile; drain first and preserve Nighttime |
| `~/daytime-27b` | Select the saved 27B Q8 profile at 160K; preserve Nighttime |
| `~/local-ai-config.sh list` | List the supported configurations, generated from the profile registry |
| `~/local-ai-config.sh gpus` | Host GPU status |
| `docker exec local-ai-runtime python3 -m runtime check` | Exit nonzero unless the deployed runtime and router discovery are ready |
| `docker exec local-ai-runtime python3 -m runtime publish` | Verify and republish the selected catalog |
| `scripts/recover-update.sh` | Complete or recover an interrupted portal update |

`nighttime`, `daytime-256`, and `nighttime-256` retain their existing compatibility meaning: ensure the selected pair. Historical `deploy` and `rollback` commands are retired. Stopping the controller does not unload the model servers. Backend restart policies remain in place. Controller startup is idempotent and rejects stale source revisions or unfinished maintenance rather than overwriting them.

### HTTP interface

- `GET /api/status`: revision, active profile, configuration registry, GPU assignments, model readiness, request counts, source deployment, latest browser operation, switching availability, and a process-scoped CSRF token.
- `POST /api/profile-switch`: JSON `{profile, request_id, expected_profile, expected_revision}`. `request_id` is a lowercase UUID; the expected values come from status. Send the same-origin `Origin` and `X-Runtime-CSRF` headers. Returns `202 {operation_id, operation}` immediately; repeated identical IDs return `200` and the existing receipt. Reusing an ID for different parameters, stale selection, or maintenance conflicts returns `409`. Invalid bodies return `400`, cross-origin/token failures `403`, and non-JSON requests `415`.
- `GET /api/operations/<id>`: a sanitized durable receipt, or `404`. Status is `running`, `succeeded`, `failed`, `recovered`, or `needs-attention`; live phases are `checking`, `draining`, `loading`, `verifying`, and `restoring`.
- `GET /healthz`: `200 {"ready": true}` only after startup reconciliation and verification and outside an active browser switch; otherwise `503`.
- `GET /`, `/app.js`, `/style.css`: runtime UI. Other mutation endpoints/methods return `405`.

This is a trusted-LAN administration surface without a login: anyone who can reach it can switch profiles. The current bind address is preserved. JSON, same-origin checks, a CSRF token, and a Host allowlist protect browser requests; they are not user authentication. `RUNTIME_ALLOWED_HOSTS` in the existing Git-excluded `.env` accepts comma-separated hostnames/IPs without ports. It defaults to `RUNTIME_BIND_IP` and loopback addresses. Add any hostname used to open the page to this setting; unrecognized hosts return `421`. The direct HTTP listener does not trust forwarded origin headers.

The runtime and updater locks cover admission through completion, including validation and recovery. No switch is queued. Operation receipts live in `.state/operations/` and contain private diagnostic errors; HTTP responses expose only safe status messages. A restarted controller reconciles receipts against the existing runtime transaction and never automatically retries an interrupted switch. Full failure diagnostics remain in the private transaction/operation journals and the CLI. Source-deployment results come from the Portal updater journal and are displayed separately from profile-switch results.

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
