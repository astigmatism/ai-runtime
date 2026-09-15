# AI Runtime

Versioned startup, launch configuration, and catalog publication for Rosalina's two resident llama.cpp services. The controller has a read-only status page and integrates with Service Portal's **Update and restart** action.

## Current profiles

| Profile | Daytime model | Context | GPU pair |
| --- | --- | --- | --- |
| `daytime-swift` | Swift-Qwen3.8-27B Q8 with embedded MTP3 | 128K | RTX 3090 + RTX 4080 SUPER |
| `daytime` | Qwen3.8-27B Q8 with separate Q4 MTP3 draft | 160K | RTX 3090 + RTX 4080 SUPER |

Both include Nighttime: Qwen3.8-27B Abliterated Q6_K, 128K, RTX 4080 + RTX 3080 Ti. Each backend has one slot. `primary` means the selected Daytime configuration plus Nighttime; it is not a clock-based schedule.

The initial migration preserves `qwen38-daytime`, `qwen38-nighttime`, the `local-ai-primary` backend Compose project, `local-ai-ollama_default`, loopback inference ports 18080/18081, existing model files, and the pinned llama.cpp image. The controller uses a separate Compose project, `local-ai-runtime`, and no GPUs.

## Edit, publish, deploy

1. Edit and test on the development Mac.
2. Commit and push `main` to this public repository.
3. Select **Update and restart** beside AI Runtime in Service Portal.

Rosalina reads public GitHub over HTTPS. It requires no GitHub write credentials. The updater fetches only fast-forward changes, builds an image marked with the exact source commit, validates it before draining work, and reconciles only changed backends. The source checkout advances after the candidate controller is healthy. A failed update retains the previous image and configuration, and reports its recovery state. Read [deployment and recovery](docs/deployment.md) before the initial migration.

### Where to change settings

- `config/shared.json`: shared launch arguments, container defaults, output/reasoning policy, network, and pinned engine identity.
- `config/profiles/*.json`: one definition each for original Daytime, Swift Daytime, and Nighttime. Change `context_tokens` to update both launch flags, the manifest, and router discovery together. Per-profile `arguments` override shared arguments. `argument_order` preserves the exact qualified invocation and must list each effective argument once.
- `.state/host.json`: private host deployment settings, GPU UUIDs, model root, and router integration paths. Start with `config/host.example.json` for another machine. The migration imports Rosalina's existing settings automatically.

For example, changing Swift's `context_tokens` to `98304` publishes 96K consistently and recreates only Daytime after drain. Model files are referenced by path relative to the model root and SHA-256; they are never included in Git or the image. A changed file invalidates the checksum cache even when its size stays the same.

The initial implementation preserves the existing single-slot and unrestricted generation contracts. Context is limited to 163840 tokens by the current router contract. Changes outside those contracts require a coordinated router/runtime release. The status page does not select models or edit configuration. Existing SSH profile commands remain available.

Integrity checks prove that the launch configuration, engine, and model artifacts agree. The short post-change generation check establishes basic operation. Neither is a new throughput benchmark, full-context qualification, or VRAM soak test. `docs/import-provenance.json` records the original import without fabricating new qualification receipts.

## Status and commands

After cutover, open `http://192.168.1.21:11436` for deployed revision, selected profile, model/context details, GPU assignments, readiness, request counts, and the latest deployment result. The page polls same-origin status; it never receives router credentials.

| Command | Behavior |
| --- | --- |
| `~/primary status` | Current pair, revision, and health |
| `~/primary` | Ensure the selected pair is running |
| `~/daytime` / `~/daytime-swift` | Select that existing profile; drain first and preserve Nighttime |
| `~/local-ai-config.sh list` | List supported configurations |
| `~/local-ai-config.sh gpus` | Host GPU status |
| `docker exec local-ai-runtime python3 -m runtime check` | Exit nonzero unless the deployed runtime and router discovery are ready |
| `docker exec local-ai-runtime python3 -m runtime publish` | Verify and republish the selected catalog |
| `scripts/recover-update.sh` | Complete or recover an interrupted portal update |

`nighttime`, `daytime-256`, and `nighttime-256` retain their existing compatibility meaning: ensure the selected pair. Historical `deploy` and `rollback` commands are retired. Stopping the controller does not unload the model servers. Backend restart policies remain in place. Controller startup is idempotent and rejects stale source revisions or unfinished maintenance rather than overwriting them.

### HTTP interface

- `GET /api/status`: current revision, deployed revision, selected profile, per-service identities/readiness, maintenance counts, and deployment result.
- `GET /healthz`: `200 {"ready": true}` only after startup reconciliation and verification; otherwise `503`.
- `GET /`, `/app.js`, `/style.css`: read-only status UI.
- Mutating HTTP methods return 405. Runtime administration is available only through the container CLI and the existing Service Portal runner.

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

The runtime image includes Python, Git, Docker CLI, Buildx, and Compose. Service Portal runs the updater as the checkout owner, with the Docker socket group. The controller mounts only its state, model root (read-only), router catalog directory, and Docker socket. It does not mount the host home or manage systemd during routine updates.

`.state/` contains the private router credential, host settings, selected release, verified artifact receipts, generated Compose files, operation journals, prior release source, and migration backups. It is excluded from Git and the Docker build context. The pinned inference image is an existing local artifact; ordinary updates never pull/rebuild that engine or download model weights. Retain it for recovery. Driver installation and model provisioning remain host responsibilities.

The router owns inference APIs, request queues, client policy, and conversation archives. Runtime owns launch definitions and catalog publication. After migration, router-only deployment uses `router-maintenance-begin`, `publish`, and `router-maintenance-end` to preserve this boundary. An interrupted router deployment keeps its reservation until the router is repaired and the end command verifies readiness.
