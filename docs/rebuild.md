# Rebuilding after loss of the system disk

Containerization makes the application replaceable. It does not back up model weights, Docker volumes, credentials, custom images, or host configuration. This procedure restores AI Runtime without the retired host Python application. A full-machine recovery also needs the router and each client application's own deployment and data backups.

## Recovery inventory

| Item | Recovery source |
| --- | --- |
| Runtime source, launch definitions, catalog generation, home wrappers | Public repository and `runtime.git.bundle` |
| Selected profile, GPU UUIDs, private router token, deployment/recovery journals | Private `.state/` and `.env` in `private-state.tar.gz` |
| Exact pinned inference engines, current and previous controller images | `images.tar`; verify IDs against `inventory.json` after loading |
| Model artifacts | Separate model backup, or immutable download URLs in `model-downloads.json`; validate sizes and SHA-256 |
| Router API, credentials, runtime catalog directory, network | Router deployment and private configuration backup |
| Router conversation data, Open WebUI data, other application volumes | Separate consistent application backups |
| OS, LAN address, user IDs, NVIDIA driver/toolkit, Docker/Compose | Host provisioning; record installed versions in the private recovery inventory |

The current model definitions require 40 files (152,039,442,720 bytes) for every profile, or five files (54,702,891,968 bytes) for `daytime-27b` plus Nighttime. The exporter does **not** copy these large files. Download URLs were imported from the historical artifact manifests, and future availability is not guaranteed. Keep an off-machine model backup if recovery must work offline or finish quickly.

The custom inference images have exact local image identity pins. Loading saved images preserves the known artifacts; rebuilding llama.cpp may produce a different image ID even from the same source commit. Validate and deliberately update the engine definition before using a rebuilt image. Never change an integrity hash just to bypass a failed restore, and never treat restoration as a new performance qualification.

## Make a private export

Run as the deployment user on Rosalina, with a new destination on storage that has enough free space:

```sh
cd /home/astigmatism/apps/local-ai-runtime
python3 -m runtime.recovery export --destination /path/to/private-backup/runtime-YYYYMMDD
```

The exporter holds the update/runtime locks and requires a healthy deployment with no interrupted maintenance. Generations continue. It saves runtime Git history, private state, exact engine/controller images, model inventory, and file checksums. Directory/file permissions are 0700/0600. A failed or partial export has no completed `checksums.json`; do not use it as the only backup. `--without-images` produces a metadata-only snapshot, not a complete image backup.

Copy the completed directory off Rosalina's disk. Store it as private data: `private-state.tar.gz` contains the router credential and historical migration configuration. Never commit it or upload it to a public repository. These file permissions are not encryption; use encrypted backup storage for the private kit.

## Restore onto a replacement disk

This procedure assumes the same Linux user (UID/GID 1000), absolute directory paths and GPUs. Hardware/path changes require reviewing `host.json` and regenerating a deployment; do not blindly reuse a recorded bundle for another host.

1. Install a compatible Linux OS, Docker Engine/Compose, NVIDIA driver and NVIDIA Container Toolkit. Restore the LAN address. Verify all four GPUs with `nvidia-smi` and GPU access from a disposable container. Enable Docker at boot.
2. Verify each recovery file against `checksums.json`. Load `images.tar` using `docker image load -i /path/to/kit/images.tar`. Compare every saved image's ID and source revision with `inventory.json`. Do this before starting any models.
3. Restore the runtime repository at the recorded `root` path, using the public repository or `git clone /path/to/kit/runtime.git.bundle /home/astigmatism/apps/local-ai-runtime`. Set `origin` to `https://github.com/astigmatism/ai-runtime.git` for rename-aware releases. When restoring a pre-rename revision, retain `https://github.com/astigmatism/local-ai-runtime.git` until the normal updater installs a rename-aware release; the old URL redirects, and the old source rejects the new origin. See [rename compatibility](renaming.md). On this **new checkout**, select the recorded deployed commit on `main` and set its upstream to `origin/main`. Keep the exact deployed source until recovery succeeds; upgrade afterward.
4. Extract `private-state.tar.gz` into that checkout, preserving permissions/ownership. Read the archive listing first. Restore only a trusted backup. This restores `.env`, `.state/active.json`, the selected profile and all retained recovery state. Do not run `migrate.sh prepare`: that command is only for converting the former host application.
5. Restore model files under the exact `model_root` in `.state/host.json`. Use the profile paths and checksums. A copied file's changed inode/timestamps invalidate cached checksums; runtime validation will rehash it. Budget disk bandwidth and time for this step.
6. Restore the router **separately**, using its pinned image/source and private configuration, catalog directory and persistent data. Create/preserve the external Docker network `local-ai-ollama_default`. Restore its matching admin credential. Start only the router service; never run the retired all-GPU launcher or the whole historical combined inference stack. Runtime depends on the router admin API and its writable catalog directory.
7. Update only `.env`'s `DOCKER_GID` to match `stat -c %g /var/run/docker.sock` if needed. Verify `.env`'s controller image is the recorded active image. Keep model, state and router paths unchanged. Confirm that no legacy runtime startup unit/container can compete for GPUs or ports.
8. Render and validate the saved configuration, then recreate the backend pair and controller, as below. Run these commands only on the replacement host, after the prerequisites above are ready.

```sh
cd /home/astigmatism/apps/local-ai-runtime
python3 - <<'PY'
from pathlib import Path
from runtime.config import read, require, render
from runtime.system import atomic_json
from runtime.update import Updater
u = Updater(Path.cwd())
active = read(u.state / 'active.json')
host = read(u.state / 'host.json')
desired = render(u.root / 'config', host, active['bundle']['profile'])
require(desired == active['bundle'], 'Restored source/paths differ from saved release')
require(u.source_preflight() == active['revision'], 'Wrong restored source revision')
# Read-only model validation: full hashing is intentionally required on restore.
u.candidate(active['image'], 'validate', full_hash=True)
path = u.state / 'generated' / desired['config_sha256'] / 'compose.json'
atomic_json(path, desired['compose'])
u.run('docker', 'compose', '-p', desired['compose']['name'], '-f', str(path),
      'up', '-d', '--pull', 'never', '--wait', '--wait-timeout', '450',
      'coding', 'everyday', capture=False)
u.compose(u.root, active['image'], 'up', '-d', '--pull', 'never',
          '--wait', '--wait-timeout', '150', 'controller')
u.candidate(active['image'], 'check')
PY
python3 -m runtime.recovery install-wrappers --home /home/astigmatism
```

9. Check `http://192.168.1.4:11436/healthz`, profile/context/GPU identities, router catalog, both direct backends and a bounded request through the router. Confirm Service Portal discovers AI Runtime and its update action. Test a reboot in a maintenance window before declaring the machine recovery-tested.

The Docker socket grants the controller and update runner host-level Docker control. Containerization makes software ownership and packaging clearer; it does not remove the need to protect that socket or back up private application data.

## Legacy retirement on Rosalina (2026-09-16)

The retired host application's tree and stale home registry/selection/commands are archived under `~/archives/ai-runtime-legacy/20260916`, with a receipt of original paths and checksums. The compatibility directory `~/apps/local-ai-primary` retains only container bridges, a README and the ownership marker used by router tooling. It is not needed to render or run the models.

The two old runtime user-systemd units were already inactive and disabled; their installed unit files were archived. The disconnected, unloaded legacy Ollama container was stopped, and legacy inference containers have restart policy `no`. Their data and images were retained. The old all-GPU `deploy-runtime.sh` still refuses execution.

Open WebUI's host boot helper had been waiting indefinitely for the retired bootstrap unit. It now waits for the existing network and uses the current Compose port binding. Its live container was preserved. The old ComfyUI frontend helper was also stuck and referred to a container that no longer exists; that obsolete unit was disabled and archived. DeepSeek Harness already skips absent/disabled bootstrap units. Other services still have host boot helpers and persistent data; they are outside this runtime export's scope.

Archival directories and stopped legacy containers are historical recovery material, not active dependencies. Retain them until an off-machine backup and restore drill are complete. Do not revive their old launchers against the current deployment.
