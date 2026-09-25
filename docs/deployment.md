# Deployment and recovery

The source repository is now `ai-runtime`. Existing production paths and the `local-ai-runtime` container/Compose project remain intentional compatibility names. Read [rename compatibility](renaming.md) before changing an existing Git remote; the first update must use the remote accepted by the deployed updater.

## Initial staging: no production service changes

The current source offers Flash-Next `daytime` (128K), experimental `daytime-flash-f16` (128K), saved `daytime-27b` (160K), and unchanged Nighttime (128K). Initial migration compares only the two historical Daytime profiles with legacy saved profiles; the new F16 choice has no legacy counterpart. Any inspection or image prepared before this refresh is stale. For an existing clean checkout, fast-forward public `main` and repeat preparation, image build, and inspection before cutover. Do not run the earlier migration against the new host configuration.

Run on Rosalina as the existing deployment user after the repository is published:

```sh
git clone https://github.com/astigmatism/ai-runtime.git /home/astigmatism/apps/local-ai-runtime
cd /home/astigmatism/apps/local-ai-runtime
scripts/migrate.sh prepare
revision=$(git rev-parse HEAD)
docker build --build-arg SOURCE_REVISION="$revision" -t "local/ai-runtime:git-$revision" .
docker run --rm "local/ai-runtime:git-$revision" python3 -m unittest discover -s tests -v
scripts/migrate.sh inspect
```

`prepare` writes private state only in the new checkout. It imports the selected profile, GPU UUIDs, directory locations, and the router token. It checks that the published Compose generation exactly matches the existing source. It does not change the old controller, router, wrappers, or model servers.

`inspect` runs a disposable image against read-only state/model/catalog mounts. The inspection code rejects Docker or HTTP mutations. It first verifies engine identity, live arguments, both slots, and existing checksum receipts. If every artifact has a previously verified SHA-256 with unchanged size, modification time, change time, and inode, it reuses those receipts. Otherwise it fully hashes the selected artifacts before recording `.state/inspection.json`. Live configuration drift fails before the expensive hash pass. It does not generate text, drain requests, publish a catalog, or restart anything.

Full hashing of Flash-Next reads all 33 weight shards plus its projector and draft. Schedule this disk-intensive verification when it will not compete with production work. Lightweight read-only inspection without `--full-hash` checks existing receipts, artifact sizes, per-service engine identities, arguments, mounts, health, and slot capacity; it cannot certify new checksums or substitute for the migration inspection gate. Importing historical hashes never creates a verification or performance-qualification receipt.

The status page and portal entry start during cutover, not during staging. Never publish `.state/`, `.env`, historical rollback directories, or raw production logs.

## Cutover: requires its own production authorization

Before cutover, review the inspection report, publish the Service Portal friendly-name change, and publish the router integration that respects `runtime-owner.json`. A controller image/source revision mismatch or any change to the inspected legacy files/backend identities stops the migration before cutover. Both saved profile directories and the `daytime-27b` wrapper are included in the migration baseline and backup. If generations must remain uninterrupted, defer cutover: it temporarily pauses new router admissions even when backend settings are unchanged.

```sh
cd /home/astigmatism/apps/local-ai-runtime
scripts/migrate.sh cutover
docker exec local-ai-runtime python3 -m runtime check
```

The cutover:

1. Acquires the old profile lock and checks the inspected files and backend IDs.
2. Saves private backups of the old scripts, registry, selection, and startup unit.
3. Adopts the exact healthy pair, briefly drains accepted work for verified catalog publication, and starts the controller with a bounded health wait.
4. Confirms both inference container IDs remain unchanged.
5. Replaces the old all-GPU deployment launcher with a refusal stub, then disables `local-ai-primary.service` only after the new controller is healthy. The old launcher previously depended on that unit's enablement for its safety guard; it must remain blocked after the unit is retired. Initial migration restoration restores its original bytes.
6. Installs home-folder wrappers and guarded legacy entrypoints, then writes `runtime-owner.json`.

The existing Service Portal server already supports the required update labels. Deploy its friendly-name mapping through its normal updater; no inference restart is involved. The controller's published HTTP port gives the portal a link, including when “Hide services without links” is enabled. The GPU containers remain hidden.

Check the new page, `/api/status`, `/healthz`, Service Portal's row/update capability, compatibility commands, backend IDs, router model discovery, and a short request to each resident service. The initial no-change portal update verifies runner execution and health; changed-release and rollback smoke tests must be performed only within the approved maintenance scope.

## Routine releases

The updater accepts clean `main` with the expected `origin/main` upstream and this repository's origin. It fetches public HTTPS and requires fast-forward history. It exports committed source into private release directories, builds/tests the image, validates Compose and model prerequisites, then applies the selected profile under the runtime lock.

New router admissions pause during maintenance. Existing active, queued, and direct backend work must finish within five minutes; otherwise the change aborts before stopping a backend. An idle changed backend is recreated with `--no-deps --pull never --wait`. An unchanged backend is verified by identity and preserved. Generation acceptance uses an explicit eight-token test request without changing the model's output defaults.

After publication, the new controller is recreated with a 150-second health bound. The updater records image/commit/configuration identity and advances the checkout by fast-forward. No command performs Compose `down`, Git reset/stash, global pruning, or volume removal. A controller-only release retains the existing inference containers.

## Browser profile changes

The runtime page can select `daytime`, `daytime-flash-f16`, or `daytime-27b` without another source release once this profile is published. Browser changes acquire the existing update and runtime locks and use the deployed controller's validated transition. They never edit source or advance a checkout. An unexpected Nighttime replacement is refused, including during automatic recovery. Source deployments and existing CLI administration retain their existing behavior.

`daytime-flash-f16` uses the same pinned Flash-Next inference image, model weights, 128K context, and Q8 MTP draft as `daytime`. Only the main model's K/V cache is F16 and its microbatch is 1024. Its VRAM fit and throughput have not been qualified on the production GPUs. The original `daytime` profile remains available for direct comparison and recovery.

A controller-only release of the GPU overview and switching UI does not change rendered model definitions. Verify its revision, health, assignments, profile choices, and disabled controls during other maintenance using read-only requests after deployment. Test an actual production profile switch only in an agreed maintenance window; this pauses new admissions for both models and loads the alternative Daytime backend.

Operation receipts survive page refresh and controller restart. If a switch reports `needs-attention`, inspect the private journals and follow **runtime recover** below. Do not clear journals or edit selected-release receipts to make the controls available. No browser action cancels an in-progress transition or forces a backend restart. After recovery, the controller reconciles the operation result on its next status cycle.

## Recover an interrupted update

```sh
cd /home/astigmatism/apps/local-ai-runtime
scripts/recover-update.sh
```

Recovery requires clean source and an expected revision. It reads the durable update and runtime journals, restores an interrupted runtime transaction when needed, and recreates only the controller using the recorded active image. This clears stale Docker health results from a temporary revision mismatch and applies a fresh bounded readiness wait. If the candidate runtime already succeeded, it completes controller replacement and source advancement without applying the models again. If the previous runtime is restored, it keeps source changes for an explicit retry. It never discards local work.

For a runtime operation interrupted outside the portal updater:

```sh
docker exec local-ai-runtime python3 -m runtime recover
```

If the controller is unavailable, use the already installed candidate image and the same mounts as the updater. From the host checkout:

```sh
python3 - <<'PY'
from pathlib import Path
from runtime.config import read
from runtime.update import Updater
updater = Updater(Path.cwd())
transaction = read(updater.state / 'transaction.json')
updater.candidate(transaction['target']['image'], 'recover')
PY
```

A confirmed failed initialization may remain in Docker's restart loop. Recovery does not terminate an unhealthy process whose activity cannot be established. If this prevents recovery, inspect that backend's logs, confirm it is only a failed initializer and has no active work, then repair that one backend. Re-run recovery afterward. Admission stays paused until verification succeeds. This conservative case is reported as `needs-attention` rather than a successful rollback.

## Restore the initial host controller

Use this only to undo the initial migration before a subsequent runtime release or profile change:

```sh
scripts/migrate.sh restore
```

It requires the initial revision/profile and healthy matching backends. It stops only the new controller, restores the saved host files, removes runtime ownership, and restores the original startup-unit enablement. It does not stop the model servers or remove persistent data. If initial cutover stopped after creating its backup, retain that directory and inspect the failure before running restore; do not delete it to force a fresh cutover.

## Boot verification: separate reboot window

Docker owns controller/model restart supervision after cutover. The controller serves diagnostic status while waiting for Docker, backend health, the existing network, and the router. Its startup operation is idempotent. It refuses stale revisions, interrupted runtime transactions, and unfinished router maintenance.

During an agreed reboot window, verify that the selected profile remains unchanged, both GPUs per backend are assigned correctly, both services pass health checks, router discovery reports the correct aliases/context, the controller becomes healthy, and DSH/Open WebUI can still reach the router. The existing network is preserved; no boot command destroys or recreates it. This test is not performed by staging or by a normal update.

## Optional vision GPU

See [shared vision GPU configuration](shared-vision.md) for a separately selected encoder device, CUDA ordering, memory qualification, and rollback. CPU projector placement remains the default.
