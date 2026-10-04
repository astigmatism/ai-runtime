# Deployment and recovery

The source repository is now `ai-runtime`. Existing production paths and the `local-ai-runtime` container/Compose project remain intentional compatibility names. Read [rename compatibility](renaming.md) before changing an existing Git remote; the first update must use the remote accepted by the deployed updater.

## Initial staging: no production service changes

The current source offers Flash-Next `daytime` (128K), experimental `daytime-flash-f16` (128K), saved `daytime-27b` (160K), `daytime-27b-tensor-next` (160K), `daytime-27b-q6k-tensor-next` (160K), experimental `daytime-flash-next` (128K), and Nighttime (128K). Initial migration compares only the two historical Daytime profiles with legacy saved profiles; the newer choices have no legacy counterpart. Any inspection or image prepared before this refresh is stale. For an existing clean checkout, fast-forward public `main` and repeat preparation, image build, and inspection before cutover. Do not run the earlier migration against the new host configuration.

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

If a new revision retires the profile currently active on a host, the updater refuses before building the candidate and names the retired profile and the profiles selectable in that revision. Switch the active profile to one of them on the deployed revision (runtime page validated transition), then re-run the update. The deployed runtime keeps serving meanwhile; no manual state editing is required.

## Browser profile changes

The runtime page can select `daytime`, `daytime-flash-f16`, `daytime-27b`, `daytime-27b-tensor-next`, `daytime-27b-q6k-tensor-next`, or `daytime-flash-next` without another source release once this profile is published. Browser changes acquire the existing update and runtime locks and use the deployed controller's validated transition. They never edit source or advance a checkout. An unexpected Nighttime replacement is refused, including during automatic recovery. Source deployments and existing CLI administration retain their existing behavior.

`daytime-flash-f16` uses the same pinned Flash-Next inference image, model weights, 128K context, and shared-Q4_K_M MTP draft (q8_0 draft K/V) as `daytime`. Only the main model's K/V cache is F16 and its microbatch is 1024. Its VRAM fit and throughput have not been qualified on the production GPUs. The original `daytime` profile remains available for direct comparison and recovery.

`daytime-27b-tensor-next` is the saved Q8 `daytime-27b` with only `--split-mode tensor` overriding the shared `layer` split and `--tensor-split` moved from 60,40 to 55,45. In layer mode the RTX 3090 and RTX 4080 SUPER each run their share of the layer stack in turn for every token, so per-step time is the sum of both cards' weight reads. Tensor mode divides each weight tensor between the cards by the `--tensor-split` proportions so both read concurrently, at the cost of per-layer synchronization over PCIe (Gen3 x16, no NVLink); the pinned engine ships NCCL. The pinned engine (`8ea2902`) enables tensor mode for the `qwen35` architecture, including its recurrent-layer state, and requires flash attention, which is already on. It always places the MTP draft with layer split, and the unchanged `--spec-draft-device CUDA1` keeps it on the RTX 3090; `--fit off` is required because automatic fitting is not implemented for tensor mode. Because both cards read at once, each step lasts as long as the slower card's share, so the split should balance memory bandwidth (RTX 3090 936 GB/s, RTX 4080 SUPER 736 GB/s), not VRAM capacity. Monitoring at 60,40 showed the RTX 3090 at 91% SM and 72% memory-bandwidth utilization and at its 420 W power cap while the RTX 4080 SUPER sat at 75% and 50%; 55,45 moves about 1.45 GiB of Q8 weights to the RTX 4080 SUPER, which keeps roughly 0.6 GiB free at full context. Weights, Q4_0 MTP3 draft, q8_0 K/V cache, 160K context, batch sizes, projector, and RAM prompt cache are unchanged. A retired variant (2026-10-04) moved the MTP draft to the RTX 4080 SUPER at 60,40 and 128K: it loaded with about 0.85 GiB free on that card and moved the per-step bottleneck from the RTX 3090 (77% SM, ~367 W, off its power cap) to the RTX 4080 SUPER (90% SM), but steps per second were unchanged (24.73 vs 24.80–24.92), so it saved power without adding speed and gave up 32K of context. Its alias `qwen3.8-27b-q8_0-tensor-next` keeps benchmark history apart from the layer-split profile; the `daytime` and `local-active` service IDs are unchanged. On engine `836d571` it measured 26.1 steps per second on a full Bench Studio coding session (2/2 passed), against 24.8–24.9 for the same placement on the retired `8ea2902` engine; VRAM use was unchanged (21.6 GB RTX 3090, 15.6 GB RTX 4080 SUPER).

`daytime-27b-q6k-tensor-next` is `daytime-27b-tensor-next` with Unsloth UD-Q6_K_XL main weights (the former `daytime-27b-q6k` artifact); everything else is identical. It measured 27.0 steps per second (2/2 passed), about 3.5% faster than Q8 because UD-Q6_K_XL is only about 13% smaller and costs more dequantization work on the power-capped RTX 3090, with 1.5–1.7 GB more free VRAM per card. Its alias is `qwen3.8-27b-ud-q6_k_xl-tensor-next`. The layer-split `daytime-27b-q6k` and the `8ea2902` tensor profiles `daytime-27b-tensor` and `daytime-27b-q6k-tensor` were retired on 2026-10-04.

`daytime-flash-next` is `daytime` (Flash-Next AtomicChat AD-4.27bpw, 128K) on engine `qwen38-dual-836d571` instead of `flash-next-mtp`. The `flash-next-mtp` engine (`d1a92352c`) is an unmerged MTP branch on 2026-09-03 master. Upstream has since landed qwen4exp work that it lacks: the GDN normalization fix (#28068), fused hyper-connection ops (#28901), CUDA sparse attention above 32K context (#28770), PLE row prefetch for lazy mode (#29599), the attention-path fix (#29751), mask optimizations (#29824), halved indexer score memory (#29825), and its own MTP (#29761). Upstream MTP replaced the branch's (#28243, closed) and cannot borrow the target's embedding and output tensors, so the `shared-Q4_K_M` head cannot load on this engine (`token_embd.weight not found`). The profile therefore uses ggml-org's self-contained `mtp-Qwen3.8-Flash-Next-Q4_0.gguf` (revision `052beeac`, 2,199,652,640 bytes), converted by #29761. Its tensors match the unsloth heads, and its metadata marks the MTP block as a QSA layer. Only the engine, the alias `qwen3.8-flash-next-ad4.27-next` (separate benchmark history), the MTP head, the capability name, and the warnings change. Layer split, the 16,33 split on `CUDA1,CUDA0`, all 33 tensor overrides, 49 GPU layers, lazy mode, 2048 batch/microbatch, 18 threads, q8_0 K/V, MTP2 on CUDA0, the projector, and the shards are identical. The new head adds about 0.3 GB on the RTX 4080 SUPER, which had about 1.8 GiB free on `daytime`. Load, full-context VRAM headroom, and throughput are unqualified. A missing head or checksum mismatch refuses the switch before drain, and a failed load restores the previous profile.

Tensor mode is deliberately not used for Flash-Next. `flash-next-mtp` rejects it for `qwen4exp` outright; 836d571 re-enables it (#28569), but the placement works against it. Experts for 31 of 48 blocks (about 32 GiB) are in system RAM, and a FlashNext Bench Studio session decoded at about 31.5 tokens per second with a median step near 71 ms, against 25–35 ms for the 27B profiles. Tensor mode only divides the GPU-resident share of each step. The meta device it uses implements no `offload_op`, so the scheduler stops moving CPU-resident expert matmuls to the GPU for prompt batches: those 31 blocks would be computed on the CPU during prompt processing, which the 2048 microbatch is sized to avoid. Per-GPU overrides such as `blk.15.ffn_down=CUDA0` are unavailable because there is only the meta device and the CPU. Mirrored tensors (hyper-connection mixers, router gates, the indexer cache) are duplicated on both cards, which would push more expert blocks to the CPU. Upstream issue #27964 also reports `-sm tensor` asserting with this exact AD-4.27bpw quant on an earlier build. Tensor mode would be worth revisiting if the meta device gains op offload, or if enough VRAM keeps the experts off the CPU.

Engine `qwen38-dual-836d571` runs every Qwen3.8-27B backend: Nighttime, the saved `daytime-27b`, and both tensor profiles, plus the `daytime-flash-next` Flash-Next experiment. It is llama.cpp `836d57176dc699a726c55418e4f96b8ca628e1bf` (2026-10-03), the newest commit then with every CUDA CI job green, and is 460+ commits newer than the retired `8ea2902` engine (`qwen38-dual`), including the Qwen3.5 tensor-parallel split-state fix for fused QKV (#28965), CUDA graphs for the MTP draft (#28549), the AllReduce inactive-shard change (#29793), and speculative-decoding fixes (#29019, #29638). The image is built on the host from upstream `.devops/cuda.Dockerfile` at that commit, unmodified: `docker buildx build --target full --build-arg CUDA_VERSION=12.8.1 --build-arg CUDA_DOCKER_ARCH="86;89" --build-arg APP_REVISION=<commit> --label org.opencontainers.image.revision=<commit> -t local/llama.cpp:qwen38-dual-836d57176dc6`, in a dedicated buildx builder that is removed afterwards with its base images. The build receipt is kept under `~/ops/reports/20261004-llama-engine-836d571/`. Retention: the engine image is kept while any profile in `config/shared.json` references it, and when the last referencing profile is retired it is removed after checking compose image references, `.state/active.json`, `.state/previous.json`, and running containers.

Nighttime runs tensor-parallel across its RTX 4080 (CUDA0) and RTX 3080 Ti (CUDA1): `--split-mode tensor` with the existing `--tensor-split 60,40`, on engine `qwen38-dual-836d571`, which supports `qwen35` tensor mode and requires the flash attention and `--fit off` Nighttime already sets. It has no MTP draft. Before the change it ran layer split at about 29.5 tokens per second, with the RTX 4080 at 15.7 of 16.4 GB and the RTX 3080 Ti at 10.8 of 12.3 GB in use. The two cards have similar memory bandwidth (716 and 912 GB/s), so 60,40 is a VRAM-capacity split that may leave the RTX 3080 Ti underused; its throughput and full-context VRAM headroom in tensor mode have not been qualified. Changing Nighttime's placement recreates only Nighttime after drain; if it fails to load or verify, the update restores the previous release.

Every Daytime profile inherits the RAM prompt-cache cap from `config/shared.json` (`--cache-ram` 49152 MiB = 48 GiB). Nighttime alone overrides it to 24576 MiB (24 GiB): on the 123 GiB host its cache had grown to about 50 GB, about 8 GB of it swapped out, with host swap exhausted, so the worst-case combined pair is now bounded at 72 GiB instead of 96 GiB. The cap is a lazy LRU limit allocated on demand, never reserved; each model's catalog entry publishes its own `global_ram_prompt_cache_mib`. Changing the Nighttime value recreates only Nighttime after drain and discards its in-memory prompt cache.

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
