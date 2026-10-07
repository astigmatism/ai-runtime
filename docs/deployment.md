# Deployment and recovery

The source repository is now `ai-runtime`. Existing production paths and the `local-ai-runtime` container/Compose project remain intentional compatibility names. Read [rename compatibility](renaming.md) before changing an existing Git remote; the first update must use the remote accepted by the deployed updater.

## Initial staging: no production service changes

The current source offers `qwen27b-q8-with-nighttime` and `qwen27b-q6k-with-nighttime` (160K, each with Nighttime at 128K), and the exclusive `flash-next-solo-128k` and `flash-next-solo-160k` (no Nighttime). The configurations were renamed and consolidated on 2026-10-07. The [README](../README.md#current-configurations) maps the old names, and history below keeps the names in effect when it was written. The initial migration from the legacy primary deployment is complete. Its legacy `daytime` and `daytime-27b` profiles are retired, so `scripts/migrate.sh prepare` now refuses before writing private state.

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

The runtime page can select any registered configuration without another source release once it is published. Each choice shows its Daytime model and, when it includes one, its Nighttime model, along with the GPUs each uses. Names come from the profile files through `/api/status`, so the page needs no code change when a configuration is added or renamed. Browser changes acquire the existing update and runtime locks and use the deployed controller's validated transition. They never edit source or advance a checkout. An unexpected Nighttime replacement is refused, including during automatic recovery. The one exception is a switch into or out of an exclusive profile, which deliberately removes or recreates Nighttime (see below). Source deployments and existing CLI administration retain their existing behavior.

Renaming a configuration takes two releases, because the updater refuses a release that retires the active configuration. First publish the new name beside the old one; both launch the identical backend. Switch to the new name, which recreates no backend because the Compose definitions are identical. Then publish the release that drops the old name. `tests/fixtures/renamed-profiles.json` records each renamed configuration's Compose digest, so a rename cannot change a backend.

The sections below are the history of the retired and renamed profiles: `daytime`, `daytime-flash-f16`, `daytime-27b`, `daytime-flash-next`, `daytime-flash-solo`, and `daytime-flash-solo-tuned`. `daytime-27b-tensor-next` is now `qwen27b-q8-with-nighttime`, and `daytime-flash-solo-tuned-mtp3` is now `flash-next-solo-128k`.

`daytime-27b-tensor-next` is the saved Q8 `daytime-27b` with only `--split-mode tensor` overriding the shared `layer` split and `--tensor-split` moved from 60,40 to 55,45. In layer mode the RTX 3090 and RTX 4080 SUPER each run their share of the layer stack in turn for every token, so per-step time is the sum of both cards' weight reads. Tensor mode divides each weight tensor between the cards by the `--tensor-split` proportions so both read concurrently, at the cost of per-layer synchronization over PCIe (Gen3 x16, no NVLink); the pinned engine ships NCCL. The pinned engine (`8ea2902`) enables tensor mode for the `qwen35` architecture, including its recurrent-layer state, and requires flash attention, which is already on. It always places the MTP draft with layer split, and the unchanged `--spec-draft-device CUDA1` keeps it on the RTX 3090; `--fit off` is required because automatic fitting is not implemented for tensor mode. Because both cards read at once, each step lasts as long as the slower card's share, so the split should balance memory bandwidth (RTX 3090 936 GB/s, RTX 4080 SUPER 736 GB/s), not VRAM capacity. Monitoring at 60,40 showed the RTX 3090 at 91% SM and 72% memory-bandwidth utilization and at its 420 W power cap while the RTX 4080 SUPER sat at 75% and 50%; 55,45 moves about 1.45 GiB of Q8 weights to the RTX 4080 SUPER, which keeps roughly 0.6 GiB free at full context. Weights, Q4_0 MTP3 draft, q8_0 K/V cache, 160K context, batch sizes, projector, and RAM prompt cache are unchanged. A retired variant (2026-10-04) moved the MTP draft to the RTX 4080 SUPER at 60,40 and 128K: it loaded with about 0.85 GiB free on that card and moved the per-step bottleneck from the RTX 3090 (77% SM, ~367 W, off its power cap) to the RTX 4080 SUPER (90% SM), but steps per second were unchanged (24.73 vs 24.80–24.92), so it saved power without adding speed and gave up 32K of context. Its alias `qwen3.8-27b-q8_0-tensor-next` keeps benchmark history apart from the layer-split profile; the `daytime` and `local-active` service IDs are unchanged. On engine `836d571` it measured 26.1 steps per second on a full Bench Studio coding session (2/2 passed), against 24.8–24.9 for the same placement on the retired `8ea2902` engine; VRAM use was unchanged (21.6 GB RTX 3090, 15.6 GB RTX 4080 SUPER).

`daytime-27b-q6k-tensor-next` is `daytime-27b-tensor-next` with Unsloth UD-Q6_K_XL main weights (the former `daytime-27b-q6k` artifact); everything else is identical. It measured 27.0 steps per second (2/2 passed), about 3.5% faster than Q8 because UD-Q6_K_XL is only about 13% smaller and costs more dequantization work on the power-capped RTX 3090, with 1.5–1.7 GB more free VRAM per card. Its alias is `qwen3.8-27b-ud-q6_k_xl-tensor-next`. The layer-split `daytime-27b-q6k` and the `8ea2902` tensor profiles `daytime-27b-tensor` and `daytime-27b-q6k-tensor` were retired on 2026-10-04.

`daytime-flash-next` is `daytime` (Flash-Next AtomicChat AD-4.27bpw, 128K) on engine `qwen38-dual-836d571` instead of `flash-next-mtp`. The `flash-next-mtp` engine (`d1a92352c`) is an unmerged MTP branch on 2026-09-03 master. Upstream has since landed qwen4exp work that it lacks: the GDN normalization fix (#28068), fused hyper-connection ops (#28901), CUDA sparse attention above 32K context (#28770), PLE row prefetch for lazy mode (#29599), the attention-path fix (#29751), mask optimizations (#29824), halved indexer score memory (#29825), and its own MTP (#29761). Upstream MTP replaced the branch's (#28243, closed) and cannot borrow the target's embedding and output tensors, so the `shared-Q4_K_M` head cannot load on this engine (`token_embd.weight not found`). The profile therefore uses ggml-org's self-contained `mtp-Qwen3.8-Flash-Next-Q4_0.gguf` (revision `052beeac`, 2,199,652,640 bytes), converted by #29761. Its tensors match the unsloth heads, and its metadata marks the MTP block as a QSA layer. Only the engine, the alias `qwen3.8-flash-next-ad4.27-next` (separate benchmark history), the MTP head, the capability name, and the warnings change. Layer split, the 16,33 split on `CUDA1,CUDA0`, all 33 tensor overrides, 49 GPU layers, lazy mode, 2048 batch/microbatch, 18 threads, q8_0 K/V, MTP2 on CUDA0, the projector, and the shards are identical. The new head adds about 0.3 GB on the RTX 4080 SUPER, which had about 1.8 GiB free on `daytime`. Load, full-context VRAM headroom, and throughput are unqualified. A missing head or checksum mismatch refuses the switch before drain, and a failed load restores the previous profile.

Tensor mode is deliberately not used for Flash-Next. `flash-next-mtp` rejects it for `qwen4exp` outright; 836d571 re-enables it (#28569), but the placement works against it. Experts for 31 of 48 blocks (about 32 GiB) are in system RAM, and a FlashNext Bench Studio session decoded at about 31.5 tokens per second with a median step near 71 ms, against 25–35 ms for the 27B profiles. Tensor mode only divides the GPU-resident share of each step. The meta device it uses implements no `offload_op`, so the scheduler stops moving CPU-resident expert matmuls to the GPU for prompt batches: those 31 blocks would be computed on the CPU during prompt processing, which the 2048 microbatch is sized to avoid. Per-GPU overrides such as `blk.15.ffn_down=CUDA0` are unavailable because there is only the meta device and the CPU. Mirrored tensors (hyper-connection mixers, router gates, the indexer cache) are duplicated on both cards, which would push more expert blocks to the CPU. Upstream issue #27964 also reports `-sm tensor` asserting with this exact AD-4.27bpw quant on an earlier build. Tensor mode would be worth revisiting if the meta device gains op offload, or if enough VRAM keeps the experts off the CPU.

Engine `qwen38-dual-836d571` runs every Qwen3.8-27B backend: Nighttime, the saved `daytime-27b`, and both tensor profiles, plus the `daytime-flash-next` Flash-Next experiment. It is llama.cpp `836d57176dc699a726c55418e4f96b8ca628e1bf` (2026-10-03), the newest commit then with every CUDA CI job green, and is 460+ commits newer than the retired `8ea2902` engine (`qwen38-dual`), including the Qwen3.5 tensor-parallel split-state fix for fused QKV (#28965), CUDA graphs for the MTP draft (#28549), the AllReduce inactive-shard change (#29793), and speculative-decoding fixes (#29019, #29638). The image is built on the host from upstream `.devops/cuda.Dockerfile` at that commit, unmodified: `docker buildx build --target full --build-arg CUDA_VERSION=12.8.1 --build-arg CUDA_DOCKER_ARCH="86;89" --build-arg APP_REVISION=<commit> --label org.opencontainers.image.revision=<commit> -t local/llama.cpp:qwen38-dual-836d57176dc6`, in a dedicated buildx builder that is removed afterwards with its base images. The build receipt is kept under `~/ops/reports/20261004-llama-engine-836d571/`. Retention: the engine image is kept while any profile in `config/shared.json` references it, and when the last referencing profile is retired it is removed after checking compose image references, `.state/active.json`, `.state/previous.json`, and running containers.

Nighttime runs tensor-parallel across its RTX 4080 (CUDA0) and RTX 3080 Ti (CUDA1): `--split-mode tensor` with the existing `--tensor-split 60,40`, on engine `qwen38-dual-836d571`, which supports `qwen35` tensor mode and requires the flash attention and `--fit off` Nighttime already sets. It has no MTP draft. Before the change it ran layer split at about 29.5 tokens per second, with the RTX 4080 at 15.7 of 16.4 GB and the RTX 3080 Ti at 10.8 of 12.3 GB in use. The two cards have similar memory bandwidth (716 and 912 GB/s), so 60,40 is a VRAM-capacity split that may leave the RTX 3080 Ti underused; its throughput and full-context VRAM headroom in tensor mode have not been qualified. Changing Nighttime's placement recreates only Nighttime after drain; if it fails to load or verify, the update restores the previous release.

Every Daytime profile inherits the RAM prompt-cache cap from `config/shared.json` (`--cache-ram` 49152 MiB = 48 GiB). Nighttime alone overrides it to 24576 MiB (24 GiB): on the 123 GiB host its cache had grown to about 50 GB, about 8 GB of it swapped out, with host swap exhausted, so the worst-case combined pair is now bounded at 72 GiB instead of 96 GiB. The cap is a lazy LRU limit allocated on demand, never reserved; each model's catalog entry publishes its own `global_ram_prompt_cache_mib`. Changing the Nighttime value recreates only Nighttime after drain and discards its in-memory prompt cache.

A controller-only release of the GPU overview and switching UI does not change rendered model definitions. Verify its revision, health, assignments, profile choices, and disabled controls during other maintenance using read-only requests after deployment. Test an actual production profile switch only in an agreed maintenance window; this pauses new admissions for both models and loads the alternative Daytime backend.

Operation receipts survive page refresh and controller restart. If a switch reports `needs-attention`, inspect the private journals and follow **runtime recover** below. Do not clear journals or edit selected-release receipts to make the controls available. No browser action cancels an in-progress transition or forces a backend restart. After recovery, the controller reconciles the operation result on its next status cycle.

## Exclusive Flash-Next solo profiles

`daytime-flash-solo` is `daytime-flash-next` (the same weights, projector, ggml-org Q4_0 MTP head, 128K context, q8_0 K/V, lazy n-gram table, and engine `qwen38-dual-836d571`) moved onto all four text GPUs with automatic fitting. `daytime-flash-solo-tuned-mtp3` is the recommended exclusive profile: the measured tuning winners on engine `qwen38-dual-43fe9c6` with a three-token draft. `daytime-flash-solo-tuned` is the same with a two-token draft. All three set `"exclusive": true` and `"gpu_group": "all"`, so Nighttime does not run while one is selected, and each keeps its own alias (`qwen3.8-flash-next-ad4.27-solo`, `-solo-tuned`, `-solo-tuned-mtp3`) for benchmark history.

Why: on two GPUs, the experts of 31 of 48 blocks (about 32 GiB) live in pageable host memory.
- Every decode step reads the routed experts of those blocks from RAM, and MTP verification multiplies those reads. The Bench Studio session decoded at about 31.5 tokens per second, against about 90 for the 27B tensor profiles.
- Every prompt batch of at least 32 tokens (`GGML_OP_OFFLOAD_MIN_BATCH`) copies the used experts over PCIe Gen3, nearly all 512 per block. That is why the median first token was about 3.8 s even for small agent turns, against 0.9 s for the 27B.
- About 54.5 GB of the model must be resident; the attention cache is about 25 KB per token. That fits in the 68 GiB of the four text GPUs.

Placement:
- The renderer orders CUDA devices as `gpu_ids.daytime`, then `gpu_ids.nighttime`, then the vision GPU. On Rosalina that is CUDA0 RTX 3090, CUDA1 RTX 4080 SUPER, CUDA2 RTX 4080, CUDA3 RTX 3080 Ti, CUDA4 RTX 3080. It always exports that `CUDA_VISIBLE_DEVICES` order and reserves all five UUIDs.
- The language model uses `--device CUDA1,CUDA2,CUDA3,CUDA0`, so the RTX 3090 is last and holds the output layer.
- `daytime-flash-solo` uses `--fit on --fit-target 1024`, which distributes layers itself and counts the MTP draft. To reach full residency it splits two layers across devices with tensor overrides. The tuned profiles pin whole layers instead: `--fit off --n-gpu-layers all --tensor-split 11,12,8,18` (48 blocks plus the output layer). They use one host thread (`--threads 1 --threads-batch 1`; see host CPU use below). They load in about 35 s with no fitting search, and a load that does not fit fails loudly instead of spilling to the CPU. With F16 K/V, Rosalina keeps 1.3–2.0 GiB free on each text GPU.
- The microbatch is 1024. Large microbatches only paid for themselves while experts were streamed from the CPU. At 2048, fitting no longer kept every expert on the GPUs.
- The projector and the MTP draft use CUDA4; the vision GPU is not shared in this mode.
- Validation requires a configured `vision_gpu_id`, `--device` naming CUDA0–CUDA3 once each, and tensor overrides limited to CPU or CUDA0–CUDA3. A paired profile can still never place text tensors or drafts on the vision GPU.

Transitions:
- **Into an exclusive profile.** The controller drains both models and waits until both are idle. It stops and removes `qwen38-nighttime` (removal keeps its restart policy from reviving it on reassigned GPUs), then recreates Daytime and publishes a one-model catalog. Requests for the Nighttime model return `MODEL_NOT_FOUND`; there is deliberately no alias substitution.
- **Back to a paired profile.** Daytime is recreated first, so the exclusive container releases every GPU, and Nighttime second. Nighttime's in-memory prompt cache does not survive.
- **Failed switch.** Recovery follows the same order. Restoring a paired release recreates Daytime and then Nighttime. Restoring an exclusive release first removes any Nighttime the failed target started.
- **Drift.** A running Nighttime beside an exclusive release is reported as `competing Nighttime backend running`, and the next apply removes it.
- **Status.** `/api/status` reports `offline_roles: ["everyday"]` while an exclusive profile is active, and the page asks for confirmation before selecting one.
- **Unchanged rules.** Browser switches between paired profiles still refuse to touch Nighttime.

Qualification on 2026-10-06:
- **Bench Studio coding session, 2/2 passed:**
  - `daytime-flash-solo`: 80.7 tokens per second decode, median first token 0.88 s.
  - The two-GPU `daytime`, for comparison: 31.5 tokens per second and 3.78 s.
- **Fitted layout:** no layer overflowed to the CPU, and the model used about 63 GB of the four text GPUs.
- **Tuning method:** each change was measured against `daytime-flash-solo` with a fixed replay of a recorded Bench Studio session (48 requests, greedy, prompt cache on, about 18.5K generated tokens) and fixed-depth prefills of 8K, 32K, and 98K tokens. Repeating an identical configuration varied about ±0.5% in decode and ±1% in prefill.

| Change | Replay decode (tok/s) | Prefill 8K / 32K / 98K (tok/s) | Result |
| --- | --- | --- | --- |
| `daytime-flash-solo` (baseline, two runs) | 86.1, 85.7 | 1129 / 1058 / 926 | — |
| MTP depth 3 | 87.3 | 1170 / 1065 / 928 | Kept in `-tuned-mtp3` |
| MTP 3, `--spec-draft-p-min 0.6` | 72.1 | unchanged | Rejected |
| MTP 4, `--spec-draft-p-min 0.75` | 65.7 | unchanged | Rejected |
| `--batch-size 4096` | 85.9 | 1099 / 1055 / 925 | No effect: tensor overrides from fitting disable pipeline parallelism |
| `--lazy-mode off` (n-gram table page-cached in host RAM) | 86.1 | 1164 / 1077 / 938 | Kept |
| `--ubatch-size 2048` | 66.4 | 994 / 952 / 846 | Rejected: fitting no longer kept every expert on the GPUs |
| F16 K/V | 87.3 | 1140 / 1070 / 937 | Kept |
| Whole-layer split with pipeline parallelism, microbatch 512 | 90.3 (different output) | 964 / 914 / 831 | Rejected: slower prefill |
| Whole-layer split, microbatch 1024, pipelining off | 86.6 | 1141 / 1074 / 939 | Kept (deterministic) |
| Engine `qwen38-dual-43fe9c6` | 85.9 | 1152 / 1098 / 1004 | Kept |
| **`daytime-flash-solo-tuned`** (all kept changes, MTP 2) | **90.2** | **1206 / 1138 / 1038** | +5% decode, +7–12% prefill |
| **`daytime-flash-solo-tuned-mtp3`** | **94.3** | **1203 / 1135 / 1038** | **+10% decode, +7–12% prefill** |

- **Second, unseen workload** (29 later requests from another session, about 15K generated tokens): `daytime-flash-solo` 85.0, `-tuned` 91.0, `-tuned-mtp3` 95.2 tokens per second. Median prompt time was 801, 763, and 776 ms.
- **Why p-min lost:** `--spec-draft-p-min` raised acceptance (0.86–0.88) but cut decode by 16–24%. Stopping drafts early costs more steps than the cheap extra verification saves.
- **Workload sensitivity of depth 3:** it gains less on long, low-acceptance outputs. One 9,221-token answer in a Bench Studio session decoded at 61.8 tokens per second with depth 3, at 0.56 acceptance. `daytime-flash-solo-tuned` keeps depth 2 for that kind of workload.
- **Host RAM and lazy reads:** with lazy reads off, the 35.8 GiB table is held as page-cached host memory (Rosalina has about 98 GiB available in solo mode). Raising `--cache-ram` would not help single-agent work, because prompt-cache reuse already serves about 97% of each agent turn's prompt from the slot.
- **The new engine:** `qwen38-dual-43fe9c6` is llama.cpp `43fe9c64281ef735046adc025e9e7559a1f659a5` (2026-10-06, every CUDA CI job green). It adds:
  - MMVF for thin F16/BF16 matmuls at small batch (#29633)
  - whole-tile FlashAttention scheduling (#29435)
  - the tiled lightning-indexer kernel (#29901)
  - the k-pool graph-reallocation fix (#29958)
  - batch-independent CUDA graph dependency checks (#29986)
  - the MMQ fix for `n_expert >> n_ubatch` (#29941)
- **How the engine was built:** on the host from the unmodified `.devops/cuda.Dockerfile` with the `836d571` command, and the CUDA devel, CUDA runtime, and Node base images pinned to the digests of the `836d571` build. The receipt is under `~/ops/reports/20261006-llama-engine-43fe9c6/`.
- **Retired:** the one-variable experiment profiles were retired after measurement; this table is their record.

Host CPU use (2026-10-06): the first tuned profiles kept the shared `--threads 18`, and decode then used about 9–11 CPU cores where `daytime-flash-solo` used one. Measurements on Rosalina:
- **Cause:** engine `43fe9c6` alone reproduced it. On the older engine, the pinned layout, MTP depth, lazy mode, and `--poll 0` changed nothing.
- **Where the CPU went:** a scheduler trace (`GGML_SCHED_DEBUG=2`, verbosity 5) showed the decode graph's CPU input split gained `DUP` and `SET_ROWS` nodes for the new mixed embedding-and-token batch input (llama.cpp #29622), next to the token and per-layer embedding `GET_ROWS`. The work is microseconds, but it now runs as an 18-thread OpenMP region several times per decode step. Between regions, libgomp workers busy-wait, so 17 threads sat at about 57% each.
- **Proof that this is only waiting:** `OMP_WAIT_POLICY=PASSIVE` cut decode CPU from 10.7 to 1.05 cores at the same speed. `--threads 1 --threads-batch 1` also gives about 1.0 core.
- **No cost to speed:** on the session replay, both tuned profiles with one host thread produce identical output at the same speed: 93.2 against 93.1 tokens per second for `-tuned-mtp3`, with identical 221 s wall time. CPU fell from 2055 to 224 CPU-seconds, about 118 to 13 ms of CPU per generated token. Prefill speed is unchanged (1124–1132 tokens per second at 32K).
- **Why one thread is enough:** every expert is on the GPUs. The remaining host work is driving the GPUs (the main thread, about one core in every solo profile) and the tiny input split.

Concurrency and context experiments:
- **Two slots.** A profile may set `"parallel_slots": 2`, and only exclusive profiles may. The renderer then passes `--parallel 2`, sizes `--ctx-size` for both slots, and keeps `--kv-unified-per-slot` and the catalog `context_length` at the per-request window. The controller verifies the slot count and each slot's context.
- **Router admission follows the slots.** The catalog publishes `max_active_requests` as the slot count and `total_context_length` as the window times the slots. LLM Router `a2f3406` and later admits two overlapping requests for such a resident and queues the next. An older router rejects a two-slot catalog, and the switch then recovers to the previous profile.
- **`daytime-flash-solo-tuned-mtp3-2slot`** (renamed `flash-next-solo-two-requests`, then retired on 2026-10-07; the two-slot capability remains): two 128K slots with q8_0 K/V, so the pool uses the same KV memory as one F16 slot, and a 512 microbatch. At microbatch 1024, a cold 32K prefill in one slot while the other decoded hit the same RTX 4080 `top_k` out-of-memory abort as F16 at 160K.
- **`daytime-flash-solo-tuned-mtp3-160k`:** the tuned profile with a 160K window and q8_0 K/V. With F16 K/V at 160K, the RTX 4080 kept about 0.8 GiB free, and a cold 32K–98K prefill aborted the backend. The CUDA out-of-memory error (`cuMemCreate`) came from the VMM pool in `top_k` (CUB argsort) for the sparse-attention indexer, scratch that the compute-buffer reservation does not include. Docker restarted the container, and no other backend was affected.

Measured on 2026-10-06, direct to the backend: two recorded Bench Studio sessions, A (48 requests) and B (29 requests), greedy, prompt cache on.

| Configuration | Single request (tok/s, A / B) | Both at once (tok/s each, A / B) | Total throughput, both sessions |
| --- | --- | --- | --- |
| `-tuned-mtp3`, one slot | 90.1 / 97.6 | (router queues the second) | sequential |
| two slots, microbatch 1024 | 83.2 / 95.4 | 50.5 / 56.2 | 90.7 tok/s against 75.8 back to back (+20%), then aborted on a cold 32K prefill beside a decode |
| two slots, microbatch 512 | 89.5 / 90.7 | 49.0 / 48.3 | 80.6 tok/s against 74.4 back to back (+8%) |

- **Why the second request is not free.** The low GPU utilization does not translate into a free second stream. Every step carries both sequences' MTP verification batches through the same four-stage layer pipeline and the per-sequence draft passes, so each request runs at about 55% of its solo speed and median prompt time rises 25–45%.
- **Prefill blocks the other slot.** A cold 32K prefill in one slot (37 s at 892 tokens per second) held the other slot to about 2 streamed tokens per second for its duration, with gaps of up to 2.5 s.
- **160K window (q8_0 K/V), no cost from the allocation itself.** Replay decode was 88.5 tokens per second, and prefill was 1236 / 1128 / 1030 tokens per second at 8K / 32K / 98K, matching the 128K F16 profile within its noise.
- **The cost of long context comes from using it,** with either window:

  | Context depth | Decode (tok/s) | Prefill (tok/s) |
  | --- | --- | --- |
  | 8K | 90.5 | — |
  | 64K | 56.5 (59.5 at 128K F16) | 1088 |
  | 120K | 36.1 (38.2 at 128K F16) | 991 |
  | 155K | 34.5 | 942 (165 s cold) |

- **128K headroom check:** `-tuned-mtp3` completed a cold 130K prefill without errors.
- **Routed two-slot acceptance** (LLM Router `a2f3406`, 2026-10-07), with no backend restart:
  - Discovery advertised `active_request_limit: 2` at 131,072 tokens each. Three short requests ran two at a time, and the third queued.
  - Two concurrent 60K cold prompts both completed (141 s for both).
  - A 120K prompt alongside a 32K prompt both completed. Free VRAM afterwards was at least 0.8 GiB on each text GPU.

## Speculative decoding experiments

These follow the DFlash 2 review of 2026-10-07. DFlash 2 drafts a block of tokens per pass from a small model trained for Qwen3.8-27B. llama.cpp supports it ([#27342](https://github.com/ggml-org/llama.cpp/pull/27342)), and both pinned engines include it. With `--split-mode tensor`, however, it aborts in `ggml-backend-meta.cpp` ([#28777](https://github.com/ggml-org/llama.cpp/issues/28777), [#27833](https://github.com/ggml-org/llama.cpp/issues/27833)); the fix [#27858](https://github.com/ggml-org/llama.cpp/pull/27858) is an unmerged draft. Each experiment is one configuration that adds one drafter to the configuration it names, and each has its own alias for separate benchmark history. The owner benchmarks each one; the runtime only verifies that it loads and drafts.

1. **Copy drafter:** `qwen27b-q6k-copy-drafter-with-nighttime` and `flash-next-solo-128k-copy-drafter`. Retired on 2026-10-07 after the owner's planning session.
   - **Change:** `--spec-type draft-mtp,ngram-map-k4v`. llama.cpp tries the lookup drafters before draft models. When the last 12 tokens recur in the context (`--spec-ngram-map-k4v-size-n`, default 12), `ngram-map-k4v` proposes up to 48 copied tokens (`size-m`, default 48); otherwise MTP drafts its usual three. It needs no VRAM.
   - **Expected:** a published DFlash 2 study measured +68% on iterative multi-turn coding from one lookup drafter, and nothing on single fresh prompts. A long rejected lookup draft costs a larger verification batch, which matters more for the Flash-Next MoE than for the dense 27B.
   - **Engine:** the 27B configuration is on `836d571`, which predates #29924. That fix concerns lookup drafts truncated at temperature > 0, which happens only at the end of the context window.
   - **Result:**
     - **File rewrites (quick check):** 2.4× on 27B (270 against 111 tokens per second) and 1.9× on Flash-Next.
     - **Planning:** slower. Over the owner's 26-request planning session it averaged 55.4 tokens per second, with 42% of drafted tokens accepted, at contexts up to 53K.
     - **Matched planning prompt (45K context):** 57.8 against 64.6 tokens per second at temperature 0.7, with 49% against 61% acceptance. Greedy decoding gave 66.6 against 69.3.
     - **Why:** plans repeat paths and identifiers, so the lookup proposes long copies the model then departs from. Each rejection costs a large verification batch. At temperature > 0, lookup drafts also need exact matches, while MTP drafts get rejection sampling.
2. **DFlash 2 on Daytime 27B Q6_K in layer mode:** `qwen27b-q6k-dflash2-with-nighttime`.
   - **Drafter:** the MTP3 head is replaced by z-lab's `Qwen3.8-27B-DFlash2-GGUF` Q4_K_M (revision `2d9571f`, which includes the 2026-08-24 `dflash.rope.dimension_sections` update), mounted as `/weights/dflash2.gguf` on the RTX 3090.
   - **Depth:** `--spec-draft-n-max 5`. A published sweep peaked at 5; the block size is 8, so llama.cpp caps the depth at 7.
   - **Placement:** `--split-mode layer`. Layer mode gives up tensor parallelism, so it must recover about 30% to match the current configuration. Nighttime is unchanged.
   - **First load (2026-10-07) failed:** with `--device CUDA1,CUDA0`, the draft context aborted with `pre-allocated tensor (output.weight) in a buffer (CUDA0) that cannot run the operation`. The drafter shares the target's output layer, which llama.cpp places on the last device, the RTX 4080 SUPER, while the drafter runs on the RTX 3090.
     - **Recovery:** the backend restart-looped as a failed initializer, so automatic recovery timed out waiting for it (`needs-attention`). The documented repair restored the previous configuration in about 8 minutes, with Nighttime untouched: confirm no active work, stop that container, run `runtime recover`.
     - **Fix:** `--device CUDA0,CUDA1 --tensor-split 40,60` lists the RTX 3090 last, so it holds both the output layer and the drafter, with the same 60/40 share.
   - **Copy drafter:** not combined, because experiment 1 showed it hurts planning.
3. **Speculative decoding for Nighttime** (after that), which has none today: the base 27B MTP head in tensor mode, and DFlash 2 in layer mode. The Nighttime cards lack the room, so the draft goes on the shared RTX 3080 for these experiments.

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
