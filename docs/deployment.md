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

## Strata trial profile (flash-next-strata-160k)

`flash-next-strata-160k` is a **trial**, added 2026-10-11: the same four text GPUs and the same 163,840-token window as `flash-next-solo-160k`, but the model served by [Strata](https://github.com/Niko1221/Strata) instead of llama.cpp, from Unsloth's dynamic 4-bit `UD-IQ4_XS` quant (93,682,584,224 bytes, revision `38bb39ee97821de2c9009abb7e93950eec396e66`). It exists to answer one question: whether Strata's profiled expert caching beats llama.cpp's resident layout where llama.cpp is weakest, which is decode in a long context (measured below: 97 tokens per second at 4K falling to 35 at 128K).

- **Its own service ID, and never anybody else's.** It declares `"aliases": ["trial"]`: `trial` is the ID a client sends for it — `GET /v1/router/capabilities` reports `models[].service == "trial"`, and `POST /v1/chat/completions {"model": "trial"}` resolves to this backend — and it carries no `daytime`, `local-active` or `nighttime` alias, so selecting some other configuration can never route to it and no client that was told to send `daytime` lands on it by accident. Selecting it is an explicit act, and only the owner does it. What that costs is stated plainly, in the catalog itself and in `deployment_warnings`: **while this profile is active there is no `daytime` service on the router at all** — `daytime` is refused as `MODEL_NOT_FOUND`, not answered from this backend — because this configuration replaces the Daytime resident instead of adding a second one. A client that only ever sends `daytime`, which is what LLM Router's `CLIENT_CONTRACT.md` §3 tells it to do, has nothing to send until a Daytime profile is selected again; `trial` has to be a name it knows, and does not become one on its own.
- **Two properties, both tested** (`tests/test_config.py`, `ServiceIdentityTests`). Every selectable configuration has to publish a service alias a contract client can address: LLM Router names an entry by the first of its aliases that is not the compatibility alias and otherwise falls back to the canonical ID (`src/capabilities.js`), and a client forbidden from sending canonical IDs cannot ask for such a model at all — a backend that is healthy, selected, and impossible to reach. That hole existed: the trial published `aliases: []` from 2026-10-11 to 2026-10-13 and the primary client on this machine had no model while it was selected. The other half is that `trial` is only ever the trial, and never `daytime`, `local-active` or `nighttime`, in this configuration or any other.
- **Both alias shapes need LLM Router `c3a342b`.** That is the revision that stopped requiring a resident catalog's default entry to declare the `local-active` compatibility alias: the empty alias list this replaced was only publishable because of it, and a catalog carrying only `trial` needs it just as much. A router older than that refuses either shape as `INVALID_MODEL_CATALOG` — at publication, so after the drain, with the prior catalog written back and the switch restored rather than the host left drained. Selecting this profile therefore takes a router at `c3a342b` or later.
- **Engine** `strata-7695859aa13a`: Strata `v0.1.43` at `7695859aa13afb248bdb4d36f9b2e95707db079d`, image `local/strata:7695859aa13a` (`8d852ef69ad6…`), compiled for `sm_86,sm_89` on `nvidia/cuda:13.0.0-devel-ubuntu24.04` (`1e8ac7a5…`). No weights are in the image: it reads the mounted revision.
- **It is not configured by flags.** The container's entrypoint reads its settings from the environment and writes its own engine config, logs and installed state. The engine therefore declares `"interface": "environment"` plus a map from each profile argument to the variable the entrypoint reads, and is rendered from its own arguments alone; the llama.cpp vocabulary, the per-slot KV flags, and the artifact-only mount list are not applied to it. What follows from that, and what stayed:
  - The service has an empty command, `read_only: false` (its config link and logs go under `/opt/strata`), and one writable bind: `ai/strata-trial/data` at `/data`, where the engine keeps its expert pack, its MTP draft layer and its own logs. Every model file stays a read-only bind, and the mount check in `observe` now honours each mount's own read-only flag instead of assuming all of them are.
  - `GGUF_DIR` is derived from the mounted model path and `CONTEXT` from `context_tokens`, so the engine cannot be pointed at a revision the profile did not checksum.
  - Its health and idle checks are unchanged: Strata answers `/health` and `/slots` in llama.cpp's shape. Its window is checked against the published configuration instead of the argument vector, the bounded generation probe runs in the OpenAI chat dialect (there is no `/completion` and no `/tokenize`), and the published output-policy record says `docker-inspect-env` because there is no argv to read a ceiling out of.
  - A fresh container reads all 94 GB of its shards, prepares its pack and installs its draft layer before it answers a health check, so the engine names its own readiness window (3600 s, and the same start period on the healthcheck) instead of inheriting the 450 s llama.cpp waits.
- **An engine's container settings are merged over the house ones, never in place of them.** This is the lesson of the trial's first switch attempt (2026-10-11, `ops/reports/20261011-strata-trial/FAILURE-20261011-switch.md`): the engine entry declared its own `container_defaults` — entrypoint, root, ulimits, healthcheck — and those *replaced* the shared defaults, so the service arrived with no restart policy, no init, no logging and no pull policy, and the observer died on `KeyError: 'restart'`. It failed closed, 93 GB of artifact checks in and no container touched: the active profile never moved and nothing was stopped. Three things follow, and all three are in the code now: engine defaults are deep-merged over the shared ones before the profile is merged over them (`merge_settings`); every rendered service is checked against `CONTAINER_CONTRACT` at render time, so a service that cannot state how it runs is a configuration error at deploy rather than a crash at switch; and `observe` reads every expectation with `.get()`, so a bundle that lost a key produces the named drift reason instead of an exception. The same observation pass runs over the four llama.cpp profiles with no new reasons, and they render byte-identically to the revision they were deployed on.
- **What the backend answers is checked, and a mismatch says what it saw.** At the pinned revision `/health` replies `{"status": "ok", "max_context", "model", "images", "api_key", "loaded", "service": "strata"}` and `/slots` replies a list of `{"id", "n_ctx", "is_processing", "n_prompt_tokens"}` — one entry per slot, or `[]` while the engine is not loaded. `n_ctx` is the window the *engine* loaded with, which is not necessarily the one asked for, so a window mismatch now records `slot_window` (`contracted`, `reported_context`, `loaded`) in the observation as well as the `slot context` reason: a backend that came up on a smaller window is permanent and must be readable, while an empty slot list is one still loading, and those two otherwise look identical while the switch sits out its health timeout.
- **Pre-flight that needs no switch and allocates nothing:** `docker run --rm --user 1000:1000 --gpus all --entrypoint /opt/strata/.venv/bin/python local/strata:7695859aa13a setup.py --check --yes` reports the engine's own verdict on the machine without installing or downloading anything (`ops/reports/20261011-strata-trial/preflight-strata-check.log`, 2026-10-11: all five cards usable, `[ok] RAM: 123 GB`, `[ok] PCIe: 3.0 x16`, `[ok] CPU … (AVX2)`, and `UD-IQ4_XS needs ~48 GB RAM: fits with 55 GiB of its experts in RAM, the rest read from the SSD`). It is not a preview of what the profile will do — the profile hands it four cards and its own settings.
- **Build** (host `ops/strata-build/`, receipt in `ops/reports/20261011-strata-trial/`): clone, `git checkout` the pinned revision, then their own Dockerfile with `--build-arg CUDA_ARCHITECTURES=86,89 --build-arg BUILD_VISION=0` (the profile serves text only). One overlay layer is added on top: `chmod a+w /opt/strata`, plus the OCI `revision`/`source` labels the controller pins against. The engine's `BUILD.json` inside the image reports `archs [86, 89]`, `vision none`, `version 0.1.43`.
- **This backend runs as the image's own user, and not as the deployment user** (since 2026-10-12; every other backend in the house keeps the unprivileged `1000:1000`). The `chmod a+w /opt/strata` above is not enough: the engine installs its Python packages into `/opt/strata/.venv` on first start, which the overlay left root-owned at 755, so as the deployment user it died 1.3 seconds in with `PermissionError: [Errno 13] Permission denied: '/opt/strata/.venv/.strata-pip.json'`. It is stated as `runs_as_image_user` on the engine entry, which renders the service with no user at all, and the observer accepts the empty string the daemon reports for it as matching. This is a wider grant than anything else the controller starts — the container gets the image's privileges — so it is on the engine, where it can be read and removed, and not inferred.
- **The family it is told to use is `unsloth`, not `qwen`** (changed 2026-10-12, after two runs that produced nothing). The staged shards are Unsloth's UD-IQ4_XS; with `FAMILY=qwen` the setup stops at its second step — `[X] Qwen3.8-Flash-Next has no UD-IQ4_XS model file`, `Setup stopped.` — and the container exits in a second, which from a terminal watching a counter is indistinguishable from a very slow model. With the family the engine itself names for that quant, setup reaches step 5 and starts hashing the 94 GB already on disk instead of downloading anything.
- **Cards.** `CUDA_VISIBLE_DEVICES=0,1,2,3` with `GPUS=0,1,2,3` and `LAYER_SPLIT=auto`: the four LLM cards, placed by Strata by free VRAM. The RTX 3080 is not in the container at all, so this profile has no projector, no vision, and its catalog says so (`input_modalities: ["text"]`, `capability_profile.vision: false`). `memlock` is unlimited and the stop grace period is 180 s. It never co-resides with a llama.cpp backend: it is exclusive, so Nighttime is stopped and Daytime is recreated in its place.
- **Not enabled in the profile:** `--calibrate` and any auto-tuning, `VISION`, a `daytime`/`local-active`/`nighttime` alias of its own, and a second slot. The strict-determinism measurements (single-thread, cache-off) belong to the benchmark harness, not to the served configuration.
- **What `output_policy: unrestricted` means here.** The house contract requires it, and it is true in the sense it is used: the deployment sets no server-side output ceiling. It is **not** a claim that this quant produces unrestricted content: `UD-IQ4_XS` is Unsloth's dynamic quant of the official Qwen release, not an uncensored variant, so `nsfw` stays `false` and a warning to that effect ships with the catalog entry. It has not been attested on the trial quant.
- **Retention, bounded when it was created** (the rule is repeated in the profile's `retention` block): the image — keep exactly the registered tag for as long as the profile exists, superseded tags removed reference-aware and never by a blanket prune; the model revision — retained only while the profile is registered, then it is an ordinary library revision under the normal pass; the workspace `ai/strata-trial/data` — bounded by the engine and removed with the profile; reports — `ops/reports/<date>-strata-trial`, following the existing reports aging. Root filesystem headroom is checked before and after the download (the 85% warning line in the house maintenance pass); at staging it was 434 GB free of 938 GB.
- **Before selecting it:** see [the switch preparing the trial](#the-switch-prepares-the-trial-itself-and-what-the-measurements-still-do) — the switch builds the workspace itself, beside the serving model, before it drains, and waits for the engine's actual answer; the page shows on the profile what the selection will do, and a refusal, if there is one, comes with the evidence it rests on. The circuit breaker for the trial, agreed with the owner: any sustained state under 8 GiB MemAvailable or under 150 MiB free on any of the four cards is expected behaviour for a 94 GB model on 123 GB of RAM and is the signal to switch back, not an incident to debug. Switching back **is** the rollback; teardown is removing the profile and then the image reference-aware.
- **Baseline measured on 2026-10-11 from the running `flash-next-solo-160k`** (same probe, prompt set and method the trial will be measured with; three distinct prompts per length, greedy, 256 new tokens, direct to `127.0.0.1:18080`; `ops/reports/20261011-strata-trial/baseline.jsonl`):

| Prompt tokens | First token (s) | Prefill (tok/s) | Decode (tok/s) |
| --- | --- | --- | --- |
| 4,096 (4,141–4,150) | 1.44 / 3.10 / 3.41 | 1,216–2,882 | 88.2 / 97.0 / 97.3 |
| 16,384 (16,444–16,462) | 14.09 / 16.94 / 17.20 | 957–1,167 | 78.2 / 82.3 / 84.3 |
| 32,768 (32,840–32,862) | 2.77 / 3.55 / 32.25 | 1,019 (cold); 9,261–11,851 (prefix-cached) | 68.7 / 69.7 / 72.6 |
| 131,072 (131,116–131,348) | 137.2 / 137.7 / 138.8 | 946–956 | 35.4 / 35.5 / 35.9 |

  The three prompts of one length share their filler, so the 32K row shows both a cold run (1,019 tok/s) and prefix-cache hits (9,261–11,851) rather than three comparable prefills; the 16K and 128K rows are cold. The trial must be read the same way, cold run against cold run. Nothing here was estimated: every number is one recorded request.
- **Open questions the trial still has to answer**, stated at creation: how long the first start actually takes (the pack plus the ~6 GB draft download, which needs Hugging Face egress and is the engine's own mtp/rt form, not the `mtp-*.gguf` heads in the library) — since 2026-10-12 this one is answered: the first-start state builds in under ten minutes beside the serving model, and the switch does that build itself before it drains, waiting for the engine's real answer afterwards; whether `--vision no` in a `BUILD_VISION=0` image changes anything the router or `/v1/models` reports. Narrowed since: the numbering question only decides *which of the four* is card `0`, because the container is given exactly the four reserved UUIDs and `GPUS=0,1,2,3` asks for all of them — the pre-flight shows the host enumerates 0=RTX 3080 Ti, 1=RTX 4080 SUPER, 2=RTX 3090, 3=RTX 4080, and `LAYER_SPLIT=auto` places the layers by free VRAM whichever end of that list is which.

## The switch prepares the trial itself (and what the measurements still do)

The trial was switched to four times on 2026-10-12. The first attempt failed on merged container settings. The second (01:12) got the container up and spent its whole readiness window inside the engine's first-boot work — reading 94 GB of shards, building the expert pack, installing the draft layer — never reached `/health`, and was abandoned with the router drained. The third (17:42) was the first to get the switch past the drain: the engine's server came up and its own health check passed while the model was still reading, so Docker — and `up --wait` — called the load done in about two minutes, the controller's one-shot verification saw the slot list still empty, aborted the switch, and the restoration put the previous profile back in 43 s. The fourth (18:47, with the wait in place) proved the wait: the switch re-observed on its own cadence while the engine reported it was reading, and when the engine stopped itself mid-read with the cards free and the kernel quiet, the switch failed at once instead of spending the window, and the restoration put the previous profile back in about two and a half minutes. The engine's own account of the stop was in a log inside the container, and the restoration deleted it — so the failure record now carries the state, the exit code, and the last words itself, read while the container still exists. The readiness window on this engine was never a measurement; it was 3600 s because that was a plausible number, and `up --wait` was not the thing that waits it out.

What the host measurements then showed, 2026-10-12:

- The first-start state — the 94 GB shard verification pass, the expert pack, the MTP draft — is written in **under ten minutes** (559 s to the point of no return, 7.9 GB into `/data`), and the served model never noticed: `neighbour_slots_max_seconds` 0.003 s against a 0.002 s baseline, its health never leaving `healthy`, card memory unmoved, MemAvailable flat at 66 GB. **Packing beside a loaded model is not the danger; the load is.** Once the state exists the engine immediately reads the whole model onto the four cards, and with llama.cpp holding them it dies exit 1 at `starting the engine: reading the model's weights`.
- The two runs before that never built anything at all: the engine exits in 1.3 s when the profile names the wrong family for its files (`--family unsloth`, not `qwen`), and as the house user it cannot write its own image's venv (`PermissionError: /opt/strata/.venv/...`), so the engine entry states `runs_as_image_user` — a wider grant than the house's, stated on the engine rather than inferred. And when it did build, the run waited for a ready file under the name the other family would write: the declaration is now a pattern (`config/strata-*-ud-iq4_xs.json`), the receipt records the path it matched as `ready_paths`, and a container that stops by itself ends the run in seconds with its own last words in the receipt (`container_log`) instead of being polled for two hours.

The design that grew out of that: **selecting the profile is the operation.** A switch to a self-provisioning engine

1. **builds the workspace itself, before the drain** — the half that is safe beside a loaded model (measured above), in its own `preparing` phase, through the same throwaway project the read-only tool uses: same image, entrypoint, arguments, environment, user, binds and four card reservations, different container name and published port, service aliases dropped, its own project name so teardown can only ever reach its own leftovers. If the engine cannot build the workspace, the switch stops at the edge of the transaction: no drain, no touched backend, and the error carries the engine's own last words.
2. **drains and loads as any switch does**, waiting for the engine's actual answer. An engine like Strata answers `/health` with `status: ok` from server start — the model state rides in its `loaded` field, and `/slots` returns nothing until the read completes — so Docker being healthy is not the model being resident, and `up --wait` alone is not the wait. `readiness_wait_seconds` is the bound on the controller's own re-observation (every 15 s) until the slots report the contracted window. A loaded engine on the wrong window, or any other drift, is not waited out: the failure record carries the per-role differences, the slot window the engine reported, and — when the engine stops by itself, the one failure that leaves no health and no slots behind — the state, the exit code, and the engine's own last words, read while the container still exists because the restoration that follows removes it. The switch's bounded-generation acceptance still has to pass.
3. **records the boot it just waited** — the seconds the switch waited become the receipt's `warm_boot_seconds` / `healthy_warm_boot` (`via: switch`), so after one successful switch the measurements exist without anyone having taken them.

The only refusals left are the ones the machine's evidence backs, and they happen before the drain:

- a warm boot that was measured and **failed** (`never completed a warm boot from its own workspace`) — the honest move then is to fix or retire the profile, not to spend a drain and a restoration relearning it;
- an engine that **stops itself at startup** — a configuration fault, reported in the words the engine gave.

Everything else is a **note**, not a refusal: `python3 -m runtime gate --profile <name>` (read-only, nothing locked, nothing drained, exit 1 only on a refusal) reports `reasons` and `notes` — the workspace still needs building (and how long the last build here took), a readiness window thin against a measured boot — and the page shows the same verdict on the profile itself, from the per-profile `reasons`/`notes` in `/api/status`, before the selection is made. A thin window is worth knowing; it is not paperwork, because the switch waits for the answer and the recovery path stands behind a switch that turns out to be wrong.

The read-only tool remains for evidence and diagnosis:

```sh
docker exec local-ai-runtime python3 -m runtime provision --profile flash-next-strata-160k
# build phase: throwaway project, beside the serving backend; nothing stopped

docker exec local-ai-runtime python3 -m runtime provision --profile flash-next-strata-160k --phase warm
# boot phase: refuses while anything is serving; it never stops the serving backend to get the cards
```

Runs from the tool and runs from a switch merge into one receipt, `.state/provisioning/flash-next-strata-160k.json`: engine tag and image id, workspace, ready files, `provision_seconds`, `warm_boot_seconds`, `healthy_warm_boot`, `pack_bytes`, `disk_free_bytes` before and after, `neighbour_healthy`, `neighbour_slots_max_seconds`, `ready_paths`, and every run with its samples and `via` (`cli` or `switch`). A receipt counts only for the image id and workspace directory it was taken with, so a rebuilt tag or a moved workspace invalidates it. The health answer is read from the container's **name on the router network** — the route the controller uses for every backend — with the published loopback port as fallback; `health_via` records which one answered, because a measurement taken through a route that does not exist would report an engine that never came up.

Teardown between attempts is `compose down` on the provisioning project name and nothing else — never a blanket prune, and never the serving stack's project.

## What the trial measured in its first day of real traffic (2026-10-14)

Sources, all of them the machine's own words: the engine's durable log `ai/strata-trial/data/strata-engine.log` (**771 served requests** from one boot, 11 hours of live agent traffic), the config the engine generated for itself, `/v1` engine info, and `tools/strata_bench.py --long` recorded as `data/bench-results/baseline-2026-10-14/`. Nothing below is estimated; every number is one recorded request or one engine line.

- **The layout the engine chose** (boot lines): `layer split auto: K=5,26,37`, so CUDA0 (RTX 3080 Ti) runs layers 0–4, CUDA1 (the RTX 3090, 23.56 GiB) layers 5–25, CUDA2 layers 26–36 and CUDA3 layers 37–47 plus the head — **predicted 27.5 ms per decode window**, with "the caches hold 16571 of 24576 profiled pairs (**~96.5% of the routed mass**), best of 15180 placements". Expert cache `auto`: **16,420 resident experts, 37,877 MiB** across the four cards (1,895 / 7,563 / 3,812 / 3,150 slots).
- **The storage hierarchy is not in the serving path, and the RAM budget is over-provisioned by design.** Setup asked for `--resident-budget-gib 55` (the pre-flight's "fits with 55 GiB of its experts in RAM, the rest read from the SSD"), and the engine allocated a **32.90 GiB** page-locked complement instead — because the GPU tier already holds everything but 18.44 GiB of experts. Across all 771 requests: **`0 blob reads from the file`**, and cumulative **expert cache hit rate 98.84% over 367,631,500 lookups** with 0.0% of routed experts read over PCIe. KV is `k8v4` with `--kv-resident 32768` and streaming on: **96.96% of 17.2 M block reads hit VRAM** in the longest sessions.
- **PCIe is fine.** The probe reports 8.0 GB/s host→device and sets `pcie_frac 0.22` (default 0.55), which makes a cache miss cost ×2.51; `nvidia-smi` shows all four cards at **Gen3 x16**, so the probe is the measured single-stream direction and not a lane fault. Speculation is `--spec 4 --spec-min-p 0.5` on an 835 MiB draft layer, accepting 73–76% in production.
- **Two of the engine's arguments are hand edits, not profile settings.** `strata-unsloth-ud-iq4_xs.json` differs from its own `.bak-original` by `allowed_hosts`, the durable `log` path, and **`--vram-reserve-mib 2048`**. They survive because nothing has reinstalled; see the reachability note below.

**Decode is flat in context, which was the trial's whole question.** 620 requests that generated at least 120 tokens, bucketed by prompt size:

| Context | n | median | p10 | p90 | draft accepted |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0–8K | 20 | 112.0 | 97.4 | 150.7 | 90% |
| 8–16K | 19 | 114.7 | 106.3 | 137.5 | 76% |
| 16–32K | 42 | 116.8 | 103.3 | 129.4 | 73% |
| 32–48K | 27 | 113.4 | 101.9 | 130.3 | 74% |
| 48–64K | 82 | 112.1 | 101.6 | 124.1 | 74% |
| 64–80K | 140 | 110.8 | 99.1 | 125.7 | 75% |
| 80–100K | 179 | 108.4 | 99.1 | 120.6 | 73% |
| 100–131K | 112 | 107.8 | 98.1 | 117.5 | 73% |

**−3.8% from the shortest bucket to the longest**, against the llama.cpp baseline above that fell 97 → 35 on the same cards; longest prompt served 129,025 tokens. Strata's profiled expert caching plus KV streaming does remove the long-context decode collapse. That is the answer the profile was added to get.

**The floor lives in the tail, not in the median.** Overall decode median 110.7, p10 99.7, p90 124.9, max 155.5 — and **65 of the 620 long generations (10%) fell below 100 tokens per second**, with p10 at 98–99 in every bucket above 64K. The owner's reading of "108–109" is the median at working context. There is no 8% of headroom to spend on a longer session: **the tail is already at the floor**, and the useful work is raising it.

**Prefill, not decode, is where this machine spends its time: 12,487 s of prompt processing against 5,656 s of generation (2.21×), and 11,577 s of that prompt time — 94% — is in requests that reused less than half their prompt.** Fresh-prompt throughput rises with the size of the read (879 tok/s at 4K, 1,685 at 16K, 2,171 at 32K from the recorded suite; 2,415 median on 50K+ fresh reads in production), and small reads are poor: 241 tokens cost 0.92 s and 917 cost 1.90 s, because the engine chunks prompts at 8,192 tokens (`prompt chunk auto: 8192 tokens, a 96-slot ring`). The bench's own agent-session cache hit is 68.8%.

**`bench.sh` is the instrument for the next round — for prefill, and not for decode.** `/opt/strata/bench.sh` is a wrapper for `tools/strata_bench.py --long`, which drives the HTTP API of the already-running server (`--url http://127.0.0.1:8080`, `--out` into the durable workspace) and takes about three minutes. It has been run three times against the live trial:

| | 15:28 | 17:05 | 17:25 (silent) | spread |
| --- | ---: | ---: | ---: | ---: |
| 4K prompt tok/s | 879.5 | 882.5 | 882.4 | **0.4%** |
| 16K | 1,684.9 | 1,693.3 | 1,694.5 | **0.6%** |
| 32K | 2,171.0 | 2,171.3 | 2,161.7 | **0.6%** |
| 32K first token (s) | 14.81 | 14.80 | 14.88 | **0.6%** |
| agent cache hit % | 68.8 | 68.5 | 68.8 | 0.4% |
| agent decode tok/s | 109.4 | 108.2 | 104.0 | **4.9%** |
| short-chat decode tok/s | 98.7 | 93.1 | 98.2 | 5.7% (range 64–140) |

**Prefill repeats within ±0.6% and decode does not repeat at all: 4.9% across three runs, and the silent run was the slowest of the three.** Pooling the 108 agent-decode samples from the three runs (64 to 146 tok/s, CV 17%) and bootstrapping the median puts the 95% interval at **±6% for the sample size this tool produces** — so at three runs the suite cannot resolve a 4% decode change, and route tail skip costs 4–10%. The cause is in the tool's own README: "the conversation uses the model's own replies, so token counts differ a little between runs" — different text means different draft acceptance, and acceptance drives decode directly (73–90% in production). It is a load generator, not a fixed replay.

**What that means for the protocol is a split, not a verdict.** The conversation-cache and prefill experiments are prefill-shaped, so this suite is the right instrument for them and its ±0.6% is sharp enough to see them. The decode levers — pipeline windows, route tail skip, adapt-async — need a replay that resends identical text so acceptance is held constant, which is what a recorded Bench Studio session does and what `strata_bench.py` deliberately does not. Two cautions stand regardless: the short-chat decode cell spans 64 to 140 tok/s and is no signal, and the suite's `quality` subcommand computes nothing — it writes two temperature-0 answers out to be read by a human.

**One method rule, learned the hard way.** The engine under test is the engine answering the session that runs the test, so any prose generated while the suite runs is load on the measurement. A run is taken with the session silent, in one blocking call, and the log window either side is then audited for foreign requests — the silent run's window held 40 requests, all of them the suite's own, with no large one among them.

**The fixed replay is that missing instrument now, and this is what it measures.** `scripts/strata_replay.py` resends identical greedy text so draft acceptance is held constant; recorded 2026-10-14 as `data/replay-results/replay-baseline-2026-10-14-1922/`, five cells over six rounds with the first round reported apart as cold:

| cell | warm decode | 95% CI | prefill | first token | acceptance |
| --- | ---: | ---: | ---: | ---: | ---: |
| code-4k | 164.0 | [162.8, 165.1] | 871.3 | 4.60 s | 94.6% |
| tool-4k | 170.2 | [169.2, 171.7] | 881.4 | 4.58 s | 99.4% |
| code-32k | 155.7 | [153.8, 156.7] | 2,123.7 | 14.54 s | 93.0% |
| prose-32k | 159.6 | [153.9, 161.1] | 2,078.2 | 14.52 s | 98.9% |
| tool-128k | 148.1 | [144.6, 148.8] | 2,674.4 | 47.48 s | 94.8% |

**Decode is dominated by cache warmth, not by context length.** The first pass over a prompt decodes 30–60% slower than every pass after it with time-to-first-token unchanged (code-4k: 112.5 cold against 164.0 warm), so round 1 is never pooled with the rest. Warm decode here is 148–170 where the recorded `bench.sh` figure is 104–109: two regimes of one engine, and only replay-against-replay in the same regime means anything.

**Sharp inside a run, soft between runs: about 1% within one run and about 2% from one run to the next.** The same four cells moved 161.1→164.0, 168.3→170.2, 156.0→155.7 and 146.7→148.1 across two runs, so a lever has to clear roughly 3% rather than 1% — and a bootstrap interval taken inside a single run cannot show that, which is why two runs per side are the unit of comparison.

**The floor is on the cold path.** Cold decode came back at 112.5 for code-4k and 100.7 for prose-32k, so the owner's 100 tok/s floor is met with about one token per second to spare exactly where a fresh session first meets the engine. That is the argument for taking the accuracy-positive `STRATA_HC_Q8` before any lever that only moves the warm median.

**Three guards the live box taught the harness.** Cells are sized from chars-per-token measured against the served tokenizer (code 3.836, prose 4.726, JSON 2.183 — a uniform guess built a cell named 128k that was 208,561 tokens and was refused as over context); every sample records whether `/slots` was already generating, and a run taken over foreign work exits as "not a baseline" rather than as a slow run, with `--require-quiet` making idleness a precondition so an unattended run either gets its baseline or says why it did not; and a cell whose median answer is under 60% of `--max-tokens` fails, because a task that stops at 35 tokens produces a full table of plausible numbers that measure nothing.

**The first lever measured came back negative, and the instrument is the only reason it was caught.** `STRATA_PREFILL_AUTO_MAX=16384` — inherited on a list of free wins — was deployed and measured twice against the recorded suite: 4K **+0.3%**, 16K **−5.5%**, 32K **−5.9%**, with run-to-run drift of 0.06–0.16%. It was reverted. The mechanism was already visible in the boot log: the ceiling was never the limit, `auto` took 9216 tokens either way, and the larger chunk borrows more expert-cache slots to hold its buffers (6,933 against 6,281). Fewer resident experts on a box reporting `pcie_frac 0.22` costs more than the re-streaming it saves, so anything that lends experts to the prompt path has to be weighed against the experts it evicts rather than the chunk it enables. Worth noting what the old resolution would have done with this: a 5% prefill change on an instrument that repeats within 0.2% is unambiguous, and the same change against the suite's ±6% decode spread would have been filed as noise and the flag left in place.

**The second lever was also a loss, and it was caught on an axis the suite does not measure at all.** `STRATA_HC_Q8=1` reads the hyper-connection projections from the GGUF's Q8_0 bytes. It booted on all four cards and came back at mean decode **+1.1%** — under the 3% the replay resolves between runs, and negative on two of five cells — against prefill **−8.2% to −15.5%** and first token **9–18% slower**. Q8_0 bytes are wider than the quantized ones they replace, so the prompt path pays for them on every token it reads, on the box that reports `pcie_frac 0.22`. Reverted. The decisive evidence was not the speed: the replay hashes greedy answers, and they changed (the code cell stopped at 156 tokens where the baseline ran to 256), so the flag is not bit-identical and its accuracy benefit is unproven rather than established. An unproven accuracy gain does not buy a 15% prefill loss; if accuracy is to be bought, it should be bought with a measurement of accuracy.

**What is and is not reachable from configuration on this engine.** The Strata engine's `environment` map carries ten arguments -- `--family`, `--quant`, `--kv`, `--kv-streaming`, `--vision`, `--gpus`, `--layer-split`, `--low-ram`, `--host`, `--port` -- and nothing else. Everything else the running command line shows (`--spec 4`, `--mtp`, `--max-context`, `--kv-resident`, `--resident-budget-gib`) is written by the image into `config/<quant>.json`, which is regenerated at every boot, so editing it by hand is erased by the next start. That closes off the last argument that looked like a config change: `--pipeline-windows 2` passes all eight of its refusal conditions here -- `--serve`, four distinct stages, MTP drafter present with `--spec 4`, no `--batch`, no `--peer-device`, no helper caches, and `adapt_async` defaulting to 0 -- but there is no variable that reaches it, and `STRATA_PIPELINE_AUTO` cannot help, because on a non-HIP build the auto path assigns `on = false` unconditionally, so that variable can only ever turn the feature off. The consequence is worth stating plainly: **after the two flags above, every remaining performance lever on this box needs a change to the engine image, and none of them can be made from AI Runtime's configuration.**

**And the suite agrees with production, which is what makes it trustworthy and what corrected one of the levers above.** The suite's fresh 4K prompt measures 882.5 tok/s; production's fresh reads averaging 4,019 tokens measure 881 tok/s. Laying the two side by side, effective prefill throughput is strongly sublinear in the size of the read:

| Fresh read | 469 tok | 1,335 | 4,019 | 13,649 | 20,943 | 53,014 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Effective tok/s | 287 | 408 | 881 | 1,023 | 1,406 | 2,057 |

That is a 7× spread, and it is *not* a constant per-request cost — a fixed-cost-plus-rate fit over 229 fresh reads leaves 1.3 s of median residual and predicts the small buckets badly. What follows for the chunk cap is narrower than it first looked: reads below 8,192 tokens are a single chunk and the cap cannot touch them, so `--prefill auto:16384` is an experiment about the 13K–32K range only. **The lever that removes whole 25-second reads is not making prefill faster but not doing it: the conversation cache, which is off.**

**The three questions the owner asked, closed against this evidence:**

1. **A 3-bit pack has no value here, and the reason is measurable rather than theoretical.** Hit rate 98.84%, zero file reads, 96.5% of the routed mass already resident: there is no memory-system pressure for a smaller pack to relieve. And the routed experts are the small part of a decode window — upstream's breakdown puts them at 4.2 ms of 34.3 ms, with the dense Q8_0 side "about 4.8 GB in every file" regardless of quant. Upstream's yardstick against Q8_0 (4,995 teacher-forced positions) puts `UD-IQ4_XS` at 88.91% same-top-token / KL 0.101, `IQ3_S` at 87.69% / 0.144 and `IQ3_XXS` at 84.46% / 0.202. 3-bit is a pure accuracy purchase of a non-bottleneck. The accuracy direction that *is* open is the other way: `UD-Q4_K_XL` at 93.09% / 0.056, for 16 GiB more experts.
2. **128K buys nothing.** The window costs RAM (2.2 GB of `k8v4` KV at 160K) and not decode; `--kv-resident 32768` is what bounds the VRAM and it does not move with the window, which is why the measured curve is flat. Dropping the advertised window to 128K saves about 0.5 GB of RAM and nothing else.
3. **The ceiling is 262,144 and it is admissible today.** The deployed router `c3a342b` carries `RESIDENT_CONTEXT_LIMIT = 262144` with the comment "256K window of the Qwen3.8 models. The runtime controller owns per-profile qualification", the model's native `context_length` is 262,144, and `ROUTER_CONTEXT_LIMIT` in `runtime/config.py` agrees. Above it needs YaRN scaling and leaves the contract. On the measured curve the window itself is worth ~1–2 tokens per second; **what bounds it is the tail, not the ceiling** — at 100–131K actual context p10 is already 98.1.

**The levers, ranked by the headroom this machine actually shows.** All of them exist in the pinned 0.1.43 binary (checked with `--help` inside the running container) and none is enabled today:

| Change | Reach | Why it is here | What it costs |
| --- | --- | --- | --- |
| `--prefill auto:16384` / `auto:32768` (`STRATA_PREFILL_AUTO_MAX`) | profile env | only reads **above 8,192 tokens** are chunked, and those are the ones between 1,023 and 2,057 tok/s; the cap is 8,192 | borrows expert-cache slots (3.61 GiB already borrowed on CUDA0) |
| `--conversation-cache-mib N --conversation-cache-slots 4` | workspace config | 94% of prompt time re-reads text the engine has seen; parked conversations are restored instead of re-prefilled; default `0` = off | RAM, ~2.2 GB per parked 160K conversation, guarded by `--conversation-cache-min-free-mib` |
| `--pipeline-windows 2` | workspace config | the split is 5/21/11/11 layers, so the 27.5 ms window is the **sum** of four stages whose slowest is ~10 ms; upstream +14–27% on a 4-stage AMD box | 160 MiB of expert cache per card; upstream measured it **flat** on 2× RTX 3090 NVIDIA, so it is an A/B and not a gift |
| `STRATA_ROUTE_TAIL_SKIP=0` | profile env | it is **on**, and it skipped **3,957,512 of 4,116,295 missed expert entries (96.1%)**: the trial is paying accuracy for speed right now, unmeasured | 4–10% decode upstream, which lands directly on the floor |
| `STRATA_HC_Q8=1` | profile env | the pack rounds the file's Q8_0 hyper-connection projections to BF16 and doubles their bytes; reading Q8_0 improves KL 0.0562 → 0.0526 and p99 0.925 → 0.716 | none known upstream ("opt-in until decided"); verify the byte count moves |
| `--adapt-async 1` | workspace config | +6% decode measured upstream on 2× RTX 3090, prefill unchanged | answers differ slightly run to run, identical within a run |
| `--expert-profile-save P`, then `--expert-profile P` | workspace config | the shipped `/opt/strata/data/expert-profile.bin` ranks all 24,576 experts generically; the adaptive tier has 11 hours of this owner's coding traffic to save | none; the profile goes stale as the work changes |
| engine `v0.1.43` → `v0.1.44.1` | new engine entry | "dense Q8_0 matrices read one row per block where that measured faster (RDNA4 and **sm_86**) … **the same bits as before**: decode +2.8 to +3.8%" — two of the four cards are sm_86 | new image entry and digests, a fresh provisioning receipt (it is keyed on image id, so it self-invalidates), a warm boot |

**Rejected before spending a drain on them, with the reason:** `q4_0` KV (+4% at 128K, perplexity +8–12% on long documents); 3-bit (above); `--batch 2` / `--batch-groups` (each conversation at ~55% speed, which breaks the floor outright); raising `--spec-min-p` (the house measured −16–24% decode for that change on llama.cpp, acceptance up and speed down); `--kv-resident` up (streaming already hits 97% at 129K, so the cells are not what is missing).

**How these are reached is a real constraint, not a detail.** `STRATA_*` variables and the already-mapped ones go in the profile's `container.environment` and are rendered by existing code with no new mechanism. `--prefill` is mapped in neither direction, and `--pipeline-windows`, `--conversation-cache-*`, `--adapt-async` and `--expert-profile*` have **no environment variable in the entrypoint at all**: they exist only inside `ai/strata-trial/data/config/strata-unsloth-ud-iq4_xs.json`, which is the engine's own state, is what the entrypoint starts from on every fresh container, and is invisible to the controller except through the slot window it reports. `REINSTALL=1` regenerates that file from the environment and discards the edits. **A tuned argument that earns its place has to end up declared in the profile — the same way `--vram-reserve-mib` should have been — or it is one reinstall away from silently gone.** Each of them is a drain to apply: the warm boot now measures **115.3 s** and provisioning 1.6 s against a 3,600 s readiness bound, so an experiment is cheap, and the agreed circuit breaker (under 8 GiB MemAvailable, or under 150 MiB free on any of the four cards) still means switching back rather than debugging.

**What is still unmeasured, and which instrument closes it.** Every number above is speed, and the trial has never been through Bench Studio — the 27B configurations each carry "a full Bench Studio coding session (2/2 passed)" while the trial carries a catalog warning that its output "has not been attested on this quant". Three instruments, at three prices, and they are not interchangeable:

- **`bench.sh` (`tools/strata_bench.py --long`)** — the speed gate, for prefill. Three minutes, attaches to the running server, needs no drain, repeats within ±0.6% on prompt throughput and time-to-first-token. **Its decode cells repeat only within about ±6%**, so they rank nothing that trades 4–10% of decode; run it silently and audit the window, because the session driving it is served by the engine it is measuring.
- **Bench Studio** — the only scored instrument on this machine, and the one that decides whether a change is *allowed*. It is not part of Strata and not a dependency of any speed experiment; it is needed for exactly one class of decision, the ones that buy speed with accuracy, of which route tail skip (96.1% of missed experts skipped, active right now, never measured here) is the only known instance.
- **Strata's own gates, both needing a GPU window rather than a measurement slot** — `bench/bench_vs_llama.py quality` generates the same two prompts from both engines at temperature 0 and writes them out **to be read by a human; it computes no score**, so it is a cheap greedy-output diff and no more. `tools/conversation_cache_parity.py` is the real thing — a full-model A/B/A parity gate over the engine's internal state — but it loads private engines sequentially and "requires an available GPU/model-loading window", i.e. a drain, and it is dry-run by default.

The order that follows is therefore set by what each decision needs: the speed-only levers go through `bench.sh` and cost nothing but a drain each; route tail skip waits for a scored run, because there is no honest way to price 4–10% of decode against an accuracy cost nobody has measured on this box.

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
2. **DFlash 2 on Daytime 27B Q6_K in layer mode:** `qwen27b-q6k-dflash2-with-nighttime`. Retired.
   - **Drafter:** the MTP3 head is replaced by z-lab's `Qwen3.8-27B-DFlash2-GGUF` Q4_K_M (revision `2d9571f`, which includes the 2026-08-24 `dflash.rope.dimension_sections` update), mounted as `/weights/dflash2.gguf` on the RTX 3090.
   - **Depth:** `--spec-draft-n-max 5`. A published sweep peaked at 5; the block size is 8, so llama.cpp caps the depth at 7.
   - **Placement:** `--split-mode layer`. Layer mode gives up tensor parallelism, so it must recover about 30% to match the current configuration. Nighttime is unchanged.
   - **First load (2026-10-07) failed:** with `--device CUDA1,CUDA0`, the draft context aborted with `pre-allocated tensor (output.weight) in a buffer (CUDA0) that cannot run the operation`. The drafter shares the target's output layer, which llama.cpp places on the last device, the RTX 4080 SUPER, while the drafter runs on the RTX 3090.
     - **Recovery:** the backend restart-looped as a failed initializer, so automatic recovery timed out waiting for it (`needs-attention`). The documented repair restored the previous configuration in about 8 minutes, with Nighttime untouched: confirm no active work, stop that container, run `runtime recover`.
     - **Fix:** `--device CUDA0,CUDA1 --tensor-split 40,60` lists the RTX 3090 last, so it holds both the output layer and the drafter, with the same 60/40 share.
   - **Copy drafter:** not combined, because experiment 1 showed it hurts planning.
   - **Result (quick check, against tensor split + MTP3):**

     | Prompt | DFlash 2, layer split | Tensor split + MTP3 |
     | --- | --- | --- |
     | File rewrite | 110 tokens per second | 111 |
     | Short fresh answer | 57.7 | 75.5 |
     | Planning prompt, 45K, temperature 0.7 | 49.7 | 64.6 |
     | Same prompt, greedy | 54.0 | 69.3 |

     DFlash 2 accepted only 40–48% of its drafted tokens on planning text, so it did not recover what layer split loses. The owner declined a benchmark, and the configuration was retired on 2026-10-07. The drafter file stays on the host under `llm/z-lab/Qwen3.8-27B-DFlash2-GGUF/revisions/2d9571f…/`. Revisit it if llama.cpp fixes DFlash 2 under tensor split.
3. **Speculative decoding for Nighttime:** `qwen27b-q6k-with-nighttime-mtp3`. **Adopted:** at the owner's request, Nighttime itself has run this MTP3 setup in every paired configuration since 2026-10-07, under its usual alias `qwen3.8-27b-abliterated-q6_k`. The variant profile and its configuration were then retired. On a host without a vision GPU, Nighttime runs without the draft. Before this change Nighttime had no draft model, while Daytime 27B already got its speed from MTP3.
   - **Mechanism:** the configuration keeps the 27B Q6_K Daytime unchanged and names the Nighttime variant `nighttime-mtp3`. That variant is Nighttime plus the same self-contained base Qwen3.8-27B MTP3 head Daytime uses (unsloth Q4_0, with its own embeddings and output layer), keeping tensor mode.
   - **Draft placement:** the Nighttime pair has about 0.7 and 1.3 GB free, so the head runs on the shared RTX 3080 (`--spec-draft-device CUDA2`) beside both projectors. Only a profile that declares `draft_on_vision_gpu` may do that, and such a profile requires a configured vision GPU. The text model still never uses it.
   - **Context:** the first load at 128K failed: `cudaMalloc` of 564 MiB on the RTX 4080 failed while llama.cpp allocated speculative-decoding buffers. A watchdog stopped the failed initializer, and the runtime restored the previous configuration in 40 s. The variant now runs 96K, which frees about 0.9 GB of K/V across the pair.
   - **Quick check (2026-10-07, before benchmarking):** Nighttime without a draft, then with MTP3.

     | Prompt | No draft | MTP3 | Acceptance |
     | --- | --- | --- | --- |
     | File rewrite | 39.3 tokens per second | 101.8 | — |
     | Short fresh answer | 39.4 | 66.5 | — |
     | 16K planning prompt, temperature 0.7 | 37.8 | 65.7 | 57% |
     | Same prompt, greedy | 37.8 | 64.5 | 55% |

     After load the RTX 4080 has 1.06 GB free, the RTX 3080 Ti 1.5 GB, and the RTX 3080 4.2 GB in use.
   - **Switching:** the head was trained for the original model, not the abliterated one, so acceptance may be lower than Daytime's. In the browser, switching between configurations that differ only in their Nighttime variant recreates only Nighttime; an unexpected Nighttime change is still refused. DFlash 2 for Nighttime is not planned, given experiment 2.

## Service restoration: the host never rests with nothing serving

Every destructive transition on this host has to be survivable without a person, and the path that
puts the machine back has to be exercised by tests against the failures the deploy paths can
actually produce. This house is run by agents. A human in front of a drained router with no lever is
the failure mode, not the fallback.

So the runtime never comes to rest with no backend serving and no automatic action pending. Either a
healthy backend answers with admissions open, or the last-known-good release is being brought back —
now, or on a schedule that does not need anyone at the keyboard.

### What that means when a transition fails

| Situation | Behaviour |
| --- | --- |
| A backend's `/slots` cannot be fetched | It is serving nothing. It is recorded as `unreachable` and never holds a drain open. A response that is not a slot list is recorded as `malformed`, which is not evidence of work either. |
| A drain deadline passes during a restoration | The restoration continues. The router already rejects everything, so there is nobody left to be polite to; aborting there is exactly how a host ends up drained with no backend. |
| A drain deadline passes before any backend was replaced | The operation refuses and has touched nothing. A generation on a backend's own port is legitimate work, and backing away costs nothing yet. |
| A deployment fails after replacing a backend | The previous release is restored through the same finalize path a healthy boot uses: recreate, verify health and generation, republish the catalog, undrain, write the receipt. |
| That restoration fails too | The next attempt is scheduled on a capped backoff (30s, 60s, 2m, 5m, 10m, 15m) and the controller's supervision loop takes it up. Attempts continue until the service comes back. |
| The router rejects a newly published catalog | Publication restores the prior catalog before backend recovery checks drain ownership. The failed catalog must not disable the admin API that recovery uses. |
| The router admin API fails before restoration can begin | The failure advances the retry count and backoff, including failures while checking drain ownership. |

The failure is never left as a condition to notice. `.state/restoration.json` is the durable action:
the profile to bring back, how many attempts have run, the reason, `next_attempt_at`, and the phases
reached. It is cleared only once a healthy backend is serving with admissions open, and it holds off
`python3 -m runtime.recovery export` while it is outstanding. Each phase a restoration reaches is
also written into `transaction.json` under `restoration`, so what an attempt did is readable
afterwards without a journal.

### Three ways to run the same restoration

There is one implementation — the boot-time path that brought this host back on llama.cpp by itself
after the 2026-10-12 outage — and three ways to reach it:

| Way | When it is used |
| --- | --- |
| The supervision loop | Automatic. A restoration that has come due runs on its own and the page reflects it. |
| **Restore last-known-good** in the runtime page (`POST /api/restore-last-known-good`) | The one action for a machine that is not serving. It names no profile and expects nothing to have been observed first, and appears on the page whenever the runtime is not serving. |
| `docker exec local-ai-runtime python3 -m runtime restore` | The same call from a shell, including one the browser cannot reach. |

All three hold `runtime.lock` for the whole pass, refuse a router release that has reserved the
runtime, and refuse to start while another operation is running. They do not refuse a runtime that
is mid-transaction, not ready, or already reporting failures — that is the moment restoration is
needed. A drain left set by an attempt that stopped belongs to no live operation, so the
restoration takes it over and clears it.

`GET /api/status` carries the contract: `service` (`serving`, `restorable`, `pending`, `profile`)
and `restoration` (attempts, reason, next attempt, phases). Raw Docker errors stay in the private
journals; the browser is shown the shape of the failure, never its text.

### The release gates

`tests/test_restoration.py` is what keeps this honest, and it runs in CI as part of
`python3 -m unittest discover -s tests -v`:

- a target that starts but never becomes healthy → the previous configuration is restored inside the
  same call, the router is undrained, and a healthy backend is serving;
- the restoration's own first attempt fails → it is scheduled, re-entered, and succeeds with nobody
  touching anything;
- a `/slots` response of the wrong shape → a bounded wait and a recorded decision, never a hang;
- a backend that answers no port at all → treated as idle, so the drain can settle.

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

A failed initializer may sit in Docker's restart loop while recovery runs. That no longer stalls it: a backend whose `/slots` cannot be read is serving nothing, and once the router reports it is relaying no work the restoration continues rather than arguing about activity it cannot see. If a pass still fails — a protected backend has drifted, a router release has reserved the runtime, or the release cannot be brought up at all — the attempt is recorded and the next one is scheduled, and the runtime page offers the same restoration as one action. Read the failed backend's logs when a specific backend needs repair, then run the restoration; admission stays paused until verification succeeds. `needs-attention` now means an attempt failed and another is coming, not that the machine needs to be restarted.

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
