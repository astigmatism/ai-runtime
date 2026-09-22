# Optional shared vision GPU

Hosts without `vision_gpu_id` keep CPU vision and the original two-device backends. In `.state/host.json`, set `vision_gpu_id` to a full UUID outside both `gpu_ids` pairs, optionally set `vision_gpu_name`, and supply `cuda_order.daytime` and `cuda_order.nighttime` with the two text UUIDs in their existing CUDA order. Runtime appends the shared vision UUID as CUDA2 and exports that three-UUID `CUDA_VISIBLE_DEVICES` list. The four text UUIDs remain exclusive.

The existing CPU defaults in profiles remain authoritative for installations without this opt-in. With it enabled, rendering changes only projector offload, GPU visibility/reservations, and corresponding discovery metadata: `--mmproj-offload --mmproj-device CUDA2`. Text model devices, tensor overrides, KV cache, and MTP devices must remain on CUDA0/CUDA1 (or existing CPU tensor overrides). Both immutable inference engines already support this option.

Each backend loads its own projector. This is shared hardware, not a separate encoder service. CPU image preprocessing remains normal. The catalog identifies `text_gpu_uuids`, `vision_gpu_uuid`, `vision_device`, `vision_gpu_shared`, and `cuda_visible_devices`; `gpu_uuids` contains all reserved devices. Runtime status separates text and vision hardware. Live readiness checks validate the UUID order as well as the device reservation.

Before enabling on another host, measure simultaneous and repeated image requests, maximum supported inputs, retained buffers, and profile changes. Encoder file sizes do not predict peak VRAM. Independent router queues do not coordinate vision memory. If coexistence fails, restore CPU configuration; do not silently shrink contexts or images.

Use the normal immutable candidate workflow: commit source, export the commit, build with `SOURCE_REVISION`, run tests, validate the candidate, invoke candidate `deploy`, then replace the controller and persist its image. Explicit deployment establishes ownership of the new revision; never edit active revision receipts to bypass safeguards. Keep prior host configuration, image, and source for rollback. For an unpublished local release, future public fast-forward updates require that its commits first reach upstream.

To return to CPU, remove `vision_gpu_id`, `vision_gpu_name`, and `cuda_order` from host settings and run `apply` with the current controller. Runtime drains accepted work before replacing affected backends. Removing host settings alone does not change a running backend. Do not terminate active generations on drain timeout.
