# comfy-remote-nodes

ComfyUI custom-node package that fetches node descriptors from an
**RNP/1** (Remote Node Protocol v1) server and exposes them as native
V3 nodes inside ComfyUI.

The reference server lives at
[`Comfy-Org/comfy-rnp-server`](https://github.com/Comfy-Org/comfy-rnp-server)
(private). This client is intentionally provider-agnostic: any server
that speaks RNP/1 can publish nodes through it.

## Install

Clone into your ComfyUI `custom_nodes/` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Comfy-Org/comfy-remote-nodes.git
```

(The directory name on disk doesn't matter — ComfyUI loads the package
via `spec_from_file_location`, so `comfy-remote-nodes/` is fine.)

## Configure

Point the extension at one RNP server URL via env var before starting
ComfyUI:

```bash
export RNP_SERVER_URL=http://127.0.0.1:9190
python main.py
```

If `RNP_SERVER_URL` is unset, the extension registers zero nodes and
logs a single line — ComfyUI starts normally with no remote nodes.

## What works today

* Manifest + capability handshake
* `object_info` fetch and dynamic V3 class generation (with ETag
  conditional GETs)
* Sync `execute` and async `execute_async` + poll/cancel
* Image / mask / audio / video envelope encode + decode
* Presigned upload helper for >8 MiB inputs
* **Tier 3** (recent):
  * `TASK_LOST` detection with idempotent resubmission — survives an
    RNP server restart mid-task without losing the workflow
  * `SERVER_BUSY` / `MAINTENANCE` 503 envelopes treated as rate
    limits with `Retry-After` honoring and friendly per-second
    progress text ("Server busy, retrying in 30s...")
  * Out-of-band `url_fetch` for URI-only output envelopes, driven by
    the descriptor's `remote.url_fetch` policy
  * Per-poll auth refresh so long-running tasks can rotate tokens

## What does not (yet)

* `validate_inputs` / `fingerprint_inputs` / `check_lazy_status` —
  the server returns `501 NOT_IMPLEMENTED`; the client falls through
  to ComfyUI's defaults
* Multi-server config file (`comfy_remote_nodes.yaml` or similar) —
  today's setup is one server per ComfyUI process via env var
* Streaming execute (server-sent events for incremental previews)
