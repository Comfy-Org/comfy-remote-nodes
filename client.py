"""HTTP client for the RNP/1 server.

Builds on ``comfy_api_nodes.util.client.sync_op_raw`` so RNP requests
get the same retry / cancellation / friendly-error treatment as local
partner-node calls. Adds the RNP-specific bits ``sync_op_raw`` doesn't
know about: protocol/capability headers, ``If-None-Match`` 304 handling
on ``/object_info``, and the two-phase upload helper (the GCS PUT runs
on plain aiohttp because presigned URLs bypass api.comfy.org).
"""
from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import urljoin

import aiohttp

from comfy_api_nodes.util.client import ApiEndpoint, sync_op_raw

from .protocol import (
    Capability,
    Header,
    PROTOCOL_VERSION,
    RnpProtocolError,
)

log = logging.getLogger("comfy_remote_nodes.client")


# Capability tokens advertised on every outbound request. Currently
# covers the input/output encodings the registered providers (LTXV,
# Gemini, Ideogram, ...) need, plus the ``schema:v3`` marker so
# descriptors come back in the V3 ``get_v1_info()`` shape. Servers use
# this list to negotiate per-request: e.g. a provider only emits the
# multi-frame ``image:png_base64_batch`` envelope when this list
# contains :data:`Capability.IMAGE_PNG_BASE64_BATCH`, otherwise it
# falls back to the single-frame ``image:png_base64`` shape (with a
# server-side truncation warning logged when extras are dropped).
CLIENT_CAPABILITIES = [
    Capability.SCHEMA_V3,
    Capability.VIDEO_MP4_INLINE,
    Capability.VIDEO_MP4_BASE64,
    Capability.IMAGE_PNG_BASE64,
    Capability.IMAGE_PNG_BASE64_BATCH,
    # Advertises that this client can emit a multi-frame IMAGE
    # envelope with per-frame presigned-GET URIs instead of inline
    # base64 frames when the summed batch payload would overflow the
    # 8 MiB inline cap (see ``serialization.maybe_externalize``).
    # Opportunistic: the client falls back to it whenever a batch
    # envelope it produced exceeds the cap, so descriptors don't
    # need to publish it in their ``input_serialization`` to
    # benefit — any input that already accepts ``png_base64_batch``
    # transparently accepts the externalised variant on the server
    # via the shared ``resolve_image_envelope_pngs`` dispatch.
    Capability.IMAGE_PNG_BASE64_BATCH_URI,
    Capability.MASK_PNG_BASE64,
    Capability.AUDIO_MP3_INLINE,
    Capability.AUDIO_MP3_BASE64,
    Capability.AUDIO_WAV_BASE64,
    Capability.AUDIO_WAV_INLINE,
    # OGG/Opus decode path. PyAV / FFmpeg handle OGG/Opus natively in
    # ``audio_bytes_to_audio_input`` so the proxy_node side is the
    # same as MP3/WAV — advertising the token unlocks per-provider
    # ``output_format`` choices that emit Opus (e.g. ElevenLabs'
    # ``opus_48000_192``) without requiring servers to label-launder
    # the upstream container as ``mp3_base64``.
    Capability.AUDIO_OPUS_BASE64,
    # Advertises that this client can drive the
    # ``execute_async`` + poll-loop lifecycle (submit, poll, cancel,
    # TASK_LOST replay). Servers gate ``async_polling``-mode
    # descriptors on the presence of this token so an older client
    # that only knows ``/execute`` doesn't silently submit a
    # long-poll node into the sync endpoint.
    Capability.ASYNC_EXECUTE,
    # Generic opaque pass-through bucket for partner helper-config
    # custom IO types (OPENAI_INPUT_FILES, OPENAI_CHAT_CONFIG,
    # GEMINI_INPUT_FILES, RECRAFT_*, future partner helper types).
    # Servers may gate descriptors that publish ``input_serialization``
    # / ``output_serialization`` entries with the ``"opaque"`` marker
    # on the presence of this token; legacy clients that don't know
    # about IO.Custom fallthrough get filtered out before they try to
    # parse a descriptor whose io_type strings they wouldn't recognise.
    Capability.IO_OPAQUE,
    # 3D MODEL envelope (binary glTF 2.0 — single-mesh GLB). The
    # decoder in ``serialization.decode_model3d_envelope`` hands raw
    # GLB bytes to ``comfy_api.latest._util.File3D`` so downstream
    # nodes (preview3d / SaveGLB / Load3D) see the same value shape
    # they would from a local Tripo / Rodin / Meshy node. Servers
    # gate ``MODEL_3D`` provider descriptors on this token so a
    # legacy client without the decoder surfaces NEGOTIATION_FAILED
    # at descriptor-load time rather than crashing at execute.
    Capability.MODEL_3D_GLB_INLINE,
    # 3D MODEL BUNDLE envelope (multi-file: primary mesh plus N
    # companion files — textures / .mtl / .bin / etc. — as a single
    # coherent unit). The decoder in
    # ``serialization.decode_model3d_envelope`` dispatches on
    # ``encoding="bundle_inline"`` and returns a ``BundledFile3D``
    # (a ``File3D`` subclass) whose ``get_source()`` / ``save_to()``
    # materialise the whole bundle on disk so cross-file refs (.obj
    # -> .mtl -> textures; .gltf -> .bin + textures) resolve via
    # the filesystem. Servers gate bundle-emitting descriptors on
    # this token; legacy clients surface NEGOTIATION_FAILED at
    # descriptor-load time rather than crashing at execute.
    Capability.MODEL_3D_BUNDLE_INLINE,
]

# Reported in the X-RNP-Client-Version header.
CLIENT_VERSION = "0.2.0"


def _default_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {
        Header.PROTOCOL_VERSION: PROTOCOL_VERSION,
        Header.CLIENT_VERSION: CLIENT_VERSION,
        Header.CLIENT_CAPABILITIES: json.dumps(CLIENT_CAPABILITIES),
    }
    if extra:
        headers.update(extra)
    return headers


async def fetch_manifest(server_url: str, *, timeout_s: float = 30.0) -> dict[str, Any]:
    """``GET /rnp/v1/manifest`` — protocol + capability handshake."""
    body = await sync_op_raw(
        None,
        ApiEndpoint("rnp/v1/manifest", "GET", headers=_default_headers()),
        timeout=timeout_s,
        as_binary=False,
        max_retries=2,
        wait_label="Fetching manifest",
        final_label_on_success=None,
        monitor_progress=False,
        base_url=server_url,
        auth_headers={},
        error_parser=_parse_rnp_error,
        is_rate_limited=_is_rnp_backpressure,
    )
    if not isinstance(body, dict):
        raise RnpProtocolError(
            f"Expected JSON manifest, got {type(body).__name__}",
        )
    return body


async def fetch_object_info(
    server_url: str,
    *,
    timeout_s: float = 30.0,
    etag: str | None = None,
) -> tuple[dict[str, Any] | None, str | None, int | None]:
    """``GET /rnp/v1/object_info`` — node descriptor map.

    Returns ``(body, etag, max_age_s)``. When the server replies ``304
    Not Modified`` (``If-None-Match`` matched), ``body`` is ``None`` so
    the caller can keep its cached descriptor map. ``max_age_s`` is the
    Cache-Control hint (or None if absent).
    """
    headers = _default_headers()
    if etag:
        headers["If-None-Match"] = etag

    captured: dict[str, Any] = {"etag": None, "max_age": None}

    def _capture(resp_headers: dict[str, str]) -> None:
        captured["etag"] = resp_headers.get("etag")
        captured["max_age"] = _parse_max_age(resp_headers.get("cache-control") or "")

    body = await sync_op_raw(
        None,
        ApiEndpoint("rnp/v1/object_info", "GET", headers=headers),
        timeout=timeout_s,
        as_binary=False,
        max_retries=2,
        wait_label="Fetching object_info",
        final_label_on_success=None,
        monitor_progress=False,
        base_url=server_url,
        auth_headers={},
        allow_304=True,
        response_header_validator=_capture,
        error_parser=_parse_rnp_error,
        is_rate_limited=_is_rnp_backpressure,
    )
    if body is None:  # 304 Not Modified
        return None, captured["etag"], captured["max_age"]
    if not isinstance(body, dict):
        raise RnpProtocolError(
            f"Expected JSON object_info, got {type(body).__name__}",
        )
    return body, captured["etag"], captured["max_age"]


def _parse_max_age(cache_control: str) -> int | None:
    """Extract ``max-age=<n>`` from a Cache-Control header.

    Returns ``None`` when the directive is missing or malformed; the
    caller treats that as 'no caching hint'.
    """
    for part in cache_control.split(","):
        part = part.strip().lower()
        if part.startswith("max-age="):
            try:
                return int(part.split("=", 1)[1])
            except (IndexError, ValueError):
                return None
    return None


async def upload_asset(
    server_url: str,
    data: bytes,
    *,
    content_type: str | None = None,
    timeout_s: float = 120.0,
    auth_headers: dict[str, str] | None = None,
) -> str:
    """Stage ``data`` on the RNP server and return its download URL.

    Two-phase: ``POST /rnp/v1/uploads`` to allocate the slot and get
    back ``{upload_url, download_url}``, then ``PUT upload_url`` with
    the raw bytes. ``auth_headers`` are forwarded on the allocate call
    so the server can delegate to api.comfy.org/customers/storage with
    the user's credentials. The PUT runs on plain aiohttp because the
    presigned URL bypasses api.comfy.org.
    """
    meta_body = await sync_op_raw(
        None,
        ApiEndpoint(
            "rnp/v1/uploads", "POST",
            headers=_default_headers(),
        ),
        data={"content_type": content_type or "application/octet-stream"},
        timeout=timeout_s,
        as_binary=False,
        max_retries=2,
        wait_label="Allocating upload slot",
        final_label_on_success=None,
        monitor_progress=False,
        base_url=server_url,
        auth_headers=auth_headers or {},
        error_parser=_parse_rnp_error,
        is_rate_limited=_is_rnp_backpressure,
    )
    if not isinstance(meta_body, dict):
        raise RnpProtocolError(
            f"Expected JSON upload response, got {type(meta_body).__name__}",
        )
    upload_url = meta_body.get("upload_url")
    download_url = meta_body.get("download_url")
    if not isinstance(upload_url, str) or not isinstance(download_url, str):
        raise RnpProtocolError(
            f"upload response missing upload_url/download_url: {meta_body!r}",
        )

    put_headers = {"Content-Type": content_type or "application/octet-stream"}
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.put(upload_url, data=data, headers=put_headers) as put_resp:
            if put_resp.status >= 400:
                raise RnpProtocolError(
                    f"PUT to upload_url failed: HTTP {put_resp.status}",
                    status=put_resp.status,
                )
    return download_url


async def execute_remote(
    server_url: str,
    schema_hash: str | None,
    endpoint_path: str,
    inputs: dict[str, Any],
    context: dict[str, Any],
    *,
    timeout_s: float = 600.0,
    extra_headers: dict[str, str] | None = None,
    node_cls: type | None = None,
    rate_limit_label: Any = None,
) -> dict[str, Any]:
    """``POST /rnp/v1/nodes/<id>/execute`` — run the remote node.

    Returns the parsed JSON body (typically ``{"outputs": [...]}``).
    Raises ``RnpProtocolError`` on non-2xx responses with the
    structured-error fields populated. ``node_cls`` is forwarded to
    ``sync_op_raw`` so progress/interrupt checks key off the right
    node id; ``None`` is fine for callers without a workflow node.
    """
    headers = _default_headers()
    if schema_hash:
        headers[Header.SCHEMA_HASH] = schema_hash
    if extra_headers:
        headers.update(extra_headers)
    body = await sync_op_raw(
        node_cls,
        ApiEndpoint(endpoint_path.lstrip("/"), "POST", headers=headers),
        data={"inputs": inputs, "context": context},
        timeout=timeout_s,
        as_binary=False,
        max_retries=2,
        wait_label="Processing",
        final_label_on_success=None,
        monitor_progress=False,
        base_url=server_url,
        auth_headers=extra_headers or {},
        error_parser=_parse_rnp_error,
        is_rate_limited=_is_rnp_backpressure,
        rate_limit_label=rate_limit_label or _rnp_rate_limit_label,
    )
    if not isinstance(body, dict):
        raise RnpProtocolError(
            f"Expected JSON execute response, got {type(body).__name__}",
        )
    return body


async def execute_remote_async(
    server_url: str,
    schema_hash: str | None,
    node_id: str,
    inputs: dict[str, Any],
    context: dict[str, Any],
    *,
    timeout_s: float = 60.0,
    extra_headers: dict[str, str] | None = None,
    idempotency_key: str | None = None,
    node_cls: type | None = None,
    rate_limit_label: Any = None,
) -> dict[str, Any]:
    """``POST /rnp/v1/nodes/<node_id>/execute_async`` — kick off the
    background runner.

    Returns ``{"task_id": "...", "poll_interval": <float>}``. The caller
    drives ``poll_op`` until a terminal status. ``idempotency_key`` is
    forwarded as ``X-RNP-Idempotency-Key`` so a network-retry of this
    POST returns the same task_id rather than spawning a duplicate run.
    """
    headers = _default_headers()
    if schema_hash:
        headers[Header.SCHEMA_HASH] = schema_hash
    if idempotency_key:
        headers[Header.IDEMPOTENCY_KEY] = idempotency_key
    if extra_headers:
        headers.update(extra_headers)
    body = await sync_op_raw(
        node_cls,
        ApiEndpoint(
            f"rnp/v1/nodes/{node_id}/execute_async", "POST", headers=headers,
        ),
        data={"inputs": inputs, "context": context},
        timeout=timeout_s,
        as_binary=False,
        max_retries=2,
        wait_label="Submitting",
        final_label_on_success=None,
        monitor_progress=False,
        base_url=server_url,
        auth_headers=extra_headers or {},
        error_parser=_parse_rnp_error,
        is_rate_limited=_is_rnp_backpressure,
    )
    if not isinstance(body, dict) or "task_id" not in body:
        raise RnpProtocolError(
            f"execute_async response missing task_id: {body!r}",
        )
    return body


async def poll_op(
    server_url: str,
    task_id: str,
    *,
    timeout_s: float = 30.0,
    node_cls: type | None = None,
) -> dict[str, Any]:
    """``GET /rnp/v1/tasks/<task_id>`` — single poll-response read.

    Returns the canonical poll-response dict (see
    ``comfy_rnp_protocol.tasks``). Caller schedules the loop and the
    sleep between polls.
    """
    body = await sync_op_raw(
        node_cls,
        ApiEndpoint(
            f"rnp/v1/tasks/{task_id}", "GET", headers=_default_headers(),
        ),
        timeout=timeout_s,
        as_binary=False,
        max_retries=2,
        wait_label=None,
        final_label_on_success=None,
        monitor_progress=False,
        base_url=server_url,
        auth_headers={},
        error_parser=_parse_rnp_error,
        is_rate_limited=_is_rnp_backpressure,
    )
    if not isinstance(body, dict) or "status" not in body:
        raise RnpProtocolError(
            f"poll response missing status: {body!r}",
        )
    return body


async def cancel_op(
    server_url: str,
    task_id: str,
    *,
    timeout_s: float = 10.0,
) -> None:
    """``POST /rnp/v1/tasks/<task_id>/cancel`` — best-effort cancel.

    Failures are swallowed: if the task already finished or the server
    is unreachable, the user has already pressed cancel and we don't
    want to surface a second exception over their original interrupt.
    """
    try:
        await sync_op_raw(
            None,
            ApiEndpoint(
                f"rnp/v1/tasks/{task_id}/cancel", "POST",
                headers=_default_headers(),
            ),
            timeout=timeout_s,
            as_binary=False,
            max_retries=0,
            wait_label=None,
            final_label_on_success=None,
            monitor_progress=False,
            base_url=server_url,
            auth_headers={},
            error_parser=_parse_rnp_error,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("RNP cancel %s failed (best-effort): %s", task_id, e)


def _parse_rnp_error(status: int, body: Any) -> Exception | None:
    """``sync_op_raw`` error_parser hook.

    Surfaces structured RNP error envelopes (``{"error": {...}}``) as
    ``RnpProtocolError`` so the proxy_node can map ``code`` /
    ``details.upstream_status`` to the literal frontend dialog strings.
    Returns ``None`` for unstructured 4xx/5xx so the default
    ``_friendly_http_message`` path keeps firing — that already
    produces the magic 401/402 strings for missing-auth / no-credits
    failures the server reflects with the real upstream status.
    """
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        return RnpProtocolError.from_response_body(body, status)
    return None


# Wire codes the server emits when it wants the client to wait and
# retry (operator drain / capacity ceiling). Treated identically to a
# 429 by ``sync_op_raw`` so they consume the rate-limit retry counter
# and honor ``Retry-After`` instead of falling through the generic
# 5xx exponential-backoff path.
_BACKPRESSURE_CODES = frozenset({"SERVER_BUSY", "MAINTENANCE"})


def _is_rnp_backpressure(status: int, body: Any) -> bool:
    """``sync_op_raw`` is_rate_limited hook for SERVER_BUSY / MAINTENANCE.

    Both arrive as 503 with a structured RNP error envelope; promoting
    them to "rate limited" makes the client honor ``Retry-After`` and
    use ``max_retries_on_rate_limit`` instead of the generic 5xx
    counter, matching the spec's "the server is asking you to wait"
    semantics.
    """
    if status != 503 or not isinstance(body, dict):
        return False
    err = body.get("error")
    if not isinstance(err, dict):
        return False
    return err.get("code") in _BACKPRESSURE_CODES


def _rnp_rate_limit_label(status: int, body: Any, retry_after_s: float) -> str | None:
    """``sync_op_raw`` rate_limit_label hook for SERVER_BUSY / MAINTENANCE.

    Returns the per-second progress label shown while waiting out a
    backpressure 503 — distinct copy for capacity vs. operator drain so
    users can tell why the workflow is stalled. ``None`` lets the
    default ``cfg.wait_label`` ("Waiting for server") stand for plain
    429s and any 503 the RNP server didn't tag with a known code.
    """
    secs = max(1, int(round(retry_after_s)))
    if isinstance(body, dict):
        err = body.get("error") if isinstance(body.get("error"), dict) else None
        code = err.get("code") if err else None
        if code == "SERVER_BUSY":
            return f"Server busy, retrying in {secs}s..."
        if code == "MAINTENANCE":
            return f"Server in maintenance, back in {secs}s..."
    if status == 429:
        return f"Rate limited, retrying in {secs}s..."
    return None
