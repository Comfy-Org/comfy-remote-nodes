"""Vendored copy of the parts of ``comfy_rnp_protocol`` the client needs.

The full protocol package lives in ``comfy-rnp-server/comfy_rnp_protocol``
and is the source of truth for these constants and helpers — this file
is a thin mirror so the ComfyUI extension can ship without depending on
the server package being on ``PYTHONPATH``. Keep it byte-compatible with
the upstream module.
"""
from __future__ import annotations

import base64
from typing import Any

PROTOCOL_NAME = "comfy-rnp"
PROTOCOL_VERSION = "1.0"
SCHEMA_HASH_ALGORITHM = "sha256"
DEFAULT_MAX_INLINE_PAYLOAD_BYTES = 8 * 1024 * 1024
DEFAULT_POLL_INTERVAL_S = 1.5

# Defaults for descriptor.remote.execution fields the client must honour.
# Values mirror comfy_api_nodes.util.client.poll_op_raw so RNP-routed
# nodes behave identically to a direct upstream call when the descriptor
# leaves a field unset.
#
# Lifetime budget design: the wire format declares ``soft_timeout_s``
# (server-side stuck threshold) and ``hard_timeout_s`` (server-side
# hard kill) as durations. The client's poll-loop cap is a derived
# value — ``max_poll_attempts = ceil((hard_timeout_s + DEFAULT_POLL_GRACE_S)
# / poll_interval_s)`` — not a separate wire field, so changing the
# poll cadence can never silently shrink or stretch the contract.
# Industry precedent: OAuth2 device flow §3.5, Kubernetes
# ``--timeout``, gRPC deadlines all express bounds as durations.
DEFAULT_SOFT_TIMEOUT_S = 240.0
DEFAULT_HARD_TIMEOUT_S = 300.0
DEFAULT_POLL_GRACE_S = 30.0
DEFAULT_TIMEOUT_PER_POLL_S = 120.0
DEFAULT_MAX_RETRIES_PER_POLL = 10
DEFAULT_RETRY_DELAY_PER_POLL_S = 1.0
DEFAULT_RETRY_BACKOFF_PER_POLL = 1.4
DEFAULT_CANCEL_TIMEOUT_S = 10.0
DEFAULT_SUBMIT_MAX_RETRIES = 2
DEFAULT_SUBMIT_RETRY_BACKOFF = 2.0
DEFAULT_SUBMIT_RETRY_DELAY_S = 1.0


class IdempotencyMode:
    CLIENT_KEY = "client_key"
    NONE = "none"


class ExecutionMode:
    REQUEST_RESPONSE = "request_response"
    ASYNC_POLLING = "async_polling"


class TaskStatus:
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


TERMINAL_TASK_STATES = frozenset({
    TaskStatus.DONE,
    TaskStatus.ERROR,
    TaskStatus.CANCELLED,
})


class ErrorCode:
    INPUT_INVALID = "INPUT_INVALID"
    AUTH_FAILED = "AUTH_FAILED"
    INSUFFICIENT_CREDITS = "INSUFFICIENT_CREDITS"
    RATE_LIMITED = "RATE_LIMITED"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    VERSION_INCOMPATIBLE = "VERSION_INCOMPATIBLE"
    SCHEMA_HASH_MISMATCH = "SCHEMA_HASH_MISMATCH"
    TIMEOUT = "TIMEOUT"
    INTERNAL = "INTERNAL"
    NOT_FOUND = "NOT_FOUND"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    # 404 from a poll/cancel that carried X-RNP-Idempotency-Key — the
    # server lost the task (typically a restart). The client replays
    # the original submit with the same key to spawn a fresh task.
    TASK_LOST = "TASK_LOST"
    # Server-side back-pressure / operator drain — both ride 503 with
    # Retry-After. SERVER_BUSY = transient saturation, MAINTENANCE =
    # operator action.
    SERVER_BUSY = "SERVER_BUSY"
    MAINTENANCE = "MAINTENANCE"


class Header:
    PROTOCOL_VERSION = "X-RNP-Protocol-Version"
    CLIENT_VERSION = "X-RNP-Client-Version"
    CLIENT_CAPABILITIES = "X-RNP-Client-Capabilities"
    SCHEMA_HASH = "X-RNP-Schema-Hash"
    IDEMPOTENCY_KEY = "X-RNP-Idempotency-Key"


class Capability:
    IMAGE_PNG_BASE64 = "image:png_base64"
    IMAGE_JPEG_BASE64 = "image:jpeg_base64"
    IMAGE_RAW_CHW_UINT8 = "image:raw_b64_chw_uint8"
    # Multi-frame IMAGE envelope: a single value carries N PNG-base64
    # frames sharing the same H/W/C, decoded into a [B,H,W,C] tensor on
    # this client. Mirrors the AUDIO sweep precedent (parallel encodings
    # rather than extending ``png_base64``) so legacy server emits still
    # decode correctly when the new token isn't advertised.
    IMAGE_PNG_BASE64_BATCH = "image:png_base64_batch"
    # Externalised counterpart of ``image:png_base64_batch``: same
    # multi-frame IMAGE envelope shape, but every frame's PNG bytes
    # are uploaded out-of-band via the presigned-PUT path (see
    # ``serialization.maybe_externalize``) and the envelope carries
    # a parallel ``uris`` list instead of inline ``frames``. Required
    # because the original ``maybe_externalize`` only inspected the
    # singleton ``data`` field, so a batch of N PNGs whose summed
    # base64 payload crossed the 8 MiB inline cap silently rode the
    # wire as a 30+ MiB POST body. Per-frame URIs (NOT a tar/zip
    # blob) keep this composable with the existing per-frame fetch
    # helpers and let the server fetch in parallel. Decoder dispatch
    # keys off the ``encoding`` string — ``frames`` vs ``uris`` is a
    # payload field, not a shape selector.
    IMAGE_PNG_BASE64_BATCH_URI = "image:png_base64_batch_uri"
    VIDEO_MP4_INLINE = "video:mp4_inline"
    VIDEO_MP4_BASE64 = "video:mp4_base64"
    AUDIO_MP3_INLINE = "audio:mp3_inline"
    AUDIO_MP3_BASE64 = "audio:mp3_base64"
    AUDIO_WAV_BASE64 = "audio:wav_base64"
    AUDIO_WAV_INLINE = "audio:wav_inline"
    # OGG/Opus container as base64 in ``data``. Mirrors the parallel-
    # encoding negotiation pattern already used for the AUDIO sweep
    # (mp3 vs. wav) — servers only emit ``audio:opus_base64`` when
    # this client advertises the token, otherwise they fall back to
    # ``mp3_base64``. PyAV/FFmpeg decodes OGG/Opus natively so
    # ``decode_audio_envelope`` ships the bytes through
    # ``audio_bytes_to_audio_input`` the same way it ships MP3/WAV.
    AUDIO_OPUS_BASE64 = "audio:opus_base64"
    MASK_PNG_BASE64 = "mask:png_base64"
    SVG_XML_BASE64 = "svg:svg_xml_base64"
    SCHEMA_V3 = "schema:v3"
    ASYNC_EXECUTE = "execute:async"
    # Pass-through custom-IO bucket: when an input/output type string
    # is not in the standard primitive/media set, the proxy_node parser
    # falls back to ``IO.Custom(<type>)`` so partner helper-config nodes
    # (OpenAIInputFiles, OpenAIChatConfig, GeminiInputFiles, Recraft
    # style/color/controls helpers, future partner helper-config types)
    # round-trip without per-type encoder plumbing. The descriptor's
    # original io_type string is preserved on the V3 input so node-to-
    # node connections only chain when the strings match. Encoder
    # dispatch passes opaque dict/list/string values through
    # ``_encode_one`` unchanged — they're already JSON-serialisable.
    IO_OPAQUE = "io:opaque"
    # 3D MODEL envelope: a single-mesh GLB (binary glTF 2.0, magic
    # b"glTF") wrapped opaquely in ``data`` (base64). Mirrors the
    # parallel-encoding pattern already used for VIDEO_MP4_INLINE — the
    # server emits this token only when the inbound request advertised
    # it; legacy clients without the token will surface
    # NEGOTIATION_FAILED at descriptor-load time. The decoder on the
    # client side hands raw GLB bytes straight to ComfyUI's
    # ``Types.File3D`` / ``IO.File3DGLB`` machinery (no in-server GLB
    # parse — the magic-byte check at offset 0 is the only validity
    # guard the server applies).
    MODEL_3D_GLB_INLINE = "model_3d:glb_inline"
    # 3D MODEL BUNDLE envelope: a primary mesh file plus N companion
    # files (textures / .mtl / .bin / other PBR maps) carried inline as
    # a single coherent unit. Wire shape:
    #
    #   {
    #     "type":         "model_3d",
    #     "encoding":     "bundle_inline",
    #     "format":       "obj",
    #     "primary_path": "model.obj",
    #     "files": [
    #       {"path": "model.obj",            "format": "obj", "role": "mesh",
    #        "data": "<b64>", "byte_size": 12345},
    #       {"path": "model.mtl",            "format": "mtl", "role": "material",
    #        "data": "<b64>"},
    #       {"path": "textures/albedo.png",  "format": "png", "role": "texture_diffuse",
    #        "data": "<b64>"}
    #     ]
    #   }
    #
    # ``path`` is the authoritative cross-file reference key — a POSIX-
    # relative path (no ``..``, no absolute prefix, no Windows drive
    # letter, no backslashes, no case-insensitive collisions). The
    # client decoder materialises files into a temp directory using
    # exactly these paths so a ``.obj`` that references its ``.mtl``
    # via filename (or a ``.gltf`` that references its ``.bin`` /
    # textures via relative URI) resolves on disk.
    #
    # ``role`` is an OPTIONAL producer hint (NOT a closed enum) —
    # well-known values include ``mesh``, ``material``, ``buffer``,
    # ``texture_diffuse``, ``texture_metallic``, ``texture_normal``,
    # ``texture_roughness``, ``texture_ao``, ``texture_emissive``,
    # ``texture_height``. Unknown roles round-trip without error;
    # consumers may use ``role`` for convenience but ``path`` is the
    # primary identifier.
    #
    # ``primary_path`` MUST appear in ``files`` exactly once and pins
    # the entry-point file (the one passed to ComfyUI's
    # ``Types.File3D`` constructor by the client decoder). The
    # top-level ``format`` (if set) MUST match that primary file's
    # ``format``.
    #
    # The client decoder returns a ``BundledFile3D`` (a thin
    # ``File3D``-compatible wrapper) — existing downstream loaders
    # that call ``get_source()`` / ``get_data()`` / ``save_to()`` get
    # the bundle's primary file with all sidecars materialised on
    # disk first so cross-file refs resolve correctly. Override of
    # ``save_to()`` copies the *whole bundle* into the destination
    # directory, not just the primary file.
    #
    # Negotiation: the server emits this encoding ONLY when the
    # inbound request advertised this token; legacy clients without
    # the bundle decoder will surface NEGOTIATION_FAILED at
    # descriptor-load time. Whole-descriptor gating — if a node has
    # both bundle and non-bundle outputs and the cap is absent, the
    # whole descriptor is skipped rather than synthesising a
    # partial-socket variant.
    #
    # A future sibling ``Capability.MODEL_3D_BUNDLE_URI`` would carry
    # ``uri`` per file for URL-fetch-out-of-band semantics; not in
    # this PR.
    MODEL_3D_BUNDLE_INLINE = "model_3d:bundle_inline"


HEAVY_TYPES = frozenset({"image", "video", "audio", "mask", "model_3d"})


def is_envelope(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("type"), str)
        and value.get("type") in HEAVY_TYPES
        and isinstance(value.get("encoding"), str)
    )


def decode_envelope_data(envelope: dict[str, Any]) -> bytes:
    """Decode the inline base64 ``data`` of an envelope.

    URI envelopes are resolved out-of-band by the caller (see
    ``proxy_node._deserialize_output``); by the time they reach this
    function the bytes have already been fetched and re-stamped onto
    ``data``.
    """
    if "data" not in envelope:
        raise ValueError(
            "envelope has no inline 'data' — caller must resolve "
            f"{sorted(k for k in envelope.keys() if k != 'data')!r} first"
        )
    return base64.b64decode(envelope["data"])


class RnpProtocolError(Exception):
    """Raised when a server response can't be parsed as RNP/1, or when
    a structured-error response is returned. Carries the parsed error
    envelope so the caller can surface ``user_facing`` messages.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = ErrorCode.INTERNAL,
        status: int | None = None,
        user_facing: bool = False,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.user_facing = user_facing
        self.retryable = retryable
        self.details = details

    @classmethod
    def from_response_body(
        cls, body: dict[str, Any] | None, status: int
    ) -> "RnpProtocolError":
        err = (body or {}).get("error") if isinstance(body, dict) else None
        if not isinstance(err, dict):
            return cls(
                f"Unstructured error response (HTTP {status})",
                code=ErrorCode.INTERNAL,
                status=status,
            )
        return cls(
            err.get("message") or f"HTTP {status}",
            code=err.get("code") or ErrorCode.INTERNAL,
            status=status,
            user_facing=bool(err.get("user_facing")),
            retryable=bool(err.get("retryable")),
            details=err.get("details") if isinstance(err.get("details"), dict) else None,
        )
