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
DEFAULT_MAX_POLL_ATTEMPTS = 160
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
    VIDEO_MP4_INLINE = "video:mp4_inline"
    AUDIO_MP3_INLINE = "audio:mp3_inline"
    MASK_PNG_BASE64 = "mask:png_base64"
    SCHEMA_V3 = "schema:v3"
    ASYNC_EXECUTE = "execute:async"


HEAVY_TYPES = frozenset({"image", "video", "audio", "mask"})


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
