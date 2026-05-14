"""Tensor <-> RNP value-envelope helpers.

The proxy node receives raw ComfyUI Python values from the workflow
runtime (image tensors, mask tensors, audio dicts, Video objects) and
must turn them into wire-format envelopes before POSTing to the RNP
server. Conversely, each ``execute`` response returns envelopes that
must be deserialized into the native ComfyUI types so downstream nodes
in the workflow get exactly what they would have from a local node.

This module is the single home for that round-trip, so the proxy_node
only ever sees ``encode_*`` / ``decode_*`` pairs and stays focused on
schema construction and HTTP dispatch.

Byte-level encodings reuse ``comfy_api_nodes.util.conversions`` so a
remote node and a local one produce identical bytes for the same
tensor — no parallel dialect.
"""
from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import Any

from . import client as rnp_client
from .protocol import (
    Capability,
    ErrorCode,
    RnpProtocolError,
    decode_envelope_data,
    is_envelope,
)

log = logging.getLogger("comfy_remote_nodes.serialization")


# ---------------------------------------------------------------------------
# Externalization (inline → presigned-upload swap)
# ---------------------------------------------------------------------------

# Per-type MIME content-type used when uploading bytes that an envelope
# would otherwise carry inline. The server stamps these on the asset
# record so subsequent fetches arrive with the right Content-Type.
_CONTENT_TYPE_FOR_TYPE = {
    "image": "image/png",
    "mask":  "image/png",
    "video": "video/mp4",
    "audio": "audio/mpeg",
}


async def maybe_externalize(
    envelope: dict[str, Any],
    *,
    server_url: str | None,
    max_inline_bytes: int | None,
    auth_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """If the envelope's inline data exceeds ``max_inline_bytes``,
    upload the bytes and replace ``data`` with ``uri``.

    No-op when either argument is falsy, the envelope already has a
    ``uri``, or the inline payload fits under the cap.
    """
    if not server_url or not max_inline_bytes:
        return envelope
    data = envelope.get("data")
    if not isinstance(data, str):
        return envelope
    # Cap is on raw bytes, not the base64 expansion.
    if len(data) * 3 // 4 <= max_inline_bytes:
        return envelope
    raw = base64.b64decode(data)
    content_type = _CONTENT_TYPE_FOR_TYPE.get(envelope.get("type", ""), "application/octet-stream")
    download_url = await rnp_client.upload_asset(
        server_url, raw, content_type=content_type, auth_headers=auth_headers,
    )
    log.info(
        "RNP: externalized %s envelope (%d bytes) -> %s",
        envelope.get("type"), len(raw), download_url,
    )
    out = {k: v for k, v in envelope.items() if k != "data"}
    out["uri"] = download_url
    return out


# ---------------------------------------------------------------------------
# Image encode / decode
# ---------------------------------------------------------------------------

def encode_image_tensor(tensor: Any) -> dict[str, Any]:
    """Encode a ComfyUI IMAGE tensor (B,H,W,C float32 in 0..1) as a
    PNG-base64 image envelope.

    Multi-image batches send only the first frame for now; the
    framework hasn't grown batch-of-envelopes semantics yet.
    """
    import numpy as np
    from PIL import Image

    # Strip the batch dim if present.
    if hasattr(tensor, "dim") and tensor.dim() == 4:
        frame = tensor[0]
    else:
        frame = tensor
    arr = (frame.cpu().clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)
    height, width = arr.shape[:2]
    channels = arr.shape[2] if arr.ndim == 3 else 1
    pil_mode = {1: "L", 3: "RGB", 4: "RGBA"}.get(channels, "RGB")
    img = Image.fromarray(arr, mode=pil_mode)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return {
        "type":        "image",
        "encoding":    "png_base64",
        "data":        base64.b64encode(buf.getvalue()).decode("ascii"),
        "shape":       [1, height, width, channels],
        "dtype":       "uint8",
        "color_space": "srgb",
    }


def decode_image_envelope(envelope: dict[str, Any]) -> Any:
    """Decode an image envelope into a ComfyUI IMAGE tensor.

    Single-frame encodings (``png_base64``, ``jpeg_base64``,
    ``webp_base64``) decode into ``[1, H, W, C]``. The multi-frame
    ``png_base64_batch`` encoding (a list of base64-PNG ``frames``)
    stacks each frame into ``[B, H, W, C]`` — the native ComfyUI
    image-batch shape — so providers whose upstream returns ``n > 1``
    surface as a real batched tensor in the workflow runtime.
    """
    encoding = envelope.get("encoding")
    if encoding == "png_base64_batch":
        return _decode_image_batch_envelope(envelope)
    if encoding not in ("png_base64", "jpeg_base64", "webp_base64"):
        raise RnpProtocolError(
            f"Unsupported image encoding: {encoding!r}",
            code=ErrorCode.INTERNAL,
        )
    import numpy as np
    import torch
    from PIL import Image

    raw = decode_envelope_data(envelope)
    # Server emits PNG (sRGB); RGB is enough for IMAGE — RGBA path
    # would force downstream models to receive an alpha channel they
    # don't expect.
    img = Image.open(BytesIO(raw)).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _decode_image_batch_envelope(envelope: dict[str, Any]) -> Any:
    """Decode a ``png_base64_batch`` envelope into ``[B, H, W, C]``.

    The wire shape is ``{"type":"image", "encoding":"png_base64_batch",
    "frames":[<png_b64>, ...], "shape":[B,H,W,C], ...}``. Each frame is
    a PNG-encoded image; all frames must share the same H/W after the
    RGB conversion (the server-side encoder enforces this). A frame
    count mismatch against the ``shape[0]`` field surfaces as a hard
    parse error rather than silently dropping frames — matches the SVG
    decoder's ``count`` integrity check.
    """
    import base64 as _b64
    import numpy as np
    import torch
    from PIL import Image

    frames = envelope.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RnpProtocolError(
            "image batch envelope missing or empty 'frames' list",
            code=ErrorCode.INTERNAL,
        )
    declared_shape = envelope.get("shape")
    if isinstance(declared_shape, list) and len(declared_shape) >= 1:
        declared_b = int(declared_shape[0])
        if declared_b != len(frames):
            raise RnpProtocolError(
                f"image batch envelope frame count {len(frames)} does not "
                f"match declared shape[0]={declared_b}",
                code=ErrorCode.INTERNAL,
            )

    arrs: list[Any] = []
    base_hw: tuple[int, int] | None = None
    for idx, frame_b64 in enumerate(frames):
        if not isinstance(frame_b64, str):
            raise RnpProtocolError(
                f"image batch frame {idx} is not a base64 string",
                code=ErrorCode.INTERNAL,
            )
        try:
            raw = _b64.b64decode(frame_b64)
            img = Image.open(BytesIO(raw)).convert("RGB")
        except Exception as e:
            raise RnpProtocolError(
                f"image batch frame {idx} failed to decode: {e}",
                code=ErrorCode.INTERNAL,
            ) from e
        arr = np.array(img).astype(np.float32) / 255.0
        if base_hw is None:
            base_hw = (arr.shape[0], arr.shape[1])
        elif (arr.shape[0], arr.shape[1]) != base_hw:
            raise RnpProtocolError(
                f"image batch frame {idx} size {arr.shape[:2]} does not "
                f"match frame 0 size {base_hw} (server should have rejected "
                f"this; likely a wire bug)",
                code=ErrorCode.INTERNAL,
            )
        arrs.append(arr)
    stacked = np.stack(arrs, axis=0)
    return torch.from_numpy(stacked)


# ---------------------------------------------------------------------------
# Mask encode / decode
# ---------------------------------------------------------------------------

def encode_mask_tensor(tensor: Any) -> dict[str, Any]:
    """Encode a ComfyUI MASK tensor (B,H,W float32 in 0..1) as a
    PNG-base64 mask envelope (single-channel, 0..255 grayscale)."""
    import numpy as np
    import torch  # noqa: F401  (typing only)
    from PIL import Image

    if hasattr(tensor, "dim") and tensor.dim() == 3:
        # Batches of masks: take first; multi-mask batches are wrapped
        # by the caller as a list of envelopes when needed.
        frame = tensor[0]
    else:
        frame = tensor
    arr = (frame.cpu().clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = BytesIO()
    img.save(buf, format="PNG")
    height, width = arr.shape[:2]
    return {
        "type":     "mask",
        "encoding": "png_base64",
        "data":     base64.b64encode(buf.getvalue()).decode("ascii"),
        "shape":    [1, height, width],
        "dtype":    "uint8",
    }


def decode_mask_envelope(envelope: dict[str, Any]) -> Any:
    """Decode a mask envelope into a ComfyUI MASK tensor (B,H,W float32)."""
    import numpy as np
    import torch
    from PIL import Image

    encoding = envelope.get("encoding")
    if encoding != "png_base64":
        raise RnpProtocolError(
            f"Unsupported mask encoding: {encoding!r}",
            code=ErrorCode.INTERNAL,
        )
    raw = decode_envelope_data(envelope)
    img = Image.open(BytesIO(raw)).convert("L")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


# ---------------------------------------------------------------------------
# Audio encode / decode
# ---------------------------------------------------------------------------

def encode_audio_input(audio: dict[str, Any]) -> dict[str, Any]:
    """Encode a ComfyUI AUDIO dict ({waveform, sample_rate}) as an
    mp3-base64 audio envelope.
    """
    from comfy_api_nodes.util.conversions import audio_input_to_mp3

    mp3_buf = audio_input_to_mp3(audio)
    sample_rate = int(audio.get("sample_rate", 0))
    waveform = audio.get("waveform")
    channels = int(waveform.shape[-2]) if waveform is not None else None
    duration_s = (
        float(waveform.shape[-1]) / sample_rate
        if (waveform is not None and sample_rate)
        else None
    )
    extra: dict[str, Any] = {}
    if channels is not None:
        extra["channels"] = channels
    if sample_rate:
        extra["sample_rate"] = sample_rate
    if duration_s is not None:
        extra["duration_s"] = duration_s
    return {
        "type":     "audio",
        "encoding": "mp3_base64",
        "data":     base64.b64encode(mp3_buf.getvalue()).decode("ascii"),
        **extra,
    }


def decode_audio_envelope(envelope: dict[str, Any]) -> dict[str, Any]:
    """Decode an audio envelope back into a ComfyUI AUDIO dict."""
    encoding = envelope.get("encoding")
    if encoding not in ("mp3_base64", "mp3_inline", "wav_base64", "wav_inline"):
        raise RnpProtocolError(
            f"Unsupported audio encoding: {encoding!r}",
            code=ErrorCode.INTERNAL,
        )
    from comfy_api_nodes.util.conversions import audio_bytes_to_audio_input
    return audio_bytes_to_audio_input(decode_envelope_data(envelope))


# ---------------------------------------------------------------------------
# Video decode (encode lands when a remote node accepts VIDEO inputs)
# ---------------------------------------------------------------------------

def decode_video_envelope(envelope: dict[str, Any]) -> Any:
    """Decode a video envelope into a ComfyUI Video object (mp4 inline)."""
    encoding = envelope.get("encoding")
    if encoding not in ("mp4_inline", "mp4_base64"):
        raise RnpProtocolError(
            f"Unsupported video encoding: {encoding!r}",
            code=ErrorCode.INTERNAL,
        )
    from comfy_api.latest import InputImpl
    return InputImpl.VideoFromFile(BytesIO(decode_envelope_data(envelope)))


# ---------------------------------------------------------------------------
# Dispatch by envelope type
# ---------------------------------------------------------------------------

_DECODERS = {
    "image": decode_image_envelope,
    "mask":  decode_mask_envelope,
    "audio": decode_audio_envelope,
    "video": decode_video_envelope,
}


def decode_envelope(value: dict[str, Any]) -> Any:
    """Top-level dispatch: convert any envelope to its native ComfyUI type."""
    decoder = _DECODERS.get(value.get("type", ""))
    if decoder is None:
        raise RnpProtocolError(
            f"Unknown envelope type: {value.get('type')!r}",
            code=ErrorCode.INTERNAL,
        )
    return decoder(value)


def is_image_tensor(value: Any) -> bool:
    """Best-effort check for a ComfyUI IMAGE tensor (B,H,W,C float)."""
    return _is_torch_tensor(value) and getattr(value, "dim", lambda: 0)() in (3, 4)


def is_mask_tensor(value: Any) -> bool:
    """Best-effort check for a ComfyUI MASK tensor (B,H,W float or H,W)."""
    return _is_torch_tensor(value) and getattr(value, "dim", lambda: 0)() in (2, 3)


def is_audio_input(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and "waveform" in value
        and "sample_rate" in value
    )


def _is_torch_tensor(value: Any) -> bool:
    try:
        import torch
    except ImportError:
        return False
    return isinstance(value, torch.Tensor)


# Re-export for callers that already imported via this module.
__all__ = [
    "encode_image_tensor",
    "decode_image_envelope",
    "encode_mask_tensor",
    "decode_mask_envelope",
    "encode_audio_input",
    "decode_audio_envelope",
    "decode_video_envelope",
    "decode_envelope",
    "is_envelope",
    "is_image_tensor",
    "is_mask_tensor",
    "is_audio_input",
]
