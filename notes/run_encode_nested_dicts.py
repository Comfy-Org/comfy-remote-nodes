"""Smoke test for nested dict recursion in ``_encode_one`` /
``_encode_inputs`` / ``_externalize_nested``.

Background: ComfyUI's V3 ``build_nested_inputs`` rebuilds AUTOGROW and
DYNAMIC_COMBO runtime kwargs into nested dicts (e.g. ``{"reference_images":
{"image1": <tensor>, "image2": <tensor>}}`` for a top-level AG, or
``{"model": {"branch": "wan2.7-r2v", "video_refs": {"video1":
<VideoInput>}}}`` for AG-inside-DC). Before this fix, ``_encode_one``
returned a non-tensor non-VideoInput non-audio dict unchanged, which
meant per-slot tensors / VideoInputs survived past encoding and would
either crash at ``json.dumps`` time or violate the server's contract
(comfy_rnp_server's ``_ordered_wan_videoedit_image_envelopes`` etc.
expect each slot value to already be an RNP envelope).

Asserts:

1. Top-level AG-of-IMAGE dict: each ``image<N>`` tensor becomes an
   image envelope; the wrapping dict structure is preserved.
2. Top-level AG-of-VIDEO dict: each ``video<N>`` VideoInput becomes a
   video envelope.
3. DC-wrapping-AG: top-level dict carries a branch key (string) AND a
   nested AG dict of tensors / VideoInputs; encoder recurses one level
   deeper and encodes the leaves while leaving the branch key untouched.
4. AUDIO input is NOT swallowed by the dict-recursion branch: the
   ``{"waveform": ..., "sample_rate": ...}`` shape still routes to
   ``encode_audio_input``.
5. Already-encoded envelope dicts pass through unchanged (no double-
   encoding) — protects against re-running the encoder on a value that
   the caller pre-encoded.
6. Plain scalar config dicts (``{"width": 1024, "height": 576}``) pass
   through unchanged.
7. ``_externalize_nested`` walks the same shape: a nested AG dict of
   oversize envelopes uploads each envelope via the stubbed externalize
   path and the envelope's ``data`` field is swapped for ``uri``;
   non-envelope siblings are untouched; the wrapping dict structure is
   preserved.
8. JSON-serializability post-condition: ``json.dumps`` round-trips the
   full encoded payload (regression guard for the exact bug fixed).
9. Single-tensor / single-VideoInput / single-audio top-level inputs
   still encode the same way they did before (no regression for the
   existing leaf path).

Run with any python that has ``torch`` available (uses the ComfyUI
venv on this workstation):

    python notes/run_encode_nested_dicts.py

The test loads ``_encode_one`` / ``_encode_inputs`` / ``_externalize_nested``
out of ``proxy_node.py`` via AST extraction so we don't pay the cost
of importing ComfyUI's ``comfy_api`` / ``comfy_api_nodes`` packages
(which those helpers don't touch). Same standalone pattern as
``run_image_max_batch.py``.
"""
from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import os
import sys
import types

import torch

# ---------------------------------------------------------------------------
# Locate the worktree under test (this file's parent's parent).
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_CLIENT_DIR = os.path.dirname(_HERE)


def _load_protocol() -> types.ModuleType:
    """Load ``protocol.py`` standalone (no relative imports)."""
    path = os.path.join(_CLIENT_DIR, "protocol.py")
    spec = importlib.util.spec_from_file_location("rnp_smoke_protocol", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakeVideoInput:
    """Duck-typed VideoInput stand-in: ``save_to`` + ``get_duration``
    are the only attrs ``serialization.is_video_input`` checks for."""
    def __init__(self, tag: str, duration: float = 4.0) -> None:
        self.tag = tag
        self._duration = duration

    def save_to(self, *args, **kwargs):  # pragma: no cover — never called by stub
        return None

    def get_duration(self) -> float:
        return self._duration


def _build_serialization_stub(protocol: types.ModuleType) -> types.ModuleType:
    """Minimal ``serialization`` shim covering exactly the surface the
    helpers under test use."""
    mod = types.ModuleType("rnp_smoke_serialization")

    def _is_torch_tensor(value):  # noqa: ANN001
        return isinstance(value, torch.Tensor)

    def is_audio_input(value):  # noqa: ANN001
        return (
            isinstance(value, dict)
            and {"waveform", "sample_rate"} <= set(value.keys())
        )

    def is_video_input(value):  # noqa: ANN001
        return (
            callable(getattr(value, "save_to", None))
            and callable(getattr(value, "get_duration", None))
        )

    def is_model3d_input(value):  # noqa: ANN001
        return False  # unused in this test

    def encode_image_tensor(tensor, *, accepts_batch=False):  # noqa: ANN001, ARG001
        return {
            "type": "image",
            "encoding": "png_base64",
            "data": f"img:{int(tensor.shape[0])}",
        }

    def encode_mask_tensor(tensor):  # noqa: ANN001, ARG001
        return {"type": "mask", "encoding": "png_base64", "data": "mask"}

    def encode_audio_input(value):  # noqa: ANN001, ARG001
        return {"type": "audio", "encoding": "mp3_base64", "data": "aud"}

    def encode_video_input(value):  # noqa: ANN001
        return {
            "type": "video",
            "encoding": "mp4_base64",
            "data": f"vid:{getattr(value, 'tag', '?')}",
        }

    def encode_model3d_input(value):  # noqa: ANN001, ARG001  # pragma: no cover
        return {"type": "model_3d", "encoding": "glb_base64", "data": "m3d"}

    HEAVY_TYPES = {"image", "mask", "audio", "video", "model_3d"}

    def is_envelope(value):  # noqa: ANN001
        return (
            isinstance(value, dict)
            and isinstance(value.get("type"), str)
            and value.get("type") in HEAVY_TYPES
            and isinstance(value.get("encoding"), str)
        )

    # Cap-aware ``maybe_externalize`` stub: swap ``data`` for ``uri`` when
    # the inline payload would exceed ``max_inline_bytes``. Lets us prove
    # that ``_externalize_nested`` reaches every per-slot envelope.
    upload_log: list[str] = []

    async def maybe_externalize(
        envelope,  # noqa: ANN001
        *, server_url=None, max_inline_bytes=None, auth_headers=None,  # noqa: ANN001, ARG001
    ):
        if not server_url or max_inline_bytes is None:
            return envelope
        data = envelope.get("data")
        if not isinstance(data, str):
            return envelope
        if len(data) <= max_inline_bytes:
            return envelope
        upload_log.append(f"{envelope.get('type')}:{data}")
        out = {k: v for k, v in envelope.items() if k != "data"}
        out["uri"] = f"https://upload/{envelope.get('type')}/{len(upload_log)}"
        return out

    mod._is_torch_tensor = _is_torch_tensor
    mod.is_audio_input = is_audio_input
    mod.is_video_input = is_video_input
    mod.is_model3d_input = is_model3d_input
    mod.encode_image_tensor = encode_image_tensor
    mod.encode_mask_tensor = encode_mask_tensor
    mod.encode_audio_input = encode_audio_input
    mod.encode_video_input = encode_video_input
    mod.encode_model3d_input = encode_model3d_input
    mod.is_envelope = is_envelope
    mod.maybe_externalize = maybe_externalize
    mod._upload_log = upload_log  # exposed for assertions
    return mod


def _build_proxy_node_stub(
    protocol: types.ModuleType, serialization: types.ModuleType,
) -> types.ModuleType:
    """Extract the helpers under test from ``proxy_node.py``."""
    mod = types.ModuleType("rnp_smoke_proxy_node")
    mod.__dict__.update({
        "Any": object,
        "log": types.SimpleNamespace(warning=lambda *a, **kw: None),
        "serialization": serialization,
        "RnpProtocolError": protocol.RnpProtocolError,
        "ErrorCode": protocol.ErrorCode,
    })
    src_path = os.path.join(_CLIENT_DIR, "proxy_node.py")
    with open(src_path, "r", encoding="utf-8") as fh:
        src = fh.read()
    tree = ast.parse(src)
    wanted = {
        "_enforce_local_validate",
        "_check_image_max_batch",
        "_encode_one",
        "_encode_inputs",
        "_externalize_nested",
    }
    nodes = [
        n for n in tree.body
        if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))
        and n.name in wanted
    ]
    snippet = ast.Module(body=nodes, type_ignores=[])
    code = compile(snippet, src_path, "exec")
    exec(code, mod.__dict__)
    missing = wanted - set(mod.__dict__)
    if missing:
        raise RuntimeError(f"proxy_node helpers missing from extract: {missing}")
    return mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def main() -> int:
    protocol = _load_protocol()
    serialization = _build_serialization_stub(protocol)
    proxy_node = _build_proxy_node_stub(protocol, serialization)

    img1 = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    img2 = torch.zeros((1, 32, 32, 3), dtype=torch.float32)
    vid1 = _FakeVideoInput("v1")
    vid2 = _FakeVideoInput("v2")
    audio = {
        "waveform": torch.zeros((1, 1, 1000), dtype=torch.float32),
        "sample_rate": 16000,
    }

    # ---- 1. Top-level AG-of-IMAGE dict encodes per-slot
    out = await proxy_node._encode_inputs(
        {"reference_images": {"image1": img1, "image2": img2}},
    )
    refs = out["reference_images"]
    assert isinstance(refs, dict) and set(refs.keys()) == {"image1", "image2"}, refs
    assert refs["image1"]["type"] == "image" and refs["image1"]["data"] == "img:1"
    assert refs["image2"]["type"] == "image" and refs["image2"]["data"] == "img:1"
    print("ok: top-level AG-of-IMAGE encoded per slot")

    # ---- 2. Top-level AG-of-VIDEO dict encodes per-slot
    out = await proxy_node._encode_inputs(
        {"video_refs": {"video1": vid1, "video2": vid2}},
    )
    vrefs = out["video_refs"]
    assert isinstance(vrefs, dict) and set(vrefs.keys()) == {"video1", "video2"}, vrefs
    assert vrefs["video1"]["type"] == "video" and vrefs["video1"]["data"] == "vid:v1"
    assert vrefs["video2"]["type"] == "video" and vrefs["video2"]["data"] == "vid:v2"
    print("ok: top-level AG-of-VIDEO encoded per slot")

    # ---- 3. DC-wrapping-AG: branch key untouched, nested AG dict encoded
    dc_value = {
        "model": "wan2.7-r2v",
        "video_refs": {"video1": vid1},
        "reference_images": {"image1": img1, "image2": img2},
        "duration": 5,
    }
    out = await proxy_node._encode_inputs({"model": dc_value})
    enc = out["model"]
    assert enc["model"] == "wan2.7-r2v", enc
    assert enc["duration"] == 5, enc
    assert enc["video_refs"]["video1"]["type"] == "video"
    assert enc["reference_images"]["image1"]["type"] == "image"
    assert enc["reference_images"]["image2"]["type"] == "image"
    print("ok: DC-wrapping-AG encoded leaves at every level")

    # ---- 4. AUDIO dict is encoded, NOT recursed into
    out = await proxy_node._encode_inputs({"audio": audio})
    assert out["audio"] == {"type": "audio", "encoding": "mp3_base64", "data": "aud"}
    print("ok: AUDIO dict still routes to encode_audio_input")

    # ---- 5. Already-encoded envelope passes through (no double-encode)
    pre_encoded = {"type": "image", "encoding": "png_base64", "data": "preimg"}
    out = await proxy_node._encode_inputs({"images": pre_encoded})
    assert out["images"] == pre_encoded, out
    print("ok: pre-encoded envelope passed through unchanged")

    # ---- 6. Plain scalar config dict passes through unchanged
    out = await proxy_node._encode_inputs(
        {"config": {"width": 1024, "height": 576, "name": "x"}},
    )
    assert out["config"] == {"width": 1024, "height": 576, "name": "x"}, out
    print("ok: scalar config dict passed through unchanged")

    # ---- 7. _externalize_nested walks per-slot envelopes
    serialization._upload_log.clear()
    big_img = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    # encode_image_tensor stub emits ``data="img:1"`` (5 chars). Cap at 3
    # to force every slot to externalize.
    out = await proxy_node._encode_inputs(
        {"reference_images": {"image1": big_img, "image2": big_img}},
        server_url="https://fake.server",
        max_inline_bytes=3,
    )
    refs = out["reference_images"]
    assert refs["image1"].get("uri", "").startswith("https://upload/image/"), refs
    assert refs["image2"].get("uri", "").startswith("https://upload/image/"), refs
    assert "data" not in refs["image1"] and "data" not in refs["image2"]
    assert len(serialization._upload_log) == 2, serialization._upload_log
    print("ok: nested AG envelopes externalized per slot")

    # ---- 7b. _externalize_nested leaves non-envelope siblings alone
    serialization._upload_log.clear()
    out = await proxy_node._encode_inputs(
        {"model": {
            "model": "wan2.7-r2v",
            "duration": 5,
            "reference_images": {"image1": big_img},
        }},
        server_url="https://fake.server",
        max_inline_bytes=3,
    )
    enc = out["model"]
    assert enc["model"] == "wan2.7-r2v" and enc["duration"] == 5, enc
    assert enc["reference_images"]["image1"]["uri"].startswith("https://upload/image/")
    print("ok: scalar siblings preserved alongside externalized envelopes")

    # ---- 8. JSON-serializability post-condition
    payload = await proxy_node._encode_inputs({
        "reference_images": {"image1": img1, "image2": img2},
        "video_refs": {"video1": vid1},
        "audio": audio,
        "config": {"width": 1024},
        "scalar": "hello",
        "n": 42,
    })
    json.dumps(payload)  # would raise TypeError pre-fix
    print("ok: full encoded payload is JSON-serializable")

    # ---- 9. Leaf-path regression: single-tensor / single-video / single-audio
    out = await proxy_node._encode_inputs({"image": img1})
    assert out["image"]["type"] == "image", out
    out = await proxy_node._encode_inputs({"video": vid1})
    assert out["video"]["type"] == "video", out
    out = await proxy_node._encode_inputs({"audio": audio})
    assert out["audio"]["type"] == "audio", out
    out = await proxy_node._encode_inputs({"prompt": "hi", "seed": 7})
    assert out == {"prompt": "hi", "seed": 7}, out
    print("ok: top-level leaf path unchanged")

    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
