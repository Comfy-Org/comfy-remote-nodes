"""Smoke test for the descriptor-driven ``local_validate.image_max_batch``
client-side fail-fast.

Exercises the proxy_node's ``_collect_local_validate`` /
``_enforce_local_validate`` pair against a fake descriptor that mirrors
the Flux2 wire shape. Asserts that:

1. ``_collect_local_validate`` extracts ``{"images": {"image_max_batch": 2}}``
   from the descriptor's IMAGE-input options dict.
2. Encoding a 3-frame tensor for that input raises ``RnpProtocolError``
   with ``code=INPUT_INVALID`` and ``user_facing=True`` *before* any
   serialization / HTTP attempt.
3. A 1-frame tensor passes through cleanly.
4. A descriptor without ``local_validate`` is a no-op.
5. ``cap=1`` surfaces the single-frame copy.
6. A ``local_validate`` block missing the ``image_max_batch`` key is a no-op.

Run with any python that has ``torch`` available:

    python notes/run_image_max_batch.py

The test loads the helper functions out of ``proxy_node.py`` via AST
extraction so it doesn't pay the cost of importing ComfyUI's
``comfy_api`` / ``comfy_api_nodes`` packages (which the helper
functions don't touch). Standalone — no test runner / fixtures needed.
"""
from __future__ import annotations

import ast
import asyncio
import importlib.util
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


def _build_serialization_stub(protocol: types.ModuleType) -> types.ModuleType:
    """Minimal ``serialization`` shim covering exactly the surface
    ``_encode_one`` / ``_enforce_local_validate`` / ``_encode_inputs`` use:

    * ``_is_torch_tensor(value)`` — duck-type check for torch tensors.
    * ``is_audio_input(value)`` — always False (no audio in this test).
    * ``encode_image_tensor(tensor, *, accepts_batch)`` — fake envelope.
    * ``encode_mask_tensor(tensor)`` — fake envelope.
    * ``is_envelope(value)`` — False so ``maybe_externalize`` is skipped.
    """
    mod = types.ModuleType("rnp_smoke_serialization")

    def _is_torch_tensor(value):  # noqa: ANN001
        try:
            import torch as _torch
        except ImportError:
            return False
        return isinstance(value, _torch.Tensor)

    def is_audio_input(value):  # noqa: ANN001
        return isinstance(value, dict) and {"waveform", "sample_rate"} <= set(value.keys())

    def encode_image_tensor(tensor, *, accepts_batch=False):  # noqa: ANN001, ARG001
        return {"type": "image", "encoding": "png_base64", "frames": int(tensor.shape[0])}

    def encode_mask_tensor(tensor):  # noqa: ANN001, ARG001
        return {"type": "mask", "encoding": "png_base64"}

    def encode_audio_input(value):  # noqa: ANN001, ARG001
        return {"type": "audio", "encoding": "mp3_base64"}

    def is_envelope(value):  # noqa: ANN001
        return False  # smoke test never externalizes

    mod._is_torch_tensor = _is_torch_tensor
    mod.is_audio_input = is_audio_input
    mod.encode_image_tensor = encode_image_tensor
    mod.encode_mask_tensor = encode_mask_tensor
    mod.encode_audio_input = encode_audio_input
    mod.is_envelope = is_envelope

    async def maybe_externalize(*args, **kwargs):  # pragma: no cover — unused
        return args[0] if args else None
    mod.maybe_externalize = maybe_externalize
    return mod


def _build_proxy_node_stub(
    protocol: types.ModuleType, serialization: types.ModuleType,
) -> types.ModuleType:
    """Extract the four helper functions under test from ``proxy_node.py``
    and exec them in a minimal namespace so we don't drag the rest of
    the module's heavy ComfyUI imports into the test process.
    """
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
        "_collect_local_validate",
        "_enforce_local_validate",
        "_encode_one",
        "_encode_inputs",
    }
    nodes = [
        n for n in tree.body
        if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef)) and n.name in wanted
    ]
    snippet = ast.Module(body=nodes, type_ignores=[])
    code = compile(snippet, src_path, "exec")
    exec(code, mod.__dict__)
    missing = wanted - set(mod.__dict__)
    if missing:
        raise RuntimeError(f"proxy_node helpers missing from extract: {missing}")
    return mod


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _flux2_like_descriptor(cap: int) -> dict:
    """Mirrors the Flux2 ``images`` input descriptor surface."""
    return {
        "name": "FakeFlux2_RNP",
        "input": {
            "required": {
                "prompt": ["STRING", {"multiline": True, "default": ""}],
            },
            "optional": {
                "images": ["IMAGE", {
                    "tooltip": "Up to N images to be used as references.",
                    "local_validate": {"image_max_batch": cap},
                }],
            },
            "hidden": {},
        },
    }


def _no_validate_descriptor() -> dict:
    return {
        "name": "FakeNoValidate_RNP",
        "input": {
            "required": {},
            "optional": {
                "images": ["IMAGE", {"tooltip": "no cap"}],
            },
            "hidden": {},
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def main() -> int:
    protocol = _load_protocol()
    serialization = _build_serialization_stub(protocol)
    proxy_node = _build_proxy_node_stub(protocol, serialization)

    # ---- 1. _collect_local_validate extracts the rule
    desc = _flux2_like_descriptor(cap=2)
    rules = proxy_node._collect_local_validate(desc)
    assert rules == {"images": {"image_max_batch": 2}}, rules
    print("ok: _collect_local_validate extracted", rules)

    # ---- 2. over-cap tensor raises INPUT_INVALID before encoding
    over_cap = torch.zeros((3, 64, 64, 3), dtype=torch.float32)
    raised = False
    try:
        await proxy_node._encode_inputs(
            {"images": over_cap},
            input_serialization={"images": ["png_base64", "png_base64_batch"]},
            local_validate=rules,
            node_id="FakeFlux2_RNP",
        )
    except protocol.RnpProtocolError as e:
        raised = True
        assert e.code == protocol.ErrorCode.INPUT_INVALID, e.code
        assert e.user_facing is True, e.user_facing
        msg = str(e)
        assert "FakeFlux2_RNP" in msg, msg
        assert "3" in msg and "2" in msg, msg
        print("ok: over-cap raised INPUT_INVALID:", msg)
    assert raised, "over-cap tensor did not raise"

    # ---- 3. at-cap tensor passes through (no exception)
    at_cap = torch.zeros((2, 64, 64, 3), dtype=torch.float32)
    out = await proxy_node._encode_inputs(
        {"images": at_cap},
        input_serialization={"images": ["png_base64", "png_base64_batch"]},
        local_validate=rules,
        node_id="FakeFlux2_RNP",
    )
    assert "images" in out
    print("ok: at-cap (B=2) encoded without error")

    # ---- 4. cap=1 / multi-frame raises with the single-frame copy
    desc1 = _flux2_like_descriptor(cap=1)
    rules1 = proxy_node._collect_local_validate(desc1)
    raised = False
    try:
        await proxy_node._encode_inputs(
            {"images": over_cap},
            local_validate=rules1,
            node_id="FakeI2V_RNP",
        )
    except protocol.RnpProtocolError as e:
        raised = True
        assert e.code == protocol.ErrorCode.INPUT_INVALID
        assert "single image" in str(e), str(e)
        print("ok: cap=1 single-frame copy fired:", str(e))
    assert raised

    # ---- 5. descriptor without local_validate is a no-op
    desc_n = _no_validate_descriptor()
    rules_n = proxy_node._collect_local_validate(desc_n)
    assert rules_n == {}, rules_n
    out = await proxy_node._encode_inputs(
        {"images": over_cap},
        input_serialization={"images": ["png_base64", "png_base64_batch"]},
        local_validate=rules_n,
        node_id="FakeNoValidate_RNP",
    )
    assert "images" in out
    print("ok: missing local_validate is a no-op")

    # ---- 6. local_validate block without image_max_batch sub-key is a no-op
    out = await proxy_node._encode_inputs(
        {"images": over_cap},
        input_serialization={"images": ["png_base64", "png_base64_batch"]},
        local_validate={"images": {"some_other_rule": 7}},
        node_id="FakeOther_RNP",
    )
    assert "images" in out
    print("ok: missing image_max_batch sub-key is a no-op")

    print("PASS: image_max_batch local_validate smoke test")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
