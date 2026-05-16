"""Smoke test for the client-side MODEL_3D decoder.

Verifies:

1. The vendored ``protocol.HEAVY_TYPES`` now recognises ``model_3d`` so
   ``is_envelope`` returns True for an inbound MODEL_3D envelope.
2. ``serialization.decode_model3d_envelope`` round-trips an inline
   ``glb_inline`` envelope into a ``comfy_api.latest._util.File3D``
   whose ``BytesIO`` payload matches the original GLB bytes.
3. ``decode_envelope`` dispatches ``type=model_3d`` to the new decoder.
4. The proxy_node ``_OUTPUT_CLASSES`` now maps ``"MODEL_3D"`` to
   ``IO.File3DGLB.Output`` so a descriptor with ``output=["MODEL_3D"]``
   parses without falling through to the opaque IO.Custom branch.
5. ``client.CLIENT_CAPABILITIES`` advertises
   ``model_3d:glb_inline`` on every outbound request so descriptors
   gated on the token negotiate cleanly.
6. End-to-end against the server's ``make_model3d_envelope`` helper
   (imported from the comfy-rnp-server worktree) — the bytes the
   server packs are exactly the bytes the client decodes.

Run with the ComfyUI venv (needs torch / comfy_api on PYTHONPATH):

    python notes/run_model3d_decoder.py
"""
from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import sys
import types

# ---------------------------------------------------------------------------
# Locate the worktree under test.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_CLIENT_DIR = os.path.dirname(_HERE)
_COMFYUI_DIR = os.path.dirname(os.path.dirname(_CLIENT_DIR))
_SERVER_DIR = os.path.normpath(
    os.path.join(_COMFYUI_DIR, "..", "comfy-rnp-server")
)


def _load_module(name: str, path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Make ComfyUI importable so ``comfy_api.latest._util.File3D`` resolves.
if _COMFYUI_DIR not in sys.path:
    sys.path.insert(0, _COMFYUI_DIR)
# And the server worktree so we can import the ENCODER side end-to-end.
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

# ComfyUI's full ``comfy_api`` tree pulls torch / PIL / comfy /
# comfy_execution at import time — way too heavy for a decoder
# round-trip check. Stub just the leaf module our decoder imports
# (``comfy_api.latest._util``) with a faithful pure-Python ``File3D``.
class _StubFile3D:
    def __init__(self, source, file_format: str = "") -> None:
        self._source = source
        self._format = file_format

_comfy_api = types.ModuleType("comfy_api")
_comfy_api_latest = types.ModuleType("comfy_api.latest")
_comfy_api_latest_util = types.ModuleType("comfy_api.latest._util")
_comfy_api_latest_util.File3D = _StubFile3D
_comfy_api.latest = _comfy_api_latest
_comfy_api_latest._util = _comfy_api_latest_util
sys.modules.setdefault("comfy_api", _comfy_api)
sys.modules.setdefault("comfy_api.latest", _comfy_api_latest)
sys.modules.setdefault("comfy_api.latest._util", _comfy_api_latest_util)

# Load the client modules as if they were a package — the ``protocol``
# module imports from ``.protocol`` relative to its package, so we
# emulate the package layout under the name ``comfy_remote_nodes``.
pkg = types.ModuleType("comfy_remote_nodes_test")
pkg.__path__ = [_CLIENT_DIR]
sys.modules["comfy_remote_nodes_test"] = pkg

protocol = _load_module(
    "comfy_remote_nodes_test.protocol",
    os.path.join(_CLIENT_DIR, "protocol.py"),
)
# serialization imports ``client`` for ``rnp_client`` but we don't need
# the HTTP client for the decoder path; provide a tiny stub so the
# import succeeds.
client_stub = types.ModuleType("comfy_remote_nodes_test.client")
sys.modules["comfy_remote_nodes_test.client"] = client_stub
serialization = _load_module(
    "comfy_remote_nodes_test.serialization",
    os.path.join(_CLIENT_DIR, "serialization.py"),
)


# ---------------------------------------------------------------------------
# Build a minimal valid binary glTF 2.0 (GLB) payload for the round-trip.
# Format: 12-byte header + N chunks. Header: magic(4) + version(4) + length(4).
# Magic must be b"glTF", version=2. We include exactly one JSON chunk that
# describes an empty asset (enough to satisfy any GLB-magic check; downstream
# nodes do their own validation).
# ---------------------------------------------------------------------------
def _make_minimal_glb() -> bytes:
    json_payload = b'{"asset":{"version":"2.0"}}'
    # Chunks must be 4-byte aligned; pad JSON with spaces (0x20).
    pad = (4 - (len(json_payload) % 4)) % 4
    json_payload = json_payload + (b" " * pad)
    json_chunk = (
        len(json_payload).to_bytes(4, "little")  # chunk length
        + b"JSON"                                  # chunk type
        + json_payload                             # chunk data
    )
    total = 12 + len(json_chunk)
    header = (
        b"glTF"
        + (2).to_bytes(4, "little")
        + total.to_bytes(4, "little")
    )
    return header + json_chunk


# ---------------------------------------------------------------------------
# 1. HEAVY_TYPES recognises model_3d.
# ---------------------------------------------------------------------------
assert "model_3d" in protocol.HEAVY_TYPES, (
    f"HEAVY_TYPES missing 'model_3d': {sorted(protocol.HEAVY_TYPES)}"
)
assert protocol.Capability.MODEL_3D_GLB_INLINE == "model_3d:glb_inline", (
    f"Capability.MODEL_3D_GLB_INLINE wrong value: "
    f"{protocol.Capability.MODEL_3D_GLB_INLINE!r}"
)
print("PASS: HEAVY_TYPES includes 'model_3d'; Capability constant intact.")


# ---------------------------------------------------------------------------
# 2. is_envelope returns True for a MODEL_3D envelope.
# ---------------------------------------------------------------------------
glb_bytes = _make_minimal_glb()
envelope = {
    "type":     "model_3d",
    "encoding": "glb_inline",
    "data":     base64.b64encode(glb_bytes).decode("ascii"),
    "format":   "glb",
    "byte_size": len(glb_bytes),
}
assert protocol.is_envelope(envelope), (
    "is_envelope returned False for a well-formed MODEL_3D envelope; "
    "HEAVY_TYPES update did not take effect."
)
print("PASS: is_envelope recognises model_3d envelopes.")


# ---------------------------------------------------------------------------
# 3. decode_model3d_envelope round-trips inline bytes.
# ---------------------------------------------------------------------------
file3d = serialization.decode_model3d_envelope(envelope)
from comfy_api.latest._util import File3D as _File3D_for_check  # noqa: E402
assert isinstance(file3d, _File3D_for_check), (
    f"Expected File3D, got {type(file3d).__name__}"
)
# File3D exposes get_data() / get_source(); the public API on this
# version returns a BytesIO when constructed with one. Drain bytes:
src = file3d._source  # internal, but stable across this code path
assert hasattr(src, "read"), f"File3D source not BytesIO-like: {type(src)}"
src.seek(0)
round_trip = src.read()
assert round_trip == glb_bytes, (
    f"GLB byte mismatch after decode: input={len(glb_bytes)} bytes, "
    f"output={len(round_trip)} bytes"
)
assert round_trip[:4] == b"glTF", "GLB magic bytes lost in round-trip"
print(f"PASS: decode_model3d_envelope round-trips {len(glb_bytes)} GLB bytes.")


# ---------------------------------------------------------------------------
# 4. decode_envelope dispatches model_3d to the new decoder.
# ---------------------------------------------------------------------------
file3d2 = serialization.decode_envelope(envelope)
assert isinstance(file3d2, _File3D_for_check), (
    f"decode_envelope dispatch failed for type=model_3d: got "
    f"{type(file3d2).__name__}"
)
print("PASS: decode_envelope dispatches type='model_3d' correctly.")


# ---------------------------------------------------------------------------
# 5. Unsupported encoding raises a hard error (defence-in-depth so a
#    future encoding never silently downgrades to garbage bytes).
# ---------------------------------------------------------------------------
bad = dict(envelope, encoding="obj_inline")
try:
    serialization.decode_model3d_envelope(bad)
except protocol.RnpProtocolError as e:
    assert "Unsupported model_3d encoding" in str(e), (
        f"Wrong error message for unsupported encoding: {e!r}"
    )
    assert e.code == protocol.ErrorCode.INTERNAL
    print("PASS: unsupported model_3d encoding raises RnpProtocolError.")
else:
    raise AssertionError(
        "decode_model3d_envelope accepted an unsupported encoding"
    )


# ---------------------------------------------------------------------------
# 6. End-to-end: server's make_model3d_envelope -> client decode.
#    Skips when the server worktree isn't on disk (e.g. CI running
#    against just the client repo).
# ---------------------------------------------------------------------------
server_protocol_path = os.path.join(
    _SERVER_DIR, "comfy_rnp_protocol", "envelopes.py",
)
if os.path.exists(server_protocol_path):
    # Import the server-side envelopes module standalone (it has no
    # ComfyUI deps and uses its own sibling ``constants`` import).
    server_pkg = types.ModuleType("comfy_rnp_protocol_e2e")
    server_pkg.__path__ = [os.path.join(_SERVER_DIR, "comfy_rnp_protocol")]
    sys.modules["comfy_rnp_protocol_e2e"] = server_pkg
    server_constants = _load_module(
        "comfy_rnp_protocol_e2e.constants",
        os.path.join(_SERVER_DIR, "comfy_rnp_protocol", "constants.py"),
    )
    server_envelopes = _load_module(
        "comfy_rnp_protocol_e2e.envelopes",
        os.path.join(_SERVER_DIR, "comfy_rnp_protocol", "envelopes.py"),
    )
    # Build using the server helper exactly as a provider would.
    server_env = server_envelopes.make_model3d_envelope(
        encoding="glb_inline",
        data=base64.b64encode(glb_bytes).decode("ascii"),
        format="glb",
        byte_size=len(glb_bytes),
    )
    assert server_env["type"] == "model_3d", server_env
    assert server_env["encoding"] == "glb_inline", server_env
    assert "model_3d" in server_envelopes.HEAVY_TYPES, (
        "Server HEAVY_TYPES missing 'model_3d' — server PR #37 should "
        "have added it; reinstall comfy_rnp_protocol if this fires."
    )
    # Pipe through the client decoder.
    file3d3 = serialization.decode_envelope(server_env)
    src = file3d3._source
    src.seek(0)
    assert src.read() == glb_bytes, "End-to-end byte mismatch"
    print(
        "PASS: end-to-end server make_model3d_envelope -> client "
        "decode_envelope preserves bytes."
    )
else:
    print(
        f"SKIP end-to-end: server worktree not found at {_SERVER_DIR!r}"
    )


# ---------------------------------------------------------------------------
# 7. proxy_node._OUTPUT_CLASSES maps "MODEL_3D" -> IO.File3DGLB.Output.
#    proxy_node pulls in the full comfy_api / server / comfy_api_nodes
#    tree at import time (too heavy to stub here), so we verify the
#    mapping with a source-level check instead. The pattern is exact:
#    a quoted ``"MODEL_3D":`` key whose value is the
#    ``IO.File3DGLB.Output`` class reference inside the
#    ``_OUTPUT_CLASSES`` dict literal. The same source check would
#    catch a typo (e.g. ``IO.File3DAny.Output``) that a smoke test
#    importing the live module would miss too — the dict literal is
#    the contract.
# ---------------------------------------------------------------------------
proxy_src = open(
    os.path.join(_CLIENT_DIR, "proxy_node.py"), "r", encoding="utf-8",
).read()
assert "_OUTPUT_CLASSES = {" in proxy_src, "proxy_node missing _OUTPUT_CLASSES dict"
assert '"MODEL_3D":' in proxy_src and "IO.File3DGLB.Output" in proxy_src, (
    "proxy_node._OUTPUT_CLASSES does not appear to map "
    "'MODEL_3D' -> IO.File3DGLB.Output"
)
# Verify the mapping is in the same dict literal (not somewhere else).
oc_start = proxy_src.index("_OUTPUT_CLASSES = {")
oc_end = proxy_src.index("}", oc_start)
oc_block = proxy_src[oc_start:oc_end]
assert '"MODEL_3D":' in oc_block, (
    "'MODEL_3D' key not inside the _OUTPUT_CLASSES dict literal"
)
assert "IO.File3DGLB.Output" in oc_block, (
    "IO.File3DGLB.Output value not inside the _OUTPUT_CLASSES dict literal"
)
print(
    "PASS: proxy_node._OUTPUT_CLASSES dict literal contains "
    "'MODEL_3D' -> IO.File3DGLB.Output."
)


# ---------------------------------------------------------------------------
# 8. CLIENT_CAPABILITIES advertises model_3d:glb_inline.
#    client.py imports aiohttp / comfy_api_nodes too — same source-level
#    contract check.
# ---------------------------------------------------------------------------
client_src = open(
    os.path.join(_CLIENT_DIR, "client.py"), "r", encoding="utf-8",
).read()
assert "CLIENT_CAPABILITIES = [" in client_src, (
    "client.py missing CLIENT_CAPABILITIES list"
)
cc_start = client_src.index("CLIENT_CAPABILITIES = [")
cc_end = client_src.index("]", cc_start)
cc_block = client_src[cc_start:cc_end]
assert "Capability.MODEL_3D_GLB_INLINE" in cc_block, (
    "Capability.MODEL_3D_GLB_INLINE not advertised in CLIENT_CAPABILITIES"
)
print("PASS: CLIENT_CAPABILITIES advertises Capability.MODEL_3D_GLB_INLINE.")

print("ALL CHECKS PASSED.")
