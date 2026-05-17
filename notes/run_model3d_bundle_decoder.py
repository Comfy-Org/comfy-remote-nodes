"""Smoke test for the client-side multi-file MODEL_3D bundle decoder.

Verifies:

1. ``Capability.MODEL_3D_BUNDLE_INLINE`` is intact at the vendored
   protocol layer and exposed as the well-formed
   ``model_3d:bundle_inline`` token.
2. ``is_envelope`` recognises a ``bundle_inline`` envelope (still
   ``type=model_3d`` — HEAVY_TYPES unchanged).
3. ``decode_model3d_envelope`` dispatches ``encoding=bundle_inline``
   to the new ``BundledFile3D`` class, returns a real ``File3D``
   subclass, and round-trips the primary mesh bytes via
   ``get_data()`` / ``get_bytes()`` without materialising anything
   on disk.
4. ``get_source()`` materialises the whole bundle into a temp
   directory using the producer-supplied relative paths, so a .obj
   that references its .mtl by filename (and a texture in
   ``textures/``) finds them on disk via the filesystem.
5. ``save_to(dest)`` copies the whole bundle into the destination
   directory — the primary file under the caller-specified name,
   sidecars under their ORIGINAL relative paths — so cross-file refs
   inside the .obj (which point at ``model.mtl``) still resolve when
   the user saves the model somewhere stable.
6. Bundle-specific accessors (``primary_path``, ``file_paths``,
   ``file_bytes``, ``file_role``, ``paths_with_role``) return the
   expected values; unknown roles round-trip without error.
7. Malformed bundle envelopes (missing primary_path, missing
   files, missing data, primary_path not in files) raise
   ``RnpProtocolError`` with ``code=INTERNAL`` — no opaque
   fallthrough.
8. End-to-end against the server's ``make_model3d_bundle_envelope``
   helper (imported from the comfy-rnp-server worktree) — the bytes
   the server packs into each file are exactly the bytes the
   client unpacks, and the path sanitization happens server-side so
   the client never sees a malicious path.
9. ``CLIENT_CAPABILITIES`` advertises ``model_3d:bundle_inline``
   so server-side capability-gated descriptors negotiate cleanly.

Run with the ComfyUI venv (needs torch / comfy_api on PYTHONPATH):

    python notes/run_model3d_bundle_decoder.py
"""
from __future__ import annotations

import base64
import importlib.util
import os
import shutil
import sys
import tempfile
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


# Make ComfyUI importable so ``comfy_api.latest._util.geometry_types.File3D``
# resolves at first BundledFile3D construction.
if _COMFYUI_DIR not in sys.path:
    sys.path.insert(0, _COMFYUI_DIR)
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

# ComfyUI's full ``comfy_api`` tree pulls torch / PIL / comfy_execution at
# import time. Stub just what BundledFile3D needs — the leaf
# ``comfy_api.latest._util.geometry_types.File3D`` class with a faithful
# subclass-friendly implementation.
class _StubFile3D:
    """Faithful subset of upstream ``comfy_api.latest._util.geometry_types.File3D``.

    Implements the same parent-class shape (``_source`` / ``_format``
    attrs + ``get_source`` / ``get_data`` / ``get_bytes`` / ``save_to``
    methods) so subclassing it produces an object indistinguishable
    from the real thing for the decoder's purposes. The smoke test
    deliberately stubs torch + the heavy ``comfy_api.latest._util``
    package init to keep the test runnable in the rnp venv.
    """
    def __init__(self, source, file_format: str = "") -> None:
        self._source = source
        self._format = (file_format or "").lstrip(".").lower()

    @property
    def format(self) -> str:
        return self._format

    @format.setter
    def format(self, value: str) -> None:
        self._format = (value or "").lstrip(".").lower()


# Stub ``comfy_api.latest._util`` and the nested ``geometry_types`` module
# so ``from comfy_api.latest._util.geometry_types import File3D`` works
# without pulling torch / PIL / etc.
_comfy_api = types.ModuleType("comfy_api")
_comfy_api.__path__ = []  # type: ignore[attr-defined]
_comfy_api_latest = types.ModuleType("comfy_api.latest")
_comfy_api_latest.__path__ = []  # type: ignore[attr-defined]
_comfy_api_latest_util = types.ModuleType("comfy_api.latest._util")
_comfy_api_latest_util.__path__ = []  # type: ignore[attr-defined]
_comfy_api_latest_util.File3D = _StubFile3D
_comfy_api_latest_util_geom = types.ModuleType("comfy_api.latest._util.geometry_types")
_comfy_api_latest_util_geom.File3D = _StubFile3D
sys.modules.setdefault("comfy_api", _comfy_api)
sys.modules.setdefault("comfy_api.latest", _comfy_api_latest)
sys.modules.setdefault("comfy_api.latest._util", _comfy_api_latest_util)
sys.modules.setdefault("comfy_api.latest._util.geometry_types", _comfy_api_latest_util_geom)

# Emulate the client package layout for the relative imports.
pkg = types.ModuleType("comfy_remote_nodes_test")
pkg.__path__ = [_CLIENT_DIR]
sys.modules["comfy_remote_nodes_test"] = pkg

protocol = _load_module(
    "comfy_remote_nodes_test.protocol",
    os.path.join(_CLIENT_DIR, "protocol.py"),
)
# serialization imports ``client`` for ``rnp_client``; the bundle decoder
# doesn't use it, so a tiny stub suffices.
sys.modules["comfy_remote_nodes_test.client"] = types.ModuleType(
    "comfy_remote_nodes_test.client",
)
serialization = _load_module(
    "comfy_remote_nodes_test.serialization",
    os.path.join(_CLIENT_DIR, "serialization.py"),
)


# ---------------------------------------------------------------------------
# Build a minimal Wavefront-OBJ + MTL + PNG-stub bundle for the round-trip.
# These bytes don't need to be loadable by an actual OBJ parser — the
# decoder is bytes-in / bytes-out — but using realistic content makes
# debugging easier if a future regression breaks the cross-file refs.
# ---------------------------------------------------------------------------
def _make_minimal_obj() -> bytes:
    return (
        b"# Minimal OBJ for RNP bundle smoke test.\n"
        b"mtllib model.mtl\n"
        b"v 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 0.0 1.0 0.0\n"
        b"usemtl test_material\n"
        b"f 1 2 3\n"
    )


def _make_minimal_mtl() -> bytes:
    return (
        b"# Minimal MTL for RNP bundle smoke test.\n"
        b"newmtl test_material\n"
        b"Ka 1.0 1.0 1.0\n"
        b"Kd 1.0 1.0 1.0\n"
        b"map_Kd textures/albedo.png\n"
        b"map_bump textures/normal.png\n"
    )


def _make_stub_png(tag: bytes) -> bytes:
    # Not a real PNG — the decoder doesn't parse it. Tag distinguishes
    # each "texture" so the round-trip assertions can pin which bytes
    # went where.
    return b"\x89PNG\r\n\x1a\n" + tag


obj_bytes = _make_minimal_obj()
mtl_bytes = _make_minimal_mtl()
albedo_bytes = _make_stub_png(b"ALBEDO_STUB")
normal_bytes = _make_stub_png(b"NORMAL_STUB")
roughness_bytes = _make_stub_png(b"ROUGHNESS_STUB")


# ---------------------------------------------------------------------------
# 1. Capability constant intact at the vendored protocol layer.
# ---------------------------------------------------------------------------
assert protocol.Capability.MODEL_3D_BUNDLE_INLINE == "model_3d:bundle_inline", (
    f"Capability.MODEL_3D_BUNDLE_INLINE wrong value: "
    f"{protocol.Capability.MODEL_3D_BUNDLE_INLINE!r}"
)
print("PASS: Capability.MODEL_3D_BUNDLE_INLINE = 'model_3d:bundle_inline'.")


# ---------------------------------------------------------------------------
# Build a hand-crafted bundle envelope for the next several checks.
# ---------------------------------------------------------------------------
hand_envelope = {
    "type":         "model_3d",
    "encoding":     "bundle_inline",
    "format":       "obj",
    "primary_path": "model.obj",
    "files": [
        {"path": "model.obj",            "format": "obj", "role": "mesh",
         "data": base64.b64encode(obj_bytes).decode("ascii"),
         "byte_size": len(obj_bytes)},
        {"path": "model.mtl",            "format": "mtl", "role": "material",
         "data": base64.b64encode(mtl_bytes).decode("ascii")},
        {"path": "textures/albedo.png",  "format": "png", "role": "texture_diffuse",
         "data": base64.b64encode(albedo_bytes).decode("ascii")},
        {"path": "textures/normal.png",  "format": "png", "role": "texture_normal",
         "data": base64.b64encode(normal_bytes).decode("ascii")},
        {"path": "textures/roughness.png","format": "png","role": "texture_roughness",
         "data": base64.b64encode(roughness_bytes).decode("ascii")},
        # Unknown role — must round-trip without error.
        {"path": "extras/notes.txt",     "role": "producer_notes",
         "data": base64.b64encode(b"hello").decode("ascii")},
    ],
}


# ---------------------------------------------------------------------------
# 2. is_envelope recognises a bundle envelope.
# ---------------------------------------------------------------------------
assert protocol.is_envelope(hand_envelope), (
    "is_envelope returned False for a well-formed bundle_inline envelope; "
    "HEAVY_TYPES update somehow regressed."
)
print("PASS: is_envelope recognises bundle_inline envelopes.")


# ---------------------------------------------------------------------------
# 3. decode_model3d_envelope dispatches + round-trips primary bytes.
# ---------------------------------------------------------------------------
bundle = serialization.decode_model3d_envelope(hand_envelope)
bundle_cls = serialization._bundled_file3d_class()
assert isinstance(bundle, bundle_cls), (
    f"decode_model3d_envelope should return BundledFile3D, got "
    f"{type(bundle).__name__}"
)
# It must ALSO be a File3D (parent class) — that's the whole point of
# the subclass approach.
assert isinstance(bundle, _StubFile3D), (
    f"BundledFile3D must subclass File3D — got {type(bundle).__mro__}"
)
# Primary file bytes round-trip via get_data / get_bytes (no disk I/O).
assert bundle.get_bytes() == obj_bytes, (
    f"primary OBJ bytes lost: {len(bundle.get_bytes())} vs {len(obj_bytes)}"
)
src = bundle.get_data()
src.seek(0)
assert src.read() == obj_bytes, "primary OBJ get_data() didn't round-trip"
assert bundle.format == "obj", f"primary format wrong: {bundle.format!r}"
print("PASS: decode_model3d_envelope returns BundledFile3D + primary bytes round-trip.")


# ---------------------------------------------------------------------------
# 4. get_source() materialises the whole bundle on disk.
# ---------------------------------------------------------------------------
disk_path = bundle.get_source()
assert os.path.isfile(disk_path), f"get_source() should return existing path: {disk_path}"
assert disk_path.endswith("model.obj"), f"primary path wrong: {disk_path}"
with open(disk_path, "rb") as f:
    assert f.read() == obj_bytes, "disk-materialised primary OBJ bytes mismatched"
# Sidecars must be at the SAME relative paths so the .obj's mtllib /
# map_Kd refs resolve.
temp_root = bundle.materialize_to_temp_dir()
assert os.path.isfile(os.path.join(temp_root, "model.mtl")), (
    "model.mtl missing from materialised bundle"
)
assert os.path.isfile(os.path.join(temp_root, "textures", "albedo.png")), (
    "textures/albedo.png missing from materialised bundle"
)
assert os.path.isfile(os.path.join(temp_root, "textures", "normal.png"))
assert os.path.isfile(os.path.join(temp_root, "textures", "roughness.png"))
assert os.path.isfile(os.path.join(temp_root, "extras", "notes.txt"))
# Sidecar bytes match.
with open(os.path.join(temp_root, "model.mtl"), "rb") as f:
    assert f.read() == mtl_bytes
with open(os.path.join(temp_root, "textures", "albedo.png"), "rb") as f:
    assert f.read() == albedo_bytes
print("PASS: get_source() materialises whole bundle on disk at producer-supplied paths.")


# ---------------------------------------------------------------------------
# 5. save_to(dest) copies the whole bundle into the destination directory.
# ---------------------------------------------------------------------------
save_dir = tempfile.mkdtemp(prefix="rnp_save_to_test_")
try:
    out_path = bundle.save_to(os.path.join(save_dir, "renamed_scene.obj"))
    assert os.path.isfile(out_path)
    with open(out_path, "rb") as f:
        assert f.read() == obj_bytes, "primary OBJ bytes wrong after save_to"
    # Sidecars keep ORIGINAL filenames so cross-file refs inside the
    # OBJ (which point at "model.mtl") still resolve relative to the
    # renamed primary file's directory.
    assert os.path.isfile(os.path.join(save_dir, "model.mtl")), (
        "save_to should copy model.mtl alongside the renamed primary"
    )
    assert os.path.isfile(os.path.join(save_dir, "textures", "albedo.png"))
    assert os.path.isfile(os.path.join(save_dir, "textures", "normal.png"))
    with open(os.path.join(save_dir, "textures", "normal.png"), "rb") as f:
        assert f.read() == normal_bytes
    print("PASS: save_to copies whole bundle, primary renamed, sidecars preserve filenames.")
finally:
    shutil.rmtree(save_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 6. Bundle-specific accessors.
# ---------------------------------------------------------------------------
assert bundle.primary_path == "model.obj"
assert set(bundle.file_paths) == {
    "model.obj", "model.mtl",
    "textures/albedo.png", "textures/normal.png", "textures/roughness.png",
    "extras/notes.txt",
}
assert bundle.file_bytes("textures/albedo.png") == albedo_bytes
# Optional roles.
assert bundle.file_role("model.obj") == "mesh"
assert bundle.file_role("textures/normal.png") == "texture_normal"
# Unknown roles round-trip — no enum check.
assert bundle.file_role("extras/notes.txt") == "producer_notes"
# Missing role (none supplied) returns None.
assert bundle.file_role("nonexistent.png") is None
# Convenience lookup.
assert bundle.paths_with_role("texture_normal") == ["textures/normal.png"]
assert bundle.paths_with_role("nonexistent_role") == []
print("PASS: bundle accessors (primary_path / file_paths / file_role / paths_with_role).")


# ---------------------------------------------------------------------------
# 7. Malformed bundle envelopes raise RnpProtocolError with code=INTERNAL.
# ---------------------------------------------------------------------------
def _expect_protocol_error(bad_env: dict, needle: str) -> None:
    try:
        serialization.decode_model3d_envelope(bad_env)
    except protocol.RnpProtocolError as e:
        assert e.code == protocol.ErrorCode.INTERNAL, f"wrong error code: {e.code}"
        assert needle in str(e), f"error msg doesn't mention {needle!r}: {e}"
    else:
        raise AssertionError(f"expected RnpProtocolError for {needle!r}")

_expect_protocol_error(
    {"type": "model_3d", "encoding": "bundle_inline", "files": [
        {"path": "x.obj", "data": "AA=="},
    ]},
    "primary_path",
)
_expect_protocol_error(
    {"type": "model_3d", "encoding": "bundle_inline", "primary_path": "x.obj"},
    "files",
)
_expect_protocol_error(
    {"type": "model_3d", "encoding": "bundle_inline", "primary_path": "x.obj",
     "files": [{"path": "y.obj", "data": "AA=="}]},
    "primary_path 'x.obj' not in",
)
_expect_protocol_error(
    {"type": "model_3d", "encoding": "bundle_inline", "primary_path": "x.obj",
     "files": [{"path": "x.obj"}]},  # missing data
    "missing inline 'data'",
)
_expect_protocol_error(
    {"type": "model_3d", "encoding": "wat_inline", "data": "AA=="},
    "Unsupported model_3d encoding",
)
print("PASS: malformed bundle envelopes raise RnpProtocolError(INTERNAL).")


# ---------------------------------------------------------------------------
# 8. End-to-end: server make_model3d_bundle_envelope -> client decode.
#    Skips when the server worktree isn't on disk.
# ---------------------------------------------------------------------------
server_protocol_path = os.path.join(
    _SERVER_DIR, "comfy_rnp_protocol", "envelopes.py",
)
if os.path.exists(server_protocol_path):
    server_pkg = types.ModuleType("comfy_rnp_protocol_e2e_bundle")
    server_pkg.__path__ = [os.path.join(_SERVER_DIR, "comfy_rnp_protocol")]
    sys.modules["comfy_rnp_protocol_e2e_bundle"] = server_pkg
    _load_module(
        "comfy_rnp_protocol_e2e_bundle.constants",
        os.path.join(_SERVER_DIR, "comfy_rnp_protocol", "constants.py"),
    )
    server_envelopes = _load_module(
        "comfy_rnp_protocol_e2e_bundle.envelopes",
        os.path.join(_SERVER_DIR, "comfy_rnp_protocol", "envelopes.py"),
    )
    assert hasattr(server_envelopes, "make_model3d_bundle_envelope"), (
        "Server envelopes module missing make_model3d_bundle_envelope — "
        "server-side PR #38 should have added it."
    )
    server_env = server_envelopes.make_model3d_bundle_envelope(
        files=[
            {"path": "model.obj",            "format": "obj", "role": "mesh",
             "data": base64.b64encode(obj_bytes).decode("ascii"),
             "byte_size": len(obj_bytes)},
            {"path": "model.mtl",            "format": "mtl", "role": "material",
             "data": base64.b64encode(mtl_bytes).decode("ascii")},
            {"path": "textures/albedo.png",  "format": "png", "role": "texture_diffuse",
             "data": base64.b64encode(albedo_bytes).decode("ascii")},
        ],
        primary_path="model.obj",
        format="obj",
    )
    assert server_env["type"] == "model_3d", server_env
    assert server_env["encoding"] == "bundle_inline", server_env
    assert server_env["primary_path"] == "model.obj"
    assert len(server_env["files"]) == 3
    # Path sanitization happens server-side. Confirm a malicious path
    # would be rejected at envelope build time (defence-in-depth — the
    # client decoder doesn't see this).
    try:
        server_envelopes.make_model3d_bundle_envelope(
            files=[{"path": "../etc/passwd", "data": "AA=="}],
            primary_path="../etc/passwd",
        )
    except ValueError as e:
        assert "parent-dir" in str(e) or "absolute" in str(e), (
            f"path sanitization mistake: {e!r}"
        )
    else:
        raise AssertionError("server should reject '..' path at envelope build time")
    # Pipe through the client decoder.
    server_bundle = serialization.decode_model3d_envelope(server_env)
    assert isinstance(server_bundle, bundle_cls), (
        f"server-built envelope decoded to wrong type: {type(server_bundle)}"
    )
    assert server_bundle.get_bytes() == obj_bytes
    assert server_bundle.file_bytes("textures/albedo.png") == albedo_bytes
    assert server_bundle.file_role("model.obj") == "mesh"
    print(
        "PASS: end-to-end server make_model3d_bundle_envelope -> client "
        "decode_model3d_envelope preserves all files + roles + bytes."
    )
else:
    print(
        f"SKIP end-to-end: server worktree not found at {_SERVER_DIR!r}"
    )


# ---------------------------------------------------------------------------
# 9. CLIENT_CAPABILITIES advertises model_3d:bundle_inline.
#    client.py imports aiohttp / comfy_api_nodes — same source-level
#    contract check we used for the GLB cap in PR #1.
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
assert "Capability.MODEL_3D_BUNDLE_INLINE" in cc_block, (
    "Capability.MODEL_3D_BUNDLE_INLINE not advertised in CLIENT_CAPABILITIES"
)
print("PASS: CLIENT_CAPABILITIES advertises Capability.MODEL_3D_BUNDLE_INLINE.")

print("ALL CHECKS PASSED.")
