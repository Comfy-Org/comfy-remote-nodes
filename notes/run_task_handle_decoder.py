"""Smoke test for the client-side cross-node task_handle decoder.

Verifies:

1. ``Capability.IO_TASK_HANDLE`` is intact at the vendored protocol
   layer and exposed as the well-formed ``io:task_handle`` token.
2. ``"task_handle"`` is in ``HEAVY_TYPES`` and ``is_envelope`` returns
   True for a well-formed task_handle envelope.
3. ``decode_task_handle_envelope`` returns a ``TaskHandle`` dataclass
   with every field round-tripped verbatim (including a non-empty
   ``parent_chain``).
4. ``str(handle)`` surfaces the native_id (so UI previews / log lines
   stay human-meaningful) without leaking the lineage chain.
5. ``encode_task_handle(decode_task_handle_envelope(env)) == env`` —
   the round-trip is lossless so a downstream provider node can re-
   emit the handle unchanged.
6. The top-level ``decode_envelope`` dispatcher routes task_handle
   envelopes to the new decoder (so proxy_node's generic
   ``_deserialize_output`` picks it up without provider-specific
   plumbing).
7. Malformed task_handle envelopes (missing fields, wrong types,
   comma-union ``kind``, non-list ``parent_chain``, bad parent_chain
   entries, wrong encoding) raise ``RnpProtocolError`` with
   ``code=INTERNAL`` — no opaque fallthrough.
8. End-to-end against the server's ``make_task_handle_envelope``
   helper (imported from the comfy-rnp-server worktree) — the
   envelope the server packs is exactly what the client decoder
   unpacks, and ``encode_task_handle`` of the decoded result
   re-produces the same wire shape.
9. ``CLIENT_CAPABILITIES`` advertises ``io:task_handle`` so server-
   side capability-gated descriptors negotiate cleanly.

Run with the ComfyUI venv (needs torch / comfy_api on PYTHONPATH):

    python notes/run_task_handle_decoder.py
"""
from __future__ import annotations

import importlib.util
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


# Emulate the client package layout so the relative ``from . import
# client as rnp_client`` at the top of serialization.py resolves
# without pulling aiohttp / comfy_api_nodes.
pkg = types.ModuleType("comfy_remote_nodes_test")
pkg.__path__ = [_CLIENT_DIR]
sys.modules["comfy_remote_nodes_test"] = pkg

protocol = _load_module(
    "comfy_remote_nodes_test.protocol",
    os.path.join(_CLIENT_DIR, "protocol.py"),
)
# serialization imports ``client`` for ``rnp_client``; the task_handle
# decoder doesn't use it, so a tiny stub suffices.
sys.modules["comfy_remote_nodes_test.client"] = types.ModuleType(
    "comfy_remote_nodes_test.client",
)
serialization = _load_module(
    "comfy_remote_nodes_test.serialization",
    os.path.join(_CLIENT_DIR, "serialization.py"),
)


# ---------------------------------------------------------------------------
# 1. Capability constant intact + well-formed token.
# ---------------------------------------------------------------------------
assert protocol.Capability.IO_TASK_HANDLE == "io:task_handle", (
    f"Capability.IO_TASK_HANDLE wrong value: "
    f"{protocol.Capability.IO_TASK_HANDLE!r}"
)
print("PASS: Capability.IO_TASK_HANDLE = 'io:task_handle'.")


# ---------------------------------------------------------------------------
# 2. HEAVY_TYPES + is_envelope.
# ---------------------------------------------------------------------------
assert "task_handle" in protocol.HEAVY_TYPES, (
    f"'task_handle' missing from HEAVY_TYPES: {sorted(protocol.HEAVY_TYPES)}"
)

hand_envelope = {
    "type":           "task_handle",
    "encoding":       "vendor_inline",
    "vendor":         "tripo",
    "kind":           "MODEL_TASK_ID",
    "native_id":      "abc123-tripo-uuid",
    "origin_node_id": "42",
    "parent_chain": [
        {"vendor": "tripo", "kind": "MODEL_TASK_ID",
         "native_id": "root-uuid", "origin_node_id": "9"},
        {"vendor": "tripo", "kind": "MODEL_TASK_ID",
         "native_id": "prev-uuid", "origin_node_id": "17"},
    ],
}

assert protocol.is_envelope(hand_envelope), (
    "is_envelope returned False for a well-formed task_handle envelope; "
    "HEAVY_TYPES update somehow regressed."
)
print("PASS: 'task_handle' in HEAVY_TYPES and is_envelope recognises it.")


# ---------------------------------------------------------------------------
# 3. decode_task_handle_envelope returns a populated TaskHandle.
# ---------------------------------------------------------------------------
handle = serialization.decode_task_handle_envelope(hand_envelope)
assert isinstance(handle, serialization.TaskHandle), (
    f"decode_task_handle_envelope should return TaskHandle, got "
    f"{type(handle).__name__}"
)
assert handle.vendor == "tripo"
assert handle.kind == "MODEL_TASK_ID"
assert handle.native_id == "abc123-tripo-uuid"
assert handle.origin_node_id == "42"
assert len(handle.parent_chain) == 2
assert handle.parent_chain[0] == {
    "vendor": "tripo", "kind": "MODEL_TASK_ID",
    "native_id": "root-uuid", "origin_node_id": "9",
}
assert handle.parent_chain[1] == {
    "vendor": "tripo", "kind": "MODEL_TASK_ID",
    "native_id": "prev-uuid", "origin_node_id": "17",
}
print("PASS: decode_task_handle_envelope returns TaskHandle with full lineage.")


# ---------------------------------------------------------------------------
# 4. str(handle) surfaces the native_id.
# ---------------------------------------------------------------------------
assert str(handle) == "abc123-tripo-uuid", (
    f"str(TaskHandle) should be the native_id; got {str(handle)!r}"
)
print("PASS: str(TaskHandle) surfaces the native_id.")


# ---------------------------------------------------------------------------
# 5. encode(decode(env)) == env — lossless round-trip.
# ---------------------------------------------------------------------------
re_encoded = serialization.encode_task_handle(handle)
assert re_encoded == hand_envelope, (
    f"round-trip mismatch:\n  original: {hand_envelope}\n  "
    f"re-encoded: {re_encoded}"
)
# Empty chain default is preserved.
solo = serialization.decode_task_handle_envelope({
    "type":           "task_handle",
    "encoding":       "vendor_inline",
    "vendor":         "kling",
    "kind":           "KLING_VIDEO_ID",
    "native_id":      "video-xyz",
    "origin_node_id": "5",
    "parent_chain":   [],
})
assert solo.parent_chain == [], "empty parent_chain should round-trip empty"
assert serialization.encode_task_handle(solo)["parent_chain"] == []
print("PASS: encode_task_handle round-trips lossless (including empty chain).")


# ---------------------------------------------------------------------------
# 6. decode_envelope dispatcher routes to the new decoder.
# ---------------------------------------------------------------------------
dispatched = serialization.decode_envelope(hand_envelope)
assert isinstance(dispatched, serialization.TaskHandle), (
    f"decode_envelope didn't dispatch task_handle to the right decoder; "
    f"got {type(dispatched).__name__}"
)
assert dispatched.native_id == "abc123-tripo-uuid"
print("PASS: decode_envelope dispatches 'task_handle' -> decode_task_handle_envelope.")


# ---------------------------------------------------------------------------
# 7. Malformed envelopes raise RnpProtocolError(INTERNAL).
# ---------------------------------------------------------------------------
def _expect_protocol_error(bad_env: dict, needle: str) -> None:
    try:
        serialization.decode_task_handle_envelope(bad_env)
    except protocol.RnpProtocolError as e:
        assert e.code == protocol.ErrorCode.INTERNAL, f"wrong error code: {e.code}"
        assert needle in str(e), f"error msg doesn't mention {needle!r}: {e}"
    else:
        raise AssertionError(f"expected RnpProtocolError for {needle!r}")


# Wrong encoding.
_expect_protocol_error(
    {"type": "task_handle", "encoding": "replay_ticket_inline",
     "vendor": "tripo", "kind": "MODEL_TASK_ID",
     "native_id": "x", "origin_node_id": "1"},
    "Unsupported task_handle encoding",
)
# Missing top-level fields.
for missing in ("vendor", "kind", "native_id", "origin_node_id"):
    bad = {
        "type": "task_handle", "encoding": "vendor_inline",
        "vendor": "tripo", "kind": "MODEL_TASK_ID",
        "native_id": "x", "origin_node_id": "1",
    }
    del bad[missing]
    _expect_protocol_error(bad, repr(missing))
# Empty string fields.
_expect_protocol_error(
    {"type": "task_handle", "encoding": "vendor_inline",
     "vendor": "", "kind": "MODEL_TASK_ID",
     "native_id": "x", "origin_node_id": "1"},
    "'vendor'",
)
# Comma-union kind rejected (input-acceptance string, not emit kind).
_expect_protocol_error(
    {"type": "task_handle", "encoding": "vendor_inline",
     "vendor": "tripo",
     "kind": "MODEL_TASK_ID,RIG_TASK_ID,RETARGET_TASK_ID",
     "native_id": "x", "origin_node_id": "1"},
    "singular socket kind",
)
# parent_chain not a list.
_expect_protocol_error(
    {"type": "task_handle", "encoding": "vendor_inline",
     "vendor": "tripo", "kind": "MODEL_TASK_ID",
     "native_id": "x", "origin_node_id": "1",
     "parent_chain": "not-a-list"},
    "'parent_chain'",
)
# Bad parent_chain entry — missing field.
_expect_protocol_error(
    {"type": "task_handle", "encoding": "vendor_inline",
     "vendor": "tripo", "kind": "MODEL_TASK_ID",
     "native_id": "x", "origin_node_id": "1",
     "parent_chain": [{"vendor": "tripo", "kind": "MODEL_TASK_ID",
                       "native_id": "y"}]},  # missing origin_node_id
    "parent_chain[0]",
)
# Bad parent_chain entry — comma-union kind in lineage.
_expect_protocol_error(
    {"type": "task_handle", "encoding": "vendor_inline",
     "vendor": "tripo", "kind": "MODEL_TASK_ID",
     "native_id": "x", "origin_node_id": "1",
     "parent_chain": [{"vendor": "tripo",
                       "kind": "MODEL_TASK_ID,RIG_TASK_ID",
                       "native_id": "y", "origin_node_id": "3"}]},
    "parent_chain[0]",
)
# Non-dict parent_chain entry.
_expect_protocol_error(
    {"type": "task_handle", "encoding": "vendor_inline",
     "vendor": "tripo", "kind": "MODEL_TASK_ID",
     "native_id": "x", "origin_node_id": "1",
     "parent_chain": ["not-a-dict"]},
    "parent_chain[0]",
)
print("PASS: malformed task_handle envelopes raise RnpProtocolError(INTERNAL).")


# ---------------------------------------------------------------------------
# 8. End-to-end: server make_task_handle_envelope -> client decode.
#    Skips when the server worktree isn't on disk.
# ---------------------------------------------------------------------------
server_envelopes_path = os.path.join(
    _SERVER_DIR, "comfy_rnp_protocol", "envelopes.py",
)
if os.path.exists(server_envelopes_path):
    server_pkg = types.ModuleType("comfy_rnp_protocol_e2e_handle")
    server_pkg.__path__ = [os.path.join(_SERVER_DIR, "comfy_rnp_protocol")]
    sys.modules["comfy_rnp_protocol_e2e_handle"] = server_pkg
    _load_module(
        "comfy_rnp_protocol_e2e_handle.constants",
        os.path.join(_SERVER_DIR, "comfy_rnp_protocol", "constants.py"),
    )
    server_envelopes = _load_module(
        "comfy_rnp_protocol_e2e_handle.envelopes",
        os.path.join(_SERVER_DIR, "comfy_rnp_protocol", "envelopes.py"),
    )
    assert hasattr(server_envelopes, "make_task_handle_envelope"), (
        "Server envelopes module missing make_task_handle_envelope — "
        "server-side PR #39 should have added it."
    )
    server_env = server_envelopes.make_task_handle_envelope(
        vendor="tripo",
        kind="MODEL_TASK_ID",
        native_id="server-built-uuid",
        origin_node_id="100",
        parent_chain=[
            {"vendor": "tripo", "kind": "MODEL_TASK_ID",
             "native_id": "ancestor-uuid", "origin_node_id": "50"},
        ],
    )
    assert server_env["type"] == "task_handle", server_env
    assert server_env["encoding"] == "vendor_inline", server_env
    # Client decode of the server-packed bytes.
    server_decoded = serialization.decode_task_handle_envelope(server_env)
    assert isinstance(server_decoded, serialization.TaskHandle)
    assert server_decoded.native_id == "server-built-uuid"
    assert server_decoded.origin_node_id == "100"
    assert len(server_decoded.parent_chain) == 1
    assert server_decoded.parent_chain[0]["native_id"] == "ancestor-uuid"
    # Round-trip back through encode_task_handle reproduces the server's wire shape.
    re_encoded_server = serialization.encode_task_handle(server_decoded)
    assert re_encoded_server == server_env, (
        f"client encode -> server wire mismatch:\n"
        f"  server: {server_env}\n  client: {re_encoded_server}"
    )
    # Server-side rejection paths fire as expected (defence in depth —
    # the client decoder mirrors the same rejections, but a malicious /
    # buggy producer can't sneak a bad envelope past the server-side
    # builder either).
    try:
        server_envelopes.make_task_handle_envelope(
            vendor="tripo",
            kind="MODEL_TASK_ID,RIG_TASK_ID",  # comma-union rejected
            native_id="x",
            origin_node_id="1",
        )
    except ValueError as e:
        assert "singular socket kind" in str(e), (
            f"server-side comma-union rejection wrong msg: {e!r}"
        )
    else:
        raise AssertionError(
            "server should reject comma-union 'kind' at envelope build time"
        )
    # Server enforces parent_chain depth cap (32).
    try:
        server_envelopes.make_task_handle_envelope(
            vendor="tripo",
            kind="MODEL_TASK_ID",
            native_id="x",
            origin_node_id="1",
            parent_chain=[
                {"vendor": "tripo", "kind": "MODEL_TASK_ID",
                 "native_id": f"u{i}", "origin_node_id": str(i)}
                for i in range(33)
            ],
        )
    except ValueError as e:
        assert "depth" in str(e), (
            f"server-side depth-cap rejection wrong msg: {e!r}"
        )
    else:
        raise AssertionError(
            "server should reject parent_chain over MAX_CHAIN_DEPTH"
        )
    print(
        "PASS: end-to-end server make_task_handle_envelope -> client "
        "decode_task_handle_envelope -> encode_task_handle round-trips "
        "losslessly, and server-side rejections fire."
    )
else:
    print(
        f"SKIP end-to-end: server worktree not found at {_SERVER_DIR!r}"
    )


# ---------------------------------------------------------------------------
# 9. CLIENT_CAPABILITIES advertises io:task_handle.
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
assert "Capability.IO_TASK_HANDLE" in cc_block, (
    "Capability.IO_TASK_HANDLE not advertised in CLIENT_CAPABILITIES"
)
print("PASS: CLIENT_CAPABILITIES advertises Capability.IO_TASK_HANDLE.")

print("ALL CHECKS PASSED.")
