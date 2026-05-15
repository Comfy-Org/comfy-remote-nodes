"""Smoke test for the three batch-URI followups (PR-2/PR-3 cleanup):

1. ``serialization.maybe_externalize`` keys batch dispatch off the
   ``encoding`` string (``png_base64_batch``) — NOT field presence
   (``frames``).
2. ``serialization.maybe_externalize`` rejects an empty/missing
   ``frames`` list on a ``png_base64_batch`` envelope as a hard
   ``RnpProtocolError`` rather than silently no-op-ing.
3. ``proxy_node._deserialize_output`` accepts the externalised
   ``png_base64_batch_uri`` shape on the response path: fetches every
   ``uri`` (in parallel via ``asyncio.gather``), rewrites to inline
   ``frames``, and decodes into a ``[B,H,W,C]`` tensor with frame
   ordering preserved.

Run with any python that has ``torch`` and ``Pillow`` available:

    python notes/run_batch_uri_followups.py
"""
from __future__ import annotations

import asyncio
import base64
import http.server
import importlib.util
import io
import os
import socketserver
import sys
import threading
import time
import types

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Locate the worktree under test.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_CLIENT_DIR = os.path.dirname(_HERE)


def _load_module(name: str, path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Stub out client.upload_asset so the dispatch tests don't need a real
# RNP server — the externalize batch path will call this once per frame.
_uploads: list[bytes] = []


async def _fake_upload_asset(server_url, raw, *, content_type, auth_headers=None):  # noqa: ANN001
    _uploads.append(bytes(raw))
    return f"https://fake-storage/{len(_uploads) - 1}"


def _install_fake_client() -> None:
    fake_client = types.ModuleType("comfy_remote_nodes.client")
    fake_client.upload_asset = _fake_upload_asset
    sys.modules["comfy_remote_nodes.client"] = fake_client


def _png_b64_for_color(color: tuple[int, int, int]) -> str:
    img = Image.new("RGB", (4, 4), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _png_bytes_for_color(color: tuple[int, int, int]) -> bytes:
    img = Image.new("RGB", (4, 4), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Test 1+2: maybe_externalize dispatch + empty-frames
# ---------------------------------------------------------------------------

async def _test_maybe_externalize() -> list[str]:
    failures: list[str] = []

    # Re-import serialization fresh because it caches a `client` ref.
    sys.modules.pop("comfy_remote_nodes", None)
    sys.modules.pop("comfy_remote_nodes.serialization", None)
    sys.modules.pop("comfy_remote_nodes.protocol", None)

    # Build a minimal package shim so `from . import client` works.
    pkg = types.ModuleType("comfy_remote_nodes")
    pkg.__path__ = [_CLIENT_DIR]
    sys.modules["comfy_remote_nodes"] = pkg
    _install_fake_client()
    protocol = _load_module(
        "comfy_remote_nodes.protocol",
        os.path.join(_CLIENT_DIR, "protocol.py"),
    )
    serialization = _load_module(
        "comfy_remote_nodes.serialization",
        os.path.join(_CLIENT_DIR, "serialization.py"),
    )

    # --- Case A: batch envelope with payload bigger than cap → externalises
    _uploads.clear()
    big_frames = [_png_b64_for_color((i * 30, 0, 0)) for i in range(4)]
    env = {
        "type": "image",
        "encoding": "png_base64_batch",
        "frames": big_frames,
        "shape": [4, 4, 4, 3],
        "dtype": "uint8",
        "color_space": "srgb",
    }
    out = await serialization.maybe_externalize(
        env, server_url="http://fake", max_inline_bytes=1, auth_headers={},
    )
    if out.get("encoding") != "png_base64_batch_uri":
        failures.append(
            f"A: expected encoding=png_base64_batch_uri, got {out.get('encoding')!r}"
        )
    if "frames" in out:
        failures.append("A: 'frames' field still present after externalise")
    if not isinstance(out.get("uris"), list) or len(out["uris"]) != 4:
        failures.append(f"A: expected 4 uris, got {out.get('uris')!r}")
    if len(_uploads) != 4:
        failures.append(f"A: expected 4 uploads, got {len(_uploads)}")

    # --- Case B: dispatch keys off encoding, NOT field presence.
    # An envelope with a stray `frames` field but a non-batch encoding
    # must NOT be treated as a batch (would have under the old
    # `isinstance(frames, list) and frames` dispatch).
    _uploads.clear()
    rogue = {
        "type": "image",
        "encoding": "png_base64",  # singleton encoding
        "data": _png_b64_for_color((10, 20, 30)) * 200,  # > cap
        "frames": ["bogus-extra-field"],
    }
    out_b = await serialization.maybe_externalize(
        rogue, server_url="http://fake", max_inline_bytes=1, auth_headers={},
    )
    if out_b.get("encoding") == "png_base64_batch_uri":
        failures.append(
            "B: dispatch took batch branch on a singleton encoding (rogue frames field)"
        )
    if "uri" not in out_b:
        failures.append("B: singleton encoding should have externalised to 'uri'")
    if len(_uploads) != 1:
        failures.append(f"B: expected 1 upload (singleton path), got {len(_uploads)}")

    # --- Case C: empty frames on a batch encoding is a hard error.
    bad = {
        "type": "image",
        "encoding": "png_base64_batch",
        "frames": [],
        "shape": [0, 4, 4, 3],
    }
    try:
        await serialization.maybe_externalize(
            bad, server_url="http://fake", max_inline_bytes=1, auth_headers={},
        )
    except protocol.RnpProtocolError:
        pass
    else:
        failures.append("C: empty 'frames' on batch encoding did not raise RnpProtocolError")

    # --- Case D: under-cap batch: no upload, envelope unchanged.
    _uploads.clear()
    small = {
        "type": "image",
        "encoding": "png_base64_batch",
        "frames": big_frames,
        "shape": [4, 4, 4, 3],
    }
    out_d = await serialization.maybe_externalize(
        small,
        server_url="http://fake",
        max_inline_bytes=10 * 1024 * 1024,
        auth_headers={},
    )
    if out_d is not small and out_d != small:
        failures.append("D: under-cap batch should be a no-op")
    if _uploads:
        failures.append(f"D: under-cap batch should not upload, got {len(_uploads)}")

    return failures


# ---------------------------------------------------------------------------
# Test 3: _deserialize_output handles png_base64_batch_uri on response
# ---------------------------------------------------------------------------

DELAY_S = 0.4
N_FRAMES = 4


class _SlowPngHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        time.sleep(DELAY_S)
        try:
            idx = int(self.path.rsplit("/", 1)[-1])
        except ValueError:
            self.send_response(404); self.end_headers(); return
        # Distinct color per frame so we can verify ordering after decode.
        body = _png_bytes_for_color((idx * 60 % 256, 80, 120))
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_a, **_kw) -> None:
        return


async def _test_deserialize_output() -> list[str]:
    failures: list[str] = []

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _SlowPngHandler)
    host, port = server.server_address
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        # Stub the proxy_node's _fetch_url_envelope so we don't have to
        # bring up the whole comfy_api_nodes.util.client stack.
        import urllib.request as _urlreq

        async def fake_fetch(envelope, policy, *, auth_headers, node_cls):  # noqa: ANN001, ARG001
            # Real ``_fetch_url_envelope`` is properly async (aiohttp);
            # the smoke stub uses sync urllib so we hop to a thread to
            # preserve the parallel-gather semantics under test.
            uri = envelope["uri"]
            def _do_fetch() -> bytes:
                with _urlreq.urlopen(uri, timeout=10.0) as resp:
                    return resp.read()
            return await asyncio.to_thread(_do_fetch)

        # Re-import proxy_node helpers via AST to avoid loading ComfyUI.
        import ast

        with open(os.path.join(_CLIENT_DIR, "proxy_node.py"), "r", encoding="utf-8") as fh:
            src = fh.read()
        tree = ast.parse(src)
        wanted = {"_deserialize_output"}
        nodes = [n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name in wanted]
        if len(nodes) != len(wanted):
            failures.append(f"missing AST nodes; found {[n.name for n in nodes]}")
            return failures

        # Re-import the protocol + serialization modules with the package
        # shim so `from . import` works inside serialization.
        sys.modules.pop("comfy_remote_nodes.serialization", None)
        sys.modules.pop("comfy_remote_nodes.protocol", None)
        pkg = types.ModuleType("comfy_remote_nodes")
        pkg.__path__ = [_CLIENT_DIR]
        sys.modules["comfy_remote_nodes"] = pkg
        _install_fake_client()
        protocol = _load_module(
            "comfy_remote_nodes.protocol",
            os.path.join(_CLIENT_DIR, "protocol.py"),
        )
        serialization = _load_module(
            "comfy_remote_nodes.serialization",
            os.path.join(_CLIENT_DIR, "serialization.py"),
        )

        ns: dict = {
            "Any": object,
            "RnpProtocolError": protocol.RnpProtocolError,
            "ErrorCode": protocol.ErrorCode,
            "is_envelope": protocol.is_envelope,
            "serialization": serialization,
            "_fetch_url_envelope": fake_fetch,
            "log": types.SimpleNamespace(warning=lambda *a, **kw: None),
        }
        mod_src = ast.Module(body=nodes, type_ignores=[])
        exec(compile(mod_src, "<batch_uri_smoke>", "exec"), ns)
        deserialize = ns["_deserialize_output"]

        # Build the externalised batch envelope on the response path.
        uris = [f"http://{host}:{port}/frame/{i}" for i in range(N_FRAMES)]
        envelope = {
            "type": "image",
            "encoding": "png_base64_batch_uri",
            "uris": uris,
            "shape": [N_FRAMES, 4, 4, 3],
            "dtype": "uint8",
            "color_space": "srgb",
        }
        url_fetch_policy = {"image": {"url_kind": "presigned", "timeout_s": 10}}

        t0 = time.monotonic()
        tensor = await deserialize(
            envelope, meta=None,
            url_fetch_policy=url_fetch_policy,
            server_url="http://fake",
            auth_headers={},
            node_cls=None,
        )
        elapsed = time.monotonic() - t0

        if not isinstance(tensor, torch.Tensor):
            failures.append(f"expected torch.Tensor, got {type(tensor).__name__}")
            return failures
        if tuple(tensor.shape) != (N_FRAMES, 4, 4, 3):
            failures.append(f"shape {tuple(tensor.shape)} != ({N_FRAMES}, 4, 4, 3)")

        # Each frame's R-channel should match the expected color.
        for i in range(N_FRAMES):
            expected_r = (i * 60 % 256) / 255.0
            actual_r = float(tensor[i, 0, 0, 0])
            if abs(actual_r - expected_r) > 1e-3:
                failures.append(
                    f"frame {i}: R channel {actual_r:.3f} != expected {expected_r:.3f} "
                    f"(ordering broken or wrong fetch)"
                )

        # Wall-clock: parallel should be ~DELAY_S, sequential would be ~N*DELAY_S.
        sequential_lower = N_FRAMES * DELAY_S * 0.7
        if elapsed >= sequential_lower:
            failures.append(
                f"wall clock {elapsed:.2f}s >= sequential lower bound "
                f"{sequential_lower:.2f}s — gather not running in parallel"
            )

        return failures
    finally:
        server.shutdown(); server.server_close()


def main() -> int:
    failures: list[str] = []
    failures += asyncio.run(_test_maybe_externalize())
    failures += asyncio.run(_test_deserialize_output())
    if failures:
        print()
        print("FAIL:")
        for f in failures:
            print(" -", f)
        return 1
    print("PASS: maybe_externalize dispatches on encoding + rejects empty frames; "
          "_deserialize_output handles png_base64_batch_uri in parallel")
    return 0


if __name__ == "__main__":
    sys.exit(main())
