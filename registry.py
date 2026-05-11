"""ComfyExtension entry point — handshake with the RNP server and
register dynamic node classes with ComfyUI.

Reads the server URL from ``RNP_SERVER_URL``. When the env var is unset
the extension registers zero nodes and logs a single "no server
configured" line — never crashes ComfyUI startup.

If the server is unreachable or the manifest/object_info fetch fails,
the extension degrades silently with a log line so a flaky deployment
doesn't take ComfyUI down with it.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from comfy_api.latest import ComfyExtension, IO
from typing_extensions import override

from . import client as rnp_client
from .protocol import PROTOCOL_VERSION, RnpProtocolError
from .proxy_node import build_node_class

log = logging.getLogger("comfy_remote_nodes")


SERVER_URL_ENV = "RNP_SERVER_URL"


class _RemoteNodesExtension(ComfyExtension):
    """Concrete ComfyExtension that pulls node descriptors from one
    RNP server URL and exposes the resulting dynamic classes to
    ComfyUI."""

    def __init__(self, server_url: str) -> None:
        self._server_url = server_url
        self._node_classes: list[type[IO.ComfyNode]] = []
        # ETag + max-age from the latest object_info fetch — used by a
        # future re-discovery loop to do conditional GETs.
        self._object_info_etag: str | None = None
        self._object_info_max_age: int | None = None
        # Inline-payload cap from the manifest. Populated in on_load().
        self._max_inline_bytes: int | None = None

    @override
    async def on_load(self) -> None:
        try:
            manifest = await rnp_client.fetch_manifest(self._server_url)
        except RnpProtocolError as e:
            log.warning(
                "RNP manifest fetch from %s failed (%s): %s",
                self._server_url, e.code, e,
            )
            return
        except Exception as e:  # noqa: BLE001
            log.warning(
                "RNP manifest fetch from %s failed: %s",
                self._server_url, e,
            )
            return

        server_proto = manifest.get("protocol_version") or ""
        if not _major_compatible(server_proto, PROTOCOL_VERSION):
            log.warning(
                "RNP server %s advertises protocol %s; client speaks %s — skipping",
                self._server_url, server_proto, PROTOCOL_VERSION,
            )
            return
        # Cap on inline base64 payloads in /execute requests. Envelopes
        # larger than this are uploaded via /rnp/v1/uploads and the
        # payload is rewritten to a download URL. ``None`` disables.
        self._max_inline_bytes = manifest.get("max_inline_payload_bytes")
        log.info(
            "RNP server %s ok: provider=%s protocol=%s max_inline_bytes=%s",
            self._server_url,
            (manifest.get("provider") or {}).get("id"),
            server_proto,
            self._max_inline_bytes,
        )

        try:
            object_info, etag, max_age = await rnp_client.fetch_object_info(
                self._server_url
            )
        except RnpProtocolError as e:
            log.warning(
                "RNP object_info fetch from %s failed (%s): %s",
                self._server_url, e.code, e,
            )
            return
        except Exception as e:  # noqa: BLE001
            log.warning(
                "RNP object_info fetch from %s failed: %s",
                self._server_url, e,
            )
            return

        # First-load case: 304 should not happen because we sent no
        # If-None-Match. Defensive check anyway.
        if object_info is None:
            log.warning(
                "RNP object_info from %s returned 304 with no cached body",
                self._server_url,
            )
            return

        # Stash the cache state on the extension instance so a future
        # rediscovery loop can issue cheap conditional GETs against
        # ``/object_info``.
        self._object_info_etag = etag
        self._object_info_max_age = max_age
        log.debug(
            "RNP object_info etag=%s max_age=%s",
            etag, max_age,
        )

        for node_id, descriptor in object_info.items():
            if not isinstance(descriptor, dict):
                log.warning("Skipping %s: descriptor is not a dict", node_id)
                continue
            try:
                node_cls = build_node_class(
                    node_id,
                    descriptor,
                    self._server_url,
                    max_inline_bytes=self._max_inline_bytes,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("Failed to build remote node %s: %s", node_id, e)
                continue
            if node_cls is not None:
                self._node_classes.append(node_cls)
                log.info("Registered remote node: %s", node_id)
        log.info(
            "comfy_remote_nodes ready: %d node(s) loaded from %s",
            len(self._node_classes), self._server_url,
        )

    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return list(self._node_classes)


def _major_compatible(server_version: str, client_version: str) -> bool:
    """True iff the major component of two semver strings matches.

    A major-version mismatch is a hard reject; minor / patch
    differences are tolerated (additive evolution only).
    """
    if not server_version or not client_version:
        return False
    try:
        return server_version.split(".")[0] == client_version.split(".")[0]
    except Exception:  # noqa: BLE001
        return False


async def comfy_entrypoint() -> ComfyExtension:
    """ComfyUI calls this at custom-node load time."""
    server_url = (os.environ.get(SERVER_URL_ENV) or "").strip()
    if not server_url:
        log.info(
            "comfy_remote_nodes: %s not set — no remote nodes will be registered",
            SERVER_URL_ENV,
        )
        return _NullExtension()
    log.info("comfy_remote_nodes: connecting to RNP server %s", server_url)
    return _RemoteNodesExtension(server_url)


class _NullExtension(ComfyExtension):
    """Used when no server URL is configured — registers zero nodes."""

    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return []
