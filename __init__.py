"""comfy_remote_nodes — ComfyUI client for the Remote Node Protocol (RNP/1).

* On boot, fetches ``GET /rnp/v1/manifest`` then
  ``GET /rnp/v1/object_info`` from the server URL set via
  ``RNP_SERVER_URL``.
* Builds dynamic V3 ``IO.ComfyNode`` subclasses for each descriptor
  with a ``remote.endpoints.execute`` entry and registers them with
  ComfyUI through the standard ``ComfyExtension`` interface.
* Only ``execute`` is wired today; ``validate_inputs``,
  ``fingerprint_inputs`` and ``check_lazy_status`` fall through to
  ComfyUI's defaults.
"""
from __future__ import annotations

from .registry import comfy_entrypoint  # noqa: F401  — exposed to ComfyUI
