"""Build dynamic V3 ``IO.ComfyNode`` subclasses from RNP NodeDescriptors.

The descriptor wire format is V3 ``Schema.get_v1_info()`` verbatim, so
per-input dicts are handed straight to V3's IO classes without
reinterpretation. ``_parse_input_spec`` knows STRING / INT / FLOAT /
BOOLEAN / COMBO / IMAGE / VIDEO / AUDIO / MASK; any other io_type
string falls through to ``IO.Custom(io_type)`` (the §B opaque
pass-through bucket — partner helper-config types like RECRAFT_*,
OPENAI_INPUT_FILES, OPENAI_CHAT_CONFIG, GEMINI_INPUT_FILES round-trip
as raw JSON blobs with the original io_type preserved on the V3
socket, so node-to-node connections only chain when the strings
match). Only a non-string / empty io_type causes a skip.

Hidden inputs (``auth_token_comfy_org`` / ``api_key_comfy_org`` /
``unique_id`` / ``prompt`` / ``extra_pnginfo`` / ``dynprompt``) are
declared on the schema so the executor populates ``cls.hidden``; auth
credentials are then lifted into outbound HTTP headers and everything
else flows into the request ``context`` map.

Only ``execute`` is wired today; the validate / fingerprint /
check_lazy_status RNP methods fall through to ComfyUI's defaults until
their server endpoints exist.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import time
import uuid
from typing import Any

from comfy_api.latest import IO
from server import PromptServer

from comfy_api_nodes.util.client import ApiEndpoint, poll_op_raw
from comfy_api_nodes.util.common_exceptions import ProcessingInterrupted

from . import client as rnp_client
from . import serialization
from .protocol import (
    DEFAULT_CANCEL_TIMEOUT_S,
    DEFAULT_HARD_TIMEOUT_S,
    DEFAULT_MAX_RETRIES_PER_POLL,
    DEFAULT_POLL_GRACE_S,
    DEFAULT_POLL_INTERVAL_S,
    DEFAULT_RETRY_BACKOFF_PER_POLL,
    DEFAULT_RETRY_DELAY_PER_POLL_S,
    DEFAULT_TIMEOUT_PER_POLL_S,
    ErrorCode,
    ExecutionMode,
    Header,
    PROTOCOL_VERSION,
    RnpProtocolError,
    TaskStatus,
    is_envelope,
)


# ComfyUI's frontend matches these strings literally on
# ``exception_message.includes(...)`` to fire dedicated dialogs, so the
# wording here must stay byte-identical with
# ``comfy_api_nodes.util.client._friendly_http_message``.
_AUTH_FAILED_MESSAGE = "Unauthorized: Please login first to use this node."
_INSUFFICIENT_CREDITS_MESSAGE = (
    "Payment Required: Please add credits to your account to use this node."
)
_ACCOUNT_PROBLEM_MESSAGE = (
    "There is a problem with your account. Please contact support@comfy.org."
)
_RATE_LIMIT_AFTER_RETRY_MESSAGE = (
    "Rate Limit Exceeded: The server returned 429 after all retry attempts. "
    "Please wait and try again."
)
_PROVIDER_UNAVAILABLE_MESSAGE = (
    "The remote node provider is temporarily unavailable. Please try again "
    "in a few minutes."
)
_SERVER_BUSY_MESSAGE = (
    "The remote node server is at capacity. Please try again in a moment."
)
_MAINTENANCE_MESSAGE = (
    "The remote node server is in maintenance. Please try again shortly."
)
_TASK_LOST_MESSAGE = (
    "The remote node server lost the running task and the resume attempt "
    "did not succeed. Please re-run the workflow."
)


def _to_friendly_message(e: RnpProtocolError, node_id: str) -> str:
    upstream = (e.details or {}).get("upstream_status") if e.details else None
    if e.code == ErrorCode.AUTH_FAILED or upstream in (401, 403):
        return _AUTH_FAILED_MESSAGE
    if e.code == ErrorCode.INSUFFICIENT_CREDITS or upstream == 402:
        return _INSUFFICIENT_CREDITS_MESSAGE
    # 409 from the upstream gateway means "account in bad standing".
    # comfy_api_nodes.util.client._friendly_http_message uses a dedicated
    # support-contact prompt for this; mirror it byte-identically so the
    # same frontend dialog fires.
    if upstream == 409:
        return _ACCOUNT_PROBLEM_MESSAGE
    # 429 reaches the client only after the server has exhausted its own
    # rate-limit retries — surface the post-retry copy so users don't
    # think a single transient 429 leaked through.
    if e.code == ErrorCode.RATE_LIMITED or upstream == 429:
        return _RATE_LIMIT_AFTER_RETRY_MESSAGE
    if e.code == ErrorCode.SERVER_BUSY:
        return _SERVER_BUSY_MESSAGE
    if e.code == ErrorCode.MAINTENANCE:
        # Server-friendly message often embeds an "until" timestamp in
        # the wire ``message`` — prefer it over the canned copy.
        return str(e) if e.user_facing else _MAINTENANCE_MESSAGE
    if e.code == ErrorCode.TASK_LOST:
        return _TASK_LOST_MESSAGE
    # PROVIDER_UNAVAILABLE with a retryable hint or 5xx upstream
    # collapses to the generic "try again" copy.
    if e.code == ErrorCode.PROVIDER_UNAVAILABLE and (
        upstream in (502, 503, 504) or e.retryable
    ):
        return _PROVIDER_UNAVAILABLE_MESSAGE
    if e.user_facing:
        return str(e)
    return f"Remote node {node_id} failed ({e.code})"


def _coerce_pos(value: Any, default: float) -> float:
    """Best-effort numeric coerce — falls back to ``default`` for anything
    unparseable or non-positive (the descriptor came over the wire so we
    don't trust it)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float(default)
    return f if f > 0 else float(default)


def _coerce_pos_int(value: Any, default: int) -> int:
    try:
        i = int(value)
    except (TypeError, ValueError):
        return int(default)
    return i if i > 0 else int(default)


def _extract_execution_policy(execution: dict[str, Any]) -> dict[str, Any]:
    """Lift ``descriptor.remote.execution`` policy fields into a flat dict
    of ``poll_op_raw`` kwargs.

    Defaults mirror ``comfy_api_nodes.util.client.poll_op_raw`` so an RNP
    node with an empty ``execution`` block behaves identically to a
    direct upstream call. The schema is the wire contract from
    ``comfy_rnp_protocol.constants``: ``poll_interval_s``,
    ``soft_timeout_s``, ``hard_timeout_s``, ``timeout_per_poll_s``,
    ``cancel_timeout_s``, ``max_retries_per_poll``,
    ``retry_delay_per_poll_s``, ``retry_backoff_per_poll``,
    ``retry: {max, delay_s, backoff, retry_on}``, ``idempotency``.

    The client's poll-loop cap (``max_poll_attempts``) is **derived**
    from ``hard_timeout_s + DEFAULT_POLL_GRACE_S`` and
    ``poll_interval_s`` — there is no wire field for it. Coupling a
    contract bound to a cadence value silently breaks every time you
    tune the cadence; expressing the bound as a duration keeps the
    contract invariant under cadence changes (see protocol.py
    "Lifetime budget design" comment).
    """
    retry_block = execution.get("retry") if isinstance(execution.get("retry"), dict) else {}
    poll_interval_s = _coerce_pos(
        execution.get("poll_interval_s"), DEFAULT_POLL_INTERVAL_S,
    )
    hard_timeout_s = _coerce_pos(
        execution.get("hard_timeout_s"),
        _coerce_pos(execution.get("soft_timeout_s"), DEFAULT_HARD_TIMEOUT_S),
    )
    poll_budget_s = hard_timeout_s + DEFAULT_POLL_GRACE_S
    max_poll_attempts = max(1, math.ceil(poll_budget_s / poll_interval_s))
    return {
        "poll_interval_s": poll_interval_s,
        "max_poll_attempts": max_poll_attempts,
        "timeout_per_poll_s": _coerce_pos(
            execution.get("timeout_per_poll_s"), DEFAULT_TIMEOUT_PER_POLL_S,
        ),
        "cancel_timeout_s": _coerce_pos(
            execution.get("cancel_timeout_s"), DEFAULT_CANCEL_TIMEOUT_S,
        ),
        "max_retries_per_poll": _coerce_pos_int(
            execution.get("max_retries_per_poll"), DEFAULT_MAX_RETRIES_PER_POLL,
        ),
        "retry_delay_per_poll_s": _coerce_pos(
            execution.get("retry_delay_per_poll_s"), DEFAULT_RETRY_DELAY_PER_POLL_S,
        ),
        "retry_backoff_per_poll": _coerce_pos(
            execution.get("retry_backoff_per_poll"), DEFAULT_RETRY_BACKOFF_PER_POLL,
        ),
        # Submit-side retry block — currently unused by execute_remote /
        # execute_remote_async (they hard-code max_retries=2 in
        # rnp_client) but lifted out here so future steps can plumb it
        # through without re-parsing the descriptor.
        "submit_retry": {
            "max": _coerce_pos_int(retry_block.get("max"), 0)
            if retry_block.get("max") is not None else 0,
            "delay_s": _coerce_pos(retry_block.get("delay_s"), 1.0),
            "backoff": _coerce_pos(retry_block.get("backoff"), 2.0),
            "retry_on": list(retry_block.get("retry_on") or []),
        },
        "idempotency": execution.get("idempotency") or "none",
    }


def _node_id_for_ticker(cls: type[IO.ComfyNode]) -> str | None:
    """Lift ``cls.hidden.unique_id`` for the per-second progress ticker.

    Returns ``None`` if the hidden holder isn't populated (test paths
    that build a class but don't run it through the executor).
    """
    hidden = getattr(cls, "hidden", None)
    if hidden is None:
        return None
    uid = getattr(hidden, "unique_id", None)
    return str(uid) if uid is not None else None


async def _progress_ticker(
    node_id: str | None,
    estimated_duration_s: int | float | None,
    *,
    status_override: dict[str, Any] | None = None,
) -> None:
    """Emit one ``send_progress_text`` per second until cancelled.

    Format mirrors ``comfy_api_nodes.util.client._display_time_progress``
    so the progress badge looks identical to local partner nodes:
    ``Status: Processing\\nTime elapsed: <N>s`` (with
    ``(~Ns remaining)`` when the descriptor declared an estimate).

    ``status_override`` is a shared mutable holder
    ``{"label": str|None, "expires_at": float|None}`` the caller can
    mutate to swap "Processing" for a transient message like
    "Server busy, retrying in 30s..." while a backpressure 503 is
    being waited out. The ticker auto-clears the override once
    ``time.monotonic() >= expires_at`` so a stale label doesn't linger
    if the post-wait request takes longer than the announced
    Retry-After window.
    """
    if node_id is None:
        return
    started = time.monotonic()
    while True:
        elapsed = int(time.monotonic() - started)
        if estimated_duration_s is not None and estimated_duration_s > 0:
            remaining = max(0, int(estimated_duration_s) - elapsed)
            time_line = f"Time elapsed: {elapsed}s (~{remaining}s remaining)"
        else:
            time_line = f"Time elapsed: {elapsed}s"
        label = "Processing"
        if status_override is not None:
            expires_at = status_override.get("expires_at")
            if expires_at is not None and time.monotonic() >= expires_at:
                status_override["label"] = None
                status_override["expires_at"] = None
            override = status_override.get("label")
            if override:
                label = override
        text = f"Status: {label}\n{time_line}"
        with contextlib.suppress(Exception):
            instance = PromptServer.instance
            if instance is not None:
                instance.send_progress_text(text, node_id)
        try:
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            return

log = logging.getLogger("comfy_remote_nodes.proxy_node")


def build_node_class(
    node_id: str,
    descriptor: dict[str, Any],
    server_url: str,
    *,
    max_inline_bytes: int | None = None,
) -> type[IO.ComfyNode] | None:
    """Build an ``IO.ComfyNode`` subclass for one descriptor.

    Returns ``None`` if the descriptor can't be parsed (e.g. an input
    type ``_parse_input_spec`` doesn't recognise). The caller logs and
    skips the node so one bad descriptor doesn't break the whole
    extension.
    """
    remote = descriptor.get("remote") or {}
    if not isinstance(remote, dict):
        log.warning("Descriptor %s missing 'remote' block — skipping", node_id)
        return None
    endpoints = remote.get("endpoints") or {}
    execute_endpoint = endpoints.get("execute")
    if not isinstance(execute_endpoint, dict) or not execute_endpoint.get("path"):
        log.warning("Descriptor %s has no remote.endpoints.execute — skipping", node_id)
        return None

    inputs, hidden_names = _parse_inputs(descriptor)
    if inputs is None:
        return None
    outputs = _parse_outputs(descriptor)
    if outputs is None:
        return None

    schema_hash = remote.get("schema_hash")
    execute_path = execute_endpoint["path"]
    execution = remote.get("execution") or {}
    # ``descriptor.remote.input_serialization`` advertises the wire
    # encodings each input accepts. The wire shape is
    # ``{input_name: <str | list[str]>}`` — single string for a
    # single-encoding input (e.g. ``"png_base64"``), list when the
    # input accepts multiple encodings (e.g.
    # ``["png_base64", "png_base64_batch"]`` for a multi-frame IMAGE).
    # Normalize to ``dict[str, list[str]]`` so the encoder can do a
    # single ``in`` check per input without re-parsing per call.
    raw_input_serialization = (
        remote.get("input_serialization") if isinstance(remote.get("input_serialization"), dict) else {}
    )
    input_serialization: dict[str, list[str]] = {}
    for name, val in raw_input_serialization.items():
        if isinstance(val, str):
            input_serialization[name] = [val]
        elif isinstance(val, list):
            input_serialization[name] = [v for v in val if isinstance(v, str)]

    # ``descriptor.input.<section>.<name>[1].local_validate`` carries
    # declarative client-side validation rules (e.g.
    # ``{"image_max_batch": 9}`` for a Flux2 reference-image input).
    # The encoder consults this map and raises ``INPUT_INVALID``
    # locally before any HTTP round trip when an over-cap batch is
    # wired in. Lookup is defensive: missing ``local_validate`` →
    # no validation; missing per-rule key → no validation.
    local_validate = _collect_local_validate(descriptor)
    timeout_s = float(
        execution.get("hard_timeout_s")
        or execution.get("soft_timeout_s")
        or 600.0
    )
    estimated_duration_s = execution.get("estimated_duration_s")
    if not isinstance(estimated_duration_s, (int, float)):
        estimated_duration_s = None
    execution_mode = execution.get("mode") or ExecutionMode.REQUEST_RESPONSE
    execution_policy = _extract_execution_policy(execution)
    # ``descriptor.remote.url_fetch`` declares per-output-type fetch
    # posture for envelopes that arrive with ``uri`` instead of inline
    # ``data``. Empty / absent → all outputs must inline their bytes.
    url_fetch_policy = (
        remote.get("url_fetch") if isinstance(remote.get("url_fetch"), dict) else {}
    )

    display_name = descriptor.get("display_name") or node_id
    category = descriptor.get("category") or "remote"
    description = descriptor.get("description") or ""
    is_api_node = bool(descriptor.get("api_node", False))
    is_experimental = bool(descriptor.get("experimental", False))
    is_deprecated = bool(descriptor.get("deprecated", False))

    schema_inputs = list(inputs)
    schema_outputs = list(outputs)
    hidden_decls = _hidden_decls_from_names(hidden_names, is_api_node)

    # Forward every other V3 NodeInfoV1 field the descriptor surfaces.
    # Anything not constructed here is silently dropped; the
    # ``_parse_input_spec`` / ``_parse_outputs`` warnings are the
    # signal a deployment needs to add a missing type to this builder.
    is_input_list = bool(descriptor.get("is_input_list", False))
    is_output_node = bool(descriptor.get("output_node", False))
    is_dev_only = bool(descriptor.get("dev_only", False))
    has_intermediate_output = bool(descriptor.get("has_intermediate_output", False))
    search_aliases = descriptor.get("search_aliases") or []
    essentials_category = descriptor.get("essentials_category")
    price_badge = _build_price_badge(descriptor.get("price_badge"))

    class RemoteProxyNode(IO.ComfyNode):
        @classmethod
        def define_schema(cls) -> IO.Schema:
            return IO.Schema(
                node_id=node_id,
                display_name=display_name,
                category=category,
                description=description,
                inputs=schema_inputs,
                outputs=schema_outputs,
                hidden=hidden_decls,
                is_input_list=is_input_list,
                is_output_node=is_output_node,
                is_api_node=is_api_node,
                is_experimental=is_experimental,
                is_deprecated=is_deprecated,
                is_dev_only=is_dev_only,
                has_intermediate_output=has_intermediate_output,
                search_aliases=list(search_aliases),
                essentials_category=essentials_category,
                price_badge=price_badge,
            )

        @classmethod
        async def execute(cls, **kwargs: Any) -> IO.NodeOutput:
            return await _execute_remote(
                cls,
                node_id=node_id,
                server_url=server_url,
                execute_path=execute_path,
                schema_hash=schema_hash,
                timeout_s=timeout_s,
                outputs_meta=schema_outputs,
                hidden_names=hidden_names,
                inputs=kwargs,
                max_inline_bytes=max_inline_bytes,
                estimated_duration_s=estimated_duration_s,
                execution_mode=execution_mode,
                execution_policy=execution_policy,
                url_fetch_policy=url_fetch_policy,
                input_serialization=input_serialization,
                local_validate=local_validate,
            )

    RemoteProxyNode.__name__ = node_id
    RemoteProxyNode.__qualname__ = node_id
    RemoteProxyNode.RELATIVE_PYTHON_MODULE = "custom_nodes.comfy_remote_nodes"
    return RemoteProxyNode


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

# The hidden input markers V3 emits in get_v1_info() — clients forward
# their values through the request ``context`` instead of treating them
# as user-facing inputs.
_HIDDEN_MARKERS = {
    "AUTH_TOKEN_COMFY_ORG": IO.Hidden.auth_token_comfy_org,
    "API_KEY_COMFY_ORG":    IO.Hidden.api_key_comfy_org,
    "UNIQUE_ID":            IO.Hidden.unique_id,
    "PROMPT":               IO.Hidden.prompt,
    "EXTRA_PNGINFO":        IO.Hidden.extra_pnginfo,
    "DYNPROMPT":            IO.Hidden.dynprompt,
}


def _collect_local_validate(descriptor: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Pull ``local_validate`` rule dicts off every required/optional input
    spec in the descriptor, recursing into AUTOGROW templates and
    DYNAMIC_COMBO branches.

    Returns ``{input_name: rules_dict}`` for every input that carries a
    non-empty ``local_validate`` block (or whose nested template /
    branches do). The returned ``rules_dict`` may carry the literal
    ``local_validate`` keys (e.g. ``{"image_max_batch": 9}``) plus two
    private composite keys consumed by ``_enforce_local_validate``:

    * ``__template__`` — for AUTOGROW inputs, the rules harvested from
      the template's options dict. The enforcer applies these to every
      slot value in the runtime Autogrow dict
      (``{"image0": <env>, "image1": <env>, ...}``).
    * ``__branches__`` — for DYNAMIC_COMBO inputs, a
      ``{branch_key: {sub_name: rules}}`` map. The enforcer reads the
      selected branch from ``value[<name>]`` and walks the matching
      branch's sub-inputs (used by GPT Image V2 / Grok ImageEditV2 /
      Grok VideoReference, where the cap lives on a nested Autogrow
      template inside a DynamicCombo branch).

    Inputs whose entire subtree is empty are omitted so the encoder can
    use a single ``in`` check per call.
    """
    out: dict[str, dict[str, Any]] = {}
    input_def = descriptor.get("input") or {}
    for kind in ("required", "optional"):
        defs = input_def.get(kind) or {}
        for name, spec in defs.items():
            rules = _rules_from_input_spec(spec)
            if rules:
                out[name] = rules
    return out


def _rules_from_input_spec(spec: Any) -> dict[str, Any]:
    """Extract ``local_validate`` rules from one V3 input spec.

    Accepts both wire shapes:

    * ``[io_type, options_dict]`` — top-level inputs (the descriptor's
      ``input.required[name]`` value, or an AUTOGROW template).
    * ``[name, io_type, options_dict]`` — DynamicCombo branch sub-inputs
      (each entry of ``options.options[i].inputs``).

    For ``AUTOGROW`` specs, recurses into ``options["template"]`` and
    stashes the result under ``__template__``. For ``DYNAMIC_COMBO``,
    walks each branch's sub-inputs and stashes the resulting
    ``{branch_key: {sub_name: sub_rules}}`` map under ``__branches__``.
    """
    if not isinstance(spec, (list, tuple)) or len(spec) < 2:
        return {}
    if (
        len(spec) >= 3
        and isinstance(spec[1], str)
        and isinstance(spec[2], dict)
    ):
        # [name, io_type, options]
        io_type, options = spec[1], spec[2]
    elif isinstance(spec[1], dict):
        # [io_type, options]
        io_type, options = spec[0], spec[1]
    else:
        return {}
    rules: dict[str, Any] = {}
    direct = options.get("local_validate")
    if isinstance(direct, dict) and direct:
        rules.update(direct)
    if io_type == "AUTOGROW":
        tmpl_spec = options.get("template")
        tmpl_rules = (
            _rules_from_input_spec(tmpl_spec)
            if isinstance(tmpl_spec, (list, tuple))
            else {}
        )
        if tmpl_rules:
            rules["__template__"] = tmpl_rules
    elif io_type == "DYNAMIC_COMBO":
        branches: dict[str, dict[str, Any]] = {}
        for opt in (options.get("options") or []):
            if not isinstance(opt, dict):
                continue
            key = opt.get("key")
            if not isinstance(key, str):
                continue
            sub_rules: dict[str, dict[str, Any]] = {}
            for sub_spec in (opt.get("inputs") or []):
                if (
                    not isinstance(sub_spec, (list, tuple))
                    or len(sub_spec) < 1
                    or not isinstance(sub_spec[0], str)
                ):
                    continue
                sub_name = sub_spec[0]
                r = _rules_from_input_spec(sub_spec)
                if r:
                    sub_rules[sub_name] = r
            if sub_rules:
                branches[key] = sub_rules
        if branches:
            rules["__branches__"] = branches
    return rules


def _parse_inputs(descriptor: dict[str, Any]) -> tuple[list[Any] | None, list[str]]:
    """Return ``(input_objs, hidden_input_names)``.

    Returns ``(None, [])`` if any required/optional input can't be
    parsed (caller logs and skips).
    """
    input_def = descriptor.get("input") or {}
    input_order = descriptor.get("input_order") or {}
    out: list[Any] = []
    for kind, optional in (("required", False), ("optional", True)):
        order = input_order.get(kind) or list((input_def.get(kind) or {}).keys())
        defs = input_def.get(kind) or {}
        for name in order:
            spec = defs.get(name)
            if spec is None:
                continue
            inp = _parse_input_spec(name, spec, optional=optional)
            if inp is None:
                log.warning(
                    "Skipping descriptor: unsupported input %r (%s)",
                    name, spec,
                )
                return None, []
            out.append(inp)
    hidden_names = list((input_def.get("hidden") or {}).keys())
    return out, hidden_names


def _parse_input_spec(name: str, spec: list[Any], optional: bool) -> Any | None:
    """Map one V3-shaped ``[io_type, options_dict]`` entry to a V3 IO Input."""
    if not isinstance(spec, (list, tuple)) or len(spec) < 1:
        return None
    io_type = spec[0]
    options: dict[str, Any] = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    common = {
        "tooltip": options.get("tooltip"),
        "optional": optional,
    }
    if options.get("advanced") is not None:
        common["advanced"] = bool(options.get("advanced"))

    if isinstance(io_type, list):
        # Legacy V1-style combo: io_type *is* the options list.
        return IO.Combo.Input(name, options=list(io_type),
                              default=options.get("default"), **common)
    if io_type == "COMBO":
        opts = options.get("options")
        if not isinstance(opts, (list, tuple)):
            return None
        return IO.Combo.Input(name, options=list(opts),
                              default=options.get("default"), **common)
    if io_type == "STRING":
        return IO.String.Input(
            name,
            default=options.get("default", ""),
            multiline=bool(options.get("multiline", False)),
            **common,
        )
    if io_type == "INT":
        return IO.Int.Input(
            name,
            default=options.get("default", 0),
            min=options.get("min", 0),
            max=options.get("max", 2147483647),
            step=options.get("step", 1),
            **common,
        )
    if io_type == "FLOAT":
        return IO.Float.Input(
            name,
            default=options.get("default", 0.0),
            min=options.get("min", 0.0),
            max=options.get("max", 1.0),
            step=options.get("step", 0.01),
            **common,
        )
    if io_type == "BOOLEAN":
        return IO.Boolean.Input(
            name, default=bool(options.get("default", False)), **common,
        )
    if io_type == "IMAGE":
        return IO.Image.Input(name, **common)
    if io_type == "VIDEO":
        return IO.Video.Input(name, **common)
    if io_type == "AUDIO":
        return IO.Audio.Input(name, **common)
    if io_type == "MASK":
        return IO.Mask.Input(name, **common)
    # Opaque custom-IO fallthrough — any unknown io_type string becomes
    # an ``IO.Custom(io_type)`` socket so partner helper-config nodes
    # (OpenAIInputFiles → OPENAI_INPUT_FILES, OpenAIChatConfig →
    # OPENAI_CHAT_CONFIG, GeminiInputFiles → GEMINI_INPUT_FILES, the
    # already-shipped Recraft RECRAFT_* helpers, future partner types)
    # round-trip without per-type encoder plumbing. The descriptor's
    # original ``io_type`` string is preserved on ``IO.Custom``'s
    # ComfyType.io_type via the V3 ``@comfytype`` decorator, so the
    # frontend's connection-validity check only chains sockets when the
    # strings match — an OPENAI_INPUT_FILES output cannot connect to a
    # GEMINI_INPUT_FILES input. Encoder dispatch in
    # ``serialization`` already passes non-tensor / non-audio values
    # through ``_encode_one`` unchanged, and the deserializer treats
    # non-envelope values as already-native Python objects.
    if isinstance(io_type, str) and io_type:
        return IO.Custom(io_type).Input(name, **common)
    return None


def _build_price_badge(wire: Any) -> Any | None:
    """Convert the wire-format ``price_badge`` dict back to an
    ``IO.PriceBadge`` instance.

    The wire shape (from ``PriceBadge.as_dict()``) carries
    ``depends_on.widgets`` as a list of ``{name, type}`` objects so the
    frontend can resolve types; the V3 constructor only needs the
    ``name`` strings since it re-derives types from the schema's actual
    inputs at ``as_dict()`` time. Inputs / input_groups round-trip as-is.

    Returns ``None`` if the descriptor has no ``price_badge`` field.
    """
    if not isinstance(wire, dict):
        return None
    expr = wire.get("expr")
    if not isinstance(expr, str) or not expr.strip():
        return None
    dep = wire.get("depends_on") or {}
    widgets = [
        w["name"] for w in (dep.get("widgets") or []) if isinstance(w, dict) and "name" in w
    ]
    inputs = list(dep.get("inputs") or [])
    input_groups = list(dep.get("input_groups") or [])
    engine = wire.get("engine", "jsonata")
    try:
        return IO.PriceBadge(
            expr=expr,
            depends_on=IO.PriceBadgeDepends(
                widgets=widgets, inputs=inputs, input_groups=input_groups,
            ),
            engine=engine,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("Skipping price_badge — could not construct: %s", e)
        return None


def _hidden_decls_from_names(hidden_names: list[str], is_api_node: bool) -> list[Any]:
    """Build the ``hidden=`` list for IO.Schema.

    The descriptor lists each hidden input by name (e.g.
    ``auth_token_comfy_org``); the ``_HIDDEN_MARKERS`` table above
    resolves those names back to V3 ``IO.Hidden`` enum members. Names
    that aren't in the table are dropped with a debug log so a server
    that adds a new hidden marker doesn't break older clients.
    """
    out: list[Any] = []
    seen: set[Any] = set()
    for name in hidden_names:
        marker = getattr(IO.Hidden, name, None)
        if marker is None:
            log.debug("Unknown hidden input %s — dropping", name)
            continue
        if marker not in seen:
            out.append(marker)
            seen.add(marker)
    # IO.Schema.finalize() will add auth/api hidden if is_api_node, so
    # we don't need to add them again here.
    if not is_api_node:
        # Non-api-node descriptors that happened to advertise auth
        # hidden are forwarded as-is.
        pass
    return out


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _parse_outputs(descriptor: dict[str, Any]) -> list[Any] | None:
    output_types = descriptor.get("output") or []
    output_names = descriptor.get("output_name") or output_types
    output_is_list = descriptor.get("output_is_list") or [False] * len(output_types)
    output_tooltips = descriptor.get("output_tooltips") or [None] * len(output_types)
    out: list[Any] = []
    for i, io_type in enumerate(output_types):
        name = output_names[i] if i < len(output_names) else io_type
        is_list = bool(output_is_list[i]) if i < len(output_is_list) else False
        tip = output_tooltips[i] if i < len(output_tooltips) else None
        cls = _OUTPUT_CLASSES.get(io_type)
        if cls is None:
            # Opaque custom-IO fallthrough (mirror of
            # ``_parse_input_spec``) — any unknown output io_type
            # string becomes ``IO.Custom(io_type).Output`` so partner
            # helper-config nodes can publish their own opaque
            # outputs (RECRAFT_*, OPENAI_INPUT_FILES, GEMINI_INPUT_FILES,
            # OPENAI_CHAT_CONFIG, …). The original io_type string
            # rides on the resulting Output via V3's ``@comfytype``
            # decorator, so the frontend only chains output→input
            # sockets when the strings match.
            if isinstance(io_type, str) and io_type:
                cls = IO.Custom(io_type).Output
            else:
                log.warning(
                    "Skipping descriptor: unsupported output type %r",
                    io_type,
                )
                return None
        out.append(cls(display_name=name, is_output_list=is_list, tooltip=tip))
    return out


_OUTPUT_CLASSES = {
    "VIDEO":   IO.Video.Output,
    "IMAGE":   IO.Image.Output,
    "AUDIO":   IO.Audio.Output,
    "MASK":    IO.Mask.Output,
    "STRING":  IO.String.Output,
    "INT":     IO.Int.Output,
    "FLOAT":   IO.Float.Output,
    "BOOLEAN": IO.Boolean.Output,
}


# ---------------------------------------------------------------------------
# Execute dispatch
# ---------------------------------------------------------------------------

def _extract_progress_pct(response: dict[str, Any]) -> int | None:
    """Map the RNP ``progress`` block to the 0-100 int that
    ``poll_op_raw`` feeds to its (single) ``ProgressBar(100)``.

    Returns ``None`` when the server doesn't yet have progress to
    report so the bar stays at its current position rather than
    snapping back to 0.
    """
    progress = response.get("progress")
    if not isinstance(progress, dict):
        return None
    value = progress.get("value")
    total = progress.get("max")
    if not isinstance(value, (int, float)) or not isinstance(total, (int, float)) or total <= 0:
        return None
    pct = int(round((float(value) / float(total)) * 100))
    return max(0, min(100, pct))


def _rnp_protocol_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Headers every RNP request carries — protocol-version + client
    version. Pass ``extra`` to merge per-request headers (e.g. the
    Idempotency-Key the server uses to authorize a TASK_LOST resume,
    or the freshly-rotated auth token a long-running poll wants to
    forward to the server)."""
    headers = {
        Header.PROTOCOL_VERSION: PROTOCOL_VERSION,
        Header.CLIENT_VERSION: rnp_client.CLIENT_VERSION,
    }
    if extra:
        headers.update(extra)
    return headers


async def _run_async_polling(
    cls: type[IO.ComfyNode],
    *,
    node_id: str,
    server_url: str,
    schema_hash: str | None,
    payload_inputs: dict[str, Any],
    context: dict[str, Any],
    auth_headers: dict[str, str],
    idempotency_key: str,
    poll_interval_s: float,
    estimated_duration_s: int | float | None,
    max_poll_attempts: int,
    timeout_per_poll_s: float,
    cancel_timeout_s: float,
    max_retries_per_poll: int,
    retry_delay_per_poll_s: float,
    retry_backoff_per_poll: float,
) -> dict[str, Any]:
    """Drive the ``execute_async`` + ``poll`` flow via ``poll_op_raw``.

    Submits the task with ``execute_async``, then hands the polling
    loop off to ``comfy_api_nodes.util.client.poll_op_raw`` so RNP
    inherits the same per-poll retry/backoff, queued-vs-active
    accounting, max-poll-attempts cap, ``ProgressBar`` plumbing,
    ticker UI, and cancel-endpoint propagation as local partner
    nodes. All of DONE/ERROR/CANCELLED are mapped to poll_op_raw's
    "completed" set so the function returns the terminal poll-response
    dict and we dispatch on ``status`` here — that preserves
    ``RnpProtocolError`` typing for ERROR and ``ProcessingInterrupted``
    for CANCELLED instead of flattening both to
    ``Exception("Task failed: …")``.

    On ``TASK_LOST`` (the server lost the task store between submit
    and poll, typically a restart), the function resubmits exactly
    once with the same idempotency key and resumes polling. A second
    ``TASK_LOST`` from the resume gives up — at that point the failure
    is structural rather than transient.
    """
    resume_attempted = False
    while True:
        submit = await rnp_client.execute_remote_async(
            server_url,
            schema_hash,
            node_id,
            payload_inputs,
            context,
            extra_headers=auth_headers or None,
            idempotency_key=idempotency_key,
            node_cls=cls,
        )
        task_id = submit["task_id"]
        interval = float(submit.get("poll_interval") or poll_interval_s)
        try:
            response = await _poll_until_terminal(
                cls,
                node_id=node_id,
                server_url=server_url,
                task_id=task_id,
                auth_headers=auth_headers,
                idempotency_key=idempotency_key,
                poll_interval_s=interval,
                estimated_duration_s=estimated_duration_s,
                max_poll_attempts=max_poll_attempts,
                timeout_per_poll_s=timeout_per_poll_s,
                cancel_timeout_s=cancel_timeout_s,
                max_retries_per_poll=max_retries_per_poll,
                retry_delay_per_poll_s=retry_delay_per_poll_s,
                retry_backoff_per_poll=retry_backoff_per_poll,
            )
        except RnpProtocolError as e:
            if e.code == ErrorCode.TASK_LOST and not resume_attempted:
                resume_attempted = True
                _send_progress_text(
                    cls,
                    "Server restarted, resuming…",
                )
                log.warning(
                    "RNP %s: TASK_LOST during poll (task_id=%s); "
                    "resubmitting with idempotency_key=%s",
                    node_id, task_id, idempotency_key,
                )
                continue
            raise
        return _interpret_terminal_response(task_id, response)


def _send_progress_text(cls: type[IO.ComfyNode], text: str) -> None:
    """Best-effort progress badge update — invisible when the executor
    isn't running this class through PromptServer (e.g. unit tests)."""
    node_id = _node_id_for_ticker(cls)
    if node_id is None:
        return
    with contextlib.suppress(Exception):
        instance = PromptServer.instance
        if instance is not None:
            instance.send_progress_text(text, node_id)


async def _poll_until_terminal(
    cls: type[IO.ComfyNode],
    *,
    node_id: str,
    server_url: str,
    task_id: str,
    auth_headers: dict[str, str],
    idempotency_key: str,
    poll_interval_s: float,
    estimated_duration_s: int | float | None,
    max_poll_attempts: int,
    timeout_per_poll_s: float,
    cancel_timeout_s: float,
    max_retries_per_poll: int,
    retry_delay_per_poll_s: float,
    retry_backoff_per_poll: float,
) -> dict[str, Any]:
    """Run ``poll_op_raw`` against an existing task_id and return the
    terminal poll-response dict. Wraps the per-request header build so
    the resume path in ``_run_async_polling`` can reuse it."""
    # Forward the idempotency key on every poll/cancel: the server
    # uses it to gate TASK_LOST recovery (a typo'd task_id without an
    # idempotency key falls through to a generic NOT_FOUND so it
    # cannot trigger a phantom resubmit). Auth headers ride alongside
    # so a token rotated mid-task picks up on the next upstream call
    # the server makes.
    per_request_headers = _rnp_protocol_headers({
        Header.IDEMPOTENCY_KEY: idempotency_key,
        **(auth_headers or {}),
    })
    poll_endpoint = ApiEndpoint(
        f"rnp/v1/tasks/{task_id}", "GET",
        headers=per_request_headers,
    )
    cancel_endpoint = ApiEndpoint(
        f"rnp/v1/tasks/{task_id}/cancel", "POST",
        headers=per_request_headers,
    )
    return await poll_op_raw(
        cls,
        poll_endpoint,
        status_extractor=lambda r: r.get("status"),
        progress_extractor=_extract_progress_pct,
        completed_statuses=[TaskStatus.DONE, TaskStatus.ERROR, TaskStatus.CANCELLED],
        failed_statuses=[],
        queued_statuses=[TaskStatus.PENDING],
        poll_interval=poll_interval_s,
        max_poll_attempts=max_poll_attempts,
        timeout_per_poll=timeout_per_poll_s,
        max_retries_per_poll=max_retries_per_poll,
        retry_delay_per_poll=retry_delay_per_poll_s,
        retry_backoff_per_poll=retry_backoff_per_poll,
        estimated_duration=int(estimated_duration_s) if estimated_duration_s else None,
        cancel_endpoint=cancel_endpoint,
        cancel_timeout=cancel_timeout_s,
        base_url=server_url,
        auth_headers={},
        error_parser=rnp_client._parse_rnp_error,
        is_rate_limited=rnp_client._is_rnp_backpressure,
        rate_limit_label=rnp_client._rnp_rate_limit_label,
    )


def _interpret_terminal_response(
    task_id: str, response: dict[str, Any],
) -> dict[str, Any]:
    """Map a terminal poll-response dict to the same ``{"outputs": [...]}``
    shape sync ``execute`` returns, or raise the appropriate typed
    exception for ERROR / CANCELLED."""
    status = response.get("status")
    if status == TaskStatus.DONE:
        outputs = response.get("outputs")
        if not isinstance(outputs, list):
            raise RnpProtocolError(
                f"task {task_id} done but no outputs list: {response!r}",
            )
        return {"outputs": outputs}
    if status == TaskStatus.ERROR:
        err_body = response.get("exception")
        if not isinstance(err_body, dict):
            raise RnpProtocolError(
                f"task {task_id} errored without exception body",
            )
        raise RnpProtocolError.from_response_body(err_body, 500)
    if status == TaskStatus.CANCELLED:
        raise ProcessingInterrupted(f"task {task_id} cancelled")
    raise RnpProtocolError(
        f"task {task_id} returned unrecognised terminal status: {status!r}",
    )


async def _execute_remote(
    cls: type[IO.ComfyNode],
    *,
    node_id: str,
    server_url: str,
    execute_path: str,
    schema_hash: str | None,
    timeout_s: float,
    outputs_meta: list[Any],
    hidden_names: list[str],
    inputs: dict[str, Any],
    max_inline_bytes: int | None = None,
    estimated_duration_s: int | float | None = None,
    execution_mode: str = ExecutionMode.REQUEST_RESPONSE,
    execution_policy: dict[str, Any] | None = None,
    url_fetch_policy: dict[str, Any] | None = None,
    input_serialization: dict[str, list[str]] | None = None,
    local_validate: dict[str, dict[str, Any]] | None = None,
) -> IO.NodeOutput:
    policy = execution_policy or _extract_execution_policy({})
    url_fetch_policy = url_fetch_policy or {}
    # Strip hidden inputs from the user-supplied kwargs. Auth credentials
    # (``auth_token_comfy_org`` / ``api_key_comfy_org``) travel as standard
    # HTTP headers so the server can forward them to api.comfy.org without
    # parsing the body; everything else rides in the request ``context``.
    context: dict[str, Any] = {}
    auth_headers: dict[str, str] = {}
    hidden_holder = getattr(cls, "hidden", None)
    for hname in hidden_names:
        # Prefer cls.hidden (the standard executor path); fall back to a
        # value piped directly into kwargs for advanced workflows that
        # wire strings into hidden slots manually. Either way, strip the
        # name from the user-input kwargs so it doesn't leak into the
        # request body.
        hidden_value = (
            getattr(hidden_holder, hname, None) if hidden_holder is not None else None
        )
        piped_value = inputs.pop(hname, None)
        value = hidden_value if hidden_value is not None else piped_value
        if value is None:
            continue
        if hname == "auth_token_comfy_org":
            auth_headers["Authorization"] = f"Bearer {value}"
        elif hname == "api_key_comfy_org":
            auth_headers["X-API-KEY"] = str(value)
        else:
            context[hname] = value

    # Encode heavy-typed inputs (IMAGE/MASK/AUDIO) as RNP value
    # envelopes; scalars pass through. Oversized envelopes are
    # uploaded out-of-band and rewritten to ``{..., uri: <url>}``.
    # ``input_serialization`` carries the per-input encoding allow-list
    # from the descriptor — the encoder uses it to decide whether to
    # ship a multi-frame ``png_base64_batch`` envelope or fall back
    # to single-frame ``png_base64``.
    payload_inputs = await _encode_inputs(
        inputs,
        server_url=server_url,
        max_inline_bytes=max_inline_bytes,
        auth_headers=auth_headers,
        input_serialization=input_serialization or {},
        local_validate=local_validate or {},
        node_id=node_id,
    )

    ticker_node_id = _node_id_for_ticker(cls)
    # Stable per-execute() idempotency key — when the underlying HTTP
    # client retries the POST after a connection drop, the server sees
    # the same key and returns the original task_id rather than
    # spawning a duplicate run.
    idempotency_key = uuid.uuid4().hex

    async def _run_with_ticker(
        execute_schema_hash: str | None,
    ) -> dict[str, Any]:
        if execution_mode == ExecutionMode.ASYNC_POLLING:
            # poll_op_raw runs its own _display_time_progress ticker,
            # so we don't start a parallel one here — the sync path
            # below still uses the local ticker since execute_remote
            # blocks on a single HTTP request without UI feedback.
            return await _run_async_polling(
                cls,
                node_id=node_id,
                server_url=server_url,
                schema_hash=execute_schema_hash,
                payload_inputs=payload_inputs,
                context=context,
                auth_headers=auth_headers,
                idempotency_key=idempotency_key,
                poll_interval_s=policy["poll_interval_s"],
                estimated_duration_s=estimated_duration_s,
                max_poll_attempts=policy["max_poll_attempts"],
                timeout_per_poll_s=policy["timeout_per_poll_s"],
                cancel_timeout_s=policy["cancel_timeout_s"],
                max_retries_per_poll=policy["max_retries_per_poll"],
                retry_delay_per_poll_s=policy["retry_delay_per_poll_s"],
                retry_backoff_per_poll=policy["retry_backoff_per_poll"],
            )
        # Shared mutable holder the local ticker watches so a
        # SERVER_BUSY/MAINTENANCE retry inside execute_remote can swap
        # the per-second "Status: Processing" line for the friendly
        # backpressure copy without restarting the ticker. The ticker
        # auto-expires the override once ``expires_at`` passes so a
        # stale "Server busy, retrying in 30s..." doesn't linger if the
        # post-wait request itself takes longer than the announced
        # Retry-After window.
        status_override: dict[str, Any] = {"label": None, "expires_at": None}

        def _local_rate_limit_label(
            status: int, body: Any, retry_after_s: float,
        ) -> str | None:
            label = rnp_client._rnp_rate_limit_label(status, body, retry_after_s)
            status_override["label"] = label  # may be None for unknown codes
            status_override["expires_at"] = (
                time.monotonic() + max(1.0, retry_after_s) if label else None
            )
            return label

        ticker = asyncio.create_task(
            _progress_ticker(
                ticker_node_id,
                estimated_duration_s,
                status_override=status_override,
            ),
        )
        try:
            return await rnp_client.execute_remote(
                server_url,
                execute_schema_hash,
                execute_path,
                payload_inputs,
                context,
                timeout_s=timeout_s,
                extra_headers=auth_headers or None,
                node_cls=cls,
                rate_limit_label=_local_rate_limit_label,
            )
        finally:
            ticker.cancel()
            with contextlib.suppress(BaseException):
                await ticker

    try:
        body = await _run_with_ticker(schema_hash)
    except RnpProtocolError as e:
        # Schema-hash mismatch → re-fetch object_info, find the new
        # hash for this node, and retry once. This recovers from the
        # common deploy ordering of "server updated descriptor ⇒
        # client still using cached one" without forcing a ComfyUI
        # restart.
        if e.code == ErrorCode.SCHEMA_HASH_MISMATCH:
            new_hash = await _refresh_schema_hash(server_url, node_id)
            if new_hash and new_hash != schema_hash:
                log.info(
                    "RNP %s: schema_hash changed (%s → %s); retrying execute",
                    node_id, schema_hash, new_hash,
                )
                try:
                    body = await _run_with_ticker(new_hash)
                except RnpProtocolError as e2:
                    raise RuntimeError(
                        _to_friendly_message(e2, node_id),
                    ) from e2
            else:
                raise RuntimeError(
                    f"Remote node {node_id}: schema-hash mismatch and no "
                    f"updated descriptor found upstream"
                ) from e
        else:
            log.warning("RNP execute %s failed: %s [%s]", node_id, e, e.code)
            raise RuntimeError(_to_friendly_message(e, node_id)) from e

    outputs = body.get("outputs")
    if not isinstance(outputs, list):
        raise RuntimeError(
            f"Remote node {node_id} returned no 'outputs' list"
        )
    if len(outputs) != len(outputs_meta):
        raise RuntimeError(
            f"Remote node {node_id} returned {len(outputs)} outputs; "
            f"schema declared {len(outputs_meta)}"
        )

    decoded = []
    for value, meta in zip(outputs, outputs_meta):
        decoded.append(await _deserialize_output(
            value,
            meta,
            url_fetch_policy=url_fetch_policy,
            server_url=server_url,
            auth_headers=auth_headers,
            node_cls=cls,
        ))
    return IO.NodeOutput(*decoded)


async def _refresh_schema_hash(server_url: str, node_id: str) -> str | None:
    """Re-fetch ``/object_info`` and return the freshly published
    ``remote.schema_hash`` for ``node_id`` (or ``None`` if the node has
    been removed upstream).
    """
    try:
        body, _etag, _max_age = await rnp_client.fetch_object_info(server_url)
    except Exception as e:  # noqa: BLE001
        log.warning("schema-hash refresh failed for %s: %s", node_id, e)
        return None
    if not isinstance(body, dict):
        return None
    descriptor = body.get(node_id)
    if not isinstance(descriptor, dict):
        return None
    remote = descriptor.get("remote") or {}
    return remote.get("schema_hash") if isinstance(remote, dict) else None


async def _deserialize_output(
    value: Any,
    meta: Any,
    *,
    url_fetch_policy: dict[str, Any],
    server_url: str,
    auth_headers: dict[str, str],
    node_cls: type[IO.ComfyNode] | None,
) -> Any:
    """Convert one output value (envelope or scalar) to its native Python form.

    Inline envelopes (with ``data``) decode immediately. URI envelopes
    look up their fetch policy from ``url_fetch_policy`` (keyed by the
    envelope's ``type``) and fetch the bytes via ``sync_op_raw`` so the
    GET inherits ``ProcessingInterrupted`` cancellation, retry/backoff
    on transient errors, and the per-policy timeout. Scalars pass
    through unchanged."""
    if not is_envelope(value):
        # Scalars pass through unchanged — the schema's output type is
        # advisory here, the value already matches the native ComfyUI
        # type for STRING / INT / FLOAT / BOOLEAN.
        return value
    encoding = value.get("encoding")
    env_type = value.get("type", "")
    if encoding == "png_base64_batch_uri":
        # Externalised IMAGE batch on the response path (mirror of the
        # request-side ``serialization.maybe_externalize`` behaviour).
        # No server emits this shape today, but the decoder owns the
        # symmetry: any envelope shape the client may externalise on
        # the way out, the client must also accept on the way in.
        # Per-frame URIs fetched in parallel via ``asyncio.gather`` so
        # a 9-frame batch import is one round-trip rather than N — the
        # GET fan-out is the dominant latency for an out-of-band batch.
        # Frame ordering is preserved (gather returns positional
        # results) — critical because the IMAGE tensor's batch axis
        # carries semantic order.
        policy = url_fetch_policy.get(env_type)
        if not isinstance(policy, dict):
            raise RnpProtocolError(
                f"Server returned a URL batch envelope for output type "
                f"{env_type!r} but the descriptor declared no "
                f"remote.url_fetch policy for that type",
                code=ErrorCode.INTERNAL,
            )
        uris = value.get("uris")
        if not isinstance(uris, list) or not uris:
            raise RnpProtocolError(
                "png_base64_batch_uri envelope requires a non-empty 'uris' list",
                code=ErrorCode.INTERNAL,
            )
        import asyncio as _asyncio
        import base64 as _b64
        per_frame_envs = [
            {"type": env_type, "encoding": "png_base64", "uri": u}
            for u in uris
        ]
        raws = await _asyncio.gather(*[
            _fetch_url_envelope(
                env, policy,
                auth_headers=auth_headers,
                node_cls=node_cls,
            )
            for env in per_frame_envs
        ])
        # Re-stamp as the inline batch shape so the existing
        # ``_decode_image_batch_envelope`` path handles it uniformly.
        value = {k: v for k, v in value.items() if k != "uris"}
        value["encoding"] = "png_base64_batch"
        value["frames"] = [_b64.b64encode(r).decode("ascii") for r in raws]
    elif "data" not in value and "uri" in value:
        policy = url_fetch_policy.get(env_type)
        if not isinstance(policy, dict):
            raise RnpProtocolError(
                f"Server returned a URL envelope for output type {env_type!r} "
                "but the descriptor declared no remote.url_fetch policy "
                "for that type",
                code=ErrorCode.INTERNAL,
            )
        raw = await _fetch_url_envelope(
            value, policy,
            auth_headers=auth_headers,
            node_cls=node_cls,
        )
        # Re-stamp the envelope with inline ``data`` so the existing
        # decoder dispatch in ``serialization`` sees a homogeneous
        # input shape regardless of how the bytes arrived.
        import base64 as _b64
        value = {k: v for k, v in value.items() if k != "uri"}
        value["data"] = _b64.b64encode(raw).decode("ascii")
    return serialization.decode_envelope(value)


async def _fetch_url_envelope(
    envelope: dict[str, Any],
    policy: dict[str, Any],
    *,
    auth_headers: dict[str, str],
    node_cls: type[IO.ComfyNode] | None,
) -> bytes:
    """GET ``envelope["uri"]`` via ``sync_op_raw`` honoring the descriptor's
    url_fetch policy (retries, backoff, timeout, optional auth).

    ``url_kind`` drives the auth posture: ``presigned`` / ``public`` →
    no auth header (the URL embeds its own credentials or needs none);
    ``comfy_api`` → forward the same Authorization / X-API-KEY the
    polls use. ``auth_required`` overrides ``url_kind`` when explicitly
    set to ``True``.
    """
    from comfy_api_nodes.util.client import ApiEndpoint, sync_op_raw

    uri = envelope["uri"]
    url_kind = policy.get("url_kind") or "presigned"
    auth_required = bool(policy.get(
        "auth_required",
        url_kind == "comfy_api",
    ))
    fetch_auth: dict[str, str] = (
        dict(auth_headers) if (auth_required and auth_headers) else {}
    )
    timeout_s = float(policy.get("timeout_s") or 300)
    max_retries = int(policy.get("max_retries") or 3)
    retry_delay = float(policy.get("retry_delay_s") or 1.0)
    retry_backoff = float(policy.get("retry_backoff") or 2.0)

    raw = await sync_op_raw(
        node_cls,
        ApiEndpoint(uri, "GET"),
        timeout=timeout_s,
        max_retries=max_retries,
        retry_delay=retry_delay,
        retry_backoff=retry_backoff,
        wait_label="Downloading",
        as_binary=True,
        final_label_on_success=None,
        monitor_progress=False,
        auth_headers=fetch_auth,
    )
    if not isinstance(raw, (bytes, bytearray)):
        raise RnpProtocolError(
            f"url_fetch GET {uri!r} returned non-bytes payload",
            code=ErrorCode.INTERNAL,
        )
    return bytes(raw)


async def _encode_inputs(
    inputs: dict[str, Any],
    *,
    server_url: str | None = None,
    max_inline_bytes: int | None = None,
    auth_headers: dict[str, str] | None = None,
    input_serialization: dict[str, list[str]] | None = None,
    local_validate: dict[str, dict[str, Any]] | None = None,
    node_id: str | None = None,
) -> dict[str, Any]:
    """Convert in-place: replace heavy-typed input values with the
    matching RNP value envelope. Scalars pass through unchanged.

    The duck-typing rules:

    * ``torch.Tensor`` rank 4 (B,H,W,C) → image envelope.
    * ``torch.Tensor`` rank 3 with last-dim != 3/4 → mask envelope (B,H,W).
      Rank-2 (H,W) tensors are also treated as masks.
    * ``dict`` with ``waveform`` + ``sample_rate`` → audio envelope.
    * Everything else passes through as-is (already JSON-serializable).

    When ``max_inline_bytes`` is set, any envelope whose inline payload
    exceeds the threshold is externalized via ``POST /rnp/v1/uploads``
    (see ``serialization.maybe_externalize``).

    ``input_serialization`` is the descriptor's per-input encoding
    allow-list (e.g. ``{"images": ["png_base64", "png_base64_batch"]}``).
    The encoder consults it to decide whether a multi-frame IMAGE input
    can ride a single batch envelope vs. falling back to single-frame.

    ``local_validate`` is the per-input declarative-rule map
    (e.g. ``{"images": {"image_max_batch": 9}}``). For each rule the
    encoder enforces the constraint locally and raises
    ``RnpProtocolError(code=INPUT_INVALID, user_facing=True)`` so the
    proxy_node's friendly-message dispatch surfaces a per-provider
    error before any paid HTTP round trip. Lookup is defensive: a
    missing ``local_validate`` entry or missing per-rule key is a
    no-op.
    """
    serialization_map = input_serialization or {}
    validate_map = local_validate or {}
    out: dict[str, Any] = {}
    for name, value in inputs.items():
        _enforce_local_validate(
            name, value, validate_map.get(name) or {}, node_id=node_id,
        )
        encoded = _encode_one(name, value, serialization_map.get(name))
        # Only envelope-shaped values can be externalized; scalars pass
        # straight through ``maybe_externalize`` unchanged.
        if serialization.is_envelope(encoded):
            encoded = await serialization.maybe_externalize(
                encoded,
                server_url=server_url,
                max_inline_bytes=max_inline_bytes,
                auth_headers=auth_headers,
            )
        out[name] = encoded
    return out


def _enforce_local_validate(
    name: str,
    value: Any,
    rules: dict[str, Any],
    *,
    node_id: str | None,
) -> None:
    """Apply each ``local_validate.*`` rule to ``value`` and raise
    ``RnpProtocolError(INPUT_INVALID)`` on a violation.

    Rules currently supported:

    * ``image_max_batch`` — for IMAGE inputs (rank-4 tensors), assert
      ``tensor.shape[0] <= cap``. Mirrors the per-call cap raised on
      the server (e.g. Flux2 raises "The current maximum number of
      supported images is 9.") so a workflow with N>cap reference
      frames fails fast on the client before any paid HTTP round trip.

    Two private composite keys are walked transparently:

    * ``__template__`` (AUTOGROW) — ``value`` is expected to be a
      runtime Autogrow dict (``{"image0": <env>, "image1": <env>, ...}``).
      The template's rules are applied to each non-``None`` slot value.
    * ``__branches__`` (DYNAMIC_COMBO) — ``value`` is expected to be the
      DynamicCombo runtime dict whose ``value[<name>]`` carries the
      selected branch key. The matching branch's per-sub-input rules are
      applied recursively. This is what makes GPT Image V2 / Grok
      ImageEditV2 / Grok VideoReference's nested per-slot caps fire
      client-side before any HTTP round trip.
    """
    if not rules:
        return
    _check_image_max_batch(name, value, rules.get("image_max_batch"), node_id=node_id)
    tmpl = rules.get("__template__")
    if isinstance(tmpl, dict) and isinstance(value, dict):
        for slot_key, slot_value in value.items():
            if slot_value is None:
                continue
            _enforce_local_validate(
                f"{name}.{slot_key}", slot_value, tmpl, node_id=node_id,
            )
    branches = rules.get("__branches__")
    if isinstance(branches, dict) and isinstance(value, dict):
        branch_key = value.get(name)
        if isinstance(branch_key, str):
            sub_rules_map = branches.get(branch_key)
            if isinstance(sub_rules_map, dict):
                for sub_name, sub_rules in sub_rules_map.items():
                    if sub_name in value:
                        _enforce_local_validate(
                            f"{name}.{sub_name}", value[sub_name], sub_rules,
                            node_id=node_id,
                        )


def _check_image_max_batch(
    name: str,
    value: Any,
    cap: Any,
    *,
    node_id: str | None,
) -> None:
    """Apply the ``image_max_batch`` rule. No-op if cap is missing /
    non-int / value is not a recognisable image tensor."""
    if cap is None:
        return
    try:
        cap_int = int(cap)
    except (TypeError, ValueError):
        return
    if not serialization._is_torch_tensor(value):
        return
    rank = value.dim()
    last_dim = int(value.shape[-1]) if rank >= 1 else 0
    if not (rank == 4 or (rank == 3 and last_dim in (3, 4))):
        return
    batch = int(value.shape[0]) if rank == 4 else 1
    if batch <= cap_int:
        return
    label = f" ({node_id})" if node_id else ""
    if cap_int == 1:
        msg = (
            f"Remote node{label}: input {name!r} expects a single image "
            f"frame, got a batch of {batch}. Wire a single-frame IMAGE "
            f"into this input."
        )
    else:
        msg = (
            f"Remote node{label}: input {name!r} accepts at most "
            f"{cap_int} image frames, got a batch of {batch}. The "
            f"current maximum number of supported images is "
            f"{cap_int}."
        )
    raise RnpProtocolError(
        msg,
        code=ErrorCode.INPUT_INVALID,
        user_facing=True,
    )


def _encode_one(
    name: str,
    value: Any,
    accepted_encodings: list[str] | None = None,
) -> Any:
    if serialization.is_audio_input(value):
        return serialization.encode_audio_input(value)
    if not serialization._is_torch_tensor(value):
        return value
    rank = value.dim()
    last_dim = int(value.shape[-1]) if rank >= 1 else 0
    if rank == 4 or (rank == 3 and last_dim in (3, 4)):
        # Per-image tensors: a length-1 batch is the common case.
        # When the descriptor advertises ``"png_base64_batch"`` for
        # this input AND the tensor has > 1 frame, ship the multi-
        # frame envelope; otherwise fall back to single-frame
        # (``tensor[0]`` for batches >1, full tensor for length-1).
        accepts_batch = bool(
            accepted_encodings and "png_base64_batch" in accepted_encodings
        )
        return serialization.encode_image_tensor(
            value, accepts_batch=accepts_batch,
        )
    if rank in (2, 3):
        return serialization.encode_mask_tensor(value)
    log.warning(
        "Input %s: unrecognised tensor rank %d — sending as-is",
        name, rank,
    )
    return value
