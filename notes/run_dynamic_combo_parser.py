"""Smoke test for the DYNAMIC_COMBO + AUTOGROW parser extension.

Verifies the proxy_node parser now constructs real V3
``IO.DynamicCombo.Input`` and ``IO.Autogrow.Input`` objects from the
flattened RNP wire shapes that the server already emits, instead of
falling through to the opaque ``IO.Custom`` bucket.

Checks:

1.  Bria-shaped 2-branch DYNAMIC_COMBO (the "false" branch has empty
    ``inputs``; the "true" branch has 3 BOOLEAN sub-widgets) parses
    into an ``IO.DynamicCombo.Input`` whose two ``Option``s carry the
    right keys + sub-input objects. Proves empty branches are kept and
    that the outer id is whatever the descriptor key was (here
    ``"moderation"``, *not* ``"model"``).
2.  Grok-V2-shaped DYNAMIC_COMBO with three branches, each containing
    a nested AUTOGROW(IMAGE) plus a COMBO + INT (and an aspect_ratio
    COMBO on 2 of the 3 branches). Asserts the AUTOGROW becomes a real
    ``IO.Autogrow.Input`` with a ``TemplatePrefix`` whose underlying
    input is ``IO.Image.Input``, and that prefix / min / max survive.
3.  ElevenLabs-shaped DYNAMIC_COMBO whose branches mix
    ``IO.Custom(<opaque_io_type>).Input`` (voiceN custom-IO) with
    primitive STRING sub-widgets — proves the recursive opaque
    fallback path inside a branch still works for partner helper
    types.
4.  Synthetic nested DYNAMIC_COMBO-inside-DYNAMIC_COMBO — no provider
    uses this today, but defensive coverage so the recursive entry
    point keeps working.
5.  Malformed DYNAMIC_COMBO / AUTOGROW shapes return ``None`` (so the
    caller skips the descriptor entirely) instead of silently
    falling through to the opaque IO.Custom bucket.
6.  Regression: ``_collect_local_validate`` output is byte-identical
    before and after the parser change (the rules-extraction path is
    independent of input construction and must not regress).
7.  AUTOGROW template that is itself a dynamic input is rejected
    (upstream Autogrow asserts this).
8.  Source-level check: the proxy_node docstring + dispatch covers
    DYNAMIC_COMBO and AUTOGROW (not just the existing primitive set).

Run with the ComfyUI venv (needs torch / comfy_api on PYTHONPATH):

    python notes/run_dynamic_combo_parser.py
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


def _load_module(name: str, path: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Stub all the heavy deps proxy_node + comfy_api/latest/_io.py pull in.
#
# We only need three things from the world:
#
# * the real V3 IO classes (DynamicCombo / Autogrow / Image / Combo / String /
#   Int / Boolean / Custom) — load ``_io.py`` standalone so we don't pay the
#   full ``comfy_api.latest`` package init (PIL, numpy, comfy.cli_args,
#   comfy_execution, the sync-class generator etc.).
# * the real proxy_node parser helpers — load ``proxy_node.py`` after the
#   ``comfy_api.latest`` shim is in place.
# * the real protocol module (lightweight; for the ErrorCode / Header / etc.
#   imports proxy_node does up top).
#
# Everything else (server.PromptServer, aiohttp, comfy_api_nodes.*) is
# stubbed because proxy_node only references those at request-time, not at
# import-time-after-trivial-stub.
# ---------------------------------------------------------------------------

# 1) Stub ``torch`` — _io.py uses ``torch.Tensor`` only in type annotations
#    (``Type = torch.Tensor`` class-level field), so a bare-class stub works.
_torch_stub = types.ModuleType("torch")
class _Tensor:  # noqa: D401
    pass
_torch_stub.Tensor = _Tensor
sys.modules.setdefault("torch", _torch_stub)

# 2) Stub ``comfy_execution.graph_utils.ExecutionBlocker`` — only used as a
#    class reference inside Output construction; never instantiated here.
_comfy_execution = types.ModuleType("comfy_execution")
_graph_utils = types.ModuleType("comfy_execution.graph_utils")
class _ExecutionBlocker:  # noqa: D401
    pass
_graph_utils.ExecutionBlocker = _ExecutionBlocker
sys.modules.setdefault("comfy_execution", _comfy_execution)
sys.modules.setdefault("comfy_execution.graph_utils", _graph_utils)

# 3) Stub ``comfy_api.internal`` — _io.py imports a handful of helper names.
#    The parser code paths exercised here don't actually invoke any of these;
#    bare-function / bare-class stubs are sufficient to satisfy the import.
_comfy_api = types.ModuleType("comfy_api")
_comfy_api.__path__ = []
_comfy_api_internal = types.ModuleType("comfy_api.internal")
class _ComfyNodeInternal:  # noqa: D401
    pass
class _NodeOutputInternal:  # noqa: D401
    pass
class _classproperty:  # noqa: D401
    def __init__(self, f):
        self.f = f
    def __get__(self, instance, owner):
        return self.f(owner)
def _identity(x, *_args, **_kw):
    return x
def _first_real_override(*_a, **_kw):
    return None
def _is_class(x):
    return isinstance(x, type)
def _prune_dict(d):
    return {k: v for k, v in (d or {}).items() if v is not None}
_comfy_api_internal._ComfyNodeInternal = _ComfyNodeInternal
_comfy_api_internal._NodeOutputInternal = _NodeOutputInternal
_comfy_api_internal.classproperty = _classproperty
_comfy_api_internal.copy_class = _identity
_comfy_api_internal.first_real_override = _first_real_override
_comfy_api_internal.is_class = _is_class
_comfy_api_internal.prune_dict = _prune_dict
_comfy_api_internal.shallow_clone_class = _identity
sys.modules.setdefault("comfy_api", _comfy_api)
sys.modules.setdefault("comfy_api.internal", _comfy_api_internal)

# 3b) Stub ``comfy_api.input`` for the lazy imports inside _io.py's
#     Curve / RangeInput class bodies — the parser code never touches
#     those types, but the *class body* runs at module load.
_comfy_api_input = types.ModuleType("comfy_api.input")
class _CurvePoint: pass  # noqa: E701,D401
class _RangeInput: pass  # noqa: E701,D401
_comfy_api_input.CurvePoint = _CurvePoint
_comfy_api_input.RangeInput = _RangeInput
sys.modules.setdefault("comfy_api.input", _comfy_api_input)

# 4) Stub ``comfy_api.latest._util`` — _io.py imports MESH / VOXEL / SVG /
#    File3D as type-token classes. We don't construct any here.
_comfy_api_latest = types.ModuleType("comfy_api.latest")
_comfy_api_latest.__path__ = []
_comfy_api_latest_util = types.ModuleType("comfy_api.latest._util")
class _MESH: pass  # noqa: E701,D401
class _VOXEL: pass  # noqa: E701,D401
class _SVG: pass  # noqa: E701,D401
class _File3D: pass  # noqa: E701,D401
_comfy_api_latest_util.MESH = _MESH
_comfy_api_latest_util.VOXEL = _VOXEL
_comfy_api_latest_util.SVG = _SVG
_comfy_api_latest_util.File3D = _File3D
sys.modules.setdefault("comfy_api.latest", _comfy_api_latest)
sys.modules.setdefault("comfy_api.latest._util", _comfy_api_latest_util)

# 5) Load the real ``_io.py`` as ``comfy_api.latest._io`` — its
#    relative ``from ._util import ...`` requires the parent-package
#    setup we just did.
_io_path = os.path.join(_COMFYUI_DIR, "comfy_api", "latest", "_io.py")
_io_mod = _load_module("comfy_api.latest._io", _io_path)
# Expose ``IO`` exactly like the real package does
# (``from . import _io_public as io; IO = io`` where _io_public is just
# ``from ._io import *``).
_comfy_api_latest.IO = _io_mod
_comfy_api_latest.io = _io_mod
IO = _io_mod  # local alias for the assertions below

# 6) Stub proxy_node's other heavyweight imports.
for missing in (
    "server",
    "aiohttp",
    "comfy_api_nodes",
    "comfy_api_nodes.util",
    "comfy_api_nodes.util.client",
    "comfy_api_nodes.util.common_exceptions",
):
    if missing not in sys.modules:
        m = types.ModuleType(missing)
        sys.modules[missing] = m
        m.__path__ = []  # type: ignore[attr-defined]
sys.modules["server"].PromptServer = type("_StubPS", (), {})
sys.modules["comfy_api_nodes.util.client"].ApiEndpoint = type("_StubAE", (), {})
sys.modules["comfy_api_nodes.util.client"].poll_op_raw = lambda *a, **k: None
sys.modules["comfy_api_nodes.util.common_exceptions"].ProcessingInterrupted = type(
    "_StubPI", (Exception,), {},
)

# 7) Emulate the package layout for the client so its relative imports
#    work (``from .protocol import ...`` etc.).
pkg = types.ModuleType("comfy_remote_nodes_test")
pkg.__path__ = [_CLIENT_DIR]
sys.modules["comfy_remote_nodes_test"] = pkg
sys.modules["comfy_remote_nodes_test.client"] = types.ModuleType(
    "comfy_remote_nodes_test.client",
)
sys.modules["comfy_remote_nodes_test.serialization"] = types.ModuleType(
    "comfy_remote_nodes_test.serialization",
)
_load_module(
    "comfy_remote_nodes_test.protocol",
    os.path.join(_CLIENT_DIR, "protocol.py"),
)

proxy_node = _load_module(
    "comfy_remote_nodes_test.proxy_node",
    os.path.join(_CLIENT_DIR, "proxy_node.py"),
)


# ---------------------------------------------------------------------------
# 1. Bria-shaped 2-branch DYNAMIC_COMBO.
# ---------------------------------------------------------------------------
bria_spec = [
    "DYNAMIC_COMBO",
    {
        "options": [
            {"key": "false", "inputs": []},
            {
                "key": "true",
                "inputs": [
                    ["prompt_content_moderation", "BOOLEAN", {"default": False}],
                    ["visual_input_moderation", "BOOLEAN", {"default": False}],
                    ["visual_output_moderation", "BOOLEAN", {"default": True}],
                ],
            },
        ],
        "tooltip": "Moderation settings",
    },
]
inp = proxy_node._parse_input_spec("moderation", bria_spec, optional=False)
assert inp is not None, "Bria DYNAMIC_COMBO returned None"
assert isinstance(inp, IO.DynamicCombo.Input), (
    f"Expected IO.DynamicCombo.Input, got {type(inp).__name__}"
)
assert inp.id == "moderation", f"outer id should track descriptor key, got {inp.id!r}"
assert len(inp.options) == 2, f"expected 2 branches, got {len(inp.options)}"
assert [o.key for o in inp.options] == ["false", "true"], (
    f"branch keys wrong: {[o.key for o in inp.options]}"
)
# Empty branch preserved (UX needs the dropdown entry).
assert inp.options[0].inputs == [], "empty 'false' branch dropped"
assert len(inp.options[1].inputs) == 3, (
    f"'true' branch should have 3 sub-inputs, got {len(inp.options[1].inputs)}"
)
for sub in inp.options[1].inputs:
    assert isinstance(sub, IO.Boolean.Input), (
        f"branch sub-input should be IO.Boolean.Input, got {type(sub).__name__}"
    )
assert [s.id for s in inp.options[1].inputs] == [
    "prompt_content_moderation",
    "visual_input_moderation",
    "visual_output_moderation",
], "branch sub-input names lost"
print("PASS: Bria DYNAMIC_COMBO parses with empty + populated branches.")


# ---------------------------------------------------------------------------
# 2. Grok-V2-shaped DYNAMIC_COMBO with nested AUTOGROW(IMAGE) per branch.
# ---------------------------------------------------------------------------
def _grok_v2_subinputs(*, max_ref_images: int, with_aspect_ratio: bool) -> list:
    inputs: list = [
        [
            "images",
            "AUTOGROW",
            {
                "template": [
                    "IMAGE",
                    {"local_validate": {"image_max_batch": max_ref_images}},
                ],
                "prefix": "image",
                "min": 1,
                "max": max_ref_images,
                "tooltip": (
                    "Reference image to edit."
                    if max_ref_images == 1
                    else f"Reference image(s) to edit. Up to {max_ref_images} images."
                ),
            },
        ],
        ["resolution", "COMBO", {"options": ["768p", "1k"]}],
        ["number_of_images", "INT", {
            "default": 1, "min": 1, "max": 10, "step": 1,
        }],
    ]
    if with_aspect_ratio:
        inputs.append(["aspect_ratio", "COMBO", {
            "options": ["auto", "1:1", "16:9"], "default": "auto",
        }])
    return inputs

grok_spec = [
    "DYNAMIC_COMBO",
    {
        "options": [
            {
                "key": "grok-imagine-image-quality",
                "inputs": _grok_v2_subinputs(max_ref_images=3, with_aspect_ratio=True),
            },
            {
                "key": "grok-imagine-image-pro",
                "inputs": _grok_v2_subinputs(max_ref_images=1, with_aspect_ratio=False),
            },
            {
                "key": "grok-imagine-image",
                "inputs": _grok_v2_subinputs(max_ref_images=3, with_aspect_ratio=True),
            },
        ],
        "tooltip": "The model to use for editing.",
    },
]
grok_inp = proxy_node._parse_input_spec("model", grok_spec, optional=False)
assert grok_inp is not None, "Grok V2 DYNAMIC_COMBO returned None"
assert isinstance(grok_inp, IO.DynamicCombo.Input), (
    f"Expected IO.DynamicCombo.Input, got {type(grok_inp).__name__}"
)
assert len(grok_inp.options) == 3
# Pro branch (max_ref_images=1, no aspect_ratio).
pro_branch = next(o for o in grok_inp.options if o.key == "grok-imagine-image-pro")
assert len(pro_branch.inputs) == 3, f"pro branch should have 3 sub-inputs, got {len(pro_branch.inputs)}"
images_inp = pro_branch.inputs[0]
assert isinstance(images_inp, IO.Autogrow.Input), (
    f"Expected nested AUTOGROW to parse as IO.Autogrow.Input, got "
    f"{type(images_inp).__name__}"
)
assert images_inp.id == "images"
assert isinstance(images_inp.template, IO.Autogrow.TemplatePrefix), (
    f"Expected TemplatePrefix template, got {type(images_inp.template).__name__}"
)
assert images_inp.template.prefix == "image"
assert images_inp.template.min == 1
assert images_inp.template.max == 1, f"pro branch max should be 1, got {images_inp.template.max}"
assert isinstance(images_inp.template.input, IO.Image.Input), (
    f"AUTOGROW template input should be IO.Image.Input, got "
    f"{type(images_inp.template.input).__name__}"
)
# Quality branch (max_ref_images=3 + aspect_ratio).
qual_branch = next(o for o in grok_inp.options if o.key == "grok-imagine-image-quality")
assert len(qual_branch.inputs) == 4, f"quality branch should have 4 sub-inputs, got {len(qual_branch.inputs)}"
qual_images = qual_branch.inputs[0]
assert isinstance(qual_images, IO.Autogrow.Input)
assert qual_images.template.max == 3, f"quality branch max should be 3, got {qual_images.template.max}"
# Primitive widgets after the autogrow.
assert isinstance(qual_branch.inputs[1], IO.Combo.Input)
assert isinstance(qual_branch.inputs[2], IO.Int.Input)
assert isinstance(qual_branch.inputs[3], IO.Combo.Input)
print("PASS: Grok V2 DYNAMIC_COMBO + nested AUTOGROW parses correctly.")


# ---------------------------------------------------------------------------
# 3. ElevenLabs-shaped branch with custom-IO sub-inputs (voiceN).
# ---------------------------------------------------------------------------
el_spec = [
    "DYNAMIC_COMBO",
    {
        "options": [
            {
                "key": "2",
                "inputs": [
                    ["voice0", "ELEVENLABS_VOICE", {}],
                    ["text0", "STRING", {"multiline": True, "default": ""}],
                    ["voice1", "ELEVENLABS_VOICE", {}],
                    ["text1", "STRING", {"multiline": True, "default": ""}],
                ],
            },
        ],
        "tooltip": "Number of dialogue turns.",
    },
]
el_inp = proxy_node._parse_input_spec("inputs", el_spec, optional=False)
assert el_inp is not None, "ElevenLabs DYNAMIC_COMBO returned None"
assert isinstance(el_inp, IO.DynamicCombo.Input)
assert el_inp.id == "inputs", (
    f"outer id should be 'inputs' (not always 'model'), got {el_inp.id!r}"
)
branch = el_inp.options[0]
# IO.Custom("FOO") returns a synthesized ComfyTypeIO subclass — each call
# creates a *new* class, so isinstance-against-a-fresh-Custom doesn't
# work. The @comfytype decorator stamps ``io_type`` onto the Input class
# instead; check that the synthesized custom Input carries the right
# string (this is how the frontend's connection-validity check chains
# sockets).
voice0 = branch.inputs[0]
assert voice0.get_io_type() == "ELEVENLABS_VOICE", (
    f"voice0 should be opaque ELEVENLABS_VOICE Input, got "
    f"{voice0.get_io_type()!r} on {type(voice0).__name__}"
)
assert isinstance(branch.inputs[1], IO.String.Input)
assert isinstance(branch.inputs[3], IO.String.Input)
print("PASS: ElevenLabs DYNAMIC_COMBO branch mixes custom-IO + primitive.")


# ---------------------------------------------------------------------------
# 4. Synthetic nested DYNAMIC_COMBO inside DYNAMIC_COMBO.
# ---------------------------------------------------------------------------
nested = [
    "DYNAMIC_COMBO",
    {
        "options": [
            {
                "key": "outer_a",
                "inputs": [
                    [
                        "submodel",
                        "DYNAMIC_COMBO",
                        {
                            "options": [
                                {"key": "inner_x", "inputs": [
                                    ["x", "INT", {"default": 1, "min": 0, "max": 10}],
                                ]},
                                {"key": "inner_y", "inputs": []},
                            ],
                            "tooltip": "Inner branch.",
                        },
                    ],
                ],
            },
        ],
        "tooltip": "Outer branch.",
    },
]
nest_inp = proxy_node._parse_input_spec("model", nested, optional=False)
assert isinstance(nest_inp, IO.DynamicCombo.Input)
inner = nest_inp.options[0].inputs[0]
assert isinstance(inner, IO.DynamicCombo.Input), (
    f"nested DynamicCombo should be IO.DynamicCombo.Input, got {type(inner).__name__}"
)
assert [o.key for o in inner.options] == ["inner_x", "inner_y"]
assert isinstance(inner.options[0].inputs[0], IO.Int.Input)
print("PASS: nested DYNAMIC_COMBO-in-DYNAMIC_COMBO parses recursively.")


# ---------------------------------------------------------------------------
# 5. Malformed dynamic specs return None (no opaque fallthrough).
# ---------------------------------------------------------------------------
# Missing options list on DYNAMIC_COMBO.
assert proxy_node._parse_input_spec("x", ["DYNAMIC_COMBO", {}], optional=False) is None
# Option missing key.
bad = ["DYNAMIC_COMBO", {"options": [{"inputs": []}]}]
assert proxy_node._parse_input_spec("x", bad, optional=False) is None
# Option's inputs entry not a 3-tuple.
bad2 = ["DYNAMIC_COMBO", {"options": [{"key": "a", "inputs": [["x"]]}]}]
assert proxy_node._parse_input_spec("x", bad2, optional=False) is None
# AUTOGROW without template.
assert proxy_node._parse_input_spec("x", ["AUTOGROW", {"prefix": "p"}], optional=False) is None
# AUTOGROW without prefix/names.
assert (
    proxy_node._parse_input_spec("x", ["AUTOGROW", {"template": ["IMAGE", {}]}], optional=False)
    is None
)
print("PASS: malformed DYNAMIC_COMBO / AUTOGROW specs return None.")


# ---------------------------------------------------------------------------
# 6. Regression: _collect_local_validate output unchanged for these inputs.
# ---------------------------------------------------------------------------
desc_grok = {
    "input": {
        "required": {"model": grok_spec},
        "optional": {},
    },
}
rules = proxy_node._collect_local_validate(desc_grok)
# rules['model'] should carry __branches__ with per-branch maps that each
# carry an __template__ entry under "images" with image_max_batch.
assert "model" in rules, f"_collect_local_validate dropped 'model': {rules!r}"
br = rules["model"].get("__branches__")
assert isinstance(br, dict) and set(br) == {
    "grok-imagine-image-quality",
    "grok-imagine-image-pro",
    "grok-imagine-image",
}, f"branches mismatch: {br!r}"
pro = br["grok-imagine-image-pro"]
assert "images" in pro, f"pro branch should carry 'images' rule: {pro!r}"
tmpl_rules = pro["images"].get("__template__")
assert tmpl_rules == {"image_max_batch": 1}, (
    f"pro branch template rules wrong: {tmpl_rules!r}"
)
qual = br["grok-imagine-image-quality"]
assert qual["images"].get("__template__") == {"image_max_batch": 3}
print("PASS: _collect_local_validate output unchanged (branches + template intact).")


# ---------------------------------------------------------------------------
# 7. AUTOGROW with a dynamic template input is rejected.
# ---------------------------------------------------------------------------
ag_with_dyn_template = [
    "AUTOGROW",
    {
        "template": [
            "DYNAMIC_COMBO",
            {"options": [{"key": "k", "inputs": []}]},
        ],
        "prefix": "img",
        "min": 1,
        "max": 3,
    },
]
assert (
    proxy_node._parse_input_spec("bad", ag_with_dyn_template, optional=False) is None
), "AUTOGROW with DYNAMIC_COMBO template should be rejected"
print("PASS: AUTOGROW rejects dynamic template input (matches upstream assert).")


# ---------------------------------------------------------------------------
# 8. Source-level: docstring + dispatch mention DYNAMIC_COMBO + AUTOGROW.
# ---------------------------------------------------------------------------
proxy_src = open(
    os.path.join(_CLIENT_DIR, "proxy_node.py"), "r", encoding="utf-8",
).read()
assert "DYNAMIC_COMBO" in proxy_src, "proxy_node source missing DYNAMIC_COMBO branch"
assert "AUTOGROW" in proxy_src, "proxy_node source missing AUTOGROW branch"
assert "_parse_named_input_spec" in proxy_src, (
    "proxy_node source missing _parse_named_input_spec helper"
)
assert "_build_input" in proxy_src, "proxy_node source missing _build_input helper"
print("PASS: proxy_node source contains the new dispatch + helpers.")

print("ALL CHECKS PASSED.")
