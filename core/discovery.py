"""Model + LoRA discovery for the categorized model dropdown.

Two backends:
  - native: Wan2GP image models (Flux / Z-Image / Qwen) — the list is supplied by
    the plugin at runtime (from wgp globals), categorized by model_type name here.
  - sd:     SDXL/Pony/Illustrious checkpoints scanned from paths.sdxl_models_dir().
            Architecture is SDXL for all; Pony/Illustrious are name-based.

Dropdown choices are (label, value) pairs; value encodes the backend so the
generator can route: "native::<model_type>" or "sd::<checkpoint_path>".
"""
from __future__ import annotations

import re
from pathlib import Path

from . import paths

_CKPT_EXTS = (".safetensors", ".ckpt")


def categorize_native(model_type: str) -> str | None:
    """Bucket a Wan2GP image model_type into a dropdown category, or None if it
    isn't an image model we surface."""
    mt = (model_type or "").lower()
    if "z_image" in mt or "zimage" in mt or "z-image" in mt:
        return "Z-Image"
    if "qwen_image" in mt or "qwen-image" in mt or mt.startswith("qwen_image"):
        return "Qwen"
    if "flux" in mt:
        return "Flux"
    return None


def _has_marker(name: str, markers) -> bool:
    """True if any marker starts on a token boundary in name (preceded by the
    string start or a non-alphanumeric char). Anchoring the *leading* edge lets
    the common 'ponyDiffusionV6XL' / 'noobaiXL' / 'illustriousXL' naming style
    match while rejecting unrelated names where the marker is embedded mid-token
    (e.g. 'noob' inside 'snooball')."""
    for m in markers:
        if re.search(r"(?<![a-z0-9])" + re.escape(m), name):
            return True
    return False


def categorize_sdxl(name: str) -> str:
    """Name-based bucket for an SDXL-architecture checkpoint.

    Markers are matched on a leading token boundary rather than as raw
    substrings so unrelated names aren't misclassified. The over-broad generic
    'illust' marker is intentionally dropped (it caught
    'illustration'/'illustrated'); the full 'illustrious' marker is still
    recognized."""
    n = (name or "").lower()
    if _has_marker(n, ("pony",)):
        return "Pony"
    if _has_marker(n, ("illustrious", "noob")):
        return "Illustrious"
    return "SDXL"


def discover_sdxl_models(models_dir=None) -> list[dict]:
    """Scan the SDXL models dir for checkpoints, categorized."""
    d = Path(models_dir) if models_dir else paths.sdxl_models_dir()
    out = []
    if d and Path(d).is_dir():
        for p in sorted(Path(d).rglob("*")):
            if p.is_file() and p.suffix.lower() in _CKPT_EXTS:
                out.append({"backend": "sd", "category": categorize_sdxl(p.stem),
                            "name": p.stem, "path": str(p)})
    return out


def discover_sdxl_loras(loras_dir=None) -> list[dict]:
    d = Path(loras_dir) if loras_dir else paths.sdxl_loras_dir()
    out = []
    if d and Path(d).is_dir():
        for p in sorted(Path(d).rglob("*")):
            if p.is_file() and p.suffix.lower() in _CKPT_EXTS:
                out.append({"name": p.stem, "path": str(p)})
    return out


# Category display order in the dropdown.
_ORDER = ["Flux", "Z-Image", "Qwen", "SDXL", "Pony", "Illustrious"]


def native_label_suffix(model_type: str) -> str:
    """A clarifying " (Turbo)"/" (Inpaint)" suffix for a native model label, so
    e.g. z_image reads "(Turbo)" and z_image_control2_1 reads "(Inpaint)"."""
    mt = (model_type or "").lower()
    tags = []
    if any(t in mt for t in ("control", "inpaint", "edit", "fill", "kontext")):
        tags.append("Inpaint")
    if mt == "z_image" or any(t in mt for t in ("turbo", "schnell", "distill",
                                                "lightning")):
        tags.append("Turbo")
    return " (" + ", ".join(tags) + ")" if tags else ""

# Aggressive (≈4-bit) quantization — shrinks even a 20B/32B enough to fit a
# low-VRAM card, so these OVERRIDE the heavy size hints below.
_STRONG_QUANT_HINTS = ("nunchaku", "nvfp4", "fp4", "int4", "gguf", "_q4", "_q5",
                       "q4_k", "q5_k")
# Milder light hints: small param counts / fast distills / fp8 (only enough on
# their own, NOT enough to rescue a 20B+ base — hence checked after heavy).
_LOW_VRAM_HINTS = ("klein_4b", "klein 4b", "_4b", "schnell", "fp8", "distill",
                   "turbo", "lightning")
# Substrings that mark a model as too heavy unless strongly quantized above.
_HEAVY_HINTS = ("32b", "_20b", "20b", "klein_9b", "_9b")
# Whole native families that are always light enough for a low-VRAM card,
# regardless of variant naming (the model_type often omits a size hint).
_LOW_VRAM_CATEGORIES = ("Z-Image",)  # Z-Image is a 6B family (≈6GB at int8)


def is_low_vram_native(model_type: str) -> bool:
    """Heuristic: is this native image model light enough for a low-VRAM card?

    Order matters: aggressive 4-bit quantization (nunchaku/int4/fp4/gguf-q4)
    fits even a 20B, so it wins first. Then heavy size hints (20B/32B/9B)
    disqualify. Whole small families (Z-Image, 6B) always qualify. Otherwise a
    milder light hint (klein_4b/schnell/fp8/distill/turbo) is required."""
    mt = (model_type or "").lower()
    if any(q in mt for q in _STRONG_QUANT_HINTS):
        return True
    if any(h in mt for h in _HEAVY_HINTS):
        return False
    if categorize_native(model_type) in _LOW_VRAM_CATEGORIES:
        return True
    return any(h in mt for h in _LOW_VRAM_HINTS)


def build_model_choices(native_model_types=None, models_dir=None,
                        low_vram_only=False) -> list:
    """Return Gradio Dropdown choices [(label, value), ...] grouped by category.

    native_model_types: iterable of Wan2GP image model_type strings (from the app).
    low_vram_only: when True, drop heavy native models (SDXL-family always kept).
    """
    entries: list[dict] = []
    for mt in (native_model_types or []):
        cat = categorize_native(mt)
        if not cat:
            continue
        if low_vram_only and not is_low_vram_native(mt):
            continue
        entries.append({"backend": "native", "category": cat, "name": mt,
                        "value": f"native::{mt}"})
    for m in discover_sdxl_models(models_dir):
        entries.append({**m, "value": f"sd::{m['path']}"})

    entries.sort(key=lambda e: (_ORDER.index(e["category"]) if e["category"] in _ORDER
                                else len(_ORDER), e["name"].lower()))

    def _label(e):
        suffix = native_label_suffix(e["name"]) if e["backend"] == "native" else ""
        return f"{e['category']} · {e['name']}{suffix}"
    return [(_label(e), e["value"]) for e in entries]


def parse_model_value(value: str) -> tuple[str, str]:
    """('native', model_type) or ('sd', checkpoint_path) from a dropdown value."""
    if not value:
        return "", ""
    backend, _, ident = value.partition("::")
    return backend, ident


# Common generation resolutions per model family, for the outpaint target-size
# dropdown. All current families are ~1024-base; Flux tolerates larger/wider.
_SIZES_SDXL = ["1024×1024", "1152×896", "896×1152", "1216×832", "832×1216",
               "1344×768", "768×1344", "1536×640", "640×1536"]
_SIZES_FLUX = ["1024×1024", "1216×832", "832×1216", "1344×768", "768×1344",
               "1536×1024", "1024×1536", "1920×1088", "1088×1920"]


def common_sizes(model_value: str) -> list[str]:
    """Family-appropriate 'W×H' target sizes for the selected model."""
    backend, ident = parse_model_value(model_value)
    if backend == "native" and categorize_native(ident) == "Flux":
        return _SIZES_FLUX
    return _SIZES_SDXL  # SDXL/Pony/Illustrious + Z-Image/Qwen (all 1024-base)


def parse_size(label: str) -> tuple[int, int]:
    """'1216×832' (or with 'x') → (1216, 832); anything else → (0, 0)."""
    if not label:
        return 0, 0
    for sep in ("×", "x", "X"):
        if sep in label:
            a, _, b = label.partition(sep)
            try:
                return int(a.strip()), int(b.strip())
            except ValueError:
                return 0, 0
    return 0, 0


def resolution_presets(model_value: str) -> list[tuple[str, str]]:
    """(label, 'W×H') resolution choices for the selected model's FAMILY — the
    trained 1024-base buckets for SDXL/Pony/Illustrious/Z-Image/Qwen, the wider set
    for Flux — tagged by orientation (1:1 / Portrait / Landscape) and ordered
    square → portrait → landscape. The value is a 'W×H' string parse_size reads;
    plugin.py wires the dropdown to drop those into Width/Height. Repopulated on
    model select (mirrors common_sizes / the outpaint target-size dropdown)."""
    tagged = []
    for s in common_sizes(model_value):
        w, h = parse_size(s)
        if not (w and h):
            continue
        if w == h:
            tag, rank = "⬛ 1:1", 0
        elif h > w:
            tag, rank = "📱 Portrait", 1
        else:
            tag, rank = "🖼 Landscape", 2
        tagged.append((rank, f"{tag} · {w}×{h}", s))
    tagged.sort(key=lambda t: t[0])  # stable → keeps common_sizes order within a tag
    return [(label, val) for _rank, label, val in tagged]


def categorize_lora(name: str) -> str:
    """Same name-based family bucket as checkpoints (Pony/Illustrious/SDXL)."""
    return categorize_sdxl(name)


def model_family(model_value: str) -> str | None:
    """The SD family of the selected model (Pony/Illustrious/SDXL), or None for
    native models (Flux/Z-Image/Qwen) — which the SDXL LoRAs don't apply to."""
    backend, ident = parse_model_value(model_value)
    if backend == "sd":
        return categorize_sdxl(Path(ident).stem)
    return None


def lora_choices(loras_dir=None, family: str | None = None) -> list:
    """Categorized LoRA dropdown choices, optionally filtered to one family.
    SDXL/Pony/Illustrious LoRAs are NOT cross-compatible, so when a model is
    selected we show only its family."""
    out = []
    for m in discover_sdxl_loras(loras_dir):
        cat = categorize_lora(m["name"])
        if family and cat != family:
            continue
        out.append((f"{cat} · {m['name']}", m["path"]))
    return out
