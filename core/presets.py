"""Recommended generation settings per model family, applied when a model is
selected. Three layers, highest priority first:

  1. User overrides — edited in Settings → OrphanSuite → "Default Generation
     Values" and persisted to the cross-plugin ``.orphansuite.json`` (shared by
     every saintorphan plugin). Per family.
  2. Native model defaults — for Flux/Z-Image/Qwen, the model's own Wan2GP
     defaults (steps / CFG / resolution) are more accurate per-model than a generic
     family number (Flux 2 Klein is 4-step, Z-Image Turbo 8-step), so they win over
     the factory family numbers.
  3. Factory family defaults — the curated starting points below.

A preset is a plain dict with any of FIELDS (+ denoise for img2img/inpaint).
Missing keys mean "leave the control as-is". This schema (the ``gen_defaults`` key,
family names and FIELDS) matches Replicant CharLab so the saved defaults are shared.
"""
from __future__ import annotations

from pathlib import Path

from . import discovery

FIELDS = ["steps", "cfg", "sampler", "scheduler", "clip_skip", "width", "height"]
# Editor order (native families first, then SD-family).
FAMILIES = ["Flux", "Z-Image", "Qwen", "SDXL", "Pony", "Illustrious"]
_NATIVE_FAMILIES = {"Flux", "Z-Image", "Qwen"}
_CONFIG_KEY = "gen_defaults"  # key in .orphansuite.json (shared across plugins)

# Curated factory starting points. Native pipelines own their sampler/scheduler so
# those are neutral ("default"/""); per-model step/cfg/resolution still override
# these at selection time (layer 2 above).
FACTORY = {
    "Flux":        {"steps": 20, "cfg": 3.5, "sampler": "default",
                    "scheduler": "", "clip_skip": 1, "width": 896, "height": 1152},
    "Z-Image":     {"steps": 8, "cfg": 1.0, "sampler": "default",
                    "scheduler": "", "clip_skip": 1, "width": 896, "height": 1152},
    "Qwen":        {"steps": 20, "cfg": 4.0, "sampler": "default",
                    "scheduler": "", "clip_skip": 1, "width": 896, "height": 1152},
    "SDXL":        {"steps": 30, "cfg": 7.0, "sampler": "DPM++ 2M",
                    "scheduler": "Karras", "clip_skip": 2, "width": 1024, "height": 1024},
    "Pony":        {"steps": 28, "cfg": 7.0, "sampler": "DPM++ 2M SDE",
                    "scheduler": "Karras", "clip_skip": 2, "width": 1024, "height": 1024},
    "Illustrious": {"steps": 28, "cfg": 6.0, "sampler": "Euler a",
                    "scheduler": "Normal", "clip_skip": 2, "width": 832, "height": 1216},
}

# Per-mode denoise starting points (img2img / inpaint pages).
_DENOISE = {"img2img": 0.6, "inpaint": 0.75}

# Keep within the UI slider bounds.
_W_MIN, _W_MAX, _STEPS_MAX, _CFG_MIN, _CFG_MAX = 256, 2048, 60, 1.0, 15.0


def _snap(v, lo=_W_MIN, hi=_W_MAX, step=64) -> int:
    return max(lo, min(hi, int(round(v / step) * step)))


def is_native_family(family: str) -> bool:
    return family in _NATIVE_FAMILIES


def family_of(model_value: str) -> str | None:
    """Family for a dropdown value: SD → SDXL/Pony/Illustrious; native →
    Flux/Z-Image/Qwen; None if no model."""
    backend, ident = discovery.parse_model_value(model_value)
    if backend == "sd":
        return discovery.categorize_sdxl(Path(ident).stem)
    if backend == "native":
        return discovery.categorize_native(ident)
    return None


# --- persisted user overrides (shared .orphansuite.json) -------------------

def user_overrides() -> dict:
    """All per-family overrides: {family: {field: value}}."""
    from . import paths
    d = paths.get_shared(_CONFIG_KEY, {})
    return d if isinstance(d, dict) else {}


def set_overrides(family: str, values: dict) -> None:
    """Persist a family's override (known FIELDS, non-None) to the shared config."""
    from . import paths
    cur = dict(user_overrides())
    cur[family] = {k: values[k] for k in FIELDS if values.get(k) is not None}
    paths.set_shared(_CONFIG_KEY, cur)


def clear_overrides(family: str) -> None:
    """Drop a family's override → reverts to factory."""
    from . import paths
    cur = dict(user_overrides())
    if family in cur:
        cur.pop(family, None)
        paths.set_shared(_CONFIG_KEY, cur)


def has_override(family: str) -> bool:
    return bool(user_overrides().get(family))


# --- resolved values -------------------------------------------------------

def factory(family: str) -> dict:
    """Concrete factory defaults for a family (what the editor shows by default)."""
    return dict(FACTORY.get(family, FACTORY["SDXL"]))


def effective(family: str) -> dict:
    """Factory family defaults + the user's saved override (no per-model data) —
    used by the editor to show the current default."""
    base = factory(family)
    base.update(user_overrides().get(family) or {})
    return base


def _native_from_model(model_value, get_default_settings) -> dict:
    """A native model's own Wan2GP defaults → preset dict (portrait-oriented)."""
    backend, ident = discovery.parse_model_value(model_value)
    out: dict = {}
    if backend != "native" or not callable(get_default_settings):
        return out
    try:
        d = dict(get_default_settings(ident) or {})
    except Exception:
        return out
    steps = d.get("num_inference_steps")
    if isinstance(steps, (int, float)) and steps > 0:
        out["steps"] = int(min(steps, _STEPS_MAX))
    g = d.get("guidance_scale")
    if isinstance(g, (int, float)):
        # Distilled native models use guidance 0/None; clamp into the slider.
        out["cfg"] = float(max(_CFG_MIN, min(_CFG_MAX, g))) if g else _CFG_MIN
    try:
        w, h = str(d.get("resolution")).lower().split("x")
        out["width"], out["height"] = _snap(int(w)), _snap(int(h))
    except Exception:
        pass
    return out


def for_model(model_value, mode=None, get_default_settings=None) -> dict:
    """Recommended settings to apply on model selection. {} if no model.
    Priority: user override > native per-model defaults > factory family. Adds a
    per-mode denoise on the img2img / inpaint pages."""
    fam = family_of(model_value)
    if not fam:
        return {}
    base = factory(fam)
    if is_native_family(fam):
        base.update(_native_from_model(model_value, get_default_settings))
    base.update(user_overrides().get(fam) or {})  # user choice wins
    if mode in _DENOISE:
        base["denoise"] = _DENOISE[mode]
    return base
