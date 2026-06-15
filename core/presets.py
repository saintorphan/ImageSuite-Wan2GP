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

# --- distilled "Turbo" presets (SD/SDXL family only) -----------------------
# Each non-Off mode pins a distill LoRA (matched by filename hint against the
# user's SDXL LoRA dir), the scheduler the distillation expects (LCM/TCD), and
# clamped few-step / low-CFG numbers. ``hints`` are case-insensitive substrings
# searched in LoRA filenames; the first match wins. ``sampler``/``scheduler`` are
# create_scheduler() keys (LCM → LCMScheduler, TCD → DPM++ family at CFG~1).
# At CFG ~1 classifier-free guidance is effectively off, so the negative prompt
# is ignored — the UI surfaces this.
TURBO_OFF = "Off"
TURBO_PROFILES = {
    # name: distill LoRA filename hints, scheduler sampler, scheduler variant,
    #       steps, cfg.
    "LCM": {
        "hints": ["lcm"],
        "sampler": "LCM", "scheduler": "Normal",
        "steps": 6, "cfg": 1.5,
    },
    "Hyper-SD": {
        # Hyper-SD ships LCM-compatible distill LoRAs (and an 8-step CFG variant);
        # the LCM scheduler is the safe default for the common few-step weights.
        "hints": ["hyper", "hypersd", "hyper-sd", "hyper_sd"],
        "sampler": "LCM", "scheduler": "Normal",
        "steps": 8, "cfg": 1.0,
    },
    "Lightning": {
        # SDXL-Lightning distillation runs on Euler at CFG 1 (no extra scheduler).
        "hints": ["lightning", "lightnig", "sdxl_lightning", "sdxl-lightning"],
        "sampler": "Euler", "scheduler": "SGM Uniform",
        "steps": 8, "cfg": 1.0,
    },
}
# Turbo step counts are intentionally tiny; clamp to a sane distill range so a
# stray profile edit can never request 60 steps from a 4-8-step distillation.
_TURBO_STEPS_MIN, _TURBO_STEPS_MAX = 4, 12


def turbo_choices() -> list[str]:
    """Dropdown values: Off (default) + each distilled profile name."""
    return [TURBO_OFF] + list(TURBO_PROFILES.keys())


def _find_distill_lora(hints, lora_dir) -> str | None:
    """First LoRA filename (stem) under *lora_dir* whose name contains any hint
    (case-insensitive), searched recursively. None if the dir is missing or no
    file matches — the caller then warns and falls back to normal generation."""
    try:
        lora_dir = Path(lora_dir)
        if not lora_dir.is_dir():
            return None
        hints = [h.lower() for h in (hints or [])]
        for ext in (".safetensors", ".ckpt", ".pt"):
            for f in sorted(lora_dir.rglob(f"*{ext}")):
                name = f.name.lower()
                if any(h in name for h in hints):
                    return f.stem
    except Exception:
        return None
    return None


def resolve_turbo(turbo, lora_dir) -> dict:
    """Resolve a Turbo selection into an applicable profile against *lora_dir*.

    Returns a dict:
      - {"enabled": False}                          — Off / unknown (unchanged).
      - {"enabled": True, ...}                       — matched: carries ``lora``
        (stem), ``sampler``, ``scheduler``, ``steps`` (clamped), ``cfg``, ``name``.
      - {"enabled": False, "warn": "..."}            — selected but no distill LoRA
        found; caller warns and runs normally.

    Does NOT touch the GPU or load anything — pure file lookup + clamping, so it's
    safe to call before deciding whether to apply.
    """
    name = (turbo or TURBO_OFF)
    prof = TURBO_PROFILES.get(name)
    if not prof:
        return {"enabled": False}
    lora = _find_distill_lora(prof.get("hints"), lora_dir)
    if not lora:
        return {"enabled": False, "warn": (
            f"Turbo '{name}' selected but no matching distill LoRA "
            f"(hint: {', '.join(prof.get('hints') or [])}) was found in the SDXL "
            f"LoRA dir — generating normally.")}
    steps = max(_TURBO_STEPS_MIN, min(_TURBO_STEPS_MAX, int(prof["steps"])))
    cfg = float(max(_CFG_MIN, min(_CFG_MAX, prof["cfg"])))
    return {
        "enabled": True, "name": name, "lora": lora,
        "sampler": prof["sampler"], "scheduler": prof["scheduler"],
        "steps": steps, "cfg": cfg,
    }


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
    w, h = discovery.parse_size(str(d.get("resolution") or ""))
    if w > 0 and h > 0:
        out["width"], out["height"] = _snap(w), _snap(h)
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
