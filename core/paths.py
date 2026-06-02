"""Filesystem layout for Image Suite.

Two kinds of location:

  * Plugin-specific — under ``<wan2gp_root>/image_suite/`` (override IMAGESUITE_DIR):
        outputs_dir    generated images
        overlays_dir   the overlay library
        cache_dir      scratch

  * OrphanSuite SHARED — resources several saintorphan plugins use, kept in ONE
    central place ``<wan2gp_root>/orphansuite/`` (override ORPHANSUITE_DIR) so they
    aren't duplicated per plugin:
        sdxl_models_dir  SDXL/Pony/Illustrious checkpoints
        sdxl_loras_dir   SDXL-family LoRAs
        models_dir       face / ADetailer / face-swap weights

Any dir can be repointed from the Settings panel. The SHARED dirs persist to the
cross-plugin ``<cwd>/.orphansuite.json`` (so every saintorphan plugin follows),
while plugin-specific paths (outputs) and per-tab UI state persist to
``<cwd>/.imagesuite.json``. Shared dirs fall back to a legacy
``<root>/character_lab/<leaf>`` location for back-compat, and the Settings
"link existing folder" action symlinks models you already have (a1111/Forge/…)
into the shared area without copying.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("imagesuite.paths")

_DEFAULT_SUBDIR = "image_suite"
_CONFIG_NAME = ".imagesuite.json"
_config: dict | None = None


# --- config persistence ----------------------------------------------------

def _config_path() -> Path:
    # Stable location (cwd = Wan2GP root), independent of the configurable dirs.
    return Path(os.getcwd()) / _CONFIG_NAME


def load_config() -> dict:
    global _config
    if _config is None:
        try:
            _config = json.loads(_config_path().read_text())
        except Exception:
            _config = {}
    return _config


def save_config() -> None:
    try:
        _config_path().write_text(json.dumps(load_config(), indent=2))
    except Exception:
        logger.warning("Could not write %s", _config_path(), exc_info=True)


def low_vram_only() -> bool:
    """Whether the native model list is filtered to low-VRAM-friendly models."""
    return bool(load_config().get("low_vram_only", False))


def set_low_vram_only(value: bool) -> None:
    load_config()["low_vram_only"] = bool(value)
    save_config()


def get_ui_state() -> dict:
    """Last-used per-tab settings ``{mode: {field: value}}`` so the image tabs
    come back as the user left them after a restart. Persisted in the config."""
    st = load_config().get("ui_state")
    return st if isinstance(st, dict) else {}


def set_ui_state(mode: str, values: dict) -> None:
    cfg = load_config()
    st = cfg.get("ui_state")
    if not isinstance(st, dict):
        st = {}
    st[str(mode)] = dict(values)
    cfg["ui_state"] = st
    save_config()


def set_dirs(*, outputs=None, models=None, sdxl_models=None, sdxl_loras=None) -> None:
    """Persist directory overrides. SHARED resources (face/ADetailer weights, SDXL
    checkpoints + LoRAs) go to the cross-plugin .orphansuite.json so every plugin
    follows; plugin-specific (outputs) goes to .imagesuite.json. '' clears it."""
    if outputs is not None:
        cfg = load_config()
        cfg["outputs_dir"] = str(Path(outputs).expanduser()) if outputs else ""
        save_config()
    for key, val in (("models_dir", models), ("sdxl_models_dir", sdxl_models),
                     ("sdxl_loras_dir", sdxl_loras)):
        if val is not None:
            set_shared_dir(key, val)
    ensure_dirs()


# --- roots -----------------------------------------------------------------

def lab_root() -> Path:
    override = os.environ.get("IMAGESUITE_DIR")
    if override:
        return Path(override).expanduser()
    return Path(os.getcwd()) / _DEFAULT_SUBDIR


def _dir(key: str, default_leaf: str) -> Path:
    val = load_config().get(key)
    return Path(val).expanduser() if val else lab_root() / default_leaf


# --- OrphanSuite shared resources -------------------------------------------
# SDXL/Pony/Illustrious checkpoints, SDXL LoRAs and face/ADetailer/face-swap
# weights are SHARED across the saintorphan plugins, so their paths live in ONE
# central config — ``.orphansuite.json`` at the Wan2GP root, NOT each plugin's own
# config. Set it once (Settings → OrphanSuite) and every plugin (Image Suite,
# Replicant CharLab, Reel2Reel) reads the same value. Override the shared root
# directory with ORPHANSUITE_DIR.
#
# Resolution for a shared dir (first hit wins):
#   1. .orphansuite.json[key]       — the shared, canonical cross-plugin setting
#   2. .imagesuite.json[key]        — legacy per-plugin override
#   3. <root>/orphansuite/<leaf>    — the shared area, if it already holds files
#   4. <root>/character_lab/<leaf>  — legacy CharLab location (back-compat)
#   5. <root>/orphansuite/<leaf>    — default (created/symlinked on demand)
_ORPHAN_CONFIG_NAME = ".orphansuite.json"
_ORPHAN_SUBDIR = "orphansuite"
_LEGACY_SUBDIR = "character_lab"
_orphan_cfg: dict | None = None
_orphan_cfg_mtime: float | None = None


def orphansuite_root() -> Path:
    """Central shared-resource root for all saintorphan plugins."""
    override = os.environ.get("ORPHANSUITE_DIR")
    if override:
        return Path(override).expanduser()
    return Path(os.getcwd()) / _ORPHAN_SUBDIR


def reload_shared_config() -> dict:
    """Force a re-read of the cross-plugin .orphansuite.json from disk."""
    global _orphan_cfg, _orphan_cfg_mtime
    p = Path(os.getcwd()) / _ORPHAN_CONFIG_NAME
    try:
        _orphan_cfg_mtime = p.stat().st_mtime
        _orphan_cfg = json.loads(p.read_text())
    except Exception:
        _orphan_cfg = {}
        _orphan_cfg_mtime = None
    return _orphan_cfg if isinstance(_orphan_cfg, dict) else {}


def load_shared_config() -> dict:
    """The cross-plugin .orphansuite.json (shared model/dir settings). Cached by
    mtime and reloaded when the file changes on disk, so foreign writes from a
    sibling saintorphan plugin (or a manual edit) are picked up within a session."""
    global _orphan_cfg, _orphan_cfg_mtime
    p = Path(os.getcwd()) / _ORPHAN_CONFIG_NAME
    try:
        mtime = p.stat().st_mtime
    except Exception:
        mtime = None
    if _orphan_cfg is None or mtime != _orphan_cfg_mtime:
        return reload_shared_config()
    return _orphan_cfg if isinstance(_orphan_cfg, dict) else {}


def get_shared(key, default=None):
    """Read any value from the cross-plugin .orphansuite.json."""
    v = load_shared_config().get(key)
    return default if v is None else v


def set_shared(key, value) -> None:
    """Persist any JSON value to the cross-plugin .orphansuite.json (shared across
    saintorphan plugins — e.g. shared dirs and per-family gen_defaults). Re-reads
    the file fresh and MERGES, so a concurrent sibling plugin's keys aren't
    clobbered (last-writer-wins)."""
    global _orphan_cfg, _orphan_cfg_mtime
    cfg = reload_shared_config()  # fresh from disk, not the (possibly stale) cache
    cfg[key] = value
    _orphan_cfg = cfg
    p = Path(os.getcwd()) / _ORPHAN_CONFIG_NAME
    try:
        p.write_text(json.dumps(cfg, indent=2))
        _orphan_cfg_mtime = p.stat().st_mtime  # keep cache fresh after our own write
    except Exception:
        logger.warning("could not write %s", _ORPHAN_CONFIG_NAME, exc_info=True)


def set_shared_dir(key: str, path: str) -> None:
    """Persist a shared resource dir to the cross-plugin .orphansuite.json."""
    set_shared(key, str(Path(path).expanduser()) if path else "")


def _has_files(d: Path) -> bool:
    try:
        return d.is_dir() and any(d.iterdir())
    except Exception:
        return False


def _shared_dir(key: str, default_leaf: str) -> Path:
    """Resolve a shared (cross-plugin) resource dir — see resolution order above."""
    sv = load_shared_config().get(key)
    if sv:
        return Path(sv).expanduser()
    own = load_config().get(key)            # legacy per-plugin override
    if own:
        return Path(own).expanduser()
    shared = orphansuite_root() / default_leaf
    legacy = Path(os.getcwd()) / _LEGACY_SUBDIR / default_leaf
    if _has_files(shared):
        return shared
    if _has_files(legacy):
        return legacy
    return shared


# Logical link targets → the dir the loader actually scans. Resolved through the
# SAME functions consumers use (honours a configured custom dir) AND placing the
# face/body/birefnet weights in the partitioned subdirs the loaders expect.
LINK_TARGETS = ["sdxl_models", "sdxl_loras", "face", "body", "birefnet"]


def link_target_dir(target: str) -> Path:
    """Resolve a logical link target to the exact dir its loader scans."""
    if target == "sdxl_models":
        return sdxl_models_dir()
    if target == "sdxl_loras":
        return sdxl_loras_dir()
    if target in ("face", "body", "birefnet"):
        return models_dir() / target  # loaders read models_dir()/<face|body|birefnet>
    return models_dir() / target


def shared_resource_dir(leaf: str) -> Path:
    """Back-compat shim: the canonical home for a shared resource. Prefer
    ``link_target_dir`` (which honours configured dirs + subdir layout)."""
    return orphansuite_root() / leaf


def link_existing_into_shared(
        source_dir: str, target: str,
        exts=(".safetensors", ".ckpt", ".pt", ".pth", ".gguf", ".bin",
              ".onnx", ".sft", ".vae")) -> str:
    """Symlink model files from ``source_dir`` into the dir its loader actually
    scans (``link_target_dir`` — honours a configured custom dir AND the
    face/body/birefnet subdir layout), so models you already keep (a1111 / Forge /
    a drive folder) are reused without copying or moving the originals. Real
    symlinks on POSIX; copies as a fallback on Windows. Returns a summary."""
    import shutil
    src = Path(source_dir).expanduser()
    if not src.is_dir():
        raise ValueError(f"Not a folder: {source_dir}")
    dst_root = link_target_dir(target)
    dst_root.mkdir(parents=True, exist_ok=True)
    linked = copied = skipped = 0
    for f in sorted(src.iterdir()):
        if not f.is_file() or (exts and f.suffix.lower() not in exts):
            continue
        dst = dst_root / f.name
        if dst.exists() or dst.is_symlink():
            skipped += 1
            continue
        target = f.resolve()  # follow existing symlinks to the real file
        try:
            dst.symlink_to(target)
            linked += 1
        except OSError:
            try:
                shutil.copy2(target, dst)
                copied += 1
            except Exception:
                logger.warning("could not link/copy %s", f, exc_info=True)
    bits = []
    if linked:
        bits.append(f"linked {linked}")
    if copied:
        bits.append(f"copied {copied}")
    if skipped:
        bits.append(f"skipped {skipped} (already there)")
    return (", ".join(bits) or "no model files found") + f" → {dst_root}"


def outputs_dir() -> Path:
    # Image-Suite-specific: keep generated stills under our own root by default.
    return _dir("outputs_dir", "outputs")


def models_dir() -> Path:
    return _shared_dir("models_dir", "models")


def sdxl_models_dir() -> Path:
    return _shared_dir("sdxl_models_dir", "sdxl_models")


def sdxl_loras_dir() -> Path:
    return _shared_dir("sdxl_loras_dir", "sdxl_loras")


def overlays_dir() -> Path:
    # User-managed library of overlay PNGs (transparent stickers, frames, etc.)
    # organised into folders via the Overlays tab. Own root by default.
    return _dir("overlays_dir", "overlays")


def cache_dir() -> Path:
    return lab_root() / ".cache"


def ensure_dirs() -> Path:
    """Create the directory tree if missing. Idempotent; called on plugin setup."""
    for d in (outputs_dir(), models_dir(), sdxl_models_dir(), sdxl_loras_dir(),
              overlays_dir(), cache_dir()):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.warning("Could not create %s", d, exc_info=True)
    return lab_root()
