"""Named Prompt Library — save and recall a full generation setup (prompt, negative,
every generation setting, model, LoRAs and post-process settings) and reuse it on any
of the three image tabs.

ONE shared collection across txt2img / img2img / MultiCanvas. Persisted under the
``"prompt_library"`` key of the already-gitignored ``.imagesuite.json`` (the same file
that holds ui_state), so saved prompts never reach the repo — no new file, no new
gitignore entry. Values are JSON scalars/lists only; reference images aren't stored.
"""
from __future__ import annotations

import threading

from . import paths

_LOCK = threading.Lock()
_KEY = "prompt_library"


def _store() -> dict:
    pl = paths.load_config().get(_KEY)
    return pl if isinstance(pl, dict) else {}


def names() -> list[str]:
    """Saved entry names, sorted (the dropdown's choices)."""
    with _LOCK:
        return sorted(_store().keys())


def get(name: str) -> dict | None:
    """The stored field→value dict for ``name`` (a copy), or None if absent."""
    if not name:
        return None
    with _LOCK:
        entry = _store().get(name)
        return dict(entry) if isinstance(entry, dict) else None


def save(name: str, entry: dict) -> list[str]:
    """Create or overwrite ``name`` with ``entry`` (filtered to JSON-able values).
    Returns the updated sorted name list."""
    name = (name or "").strip()
    if not name:
        return names()
    with _LOCK:
        cfg = paths.load_config()
        pl = cfg.get(_KEY)
        if not isinstance(pl, dict):
            pl = {}
        pl[name] = _jsonable(entry)
        cfg[_KEY] = pl
        paths.save_config()
        return sorted(pl.keys())


def delete(name: str) -> list[str]:
    """Remove ``name`` if present. Returns the updated sorted name list."""
    with _LOCK:
        cfg = paths.load_config()
        pl = cfg.get(_KEY)
        if isinstance(pl, dict) and name in pl:
            del pl[name]
            cfg[_KEY] = pl
            paths.save_config()
        return sorted(pl.keys()) if isinstance(pl, dict) else []


def _jsonable(entry: dict) -> dict:
    """Keep only JSON-serializable scalars / lists of scalars (drop images, objects,
    and any odd values that slip through) so the config stays clean + reloadable."""
    out: dict = {}
    for k, v in (entry or {}).items():
        if v is None or isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [x for x in v if isinstance(x, (str, int, float, bool))]
    return out
