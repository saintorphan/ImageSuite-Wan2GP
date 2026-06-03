"""Project storage — full-workspace snapshots saved under ``image_suite/projects/``.

A project bundles everything you can currently see: the prompts + all generation and
post-process parameters for every tab (txt2img / img2img / MultiCanvas), the
currently-displayed result images per tab, the post-process reference images
(face / body / colour), and the full MultiCanvas state (base + every layer + mask) as
a serialized JSON blob. It deliberately does NOT archive every generation ever made —
only the visible ones.

On-disk layout (one dir per project, name-sanitised)::

    projects/<name>/
      project.json         # {name, version, tabs:{mode:{param:val}}, results:{mode:[file]},
                           #  refs:{mode:{slot:file}}, has_canvas}
      results/<mode>/*.png # copies of the currently-displayed outputs
      refs/<mode>/<slot>.* # copies of face/body/colour reference images
      canvas/state.json    # serialized MultiCanvas state (data-URLs inside)

This module is pure filesystem/JSON — the UI coupling (collecting param values,
applying them back, driving the canvas bridge) lives in plugin.py.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

from . import paths as _paths

VERSION = 1
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")


# --- naming -----------------------------------------------------------------

def sanitize(name: str) -> str:
    """A safe, single-segment folder name (no traversal, no separators)."""
    s = re.sub(r"[^\w\-. ]+", "_", str(name or "").strip())
    s = s.strip(". ").strip()
    return s[:80] or "Untitled"


def project_path(name: str) -> Path:
    return _paths.projects_dir() / sanitize(name)


def exists(name: str) -> bool:
    p = project_path(name)
    return p.is_dir() and (p / "project.json").exists()


def list_projects() -> list[str]:
    root = _paths.projects_dir()
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "project.json").exists():
            try:
                out.append(json.loads((d / "project.json").read_text()).get("name") or d.name)
            except Exception:
                out.append(d.name)
    return out


# --- save / load ------------------------------------------------------------

def _copy_into(src, dst_dir: Path, stem: str) -> str | None:
    """Copy ``src`` into ``dst_dir`` keeping its extension; return the basename."""
    try:
        sp = Path(str(src))
        if not sp.is_file():
            return None
        dst_dir.mkdir(parents=True, exist_ok=True)
        ext = sp.suffix.lower() if sp.suffix else ".png"
        out = dst_dir / f"{stem}{ext}"
        shutil.copy2(sp, out)
        return out.name
    except Exception:
        return None


def save_project(name: str, *, tabs: dict, results: dict, refs: dict,
                 canvas_state: str | None) -> str:
    """Write/overwrite a project. ``tabs`` = {mode:{param:val}} (JSON-able),
    ``results`` = {mode:[src_path,...]}, ``refs`` = {mode:{slot:src_path}},
    ``canvas_state`` = serialized JSON or None. Returns the project's display name."""
    name = sanitize(name)
    root = project_path(name)
    # fresh tree each save (update-in-place semantics) so stale files don't linger
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    meta = {"name": name, "version": VERSION, "saved_at": int(time.time()),
            "tabs": {}, "results": {}, "refs": {}, "has_canvas": False}

    for mode, params in (tabs or {}).items():
        meta["tabs"][mode] = _jsonable(params)

    for mode, files in (results or {}).items():
        kept = []
        for i, src in enumerate(files or []):
            bn = _copy_into(src, root / "results" / mode, f"{i:03d}")
            if bn:
                kept.append(bn)
        if kept:
            meta["results"][mode] = kept

    for mode, slots in (refs or {}).items():
        rec = {}
        for slot, src in (slots or {}).items():
            if not src:
                continue
            bn = _copy_into(src, root / "refs" / mode, slot)
            if bn:
                rec[slot] = bn
        if rec:
            meta["refs"][mode] = rec

    if canvas_state:
        try:
            cdir = root / "canvas"
            cdir.mkdir(parents=True, exist_ok=True)
            (cdir / "state.json").write_text(canvas_state)
            meta["has_canvas"] = True
        except Exception:
            pass

    (root / "project.json").write_text(json.dumps(meta, indent=2))
    return name


def load_project(name: str) -> dict | None:
    """Return the project's data with on-disk absolute paths for results/refs, plus the
    raw canvas-state JSON string (or None). Returns None if the project is missing."""
    root = project_path(name)
    meta_f = root / "project.json"
    if not meta_f.exists():
        return None
    try:
        meta = json.loads(meta_f.read_text())
    except Exception:
        return None

    results = {}
    for mode, files in (meta.get("results") or {}).items():
        results[mode] = [str(root / "results" / mode / f) for f in files
                         if (root / "results" / mode / f).exists()]
    refs = {}
    for mode, slots in (meta.get("refs") or {}).items():
        refs[mode] = {slot: str(root / "refs" / mode / f)
                      for slot, f in slots.items()
                      if (root / "refs" / mode / f).exists()}
    canvas_state = None
    cs = root / "canvas" / "state.json"
    if meta.get("has_canvas") and cs.exists():
        try:
            canvas_state = cs.read_text()
        except Exception:
            canvas_state = None

    return {"name": meta.get("name") or sanitize(name),
            "tabs": meta.get("tabs") or {}, "results": results,
            "refs": refs, "canvas_state": canvas_state}


def rename_project(old: str, new: str) -> str | None:
    """Rename a project folder + its stored name. Returns the new name, or None on failure."""
    src = project_path(old)
    if not src.is_dir():
        return None
    new = sanitize(new)
    dst = project_path(new)
    if dst.exists():
        return None  # don't clobber an existing project
    try:
        src.rename(dst)
        mf = dst / "project.json"
        if mf.exists():
            meta = json.loads(mf.read_text())
            meta["name"] = new
            mf.write_text(json.dumps(meta, indent=2))
        return new
    except Exception:
        return None


def delete_project(name: str) -> bool:
    root = project_path(name)
    if not root.is_dir():
        return False
    shutil.rmtree(root, ignore_errors=True)
    return not root.exists()


# --- flush orphaned outputs -------------------------------------------------

def orphaned_outputs() -> tuple[list[Path], int]:
    """Image files sitting in the raw generation output dirs (sd_gen + inpaint).
    Projects keep their own copies and the on-screen galleries restore from
    .cache/persist/results — both elsewhere — so everything here is reclaimable.
    Returns (files, total_bytes)."""
    files, total = [], 0
    for d in _paths.gen_output_dirs():
        try:
            if not d.is_dir():
                continue
            for f in d.iterdir():
                if f.is_file() and f.suffix.lower() in _IMG_EXTS:
                    try:
                        total += f.stat().st_size
                    except Exception:
                        pass
                    files.append(f)
        except Exception:
            pass
    return files, total


def flush_outputs() -> tuple[int, int]:
    """Delete every orphaned output image. Returns (count_deleted, bytes_freed)."""
    files, total = orphaned_outputs()
    n, freed = 0, 0
    for f in files:
        try:
            sz = f.stat().st_size
            f.unlink()
            n += 1
            freed += sz
        except Exception:
            pass
    return n, freed


def human_size(n: int) -> str:
    """Bytes → a compact human-readable string (KB/MB/GB)."""
    n = float(max(0, int(n or 0)))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return (f"{int(n)} {unit}" if unit == "B" else f"{n:.1f} {unit}")
        n /= 1024
    return f"{n:.1f} TB"


def flush_label(count: int, total_bytes: int) -> str:
    """The line shown beside the Flush Outputs button."""
    if not count:
        return "✅ No orphaned generations — the output cache is clean."
    return (f"**≈ {human_size(total_bytes)}** in **{count}** orphaned generation"
            + ("s" if count != 1 else "") + " (not in any project, not on screen).")


def _jsonable(d: dict) -> dict:
    """Drop non-JSON-serialisable values (mirrors prompt_library's filter)."""
    out = {}
    for k, v in (d or {}).items():
        try:
            json.dumps(v)
            out[k] = v
        except Exception:
            pass
    return out
