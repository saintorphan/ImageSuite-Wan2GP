"""Filesystem-backed overlay library (managed by the Overlays tab).

A flat set of folders under ``paths.overlays_dir()``, each holding image files
(transparent PNGs, frames, stickers — anything Pillow can open). The Overlays
tab does full CRUD; the per-page Overlays strip is read-only.

Every public function takes plain folder/name strings from the UI and is guarded
against path traversal: resolved targets must stay inside the overlays root.
"""
from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

from . import paths

logger = logging.getLogger("imagesuite.overlays")

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
ROOT_LABEL = "(root)"  # the overlays_dir itself, shown as a pseudo-folder


def _root() -> Path:
    d = paths.overlays_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d.resolve()


def _safe(*parts: str) -> Path:
    """Resolve a path under the overlays root, rejecting traversal/escape."""
    root = _root()
    p = root
    for part in parts:
        part = (part or "").strip()
        if part and part != ROOT_LABEL:
            p = p / part
    p = p.resolve()
    if p != root and root not in p.parents:
        raise ValueError("Path escapes the overlays library.")
    return p


def _folder_dir(folder: str) -> Path:
    return _root() if (not folder or folder == ROOT_LABEL) else _safe(folder)


def list_folders() -> list[str]:
    """Folder names (one level), with the root pseudo-folder first."""
    root = _root()
    subs = sorted(p.name for p in root.iterdir()
                  if p.is_dir() and not p.is_symlink())
    return [ROOT_LABEL] + subs


def list_images(folder: str) -> list[str]:
    """Absolute paths of image files in ``folder`` (sorted by name)."""
    d = _folder_dir(folder)
    if not d.is_dir():
        return []
    return [str(p) for p in sorted(d.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def create_folder(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise ValueError("Enter a folder name.")
    if "/" in name or "\\" in name:
        raise ValueError("Folder names can't contain slashes.")
    d = _safe(name)
    if d.exists():
        raise ValueError(f"Folder '{name}' already exists.")
    d.mkdir(parents=True)
    return name


def delete_folder(name: str) -> None:
    if not name or name == ROOT_LABEL:
        raise ValueError("Pick a folder to delete (not the root).")
    d = _safe(name)
    if not d.is_dir():
        raise ValueError(f"No such folder '{name}'.")
    shutil.rmtree(d)


def save_uploads(folder: str, file_paths: list[str]) -> int:
    """Copy uploaded files (Gradio temp paths) into ``folder``. Returns the count
    of images actually saved (non-images are skipped)."""
    d = _folder_dir(folder)
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    for fp in file_paths or []:
        src = Path(fp)
        if src.suffix.lower() not in IMAGE_EXTS:
            continue
        dst = _unique(d / _safe_name(src.name))
        shutil.copy2(src, dst)
        n += 1
    return n


def _safe_name(name: str) -> str:
    """Sanitize an uploaded filename: keep the (allow-listed image) extension,
    strip path separators, allow only safe characters, and cap the length."""
    base = Path(name or "").name  # drop any path separators / traversal
    stem, suf = Path(base).stem, Path(base).suffix
    if suf.lower() not in IMAGE_EXTS:
        stem, suf = base, ""
    # Allow-list: letters, digits, and a few safe punctuation marks.
    stem = re.sub(r"[^A-Za-z0-9 ._()\-]", "_", stem).strip(" .") or "overlay"
    return (stem[:120] + suf)


def _unique(dst: Path) -> Path:
    """Avoid clobbering: foo.png → foo (1).png if it exists."""
    if not dst.exists():
        return dst
    stem, suf, i = dst.stem, dst.suffix, 1
    while True:
        cand = dst.with_name(f"{stem} ({i}){suf}")
        if not cand.exists():
            return cand
        i += 1


def rename_image(folder: str, name: str, new_name: str) -> str:
    src = _safe(folder, name) if folder and folder != ROOT_LABEL else _safe(name)
    if not src.is_file():
        raise ValueError("Pick an image to rename.")
    new_name = (new_name or "").strip()
    if not new_name:
        raise ValueError("Enter a new name.")
    if "/" in new_name or "\\" in new_name:
        raise ValueError("Names can't contain slashes.")
    if not Path(new_name).suffix:
        new_name += src.suffix  # keep the extension if the user dropped it
    dst = src.with_name(new_name)
    if dst.exists():
        raise ValueError(f"'{new_name}' already exists here.")
    src.rename(dst)
    return dst.name


def delete_image(folder: str, name: str) -> None:
    src = _safe(folder, name) if folder and folder != ROOT_LABEL else _safe(name)
    if not src.is_file():
        raise ValueError("Pick an image to delete.")
    src.unlink()


def move_image(folder: str, name: str, dest_folder: str) -> str:
    src = _safe(folder, name) if folder and folder != ROOT_LABEL else _safe(name)
    if not src.is_file():
        raise ValueError("Pick an image to move.")
    dst_dir = _folder_dir(dest_folder)
    if dst_dir == src.parent:
        return src.name  # same folder, no-op
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = _unique(dst_dir / src.name)
    shutil.move(str(src), str(dst))
    return dst.name
