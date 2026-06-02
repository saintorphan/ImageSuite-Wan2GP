"""Auto-install missing runtime *code* dependencies so features self-heal instead
of hard-failing mid-run. (Models are never auto-pulled — that's core.models.)"""
from __future__ import annotations

import importlib
import logging
import subprocess
import sys

logger = logging.getLogger("imagesuite.deps")

# import-name -> pip spec. The body-swap / SD-image path needs these beyond what
# Wan2GP bundles.
BODY_SWAP_DEPS = {
    "kornia": "kornia",              # BiRefNet segmentation custom modeling code
    "controlnet_aux": "controlnet_aux",  # OpenPose preprocessor
    "ultralytics": "ultralytics",    # YOLOv8 (ADetailer / person detection)
}


def ensure(import_to_pip: dict, progress=None, label="dependencies") -> bool:
    """pip-install any of import_to_pip whose import-name isn't importable. If a
    Gradio ``progress`` is passed, show a status so it doesn't look frozen.

    Returns True once every requested import is importable. Raises RuntimeError
    with a manual-pip hint if the install ran but the imports still fail, so
    callers fail with actionable guidance instead of a cryptic downstream
    ``ModuleNotFoundError``."""
    # import-name -> pip spec for the entries that aren't importable yet.
    missing = {}
    for imp, pip in import_to_pip.items():
        try:
            importlib.import_module(imp)
        except Exception:
            missing[imp] = pip
    if not missing:
        return True
    missing_pip = list(missing.values())
    logger.info("Replicant: auto-installing missing deps: %s", missing_pip)
    if progress is not None:
        try:
            progress(0.0, desc=f"Installing {label}: {', '.join(missing_pip)} "
                               f"(first run only — see console for progress)…")
        except Exception:
            pass
    install_error = None
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_pip])
    except Exception as exc:
        install_error = exc
        logger.warning("auto-install failed for %s", missing_pip, exc_info=True)
    # Re-verify importability — pip "succeeding" doesn't guarantee the import
    # works (partial install, resolution conflict, offline), and we must not
    # let a swallowed failure surface later as a cryptic ModuleNotFoundError.
    importlib.invalidate_caches()
    still_missing = {}
    for imp, pip in missing.items():
        try:
            importlib.import_module(imp)
        except Exception:
            still_missing[imp] = pip
    if still_missing:
        hint = f"pip install {' '.join(still_missing.values())}"
        raise RuntimeError(
            f"Could not install {label}: {', '.join(still_missing.values())}. "
            f"Install manually with: {hint}"
        ) from install_error
    return True


def ensure_body_swap(progress=None) -> bool:
    return ensure(BODY_SWAP_DEPS, progress=progress, label="body-swap dependencies")
