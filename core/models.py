"""Registry of models this extension needs that Wan2GP does NOT bundle.

The Prereqs Models panel reads this to show per-model status + a Download button.
NOTHING downloads without an explicit button press: generation runs under
``no_auto_download()`` (HF offline), so a missing model raises a clear error
telling the user to fetch it here first — it never silently pulls.

Source kinds per entry:
  - url:  single-file download → ``subpath`` under models_dir() (face models).
  - repo: HuggingFace repo. ``repo_local_dir`` set → snapshot into that dir
          (BiRefNet); else into the HF cache (ControlNet / IP-Adapter / annotator).
  - url + extract: a .zip fetched and unpacked into ``extract_to`` (buffalo_l).
"""
from __future__ import annotations

import contextlib
import logging
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from . import paths

logger = logging.getLogger("imagesuite.models")

# BiRefNet (body-swap segmentation) downloads into the plugin's own models dir;
# segment_foreground falls back to the HF model id if it isn't present there.
# Resolved lazily (NOT at import time) so it follows a later-repointed models dir —
# otherwise the Models panel would download/report under the old dir while the
# runtime body-swap path looks under the new one (see analysis M9).
_BUFFALO_DIR = str(Path.home() / ".insightface" / "models" / "buffalo_l")


def _birefnet_dir() -> str:
    return str(paths.models_dir() / "birefnet")


@dataclass
class ModelSpec:
    key: str
    name: str
    purpose: str
    required: bool
    url: str | None = None       # single-file (or .zip with extract=True)
    subpath: str = ""            # rel to models_dir() for url entries
    repo: str = ""               # HF repo id
    repo_local_dir: str = ""     # absolute local_dir for snapshot (else HF cache)
    extract: bool = False        # url is a .zip → unpack into extract_to
    extract_to: str = ""         # absolute dir for extracted contents
    note: str = ""

    @property
    def downloadable(self) -> bool:
        return bool(self.url or self.repo)

    def local_dir(self) -> str:
        """Resolve the snapshot target dir lazily so it tracks a later-repointed
        models dir. BiRefNet lives under models_dir()/birefnet; other repo
        entries snapshot into the HF cache (empty)."""
        if self.repo == "ZhengPeng7/BiRefNet":
            return _birefnet_dir()
        return self.repo_local_dir

    def display_path(self) -> str:
        if self.repo and self.local_dir():
            return self.local_dir()
        if self.repo:
            return f"HF cache · {self.repo}"
        if self.extract:
            return self.extract_to
        return str(paths.models_dir() / self.subpath)

    def is_present(self) -> bool:
        if self.repo:
            local = self.local_dir()
            if local and Path(local).is_dir() and any(Path(local).iterdir()):
                return True
            # Fall back to the HF cache (the loader uses the repo id when the local
            # dir is absent, e.g. BiRefNet) so a cached model isn't read as missing.
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(self.repo, local_files_only=True)
                return True
            except Exception:
                return False
        if self.extract:
            d = Path(self.extract_to)
            if d.is_dir() and any(d.iterdir()):
                return True
            if self.key == "buffalo_l":
                # Also accept the shared-dir layout the face loader looks in.
                face = paths.models_dir() / "face"
                for alt in (face / "buffalo_l", face / "models" / "buffalo_l"):
                    if alt.is_dir() and any(alt.iterdir()):
                        return True
            return False
        return (paths.models_dir() / self.subpath).is_file()


REGISTRY: list[ModelSpec] = [
    # --- face swap / enhancers ---
    ModelSpec("inswapper_128", "InSwapper 128 (face swap)",
              "Face swap onto the base + base-face→poses identity lock.",
              required=True, subpath="face/inswapper_128.onnx",
              url="https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.onnx"),
    ModelSpec("gfpgan", "GFPGAN v1.4 (face enhancer)",
              "Optional face restoration after swaps.", required=False,
              subpath="face/GFPGANv1.4.onnx",
              url="https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GFPGANv1.4.onnx"),
    ModelSpec("codeformer", "CodeFormer (face enhancer)",
              "Optional face restoration after swaps.", required=False,
              subpath="face/codeformer.onnx",
              url="https://huggingface.co/facefusion/models-3.0.0/resolve/main/codeformer.onnx"),
    ModelSpec("face_yolov8s", "ADetailer face_yolov8s",
              "Better face detection on tough angles.", required=False,
              subpath="face/face_yolov8s.pt",
              url="https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8s.pt"),
    ModelSpec("person_yolov8s_seg", "ADetailer person_yolov8s-seg",
              "Body detection/segmentation — lets body-ADetailer use a person model "
              "(not the face model).", required=False,
              subpath="body/person_yolov8s-seg.pt",
              url="https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8s-seg.pt"),
    ModelSpec("buffalo_l", "InsightFace buffalo_l (face detect)",
              "Face detection for swaps + dataset crops.", required=True,
              url="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
              extract=True, extract_to=_BUFFALO_DIR,
              note="InsightFace would otherwise auto-fetch this on first use."),
    # --- body swap (SD-family only) ---
    ModelSpec("birefnet", "BiRefNet (body-swap segmentation)",
              "Body swap: person segmentation mask.", required=False,
              repo="ZhengPeng7/BiRefNet",
              note="Loaded from a local dir, not HF cache."),
    ModelSpec("openpose_annotator", "OpenPose annotator (body swap)",
              "Body swap: extract a pose control image from the base.",
              required=False, repo="lllyasviel/ControlNet"),
    ModelSpec("controlnet_openpose_sdxl", "ControlNet OpenPose (SDXL)",
              "Body swap: pose ControlNet for SDXL/Pony/Illustrious.",
              required=False, repo="thibaud/controlnet-openpose-sdxl-1.0"),
    ModelSpec("controlnet_openpose_sd15", "ControlNet OpenPose (SD1.5)",
              "Body swap: pose ControlNet for SD1.5.", required=False,
              repo="lllyasviel/control_v11p_sd15_openpose"),
    ModelSpec("ip_adapter", "IP-Adapter (body swap identity)",
              "Body swap: applies the source person's identity.", required=False,
              repo="h94/IP-Adapter"),
    # --- ControlNet (txt2img/img2img guidance, SDXL family) ---
    # Per-type standalone SDXL ControlNets used by the ControlNet accordion. Each is
    # downloadable on demand from the Prereqs panel; generation runs offline and
    # raises a clear "download it first" error if absent (never auto-pulls).
    ModelSpec("controlnet_canny_sdxl", "ControlNet Canny (SDXL)",
              "ControlNet guidance: edge/Canny conditioning for SDXL/Pony/Illustrious.",
              required=False, repo="diffusers/controlnet-canny-sdxl-1.0"),
    ModelSpec("controlnet_depth_sdxl", "ControlNet Depth (SDXL)",
              "ControlNet guidance: depth (Midas) conditioning for SDXL family.",
              required=False, repo="diffusers/controlnet-depth-sdxl-1.0"),
    ModelSpec("controlnet_union_sdxl", "ControlNet Union (SDXL, all-in-one)",
              "ControlNet guidance: xinsir all-in-one SDXL ControlNet (canny/depth/"
              "pose/lineart/tile in one model). Recommended default.",
              required=False, repo="xinsir/controlnet-union-sdxl-1.0"),
    ModelSpec("ip_adapter_faceid", "IP-Adapter FaceID (true identity)",
              "Colour Reference 'FaceID' variants: transfers the reference face's "
              "identity (InsightFace embeddings) instead of just its look.",
              required=False, repo="h94/IP-Adapter-FaceID"),
]

# Models the body-swap path needs present (or it errors, not auto-downloads).
# Now an IP-Adapter masked inpaint (no ControlNet/openpose): BiRefNet (body mask),
# buffalo_l (head detect), IP-Adapter (source appearance).
BODY_SWAP_KEYS = ["birefnet", "buffalo_l", "ip_adapter"]


def replace_person_keys(is_sdxl: bool) -> list[str]:
    """Models the 'Replace Person' path needs present (or it errors, never
    auto-downloads). It copies the target's POSE via an OpenPose ControlNet and
    the reference's IDENTITY via IP-Adapter FaceID (InsightFace embeddings):
      - birefnet         → person segmentation mask (the region to replace)
      - buffalo_l        → InsightFace face embeddings for FaceID identity
      - ip_adapter_faceid→ FaceID adapter weights (true identity transfer)
      - openpose_annotator → extract the pose control map from the target
      - the OpenPose ControlNet matching the chosen checkpoint's family.
    The ControlNet key depends on SDXL vs SD1.5, hence the parameter."""
    cn = "controlnet_openpose_sdxl" if is_sdxl else "controlnet_openpose_sd15"
    return ["birefnet", "buffalo_l", "ip_adapter_faceid", "openpose_annotator", cn]


def by_key(key: str) -> ModelSpec | None:
    return next((m for m in REGISTRY if m.key == key), None)


def status() -> list[dict]:
    return [{"key": m.key, "name": m.name, "present": m.is_present(),
             "required": m.required, "downloadable": m.downloadable,
             "path": m.display_path(), "purpose": m.purpose, "note": m.note}
            for m in REGISTRY]


def missing(keys) -> list[str]:
    """Names of registry models in ``keys`` that are not present."""
    return [m.name for m in REGISTRY if m.key in keys and not m.is_present()]


@contextlib.contextmanager
def no_auto_download():
    """Force HF/transformers offline so generation never silently pulls a model;
    a missing one raises instead. Download buttons run OUTSIDE this guard."""
    env = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    old = {k: os.environ.get(k) for k in env}
    for k in env:
        os.environ[k] = "1"
    try:
        yield
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def download(key: str, progress=None) -> str:
    spec = by_key(key)
    if spec is None:
        return f"[Error] Unknown model '{key}'."
    if spec.is_present():
        return f"[OK] {spec.name} already present."
    try:
        if spec.repo:
            return _download_repo(spec, progress)
        if spec.extract:
            return _download_zip(spec, progress)
        return _download_file(spec, progress)
    except Exception as e:
        logger.warning("download failed for %s", key, exc_info=True)
        return f"[Error] Download failed for {spec.name}: {e}"


def _gr_tqdm(progress, label):
    """A tqdm subclass that pumps its progress into a Gradio progress bar so HF
    downloads don't look frozen."""
    from tqdm.auto import tqdm as _tqdm

    class _GrTqdm(_tqdm):
        def update(self, n=1):
            r = super().update(n)
            try:
                if progress is not None and self.total:
                    progress(min(1.0, self.n / self.total),
                             desc=f"Downloading {label} — {self.desc or ''}".strip(" —"))
            except Exception:
                pass
            return r
    return _GrTqdm


def _download_repo(spec, progress) -> str:
    from huggingface_hub import snapshot_download
    if progress is not None:
        try:
            progress(0.0, desc=f"Fetching {spec.name} ({spec.repo})…")
        except Exception:
            pass
    kwargs = {"tqdm_class": _gr_tqdm(progress, spec.name)}
    local = spec.local_dir()
    if local:
        Path(local).mkdir(parents=True, exist_ok=True)
        kwargs["local_dir"] = local
    snapshot_download(spec.repo, **kwargs)
    return f"[Success] {spec.name} → {spec.display_path()}"


def _download_file(spec, progress) -> str:
    dst = paths.models_dir() / spec.subpath
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")

    def _hook(blocks, bsize, total):
        if progress is not None and total > 0:
            try:
                progress(min(1.0, blocks * bsize / total), desc=f"Downloading {spec.name}")
            except Exception:
                pass
    urllib.request.urlretrieve(spec.url, tmp, _hook)
    tmp.replace(dst)
    return f"[Success] {spec.name} → {dst}"


def _download_zip(spec, progress) -> str:
    dst_dir = Path(spec.extract_to)
    dst_dir.mkdir(parents=True, exist_ok=True)
    tmp = dst_dir.parent / (dst_dir.name + ".zip")

    def _hook(blocks, bsize, total):
        if progress is not None and total > 0:
            try:
                progress(min(1.0, blocks * bsize / total), desc=f"Downloading {spec.name}")
            except Exception:
                pass
    urllib.request.urlretrieve(spec.url, tmp, _hook)
    with zipfile.ZipFile(tmp) as z:
        # Guard against zip-slip: every member must resolve under dst_dir.
        base = dst_dir.resolve()
        for member in z.namelist():
            target = (dst_dir / member).resolve()
            if base != target and base not in target.parents:
                tmp.unlink(missing_ok=True)
                raise ValueError(f"Unsafe path in archive: {member!r}")
        z.extractall(dst_dir)
    tmp.unlink(missing_ok=True)
    return f"[Success] {spec.name} → {dst_dir}"


# --- on-demand scan (button-triggered) + link-from-disk ---------------------
# NOTHING here runs at startup. The Settings panel calls scan() only when the user
# presses "Scan for models", because finding a weight that's "on disk but not linked"
# means walking the (optional) search folder — which can be slow on a big/network
# drive. status()/is_present() above stay cheap (file-exists + HF-cache check).

def _alt_roots(search_dir):
    p = Path(search_dir).expanduser() if search_dir else None
    return [p] if (p and p.is_dir()) else []


def _walk_find(root, *, filename=None, dirname=None, cap=50000):
    """First file named ``filename`` (or first non-empty dir named ``dirname``) under
    ``root`` — capped traversal so a huge/network drive can't hang the scan."""
    seen = 0
    for dp, dirnames, filenames in os.walk(root):
        if dirname is not None and os.path.basename(dp) == dirname:
            try:
                with os.scandir(dp) as it:
                    if any(it):
                        return dp
            except OSError:
                pass
        if filename is not None and filename in filenames:
            return os.path.join(dp, filename)
        seen += len(filenames) + len(dirnames)
        if seen > cap:
            break
    return None


def _hf_cached(repo):
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(repo, local_files_only=True)
    except Exception:
        return None


def _find_elsewhere(spec: ModelSpec, roots):
    """Where this model exists on disk but is NOT in the dir our loader uses, so we
    can offer to symlink it in. Known spots (HF cache) + the user's search folder."""
    if spec.url and not spec.extract and spec.subpath:           # a single file
        name = Path(spec.subpath).name
        for r in roots:
            hit = _walk_find(r, filename=name)
            if hit:
                return hit
    elif spec.extract:                                           # a dir (buffalo_l)
        name = Path(spec.extract_to).name
        for r in roots:
            hit = _walk_find(r, dirname=name)
            if hit:
                return hit
    elif spec.repo and spec.local_dir():                         # BiRefNet: HF cache
        cached = _hf_cached(spec.repo)
        if cached:
            return cached
        for r in roots:
            hit = _walk_find(r, dirname=Path(spec.local_dir()).name)
            if hit:
                return hit
    return None


def scan(search_dir: str | None = None) -> list[dict]:
    """Per-model state for the Settings panel — RUN ON DEMAND (button), never at
    startup. state: 'linked' = in the dir our loader uses; 'elsewhere' = found on
    disk (HF cache / your search folder) but not linked → offer a symlink; 'missing'
    = not found → offer Download. ``found_at`` is where an 'elsewhere' model lives."""
    roots = _alt_roots(search_dir)
    out = []
    for m in REGISTRY:
        present = m.is_present()
        found_at = None if present else _find_elsewhere(m, roots)
        out.append({
            "key": m.key, "name": m.name, "required": m.required,
            "downloadable": m.downloadable,
            "state": "linked" if present else ("elsewhere" if found_at else "missing"),
            "found_at": found_at, "path": m.display_path(),
        })
    return out


def link_found(key: str, found_at: str | None = None, progress=None) -> str:
    """Symlink (copy fallback) a model found elsewhere on disk into the dir our loader
    uses. For an HF-repo model (BiRefNet) this materializes the cached snapshot into
    the local dir via the normal repo download (offline-fast when already cached)."""
    spec = by_key(key)
    if spec is None:
        return f"[Error] Unknown model '{key}'."
    if spec.is_present():
        return f"[OK] {spec.name} is already in place."
    try:
        if spec.repo and spec.local_dir():
            return _download_repo(spec, progress)   # snapshot from cache → local dir
        if not found_at or not Path(found_at).exists():
            return f"[Error] {spec.name}: nothing found on disk to link."
        dst = Path(spec.extract_to) if spec.extract else (paths.models_dir() / spec.subpath)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            return f"[OK] {spec.name} is already in place."
        src = Path(found_at).resolve()
        try:
            dst.symlink_to(src)
            return f"[Success] Linked {spec.name} → {dst}"
        except OSError:
            import shutil
            (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)
            return f"[Success] Copied {spec.name} → {dst}"
    except Exception as e:
        logger.warning("link_found failed for %s", key, exc_info=True)
        return f"[Error] Could not link {spec.name}: {e}"
