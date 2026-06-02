"""Image interrogation → prompt text. Family-routed:

  * booru families (Pony / Illustrious / anime) → WD14 danbooru-style TAGS
    (SmilingWolf/wd-swinv2-tagger-v3, ONNX).
  * everything else (photoreal SDXL, Flux, Z-Image, Qwen) → BLIP natural-language
    CAPTION (Salesforce/blip-image-captioning-large).

Ported from SupremeDiffusionQt's interrogate worker, reduced to plain functions
with an optional Gradio progress callback. Models download on first use (the
button press is the explicit user action — same as the Qwen prompt enhancer);
WD14 falls back to BLIP if its deps/files are unavailable.
"""
from __future__ import annotations

import csv
import logging

logger = logging.getLogger("imagesuite.interrogate")

# Model families whose prompts are danbooru tag lists rather than prose.
BOORU_FAMILIES = {"Pony", "Illustrious"}

_WD14_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
_BLIP_REPO = "Salesforce/blip-image-captioning-large"


def _say(progress, frac, msg):
    if progress is not None:
        try:
            progress(frac, desc=msg)
        except Exception:
            pass


def interrogate(image_path: str, family: str | None = None,
                threshold: float = 0.35, progress=None) -> str:
    """Route by model family and return prompt text (tags or a caption)."""
    if family in BOORU_FAMILIES:
        try:
            return wd14_tags(image_path, threshold, progress)
        except Exception:
            logger.warning("WD14 unavailable, falling back to BLIP", exc_info=True)
            # WD14 may have partially loaded a CUDA ORT session before raising;
            # reclaim any VRAM it grabbed before BLIP loads on the same device.
            _free_cuda()
            _say(progress, 0.3, "WD14 unavailable — using BLIP…")
    return blip_caption(image_path, progress)


def _free_cuda():
    """Best-effort GPU memory reclaim (used before falling back to BLIP)."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def wd14_tags(image_path: str, threshold: float = 0.35, progress=None) -> str:
    """Danbooru-style comma-separated tags via the WD14 SwinV2 v3 ONNX tagger."""
    import numpy as np
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from PIL import Image

    _say(progress, 0.15, "Loading WD14 tagger… (first run downloads ~400MB)")
    model_path = hf_hub_download(_WD14_REPO, "model.onnx")
    tags_path = hf_hub_download(_WD14_REPO, "selected_tags.csv")
    # CSV layout: tag_id, name, category, count. Rating tags carry category 9
    # (general=0, character=4). Read the names + categories defensively rather
    # than assuming a fixed header/column/rating-count layout.
    tags, cats = [], []
    with open(tags_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"WD14 tags file is empty: {tags_path}")
        # Locate the name/category columns from the header when possible.
        name_idx, cat_idx = 1, 2
        lower = [str(c).strip().lower() for c in header]
        if "name" in lower:
            name_idx = lower.index("name")
        if "category" in lower:
            cat_idx = lower.index("category")
        for row in reader:
            if len(row) <= name_idx:
                continue  # skip short/blank rows
            tags.append(row[name_idx])
            cat = None
            if len(row) > cat_idx:
                try:
                    cat = int(row[cat_idx])
                except (ValueError, TypeError):
                    cat = None
            cats.append(cat)
    if not tags:
        raise ValueError(f"WD14 tags file contained no usable rows: {tags_path}")

    session = None
    try:
        _say(progress, 0.5, "Tagging…")
        session = ort.InferenceSession(
            model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        # The model takes a 448² BGR image (height-major, batch of 1).
        img = Image.open(image_path).convert("RGB").resize((448, 448), Image.LANCZOS)
        arr = np.array(img).astype(np.float32)[:, :, ::-1].copy()
        arr = np.expand_dims(arr, 0)
        probs = session.run(None, {session.get_inputs()[0].name: arr})[0][0]
    finally:
        # Drop the CUDA ORT session promptly so its VRAM is reclaimed even on
        # error (mirrors blip_caption's cleanup).
        if session is not None:
            del session
        _free_cuda()

    # Skip rating tags (category 9) when categories are known; otherwise fall
    # back to the historical "first 4 rows are ratings" assumption.
    have_cats = any(c is not None for c in cats)
    scored = []
    for i, (t, p) in enumerate(zip(tags, probs)):
        if have_cats:
            if cats[i] == 9:
                continue
        elif i < 4:
            continue
        if p >= threshold:
            scored.append((t, p))
    scored.sort(key=lambda x: x[1], reverse=True)
    _say(progress, 1.0, "Done")
    return ", ".join(t.replace("_", " ") for t, _ in scored)


def blip_caption(image_path: str, progress=None) -> str:
    """One natural-language caption via BLIP-large."""
    import torch
    from PIL import Image
    from transformers import BlipForConditionalGeneration, BlipProcessor

    _say(progress, 0.15, "Loading BLIP… (first run downloads ~1.8GB)")
    processor = BlipProcessor.from_pretrained(_BLIP_REPO)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = BlipForConditionalGeneration.from_pretrained(
        _BLIP_REPO, torch_dtype=dtype).to(device)
    try:
        _say(progress, 0.5, "Captioning…")
        img = Image.open(image_path).convert("RGB")
        inputs = processor(img, return_tensors="pt").to(device, dtype)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=75)
        caption = processor.decode(out[0], skip_special_tokens=True)
    finally:
        del model, processor
        try:
            import torch as _t
            if _t.cuda.is_available():
                _t.cuda.empty_cache()
        except Exception:
            pass
    _say(progress, 1.0, "Done")
    return caption.strip()
