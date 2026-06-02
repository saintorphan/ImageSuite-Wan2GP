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
            _say(progress, 0.3, "WD14 unavailable — using BLIP…")
    return blip_caption(image_path, progress)


def wd14_tags(image_path: str, threshold: float = 0.35, progress=None) -> str:
    """Danbooru-style comma-separated tags via the WD14 SwinV2 v3 ONNX tagger."""
    import numpy as np
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from PIL import Image

    _say(progress, 0.15, "Loading WD14 tagger… (first run downloads ~400MB)")
    model_path = hf_hub_download(_WD14_REPO, "model.onnx")
    tags_path = hf_hub_download(_WD14_REPO, "selected_tags.csv")
    with open(tags_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # header
        tags = [row[1] for row in reader]

    _say(progress, 0.5, "Tagging…")
    session = ort.InferenceSession(
        model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    # The model takes a 448² BGR image (height-major, batch of 1).
    img = Image.open(image_path).convert("RGB").resize((448, 448), Image.LANCZOS)
    arr = np.array(img).astype(np.float32)[:, :, ::-1].copy()
    arr = np.expand_dims(arr, 0)
    probs = session.run(None, {session.get_inputs()[0].name: arr})[0][0]

    # First 4 entries are rating tags — skip them; keep tag>=threshold, sorted.
    scored = [(t, p) for t, p in list(zip(tags, probs))[4:] if p >= threshold]
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
