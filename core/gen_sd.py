"""SD/SDXL/Pony/Illustrious generation backend.

Wraps the bundled ``core.sd.SDImagePipeline`` (a self-contained diffusers backend
ported from SupremeDiffusion — Wan2GP has no SD-family support). The pipeline only
needs ``config.model_paths`` with sd_checkpoint_dir / sd_lora_dir / sd_vae_dir /
sd_refiner_dir, so the shim is tiny.

Generation is GPU-heavy; callers must hold the Wan2GP GPU lock (and ideally unload
the main model) around these calls — see plugin.acquire_gpu/release_gpu.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from . import models, paths

logger = logging.getLogger("imagesuite.gen_sd")

# The SD/SDXL pipeline is bundled in ``core/sd`` — no external checkout needed.


class _SDConfig:
    """Minimal stand-in for SupremeDiffusion's global_config — only model_paths
    is read by SDImagePipeline."""
    def __init__(self):
        self.model_paths = {
            "sd_checkpoint_dir": str(paths.sdxl_models_dir()),
            "sd_lora_dir": str(paths.sdxl_loras_dir()),
            "sd_vae_dir": "",
            "sd_refiner_dir": "",
        }


_pipeline = None


def _import_pipeline_cls():
    # Lazy import so torch/diffusers only load when the SD backend is first used.
    from .sd.sd_pipeline import SDImagePipeline
    return SDImagePipeline


def get_pipeline():
    """Lazily build the (cached) SDImagePipeline. Raises ImportError if the SD
    checkout isn't available."""
    global _pipeline
    if _pipeline is None:
        cls = _import_pipeline_cls()
        _pipeline = cls(_SDConfig())
    return _pipeline


def _free_torch():
    import gc
    import torch
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def sharpen(image_path, radius=2.0, percent=120, threshold=3, out_dir=None) -> str:
    """Crisp the WHOLE image without changing its resolution (PIL unsharp mask).
    No model, no GPU, instant. Boosts edge contrast — keep params modest so it
    doesn't ring/halo. (Don't run this on the post-downscale training export.)"""
    import time
    from PIL import Image, ImageFilter
    try:
        img = Image.open(image_path).convert("RGB").filter(
            ImageFilter.UnsharpMask(radius=float(radius), percent=int(percent),
                                    threshold=int(threshold)))
        out = Path(out_dir) if out_dir else (paths.cache_dir() / "poses")
        out.mkdir(parents=True, exist_ok=True)
        f = out / f"sharp_{int(time.time() * 1000)}.png"
        img.save(f)
        return str(f)
    except Exception:
        logger.warning("sharpen failed", exc_info=True)
        return image_path


def release_sd():
    """Unload the cached SD txt2img pipeline + free its VRAM (the SDXL checkpoint
    is ~6.5GB). Call before a different heavy model needs the GPU."""
    global _pipeline
    try:
        if _pipeline is not None:
            _pipeline.unload()
    except Exception:
        logger.debug("SD unload failed", exc_info=True)
    _free_torch()


def available() -> bool:
    """True if the SD/SDXL backend can run — i.e. diffusers is installed and the
    bundled pipeline imports. (The pipeline ships with the plugin; the gate is the
    optional ``pip install -r requirements.txt`` deps.)"""
    try:
        import importlib.util
        if importlib.util.find_spec("diffusers") is None:
            return False
        _import_pipeline_cls()
        return True
    except Exception:
        return False


def _apply_loras(pipe, loras):
    """Apply a [{"name","weight"}] LoRA list to the loaded SD pipe (no-op if empty)."""
    if not loras:
        return
    try:
        pipe.apply_loras(loras)
    except Exception:
        logger.warning("failed applying SD LoRAs %s", loras, exc_info=True)


def generate_txt2img(checkpoint_path, prompt, negative, width, height, steps, cfg,
                     seed, sampler="DPM++ 2M", scheduler="", batch_size=1,
                     clip_skip=1, out_dir=None, callback=None, loras=None) -> list[str]:
    """Generate image(s) with an SD-family checkpoint; returns saved file paths."""
    import random as _random
    pipe = get_pipeline()
    if seed is None or int(seed) < 0:
        seed = _random.randint(0, 2**31 - 1)
    with models.no_auto_download():  # never silently pull weights/configs
        pipe.load(checkpoint_path)
        _apply_loras(pipe, loras)
        images = pipe.generate_txt2img(
            prompt=prompt, negative_prompt=negative or "",
            width=int(width), height=int(height), steps=int(steps),
            cfg_scale=float(cfg), seed=int(seed), sampler=sampler, scheduler=scheduler,
            batch_size=int(batch_size), clip_skip=int(clip_skip), callback=callback,
        )
    out = Path(out_dir) if out_dir else (paths.cache_dir() / "sd_gen")
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, img in enumerate(images or []):
        f = out / f"sd_{int(seed)}_{i}.png"
        try:
            img.save(f)
            saved.append(str(f))
        except Exception:
            logger.warning("failed saving SD image %d", i, exc_info=True)
    return saved


def generate_img2img(checkpoint_path, image_path, prompt, negative, width, height,
                     steps, cfg, seed, denoise=0.6, sampler="DPM++ 2M", scheduler="Karras",
                     resize_mode=0, batch_size=1, clip_skip=1, out_dir=None,
                     loras=None) -> list[str]:
    """Reimagine an init image with an SD-family checkpoint (img2img). resize_mode
    fits the init image to width×height: 0 just-resize / 1 crop-and-resize /
    2 resize-and-fill (A1111/reForge codes)."""
    import random as _random
    pipe = get_pipeline()
    if seed is None or int(seed) < 0:
        seed = _random.randint(0, 2**31 - 1)
    with models.no_auto_download():
        pipe.load(checkpoint_path)
        _apply_loras(pipe, loras)
        images = pipe.generate_img2img(
            image=image_path, prompt=prompt, negative_prompt=negative or "",
            denoising_strength=float(denoise), width=int(width), height=int(height),
            steps=int(steps), cfg_scale=float(cfg), seed=int(seed), sampler=sampler,
            scheduler=scheduler, resize_mode=int(resize_mode),
            batch_size=int(batch_size), clip_skip=int(clip_skip))
    out = Path(out_dir) if out_dir else (paths.cache_dir() / "sd_gen")
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, img in enumerate(images or []):
        f = out / f"i2i_{int(seed)}_{i}.png"
        try:
            img.save(f); saved.append(str(f))
        except Exception:
            logger.warning("failed saving img2img %d", i, exc_info=True)
    return saved


def inpaint(checkpoint_path, image_path, mask_image, prompt, negative, denoise=0.75,
            steps=30, cfg=6.0, seed=-1, sampler="DPM++ 2M", scheduler="Karras",
            clip_skip=1, mask_blur=4, inpainting_fill=1, full_res=False, padding=32,
            batch_size=1, loras=None, out_dir=None, progress=None) -> list[str]:
    """Prompt-driven masked inpaint for manual touch-ups (no IP-Adapter). Returns the
    saved image paths (``batch_size`` of them).

    Inpaint-specific params mirror SupremeDiffusion's generate_inpaint: ``mask_blur``
    (px), ``inpainting_fill`` (0 fill / 1 original / 2 latent-noise / 3 latent-nothing),
    ``full_res`` (inpaint only the masked region at full res) and ``padding`` (px)."""
    import random as _random
    from PIL import Image
    pipe = get_pipeline()
    if seed is None or int(seed) < 0:
        seed = _random.randint(0, 2**31 - 1)
    img = Image.open(image_path).convert("RGB") if isinstance(image_path, str) else image_path
    w, h = img.size
    with models.no_auto_download():
        pipe.load(checkpoint_path)
        # LoRAs (independent to the Touch Up tab): generate_inpaint takes no loras arg,
        # so apply them to the inpaint pipe directly, then remove afterward.
        ip = None
        if loras:
            try:
                ip = pipe._get_inpaint_pipe()
                pipe._apply_loras_to_pipe(ip, loras)
            except Exception:
                logger.warning("failed applying inpaint LoRAs", exc_info=True)
                ip = None
        try:
            images = pipe.generate_inpaint(
                image=img, mask=mask_image, prompt=prompt or "", negative_prompt=negative or "",
                denoising_strength=float(denoise), width=int(w), height=int(h),
                steps=int(steps), cfg_scale=float(cfg), seed=int(seed), sampler=sampler,
                scheduler=scheduler, clip_skip=int(clip_skip), mask_blur=int(mask_blur),
                inpainting_fill=int(inpainting_fill), full_res=bool(full_res),
                padding=int(padding), batch_size=int(batch_size))
        finally:
            if ip is not None:
                try:
                    pipe._remove_loras_from_pipe(ip)
                except Exception:
                    logger.warning("failed removing inpaint LoRAs", exc_info=True)
    out = Path(out_dir) if out_dir else (paths.cache_dir() / "inpaint")
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, im in enumerate(images or []):
        f = out / f"inp_{int(seed)}_{i}.png"
        try:
            im.save(f)
            saved.append(str(f))
        except Exception:
            logger.warning("failed saving inpaint", exc_info=True)
    return saved


# ---- Shared IP-Adapter masked-inpaint identity transfer -------------------
# One primitive for both: body swap (ref = source body) and pose identity
# (ref = base). diffusers SDXL inpaint + IP-Adapter (plus, ViT-H). No ControlNet,
# no pose copy, no sequential CPU offload.
_inpaint_pipe = None
_inpaint_ckpt = None
_inpaint_with_ip = None

# Cooperative abort: the UI sets the flag; the diffusers step callback below trips
# the pipeline's _interrupt so the denoising loop bails out early.
_abort_flag = False


def request_abort():
    global _abort_flag
    _abort_flag = True


def clear_abort():
    global _abort_flag
    _abort_flag = False


def was_aborted() -> bool:
    return _abort_flag


def _abort_callback(pipe, step, timestep, kwargs):
    if _abort_flag:
        pipe._interrupt = True
    return kwargs


def release_inpaint():
    global _inpaint_pipe, _inpaint_ckpt, _inpaint_with_ip
    _inpaint_pipe = None
    _inpaint_ckpt = None
    _inpaint_with_ip = None
    _free_torch()


def _get_inpaint(checkpoint_path, with_ip=True):
    """Cached diffusers SDXL inpaint pipeline. With ``with_ip`` it also loads the
    IP-Adapter (for body swap); without it (ADetailer) it's a plain inpaint."""
    global _inpaint_pipe, _inpaint_ckpt, _inpaint_with_ip
    if (_inpaint_pipe is not None and _inpaint_ckpt == checkpoint_path
            and _inpaint_with_ip == with_ip):
        return _inpaint_pipe
    release_inpaint()
    import torch
    from diffusers import StableDiffusionXLInpaintPipeline
    with models.no_auto_download():
        pipe = StableDiffusionXLInpaintPipeline.from_single_file(
            checkpoint_path, torch_dtype=torch.float16)
        if with_ip:
            # The *_vit-h weights expect the ViT-H image encoder (1280-dim) at
            # models/image_encoder — NOT the bigG encoder (1664) in sdxl_models/.
            pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
                                 weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
                                 image_encoder_folder="models/image_encoder")
    # 12 GB-class cards can't hold the full SDXL inpaint + IP-Adapter (UNet + 2 text
    # encoders + image encoder + VAE) resident. Model CPU offload keeps only the
    # active submodule on GPU (per-module, not the slow per-layer sequential variant),
    # which fits comfortably. Don't call .to("cuda") alongside it — they conflict.
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
        if total <= 16 * 1024 ** 3:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
    except Exception:
        pipe.enable_model_cpu_offload()
    try:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
    except Exception:
        pass
    _inpaint_pipe, _inpaint_ckpt, _inpaint_with_ip = pipe, checkpoint_path, with_ip
    return pipe


def head_excluded_body_mask(base_path, models_root, hair_up=2.0,
                            exclude_hands=True) -> "object":
    """White = body to inpaint; black = whole head (face+hair) + hands + background, kept.
    Person via BiRefNet; head box = detected face dilated up/out, subtracted. Hands
    (which the reference's pose can't supply) are detected via MediaPipe Pose and
    carved out too, so e.g. hands-on-hips survive a differently-posed reference."""
    import os
    import numpy as np
    from PIL import Image
    from .sd.segmentation import (segment_foreground,
                                  release_segmentation_model)
    from . import deps
    deps.ensure({"kornia": "kornia"})  # BiRefNet modeling code
    mp = segment_foreground(base_path, models_root)
    release_segmentation_model()
    _free_torch()
    person = np.array(Image.open(mp).convert("L"))
    try:
        os.unlink(mp)
    except Exception:
        pass
    mask = (person > 127).astype("uint8") * 255
    try:
        import cv2
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l",
                           providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        faces = app.get(cv2.imread(base_path))
        if faces:
            f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            x1, y1, x2, y2 = (int(v) for v in f.bbox)
            bw, bh = x2 - x1, y2 - y1
            H, W = mask.shape
            hx1, hx2 = max(0, int(x1 - bw * 0.6)), min(W, int(x2 + bw * 0.6))
            hy1, hy2 = max(0, int(y1 - bh * hair_up)), min(H, int(y2 + bh * 0.25))
            mask[hy1:hy2, hx1:hx2] = 0  # exclude the whole head from the inpaint region
    except Exception:
        logger.warning("head detection failed; inpainting the full person", exc_info=True)
    if exclude_hands:
        try:
            import mediapipe as mp
            # mediapipe's legacy `solutions` API is incompatible with protobuf>=5
            # (and is absent from some wheels). When unavailable, skip hand
            # preservation rather than spam a traceback — body swap still runs.
            if not hasattr(mp, "solutions"):
                logger.info("mediapipe.solutions unavailable (protobuf %s); skipping "
                            "hand preservation — hands may be regenerated.",
                            __import__("google.protobuf", fromlist=["__version__"]).__version__)
                return Image.fromarray(mask, "L")
            import cv2
            bgr = cv2.imread(base_path)
            H, W = mask.shape
            with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2,
                                        min_detection_confidence=0.3) as pose:
                res = pose.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                # Per hand: wrist + pinky/index/thumb landmarks bound the hand.
                for wrist, fingers in ((15, (17, 19, 21)), (16, (18, 20, 22))):
                    pts = [(lm[i].x * W, lm[i].y * H) for i in (wrist, *fingers)
                           if lm[i].visibility > 0.3]
                    if len(pts) < 2:
                        continue
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
                    # Radius from hand spread, padded; floor relative to image size.
                    spread = max(max(xs) - min(xs), max(ys) - min(ys))
                    r = max(spread * 1.4, W * 0.06)
                    x1, x2 = max(0, int(cx - r)), min(W, int(cx + r))
                    y1, y2 = max(0, int(cy - r)), min(H, int(cy + r))
                    mask[y1:y2, x1:x2] = 0  # keep hands from the base
        except Exception:
            logger.warning("hand detection failed; hands may be regenerated",
                           exc_info=True)
    return Image.fromarray(mask, "L")


def _apply_loras_to_diffusers(pipe, loras) -> bool:
    """Apply [{'name','weight'}] LoRAs to a raw diffusers pipe, reusing the SD
    pipeline's loader (resolves files under sd_lora_dir). Returns True if applied.
    Used by the enhancement passes (body swap / colour reference / ADetailer), which
    run on their own diffusers inpaint pipe rather than the main txt2img pipe."""
    if not loras:
        return False
    try:
        get_pipeline()._apply_loras_to_pipe(pipe, loras)
        return True
    except Exception:
        logger.warning("failed applying LoRAs to enhancement pipe", exc_info=True)
        return False


def _remove_loras_from_diffusers(pipe, applied) -> None:
    if not applied:
        return
    try:
        get_pipeline()._remove_loras_from_pipe(pipe)
    except Exception:
        logger.debug("failed removing enhancement LoRAs", exc_info=True)


def ip_adapter_inpaint(checkpoint_path, target_path, reference_path, mask_image,
                       prompt, negative, denoise=0.7, ip_scale=0.8, steps=30,
                       cfg=6.0, seed=-1, out_dir=None, loras=None, progress=None) -> str | None:
    """Apply a reference identity onto the masked region of a target via IP-Adapter
    inpaint. Optional SDXL ``loras`` are applied for the pass. Returns the saved path."""
    import random as _random
    import time
    import torch
    from PIL import Image
    if seed is None or int(seed) < 0:
        seed = _random.randint(0, 2**31 - 1)
    pipe = _get_inpaint(checkpoint_path, with_ip=True)
    pipe.set_ip_adapter_scale(float(ip_scale))
    target = Image.open(target_path).convert("RGB")
    ref = Image.open(reference_path).convert("RGB")
    w, h = target.size
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    _remove_loras_from_diffusers(pipe, True)   # clear any stale from a prior pass
    _la = _apply_loras_to_diffusers(pipe, loras)
    try:
        with models.no_auto_download():
            img = pipe(prompt=prompt or "", negative_prompt=negative or "",
                       image=target, mask_image=mask_image, ip_adapter_image=ref,
                       strength=float(denoise), num_inference_steps=int(steps),
                       guidance_scale=float(cfg), width=w, height=h, generator=gen,
                       callback_on_step_end=_abort_callback).images[0]
    finally:
        _remove_loras_from_diffusers(pipe, _la)
    if _abort_flag:  # interrupted mid-run — discard the partial result
        return None
    out = Path(out_dir) if out_dir else (paths.cache_dir() / "swap")
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"ipinpaint_{int(seed)}_{int(time.time())}.png"
    try:
        img.save(f)
        return str(f)
    except Exception:
        logger.warning("failed saving ip-adapter inpaint", exc_info=True)
        return None


def _detect_face_boxes(image_path, threshold=0.4):
    """Return [(x1,y1,x2,y2), ...] face boxes via InsightFace (buffalo_l)."""
    try:
        import cv2
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l",
                           providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        boxes = []
        for f in app.get(cv2.imread(image_path)):
            if float(getattr(f, "det_score", 1.0)) < threshold:
                continue
            x1, y1, x2, y2 = (int(v) for v in f.bbox)
            boxes.append((x1, y1, x2, y2))
        return boxes
    except Exception:
        logger.warning("ADetailer face detection failed", exc_info=True)
        return []


def _detect_person_regions(image_path, threshold=0.3):
    """Largest person via the gated person_yolov8s-seg model → [((x1,y1,x2,y2), L-mask)].
    Returns [] if the model isn't downloaded (body-ADetailer then passes through)."""
    try:
        import numpy as np
        import torch
        from PIL import Image, ImageDraw
        from ultralytics import YOLO
        mp = paths.models_dir() / "body" / "person_yolov8s-seg.pt"
        if not mp.is_file():
            logger.info("person_yolov8s-seg.pt not downloaded — body ADetailer skipped.")
            return []
        src = Image.open(image_path).convert("RGB")
        W, H = src.size
        model = YOLO(str(mp))
        dev = 0 if torch.cuda.is_available() else "cpu"
        res = model.predict(np.array(src), conf=float(threshold), verbose=False, device=dev)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []
        msk = res.masks.data.cpu().numpy() if getattr(res, "masks", None) is not None else None
        out = []
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = (int(v) for v in b)
            if msk is not None and i < len(msk):
                m = Image.fromarray((msk[i] * 255).astype("uint8"), "L").resize((W, H))
            else:
                m = Image.new("L", (W, H), 0)
                ImageDraw.Draw(m).rectangle((x1, y1, x2, y2), fill=255)
            out.append(((x1, y1, x2, y2), m))
        out.sort(key=lambda r: (r[0][2] - r[0][0]) * (r[0][3] - r[0][1]), reverse=True)
        return out[:1]  # largest person only
    except Exception:
        logger.warning("ADetailer person detection failed", exc_info=True)
        return []


def run_adetailer(checkpoint_path, image_path, prompt, negative, sampler=None,
                  scheduler=None, steps=24, cfg=7.0, clip_skip=1, denoise=0.4,
                  pad=0.35, detector="face", out_dir=None, loras=None) -> str | None:
    """Detect a region (``detector`` = "face" via InsightFace, or "person" via the
    gated person-seg YOLO) and re-inpaint it at higher detail. Self-contained:
    diffusers SDXL inpaint, crop → inpaint → feathered paste at full res. Returns the
    refined path (or the original on no-op)."""
    import time
    from PIL import Image, ImageDraw, ImageFilter
    src = Image.open(image_path).convert("RGB")
    W, H = src.size
    if detector == "person":
        regions = _detect_person_regions(image_path)  # [((box), full-image mask)]
        noun = "person"
    else:
        regions = [(b, None) for b in _detect_face_boxes(image_path)]
        noun = "face"
    if not regions:
        logger.info("ADetailer(%s): nothing detected; passing through.", noun)
        return image_path
    release_sd()  # free the txt2img checkpoint
    pipe = _get_inpaint(checkpoint_path, with_ip=False)
    _remove_loras_from_diffusers(pipe, True)   # clear any stale from a prior pass
    _la = _apply_loras_to_diffusers(pipe, loras)
    result = src.copy()
    for (x1, y1, x2, y2), full_mask in regions:
        bw, bh = x2 - x1, y2 - y1
        px, py = bw * pad, bh * pad
        cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
        cx2, cy2 = min(W, int(x2 + px)), min(H, int(y2 + py))
        crop = result.crop((cx1, cy1, cx2, cy2))
        cw, ch = crop.size
        if cw < 16 or ch < 16:
            continue
        if full_mask is not None:  # person: use the seg mask cropped to this region
            m = full_mask.crop((cx1, cy1, cx2, cy2))
        else:  # face: white rect over the (unpadded) face within the crop
            m = Image.new("L", (cw, ch), 0)
            ImageDraw.Draw(m).rectangle(
                (int(x1 - cx1), int(y1 - cy1), int(x2 - cx1), int(y2 - cy1)), fill=255)
        m = m.filter(ImageFilter.GaussianBlur(radius=max(4, cw // 25)))
        # Inpaint at ~1024 on the long side, snapped to multiples of 8.
        scale = 1024.0 / max(cw, ch)
        tw = max(8, (int(round(cw * scale)) // 8) * 8)
        th = max(8, (int(round(ch * scale)) // 8) * 8)
        with models.no_auto_download():
            out_img = pipe(prompt=prompt or "", negative_prompt=negative or "",
                           image=crop.resize((tw, th), Image.LANCZOS),
                           mask_image=m.resize((tw, th), Image.LANCZOS),
                           strength=float(denoise), num_inference_steps=int(steps),
                           guidance_scale=float(cfg), width=tw, height=th).images[0]
        result.paste(out_img.resize((cw, ch), Image.LANCZOS), (cx1, cy1), m)
    _remove_loras_from_diffusers(pipe, _la)
    out = Path(out_dir) if out_dir else (paths.cache_dir() / "swap")
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"adetail_{int(time.time())}.png"
    try:
        result.save(f)
        return str(f)
    except Exception:
        logger.warning("failed saving ADetailer result", exc_info=True)
        return image_path


def body_swap(checkpoint_path, base_path, source_person_path, prompt, negative,
              cn_strength=0.7, ip_scale=0.8, denoise=0.75, cfg=7.0, steps=30,
              seed=-1, sampler="DPM++ 3M SDE", scheduler="Karras",
              adetailer=True, adet_prompt="", adet_neg="", loras=None,
              progress=None) -> str | None:
    """Transfer the source person's skin tone / body texture onto the base's body
    (whole head excluded → face + hair preserved). No pose copy, no ControlNet —
    an IP-Adapter masked inpaint. SD-family checkpoints only. With ``adetailer``,
    a final face-detail pass runs (using ``adet_prompt``/``adet_neg`` if given)."""
    def _say(frac, msg):
        if progress is not None:
            try:
                progress(frac, desc=msg)
            except Exception:
                pass
    clear_abort()
    _import_pipeline_cls()
    models_root = str(paths.models_dir())
    release_sd()  # free the txt2img checkpoint
    _say(0.2, "Segmenting body (head excluded)…")
    mask = head_excluded_body_mask(base_path, models_root)
    if was_aborted():
        return None
    _say(0.55, "Applying source skin/texture (IP-Adapter inpaint)…")
    res = ip_adapter_inpaint(checkpoint_path, base_path, source_person_path, mask,
                             prompt, negative, denoise=float(denoise),
                             ip_scale=float(ip_scale), steps=int(steps),
                             cfg=float(cfg), seed=int(seed), loras=loras,
                             progress=progress)
    if adetailer and res and not was_aborted():
        _say(0.9, "ADetailer body refine…")
        try:  # body swap → re-detail the BODY with the person model
            return run_adetailer(checkpoint_path, res,
                                 adet_prompt or prompt, adet_neg or negative,
                                 sampler, scheduler, steps=int(steps), cfg=float(cfg),
                                 detector="person", loras=loras)
        except Exception:
            logger.warning("ADetailer pass failed; returning un-refined body swap",
                           exc_info=True)
    return res
