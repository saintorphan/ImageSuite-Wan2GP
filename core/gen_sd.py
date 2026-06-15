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
            "sd_vae_dir": str(paths.sdxl_vae_dir()),
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
    # Refresh the persisted memory policy each access (cheap; reads .orphansuite.json)
    # so a Settings change applies to the next load() without a restart. Defaults to
    # "balanced" == current behaviour, so this never regresses an existing setup.
    try:
        _pipeline._mem_policy = paths.get_sd_mem_policy()
    except Exception:
        logger.debug("could not read sd_mem_policy", exc_info=True)
    return _pipeline


def _free_torch():
    import gc
    import torch
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


_face_app = None  # cached InsightFace buffalo_l detector (CUDA/ONNX) — see _get_face_app


def _get_face_app():
    """Cached InsightFace buffalo_l detector, built once and reused. Body-swap
    masking (head_excluded_body_mask) and face ADetailer (_detect_face_boxes) used to
    each spin up a fresh FaceAnalysis on the CUDAExecutionProvider per call — a CUDA
    ONNX session that was a function-local, never released, and invisible to the VRAM
    release path. Sharing one instance (freed via release_face_analysis, wired into
    release_all) keeps it reclaimable at the GPU handoff."""
    global _face_app
    if _face_app is None:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l",
                           providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        _face_app = app
    return _face_app


def release_face_analysis():
    """Drop the cached InsightFace session so its CUDA/ONNX arena is reclaimed.
    Dropping the only Python reference is the load-bearing step — empty_cache() does
    NOT free onnxruntime's CUDA arena."""
    global _face_app
    _face_app = None
    _free_torch()


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


def release_sd(force: bool = True):
    """Unload the cached SD txt2img pipeline + free its VRAM (the SDXL checkpoint
    is ~6.5GB). Call before a different heavy model needs the GPU.

    ``force`` defaults True so every existing caller (GPU handoff, stacking the
    inpaint/face-swap pipe) frees exactly as before — no regression. Pass
    ``force=False`` to let the "keep resident" memory policy keep the pipe loaded
    for back-to-back SD gens, but only when no other model is about to claim the
    GPU."""
    global _pipeline
    try:
        if _pipeline is not None:
            _pipeline.unload(force=force)
    except Exception:
        logger.debug("SD unload failed", exc_info=True)
    _free_torch()


def release_controlnet():
    """Free the cached ControlNet pipeline on the main SD pipeline (best-effort).
    Operates only on an already-built pipeline so it never forces a load."""
    global _pipeline
    try:
        if _pipeline is not None:
            _pipeline.unload_controlnet()
    except Exception:
        logger.debug("ControlNet unload failed", exc_info=True)
    _free_torch()


def release_body_double():
    """Free the cached body-double pipeline on the main SD pipeline (best-effort).
    Operates only on an already-built pipeline so it never forces a load."""
    global _pipeline
    try:
        if _pipeline is not None:
            _pipeline.unload_body_double()
    except Exception:
        logger.debug("body-double unload failed", exc_info=True)
    _free_torch()


# ---- TAESD live latent preview --------------------------------------------
# Tiny autoencoder (madebyollin/taesd[xl], ~10MB) used to decode in-progress
# latents to a small preview image during sampling. Entirely OPTIONAL and
# default-OFF: if the weights aren't downloaded, or anything throws, the decode
# returns None and generation is unaffected. Cached per model-type and freed in
# release_all so it never leaks into video generation.
_taesd = {}  # {"sdxl"|"sd15": AutoencoderTiny}


def _get_taesd(model_type: str):
    """Lazily build & cache the tiny VAE for previewing latents. Returns None on
    any failure (missing weights, no diffusers, no CUDA) so the caller no-ops."""
    mt = "sd15" if str(model_type) == "sd15" else "sdxl"
    pipe = _taesd.get(mt)
    if pipe is not None:
        return pipe
    try:
        import torch
        from diffusers import AutoencoderTiny
        repo = "madebyollin/taesd" if mt == "sd15" else "madebyollin/taesdxl"
        # TAESD is tiny (~10MB) and the live preview is explicitly opt-in (default
        # OFF), so we DON'T wrap this in models.no_auto_download() — let it fetch on
        # first use. If it's unreachable (no network / HF down) the except below
        # swallows it and the preview is simply skipped; generation is unaffected.
        tae = AutoencoderTiny.from_pretrained(repo, torch_dtype=torch.float16)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tae = tae.to(dev)
        _taesd[mt] = tae
        return tae
    except Exception:
        logger.info("TAESD preview unavailable (%s couldn't load/download) — live "
                    "preview disabled for this run.", mt, exc_info=True)
        # Leave it absent (don't cache None) so a later attempt can retry the fetch.
        return None


def taesd_preview(latents, model_type="sdxl", max_side=384):
    """Decode in-progress diffusion ``latents`` to a small PIL preview via TAESD.

    Fully best-effort: returns None on ANY failure so a live-preview attempt can
    never break generation. ``latents`` is the tensor diffusers hands the step
    callback in ``callback_kwargs['latents']`` (one or more images in a batch — we
    preview only the first). ``max_side`` caps the longest edge of the returned
    image (TAESD output is already 1:8 of native, so this is usually a downscale)."""
    if latents is None:
        return None
    try:
        import torch
        from PIL import Image as _Img
        tae = _get_taesd(model_type)
        if tae is None:
            return None
        with torch.inference_mode():
            x = latents
            if x.dim() == 3:
                x = x.unsqueeze(0)
            x = x[:1].to(device=tae.device, dtype=tae.dtype)
            # AutoencoderTiny.decode handles its own latent (un)scaling — feed the
            # raw denoising latents straight in. Output is in [-1, 1].
            img = tae.decode(x).sample
        img = (img.clamp(-1, 1) + 1) / 2          # → [0, 1]
        img = (img[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype("uint8")
        pil = _Img.fromarray(img, "RGB")
        if max_side and max(pil.size) > int(max_side):
            s = int(max_side) / max(pil.size)
            pil = pil.resize((max(1, int(pil.width * s)), max(1, int(pil.height * s))),
                             _Img.BILINEAR)
        return pil
    except Exception:
        logger.debug("TAESD preview decode failed", exc_info=True)
        return None


def release_taesd():
    """Drop the cached tiny preview VAE(s) and free their (~10MB each) VRAM."""
    global _taesd
    _taesd = {}
    _free_torch()


def release_all():
    """Free *all* plugin-held GPU consumers: the main SD txt2img pipeline, the
    standalone inpaint/IP-Adapter pipe, the ControlNet + body-double pipes, the
    TAESD live-preview VAE, and (best-effort) the bundled BiRefNet segmentation
    model. Registered as the GPU release_vram_callback so handing the lock back to
    Wan2GP doesn't leak any of these into video generation."""
    release_sd()
    release_inpaint()
    release_controlnet()
    release_body_double()
    release_taesd()
    try:
        from .sd.segmentation import release_segmentation_model
        release_segmentation_model()
    except Exception:
        logger.debug("segmentation release failed", exc_info=True)
    release_face_analysis()


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


def _vae_name() -> str:
    """The custom-VAE name to hand SDImagePipeline.load(). The persisted setting is
    a file stem ('' = none); map '' to "Automatic" (the checkpoint's own VAE — current
    behaviour, no regression)."""
    return paths.get_sd_vae() or "Automatic"


def _apply_loras(pipe, loras):
    """Apply a [{"name","weight"}] LoRA list to the loaded SD pipe (no-op if empty)."""
    if not loras:
        return
    try:
        pipe.apply_loras(loras)
    except Exception:
        logger.warning("failed applying SD LoRAs %s", loras, exc_info=True)


def _apply_turbo(turbo, sampler, scheduler, steps, cfg, loras):
    """Resolve a "Turbo" distilled preset and, if a matching distill LoRA exists,
    override sampler/scheduler/steps/CFG and prepend the distill LoRA to ``loras``.

    Returns ``(sampler, scheduler, steps, cfg, loras)`` — unchanged when Turbo is
    Off or no matching distill LoRA is found in the SDXL LoRA dir (a warning is
    logged in the latter case so generation never breaks if the LoRA is absent).
    The distill LoRA is fused at weight 1.0 through the normal apply/fuse path; at
    the forced CFG ~1 the negative prompt has no effect (CFG off)."""
    from . import presets
    if not turbo or turbo == presets.TURBO_OFF:
        return sampler, scheduler, steps, cfg, loras
    info = presets.resolve_turbo(turbo, str(paths.sdxl_loras_dir()))
    if not info.get("enabled"):
        if info.get("warn"):
            logger.warning("%s", info["warn"])
        return sampler, scheduler, steps, cfg, loras
    # Prepend the distill LoRA (so user LoRAs still apply on top of it).
    turbo_loras = [{"name": info["lora"], "weight": 1.0}] + list(loras or [])
    logger.info("Turbo '%s': distill LoRA=%s sampler=%s scheduler=%s steps=%d cfg=%.2f",
                info["name"], info["lora"], info["sampler"], info["scheduler"],
                info["steps"], info["cfg"])
    return (info["sampler"], info["scheduler"], info["steps"], info["cfg"],
            turbo_loras)


def _pnginfo(checkpoint_path, prompt, negative, seed, steps, cfg, sampler,
             scheduler, clip_skip, img):
    """Build a PIL PngInfo carrying an A1111 ``parameters`` chunk for an SD result,
    or None on any failure (so a save never breaks if metadata can't be built).

    Embeds only what the SD backend knows — prompt/neg/seed/steps/cfg/sampler/
    scheduler/model (checkpoint stem)/size/clip_skip. Native (Flux/Z-Image/Qwen)
    saves go through the host and never reach here."""
    try:
        from PIL import PngImagePlugin
        from . import metadata
        w, h = (img.size if img is not None else (None, None))
        text = metadata.format_a1111({
            "prompt": prompt, "negative": negative, "seed": seed, "steps": steps,
            "cfg": cfg, "sampler": sampler, "scheduler": scheduler,
            "model": Path(str(checkpoint_path)).stem if checkpoint_path else None,
            "width": w, "height": h, "clip_skip": clip_skip,
        })
        if not text:
            return None
        info = PngImagePlugin.PngInfo()
        info.add_text("parameters", text)
        return info
    except Exception:
        logger.debug("failed building PNG metadata", exc_info=True)
        return None


def _hires_fix(pipe, images, prompt, negative, width, height, hr_scale, hr_denoise,
               steps, cfg, seed, sampler, scheduler, clip_skip, callback):
    """Latent hi-res fix: upscale each base txt2img image by ``hr_scale`` and run a
    low-denoise img2img second pass on the SAME loaded pipe (its img2img variant
    shares the UNet/VAE/text-encoders — no second model, minimal extra VRAM).

    Returns the refined PIL list. Best-effort: any failure logs and returns the
    untouched base images so a gen never breaks. LoRAs are assumed already applied
    by the caller (the variant pipe shares them). Called only when hr_scale > 1.0."""
    if not images:
        return images
    try:
        from PIL import Image as _Img
    except Exception:
        return images
    hw = max(8, (int(round(int(width) * float(hr_scale))) // 8) * 8)
    hh = max(8, (int(round(int(height) * float(hr_scale))) // 8) * 8)
    # img2img's effective step count is steps*denoise; bump nominal steps so the
    # short hi-res pass still does enough work (clamped, mirrors A1111 behaviour).
    hr_steps = max(int(steps), int(round(int(steps) * float(hr_denoise))) + 1)
    refined = []
    for img in images:
        try:
            up = img.resize((hw, hh), _Img.LANCZOS)
            out = pipe.generate_img2img(
                image=up, prompt=prompt, negative_prompt=negative or "",
                denoising_strength=float(hr_denoise), width=hw, height=hh,
                steps=hr_steps, cfg_scale=float(cfg), seed=int(seed),
                sampler=sampler, scheduler=scheduler or "Karras", resize_mode=0,
                batch_size=1, clip_skip=int(clip_skip), callback=callback)
            refined.append(out[0] if out else img)
        except Exception:
            logger.warning("hi-res fix pass failed; using base image", exc_info=True)
            refined.append(img)
    return refined


def generate_txt2img(checkpoint_path, prompt, negative, width, height, steps, cfg,
                     seed, sampler="DPM++ 2M", scheduler="", batch_size=1,
                     clip_skip=1, out_dir=None, callback=None, loras=None,
                     turbo="Off", hr_scale=1.0, hr_denoise=0.4,
                     refiner_checkpoint="", refiner_switch_at=0.8,
                     refiner_steps=10, refiner_cfg=7.0) -> list[str]:
    """Generate image(s) with an SD-family checkpoint; returns saved file paths.

    ``turbo`` ("Off"/LCM/Hyper-SD/Lightning) applies a distilled few-step preset:
    it fuses the matched distill LoRA and forces sampler/scheduler/steps/CFG. Off
    (default) is unchanged; a missing distill LoRA warns and falls back to normal.

    ``hr_scale`` (1.0 = OFF, default) enables a latent hi-res fix: after the base
    gen each image is upscaled by hr_scale and run through a low-denoise img2img
    second pass (``hr_denoise``) on the same loaded pipe — sharper, larger output
    with no second model. hr_scale==1.0 leaves behaviour byte-for-byte unchanged.

    ``refiner_checkpoint`` ("" = OFF, default) enables the SDXL ensemble-of-experts
    refiner: the base pass stops at ``refiner_switch_at`` (fraction) and hands its
    latents to a second SDXL checkpoint that finishes the schedule with
    ``refiner_steps`` / ``refiner_cfg``. SDXL-family only and ignored when the
    refiner checkpoint is empty or can't be resolved, so behaviour never regresses."""
    import random as _random
    pipe = get_pipeline()
    sampler, scheduler, steps, cfg, loras = _apply_turbo(
        turbo, sampler, scheduler, steps, cfg, loras)
    if seed is None or int(seed) < 0:
        seed = _random.randint(0, 2**31 - 1)
    hires = float(hr_scale or 1.0) > 1.0
    with models.no_auto_download():  # never silently pull weights/configs
        pipe.load(checkpoint_path, vae_name=_vae_name())
        # Clear any LoRAs left fused on the cached pipe from a prior gen (the empty
        # list must clear too, so this can't live inside _apply_loras), apply the
        # requested set, and remove them again afterward so they never contaminate
        # the next generation on the same checkpoint.
        pipe.remove_loras()
        _apply_loras(pipe, loras)
        try:
            images = pipe.generate_txt2img(
                prompt=prompt, negative_prompt=negative or "",
                width=int(width), height=int(height), steps=int(steps),
                cfg_scale=float(cfg), seed=int(seed), sampler=sampler, scheduler=scheduler,
                batch_size=int(batch_size), clip_skip=int(clip_skip), callback=callback,
                # SDXL refiner second pass — OFF when refiner_checkpoint is "" (the
                # pipeline also no-ops it on non-SDXL or an unresolved path).
                refiner_checkpoint=refiner_checkpoint or "",
                refiner_switch_at=float(refiner_switch_at),
                refiner_steps=int(refiner_steps),
                refiner_cfg_scale=float(refiner_cfg),
            )
            # Hi-res fix second pass reuses the loaded pipe's img2img variant (same
            # UNet/VAE — no extra model). LoRAs are still applied here (shared with
            # the variant pipe), so it runs inside this try before remove_loras.
            if hires:
                images = _hires_fix(
                    pipe, images, prompt, negative, width, height, hr_scale,
                    hr_denoise, steps, cfg, seed, sampler, scheduler, clip_skip,
                    callback)
        finally:
            pipe.remove_loras()
    out = Path(out_dir) if out_dir else (paths.cache_dir() / "sd_gen")
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, img in enumerate(images or []):
        f = out / f"sd_{int(seed)}_{i}.png"
        try:
            info = _pnginfo(checkpoint_path, prompt, negative, seed, steps, cfg,
                            sampler, scheduler, clip_skip, img)
            img.save(f, pnginfo=info)
            saved.append(str(f))
        except Exception:
            logger.warning("failed saving SD image %d", i, exc_info=True)
    return saved


def generate_img2img(checkpoint_path, image_path, prompt, negative, width, height,
                     steps, cfg, seed, denoise=0.6, sampler="DPM++ 2M", scheduler="Karras",
                     resize_mode=0, batch_size=1, clip_skip=1, out_dir=None,
                     callback=None, loras=None, turbo="Off") -> list[str]:
    """Reimagine an init image with an SD-family checkpoint (img2img). resize_mode
    fits the init image to width×height: 0 just-resize / 1 crop-and-resize /
    2 resize-and-fill (A1111/reForge codes).

    ``turbo`` applies a distilled few-step preset (see generate_txt2img)."""
    import random as _random
    pipe = get_pipeline()
    sampler, scheduler, steps, cfg, loras = _apply_turbo(
        turbo, sampler, scheduler, steps, cfg, loras)
    if seed is None or int(seed) < 0:
        seed = _random.randint(0, 2**31 - 1)
    with models.no_auto_download():
        pipe.load(checkpoint_path, vae_name=_vae_name())
        # See generate_txt2img: clear stale LoRAs, apply, and remove afterward.
        pipe.remove_loras()
        _apply_loras(pipe, loras)
        try:
            images = pipe.generate_img2img(
                image=image_path, prompt=prompt, negative_prompt=negative or "",
                denoising_strength=float(denoise), width=int(width), height=int(height),
                steps=int(steps), cfg_scale=float(cfg), seed=int(seed), sampler=sampler,
                scheduler=scheduler, resize_mode=int(resize_mode),
                batch_size=int(batch_size), clip_skip=int(clip_skip), callback=callback)
        finally:
            pipe.remove_loras()
    out = Path(out_dir) if out_dir else (paths.cache_dir() / "sd_gen")
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, img in enumerate(images or []):
        f = out / f"i2i_{int(seed)}_{i}.png"
        try:
            info = _pnginfo(checkpoint_path, prompt, negative, seed, steps, cfg,
                            sampler, scheduler, clip_skip, img)
            img.save(f, pnginfo=info); saved.append(str(f))
        except Exception:
            logger.warning("failed saving img2img %d", i, exc_info=True)
    return saved


def inpaint(checkpoint_path, image_path, mask_image, prompt, negative, denoise=0.75,
            steps=30, cfg=6.0, seed=-1, sampler="DPM++ 2M", scheduler="Karras",
            clip_skip=1, mask_blur=4, inpainting_fill=1, full_res=False, padding=32,
            seamless=False, batch_size=1, loras=None, out_dir=None, callback=None,
            progress=None, turbo="Off") -> list[str]:
    """Prompt-driven masked inpaint for manual touch-ups (no IP-Adapter). Returns the
    saved image paths (``batch_size`` of them).

    Inpaint-specific params mirror SupremeDiffusion's generate_inpaint: ``mask_blur``
    (px), ``inpainting_fill`` (0 fill / 1 original / 2 latent-noise / 3 latent-nothing),
    ``full_res`` (inpaint only the masked region at full res) and ``padding`` (px).
    ``seamless`` opts the paste-back into Laplacian pyramid blending (default off uses
    the plain feathered composite). ``turbo`` applies a distilled few-step preset
    (see generate_txt2img); its distill LoRA is fused with any user LoRAs for the pass."""
    import random as _random
    from PIL import Image
    pipe = get_pipeline()
    sampler, scheduler, steps, cfg, loras = _apply_turbo(
        turbo, sampler, scheduler, steps, cfg, loras)
    if seed is None or int(seed) < 0:
        seed = _random.randint(0, 2**31 - 1)
    img = Image.open(image_path).convert("RGB") if isinstance(image_path, str) else image_path
    w, h = img.size
    with models.no_auto_download():
        pipe.load(checkpoint_path, vae_name=_vae_name())
        # Clear any LoRAs left fused from a prior gen before applying this set —
        # same as generate_txt2img/generate_img2img. The SDXL inpaint/img2img
        # variant pipes share the main pipe's UNet/VAE/text-encoders, so LoRA
        # load/unload is GLOBAL across all three; clearing via the main pipe
        # clears the inpaint pipe too.
        pipe.remove_loras()
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
        # Opt the paste-back composite into Laplacian pyramid blending for this call
        # only — generate_inpaint / _inpaint_full_res read self._use_laplacian_blend.
        # Restore the prior value afterward so the flag never leaks across calls.
        prev_lap_blend = getattr(pipe, "_use_laplacian_blend", False)
        pipe._use_laplacian_blend = bool(seamless)
        try:
            images = pipe.generate_inpaint(
                image=img, mask=mask_image, prompt=prompt or "", negative_prompt=negative or "",
                denoising_strength=float(denoise), width=int(w), height=int(h),
                steps=int(steps), cfg_scale=float(cfg), seed=int(seed), sampler=sampler,
                scheduler=scheduler, clip_skip=int(clip_skip), mask_blur=int(mask_blur),
                inpainting_fill=int(inpainting_fill), full_res=bool(full_res),
                padding=int(padding), batch_size=int(batch_size), callback=callback)
        finally:
            pipe._use_laplacian_blend = prev_lap_blend
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
            info = _pnginfo(checkpoint_path, prompt, negative, seed, steps, cfg,
                            sampler, scheduler, clip_skip, im)
            im.save(f, pnginfo=info)
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
_inpaint_ip_variant = None  # which IP-Adapter variant the cached pipe loaded

# Cooperative abort: the UI sets the flag; the diffusers step callback below trips
# the pipeline's _interrupt so the denoising loop bails out early.
_abort_flag = False

# Live-preview hand-off: the step callback (running inside the blocking diffusers
# loop, possibly on a worker thread) stashes the latest decoded PIL preview here;
# the streaming Gradio handler polls it to update the on-page preview Image. A
# monotonically-rising counter lets the poller detect a new frame cheaply. Nothing
# here touches generation correctness — it's all preview-only and default-OFF.
import threading as _threading
_preview_lock = _threading.Lock()
_preview_img = None
_preview_seq = 0


def set_preview(img) -> None:
    """Publish the latest live preview PIL image (or None to clear)."""
    global _preview_img, _preview_seq
    with _preview_lock:
        _preview_img = img
        _preview_seq += 1


def get_preview():
    """Return ``(seq, img)`` — the current preview frame and its sequence number.
    The poller compares ``seq`` to know when a new frame is ready."""
    with _preview_lock:
        return _preview_seq, _preview_img


def clear_preview() -> None:
    """Reset the live-preview hand-off at the start/end of a generation run."""
    set_preview(None)


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
    global _inpaint_pipe, _inpaint_ckpt, _inpaint_with_ip, _inpaint_ip_variant
    _inpaint_pipe = None
    _inpaint_ckpt = None
    _inpaint_with_ip = None
    _inpaint_ip_variant = None
    _free_torch()


def _get_inpaint(checkpoint_path, with_ip=True, ip_variant="plus"):
    """Cached diffusers SDXL inpaint pipeline. With ``with_ip`` it also loads the
    IP-Adapter (for body swap / colour reference); without it (ADetailer) it's a
    plain inpaint.

    ``ip_variant`` selects the IP-Adapter weights:
      - "plus" (default): CLIP ViT-H look transfer (texture/colour — current behaviour).
      - "faceid" / "faceid_plus": InsightFace-embedding identity transfer. These load
        from the separate ``h94/IP-Adapter-FaceID`` repo and expect the caller to pass
        ``ip_adapter_image_embeds`` (face embeddings) rather than ``ip_adapter_image``.
        ``faceid_plus`` (FaceID-Plus-v2 on SDXL) also loads a LAION CLIP ViT-H encoder
        for its dual-guidance look channel.
    """
    global _inpaint_pipe, _inpaint_ckpt, _inpaint_with_ip, _inpaint_ip_variant
    if (_inpaint_pipe is not None and _inpaint_ckpt == checkpoint_path
            and _inpaint_with_ip == with_ip
            and (not with_ip or _inpaint_ip_variant == ip_variant)):
        return _inpaint_pipe
    release_inpaint()
    import torch
    from diffusers import StableDiffusionXLInpaintPipeline
    is_faceid = with_ip and str(ip_variant).startswith("faceid")
    with models.no_auto_download():
        # FaceID-Plus-v2 needs a LAION CLIP ViT-H encoder baked into the pipe for its
        # dual-guidance look channel; load it at construction (other variants don't).
        image_encoder = None
        if is_faceid and ip_variant == "faceid_plus":
            from transformers import CLIPVisionModelWithProjection
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                "laion/CLIP-ViT-H-14-laion2B-s32B-b79K", torch_dtype=torch.float16)
        if image_encoder is not None:
            pipe = StableDiffusionXLInpaintPipeline.from_single_file(
                checkpoint_path, torch_dtype=torch.float16, image_encoder=image_encoder)
        else:
            pipe = StableDiffusionXLInpaintPipeline.from_single_file(
                checkpoint_path, torch_dtype=torch.float16)
        if with_ip and is_faceid:
            # FaceID weights live in a separate repo and condition on InsightFace
            # embeddings (no CLIP image encoder for the plain faceid variant).
            from .sd.controlnet_types import get_ip_adapter_config
            _subfolder, weight = get_ip_adapter_config(ip_variant, True)  # SDXL
            load_kwargs = {"weight_name": weight}
            if ip_variant != "faceid_plus":
                load_kwargs["image_encoder_folder"] = None
            pipe.load_ip_adapter("h94/IP-Adapter-FaceID", **load_kwargs)
        elif with_ip:
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
    _inpaint_ip_variant = ip_variant if with_ip else None
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
        app = _get_face_app()
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
                       cfg=6.0, seed=-1, out_dir=None, loras=None, ip_variant="plus",
                       progress=None) -> str | None:
    """Apply a reference identity onto the masked region of a target via IP-Adapter
    inpaint. Optional SDXL ``loras`` are applied for the pass. Returns the saved path.

    ``ip_variant`` chooses the IP-Adapter weights:
      - "plus" (default): CLIP ViT-H look transfer (texture/colour — current behaviour;
        body swap relies on this).
      - "faceid" / "faceid_plus": true identity transfer via InsightFace face
        embeddings. Needs a clearly visible face in the reference and the
        ``IP-Adapter FaceID`` weights from Settings → Models.
    """
    import random as _random
    import time
    import torch
    from PIL import Image
    if seed is None or int(seed) < 0:
        seed = _random.randint(0, 2**31 - 1)
    is_faceid = str(ip_variant).startswith("faceid")
    release_sd()  # free the txt2img checkpoint before stacking the inpaint+IP pipe
    pipe = _get_inpaint(checkpoint_path, with_ip=True, ip_variant=ip_variant)
    pipe.set_ip_adapter_scale(float(ip_scale))
    target = Image.open(target_path).convert("RGB")
    ref = Image.open(reference_path).convert("RGB")
    w, h = target.size
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=dev).manual_seed(int(seed))

    # FaceID conditions on InsightFace face embeddings (not the raw image) — build
    # them up front so a "no face detected" error surfaces before the diffusion run.
    ip_kwargs: dict = {}
    if is_faceid:
        from .sd.sd_pipeline import SDImagePipeline
        face_embeds = SDImagePipeline._extract_face_embeddings(ref)
        ip_kwargs["ip_adapter_image_embeds"] = face_embeds
        if ip_variant == "faceid_plus":
            # FaceID-Plus-v2 dual guidance: feed the reference's CLIP embeds into the
            # adapter's look channel (mirrors generate_body_double's faceid_plus path).
            clip_embeds = pipe.prepare_ip_adapter_image_embeds(
                [ref], None, torch.device(dev), 1, True)[0]
            proj = pipe.unet.encoder_hid_proj.image_projection_layers[0]
            proj.clip_embeds = clip_embeds.to(dtype=torch.float16)
            proj.shortcut = True  # FaceID-Plus-v2 uses the shortcut path
    else:
        ip_kwargs["ip_adapter_image"] = ref

    _remove_loras_from_diffusers(pipe, True)   # clear any stale from a prior pass
    _la = _apply_loras_to_diffusers(pipe, loras)
    try:
        with models.no_auto_download():
            img = pipe(prompt=prompt or "", negative_prompt=negative or "",
                       image=target, mask_image=mask_image,
                       strength=float(denoise), num_inference_steps=int(steps),
                       guidance_scale=float(cfg), width=w, height=h, generator=gen,
                       callback_on_step_end=_abort_callback, **ip_kwargs).images[0]
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
        app = _get_face_app()
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
    if sampler:  # honour the requested sampler/scheduler on the standalone pipe
        try:
            from .sd.sd_samplers import create_scheduler
            sched_config = dict(pipe.scheduler.config)
            pipe.scheduler = create_scheduler(sampler, scheduler or "Automatic", sched_config)
        except Exception:
            logger.warning("failed setting ADetailer scheduler", exc_info=True)
    _remove_loras_from_diffusers(pipe, True)   # clear any stale from a prior pass
    _la = _apply_loras_to_diffusers(pipe, loras)
    result = src.copy()
    try:
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
    finally:
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
    a final person/body-detail pass runs (using ``adet_prompt``/``adet_neg`` if given)."""
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
