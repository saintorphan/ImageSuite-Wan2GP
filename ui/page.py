"""One reusable builder for all three Image Suite pages.

The pages share a layout — settings bar, prompt pair, an optional input image,
a Generate button, a result gallery and a "send to…" action row — and differ
only in:

  * txt2img : no input image.
  * img2img : a single init image + denoise strength.
  * inpaint : a full PaintShop draw+mask canvas (see ui/canvas.py) + denoise.

build_page() returns (page_components_dict). All cross-page wiring (generation,
send-to-page, send-to-Img2Vid, save) is done in plugin.py, which owns the
Wan2GP api_session and globals.
"""
from __future__ import annotations

import gradio as gr

from . import canvas as _canvas
from . import modify_canvas as _modify
from . import enhance as _enhance
from ..core import overlays as _ovcore
from ..core import presets as _presets
from ..core import prompt_library as _plib
from ..core.sd import sd_samplers as _sd_samplers

# Sourced from the SD backend so only sampler/scheduler values create_scheduler()
# actually supports are offered (anything else silently falls back to Euler /
# Automatic). Each includes the neutral native sentinel ("default" / "") used by
# Flux/Z-Image/Qwen, whose pipelines own their own scheduler.
SAMPLERS = _sd_samplers.list_samplers()
SCHEDULERS = _sd_samplers.list_schedulers()
# Initial outpaint target sizes (SDXL set); refreshed per model family on change.
_OUTPAINT_SIZES_INIT = ["1024×1024", "1152×896", "896×1152", "1216×832", "832×1216",
                        "1344×768", "768×1344", "1536×640", "640×1536"]
# Initial resolution presets (SDXL buckets, grouped by orientation); refreshed per
# model family on model select (plugin._on_model → discovery.resolution_presets).
# (label, 'W×H' value) — must mirror discovery.resolution_presets("") output.
_RES_PRESETS_INIT = [
    ("⬛ 1:1 · 1024×1024", "1024×1024"),
    ("📱 Portrait · 896×1152", "896×1152"),
    ("📱 Portrait · 832×1216", "832×1216"),
    ("📱 Portrait · 768×1344", "768×1344"),
    ("📱 Portrait · 640×1536", "640×1536"),
    ("🖼 Landscape · 1152×896", "1152×896"),
    ("🖼 Landscape · 1216×832", "1216×832"),
    ("🖼 Landscape · 1344×768", "1344×768"),
    ("🖼 Landscape · 1536×640", "1536×640"),
]

MODES = ("txt2img", "img2img", "inpaint", "modify")


def _settings_bar(model_choices, lora_choices, mode, sdxl_choices=None):
    c = {}
    # Universal generation settings — laid out identically on every tab so the
    # panel reads the same whichever sub-tab you're on. Two controls per row
    # (sliders never share a row with radios) keeps the columns aligned.
    with gr.Accordion("Generation settings", open=True, elem_classes="imagesuite-acc"):
        with gr.Row():
            c["model"] = gr.Dropdown(label="Model", choices=model_choices or [], scale=3)
            c["sampler"] = gr.Dropdown(label="Sampler", choices=SAMPLERS,
                                       value="DPM++ 2M", scale=1)
            c["scheduler"] = gr.Dropdown(label="Scheduler", choices=SCHEDULERS,
                                         value="Karras", scale=1)
        with gr.Row():
            c["steps"] = gr.Slider(1, 60, value=28, step=1, label="Steps")
            c["cfg"] = gr.Slider(1.0, 15.0, value=6.0, step=0.5, label="CFG")
            c["clip_skip"] = gr.Slider(1, 4, value=2, step=1, label="Clip skip")
            c["seed"] = gr.Number(value=-1, label="Seed (-1=random)", precision=0)
        # Read-only "Last seed" + Reuse: when Seed is -1 the actual random seed used
        # is otherwise lost (it's only in the sd_<seed>_*.png filename). The gen
        # handlers surface it here; ♻ copies it back into Seed above (wired in
        # plugin._wire_page).
        with gr.Row():
            c["last_seed"] = gr.Number(value=None, label="Last seed", precision=0,
                                       interactive=False, scale=3)
            c["reuse_seed"] = gr.Button("♻ Reuse seed", size="sm", scale=1,
                                        min_width=110)
        # Resolution: a per-model preset picker (trained buckets by orientation) +
        # an aspect lock. Picking a preset fills Width/Height; with Lock on, dragging
        # one slider scales the other to hold the ratio (wired in plugin._wire_page).
        with gr.Row():
            c["res_preset"] = gr.Dropdown(
                label="Resolution preset (follows the selected model)",
                choices=_RES_PRESETS_INIT, value=None, scale=3)
            c["res_lock"] = gr.Checkbox(label="🔒 Lock ratio", value=False,
                                        scale=1, min_width=90)
        with gr.Row():
            c["width"] = gr.Slider(256, 2048, value=832, step=64, label="Width")
            c["height"] = gr.Slider(256, 2048, value=1216, step=64, label="Height")
        # Aspect ratio captured when Lock is engaged (or a preset is picked); the
        # slider-sync handlers in plugin.py use it. Holds None when unlocked.
        c["res_ratio"] = gr.State(None)
        # Batch count always sits in the row below Width/Height (with Denoise
        # beside it on img2img/inpaint) so it's in the same spot on every tab.
        with gr.Row():
            if mode != "txt2img":
                c["denoise"] = gr.Slider(
                    0.0, 1.0, value=0.6 if mode == "img2img" else 0.75, step=0.05,
                    label=("Denoise (paint adherence: lower = follow paint)"
                           if mode == "inpaint" else "Denoise strength"))
            c["count"] = gr.Slider(1, 8, value=1, step=1, label="Batch count")
        if mode == "img2img":
            # How the init image is fit to Width×Height (reForge "Resize mode").
            c["resize_mode"] = gr.Radio(
                ["Just resize", "Crop and resize", "Resize and fill"],
                value="Just resize", label="Resize mode")
        with gr.Row():
            c["loras"] = gr.Dropdown(label="LoRAs (SDXL family)", multiselect=True,
                                     choices=lora_choices or [], scale=3)
            c["lora_mult"] = gr.Textbox(label="Multipliers", placeholder="0.8, 1.0", scale=1)
        # Distilled few-step presets (SDXL family). Off (default) = unchanged. Each
        # other mode fuses a matched distill LoRA from your SDXL LoRA dir and forces
        # the scheduler + clamped steps/CFG; if no matching distill LoRA is found it
        # warns and generates normally. Native Flux/Z-Image/Qwen ignore it.
        with gr.Row():
            c["turbo"] = gr.Dropdown(
                label="⚡ Turbo (distilled few-step)",
                choices=_presets.turbo_choices(), value=_presets.TURBO_OFF, scale=3,
                info="SDXL family only. Fuses a matched distill LoRA (LCM / Hyper-SD "
                     "/ Lightning) and forces sampler + steps + CFG. CFG ~1 means "
                     "classifier-free guidance is off, so the NEGATIVE prompt is "
                     "IGNORED. No matching LoRA → warns and runs normally.")
        # Hi-res fix (txt2img only, SDXL family): base gen → upscale by HR scale →
        # low-denoise img2img second pass on the same loaded pipe (no second model).
        # HR scale 1.0 (default) = OFF → behaviour unchanged. Native models ignore it.
        if mode == "txt2img":
            with gr.Row():
                c["hr_scale"] = gr.Slider(
                    1.0, 2.0, value=1.0, step=0.05, label="🔍 Hi-res fix scale",
                    info="1.0 = off. >1.0 upscales then runs a low-denoise img2img "
                         "refine pass (sharper, larger). SDXL family only.")
                c["hr_denoise"] = gr.Slider(
                    0.2, 0.7, value=0.4, step=0.05, label="Hi-res denoise",
                    info="Refine-pass strength. Lower keeps the base composition; "
                         "higher adds detail (and can drift). Used when scale > 1.0.")
            # SDXL refiner (txt2img only, SDXL family): the base pass stops at the
            # switch-at fraction and a SECOND SDXL checkpoint finishes the schedule
            # (diffusers ensemble-of-experts). Disabled by default → single-pass,
            # behaviour unchanged. Native Flux/Z-Image/Qwen ignore it.
            with gr.Accordion("🪄 Refiner (SDXL only)", open=False,
                              elem_classes="imagesuite-acc"):
                gr.Markdown(
                    "Run a second **SDXL / Pony / Illustrious** checkpoint over the "
                    "tail end of the schedule for crisper detail (diffusers "
                    "ensemble-of-experts). Off = single pass. Native "
                    "Flux / Z-Image / Qwen ignore this.",
                    elem_classes="imagesuite-help")
                c["refiner_enable"] = gr.Checkbox(
                    value=False, label="Enable refiner pass")
                c["refiner_model"] = gr.Dropdown(
                    label="Refiner checkpoint (SDXL family)",
                    choices=sdxl_choices or [], value=None)
                with gr.Row():
                    c["refiner_switch_at"] = gr.Slider(
                        0.5, 0.95, value=0.8, step=0.05, label="Switch at",
                        info="Fraction of the schedule done on the BASE model before "
                             "the refiner takes over (0.8 = base does 80%).")
                    c["refiner_steps"] = gr.Slider(
                        4, 40, value=10, step=1, label="Refiner steps")
                    c["refiner_cfg"] = gr.Slider(
                        1.0, 15.0, value=7.0, step=0.5, label="Refiner CFG")

    if mode == "inpaint":
        # Mask + inpaint knobs live in their own block, so the generation settings
        # above stay byte-for-byte identical to the other tabs. The SDXL-only knobs
        # are nested but open by default (this workbench is used SDXL-first; native
        # Flux/Z-Image/Qwen simply ignore them).
        with gr.Accordion("Mask & inpaint", open=True, elem_classes="imagesuite-acc"):
            c["mask_mode"] = gr.Radio(
                ["Inpaint masked", "Inpaint not masked"],
                value="Inpaint masked", label="Mask mode")
            c["feather"] = gr.Slider(0, 64, value=4, step=1, label="Mask blur (px)")
            with gr.Accordion("SDXL inpaint options", open=True,
                              elem_classes="imagesuite-acc"):
                gr.Markdown(
                    "Used only by **SDXL / Pony / Illustrious**. Native "
                    "Flux / Z-Image / Qwen ignore these — they use denoise + mask "
                    "blur only.", elem_classes="imagesuite-help")
                c["inpaint_fill"] = gr.Radio(
                    ["fill", "original", "latent noise", "latent nothing"],
                    value="original", label="Masked content")
                c["inpaint_area"] = gr.Radio(
                    ["Whole picture", "Only masked"], value="Whole picture",
                    label="Inpaint area")
                c["padding"] = gr.Slider(0, 256, value=32, step=8,
                                         label="Only masked padding (px)")
                c["seamless"] = gr.Checkbox(
                    value=False, label="Seamless blend",
                    info="Laplacian pyramid paste-back — hides the mask seam at "
                         "every frequency band instead of just feathering its edge. "
                         "Off uses the plain feathered composite.")
    return c


def _prompt_library_block(c, mode):
    """Collapsible 'Prompt Library' (above the generation settings, collapsed by
    default) — save / update / load / delete a full setup: prompt, negative, every
    generation setting, model, LoRAs and post-process settings, to ONE shared,
    reusable list across all three tabs. These components are created here; plugin.py
    wires the CRUD against the whole page (and keeps every tab's list in sync)."""
    with gr.Accordion("📚 Prompt Library", open=False, elem_classes="imagesuite-acc"):
        gr.Markdown(
            "Save the current prompt + **all** settings (model, LoRAs, generation "
            "and post-process) and reload them on any tab. Reference images aren't "
            "stored. **Save as** a new name, **Update** the selected one, **Load** it "
            "into the form, or **Delete** it.", elem_classes="imagesuite-help")
        with gr.Row():
            c["pl_name"] = gr.Textbox(label="Name", placeholder="e.g. moody portrait",
                                      scale=2)
            c["pl_saved"] = gr.Dropdown(label="Saved prompts", choices=_plib.names(),
                                        value=None, scale=2)
        with gr.Row():
            c["pl_save"] = gr.Button("💾 Save as", size="sm")
            c["pl_update"] = gr.Button("⟳ Update", size="sm")
            c["pl_load"] = gr.Button("📥 Load", variant="primary", size="sm")
            c["pl_delete"] = gr.Button("🗑 Delete", variant="stop", size="sm")
        # Recall A1111-style "parameters" PNG metadata from any image (one written by
        # this plugin's SDXL backend, or any A1111/Forge/reForge export) straight into
        # the form. UploadButton hands a temp file path to the handler in plugin.py;
        # an image with no readable metadata is a friendly no-op.
        c["pl_read"] = gr.UploadButton("🔎 Read params from PNG", file_count="single",
                                       file_types=["image"], size="sm")
        c["pl_status"] = gr.Markdown("", elem_classes="imagesuite-genstatus")


def _advanced_prompt_default() -> bool:
    """Default the 'Advanced prompt' toggle ON only when compel is already importable
    (so SDXL gets weights/BREAK/long-prompt out of the box where it works) and OFF
    otherwise (no surprise install, raw-string path = current behaviour). Best-effort;
    a build never breaks on this check."""
    try:
        from ..core import deps as _deps
        return bool(_deps.has("compel"))
    except Exception:
        return False


def _prompt_block(c):
    """Prompt + negative + Qwen-abliterated Enhance buttons (wired in plugin.py)."""
    c["pos"] = gr.Textbox(label="Prompt", lines=3, placeholder="Describe the image…")
    c["neg"] = gr.Textbox(label="Negative prompt", lines=2,
                          placeholder="low quality, blurry…")
    # Advanced prompting (SDXL / Pony / Illustrious only): parse (token:1.3) weights,
    # the BREAK chunk separator, and >77-token prompts via compel embeds. Default ON
    # when compel is present, else OFF. Off = the exact raw-string path (weights
    # literal, >77 tokens truncated) — current behaviour. Native models ignore it.
    c["adv_prompt"] = gr.Checkbox(
        value=_advanced_prompt_default(),
        label="🎚 Advanced prompt (weights / BREAK / long)",
        info="SDXL family only. Reads (token:1.3) emphasis, BREAK to start a new "
             "77-token chunk, and prompts past 77 tokens. Off = raw text (weights "
             "literal, long prompts truncated). Installs 'compel' on first use.")
    with gr.Row():
        c["enhance_pos"] = gr.Button("✨ Enhance prompt")
        c["enhance_neg"] = gr.Button("✨ Enhance negative")
        c["interrogate"] = gr.Button("🔍 Interrogate image")
    with gr.Row():
        # WD14 tag confidence (booru families only; BLIP captions ignore it) and
        # whether interrogation overwrites the prompt (default) or appends tags.
        c["interro_thresh"] = gr.Slider(0.1, 0.9, value=0.35, step=0.05,
                                        label="Interrogate tag threshold")
        c["interro_mode"] = gr.Radio(["Replace", "Append"], value="Replace",
                                     label="Interrogate result")


# ControlNet conditioning types offered in the UI. SINGLE ControlNet for now.
# canny/depth/openpose have verified standalone SDXL ControlNet models; lineart/tile
# need the all-in-one ControlNet Union model (download it from the Prereqs panel).
_CONTROLNET_TYPES = [
    ("Canny (edges)", "canny"),
    ("Depth (Midas)", "depth"),
    ("OpenPose (pose)", "openpose"),
    ("Lineart", "lineart"),
    ("Tile / raw", "tile"),
]


def _controlnet_block(c, mode):
    """ControlNet accordion (txt2img / img2img, SDXL family). Default OFF: when the
    Enable checkbox is unchecked the generation path is byte-for-byte unchanged.
    Native Flux / Z-Image / Qwen ignore ControlNet entirely. SINGLE ControlNet for
    now (the wrapper + pipeline already support multi — extend the inputs here)."""
    with gr.Accordion("🎛 ControlNet (SDXL only)", open=False,
                      elem_classes="imagesuite-acc"):
        gr.Markdown(
            "Guide the generation with a **control image** (edges / depth / pose / "
            "lineart). **SDXL / Pony / Illustrious only** — native models ignore it. "
            "Off (default) = no ControlNet. Download the ControlNet model for your "
            "type from the **Prereqs** panel first (canny/depth/pose have their own "
            "SDXL models; lineart/tile need **ControlNet Union**).",
            elem_classes="imagesuite-help")
        c["cn_enable"] = gr.Checkbox(value=False, label="Enable ControlNet")
        with gr.Row():
            c["cn_type"] = gr.Dropdown(
                label="Type", choices=_CONTROLNET_TYPES, value="canny", scale=2)
            c["cn_use_raw"] = gr.Checkbox(
                value=False, label="Use raw image (skip preprocessor)", scale=1,
                info="Feed the control image straight in (it's already a "
                     "condition map, or for Tile).")
        c["cn_image"] = gr.Image(
            label="Control image", type="filepath", height=200,
            elem_classes="imagesuite-initthumb")
        with gr.Row():
            c["cn_preprocess"] = gr.Button("🔎 Preprocess preview", size="sm", scale=1)
        c["cn_preview"] = gr.Image(
            label="Preprocessed (condition map)", type="filepath", height=200,
            interactive=False)
        with gr.Row():
            c["cn_strength"] = gr.Slider(
                0.0, 2.0, value=0.7, step=0.05, label="ControlNet strength",
                info="Conditioning scale — how strongly the control image steers.")
        with gr.Row():
            c["cn_guidance_start"] = gr.Slider(
                0.0, 1.0, value=0.0, step=0.05, label="Guidance start",
                info="Fraction of the schedule before ControlNet kicks in.")
            c["cn_guidance_end"] = gr.Slider(
                0.0, 1.0, value=1.0, step=0.05, label="Guidance end",
                info="Fraction of the schedule after which ControlNet stops.")
        with gr.Accordion("Preprocessor options", open=False,
                          elem_classes="imagesuite-acc"):
            with gr.Row():
                c["cn_detect_res"] = gr.Slider(
                    256, 1024, value=512, step=64, label="Detect resolution")
                c["cn_image_res"] = gr.Slider(
                    256, 1536, value=512, step=64, label="Output resolution")
            with gr.Row():
                c["cn_canny_low"] = gr.Slider(
                    1, 255, value=100, step=1, label="Canny low threshold")
                c["cn_canny_high"] = gr.Slider(
                    1, 255, value=200, step=1, label="Canny high threshold")
        c["cn_status"] = gr.Markdown("", elem_classes="imagesuite-genstatus")


# XYZ-sweep axes. Each value is the per-image SD param the axis varies; the labels
# read naturally in the accordion. "none" disables that axis. sampler/scheduler take
# names from the dropdowns above (comma-separated); the rest take numbers.
_SWEEP_AXES = [
    ("(none)", "none"),
    ("Seed", "seed"),
    ("Steps", "steps"),
    ("CFG", "cfg"),
    ("Sampler", "sampler"),
    ("Scheduler", "scheduler"),
    ("Denoise", "denoise"),
    ("Clip skip", "clip_skip"),
]


def _xyz_sweep_block(c, mode):
    """Parameter sweep / XYZ grid (txt2img + img2img, SDXL family). Default OFF: when
    the Enable checkbox is unchecked the normal single/batch Generate path runs
    byte-for-byte unchanged. Enabled, it runs the cartesian product of the X (and
    optional Y) value lists through the EXISTING per-image SD generate path and
    assembles one labelled contact-sheet grid (saved alongside the individual images).
    SD-path only — native Flux / Z-Image / Qwen ignore the sweep and gen normally."""
    with gr.Accordion("🔬 XYZ sweep (parameter grid)", open=False,
                      elem_classes="imagesuite-acc"):
        gr.Markdown(
            "Vary 1–2 settings across a **value list** and assemble a labelled "
            "contact-sheet grid (plus the individual images). Pick an axis, enter a "
            "comma-separated list (e.g. Steps `20, 30, 40` or Sampler "
            "`Euler, DPM++ 2M`). Off (default) = normal Generate. **SDXL / Pony / "
            "Illustrious only** — native models ignore it. Each cell reuses the "
            "page's model / LoRAs / prompt; the per-image generator runs once per "
            "combination, so a big grid takes a while." +
            ("" if mode != "txt2img" else
             " *Denoise* only applies on Img2Img."),
            elem_classes="imagesuite-help")
        c["sweep_enable"] = gr.Checkbox(value=False, label="Enable XYZ sweep")
        with gr.Row():
            c["sweep_x_axis"] = gr.Dropdown(
                label="X axis", choices=_SWEEP_AXES, value="steps", scale=1)
            c["sweep_x_values"] = gr.Textbox(
                label="X values (comma-separated)", placeholder="20, 30, 40", scale=2)
        with gr.Row():
            c["sweep_y_axis"] = gr.Dropdown(
                label="Y axis", choices=_SWEEP_AXES, value="none", scale=1)
            c["sweep_y_values"] = gr.Textbox(
                label="Y values (comma-separated)", placeholder="(leave blank for 1-D)",
                scale=2)


def _overlays_strip_block(c):
    """Folder picker for the overlay strip rendered at the bottom of the canvas
    iframe. The thumbnails live in the canvas (so they're draggable onto it);
    this just chooses which folder to load (wired in plugin.py)."""
    folders = _ovcore.list_folders()
    with gr.Accordion("Overlays — drag onto the canvas", open=False,
                      elem_classes="imagesuite-acc"):
        with gr.Row():
            c["ov_folder"] = gr.Dropdown(
                label="Overlay folder", choices=folders,
                value=folders[0] if folders else None, scale=4)
            c["ov_reload"] = gr.Button("🔄 Load", scale=1)
        gr.Markdown("Pick a folder, then drag thumbnails from the strip at the "
                    "**bottom of the canvas** onto your image — each drop adds a new "
                    "layer (below the mask). Double-click a thumbnail to preview. "
                    "Manage the library in the **Overlays** tab.",
                    elem_classes="imagesuite-help")


def _touchup_block(c):
    """Outpaint (its own operation — the reForge "Outpainting" script analog):
    extend the canvas image outward and fill the new edges with the model. This is
    NOT a resize mode (that's the img2img radio); it adds new canvas + generates."""
    with gr.Accordion("Outpaint", open=False, elem_classes="imagesuite-acc"):
        gr.Markdown("Extend the canvas image outward and fill the new edges with the "
                    "page's **model + prompt** (runs at full denoise on the new area). "
                    "Pick a target size (centered) or set per-side pixels.",
                    elem_classes="imagesuite-help")
        c["out_size"] = gr.Dropdown(
            ["Custom (use px below)"] + _OUTPAINT_SIZES_INIT,
            value="Custom (use px below)",
            label="Target size (centered — choices follow the selected model's family)")
        with gr.Row():
            c["out_top"] = gr.Slider(0, 512, value=0, step=8, label="Top px")
            c["out_bottom"] = gr.Slider(0, 512, value=0, step=8, label="Bottom px")
        with gr.Row():
            c["out_left"] = gr.Slider(0, 512, value=0, step=8, label="Left px")
            c["out_right"] = gr.Slider(0, 512, value=0, step=8, label="Right px")
        with gr.Row():
            c["out_feather"] = gr.Slider(0, 128, value=24, step=4,
                                         label="Edge blend (mask blur, px)")
        with gr.Row():
            c["out_run"] = gr.Button("Outpaint", variant="primary", scale=4)
            c["out_abort"] = gr.Button("⛔", variant="stop", scale=1)


def _persisted_results(mode):
    """A tab's persisted result images (as PIL) + the first path, so the gallery comes
    back populated after a restart by being the gallery's INITIAL value (no load-event
    needed). Best-effort: returns (None, None) on any problem so a build never breaks."""
    try:
        import os
        from PIL import Image
        from ..core import paths as _paths
        kept = [p for p in _paths.get_results(mode) if os.path.exists(p)]
        imgs = []
        for p in kept:
            try:
                with Image.open(p) as im:
                    imgs.append(im.copy())
            except Exception:
                pass
        return (imgs or None), (kept[0] if kept else None)
    except Exception:
        return None, None


def _results_gallery(c, mode):
    """Just the results gallery. Split out so MultiCanvas can place it in its LEFT
    column (between Overlays and Enhancement) while the send/save row stays on the
    right. NOTE: no preview=True — it auto-selects the first item on populate, firing a
    select round-trip every generation and tripping Gradio's cache check on the native
    backend's out-of-cache output path. Restore the tab's last results as the INITIAL
    value (survives a restart without a load event — see _persisted_results)."""
    gallery_h = {"txt2img": 520, "img2img": 620}.get(mode, 460)
    _init_imgs, _ = _persisted_results(mode)
    c["gallery"] = gr.Gallery(label="Results", columns=2, height=gallery_h,
                              value=_init_imgs,
                              elem_classes=["imagesuite-gallery", "imagesuite-results"],
                              object_fit="contain")


def _results_block(c, mode, include_gallery=True, send_panel_fn=None):
    """The shared results column: the gallery (unless already placed elsewhere —
    MultiCanvas puts it in the left column), then (Txt2Img/Img2Img) the Generate/Abort
    row + status, then the send-to row + Save As. Click a gallery item to make it the
    selection (``picked``). MultiCanvas builds its own 'Inpaint' button beside the
    canvas, so this block skips the Generate row for it.

    send_panel_fn(picked) — optional: renders the unified SendTo picker for
    cross-destination sends (img2vid, other plugins). The in-suite buttons above
    stay for instant same-plugin (Txt2Img/Img2Img/MultiCanvas/Modify) moves, which
    the SendTo inbox can't do (a same-tab switch doesn't fire the drain)."""
    _, _init_path = _persisted_results(mode)
    if include_gallery:
        _results_gallery(c, mode)
    # Live latent preview (txt2img only): a small TAESD-decoded image that updates
    # every few sampling steps while generating. Hidden unless the Settings toggle
    # is on (default OFF) so nothing changes for existing users. Wired to the
    # streaming Generate handler in plugin.py.
    if mode == "txt2img":
        try:
            from ..core import paths as _paths_live
            _live_on = bool(_paths_live.get_sd_live_preview())
        except Exception:
            _live_on = False
        c["live_preview"] = gr.Image(
            label="Live preview (sampling…)", visible=_live_on, height=256,
            interactive=False, show_download_button=False,
            elem_classes="imagesuite-live-preview")
    c["picked"] = gr.State(_init_path)  # path of the selected (clicked) result
    if mode not in ("inpaint", "modify"):   # Generate/Abort + status sit directly under the gallery
        with gr.Row(elem_classes="imagesuite-genrow"):
            c["generate"] = gr.Button(
                {"txt2img": "Generate", "img2img": "Reimagine (img2img)"}[mode],
                variant="primary", scale=4)
            c["abort"] = gr.Button("⛔ Abort", variant="stop", scale=1)
        c["gen_status"] = gr.Markdown("", elem_classes="imagesuite-genstatus")
    gr.Markdown("**Send selected result to →**")
    with gr.Row(elem_classes="imagesuite-sendrow"):
        # Txt2Img/Img2Img hide their own-page button; MultiCanvas keeps it as
        # "Back to canvas" so the inpainted result can be re-loaded for another pass.
        c["to_t2i"] = gr.Button("Txt2Img", visible=mode not in ("txt2img", "modify"))
        c["to_i2i"] = gr.Button("Img2Img", visible=mode != "img2img")
        c["to_inp"] = gr.Button("↻ Back to canvas" if mode == "inpaint"
                                else "MultiCanvas")
        c["to_mod"] = gr.Button("Modify", visible=mode != "modify")
        # Img2Vid + cross-plugin sends live in the unified SendTo panel below when the
        # SendTo plugin is installed; this standalone button is the no-SendTo fallback
        # (its handler stays wired either way).
        c["to_i2v"] = gr.Button("Img2Vid", visible=send_panel_fn is None)
    c["save"] = gr.DownloadButton("Save As…", value=_init_path,
                                  elem_classes="imagesuite-savebtn")
    # Unified SendTo picker (cross-destination: img2vid + other plugins). Rendered
    # only when SendTo is present (plugin.py passes send_panel_fn=None otherwise).
    c["_sendto_panel"] = send_panel_fn(c["picked"]) if send_panel_fn else None


def build_page(mode, model_choices=None, lora_choices=None, sdxl_choices=None,
               send_panel_fn=None):
    assert mode in MODES, mode
    c = {"mode": mode}

    if mode == "inpaint":
        # Editor layout: big canvas (with its own tool rail) on the left; a single
        # column of generation controls + results on the right.
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown(
                    "Paint on the lower layer; use the **Mask** tools (or **Auto-mask**) "
                    "to mark what to regenerate. Inside the mask your paint *guides* the "
                    "inpaint; outside it's a direct edit. **Denoise** = how loosely the "
                    "model follows your paint.", elem_classes="imagesuite-help")
                c.update(_canvas.build_canvas("inpaint"))
                with gr.Row(elem_classes="imagesuite-genrow"):
                    c["mc_magicmask"] = gr.Button(
                        "🪄 Magic select subject",
                        elem_classes="imagesuite-help")
                    c["mc_mask_status"] = gr.Markdown(
                        "", elem_classes="imagesuite-genstatus")
                _overlays_strip_block(c)                           # folder picker
                _results_block(c, mode, send_panel_fn=send_panel_fn)                            # results between overlays & enhance
                _touchup_block(c)                                  # under the canvas
                c.update(_enhance.build_enhance_sections(mode, sdxl_choices, lora_choices))  # under the canvas
            with gr.Column(scale=2):
                _prompt_library_block(c, mode)                     # above settings
                c.update(_settings_bar(model_choices, lora_choices, mode))
                _prompt_block(c)
                with gr.Row(elem_classes="imagesuite-genrow"):
                    c["generate"] = gr.Button("Inpaint", variant="primary", scale=4)
                    c["abort"] = gr.Button("⛔ Abort", variant="stop", scale=1)
                c["gen_status"] = gr.Markdown("", elem_classes="imagesuite-genstatus")
        return c

    if mode == "modify":
        # Editor layout mirrors MultiCanvas: the Modify canvas + results on the
        # left; the load / colour-match / save controls on the right. No prompt or
        # model settings — this page only edits an existing image.
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown(
                    "Crop, zoom and colour-correct an image. Load one on the right "
                    "or **Send to Modify** from any result. Adjustments preview live; "
                    "click **Save to results** to keep the edited image, then send it "
                    "anywhere.", elem_classes="imagesuite-help")
                c.update(_modify.build_modify_canvas("modify"))
                _results_block(c, mode, send_panel_fn=send_panel_fn)
            with gr.Column(scale=2):
                c["mod_input"] = gr.Image(
                    label="Image to modify (double-click to enlarge)",
                    type="filepath", height=220, elem_classes="imagesuite-initthumb")
                with gr.Accordion("Colour match", open=True,
                                  elem_classes="imagesuite-acc"):
                    gr.Markdown(
                        "Match the edited image's colours to a reference (LAB "
                        "mean/std transfer), then keep editing on top.",
                        elem_classes="imagesuite-help")
                    c["mod_ref"] = gr.Image(label="Reference image",
                                            type="filepath", height=180)
                    c["mod_match"] = gr.Button("🎨 Apply colour match")
                with gr.Row(elem_classes="imagesuite-genrow"):
                    c["mod_removebg"] = gr.Button("✂️ Remove background")
                with gr.Accordion("Upscale", open=False,
                                  elem_classes="imagesuite-acc"):
                    gr.Markdown(
                        "Enlarge the edited image. **Fast** is a high-quality "
                        "Lanczos resize (no model, instant). **AI refine** adds "
                        "detail with a low-denoise SDXL pass run in tiles "
                        "(needs an SDXL / Pony / Illustrious model below).",
                        elem_classes="imagesuite-help")
                    with gr.Row():
                        c["up_scale"] = gr.Radio(
                            ["2x", "4x"], value="2x", label="Scale", scale=1)
                        c["up_mode"] = gr.Radio(
                            ["Fast (Lanczos)", "AI refine"],
                            value="Fast (Lanczos)", label="Mode", scale=2)
                    c["up_model"] = gr.Dropdown(
                        label="AI-refine model (SDXL family)",
                        choices=sdxl_choices or [], value=None)
                    c["up_denoise"] = gr.Slider(
                        0.05, 0.6, value=0.25, step=0.05,
                        label="AI-refine strength (denoise)")
                    c["mod_upscale"] = gr.Button("🔼 Upscale")
                with gr.Row(elem_classes="imagesuite-genrow"):
                    c["mod_save"] = gr.Button("💾 Save to results", variant="primary")
                c["mod_status"] = gr.Markdown("", elem_classes="imagesuite-genstatus")
        return c

    with gr.Row():
        # -- left: inputs --
        with gr.Column(scale=1):
            _prompt_library_block(c, mode)                         # above settings
            c.update(_settings_bar(model_choices, lora_choices, mode,
                                   sdxl_choices=sdxl_choices))
            if mode == "img2img":
                # Init image sits to the RIGHT of the prompt/negative (same row),
                # not as a full-width block below it — groups the inputs and reclaims
                # vertical space.
                with gr.Row():
                    with gr.Column(scale=2):
                        _prompt_block(c)
                    with gr.Column(scale=1, min_width=160):
                        c["input_image"] = gr.Image(
                            label="Init image (double-click to enlarge)",
                            type="filepath", height=200,
                            elem_classes="imagesuite-initthumb")
            else:
                _prompt_block(c)
            # ControlNet (SDXL family, default OFF) — txt2img + img2img only.
            _controlnet_block(c, mode)
            # XYZ sweep / parameter grid (SDXL family, default OFF) — txt2img + img2img.
            _xyz_sweep_block(c, mode)
            # Generate/Abort moved to the right column, directly under the results
            # gallery (see _results_block). The left column is inputs only now.
            c.update(_enhance.build_enhance_sections(mode, sdxl_choices, lora_choices))

        # -- right: results + Generate/Abort --
        with gr.Column(scale=1):
            _results_block(c, mode, send_panel_fn=send_panel_fn)
    return c
