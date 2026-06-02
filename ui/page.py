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
from . import enhance as _enhance
from ..core import overlays as _ovcore

SAMPLERS = ["DPM++ 2M", "DPM++ 2M SDE", "DPM++ 3M SDE", "Euler a", "Euler",
            "Heun", "DDIM", "UniPC", "LCM", "default"]
SCHEDULERS = ["", "Karras", "Exponential", "Normal", "SGM Uniform", "Simple"]
# Initial outpaint target sizes (SDXL set); refreshed per model family on change.
_OUTPAINT_SIZES_INIT = ["1024×1024", "1152×896", "896×1152", "1216×832", "832×1216",
                        "1344×768", "768×1344", "1536×640", "640×1536"]

MODES = ("txt2img", "img2img", "inpaint")


def _settings_bar(model_choices, lora_choices, mode):
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
        with gr.Row():
            c["width"] = gr.Slider(256, 2048, value=832, step=64, label="Width")
            c["height"] = gr.Slider(256, 2048, value=1216, step=64, label="Height")
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

    if mode == "inpaint":
        # Mask + inpaint knobs live in their own block, so the generation settings
        # above stay byte-for-byte identical to the other tabs. The SDXL-only knobs
        # are nested + collapsed (native Flux/Z-Image/Qwen ignore them).
        with gr.Accordion("Mask & inpaint", open=True, elem_classes="imagesuite-acc"):
            c["mask_mode"] = gr.Radio(
                ["Inpaint masked", "Inpaint not masked"],
                value="Inpaint masked", label="Mask mode")
            c["feather"] = gr.Slider(0, 64, value=4, step=1, label="Mask blur (px)")
            with gr.Accordion("SDXL inpaint options", open=False,
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
    return c


def _prompt_block(c):
    """Prompt + negative + Qwen-abliterated Enhance buttons (wired in plugin.py)."""
    c["pos"] = gr.Textbox(label="Prompt", lines=3, placeholder="Describe the image…")
    c["neg"] = gr.Textbox(label="Negative prompt", lines=2,
                          placeholder="low quality, blurry…")
    with gr.Row():
        c["enhance_pos"] = gr.Button("✨ Enhance prompt")
        c["enhance_neg"] = gr.Button("✨ Enhance negative")
        c["interrogate"] = gr.Button("🔍 Interrogate image")


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


def _results_block(c, mode):
    """The shared results column: gallery (selection = the picked result), send-to
    row, save. No separate 'Selected' box — a clicked gallery item IS the selection.
    The side-by-side tabs get a taller gallery so the Save As button lands level
    with the Generate button in the left column (tune the heights to taste)."""
    gallery_h = {"txt2img": 520, "img2img": 620}.get(mode, 460)
    # NOTE: no preview=True. It auto-opens/selects the first item the moment the
    # gallery is populated, which fires a select round-trip on every generation —
    # and Gradio then runs check_all_files_in_cache on the native backend's output
    # path (outside the cache dir) → "File … is not in the cache folder". The
    # working sibling plugin (CharLab) uses a plain grid for the same reason.
    c["gallery"] = gr.Gallery(label="Results", columns=2, height=gallery_h,
                              elem_classes="imagesuite-gallery", object_fit="contain")
    c["picked"] = gr.State(None)  # path of the selected (clicked) result
    gr.Markdown("**Send selected result to →**")
    with gr.Row(elem_classes="imagesuite-sendrow"):
        # Txt2Img/Img2Img hide their own-page button; MultiCanvas keeps it as
        # "Back to canvas" so the inpainted result can be re-loaded for another pass.
        c["to_t2i"] = gr.Button("Txt2Img", visible=mode != "txt2img")
        c["to_i2i"] = gr.Button("Img2Img", visible=mode != "img2img")
        c["to_inp"] = gr.Button("↻ Back to canvas" if mode == "inpaint"
                                else "MultiCanvas")
        c["to_i2v"] = gr.Button("Img2Vid")
    c["save"] = gr.DownloadButton("Save As…", elem_classes="imagesuite-savebtn")


def build_page(mode, model_choices=None, lora_choices=None, sdxl_choices=None):
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
                _overlays_strip_block(c)                           # folder picker
                _touchup_block(c)                                  # under the canvas
                c.update(_enhance.build_enhance_sections(mode, sdxl_choices, lora_choices))  # under the canvas
            with gr.Column(scale=2):
                c.update(_settings_bar(model_choices, lora_choices, mode))
                _prompt_block(c)
                with gr.Row(elem_classes="imagesuite-genrow"):
                    c["generate"] = gr.Button("Inpaint", variant="primary", scale=4)
                    c["abort"] = gr.Button("⛔ Abort", variant="stop", scale=1)
                c["gen_status"] = gr.Markdown("", elem_classes="imagesuite-genstatus")
                _results_block(c, mode)
        return c

    with gr.Row():
        # -- left: inputs --
        with gr.Column(scale=1):
            c.update(_settings_bar(model_choices, lora_choices, mode))
            _prompt_block(c)
            if mode == "img2img":
                c["input_image"] = gr.Image(
                    label="Init image (double-click to enlarge)", type="filepath",
                    height=200, elem_classes="imagesuite-initthumb")
            with gr.Row(elem_classes="imagesuite-genrow"):
                c["generate"] = gr.Button(
                    {"txt2img": "Generate", "img2img": "Reimagine (img2img)"}[mode],
                    variant="primary", scale=4)
                c["abort"] = gr.Button("⛔ Abort", variant="stop", scale=1)
            c["gen_status"] = gr.Markdown("", elem_classes="imagesuite-genstatus")
            c.update(_enhance.build_enhance_sections(mode, sdxl_choices, lora_choices))  # under the settings

        # -- right: results --
        with gr.Column(scale=1):
            _results_block(c, mode)
    return c
