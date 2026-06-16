"""Collapsible post-process / enhancement sections shared by every page.

ADetailer (face + body, each with its own prompts), Face Swap, Body Swap and
Color Reference. Each section has a "Run on the selected result" button (plugin.py wires
it). Backends are CharLab's proven gen_sd / faceswap functions. Reference images
use a small square thumbnail.

build_enhance_sections(mode, sdxl_choices) returns a flat dict of components,
keys prefixed by feature (adetf_/adetb_/face_/body_/color_).
"""
from __future__ import annotations

import gradio as gr

_FACE_ENHANCERS = ["", "gfpgan", "codeformer"]


def build_enhance_sections(mode, sdxl_choices=None, lora_choices=None):
    c = {}
    with gr.Accordion("Enhancements (post-process)", open=False,
                      elem_classes="imagesuite-acc"):
        gr.Markdown("Applied to the **selected** result (click one in the gallery; "
                    "defaults to the first) with each section's Run "
                    "button. Face/body/ADetailer need an SDXL/Pony/Illustrious "
                    "model + the weights from Settings → Models.",
                    elem_classes="imagesuite-help")

        # -- ADetailer: Face + Body, each its own pos/neg --
        with gr.Accordion("ADetailer — detail refine", open=False):
            with gr.Tab("Face"):
                c["adetf_pos"] = gr.Textbox(label="Face prompt", lines=2,
                                            placeholder="detailed face, sharp eyes…")
                c["adetf_neg"] = gr.Textbox(label="Face negative", lines=1)
                c["adetf_run"] = gr.Button("Run face ADetailer on the selected result", size="sm")
            with gr.Tab("Body"):
                c["adetb_pos"] = gr.Textbox(label="Body prompt", lines=2,
                                            placeholder="detailed skin, anatomy…")
                c["adetb_neg"] = gr.Textbox(label="Body negative", lines=1)
                c["adetb_run"] = gr.Button("Run body ADetailer on the selected result", size="sm")

        # -- Face swap --
        with gr.Accordion("Face Swap", open=False):
            with gr.Row():
                c["face_ref"] = gr.Image(label="Reference face", type="filepath",
                                         height=150, elem_classes="imagesuite-refthumb")
                with gr.Column():
                    c["face_enhancer"] = gr.Dropdown(_FACE_ENHANCERS, value="",
                                                     label="Face enhancer")
                    c["face_blend"] = gr.Slider(0.0, 1.0, value=0.8, step=0.05,
                                                label="Blend ratio")
                    c["face_strength"] = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                                                   label="Enhancer strength")
            c["face_run"] = gr.Button("Run face swap on the selected result", size="sm")

        # -- Body swap (SD-family) — copies skin tone + texture, head preserved --
        with gr.Accordion("Body Swap", open=False):
            gr.Markdown("Transfers the reference body's **skin tone & texture** onto "
                        "the subject (face & hair preserved). Runs on the SDXL model "
                        "picked below — independent of the page's model — so it works "
                        "even when generating with Flux / Z-Image. Needs the body-swap "
                        "weights from Settings → Models.", elem_classes="imagesuite-help")
            with gr.Row():
                c["body_ref"] = gr.Image(label="Reference body", type="filepath",
                                         height=150, elem_classes="imagesuite-refthumb")
                with gr.Column():
                    c["body_model"] = gr.Dropdown(
                        label="SDXL / Pony / Illustrious model (body swap runs on this)",
                        choices=sdxl_choices or [])
                    # Body swap runs on ITS OWN SDXL model above (not the page model),
                    # so it gets its own SDXL-family LoRA picker.
                    c["body_loras"] = gr.Dropdown(
                        label="LoRAs for the body-swap model", multiselect=True,
                        choices=lora_choices or [])
                    c["body_lora_mult"] = gr.Textbox(label="LoRA multipliers",
                                                     placeholder="0.8, 1.0")
                    c["body_run"] = gr.Button("Run body swap on the selected result", size="sm")

        # -- Replace Person (pose-copy ControlNet + IP-Adapter FaceID) --
        # Distinct from Body Swap: that copies only skin/texture and keeps the
        # original head; this regenerates the WHOLE person (head included) wearing
        # the reference person's identity, re-posed to match the base.
        with gr.Accordion("Replace Person (pose + identity)", open=False):
            gr.Markdown("Replaces the **entire person** with the reference person, "
                        "keeping the base's **pose** (copied via OpenPose ControlNet) "
                        "and transferring the reference's **face identity** "
                        "(IP-Adapter FaceID). Runs on the SDXL model picked below — "
                        "independent of the page's model. Needs a clear reference face "
                        "plus the OpenPose ControlNet + FaceID weights from "
                        "Settings → Models.", elem_classes="imagesuite-help")
            with gr.Row():
                c["rep_ref"] = gr.Image(label="Reference person (face visible)",
                                        type="filepath", height=150,
                                        elem_classes="imagesuite-refthumb")
                with gr.Column():
                    c["rep_model"] = gr.Dropdown(
                        label="SDXL / Pony / Illustrious model (replace runs on this)",
                        choices=sdxl_choices or [])
                    c["rep_variant"] = gr.Dropdown(
                        choices=[("FaceID-Plus-v2 (identity, recommended)", "faceid_plus"),
                                 ("FaceID", "faceid")],
                        value="faceid_plus", label="FaceID variant")
                    c["rep_ip_scale"] = gr.Slider(0.0, 1.0, value=0.7, step=0.05,
                                                  label="Identity strength")
                    c["rep_cn_strength"] = gr.Slider(0.0, 1.5, value=0.7, step=0.05,
                                                     label="Pose strength")
                    c["rep_denoise"] = gr.Slider(0.0, 1.0, value=0.85, step=0.05,
                                                 label="Denoise")
            with gr.Row():
                c["rep_pos"] = gr.Textbox(label="Prompt (optional)", lines=1,
                                          placeholder="describe clothing / scene…")
                c["rep_neg"] = gr.Textbox(label="Negative (optional)", lines=1)
            with gr.Row():
                # Replace runs on ITS OWN SDXL model (not the page model) → own LoRAs.
                c["rep_loras"] = gr.Dropdown(
                    label="LoRAs for the replace model", multiselect=True,
                    choices=lora_choices or [])
                c["rep_lora_mult"] = gr.Textbox(label="LoRA multipliers",
                                                placeholder="0.8, 1.0")
            c["rep_run"] = gr.Button("Replace person on the selected result", size="sm")

        # -- Color / style reference (IP-Adapter) --
        with gr.Accordion("Color Reference", open=False):
            with gr.Row():
                c["color_ref"] = gr.Image(label="Color / style reference", type="filepath",
                                          height=150, elem_classes="imagesuite-refthumb")
                with gr.Column():
                    # Plus = look/texture only (default, current behaviour). FaceID
                    # variants transfer the reference face's *identity* via InsightFace
                    # embeddings — need a clear face + the FaceID weights (Settings → Models).
                    c["color_variant"] = gr.Dropdown(
                        choices=[("Plus (look)", "plus"),
                                 ("FaceID", "faceid"),
                                 ("FaceID-Plus-v2 (identity)", "faceid_plus")],
                        value="plus", label="IP-Adapter variant")
                    c["color_scale"] = gr.Slider(0.0, 1.0, value=0.6, step=0.05,
                                                 label="Reference strength")
                    c["color_denoise"] = gr.Slider(0.0, 1.0, value=0.6, step=0.05,
                                                   label="Denoise")
            c["color_run"] = gr.Button("Apply color reference on the selected result", size="sm")

        # -- Batch apply over a folder or the current results --
        # Opt-in: runs ONE of the passes above over many images at once, reusing
        # that section's settings (reference image, prompts, model, LoRAs…). Default
        # closed and nothing here touches the single-image flow above.
        _build_batch(c)

        _build_compare(c)
    return c


_BATCH_OPS = [
    ("Face swap (uses Face Swap settings above)", "face"),
    ("ADetailer — face (uses ADetailer ▸ Face above)", "adet_face"),
    ("ADetailer — body (uses ADetailer ▸ Body above)", "adet_body"),
    ("Colour reference (uses Color Reference above)", "color"),
    ("Upscale (Lanczos, no model)", "upscale"),
]


def _build_batch(c):
    """Run one enhancement pass over a whole folder or the current gallery.

    Reuses the per-section settings already on screen (the Face-Swap reference,
    the ADetailer prompts, the page model + LoRAs, the Colour reference + sliders),
    so there's nothing new to configure — pick a source, pick an op, run. Outputs
    land in a timestamped batch subfolder; per-image failures are skipped so one
    bad file never aborts the run. plugin.py wires ``batch_run``."""
    with gr.Accordion("Batch apply (folder / all results)", open=False,
                      elem_classes="imagesuite-acc"):
        gr.Markdown("Run **one** of the passes above over **many** images at once. "
                    "Each image reuses that section's settings (reference image, "
                    "prompts, model, LoRAs). Outputs are written to a new "
                    "`batch_…` subfolder and the count is reported below. "
                    "A bad image is skipped, not fatal.",
                    elem_classes="imagesuite-help")
        c["batch_op"] = gr.Dropdown(choices=_BATCH_OPS, value="face",
                                    label="Operation")
        c["batch_source"] = gr.Radio(
            choices=[("A folder of images", "folder"),
                     ("All current results (the gallery)", "gallery")],
            value="folder", label="Source")
        c["batch_folder"] = gr.Textbox(
            label="Source folder (full path)",
            placeholder="/path/to/images — used when Source = A folder")
        c["batch_upscale"] = gr.Radio(
            choices=[("2x", "2"), ("4x", "4")], value="2",
            label="Upscale factor (Upscale op only)")
        c["batch_run"] = gr.Button("Run batch", size="sm", variant="primary")
        c["batch_status"] = gr.Markdown("", elem_classes="imagesuite-help")


def _build_compare(c):
    """Before/after compare + Accept/Revert for the destructive enhancement passes.

    Hidden until a pass runs (so the normal gallery flow is untouched when unused).
    Uses gr.ImageSlider where the running Gradio offers it (a single drag-to-compare
    widget), else falls back to two side-by-side gr.Image previews. plugin.py reads
    ``cmp_kind`` to know which it built, fills ``cmp_before``/``cmp_after`` (the
    original + enhanced paths) on every pass, and wires Accept (keep) / Revert
    (restore the original into the gallery)."""
    # original ("before") + enhanced ("after") paths held for the Revert handler.
    c["cmp_before"] = gr.State(None)
    c["cmp_after"] = gr.State(None)
    with gr.Accordion("Before / after — Accept or Revert", open=True,
                      visible=False, elem_classes="imagesuite-acc") as cmp_acc:
        c["cmp_panel"] = cmp_acc
        gr.Markdown("Drag the handle to compare the **original** (left) with the "
                    "**enhanced** result (right). The enhanced result is already in "
                    "the gallery — click **Revert** to put the original back, or "
                    "**Accept** to keep it and close this.",
                    elem_classes="imagesuite-help")
        if hasattr(gr, "ImageSlider"):
            c["cmp_kind"] = "slider"
            c["cmp_slider"] = gr.ImageSlider(label="Original ↔ Enhanced",
                                             type="filepath", height=420,
                                             show_download_button=False,
                                             interactive=False)
        else:  # older Gradio without ImageSlider → two side-by-side previews
            c["cmp_kind"] = "pair"
            with gr.Row():
                c["cmp_before_img"] = gr.Image(label="Original", type="filepath",
                                               height=360, interactive=False)
                c["cmp_after_img"] = gr.Image(label="Enhanced", type="filepath",
                                              height=360, interactive=False)
        with gr.Row():
            c["cmp_accept"] = gr.Button("✅ Accept (keep enhanced)", size="sm",
                                        variant="primary")
            c["cmp_revert"] = gr.Button("↩ Revert (restore original)", size="sm",
                                        variant="stop")
