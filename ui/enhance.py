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

        # -- Color / style reference (IP-Adapter) --
        with gr.Accordion("Color Reference", open=False):
            with gr.Row():
                c["color_ref"] = gr.Image(label="Color / style reference", type="filepath",
                                          height=150, elem_classes="imagesuite-refthumb")
                with gr.Column():
                    c["color_scale"] = gr.Slider(0.0, 1.0, value=0.6, step=0.05,
                                                 label="Reference strength")
                    c["color_denoise"] = gr.Slider(0.0, 1.0, value=0.6, step=0.05,
                                                   label="Denoise")
            c["color_run"] = gr.Button("Apply color reference on the selected result", size="sm")
    return c
