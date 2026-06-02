"""The Overlays library tab — a Gradio-native file manager for overlay images.

Folders + thumbnails with full CRUD (create/delete folders, upload, rename, move,
delete, preview). The Overlays strip beneath the MultiCanvas canvas is a
read-only view of the same library. Wiring lives in plugin.py.
"""
from __future__ import annotations

import gradio as gr

from ..core import overlays as _ov


def build_overlays_panel():
    c = {}
    gr.Markdown(
        "Your **overlay library** — transparent PNGs, frames, stickers, watermarks "
        "(any image). Organise into folders here; they show up in the **Overlays** "
        "strip beneath the **MultiCanvas** canvas, where you can drag one onto the "
        "image as a new layer.", elem_classes="imagesuite-help")

    folders = _ov.list_folders()
    cur = folders[0] if folders else _ov.ROOT_LABEL

    # -- Folders: pick/delete on one row, create on the next. Each input pairs
    #    with its own button (input scale 3 + button scale 1) like the rest of
    #    the app, so nothing floats. --
    with gr.Accordion("Folders", open=True, elem_classes="imagesuite-acc"):
        with gr.Row():
            c["folder"] = gr.Dropdown(label="Folder", choices=folders, value=cur, scale=3)
            c["delete_folder"] = gr.Button("🗑 Delete folder", variant="stop", scale=1)
        with gr.Row():
            c["new_folder"] = gr.Textbox(label="New folder", placeholder="e.g. frames",
                                         scale=3)
            c["create_folder"] = gr.Button("➕ Create", scale=1)

    # -- Add images to the selected folder --
    with gr.Accordion("Add images", open=True, elem_classes="imagesuite-acc"):
        with gr.Row():
            c["upload"] = gr.File(label="Images (PNG / JPG / WEBP / GIF / BMP)",
                                  file_count="multiple", file_types=["image"], scale=3)
            c["upload_btn"] = gr.Button("⬆ Add to folder", scale=1)

    # -- The library itself --
    c["gallery"] = gr.Gallery(label="Overlays — click to select (click again to enlarge)",
                              columns=6, height=360, object_fit="contain",
                              value=_ov.list_images(cur), elem_classes="imagesuite-gallery")
    c["selected"] = gr.State(None)  # basename of the selected overlay

    # -- Actions on the selected overlay: preview left, paired action rows right. --
    with gr.Accordion("Selected overlay", open=True, elem_classes="imagesuite-acc"):
        with gr.Row():
            c["preview"] = gr.Image(label="Preview", height=200,
                                    interactive=False, scale=1)
            with gr.Column(scale=2):
                with gr.Row():
                    c["rename_to"] = gr.Textbox(label="Rename to",
                                                placeholder="new-name.png", scale=3)
                    c["rename_btn"] = gr.Button("✎ Rename", scale=1)
                with gr.Row():
                    c["move_to"] = gr.Dropdown(label="Move to folder",
                                               choices=folders, value=cur, scale=3)
                    c["move_btn"] = gr.Button("➡ Move", scale=1)
                c["delete_btn"] = gr.Button("🗑 Delete selected", variant="stop")

    c["status"] = gr.Markdown("", elem_classes="imagesuite-help")
    return c
