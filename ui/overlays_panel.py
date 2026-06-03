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

    # -- Compact toolbar: pick a folder + create / delete folders on ONE line, so the
    #    library gallery below becomes the centerpiece instead of being shoved down. --
    with gr.Row():
        c["folder"] = gr.Dropdown(label="Folder", choices=folders, value=cur, scale=3)
        c["new_folder"] = gr.Textbox(label="New folder", placeholder="e.g. frames",
                                     scale=2)
        c["create_folder"] = gr.Button("➕ Create", scale=1, min_width=90)
        c["delete_folder"] = gr.Button("🗑 Delete folder", variant="stop", scale=1,
                                        min_width=120)

    # -- The library itself: the big, prominent centerpiece. Keeps .imagesuite-gallery
    #    (so the scroll fix applies) but NOT .imagesuite-results — its dense 6-column
    #    thumbnails stay compact rather than getting the results' 180px min-row. --
    c["gallery"] = gr.Gallery(
        label="Overlays — click to select (click again to enlarge)",
        columns=6, height=480, object_fit="contain",
        value=_ov.list_images(cur), elem_classes="imagesuite-gallery")
    c["selected"] = gr.State(None)  # basename of the selected overlay

    # -- Actions on the selected overlay: one compact row under the gallery
    #    (preview on the left, paired rename / move / delete on the right). --
    with gr.Row():
        c["preview"] = gr.Image(label="Selected", height=128,
                                interactive=False, scale=1)
        with gr.Column(scale=3):
            with gr.Row():
                c["rename_to"] = gr.Textbox(label="Rename to",
                                            placeholder="new-name.png", scale=3)
                c["rename_btn"] = gr.Button("✎ Rename", scale=1, min_width=90)
            with gr.Row():
                c["move_to"] = gr.Dropdown(label="Move to folder",
                                           choices=folders, value=cur, scale=3)
                c["move_btn"] = gr.Button("➡ Move", scale=1, min_width=80)
                c["delete_btn"] = gr.Button("🗑 Delete", variant="stop", scale=1,
                                            min_width=90)

    # -- Add images: tucked into a collapsed accordion (the file drop-zone is bulky
    #    and rarely the first thing you reach for). --
    with gr.Accordion("➕ Add images to this folder", open=False,
                      elem_classes="imagesuite-acc"):
        with gr.Row():
            c["upload"] = gr.File(label="Images (PNG / JPG / WEBP / GIF / BMP)",
                                  file_count="multiple", file_types=["image"], scale=3)
            c["upload_btn"] = gr.Button("⬆ Add to folder", scale=1)

    c["status"] = gr.Markdown("", elem_classes="imagesuite-help")
    return c
