"""Assemble the Image Suite tab: three sub-tabs (Txt2Img / Img2Img / MultiCanvas).

Returns a dict:
    {
      "subtabs": gr.Tabs,             # so send-to-page can switch the active sub-tab
      "tab_ids": {"txt2img": ..., "img2img": ..., "inpaint": ...},
      "pages": {"txt2img": {...}, "img2img": {...}, "inpaint": {...}},
    }
"""
from __future__ import annotations

import gradio as gr

from . import page
from .overlays_panel import build_overlays_panel
from .settings_panel import build_settings_panel

_TAB_IDS = {"txt2img": "imagesuite-t2i", "img2img": "imagesuite-i2i",
            "inpaint": "imagesuite-inp", "modify": "imagesuite-mod",
            "overlays": "imagesuite-overlays", "settings": "imagesuite-settings"}
_LABELS = {"txt2img": "Txt2Img", "img2img": "Img2Img", "inpaint": "MultiCanvas",
           "modify": "Modify"}


def build_suite(model_choices_by_mode=None, lora_choices=None, native_dl_choices=None,
                sdxl_choices=None, send_panel_fn=None):
    mc = model_choices_by_mode or {}
    pages = {}
    with gr.Tabs() as subtabs:
        for mode in page.MODES:
            with gr.Tab(_LABELS[mode], id=_TAB_IDS[mode]):
                pages[mode] = page.build_page(mode, mc.get(mode), lora_choices,
                                              sdxl_choices, send_panel_fn=send_panel_fn)
        with gr.Tab("🖼 Overlays", id=_TAB_IDS["overlays"]):
            overlays = build_overlays_panel()
        with gr.Tab("⚙ Settings", id=_TAB_IDS["settings"]):
            settings = build_settings_panel(native_dl_choices=native_dl_choices)
    return {"subtabs": subtabs, "tab_ids": _TAB_IDS, "pages": pages,
            "overlays": overlays, "settings": settings}
