"""Settings + Models sub-tab.

Two jobs:
  * Directories — view/repoint the output, SDXL-checkpoint, SDXL-LoRA and
    face-model dirs (defaults shared with Replicant Character Lab). The shared
    dirs persist to .orphansuite.json; only the outputs dir persists to
    .imagesuite.json. A Rescan refreshes every page's model + LoRA dropdowns.
  * Models — show the status of the optional face / ADetailer weights and
    download them on demand (nothing auto-downloads).

build_settings_panel() returns a components dict; plugin.py wires the actions
(it owns paths + the native-model list + the per-page dropdowns to refresh).
"""
from __future__ import annotations

import gradio as gr

from ..core import models, paths, presets, projects
from .page import SAMPLERS, SCHEDULERS


_VRAM_ALL = "All models"
_VRAM_LOW = "Low-VRAM only (light/quantized native models)"


def build_settings_panel(native_dl_choices=None):
    c = {}
    with gr.Column():
        # -- VRAM / model-list radio (global; filters native models app-wide) --
        c["vram_mode"] = gr.Radio(
            choices=[_VRAM_ALL, _VRAM_LOW], label="Native model list",
            value=_VRAM_LOW if paths.low_vram_only() else _VRAM_ALL,
            info="Wan2GP already runs native models under a low-VRAM/int8 profile "
                 "(global, set in the Configuration tab). This only trims the model "
                 "dropdown: 'Low-VRAM only' hides heavy native models (e.g. 32B Flux 2 "
                 "Dev, 20B Qwen) and keeps light ones (Flux 2 Klein 4B, Flux Schnell, "
                 "nvfp4/int4 variants). SDXL/Pony/Illustrious are always shown.")

        # -- OrphanSuite shared resources (collapsible) --
        with gr.Accordion("OrphanSuite — shared models & folders", open=False,
                          elem_classes="imagesuite-acc"):
            gr.Markdown(
                "**Shared across all saintorphan plugins** (Image Suite, Replicant "
                "CharLab, Reel2Reel) via `.orphansuite.json` — set a folder here and "
                "every plugin follows. Point them anywhere you already keep models so "
                "nothing's duplicated.")
            with gr.Row():
                c["sdxl_models_dir"] = gr.Textbox(
                    label="SDXL / Pony / Illustrious checkpoints (shared)",
                    value=str(paths.sdxl_models_dir()))
                c["sdxl_loras_dir"] = gr.Textbox(label="SDXL-family LoRAs (shared)",
                                                 value=str(paths.sdxl_loras_dir()))
            with gr.Row():
                c["models_dir"] = gr.Textbox(
                    label="Face / ADetailer / face-swap weights (shared)",
                    value=str(paths.models_dir()))
                c["outputs_dir"] = gr.Textbox(label="Outputs (this plugin only)",
                                              value=str(paths.outputs_dir()))
            with gr.Row():
                c["save_dirs"] = gr.Button("Save folders", variant="primary")
                c["rescan"] = gr.Button("Rescan models & LoRAs")
            gr.Markdown(
                "**Link an existing folder** — symlink models you already keep "
                "(a1111 / Forge / anywhere on disk) into the shared area. Works with "
                "physical files *or* symlinks and never moves the originals.",
                elem_classes="imagesuite-help")
            with gr.Row():
                c["link_src"] = gr.Textbox(label="Folder to link from",
                                           placeholder="/path/to/your/models", scale=3)
                c["link_target"] = gr.Dropdown(
                    label="Into", value="sdxl_models",
                    # Resolve to the exact dir each loader scans (incl. the
                    # face/body/birefnet subdirs) — see paths.link_target_dir.
                    choices=[("SDXL checkpoints", "sdxl_models"),
                             ("SDXL LoRAs", "sdxl_loras"),
                             ("Face / swap weights", "face"),
                             ("ADetailer / person-seg (body)", "body"),
                             ("BiRefNet (body-swap seg)", "birefnet"),
                             ("InsightFace buffalo_l (face detect)", "buffalo_l")],
                    scale=2)
                c["link_btn"] = gr.Button("🔗 Link", scale=1)
            c["dirs_status"] = gr.Markdown("")

        # -- Storage: reclaim disk from dead (orphaned) generations --
        with gr.Accordion("Storage — flush dead generations", open=False,
                          elem_classes="imagesuite-acc"):
            gr.Markdown(
                "Every generation lands in the output cache (txt2img / img2img / "
                "MultiCanvas). **Saved projects keep their own copies** and the "
                "on-screen galleries restore independently — so flushing only removes "
                "*orphaned* generations (not in any project, not currently shown). "
                "Nothing scans automatically; press **Rescan** to measure.",
                elem_classes="imagesuite-help")
            _ff, _fb = projects.orphaned_outputs()
            with gr.Row():
                c["flush_size"] = gr.Markdown(projects.flush_label(len(_ff), _fb))
                c["flush_rescan"] = gr.Button("↻ Rescan", scale=0)
            c["flush_btn"] = gr.Button("🗑 Flush Outputs", variant="stop")
            c["flush_status"] = gr.Markdown("", elem_classes="imagesuite-help")

        # -- Default Generation Values (per family; shared via .orphansuite.json) --
        with gr.Accordion("Default Generation Values (per family)", open=False,
                          elem_classes="imagesuite-acc"):
            gr.Markdown(
                "Recommended cfg / steps / sampler / scheduler + resolution that "
                "auto-fill **Generation settings** when you pick a model. Edit and "
                "**Save** to set your own defaults (shared across all saintorphan "
                "plugins via `.orphansuite.json`); **Reset** restores factory values. "
                "For Flux / Z-Image / Qwen the model's own steps/CFG still take "
                "precedence unless you save an override here.",
                elem_classes="imagesuite-help")
            _f0 = presets.FAMILIES[0]
            c["gd_fam"] = gr.Dropdown(label="Model family", choices=presets.FAMILIES,
                                      value=_f0)
            _e0 = presets.effective(_f0)
            with gr.Row():
                c["gd_steps"] = gr.Slider(1, 60, value=_e0["steps"], step=1, label="Steps")
                c["gd_cfg"] = gr.Slider(1.0, 15.0, value=_e0["cfg"], step=0.5, label="CFG")
                c["gd_clip"] = gr.Slider(1, 4, value=_e0["clip_skip"], step=1,
                                         label="Clip skip")
            with gr.Row():
                c["gd_sampler"] = gr.Dropdown(SAMPLERS, value=_e0["sampler"], label="Sampler")
                c["gd_scheduler"] = gr.Dropdown(SCHEDULERS, value=_e0["scheduler"],
                                                label="Scheduler")
            with gr.Row():
                c["gd_width"] = gr.Slider(256, 2048, value=_e0["width"], step=64, label="Width")
                c["gd_height"] = gr.Slider(256, 2048, value=_e0["height"], step=64, label="Height")
            with gr.Row():
                c["gd_save"] = gr.Button("Save as my default", variant="primary")
                c["gd_reset"] = gr.Button("Reset to factory")
            c["gd_status"] = gr.Markdown(
                f"Showing **{_f0}** "
                + ("(your override)." if presets.has_override(_f0) else "(factory)."),
                elem_classes="imagesuite-help")

        # -- Models (collapsible): native image models + optional face weights --
        with gr.Accordion("Models", open=False, elem_classes="imagesuite-acc"):
            gr.Markdown("### Native image models\n"
                        "Flux / Z-Image / Qwen weights, downloaded on demand (the "
                        "list follows the Low-VRAM filter above). Pick one and press "
                        "Download to fetch it ahead of time instead of waiting on the "
                        "first generation — nothing downloads without a button press.")
            with gr.Row():
                c["native_key"] = gr.Dropdown(
                    label="Native model", choices=native_dl_choices or [], scale=3)
                c["native_download"] = gr.Button("Download", scale=1)
            c["native_log"] = gr.Markdown("")

            gr.Markdown(
                "### Face / body / ADetailer helper weights\n"
                "Only needed for the optional swap + detail passes. **Scanned on "
                "your machine** — nothing runs until you press **Scan**. Each row "
                "shows whether a weight is **on disk**, **found elsewhere** (→ Link "
                "it in) or **not downloaded** (→ Download).")
            with gr.Row():
                c["scan_btn"] = gr.Button("🔍 Scan for models", variant="primary",
                                          scale=1)
                c["scan_search"] = gr.Textbox(
                    label="Also search this folder (optional)",
                    placeholder="/your/a1111/models — find weights you already have",
                    scale=3)
            c["scan_found"] = gr.State({})   # key -> path found elsewhere on disk
            c["scan_status"] = gr.Markdown(
                "_Press **Scan for models** to check what's on disk._",
                elem_classes="imagesuite-help")
            # One row per registry weight, built once; the Scan handler fills each
            # status + enables/greys its Download / Link buttons (see plugin.py).
            c["model_row_keys"] = [m.key for m in models.REGISTRY]
            for m in models.REGISTRY:
                with gr.Row(elem_classes="imagesuite-modelrow"):
                    c[f"m_{m.key}_status"] = gr.Markdown(
                        f"**{m.name}** — _not scanned_")
                    c[f"m_{m.key}_dl"] = gr.Button(
                        "⬇ Download", size="sm", interactive=False,
                        scale=0, min_width=120)
                    c[f"m_{m.key}_link"] = gr.Button(
                        "🔗 Link", size="sm", interactive=False,
                        scale=0, min_width=90)
    return c


def vram_is_low(value) -> bool:
    return value == _VRAM_LOW
