"""Image Suite — a Wan2GP plugin.

A three-in-one image workbench rendered as one main-webui tab with three
sub-tabs: Txt2Img, Img2Img and MultiCanvas. Each page supports the native
Wan2GP image models (Flux / Z-Image / Qwen) and the SDXL lineup (SDXL / Pony /
Illustrious) — the latter via a self-contained diffusers backend bundled in
``core/sd`` (no external checkout). Any result can be pushed to the other two pages,
handed to the Video Generator (Img2Vid), or downloaded (Save As).

NOTE: not an official plugin. Distribute via the plugin-manager "add from GitHub
URL" flow; do not add to the bundled plugins.json without dbm's approval.
"""
from __future__ import annotations

import functools
import os
import time
import traceback

import gradio as gr

from shared.utils.plugins import WAN2GPPlugin

try:  # GPU arbitration with the main Video Generator (see wan2gp-sample)
    from shared.utils.process_locks import (acquire_GPU_ressources,
                                            any_GPU_process_running,
                                            release_GPU_ressources)
    try:
        from shared.utils.process_locks import set_main_generation_running
    except Exception:
        set_main_generation_running = None
    _HAVE_LOCKS = True
except Exception:  # pragma: no cover
    _HAVE_LOCKS = False
    set_main_generation_running = None

from .core import discovery, gen_sd, inbox, models, paths, presets, projects
from .ui import canvas, contextmenu, logo, modify_canvas, suite
from .ui.styles import CSS, LIGHTBOX_HTML

PLUGIN_ID = "ImageSuite"
PLUGIN_NAME = "Image Suite"

# Hidden Textbox the shared right-click menu relays into ({a:action,s:src,t}).
CTX_RELAY = contextmenu.RELAY_ID


class ImageSuite(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = PLUGIN_NAME
        self.version = "0.2.1"
        self.description = ("Txt2Img / Img2Img / MultiCanvas workbench for Flux, "
                            "Z-Image and the SDXL lineup, with send-to-page, "
                            "send-to-Img2Vid and Save As.")

    # -- lifecycle ----------------------------------------------------------
    def setup_ui(self):
        try:
            paths.ensure_dirs()
        except Exception:
            traceback.print_exc()

        self.request_component("state")
        self.request_component("main_tabs")
        self.request_component("refresh_form_trigger")
        self.request_component("output")  # main preview gallery (Send current frame)
        self.request_global("get_current_model_settings")
        self.request_global("get_default_settings")  # native image generation
        self.request_global("get_model_def")          # per-model capability flags
        self.request_global("models_def")             # list native image models
        # Native-model manual downloads (Models panel) — mirror wgp's load path
        # WITHOUT loading into VRAM. These are the same globals wgp uses.
        self.request_global("download_models")
        self.request_global("get_model_filename")
        self.request_global("get_model_recursive_prop")
        self.request_global("transformer_quantization")
        self.request_global("transformer_dtype_policy")
        self.request_global("text_encoder_quantization")
        # Unloads Wan2GP's own (native) model to free VRAM before our SDXL loads.
        self.request_global("release_model")
        self.request_global("get_lora_dir")  # per-native-model LoRA folder
        self.request_global("get_state_model_type")  # preserve the user's video-gen pick
        self.request_global("exec_prompt_enhancer_engine")  # Qwen prompt enhancer

        self.add_tab(tab_id=PLUGIN_ID, label=PLUGIN_NAME,
                     component_constructor=self.create_ui)
        # Inject our own "Send current frame" section under the preview gallery —
        # but ONLY when the standalone SendTo plugin isn't installed. When SendTo is
        # present it owns that panel (and routes here via our sendto.json inbox), so
        # we skip ours to avoid a duplicate. Either way we still drain the inbox in
        # on_tab_select, so frames sent to us land regardless.
        if not self._sendto_installed():
            self.insert_after("gallery_tabs", self._build_send_frame_section)

    @staticmethod
    def _sendto_installed() -> bool:
        """True if the standalone SendTo-Wan2GP plugin folder is present (host runs
        from the repo root with ``plugins`` on sys.path)."""
        base = os.path.abspath("plugins")
        return os.path.isdir(os.path.join(base, "SendTo-Wan2GP"))

    def _sendto_panel_fn(self):
        """fn(picked_state) -> the unified SendTo picker for CROSS-destination sends
        (img2vid + other plugins). None when SendTo isn't installed — then each tab's
        Img2Vid button shows as the fallback. Same-plugin moves stay on the direct
        buttons (Txt2Img/Img2Img/MultiCanvas/Modify): SendTo's inbox can't do a
        same-tab hop, so we exclude our own tab from the panel."""
        try:
            from sendto.embed import build_send_panel
        except Exception:
            return None

        def _fn(picked):
            try:
                return build_send_panel(
                    self.state, self.main_tabs, [picked], (lambda p: p),
                    refresh_trigger=getattr(self, "refresh_form_trigger", None),
                    get_settings=getattr(self, "get_current_model_settings", None),
                    exclude_tab=PLUGIN_ID, include_save=False, title="📤 Send to")
            except Exception:
                traceback.print_exc()
                return None
        return _fn

    def on_tab_select(self, state: dict):
        # Warn if the GPU looks busy, but DON'T bounce out — a stale lock would
        # otherwise trap the user out of the tab (and away from ⛔ Abort, which
        # clears it). The gen handlers guard the GPU themselves.
        if _HAVE_LOCKS and any_GPU_process_running(state, PLUGIN_ID):
            gr.Warning("A generation appears to be running — if it's actually stuck, "
                       "hit ⛔ Abort to clear it.")
        # Drain any frames handed to us by SendTo (or another sender using our
        # sendto.json inbox) and load them into the right slot.
        return self._drain_inbox(state)

    def _drain_inbox(self, state):
        """on_tab_select handler: pull queued frames from state['imagesuite_inbox']
        and route each to its slot. Returns updates for self.on_tab_outputs:
        [img2img input, inpaint bg_bridge, modify bg_bridge, subtabs]."""
        noop = gr.update()
        out = [noop, noop, noop, noop]
        ui = getattr(self, "_ui", None)
        if not ui:
            return out
        try:
            items = inbox.drain(state)
        except Exception:
            traceback.print_exc()
            return out
        if not items:
            return out
        pages, tab_ids = ui.get("pages", {}), ui.get("tab_ids", {})
        last_slot = None
        for it in items:
            path = (it or {}).get("path")
            slot = (it or {}).get("slot")
            if not path or not os.path.exists(path):
                continue
            try:
                if slot == "img2img" and "img2img" in pages:
                    out[0] = gr.update(value=path)
                elif slot == "inpaint" and "inpaint" in pages:
                    out[1] = canvas.bg_bridge_html(self._file_to_dataurl(path),
                                                   "inpaint", nonce=time.time())
                elif slot == "modify" and "modify" in pages:
                    out[2] = modify_canvas.modify_bg_bridge_html(
                        self._file_to_dataurl(path), "modify", nonce=time.time())
                else:
                    continue
                last_slot = slot
            except Exception:
                traceback.print_exc()
        if last_slot is not None and last_slot in tab_ids:
            out[3] = gr.update(selected=tab_ids[last_slot])
        return out

    # -- GPU arbitration ----------------------------------------------------
    # Wan2GP's GPU manager only knows about its OWN models (wan_model/offloadobj),
    # not our external SDXL/diffusers pipeline — so we bridge both directions:
    #   acquire: take the lock, then release_model() to free Wan2GP's native model
    #            so our SDXL checkpoint has room.
    #   release: keep our SDXL cached but register _release_all_vram as a GPU-resident
    #            callback (frees the main SD pipe + standalone inpaint/IP pipe +
    #            segmentation + faceswap sessions), so when the main app / another
    #            plugin next acquires the GPU our VRAM is reclaimed automatically
    #            (acquire_main runs the callbacks).
    def _gpu_busy(self, state) -> bool:
        return bool(_HAVE_LOCKS and any_GPU_process_running(state, PLUGIN_ID))

    def _abort(self, state):
        """Stop the current generation — native via the session's cancel(), SD via
        the cooperative abort flag (checked between batch images and mid-image via
        the diffusers step callback), plus the base app's gen abort flag."""
        try:
            gen_sd.request_abort()
        except Exception:
            pass
        try:
            if getattr(self, "_api", None) is not None:
                self._api.cancel()
        except Exception:
            traceback.print_exc()
        # Only clear the host-GLOBAL generation lock when WE own it — otherwise a
        # legitimate Video Generator render (or another plugin's gen) would be
        # cancelled out from under it. Ownership is the recorded lock holder; an
        # absent/None holder means there's no foreign owner to protect, so a stale
        # empty lock is safe to clear.
        gen = state.get("gen") if isinstance(state, dict) else None
        owner = gen.get("main_process_running") if isinstance(gen, dict) else None
        we_own = owner in (None, "", PLUGIN_ID, PLUGIN_NAME)
        # Always set our own abort flag so OUR queued/running gen stops cooperatively.
        try:
            if isinstance(gen, dict):
                gen["abort"] = True
        except Exception:
            pass
        if we_own:
            try:
                if callable(set_main_generation_running):
                    set_main_generation_running(state, False)
            except Exception:
                pass
            try:
                if isinstance(gen, dict):
                    gen["process_status"] = None
                    gen.pop("main_process_running", None)
            except Exception:
                pass
            gr.Info("Abort requested — stopping the current generation (and clearing "
                    "any stuck lock).")
        else:
            gr.Info("Abort requested — stopping Image Suite's work. (A Video Generator "
                    "render appears to own the GPU; its lock was left intact.)")

    def acquire_gpu(self, state):
        if not _HAVE_LOCKS:
            return True
        if any_GPU_process_running(state, PLUGIN_ID):
            # raise (not bare gr.Error) so the user actually sees why nothing ran.
            raise gr.Error("The GPU is busy with another generation — wait for it "
                           "to finish, then try again.")
        acquire_GPU_ressources(state, PLUGIN_ID, PLUGIN_NAME, gr=gr)
        rm = getattr(self, "release_model", None)  # free Wan2GP's native model
        if callable(rm):
            try:
                rm()
            except Exception:
                traceback.print_exc()
        return True

    def _release_all_vram(self):
        """Combined GPU-resident release callback: free EVERY plugin-held GPU
        consumer — the main SD pipe, the standalone inpaint+IP pipe and the
        segmentation model (gen_sd.release_all), plus the InsightFace/ONNX face
        swap sessions — so when the host/another plugin next acquires the GPU,
        none of our VRAM leaks into video generation."""
        try:
            gen_sd.release_all()
        except Exception:
            traceback.print_exc()
        try:
            self._release_faceswap()
        except Exception:
            traceback.print_exc()

    def release_gpu(self, state):
        if not _HAVE_LOCKS:
            return
        try:
            release_GPU_ressources(
                state, PLUGIN_ID, keep_resident=True, process_name=PLUGIN_NAME,
                release_vram_callback=self._release_all_vram, force_release_on_acquire=True)
        except TypeError:  # older process_locks without the keep_resident kwargs
            release_GPU_ressources(state, PLUGIN_ID)
            self._release_all_vram()

    def _release_native_model(self):
        """Evict Wan2GP's native transformer (wan_model/offloadobj) after one of our
        native image gens. The native path borrows the host engine WITHOUT going
        through acquire_gpu (which is what calls release_model on the SD path), so
        without this the Flux/Z-Image/Qwen model stays resident and stacks with a
        later SDXL load -> OOM. Uses the host's own release_model (requested in
        setup_ui); the next native gen reloads from disk (release_model sets
        reload_needed)."""
        rm = getattr(self, "release_model", None)
        if callable(rm):
            try:
                rm()
            except Exception:
                traceback.print_exc()

    # -- preserve the Video Generator's setup across a native gen -----------
    # A native gen borrows Wan2GP's shared engine, which can change the current
    # model + its (large) settings dict. Snapshot the WHOLE dict + model_type
    # before, restore them after, so the user's Video Generator place — model,
    # prompt, sliding windows, anchors, every param — is left exactly as it was.
    def _snapshot_video_state(self, state):
        import copy
        snap = {"model_type": None, "settings": None}
        try:
            getter = getattr(self, "get_state_model_type", None)
            snap["model_type"] = (getter(state) if callable(getter)
                                  else (state.get("model_type") if isinstance(state, dict) else None))
        except Exception:
            pass
        try:
            cur = getattr(self, "get_current_model_settings", None)
            if callable(cur):
                live = cur(state)
                if isinstance(live, dict):
                    snap["settings"] = copy.deepcopy(live)
        except Exception:
            pass
        return snap

    def _restore_video_state(self, state, snap):
        if not snap:
            return
        try:
            mt = snap.get("model_type")
            if mt and isinstance(state, dict):
                state["model_type"] = mt          # key get_state_model_type reads
                if "edit_model_type" in state:
                    state["edit_model_type"] = mt
            settings = snap.get("settings")
            if settings is not None:
                cur = getattr(self, "get_current_model_settings", None)
                if callable(cur):
                    live = cur(state)             # now resolves to the restored model
                    if isinstance(live, dict):
                        live.clear()
                        live.update(settings)
        except Exception:
            traceback.print_exc()

    # -- enhancement post-process (face/body swap, ADetailer, colour ref) ---
    def _face_pipe(self):
        if getattr(self, "_faceswap", None) is None:
            from .core import faceswap
            self._faceswap = faceswap.FaceSwapPipeline(str(paths.models_dir() / "face"))
        return self._faceswap

    def _release_faceswap(self):
        try:
            if getattr(self, "_faceswap", None) is not None:
                self._faceswap.release()
        except Exception:
            pass
        self._faceswap = None
        gen_sd._free_torch()

    @staticmethod
    def _require(keys, what):
        miss = models.missing(keys)
        if miss:
            raise gr.Error(f"{what} needs models you haven't downloaded yet — get them "
                           f"in Settings → Models first: " + ", ".join(miss))

    @staticmethod
    def _sd_ident(model):
        backend, ident = discovery.parse_model_value(model)
        if backend != "sd":
            raise gr.Error("This needs an SDXL / Pony / Illustrious model selected.")
        return ident

    def _save_enh(self, img_or_path, tag):
        """Normalise an enhancement result (PIL or path) → a saved path."""
        from PIL import Image
        out = paths.cache_dir() / "enhance"
        out.mkdir(parents=True, exist_ok=True)
        if isinstance(img_or_path, str):
            return img_or_path
        p = out / f"{tag}_{int(time.time() * 1000)}.png"
        img_or_path.save(p)
        return str(p)

    def _enh_result(self, res):
        if not res:
            raise gr.Error("Enhancement produced no image.")
        return self._gallery_result(res)

    def _enh_faceswap(self, state, picked, ref, enhancer, blend, strength,
                      progress=gr.Progress()):
        if not picked:
            raise gr.Error("Select a result first.")
        if not ref:
            raise gr.Error("Add a reference face image.")
        keys = ["inswapper_128", "buffalo_l"]
        if enhancer and enhancer.lower() != "none":
            keys.append(enhancer.lower())
        self._require(keys, "Face swap")
        gen_sd.release_sd()
        self.acquire_gpu(state)
        try:
            progress(0.3, desc="Face swap…")
            img = self._face_pipe().swap(
                source_path=ref, target_path=picked, enhancer=(enhancer or None),
                blend_ratio=float(blend), enhancer_strength=float(strength))
        finally:
            self._release_faceswap()  # don't leak InsightFace/ONNX VRAM
            self.release_gpu(state)
        return self._enh_result(self._save_enh(img, "face"))

    def _enh_adetailer(self, state, model, picked, pos, neg, detector,
                       loras=None, mult="", progress=gr.Progress()):
        if not picked:
            raise gr.Error("Select a result first.")
        # Clear any stale abort flag from a prior cancelled gen (mirrors _enh_color /
        # _enh_bodyswap) — otherwise the callback can trip _interrupt and discard the
        # result until an unrelated handler clears it.
        gen_sd.clear_abort()
        ident = self._sd_ident(model)
        self._require(["buffalo_l"] if detector == "face" else ["person_yolov8s_seg"],
                      f"{detector.title()} ADetailer")
        self._release_faceswap()
        self.acquire_gpu(state)
        try:
            progress(0.3, desc=f"{detector.title()} ADetailer…")
            res = gen_sd.run_adetailer(ident, picked, pos, neg, "DPM++ 2M", "Karras",
                                       detector=detector,
                                       loras=self._lora_list(loras, mult))
        finally:
            self.release_gpu(state)
        return self._enh_result(res)

    def _enh_bodyswap(self, state, body_model, picked, ref, body_loras=None,
                      body_lora_mult="", progress=gr.Progress()):
        """Skin-tone + texture transfer (head preserved). Runs on the SDXL model
        picked in the Body Swap section — NOT the page model — so it works while
        the page generates with Flux/Z-Image. Settings are hardcoded to CharLab's
        proven defaults (ip_scale 0.8, denoise 0.75, face ADetailer after)."""
        if not picked:
            raise gr.Error("Select a result first.")
        if not ref:
            raise gr.Error("Add a reference body image.")
        if not body_model:
            raise gr.Error("Pick an SDXL / Pony / Illustrious model for the body swap.")
        ident = self._sd_ident(body_model)
        self._require(models.BODY_SWAP_KEYS, "Body swap")
        # The final body-ADetailer refine needs person_yolov8s_seg; it's gated
        # separately from BODY_SWAP_KEYS, so warn (rather than silently no-op the
        # refine) if it isn't downloaded.
        if models.missing(["person_yolov8s_seg"]):
            gr.Warning("ADetailer 'person_yolov8s-seg' isn't downloaded — the final "
                       "body refine pass will be skipped. Get it in Settings → Models "
                       "for a cleaner result.")
        self._release_faceswap()
        self.acquire_gpu(state)
        try:
            progress(0.2, desc="Body swap…")
            res = gen_sd.body_swap(ident, picked, ref, "", "", ip_scale=0.8,
                                   denoise=0.75, adetailer=True,
                                   loras=self._lora_list(body_loras, body_lora_mult),
                                   progress=progress)
        finally:
            self.release_gpu(state)
        return self._enh_result(res)

    def _enh_color(self, state, model, picked, ref, scale, denoise,
                   loras=None, mult="", progress=gr.Progress()):
        if not picked:
            raise gr.Error("Select a result first.")
        if not ref:
            raise gr.Error("Add a colour / style reference.")
        ident = self._sd_ident(model)
        self._require(["ip_adapter"], "Colour reference")
        # Clear any stale abort flag from a prior cancelled gen — otherwise the
        # IP-Adapter callback trips _interrupt on step 1 and discards the result,
        # persistently breaking Colour Reference until an unrelated handler clears it.
        gen_sd.clear_abort()
        from PIL import Image
        base = Image.open(picked).convert("RGB")
        mask = Image.new("L", base.size, 255)  # whole-image IP-Adapter restyle
        self._release_faceswap()
        self.acquire_gpu(state)
        try:
            progress(0.2, desc="Colour reference…")
            res = gen_sd.ip_adapter_inpaint(ident, picked, ref, mask, "", "",
                                            denoise=float(denoise), ip_scale=float(scale),
                                            loras=self._lora_list(loras, mult),
                                            progress=progress)
        finally:
            self.release_gpu(state)
        return self._enh_result(res)

    def _wire_enhance(self, c):
        out = [c["gallery"], c["picked"], c["save"]]
        st, model, picked = self.state, c["model"], c["picked"]
        lo, mu = c["loras"], c["lora_mult"]  # page LoRAs: ADetailer/colour run on the page model
        c["adetf_run"].click(
            lambda s, m, p, pos, neg, l, u: self._enh_adetailer(s, m, p, pos, neg, "face", l, u),
            inputs=[st, model, picked, c["adetf_pos"], c["adetf_neg"], lo, mu], outputs=out)
        c["adetb_run"].click(
            lambda s, m, p, pos, neg, l, u: self._enh_adetailer(s, m, p, pos, neg, "person", l, u),
            inputs=[st, model, picked, c["adetb_pos"], c["adetb_neg"], lo, mu], outputs=out)
        c["face_run"].click(
            self._enh_faceswap,
            inputs=[st, picked, c["face_ref"], c["face_enhancer"], c["face_blend"],
                    c["face_strength"]], outputs=out)
        # Body swap runs on ITS OWN SDXL model → its own LoRA picker (body_loras).
        c["body_run"].click(
            self._enh_bodyswap,
            inputs=[st, c["body_model"], picked, c["body_ref"], c["body_loras"],
                    c["body_lora_mult"]], outputs=out)
        c["color_run"].click(
            self._enh_color,
            inputs=[st, model, picked, c["color_ref"], c["color_scale"],
                    c["color_denoise"], lo, mu], outputs=out)

    # -- prompt enhancer (Wan2GP's native Qwen-abliterated enhancer) --------
    def _enhance(self, state, text, progress=gr.Progress()):
        if not (text and text.strip()):
            raise gr.Error("Enter a prompt to enhance first.")
        if not all(hasattr(self, a) for a in ("exec_prompt_enhancer_engine",
                                              "get_state_model_type", "get_model_def")):
            raise gr.Error("The prompt enhancer isn't available in this Wan2GP build.")
        progress(0.0, desc="Enhancing with Qwen… (first run downloads the model — "
                           "watch the console)")
        model_type = self.get_state_model_type(state)
        model_def = self.get_model_def(model_type)
        out = self.exec_prompt_enhancer_engine(
            state, model_type, model_def,
            "T",            # text-only enhancement mode
            [text],         # original_prompts
            [None],         # image_start
            None,           # original_image_refs
            True,           # is_image
            False,          # audio_only
            -1,             # seed
            progress,
            -1,             # override_profile
            enhancer_kwargs={"image_prompt_type": "", "video_prompt_type": "",
                             "audio_prompt_type": ""},
        )
        if out and out[0]:
            res = out[0]
            return res[0] if isinstance(res, (list, tuple)) else res
        return gr.update()

    def _interrogate(self, state, model, src, progress=gr.Progress(), *, mode):
        """Interrogate the page's image → prompt text. WD14 tags for booru
        families (Pony/Illustrious), BLIP caption otherwise. src is a filepath
        (txt2img result / img2img init) or the inpaint canvas composite data-URL."""
        if mode == "inpaint":
            img = self._decode_dataurl(src)
            if img is None:
                raise gr.Error("Load an image into the canvas first.")
            path = self._save_img(img.convert("RGB"), "interrogate")
        else:
            path = src
            if not path:
                raise gr.Error("Provide an image first — select a result (Txt2Img) "
                               "or load an init image (Img2Img).")
        if self._gpu_busy(state):
            raise gr.Error("A generation is running — wait for it to finish, then "
                           "interrogate.")
        self._release_faceswap()
        from .core import interrogate as _interro
        fam = discovery.model_family(model)  # Pony/Illustrious/SDXL, or None (native)
        kind = "tags" if fam in _interro.BOORU_FAMILIES else "caption"
        progress(0.05, desc=f"Interrogating ({kind})…")
        # Take the shared GPU lock like every other heavy op: registers interrogate
        # as a GPU process (so it's visible to any_GPU_process_running), evicts the
        # native model for headroom, and frees the ONNX/BLIP VRAM afterward.
        self.acquire_gpu(state)
        try:
            text = _interro.interrogate(path, fam, progress=progress)
        except Exception as e:
            traceback.print_exc()
            raise gr.Error(f"Interrogation failed: {e}")
        finally:
            self.release_gpu(state)
        if not (text and text.strip()):
            raise gr.Error("Interrogation produced no text.")
        return text

    def _native_model_types(self):
        defs = getattr(self, "models_def", None) or {}
        try:
            return [mt for mt in defs if discovery.categorize_native(mt)]
        except Exception:
            return []

    def _native_caps(self, model_type) -> dict:
        """The model's capability/def dict (inpaint_support, inpaint_video_prompt_type,
        image_modes, …). Prefer get_model_def; fall back to the models_def entry."""
        getter = getattr(self, "get_model_def", None)
        if callable(getter):
            try:
                d = getter(model_type)
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
        defs = getattr(self, "models_def", None) or {}
        d = defs.get(model_type)
        return d if isinstance(d, dict) else {}

    # -- native model manual downloads (Models panel) -----------------------
    def _native_dl_model_types(self):
        """Native image model_types offered for download — same set as the
        generation dropdown (honouring the low-VRAM filter)."""
        low = paths.low_vram_only()
        return [mt for mt in self._native_model_types()
                if not low or discovery.is_low_vram_native(mt)]

    def _native_present(self, model_type) -> bool:
        """Is this native model's transformer already downloaded? Uses the same
        check wgp's model-status code does (get_model_filename + files_locator)."""
        try:
            from shared.utils import files_locator as fl
            fn = self.get_model_filename(
                model_type, quantization=self.transformer_quantization,
                dtype_policy=self.transformer_dtype_policy)
            return bool(fn) and fl.get_local_model_filename(fn) is not None
        except Exception:
            return False

    def _native_dl_choices(self):
        """[(decorated_label, model_type)] for the native-download dropdown."""
        out = []
        for mt in self._native_dl_model_types():
            cat = discovery.categorize_native(mt) or "Native"
            mark = "✅ downloaded" if self._native_present(mt) else "⬇️ not downloaded"
            out.append((f"{cat} · {mt} — {mark}", mt))
        return out

    def _download_native(self, model_type, progress=None) -> bool:
        """Download a native model's weights WITHOUT loading it into VRAM —
        mirrors wgp.load_models' download steps (transformer [+URLs2] + text
        encoder; preload/VAE URLs are pulled inside download_models). Any
        remaining bits are completed by Wan2GP on first generation."""
        q, dp = self.transformer_quantization, self.transformer_dtype_policy
        mdef = self.get_model_def(model_type) or {}
        if progress:
            progress(0.05, desc=f"{model_type}: transformer + VAE…")
        fn = self.get_model_filename(model_type, quantization=q, dtype_policy=dp)
        self.download_models(fn, model_type, 0, 1)
        if "URLs2" in mdef:
            fn2 = self.get_model_filename(model_type, quantization=q,
                                          dtype_policy=dp, submodel_no=2)
            if fn2:
                self.download_models(fn2, model_type, 0, 2)
        te = self.get_model_recursive_prop(model_type, "text_encoder_URLs",
                                           return_list=True)
        if te:
            if progress:
                progress(0.7, desc=f"{model_type}: text encoder…")
            te_fn = self.get_model_filename(
                model_type, quantization=self.text_encoder_quantization,
                dtype_policy=dp, URLs=te)
            if te_fn:
                self.download_models(te_fn, model_type, 2, -1,
                                     force_path=mdef.get("text_encoder_folder"))
        return self._native_present(model_type)

    # -- UI -----------------------------------------------------------------
    def _model_choices(self, mode="txt2img"):
        """Categorized dropdown choices for a page, honouring the low-VRAM filter
        AND the mode's capability: txt2img shows every model; img2img/inpaint show
        only guide/mask-capable ones (all SDXL-family + native models whose
        inpaint_support is set), so an unusable model isn't even offered."""
        if mode == "txt2img":
            native = self._native_model_types()
        else:
            native = [mt for mt in self._native_model_types()
                      if self._native_caps(mt).get("inpaint_support")]
        return discovery.build_model_choices(native, low_vram_only=paths.low_vram_only())

    @staticmethod
    def _cache_allow_dirs():
        """Real (symlink-resolved) directories whose files the relaxed file-cache
        check accepts: Gradio's own cache dir, our .cache + outputs, and Wan2GP's
        outputs. Anything outside these stays rejected (the original boundary)."""
        import os
        import tempfile
        dirs = []

        def _add(p):
            try:
                if p:
                    dirs.append(os.path.realpath(str(p)))
            except Exception:
                pass

        try:
            from gradio.context import Context
            _add(getattr(Context.root_block, "GRADIO_CACHE", None))
        except Exception:
            pass
        try:
            from gradio.utils import get_upload_folder
            _add(get_upload_folder())
        except Exception:
            pass
        _add(os.environ.get("GRADIO_TEMP_DIR"))
        _add(os.path.join(tempfile.gettempdir(), "gradio"))
        try:
            _add(paths.cache_dir())
        except Exception:
            pass
        try:
            od = os.path.realpath(str(paths.outputs_dir()))
            home = os.path.realpath(os.path.expanduser("~"))
            # A user-repointed outputs_dir of '/', '~', or an ancestor of home would
            # widen this read allow-list into an arbitrary-local-file primitive once
            # exposed — skip those (cwd/outputs below still covers the normal case).
            if (od not in ("/", home, os.path.dirname(home))
                    and os.path.commonpath([od, home]) != od):
                _add(od)
        except Exception:
            pass
        _add(os.path.join(os.getcwd(), "outputs"))  # Wan2GP's own outputs
        return [d for d in dirs if d]

    @staticmethod
    def _install_file_cache_patch():
        """Relax Gradio 5's check_all_files_in_cache so it also accepts files that
        actually exist on disk AND resolve inside an explicit allow-list (Gradio's
        cache, our .cache/outputs, Wan2GP's outputs) — NOT every existing path.

        Wan2GP's galleries hold relative 'outputs/...' paths (its own generated
        files); any event that carries one (a gallery select / auto-select on the
        native result) otherwise throws 'File … is not in the cache folder and
        cannot be accessed'. We can't control Wan2GP's gallery contents, so we
        loosen the check — but only for files under the allow-list, so the patch
        can't become an arbitrary-local-file-read primitive once Wan2GP is exposed
        via --listen/--share. Idempotent."""
        try:
            import os
            import gradio.processing_utils as _pu
            from gradio_client import utils as _cu
        except Exception:
            # Gradio internals moved/absent — leave the original check untouched
            # rather than silently turning the patch into a no-op via a broad catch.
            print(">>> ImageSuite: could not locate Gradio file-cache internals; "
                  "leaving the original check in place. <<<", flush=True)
            traceback.print_exc()
            return
        if getattr(_pu, "_imagesuite_cache_patch", False):
            return
        # The specific violation the original check raises (a ValueError). Narrow
        # the except to it so unrelated failures aren't downgraded into the
        # permissive path; re-raise everything else unchanged.
        _orig = _pu.check_all_files_in_cache

        def _within_allow_list(p):
            try:
                rp = os.path.realpath(p)
            except Exception:
                return False
            for base in ImageSuite._cache_allow_dirs():
                try:
                    if os.path.commonpath([rp, base]) == base:
                        return True
                except (ValueError, OSError):
                    continue
            return False

        def _lenient(data):
            try:
                _orig(data)            # fast path: unchanged when already valid
            except ValueError:
                def _ok(d):
                    p = d.get("path", "") if isinstance(d, dict) else ""
                    if not p or _cu.is_http_url_like(p):
                        return
                    if not (os.path.exists(p) and _within_allow_list(p)):
                        raise gr.Error(f"File {p} is not accessible.")
                _cu.traverse(data, _ok, _cu.is_file_obj)

        _pu.check_all_files_in_cache = _lenient
        _pu._imagesuite_cache_patch = True
        print(">>> ImageSuite: relaxed Gradio file-cache check "
              "(existing local files under the allow-list now allowed) <<<", flush=True)

    def create_ui(self, api_session):
        # The host's plugin-tab loop calls this constructor with no guard, so a
        # raise (e.g. PermissionError mid-discovery over a mis-pointed dir) would
        # otherwise abort the whole shared Tabs build. Fall back to a minimal
        # message UI instead of taking the rest of the app down.
        try:
            return self._build_ui(api_session)
        except Exception:
            traceback.print_exc()
            return gr.Markdown(
                "### Image Suite failed to load\n\n"
                "Image Suite hit an error while building its UI — the rest of "
                "Wan2GP is unaffected. Check the console/logs for the traceback "
                "(commonly a mis-pointed models/outputs directory in Settings).")

    def _build_ui(self, api_session):
        print("\n>>> ImageSuite UI build ORPHANSUITE-5 loaded "
              "(LoRAs in enhancements + per-family Default Generation Values) <<<\n",
              flush=True)
        self._install_file_cache_patch()
        self._api = api_session
        self._faceswap = None  # lazy FaceSwapPipeline (face swap / body-swap ADetailer)
        choices_by_mode = {m: self._model_choices(m)
                           for m in ("txt2img", "img2img", "inpaint")}
        lora_choices = discovery.lora_choices()
        # SDXL-only list for the Body Swap section (it always needs an SD-family
        # checkpoint, regardless of which model the page is generating with).
        sdxl_choices = discovery.build_model_choices()

        gr.HTML(f"<style>{CSS}</style>", elem_classes="imagesuite-hidden")
        # Tag our main-webui tab button so the accent CSS can target only us.
        gr.HTML(
            "<img src=x style='display:none' onerror=\"(function(){"
            "var NAME=" + repr(PLUGIN_NAME) + ";"
            "function mark(){document.querySelectorAll("
            "'.tab-nav button,button[role=&quot;tab&quot;]').forEach(function(b){"
            "if(b.textContent.trim()===NAME)b.classList.add('imagesuite-tabbtn');});}"
            "mark();new MutationObserver(mark).observe(document.body,"
            "{childList:true,subtree:true});})()\">",
            elem_classes="imagesuite-hidden")
        # Double-click an init-image thumbnail to enlarge it (shared lightbox).
        gr.HTML(LIGHTBOX_HTML, elem_classes="imagesuite-hidden")

        with gr.Column(elem_id="imagesuite-root"):
            gr.HTML(logo.banner_html())
            # Project bar — centered name + Save / Projects buttons. The list lives in
            # POPUPS (modeled on Replicant: gr.Group(visible=False) toggled by clicks),
            # NOT a dropdown on the main screen. Wired in _wire_projects.
            _active = paths.get_active_project()
            with gr.Row(elem_id="imagesuite-projbar"):
                self._proj_name = gr.Markdown(self._projname_md(_active),
                                              elem_id="imagesuite-projname")
                self._proj_save = gr.Button("💾 Save", scale=0, variant="primary")
                self._proj_open = gr.Button("📂 Projects…", scale=0)
            self._proj_active = gr.Textbox(_active, visible=False,
                                           elem_id="imagesuite-proj-active")
            self._proj_status = gr.Markdown("", elem_classes="imagesuite-help")
            # Save-as popup (used for a new/unnamed project).
            with gr.Group(visible=False, elem_classes="imagesuite-modal") as self._proj_savepop:
                gr.Markdown("### 💾 Save project as")
                self._proj_saveas = gr.Textbox(label="Project name",
                                               elem_id="imagesuite-proj-saveas")
                with gr.Row():
                    self._proj_saveas_ok = gr.Button("Save", variant="primary")
                    self._proj_saveas_cancel = gr.Button("Cancel")
            # Projects popup — the list + Load / Rename / Delete live here (off-screen).
            with gr.Group(visible=False, elem_classes="imagesuite-modal") as self._proj_managepop:
                gr.Markdown("### 📂 Projects")
                self._proj_pick = gr.Dropdown(label="Project", elem_id="imagesuite-projpick")
                with gr.Row():
                    self._proj_load = gr.Button("📂 Load", variant="primary")
                    self._proj_rename = gr.Button("✏ Rename")
                    self._proj_delete = gr.Button("🗑 Delete", variant="stop")
                    self._proj_manage_close = gr.Button("Close")
                with gr.Row(visible=False) as self._proj_rename_row:
                    self._proj_rename_to = gr.Textbox(label="New name", scale=3)
                    self._proj_rename_ok = gr.Button("Confirm", scale=1)
                with gr.Row(visible=False) as self._proj_del_row:
                    self._proj_del_ok = gr.Button("🗑 Yes, delete permanently",
                                                  variant="stop")
                    self._proj_del_no = gr.Button("Cancel")
                self._proj_manage_status = gr.Markdown("", elem_classes="imagesuite-help")
            # Shared app-wide right-click menu (idempotent engine + our items) and
            # the hidden relay it writes into.
            gr.HTML(contextmenu.imagesuite_ctx_html(paths.get_ctx_plugin_only()),
                    elem_classes="imagesuite-hidden")
            self._ctx_relay = gr.Textbox(visible=False, elem_id=CTX_RELAY)
            ui = suite.build_suite(model_choices_by_mode=choices_by_mode,
                                   lora_choices=lora_choices,
                                   native_dl_choices=self._native_dl_choices(),
                                   sdxl_choices=sdxl_choices,
                                   send_panel_fn=self._sendto_panel_fn())
        # Kept so the main-page "Send current frame" section (injected under the
        # preview gallery BEFORE this tab is built) can reach our pages/subtabs when
        # we wire it, just below.
        self._ui = ui
        self._wire(ui)
        # The section's widgets were created at insert time (self._sf); now that our
        # pages exist, connect its handlers. Guarded so a wiring hiccup can't take
        # down the whole tab (create_ui falls back to an error panel on raise).
        try:
            self._wire_send_frame_section()
        except Exception:
            traceback.print_exc()
        # on_tab_select drains the SendTo inbox into these (see _drain_inbox):
        # [img2img input, inpaint bg_bridge, modify bg_bridge, subtabs].
        try:
            pages = ui["pages"]
            self.on_tab_outputs = [pages["img2img"]["input_image"],
                                   pages["inpaint"]["bg_bridge"],
                                   pages["modify"]["bg_bridge"],
                                   ui["subtabs"]]
        except Exception:
            traceback.print_exc()
            self.on_tab_outputs = None
        return ui

    # -- LoRA helpers -------------------------------------------------------
    @staticmethod
    def _sd_loading(progress):
        """Signal that the SDXL checkpoint is loading (it's slow + gives no
        step feedback during load), so Generate doesn't look frozen."""
        try:
            progress(0.02, desc="Loading the SDXL model… (first run reads the "
                                "checkpoint, ~6.5 GB)")
        except Exception:
            pass
        try:
            gr.Info("Loading the SDXL model… first run reads the checkpoint "
                    "(~6.5 GB); subsequent gens are fast.")
        except Exception:
            pass

    @staticmethod
    def _sd_step_cb(progress, img_index, steps, total):
        """A diffusers ``callback_on_step_end`` that advances the Gradio progress
        bar as SD sampling runs (the load is silent; this fills once steps begin)
        AND honours Abort mid-image: when gen_sd.was_aborted() it sets
        pipe._interrupt so diffusers bails out of the current denoise loop.
        Must return the callback_kwargs dict diffusers hands it."""
        def _cb(pipe, step, timestep, cb_kwargs):
            try:
                if gen_sd.was_aborted():
                    pipe._interrupt = True  # interrupt the running diffusers loop
            except Exception:
                pass
            try:
                done = img_index * steps + step + 1
                progress(done / max(1, total), desc=f"Generating… step {done}/{total}")
            except Exception:
                pass
            return cb_kwargs
        return _cb

    @staticmethod
    def _lora_list(selected_paths, mult_str):
        """UI (paths + "0.8, 1.0" multipliers) → SD-pipeline [{"name","weight"}].
        Names are stems; the SD pipeline resolves them under sd_lora_dir."""
        from pathlib import Path as _P
        paths_ = list(selected_paths or [])
        weights = []
        for tok in (mult_str or "").replace(";", ",").split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                weights.append(float(tok))
            except ValueError:
                pass
        out = []
        for i, p in enumerate(paths_):
            w = weights[i] if i < len(weights) else 1.0
            out.append({"name": _P(p).stem, "weight": w})
        return out

    def _native_loras(self, model_type):
        """LoRA files in this native model's Wan2GP LoRA dir → dropdown (label, value)
        where value is the filename (what activated_loras expects)."""
        getter = getattr(self, "get_lora_dir", None)
        if not callable(getter):
            return []
        try:
            import os
            d = getter(model_type)
            if not (d and os.path.isdir(d)):
                return []
            return [(f, f) for f in sorted(os.listdir(d))
                    if f.lower().endswith((".safetensors", ".sft", ".lora", ".ckpt", ".pt"))]
        except Exception:
            return []

    @staticmethod
    def _native_lora_settings(selected_paths, mult_str):
        """activated_loras (filenames, as the dropdown supplies) + loras_multipliers
        (space-separated) for the native Wan2GP task settings. Native image models
        may ignore these (Flux/Z-Image don't take user LoRAs); harmless when
        unsupported."""
        paths_ = list(selected_paths or [])
        if not paths_:
            return {}
        weights = [t.strip() for t in (mult_str or "").replace(";", ",").split(",")
                   if t.strip()]
        mult = " ".join(weights[:len(paths_)]) if weights else ""
        return {"activated_loras": paths_, "loras_multipliers": mult}

    # -- generation backends ------------------------------------------------
    def _gen_native(self, model_type, pos, neg, w, h, steps, cfg, seed, *,
                    mode="txt2img", denoise=1.0, guide_path=None, mask_path=None,
                    loras=None, mult_str="", progress=None, api=None, state=None):
        """One native (Flux/Z-Image/Qwen) image via the Wan2GP task API.

        mode: 'txt2img' (image_mode 1), 'img2img' (image_mode 1 + image_guide +
        denoise) or 'inpaint' (image_mode 2 + image_guide + image_mask + denoise).
        Image fields are file paths — the session API absolutizes/loads them.
        NOTE: native i2i/inpaint plumbing is model-dependent and best-effort;
        SDXL-family is the well-trodden path."""
        import random as _rng
        sd = int(seed) if int(seed) >= 0 else _rng.randint(0, 2**31 - 1)
        settings = dict(self.get_default_settings(model_type))
        settings.update({
            "model_type": model_type, "prompt": pos,
            "negative_prompt": neg or "", "resolution": f"{int(w)}x{int(h)}",
            "num_inference_steps": int(steps), "guidance_scale": float(cfg),
            "seed": int(sd), "video_length": 1, "batch_size": 1,
        })
        if mode == "txt2img":
            settings["image_mode"] = 1
        elif mode in ("img2img", "inpaint"):
            # Guide/mask are only honoured by native models that declare
            # inpaint_support (this tracks guide-image capability — e.g. plain
            # z_image_base supports neither, while flux / qwen_image /
            # z_image_control* do). Reject early with a clear message rather than
            # submit a task the backend will ignore or error on.
            caps = self._native_caps(model_type)
            if not caps.get("inpaint_support"):
                raise gr.Error(
                    f"'{model_type}' doesn't support {mode} — it has no guide/mask "
                    "input. Use a Flux/Qwen-Image model, a Z-Image *Control* model, "
                    "or an SDXL/Pony/Illustrious checkpoint.")
            if mode == "img2img":
                settings.update({"image_mode": 1, "image_guide": guide_path,
                                 "denoising_strength": float(denoise),
                                 "video_prompt_type": "V"})
            else:  # inpaint — use the letters the model itself declares ("VA"/"VAGI")
                vpt = caps.get("inpaint_video_prompt_type") or "VA"
                settings.update({"image_mode": 2, "image_guide": guide_path,
                                 "image_mask": mask_path,
                                 "denoising_strength": float(denoise),
                                 "video_prompt_type": vpt, "model_mode": 0})
        settings.update(self._native_lora_settings(loras, mult_str))
        # Pass NO callbacks: the webui bridge installs its own that surface task
        # progress onto the wrapped click's outputs (the visible progress bar).
        # Supplying our own here replaced that default and killed the bar.
        try:
            job = (api or self._api).submit_task(settings)
            result = job.result()
        except Exception as e:
            if "generation in progress" in str(e).lower():
                raise gr.Error(
                    "Another generation is still pending — let it finish, then "
                    "try again.")
            raise
        if result.success and result.generated_files:
            return list(result.generated_files)
        if result.errors:
            raise gr.Error(str(list(result.errors)[0]))
        return []

    # -- PaintShop canvas helpers ------------------------------------------
    # Decompression-bomb guard: cap decoded image area + the base64 payload we'll
    # even attempt to decode (a ~64 MP cap covers any realistic canvas/gallery).
    _MAX_DECODE_PIXELS = 64 * 1024 * 1024  # ~64 megapixels
    _MAX_DATAURL_BYTES = 96 * 1024 * 1024  # base64 string length cap (~72 MB raw)

    @staticmethod
    def _decode_dataurl(url):
        """data:image/png;base64,… → PIL.Image (or None). Guards against
        decompression-bomb payloads (size + pixel cap)."""
        if not url or "," not in url:
            return None
        import base64
        import io
        from PIL import Image
        b64 = url.split(",", 1)[1]
        if len(b64) > ImageSuite._MAX_DATAURL_BYTES:
            raise gr.Error("That image is too large to process.")
        try:
            raw = base64.b64decode(b64)
        except Exception:
            return None
        prev = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = ImageSuite._MAX_DECODE_PIXELS
        try:
            im = Image.open(io.BytesIO(raw))
            im.load()
            return im
        except Image.DecompressionBombError:
            raise gr.Error("That image is too large to process.")
        except Exception:
            return None
        finally:
            Image.MAX_IMAGE_PIXELS = prev

    @staticmethod
    def _mask_nonempty(mask) -> bool:
        try:
            import numpy as np
            return bool(np.asarray(mask.convert("L")).max() > 10)
        except Exception:
            return True

    @staticmethod
    def _save_img(img, tag) -> str:
        """Persist a PIL image to the cache and return its path (native paths
        take file paths for image_guide/image_mask)."""
        out = paths.cache_dir() / "canvas"
        out.mkdir(parents=True, exist_ok=True)
        p = out / f"{tag}_{int(time.time() * 1000)}.png"
        img.save(p)
        return str(p)

    @staticmethod
    def _serve(files):
        """Copy result files into Gradio's cache folder and return the new paths.
        Gradio 5 only serves — and only accepts back via the gallery's select event
        (check_all_files_in_cache) — files that live INSIDE its cache. Results land
        in Wan2GP's outputs dir (native) or our .cache (SD), both outside it, hence
        'File … is not in the cache folder and cannot be accessed'. Copying them
        into the cache is what makes the gallery display + selection + Save As work."""
        import os
        import shutil
        import tempfile
        cache = None
        try:  # the exact dir the running app uses
            from gradio.context import Context
            cache = getattr(Context.root_block, "GRADIO_CACHE", None)
        except Exception:
            cache = None
        if not cache:
            try:
                from gradio.utils import get_upload_folder
                cache = get_upload_folder()
            except Exception:
                cache = os.environ.get("GRADIO_TEMP_DIR") or os.path.join(
                    tempfile.gettempdir(), "gradio")
        dest = os.path.join(cache, "imagesuite")
        os.makedirs(dest, exist_ok=True)
        served = []
        for idx, f in enumerate(files or []):
            try:
                src = os.path.abspath(str(f))  # resolve relative paths against cwd
                # Prefix a unique token so a fixed-seed batch (deterministic SD
                # filenames like sd_<seed>_0.png) doesn't collapse N results into
                # one overwritten cache file (which would alias every gallery entry).
                base = os.path.basename(src)
                dst = os.path.join(dest, f"{int(time.time()*1000)}_{idx}_{base}")
                if src != os.path.abspath(dst):
                    shutil.copy2(src, dst)
                served.append(dst)
            except Exception:
                traceback.print_exc()
                served.append(os.path.abspath(str(f)))
        return served

    def _persist_results(self, mode, served):
        """Copy a tab's just-served results into a per-mode persist dir (cleared each
        time, so it never grows) + record the list in the gitignored .imagesuite.json,
        so the gallery is restored on the next app load. Best-effort — a hiccup here
        never breaks a generation."""
        import os
        import shutil
        try:
            d = paths.cache_dir() / "persist" / "results" / str(mode)
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
            kept = []
            for sp in served or []:
                try:
                    dst = d / os.path.basename(sp)
                    shutil.copy2(sp, dst)
                    kept.append(str(dst))
                except Exception:
                    pass
            paths.set_results(mode, kept)
        except Exception:
            traceback.print_exc()

    def _gallery_result(self, files, mode=None):
        """The (gallery, picked, save) tuple every generate handler returns.

        The gallery is fed PIL images, NOT paths: Gradio then owns the cache entry
        for each, so the select event's file-in-cache check can't trip on an
        external/relative path. picked + Save As still get a real cached file path
        (via _serve) so send-to / download work. When ``mode`` is given, the results
        are also persisted so the tab's gallery survives a restart."""
        served = self._serve(files if isinstance(files, (list, tuple)) else [files])
        if not served:
            raise gr.Error("Generation produced no images.")
        if mode:
            self._persist_results(mode, served)
        from PIL import Image
        gallery = []
        for p in served:
            try:
                with Image.open(p) as im:
                    gallery.append(im.copy())
            except Exception:
                traceback.print_exc()
                gallery.append(p)
        return gallery, served[0], gr.update(value=served[0])

    @staticmethod
    def _file_to_dataurl(path) -> str:
        """Local image path → PNG data-URL, for pushing into the canvas frame / overlay
        strip. Keep RGBA so a transparent overlay stays transparent — convert('RGB')
        would drop the alpha and (since clear pixels are usually stored white) paint a
        solid white background. Opaque images just get a full-alpha channel (harmless)."""
        import base64
        from PIL import Image
        with Image.open(path) as im:
            import io
            buf = io.BytesIO()
            im.convert("RGBA").save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    # -- wiring -------------------------------------------------------------
    def _wire(self, ui):
        pages = ui["pages"]
        for mode, c in pages.items():
            if mode == "modify":          # settings-less editor page, wired apart
                self._wire_modify_page(c)
            else:
                self._wire_page(mode, c)
        self._wire_prompt_library(pages)
        self._wire_sends(ui)
        self._wire_settings(ui)
        self._wire_ctx(ui)
        self._wire_overlays(ui)
        self._wire_persist(ui)
        self._wire_projects(ui)

    @staticmethod
    def _projname_md(name) -> str:
        return f"📁 **{name}**" if name else "Unsaved Project"

    def _wire_projects(self, ui):
        """Header CRUD via popups (Replicant-style gr.Group toggles — no on-screen
        dropdown). A project bundles, for the whole workspace: every JSON-able param per
        tab (same capture as the Prompt Library), every filepath image (img2img init +
        face/body/colour refs), each tab's currently-displayed results, and the full
        MultiCanvas layer stack. Only the Save commit needs a js= (to flush the live
        canvas + return inputs explicitly, per the Gradio-5.29 payload behaviour);
        Load/Rename/Delete read the in-popup dropdown as normal inputs, immune to it."""
        pages = ui["pages"]
        param_comps, param_idx = [], []
        for m, c in pages.items():
            for k, comp in self._pl_items(c):
                param_comps.append(comp); param_idx.append((m, k))
        img_comps, img_idx = [], []
        for m, c in pages.items():
            for k, comp in c.items():
                if isinstance(comp, gr.Image):
                    img_comps.append(comp); img_idx.append((m, k))
        gal_out = []
        for m in pages:
            gal_out += [pages[m]["gallery"], pages[m]["picked"], pages[m]["save"]]
        inp = pages.get("inpaint", {})
        canvas_state = inp.get("state")          # filled by __is_inpaint_pushstate
        canvas_bridge = inp.get("state_bridge")  # rebuilds the canvas on load
        pick = self._proj_pick

        def _names():
            return projects.list_projects()

        # ---- open / close popups ----
        # Save → name popup (prefilled with the open project's name so it re-saves).
        self._proj_save.click(
            lambda a: (gr.update(visible=True), gr.update(value=a or "")),
            inputs=[self._proj_active],
            outputs=[self._proj_savepop, self._proj_saveas])
        self._proj_saveas_cancel.click(lambda: gr.update(visible=False),
                                       outputs=[self._proj_savepop])
        # Projects → manage popup (refresh the list; reset the sub-rows + status).
        self._proj_open.click(
            lambda: (gr.update(visible=True), gr.update(choices=_names()),
                     gr.update(visible=False), gr.update(visible=False), ""),
            outputs=[self._proj_managepop, pick, self._proj_rename_row,
                     self._proj_del_row, self._proj_manage_status])
        self._proj_manage_close.click(lambda: gr.update(visible=False),
                                      outputs=[self._proj_managepop])

        # ---- Save commit (the only js= handler: flush canvas + explicit return) ----
        def _save(name, cstate, *vals):
            name = (name or "").strip()
            if not name:
                return (gr.update(), gr.update(), gr.update(visible=True),
                        "Enter a project name.")
            pvals, ivals = vals[:len(param_comps)], vals[len(param_comps):]
            tabs = {}
            for (m, k), v in zip(param_idx, pvals):
                tabs.setdefault(m, {})[k] = v
            refs = {}
            for (m, k), v in zip(img_idx, ivals):
                if v:
                    refs.setdefault(m, {})[k] = v
            results = {m: paths.get_results(m) for m in pages}
            try:
                saved = projects.save_project(name, tabs=tabs, results=results,
                                              refs=refs, canvas_state=(cstate or None))
            except Exception as e:
                traceback.print_exc()
                return (gr.update(), gr.update(), gr.update(visible=True),
                        f"⚠️ Save failed: {e}")
            paths.set_active_project(saved)
            return (self._projname_md(saved), saved, gr.update(visible=False),
                    f"💾 Saved “{saved}”.")
        self._proj_saveas_ok.click(
            _save, inputs=[self._proj_saveas, canvas_state] + param_comps + img_comps,
            outputs=[self._proj_name, self._proj_active, self._proj_savepop,
                     self._proj_status],
            # Gradio 5.29: a js= with a backend fn REPLACES the input payload with its
            # return value, so we return every input explicitly (rest-spread passes the
            # params/images through; we substitute the FRESH canvas state after pushstate).
            js="(name, cstate, ...rest) => {"
               " var st=cstate;"
               " try{ if(window.__is_inpaint_pushstate){ window.__is_inpaint_pushstate();"
               " var el=document.querySelector('#imagesuite-inpaint-state textarea')"
               "||document.querySelector('#imagesuite-inpaint-state input');"
               " if(el) st=el.value; } }catch(e){}"
               " return [name, st].concat(rest); }")

        # ---- Load (reads the in-popup dropdown as a NORMAL input — no js=) ----
        def _load(sel):
            data = projects.load_project(sel) if sel else None
            if not data:
                return ([gr.update() for _ in param_comps]
                        + [gr.update() for _ in img_comps]
                        + [gr.update() for _ in gal_out]
                        + [gr.update(), gr.update(), gr.update(),
                           "Pick a project to load.", gr.update()])
            tabs, refs = data["tabs"], data["refs"]
            op = []
            for (m, k) in param_idx:
                ent = tabs.get(m, {})
                if k == "loras" and "loras" in ent:
                    op.append(gr.update(choices=self._lora_choices_for(ent.get("model")),
                                        value=ent.get("loras") or []))
                elif k in ent:
                    op.append(gr.update(value=ent[k]))
                else:
                    op.append(gr.update())
            oi = []
            for (m, k) in img_idx:
                path = (refs.get(m) or {}).get(k)
                oi.append(gr.update(value=path) if path else gr.update(value=None))
            gals = []
            for m in pages:
                files = data["results"].get(m) or []
                if files:
                    g, picked, save = self._gallery_result(files, m)
                    gals += [g, picked, save]
                else:
                    paths.set_results(m, [])   # clear stale persist so a restart shows empty
                    gals += [gr.update(value=None), gr.update(value=None),
                             gr.update(value=None)]
            cs = data.get("canvas_state")
            cb = (canvas.state_bridge_html(cs, "inpaint", nonce=time.time())
                  if (cs and canvas_bridge is not None) else gr.update())
            paths.set_active_project(data["name"])
            return (op + oi + gals
                    + [cb, self._projname_md(data["name"]), data["name"],
                       f"📂 Loaded “{data['name']}”.", gr.update(visible=False)])
        self._proj_load.click(
            _load, inputs=[pick],
            outputs=(param_comps + img_comps + gal_out
                     + [canvas_bridge, self._proj_name, self._proj_active,
                        self._proj_status, self._proj_managepop]))

        # ---- Rename (reveal a name row inside the popup, then confirm — no js=) ----
        self._proj_rename.click(lambda: gr.update(visible=True),
                                outputs=[self._proj_rename_row])
        def _rename(sel, new):
            new = (new or "").strip()
            if not sel or not new:
                return (gr.update(), gr.update(), gr.update(), gr.update(),
                        "Pick a project and enter a new name.")
            res = projects.rename_project(sel, new)
            if not res:
                return (gr.update(), gr.update(), gr.update(), gr.update(),
                        f"⚠️ “{projects.sanitize(new)}” already exists.")
            name_up, active_up = gr.update(), gr.update()
            if paths.get_active_project() == projects.sanitize(sel):
                paths.set_active_project(res)
                name_up, active_up = self._projname_md(res), res
            return (gr.update(choices=_names(), value=res), gr.update(visible=False),
                    name_up, active_up, f"✏ Renamed to “{res}”.")
        self._proj_rename_ok.click(
            _rename, inputs=[pick, self._proj_rename_to],
            outputs=[pick, self._proj_rename_row, self._proj_name,
                     self._proj_active, self._proj_manage_status])

        # ---- Delete (reveal a confirm row, then delete — no js=) ----
        self._proj_delete.click(lambda: gr.update(visible=True),
                                outputs=[self._proj_del_row])
        self._proj_del_no.click(lambda: gr.update(visible=False),
                                outputs=[self._proj_del_row])
        def _delete(sel):
            if not sel:
                return (gr.update(), gr.update(), gr.update(), gr.update(),
                        "Pick a project to delete.")
            ok = projects.delete_project(sel)
            name_up, active_up = gr.update(), gr.update()
            if ok and paths.get_active_project() == projects.sanitize(sel):
                paths.set_active_project("")
                name_up, active_up = self._projname_md(""), ""
            return (gr.update(choices=_names(), value=None), gr.update(visible=False),
                    name_up, active_up,
                    f"🗑 Deleted “{sel}”." if ok else "⚠️ Couldn't delete.")
        self._proj_del_ok.click(
            _delete, inputs=[pick],
            outputs=[pick, self._proj_del_row, self._proj_name,
                     self._proj_active, self._proj_manage_status])

    # -- Prompt Library (shared across all three tabs) ----------------------
    # Every JSON-able scalar input is saved/loaded by inspecting the page's
    # components by type — so it captures ALL generation settings, model, LoRAs and
    # post-process settings without a hand-maintained list. Images/buttons/galleries/
    # State/Markdown are excluded by type; these keys are excluded explicitly: the
    # library's own controls, the transient resolution helpers, and the outpaint knobs
    # (a separate operation, not part of a saved "prompt").
    _PL_SAVABLE = (gr.Textbox, gr.Dropdown, gr.Slider, gr.Number, gr.Checkbox, gr.Radio)
    _PL_EXCLUDE = {"pl_name", "pl_saved", "res_preset", "res_lock", "ov_folder",
                   "out_size", "out_top", "out_bottom", "out_left", "out_right",
                   "out_feather",
                   # canvas bridge textboxes — transient data-URLs, not settings
                   # (the MultiCanvas state is saved separately as canvas_state)
                   "composite", "mask", "state",
                   # Modify tab's edited-image export — a multi-MB base64 data-URL,
                   # not a setting; would bloat project.json if serialized.
                   "out"}

    def _pl_items(self, c):
        """Ordered (key, component) pairs this page saves/loads."""
        return [(k, comp) for k, comp in c.items()
                if isinstance(comp, self._PL_SAVABLE) and k not in self._PL_EXCLUDE]

    def _lora_choices_for(self, model_value):
        """LoRA dropdown choices for a model (mirrors _on_model) so Load can refresh
        the list to the saved model's family WITHOUT firing model.change (which would
        reset the freshly-loaded settings)."""
        backend, ident = discovery.parse_model_value(model_value or "")
        if backend == "sd":
            return discovery.lora_choices(family=discovery.model_family(model_value))
        if backend == "native":
            return self._native_loras(ident)
        return []

    def _wire_prompt_library(self, pages):
        """Wire Save-as / Update / Load / Delete for the shared library. Save/Update/
        Delete refresh EVERY tab's dropdown (one shared collection); Load applies the
        saved fields to the current tab only. Setting the model via a handler output
        does not fire model.change, so loaded settings are not clobbered by the
        model's defaults; the LoRA list's choices are refreshed explicitly."""
        from .core import prompt_library as plib
        all_saved = [c["pl_saved"] for c in pages.values() if "pl_saved" in c]
        if not all_saved:
            return

        def _refresh(names, cur, value):
            # choices on every tab's dropdown; value only on the active tab's.
            return [(gr.update(choices=names, value=value) if s is cur
                     else gr.update(choices=names)) for s in all_saved]

        for mode, c in pages.items():
            if "pl_save" not in c:
                continue
            items = self._pl_items(c)
            keys = [k for k, _ in items]
            comps = [comp for _, comp in items]
            cur = c["pl_saved"]

            def _save(name, *vals, _keys=keys, _cur=cur):
                name = (name or "").strip()
                if not name:
                    return [gr.update() for _ in all_saved] + ["Enter a name first."]
                names = plib.save(name, dict(zip(_keys, vals)))
                return _refresh(names, _cur, name) + [f"Saved “{name}”."]
            c["pl_save"].click(_save, inputs=[c["pl_name"]] + comps,
                               outputs=all_saved + [c["pl_status"]])

            def _update(sel, *vals, _keys=keys, _cur=cur):
                if not sel:
                    return [gr.update() for _ in all_saved] + ["Pick a saved entry to update."]
                names = plib.save(sel, dict(zip(_keys, vals)))
                return _refresh(names, _cur, sel) + [f"Updated “{sel}”."]
            c["pl_update"].click(_update, inputs=[c["pl_saved"]] + comps,
                                 outputs=all_saved + [c["pl_status"]])

            def _delete(sel, _cur=cur):
                if not sel:
                    return [gr.update() for _ in all_saved] + ["Pick a saved entry to delete."]
                names = plib.delete(sel)
                return _refresh(names, _cur, None) + [f"Deleted “{sel}”."]
            c["pl_delete"].click(_delete, inputs=[c["pl_saved"]],
                                 outputs=all_saved + [c["pl_status"]])

            def _load(sel, _keys=keys):
                entry = plib.get(sel) if sel else None
                if not entry:
                    return ([gr.update() for _ in _keys]
                            + [gr.update(), "Pick a saved entry to load."])
                outs = []
                for k in _keys:
                    if k == "loras" and "loras" in entry:  # refresh to saved family
                        outs.append(gr.update(
                            choices=self._lora_choices_for(entry.get("model")),
                            value=entry.get("loras") or []))
                    elif k in entry:
                        outs.append(gr.update(value=entry[k]))
                    else:  # field not in this entry (saved from another tab) → leave
                        outs.append(gr.update())
                return outs + [gr.update(value=sel), f"Loaded “{sel}”."]
            c["pl_load"].click(_load, inputs=[c["pl_saved"]],
                               outputs=comps + [c["pl_name"], c["pl_status"]])

    def _push_overlays(self, folder, mode):
        """Load a folder's overlay images (as data-URLs) into the canvas strip."""
        import os
        from .core import overlays as ov
        items = []
        for p in ov.list_images(folder):
            try:
                items.append({"name": os.path.basename(p),
                              "url": self._file_to_dataurl(p)})
            except Exception:
                traceback.print_exc()
        return canvas.overlays_bridge_html(items, mode, nonce=time.time())

    def _wire_overlays(self, ui):
        """Overlays library tab: folder + thumbnail CRUD over core.overlays."""
        import os
        o = ui.get("overlays")
        if not o:
            return
        from .core import overlays as ov

        def _refresh(folder):
            folders = ov.list_folders()
            if folder not in folders:
                folder = ov.ROOT_LABEL
            return (gr.update(choices=folders, value=folder),
                    gr.update(value=ov.list_images(folder)))

        o["folder"].change(_refresh, inputs=[o["folder"]],
                           outputs=[o["folder"], o["gallery"]])

        # (The native folder create/delete, image rename/move/delete and upload
        # widgets were removed — the file-browser right-click menus + drag-drop below
        # replace them. The folder dropdown stays as the navigator; .select isn't
        # wired since file ops carry the thumbnail index instead.)

        # -- file-browser bridges: the JS in overlays_panel.py drives two hidden
        #    textboxes — ov_action ({op,idx,arg,nonce}) for right-click menu ops, and
        #    ov_upload ({items:[{name,dataurl}],nonce}) for drag-drop / picked images.
        #    File ops carry the thumbnail INDEX; we resolve index→name server-side. --
        import json as _json

        def _ov_action(payload, folder):
            try:
                d = _json.loads(payload) if payload else {}
            except Exception:
                d = {}
            op, arg, idx = d.get("op"), d.get("arg"), d.get("idx")
            try:
                if op == "new_folder":
                    folder = ov.create_folder(arg); msg = f"Created folder '{folder}'."
                elif op == "rename_folder":
                    folder = ov.rename_folder(folder, arg)
                    msg = f"Renamed folder to '{folder}'."
                elif op == "delete_folder":
                    ov.delete_folder(folder); folder = ov.ROOT_LABEL
                    msg = "Deleted folder."
                elif op in ("rename_file", "delete_file", "moveup_file", "move_file"):
                    imgs = ov.list_images(folder)
                    i = int(idx) if idx is not None else -1
                    if not (0 <= i < len(imgs)):
                        raise ValueError("That image is no longer here — try again.")
                    name = os.path.basename(imgs[i])
                    if op == "rename_file":
                        name = ov.rename_image(folder, name, arg)
                        msg = f"Renamed to '{name}'."
                    elif op == "delete_file":
                        ov.delete_image(folder, name); msg = f"Deleted '{name}'."
                    elif op == "moveup_file":
                        ov.move_image(folder, name, ov.ROOT_LABEL)
                        msg = f"Moved '{name}' up to {ov.ROOT_LABEL}."
                    else:  # move_file → a named folder (blank → root)
                        dest = (arg or "").strip() or ov.ROOT_LABEL
                        ov.move_image(folder, name, dest)
                        msg = f"Moved '{name}' to '{dest}'."
                else:
                    return (gr.update(), gr.update(), gr.update())
            except Exception as e:
                return (gr.update(), gr.update(), f"⚠ {e}")
            return (*_refresh(folder), f"✅ {msg}")
        o["ov_action"].input(
            _ov_action, inputs=[o["ov_action"], o["folder"]],
            outputs=[o["folder"], o["gallery"], o["status"]])

        def _ov_upload(payload, folder):
            import base64
            import io
            import tempfile
            from PIL import Image
            try:
                items = (_json.loads(payload) or {}).get("items", []) if payload else []
            except Exception:
                items = []
            tmpd = tempfile.mkdtemp(prefix="ovup_")
            fps = []
            for it in (items or [])[:50]:
                try:
                    b64 = (it.get("dataurl") or "").split(",", 1)[-1]
                    if len(b64) > 24 * 1024 * 1024:     # ~18 MB encoded cap per image
                        continue
                    raw = base64.b64decode(b64)
                    prev = Image.MAX_IMAGE_PIXELS       # decompression-bomb guard
                    Image.MAX_IMAGE_PIXELS = 50_000_000
                    try:
                        with Image.open(io.BytesIO(raw)) as im:
                            im.load()
                    finally:
                        Image.MAX_IMAGE_PIXELS = prev
                    fp = os.path.join(tmpd,
                                      ov._safe_name(it.get("name") or "overlay.png"))
                    with open(fp, "wb") as f:
                        f.write(raw)
                    fps.append(fp)
                except Exception:
                    continue
            try:
                n = ov.save_uploads(folder, fps); msg = f"✅ Added {n} image(s)."
            except Exception as e:
                msg = f"⚠ {e}"
            return (*_refresh(folder), msg)
        o["ov_upload"].input(
            _ov_upload, inputs=[o["ov_upload"], o["folder"]],
            outputs=[o["folder"], o["gallery"], o["status"]])

        # Top toolbar → the same JS flows as the right-click menu, acting on the
        # currently-selected thumbnail. window.ovTool (overlays_panel.py) does the
        # prompt/confirm then drives the bridges above; these are js-only (no server fn).
        for _b, _op in (("tb_upload", "upload"), ("tb_newfolder", "newfolder"),
                        ("tb_rename", "rename"), ("tb_move", "move"),
                        ("tb_delete", "delete")):
            o[_b].click(None, js="() => window.ovTool && window.ovTool('" + _op + "')")

        # -- Preview + Send-to: clicking a thumbnail fills the preview pane and arms
        #    the selection; the buttons push it into the editor (same targets as the
        #    cross-plugin Send menu + the new add-layer canvas bridge), then jump to
        #    that tab.
        def _ov_select(evt: gr.EventData):
            # gr.SelectData.__init__ KeyErrors when Gradio sends a select event
            # without "value"; read the raw payload off gr.EventData instead.
            data = getattr(evt, "_data", {}) or {}
            v = data.get("value") if isinstance(data, dict) else None
            path = ((v.get("image", {}).get("path") or v.get("path"))
                    if isinstance(v, dict) else v if isinstance(v, str) else None)
            return path, gr.update(value=path)
        o["gallery"].select(_ov_select, outputs=[o["selected"], o["preview"]])

        pages, subtabs, tab_ids = ui["pages"], ui["subtabs"], ui["tab_ids"]
        send_out = [pages["img2img"]["input_image"], pages["inpaint"]["bg_bridge"],
                    pages["inpaint"]["addlayer_bridge"], subtabs, self.main_tabs]

        def _need(sel):
            if not sel:
                gr.Warning("Click an overlay to select it first.")
            return bool(sel)

        def _send_i2i(sel):
            if not _need(sel):
                return [gr.update()] * 5
            return [gr.update(value=sel), gr.update(), gr.update(),
                    gr.update(selected=tab_ids["img2img"]),
                    gr.update(selected=PLUGIN_ID)]
        o["send_i2i"].click(_send_i2i, inputs=[o["selected"]], outputs=send_out)

        def _send_mc_canvas(sel):
            if not _need(sel):
                return [gr.update()] * 5
            html = canvas.bg_bridge_html(self._file_to_dataurl(sel), "inpaint",
                                         nonce=time.time())
            return [gr.update(), gr.update(value=html), gr.update(),
                    gr.update(selected=tab_ids["inpaint"]),
                    gr.update(selected=PLUGIN_ID)]
        o["send_mc_canvas"].click(_send_mc_canvas, inputs=[o["selected"]],
                                  outputs=send_out)

        def _send_mc_layer(sel):
            if not _need(sel):
                return [gr.update()] * 5
            html = canvas.addlayer_bridge_html(self._file_to_dataurl(sel), "inpaint",
                                               nonce=time.time())
            return [gr.update(), gr.update(), gr.update(value=html),
                    gr.update(selected=tab_ids["inpaint"]),
                    gr.update(selected=PLUGIN_ID)]
        o["send_mc_layer"].click(_send_mc_layer, inputs=[o["selected"]],
                                 outputs=send_out)

    # Settings persisted per tab so the image tabs come back as you left them
    # after a restart (written on Generate, restored on page load).
    _PERSIST_KEYS = ["model", "sampler", "scheduler", "steps", "cfg", "clip_skip",
                     "seed", "width", "height", "count", "loras", "lora_mult",
                     "pos", "neg", "denoise", "resize_mode", "feather", "mask_mode",
                     "inpaint_fill", "inpaint_area", "padding"]

    def _wire_persist(self, ui):
        """Persist each tab's settings to <wan2gp_root>/.imagesuite.json on Generate and
        restore them on page load. File-backed (survives restart) and isolated
        from the generation handlers, so a persistence hiccup can't break a gen."""
        pages = ui["pages"]
        spec, comps = [], []  # spec[i] = (mode, key); comps[i] = component
        for mode, c in pages.items():
            if "generate" not in c:       # non-gen pages (Modify) have nothing to persist
                continue
            keys = [k for k in self._PERSIST_KEYS if k in c]

            def _save(*vals, _mode=mode, _keys=keys):
                try:
                    paths.set_ui_state(_mode, dict(zip(_keys, vals)))
                except Exception:
                    traceback.print_exc()

            # A second, side-effect-only click on the same Generate button.
            c["generate"].click(_save, inputs=[c[k] for k in keys], outputs=[])
            for k in keys:
                spec.append((mode, k))
                comps.append(c[k])

        self._persist_spec = spec
        # out_size components follow the model family (per page) but aren't a
        # persisted key — restore them too so a restored SD model doesn't leave a
        # stale (SDXL-default) outpaint size list.
        out_size_extra = [(mode, c["out_size"]) for mode, c in pages.items()
                          if "out_size" in c]
        # res_preset choices follow the model family too (and _on_model won't fire
        # on a programmatic model set) — refresh them per page alongside out_size.
        res_preset_extra = [(mode, c["res_preset"]) for mode, c in pages.items()
                            if "res_preset" in c]

        def _restore():
            saved = {}
            try:
                saved = paths.get_ui_state()
            except Exception:
                traceback.print_exc()
            # Resolve, per mode, the model value we'll actually restore: a saved
            # model is only valid if it's still in that mode's choices (it may have
            # been deleted, the dir repointed, or hidden by the low-VRAM filter).
            valid_model = {}
            for mode in {m for m, _ in self._persist_spec}:
                m = saved.get(mode) or {}
                mv = m.get("model")
                choices = {v for _, v in self._model_choices(mode)}
                valid_model[mode] = mv if (mv and mv in choices) else None
            ups = []
            for mode, key in self._persist_spec:
                m = saved.get(mode) or {}
                mv = valid_model.get(mode)
                if key == "model":
                    # Skip an invalid/absent model rather than pushing an
                    # out-of-choices value Gradio would silently drop.
                    ups.append(gr.update(value=mv) if mv else gr.update())
                elif key == "loras":
                    # Re-scope choices/label to the restored model's family (the
                    # programmatic value-set doesn't fire _on_model), keeping only
                    # restored values that are still valid for that family.
                    if mv:
                        backend, ident = discovery.parse_model_value(mv)
                        fam = discovery.model_family(mv)
                        if backend == "native":
                            cat = discovery.categorize_native(ident) or "Native"
                            choices, label = self._native_loras(ident), f"{cat} LoRAs"
                        else:
                            choices = discovery.lora_choices(family=fam)
                            label = f"{fam} LoRAs" if fam else "LoRAs (SDXL family)"
                        keep = {v for _, v in choices}
                        saved_loras = [v for v in (m.get("loras") or []) if v in keep]
                        ups.append(gr.update(choices=choices, label=label,
                                             value=saved_loras))
                    else:
                        ups.append(gr.update())
                else:
                    ups.append(gr.update(value=m[key]) if key in m else gr.update())
            # Trailing out_size updates (re-scoped to the restored model family).
            for mode, _comp in out_size_extra:
                mv = valid_model.get(mode)
                if mv:
                    ups.append(gr.update(choices=["Custom (use px below)"]
                                         + discovery.common_sizes(mv)))
                else:
                    ups.append(gr.update())
            # Trailing res_preset updates (re-scoped to the restored model family).
            for mode, _comp in res_preset_extra:
                mv = valid_model.get(mode)
                if mv:
                    ups.append(gr.update(choices=discovery.resolution_presets(mv),
                                         value=None))
                else:
                    ups.append(gr.update())
            return ups

        # (Result galleries are restored at BUILD time as each gallery's initial value
        # — see page._persisted_results / _results_block — which is more reliable than a
        # load event. Only the settings restore needs root.load.)
        try:
            from gradio.context import Context
            root = Context.root_block
            outputs = (comps + [comp for _, comp in out_size_extra]
                       + [comp for _, comp in res_preset_extra])
            if root is not None and comps:
                root.load(_restore, inputs=None, outputs=outputs)
        except Exception:
            traceback.print_exc()

    def _wire_page(self, mode, c):
        # On model change: (1) filter LoRAs to the model's family, (2) populate
        # the generation settings with the recommended values for that model.
        setting_keys = ["steps", "cfg", "sampler", "scheduler", "clip_skip",
                        "width", "height"]
        if "denoise" in c:
            setting_keys.append("denoise")

        def _on_model(mv):
            backend, ident = discovery.parse_model_value(mv)
            fam = discovery.model_family(mv)  # Pony/Illustrious/SDXL, or None for native
            if backend == "sd":
                lora_up = gr.update(
                    choices=discovery.lora_choices(family=fam), value=[],
                    interactive=True,
                    label=f"{fam} LoRAs" if fam else "LoRAs (SDXL family)")
            elif backend == "native":  # show that family's LoRA folder (Flux/Z-Image/Qwen)
                cat = discovery.categorize_native(ident) or "Native"
                lora_up = gr.update(
                    choices=self._native_loras(ident), value=[], interactive=True,
                    label=f"{cat} LoRAs")
            else:
                lora_up = gr.update(choices=[], value=[], interactive=True,
                                    label="LoRAs")
            rec = presets.for_model(mv, mode, self.get_default_settings)
            ups = [(gr.update(value=rec[k]) if k in rec else gr.update())
                   for k in setting_keys]
            result = [lora_up] + ups
            if "out_size" in c:  # outpaint target sizes follow the model family
                result.append(gr.update(choices=["Custom (use px below)"]
                                        + discovery.common_sizes(mv)))
            # Loading notice: SDXL-family checkpoints are big and give no feedback
            # while loading, so the first Generate looks frozen. Flag it on select.
            note = ("⏳ **SDXL / Pony / Illustrious selected.** The first **Generate** "
                    "loads the checkpoint (~6.5 GB) — it can sit ~30–60s with no step "
                    "counter while loading. That's normal, not a freeze; later gens "
                    "reuse it and are fast." if backend == "sd" else "")
            result.append(gr.update(value=note))
            # Resolution presets follow the model family (1:1 / portrait / landscape);
            # cleared to no selection so the per-model defaults (above) stand until
            # the user picks one.
            result.append(gr.update(choices=discovery.resolution_presets(mv),
                                    value=None))
            return result

        model_outputs = [c["loras"]] + [c[k] for k in setting_keys]
        if "out_size" in c:
            model_outputs.append(c["out_size"])
        model_outputs.append(c["gen_status"])
        model_outputs.append(c["res_preset"])
        # .input (user-only), NOT .change: a model set programmatically by the Prompt
        # Library Load (or persist-restore) must NOT trigger _on_model, which would
        # reset every setting to the model's defaults and clobber the loaded values.
        # (Gradio: handle_change dispatches "change" on backend updates too, but
        # "input" only on user interaction — verified in the dropdown source.)
        c["model"].input(_on_model, inputs=[c["model"]], outputs=model_outputs)

        # --- Resolution preset + aspect lock (shared by every page) ---------
        def _snap(v):  # clamp to the slider range and snap to its 64-px step
            return max(256, min(2048, int(round(float(v) / 64.0)) * 64))

        def _apply_res_preset(val, cur_ratio):
            """Preset 'W×H' → Width/Height; re-baseline the lock ratio to it. (Returns
            a plain ratio value, never gr.update(), since res_ratio is a gr.State.)"""
            w, h = discovery.parse_size(val or "")
            if not (w and h):
                return gr.update(), gr.update(), cur_ratio
            return gr.update(value=w), gr.update(value=h), w / h
        c["res_preset"].change(
            _apply_res_preset, inputs=[c["res_preset"], c["res_ratio"]],
            outputs=[c["width"], c["height"], c["res_ratio"]])

        def _set_lock(locked, w, h):
            """Capture the current W:H when Lock is engaged; clear it when released."""
            return (float(w) / float(h)) if (locked and w and h) else None
        c["res_lock"].change(_set_lock,
                             inputs=[c["res_lock"], c["width"], c["height"]],
                             outputs=[c["res_ratio"]])

        # While locked, dragging one slider scales the other to hold the ratio. We
        # use .release (fires once on mouse-up, not every drag tick), and a
        # programmatic value-set via a handler output does NOT re-fire the other
        # slider's .release — so width↔height can't loop.
        def _sync_h(locked, ratio, w):
            return (gr.update(value=_snap(float(w) / ratio))
                    if (locked and ratio) else gr.update())

        def _sync_w(locked, ratio, h):
            return (gr.update(value=_snap(float(h) * ratio))
                    if (locked and ratio) else gr.update())
        c["width"].release(_sync_h,
                           inputs=[c["res_lock"], c["res_ratio"], c["width"]],
                           outputs=[c["height"]])
        c["height"].release(_sync_w,
                            inputs=[c["res_lock"], c["res_ratio"], c["height"]],
                            outputs=[c["width"]])

        SET = [c["model"], c["sampler"], c["scheduler"], c["steps"], c["cfg"],
               c["clip_skip"], c["seed"], c["width"], c["height"], c["count"],
               c["loras"], c["lora_mult"]]
        gen_js = None
        if mode == "txt2img":
            gen_inputs = [self.state] + SET + [c["pos"], c["neg"]]
            fn = self._make_txt2img(mode)
        elif mode == "img2img":
            gen_inputs = [self.state] + SET + [c["denoise"], c["resize_mode"],
                                               c["pos"], c["neg"], c["input_image"]]
            fn = self._make_img2img(mode)
        else:  # inpaint — inputs come from the PaintShop canvas (composite + mask)
            gen_inputs = [self.state] + SET + [
                c["denoise"], c["feather"], c["mask_mode"], c["inpaint_fill"],
                c["inpaint_area"], c["padding"], c["pos"], c["neg"],
                c["composite"], c["mask"]]
            fn = self._make_inpaint(mode)
            # Pre-flush the canvas before reading composite/mask. Most edits push the
            # hidden fields through a 120ms-debounced pushExport (only pointer-up
            # flushes synchronously), so a Generate within 120ms of a slider/opacity/
            # grow/invert/undo/flip would otherwise send STALE pre-edit data. So force
            # a synchronous exportNow(), then re-read the freshly-written composite/
            # mask textareas (the last two inputs) and substitute them.
            # Gradio 5.29: a js= that returns undefined NULLS the inputs (model arrives
            # empty → "Select a model"), so we ...spread and RETURN every input value;
            # only composite/mask are replaced. Mirrors the Modify save fix (~2644).
            gen_js = (
                "(...args) => { try{ window.__is_inpaint_exportnow(); }catch(e){}"
                " function rd(id){ var el=document.querySelector('#'+id+' textarea')"
                "||document.querySelector('#'+id+' input'); return el?el.value:null; }"
                " var n=args.length;"
                " var comp=rd('imagesuite-inpaint-composite');"
                " if(comp!=null) args[n-2]=comp;"
                " var msk=rd('imagesuite-inpaint-mask');"
                " if(msk!=null) args[n-1]=msk;"
                " return args; }")

        c["generate"].click(
            fn, inputs=gen_inputs,
            outputs=[c["gallery"], c["picked"], c["save"]], js=gen_js)

        # Click a gallery result to make IT the active selection — so Send-to /
        # Save As / the enhancement passes act on whichever image you pick, not just
        # the first. Safe: the gallery is fed PIL images, so Gradio owns each entry
        # as a CACHE path, and the select payload's path is therefore inside the
        # Gradio cache — which both the stock check_all_files_in_cache and our
        # allow-list patch (_cache_allow_dirs) accept. 'picked' is still armed to the
        # first result at generation time, so it's correct before any click.
        def _pick(evt: gr.EventData):
            # gr.SelectData.__init__ KeyErrors when Gradio sends a select event
            # without "value"; read the raw payload off gr.EventData instead.
            data = getattr(evt, "_data", {}) or {}
            v = data.get("value") if isinstance(data, dict) else None
            p = None
            if isinstance(v, dict):
                p = (v.get("image") or {}).get("path") or v.get("path")
            elif isinstance(v, str):
                p = v
            return (p, gr.update(value=p)) if p else (gr.update(), gr.update())
        c["gallery"].select(_pick, outputs=[c["picked"], c["save"]])

        # Qwen-abliterated prompt enhance (Wan2GP's native enhancer).
        if "enhance_pos" in c:
            c["enhance_pos"].click(self._enhance, inputs=[self.state, c["pos"]],
                                   outputs=[c["pos"]])
            c["enhance_neg"].click(self._enhance, inputs=[self.state, c["neg"]],
                                   outputs=[c["neg"]])

        # Interrogate the page's image → prompt. WD14 tags for booru families
        # (Pony/Illustrious), BLIP caption otherwise. Image source per page:
        # txt2img = selected result, img2img = init image, inpaint = canvas.
        if "interrogate" in c:
            src = {"txt2img": c.get("picked"), "img2img": c.get("input_image"),
                   "inpaint": c.get("composite")}.get(mode)
            if src is not None:
                # Wire directly (functools.partial binds the mode) instead of a
                # lambda, so Gradio injects its progress object into _interrogate's
                # trailing progress= param — the lambda would drop it.
                c["interrogate"].click(
                    functools.partial(self._interrogate, mode=mode),
                    inputs=[self.state, c["model"], src], outputs=[c["pos"]])

        # Face/body swap, ADetailer (face+body), colour reference — Run on selected.
        if "face_run" in c:
            self._wire_enhance(c)

        # Abort the running gen (native cancel + SD abort flag).
        if "abort" in c:
            c["abort"].click(self._abort, inputs=[self.state])

        # Touch-Up (inpaint canvas only): outpaint + resize + crop&resize.
        if "out_run" in c:
            self._wire_touchup(mode, c)

        # Overlays strip: load a folder's thumbnails into the canvas strip.
        if "ov_folder" in c:
            push = lambda folder, _mode=mode: self._push_overlays(folder, _mode)
            c["ov_folder"].change(push, inputs=[c["ov_folder"]], outputs=[c["ov_bridge"]])
            c["ov_reload"].click(push, inputs=[c["ov_folder"]], outputs=[c["ov_bridge"]])

    def _wire_touchup(self, mode, c):
        out = [c["gallery"], c["picked"], c["save"]]
        c["out_run"].click(
            self._make_outpaint(mode),
            inputs=[self.state, c["model"], c["sampler"], c["scheduler"], c["steps"],
                    c["cfg"], c["clip_skip"], c["seed"], c["loras"], c["lora_mult"],
                    c["out_feather"], c["pos"], c["neg"], c["composite"], c["out_size"],
                    c["out_top"], c["out_bottom"], c["out_left"], c["out_right"]],
            outputs=out)
        c["out_abort"].click(self._abort, inputs=[self.state])

    def _make_txt2img(self, mode):
        import random as _rng
        api = self._api  # closed over so the click is webui-wrapped (see _callback_uses_api_session)

        def _run(state, model, sampler, scheduler, steps, cfg, clip_skip, seed,
                 width, height, count, loras, mult, pos, neg, progress=gr.Progress()):
            if not (pos and pos.strip()):
                raise gr.Error("Enter a prompt first.")
            backend, ident = discovery.parse_model_value(model)
            if not backend:
                raise gr.Error("Select a model from the dropdown first.")
            gen_sd.clear_abort()
            files, n = [], int(count)
            if backend == "native":
                if self._gpu_busy(state):
                    raise gr.Error("A generation is already running — wait for it to finish.")
                self._release_all_vram()  # free ALL our GPU consumers (SD + inpaint/IP + seg + faceswap)
                snap = self._snapshot_video_state(state)
                try:
                    for i in range(n):
                        progress((i, n), desc=f"Generating {i + 1}/{n}")
                        files += self._gen_native(ident, pos, neg, width, height, steps,
                                                  cfg, (int(seed) + i if int(seed) >= 0 else seed),
                                                  mode="txt2img",
                                                  loras=loras, mult_str=mult,
                                                  progress=progress, api=api, state=state)
                finally:
                    self._restore_video_state(state, snap)
                    self._release_native_model()  # don't leave the native model stacked under a later SDXL load
            else:  # sd
                lora_list = self._lora_list(loras, mult)
                self.acquire_gpu(state)
                try:
                    self._sd_loading(progress)
                    nsteps, total = int(steps), max(1, n) * max(1, int(steps))
                    for i in range(n):
                        if gen_sd.was_aborted():
                            break
                        sd = (int(seed) + i) if int(seed) >= 0 else _rng.randint(0, 2**31 - 1)
                        files += gen_sd.generate_txt2img(
                            ident, pos, neg, width, height, steps, cfg, sd,
                            sampler=sampler or "DPM++ 2M", scheduler=scheduler or "",
                            clip_skip=int(clip_skip), loras=lora_list,
                            callback=self._sd_step_cb(progress, i, nsteps, total))
                finally:
                    self.release_gpu(state)
            if gen_sd.was_aborted():
                raise gr.Error("Txt2img aborted.")
            if not files:
                raise gr.Error("Generation produced no images.")
            return self._gallery_result(files, mode)
        return _run

    def _make_img2img(self, mode):
        import random as _rng
        api = self._api  # closed over so the click is webui-wrapped

        RESIZE = {"Just resize": 0, "Crop and resize": 1, "Resize and fill": 2}

        def _run(state, model, sampler, scheduler, steps, cfg, clip_skip, seed,
                 width, height, count, loras, mult, denoise, resize_mode, pos, neg,
                 init_image, progress=gr.Progress()):
            backend, ident = discovery.parse_model_value(model)
            if not backend:
                raise gr.Error("Select a model from the dropdown first.")
            if not init_image:
                raise gr.Error("Load an init image first.")
            if not (pos and pos.strip()):
                raise gr.Error("Enter a prompt first.")
            gen_sd.clear_abort()
            files, n = [], int(count)
            if backend == "native":
                if self._gpu_busy(state):
                    raise gr.Error("A generation is already running — wait for it to finish.")
                self._release_all_vram()  # free ALL our GPU consumers (SD + inpaint/IP + seg + faceswap)
                snap = self._snapshot_video_state(state)
                try:
                    for i in range(n):
                        progress((i, n), desc=f"Reimagining {i + 1}/{n}")
                        files += self._gen_native(ident, pos, neg, width, height, steps,
                                                  cfg, (int(seed) + i if int(seed) >= 0 else seed),
                                                  mode="img2img",
                                                  denoise=denoise, guide_path=init_image,
                                                  loras=loras, mult_str=mult,
                                                  progress=progress, api=api, state=state)
                finally:
                    self._restore_video_state(state, snap)
                    self._release_native_model()  # don't leave the native model stacked under a later SDXL load
            else:  # sd
                lora_list = self._lora_list(loras, mult)
                self.acquire_gpu(state)
                try:
                    self._sd_loading(progress)
                    nsteps, total = int(steps), max(1, n) * max(1, int(steps))
                    for i in range(n):
                        if gen_sd.was_aborted():
                            break
                        sd = (int(seed) + i) if int(seed) >= 0 else _rng.randint(0, 2**31 - 1)
                        progress((i, n), desc=f"Reimagining {i + 1}/{n}")
                        files += gen_sd.generate_img2img(
                            ident, init_image, pos, neg, width, height, steps, cfg, sd,
                            denoise=float(denoise), sampler=sampler or "DPM++ 2M",
                            scheduler=scheduler, resize_mode=RESIZE.get(resize_mode, 0),
                            clip_skip=int(clip_skip), loras=lora_list,
                            callback=self._sd_step_cb(progress, i, nsteps, total))
                finally:
                    self.release_gpu(state)
            # An abort interrupts the diffusers loop mid-image, leaving a partial
            # (or no) result — discard it rather than presenting a half-denoised image.
            if gen_sd.was_aborted():
                raise gr.Error("img2img aborted.")
            if not files:
                raise gr.Error("img2img produced no images.")
            return self._gallery_result(files, mode)
        return _run

    # generate_inpaint's inpainting_fill code: 0 fill / 1 original /
    # 2 latent-noise / 3 latent-nothing (mirrors A1111 "Masked content").
    _FILL = {"fill": 0, "original": 1, "latent noise": 2, "latent nothing": 3}

    def _inpaint_core(self, state, backend, ident, comp, mask, pos, neg, *,
                      denoise, steps, cfg, clip_skip, seed, sampler, scheduler,
                      loras, mult, feather=4, inpaint_fill="original",
                      full_res=False, padding=32, count=1, api=None, progress=None):
        """``count`` inpaint passes over a prepared composite(RGB) + mask(L), under a
        SINGLE GPU acquisition — one busy-check, a continuous progress bar across the
        batch, abort honoured between images (and the in-flight partial dropped), and
        a fresh random seed per image when ``seed`` < 0. Shared by the canvas Generate
        (count = Batch count) and the Touch-Up outpaint (count 1). Returns a LIST of
        result paths (empty if aborted before the first completed)."""
        import random as _rng
        n = max(1, int(count))

        def _say(f, d):
            if progress is not None:
                try: progress(f, desc=d)
                except Exception: pass
        if backend == "native":
            if self._gpu_busy(state):
                raise gr.Error("A generation is already running — wait for it to finish.")
            self._release_all_vram()  # free ALL our GPU consumers (SD + inpaint/IP + seg + faceswap)
            m = mask
            if feather and int(feather) > 0:  # native has no pipeline mask_blur
                from PIL import ImageFilter
                m = mask.filter(ImageFilter.GaussianBlur(radius=int(feather)))
            guide_path = self._save_img(comp, "composite")  # same for every image
            mask_path = self._save_img(m, "mask")
            files, snap = [], self._snapshot_video_state(state)
            try:
                for i in range(n):
                    if gen_sd.was_aborted():
                        break
                    _say((i, n), f"Inpainting {i + 1}/{n}…")
                    # _gen_native randomizes seed<0 itself; for a fixed seed we add
                    # the pass index so a batch yields varied images, not duplicates.
                    files += self._gen_native(
                        ident, pos, neg, comp.width, comp.height, steps, cfg,
                        (int(seed) + i if int(seed) >= 0 else seed),
                        mode="inpaint", denoise=denoise, guide_path=guide_path,
                        mask_path=mask_path, loras=loras, mult_str=mult,
                        progress=progress, api=api, state=state)
            finally:
                self._restore_video_state(state, snap)
                self._release_native_model()  # don't leave the native model stacked under a later SDXL load
            return files
        # sd — one acquisition, loop count times for a real batch.
        lora_list = self._lora_list(loras, mult)
        self.acquire_gpu(state)
        files = []
        try:
            self._sd_loading(progress)
            nsteps, total = int(steps), n * max(1, int(steps))
            for i in range(n):
                if gen_sd.was_aborted():
                    break
                sd = (int(seed) + i) if int(seed) >= 0 else _rng.randint(0, 2**31 - 1)
                cb = (self._sd_step_cb(progress, i, nsteps, total)
                      if progress is not None else None)
                outs = gen_sd.inpaint(
                    ident, comp, mask, pos, neg, denoise=float(denoise),
                    steps=int(steps), cfg=float(cfg), seed=int(sd),
                    sampler=sampler or "DPM++ 2M", scheduler=scheduler or "Karras",
                    clip_skip=int(clip_skip), mask_blur=int(feather),
                    inpainting_fill=self._FILL.get(inpaint_fill, 1),
                    full_res=bool(full_res), padding=int(padding),
                    progress=progress, loras=lora_list, callback=cb)
                # Abort fired mid-image → drop this partial, keep the completed ones.
                if gen_sd.was_aborted():
                    break
                if outs:
                    files += outs
        finally:
            self.release_gpu(state)
        return files

    def _make_inpaint(self, mode):
        api = self._api  # closed over so the click is webui-wrapped

        def _run(state, model, sampler, scheduler, steps, cfg, clip_skip, seed,
                 width, height, count, loras, mult, denoise, feather, mask_mode,
                 inpaint_fill, inpaint_area, padding, pos, neg,
                 composite_url, mask_url, progress=gr.Progress()):
            backend, ident = discovery.parse_model_value(model)
            if not backend:
                raise gr.Error("Select a model from the dropdown first.")
            gen_sd.clear_abort()
            comp = self._decode_dataurl(composite_url)
            mask = self._decode_dataurl(mask_url)
            # Missing canvas/mask is normal user-input validation, not an error —
            # show a popup (gr.Warning) and no-op rather than raising (which the webui
            # worker re-raises into a console traceback).
            if comp is None:
                gr.Warning("Load an image into the canvas first (upload one or use a "
                           "'Send to MultiCanvas' button).")
                return gr.update(), gr.update(), gr.update()
            if mask is None or not self._mask_nonempty(mask):
                gr.Warning("Mark a mask region first — use Mask mode (paint the area "
                           "to change) or turn on Auto-mask.")
                return gr.update(), gr.update(), gr.update()
            comp = comp.convert("RGB")
            mask = mask.convert("L")
            if mask_mode == "Inpaint not masked":
                from PIL import ImageOps
                mask = ImageOps.invert(mask)
            outs = self._inpaint_core(
                state, backend, ident, comp, mask, pos, neg, denoise=denoise,
                steps=steps, cfg=cfg, clip_skip=clip_skip, seed=seed,
                sampler=sampler, scheduler=scheduler, loras=loras, mult=mult,
                feather=feather, inpaint_fill=inpaint_fill,
                full_res=(inpaint_area == "Only masked"), count=count,
                padding=padding, api=api, progress=progress)
            if not outs:
                raise gr.Error("Inpaint aborted." if gen_sd.was_aborted()
                               else "Inpaint produced no image.")
            return self._gallery_result(outs, mode)
        return _run

    def _make_outpaint(self, mode):
        """Touch-Up outpaint: extend the canvas image outward (edge-replicated for
        context) and fill the new border with the model. Reuses _inpaint_core at
        denoise 1.0 (the new pixels have no original content to preserve)."""
        api = self._api  # closed over so the click is webui-wrapped

        def _run(state, model, sampler, scheduler, steps, cfg, clip_skip, seed,
                 loras, mult, feather, pos, neg, composite_url, out_size,
                 top, bottom, left, right, progress=gr.Progress()):
            backend, ident = discovery.parse_model_value(model)
            if not backend:
                raise gr.Error("Select a model from the dropdown first.")
            comp = self._decode_dataurl(composite_url)
            if comp is None:
                raise gr.Error("Load an image into the canvas first.")
            comp = comp.convert("RGB")
            # A chosen target size (family dropdown) outpaints CENTERED to that
            # canvas; "Custom" uses the per-side px sliders.
            tw, th = discovery.parse_size(out_size)
            if tw and th:
                ew, eh = max(0, tw - comp.width), max(0, th - comp.height)
                l, r = ew // 2, ew - ew // 2
                t, b = eh // 2, eh - eh // 2
                if ew == 0 and eh == 0:
                    raise gr.Error("That size isn't larger than the current image "
                                   "in any dimension — pick a bigger target.")
            else:
                t, b, l, r = (max(0, int(x)) for x in (top, bottom, left, right))
                if t + b + l + r == 0:
                    raise gr.Error("Pick a target size, or set a per-side px amount.")
            import numpy as np
            from PIL import Image
            # Edge-replicate the original into the new border so the model has
            # local context to extend from, then mask exactly the new region.
            ext = Image.fromarray(
                np.pad(np.array(comp), ((t, b), (l, r), (0, 0)), mode="edge"))
            mask = Image.new("L", ext.size, 255)
            mask.paste(Image.new("L", comp.size, 0), (l, t))
            gen_sd.clear_abort()
            outs = self._inpaint_core(  # outpaint is always a single pass (count 1)
                state, backend, ident, ext, mask, pos, neg, denoise=1.0,
                steps=steps, cfg=cfg, clip_skip=clip_skip, seed=seed,
                sampler=sampler, scheduler=scheduler, loras=loras, mult=mult,
                feather=max(8, int(feather)), inpaint_fill="fill",
                full_res=False, padding=0, api=api, progress=progress)
            if not outs:
                raise gr.Error("Outpaint aborted." if gen_sd.was_aborted()
                               else "Outpaint produced no image.")
            return self._gallery_result(outs, mode)
        return _run

    # -- right-click context menu router -----------------------------------
    @staticmethod
    def _allowed_local_file(path) -> str | None:
        """Return path iff it's an existing file whose realpath resolves inside the
        cache/outputs allow-list — so a client-supplied src can't read arbitrary
        local files (e.g. /etc/passwd, ~/.ssh/id_rsa) once Wan2GP is exposed."""
        import os
        if not path:
            return None
        try:
            rp = os.path.realpath(path)
        except Exception:
            return None
        if not os.path.isfile(rp):
            return None
        for base in ImageSuite._cache_allow_dirs():
            try:
                if os.path.commonpath([rp, base]) == base:
                    return rp
            except (ValueError, OSError):
                continue
        return None

    @staticmethod
    def _resolve_safe_addr(host, port):
        """getaddrinfo(host, port) iff EVERY resolved IP is a public/global address;
        else None. Blocks SSRF to loopback / private / link-local / reserved /
        multicast targets (e.g. 127.0.0.1, 169.254.169.254 cloud-metadata, 10/8)
        once the app is exposed via --listen/--share. Best-effort: validates at
        resolve time; full DNS-rebinding defence would need connect-time IP pinning."""
        import ipaddress
        import socket
        try:
            infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except Exception:
            return None
        for info in infos:
            try:
                addr = ipaddress.ip_address(info[4][0])
            except ValueError:
                return None
            m = getattr(addr, "ipv4_mapped", None) or addr  # unwrap ::ffff:1.2.3.4
            if (m.is_private or m.is_loopback or m.is_link_local or m.is_reserved
                    or m.is_multicast or m.is_unspecified):
                return None
        return infos or None

    def _resolve_src(self, src) -> str | None:
        """A media src from the context menu → a local file path. Handles PNG
        data-URLs (canvas export), Gradio-served '…file=<path>' URLs, plain local
        paths (containment-checked against the cache/outputs allow-list), and
        remote http(s) URLs (downloaded to the cache, capped + timed out)."""
        import base64
        import os
        import urllib.parse
        import urllib.request
        if not src:
            return None
        try:
            if src.startswith("data:image/") and ";base64," in src:
                b64 = src.split(",", 1)[1]
                if len(b64) > ImageSuite._MAX_DATAURL_BYTES:
                    raise gr.Error("That image is too large to process.")
                try:
                    raw = base64.b64decode(b64)
                except Exception:
                    return None
                # Validate it really decodes as an image under the pixel cap before
                # persisting (no decompression-bomb).
                import io
                from PIL import Image
                prev = Image.MAX_IMAGE_PIXELS
                Image.MAX_IMAGE_PIXELS = ImageSuite._MAX_DECODE_PIXELS
                try:
                    with Image.open(io.BytesIO(raw)) as _im:
                        _im.load()
                except Image.DecompressionBombError:
                    raise gr.Error("That image is too large to process.")
                except Exception:
                    return None
                finally:
                    Image.MAX_IMAGE_PIXELS = prev
                out = paths.cache_dir() / "ctx"
                out.mkdir(parents=True, exist_ok=True)
                p = out / f"ctx_{int(time.time()*1000)}.png"
                p.write_bytes(raw)
                return str(p)
            if "file=" in src:
                path = urllib.parse.unquote(src.split("file=", 1)[1].split("&")[0])
                allowed = self._allowed_local_file(path)
                if allowed:
                    return allowed
            allowed = self._allowed_local_file(src)
            if allowed:
                return allowed
            parsed = urllib.parse.urlparse(src)
            scheme = parsed.scheme.lower()
            if scheme in ("http", "https"):
                import urllib.error
                host = parsed.hostname
                port = parsed.port or (443 if scheme == "https" else 80)
                # SSRF guard: only fetch PUBLIC hosts on the standard web ports, so an
                # exposed instance can't be coerced via a client-supplied context-menu
                # URL into hitting internal / cloud-metadata endpoints (169.254.169.254,
                # localhost, 10/8, …). Redirects are re-validated by the handler below.
                if port not in (80, 443) or not host or not self._resolve_safe_addr(host, port):
                    raise gr.Error("That URL isn't allowed — only public http(s) image "
                                   "hosts on standard ports can be fetched.")
                out = paths.cache_dir() / "ctx"
                out.mkdir(parents=True, exist_ok=True)
                ext = os.path.splitext(parsed.path)[1] or ".png"
                p = out / f"ctx_{int(time.time()*1000)}{ext}"
                # Bounded fetch: connect timeout + a hard size cap so a hostile/large
                # URL can't hang the handler or fill the disk.
                MAX_BYTES = 64 * 1024 * 1024

                class _NoSSRFRedirect(urllib.request.HTTPRedirectHandler):
                    # A public URL must not 30x-bounce the fetch into a private target.
                    def redirect_request(self, rq, fp, code, msg, hdrs, newurl):
                        pr = urllib.parse.urlparse(newurl)
                        pt = pr.port or (443 if pr.scheme.lower() == "https" else 80)
                        if (pr.scheme.lower() not in ("http", "https") or pt not in (80, 443)
                                or not pr.hostname
                                or not ImageSuite._resolve_safe_addr(pr.hostname, pt)):
                            raise urllib.error.URLError("redirect to a disallowed host")
                        return super().redirect_request(rq, fp, code, msg, hdrs, newurl)

                req = urllib.request.Request(src, headers={"User-Agent": "ImageSuite"})
                opener = urllib.request.build_opener(_NoSSRFRedirect())
                with opener.open(req, timeout=15) as resp:
                    clen = resp.headers.get("Content-Length")
                    if clen and int(clen) > MAX_BYTES:
                        raise gr.Error("That remote image is too large.")
                    data = resp.read(MAX_BYTES + 1)
                if len(data) > MAX_BYTES:
                    raise gr.Error("That remote image is too large.")
                p.write_bytes(data)
                return str(p)
        except gr.Error:
            raise
        except Exception:
            traceback.print_exc()
        return None

    def _wire_ctx(self, ui):
        pages, subtabs, tab_ids = ui["pages"], ui["subtabs"], ui["tab_ids"]
        i2i_input = pages["img2img"]["input_image"]
        inp_bridge = pages["inpaint"]["bg_bridge"]

        def _route(state, payload_json):
            import json
            noop = gr.update()
            outs = [noop, noop, noop, noop, noop]  # i2i_input, inp_bridge, subtabs, main_tabs, refresh
            try:
                d = json.loads(payload_json or "{}")
            except Exception:
                d = {}
            action, src = d.get("a"), d.get("s")  # relay payload {a:action, s:src, t:nonce}
            if not action or not src:
                return outs
            path = self._resolve_src(src)
            if not path:
                gr.Warning("Couldn't load that image.")
                return outs
            if action == "img2vid":
                ts, nav = self._send_to_img2vid(state, path)
                return [noop, noop, noop, nav, ts]
            if action == "img2img":
                return [path, noop, gr.update(selected=tab_ids["img2img"]),
                        gr.update(selected=PLUGIN_ID), noop]
            if action == "inpaint":
                html = canvas.bg_bridge_html(self._file_to_dataurl(path), "inpaint",
                                             nonce=time.time())
                return [noop, html, gr.update(selected=tab_ids["inpaint"]),
                        gr.update(selected=PLUGIN_ID), noop]
            return outs

        self._ctx_relay.change(
            _route, inputs=[self.state, self._ctx_relay],
            outputs=[i2i_input, inp_bridge, subtabs, self.main_tabs,
                     self.refresh_form_trigger])

    # Settings every page shares — carried verbatim by the Send-To buttons.
    # Mode-only fields (denoise / feather / inpaint_fill / …) are handled apart.
    CARRY_KEYS = ["model", "sampler", "scheduler", "steps", "cfg", "clip_skip",
                  "seed", "width", "height", "count", "loras", "lora_mult",
                  "pos", "neg"]

    def _carry_settings(self, target_mode, vals):
        """Source page values (in CARRY_KEYS order) → a list of gr.update for the
        target page's matching components, PLUS one trailing res_preset update (the
        target's _on_model won't fire on a programmatic model set). The model is
        applied only if the target offers it (Img2Img/Inpaint list only guide-capable
        models), and LoRAs/res_preset are re-scoped to the carried model's family so
        their values stay valid."""
        d = dict(zip(self.CARRY_KEYS, vals))
        valid = {v for _, v in self._model_choices(target_mode)}
        ups = {}
        mv = d["model"]
        if mv and mv in valid:
            ups["model"] = gr.update(value=mv)
            backend, ident = discovery.parse_model_value(mv)
            lchoices = (self._native_loras(ident) if backend == "native"
                        else discovery.lora_choices(family=discovery.model_family(mv)))
            keep = {c for _, c in lchoices}
            ups["loras"] = gr.update(
                choices=lchoices, value=[v for v in (d["loras"] or []) if v in keep])
        else:
            if mv:
                gr.Warning("That model isn't offered on the target tab (Img2Img / "
                           "Inpaint list only guide-capable models) — kept its model.")
            ups["model"] = gr.update()
            ups["loras"] = gr.update()
        for k in ("sampler", "scheduler", "steps", "cfg", "clip_skip", "seed",
                  "width", "height", "count", "lora_mult", "pos", "neg"):
            ups[k] = gr.update(value=d[k])
        # res_preset choices follow the carried model's family; cleared to no
        # selection so the carried width/height stand. Trailing (not in CARRY_KEYS).
        if mv and mv in valid:
            res_up = gr.update(choices=discovery.resolution_presets(mv), value=None)
        else:
            res_up = gr.update()
        return [ups[k] for k in self.CARRY_KEYS] + [res_up]

    def _wire_carry(self, btn, src, dst_mode, dst, subtabs, tab_ids,
                    with_image=False, bridge=False):
        """Wire a Send-To button to carry model + all params + prompt from the
        source page to the target page (and optionally the picked image into an
        init-image slot, or onto the inpaint canvas), then switch sub-tab."""
        keys = self.CARRY_KEYS
        has_denoise = "denoise" in src and "denoise" in dst
        inputs = [src[k] for k in keys]
        if has_denoise:
            inputs.append(src["denoise"])
        if with_image or bridge:
            inputs.append(src["picked"])
        # Mirrors _carry_settings' trailing res_preset update (re-scoped per model).
        outputs = [dst[k] for k in keys] + [dst["res_preset"]]
        if has_denoise:
            outputs.append(dst["denoise"])
        if with_image:
            outputs.append(dst["input_image"])
        if bridge:
            outputs.append(dst["bg_bridge"])
        outputs.append(subtabs)

        def _h(*vals):
            vals = list(vals)
            picked = vals.pop() if (with_image or bridge) else None
            dn = vals.pop() if has_denoise else None
            ups = self._carry_settings(dst_mode, vals)
            if has_denoise:
                ups.append(gr.update(value=dn))
            if with_image:
                ups.append(picked or gr.update())
            if bridge:
                if not picked:
                    raise gr.Error("Select a result first.")
                ups.append(canvas.bg_bridge_html(
                    self._file_to_dataurl(picked), "inpaint", nonce=time.time()))
            ups.append(gr.update(selected=tab_ids[dst_mode]))
            return ups
        btn.click(_h, inputs=inputs, outputs=outputs)

    # -- cross-page + Img2Vid sends ----------------------------------------
    def _wire_image_send(self, btn, src, dst, dst_mode, subtabs, tab_ids, target):
        """Send the picked result IMAGE without carrying any generation settings —
        used by the settings-less Modify page and by every page's 'Modify' button.
        target: 'image' (into an init-image slot) | 'inpaint_bridge' | 'modify_bridge'
        (load onto a canvas via its bg bridge), then switch sub-tab."""
        if target == "image":
            outs = [dst["input_image"], subtabs]

            def _h(picked):
                if not picked:
                    raise gr.Error("Select a result first.")
                return picked, gr.update(selected=tab_ids[dst_mode])
        else:
            outs = [dst["bg_bridge"], subtabs]
            fam = "modify" if target == "modify_bridge" else "inpaint"
            bridge = (modify_canvas.modify_bg_bridge_html if target == "modify_bridge"
                      else canvas.bg_bridge_html)

            def _h(picked):
                if not picked:
                    raise gr.Error("Select a result first.")
                return (bridge(self._file_to_dataurl(picked), fam, nonce=time.time()),
                        gr.update(selected=tab_ids[dst_mode]))
        btn.click(_h, inputs=[src["picked"]], outputs=outs)

    def _wire_modify_page(self, c):
        """The Modify tab: load an image into the editor, colour-match it to a
        reference, and save the edited (cropped + colour-corrected) result into the
        gallery. Crop/zoom/colour-correction live in the editor iframe; Python only
        loads images in and reads the exported data-URL (c['out']) back out."""
        import base64
        import io
        from PIL import Image

        def _durl_to_pil(durl):
            if not durl or "," not in durl:
                return None
            try:
                raw = base64.b64decode(durl.split(",", 1)[1])
                return Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                traceback.print_exc()
                return None

        # Load an uploaded / dropped image into the editor.
        def _load_input(path):
            if not path:
                return gr.update()
            return modify_canvas.modify_bg_bridge_html(
                self._file_to_dataurl(path), "modify", nonce=time.time())
        c["mod_input"].change(_load_input, inputs=[c["mod_input"]],
                              outputs=[c["bg_bridge"]])

        # Colour-match the current edited image to a reference, then reload it onto the
        # canvas so further crop/colour edits stack. The js flushes the editor's export
        # first so c['out'] is current, not the debounced previous value.
        def _match(edited, ref_path):
            src = _durl_to_pil(edited)
            if src is None:
                raise gr.Error("Load an image into the Modify canvas first.")
            if not ref_path:
                raise gr.Error("Pick a reference image first.")
            try:
                import cv2
                import numpy as np
                from .core import faceswap
                ref = Image.open(ref_path).convert("RGB")
                src_bgr = cv2.cvtColor(np.asarray(src), cv2.COLOR_RGB2BGR)
                ref_bgr = cv2.cvtColor(np.asarray(ref), cv2.COLOR_RGB2BGR)
                out_bgr = faceswap._color_transfer_lab(src_bgr, ref_bgr)
                out = Image.fromarray(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
            except Exception as e:
                traceback.print_exc()
                raise gr.Error(f"Colour match failed: {e}")
            buf = io.BytesIO()
            out.save(buf, format="PNG")
            durl = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            return (modify_canvas.modify_bg_bridge_html(durl, "modify", nonce=time.time()),
                    "🎨 Colour-matched to the reference.")
        c["mod_match"].click(
            _match, inputs=[c["out"], c["mod_ref"]],
            outputs=[c["bg_bridge"], c["mod_status"]],
            js=("(edited, ref) => { try{ window.__is_modify_exportnow(); }catch(e){} "
                "var el=document.querySelector('#imagesuite-modify-out textarea')"
                "||document.querySelector('#imagesuite-modify-out input'); "
                "return [el ? el.value : edited, ref]; }"))

        # Save the edited image into the results gallery (so send-to can act on it).
        def _save(edited):
            img = _durl_to_pil(edited)
            if img is None:
                raise gr.Error("Nothing to save — load and edit an image first.")
            gallery, picked, save = self._gallery_result(
                [self._save_img(img, "modify")], "modify")
            return gallery, picked, save, "💾 Saved to results."
        # js MUST return the inputs — a js= that returns undefined nulls them (Gradio
        # 5.29). So flush the editor's export, then read the freshly-written hidden
        # textbox and return it as `edited` (else _save receives None → saves nothing).
        c["mod_save"].click(
            _save, inputs=[c["out"]],
            outputs=[c["gallery"], c["picked"], c["save"], c["mod_status"]],
            js=("(edited) => { try{ window.__is_modify_exportnow(); }catch(e){} "
                "var el=document.querySelector('#imagesuite-modify-out textarea')"
                "||document.querySelector('#imagesuite-modify-out input'); "
                "return el ? el.value : edited; }"))

        # Gallery select → picked (mirrors _wire_page's _pick) so send-to / Save As act
        # on whichever saved result is clicked.
        def _pick(evt: gr.EventData):
            data = getattr(evt, "_data", {}) or {}
            v = data.get("value") if isinstance(data, dict) else None
            p = None
            if isinstance(v, dict):
                p = (v.get("image") or {}).get("path") or v.get("path")
            elif isinstance(v, str):
                p = v
            return (p, gr.update(value=p)) if p else (gr.update(), gr.update())
        c["gallery"].select(_pick, outputs=[c["picked"], c["save"]])

    def _wire_sends(self, ui):
        pages, subtabs, tab_ids = ui["pages"], ui["subtabs"], ui["tab_ids"]

        for mode, c in pages.items():
            if mode == "modify":
                # Modify has no model/prompt to carry — send the edited result IMAGE only.
                self._wire_image_send(c["to_i2i"], c, pages["img2img"], "img2img",
                                      subtabs, tab_ids, "image")
                self._wire_image_send(c["to_inp"], c, pages["inpaint"], "inpaint",
                                      subtabs, tab_ids, "inpaint_bridge")
            else:
                # → Img2Img: carry model+params+prompt + drop the picked image in.
                if "to_i2i" in c and c["to_i2i"].visible:
                    self._wire_carry(c["to_i2i"], c, "img2img", pages["img2img"],
                                     subtabs, tab_ids, with_image=True)

                # → MultiCanvas: carry settings + load the picked image as the canvas bg.
                if "to_inp" in c and c["to_inp"].visible:
                    self._wire_carry(c["to_inp"], c, "inpaint", pages["inpaint"],
                                     subtabs, tab_ids, bridge=True)

                # → Txt2Img: carry model + all params + prompt (no image slot).
                if "to_t2i" in c and c["to_t2i"].visible:
                    self._wire_carry(c["to_t2i"], c, "txt2img", pages["txt2img"],
                                     subtabs, tab_ids)

            # → Modify: load the picked image onto the Modify canvas (every page).
            if "to_mod" in c and c["to_mod"].visible:
                self._wire_image_send(c["to_mod"], c, pages["modify"], "modify",
                                      subtabs, tab_ids, "modify_bridge")

            # → Img2Vid: hand the picked image to the Video Generator as its start frame.
            c["to_i2v"].click(self._send_to_img2vid, inputs=[self.state, c["picked"]],
                             outputs=[self.refresh_form_trigger, self.main_tabs])

    # -- Settings + Models panel -------------------------------------------
    def _wire_settings(self, ui):
        s, pages = ui["settings"], ui["pages"]
        from .ui import settings_panel

        # Rescan rebuilds every page's model + LoRA dropdown choices. Each page's
        # model list is mode-specific (txt2img shows all; img2img/inpaint only
        # guide-capable), so build per-mode updates in pages-dict order.
        modes = [m for m in pages if "model" in pages[m]]   # gen pages only (skip Modify)
        model_dds = [pages[m]["model"] for m in modes]
        lora_dds = [pages[m]["loras"] for m in modes]

        def _model_updates():
            return [gr.update(choices=self._model_choices(m)) for m in modes]

        def _fresh_choices():
            lc = discovery.lora_choices()
            return _model_updates() + [gr.update(choices=lc) for _ in lora_dds]

        # Low-VRAM radio: persist + refilter every page's model dropdown AND the
        # native-download dropdown (it follows the same filter).
        def _set_vram(vram_val):
            paths.set_low_vram_only(settings_panel.vram_is_low(vram_val))
            return _model_updates() + [gr.update(choices=self._native_dl_choices())]
        s["vram_mode"].change(_set_vram, inputs=[s["vram_mode"]],
                             outputs=model_dds + [s["native_key"]])

        # Right-click menu scope toggle: persist + apply LIVE (no reload). The js
        # rewrites the registered items' match in place; Python just remembers the choice.
        def _set_ctx_scope(v):
            paths.set_ctx_plugin_only(bool(v))
        s["ctx_scope"].change(
            _set_ctx_scope, inputs=[s["ctx_scope"]],
            js="(v) => { try{ if(window.__imagesuiteScope) window.__imagesuiteScope(!!v); }"
               "catch(e){} return [v]; }")

        # Native model manual download.
        def _dl_native(model_type, progress=gr.Progress()):
            if not model_type:
                raise gr.Error("Pick a native model to download first.")
            try:
                ok = self._download_native(model_type, progress=progress)
            except Exception as e:
                traceback.print_exc()
                return f"⚠️ Download failed for {model_type}: {e}", gr.update()
            msg = (f"✅ {model_type} is ready." if ok else
                   f"Downloaded what was available for {model_type}; Wan2GP will "
                   "finish any remaining files on first generation.")
            return msg, gr.update(choices=self._native_dl_choices(), value=model_type)
        s["native_download"].click(_dl_native, inputs=[s["native_key"]],
                                  outputs=[s["native_log"], s["native_key"]])

        # Flush Outputs — reclaim disk from orphaned generations (sd_gen + inpaint).
        # Projects keep their own copies and galleries restore from persist/results,
        # so this never touches a project or the current view. JS confirm gates it.
        def _flush_rescan():
            f, b = projects.orphaned_outputs()
            return projects.flush_label(len(f), b)
        s["flush_rescan"].click(_flush_rescan, outputs=[s["flush_size"]])

        def _flush():
            n, freed = projects.flush_outputs()
            msg = (f"✅ Freed **{projects.human_size(freed)}** — deleted {n} orphaned "
                   f"generation{'s' if n != 1 else ''}." if n
                   else "Nothing to flush — already clean.")
            f, b = projects.orphaned_outputs()
            return projects.flush_label(len(f), b), msg
        s["flush_btn"].click(
            _flush, outputs=[s["flush_size"], s["flush_status"]],
            js=("() => { if(!confirm('Permanently delete all orphaned generations? "
                "Saved projects are unaffected and the images you currently see stay. "
                "This cannot be undone.')) throw new Error('flush cancelled'); }"))

        # Unload the bundled SDXL + helper models from VRAM (Wan2GP's unload misses
        # these). release_all() = the same callback handed to Wan2GP when it needs the GPU.
        def _unload_models():
            freed = ""
            torch = None
            try:
                import torch as _t
                torch = _t
            except Exception:
                torch = None
            before = None
            if torch is not None and torch.cuda.is_available():
                try:
                    before = torch.cuda.memory_allocated()
                except Exception:
                    before = None
            try:
                gen_sd.release_all()
            except Exception as e:
                traceback.print_exc()
                return f"⚠️ Unload failed: {e}"
            if torch is not None and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    if before is not None:
                        freed = (" — freed ~"
                                 + projects.human_size(max(0, before - torch.cuda.memory_allocated())))
                except Exception:
                    pass
            return f"✅ Unloaded the SDXL + helper models from VRAM{freed}."
        s["unload_models"].click(_unload_models, outputs=[s["unload_status"]])

        def _save_dirs(outputs, sdxl_models, sdxl_loras, models_d):
            try:
                paths.set_dirs(outputs=outputs or None, sdxl_models=sdxl_models or None,
                               sdxl_loras=sdxl_loras or None, models=models_d or None)
                status = ("✅ Directories saved & created. "
                          "(Re-run Scan to refresh the helper-weights status.)")
            except Exception as e:
                status = f"⚠️ Could not save directories: {e}"
            return [status] + _fresh_choices()

        s["save_dirs"].click(
            _save_dirs,
            inputs=[s["outputs_dir"], s["sdxl_models_dir"], s["sdxl_loras_dir"],
                    s["models_dir"]],
            outputs=[s["dirs_status"]] + model_dds + lora_dds)

        s["rescan"].click(
            lambda: ["🔄 Rescanned."] + _fresh_choices()
            + [gr.update(choices=self._native_dl_choices())],
            outputs=[s["dirs_status"]] + model_dds + lora_dds + [s["native_key"]])

        # Link an existing models folder into the shared OrphanSuite area, then
        # refresh the page dropdowns + the shared-dir textboxes (which now resolve
        # to orphansuite once it holds files).
        def _link_existing(src, leaf):
            try:
                # Links straight into the dir the loader scans (resolved through the
                # configured/shared dirs incl. the face/body/birefnet subdirs), so no
                # separate shared-dir override is needed — see paths.link_target_dir.
                msg = f"🔗 {paths.link_existing_into_shared(src, leaf)}"
            except Exception as e:
                msg = f"⚠️ {e}"
            return ([msg] + _fresh_choices()
                    + [gr.update(value=str(paths.sdxl_models_dir())),
                       gr.update(value=str(paths.sdxl_loras_dir())),
                       gr.update(value=str(paths.models_dir()))])
        s["link_btn"].click(
            _link_existing, inputs=[s["link_src"], s["link_target"]],
            outputs=([s["dirs_status"]] + model_dds + lora_dds
                     + [s["sdxl_models_dir"], s["sdxl_loras_dir"], s["models_dir"]]))

        # -- Helper-weights manager: button-triggered scan + per-row Download / Link.
        row_keys = s["model_row_keys"]

        def _row_update(r):
            st = r["state"]
            if st == "linked":
                txt, dl_on, ln_on = "✅ on disk", False, False
            elif st == "elsewhere":
                txt, dl_on, ln_on = "📁 on disk — not linked", bool(r["downloadable"]), True
            else:
                txt, dl_on, ln_on = "⬇ not downloaded", bool(r["downloadable"]), False
            return [gr.update(value=f"**{r['name']}** — {txt}  \n`{r['path']}`"),
                    gr.update(interactive=dl_on), gr.update(interactive=ln_on)]

        def _row_comps():
            comps = []
            for k in row_keys:
                comps += [s[f"m_{k}_status"], s[f"m_{k}_dl"], s[f"m_{k}_link"]]
            return comps

        def _scan(search_dir, progress=gr.Progress()):
            progress(0.2, desc="Scanning for models…")
            results = models.scan(search_dir or None)
            progress(1.0, desc="done")
            found = {r["key"]: r["found_at"] for r in results}
            n_disk = sum(r["state"] == "linked" for r in results)
            n_link = sum(r["state"] == "elsewhere" for r in results)
            n_dl = sum(r["state"] == "missing" for r in results)
            msg = (f"✅ Scan complete — {n_disk} on disk, {n_link} linkable, "
                   f"{n_dl} to download.")
            ups = []
            for r in results:
                ups += _row_update(r)
            return [found, msg] + ups
        s["scan_btn"].click(_scan, inputs=[s["scan_search"]],
                            outputs=[s["scan_found"], s["scan_status"]] + _row_comps())

        def _post(key, msg):
            """Row state after a Download/Link: present → linked + buttons off; on
            failure leave the buttons as they were and just surface the message."""
            spec = models.by_key(key)
            if spec is not None and spec.is_present():
                return [msg,
                        gr.update(value=f"**{spec.name}** — ✅ on disk  \n"
                                        f"`{spec.display_path()}`"),
                        gr.update(interactive=False), gr.update(interactive=False)]
            return [msg, gr.update(), gr.update(), gr.update()]

        def _mk_dl(key):
            def _dl(progress=gr.Progress()):
                return _post(key, models.download(key, progress=progress))
            return _dl

        def _mk_link(key):
            def _ln(found, progress=gr.Progress()):
                fa = (found or {}).get(key)
                return _post(key, models.link_found(key, fa, progress=progress))
            return _ln

        for k in row_keys:
            row_out = [s["scan_status"], s[f"m_{k}_status"],
                       s[f"m_{k}_dl"], s[f"m_{k}_link"]]
            s[f"m_{k}_dl"].click(_mk_dl(k), outputs=row_out)
            s[f"m_{k}_link"].click(_mk_link(k), inputs=[s["scan_found"]],
                                   outputs=row_out)

        # Default Generation Values editor → shared .orphansuite.json gen_defaults.
        _GD = [s["gd_steps"], s["gd_cfg"], s["gd_sampler"], s["gd_scheduler"],
               s["gd_clip"], s["gd_width"], s["gd_height"]]

        def _gd_fields(e):
            return [gr.update(value=e[k]) for k in
                    ("steps", "cfg", "sampler", "scheduler", "clip_skip", "width", "height")]

        def _gd_load(fam):
            tag = "(your override)" if presets.has_override(fam) else "(factory)"
            return _gd_fields(presets.effective(fam)) + [f"Showing **{fam}** {tag}."]
        s["gd_fam"].change(_gd_load, inputs=[s["gd_fam"]], outputs=_GD + [s["gd_status"]])

        def _gd_save(fam, steps, cfg, sampler, scheduler, clip, w, h):
            presets.set_overrides(fam, {
                "steps": int(steps), "cfg": float(cfg), "sampler": sampler,
                "scheduler": scheduler, "clip_skip": int(clip),
                "width": int(w), "height": int(h)})
            return f"✅ Saved **{fam}** defaults (applies on the next model select)."
        s["gd_save"].click(_gd_save, inputs=[s["gd_fam"]] + _GD, outputs=[s["gd_status"]])

        def _gd_reset(fam):
            presets.clear_overrides(fam)
            return _gd_fields(presets.effective(fam)) + [f"↺ **{fam}** reset to factory."]
        s["gd_reset"].click(_gd_reset, inputs=[s["gd_fam"]], outputs=_GD + [s["gd_status"]])

    def _send_to_img2vid(self, state, picked):
        if not picked:
            raise gr.Error("Select a result first.")
        try:
            settings = self.get_current_model_settings(state)
            settings["image_start"] = [picked]
            ipt = settings.get("image_prompt_type") or ""
            if "S" not in ipt:
                settings["image_prompt_type"] = ("S" + ipt) if ipt else "S"
            gr.Info("Image sent to the Video Generator as the start frame. "
                    "Pick an Image-to-Video model there if the current one isn't i2v.")
        except Exception:
            traceback.print_exc()
            raise gr.Error("Could not push the image to the Video Generator.")
        nav = gr.update()
        try:
            nav = self.goto_media_tab(state)
        except Exception:
            pass
        return time.time(), nav

    # -- main-page "Send current frame to Image Suite" ---------------------
    # A section injected directly under the host's video preview gallery (see the
    # insert_after in setup_ui). Pulls the result currently selected in the preview
    # player, lets the user pick a frame, and routes it into Img2Img's init slot or
    # the MultiCanvas background — the inbound mirror of _send_to_img2vid.
    _VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".avi", ".mov", ".m4v", ".gif")

    def _current_selection_path(self, state):
        """(path, kind) for the item selected in the main preview gallery. kind is
        'audio' (ignored), 'file', or None when nothing is selected. Mirrors the
        host's gen bookkeeping (get_gen_info / set_file_choice in wgp.py)."""
        gen = (state or {}).get("gen", {}) or {}
        if gen.get("current_gallery_source") == "audio":
            return None, "audio"
        files = gen.get("file_list") or []
        if not files:
            return None, None
        idx = gen.get("selected", 0) or 0
        if idx < 0 or idx >= len(files):
            idx = 0
        return files[idx], "file"

    def _color_match_to_init(self, state, frame_img):
        """Recolour `frame_img` (PIL) to match the host's current img2vid init image
        (image_start) via LAB mean/std transfer. Returns a PIL image, or None when
        there's no usable init image or the transfer fails."""
        if frame_img is None:
            return None
        try:
            settings = self.get_current_model_settings(state) or {}
        except Exception:
            return None
        init = settings.get("image_start")
        while isinstance(init, (list, tuple)):   # unwrap [path] or [(img, caption)]
            init = init[0] if init else None
        if init is None:
            return None
        try:
            import numpy as np, cv2
            from PIL import Image
            from .core import faceswap
            if isinstance(init, str):
                init_img = Image.open(init).convert("RGB")
            elif isinstance(init, Image.Image):
                init_img = init.convert("RGB")
            else:
                init_img = Image.fromarray(np.asarray(init)).convert("RGB")
            src_bgr = cv2.cvtColor(np.asarray(frame_img.convert("RGB")),
                                   cv2.COLOR_RGB2BGR)
            ref_bgr = cv2.cvtColor(np.asarray(init_img), cv2.COLOR_RGB2BGR)
            out_bgr = faceswap._color_transfer_lab(src_bgr, ref_bgr)
            return Image.fromarray(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
        except Exception:
            traceback.print_exc()
            return None

    def _build_send_frame_section(self):
        """BUILD ONLY (no wiring). The host runs this insert_after constructor during
        component insertion (wgp.py ~13034), which is BEFORE it builds our plugin tab
        (~13365) — so self._ui and our page components don't exist yet. We render the
        section's widgets, stash them on self._sf, and wire the handlers later in
        _wire_send_frame_section (end of _build_ui, once the pages exist). Renders
        exactly one top-level Accordion (what insert_after relocates)."""
        with gr.Accordion("📤 Send current frame", open=False) as box:
            gr.Markdown(
                "Pick a frame from the result selected in the gallery above and send it "
                "to Image Suite (Img2Img / MultiCanvas) or to the Video Generator "
                "(img2vid init / end image). The frame is chosen with the slider (the "
                "player can't report the exact on-screen frame); stills are sent as-is. "
                "When the Video Generator has an init image, a colour-matched "
                "'Corrected frame' is shown too — choose which one to send.",
                elem_classes="imagesuite-help")
            src_path = gr.State(None)
            with gr.Row():
                target = gr.Dropdown(
                    ["Img2Img", "MultiCanvas", "Modify", "img2vid (init)", "img2vid (end image)"],
                    value="Img2Img", label="Send current frame to", scale=2)
                load_btn = gr.Button("⟳ Load selected", scale=1)
            frame_no = gr.Slider(0, 0, value=0, step=1, label="Frame to send",
                                 visible=False)
            with gr.Row():
                preview = gr.Image(label="Current frame", type="pil",
                                   interactive=False, height=220, visible=False)
                corrected = gr.Image(label="Corrected frame (colour-matched to init)",
                                     type="pil", interactive=False, height=220,
                                     visible=False)
            which = gr.Radio(["Current frame", "Corrected frame"],
                             value="Current frame", label="Send which frame",
                             visible=False)
            send_btn = gr.Button("Send frame →", variant="primary", interactive=False)
        self._sf = {"src_path": src_path, "target": target, "load_btn": load_btn,
                    "frame_no": frame_no, "preview": preview, "corrected": corrected,
                    "which": which, "send_btn": send_btn}
        return box

    def _wire_send_frame_section(self):
        """Wire the injected 'Send current frame' section. Called at the end of
        _build_ui: by now the section widgets exist (built at insert time) AND our
        pages exist (just built), so the cross-tab handlers can connect (Gradio event
        wiring is context-independent). No-op if the section wasn't injected (older
        host without insert_after) or it's already wired."""
        sf = getattr(self, "_sf", None)
        ui = getattr(self, "_ui", None)
        if not sf or not ui or getattr(self, "_sf_wired", False):
            return
        self._sf_wired = True
        pages, subtabs, tab_ids = ui["pages"], ui["subtabs"], ui["tab_ids"]
        i2i_input = pages["img2img"]["input_image"]
        inp_bridge = pages["inpaint"]["bg_bridge"]
        mod_bridge = pages["modify"]["bg_bridge"]
        src_path, target = sf["src_path"], sf["target"]
        load_btn, frame_no = sf["load_btn"], sf["frame_no"]
        preview, send_btn = sf["preview"], sf["send_btn"]
        corrected, which = sf["corrected"], sf["which"]

        # 6-tuple matching load_outs below:
        # preview, frame_no, send_btn, src_path, corrected, which
        _HIDE = (gr.update(visible=False), gr.update(visible=False),
                 gr.update(interactive=False), None,
                 gr.update(value=None, visible=False),
                 gr.update(visible=False, value="Current frame"))

        def _corrected_updates(state, frame_img):
            """(corrected_preview, which_radio) updates for a frame: show the
            colour-matched-to-init version when an init image is available, else
            hide both and force the radio back to 'Current frame'."""
            corr = self._color_match_to_init(state, frame_img)
            if corr is None:
                return (gr.update(value=None, visible=False),
                        gr.update(visible=False, value="Current frame"))
            return (gr.update(value=corr, visible=True), gr.update(visible=True))

        def _updates_for(state, path, warn):
            """Resolve a selected path into the 6 load_outs updates. `warn` toggles
            the 'select something' hint (off for auto-sync, which fires constantly)."""
            from shared.utils.utils import get_video_frame, get_video_info
            if not path:
                if warn:
                    gr.Warning("Select a result in the gallery above first.")
                return _HIDE
            if not str(path).lower().endswith(self._VIDEO_EXTS):
                try:
                    from PIL import Image
                    img = Image.open(path).convert("RGB")
                except Exception:
                    traceback.print_exc()
                    if warn:
                        gr.Warning("Couldn't open that result.")
                    return _HIDE
                corr_p, which_u = _corrected_updates(state, img)
                return (gr.update(value=img, visible=True), gr.update(visible=False),
                        gr.update(interactive=True), path, corr_p, which_u)
            try:
                _fps, _w, _h, frames = get_video_info(path)
                frames = int(frames) or 1
                first = get_video_frame(path, 0, return_last_if_missing=True,
                                        return_PIL=True)
            except Exception:
                traceback.print_exc()
                if warn:
                    gr.Warning("Couldn't read that video.")
                return _HIDE
            corr_p, which_u = _corrected_updates(state, first)
            return (gr.update(value=first, visible=True),
                    gr.update(minimum=0, maximum=max(frames - 1, 0), value=0,
                              visible=True),
                    gr.update(interactive=True), path, corr_p, which_u)

        def _load(state):  # explicit "Load selected" button
            path, kind = self._current_selection_path(state)
            if kind == "audio":
                gr.Warning("That's an audio selection — pick a video or image result.")
                return _HIDE
            return _updates_for(state, path, warn=True)

        def _load_sel(state, evt: gr.EventData):
            # Auto-sync: resolve from the click's own index (gen['selected'] may not
            # be updated yet — the host's select_video runs as a separate listener).
            # Use gr.EventData (not gr.SelectData, whose __init__ KeyErrors when the
            # select payload has no "value") and read the index off the raw payload.
            files = ((state or {}).get("gen", {}) or {}).get("file_list") or []
            data = getattr(evt, "_data", {}) or {}
            idx = data.get("index") if isinstance(data, dict) else None
            if isinstance(idx, (list, tuple)):
                idx = idx[0] if idx else None
            path = files[idx] if isinstance(idx, int) and 0 <= idx < len(files) else None
            return _updates_for(state, path, warn=False)

        def _scrub(state, path, n):
            from shared.utils.utils import get_video_frame
            if not path or not str(path).lower().endswith(self._VIDEO_EXTS):
                return gr.update(), gr.update(), gr.update()
            try:
                img = get_video_frame(path, int(n), return_last_if_missing=True,
                                      return_PIL=True)
            except Exception:
                traceback.print_exc()
                return gr.update(), gr.update(), gr.update()
            corr_p, which_u = _corrected_updates(state, img)
            return gr.update(value=img), corr_p, which_u

        def _send(state, tgt, which_val, frame_img, corrected_img):
            noop = gr.update()
            if which_val == "Corrected frame":
                send_img = corrected_img if corrected_img is not None else frame_img
                if corrected_img is None:
                    gr.Warning("No corrected frame available — sending the current frame.")
            else:
                send_img = frame_img
            if send_img is None:
                gr.Warning("Load a frame first.")
                return [noop, noop, noop, noop, noop, noop]
            try:
                from PIL import Image
                if not isinstance(send_img, Image.Image):
                    send_img = Image.fromarray(send_img)
                path = self._save_img(send_img.convert("RGB"), "sentframe")
            except Exception:
                traceback.print_exc()
                raise gr.Error("Couldn't save that frame.")
            if tgt == "MultiCanvas":
                html = canvas.bg_bridge_html(self._file_to_dataurl(path), "inpaint",
                                             nonce=time.time())
                gr.Info("Frame sent to the MultiCanvas background.")
                return [noop, html, noop, gr.update(selected=tab_ids["inpaint"]),
                        gr.update(selected=PLUGIN_ID), noop]
            if tgt == "Modify":
                html = modify_canvas.modify_bg_bridge_html(
                    self._file_to_dataurl(path), "modify", nonce=time.time())
                gr.Info("Frame sent to the Modify canvas.")
                return [noop, noop, html, gr.update(selected=tab_ids["modify"]),
                        gr.update(selected=PLUGIN_ID), noop]
            if tgt in ("img2vid (init)", "img2vid (end image)"):
                is_end = tgt == "img2vid (end image)"
                try:
                    settings = self.get_current_model_settings(state)
                    settings["image_end" if is_end else "image_start"] = [path]
                    letter = "E" if is_end else "S"
                    ipt = settings.get("image_prompt_type") or ""
                    if letter not in ipt:
                        settings["image_prompt_type"] = (letter + ipt) if ipt else letter
                except Exception:
                    traceback.print_exc()
                    raise gr.Error("Could not push the frame to the Video Generator.")
                nav = gr.update()
                try:
                    nav = self.goto_media_tab(state)
                except Exception:
                    pass
                slot = "end image" if is_end else "init"
                gr.Info(f"Frame sent to the Video Generator as the img2vid {slot}. "
                        "Pick an Image-to-Video model there if the current one isn't i2v.")
                return [noop, noop, noop, noop, nav, time.time()]
            gr.Info("Frame sent to Img2Img as the init image.")
            return [path, noop, noop, gr.update(selected=tab_ids["img2img"]),
                    gr.update(selected=PLUGIN_ID), noop]

        load_outs = [preview, frame_no, send_btn, src_path, corrected, which]
        load_btn.click(_load, inputs=[self.state], outputs=load_outs)
        # Auto-reflect the gallery's current selection (what the user asked for).
        if getattr(self, "output", None) is not None:
            self.output.select(_load_sel, inputs=[self.state], outputs=load_outs)
        # .release (not .change) so we decode on drag-end, not every pixel.
        frame_no.release(_scrub, inputs=[self.state, src_path, frame_no],
                         outputs=[preview, corrected, which])
        send_btn.click(_send, inputs=[self.state, target, which, preview, corrected],
                       outputs=[i2i_input, inp_bridge, mod_bridge, subtabs,
                                self.main_tabs, self.refresh_form_trigger])
