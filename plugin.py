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
import functools
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

from .core import discovery, gen_sd, models, paths, presets
from .ui import canvas, contextmenu, logo, suite
from .ui.styles import CSS, LIGHTBOX_HTML

PLUGIN_ID = "ImageSuite"
PLUGIN_NAME = "Image Suite"

# Hidden Textbox the shared right-click menu relays into ({a:action,s:src,t}).
CTX_RELAY = contextmenu.RELAY_ID


class ImageSuite(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = PLUGIN_NAME
        self.version = "0.1.0"
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

    def on_tab_select(self, state: dict):
        # Warn if the GPU looks busy, but DON'T bounce out — a stale lock would
        # otherwise trap the user out of the tab (and away from ⛔ Abort, which
        # clears it). The gen handlers guard the GPU themselves.
        if _HAVE_LOCKS and any_GPU_process_running(state, PLUGIN_ID):
            gr.Warning("A generation appears to be running — if it's actually stuck, "
                       "hit ⛔ Abort to clear it.")
        return gr.update()

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
            _add(paths.outputs_dir())
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
            # Shared app-wide right-click menu (idempotent engine + our items) and
            # the hidden relay it writes into.
            gr.HTML(contextmenu.imagesuite_ctx_html(), elem_classes="imagesuite-hidden")
            self._ctx_relay = gr.Textbox(visible=False, elem_id=CTX_RELAY)
            ui = suite.build_suite(model_choices_by_mode=choices_by_mode,
                                   lora_choices=lora_choices,
                                   native_dl_choices=self._native_dl_choices(),
                                   sdxl_choices=sdxl_choices)
        self._wire(ui)
        self.on_tab_outputs = [self.main_tabs] if hasattr(self, "main_tabs") else None
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
                    loras=None, mult_str="", progress=None, api=None):
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
            result = (api or self._api).submit_task(settings).result()
        except Exception as e:
            if "generation in progress" in str(e).lower():
                raise gr.Error(
                    "Another generation is still pending. Native (Flux/Z-Image/Qwen) "
                    "gens run in Wan2GP's queue and PAUSE if the browser loses focus — "
                    "click the Video Generator tab to let it finish, then try again "
                    "(or restart Wan2GP).")
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

    def _gallery_result(self, files):
        """The (gallery, picked, save) tuple every generate handler returns.

        The gallery is fed PIL images, NOT paths: Gradio then owns the cache entry
        for each, so the select event's file-in-cache check can't trip on an
        external/relative path. picked + Save As still get a real cached file path
        (via _serve) so send-to / download work."""
        served = self._serve(files if isinstance(files, (list, tuple)) else [files])
        if not served:
            raise gr.Error("Generation produced no images.")
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
        """Local image path → PNG data-URL, for pushing into the canvas frame."""
        import base64
        from PIL import Image
        with Image.open(path) as im:
            import io
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    # -- wiring -------------------------------------------------------------
    def _wire(self, ui):
        pages = ui["pages"]
        for mode, c in pages.items():
            self._wire_page(mode, c)
        self._wire_sends(ui)
        self._wire_settings(ui)
        self._wire_ctx(ui)
        self._wire_overlays(ui)
        self._wire_persist(ui)

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
                    gr.update(value=ov.list_images(folder)),
                    gr.update(choices=folders))

        o["folder"].change(_refresh, inputs=[o["folder"]],
                           outputs=[o["folder"], o["gallery"], o["move_to"]])

        def _create(name, folder):
            try:
                folder = ov.create_folder(name); msg = f"Created folder '{folder}'."
            except Exception as e:
                msg = f"⚠ {e}"
            return (*_refresh(folder), gr.update(value=""), msg)
        o["create_folder"].click(
            _create, inputs=[o["new_folder"], o["folder"]],
            outputs=[o["folder"], o["gallery"], o["move_to"], o["new_folder"], o["status"]])

        def _del_folder(folder):
            try:
                ov.delete_folder(folder); msg = f"Deleted folder '{folder}'."
            except Exception as e:
                msg = f"⚠ {e}"
            return (*_refresh(ov.ROOT_LABEL), msg)
        o["delete_folder"].click(
            _del_folder, inputs=[o["folder"]],
            outputs=[o["folder"], o["gallery"], o["move_to"], o["status"]])

        def _upload(files, folder):
            try:
                fps = [getattr(f, "name", f) for f in (files or [])]
                n = ov.save_uploads(folder, fps); msg = f"Added {n} image(s)."
            except Exception as e:
                msg = f"⚠ {e}"
            return (*_refresh(folder), None, msg)
        o["upload_btn"].click(
            _upload, inputs=[o["upload"], o["folder"]],
            outputs=[o["folder"], o["gallery"], o["move_to"], o["upload"], o["status"]])

        def _select(evt: gr.SelectData):
            v = evt.value
            path = (v.get("image", {}).get("path") or v.get("path")
                    if isinstance(v, dict) else v if isinstance(v, str) else None)
            name = os.path.basename(path) if path else None
            return name, gr.update(value=path), gr.update(value=name or "")
        o["gallery"].select(_select,
                            outputs=[o["selected"], o["preview"], o["rename_to"]])

        def _rename(folder, sel, new):
            try:
                if not sel:
                    raise ValueError("Select an overlay first.")
                sel = ov.rename_image(folder, sel, new); msg = f"Renamed to '{sel}'."
            except Exception as e:
                msg = f"⚠ {e}"
            return (*_refresh(folder), sel, msg)
        o["rename_btn"].click(
            _rename, inputs=[o["folder"], o["selected"], o["rename_to"]],
            outputs=[o["folder"], o["gallery"], o["move_to"], o["selected"], o["status"]])

        def _move(folder, sel, dest):
            try:
                if not sel:
                    raise ValueError("Select an overlay first.")
                name = ov.move_image(folder, sel, dest)
                msg = (f"Moved '{sel}' to '{dest}' (renamed '{name}')."
                       if name and name != sel else f"Moved '{sel}' to '{dest}'.")
            except Exception as e:
                msg = f"⚠ {e}"
            return (*_refresh(folder), None, msg)
        o["move_btn"].click(
            _move, inputs=[o["folder"], o["selected"], o["move_to"]],
            outputs=[o["folder"], o["gallery"], o["move_to"], o["selected"], o["status"]])

        def _delete(folder, sel):
            try:
                if not sel:
                    raise ValueError("Select an overlay first.")
                ov.delete_image(folder, sel); msg = f"Deleted '{sel}'."
            except Exception as e:
                msg = f"⚠ {e}"
            return (*_refresh(folder), None, None, msg)
        o["delete_btn"].click(
            _delete, inputs=[o["folder"], o["selected"]],
            outputs=[o["folder"], o["gallery"], o["move_to"], o["selected"],
                     o["preview"], o["status"]])

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
            return ups

        try:
            from gradio.context import Context
            root = Context.root_block
            outputs = comps + [comp for _, comp in out_size_extra]
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
        c["model"].change(_on_model, inputs=[c["model"]], outputs=model_outputs)

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

        # NOTE: no js= here. A js hook that returns undefined nulls the handler's
        # inputs in Gradio (model arrived empty → "Select a model"). The canvas
        # already pushes composite/mask into the hidden fields on every stroke, so
        # the latest is present at click time without a pre-flush hook.
        c["generate"].click(
            fn, inputs=gen_inputs,
            outputs=[c["gallery"], c["picked"], c["save"]])

        # Click a gallery result to make IT the active selection — so Send-to /
        # Save As / the enhancement passes act on whichever image you pick, not just
        # the first. Safe: the gallery is fed PIL images, so Gradio owns each entry
        # as a CACHE path, and the select payload's path is therefore inside the
        # Gradio cache — which both the stock check_all_files_in_cache and our
        # allow-list patch (_cache_allow_dirs) accept. 'picked' is still armed to the
        # first result at generation time, so it's correct before any click.
        def _pick(evt: gr.SelectData):
            v = evt.value
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
                gen_sd.release_sd()  # free our SDXL so the native model has room
                snap = self._snapshot_video_state(state)
                try:
                    for i in range(n):
                        progress((i, n), desc=f"Generating {i + 1}/{n}")
                        files += self._gen_native(ident, pos, neg, width, height, steps,
                                                  cfg, seed, mode="txt2img",
                                                  loras=loras, mult_str=mult,
                                                  progress=progress, api=api)
                finally:
                    self._restore_video_state(state, snap)
            else:  # sd
                lora_list = self._lora_list(loras, mult)
                self.acquire_gpu(state)
                try:
                    self._sd_loading(progress)
                    nsteps, total = int(steps), max(1, n) * max(1, int(steps))
                    for i in range(n):
                        if gen_sd.was_aborted():
                            break
                        sd = int(seed) if int(seed) >= 0 else _rng.randint(0, 2**31 - 1)
                        files += gen_sd.generate_txt2img(
                            ident, pos, neg, width, height, steps, cfg, sd,
                            sampler=sampler or "DPM++ 2M", scheduler=scheduler or "",
                            clip_skip=int(clip_skip), loras=lora_list,
                            callback=self._sd_step_cb(progress, i, nsteps, total))
                finally:
                    self.release_gpu(state)
            if not files:
                raise gr.Error("Generation produced no images.")
            return self._gallery_result(files)
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
                gen_sd.release_sd()  # free our SDXL so the native model has room
                snap = self._snapshot_video_state(state)
                try:
                    for i in range(n):
                        progress((i, n), desc=f"Reimagining {i + 1}/{n}")
                        files += self._gen_native(ident, pos, neg, width, height, steps,
                                                  cfg, seed, mode="img2img",
                                                  denoise=denoise, guide_path=init_image,
                                                  loras=loras, mult_str=mult,
                                                  progress=progress, api=api)
                finally:
                    self._restore_video_state(state, snap)
            else:  # sd
                lora_list = self._lora_list(loras, mult)
                self.acquire_gpu(state)
                try:
                    self._sd_loading(progress)
                    nsteps, total = int(steps), max(1, n) * max(1, int(steps))
                    for i in range(n):
                        if gen_sd.was_aborted():
                            break
                        sd = int(seed) if int(seed) >= 0 else _rng.randint(0, 2**31 - 1)
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
            return self._gallery_result(files)
        return _run

    # generate_inpaint's inpainting_fill code: 0 fill / 1 original /
    # 2 latent-noise / 3 latent-nothing (mirrors A1111 "Masked content").
    _FILL = {"fill": 0, "original": 1, "latent noise": 2, "latent nothing": 3}

    def _inpaint_core(self, state, backend, ident, comp, mask, pos, neg, *,
                      denoise, steps, cfg, clip_skip, seed, sampler, scheduler,
                      loras, mult, feather=4, inpaint_fill="original",
                      full_res=False, padding=32, api=None, progress=None):
        """One inpaint pass over a prepared composite(RGB) + mask(L). Shared by the
        canvas Generate and the Touch-Up outpaint so both get identical GPU
        lifecycle, native/SD branching and abort handling. Returns the path."""
        def _say(f, d):
            if progress is not None:
                try: progress(f, desc=d)
                except Exception: pass
        if backend == "native":
            if self._gpu_busy(state):
                raise gr.Error("A generation is already running — wait for it to finish.")
            gen_sd.release_sd()  # free our SDXL so the native model has room
            _say(0.1, "Inpainting…")
            m = mask
            if feather and int(feather) > 0:  # native has no pipeline mask_blur
                from PIL import ImageFilter
                m = mask.filter(ImageFilter.GaussianBlur(radius=int(feather)))
            snap = self._snapshot_video_state(state)
            try:
                files = self._gen_native(
                    ident, pos, neg, comp.width, comp.height, steps, cfg, seed,
                    mode="inpaint", denoise=denoise,
                    guide_path=self._save_img(comp, "composite"),
                    mask_path=self._save_img(m, "mask"),
                    loras=loras, mult_str=mult, progress=progress, api=api)
            finally:
                self._restore_video_state(state, snap)
            return files[0] if files else None
        # sd
        lora_list = self._lora_list(loras, mult)
        self.acquire_gpu(state)
        try:
            self._sd_loading(progress)
            nsteps = int(steps)
            cb = (self._sd_step_cb(progress, 0, nsteps, nsteps)
                  if progress is not None else None)
            outs = gen_sd.inpaint(
                ident, comp, mask, pos, neg, denoise=float(denoise),
                steps=int(steps), cfg=float(cfg), seed=int(seed),
                sampler=sampler or "DPM++ 2M", scheduler=scheduler or "Karras",
                clip_skip=int(clip_skip), mask_blur=int(feather),
                inpainting_fill=self._FILL.get(inpaint_fill, 1),
                full_res=bool(full_res), padding=int(padding),
                progress=progress, loras=lora_list, callback=cb)
        finally:
            self.release_gpu(state)
        # Mid-image abort interrupted the diffusers loop → discard the partial; the
        # callers already surface "Inpaint/Outpaint aborted." when was_aborted().
        if gen_sd.was_aborted():
            return None
        return outs[0] if outs else None

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
            if comp is None:
                raise gr.Error("Load an image into the canvas first (upload one or "
                               "use a 'Send to MultiCanvas' button).")
            if mask is None or not self._mask_nonempty(mask):
                raise gr.Error("Mark a mask region first — use Mask mode (paint the "
                               "area to change) or turn on Auto-mask.")
            comp = comp.convert("RGB")
            mask = mask.convert("L")
            if mask_mode == "Inpaint not masked":
                from PIL import ImageOps
                mask = ImageOps.invert(mask)
            out = self._inpaint_core(
                state, backend, ident, comp, mask, pos, neg, denoise=denoise,
                steps=steps, cfg=cfg, clip_skip=clip_skip, seed=seed,
                sampler=sampler, scheduler=scheduler, loras=loras, mult=mult,
                feather=feather, inpaint_fill=inpaint_fill,
                full_res=(inpaint_area == "Only masked"),
                padding=padding, api=api, progress=progress)
            if not out:
                raise gr.Error("Inpaint aborted." if gen_sd.was_aborted()
                               else "Inpaint produced no image.")
            return self._gallery_result(out)
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
            out = self._inpaint_core(
                state, backend, ident, ext, mask, pos, neg, denoise=1.0,
                steps=steps, cfg=cfg, clip_skip=clip_skip, seed=seed,
                sampler=sampler, scheduler=scheduler, loras=loras, mult=mult,
                feather=max(8, int(feather)), inpaint_fill="fill",
                full_res=False, padding=0, api=api, progress=progress)
            if not out:
                raise gr.Error("Outpaint aborted." if gen_sd.was_aborted()
                               else "Outpaint produced no image.")
            return self._gallery_result(out)
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
            if src.startswith("data:"):
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
            scheme = urllib.parse.urlparse(src).scheme.lower()
            if scheme in ("http", "https"):
                out = paths.cache_dir() / "ctx"
                out.mkdir(parents=True, exist_ok=True)
                ext = os.path.splitext(urllib.parse.urlparse(src).path)[1] or ".png"
                p = out / f"ctx_{int(time.time()*1000)}{ext}"
                # Bounded fetch: connect timeout + a hard size cap so a hostile/large
                # URL can't hang the handler or fill the disk.
                MAX_BYTES = 64 * 1024 * 1024
                req = urllib.request.Request(src, headers={"User-Agent": "ImageSuite"})
                with urllib.request.urlopen(req, timeout=15) as resp:
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
        target page's matching components. The model is applied only if the target
        offers it (Img2Img/Inpaint list only guide-capable models), and LoRAs are
        re-scoped to the carried model's family so their values stay valid."""
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
        return [ups[k] for k in self.CARRY_KEYS]

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
        outputs = [dst[k] for k in keys]
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
    def _wire_sends(self, ui):
        pages, subtabs, tab_ids = ui["pages"], ui["subtabs"], ui["tab_ids"]

        for mode, c in pages.items():
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
        modes = list(pages.keys())
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

        def _save_dirs(outputs, sdxl_models, sdxl_loras, models_d):
            try:
                paths.set_dirs(outputs=outputs or None, sdxl_models=sdxl_models or None,
                               sdxl_loras=sdxl_loras or None, models=models_d or None)
                status = "✅ Directories saved & created."
            except Exception as e:
                status = f"⚠️ Could not save directories: {e}"
            return [status] + _fresh_choices() + [settings_panel.status_md()]

        s["save_dirs"].click(
            _save_dirs,
            inputs=[s["outputs_dir"], s["sdxl_models_dir"], s["sdxl_loras_dir"],
                    s["models_dir"]],
            outputs=[s["dirs_status"]] + model_dds + lora_dds + [s["models_status"]])

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

        def _download(key, progress=gr.Progress()):
            if not key:
                raise gr.Error("Pick a model to download first.")
            msg = models.download(key, progress=progress)
            return msg, settings_panel.status_md()

        s["download"].click(_download, inputs=[s["model_key"]],
                            outputs=[s["download_log"], s["models_status"]])

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
            nav = self.goto_video_tab(state)
        except Exception:
            pass
        return time.time(), nav
