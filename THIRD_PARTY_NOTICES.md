# Third-Party Notices

ImageSuite-Wan2GP is distributed under the **WanGP Community License 2.0** (see
[`LICENSE.txt`](LICENSE.txt)), the same license as its host, Wan2GP. That license
applies only to this plugin's own code; the third-party components bundled in-tree
below retain their own licenses, and nothing here reduces the rights those licenses
grant you.

## Bundled code

### `core/sd/laplacian_blend.py` — Apache-2.0
Pure-torch Laplacian-pyramid blending ported from **Code2Collapse's
ComfyUI-CustomNodePacks** (`inpaint_suite.py`), licensed under the **Apache License,
Version 2.0**. The numerical method is unchanged; the ComfyUI node wrappers were
removed and the code reduced to a single `laplacian_pyramid_blend` function.
Apache-2.0 full text: <https://www.apache.org/licenses/LICENSE-2.0>.

### `core/sd/` (SD/SDXL backend) — original work
The diffusers-based SD/SDXL pipeline was ported from the author's own prior project,
**SupremeDiffusion**, and is bundled here so the plugin is self-contained. It is the
author's own code, released under this repository's license; **no external
SupremeDiffusion checkout is required.**

## Re-implemented techniques (no third-party code copied)

### `core/faceswap.py` — algorithm credit
The face-swap pipeline is a **clean-room implementation** that depends on no
roop-unleashed source code; its algorithms were adapted in spirit from
roop-unleashed's `face_util` / `FaceSwapInsightFace` / `ProcessMgr` / `Enhance_*`
modules. Credit is given here for provenance; no roop-unleashed code is included.

## Runtime dependencies

Third-party packages installed via `requirements.txt` or `core/deps.py`
(diffusers, transformers, InsightFace, ONNX Runtime, BiRefNet, MediaPipe, kornia,
controlnet_aux, ultralytics, etc.) are **not redistributed** in this repository and
remain governed by their own licenses, obtained from their respective distributors.
