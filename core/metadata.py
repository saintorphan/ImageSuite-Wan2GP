"""A1111-style PNG ``parameters`` text-chunk format + embed/recall.

AUTOMATIC1111 (and reForge/Forge/ComfyUI's A1111 export) stores generation
settings as a single ``parameters`` PNG tEXt chunk, in this convention::

    <positive prompt>
    Negative prompt: <negative prompt>
    Steps: 28, Sampler: DPM++ 2M, Schedule type: Karras, CFG scale: 6.0, \
Seed: 1234, Size: 832x1216, Model: myCheckpoint, Clip skip: 2

The positive prompt is the leading block; an optional ``Negative prompt:`` line
follows; the trailing line is comma-separated ``Key: value`` pairs. We format and
parse exactly that shape so images carry params other A1111-family tools can read,
and so we can recall params from any A1111-produced PNG.

Pure stdlib — no PIL needed here (the caller owns the PngInfo write); this module
only does the text<->dict conversion.
"""
from __future__ import annotations

import re

# Form-field key  ->  A1111 trailing-line label. Order is the canonical A1111
# order; we emit in this order so output reads like a real A1111 chunk.
_KV_FIELDS = [
    ("steps", "Steps"),
    ("sampler", "Sampler"),
    ("scheduler", "Schedule type"),
    ("cfg", "CFG scale"),
    ("seed", "Seed"),
    ("size", "Size"),
    ("model", "Model"),
    ("clip_skip", "Clip skip"),
]
# label (lower-cased) -> field key, for parsing. Include a couple of common
# aliases so chunks from older/other tools still resolve.
_LABEL_TO_KEY = {label.lower(): key for key, label in _KV_FIELDS}
_LABEL_TO_KEY.update({
    "cfg scale": "cfg",
    "sampler name": "sampler",
    "schedule type": "scheduler",
    "model name": "model",
    "clip skip": "clip_skip",
})


def format_a1111(params: dict) -> str:
    """Render ``params`` (a form-style dict) as an A1111 ``parameters`` string.

    Recognised keys: ``prompt`` (or ``pos``), ``negative`` (or ``neg``), ``seed``,
    ``steps``, ``cfg`` (or ``guidance``), ``sampler``, ``scheduler``, ``model``,
    ``width``+``height`` (or ``size``) and ``clip_skip``. Missing keys are simply
    omitted, so a partial dict still produces a valid (shorter) chunk.
    """
    p = dict(params or {})
    prompt = _first(p, "prompt", "pos") or ""
    negative = _first(p, "negative", "neg") or ""

    lines = [str(prompt).strip()]
    if str(negative).strip():
        lines.append(f"Negative prompt: {str(negative).strip()}")

    # Build the canonical scalar dict, then emit the present ones in order.
    size = p.get("size")
    if not size:
        w, h = p.get("width"), p.get("height")
        if w and h:
            size = f"{_int(w)}x{_int(h)}"
    vals = {
        "steps": _int(p.get("steps")),
        "sampler": _str(p.get("sampler")),
        "scheduler": _str(_first(p, "scheduler")),
        "cfg": _num(_first(p, "cfg", "guidance")),
        "seed": _int(p.get("seed")),
        "size": _str(size),
        "model": _str(p.get("model")),
        "clip_skip": _int(p.get("clip_skip")),
    }
    kv = [f"{label}: {vals[key]}" for key, label in _KV_FIELDS
          if vals.get(key) not in (None, "")]
    if kv:
        lines.append(", ".join(kv))
    return "\n".join(lines)


def parse_a1111(text: str) -> dict:
    """Parse an A1111 ``parameters`` string into a form-style dict.

    Returns whatever it can recognise: ``prompt``, ``negative``, ``seed``,
    ``steps``, ``cfg``, ``sampler``, ``scheduler``, ``model``, ``width``,
    ``height`` and ``clip_skip``. Unparseable / empty input yields ``{}`` so
    callers can treat a falsy result as "no usable metadata".
    """
    if not text or not str(text).strip():
        return {}
    raw = str(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = raw.split("\n")

    # Split into: positive block, optional negative block, trailing KV line.
    neg_idx = next((i for i, ln in enumerate(lines)
                    if ln.strip().lower().startswith("negative prompt:")), None)
    # The KV line is the last non-blank line IF it parses as Key: value pairs.
    kv_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip():
            if _looks_like_kv(lines[i]):
                kv_idx = i
            break

    out: dict = {}
    pos_end = neg_idx if neg_idx is not None else kv_idx
    if pos_end is None:
        pos_end = len(lines)
    prompt = "\n".join(lines[:pos_end]).strip()
    if prompt:
        out["prompt"] = prompt

    if neg_idx is not None:
        neg_end = kv_idx if kv_idx is not None else len(lines)
        neg_block = lines[neg_idx:neg_end]
        neg_block[0] = neg_block[0].split(":", 1)[1] if ":" in neg_block[0] else ""
        negative = "\n".join(neg_block).strip()
        if negative:
            out["negative"] = negative

    if kv_idx is not None:
        out.update(_parse_kv_line(lines[kv_idx]))
    return out


# ---------------------------------------------------------------------------

def _parse_kv_line(line: str) -> dict:
    out: dict = {}
    for label, value in _split_kv(line):
        key = _LABEL_TO_KEY.get(label.strip().lower())
        if not key:
            continue
        value = value.strip().strip('"')
        if key == "size":
            m = re.match(r"^\s*(\d+)\s*[x×]\s*(\d+)\s*$", value)
            if m:
                out["width"], out["height"] = int(m.group(1)), int(m.group(2))
        elif key in ("steps", "seed", "clip_skip"):
            iv = _to_int(value)
            if iv is not None:
                out[key] = iv
        elif key == "cfg":
            fv = _to_float(value)
            if fv is not None:
                out[key] = fv
        else:  # sampler / scheduler / model — free text
            if value:
                out[key] = value
    return out


def _split_kv(line: str):
    """Yield (label, value) pairs from a ``a: b, c: d`` line. Splits on commas that
    sit before a ``label:`` token so commas inside a value (rare) don't break it."""
    parts = re.split(r",\s*(?=[A-Za-z][\w +/().-]*:)", line.strip())
    for part in parts:
        if ":" in part:
            label, value = part.split(":", 1)
            yield label, value


def _looks_like_kv(line: str) -> bool:
    for label, _ in _split_kv(line):
        if label.strip().lower() in _LABEL_TO_KEY:
            return True
    return False


def _first(p: dict, *keys):
    for k in keys:
        v = p.get(k)
        if v not in (None, ""):
            return v
    return None


def _int(v):
    iv = _to_int(v)
    return iv if iv is not None else None


def _num(v):
    fv = _to_float(v)
    if fv is None:
        return None
    return int(fv) if float(fv).is_integer() else fv


def _str(v):
    return str(v).strip() if v not in (None, "") else None


def _to_int(v):
    try:
        return int(float(str(v).strip()))
    except (TypeError, ValueError):
        return None


def _to_float(v):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None
