"""Advanced prompt encoding for the SD/SDXL backend via ``compel``.

Today the SD pipelines take raw prompt strings, so attention weights are ignored
and anything over CLIP's 77-token window is silently truncated. This module builds
``prompt_embeds`` (and, for SDXL, ``pooled_prompt_embeds``) from the prompt instead,
giving:

  * ``(token:1.3)`` / ``(emphasis)`` attention weighting,
  * ``BREAK`` to force a fresh 77-token chunk (A1111 convention),
  * automatic >77-token chunking (``truncate_long_prompts=False``).

Everything here is best-effort and OPT-IN. :func:`build_embeds` returns ``None`` on
*any* problem (compel missing, an exotic pipe, an encode error) so the caller can
fall back to the exact current raw-string path — behaviour never regresses.

clip_skip note: diffusers treats ``clip_skip`` and ``prompt_embeds`` as mutually
exclusive (clip_skip slices the encoder hidden states, but we hand it pre-built
embeds). So when encoding we BAKE the skip into compel's
``returned_embeddings_type`` (penultimate-layer hidden states) and the caller must
NOT also pass ``clip_skip`` alongside the returned embeds.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def compel_available() -> bool:
    """True if ``compel`` can be imported (cheap; never installs)."""
    try:
        import compel  # noqa: F401
        return True
    except Exception:
        return False


def prompt_is_plain(prompt: str | None, negative: str | None = None) -> bool:
    """True when neither string uses any advanced-prompt syntax, so the raw-string
    path is byte-for-byte equivalent and we can skip compel entirely.

    Advanced syntax = attention weights ``( ) [ ]`` / explicit ``(x):1.2`` weights /
    the ``BREAK`` chunk separator. A plain prompt that merely happens to be long is
    deliberately NOT treated as plain — compel's ``truncate_long_prompts=False`` is
    exactly the win there. We can't cheaply know the token count without a tokenizer,
    so any non-empty prompt is encoded; only truly trivial cases stay on the raw path.
    """
    # Conservative: only the empty/whitespace case is unambiguously "plain". Letting
    # everything else go through compel keeps the long-prompt and weighting wins, and
    # build_embeds still falls back to raw on any failure.
    p = (prompt or "").strip()
    n = (negative or "").strip()
    return not p and not n


def _build_compel(pipe, model_type: str, clip_skip: int):
    """Construct a ``Compel`` for *pipe*, baking clip_skip into the returned-embeds
    type. Returns ``(compel, is_sdxl)`` or raises (caller handles)."""
    from compel import Compel, ReturnedEmbeddingsType

    is_sdxl = str(model_type) == "sdxl"

    # Pick the ReturnedEmbeddingsType so the embeds MATCH what the diffusers
    # pipeline would build for the raw prompt (clip_skip + prompt_embeds are
    # mutually exclusive, so the skip must be baked here).
    try:
        n = int(clip_skip)
    except (TypeError, ValueError):
        n = 1
    if is_sdxl:
        # SDXL ALWAYS uses the penultimate (non-normalized) hidden states of both
        # encoders by default in diffusers — independent of clip_skip — so use that
        # for the common clip_skip 1/2 case to match the raw path. A deeper skip
        # (>2) can't be expressed via the two compel embedding types, so it also
        # falls back to penultimate (the SDXL norm) rather than producing a mismatch.
        emb_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED
    elif n > 1:
        # SD1.5 with skip: penultimate (A1111 clip_skip 2 == diffusers clip_skip 1).
        emb_type = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED
    else:
        # SD1.5 no-skip: the last (normalized) hidden state == diffusers default.
        emb_type = ReturnedEmbeddingsType.LAST_HIDDEN_STATES_NORMALIZED

    if is_sdxl:
        # SDXL has two tokenizers / text encoders; the second supplies the pooled
        # embedding. requires_pooled marks which encoder contributes the pooled vec.
        compel = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=emb_type,
            requires_pooled=[False, True],
            truncate_long_prompts=False,
        )
    else:
        compel = Compel(
            tokenizer=pipe.tokenizer,
            text_encoder=pipe.text_encoder,
            returned_embeddings_type=emb_type,
            truncate_long_prompts=False,
        )
    return compel, is_sdxl


def _encode_one(compel, text: str):
    """Encode a single prompt string, handling ``BREAK`` by encoding each segment
    and concatenating along the token axis (A1111 convention — BREAK pads the
    current chunk to 77 then starts a fresh one).

    Returns ``(embeds, pooled_or_None)``.
    """
    import torch

    text = text or ""
    # Split on a standalone BREAK token (whitespace-delimited, case-sensitive to
    # match A1111). Empty segments (e.g. leading/trailing BREAK) are dropped.
    parts = [seg.strip() for seg in _split_break(text)]
    parts = [seg for seg in parts if seg] or [""]

    if len(parts) == 1:
        out = compel(parts[0])
    else:
        chunks = [compel(seg) for seg in parts]
        # compel returns either a tensor (SD1.5) or (embeds, pooled) (SDXL).
        if isinstance(chunks[0], tuple):
            embeds = torch.cat([c[0] for c in chunks], dim=1)
            # Pooled is a single vector per prompt; BREAK only concatenates the
            # sequence dim, so reuse the FIRST segment's pooled (its global summary).
            out = (embeds, chunks[0][1])
        else:
            out = torch.cat(chunks, dim=1)

    if isinstance(out, tuple):
        return out[0], out[1]
    return out, None


def _split_break(text: str) -> list[str]:
    """Split *text* on the standalone ``BREAK`` keyword (its own word)."""
    import re
    return re.split(r"\bBREAK\b", text)


def _pad_to_same_length(compel, pos, neg):
    """Pad ``pos``/``neg`` embeds to the same sequence length. Tries compel's helper
    (the API moved between an instance method and ``compel.utils`` across versions),
    then a manual zero-pad on the token axis so CFG batching always lines up."""
    # 1. Instance method (older/most common compel API).
    fn = getattr(compel, "pad_conditioning_tensors_to_same_length", None)
    if callable(fn):
        try:
            out = fn([pos, neg])
            return out[0], out[1]
        except Exception:
            logger.debug("compel instance pad failed; trying utils", exc_info=True)
    # 2. Module function in compel.utils.
    try:
        from compel.utils import pad_conditioning_tensors_to_same_length as _putil
        out = _putil([pos, neg])
        return out[0], out[1]
    except Exception:
        logger.debug("compel.utils pad failed; manual pad", exc_info=True)
    # 3. Manual zero-pad on the token (dim=1) axis.
    try:
        import torch
        lp, ln = pos.shape[1], neg.shape[1]
        target = max(lp, ln)
        def _pad(t):
            if t.shape[1] >= target:
                return t
            pad = torch.zeros(t.shape[0], target - t.shape[1], t.shape[2],
                              dtype=t.dtype, device=t.device)
            return torch.cat([t, pad], dim=1)
        return _pad(pos), _pad(neg)
    except Exception:
        # Last resort: return unchanged; diffusers will raise a clear error and the
        # caller's build_embeds except-clause falls back to the raw prompt path.
        logger.debug("manual pad failed", exc_info=True)
        return pos, neg


def build_embeds(pipe, model_type: str, prompt: str, negative: str,
                 clip_skip: int = 1) -> dict | None:
    """Build diffusers ``*_embeds`` kwargs for *pipe* from *prompt*/*negative*.

    Returns a kwargs dict to MERGE into the pipeline call (replacing ``prompt`` /
    ``negative_prompt`` / ``clip_skip``):

        {"prompt_embeds", "negative_prompt_embeds"
         [, "pooled_prompt_embeds", "negative_pooled_prompt_embeds"]}

    or ``None`` on any failure — the caller MUST then fall back to the raw-string
    path (passing the original prompt/negative strings + clip_skip). The clip_skip
    is baked into the embeddings here, so the caller must NOT pass ``clip_skip``
    alongside the returned kwargs (they're mutually exclusive in diffusers).
    """
    if pipe is None:
        return None
    try:
        import torch  # noqa: F401
    except Exception:
        return None
    try:
        compel, is_sdxl = _build_compel(pipe, model_type, clip_skip)
    except Exception:
        logger.info("compel unavailable / pipe unsupported — using raw prompt path",
                    exc_info=True)
        return None

    try:
        pos_embeds, pos_pooled = _encode_one(compel, prompt or "")
        neg_embeds, neg_pooled = _encode_one(compel, negative or "")

        # Pad the positive/negative sequences to equal length so diffusers' CFG
        # batch (cat of cond+uncond) lines up — mismatched lengths from BREAK / >77
        # chunking otherwise raise. Prefer compel's own helper (instance method or
        # the compel.utils function across versions); fall back to a manual zero-pad.
        pos_embeds, neg_embeds = _pad_to_same_length(compel, pos_embeds, neg_embeds)

        kwargs: dict = {
            "prompt_embeds": pos_embeds,
            "negative_prompt_embeds": neg_embeds,
        }
        if is_sdxl:
            if pos_pooled is None or neg_pooled is None:
                # SDXL strictly needs pooled embeds; bail to the raw path rather
                # than feed a half-built call.
                logger.info("SDXL pooled embeds missing — using raw prompt path")
                return None
            kwargs["pooled_prompt_embeds"] = pos_pooled
            kwargs["negative_pooled_prompt_embeds"] = neg_pooled
        return kwargs
    except Exception:
        logger.warning("compel prompt encode failed — falling back to raw prompt",
                       exc_info=True)
        return None
    finally:
        # compel holds no GPU weights of its own (it borrows the pipe's encoders),
        # so just drop the python reference; the encoders stay loaded on the pipe.
        del compel
