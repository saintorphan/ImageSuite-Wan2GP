"""SendTo inbox for Image Suite.

The SendTo plugin (or anything honouring the same contract) hands frames to this
plugin by appending ``{"path", "slot"}`` dicts to ``state["imagesuite_inbox"]`` on
the shared per-session state — see this plugin's ``sendto.json`` manifest. We drain
them on tab entry (``on_tab_select``) and load each into its slot (img2img init,
MultiCanvas / Modify canvas background). Pure stdlib; no cross-plugin imports.
"""
from __future__ import annotations

import threading

INBOX_KEY = "imagesuite_inbox"
_MAX = 200
_lock = threading.Lock()


def drain(state) -> list[dict]:
    """Return and clear the queued frames (atomic swap so a concurrent enqueue
    can't lose items). Each item is ``{"path": str, "slot": str|None}``."""
    if not isinstance(state, dict):
        return []
    with _lock:
        box = state.get(INBOX_KEY) or []
        state[INBOX_KEY] = []
        return list(box)


def peek(state) -> list[dict]:
    if not isinstance(state, dict):
        return []
    with _lock:
        return list(state.get(INBOX_KEY) or [])


def enqueue_frame(state, path, slot=None) -> None:
    """Append one frame (mirrors the SendTo contract; handy for in-process tests
    or other local callers)."""
    if not isinstance(state, dict) or not path:
        return
    with _lock:
        box = list(state.get(INBOX_KEY) or [])
        box.append({"path": str(path), "slot": slot})
        state[INBOX_KEY] = box[-_MAX:]
