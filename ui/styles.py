"""Image Suite CSS — light touch; tab accent, sub-tab tidy + the logo banner."""

CSS = """
#imagesuite-root { position: relative; }
#imagesuite-root .imagesuite-acc { border-radius: 10px; }
#imagesuite-root .imagesuite-sendrow button { min-width: 0; }
.imagesuite-hidden { display: none !important; }
/* Gold border around our main-webui tab button (tagged from JS in plugin.py),
   the way Replicant Character Lab accents its tab in purple. */
button.imagesuite-tabbtn {
    border: 2px solid #d4af37 !important;
    border-radius: 8px !important;
    box-shadow: 0 0 7px rgba(212,175,55,0.55) !important;
}
#imagesuite-root .imagesuite-gallery { min-height: 320px; }
#imagesuite-root .imagesuite-help { font-size: 12px; opacity: 0.8; }
/* SDXL load / working notice under Generate — accent so it's noticed */
#imagesuite-root .imagesuite-genstatus { font-size: 13px; color: #e83e8c;
    margin: 2px 0 4px; min-height: 0; }
/* taller send-to + Save As + Generate/Abort buttons (uniform action-button height) */
#imagesuite-root .imagesuite-sendrow button,
#imagesuite-root .imagesuite-genrow button,
#imagesuite-root .imagesuite-savebtn { min-height: 46px; }
/* square reference thumbnails for face/body/colour swap */
#imagesuite-root .imagesuite-refthumb { aspect-ratio: 1 / 1; max-width: 160px; }
#imagesuite-root .imagesuite-refthumb img { object-fit: contain; }
/* img2img init image as a compact thumbnail — double-click to enlarge (JS below) */
#imagesuite-root .imagesuite-initthumb { max-width: 260px; }
#imagesuite-root .imagesuite-initthumb img { object-fit: contain; cursor: zoom-in; }
/* shared full-screen lightbox — appended to <body>, so NOT scoped under #imagesuite-root */
#imagesuite-lightbox { position: fixed; inset: 0; z-index: 9999; display: none;
    align-items: center; justify-content: center; cursor: zoom-out;
    background: rgba(0, 0, 0, 0.85); }
#imagesuite-lightbox img { max-width: 92%; max-height: 92%; object-fit: contain;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.6); }

/* Logo banner — a left-aligned header above the sub-tabs. width:auto +
   object-fit keep the 4:1 artwork from being squashed. */
/* Banner row: logo on the left, GitHub link far right, both bottom-aligned. */
#imagesuite-banner {
    display: flex; align-items: flex-end; justify-content: space-between;
    gap: 12px; margin: 4px 0 10px 2px;
}
#imagesuite-banner img {
    height: 104px; width: auto; max-width: 520px;
    object-fit: contain; display: block;
}
#imagesuite-banner h2 { margin: 0; color: #e83e8c; font-style: italic; }
#imagesuite-banner #imagesuite-gh {
    display: inline-flex; align-items: center; gap: 5px;
    color: #d4af37; text-decoration: none; font-size: 13px;
    padding-bottom: 6px; white-space: nowrap; flex: 0 0 auto;
}
#imagesuite-banner #imagesuite-gh:hover { text-decoration: underline; }
"""

# Page-level JS injected once (Gradio strips <script>, so we use the <img onerror>
# trick the tab-accent code uses). Delegated dblclick: double-clicking any
# `.imagesuite-initthumb` thumbnail opens its image in a full-screen lightbox;
# click anywhere to dismiss. Event delegation on document survives Gradio
# re-renders, so no MutationObserver is needed.
LIGHTBOX_HTML = (
    "<img src=x style=\"display:none\" onerror=\""
    "(function(){if(window.__isLightbox)return;window.__isLightbox=1;"
    "var ov=document.createElement('div');ov.id='imagesuite-lightbox';"
    "var im=document.createElement('img');ov.appendChild(im);"
    "document.body.appendChild(ov);"
    "ov.addEventListener('click',function(){ov.style.display='none';});"
    "document.addEventListener('dblclick',function(e){"
    "var t=e.target&&e.target.closest&&e.target.closest('.imagesuite-initthumb');"
    "if(!t)return;var s=t.querySelector('img');if(!s||!s.src)return;"
    "im.src=s.src;ov.style.display='flex';});})()\">"
)
