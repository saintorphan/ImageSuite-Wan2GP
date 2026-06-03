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
/* Galleries: scroll vertically to show EVERY image while staying a grid.
   Gradio gives the gallery .block a fixed inline height + overflow:hidden, and its
   .grid-wrap already has overflow-y:scroll — but the percentage-height chain is
   broken: the intermediate .gallery-container div ships with NO css (height:auto),
   so .grid-wrap's height:100% has no basis, collapses to content height, and the
   fixed .block clips the overflow with no scrollbar. Repairing .gallery-container's
   height lets the block's height propagate so .grid-wrap fills it and scrolls.
   !important beats Gradio's .svelte-* scoped rules (which use none) + the inline
   grid style. (Verified against Gradio 5.29 gallery DOM: .block > .gallery-container
   > .grid-wrap > .grid-container; label is an absolute overlay, not in flow.) */
#imagesuite-root .imagesuite-gallery .gallery-container {
    height: 100% !important; min-height: 0 !important;
}
#imagesuite-root .imagesuite-gallery .grid-wrap {
    height: 100% !important; max-height: 100% !important; min-height: 0 !important;
    overflow-y: auto !important; overflow-x: hidden !important;
}
#imagesuite-root .imagesuite-gallery .grid-container {
    height: auto !important; min-height: 0 !important;
}
/* Results galleries only (txt2img / img2img / MultiCanvas): keep result thumbnails
   legible — pin a usable min row height so overflow ADDS rows (and the wrapper
   scrolls) instead of the default minmax(100px,1fr) cramming them. The shared
   overlays picker (.imagesuite-gallery, 6 columns) keeps its denser default rows. */
#imagesuite-root .imagesuite-results .grid-container {
    grid-auto-rows: minmax(180px, auto) !important;
    grid-template-rows: none !important;
}
#imagesuite-root .imagesuite-help { font-size: 12px; opacity: 0.8; }
/* Helper-weights manager rows: keep the status text + Download/Link buttons on one
   centered line. */
#imagesuite-root .imagesuite-modelrow { align-items: center; }
#imagesuite-root .imagesuite-modelrow p { margin: 2px 0; }
/* Overlays file-browser: right-click menu (appended to <body>, so NOT scoped under
   #imagesuite-root) + a drop-target highlight on the gallery. */
#imagesuite-ovmenu { position: fixed; z-index: 10000; display: none; min-width: 168px;
    background: #1d1d25; border: 1px solid #444; border-radius: 8px; padding: 4px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5); font-size: 13px; }
#imagesuite-ovmenu .ovmenu-item { padding: 6px 12px; border-radius: 5px;
    cursor: pointer; color: #ddd; white-space: nowrap; }
#imagesuite-ovmenu .ovmenu-item:hover { background: #e83e8c; color: #fff; }
#imagesuite-root .imagesuite-gallery.ov-dragover { outline: 2px dashed #e83e8c;
    outline-offset: -4px; }
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
/* Project bar (under the banner): centered project name + far-right CRUD. */
#imagesuite-projbar { display: flex; align-items: center; gap: 6px;
    margin: -6px 2px 12px 2px; }
#imagesuite-projbar #imagesuite-projname { text-align: center; min-width: 0; }
#imagesuite-projbar #imagesuite-projname p { margin: 0; font-weight: 600;
    font-style: italic; color: #d4af37; }
#imagesuite-projbar #imagesuite-projname .imagesuite-unsaved p { color: #8a8a93; }
#imagesuite-projbar button { min-width: 0; }
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
