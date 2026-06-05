"""Shared "saintorphan" right-click context menu — Replicant-compatible.

The engine scaffold below is copied VERBATIM from Replicant Character Lab's
_CTX_MENU_JS so every saintorphan plugin creates the *same* window.SaintorphanMenu.
It's idempotent (`if(!window.SaintorphanMenu){…}`): whichever plugin loads first
builds it, the rest skip the create block and just register. Injected via an
`<img onerror>` because gr.HTML doesn't run <script>; the whole thing is
single-quoted JS inside a double-quoted onerror attribute.

Shared API (do NOT diverge — must match Replicant's):
    M.register(match, label, handler)  match = 'image' | 'video' | CSS selector;
        item shows when the right-clicked element matches. handler(matchedEl) on click.
    M.announce(name)                   declare this plugin present.
    M.whenPresent(name, cb)            cb fires now if present, else when name announces.
    M.srcOf(el)                        data-media-src attr → el src → child img/video src.

ImageSuite announces 'imagesuite' and registers its items against every <img>
app-wide. Because a page can only replace the native menu (not append to it), the menu
folds in the standard image actions (open / save / copy) alongside the Send-to items
(Img2Vid, Img2Img, MultiCanvas) so nothing is lost where it takes over. A persisted
Settings toggle restricts the match to '#imagesuite-root img' (ImageSuite's own images
only) so Wan2GP's native right-click stays intact elsewhere — flipped LIVE, no reload,
via window.__imagesuiteScope(bool) which rewrites the items' match in place. Send
handlers relay {a,s,t} JSON into the hidden #imagesuite-ctx-relay Textbox for the Python
router; it also attaches to Reel2Reel's '.r2r-timeline-clip' surface once that announces.
"""

RELAY_ID = "imagesuite-ctx-relay"

# --- engine scaffold: copied verbatim from Replicant's _CTX_MENU_JS -----------
_ENGINE = """
if(!window.SaintorphanMenu){var M=window.SaintorphanMenu={items:[],present:{},_w:{}};
M.announce=function(n){M.present[n]=true;(M._w[n]||[]).forEach(function(f){try{f();}catch(e){console.error(e);}});M._w[n]=[];};
M.whenPresent=function(n,cb){if(M.present[n]){try{cb();}catch(e){console.error(e);}}else{(M._w[n]||(M._w[n]=[])).push(cb);}};
M.register=function(match,label,handler){M.items.push({match:match,label:label,handler:handler});};
M.srcOf=function(el){if(!el)return '';var a=el.getAttribute&&el.getAttribute('data-media-src');if(a)return a;if(el.currentSrc||el.src)return el.currentSrc||el.src;var q=el.querySelector&&el.querySelector('img,video');return q?(q.currentSrc||q.src||''):'';};
function hit(match,el){if(match==='image')return el.closest('img');if(match==='video')return el.closest('video');try{return el.closest(match);}catch(e){return null;}}
function close(){var m=document.getElementById('saintorphan-ctx');if(m)m.remove();}
function build(x,y,hits){close();var menu=document.createElement('div');menu.id='saintorphan-ctx';menu.style.cssText='position:fixed;z-index:99999;background:#1f2430;border:1px solid #3a3f4b;border-radius:8px;padding:4px 0;box-shadow:0 6px 24px rgba(0,0,0,.5);min-width:210px;font-family:sans-serif;font-size:13px;color:#e5e7eb;';var h=document.createElement('div');h.textContent='OrphanSuite';h.style.cssText='padding:4px 14px;font-weight:700;color:#e83e8c;cursor:default;user-select:none;';menu.appendChild(h);var hr=document.createElement('div');hr.style.cssText='height:1px;background:#3a3f4b;margin:4px 0;';menu.appendChild(hr);hits.forEach(function(hk){var el=document.createElement('div');el.textContent=hk.it.label;el.style.cssText='padding:6px 14px;cursor:pointer;white-space:nowrap;';el.onmouseenter=function(){el.style.background='#2d3340';};el.onmouseleave=function(){el.style.background='';};el.addEventListener('click',function(ev){ev.stopPropagation();close();try{hk.it.handler(hk.el);}catch(err){console.error(err);}});menu.appendChild(el);});document.body.appendChild(menu);var r=menu.getBoundingClientRect();if(x+r.width>window.innerWidth)x=window.innerWidth-r.width-6;if(y+r.height>window.innerHeight)y=window.innerHeight-r.height-6;menu.style.left=x+'px';menu.style.top=y+'px';}
document.addEventListener('contextmenu',function(e){var hits=[];M.items.forEach(function(it){var el=hit(it.match,e.target);if(el)hits.push({it:it,el:el});});if(!hits.length)return;e.preventDefault();build(e.clientX,e.clientY,hits);},true);
document.addEventListener('click',close);document.addEventListener('scroll',close,true);}
"""

# --- ImageSuite's own registration (guarded by M._imagesuite) -----------------
# SCOPED to '#imagesuite-root img' — only ImageSuite's OWN images, so we never hijack
# Wan2GP's native right-click elsewhere in the app. Because suppressing the native menu
# is all a web page can do (you can't append to it), the menu also folds in the standard
# image actions (open / save / copy) so nothing is lost where we DO take over. relay()
# writes {a,s,t} JSON into #imagesuite-ctx-relay for the Python router.
_REGISTER = """
var M=window.SaintorphanMenu;if(!M._imagesuite){M._imagesuite=true;M.announce('imagesuite');
var relay=function(action,el){var src=M.srcOf(el);if(!src)return;var b=document.querySelector('#imagesuite-ctx-relay textarea')||document.querySelector('#imagesuite-ctx-relay input');if(!b)return;b.value=JSON.stringify({a:action,s:src,t:Date.now()});b.dispatchEvent(new Event('input',{bubbles:true}));};
var openTab=function(el){var s=M.srcOf(el);if(s)window.open(s,'_blank','noopener');};
var saveImg=function(el){var s=M.srcOf(el);if(!s)return;var a=document.createElement('a');a.href=s;a.download=(s.split('?')[0].split('/').pop()||'image.png');document.body.appendChild(a);a.click();a.remove();};
var copyImg=function(el){var s=M.srcOf(el);if(!s)return;if(navigator.clipboard&&window.ClipboardItem&&s.indexOf('javascript:')!==0){fetch(s).then(function(r){return r.blob();}).then(function(bl){var it={};it[bl.type||'image/png']=bl;return navigator.clipboard.write([new ClipboardItem(it)]);}).catch(function(){if(navigator.clipboard)navigator.clipboard.writeText(s);});}else if(navigator.clipboard){navigator.clipboard.writeText(s);}};
var IMG=[];
var regImg=function(label,handler){M.register('image',label,handler);IMG.push(M.items[M.items.length-1]);};
regImg('🔗 Open image in new tab',openTab);regImg('💾 Save image',saveImg);regImg('📋 Copy image',copyImg);
regImg('Send to Img2Vid',function(el){relay('img2vid',el);});regImg('ImageSuite (Img2Img)',function(el){relay('img2img',el);});regImg('ImageSuite (MultiCanvas)',function(el){relay('inpaint',el);});
window.__imagesuiteScope=function(pluginOnly){var sel=pluginOnly?'#imagesuite-root img':'image';IMG.forEach(function(it){it.match=sel;});};
M.whenPresent('reel2reel',function(){M.register('.r2r-timeline-clip','Send to Img2Vid',function(el){relay('img2vid',el);});M.register('.r2r-timeline-clip','ImageSuite (Img2Img)',function(el){relay('img2img',el);});M.register('.r2r-timeline-clip','ImageSuite (MultiCanvas)',function(el){relay('inpaint',el);});});}
"""


def _collapse(js: str) -> str:
    """One space-joined line (no string literal spans lines, so this is safe)."""
    return " ".join(ln.strip() for ln in js.splitlines() if ln.strip())


def imagesuite_ctx_html(plugin_only: bool = False) -> str:
    """The <img onerror> that installs the shared engine (idempotent) + registers
    ImageSuite's items. Render once via gr.HTML inside the plugin tab.

    The menu matches every <img> app-wide by default. ``plugin_only`` (a persisted
    Settings toggle) restricts it to '#imagesuite-root img' at load; the same scope can
    be flipped LIVE (no reload) via window.__imagesuiteScope(bool), which just rewrites
    the registered items' match — the capture-phase listener reads it on the next click.
    """
    inner = _collapse(_ENGINE) + " " + _collapse(_REGISTER)
    if plugin_only:
        inner += " try{window.__imagesuiteScope(true);}catch(e){}"
    return ("<img src=x style='display:none' onerror=\"(function(){"
            + inner + "})()\">")
