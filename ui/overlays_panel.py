"""The Overlays library tab — a Gradio-native file manager for overlay images.

Folders + thumbnails with full CRUD (create/delete folders, upload, rename, move,
delete, preview). The Overlays strip beneath the MultiCanvas canvas is a
read-only view of the same library. Wiring lives in plugin.py.
"""
from __future__ import annotations

import base64

import gradio as gr

from ..core import overlays as _ov

# File-browser interactions for the Overlays gallery: right-click menus (empty
# space / folder / image), inline rename/delete via prompt/confirm, and drag-drop
# upload. JS drives two hidden bridge textboxes (#imagesuite-ov-action carries a
# small {op,idx,arg,nonce}; #imagesuite-ov-upload carries base64 images) which
# plugin.py wires to core.overlays. File ops pass the thumbnail INDEX (computed
# from the DOM) so Python resolves index→filename — no select round-trip races.
_OV_JS = r"""
(function(){
  if (window.__ovBrowser) return;
  window.__ovBrowser = 1;
  var GAL="imagesuite-ov-gallery", FOL="imagesuite-ov-folder",
      ACT="imagesuite-ov-action", UP="imagesuite-ov-upload";
  function setBridge(id, val){
    var el = document.querySelector("#"+id+" textarea, #"+id+" input");
    if(!el) return;
    el.value = val;
    el.dispatchEvent(new Event("input", {bubbles:true}));
  }
  function act(o){ o.nonce = Date.now()+":"+Math.random(); setBridge(ACT, JSON.stringify(o)); }
  function sendUpload(items){ if(items.length) setBridge(UP, JSON.stringify({items:items, nonce:Date.now()})); }
  function gal(){ return document.getElementById(GAL); }
  function curFolder(){ var el=document.querySelector("#"+FOL+" input"); return el ? (el.value||"") : ""; }
  function inRoot(){ var f=curFolder(); return !f || f === "(root)"; }
  function thumbIndex(t){
    // .thumbnail-lg = the GRID cells only. (.thumbnail-item also matches the preview
    // strip's .thumbnail-small when a selection is open, which would double the count
    // and misalign the index → wrong/failed file ops.) Grid order == list_images order.
    var g=gal(); if(!g || !t.closest) return -1;
    var th=t.closest(".thumbnail-lg"); if(!th || !g.contains(th)) return -1;
    return Array.prototype.indexOf.call(g.querySelectorAll(".thumbnail-lg"), th);
  }
  var menu=document.createElement("div"); menu.id="imagesuite-ovmenu";
  document.body.appendChild(menu);
  function hideMenu(){ menu.style.display="none"; }
  function showMenu(e, items){
    menu.innerHTML="";
    items.forEach(function(it){
      var d=document.createElement("div"); d.className="ovmenu-item"; d.textContent=it.t;
      d.addEventListener("click", function(ev){ ev.stopPropagation(); hideMenu(); it.f(); });
      menu.appendChild(d);
    });
    menu.style.left=e.pageX+"px"; menu.style.top=e.pageY+"px"; menu.style.display="block";
  }
  document.addEventListener("click", hideMenu);
  document.addEventListener("scroll", hideMenu, true);
  function newFolder(){ var n=prompt("New folder name:"); if(n) act({op:"new_folder", arg:n}); }
  function renameFolder(){ var n=prompt("Rename folder to:", curFolder()); if(n && n!==curFolder()) act({op:"rename_folder", arg:n}); }
  function deleteFolder(){
    var g=gal(), n=g ? g.querySelectorAll(".thumbnail-lg").length : 0;
    if(n>0 && !confirm("\""+curFolder()+"\" has "+n+" image(s). Delete the folder and all of them?")) return;
    act({op:"delete_folder"});
  }
  function renameFile(i){ var n=prompt("Rename image to:"); if(n) act({op:"rename_file", idx:i, arg:n}); }
  function deleteFile(i){ if(confirm("Delete this image?")) act({op:"delete_file", idx:i}); }
  function moveUp(i){ act({op:"moveup_file", idx:i}); }
  function pickUpload(){
    var inp=document.createElement("input"); inp.type="file"; inp.accept="image/*"; inp.multiple=true;
    inp.addEventListener("change", function(){ readFiles(inp.files); });
    inp.click();
  }
  function readFiles(files){
    var arr=Array.prototype.slice.call(files||[]).filter(function(f){ return f.type && f.type.indexOf("image/")===0; }).slice(0,50);
    if(!arr.length) return;
    var out=[], done=0;
    arr.forEach(function(f){
      var r=new FileReader();
      r.onload=function(){ out.push({name:f.name, dataurl:r.result}); if(++done===arr.length) sendUpload(out); };
      r.onerror=function(){ if(++done===arr.length) sendUpload(out); };
      r.readAsDataURL(f);
    });
  }
  document.addEventListener("contextmenu", function(e){
    var g=gal(), fol=document.getElementById(FOL);
    var inFol = fol && fol.contains(e.target), inGal = g && g.contains(e.target);
    if(!inFol && !inGal) return;          // not ours — leave other menus alone
    e.preventDefault();
    // The shared SaintorphanMenu listens in the CAPTURE phase (fires before this
    // bubble handler) and matches any <img>, so it builds #saintorphan-ctx over our
    // overlay thumbnails. Remove it here so our menu owns the gallery.
    var sm = document.getElementById("saintorphan-ctx"); if(sm) sm.remove();
    if(inFol){
      var items=[{t:"➕ New folder", f:newFolder}, {t:"⬆ Upload images", f:pickUpload}];
      if(!inRoot()){ items.push({t:"✎ Rename folder", f:renameFolder}); items.push({t:"🗑 Delete folder", f:deleteFolder}); }
      showMenu(e, items); return;
    }
    var i=thumbIndex(e.target);
    if(i>=0){
      var fi=[{t:"✎ Rename", f:function(){renameFile(i);}}, {t:"🗑 Delete", f:function(){deleteFile(i);}}];
      if(!inRoot()) fi.push({t:"⬆ Move up (to root)", f:function(){moveUp(i);}});
      showMenu(e, fi);
    } else {
      showMenu(e, [{t:"➕ New folder", f:newFolder}, {t:"⬆ Upload images", f:pickUpload}]);
    }
  });
  document.addEventListener("dragover", function(e){ var g=gal(); if(g && g.contains(e.target)){ e.preventDefault(); g.classList.add("ov-dragover"); } });
  document.addEventListener("dragleave", function(e){ var g=gal(); if(g && !g.contains(e.relatedTarget)) g.classList.remove("ov-dragover"); });
  document.addEventListener("drop", function(e){ var g=gal(); if(g && g.contains(e.target)){ e.preventDefault(); g.classList.remove("ov-dragover"); if(e.dataTransfer && e.dataTransfer.files) readFiles(e.dataTransfer.files); } });
})();
"""
# Gradio strips <script>; inject via an <img onerror> that builds a <script> whose
# body is the base64-decoded JS — so no attribute-escaping of the (large) blob.
_OV_HTML = ('<img src=x alt="" style="display:none" onerror="'
            "var s=document.createElement('script');"
            # decodeURIComponent(escape(atob(...))) — atob yields a Latin-1 byte
            # string, so this turns the UTF-8 bytes back into real chars (the menu
            # emoji would otherwise be mojibake).
            "s.textContent=decodeURIComponent(escape(atob('"
            + base64.b64encode(_OV_JS.encode()).decode() + "')));"
            "document.head.appendChild(s);this.remove()"
            '">')


def build_overlays_panel():
    c = {}
    gr.Markdown(
        "Your **overlay library** — transparent PNGs, frames, stickers, watermarks "
        "(any image). Organise into folders here; they show up in the **Overlays** "
        "strip beneath the **MultiCanvas** canvas, where you can drag one onto the "
        "image as a new layer.\n\n"
        "**Right-click** an image for Rename / Delete / Move-up, right-click a folder "
        "(the picker) for Rename / Delete, right-click empty grid space to add a "
        "folder or upload — and **drag images straight onto the grid** to add them.",
        elem_classes="imagesuite-help")

    folders = _ov.list_folders()
    cur = folders[0] if folders else _ov.ROOT_LABEL

    # Folder selector = navigation (the grid shows images, not folders, so we still
    # need a way to switch folders). Everything else is right-click / drag-drop — no
    # redundant buttons: right-click the selector for New / Upload / Rename / Delete
    # folder; right-click the grid for image actions or upload; drag images in.
    c["folder"] = gr.Dropdown(label="Folder  ·  right-click for folder actions",
                              choices=folders, value=cur,
                              elem_id="imagesuite-ov-folder")
    c["gallery"] = gr.Gallery(
        label="Overlays — click to enlarge · right-click for actions · drag images in",
        columns=6, height=520, object_fit="contain",
        value=_ov.list_images(cur), elem_id="imagesuite-ov-gallery",
        elem_classes="imagesuite-gallery")
    c["status"] = gr.Markdown("", elem_classes="imagesuite-help")

    # Hidden bridges driven by the file-browser JS (wired in plugin._wire_overlays).
    c["ov_action"] = gr.Textbox(visible=False, elem_id="imagesuite-ov-action")
    c["ov_upload"] = gr.Textbox(visible=False, elem_id="imagesuite-ov-upload")
    gr.HTML(_OV_HTML, elem_classes="imagesuite-hidden")
    return c
