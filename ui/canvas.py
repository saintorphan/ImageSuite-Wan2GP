"""PaintShop draw + mask canvas for the MultiCanvas page.

A custom HTML5 canvas embedded as an <iframe srcdoc>, bridged to Gradio the same
way the bundled Multi-Angle Prompt Helper plugin does:

  * JS -> Python: writes data-URLs into hidden Gradio Textboxes via
    ``parent.document`` + the native value setter + input/change events.
  * Python -> JS: a hidden <iframe srcdoc> whose <script> calls
    ``parent.window.__is_<mode>_setbg(dataURL)`` to load a background image.

Layer model (bottom → top): a fixed **background** (the loaded image), one or more
**draw** layers (colour paint), and a **mask** layer on top. Drawing routes by the
**Mask / Draw / Draw+Mask** mode: Draw+Mask paints the active draw layer AND the
mask together, so drawn content is automatically the region that gets regenerated.
A horizontal **layers manager** below the canvas adds / deletes / hides / reorders
(drag) the draw layers.

On generate the frame exports:
  * composite = background + all visible draw layers (the inpaint init image)
  * mask      = white-on-black region to regenerate (mask layer, plus the dilated
                draw area when Auto-mask is on)
"""
from __future__ import annotations

import html as _html

_PALETTE = ["#000000", "#ffffff", "#e83e8c", "#ff0000", "#ff7f00", "#ffd400",
            "#2ecc71", "#1e90ff", "#2c3e50", "#8e44ad", "#a0522d", "#f5cba7"]

_CANVAS_DOC = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
*{margin:0;padding:0;box-sizing:border-box;font-family:system-ui,sans-serif}
html,body{height:100%}
body{background:#15151b;color:#ddd;overflow:hidden;user-select:none}
#root{display:flex;flex-direction:column;height:100%;width:100%}
#main{display:flex;flex:1;min-height:0;position:relative}
#wrap{flex:1;position:relative;overflow:auto;
  background:#101015 repeating-conic-gradient(#1a1a22 0% 25%,#141419 0% 50%) 0/24px 24px}
#stage{position:relative;margin:10px auto;box-shadow:0 0 0 1px #000,0 6px 24px rgba(0,0,0,.5)}
#stage canvas{position:absolute;top:0;left:0;display:block}
#cursor{pointer-events:none}
#disp{position:relative;cursor:crosshair}
#empty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  color:#555;font-size:13px;text-align:center;pointer-events:none;line-height:1.6}
#rail{width:172px;flex:0 0 172px;overflow-y:auto;background:#1d1d25;
  border-left:1px solid #333;padding:8px;display:flex;flex-direction:column;gap:12px}
.sec{display:flex;flex-direction:column;gap:5px}
.seclabel{font-size:10px;letter-spacing:.08em;color:#e83e8c;font-weight:700;
  border-bottom:1px solid #333;padding-bottom:3px;text-transform:uppercase}
#rail button{background:#2a2a35;border:1px solid #3a3a48;color:#cfcfe0;border-radius:6px;
  padding:6px 4px;font-size:11px;cursor:pointer;line-height:1.1}
#rail button.on{background:#e83e8c;border-color:#e83e8c;color:#fff}
#railhead{display:flex;justify-content:flex-end;margin:-2px 0 2px}
#railhead button{padding:1px 9px;font-size:15px;line-height:1.2}
#railexpand{position:absolute;top:10px;right:10px;z-index:20;background:#2a2a35;
  border:1px solid #3a3a48;color:#cfcfe0;border-radius:6px;padding:4px 10px;font-size:15px;
  line-height:1;cursor:pointer;display:none}
#main.railcollapsed #rail{display:none}
#main.railcollapsed #railexpand{display:block}
.ovicon{width:34px;height:28px;padding:0;font-size:15px;display:inline-flex;
  align-items:center;justify-content:center;line-height:1}
.modeseg{display:grid;grid-template-columns:1fr 1fr 1fr;gap:0}
.modeseg button{border-radius:0}
.modeseg button:first-child{border-radius:6px 0 0 6px}
.modeseg button:last-child{border-radius:0 6px 6px 0}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:4px}
.row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px}
.palrow{display:grid;grid-template-columns:repeat(6,1fr);gap:4px}
.sw{width:100%;aspect-ratio:1;border-radius:4px;border:2px solid #444;cursor:pointer}
.sw.on{border-color:#fff}
.fld{font-size:10px;color:#8a8a9a;display:flex;justify-content:space-between}
input[type=range]{width:100%}
input[type=color]{width:100%;height:26px;padding:0;border:1px solid #3a3a48;
  background:none;border-radius:6px;cursor:pointer}
/* horizontal layers manager */
#layers{height:74px;flex:0 0 74px;display:flex;align-items:center;gap:6px;
  padding:6px 8px;background:#1a1a22;border-top:1px solid #333;overflow-x:auto}
#layers .lyr{display:flex;align-items:center;gap:6px;background:#252531;
  border:1px solid #3a3a48;border-radius:8px;padding:5px 8px;font-size:11px;
  white-space:nowrap;cursor:grab}
#layers .lyr.active{border-color:#e83e8c;box-shadow:0 0 0 1px #e83e8c}
#layers .lyr.base{cursor:default;opacity:.85}
#layers .lyr.mask{cursor:pointer;border-color:#ff3344}
#layers .lyr .eye{cursor:pointer;opacity:.9}
#layers .lyr .eye.off{opacity:.3}
#layers .lyr .del{cursor:pointer;color:#ff6b6b;font-weight:700}
#layers .addlyr{background:#2a2a35;border:1px dashed #555;color:#9ad;border-radius:8px;
  padding:6px 10px;font-size:11px;cursor:pointer}
/* overlays strip (below the layers manager) */
#ovbar{flex:0 0 auto;background:#181820;border-top:1px solid #333}
#ovhead{display:flex;align-items:center;gap:10px;padding:5px 8px;font-size:11px}
#ovtoggle,.ovbtn{background:#2a2a35;border:1px solid #3a3a48;color:#cfcfe0;border-radius:6px;
  padding:3px 9px;font-size:11px;cursor:pointer}
#ovhint{color:#7a7a8a;font-size:10px}
#ovstrip{display:none;align-items:center;gap:8px;padding:0 8px 8px;overflow-x:auto;height:86px}
#ovstrip.open{display:flex}
#ovstrip .ov{flex:0 0 auto;width:70px;height:70px;border:1px solid #3a3a48;border-radius:8px;
  background:#101015 repeating-conic-gradient(#1a1a22 0% 25%,#141419 0% 50%) 0/12px 12px;
  cursor:grab;display:flex;align-items:center;justify-content:center;overflow:hidden}
#ovstrip .ov img{max-width:100%;max-height:100%;object-fit:contain;pointer-events:none}
#ovstrip .ovempty{color:#666;font-size:11px;padding:6px}
#ovprev{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;
  justify-content:center;z-index:50;cursor:zoom-out}
#ovprev.open{display:flex}
#ovprev img{max-width:92%;max-height:92%;object-fit:contain;
  box-shadow:0 0 0 1px #000,0 10px 40px rgba(0,0,0,.6)}
.dragover #stage{outline:2px dashed #e83e8c;outline-offset:3px}
</style></head><body>
<div id="root">
  <div id="main">
    <div id="wrap"><div id="stage">
      <canvas id="bg"></canvas>
      <canvas id="maskv"></canvas>
      <canvas id="disp"></canvas>
      <canvas id="cursor"></canvas>
      <div id="empty">No image yet.<br>Use <b>&#8593; Upload</b> at the top of the tool rail,<br>or a "Send to MultiCanvas" button.</div>
    </div></div>
    <div id="rail">
      <div id="railhead"><button id="railcollapse" title="Collapse the tool panel">&#187;</button></div>
      <div class="sec">
        <div class="seclabel">Mode</div>
        <div class="modeseg" id="modeseg">
          <button data-mode="mask" title="Mask mode — paint only the mask (the region that gets regenerated)">&#127917;<br>Mask</button>
          <button data-mode="draw" class="on" title="Draw mode — paint only the active layer">&#9999;&#65039;<br>Draw</button>
          <button data-mode="drawmask" title="Draw + Mask — paint the layer AND the mask together">&#9999;&#65039;&#127917;<br>Both</button>
        </div>
        <div class="row3">
          <button id="upload" title="Upload an image as the background">&#8593;</button>
          <button id="undo" title="Undo (Ctrl+Z)">&#8630;</button>
          <button id="redo" title="Redo (Ctrl+Y / Ctrl+Shift+Z)">&#8631;</button>
        </div>
        <div class="row3">
          <button id="fit" title="Fit to view">&#10530;</button>
          <button id="fliph" title="Flip the active layer horizontally">&#8596;</button>
          <button id="flipv" title="Flip the active layer vertically">&#8597;</button>
        </div>
        <div class="fld"><span>Zoom</span><span id="zmv">100</span>%</div>
      </div>
      <div class="sec">
        <div class="seclabel">Brush size</div>
        <div class="fld"><span>Size</span><span id="szv">14</span></div>
        <input type="range" id="size" min="1" max="200" value="14">
      </div>
      <div class="sec">
        <div class="seclabel">Mask</div>
        <div class="row3">
          <button id="auto" title="Auto-mask — also use the painted area as the mask">&#10024;</button>
          <button id="growmask" title="Grow the mask outward (dilate)">&#8853;</button>
          <button id="shrinkmask" title="Shrink the mask inward (erode)">&#8854;</button>
        </div>
        <div class="fld"><span>Auto-grow</span><span id="dilv">8</span></div>
        <input type="range" id="dil" min="0" max="60" value="8">
        <div class="fld"><span>Select tol.</span><span id="wtolv">40</span></div>
        <input type="range" id="wtol" min="0" max="120" value="40" title="Colour tolerance for the magic wand and the fill bucket (higher = matches more)">
        <div class="row3">
          <button id="invmask" title="Invert the mask">&#9680;</button>
          <button id="clrmask" title="Clear the mask">&#8856;</button>
          <button id="showmask" class="on" title="Show / hide the mask overlay">&#128065;</button>
        </div>
        <div class="row2">
          <button id="mcopy" title="Copy the masked region, flattened (Ctrl+C)">&#128203;</button>
          <button id="mpaste" title="Paste the clipboard as a new layer (Ctrl+V)">&#128229;</button>
        </div>
      </div>
      <div class="sec">
        <div class="seclabel">Tools</div>
        <div class="row3" id="tools">
          <button data-tool="brush" class="on" title="Brush (B)">&#128396;</button>
          <button data-tool="eraser" title="Eraser (E)">&#9003;</button>
          <button data-tool="restore" title="Restore brush — paint back erased parts of the active layer (R)">&#8634;</button>
          <button data-tool="fill" title="Fill bucket (G)">&#129529;</button>
          <button data-tool="eye" title="Eyedropper — pick a colour (I)">&#128167;</button>
          <button data-tool="wand" title="Magic wand — flood-select a colour into the mask (W)">&#129668;</button>
          <button data-tool="lasso" title="Lasso — freehand-select a region into the mask (L)">&#129701;</button>
          <button data-tool="clone" title="Clone stamp — Alt/Shift-click to set the source, then paint to copy (K)">&#9112;</button>
          <button data-tool="smudge" title="Smudge — drag to push / blend pixels (S)">&#8776;</button>
          <button data-tool="rect" title="Rectangle">&#9645;</button>
          <button data-tool="ellipse" title="Ellipse">&#11053;</button>
          <button data-tool="line" title="Line">&#9585;</button>
          <button data-tool="xform" title="Move / transform the active layer (drag = move)">&#10021;</button>
        </div>
      </div>
      <div class="sec">
        <div class="seclabel">Brush</div>
        <div class="palrow" id="pal"></div>
        <input type="color" id="pick" value="#e83e8c" title="Custom colour">
        <div class="fld" id="hardfld"><span>Hardness</span><span id="hdv">80</span></div>
        <input type="range" id="hard" min="0" max="100" value="80" title="Brush edge hardness (lower = softer)">
        <div class="fld"><span>Opacity</span><span id="opv">100</span></div>
        <input type="range" id="op" min="5" max="100" value="100">
      </div>
      <div class="sec">
        <div class="seclabel">Layer</div>
        <div class="row2">
          <button id="copylayer" title="Copy the masked region of the active layer">&#10697;</button>
          <button id="copyflat" title="Copy the masked region of the flattened image">&#128464;</button>
        </div>
        <div class="row2">
          <button id="paste" title="Paste the clipboard as a new layer">&#128229;</button>
          <button id="flatten" title="Merge the active layer down into the one below">&#10515;</button>
        </div>
        <button id="clrpaint" title="Clear the active draw layer">&#9114;</button>
        <div class="fld"><span>Scale</span><span id="scv">100</span>%</div>
        <input type="range" id="scale" min="10" max="300" value="100">
        <div class="fld"><span>Rotate</span><span id="rotv">0</span>&#176;</div>
        <input type="range" id="rot" min="-180" max="180" value="0">
        <div class="fld"><span>Layer opacity</span><span id="lopv">100</span>%</div>
        <input type="range" id="lopac" min="0" max="100" value="100" title="Opacity of the active layer">
      </div>
    </div>
    <button id="railexpand" title="Show the tool panel">&#171;</button>
  </div>
  <div id="layers" title="Layers — drag draw layers to reorder; click to make active"></div>
  <div id="ovbar">
    <div id="ovhead">
      <button id="ovtoggle">&#9656; Overlays</button>
      <button id="ovxform" class="ovbtn ovicon" title="Transform the active layer — move · scale · rotate">&#10021;</button>
      <button id="ovfliph" class="ovbtn ovicon" title="Flip the active layer horizontally">&#8596;</button>
      <button id="ovflipv" class="ovbtn ovicon" title="Flip the active layer vertically">&#8597;</button>
      <span id="ovhint">drag a thumbnail or an image file onto the canvas → new layer · double-click to preview</span>
    </div>
    <div id="ovstrip"></div>
  </div>
</div>
<div id="ovprev"><img id="ovprevimg" src=""></div>
<input type="file" id="fileinput" accept="image/*" style="display:none">
<script>
(function(){
var MODE="__MODE__";
var W=0,H=0,hasBg=false,baseImg=null;
var wrap=document.getElementById('wrap'),stage=document.getElementById('stage');
var bg=document.getElementById('bg'),maskv=document.getElementById('maskv'),
    disp=document.getElementById('disp'),cursor=document.getElementById('cursor');
var bgx=bg.getContext('2d'),mx=maskv.getContext('2d'),
    dx=disp.getContext('2d'),cur=cursor.getContext('2d');
var maskBuf=document.createElement('canvas'),mbx=maskBuf.getContext('2d');
var restoreBuf=document.createElement('canvas'),rbx=restoreBuf.getContext('2d');  // scratch for the restore brush
// stacking of draw layers between base (bottom) and mask (top)
var drawLayers=[],active=0,layerSeq=0;

var tool='brush',color='#e83e8c',secondaryColor=null,size=14,opacity=1.0,hardness=0.8;
var lassoPts=[],cloneSrc=null,cloneOff={dx:0,dy:0};
var mode='draw';  // 'mask' | 'draw' | 'drawmask'
var autoMask=false,dilate=8,showMask=true,wandTol=40;
var drawing=false,sx=0,sy=0,lastX=0,lastY=0;
var undoStack=[],redoStack=[];
var baseScale=1,viewScale=1;

function newLayer(name){
  var cv=document.createElement('canvas'); cv.width=W; cv.height=H;
  return {id:++layerSeq, name:name||('Layer '+(drawLayers.length+1)), cv:cv,
          ctx:cv.getContext('2d'), visible:true, opacity:1, src:null};
}
function activeLayer(){
  if(!drawLayers.length){ drawLayers.push(newLayer('Layer 1')); active=0; renderLayers(); }
  if(active<0||active>=drawLayers.length) active=drawLayers.length-1;
  return drawLayers[active];
}
// where a paint op writes, per mode: list of {ctx, isMask}
function paintTargets(){
  if(mode==='mask') return [{ctx:mbx,isMask:true}];
  if(mode==='drawmask') return [{ctx:activeLayer().ctx,isMask:false},{ctx:mbx,isMask:true}];
  return [{ctx:activeLayer().ctx,isMask:false}];
}

function setSize(w,h){
  W=w; H=h;
  [bg,maskv,disp,cursor,maskBuf,restoreBuf].forEach(function(c){c.width=w;c.height=h;});
  drawLayers.forEach(function(l){ l.cv.width=w; l.cv.height=h; });
  fitView();
}
function fitView(){ var availW=stage.parentNode.clientWidth-20;
  baseScale=Math.min(1,availW/W); viewScale=1; applyView(); }
function applyView(){ var s=baseScale*viewScale;
  stage.style.width=(W*s)+'px'; stage.style.height=(H*s)+'px';
  [bg,maskv,disp,cursor].forEach(function(c){c.style.width='100%';c.style.height='100%';});
  var z=document.getElementById('zmv'); if(z) z.textContent=Math.round(s*100); }

// undo across draw layers + mask. Each stack entry is a GROUP that restores
// atomically: a pixel group ({kind:'pixels', frames:[{ctx,tag,data}...]}) or a
// structural group ({kind:'struct', undo, redo}) for add/delete/reorder/paste.
function pushEntry(entry){ undoStack.push(entry);
  if(undoStack.length>30) undoStack.shift(); redoStack=[]; }
// snapshot one or more ctxs as a single atomic pixel group
function pushUndoGroup(ctxs){ var frames=[];
  ctxs.forEach(function(c){ try{ frames.push({ctx:c.ctx,tag:c.tag,data:c.ctx.getImageData(0,0,W,H)}); }catch(e){} });
  if(frames.length) pushEntry({kind:'pixels',frames:frames}); }
function pushUndo(ctx,tag){ pushUndoGroup([{ctx:ctx,tag:tag}]); }
// snapshot ALL paint targets together so Draw+Mask undo restores layer + mask atomically
function snapshot(){ pushUndoGroup(paintTargets().map(function(t){ return {ctx:t.ctx,tag:t.isMask?'mask':'draw'}; })); }
// record a structural change (undo()=revert, redo()=re-apply) on the active stack
function pushStruct(undo,redo){ pushEntry({kind:'struct',undo:undo,redo:redo}); }
function restore(stack,other){ if(!stack.length) return; var s=stack.pop();
  if(s.kind==='struct'){ var inv={kind:'struct',undo:s.redo,redo:s.undo};
    other.push(inv); s.undo(); compose(); renderLayers(); pushExport(); return; }
  var cur={kind:'pixels',frames:[]};
  s.frames.forEach(function(f){ try{ cur.frames.push({ctx:f.ctx,tag:f.tag,data:f.ctx.getImageData(0,0,W,H)}); }catch(e){}
    f.ctx.putImageData(f.data,0,0); });
  other.push(cur); compose(); }

function compose(){
  dx.clearRect(0,0,W,H);
  if(baseImg) dx.drawImage(baseImg,0,0,W,H);
  drawLayers.forEach(function(l){ if(l.visible){ dx.save(); dx.globalAlpha=(l.opacity==null?1:l.opacity); dx.drawImage(l.cv,0,0); dx.restore(); } });
  if(showMask){ dx.save(); dx.globalAlpha=0.5;
    var t=document.createElement('canvas'); t.width=W;t.height=H; var tc=t.getContext('2d');
    tc.drawImage(maskBuf,0,0); tc.globalCompositeOperation='source-in';
    tc.fillStyle='#ff3344'; tc.fillRect(0,0,W,H); dx.drawImage(t,0,0); dx.restore(); }
}

function pos(e){ var r=disp.getBoundingClientRect(),t=(e.touches&&e.touches.length)?e.touches[0]:((e.changedTouches&&e.changedTouches[0])||e);
  var x=(t.clientX-r.left)/r.width*W, y=(t.clientY-r.top)/r.height*H;
  // clamp inside the bitmap so getImageData/floodFill never index a row off the edge
  return {x:Math.max(0,Math.min(W-1,x)), y:Math.max(0,Math.min(H-1,y))}; }
// does this engine support 2D ctx.filter? (assigning an unsupported value is
// silently ignored rather than thrown, so feature-detect explicitly)
var _CTX_FILTER = (function(){ try{ var c=document.createElement('canvas').getContext('2d');
  c.filter='blur(1px)'; return c.filter==='blur(1px)'; }catch(e){ return false; } })();
// radial-gradient soft stamp fallback used when ctx.filter is unavailable
function softStamp(ctx,x,y,rgb){ var r=Math.max(1,size/2);
  var g=ctx.createRadialGradient(x,y,r*Math.max(0,hardness),x,y,r);
  g.addColorStop(0,'rgba('+rgb[0]+','+rgb[1]+','+rgb[2]+','+opacity+')');
  g.addColorStop(1,'rgba('+rgb[0]+','+rgb[1]+','+rgb[2]+',0)');
  ctx.fillStyle=g; ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.fill(); }
function strokeOn(ctx,isMask,x0,y0,x1,y1){
  ctx.save(); ctx.lineCap='round'; ctx.lineJoin='round'; ctx.lineWidth=size;
  if(tool==='eraser'){ ctx.globalCompositeOperation='destination-out'; ctx.strokeStyle='rgba(0,0,0,1)'; }
  else if(isMask){ ctx.strokeStyle='#ffffff'; }
  else { ctx.globalCompositeOperation='source-over'; ctx.globalAlpha=opacity; ctx.strokeStyle=color; }
  // Soft brush: blur the stroke edge when hardness < 100% (paint only, not mask/eraser).
  if(hardness<1 && tool!=='eraser' && !isMask){
    if(_CTX_FILTER){ try{ ctx.filter='blur('+(size*(1-hardness)/4).toFixed(1)+'px)'; }catch(e){} }
    else { // radial-gradient soft-stamp fallback along the segment
      var rgb=hex2rgb(color),dxs=x1-x0,dys=y1-y0,dist=Math.hypot(dxs,dys),steps=Math.max(1,Math.ceil(dist/(size/4)));
      for(var s=0;s<=steps;s++){ softStamp(ctx,x0+dxs*s/steps,y0+dys*s/steps,rgb); }
      ctx.restore(); return; } }
  ctx.beginPath(); ctx.moveTo(x0,y0); ctx.lineTo(x1,y1); ctx.stroke(); ctx.restore();
}
function stroke(x0,y0,x1,y1){ paintTargets().forEach(function(t){ strokeOn(t.ctx,t.isMask,x0,y0,x1,y1); }); }
// --- non-destructive erase/restore. Each layer keeps a pristine 'src' snapshot,
//     captured lazily just before the first erase stroke (and invalidated whenever
//     the layer gets new positive content — paint/transform/flip/flatten/clear — so
//     it re-bases). The eraser cuts holes in the layer; the restore brush paints the
//     matching src pixels back where you brush. ---
function snapSrc(l){ if(!l) return; if(!l.src) l.src=document.createElement('canvas');
  l.src.width=W; l.src.height=H; var c=l.src.getContext('2d'); c.clearRect(0,0,W,H); c.drawImage(l.cv,0,0); }
function ensureSrc(l){ if(l && !l.src) snapSrc(l); }
function restoreStamp(x0,y0,x1,y1){ var l=activeLayer(); if(!l||!l.src) return;
  rbx.clearRect(0,0,W,H); rbx.save(); rbx.lineCap='round'; rbx.lineJoin='round';
  rbx.lineWidth=size; rbx.strokeStyle='#fff'; rbx.beginPath(); rbx.moveTo(x0,y0); rbx.lineTo(x1,y1); rbx.stroke(); rbx.restore();
  rbx.globalCompositeOperation='source-in'; rbx.drawImage(l.src,0,0); rbx.globalCompositeOperation='source-over';
  l.ctx.drawImage(restoreBuf,0,0); }
function shapeOn(ctx,isMask,x0,y0,x1,y1,preview){
  ctx.save();
  if(isMask && !preview){ ctx.strokeStyle=ctx.fillStyle='#ffffff'; }
  else { ctx.globalAlpha=preview?0.7:opacity; ctx.strokeStyle=ctx.fillStyle=color; }
  ctx.lineWidth=size;
  if(tool==='line'){ ctx.beginPath(); ctx.moveTo(x0,y0); ctx.lineTo(x1,y1); ctx.stroke(); }
  else if(tool==='rect'){ ctx.fillRect(Math.min(x0,x1),Math.min(y0,y1),Math.abs(x1-x0),Math.abs(y1-y0)); }
  else if(tool==='ellipse'){ ctx.beginPath();
    ctx.ellipse((x0+x1)/2,(y0+y1)/2,Math.abs(x1-x0)/2,Math.abs(y1-y0)/2,0,0,Math.PI*2); ctx.fill(); }
  ctx.restore();
}
function commitShape(x0,y0,x1,y1){ paintTargets().forEach(function(t){ shapeOn(t.ctx,t.isMask,x0,y0,x1,y1,false); }); }
function floodFill(sxp,syp){
  paintTargets().forEach(function(t){
    var ctx=t.ctx, img=ctx.getImageData(0,0,W,H), d=img.data;
    var idx=(Math.floor(syp)*W+Math.floor(sxp))*4;
    var tr=d[idx],tg=d[idx+1],tb=d[idx+2],ta=d[idx+3];
    var fr,fg,fb,fa;
    if(t.isMask){ fr=fg=fb=255; fa=255; }
    else { var c=hex2rgb(color); fr=c[0];fg=c[1];fb=c[2];fa=Math.round(opacity*255); }
    if(tr===fr&&tg===fg&&tb===fb&&ta===fa) return;
    var stack=[idx],seen=new Uint8Array(W*H),tol=wandTol;
    function m(i){ return Math.abs(d[i]-tr)<=tol&&Math.abs(d[i+1]-tg)<=tol&&Math.abs(d[i+2]-tb)<=tol&&Math.abs(d[i+3]-ta)<=tol; }
    while(stack.length){ var i=stack.pop(),pi=i>>2; if(pi<0||pi>=W*H||seen[pi]||!m(i)) continue; seen[pi]=1;
      d[i]=fr;d[i+1]=fg;d[i+2]=fb;d[i+3]=fa; var x=pi%W;
      if(x>0)stack.push(i-4); if(x<W-1)stack.push(i+4); stack.push(i-W*4); stack.push(i+W*4); }
    ctx.putImageData(img,0,0);
  });
}
function hex2rgb(h){ h=h.replace('#',''); if(h.length===3)h=h[0]+h[0]+h[1]+h[1]+h[2]+h[2];
  return [parseInt(h.substr(0,2),16),parseInt(h.substr(2,2),16),parseInt(h.substr(4,2),16)]; }
function pickColor(xp,yp){ var t=document.createElement('canvas'); t.width=W;t.height=H; var tc=t.getContext('2d');
  if(baseImg) tc.drawImage(baseImg,0,0,W,H); drawLayers.forEach(function(l){ if(l.visible) tc.drawImage(l.cv,0,0); });
  var p=tc.getImageData(Math.floor(xp),Math.floor(yp),1,1).data;
  setColor('#'+[p[0],p[1],p[2]].map(function(v){return ('0'+v.toString(16)).slice(-2);}).join('')); }

// op: 'replace' (default — clear then set), 'add' (Shift), 'subtract' (Alt)
function magicWand(xp,yp,op){
  var t=document.createElement('canvas'); t.width=W;t.height=H; var tc=t.getContext('2d');
  if(baseImg) tc.drawImage(baseImg,0,0,W,H);
  drawLayers.forEach(function(l){ if(l.visible) tc.drawImage(l.cv,0,0); });
  var img=tc.getImageData(0,0,W,H),d=img.data;
  var sx0=Math.floor(xp),sy0=Math.floor(yp); var idx=(sy0*W+sx0)*4;
  var tr=d[idx],tg=d[idx+1],tb=d[idx+2];
  var sel=mbx.getImageData(0,0,W,H),sd=sel.data;
  if(op==='replace'){ for(var z=0;z<sd.length;z++) sd[z]=0; }   // start from an empty mask
  var on=(op!=='subtract');                                     // add/replace set; subtract clears
  var stack=[idx],seen=new Uint8Array(W*H),tol=wandTol;
  function m(i){ return Math.abs(d[i]-tr)<=tol&&Math.abs(d[i+1]-tg)<=tol&&Math.abs(d[i+2]-tb)<=tol; }
  while(stack.length){ var i=stack.pop(),pi=i>>2; if(pi<0||pi>=W*H||seen[pi]||!m(i)) continue; seen[pi]=1;
    if(on){ sd[i]=sd[i+1]=sd[i+2]=255; sd[i+3]=255; } else { sd[i]=sd[i+1]=sd[i+2]=0; sd[i+3]=0; }
    var x=pi%W;
    if(x>0)stack.push(i-4); if(x<W-1)stack.push(i+4); stack.push(i-W*4); stack.push(i+W*4); }
  mbx.putImageData(sel,0,0);
}
// -- lasso: freehand select a region into the mask --
function drawLasso(){ if(lassoPts.length<2) return; dx.save(); dx.strokeStyle='#e83e8c';
  dx.lineWidth=Math.max(1,W/stage.clientWidth*1.5); dx.setLineDash([6,4]);
  dx.beginPath(); dx.moveTo(lassoPts[0].x,lassoPts[0].y);
  for(var i=1;i<lassoPts.length;i++) dx.lineTo(lassoPts[i].x,lassoPts[i].y); dx.stroke(); dx.restore(); }
function fillLasso(){ if(lassoPts.length<3){ lassoPts=[]; return; }
  mbx.save(); mbx.fillStyle='#ffffff'; mbx.beginPath(); mbx.moveTo(lassoPts[0].x,lassoPts[0].y);
  for(var i=1;i<lassoPts.length;i++) mbx.lineTo(lassoPts[i].x,lassoPts[i].y);
  mbx.closePath(); mbx.fill(); mbx.restore(); lassoPts=[]; }
// -- clone stamp: copy from a sampled source offset onto the active layer --
function flatCanvas(){ var t=document.createElement('canvas'); t.width=W;t.height=H; var tc=t.getContext('2d');
  if(baseImg) tc.drawImage(baseImg,0,0,W,H); drawLayers.forEach(function(l){ if(l.visible) tc.drawImage(l.cv,0,0); }); return t; }
function cloneStamp(x,y){ if(!cloneSrc) return; var l=activeLayer(),r=size/2,src=flatCanvas();
  l.ctx.save(); l.ctx.beginPath(); l.ctx.arc(x,y,r,0,Math.PI*2); l.ctx.clip();
  l.ctx.drawImage(src, x+cloneOff.dx-r, y+cloneOff.dy-r, size, size, x-r, y-r, size, size); l.ctx.restore(); }
// -- smudge: drag a sampled patch along the stroke to push/blend pixels --
function smudgeStep(x0,y0,x1,y1){ var l=activeLayer(),r=size/2;
  // sample from the flattened image so smudge works over background-only content,
  // then paint the dragged patch onto the active layer
  var src=flatCanvas();
  var patch=document.createElement('canvas'); patch.width=size||1; patch.height=size||1;
  patch.getContext('2d').drawImage(src, x0-r,y0-r,size,size, 0,0,size,size);
  l.ctx.save(); l.ctx.globalAlpha=0.55; l.ctx.beginPath(); l.ctx.arc(x1,y1,r,0,Math.PI*2); l.ctx.clip();
  l.ctx.drawImage(patch, x1-r, y1-r, size, size); l.ctx.restore(); }
// -- mask morphology: grow (dilate) / shrink (erode) --
function dilateCtx(ctx,px){ if(px<=0) return; var tmp=document.createElement('canvas'); tmp.width=W;tmp.height=H;
  tmp.getContext('2d').drawImage(ctx.canvas,0,0);
  for(var a=0;a<16;a++){ var ang=a/16*Math.PI*2; ctx.drawImage(tmp,Math.cos(ang)*px,Math.sin(ang)*px); } }
// grow/shrink use the Auto-grow slider magnitude (min 1px so the button always acts)
function maskMorphPx(){ return Math.max(1,dilate); }
function growMask(){ pushUndo(mbx,'mask'); dilateCtx(mbx,maskMorphPx()); compose(); pushExport(); }
function shrinkMask(){ pushUndo(mbx,'mask');
  var t=document.createElement('canvas'); t.width=W;t.height=H; var tc=t.getContext('2d');
  tc.fillStyle='#fff'; tc.fillRect(0,0,W,H); tc.globalCompositeOperation='destination-out'; tc.drawImage(maskBuf,0,0);
  dilateCtx(tc,maskMorphPx());  // grow the inverse = erode the mask
  mbx.save(); mbx.globalCompositeOperation='source-over'; mbx.clearRect(0,0,W,H);
  mbx.fillStyle='#fff'; mbx.fillRect(0,0,W,H); mbx.globalCompositeOperation='destination-out';
  mbx.drawImage(t,0,0); mbx.restore(); compose(); pushExport(); }
// -- flip the active layer --
function flipLayer(horiz){ if(!hasBg) return; var l=activeLayer(); pushUndo(l.ctx,'draw'); l.src=null;
  if(tSession===l.id){ tSession=-1; tSnap=null; tBox=null; }   // flipped pixels supersede any pending transform snapshot
  var tmp=document.createElement('canvas'); tmp.width=W;tmp.height=H; tmp.getContext('2d').drawImage(l.cv,0,0);
  l.ctx.save(); l.ctx.clearRect(0,0,W,H); l.ctx.translate(horiz?W:0,horiz?0:H);
  l.ctx.scale(horiz?-1:1,horiz?1:-1); l.ctx.drawImage(tmp,0,0); l.ctx.restore(); compose(); pushExport(); }
// -- clipboard + transform --
var clipboard=null,tSnap=null,tSession=-1,tDx=0,tDy=0,tScale=1,tRot=0;
var tBox=null,txMode='move',txStart=null;   // on-canvas transform handles state
function maskBBox(){ var d=mbx.getImageData(0,0,W,H).data,minx=W,miny=H,maxx=-1,maxy=-1;
  for(var y=0;y<H;y++)for(var x=0;x<W;x++){ if(d[(y*W+x)*4]>127){ if(x<minx)minx=x;if(x>maxx)maxx=x;if(y<miny)miny=y;if(y>maxy)maxy=y; } }
  return maxx<0?null:{x:minx,y:miny,w:maxx-minx+1,h:maxy-miny+1}; }
function copyRegion(fromFlat){ var bb=maskBBox(); if(!bb) return false;
  var src=document.createElement('canvas'); src.width=W;src.height=H; var sc=src.getContext('2d');
  if(fromFlat){ if(baseImg) sc.drawImage(baseImg,0,0,W,H); drawLayers.forEach(function(l){if(l.visible)sc.drawImage(l.cv,0,0);}); }
  else sc.drawImage(activeLayer().cv,0,0);
  sc.globalCompositeOperation='destination-in'; sc.drawImage(maskBuf,0,0);  // keep only masked
  var clip=document.createElement('canvas'); clip.width=bb.w;clip.height=bb.h;
  clip.getContext('2d').drawImage(src,bb.x,bb.y,bb.w,bb.h,0,0,bb.w,bb.h);
  clipboard={cv:clip,x:bb.x,y:bb.y}; return true; }
// insert a layer at idx and record an atomic structural undo (remove)/redo (reinsert)
function addLayerStruct(l,idx){ if(idx==null) idx=drawLayers.length;
  drawLayers.splice(idx,0,l); active=idx;
  pushStruct(function(){ var i=drawLayers.indexOf(l); if(i>=0) drawLayers.splice(i,1);
      if(active>=drawLayers.length) active=drawLayers.length-1; if(tSession===l.id) tSession=-1; },
    function(){ drawLayers.splice(idx,0,l); active=idx; });
  renderLayers(); }
function pasteClip(){ if(!clipboard) return; var l=newLayer('Pasted');
  l.ctx.drawImage(clipboard.cv, clipboard.x||0, clipboard.y||0);
  addLayerStruct(l, drawLayers.length);
  selTool('xform'); ensureXform(); compose(); pushExport(); }
function beginTransform(){ var l=activeLayer(); pushUndo(l.ctx,'draw');
  tSnap=document.createElement('canvas'); tSnap.width=W;tSnap.height=H; tSnap.getContext('2d').drawImage(l.cv,0,0);
  tBox=contentBBox(tSnap); l.src=null; tSession=l.id; tDx=0;tDy=0;tScale=1;tRot=0;
  var s=document.getElementById('scale'),r=document.getElementById('rot');
  if(s){s.value=100;document.getElementById('scv').textContent=100;}
  if(r){r.value=0;document.getElementById('rotv').textContent=0;} }
function ensureXform(){ if(!tSnap||activeLayer().id!==tSession) beginTransform(); drawHandles(); }
function applyTransform(){ if(!tSnap) return; var l=drawLayers.filter(function(x){return x.id===tSession;})[0]; if(!l) return;
  l.ctx.clearRect(0,0,W,H); l.ctx.save(); l.ctx.translate(W/2+tDx,H/2+tDy);
  l.ctx.rotate(tRot*Math.PI/180); l.ctx.scale(tScale,tScale); l.ctx.drawImage(tSnap,-W/2,-H/2); l.ctx.restore(); compose(); drawHandles(); }
// --- on-canvas transform handles (drawn on the cursor overlay; drive the SAME
//     tScale/tRot/tDx/tDy the sliders use, so sliders + drag-move stay a fallback) ---
function hdim(px){ var ds=(baseScale*viewScale)||1; return px/ds; }   // screen px -> canvas units
function contentBBox(snap){ try{
    var d=snap.getContext('2d').getImageData(0,0,W,H).data, minx=W,miny=H,maxx=-1,maxy=-1;
    for(var y=0;y<H;y++){ var r=y*W; for(var x=0;x<W;x++){ if(d[(r+x)*4+3]>16){
      if(x<minx)minx=x; if(x>maxx)maxx=x; if(y<miny)miny=y; if(y>maxy)maxy=y; } } }
    if(maxx<0) return {x:0,y:0,w:W,h:H};
    return {x:minx,y:miny,w:maxx-minx+1,h:maxy-miny+1};
  }catch(e){ return {x:0,y:0,w:W,h:H}; } }
function txPoint(lx,ly){ var rad=tRot*Math.PI/180,cs=Math.cos(rad),sn=Math.sin(rad);
  var ox=(lx-W/2)*tScale, oy=(ly-H/2)*tScale;
  return { x:(W/2+tDx)+(ox*cs-oy*sn), y:(H/2+tDy)+(ox*sn+oy*cs) }; }
function handlePts(){ if(!tBox) return null;
  var x0=tBox.x,y0=tBox.y,x1=tBox.x+tBox.w,y1=tBox.y+tBox.h, mx=(x0+x1)/2,my=(y0+y1)/2;
  var corners=[txPoint(x0,y0),txPoint(x1,y0),txPoint(x1,y1),txPoint(x0,y1)];
  var sides=[txPoint(mx,y0),txPoint(x1,my),txPoint(mx,y1),txPoint(x0,my)];
  var top=txPoint(mx,y0), ctr=txPoint(mx,my);
  var vx=top.x-ctr.x,vy=top.y-ctr.y,vl=Math.hypot(vx,vy)||1, off=hdim(26);
  return {corners:corners, scale:corners.concat(sides), top:top,
          pin:{x:top.x+vx/vl*off, y:top.y+vy/vl*off}}; }
function drawHandles(){ if(!cur) return; cur.clearRect(0,0,W,H);
  if(tool!=='xform'||!tSnap||!hasBg) return; var h=handlePts(); if(!h) return;
  var s=hdim(5), lw=hdim(1.5), knob=hdim(6.5);
  cur.save(); cur.lineWidth=lw; cur.strokeStyle='rgba(232,62,140,.95)';
  cur.beginPath(); cur.moveTo(h.corners[0].x,h.corners[0].y);
  for(var i=1;i<4;i++) cur.lineTo(h.corners[i].x,h.corners[i].y); cur.closePath(); cur.stroke();
  cur.beginPath(); cur.moveTo(h.top.x,h.top.y); cur.lineTo(h.pin.x,h.pin.y); cur.stroke();
  cur.fillStyle='#e83e8c'; cur.beginPath(); cur.arc(h.pin.x,h.pin.y,knob,0,Math.PI*2); cur.fill();
  cur.fillStyle='#fff';
  h.scale.forEach(function(p){ cur.beginPath(); cur.rect(p.x-s,p.y-s,s*2,s*2); cur.fill(); cur.stroke(); });
  cur.restore(); }
function hitHandle(p){ if(!tBox) return null; var h=handlePts(); if(!h) return null; var r=hdim(11);
  if(Math.hypot(p.x-h.pin.x,p.y-h.pin.y)<r) return 'rotate';
  for(var i=0;i<h.scale.length;i++){ var q=h.scale[i];
    if(Math.abs(p.x-q.x)<r && Math.abs(p.y-q.y)<r) return 'scale'; }
  return null; }
function startXform(p){ var cx=W/2+tDx, cy=H/2+tDy; txMode=hitHandle(p)||'move';
  txStart={scale:tScale, rot:tRot, cx:cx, cy:cy,
           dist:(Math.hypot(p.x-cx,p.y-cy)||1), ang:Math.atan2(p.y-cy,p.x-cx)};
  drawHandles(); }
function moveXform(p){ if(!txStart) return;
  if(txMode==='scale'){ var d=Math.hypot(p.x-txStart.cx,p.y-txStart.cy);
    tScale=Math.max(0.1,Math.min(3, txStart.scale*(d/txStart.dist)));
    var ss=document.getElementById('scale'); if(ss){ ss.value=Math.round(tScale*100);
      document.getElementById('scv').textContent=ss.value; } }
  else if(txMode==='rotate'){ var a=Math.atan2(p.y-txStart.cy,p.x-txStart.cx);
    var deg=txStart.rot+(a-txStart.ang)*180/Math.PI; while(deg>180)deg-=360; while(deg<-180)deg+=360; tRot=deg;
    var rs=document.getElementById('rot'); if(rs){ rs.value=Math.round(tRot);
      document.getElementById('rotv').textContent=rs.value; } }
  else { tDx+=p.x-lastX; tDy+=p.y-lastY; lastX=p.x; lastY=p.y; }
  applyTransform(); }
function flattenDown(){ if(active<=0) return; var top=drawLayers[active],below=drawLayers[active-1];
  pushUndo(below.ctx,'draw'); below.ctx.drawImage(top.cv,0,0); below.src=null; drawLayers.splice(active,1); active=active-1;
  tSession=-1; tSnap=null; tBox=null; renderLayers(); compose(); pushExport(); }

function down(e){ if(!hasBg) return; if(e.button!==undefined && e.button!==0) return; e.preventDefault(); var p=pos(e); sx=lastX=p.x; sy=lastY=p.y;
  // erase/restore source bookkeeping (only touch the active layer for tools that need it)
  if(tool==='eraser'&&mode!=='mask') ensureSrc(activeLayer());   // snapshot so the restore brush can bring erased pixels back
  else if(mode!=='mask'&&(tool==='brush'||tool==='fill'||tool==='clone'||tool==='smudge'||tool==='rect'||tool==='ellipse'||tool==='line')) activeLayer().src=null;  // new positive content on the layer → re-base
  // A destructive edit to the active layer invalidates any pending transform
  // snapshot of it, so the next transform re-captures the edited pixels instead of
  // overwriting the layer with the stale snapshot (which would undo the erase/paint
  // done since the transform session began).
  if(mode!=='mask' && tool!=='xform' && tool!=='eye' && tool!=='wand' && tool!=='lasso'){
    var _al=activeLayer(); if(tSession===_al.id){ tSession=-1; tSnap=null; tBox=null; } }
  if(tool==='eye'){ pickColor(p.x,p.y); return; }
  if(tool==='wand'){ pushUndo(mbx,'mask');
    var wop=e.shiftKey?'add':(e.altKey?'subtract':'replace'); magicWand(p.x,p.y,wop); compose(); pushExport(); return; }
  if(tool==='xform'){ ensureXform(); drawing=true; startXform(p); return; }
  if(tool==='lasso'){ pushUndo(mbx,'mask'); lassoPts=[{x:p.x,y:p.y}]; drawing=true; return; }
  if(tool==='clone'){ if(e.altKey||e.shiftKey){ cloneSrc={x:p.x,y:p.y}; drawCursor(p.x,p.y); return; }
    if(!cloneSrc) return; cloneOff={dx:cloneSrc.x-p.x,dy:cloneSrc.y-p.y};
    snapshot(); drawing=true; cloneStamp(p.x,p.y); compose(); return; }
  if(tool==='smudge'){ snapshot(); drawing=true; return; }
  if(tool==='restore'){ var rl=activeLayer(); if(rl&&rl.src){ pushUndo(rl.ctx,'draw'); drawing=true; restoreStamp(p.x,p.y,p.x,p.y); compose(); } return; }  // nothing erased → nothing to restore (no phantom undo)
  snapshot(); drawing=true;
  if(tool==='fill'){ floodFill(p.x,p.y); drawing=false; compose(); pushExport(); return; }
  if(tool==='brush'||tool==='eraser'){ stroke(p.x,p.y,p.x,p.y); compose(); } }
function move(e){ if(!drawing) return; e.preventDefault(); var p=pos(e);
  if(tool==='xform'){ moveXform(p); }
  else if(tool==='lasso'){ lassoPts.push({x:p.x,y:p.y}); compose(); drawLasso(); }
  else if(tool==='clone'){ cloneStamp(p.x,p.y); lastX=p.x;lastY=p.y; compose(); }
  else if(tool==='smudge'){ smudgeStep(lastX,lastY,p.x,p.y); lastX=p.x;lastY=p.y; compose(); }
  else if(tool==='brush'||tool==='eraser'){ stroke(lastX,lastY,p.x,p.y); lastX=p.x;lastY=p.y; compose(); }
  else if(tool==='restore'){ restoreStamp(lastX,lastY,p.x,p.y); lastX=p.x;lastY=p.y; compose(); }
  else if(tool==='rect'||tool==='ellipse'||tool==='line'){ compose(); shapeOn(dx,false,sx,sy,p.x,p.y,true); } }
function up(e){ if(!drawing) return; drawing=false; var p=pos(e);
  if(tool==='lasso'){ fillLasso(); compose(); }
  else if(tool==='rect'||tool==='ellipse'||tool==='line'){ commitShape(sx,sy,p.x,p.y); compose(); }
  // flush the composite/mask synchronously so a Generate click right after a
  // stroke never reads the previous (debounced) export
  if(exportTimer){ clearTimeout(exportTimer); exportTimer=null; } exportNow(); }
disp.addEventListener('mousedown',down); window.addEventListener('mousemove',move);
window.addEventListener('mouseup',up);
disp.addEventListener('touchstart',down,{passive:false});
disp.addEventListener('touchmove',move,{passive:false}); window.addEventListener('touchend',up);
function drawCursor(x,y){ if(tool==='xform'){ if(!drawing) drawHandles(); return; } cur.clearRect(0,0,W,H);
  if(!hasBg||(tool!=='brush'&&tool!=='eraser'&&tool!=='clone'&&tool!=='smudge'&&tool!=='restore')) return;
  cur.save(); cur.lineWidth=Math.max(1,W/stage.clientWidth);
  cur.strokeStyle='rgba(0,0,0,.6)'; cur.beginPath(); cur.arc(x,y,size/2+1,0,Math.PI*2); cur.stroke();
  cur.strokeStyle='rgba(255,255,255,.9)'; cur.beginPath(); cur.arc(x,y,size/2,0,Math.PI*2); cur.stroke(); cur.restore(); }
disp.addEventListener('mousemove',function(e){ var p=pos(e); drawCursor(p.x,p.y); });
disp.addEventListener('mouseleave',function(){ cur.clearRect(0,0,W,H); });
wrap.addEventListener('wheel',function(e){ if(!hasBg) return; e.preventDefault();
  viewScale=Math.max(0.2,Math.min(8,viewScale*(e.deltaY<0?1.1:0.9))); applyView(); },{passive:false});
var panning=false,pSx,pSy,pL,pT;
wrap.addEventListener('mousedown',function(e){ if(e.button===1){ e.preventDefault();
  panning=true; pSx=e.clientX;pSy=e.clientY;pL=wrap.scrollLeft;pT=wrap.scrollTop; } });
window.addEventListener('mousemove',function(e){ if(panning){ wrap.scrollLeft=pL-(e.clientX-pSx); wrap.scrollTop=pT-(e.clientY-pSy); } });
window.addEventListener('mouseup',function(){ panning=false; });

// -- toolbar wiring --
function selTool(t){ tool=t; document.querySelectorAll('#tools button').forEach(function(b){ b.classList.toggle('on',b.dataset.tool===t); });
  if(t!=='xform' && cur) cur.clearRect(0,0,W,H); }
document.querySelectorAll('#tools button').forEach(function(b){ b.addEventListener('click',function(){ selTool(b.dataset.tool); if(b.dataset.tool==='xform') ensureXform(); }); });
document.getElementById('copylayer').addEventListener('click',function(){ copyRegion(false); });
document.getElementById('copyflat').addEventListener('click',function(){ copyRegion(true); });
document.getElementById('mcopy').addEventListener('click',function(){ copyRegion(true); });
document.getElementById('mpaste').addEventListener('click',pasteClip);
document.getElementById('paste').addEventListener('click',pasteClip);
document.getElementById('flatten').addEventListener('click',flattenDown);
document.getElementById('scale').addEventListener('input',function(e){ ensureXform(); tScale=(+e.target.value)/100; document.getElementById('scv').textContent=e.target.value; applyTransform(); pushExport(); });
document.getElementById('rot').addEventListener('input',function(e){ ensureXform(); tRot=+e.target.value; document.getElementById('rotv').textContent=e.target.value; applyTransform(); pushExport(); });
document.getElementById('lopac').addEventListener('input',function(e){ var l=activeLayer(); l.opacity=(+e.target.value)/100; document.getElementById('lopv').textContent=e.target.value; compose(); pushExport(); });
function syncLayerOpacity(){ var l=drawLayers[active]; if(!l) return; var v=Math.round((l.opacity==null?1:l.opacity)*100); var s=document.getElementById('lopac'); if(s){ s.value=v; document.getElementById('lopv').textContent=v; } }
// Hardness has no effect on the mask (soft edges are deliberately excluded for
// mask/eraser), so gray it out + annotate when painting the mask only.
function updateHardnessUI(){ var h=document.getElementById('hard'),f=document.getElementById('hardfld');
  var off=(mode==='mask'); if(h){ h.disabled=off; h.style.opacity=off?0.4:1; }
  if(f){ f.style.opacity=off?0.5:1; f.title=off?'No effect in Mask mode (mask edges are always hard)':''; } }
function setMode(m){ mode=m; document.querySelectorAll('#modeseg button').forEach(function(b){ b.classList.toggle('on',b.dataset.mode===m); });
  if(m==='mask' && (tool==='xform'||tool==='restore')) selTool('brush');   // can't transform/restore the mask — leave those layer-only tools
  updateHardnessUI(); renderLayers(); }
document.querySelectorAll('#modeseg button').forEach(function(b){ b.addEventListener('click',function(){ setMode(b.dataset.mode); }); });
function markSwatch(){ document.querySelectorAll('#pal .sw').forEach(function(s){ s.classList.toggle('on',s.dataset.c.toLowerCase()===color.toLowerCase()); }); }
// set the active colour, remembering the previous one as secondary (for the 'x' swap)
function setColor(c){ if(c && color && c.toLowerCase()!==color.toLowerCase()) secondaryColor=color;
  color=c; var pk=document.getElementById('pick'); if(pk) pk.value=c; markSwatch(); }
var pal=document.getElementById('pal');
__PALETTE__.forEach(function(c){ var s=document.createElement('div'); s.className='sw'; s.dataset.c=c; s.style.background=c;
  s.addEventListener('click',function(){ setColor(c); }); pal.appendChild(s); });
document.getElementById('pick').addEventListener('input',function(e){ setColor(e.target.value); });
document.getElementById('size').addEventListener('input',function(e){ size=+e.target.value; document.getElementById('szv').textContent=size; });
document.getElementById('op').addEventListener('input',function(e){ opacity=(+e.target.value)/100; document.getElementById('opv').textContent=e.target.value; });
document.getElementById('hard').addEventListener('input',function(e){ hardness=(+e.target.value)/100; document.getElementById('hdv').textContent=e.target.value; });
document.getElementById('fliph').addEventListener('click',function(){ flipLayer(true); });
document.getElementById('flipv').addEventListener('click',function(){ flipLayer(false); });
// overlay-section duplicates (under the canvas, by the layers/overlays strip)
document.getElementById('ovfliph').addEventListener('click',function(){ flipLayer(true); });
document.getElementById('ovflipv').addEventListener('click',function(){ flipLayer(false); });
// collapse / restore the tool rail (rail is on the right; >> hides it, << brings it back)
document.getElementById('railcollapse').addEventListener('click',function(){
  document.getElementById('main').classList.add('railcollapsed');
  window.dispatchEvent(new Event('resize')); });
document.getElementById('railexpand').addEventListener('click',function(){
  document.getElementById('main').classList.remove('railcollapsed');
  window.dispatchEvent(new Event('resize')); });
document.getElementById('ovxform').addEventListener('click',function(){ if(!hasBg) return; selTool('xform'); ensureXform(); });
document.getElementById('growmask').addEventListener('click',function(){ if(hasBg) growMask(); });
document.getElementById('shrinkmask').addEventListener('click',function(){ if(hasBg) shrinkMask(); });
document.getElementById('auto').addEventListener('click',function(){ autoMask=!autoMask; this.classList.toggle('on',autoMask); pushExport(); });
document.getElementById('dil').addEventListener('input',function(e){ dilate=+e.target.value; document.getElementById('dilv').textContent=dilate; pushExport(); });
document.getElementById('wtol').addEventListener('input',function(e){ wandTol=+e.target.value; document.getElementById('wtolv').textContent=wandTol; });
document.getElementById('showmask').addEventListener('click',function(){ showMask=!showMask; this.classList.toggle('on',showMask); compose(); });
document.getElementById('undo').addEventListener('click',function(){ restore(undoStack,redoStack); pushExport(); });
document.getElementById('redo').addEventListener('click',function(){ restore(redoStack,undoStack); pushExport(); });
document.getElementById('clrpaint').addEventListener('click',function(){ var l=activeLayer(); pushUndo(l.ctx,'draw'); l.ctx.clearRect(0,0,W,H); l.src=null; if(tSession===l.id){ tSession=-1; tSnap=null; tBox=null; } compose(); pushExport(); });
document.getElementById('clrmask').addEventListener('click',function(){ pushUndo(mbx,'mask'); mbx.clearRect(0,0,W,H); compose(); pushExport(); });
document.getElementById('invmask').addEventListener('click',function(){ if(!hasBg) return; pushUndo(mbx,'mask');
  var t=document.createElement('canvas'); t.width=W;t.height=H; var tc=t.getContext('2d');
  tc.fillStyle='#fff'; tc.fillRect(0,0,W,H); tc.globalCompositeOperation='destination-out'; tc.drawImage(maskBuf,0,0);
  mbx.clearRect(0,0,W,H); mbx.drawImage(t,0,0); compose(); pushExport(); });
document.getElementById('fit').addEventListener('click',function(){ if(hasBg) fitView(); });
var fileinput=document.getElementById('fileinput');
document.getElementById('upload').addEventListener('click',function(){ fileinput.click(); });
fileinput.addEventListener('change',function(e){ var f=e.target.files[0]; if(!f) return;
  var rd=new FileReader(); rd.onload=function(){ setBg(rd.result); }; rd.readAsDataURL(f); e.target.value=''; });
// nudge the brush size by a delta, clamped to the slider's min/max, keeping the
// 'size' var, the slider position and the readout in sync.
function bumpSize(delta){ var s=document.getElementById('size'); if(!s) return;
  var mn=+s.min||1, mx=+s.max||200, v=Math.max(mn,Math.min(mx,size+delta));
  size=v; s.value=v; document.getElementById('szv').textContent=v; }
window.addEventListener('keydown',function(e){
  if(e.ctrlKey&&e.key==='z'){restore(undoStack,redoStack);pushExport();return;}
  if(e.ctrlKey&&(e.key==='y'||(e.shiftKey&&e.key==='Z'))){restore(redoStack,undoStack);pushExport();return;}
  if(e.ctrlKey&&(e.key==='c'||e.key==='C')){ e.preventDefault(); copyRegion(true); return; }
  if(e.ctrlKey&&(e.key==='v'||e.key==='V')){ e.preventDefault(); pasteClip(); return; }
  if(e.key==='['){ bumpSize(-(size>20?5:1)); return; }
  if(e.key===']'){ bumpSize(size>=20?5:1); return; }
  if(e.key==='x'||e.key==='X'){ if(secondaryColor){ var prev=color; setColor(secondaryColor); secondaryColor=prev; } return; }
  var k={b:'brush',e:'eraser',r:'restore',g:'fill',i:'eye',w:'wand',l:'lasso',k:'clone',s:'smudge'}[e.key]; if(k) selTool(k); });

// -- layers manager (horizontal): mask (top) · draw layers · background (bottom) --
var lyrEl=document.getElementById('layers');
function chip(cls,label,opts){ opts=opts||{};
  var d=document.createElement('div'); d.className='lyr '+cls;
  var eye=document.createElement('span'); eye.className='eye'+(opts.visible===false?' off':''); eye.textContent='👁';
  if(opts.onEye){ eye.addEventListener('click',function(ev){ev.stopPropagation();opts.onEye();}); }
  d.appendChild(eye);
  var nm=document.createElement('span'); nm.textContent=label; d.appendChild(nm);
  if(opts.onDel){ var x=document.createElement('span'); x.className='del'; x.textContent='✕';
    x.addEventListener('click',function(ev){ev.stopPropagation();opts.onDel();}); d.appendChild(x); }
  if(opts.onClick) d.addEventListener('click',opts.onClick);
  if(opts.draggable){ d.draggable=true; d.dataset.idx=opts.idx;
    d.addEventListener('dragstart',function(ev){ ev.dataTransfer.setData('text/plain',opts.idx); });
    d.addEventListener('dragover',function(ev){ ev.preventDefault(); });
    d.addEventListener('drop',function(ev){ ev.preventDefault();
      var from=+ev.dataTransfer.getData('text/plain'), to=opts.idx;
      if(from===to) return; var prevA=active; var m=drawLayers.splice(from,1)[0]; drawLayers.splice(to,0,m);
      active=to;
      pushStruct(function(){ var i=drawLayers.indexOf(m); if(i>=0) drawLayers.splice(i,1); drawLayers.splice(from,0,m); active=prevA; },
        function(){ var i=drawLayers.indexOf(m); if(i>=0) drawLayers.splice(i,1); drawLayers.splice(to,0,m); active=to; });
      renderLayers(); compose(); pushExport(); }); }
  return d;
}
function renderLayers(){ lyrEl.innerHTML='';
  lyrEl.appendChild(chip('mask'+(mode==='mask'?' active':''),'Mask',{visible:showMask,
    onClick:function(){ setMode('mask'); },
    onEye:function(){ showMask=!showMask; document.getElementById('showmask').classList.toggle('on',showMask); compose(); renderLayers(); }}));
  // draw layers top→bottom in the panel = reverse of stacking
  for(var i=drawLayers.length-1;i>=0;i--){ (function(i){ var l=drawLayers[i];
    lyrEl.appendChild(chip('draw'+(i===active?' active':''), l.name, {visible:l.visible, idx:i, draggable:true,
      onClick:function(){ active=i; syncLayerOpacity(); if(mode==='mask'){ setMode('draw'); } else { renderLayers(); } },
      onEye:function(){ l.visible=!l.visible; compose(); renderLayers(); pushExport(); },
      onDel:function(){ var delIdx=i, delLayer=l, prevA=active;
        drawLayers.splice(delIdx,1); if(active>=drawLayers.length) active=drawLayers.length-1;
        if(tSession===delLayer.id) tSession=-1;
        // structural undo: reinsert the layer object (pixels preserved); redo deletes again
        pushStruct(function(){ drawLayers.splice(delIdx,0,delLayer); active=prevA; },
          function(){ var j=drawLayers.indexOf(delLayer); if(j>=0) drawLayers.splice(j,1); if(active>=drawLayers.length) active=drawLayers.length-1; if(tSession===delLayer.id) tSession=-1; });
        compose(); renderLayers(); pushExport(); }})); })(i); }
  lyrEl.appendChild(chip('base','Background',{visible:true}));
  var add=document.createElement('div'); add.className='addlyr'; add.textContent='+ Layer';
  add.addEventListener('click',function(){ addLayerStruct(newLayer(), drawLayers.length); }); lyrEl.appendChild(add);
}

// -- export (JS -> Python hidden Textboxes) --
function buildMask(){ var t=document.createElement('canvas'); t.width=W;t.height=H; var tc=t.getContext('2d');
  tc.fillStyle='#000'; tc.fillRect(0,0,W,H); tc.drawImage(maskBuf,0,0);
  if(autoMask){ var pa=document.createElement('canvas'); pa.width=W;pa.height=H; var pc=pa.getContext('2d');
    drawLayers.forEach(function(l){ if(l.visible) pc.drawImage(l.cv,0,0); });
    pc.globalCompositeOperation='source-in'; pc.fillStyle='#fff'; pc.fillRect(0,0,W,H);
    var r=Math.max(0,dilate); for(var a=0;a<16;a++){ var ang=a/16*Math.PI*2; tc.drawImage(pa,Math.cos(ang)*r,Math.sin(ang)*r); }
    tc.drawImage(pa,0,0); }
  var im=tc.getImageData(0,0,W,H),d=im.data;
  for(var i=0;i<d.length;i+=4){ var on=d[i]>127||d[i+1]>127||d[i+2]>127; d[i]=d[i+1]=d[i+2]=on?255:0; d[i+3]=255; }
  tc.putImageData(im,0,0); return t.toDataURL('image/png'); }
function buildComposite(){ var t=document.createElement('canvas'); t.width=W;t.height=H; var tc=t.getContext('2d');
  if(baseImg) tc.drawImage(baseImg,0,0,W,H);
  drawLayers.forEach(function(l){ if(l.visible){ tc.save(); tc.globalAlpha=(l.opacity==null?1:l.opacity); tc.drawImage(l.cv,0,0); tc.restore(); } });
  return t.toDataURL('image/png'); }
function setHidden(id,val){ try{ var pd=parent.document;
  var e=pd.querySelector('#'+id+' textarea')||pd.querySelector('#'+id+' input'); if(!e) return;
  var proto=(e.tagName==='TEXTAREA')?parent.HTMLTextAreaElement.prototype:parent.HTMLInputElement.prototype;
  Object.getOwnPropertyDescriptor(proto,'value').set.call(e,val);
  e.dispatchEvent(new Event('input',{bubbles:true})); e.dispatchEvent(new Event('change',{bubbles:true})); }catch(err){} }
function exportNow(){ if(!hasBg) return;
  setHidden('imagesuite-'+MODE+'-composite', buildComposite());
  setHidden('imagesuite-'+MODE+'-mask', buildMask()); }
var exportTimer=null;
function pushExport(){ if(!hasBg) return; clearTimeout(exportTimer); exportTimer=setTimeout(exportNow,120); }
try{ parent.window['__is_'+MODE+'_exportnow']=exportNow; }catch(e){}

function setBg(dataUrl){ var im=new Image(); im.onload=function(){ baseImg=im;
  drawLayers=[]; layerSeq=0; setSize(im.naturalWidth,im.naturalHeight);
  drawLayers=[newLayer('Layer 1')]; active=0; mbx.clearRect(0,0,W,H);
  undoStack=[]; redoStack=[]; hasBg=true; document.getElementById('empty').style.display='none';
  // clear state tied to the previous image so it can't bleed into the new one
  cloneSrc=null; clipboard=null; lassoPts=[]; tSnap=null; tBox=null; tSession=-1; tDx=0;tDy=0;tScale=1;tRot=0;
  var sc=document.getElementById('scale'),rt=document.getElementById('rot');
  if(sc){ sc.value=100; document.getElementById('scv').textContent=100; }
  if(rt){ rt.value=0; document.getElementById('rotv').textContent=0; }
  var lo=document.getElementById('lopac'); if(lo){ lo.value=100; document.getElementById('lopv').textContent=100; }
  renderLayers(); compose(); pushExport(); }; im.src=dataUrl; }
try{ parent.window['__is_'+MODE+'_setbg']=setBg; }catch(e){}

// -- Magic select subject (Python -> JS): load an alpha/grayscale mask PNG
//    (white = subject) into the mask layer. Mirrors setBg but writes mbx only,
//    leaving the background + draw layers untouched. Thresholded so the mask is
//    a clean white-on-black selection the inpaint/copy tools can act on. --
function setMask(dataUrl){ if(!hasBg||!dataUrl) return;
  var im=new Image(); im.onload=function(){
    pushUndo(mbx,'mask');
    var t=document.createElement('canvas'); t.width=W; t.height=H; var tc=t.getContext('2d');
    tc.drawImage(im,0,0,W,H);
    var d=tc.getImageData(0,0,W,H),p=d.data;
    for(var i=0;i<p.length;i+=4){ var on=(p[i]>127||p[i+1]>127||p[i+2]>127);
      p[i]=p[i+1]=p[i+2]=255; p[i+3]=on?255:0; }   // white where selected, transparent elsewhere
    tc.putImageData(d,0,0);
    mbx.clearRect(0,0,W,H); mbx.drawImage(t,0,0);
    showMask=true; var sm=document.getElementById('showmask'); if(sm) sm.classList.add('on');
    compose(); renderLayers(); pushExport(); }; im.src=dataUrl; }
try{ parent.window['__is_'+MODE+'_setmask']=setMask; }catch(e){}

// -- full MultiCanvas state for Projects: serialize ALL layers + base + mask
//    (Save Project) and rebuild them (Load Project). Distinct from the flattened
//    composite export above — this round-trips the editable layer stack. --
function serializeCanvasState(){ if(!hasBg) return '';
  var base=null; try{ if(baseImg){ var bc=document.createElement('canvas'); bc.width=W;bc.height=H;
    bc.getContext('2d').drawImage(baseImg,0,0,W,H); base=bc.toDataURL('image/png'); } }catch(e){}
  var layers=drawLayers.map(function(l){ return {name:l.name, visible:l.visible!==false,
    opacity:(l.opacity==null?1:l.opacity), data:l.cv.toDataURL('image/png')}; });
  var mask=null; try{ mask=maskBuf.toDataURL('image/png'); }catch(e){}
  return JSON.stringify({v:1,w:W,h:H,base:base,layers:layers,mask:mask,active:active,autoMask:autoMask,dilate:dilate}); }
function pushState(){ try{ setHidden('imagesuite-'+MODE+'-state', hasBg?serializeCanvasState():''); }catch(e){} }
try{ parent.window['__is_'+MODE+'_pushstate']=pushState; }catch(e){}
function resetToolUI(){ var sc=document.getElementById('scale'),rt=document.getElementById('rot'),lo=document.getElementById('lopac');
  if(sc){ sc.value=100; document.getElementById('scv').textContent=100; }
  if(rt){ rt.value=0; document.getElementById('rotv').textContent=0; }
  if(lo){ lo.value=100; document.getElementById('lopv').textContent=100; } }
function loadCanvasState(json){ var st; try{ st=JSON.parse(json); }catch(e){ return; } if(!st||!st.w||!st.h) return;
  var finish=function(){ setSize(st.w,st.h); drawLayers=[]; layerSeq=0;
    var arr=st.layers||[], pending=arr.length, done=0;
    var build=function(){ active=Math.min(Math.max(0,st.active||0),Math.max(0,drawLayers.length-1));
      mbx.clearRect(0,0,W,H);
      var fin2=function(){ autoMask=!!st.autoMask; dilate=(st.dilate==null?dilate:st.dilate);
        hasBg=true; document.getElementById('empty').style.display='none';
        undoStack=[]; redoStack=[]; cloneSrc=null; clipboard=null; tSnap=null; tBox=null; tSession=-1; tDx=0;tDy=0;tScale=1;tRot=0;
        resetToolUI(); renderLayers(); compose(); pushExport(); };
      if(st.mask){ var mi=new Image(); mi.onload=function(){ mbx.drawImage(mi,0,0); fin2(); }; mi.onerror=fin2; mi.src=st.mask; } else fin2(); };
    if(!arr.length){ drawLayers=[newLayer('Layer 1')]; build(); return; }
    arr.forEach(function(spec,i){ var l=newLayer(spec.name||('Layer '+(i+1)));
      l.visible=spec.visible!==false; l.opacity=(spec.opacity==null?1:spec.opacity); drawLayers.push(l);
      var im=new Image(); im.onload=function(){ l.ctx.drawImage(im,0,0); if(++done===pending) build(); };
      im.onerror=function(){ if(++done===pending) build(); }; im.src=spec.data; }); };
  if(st.base){ var bi=new Image(); bi.onload=function(){ baseImg=bi; finish(); }; bi.onerror=function(){ baseImg=null; finish(); }; bi.src=st.base; }
  else { baseImg=null; finish(); } }
try{ parent.window['__is_'+MODE+'_loadstate']=function(j){ loadCanvasState(j); }; }catch(e){}

// -- overlays strip: drag a thumbnail onto the canvas → a new layer below mask --
var overlays=[];  // [{name, url}]
var ovstrip=document.getElementById('ovstrip');
function renderOverlays(){ ovstrip.innerHTML='';
  if(!overlays.length){ var e=document.createElement('div'); e.className='ovempty';
    e.textContent='No overlays in this folder. Add some in the Overlays tab.'; ovstrip.appendChild(e); return; }
  overlays.forEach(function(o,i){ var d=document.createElement('div'); d.className='ov'; d.title=o.name+' — drag onto the canvas';
    var im=document.createElement('img'); im.src=o.url; d.appendChild(im); d.draggable=true;
    d.addEventListener('dragstart',function(ev){ ev.dataTransfer.setData('text/plain','ov:'+i); ev.dataTransfer.effectAllowed='copy'; });
    d.addEventListener('dblclick',function(){ document.getElementById('ovprevimg').src=o.url; document.getElementById('ovprev').classList.add('open'); });
    ovstrip.appendChild(d); }); }
function setOverlays(list){ overlays=Array.isArray(list)?list:[]; renderOverlays();
  ovstrip.classList.add('open'); document.getElementById('ovtoggle').innerHTML='&#9662; Overlays'; }
try{ parent.window['__is_'+MODE+'_setoverlays']=setOverlays; }catch(e){}
document.getElementById('ovtoggle').addEventListener('click',function(){ var open=ovstrip.classList.toggle('open');
  this.innerHTML=(open?'&#9662;':'&#9656;')+' Overlays'; });
document.getElementById('ovprev').addEventListener('click',function(){ this.classList.remove('open'); });
function addOverlayLayer(url,cx,cy){ if(!hasBg){ setBg(url); return; }
  var im=new Image(); im.onload=function(){ var l=newLayer('Overlay'); drawLayers.push(l); active=drawLayers.length-1;
    var ow=im.naturalWidth,oh=im.naturalHeight, fit=Math.min(1,(W*0.9)/ow,(H*0.9)/oh), dw=ow*fit,dh=oh*fit;
    var x=(cx==null?W/2:cx)-dw/2, y=(cy==null?H/2:cy)-dh/2;
    x=Math.max(0,Math.min(W-dw,x)); y=Math.max(0,Math.min(H-dh,y));  // keep fully on-canvas
    l.ctx.drawImage(im,x,y,dw,dh); selTool('xform'); ensureXform();
    renderLayers(); compose(); pushExport(); }; im.src=url; }
try{ parent.window['__is_'+MODE+'_addlayer']=function(u){ addOverlayLayer(u); }; }catch(e){}
// external image file drop → draw onto the active (selected) layer at the drop
// point; if the canvas is empty the image becomes the background. Scaled to fit.
function dropImageOnLayer(url,cx,cy){ if(!hasBg){ setBg(url); return; }
  var im=new Image(); im.onload=function(){ var l=activeLayer(); pushUndo(l.ctx,'draw'); l.src=null;
    if(tSession===l.id){ tSession=-1; tSnap=null; tBox=null; }   // new content invalidates a pending transform snapshot
    var ow=im.naturalWidth,oh=im.naturalHeight, fit=Math.min(1,(W*0.9)/ow,(H*0.9)/oh), dw=ow*fit,dh=oh*fit;
    var x=(cx==null?W/2:cx)-dw/2, y=(cy==null?H/2:cy)-dh/2;
    x=Math.max(0,Math.min(W-dw,x)); y=Math.max(0,Math.min(H-dh,y));
    l.ctx.drawImage(im,x,y,dw,dh); selTool('xform'); ensureXform(); renderLayers(); compose(); pushExport(); }; im.src=url; }
stage.addEventListener('dragover',function(e){ var t=e.dataTransfer?(''+e.dataTransfer.types):'';
  if(t.indexOf('text/plain')>=0||t.indexOf('Files')>=0){ e.preventDefault(); e.dataTransfer.dropEffect='copy'; document.body.classList.add('dragover'); } });
stage.addEventListener('dragleave',function(){ document.body.classList.remove('dragover'); });
stage.addEventListener('drop',function(e){ var dt=e.dataTransfer; document.body.classList.remove('dragover');
  // external image file dropped from the OS / another app
  if(dt && dt.files && dt.files.length){ var f=null;
    for(var i=0;i<dt.files.length;i++){ if(/^image\//.test(dt.files[i].type)){ f=dt.files[i]; break; } }
    if(f){ e.preventDefault(); var rf=disp.getBoundingClientRect();
      var fx=(e.clientX-rf.left)/rf.width*W, fy=(e.clientY-rf.top)/rf.height*H;
      var rd=new FileReader(); rd.onload=function(){ dropImageOnLayer(rd.result,fx,fy); }; rd.readAsDataURL(f); return; } }
  // internal overlay thumbnail → new layer
  var data=dt?dt.getData('text/plain'):'';
  if(data.indexOf('ov:')!==0) return; e.preventDefault(); var o=overlays[+data.slice(3)]; if(!o) return;
  var r=disp.getBoundingClientRect(); var x=(e.clientX-r.left)/r.width*W, y=(e.clientY-r.top)/r.height*H;
  addOverlayLayer(o.url,x,y); });

// On window/layout resize do NOT touch the pixel buffers (assigning canvas
// width/height — even to the same value — clears the bitmap). Only recompute
// the fit scale; viewport scaling is pure CSS, so this preserves zoom + pixels.
window.addEventListener('resize',function(){ if(!hasBg) return;
  var availW=stage.parentNode.clientWidth-20; baseScale=Math.min(1,availW/W); applyView(); });
markSwatch(); renderLayers(); updateHardnessUI();
})();
</script></body></html>"""


def _collapse(js: str) -> str:
    return js  # kept as-is; the doc is already a literal


def build_canvas(mode="inpaint"):
    import gradio as gr
    doc = (_CANVAS_DOC.replace("__MODE__", mode)
           .replace("__PALETTE__", str(_PALETTE)))
    iframe = ('<iframe id="imagesuite-' + mode + '-frame" srcdoc="'
              + _html.escape(doc, quote=True)
              + '" style="width:100%;height:680px;border:1px solid #333;'
              + 'border-radius:10px;background:#15151b;"></iframe>')
    c = {"mode": mode}
    c["frame_html"] = gr.HTML(iframe)
    c["composite"] = gr.Textbox(visible=False, elem_id=f"imagesuite-{mode}-composite")
    c["mask"] = gr.Textbox(visible=False, elem_id=f"imagesuite-{mode}-mask")
    c["bg_bridge"] = gr.HTML(visible=False)
    c["setmask_bridge"] = gr.HTML(visible=False)  # loads a segmentation alpha as the mask
    c["ov_bridge"] = gr.HTML(visible=False)  # pushes overlay thumbnails into the strip
    c["addlayer_bridge"] = gr.HTML(visible=False)  # adds one overlay as a new layer
    # Project save/load: 'state' carries the full layer-stack JSON (filled by
    # __is_<mode>_pushstate before a Save read); 'state_bridge' rebuilds it on Load.
    c["state"] = gr.Textbox(visible=False, elem_id=f"imagesuite-{mode}-state")
    c["state_bridge"] = gr.HTML(visible=False)
    return c


def bg_bridge_html(data_url: str, mode="inpaint", nonce="") -> str:
    inner = ("<script>/*" + str(nonce) + "*/try{parent.window['__is_" + mode
             + "_setbg'](" + _js_string(data_url) + ");}catch(e){}</script>")
    return ('<iframe srcdoc="' + _html.escape(inner, quote=True)
            + '" style="display:none;width:0;height:0;border:none"></iframe>')


def setmask_bridge_html(data_url: str, mode="inpaint", nonce="") -> str:
    """Load a segmentation alpha/grayscale PNG (white = subject) into the canvas
    mask layer — see setMask in the iframe JS. Used by Magic select subject."""
    inner = ("<script>/*" + str(nonce) + "*/try{parent.window['__is_" + mode
             + "_setmask'](" + _js_string(data_url) + ");}catch(e){}</script>")
    return ('<iframe srcdoc="' + _html.escape(inner, quote=True)
            + '" style="display:none;width:0;height:0;border:none"></iframe>')


def addlayer_bridge_html(data_url: str, mode="inpaint", nonce="") -> str:
    """Add one overlay (data-URL) to the canvas as a new, centered layer (or as the
    background if the canvas is empty) — see addOverlayLayer in the iframe JS."""
    inner = ("<script>/*" + str(nonce) + "*/try{parent.window['__is_" + mode
             + "_addlayer'](" + _js_string(data_url) + ");}catch(e){}</script>")
    return ('<iframe srcdoc="' + _html.escape(inner, quote=True)
            + '" style="display:none;width:0;height:0;border:none"></iframe>')


def state_bridge_html(state_json: str, mode="inpaint", nonce="") -> str:
    """Rebuild the whole MultiCanvas (base + every layer + mask) from a serialized
    state JSON — see loadCanvasState in the iframe JS. Used by Load Project."""
    inner = ("<script>/*" + str(nonce) + "*/try{parent.window['__is_" + mode
             + "_loadstate'](" + _js_string(state_json or "") + ");}catch(e){}</script>")
    return ('<iframe srcdoc="' + _html.escape(inner, quote=True)
            + '" style="display:none;width:0;height:0;border:none"></iframe>')


def overlays_bridge_html(items, mode="inpaint", nonce="") -> str:
    """Push a list of {name, url(data-URL)} overlays into the canvas strip."""
    import json
    # Escape the script terminator so a crafted overlay name (e.g. "</script>…")
    # can't break out of the inner <script> once srcdoc is un-escaped by the iframe.
    payload = (json.dumps(items or [])
               .replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026'))
    inner = ("<script>/*" + str(nonce) + "*/try{parent.window['__is_" + mode
             + "_setoverlays'](" + payload + ");}catch(e){}</script>")
    return ('<iframe srcdoc="' + _html.escape(inner, quote=True)
            + '" style="display:none;width:0;height:0;border:none"></iframe>')


def _js_string(s: str) -> str:
    import json
    # Escape the script terminator (symmetry with overlays_bridge_html) so the
    # JSON literal can't close the embedded <script> when un-escaped by srcdoc.
    return (json.dumps(s)
            .replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026'))
