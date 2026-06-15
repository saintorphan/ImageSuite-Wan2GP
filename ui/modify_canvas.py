"""Image-modify editor for the ImageSuite plugin.

A self-contained HTML5 image editor embedded as an <iframe srcdoc>, bridged to
Gradio exactly the way ui/canvas.py does:

  * JS -> Python: writes the edited (cropped + colour-corrected) image as a PNG
    data-URL into a hidden Gradio Textbox via ``parent.document`` + the native
    value setter + input/change events (see ``setHidden``).
  * Python -> JS: a hidden <iframe srcdoc> whose <script> calls
    ``parent.window.__is_<mode>_setbg(dataURL)`` to load an image.

Unlike canvas.py this editor has NO layers, NO painting and NO mask. It loads an
image, pans / zooms, supports an interactive CROP rectangle (8 handles + aspect
constraints) and LIVE COLOUR CORRECTION (brightness / contrast / saturation /
hue / warmth) applied with ``ctx.filter`` on the display and baked into the
export. The export is the cropped region with the colour correction burned in.
"""
from __future__ import annotations

import html as _html

_CANVAS_DOC = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
*{margin:0;padding:0;box-sizing:border-box;font-family:system-ui,sans-serif}
html,body{height:100%}
body{background:#15151b;color:#ddd;overflow:hidden;user-select:none}
#root{display:flex;flex-direction:column;height:100%;width:100%}
#main{display:flex;flex:1;min-height:0}
#wrap{flex:1;position:relative;overflow:auto;
  background:#101015 repeating-conic-gradient(#1a1a22 0% 25%,#141419 0% 50%) 0/24px 24px}
#stage{position:relative;margin:10px auto;box-shadow:0 0 0 1px #000,0 6px 24px rgba(0,0,0,.5)}
#stage canvas{position:absolute;top:0;left:0;display:block}
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
.row2{display:grid;grid-template-columns:1fr 1fr;gap:4px}
.row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px}
.aspseg{display:grid;grid-template-columns:1fr 1fr 1fr;gap:4px}
.fld{font-size:10px;color:#8a8a9a;display:flex;justify-content:space-between}
input[type=range]{width:100%}
</style></head><body>
<div id="root">
  <div id="main">
    <div id="wrap"><div id="stage">
      <canvas id="bg"></canvas>
      <canvas id="disp"></canvas>
      <div id="empty">No image yet &#8212; use <b>Send to Modify</b>,<br>or the <b>Image</b> input.</div>
    </div></div>
    <div id="rail">
      <div class="sec">
        <div class="seclabel">View</div>
        <div class="row2">
          <button id="zin" title="Zoom in">&#128269;+</button>
          <button id="zout" title="Zoom out">&#128270;&#8722;</button>
        </div>
        <div class="row2">
          <button id="fit" title="Fit to view">&#10530; Fit</button>
          <button id="full" title="Actual size (100%)">100%</button>
        </div>
        <div class="fld"><span>Zoom</span><span id="zmv">100</span>%</div>
      </div>
      <div class="sec">
        <div class="seclabel">Crop</div>
        <div class="fld"><span>Aspect</span></div>
        <div class="aspseg" id="aspseg">
          <button data-asp="free" class="on" title="Free aspect">Free</button>
          <button data-asp="1:1" title="Square">1:1</button>
          <button data-asp="4:3" title="4:3">4:3</button>
          <button data-asp="3:4" title="3:4">3:4</button>
          <button data-asp="16:9" title="16:9">16:9</button>
          <button data-asp="9:16" title="9:16">9:16</button>
        </div>
        <div class="row2">
          <button id="applycrop" title="Bake the crop region into the export">&#9986; Apply crop</button>
          <button id="resetcrop" title="Restore the full image">&#8634; Reset crop</button>
        </div>
      </div>
      <div class="sec">
        <div class="seclabel">Resize output</div>
        <label class="fld" style="cursor:pointer"><span>Resize export</span>
          <input type="checkbox" id="rs_on"></label>
        <div id="rs_body" style="display:none">
          <div class="aspseg" id="rsseg">
            <button data-px="512" title="Square 512">512</button>
            <button data-px="768" title="Square 768">768</button>
            <button data-px="1024" title="Square 1024">1024</button>
            <button data-px="1536" title="Square 1536">1536</button>
          </div>
          <div class="row2">
            <label style="font-size:11px;color:#cfcfe0">W<input type="number" id="rs_w"
              min="8" max="8192" step="8" style="width:100%"></label>
            <label style="font-size:11px;color:#cfcfe0">H<input type="number" id="rs_h"
              min="8" max="8192" step="8" style="width:100%"></label>
          </div>
          <label class="fld" style="cursor:pointer"><span>Lock to crop aspect</span>
            <input type="checkbox" id="rs_lock" checked></label>
          <div id="rs_info" class="fld" style="font-size:10px;opacity:.65"></div>
        </div>
      </div>
      <div class="sec">
        <div class="seclabel">Colour</div>
        <div class="fld"><span>Brightness</span><span id="briv">100</span></div>
        <input type="range" id="bri" min="50" max="150" value="100">
        <div class="fld"><span>Contrast</span><span id="conv">100</span></div>
        <input type="range" id="con" min="50" max="150" value="100">
        <div class="fld"><span>Saturation</span><span id="satv">100</span></div>
        <input type="range" id="sat" min="0" max="200" value="100">
        <div class="fld"><span>Hue</span><span id="huev">0</span>&#176;</div>
        <input type="range" id="hue" min="-180" max="180" value="0">
        <div class="fld"><span>Warmth</span><span id="warv">0</span></div>
        <input type="range" id="war" min="-100" max="100" value="0">
        <button id="resetcol" title="Reset colour correction">&#8634; Reset colour</button>
      </div>
    </div>
  </div>
</div>
<script>
(function(){
var MODE="__MODE__";
var W=0,H=0,hasBg=false,baseImg=null;
var wrap=document.getElementById('wrap'),stage=document.getElementById('stage');
var bg=document.getElementById('bg'),disp=document.getElementById('disp');
var bgx=bg.getContext('2d'),dx=disp.getContext('2d');
var baseScale=1,viewScale=1;
// colour-correction values (UI units)
var bri=100,con=100,sat=100,hue=0,warmth=0;
// crop state: the rectangle is in bitmap (source) coordinates of the CURRENT
// source. 'baked' offset/size lets Apply crop crop the export only.
var crop={x:0,y:0,w:0,h:0};
var aspect=0;  // 0 = free, else w/h ratio
var baked={x:0,y:0,w:0,h:0};  // region of baseImg that the export covers
var dragging=null,dragStart=null;  // crop interaction state
// resize/output: when outOn, the export is scaled to outW x outH (else native crop)
var outOn=false,outW=0,outH=0,outLock=true;

// does this engine support 2D ctx.filter? (assigning an unsupported value is
// silently ignored rather than thrown, so feature-detect explicitly)
var _CTX_FILTER = (function(){ try{ var c=document.createElement('canvas').getContext('2d');
  c.filter='blur(1px)'; return c.filter==='blur(1px)'; }catch(e){ return false; } })();

// compose the CSS filter string from the colour sliders (brightness/contrast/
// saturate/hue-rotate are native; warmth is approximated with sepia + hue-rotate).
function filterStr(){ var parts=[];
  parts.push('brightness('+(bri/100)+')');
  parts.push('contrast('+(con/100)+')');
  parts.push('saturate('+(sat/100)+')');
  if(hue!==0) parts.push('hue-rotate('+hue+'deg)');
  if(warmth!==0){ var a=Math.min(0.6,Math.abs(warmth)/100*0.6);
    // warmth>0 = warmer (sepia-ish toward orange), <0 = cooler (toward blue)
    parts.push('sepia('+a.toFixed(3)+')');
    parts.push('hue-rotate('+(warmth>0?0:200)+'deg)'); }
  return parts.join(' '); }

function setSize(w,h){ W=w; H=h; bg.width=w; bg.height=h; disp.width=w; disp.height=h; fitView(); }
function fitView(){ var availW=stage.parentNode.clientWidth-20;
  baseScale=Math.min(1,availW/W); viewScale=1; applyView(); }
function applyView(){ var s=baseScale*viewScale;
  stage.style.width=(W*s)+'px'; stage.style.height=(H*s)+'px';
  bg.style.width='100%'; bg.style.height='100%';
  disp.style.width='100%'; disp.style.height='100%';
  var z=document.getElementById('zmv'); if(z) z.textContent=Math.round(s*100); }

// -- render: paint baseImg with the colour filter on bg, then the crop overlay on disp --
function paintBg(){ bgx.clearRect(0,0,W,H);
  if(!baseImg) return; bgx.save();
  if(_CTX_FILTER){ try{ bgx.filter=filterStr(); }catch(e){} }
  bgx.drawImage(baseImg,0,0,W,H); bgx.restore(); }
function drawCropOverlay(){ dx.clearRect(0,0,W,H); if(!hasBg) return;
  // dim everything outside the crop rect
  dx.save(); dx.fillStyle='rgba(0,0,0,.5)';
  dx.fillRect(0,0,W,crop.y);
  dx.fillRect(0,crop.y+crop.h,W,H-(crop.y+crop.h));
  dx.fillRect(0,crop.y,crop.x,crop.h);
  dx.fillRect(crop.x+crop.w,crop.y,W-(crop.x+crop.w),crop.h);
  dx.restore();
  // rect border
  var ds=(baseScale*viewScale)||1, lw=Math.max(1,1.5/ds), hs=Math.max(3,6/ds);
  dx.save(); dx.strokeStyle='rgba(232,62,140,.95)'; dx.lineWidth=lw;
  dx.strokeRect(crop.x,crop.y,crop.w,crop.h);
  // thirds guides
  dx.strokeStyle='rgba(255,255,255,.35)'; dx.lineWidth=Math.max(0.5,0.75/ds);
  for(var i=1;i<3;i++){ dx.beginPath(); dx.moveTo(crop.x+crop.w*i/3,crop.y);
    dx.lineTo(crop.x+crop.w*i/3,crop.y+crop.h); dx.stroke();
    dx.beginPath(); dx.moveTo(crop.x,crop.y+crop.h*i/3);
    dx.lineTo(crop.x+crop.w,crop.y+crop.h*i/3); dx.stroke(); }
  // 8 handles
  dx.fillStyle='#e83e8c';
  handlePts().forEach(function(p){ dx.beginPath(); dx.rect(p.x-hs,p.y-hs,hs*2,hs*2); dx.fill(); });
  dx.restore(); }
function render(){ paintBg(); drawCropOverlay(); }

// handle positions (bitmap coords): corners + edge midpoints, named for hit-test
function handlePts(){ var x0=crop.x,y0=crop.y,x1=crop.x+crop.w,y1=crop.y+crop.h,
    mx=(x0+x1)/2,my=(y0+y1)/2;
  return [{x:x0,y:y0,h:'nw'},{x:mx,y:y0,h:'n'},{x:x1,y:y0,h:'ne'},
          {x:x1,y:my,h:'e'},{x:x1,y:y1,h:'se'},{x:mx,y:y1,h:'s'},
          {x:x0,y:y1,h:'sw'},{x:x0,y:my,h:'w'}]; }

function pos(e){ var r=disp.getBoundingClientRect(),t=e.touches?e.touches[0]:e;
  var x=(t.clientX-r.left)/r.width*W, y=(t.clientY-r.top)/r.height*H;
  return {x:Math.max(0,Math.min(W,x)), y:Math.max(0,Math.min(H,y))}; }
function hitHandle(p){ var ds=(baseScale*viewScale)||1, r=11/ds, pts=handlePts();
  for(var i=0;i<pts.length;i++){ if(Math.abs(p.x-pts[i].x)<r && Math.abs(p.y-pts[i].y)<r) return pts[i].h; }
  if(p.x>=crop.x && p.x<=crop.x+crop.w && p.y>=crop.y && p.y<=crop.y+crop.h) return 'move';
  return null; }

// constrain crop to the aspect ratio (keeps the given anchor edge/corner fixed)
function applyAspectResize(h){ if(!aspect) return;
  // recompute height from width for horizontal edges, width from height otherwise
  var cx=crop.x+crop.w/2, cy=crop.y+crop.h/2;
  if(h==='e'||h==='w'){ var nh=crop.w/aspect; crop.y=cy-nh/2; crop.h=nh; }
  else if(h==='n'||h==='s'){ var nw=crop.h*aspect; crop.x=cx-nw/2; crop.w=nw; }
  else { var nh2=crop.w/aspect; // corners: lock width, derive height toward the drag direction
    if(h==='nw'||h==='ne'){ crop.y=(crop.y+crop.h)-nh2; } crop.h=nh2; }
  clampCrop(); }
function clampCrop(){ if(crop.w<8) crop.w=8; if(crop.h<8) crop.h=8;
  if(crop.w>W) crop.w=W; if(crop.h>H) crop.h=H;
  if(crop.x<0) crop.x=0; if(crop.y<0) crop.y=0;
  if(crop.x+crop.w>W) crop.x=W-crop.w; if(crop.y+crop.h>H) crop.y=H-crop.h; }
function setAspect(a){ aspect=a;
  if(aspect){ // refit the current rect to the new ratio, centered
    var cx=crop.x+crop.w/2, cy=crop.y+crop.h/2;
    var nw=crop.w, nh=nw/aspect; if(nh>H){ nh=H; nw=nh*aspect; }
    if(nw>W){ nw=W; nh=nw/aspect; }
    crop.w=nw; crop.h=nh; crop.x=cx-nw/2; crop.y=cy-nh/2; clampCrop(); render(); pushExport(); } }

function down(e){ if(!hasBg) return; if(e.button!==undefined && e.button!==0) return;
  var p=pos(e); var h=hitHandle(p); if(!h){ // start a fresh crop from this point
    h='se'; crop.x=p.x; crop.y=p.y; crop.w=1; crop.h=1; }
  e.preventDefault(); dragging=h;
  dragStart={p:p, crop:{x:crop.x,y:crop.y,w:crop.w,h:crop.h}}; }
function move(e){ if(!dragging) return; e.preventDefault(); var p=pos(e);
  var s=dragStart.crop, ddx=p.x-dragStart.p.x, ddy=p.y-dragStart.p.y;
  if(dragging==='move'){ crop.x=s.x+ddx; crop.y=s.y+ddy; clampCrop(); }
  else { var x0=s.x,y0=s.y,x1=s.x+s.w,y1=s.y+s.h;
    if(dragging.indexOf('w')>=0) x0=Math.min(p.x,x1-8);
    if(dragging.indexOf('e')>=0) x1=Math.max(p.x,x0+8);
    if(dragging.indexOf('n')>=0) y0=Math.min(p.y,y1-8);
    if(dragging.indexOf('s')>=0) y1=Math.max(p.y,y0+8);
    crop.x=x0; crop.y=y0; crop.w=x1-x0; crop.h=y1-y0;
    if(aspect) applyAspectResize(dragging); clampCrop(); }
  render(); }
function up(){ if(!dragging) return; dragging=null; pushExport(); }
disp.addEventListener('mousedown',down); window.addEventListener('mousemove',move);
window.addEventListener('mouseup',up);
disp.addEventListener('touchstart',down,{passive:false});
disp.addEventListener('touchmove',move,{passive:false}); window.addEventListener('touchend',up);

// -- pan (middle mouse) + zoom (wheel) — copied from canvas.py --
wrap.addEventListener('wheel',function(e){ if(!hasBg) return; e.preventDefault();
  viewScale=Math.max(0.2,Math.min(8,viewScale*(e.deltaY<0?1.1:0.9))); applyView(); render(); },{passive:false});
var panning=false,pSx,pSy,pL,pT;
wrap.addEventListener('mousedown',function(e){ if(e.button===1){ e.preventDefault();
  panning=true; pSx=e.clientX;pSy=e.clientY;pL=wrap.scrollLeft;pT=wrap.scrollTop; } });
window.addEventListener('mousemove',function(e){ if(panning){ wrap.scrollLeft=pL-(e.clientX-pSx); wrap.scrollTop=pT-(e.clientY-pSy); } });
window.addEventListener('mouseup',function(){ panning=false; });

// -- toolbar wiring --
document.getElementById('zin').addEventListener('click',function(){ if(!hasBg) return;
  viewScale=Math.min(8,viewScale*1.2); applyView(); render(); });
document.getElementById('zout').addEventListener('click',function(){ if(!hasBg) return;
  viewScale=Math.max(0.2,viewScale*0.8); applyView(); render(); });
document.getElementById('fit').addEventListener('click',function(){ if(hasBg){ fitView(); render(); } });
document.getElementById('full').addEventListener('click',function(){ if(!hasBg) return;
  viewScale=1/(baseScale||1); applyView(); render(); });

var ASPMAP={'free':0,'1:1':1,'4:3':4/3,'3:4':3/4,'16:9':16/9,'9:16':9/16};
document.querySelectorAll('#aspseg button').forEach(function(b){ b.addEventListener('click',function(){
  document.querySelectorAll('#aspseg button').forEach(function(o){ o.classList.toggle('on',o===b); });
  setAspect(ASPMAP[b.dataset.asp]||0); }); });
document.getElementById('applycrop').addEventListener('click',function(){ if(!hasBg) return;
  // bake: the new source region is the current crop area of the live baseImg.
  baked={x:Math.round(baked.x+crop.x), y:Math.round(baked.y+crop.y),
         w:Math.round(crop.w), h:Math.round(crop.h)};
  // re-base the working image to the cropped pixels (colour filter NOT yet baked;
  // it stays live and is applied at export), then reset the crop rect to full.
  var t=document.createElement('canvas'); t.width=Math.max(1,Math.round(crop.w)); t.height=Math.max(1,Math.round(crop.h));
  t.getContext('2d').drawImage(baseImg, crop.x, crop.y, crop.w, crop.h, 0,0,t.width,t.height);
  var im=new Image(); im.onload=function(){ baseImg=im; setSize(im.naturalWidth,im.naturalHeight);
    crop={x:0,y:0,w:W,h:H}; render(); pushExport(); }; im.src=t.toDataURL('image/png'); });
document.getElementById('resetcrop').addEventListener('click',function(){ if(!hasBg) return;
  crop={x:0,y:0,w:W,h:H}; render(); pushExport(); });

function bindCol(id,vid,setter,suffix){ var el=document.getElementById(id),lab=document.getElementById(vid);
  el.addEventListener('input',function(e){ setter(+e.target.value); lab.textContent=e.target.value; render(); pushExport(); }); }
bindCol('bri','briv',function(v){ bri=v; });
bindCol('con','conv',function(v){ con=v; });
bindCol('sat','satv',function(v){ sat=v; });
bindCol('hue','huev',function(v){ hue=v; });
bindCol('war','warv',function(v){ warmth=v; });
function resetColUI(){ bri=100;con=100;sat=100;hue=0;warmth=0;
  var m=[['bri','briv',100],['con','conv',100],['sat','satv',100],['hue','huev',0],['war','warv',0]];
  m.forEach(function(o){ var el=document.getElementById(o[0]); if(el) el.value=o[2];
    var lab=document.getElementById(o[1]); if(lab) lab.textContent=o[2]; }); }
document.getElementById('resetcol').addEventListener('click',function(){ resetColUI(); render(); pushExport(); });

// -- resize / output size: crop selects the region, this sets the exported pixels.
// e.g. crop a 1:1 679x679 region and emit it at 1024x1024. --
function cropAspect(){ return (crop.h>0)? crop.w/crop.h : 1; }
function rsInfo(){ var i=document.getElementById('rs_info'); if(!i) return;
  var cw=Math.round(crop.w), ch=Math.round(crop.h);
  i.textContent=(outOn && outW>0 && outH>0)
    ? ('crop '+cw+'×'+ch+' → '+outW+'×'+outH)
    : ('crop '+cw+'×'+ch+' (native)'); }
function rsSet(w,h){ outW=Math.max(8,Math.round(w)); outH=Math.max(8,Math.round(h));
  var ew=document.getElementById('rs_w'),eh=document.getElementById('rs_h');
  if(ew) ew.value=outW; if(eh) eh.value=outH; rsInfo(); }
function resetRsUI(){ outOn=false; outW=0; outH=0; outLock=true;
  var on=document.getElementById('rs_on'); if(on) on.checked=false;
  var body=document.getElementById('rs_body'); if(body) body.style.display='none';
  var lk=document.getElementById('rs_lock'); if(lk) lk.checked=true;
  document.querySelectorAll('#rsseg button').forEach(function(o){ o.classList.remove('on'); }); }
document.getElementById('rs_on').addEventListener('change',function(e){ outOn=e.target.checked;
  document.getElementById('rs_body').style.display=outOn?'':'none';
  if(outOn && (!outW||!outH)) rsSet(Math.round(crop.w),Math.round(crop.h));
  rsInfo(); pushExport(); });
document.querySelectorAll('#rsseg button').forEach(function(b){ b.addEventListener('click',function(){
  document.querySelectorAll('#rsseg button').forEach(function(o){ o.classList.toggle('on',o===b); });
  var px=+b.dataset.px; rsSet(px, outLock? px/cropAspect() : px); pushExport(); }); });
document.getElementById('rs_w').addEventListener('input',function(e){ outW=Math.max(8,Math.round(+e.target.value||0));
  if(outLock){ outH=Math.max(8,Math.round(outW/cropAspect()));
    var eh=document.getElementById('rs_h'); if(eh) eh.value=outH; } rsInfo(); pushExport(); });
document.getElementById('rs_h').addEventListener('input',function(e){ outH=Math.max(8,Math.round(+e.target.value||0));
  if(outLock){ outW=Math.max(8,Math.round(outH*cropAspect()));
    var ew=document.getElementById('rs_w'); if(ew) ew.value=outW; } rsInfo(); pushExport(); });
document.getElementById('rs_lock').addEventListener('change',function(e){ outLock=e.target.checked; });

// -- export (JS -> Python hidden Textbox) --
function setHidden(id,val){ try{ var pd=parent.document;
  var e=pd.querySelector('#'+id+' textarea')||pd.querySelector('#'+id+' input'); if(!e) return;
  var proto=(e.tagName==='TEXTAREA')?parent.HTMLTextAreaElement.prototype:parent.HTMLInputElement.prototype;
  Object.getOwnPropertyDescriptor(proto,'value').set.call(e,val);
  e.dispatchEvent(new Event('input',{bubbles:true})); e.dispatchEvent(new Event('change',{bubbles:true})); }catch(err){} }
function buildExport(){ if(!baseImg) return '';
  // export = the current crop region of the live baseImg, with the colour filter
  // baked into the pixels via ctx.filter (or drawn raw if filters unsupported),
  // scaled to outW x outH when Resize is on (else the crop's native pixels).
  var cw=Math.max(1,Math.round(crop.w)), ch=Math.max(1,Math.round(crop.h));
  var ow=cw, oh=ch;
  if(outOn && outW>0 && outH>0){ ow=outW; oh=outH; }
  var t=document.createElement('canvas'); t.width=ow; t.height=oh; var tc=t.getContext('2d');
  if(_CTX_FILTER){ try{ tc.filter=filterStr(); }catch(e){} }
  tc.imageSmoothingQuality='high';
  tc.drawImage(baseImg, crop.x, crop.y, crop.w, crop.h, 0,0,ow,oh);
  return t.toDataURL('image/png'); }
function exportNow(){ if(!hasBg) return; setHidden('imagesuite-'+MODE+'-out', buildExport()); }
var exportTimer=null;
function pushExport(){ if(!hasBg) return; rsInfo(); clearTimeout(exportTimer); exportTimer=setTimeout(exportNow,120); }
try{ parent.window['__is_'+MODE+'_exportnow']=exportNow; }catch(e){}

function setBg(dataUrl){ var im=new Image(); im.onload=function(){ baseImg=im;
  setSize(im.naturalWidth,im.naturalHeight);
  crop={x:0,y:0,w:W,h:H}; baked={x:0,y:0,w:W,h:H};
  aspect=0; document.querySelectorAll('#aspseg button').forEach(function(b){ b.classList.toggle('on',b.dataset.asp==='free'); });
  resetColUI(); resetRsUI();
  hasBg=true; document.getElementById('empty').style.display='none';
  fitView(); render(); pushExport(); }; im.src=dataUrl; }
try{ parent.window['__is_'+MODE+'_setbg']=setBg; }catch(e){}

// On window/layout resize do NOT touch the pixel buffers (assigning canvas
// width/height — even to the same value — clears the bitmap). Only recompute
// the fit scale; viewport scaling is pure CSS, so this preserves zoom + pixels.
window.addEventListener('resize',function(){ if(!hasBg) return;
  var availW=stage.parentNode.clientWidth-20; baseScale=Math.min(1,availW/W); applyView(); render(); });
})();
</script></body></html>"""


def build_modify_canvas(mode="modify"):
    import gradio as gr
    doc = _CANVAS_DOC.replace("__MODE__", mode)
    iframe = ('<iframe id="imagesuite-' + mode + '-frame" srcdoc="'
              + _html.escape(doc, quote=True)
              + '" style="width:100%;height:620px;border:1px solid #333;'
              + 'border-radius:10px;background:#15151b;"></iframe>')
    c = {"mode": mode}
    c["frame_html"] = gr.HTML(iframe)
    c["out"] = gr.Textbox(visible=False, elem_id=f"imagesuite-{mode}-out")
    c["bg_bridge"] = gr.HTML(visible=False)
    return c


def modify_bg_bridge_html(data_url: str, mode="modify", nonce="") -> str:
    inner = ("<script>/*" + str(nonce) + "*/try{parent.window['__is_" + mode
             + "_setbg'](" + _js_string(data_url) + ");}catch(e){}</script>")
    return ('<iframe srcdoc="' + _html.escape(inner, quote=True)
            + '" style="display:none;width:0;height:0;border:none"></iframe>')


def _js_string(s: str) -> str:
    import json
    # Escape the script terminator (symmetry with canvas.py) so the JSON literal
    # can't close the embedded <script> when un-escaped by srcdoc.
    return (json.dumps(s)
            .replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026'))
