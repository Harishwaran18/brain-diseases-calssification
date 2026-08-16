"""Three.js WebGL brain viewer for the NeuroCure platform.

Replaces the Plotly 3D renderer with a genuine WebGL scene built on Three.js
(the international standard for in-browser 3D visualisation, used by
BrainBrowser and the Allen Brain Atlas). The full-resolution fsaverage pial
cortex is rendered with PBR (physically-based) materials lit by a procedural
image-based environment (RoomEnvironment + PMREM), ACES Filmic tone mapping,
hemisphere + key/rim/fill lighting, and per-vertex normals so the real
gyri/sulci are crisp and anatomically lifelike. An UnrealBloom pass makes the
emissive lesion glow during the cure cascade (gracefully disabled on WebGL1).

Interactions: OrbitControls (rotate/zoom/pan), auto-rotate, a clipping plane
to slice the brain open and reveal the internal lesion, a PNG snapshot button,
an anatomical orientation gizmo (L/R/A/P/S/I), a tissue legend, and a
ResizeObserver so the view resizes correctly inside the Streamlit iframe.

The cure animation shrinks the lesion mesh toward its centroid over time while
an on-canvas overlay names the disease being treated and the curing technique
being applied, so the viewer always shows what is happening.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import streamlit.components.v1 as components

from brainframe.reconstruction.marching import MeshData


def _mesh_to_json(mesh: MeshData, decimate_to: int | None = None) -> dict:
    """Serialise a mesh to a compact JSON dict (positions + indices + normals)."""
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    normals = np.asarray(mesh.normals, dtype=np.float32) if mesh.normals is not None else None
    if decimate_to is not None and len(verts) > decimate_to:
        try:
            from brainframe.reporting.unified_report import _decimate_mesh

            verts, faces = _decimate_mesh(verts, faces, target_verts=decimate_to)
            # Recompute normals after decimation for smooth shading.
            from brainframe.data.real_brain import _compute_normals

            normals = _compute_normals(verts, faces)
        except Exception:
            pass
    out: dict[str, Any] = {
        "positions": np.asarray(verts, dtype=np.float32).flatten().tolist(),
        "indices": np.asarray(faces, dtype=np.int32).flatten().tolist(),
    }
    if normals is not None and len(normals) == len(verts):
        out["normals"] = np.asarray(normals, dtype=np.float32).flatten().tolist()
    return out


def _recenter(verts: np.ndarray) -> np.ndarray:
    """Centre a mesh on the origin for clean orbiting."""
    if len(verts) == 0:
        return verts
    return verts - verts.mean(axis=0)


def _shrink_frames(
    lesion_verts: np.ndarray, centroid: np.ndarray, n_frames: int = 9
) -> list[list[float]]:
    """Pre-compute per-frame lesion vertex positions (shrinking toward centroid)."""
    frames: list[list[float]] = []
    for fi in range(n_frames):
        frac = fi / (n_frames - 1) if n_frames > 1 else 0.0
        # Shrink toward centroid: scale = 1 - frac, translate toward centroid.
        scaled = lesion_verts * (1.0 - frac) + centroid * frac
        frames.append(scaled.astype(np.float32).flatten().tolist())
    return frames


def _phase_lesion_frames(
    lesion_verts: np.ndarray, centroid: np.ndarray, scales: list[float]
) -> list[list[float]]:
    """Per-frame lesion positions driven by the cure phase scales.

    ``scales`` are per-frame lesion_scale values (1.0 = full, 0.0 = gone) from
    the :class:`~brainframe.therapy.cure_phases.CureTimeline`.
    """
    frames: list[list[float]] = []
    for scale in scales:
        frac = 1.0 - scale  # how far toward centroid
        scaled = lesion_verts * (1.0 - frac) + centroid * frac
        frames.append(scaled.astype(np.float32).flatten().tolist())
    return frames


def _regen_frames(
    lesion_verts: np.ndarray, centroid: np.ndarray, regen_intensities: list[float]
) -> list[list[float]]:
    """Per-frame regeneration-mesh positions.

    The regeneration mesh starts at the lesion cavity and grows outward as
    healthy tissue regrows. ``regen_intensities`` are 0..1 per frame.
    """
    frames: list[list[float]] = []
    for ri in regen_intensities:
        # Regen mesh = lesion shape scaled down by (1 - regen), centred on centroid.
        # As regen -> 1 the mesh is small (cavity filled); as regen -> 0 it equals
        # the original lesion size. We invert so high regen = small filled cavity.
        scale = max(0.02, 1.0 - ri)
        scaled = (lesion_verts - centroid) * scale + centroid
        frames.append(scaled.astype(np.float32).flatten().tolist())
    return frames


_THREE_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<style>
  html,body{margin:0;padding:0;height:100%;background:#0a0e14;overflow:hidden;font-family:system-ui,sans-serif}
  #app{position:relative;width:100%;height:100%}
  #overlay{position:absolute;top:14px;left:14px;color:#e6edf3;background:rgba(13,17,23,.82);
           border:1px solid #30363d;border-radius:10px;padding:14px 18px;max-width:62%;
           font-size:13px;line-height:1.55;pointer-events:none;z-index:5}
  #overlay .disease{color:#3aa6e6;font-weight:700;font-size:15px}
  #overlay .technique{color:#3fb950;font-weight:600}
  #overlay .phase{color:#f0a040;font-weight:700;font-size:14px;margin-top:8px}
  #overlay .mechanism{color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin-top:2px}
  #overlay .desc{color:#c9d1d9;font-size:12px;margin-top:6px;font-style:italic}
  #overlay .meta{color:#8b949e;font-size:12px;margin-top:8px}
  #hud{position:absolute;top:14px;right:14px;color:#8b949e;font-size:11px;text-align:right;z-index:5}
  #err{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:#8b949e;font-size:12px;display:none;z-index:6;text-align:center;background:rgba(13,17,23,.85);border:1px solid #30363d;border-radius:8px;padding:10px 16px;max-width:70%}
  #ctrl{position:absolute;bottom:60px;left:50%;transform:translateX(-50%);display:flex;gap:8px;z-index:5;flex-wrap:wrap;justify-content:center;max-width:90%}
  #ctrl button{background:#161b22;color:#e6edf3;border:1px solid #30363d;border-radius:8px;padding:8px 14px;font-size:13px;cursor:pointer;transition:border-color .15s,background .15s}
  #ctrl button:hover{border-color:#3aa6e6}
  #ctrl button.on{background:#1f2d3a;border-color:#3aa6e6;color:#3aa6e6}
  #timeline{position:absolute;bottom:14px;left:50%;transform:translateX(-50%);width:78%;z-index:5}
  #timeline .tlbar{height:10px;background:#161b22;border:1px solid #30363d;border-radius:6px;overflow:hidden;display:flex}
  #timeline .tlseg{height:100%;opacity:.55;transition:opacity .2s}
  #timeline .tlseg.active{opacity:1}
  #timeline .tlhead{display:flex;justify-content:space-between;color:#8b949e;font-size:10px;margin-bottom:3px}
  #timeline .tllabels{display:flex;color:#8b949e;font-size:9px;margin-top:3px;gap:2px}
  #timeline .tllabels span{flex:1;text-align:center;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  #gizmo{position:absolute;bottom:14px;right:14px;width:84px;height:84px;z-index:5;pointer-events:none}
  #legend{position:absolute;top:14px;left:14px;transform:translateY(220px);color:#c9d1d9;background:rgba(13,17,23,.78);border:1px solid #30363d;border-radius:8px;padding:8px 12px;font-size:11px;z-index:5;max-width:180px}
  #legend .lgrow{display:flex;align-items:center;gap:6px;margin:3px 0}
  #legend .swatch{width:12px;height:12px;border-radius:3px;border:1px solid #30363d}
</style>
</head>
<body>
<div id="app">
  <canvas id="fallback" style="position:absolute;inset:0;width:100%;height:100%;z-index:1;display:none"></canvas>
  <div id="overlay">
    <div class="disease" id="dv"></div>
    <div class="technique" id="tv"></div>
    <div class="phase" id="pv"></div>
    <div class="mechanism" id="mc"></div>
    <div class="desc" id="ds"></div>
    <div class="meta" id="mv"></div>
  </div>
  <div id="legend"></div>
  <div id="hud">drag: rotate &middot; scroll: zoom &middot; right-drag: pan</div>
  <div id="err"></div>
  <div id="ctrl">
    <button id="play">▶ Play cure</button>
    <button id="reset">↺ Reset view</button>
    <button id="autorotate" title="Auto-rotate">⟳ Spin</button>
    <button id="clip" title="Clip plane through the brain">✂ Clip</button>
    <button id="snap" title="Save a PNG snapshot">📷 Snapshot</button>
  </div>
  <div id="timeline">
    <div class="tlhead"><span id="tlleft">Before</span><span id="tlright">Cured</span></div>
    <div class="tlbar" id="tlbar"></div>
    <div class="tllabels" id="tllabels"></div>
  </div>
  <canvas id="gizmo"></canvas>
</div>
<script type="importmap">
{ "imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.169.0/examples/jsm/"
}}
</script>
<script>
// Bootstrap: expose cure data on `window` as plain JSON so (a) the Three.js
// module below and (b) the cure-UI plain script after it can both read it
// without re-parsing. This runs synchronously before the deferred module.
window.__NC_DATA__ = __DATA__;
window.__NC_DISEASE__ = __DISEASE__;
window.__NC_TECHNIQUE__ = __TECHNIQUE__;
window.__NC_BEFORE_V__ = __BEFORE_V__;
window.__NC_AFTER_V__ = __AFTER_V__;
// Mesh-ref bag populated by the Three.js module (null until then). The cure-UI
// script reads it defensively so it works even if the module never loads.
window.__nc3d = { lesionMesh:null, lesionGeom:null, regenMesh:null, regenGeom:null, edemaMesh:null, protectMesh:null, ok:false };
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

// Cure data is exposed on `window` by a preceding plain <script> so this
// module AND the cure UI script below can both read it without duplicating
// the JSON. The cure UI runs in a plain (non-module) script so it still
// works if this Three.js module fails to load (e.g. CDN blocked in a
// sandboxed iframe).
const DATA = window.__NC_DATA__;
const DISEASE = window.__NC_DISEASE__;
const TECHNIQUE = window.__NC_TECHNIQUE__;
const BEFORE_V = window.__NC_BEFORE_V__;
const AFTER_V = window.__NC_AFTER_V__;
const TIMELINE = (DATA && DATA.timeline) || null;

const app = document.getElementById('app');
const err = document.getElementById('err');
function fail(msg){ err.style.display='block'; err.textContent='3D viewer error: '+msg; }

// Mesh refs live on window.__nc3d (initialised by the bootstrap script) so the
// plain cure-UI script can update them when WebGL succeeds; they stay null
// when WebGL is unavailable and the cure script gracefully skips 3D updates.
function showFallback(){ window.__ncShowFallback && window.__ncShowFallback(); }

try {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0e14);

  const w = app.clientWidth || 800, h = app.clientHeight || 620;
  const camera = new THREE.PerspectiveCamera(45, w/h, 0.1, 2000);

  // ---- Renderer: physically-based, tone-mapped, colour-managed ----
  const renderer = new THREE.WebGLRenderer({antialias:true, alpha:false, powerPreference:'high-performance'});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(w, h);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  // ACES Filmic tone mapping gives cinematic, film-like contrast that keeps
  // highlight detail in the bright cortical surface and the emissive lesion.
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  app.appendChild(renderer.domElement);

  // ---- Image-based lighting: procedural studio environment (no network) ----
  // PMREM-processed RoomEnvironment gives every PBR material realistic
  // specular reflections so the gyri/sulci catch the light like real tissue.
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

  // ---- Lighting: hemisphere (sky/ground) + key + rim + fill ----
  // Hemisphere light replaces flat ambient for natural sky/ground gradient.
  scene.add(new THREE.HemisphereLight(0xddeeff, 0x202830, 0.55));
  const key = new THREE.DirectionalLight(0xfff2e0, 1.5); key.position.set(120, 200, 180); scene.add(key);
  const rim = new THREE.DirectionalLight(0xb8d4ff, 0.85); rim.position.set(-160, -80, -140); scene.add(rim);
  const fill = new THREE.PointLight(0xffe0b0, 0.45, 800); fill.position.set(0, -150, 120); scene.add(fill);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.08;
  controls.minDistance = 40; controls.maxDistance = 600;

  // Shared clipping plane (toggled by the ✂ Clip button) to slice the brain
  // open and reveal the internal lesion — a core neuroimaging interaction.
  const clipPlane = new THREE.Plane(new THREE.Vector3(-1, 0, 0), 0);
  let clipOn = false;
  renderer.localClippingEnabled = true;

  function makeMesh(m, color, opacity, roughness, metalness, flat){
    if(!m || !m.positions || m.positions.length===0) return null;
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.Float32BufferAttribute(m.positions, 3));
    if(m.normals && m.normals.length===m.positions.length) geom.setAttribute('normal', new THREE.Float32BufferAttribute(m.normals, 3));
    else geom.computeVertexNormals();
    geom.setIndex(m.indices);
    const mat = new THREE.MeshStandardMaterial({color, metalness, roughness, flatShading: !!flat, transparent: opacity<1.0, opacity, side: THREE.DoubleSide, envMapIntensity:0.7, clippingPlanes: clipOn?[clipPlane]:[]});
    const mesh = new THREE.Mesh(geom, mat);
    scene.add(mesh);
    return mesh;
  }

  // Cortex (left/right hemispheres subtly tinted) — the realistic brain backdrop.
  let leftMesh=null, rightMesh=null;
  if(DATA.cortexLeft)  leftMesh  = makeMesh(DATA.cortexLeft,  0xc98a4b, 1.0, 0.45, 0.06, false);
  if(DATA.cortexRight) rightMesh = makeMesh(DATA.cortexRight, 0xc0793a, 1.0, 0.45, 0.06, false);
  // Tissue meshes.
  if(DATA.tissues) for(const t of DATA.tissues) makeMesh(t.mesh, t.color, t.opacity, 0.6, 0.0, t.flat);

  // ---- Lesion mesh (phase-coloured, shrinks per cure phase, emissive glow) ----
  const RG = window.__nc3d;
  RG.lesionMesh=null; RG.lesionGeom=null;
  if(DATA.lesion && DATA.lesion.positions && DATA.lesion.positions.length){
    RG.lesionGeom = new THREE.BufferGeometry();
    RG.lesionGeom.setAttribute('position', new THREE.Float32BufferAttribute(DATA.lesion.positions, 3));
    if(DATA.lesion.normals) RG.lesionGeom.setAttribute('normal', new THREE.Float32BufferAttribute(DATA.lesion.normals, 3));
    else RG.lesionGeom.computeVertexNormals();
    RG.lesionGeom.setIndex(DATA.lesion.indices);
    // Strong emissive so the lesion glows and picks up the bloom pass during cure.
    const lmat = new THREE.MeshStandardMaterial({color:0xff2b4a, metalness:0.1, roughness:0.35, flatShading:true, transparent:true, opacity:0.95, emissive:0xff1133, emissiveIntensity:0.6, envMapIntensity:0.4, clippingPlanes: clipOn?[clipPlane]:[]});
    RG.lesionMesh = new THREE.Mesh(RG.lesionGeom, lmat);
    scene.add(RG.lesionMesh);
  }

  // ---- Regeneration mesh (healthy tissue regrowing into the cavity) ----
  RG.regenMesh=null; RG.regenGeom=null;
  if(DATA.regenFrames && DATA.regenFrames.length && DATA.lesion && DATA.lesion.indices){
    RG.regenGeom = new THREE.BufferGeometry();
    RG.regenGeom.setAttribute('position', new THREE.Float32BufferAttribute(DATA.regenFrames[0], 3));
    RG.regenGeom.setIndex(DATA.lesion.indices);
    RG.regenGeom.computeVertexNormals();
    const rmat = new THREE.MeshStandardMaterial({color:0x3fb950, metalness:0.05, roughness:0.5, flatShading:false, transparent:true, opacity:0.0, emissive:0x0a3318, emissiveIntensity:0.5, envMapIntensity:0.6, clippingPlanes: clipOn?[clipPlane]:[]});
    RG.regenMesh = new THREE.Mesh(RG.regenGeom, rmat);
    scene.add(RG.regenMesh);
  }

  // ---- Oedema halo (inflammation sphere around the lesion) ----
  RG.edemaMesh=null;
  if(DATA.lesion && DATA.lesion.positions && DATA.lesion.positions.length){
    // Compute lesion bounding sphere for the halo.
    let mn=[1e9,1e9,1e9], mx=[-1e9,-1e9,-1e9];
    const p=DATA.lesion.positions;
    for(let i=0;i<p.length;i+=3){for(let j=0;j<3;j++){if(p[i+j]<mn[j])mn[j]=p[i+j];if(p[i+j]>mx[j])mx[j]=p[i+j];}}
    const cx=(mn[0]+mx[0])/2, cy=(mn[1]+mx[1])/2, cz=(mn[2]+mx[2])/2;
    let r=0; for(let i=0;i<p.length;i+=3){const d=Math.hypot(p[i]-cx,p[i+1]-cy,p[i+2]-cz); if(d>r)r=d;}
    r = r*1.35 || 12;
    const egeom = new THREE.SphereGeometry(r, 24, 16);
    egeom.translate(cx, cy, cz);
    const emat = new THREE.MeshBasicMaterial({color:0xff6a3a, transparent:true, opacity:0.0, side: THREE.BackSide, depthWrite:false});
    RG.edemaMesh = new THREE.Mesh(egeom, emat);
    scene.add(RG.edemaMesh);
  }

  // ---- Neuroprotective field (translucent shield around penumbra) ----
  RG.protectMesh=null;
  if(DATA.lesion && DATA.lesion.positions && DATA.lesion.positions.length){
    let mn=[1e9,1e9,1e9], mx=[-1e9,-1e9,-1e9];
    const p=DATA.lesion.positions;
    for(let i=0;i<p.length;i+=3){for(let j=0;j<3;j++){if(p[i+j]<mn[j])mn[j]=p[i+j];if(p[i+j]>mx[j])mx[j]=p[i+j];}}
    const cx=(mn[0]+mx[0])/2, cy=(mn[1]+mx[1])/2, cz=(mn[2]+mx[2])/2;
    let r=0; for(let i=0;i<p.length;i+=3){const d=Math.hypot(p[i]-cx,p[i+1]-cy,p[i+2]-cz); if(d>r)r=d;}
    r = r*1.6 || 16;
    const pgeom = new THREE.SphereGeometry(r, 24, 16);
    pgeom.translate(cx, cy, cz);
    const pmat = new THREE.MeshBasicMaterial({color:0x9a7ad0, transparent:true, opacity:0.0, side: THREE.BackSide, depthWrite:false, wireframe:true});
    RG.protectMesh = new THREE.Mesh(pgeom, pmat);
    scene.add(RG.protectMesh);
  }

  // Frame the camera on the whole scene; store the home view for Reset.
  const box = new THREE.Box3().setFromObject(scene);
  let homePos = camera.position.clone(), homeTarget = controls.target.clone();
  if(!box.isEmpty()){ const c=box.getCenter(new THREE.Vector3()), s=box.getSize(new THREE.Vector3());
    const maxDim=Math.max(s.x,s.y,s.z); camera.position.set(c.x+maxDim*0.9, c.y+maxDim*0.7, c.z+maxDim*1.3); controls.target.copy(c);
    homePos = camera.position.clone(); homeTarget = c.clone();
    // Position the clip plane through the scene centre.
    clipPlane.constant = c.x; clipPlane.normal.set(-1,0,0);
  }

  // ---- Post-processing: bloom on the emissive lesion during the cure ----
  // EffectComposer needs WebGL2; gracefully fall back to the plain renderer
  // (still tone-mapped + environment-lit) on older contexts.
  let composer = null, useComposer = false;
  try {
    if (renderer.capabilities.isWebGL2) {
      composer = new EffectComposer(renderer);
      composer.addPass(new RenderPass(scene, camera));
      const bloom = new UnrealBloomPass(new THREE.Vector2(w, h), 0.62, 0.6, 0.18);
      composer.addPass(bloom);
      composer.addPass(new OutputPass());
      useComposer = true;
    }
  } catch(e){ console.warn('Bloom unavailable, using plain render:', e); useComposer = false; }

  // ---- Orientation gizmo (anatomical L/R/A/P/S/I) drawn in a corner canvas ----
  const gizCanvas = document.getElementById('gizmo');
  let gizCtx = null;
  if(gizCanvas){ gizCanvas.width=168; gizCanvas.height=168; gizCtx = gizCanvas.getContext('2d'); }
  function drawGizmo(){
    if(!gizCtx) return;
    const W=168,H=168,cx=W/2,cy=H/2;
    gizCtx.clearRect(0,0,W,H);
    // Project the world axes through the camera quaternion to find screen directions.
    const q = camera.quaternion;
    const axes = [
      {dir:new THREE.Vector3(1,0,0),  label:'R', color:'#f85149'},   // +x = Right
      {dir:new THREE.Vector3(-1,0,0), label:'L', color:'#3fb950'},   // -x = Left
      {dir:new THREE.Vector3(0,1,0),  label:'A', color:'#d29922'},   // +y = Anterior
      {dir:new THREE.Vector3(0,-1,0), label:'P', color:'#a371f7'},   // -y = Posterior
      {dir:new THREE.Vector3(0,0,1),  label:'S', color:'#3aa6e6'},   // +z = Superior
      {dir:new THREE.Vector3(0,0,-1), label:'I', color:'#8b949e'},   // -z = Inferior
    ];
    // Rotate world-axis by inverse of camera quaternion to get view-space dir.
    const inv = q.clone().invert();
    const proj = axes.map(a => {
      const v = a.dir.clone().applyQuaternion(inv);
      return {...a, x: cx + v.x*52, y: cy - v.y*52, z: v.z};
    });
    // Draw back-to-front (negative z first).
    proj.sort((a,b)=>a.z-b.z);
    gizCtx.font='bold 15px system-ui'; gizCtx.textAlign='center'; gizCtx.textBaseline='middle';
    for(const a of proj){
      gizCtx.strokeStyle=a.color; gizCtx.lineWidth=2;
      gizCtx.beginPath(); gizCtx.moveTo(cx,cy); gizCtx.lineTo(a.x,a.y); gizCtx.stroke();
      gizCtx.fillStyle=a.color; gizCtx.beginPath(); gizCtx.arc(a.x,a.y,9,0,Math.PI*2); gizCtx.fill();
      gizCtx.fillStyle='#fff';
      gizCtx.fillText(a.label, a.x, a.y+0.5);
    }
    gizCtx.fillStyle='#e6edf3'; gizCtx.beginPath(); gizCtx.arc(cx,cy,4,0,Math.PI*2); gizCtx.fill();
  }

  // ---- Tissue legend (compact, in the overlay area) ----
  const legendEl = document.getElementById('legend');
  if(legendEl){
    const items = [
      {c:'#c98a4b', n:'Cortex'},
      {c:'#9A7AD0', n:'Gray matter'},
      {c:'#F0E6D2', n:'White matter'},
      {c:'#3A5A8A', n:'CSF'},
      {c:'#ff2b4a', n:'Lesion'},
      {c:'#3fb950', n:'Regeneration'},
    ];
    legendEl.innerHTML = items.map(i=>`<div class="lgrow"><span class="swatch" style="background:${i.c}"></span>${i.n}</div>`).join('');
  }

  // ---- Control buttons (plain-script-safe: stored on window.__nc3d) ----
  const autoBtn = document.getElementById('autorotate');
  const clipBtn = document.getElementById('clip');
  const snapBtn = document.getElementById('snap');
  let autoRotate = false;
  if(autoBtn) autoBtn.onclick = ()=>{ autoRotate=!autoRotate; controls.autoRotate=autoRotate; controls.autoRotateSpeed=1.2; autoBtn.classList.toggle('on',autoRotate); };
  if(clipBtn) clipBtn.onclick = ()=>{ clipOn=!clipOn;
    // Apply clipping planes to every material in the scene.
    scene.traverse(o=>{ if(o.isMesh && o.material){ const m=o.material; m.clippingPlanes = clipOn?[clipPlane]:[]; m.needsUpdate=true; if(o.material.length){o.material.forEach(mm=>{mm.clippingPlanes=clipOn?[clipPlane]:[]; mm.needsUpdate=true;});} } });
    clipBtn.classList.toggle('on',clipOn);
  };
  if(snapBtn) snapBtn.onclick = ()=>{ try{ renderer.render(scene,camera); const url=renderer.domElement.toDataURL('image/png'); const a=document.createElement('a'); a.href=url; a.download='neurocure_brain.png'; a.click(); }catch(e){} };

  // Expose a reset that also restores home view (the cure-UI reset button calls applyFrame(0);
  // the Reset-view button here restores the camera).
  const resetBtn = document.getElementById('reset');
  function resetCamera(){ camera.position.copy(homePos); controls.target.copy(homeTarget); controls.update(); }
  if(resetBtn) resetBtn.onclick = resetCamera;
  // Allow the cure-UI plain script's Reset button to also restore the camera.
  window.__nc3d.resetCamera = resetCamera;

  // ---- Animation loop: bloom composer + gizmo + lesion pulse during cure ----
  let t0 = performance.now();
  function animate(){ requestAnimationFrame(animate);
    controls.update();
    // Pulse the lesion emissive while the cure is playing for a "live" glow.
    if(RG.lesionMesh){
      const t = (performance.now()-t0)/1000;
      const pulse = 0.5 + 0.25*Math.sin(t*4.0);
      RG.lesionMesh.material.emissiveIntensity = 0.45 + pulse*0.35;
    }
    if(useComposer && composer) composer.render(); else renderer.render(scene, camera);
    drawGizmo();
  }
  animate();

  // ---- Resize: ResizeObserver (iframe-friendly) + window fallback ----
  function doResize(){ const W=app.clientWidth||800, H=app.clientHeight||620; camera.aspect=W/H; camera.updateProjectionMatrix(); renderer.setSize(W,H); if(composer) composer.setSize(W,H); }
  if(typeof ResizeObserver !== 'undefined'){ const ro=new ResizeObserver(doResize); ro.observe(app); }
  else window.addEventListener('resize', doResize);

  // WebGL init succeeded: tell the cure-UI script it can drive the 3D meshes.
  window.__nc3d.ok = true;

} catch(e){ fail('3D viewer unavailable — showing 2D cure timeline. (' + e.message + ')'); console.error('3D viewer:', e); showFallback(); }
</script>
<script>
// ---- Cure animation UI (plain script: runs even if the Three.js module ----
// failed to load, so Play cure / timeline / overlay always work).
(function(){
const DATA = window.__NC_DATA__;
const DISEASE = window.__NC_DISEASE__;
const TECHNIQUE = window.__NC_TECHNIQUE__;
const BEFORE_V = window.__NC_BEFORE_V__;
const AFTER_V = window.__NC_AFTER_V__;
const TIMELINE = (DATA && DATA.timeline) || null;
const NFRAMES = (DATA && DATA.lesionFrames && DATA.lesionFrames.length)
  ? DATA.lesionFrames.length
  : ((TIMELINE && TIMELINE.frames && TIMELINE.frames.length) ? TIMELINE.frames.length : 0);
const HAS_CURE = NFRAMES > 1 && BEFORE_V > 0;

document.getElementById('dv').textContent = DISEASE ? 'Disease: ' + DISEASE : '';
document.getElementById('tv').textContent = TECHNIQUE ? 'Technique: ' + TECHNIQUE : '';

// 2D canvas fallback (shown when WebGL is unavailable).
let fbCanvas=null, fbCtx=null, fbReady=false;
window.__ncShowFallback = function(){
  fbCanvas = document.getElementById('fallback');
  if(!fbCanvas) return;
  fbCanvas.style.display='block';
  const dpr=Math.min(window.devicePixelRatio||1, 2);
  fbCanvas.width=(fbCanvas.clientWidth||800)*dpr;
  fbCanvas.height=(fbCanvas.clientHeight||620)*dpr;
  fbCtx = fbCanvas.getContext('2d');
  if(fbCtx) fbCtx.scale(dpr, dpr);
  fbReady = !!fbCtx;
};
function showFallback(){ window.__ncShowFallback && window.__ncShowFallback(); }
function drawFallback(f){
  if(!fbReady || !fbCtx) return;
  const W=fbCanvas.clientWidth||800, H=fbCanvas.clientHeight||620;
  fbCtx.clearRect(0,0,W,H);
  const cx=W*0.5, cy=H*0.48, R=Math.min(W,H)*0.30;
  fbCtx.save();
  fbCtx.fillStyle='#0a0e14'; fbCtx.fillRect(0,0,W,H);
  for(let h=-1;h<=1;h+=2){
    fbCtx.beginPath();
    fbCtx.fillStyle='#3a2a18';
    fbCtx.ellipse(cx+h*R*0.55, cy, R*0.95, R*1.05, h*0.12, 0, Math.PI*2);
    fbCtx.fill();
    fbCtx.strokeStyle='#5a3e22'; fbCtx.lineWidth=2;
    fbCtx.stroke();
    for(let i=0;i<6;i++){
      fbCtx.beginPath(); fbCtx.strokeStyle='rgba(90,62,34,0.5)'; fbCtx.lineWidth=1.5;
      const a=i*Math.PI/6 + (h>0?0.1:-0.1);
      fbCtx.moveTo(cx+h*R*0.55+Math.cos(a)*R*0.2, cy+Math.sin(a)*R*0.2);
      fbCtx.quadraticCurveTo(cx+h*R*0.55+Math.cos(a)*R*0.6, cy+Math.sin(a)*R*0.6, cx+h*R*0.55+Math.cos(a)*R*0.9, cy+Math.sin(a)*R*0.9);
      fbCtx.stroke();
    }
  }
  const fi = frameInfo(f);
  const ls = fi ? (fi.lesion_scale!=null?fi.lesion_scale:1) : 1;
  const ed = fi ? (fi.edema||0) : 1;
  const rg = fi ? (fi.regen||0) : 0;
  const pr = fi ? (fi.protect||0) : 0;
  const phaseColor = fi ? (fi.phase_color||'#ff2b4a') : '#ff2b4a';
  const lx=cx-R*0.3, ly=cy-R*0.1;
  if(ed>0.02){
    fbCtx.beginPath(); fbCtx.fillStyle='rgba(255,106,58,'+(ed*0.32)+')';
    fbCtx.arc(lx, ly, R*0.22*1.5, 0, Math.PI*2); fbCtx.fill();
  }
  if(pr>0.05){
    fbCtx.beginPath(); fbCtx.strokeStyle='rgba(154,122,208,'+(pr*0.6)+')'; fbCtx.lineWidth=1.5; fbCtx.setLineDash([4,4]);
    fbCtx.arc(lx, ly, R*0.22*1.8, 0, Math.PI*2); fbCtx.stroke(); fbCtx.setLineDash([]);
  }
  if(ls>0.02){
    fbCtx.beginPath(); fbCtx.fillStyle=phaseColor;
    fbCtx.globalAlpha=0.9;
    fbCtx.arc(lx, ly, Math.max(2, R*0.22*Math.max(0.15,ls)), 0, Math.PI*2); fbCtx.fill();
    fbCtx.globalAlpha=1;
  }
  if(rg>0.03){
    fbCtx.beginPath(); fbCtx.fillStyle='rgba(63,185,80,'+(rg*0.8)+')';
    fbCtx.arc(lx, ly, Math.max(2, R*0.22*rg), 0, Math.PI*2); fbCtx.fill();
  }
  fbCtx.restore();
}

const tlbar = document.getElementById('tlbar');
const tllabels = document.getElementById('tllabels');
let phases = (TIMELINE && TIMELINE.phases) ? TIMELINE.phases : [];
if(phases.length && tlbar){
  phases.forEach((ph, i) => {
    const seg = document.createElement('div');
    seg.className = 'tlseg';
    seg.style.flex = ph.weight || 1;
    seg.style.background = ph.color || '#3aa6e6';
    seg.dataset.idx = i;
    tlbar.appendChild(seg);
    const lbl = document.createElement('span');
    lbl.textContent = (ph.name||'').split(' ')[0];
    tllabels.appendChild(lbl);
  });
}

let playing=false, frame=0, lastT=0;
const playBtn=document.getElementById('play'), resetBtn=document.getElementById('reset');
const mv=document.getElementById('mv'), pv=document.getElementById('pv');
const mc=document.getElementById('mc'), ds=document.getElementById('ds');

function frameInfo(f){
  if(TIMELINE && TIMELINE.frames && TIMELINE.frames[f]) return TIMELINE.frames[f];
  return null;
}
function curV(f){
  const fi = frameInfo(f);
  if(fi && fi.lesion_volume!=null) return fi.lesion_volume;
  if(!HAS_CURE) return BEFORE_V;
  return BEFORE_V*(1 - f/(NFRAMES-1));
}

function applyFrame(f){
  // 3D mesh updates only when the WebGL module exposed them.
  const RG = window.__nc3d;
  if(RG && RG.lesionGeom && DATA.lesionFrames && DATA.lesionFrames[f]){
    const pos=RG.lesionGeom.getAttribute('position'); const arr=DATA.lesionFrames[f];
    for(let i=0;i<arr.length;i++) pos.array[i]=arr[i]; pos.needsUpdate=true;
    RG.lesionGeom.computeVertexNormals();
  }
  const fi = frameInfo(f);
  if(RG && RG.lesionMesh && fi && fi.phase_color){
    RG.lesionMesh.material.color.set(fi.phase_color);
  }
  if(RG && RG.regenGeom && DATA.regenFrames && DATA.regenFrames[f]){
    const pos=RG.regenGeom.getAttribute('position'); const arr=DATA.regenFrames[f];
    for(let i=0;i<arr.length;i++) pos.array[i]=arr[i]; pos.needsUpdate=true;
    RG.regenGeom.computeVertexNormals();
  }
  if(RG && RG.regenMesh && fi) RG.regenMesh.material.opacity = (fi.regen||0) * 0.75;
  if(RG && RG.edemaMesh && fi) RG.edemaMesh.material.opacity = (fi.edema||0) * 0.32;
  if(RG && RG.protectMesh && fi) RG.protectMesh.material.opacity = (fi.protect||0) * 0.25;
  if(fi){
    pv.textContent = fi.phase_name || '';
    pv.style.color = fi.phase_color || '#f0a040';
    mc.textContent = (fi.mechanism||'').replace(/_/g,' ');
    ds.textContent = fi.description || '';
    const lv = fi.lesion_volume!=null ? fi.lesion_volume : curV(f);
    const pct = (fi.progress!=null ? fi.progress*100 : (f/(NFRAMES-1))*100);
    mv.textContent = 'lesion ' + lv.toFixed(0) + ' mm³ · cure ' + pct.toFixed(0) + '%';
  } else {
    mv.textContent = 'Lesion volume: ' + curV(f).toFixed(0) + ' mm³';
  }
  if(fi){
    const segs = tlbar.querySelectorAll('.tlseg');
    segs.forEach((s, i) => s.classList.toggle('active', i === fi.phase_index));
  }
  drawFallback(f);
}
applyFrame(0);

playBtn.onclick=()=>{ if(!HAS_CURE){ playBtn.textContent='No cure data'; return; } playing=!playing; playBtn.textContent=playing?'⏸ Pause':'▶ Play cure'; if(playing){frame=0; lastT=performance.now();} };
resetBtn.onclick=()=>{ frame=0; playing=false; playBtn.textContent='▶ Play cure'; applyFrame(0); if(window.__nc3d && window.__nc3d.resetCamera) window.__nc3d.resetCamera(); };
function loop(t){ requestAnimationFrame(loop);
  // Show the 2D fallback once it's clear the Three.js module didn't init WebGL
  // (the deferred module has run by the first rAF tick; if ok is still false,
  // WebGL is unavailable, so draw the schematic cure on the fallback canvas).
  if(!fbReady && window.__nc3d && !window.__nc3d.ok){ showFallback(); if(fbReady) applyFrame(frame); }
  if(playing){ if(t-lastT>380){ lastT=t; frame++; if(frame>=NFRAMES){ frame=NFRAMES-1; playing=false; playBtn.textContent='▶ Replay'; } applyFrame(frame); } } }
requestAnimationFrame(loop);
})();
</script>
</body>
</html>
"""


def render_three_brain(
    cortex_mesh: MeshData | None = None,
    tissue_meshes: list[MeshData] | None = None,
    lesion_mesh: MeshData | None = None,
    disease_name: str | None = None,
    technique_name: str | None = None,
    before_volume: float = 0.0,
    after_volume: float = 0.0,
    *,
    cure_timeline: dict | None = None,
    height: int = 640,
    decimate_cortex_to: int = 20000,
) -> None:
    """Render the interactive WebGL brain (Three.js) via a Streamlit component.

    Parameters
    ----------
    cortex_mesh
        Real fsaverage pial surface (split into hemispheres at the midline).
    tissue_meshes
        Segmented tissue meshes (gray/white matter, csf) as a backdrop.
    lesion_mesh
        Lesion mesh; if a cure animation is requested this shrinks over frames.
    disease_name / technique_name
        Shown on the on-canvas overlay during the cure animation.
    before_volume / after_volume
        Lesion volume before/after the cure; drives the animation.
    cure_timeline
        Optional :class:`~brainframe.therapy.cure_phases.CureTimeline` dict
        (``to_dict()`` output) describing the multi-phase biological cure
        cascade. When provided, the lesion is coloured per phase, a
        regeneration mesh grows into the cavity, an oedema halo fades, and a
        neuroprotective shield appears — all driven by the phase parameters.
    """
    data: dict[str, Any] = {}

    # Split cortex into left/right hemispheres at the midline for tinting.
    if cortex_mesh is not None and len(cortex_mesh.vertices) > 0:
        verts = np.asarray(cortex_mesh.vertices, dtype=np.float32)
        faces = np.asarray(cortex_mesh.faces, dtype=np.int32)
        verts = _recenter(verts)
        # fsaverage is in MNI mm: left hemisphere = negative x.
        left_mask = verts[:, 0] < 0
        right_mask = ~left_mask
        from brainframe.data.real_brain import _compute_normals

        for name, mask in (("cortexLeft", left_mask), ("cortexRight", right_mask)):
            idx = np.where(mask)[0]
            if len(idx) < 3:
                continue
            remap = {old: new for new, old in enumerate(idx)}
            sub_v = verts[idx]
            # Keep faces fully within this hemisphere.
            keep = np.all(np.isin(faces, idx), axis=1)
            sub_f = np.array([[remap[v] for v in tri] for tri in faces[keep]], dtype=np.int32)
            if len(sub_f) == 0:
                continue
            sub_n = _compute_normals(sub_v, sub_f)
            data[name] = {
                "positions": sub_v.astype(np.float32).flatten().tolist(),
                "indices": sub_f.flatten().tolist(),
                "normals": sub_n.astype(np.float32).flatten().tolist(),
            }

    tissues = []
    if tissue_meshes:
        tissue_palette = {
            "gray_matter": (0x9A7AD0, 0.85),
            "white_matter": (0xF0E6D2, 0.9),
            "csf": (0x3A5A8A, 0.55),
            "lesion": (0xFF2B4A, 0.95),
        }
        for m in tissue_meshes:
            if m is None or len(m.vertices) == 0:
                continue
            color, opacity = tissue_palette.get(m.label, (0xAAAAAA, 0.8))
            tissues.append(
                {
                    "mesh": _mesh_to_json(m, decimate_to=4000),
                    "color": color,
                    "opacity": opacity,
                    "flat": False,
                }
            )
    if tissues:
        data["tissues"] = tissues

    # Lesion mesh + cure animation frames.
    lesion_json: dict | None = None
    lesion_frames: list | None = None
    regen_frames: list | None = None
    if lesion_mesh is not None and len(lesion_mesh.vertices) > 0:
        lverts = np.asarray(lesion_mesh.vertices, dtype=np.float32).copy()
        lverts = _recenter(lverts)
        lfaces = np.asarray(lesion_mesh.faces, dtype=np.int32)
        from brainframe.data.real_brain import _compute_normals

        lnorms = _compute_normals(lverts, lfaces)
        lesion_json = {
            "positions": lverts.astype(np.float32).flatten().tolist(),
            "indices": lfaces.flatten().tolist(),
            "normals": lnorms.astype(np.float32).flatten().tolist(),
        }
        centroid = lverts.mean(axis=0)
        has_cure = before_volume > after_volume and before_volume > 0
        if cure_timeline is not None and cure_timeline.get("frames"):
            tl_frames = cure_timeline["frames"]
            scales = [f["lesion_scale"] for f in tl_frames]
            regen = [f["regen"] for f in tl_frames]
            lesion_frames = _phase_lesion_frames(lverts, centroid, scales)
            regen_frames = _regen_frames(lverts, centroid, regen)
            data["timeline"] = cure_timeline
        elif has_cure:
            lesion_frames = _shrink_frames(lverts, centroid, n_frames=9)
    if lesion_json:
        data["lesion"] = lesion_json
    if lesion_frames:
        data["lesionFrames"] = lesion_frames
    if regen_frames:
        data["regenFrames"] = regen_frames
    # Always attach the cure timeline (even without a lesion mesh) so the
    # 2D fallback + overlay can drive the multi-phase animation.
    if cure_timeline is not None:
        data["timeline"] = cure_timeline

    html = (
        _THREE_TEMPLATE.replace("__DATA__", json.dumps(data))
        .replace("__DISEASE__", json.dumps(disease_name))
        .replace("__TECHNIQUE__", json.dumps(technique_name))
        .replace("__BEFORE_V__", str(float(before_volume)))
        .replace("__AFTER_V__", str(float(after_volume)))
    )
    components.html(html, height=height, scrolling=False)
