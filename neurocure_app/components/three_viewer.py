"""Three.js WebGL brain viewer for the NeuroCure platform.

Replaces the Plotly 3D renderer with a genuine WebGL scene built on Three.js
(the international standard for in-browser 3D visualisation, used by
BrainBrowser and the Allen Brain Atlas). The full-resolution fsaverage pial
cortex is rendered with PBR (physically-based) materials, multi-light shading,
and per-vertex normals so the real gyri/sulci are crisp and anatomically
lifelike. OrbitControls let the user rotate/zoom/pan the brain freely.

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
  #err{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:#ff7b72;font-size:14px;display:none;z-index:6}
  #ctrl{position:absolute;bottom:60px;left:50%;transform:translateX(-50%);display:flex;gap:8px;z-index:5}
  #ctrl button{background:#161b22;color:#e6edf3;border:1px solid #30363d;border-radius:8px;padding:8px 18px;font-size:13px;cursor:pointer}
  #ctrl button:hover{border-color:#3aa6e6}
  #timeline{position:absolute;bottom:14px;left:50%;transform:translateX(-50%);width:78%;z-index:5}
  #timeline .tlbar{height:10px;background:#161b22;border:1px solid #30363d;border-radius:6px;overflow:hidden;display:flex}
  #timeline .tlseg{height:100%;opacity:.55;transition:opacity .2s}
  #timeline .tlseg.active{opacity:1}
  #timeline .tlhead{display:flex;justify-content:space-between;color:#8b949e;font-size:10px;margin-bottom:3px}
  #timeline .tllabels{display:flex;color:#8b949e;font-size:9px;margin-top:3px;gap:2px}
  #timeline .tllabels span{flex:1;text-align:center;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
</style>
</head>
<body>
<div id="app">
  <div id="overlay">
    <div class="disease" id="dv"></div>
    <div class="technique" id="tv"></div>
    <div class="phase" id="pv"></div>
    <div class="mechanism" id="mc"></div>
    <div class="desc" id="ds"></div>
    <div class="meta" id="mv"></div>
  </div>
  <div id="hud">drag: rotate &middot; scroll: zoom &middot; right-drag: pan</div>
  <div id="err"></div>
  <div id="ctrl"><button id="play">▶ Play cure</button><button id="reset">↺ Reset view</button></div>
  <div id="timeline">
    <div class="tlhead"><span id="tlleft">Before</span><span id="tlright">Cured</span></div>
    <div class="tlbar" id="tlbar"></div>
    <div class="tllabels" id="tllabels"></div>
  </div>
</div>
<script type="importmap">
{ "imports": {
  "three": "https://cdn.jsdelivr.net/npm/three@0.169.0/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.169.0/examples/jsm/"
}}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const DATA = __DATA__;
const DISEASE = __DISEASE__;
const TECHNIQUE = __TECHNIQUE__;
const BEFORE_V = __BEFORE_V__;
const AFTER_V = __AFTER_V__;
const TIMELINE = DATA.timeline || null;
const NFRAMES = DATA.lesionFrames ? DATA.lesionFrames.length : 0;
const HAS_CURE = NFRAMES > 1 && BEFORE_V > 0;

document.getElementById('dv').textContent = DISEASE ? 'Disease: ' + DISEASE : '';
document.getElementById('tv').textContent = TECHNIQUE ? 'Technique: ' + TECHNIQUE : '';

const app = document.getElementById('app');
const err = document.getElementById('err');
function fail(msg){ err.style.display='block'; err.textContent='3D viewer error: '+msg; }

try {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0e14);

  const w = app.clientWidth || 800, h = app.clientHeight || 620;
  const camera = new THREE.PerspectiveCamera(45, w/h, 0.1, 2000);
  const renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(w, h);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  app.appendChild(renderer.domElement);

  // Multi-light setup: ambient + key + rim + fill for depth cueing of gyri/sulci.
  scene.add(new THREE.AmbientLight(0xffffff, 0.45));
  const key = new THREE.DirectionalLight(0xfff2e0, 1.15); key.position.set(120, 200, 180); scene.add(key);
  const rim = new THREE.DirectionalLight(0xb8d4ff, 0.6);  rim.position.set(-160, -80, -140); scene.add(rim);
  const fill = new THREE.PointLight(0xffe0b0, 0.35, 600); fill.position.set(0, -150, 120); scene.add(fill);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.08;
  controls.minDistance = 40; controls.maxDistance = 600;

  function makeMesh(m, color, opacity, roughness, metalness, flat){
    if(!m || !m.positions || m.positions.length===0) return null;
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.Float32BufferAttribute(m.positions, 3));
    if(m.normals && m.normals.length===m.positions.length) geom.setAttribute('normal', new THREE.Float32BufferAttribute(m.normals, 3));
    else geom.computeVertexNormals();
    geom.setIndex(m.indices);
    const mat = new THREE.MeshStandardMaterial({color, metalness, roughness, flatShading: !!flat, transparent: opacity<1.0, opacity, side: THREE.DoubleSide});
    const mesh = new THREE.Mesh(geom, mat);
    scene.add(mesh);
    return mesh;
  }

  // Cortex (left/right hemispheres subtly tinted) — the realistic brain backdrop.
  let leftMesh=null, rightMesh=null;
  if(DATA.cortexLeft)  leftMesh  = makeMesh(DATA.cortexLeft,  0xc98a4b, 1.0, 0.55, 0.05, false);
  if(DATA.cortexRight) rightMesh = makeMesh(DATA.cortexRight, 0xc0793a, 1.0, 0.55, 0.05, false);
  // Tissue meshes.
  if(DATA.tissues) for(const t of DATA.tissues) makeMesh(t.mesh, t.color, t.opacity, 0.7, 0.0, t.flat);

  // ---- Lesion mesh (phase-coloured, shrinks per cure phase) ----
  let lesionMesh=null, lesionGeom=null;
  if(DATA.lesion && DATA.lesion.positions && DATA.lesion.positions.length){
    lesionGeom = new THREE.BufferGeometry();
    lesionGeom.setAttribute('position', new THREE.Float32BufferAttribute(DATA.lesion.positions, 3));
    if(DATA.lesion.normals) lesionGeom.setAttribute('normal', new THREE.Float32BufferAttribute(DATA.lesion.normals, 3));
    else lesionGeom.computeVertexNormals();
    lesionGeom.setIndex(DATA.lesion.indices);
    const lmat = new THREE.MeshStandardMaterial({color:0xff2b4a, metalness:0.1, roughness:0.4, flatShading:true, transparent:true, opacity:0.95, emissive:0x550011, emissiveIntensity:0.35});
    lesionMesh = new THREE.Mesh(lesionGeom, lmat);
    scene.add(lesionMesh);
  }

  // ---- Regeneration mesh (healthy tissue regrowing into the cavity) ----
  let regenMesh=null, regenGeom=null;
  if(DATA.regenFrames && DATA.regenFrames.length && DATA.lesion && DATA.lesion.indices){
    regenGeom = new THREE.BufferGeometry();
    regenGeom.setAttribute('position', new THREE.Float32BufferAttribute(DATA.regenFrames[0], 3));
    regenGeom.setIndex(DATA.lesion.indices);
    regenGeom.computeVertexNormals();
    const rmat = new THREE.MeshStandardMaterial({color:0x3fb950, metalness:0.05, roughness:0.5, flatShading:false, transparent:true, opacity:0.0, emissive:0x0a3318, emissiveIntensity:0.4});
    regenMesh = new THREE.Mesh(regenGeom, rmat);
    scene.add(regenMesh);
  }

  // ---- Oedema halo (inflammation sphere around the lesion) ----
  let edemaMesh=null;
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
    edemaMesh = new THREE.Mesh(egeom, emat);
    scene.add(edemaMesh);
  }

  // ---- Neuroprotective field (translucent shield around penumbra) ----
  let protectMesh=null;
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
    protectMesh = new THREE.Mesh(pgeom, pmat);
    scene.add(protectMesh);
  }

  // Frame the camera on the whole scene.
  const box = new THREE.Box3().setFromObject(scene);
  if(!box.isEmpty()){ const c=box.getCenter(new THREE.Vector3()), s=box.getSize(new THREE.Vector3());
    const maxDim=Math.max(s.x,s.y,s.z); camera.position.set(c.x+maxDim*0.9, c.y+maxDim*0.7, c.z+maxDim*1.3); controls.target.copy(c);
  }

  function animate(){ requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }
  animate();
  window.addEventListener('resize', ()=>{ const W=app.clientWidth, H=app.clientHeight; camera.aspect=W/H; camera.updateProjectionMatrix(); renderer.setSize(W,H); });

  // ---- Build the phase timeline bar ----
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

  // ---- Cure animation (multi-phase) ----
  let playing=false, frame=0, lastT=0;
  const playBtn=document.getElementById('play'), resetBtn=document.getElementById('reset');
  const mv=document.getElementById('mv'), pv=document.getElementById('pv');
  const mc=document.getElementById('mechanism'), ds=document.getElementById('desc');

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
    // Lesion mesh positions.
    if(lesionGeom && DATA.lesionFrames && DATA.lesionFrames[f]){
      const pos=lesionGeom.getAttribute('position'); const arr=DATA.lesionFrames[f];
      for(let i=0;i<arr.length;i++) pos.array[i]=arr[i]; pos.needsUpdate=true;
      lesionGeom.computeVertexNormals();
    }
    const fi = frameInfo(f);
    // Phase-coloured lesion.
    if(lesionMesh && fi && fi.phase_color){
      lesionMesh.material.color.set(fi.phase_color);
    }
    // Regeneration mesh grows + fades in.
    if(regenGeom && DATA.regenFrames && DATA.regenFrames[f]){
      const pos=regenGeom.getAttribute('position'); const arr=DATA.regenFrames[f];
      for(let i=0;i<arr.length;i++) pos.array[i]=arr[i]; pos.needsUpdate=true;
      regenGeom.computeVertexNormals();
    }
    if(regenMesh && fi){
      regenMesh.material.opacity = (fi.regen||0) * 0.75;
    }
    // Oedema halo fades with inflammation.
    if(edemaMesh && fi){
      edemaMesh.material.opacity = (fi.edema||0) * 0.32;
    }
    // Neuroprotective field.
    if(protectMesh && fi){
      protectMesh.material.opacity = (fi.protect||0) * 0.25;
    }
    // Overlay text.
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
    // Highlight active timeline segment.
    if(fi){
      const segs = tlbar.querySelectorAll('.tlseg');
      segs.forEach((s, i) => s.classList.toggle('active', i === fi.phase_index));
    }
  }
  applyFrame(0);

  playBtn.onclick=()=>{ if(!HAS_CURE){ playBtn.textContent='No cure data'; return; } playing=!playing; playBtn.textContent=playing?'⏸ Pause':'▶ Play cure'; if(playing){frame=0; lastT=performance.now();} };
  resetBtn.onclick=()=>{ frame=0; playing=false; playBtn.textContent='▶ Play cure'; applyFrame(0);
    if(!box.isEmpty()){ const c=box.getCenter(new THREE.Vector3()); controls.target.copy(c); }
  };
  function loop(t){ requestAnimationFrame(loop); if(playing){ if(t-lastT>380){ lastT=t; frame++; if(frame>=NFRAMES){ frame=NFRAMES-1; playing=false; playBtn.textContent='▶ Replay'; } applyFrame(frame); } } }
  requestAnimationFrame(loop);
} catch(e){ fail(e.message); console.error(e); }
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

    html = (
        _THREE_TEMPLATE.replace("__DATA__", json.dumps(data))
        .replace("__DISEASE__", json.dumps(disease_name))
        .replace("__TECHNIQUE__", json.dumps(technique_name))
        .replace("__BEFORE_V__", str(float(before_volume)))
        .replace("__AFTER_V__", str(float(after_volume)))
    )
    components.html(html, height=height, scrolling=False)
