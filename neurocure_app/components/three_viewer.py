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


_THREE_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<style>
  html,body{margin:0;padding:0;height:100%;background:#0a0e14;overflow:hidden;font-family:system-ui,sans-serif}
  #app{position:relative;width:100%;height:100%}
  #overlay{position:absolute;top:14px;left:14px;color:#e6edf3;background:rgba(13,17,23,.78);
           border:1px solid #30363d;border-radius:10px;padding:12px 16px;max-width:60%;
           font-size:13px;line-height:1.55;pointer-events:none;z-index:5}
  #overlay .disease{color:#3aa6e6;font-weight:700;font-size:15px}
  #overlay .technique{color:#3fb950;font-weight:600}
  #overlay .meta{color:#8b949e;font-size:12px;margin-top:6px}
  #hud{position:absolute;top:14px;right:14px;color:#8b949e;font-size:11px;text-align:right;z-index:5}
  #err{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:#ff7b72;font-size:14px;display:none;z-index:6}
  #ctrl{position:absolute;bottom:14px;left:50%;transform:translateX(-50%);display:flex;gap:8px;z-index:5}
  #ctrl button{background:#161b22;color:#e6edf3;border:1px solid #30363d;border-radius:8px;padding:8px 18px;font-size:13px;cursor:pointer}
  #ctrl button:hover{border-color:#3aa6e6}
</style>
</head>
<body>
<div id="app">
  <div id="overlay">
    <div class="disease" id="dv"></div>
    <div class="technique" id="tv"></div>
    <div class="meta" id="mv"></div>
  </div>
  <div id="hud">drag: rotate &middot; scroll: zoom &middot; right-drag: pan</div>
  <div id="err"></div>
  <div id="ctrl"><button id="play">▶ Play cure</button><button id="reset">↺ Reset view</button></div>
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
  // Lesion (animated if cure frames present).
  let lesionMesh=null, lesionGeom=null, lesionBasePositions=null;
  if(DATA.lesion && DATA.lesion.positions && DATA.lesion.positions.length){
    lesionGeom = new THREE.BufferGeometry();
    lesionGeom.setAttribute('position', new THREE.Float32BufferAttribute(DATA.lesion.positions, 3));
    if(DATA.lesion.normals) lesionGeom.setAttribute('normal', new THREE.Float32BufferAttribute(DATA.lesion.normals, 3));
    else lesionGeom.computeVertexNormals();
    lesionGeom.setIndex(DATA.lesion.indices);
    const lmat = new THREE.MeshStandardMaterial({color:0xff2b4a, metalness:0.1, roughness:0.4, flatShading:true, transparent:true, opacity:0.95, emissive:0x550011, emissiveIntensity:0.3});
    lesionMesh = new THREE.Mesh(lesionGeom, lmat);
    scene.add(lesionMesh);
    lesionBasePositions = new THREE.Float32BufferAttribute(DATA.lesion.positions, 3).array;
  }

  // Frame the camera on the whole scene.
  const box = new THREE.Box3().setFromObject(scene);
  if(!box.isEmpty()){ const c=box.getCenter(new THREE.Vector3()), s=box.getSize(new THREE.Vector3());
    const maxDim=Math.max(s.x,s.y,s.z); camera.position.set(c.x+maxDim*0.9, c.y+maxDim*0.7, c.z+maxDim*1.3); controls.target.copy(c);
  }

  function animate(){ requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }
  animate();
  window.addEventListener('resize', ()=>{ const W=app.clientWidth, H=app.clientHeight; camera.aspect=W/H; camera.updateProjectionMatrix(); renderer.setSize(W,H); });

  // ---- Cure animation ----
  let playing=false, frame=0, lastT=0;
  const playBtn=document.getElementById('play'), resetBtn=document.getElementById('reset');
  const mv=document.getElementById('mv');
  function curV(f){ if(!HAS_CURE) return BEFORE_V; const frac=f/(NFRAMES-1); return BEFORE_V*(1-frac); }
  function updateMeta(f){
    if(!HAS_CURE){ mv.textContent='Lesion volume: '+BEFORE_V.toFixed(0)+' mm³'; return; }
    const frac=f/(NFRAMES-1); const label = f===0 ? 'Before medicine' : (f===NFRAMES-1 && AFTER_V<BEFORE_V ? 'Cured' : 'Cure '+(frac*100).toFixed(0)+'%');
    mv.textContent = label+' · lesion '+curV(f).toFixed(0)+' mm³';
  }
  function applyFrame(f){
    if(lesionGeom && DATA.lesionFrames && DATA.lesionFrames[f]){
      const pos=lesionGeom.getAttribute('position'); const arr=DATA.lesionFrames[f];
      for(let i=0;i<arr.length;i++) pos.array[i]=arr[i]; pos.needsUpdate=true;
      lesionGeom.computeVertexNormals();
    }
    updateMeta(f);
  }
  updateMeta(0);
  playBtn.onclick=()=>{ if(!HAS_CURE){ playBtn.textContent='No cure data'; return; } playing=!playing; playBtn.textContent=playing?'⏸ Pause':'▶ Play cure'; if(playing){frame=0; lastT=performance.now();} };
  resetBtn.onclick=()=>{ frame=0; playing=false; playBtn.textContent='▶ Play cure'; applyFrame(0);
    if(!box.isEmpty()){ const c=box.getCenter(new THREE.Vector3()); controls.target.copy(c); }
  };
  function loop(t){ requestAnimationFrame(loop); if(playing){ if(t-lastT>450){ lastT=t; frame++; if(frame>=NFRAMES){ frame=0; } applyFrame(frame); } } }
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
        if before_volume > after_volume and before_volume > 0:
            centroid = lverts.mean(axis=0)
            lesion_frames = _shrink_frames(lverts, centroid, n_frames=9)
    if lesion_json:
        data["lesion"] = lesion_json
    if lesion_frames:
        data["lesionFrames"] = lesion_frames

    html = (
        _THREE_TEMPLATE.replace("__DATA__", json.dumps(data))
        .replace("__DISEASE__", json.dumps(disease_name))
        .replace("__TECHNIQUE__", json.dumps(technique_name))
        .replace("__BEFORE_V__", str(float(before_volume)))
        .replace("__AFTER_V__", str(float(after_volume)))
    )
    components.html(html, height=height, scrolling=False)
