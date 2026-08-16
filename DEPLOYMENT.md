# NeuroCure — Cloud Deployment Guide

NeuroCure runs as a **Streamlit** app. This guide covers deploying it to
**Streamlit Community Cloud** (easiest, free, always-on) or **Render**
(more control, custom domain).

---

## Prerequisites

The following files are already in the repo root and configured for cloud:

| File | Purpose |
|------|---------|
| `requirements.txt` | All Python dependencies (CPU-only PyTorch) |
| `.streamlit/config.toml` | Streamlit server + dark theme settings |
| `.python-version` | Pins Python 3.11 for the cloud runtime |
| `Procfile` | Start command for Render / Heroku |
| `render.yaml` | One-click Render Blueprint |

### Key app entry point

```
neurocure_app/app.py
```

When a platform asks for the "main file path", use `neurocure_app/app.py`.

---

## Option 1 — Streamlit Community Cloud (recommended, free)

Streamlit Cloud is the simplest path: no build commands, no Docker, automatic
redeploy on every git push.

### Steps

1. **Push to GitHub** — ensure all files are committed to the `main` branch
   of `Harishwaran18/brain-diseases-calssification`.

2. **Sign in** — go to [share.streamlit.io](https://share.streamlit.io) and
   authenticate with your GitHub account.

3. **Create a new app** — click **New app** → **From existing repo**.

4. **Configure**:
   - **Repository:** `Harishwaran18/brain-diseases-calssification`
   - **Branch:** `main`
   - **Main file path:** `neurocure_app/app.py`
   - **Python version:** 3.11
   - **Requirements file:** `requirements.txt` (auto-detected)

5. **Deploy** — click **Deploy**. The first build takes 5–10 minutes
   (downloading PyTorch + nilearn datasets).

6. **First launch note:** The trained MLP weights (`assets/models/disease_mlp.pt`)
   are not in git (too large). On the first prediction, NeuroCure auto-trains
   the model in ~60 seconds from the transparent signature-derived dataset.
   This happens once; the weights are cached in the container's ephemeral
   filesystem for the session.

### Streamlit Cloud limitations

- **Ephemeral filesystem**: files written at runtime (uploads, model weights,
  outputs) are lost when the container sleeps/restarts. This is fine for
  demos — the model retrains automatically.
- **1 GB RAM, 2 vCPU**: sufficient for CPU inference + 3D rendering.
- **Public URL**: `https://<app-name>.streamlit.app` — shareable with anyone.

---

## Option 2 — Render (more control)

Render gives you a persistent web service with configurable resources.

### One-click Blueprint

1. Go to [render.com](https://render.com) and sign in with GitHub.
2. Click **New** → **Blueprint**.
3. Select the `Harishwaran18/brain-diseases-calssification` repo.
4. Render auto-detects `render.yaml` and creates the web service.
5. Click **Apply**.

### Manual setup

1. Go to [render.com](https://render.com) → **New** → **Web Service**.
2. Connect the GitHub repo.
3. Configure:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run neurocure_app/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
   - **Plan:** Free (sleeps after 15 min idle) or Starter ($7/mo, always-on)

### Render tips

- To keep the model weights persistent across restarts, attach a
  **Disk** (persistent volume) and set the environment variable
  `NEUROCURE_MODEL_DIR` to the disk mount path, then update
  `trained_model.py` to read from it.
- Render's free tier sleeps after 15 minutes of inactivity. The first request
  after sleep takes ~30 seconds to wake. Upgrade to **Starter** for 24/7 uptime.

---

## Environment Variables

NeuroCure does not require any secret keys or API tokens. All data is
processed locally in the container — no external API calls are made.

If you later add features that need secrets (e.g., a cloud MRI storage API),
add them in:
- **Streamlit Cloud:** App settings → Secrets (stored as a `.toml` file)
- **Render:** Environment tab in the dashboard

---

## Asset Paths

All asset paths in the codebase are resolved relative to the package location
using `Path(__file__).resolve()`, so they work regardless of the working
directory the cloud platform starts the process from:

```
brainframe/data/real_brain.py     →  assets/real_brain/*.npz
brainframe/classification/trained_model.py  →  assets/models/disease_mlp.pt
neurocure_app/state.py            →  configs/default.yaml  (via REPO_ROOT)
```

The bundled assets (`fsaverage_pial.npz`, `icbm152_volume.npz`,
`deep_nuclei.npz`) are committed to git and ship with the deployment.

---

## Verifying the Deployment

After deployment, verify:

1. The homepage loads with the NeuroCure title and two-column dashboard.
2. Click **1. Upload / Demo brain** → **Load demo brain** — should load the
   ICBM152 volume in a few seconds.
3. Click **2. 3D Brain** → **Run segmentation + reconstruction** — the WebGL
   viewer should render the cortex + deep nuclei + lesion.
4. Click **3. Disease Prediction** → **Run prediction** — first run trains
   the MLP (~60s), then shows the 36-disease classification.
5. Click **5. Live Cure Simulation** → **▶ Play cure** — the animated
   cure cascade should play in the WebGL viewer.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: torch` | Ensure `requirements.txt` is in repo root; Streamlit Cloud auto-installs it. |
| App loads but 3D viewer is blank | The viewer needs WebGL — try Chrome/Firefox; some corporate proxies block WebSocket. |
| First prediction is slow (~60s) | Normal — the MLP auto-trains on first use. Subsequent predictions are instant. |
| `nilearn` download fails | The atlas data is pre-bundled in `assets/real_brain/deep_nuclei.npz`; nilearn should only need to fetch if you re-run the mesh build script. |
| Out of memory | Streamlit Cloud gives 1 GB; if the demo brain + model exceed this, reduce `decimate_cortex_to` in `three_viewer.py`. |
