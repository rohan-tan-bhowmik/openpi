"""Interactive KNN viewer — Flask server.

Loads latents + LeRobot dataset once, then serves a web UI where you can
pick an episode/frame, choose K, and see nearest neighbors with images.

Run:
  uv run python examples/robomimic/knn_server.py

View locally via SSH tunnel:
  ssh -L 7860:localhost:7860 <user>@<cluster-host>
  then open http://localhost:7860 in your browser
"""
from __future__ import annotations

import base64
import io
import os

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
from flask import Flask, jsonify, render_template_string, request
from PIL import Image

# ── config ────────────────────────────────────────────────────────────────────
NPZ_PATH = "examples/robomimic/data/latents_robomimic_2_19999.npz"
HDF5_PATH = "/iris/u/rbhowmik/cache/robomimic_datasets/square/ph/image_v15.hdf5"
PORT = 7860

# ── load data once ────────────────────────────────────────────────────────────
print(f"Loading {NPZ_PATH} ...")
_data = np.load(NPZ_PATH)
LATENTS: np.ndarray = _data["latents"].astype(np.float32)          # (N, d)
EPISODE_INDICES: np.ndarray = _data["episode_indices"]              # (N,)
FRAME_INDICES: np.ndarray = _data["frame_indices"]                  # (N,)
NPZ_IMAGES: np.ndarray | None = _data["images"] if "images" in _data else None
NPZ_WRIST: np.ndarray | None = _data["wrist_images"] if "wrist_images" in _data else None

# Pre-normalize latents for fast cosine similarity
_norms = np.linalg.norm(LATENTS, axis=1, keepdims=True)
LATENTS_NORMED: np.ndarray = LATENTS / np.maximum(_norms, 1e-8)

N = len(LATENTS)
EPISODES = sorted(int(e) for e in np.unique(EPISODE_INDICES))
print(f"  N={N}  d={LATENTS.shape[1]}  episodes={len(EPISODES)}")

import h5py
print(f"Opening HDF5 {HDF5_PATH} ...")
_hdf5 = h5py.File(HDF5_PATH, "r")
# detect wrist key once
_sample_demo = _hdf5["data/demo_0"]["obs"]
_WRIST_KEY = next((k for k in _sample_demo.keys() if "eye_in_hand" in k or "wrist" in k), None)
print(f"  agentview key: agentview_image  wrist key: {_WRIST_KEY}  Done.")

# ── helpers ───────────────────────────────────────────────────────────────────

def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)
    if arr.dtype != np.uint8:
        arr = (np.asarray(arr, np.float32) * 255).clip(0, 255).astype(np.uint8)
    return arr


def _img_to_b64(arr: np.ndarray, size: int = 168) -> str:
    img = Image.fromarray(arr).resize((size, size), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _get_images(global_idx: int) -> tuple[str | None, str | None]:
    ag = NPZ_IMAGES[global_idx] if NPZ_IMAGES is not None else None
    wr = NPZ_WRIST[global_idx] if NPZ_WRIST is not None else None
    if ag is None or wr is None:
        try:
            ep = int(EPISODE_INDICES[global_idx])
            fr = int(FRAME_INDICES[global_idx])
            demo = _hdf5[f"data/demo_{ep}"]["obs"]
            n_frames = demo["agentview_image"].shape[0]
            fr_clamped = min(fr, n_frames - 1)
            if ag is None:
                ag = _to_uint8(np.array(demo["agentview_image"][fr_clamped]))
            if wr is None and _WRIST_KEY:
                wr = _to_uint8(np.array(demo[_WRIST_KEY][fr_clamped]))
        except Exception:
            pass
    return (
        _img_to_b64(ag) if ag is not None else None,
        _img_to_b64(wr) if wr is not None else None,
    )


def _find_knn(query_idx: int, k: int):
    q = LATENTS_NORMED[query_idx]
    sims = LATENTS_NORMED @ q          # (N,)
    sims[query_idx] = -2.0
    top_k = np.argsort(sims)[::-1][:k]
    q_raw = LATENTS[query_idx]
    l2s = np.linalg.norm(LATENTS[top_k] - q_raw, axis=1)
    return top_k, sims[top_k], l2s


def _frames_for_episode(ep: int) -> list[int]:
    mask = EPISODE_INDICES == ep
    return sorted(int(f) for f in FRAME_INDICES[mask])

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>KNN Viewer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #111; color: #eee; display: flex; height: 100vh; overflow: hidden; }

  /* ── sidebar ── */
  #sidebar {
    width: 220px; min-width: 220px; background: #1a1a1a; padding: 16px 14px;
    display: flex; flex-direction: column; gap: 14px; overflow-y: auto;
    border-right: 1px solid #333;
  }
  #sidebar h2 { font-size: 15px; color: #f5a623; margin-bottom: 2px; }
  label { font-size: 12px; color: #aaa; display: block; margin-bottom: 3px; }
  select, input[type=number], input[type=range] { width: 100%; background: #252525; border: 1px solid #444; color: #eee; padding: 5px 7px; border-radius: 5px; font-size: 13px; }
  input[type=range] { padding: 0; cursor: pointer; accent-color: #4a90d9; }
  .k-row { display: flex; align-items: center; gap: 8px; }
  .k-row span { font-size: 13px; min-width: 20px; text-align: right; }
  button { width: 100%; padding: 8px; border: none; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 600; }
  #btn-query  { background: #4a90d9; color: #fff; }
  #btn-random { background: #333; color: #ccc; border: 1px solid #555; }
  button:hover { opacity: 0.85; }
  #status { font-size: 11px; color: #888; min-height: 16px; }

  /* ── main ── */
  #main { flex: 1; overflow-y: auto; padding: 16px 20px; }
  #main h3 { font-size: 13px; color: #888; margin-bottom: 14px; }

  /* ── cards ── */
  .card {
    display: flex; align-items: stretch; gap: 0;
    border-radius: 8px; overflow: hidden; margin-bottom: 10px;
    border: 2px solid #444;
  }
  .card.query      { border-color: #f5a623; }
  .card.neighbor   { border-color: #4a90d9; }
  .card.cross-ep   { border-color: #2ecc71; }
  .cross-tag { display: inline-block; font-size: 9px; font-weight: 700; color: #2ecc71; background: #0d2b1a; border: 1px solid #2ecc71; border-radius: 3px; padding: 1px 4px; margin-left: 5px; vertical-align: middle; letter-spacing: 0.03em; }

  .card-meta {
    width: 180px; min-width: 180px; padding: 10px 12px;
    background: #1e1e1e; display: flex; flex-direction: column; justify-content: center; gap: 4px;
  }
  .card.query    .card-meta { background: #211a09; }
  .card.neighbor .card-meta { background: #0d1520; }
  .card.cross-ep .card-meta { background: #0b1f12; }

  .card-meta .badge {
    font-size: 13px; font-weight: 700; margin-bottom: 4px;
  }
  .card.query    .badge { color: #f5a623; }
  .card.neighbor .badge { color: #4a90d9; }
  .card.cross-ep .badge { color: #2ecc71; }
  .card-meta .row { font-size: 11px; color: #aaa; font-family: monospace; line-height: 1.6; }
  .card-meta .row span { color: #ddd; }

  .img-pair { display: flex; gap: 2px; }
  .img-pair img { width: 168px; height: 168px; object-fit: cover; display: block; }
  .img-label { font-size: 10px; color: #666; text-align: center; padding: 2px 0; background: #111; }

  .img-slot { display: flex; flex-direction: column; }
  .img-missing { width: 168px; height: 168px; background: #1a1a1a; display: flex; align-items: center; justify-content: center; color: #555; font-size: 11px; }

  #placeholder { color: #555; font-size: 14px; margin-top: 60px; text-align: center; }
</style>
</head>
<body>

<div id="sidebar">
  <h2>KNN Viewer</h2>

  <div>
    <label>Episode</label>
    <select id="ep-sel">
      {% for ep in episodes %}
      <option value="{{ ep }}">{{ ep }}</option>
      {% endfor %}
    </select>
  </div>

  <div>
    <label>Frame</label>
    <input type="number" id="fr-inp" min="0" value="0">
  </div>

  <div>
    <label>K neighbors</label>
    <div class="k-row">
      <input type="range" id="k-slider" min="1" max="20" value="8" oninput="document.getElementById('k-val').textContent=this.value">
      <span id="k-val">8</span>
    </div>
  </div>

  <button id="btn-query" onclick="query()">Show neighbors</button>
  <button id="btn-random" onclick="random_query()">Random frame</button>

  <div id="status"></div>
  <div style="font-size:10px;color:#555;margin-top:auto">
    N={{ n }} frames<br>{{ n_ep }} episodes
  </div>
</div>

<div id="main">
  <div id="results">
    <div id="placeholder">Select an episode + frame and click "Show neighbors"</div>
  </div>
</div>

<script>
const epSel  = document.getElementById('ep-sel');
const frInp  = document.getElementById('fr-inp');
const kSlider = document.getElementById('k-slider');
const status = document.getElementById('status');

async function query(ep=null, fr=null) {
  const episode = ep ?? parseInt(epSel.value);
  const frame   = fr ?? parseInt(frInp.value);
  const k       = parseInt(kSlider.value);
  if (ep !== null) { epSel.value = ep; frInp.value = fr; }

  status.textContent = 'Searching…';
  const res = await fetch(`/query?episode=${episode}&frame=${frame}&k=${k}`);
  if (!res.ok) { status.textContent = await res.text(); return; }
  const data = await res.json();
  status.textContent = `global idx=${data.query.global_idx}`;
  render(data);
}

async function random_query() {
  status.textContent = 'Picking random frame…';
  const res = await fetch('/random');
  const {episode, frame} = await res.json();
  query(episode, frame);
}

function card(item, rank) {
  const isQuery = rank === 0;
  const isCrossEp = !isQuery && !item.same_episode;
  const cls = isQuery ? 'query' : (isCrossEp ? 'cross-ep' : 'neighbor');
  const crossTag = isCrossEp ? `<span class="cross-tag">≠ ep</span>` : '';
  const badge = isQuery ? 'QUERY' : `K${rank}${crossTag}`;

  let metaRows = `
    <div class="row">global &nbsp;<span>${item.global_idx}</span></div>
    <div class="row">ep &nbsp;&nbsp;&nbsp;&nbsp;<span>${item.episode}</span></div>
    <div class="row">frame &nbsp;<span>${item.frame}</span></div>`;
  if (!isQuery) metaRows += `
    <div class="row">cos &nbsp;&nbsp;&nbsp;<span>${item.cos.toFixed(4)}</span></div>
    <div class="row">L2 &nbsp;&nbsp;&nbsp;&nbsp;<span>${item.l2.toFixed(3)}</span></div>`;

  const imgSlot = (b64, label) => b64
    ? `<div class="img-slot"><img src="data:image/png;base64,${b64}"><div class="img-label">${label}</div></div>`
    : `<div class="img-slot"><div class="img-missing">no image</div><div class="img-label">${label}</div></div>`;

  return `
  <div class="card ${cls}">
    <div class="card-meta">
      <div class="badge">${badge}</div>
      ${metaRows}
    </div>
    <div class="img-pair">
      ${imgSlot(item.agentview, 'agentview')}
      ${imgSlot(item.wrist, 'wrist')}
    </div>
  </div>`;
}

function render(data) {
  const rows = [card(data.query, 0), ...data.neighbors.map((n, i) => card(n, i+1))];
  document.getElementById('results').innerHTML = rows.join('');
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML, episodes=EPISODES, n=N, n_ep=len(EPISODES))


@app.route("/random")
def random_frame():
    idx = int(np.random.randint(0, N))
    return jsonify(episode=int(EPISODE_INDICES[idx]), frame=int(FRAME_INDICES[idx]))


@app.route("/query")
def query():
    ep = request.args.get("episode", type=int)
    fr = request.args.get("frame", type=int)
    k  = request.args.get("k", default=8, type=int)

    if ep is None or fr is None:
        return "Missing episode or frame", 400

    matches = np.where((EPISODE_INDICES == ep) & (FRAME_INDICES == fr))[0]
    if len(matches) == 0:
        return f"No frame found for episode={ep} frame={fr}", 404
    query_idx = int(matches[0])

    ag, wr = _get_images(query_idx)
    result = {
        "query": {
            "global_idx": query_idx,
            "episode": ep,
            "frame": fr,
            "agentview": ag,
            "wrist": wr,
            "cos": 1.0,
            "l2": 0.0,
        },
        "neighbors": [],
    }

    neighbor_idxs, sims, l2s = _find_knn(query_idx, k)
    for nidx, sim, l2 in zip(neighbor_idxs, sims, l2s):
        nag, nwr = _get_images(int(nidx))
        result["neighbors"].append({
            "global_idx": int(nidx),
            "episode": int(EPISODE_INDICES[nidx]),
            "frame": int(FRAME_INDICES[nidx]),
            "agentview": nag,
            "wrist": nwr,
            "cos": float(sim),
            "l2": float(l2),
            "same_episode": int(EPISODE_INDICES[nidx]) == ep,
        })

    return jsonify(result)


if __name__ == "__main__":
    print(f"\nOpen http://localhost:{PORT} (or tunnel: ssh -L {PORT}:localhost:{PORT} <user>@<host>)\n")
    app.run(host="0.0.0.0", port=PORT, debug=False)
