"""Visualize K nearest neighbors in latent space for a query frame.

For a given query frame, finds its K nearest neighbors by cosine similarity
over VLM latents, then plots pairs of (agentview, wrist) images for the query
and each neighbor.

Usage:
  # by global frame index
  uv run python examples/robomimic/visualize_knn.py --query-idx 42 --k 8

  # by episode + frame within that episode
  uv run python examples/robomimic/visualize_knn.py --episode 3 --frame 10 --k 8
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_NPZ = "examples/robomimic/data/latents_robomimic_2_19999.npz"
DEFAULT_LEROBOT_ROOT = "/iris/u/rbhowmik/cache/huggingface/lerobot/rbhowmik/robomimic_square_ph_image"
DEFAULT_REPO_ID = "rbhowmik/robomimic_square_ph_image"


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.uint8:
        arr = (np.asarray(arr, dtype=np.float32) * 255).clip(0, 255).astype(np.uint8)
    return arr


def _chw_to_hwc(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)
    return arr


def _load_lerobot_frame(hf_dataset, global_idx: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load agentview + wrist from LeRobot HF dataset by global frame index."""
    try:
        item = hf_dataset[int(global_idx)]
        agentview = _to_uint8(_chw_to_hwc(np.asarray(item["image"]))) if "image" in item else None
        wrist = _to_uint8(_chw_to_hwc(np.asarray(item["wrist_image"]))) if "wrist_image" in item else None
        return agentview, wrist
    except Exception as e:
        print(f"  LeRobot load failed for idx={global_idx}: {e}")
        return None, None


def find_knn(latents: np.ndarray, query_idx: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(latents, axis=1, keepdims=True)
    normed = latents / np.maximum(norms, 1e-8)
    q = normed[query_idx]
    sims = normed @ q  # (N,)
    sims[query_idx] = -2.0  # exclude self
    top_k_idx = np.argsort(sims)[::-1][:k]
    return top_k_idx, sims[top_k_idx]


def get_images(
    idx: int,
    npz_images: np.ndarray | None,
    npz_wrist: np.ndarray | None,
    hf_dataset,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (agentview, wrist) for a global frame index."""
    # Prefer npz (fast, already in memory)
    agentview = npz_images[idx] if npz_images is not None else None
    wrist = npz_wrist[idx] if npz_wrist is not None else None

    # Fall back to LeRobot dataset
    if (agentview is None or wrist is None) and hf_dataset is not None:
        ag, wr = _load_lerobot_frame(hf_dataset, idx)
        if agentview is None:
            agentview = ag
        if wrist is None:
            wrist = wr

    return agentview, wrist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default=DEFAULT_NPZ)
    parser.add_argument("--lerobot-root", default=DEFAULT_LEROBOT_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--query-idx", type=int, default=None, help="Global frame index")
    parser.add_argument("--random", action="store_true", help="Pick a random query frame")
    parser.add_argument("--episode", type=int, default=None, help="Episode index (use with --frame)")
    parser.add_argument("--frame", type=int, default=None, help="Frame within episode (use with --episode)")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--output", default="examples/robomimic/data/knn_viz.png")
    args = parser.parse_args()

    print(f"Loading {args.npz} ...")
    data = np.load(args.npz)
    latents = data["latents"]                          # (N, d)
    episode_indices = data["episode_indices"]          # (N,)
    frame_indices = data["frame_indices"]              # (N,)
    npz_images = data["images"] if "images" in data else None
    npz_wrist = data["wrist_images"] if "wrist_images" in data else None
    N = len(latents)
    print(f"  N={N}  d={latents.shape[1]}  has_agentview={npz_images is not None}  has_wrist={npz_wrist is not None}")

    # Load LeRobot dataset for image fallback (offline, from local cache)
    hf_dataset = None
    if npz_images is None or npz_wrist is None:
        try:
            import os
            os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
            import lerobot.common.datasets.lerobot_dataset as ld
            print(f"Loading LeRobot dataset from {args.lerobot_root} for image fallback ...")
            ds = ld.LeRobotDataset(args.repo_id, root=args.lerobot_root)
            hf_dataset = ds.hf_dataset
            print("  Dataset loaded.")
        except Exception as e:
            print(f"  Could not load LeRobot dataset: {e}")

    # Resolve query index
    if args.query_idx is not None:
        query_idx = args.query_idx
    elif args.random:
        query_idx = int(np.random.randint(0, N))
        print(f"Random query_idx={query_idx}")
    elif args.episode is not None and args.frame is not None:
        matches = np.where((episode_indices == args.episode) & (frame_indices == args.frame))[0]
        if len(matches) == 0:
            raise ValueError(f"No frame found for episode={args.episode} frame={args.frame}")
        query_idx = int(matches[0])
    else:
        raise ValueError("Provide --query-idx, --random, or both --episode and --frame")

    print(f"Query: global_idx={query_idx}  ep={episode_indices[query_idx]}  fr={frame_indices[query_idx]}")

    neighbor_idxs, neighbor_sims = find_knn(latents, query_idx, args.k)

    all_idxs = [query_idx] + list(neighbor_idxs)
    all_sims = [1.0] + list(neighbor_sims)

    # ── compute L2 distances too ──────────────────────────────────────────────
    q_latent = latents[query_idx]
    neighbor_l2 = [float(np.linalg.norm(latents[i] - q_latent)) for i in neighbor_idxs]

    # ── plot ─────────────────────────────────────────────────────────────────
    QUERY_COLOR = "#f5a623"   # amber for query
    NEIGHBOR_COLOR = "#4a90d9"  # blue for neighbors

    n_rows = args.k + 1
    # 3 columns: [text panel | agentview | wrist], text panel is narrow
    fig = plt.figure(figsize=(10, 2.6 * n_rows))
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(
        n_rows, 3,
        width_ratios=[1.6, 2, 2],
        hspace=0.08,
        wspace=0.05,
        left=0.02, right=0.98, top=0.96, bottom=0.02,
    )

    fig.suptitle(
        f"KNN  |  query: global={query_idx}  ep={episode_indices[query_idx]}  fr={frame_indices[query_idx]}  K={args.k}",
        fontsize=11, fontweight="bold", y=0.99,
    )

    # Column headers on row 0
    for col_i, label in enumerate(["", "agentview (3rd person)", "wrist (in-hand)"]):
        if label:
            ax_h = fig.add_subplot(gs[0, col_i])
            ax_h.set_title(label, fontsize=9)
            ax_h.axis("off")

    for row, (idx, sim) in enumerate(zip(all_idxs, all_sims)):
        ep = int(episode_indices[idx])
        fr = int(frame_indices[idx])
        agentview, wrist = get_images(idx, npz_images, npz_wrist, hf_dataset)
        is_query = row == 0
        border_color = QUERY_COLOR if is_query else NEIGHBOR_COLOR
        bg_color = "#fff8ec" if is_query else "white"

        # ── text panel ──
        ax_txt = fig.add_subplot(gs[row, 0])
        ax_txt.set_facecolor(bg_color)
        ax_txt.axis("off")

        if is_query:
            lines = [
                ("QUERY", 12, "bold", QUERY_COLOR),
                (f"global idx:  {idx}", 8.5, "normal", "#333"),
                (f"episode:     {ep}", 8.5, "normal", "#333"),
                (f"frame:       {fr}", 8.5, "normal", "#333"),
            ]
        else:
            l2 = neighbor_l2[row - 1]
            lines = [
                (f"K{row}", 11, "bold", NEIGHBOR_COLOR),
                (f"cos sim:  {sim:.4f}", 8.5, "normal", "#333"),
                (f"L2 dist:  {l2:.3f}", 8.5, "normal", "#333"),
                (f"global:   {idx}", 8.5, "normal", "#555"),
                (f"ep:       {ep}   fr: {fr}", 8.5, "normal", "#555"),
            ]

        y = 0.92
        for text, size, weight, color in lines:
            ax_txt.text(0.08, y, text, transform=ax_txt.transAxes,
                        fontsize=size, fontweight=weight, color=color,
                        va="top", ha="left", family="monospace")
            y -= 0.18 if size >= 11 else 0.16

        # Border around text panel
        for spine in ax_txt.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2.5 if is_query else 1.2)

        # ── image panels ──
        for col_i, img in enumerate([agentview, wrist], start=1):
            ax = fig.add_subplot(gs[row, col_i])
            ax.set_facecolor(bg_color)
            if img is not None:
                ax.imshow(img)
            else:
                ax.set_facecolor("#222")
                ax.text(0.5, 0.5, "unavailable", color="white", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2.5 if is_query else 1.2)

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
