"""Evaluate KNN quality across different values of K.

Produces a 2x3 figure:
  Row 1 (curves vs K):
    - avg_sim_at_k       : cosine sim of k-th neighbor (elbow)
    - episode_diversity  : fraction of top-K from a different episode
    - unique_episodes_at_k: mean number of distinct episodes in top-K neighbors

  Row 2 (distributions):
    - sim@1 histogram    : spread of nearest-neighbor similarities
    - within vs cross sim: overlaid histograms of same- vs diff-episode sim
    - k_until_cross_ep   : how many neighbors before the first cross-episode hit
"""
from __future__ import annotations

import argparse
import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np


def evaluate_k(
    knn_indices: np.ndarray,      # (N, K_max)
    knn_sims: np.ndarray,         # (N, K_max)
    episode_indices: np.ndarray,  # (N,)
    k_values: list[int],
) -> dict[str, list[float]]:
    results = {
        "k": k_values,
        "avg_sim_at_k": [],
        "episode_diversity": [],
        "unique_episodes_at_k": [],
    }

    for k in k_values:
        assert k <= knn_indices.shape[1], f"k={k} exceeds saved K_max={knn_indices.shape[1]}"
        neighbors = knn_indices[:, :k]          # (N, k)
        query_eps = episode_indices[:, None]    # (N, 1)
        neighbor_eps = episode_indices[neighbors]  # (N, k)

        results["avg_sim_at_k"].append(float(knn_sims[:, k - 1].mean()))

        diff = (neighbor_eps != query_eps).astype(float)
        results["episode_diversity"].append(float(diff.mean()))

        # mean number of distinct episodes per query (including self)
        unique_ep_counts = np.array([len(np.unique(row)) for row in neighbor_eps])
        results["unique_episodes_at_k"].append(float(unique_ep_counts.mean()))

    return results


def sim_distributions(
    knn_indices: np.ndarray,      # (N, K_max)
    knn_sims: np.ndarray,         # (N, K_max)
    episode_indices: np.ndarray,  # (N,)
    k_sample: int = 20,
) -> dict:
    """Compute similarity distributions for plotting."""
    neighbors_k = knn_indices[:, :k_sample]
    sims_k = knn_sims[:, :k_sample]
    query_eps = episode_indices[:, None]
    neighbor_eps = episode_indices[neighbors_k]

    same_ep = (neighbor_eps == query_eps)
    within_sims = sims_k[same_ep].ravel()
    cross_sims = sims_k[~same_ep].ravel()

    # K until first cross-episode neighbor (per query); cap at k_sample+1 if never
    first_cross = []
    for i in range(len(knn_indices)):
        hits = np.where(neighbor_eps[i] != episode_indices[i])[0]
        first_cross.append(int(hits[0]) + 1 if len(hits) > 0 else k_sample + 1)

    return {
        "sim_at_1": knn_sims[:, 0],
        "within_sims": within_sims,
        "cross_sims": cross_sims,
        "first_cross_k": np.array(first_cross),
        "k_sample": k_sample,
    }


def main(
    data_dir: str = "examples/robomimic/data",
    k_max: int | None = None,
    output_dir: str = "examples/robomimic/data",
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    d = pathlib.Path(data_dir)

    knn_indices = np.load(d / "knn_indices.npy")
    knn_sims = np.load(d / "knn_similarities.npy")
    episode_indices = np.load(d / "episode_indices.npy")
    frame_indices = np.load(d / "frame_indices.npy")

    K_max = knn_indices.shape[1]
    if k_max is not None:
        K_max = min(k_max, K_max)

    # Use a subset of K values to keep plotting fast
    step = max(1, K_max // 50)
    k_values = list(range(1, K_max + 1, step))
    if k_values[-1] != K_max:
        k_values.append(K_max)

    logging.info("Evaluating K in 1..%d (step=%d) over N=%d samples", K_max, step, len(knn_indices))
    results = evaluate_k(knn_indices, knn_sims, episode_indices, k_values)

    k_sample = min(20, K_max)
    dists = sim_distributions(knn_indices, knn_sims, episode_indices, k_sample=k_sample)

    # Print table (sparse)
    print(f"\n{'K':>5}  {'sim@k':>8}  {'ep_diversity':>13}  {'unique_eps':>11}")
    print("-" * 44)
    for i, k in enumerate(k_values):
        print(
            f"{k:>5}  {results['avg_sim_at_k'][i]:>8.4f}  "
            f"{results['episode_diversity'][i]:>13.4f}  "
            f"{results['unique_episodes_at_k'][i]:>11.1f}"
        )

    # ── summary stats ──────────────────────────────────────────────────────────
    n_ep = len(np.unique(episode_indices))
    pct_never_cross = 100 * (dists["first_cross_k"] > k_sample).mean()
    logging.info("N=%d  episodes=%d  K_max=%d", len(knn_indices), n_ep, K_max)
    logging.info("sim@1: mean=%.4f  min=%.4f  max=%.4f",
                 dists["sim_at_1"].mean(), dists["sim_at_1"].min(), dists["sim_at_1"].max())
    logging.info("Within-ep sim: %.4f ± %.4f", dists["within_sims"].mean(), dists["within_sims"].std())
    logging.info("Cross-ep sim:  %.4f ± %.4f", dists["cross_sims"].mean(), dists["cross_sims"].std())
    logging.info("%.1f%% of queries never hit a cross-episode neighbor in top-%d", pct_never_cross, k_sample)

    # ── plot ───────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"KNN evaluation  |  N={len(knn_indices):,}  episodes={n_ep}  K_max={K_max}",
        fontsize=13,
    )

    # Row 0, col 0: elbow
    ax = axes[0, 0]
    ax.plot(k_values, results["avg_sim_at_k"], marker="o", markersize=2)
    ax.set_xlabel("K")
    ax.set_ylabel("Cosine sim of k-th neighbor")
    ax.set_title("Elbow: sim drops off at K=?")
    ax.grid(True)

    # Row 0, col 1: episode diversity
    ax = axes[0, 1]
    ax.plot(k_values, results["episode_diversity"], marker="o", markersize=2, color="orange")
    ax.set_xlabel("K")
    ax.set_ylabel("Fraction from different episode")
    ax.set_title("Episode diversity (higher = better)")
    ax.grid(True)

    # Row 0, col 2: unique episodes in top-K
    ax = axes[0, 2]
    ax.plot(k_values, results["unique_episodes_at_k"], marker="o", markersize=2, color="purple")
    ax.axhline(n_ep, color="red", linestyle="--", linewidth=1, label=f"total episodes={n_ep}")
    ax.set_xlabel("K")
    ax.set_ylabel("Mean distinct episodes in top-K")
    ax.set_title("Unique episodes covered")
    ax.legend(fontsize=8)
    ax.grid(True)

    # Row 1, col 0: sim@1 histogram
    ax = axes[1, 0]
    ax.hist(dists["sim_at_1"], bins=60, color="steelblue", edgecolor="none")
    ax.axvline(dists["sim_at_1"].mean(), color="red", linestyle="--", label=f"mean={dists['sim_at_1'].mean():.4f}")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Count")
    ax.set_title("Nearest-neighbor (K=1) similarity distribution")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y")

    # Row 1, col 1: within vs cross episode sim
    ax = axes[1, 1]
    bins = np.linspace(
        min(dists["within_sims"].min(), dists["cross_sims"].min()),
        max(dists["within_sims"].max(), dists["cross_sims"].max()),
        60,
    )
    ax.hist(dists["within_sims"], bins=bins, alpha=0.6, color="steelblue",
            label=f"within-ep (μ={dists['within_sims'].mean():.4f})", density=True)
    ax.hist(dists["cross_sims"], bins=bins, alpha=0.6, color="orange",
            label=f"cross-ep  (μ={dists['cross_sims'].mean():.4f})", density=True)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title(f"Within vs cross-episode sim (top-{k_sample} neighbors)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y")

    # Row 1, col 2: K until first cross-episode hit
    ax = axes[1, 2]
    capped = dists["first_cross_k"].clip(max=k_sample + 1)
    bins_k = np.arange(0.5, k_sample + 2.5)
    ax.hist(capped, bins=bins_k, color="green", edgecolor="none")
    ax.axvline(np.median(dists["first_cross_k"].clip(max=k_sample)),
               color="red", linestyle="--",
               label=f"median={int(np.median(capped))}")
    if pct_never_cross > 0:
        ax.bar(k_sample + 1, (capped == k_sample + 1).sum(), color="gray",
               label=f"never ({pct_never_cross:.1f}%)")
    ax.set_xlabel("K of first cross-episode neighbor")
    ax.set_ylabel("Count")
    ax.set_title(f"How deep to find cross-episode neighbor (top-{k_sample})")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y")

    plt.tight_layout()
    out = pathlib.Path(output_dir) / "knn_eval.png"
    plt.savefig(out, dpi=150)
    logging.info("Saved plot to %s", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="examples/robomimic/data")
    parser.add_argument("--k-max", type=int, default=None)
    parser.add_argument("--output-dir", default="examples/robomimic/data")
    args = parser.parse_args()
    main(args.data_dir, args.k_max, args.output_dir)
