"""Compute all pairwise cosine similarities between extracted latents and run KNN.

Loads latents.npz, L2-normalizes, computes the full (N x N) gram matrix,
and saves both the similarity matrix and top-K neighbor indices.

Output files:
  sim_matrix.npy      — (N, N) float16 cosine similarities
  knn_indices.npy     — (N, K) int32 indices of K nearest neighbors (self excluded)
  knn_similarities.npy — (N, K) float32 cosine similarities for those neighbors
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import time

import numpy as np
import tqdm


def main(
    latents_path: str = "examples/robomimic/data/latents_full_robomimic_ph.npz",
    output_dir: str = "examples/robomimic/data",
    k: int = 10,
    num_samples: int | None = None,
    seed: int = 42,
    skip_sim: bool = False,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logging.info("Loading latents from %s", latents_path)
    data = np.load(latents_path)
    latents = data["latents"].astype(np.float32)          # (N, d)
    episode_indices = data["episode_indices"]
    frame_indices = data["frame_indices"]
    N, d = latents.shape
    logging.info("Latents: N=%d  d=%d", N, d)

    if num_samples is not None and num_samples < N:
        rng = np.random.default_rng(seed)
        subset = rng.choice(N, size=num_samples, replace=False)
        subset.sort()
        latents = latents[subset]
        episode_indices = episode_indices[subset]
        frame_indices = frame_indices[subset]
        N = num_samples
        logging.info("Randomly sampled %d latents (seed=%d)", N, seed)

    sim_path = out / "sim_matrix.npy"

    if skip_sim:
        logging.info("Loading existing similarity matrix from %s", sim_path)
        sim_matrix = np.load(sim_path).astype(np.float32)
        assert sim_matrix.shape == (N, N), f"Sim matrix shape {sim_matrix.shape} != ({N}, {N})"
    else:
        # L2-normalize so dot product == cosine similarity
        norms = np.linalg.norm(latents, axis=1, keepdims=True)
        latents_normed = latents / np.clip(norms, 1e-8, None)  # (N, d)

        logging.info("Computing %dx%d gram matrix (~%.1f GB float32)...", N, N, N * N * 4 / 1e9)
        t0 = time.time()
        sim_matrix = latents_normed @ latents_normed.T         # (N, N) float32
        logging.info("Gram matrix done in %.1fs", time.time() - t0)

        np.save(sim_path, sim_matrix.astype(np.float16))
        logging.info("Saved similarity matrix to %s", sim_path)

    # KNN: for each row, find top-K excluding self (diagonal)
    logging.info("Computing KNN (k=%d)...", k)
    t0 = time.time()
    np.fill_diagonal(sim_matrix, -np.inf)                  # exclude self
    chunk = 512
    knn_indices = np.empty((N, k), dtype=np.int32)
    knn_sims = np.empty((N, k), dtype=np.float32)
    for start in tqdm.trange(0, N, chunk, desc="KNN"):
        end = min(start + chunk, N)
        rows = sim_matrix[start:end]
        idx = np.argsort(rows, axis=1)[:, -k:][:, ::-1]
        knn_indices[start:end] = idx
        knn_sims[start:end] = np.take_along_axis(rows, idx, axis=1)
    logging.info("KNN done in %.1fs", time.time() - t0)

    np.save(out / "knn_indices.npy", knn_indices)
    np.save(out / "knn_similarities.npy", knn_sims)
    # Save metadata alongside so everything is co-indexed
    np.save(out / "episode_indices.npy", episode_indices)
    np.save(out / "frame_indices.npy", frame_indices)

    logging.info(
        "Saved knn_indices %s, knn_similarities %s", knn_indices.shape, knn_sims.shape
    )

    # Quick sanity check: nearest neighbor of frame 0
    i = 0
    nn = knn_indices[i, 0]
    logging.info(
        "Nearest neighbor of frame 0 (ep=%d, f=%d): frame %d (ep=%d, f=%d) sim=%.4f",
        episode_indices[i], frame_indices[i],
        nn, episode_indices[nn], frame_indices[nn],
        knn_sims[i, 0],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents-path", default="examples/robomimic/data/latents_full_robomimic_ph.npz")
    parser.add_argument("--output-dir", default="examples/robomimic/data")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=None, help="Random subset size. Omit to use all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-sim", action="store_true", help="Skip gram matrix computation and load existing sim_matrix.npy.")
    args = parser.parse_args()
    main(args.latents_path, args.output_dir, args.k, args.num_samples, args.seed, args.skip_sim)
