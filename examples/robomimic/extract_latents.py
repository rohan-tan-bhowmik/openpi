"""Extract VLM prefix latents from the PH dataset using a trained checkpoint.

For each frame in the dataset, runs the VLM (SigLIP + Gemma) prefix forward pass
and saves a mean-pooled latent vector. The action expert is never invoked.

Output: a .npz file with:
  latents        — (N, hidden_dim) float32
  episode_indices — (N,) int32
  frame_indices   — (N,) int32
"""
from __future__ import annotations

import dataclasses
import logging
import os
import pathlib

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.7")
os.environ.setdefault("MUJOCO_GL", "osmesa")

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


@dataclasses.dataclass
class Args:
    # Training config name — must match the checkpoint.
    config_name: str = "pi0_robomimic_square_ph_image_lora"
    # Path to the specific step checkpoint directory (the one containing a 'params' folder).
    checkpoint_dir: str = "checkpoints/pi0_robomimic_square_ph_image_lora/robomimic_2/19999"
    # Where to save the output .npz file.
    output_path: str = "examples/robomimic/data/latents_robomimic_2_19999.npz"
    # Inference batch size. Reduce if OOM.
    batch_size: int = 8
    # Save agentview images alongside latents for image<->latent lookup.
    save_images: bool = True


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    config = _config.get_config(args.config_name)
    checkpoint_dir = pathlib.Path(args.checkpoint_dir)
    if not (checkpoint_dir / "params").exists():
        raise FileNotFoundError(f"No 'params' folder found at {checkpoint_dir}")

    logging.info("Loading model from %s", checkpoint_dir)
    model = config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    model.eval()

    # JIT the latent extraction — freezes model state at call time (same pattern as Policy).
    extract_fn = nnx_utils.module_jit(model.extract_prefix_latent)

    data_config = config.data.create(config.assets_dirs, config.model)

    # Raw LeRobot dataset — used only to read episode_index / frame_index metadata.
    meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
    raw_dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / meta.fps for t in range(config.model.action_horizon)]
            for key in data_config.action_sequence_keys
        },
    )

    # Transformed dataset — applies all normalization + model transforms to produce model-ready observations.
    transformed = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    transformed = _data_loader.transform_dataset(transformed, data_config)

    loader = torch.utils.data.DataLoader(
        transformed,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=_data_loader._collate_fn,
        num_workers=0,
    )

    all_latents: list[np.ndarray] = []
    all_episode_indices: list[int] = []
    all_frame_indices: list[int] = []
    all_images: list[np.ndarray] = []
    all_wrist_images: list[np.ndarray] = []

    # Detect agentview + wrist image keys in the HF dataset.
    # The convert script uses "image" (agentview) and "wrist_image".
    sample_item = raw_dataset.hf_dataset[0]
    all_keys = list(sample_item.keys())
    image_key = next(
        (k for k in ["image", "agentview_image"] if k in all_keys),
        next((k for k in all_keys if "agentview" in k), None),
    )
    wrist_key = next(
        (k for k in ["wrist_image", "wrist_image"] if k in all_keys),
        next((k for k in all_keys if "eye_in_hand" in k or "wrist" in k), None),
    )
    if image_key is None and args.save_images:
        logging.warning("No agentview image key found; save_images disabled. Keys: %s", all_keys)
        args.save_images = False
    else:
        logging.info("Agentview key: %s  Wrist key: %s", image_key, wrist_key)

    global_idx = 0
    for batch in tqdm.tqdm(loader, desc="Extracting latents"):
        batch_size = next(v for v in batch.values() if hasattr(v, "shape")).shape[0]

        # Collect metadata from the raw (untransformed) dataset in sync with the transformed batch.
        for i in range(batch_size):
            raw_item = raw_dataset.hf_dataset[global_idx + i]
            all_episode_indices.append(int(raw_item["episode_index"]))
            all_frame_indices.append(int(raw_item["frame_index"]))
            if args.save_images and image_key:
                def _to_uint8(x: np.ndarray) -> np.ndarray:
                    arr = np.asarray(x)
                    if arr.dtype != np.uint8:
                        arr = (arr * 255).clip(0, 255).astype(np.uint8)
                    return arr

                all_images.append(_to_uint8(raw_item[image_key]))
                if wrist_key:
                    all_wrist_images.append(_to_uint8(raw_item[wrist_key]))
        global_idx += batch_size

        observation = _model.Observation.from_dict(jax.tree.map(jnp.asarray, batch))
        latents = extract_fn(observation)
        all_latents.append(np.asarray(latents, dtype=np.float32))

    latents_arr = np.concatenate(all_latents, axis=0)
    episode_indices = np.array(all_episode_indices, dtype=np.int32)
    frame_indices = np.array(all_frame_indices, dtype=np.int32)

    output = pathlib.Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    save_kwargs: dict = dict(latents=latents_arr, episode_indices=episode_indices, frame_indices=frame_indices)
    if all_images:
        save_kwargs["images"] = np.stack(all_images, axis=0)  # (N, H, W, 3) uint8
        logging.info("Saving agentview images shape=%s", save_kwargs["images"].shape)
    if all_wrist_images:
        save_kwargs["wrist_images"] = np.stack(all_wrist_images, axis=0)  # (N, H, W, 3) uint8
        logging.info("Saving wrist images shape=%s", save_kwargs["wrist_images"].shape)

    np.savez(output, **save_kwargs)

    logging.info(
        "Saved %d latents shape=(%s,) to %s",
        len(latents_arr),
        latents_arr.shape[1],
        output,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
