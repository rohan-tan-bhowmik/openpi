from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np
import tyro

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def _as_float32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _as_uint8_image(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def _find_first_existing(group, keys: list[str]) -> str | None:
    for k in keys:
        if k in group:
            return k
    return None


def _make_state(obs_group) -> np.ndarray:
    """
    Build a 9D state from standard Franka RoboMimic low-dim keys:
      3  : robot0_eef_pos
      4  : robot0_eef_quat
      2  : robot0_gripper_qpos
      => 9 total
    """
    required = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    for k in required:
        if k not in obs_group:
            raise KeyError(f"Missing required obs key for state creation: {k}")

    eef_pos = _as_float32(obs_group["robot0_eef_pos"][:]) # (T, 3)
    eef_quat = _as_float32(obs_group["robot0_eef_quat"][:]) # (T, 4)
    gripper = _as_float32(obs_group["robot0_gripper_qpos"][:]) # (T, 2)

    state = np.concatenate([eef_pos, eef_quat, gripper], axis=-1)  # (T, 9)
    if state.shape[1] != 9:
        raise ValueError(f"Expected state dim 9, got shape {state.shape}")
    return state


def main(
    input_hdf5: str,
    repo_id: str,
    *,
    task: str = "insert the peg into the square hole",
    fps: int = 20,
    robot_type: str = "franka",
    overwrite: bool = False,
    push_to_hub: bool = False,
    max_episodes: int | None = None,
) -> None:
    input_path = Path(input_hdf5)
    if not input_path.exists():
        raise FileNotFoundError(f"Input HDF5 not found: {input_path}")

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output dataset already exists at {output_path}. "
                "Pass --overwrite to remove it first."
            )
        shutil.rmtree(output_path)

    # -------- First pass: inspect one demo to infer available modalities --------
    with h5py.File(input_path, "r") as f:
        if "data" not in f:
            raise KeyError("Expected top-level 'data' group in robomimic HDF5.")

        demo_keys = sorted(f["data"].keys())
        if not demo_keys:
            raise ValueError("No demos found under HDF5 group 'data'.")

        first_demo = f["data"][demo_keys[0]]
        if "obs" not in first_demo:
            raise KeyError(f"Demo {demo_keys[0]} is missing 'obs'.")
        if "actions" not in first_demo:
            raise KeyError(f"Demo {demo_keys[0]} is missing 'actions'.")

        first_obs = first_demo["obs"]
        action_dim = first_demo["actions"].shape[1]

        image_key = _find_first_existing(
            first_obs,
            [
                "agentview_image",
                "frontview_image",
                "sideview_image",
            ],
        )
        wrist_image_key = _find_first_existing(
            first_obs,
            [
                "robot0_eye_in_hand_image",
                "eye_in_hand_image",
            ],
        )

        # Infer shapes if images exist.
        image_shape = None
        wrist_image_shape = None

        if image_key is not None:
            image_shape = tuple(first_obs[image_key].shape[1:])  # (H, W, C)
            if len(image_shape) != 3:
                raise ValueError(f"Expected 3D image shape for {image_key}, got {image_shape}")

        if wrist_image_key is not None:
            wrist_image_shape = tuple(first_obs[wrist_image_key].shape[1:])
            if len(wrist_image_shape) != 3:
                raise ValueError(
                    f"Expected 3D image shape for {wrist_image_key}, got {wrist_image_shape}"
                )

        features = {
            "state": {
                "dtype": "float32",
                "shape": (9,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["actions"],
            },
        }

        if image_shape is not None:
            features["image"] = {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            }

        if wrist_image_shape is not None:
            features["wrist_image"] = {
                "dtype": "image",
                "shape": wrist_image_shape,
                "names": ["height", "width", "channel"],
            }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=fps,
        features=features,
    )

    print("Creating dataset with features:")
    for k, v in features.items():
        print(f"  {k}: {v}")

    num_episodes = 0
    num_frames = 0

    # -------- Second pass: convert all demos --------
    with h5py.File(input_path, "r") as f:
        demo_keys = sorted(f["data"].keys())
        if max_episodes is not None:
            demo_keys = demo_keys[:max_episodes]

        for demo_key in demo_keys:
            demo = f["data"][demo_key]

            if "obs" not in demo:
                raise KeyError(f"Demo {demo_key} is missing 'obs'.")
            if "actions" not in demo:
                raise KeyError(f"Demo {demo_key} is missing 'actions'.")

            obs = demo["obs"]
            actions = _as_float32(demo["actions"][:])   # (T, 7)
            state = _make_state(obs)                    # (T, 9)

            if len(state) != len(actions):
                raise ValueError(
                    f"Length mismatch in {demo_key}: "
                    f"state len {len(state)} vs actions len {len(actions)}"
                )

            image_seq = None
            wrist_image_seq = None

            if "image" in features:
                # Re-find key per demo for robustness.
                demo_image_key = _find_first_existing(
                    obs,
                    ["agentview_image", "frontview_image", "sideview_image"],
                )
                if demo_image_key is None:
                    raise KeyError(f"Demo {demo_key} is missing main image key")
                image_seq = _as_uint8_image(obs[demo_image_key][:])

            if "wrist_image" in features:
                demo_wrist_key = _find_first_existing(
                    obs,
                    ["robot0_eye_in_hand_image", "eye_in_hand_image"],
                )
                if demo_wrist_key is None:
                    raise KeyError(f"Demo {demo_key} is missing wrist image key")
                wrist_image_seq = _as_uint8_image(obs[demo_wrist_key][:])

            for t in range(len(actions)):
                frame = {
                    "state": state[t],
                    "actions": actions[t],
                    "task": task,
                }

                if image_seq is not None:
                    frame["image"] = image_seq[t]

                if wrist_image_seq is not None:
                    frame["wrist_image"] = wrist_image_seq[t]
                    
                dataset.add_frame(frame)
                num_frames += 1

            dataset.save_episode()
            num_episodes += 1
            print(f"Saved episode {num_episodes}: {demo_key} ({len(actions)} frames)")

    print(f"Done. Wrote {num_episodes} episodes and {num_frames} frames to {output_path}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["robomimic", "openpi"],
            private=False,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)