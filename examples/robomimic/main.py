from __future__ import annotations

import collections
import dataclasses
import json
import logging
import pathlib
import random
import xml.etree.ElementTree as ET
from typing import Any

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ["PYOPENGL_PLATFORM"] = os.environ["MUJOCO_GL"]
# Hide GPUs so osmesa (CPU rendering) doesn't conflict with the policy server's JAX/EGL context
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROBOMIMIC_DATASET_RESOLUTION = 84


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    replan_steps: int = 5
    resize_size: int = 224

    env_name: str = "NutAssemblySquare"
    robots: str = "Panda"
    task: str = "insert the peg into the square hole"
    num_trials: int = 10
    max_steps: int = 400
    seed: int = 7
    dataset_path: str = "/iris/u/rbhowmik/cache/robomimic_datasets/square/ph/image_v15.hdf5"

    randomize_peg: bool = False

    use_images: bool = True
    use_wrist_image: bool = True
    camera_height: int = ROBOMIMIC_DATASET_RESOLUTION
    camera_width: int = ROBOMIMIC_DATASET_RESOLUTION
    render_gpu_device_id: int = -1
    control_freq: int = 20

    video_out_path: str = "data/robomimic/videos"
    save_failures_only: bool = False


def _randomize_peg(env, x_range: float = 0.1, y_range: float = 0.1) -> None:
    """Randomize peg1 position before each episode reset."""
    x = random.uniform(-x_range, x_range)
    y = random.uniform(-y_range, y_range)

    def modify_xml(xml_str: str) -> str:
        root = ET.fromstring(xml_str)
        body = root.find(".//body[@name='peg1']")
        if body is not None:
            body.set("pos", f"{x + 0.23} {y + 0.1} 0.85")
        return ET.tostring(root, encoding="unicode")

    robosuite_env = env.env  # unwrap robomimic's EnvRobosuite
    robosuite_env.set_xml_processor(processor=modify_xml)
    robosuite_env.hard_reset = True


def eval_robomimic(args: Args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_episodes = 0
    total_successes = 0

    # Create env once and reuse — recreating per-episode exhausts EGL/GPU contexts.
    env, camera_names = _make_env(args, seed=args.seed)
    try:
        for episode_idx in range(args.num_trials):
            action_plan: collections.deque[np.ndarray] = collections.deque()
            replay_frames: dict[str, list[np.ndarray]] = {cam: [] for cam in camera_names}
            episode_success = False

            try:
                if args.randomize_peg:
                    _randomize_peg(env, x_range=0.1, y_range=0.1)
                obs = env.reset()

                missing_cams = [cam for cam in camera_names if f"{cam}_image" not in obs]
                if missing_cams:
                    logging.warning("Cameras missing from obs: %s", missing_cams)

                for step_idx in range(args.max_steps):
                    if args.use_images:
                        for cam in camera_names:
                            key = f"{cam}_image"
                            if key in obs:
                                frame = np.asarray(obs[key])
                                if frame.ndim == 3 and frame.shape[0] in (1, 3):
                                    frame = frame.transpose(1, 2, 0)  # CHW -> HWC
                                if frame.dtype != np.uint8:
                                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                                replay_frames[cam].append(frame[::-1])  # vertical flip for MuJoCo

                    if not action_plan:
                        element = _build_openpi_observation(obs, args)
                        try:
                            action_chunk = np.asarray(client.infer(element)["actions"])
                        except Exception as e:
                            logging.warning("Inference failed on episode %d step %d: %s", episode_idx, step_idx, e)
                            break
                        if step_idx == 0 and episode_idx == 0:
                            logging.info("First action chunk shape=%s sample=%s nan=%s",
                                         action_chunk.shape,
                                         np.round(action_chunk[0], 3),
                                         np.any(np.isnan(action_chunk)))
                        if len(action_chunk) < args.replan_steps:
                            raise ValueError(
                                f"Policy returned {len(action_chunk)} actions, "
                                f"but replan_steps={args.replan_steps}."
                            )
                        action_plan.extend(np.asarray(a, dtype=np.float32) for a in action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, _, env_done, _ = env.step(action.tolist())

                    episode_success = _check_success(env)
                    if episode_success or env_done:
                        break

                total_episodes += 1
                total_successes += int(episode_success)

                _maybe_save_videos(
                    replay_frames=replay_frames,
                    args=args,
                    success=episode_success,
                    episode_idx=episode_idx,
                )

                logging.info(
                    "Episode %d success=%s running_success_rate=%.3f",
                    episode_idx,
                    episode_success,
                    total_successes / total_episodes,
                )
            except Exception as e:
                logging.error("Episode %d crashed: %s", episode_idx, e, exc_info=True)

    finally:
        if hasattr(env, "close"):
            env.close()
        elif hasattr(env, "env"):
            env.env.close()

    logging.info(
        "Final success rate: %.3f (%d / %d)",
        total_successes / max(total_episodes, 1),
        total_successes,
        total_episodes,
    )


def _build_openpi_observation(obs: dict[str, Any], args: Args) -> dict[str, Any]:
    element: dict[str, Any] = {
        "observation/state": _build_state(obs),
        "prompt": args.task,
    }

    if args.use_images:
        element["observation/image"] = _preprocess_image(obs["agentview_image"], args.resize_size)
        if args.use_wrist_image and "robot0_eye_in_hand_image" in obs:
            element["observation/wrist_image"] = _preprocess_image(
                obs["robot0_eye_in_hand_image"], args.resize_size
            )

    return element


def _build_state(obs: dict[str, Any]) -> np.ndarray:
    required = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")
    missing = [key for key in required if key not in obs]
    if missing:
        raise KeyError(f"Missing required RoboMimic observation keys for state: {missing}")

    state = np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            np.asarray(obs["robot0_eef_quat"], dtype=np.float32),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ],
        axis=-1,
    )
    if state.shape != (9,):
        raise ValueError(f"Expected 9D state, got shape {state.shape}")
    return state


def _preprocess_image(image: np.ndarray, resize_size: int) -> np.ndarray:
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.transpose(1, 2, 0)  # CHW -> HWC
    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)
    image = np.ascontiguousarray(image)
    return image_tools.resize_with_pad(image, resize_size, resize_size)


def _maybe_save_videos(
    *,
    replay_frames: dict[str, list[np.ndarray]],
    args: Args,
    success: bool,
    episode_idx: int,
) -> None:
    if args.save_failures_only and success:
        return

    suffix = "success" if success else "failure"
    task_segment = args.task.replace(" ", "_")
    out_dir = pathlib.Path(args.video_out_path)
    for cam, frames in replay_frames.items():
        if not frames:
            continue
        out_path = out_dir / f"rollout_{args.env_name}_{task_segment}_ep{episode_idx:03d}_{cam}_{suffix}.mp4"
        imageio.mimwrite(out_path, frames, fps=args.control_freq)


def _make_env(args: Args, *, seed: int):
    from robomimic.config import config_factory
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils

    # Force CPU rendering so robomimic's EGL device selection doesn't override MUJOCO_GL=osmesa.
    EnvUtils.get_valid_gpu_device_id = lambda: -1

    env_meta = FileUtils.get_env_metadata_from_dataset(args.dataset_path)
    env_meta["env_kwargs"].pop("lite_physics", None)

    config = config_factory(algo_name="bc")
    config.observation.modalities.obs.rgb = [
        f"{cam}_image" for cam in env_meta["env_kwargs"]["camera_names"]
    ]
    ObsUtils.initialize_obs_utils_with_config(config)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_meta["env_name"],
        render=False,
        render_offscreen=True,
        use_image_obs=True,
    )

    camera_names = env_meta["env_kwargs"]["camera_names"]
    logging.info("Using cameras: %s", camera_names)
    return env, camera_names


def _check_success(env: Any) -> bool:
    if hasattr(env, "is_success"):
        return bool(env.is_success()["task"])
    if hasattr(env, "env") and hasattr(env.env, "is_success"):
        return bool(env.env.is_success()["task"])
    if hasattr(env, "_check_success"):
        return bool(env._check_success())
    return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(eval_robomimic)
