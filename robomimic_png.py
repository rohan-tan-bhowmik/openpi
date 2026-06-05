#!/usr/bin/env python3
"""
Save exactly one PNG from robosuite's square-peg scene camera views, headlessly.

Example:
    uv run python robomimic_png.py \
        --outdir square_peg_pngs \
        --width 512 \
        --height 512 \
        --cameras agentview robot0_eye_in_hand

Notes:
- No X11 required.
- Saves only the reset frame.
- Uses OSMesa for headless rendering.
"""

from __future__ import annotations

import os
os.environ["MUJOCO_GL"] = "osmesa"

import argparse
from pathlib import Path
from typing import Dict, List

import imageio.v2 as imageio
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, default=Path("square_peg_pngs"))
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument(
        "--cameras",
        nargs="+",
        default=["agentview", "robot0_eye_in_hand"],
        help="Camera names to save",
    )
    p.add_argument(
        "--robot",
        type=str,
        default="Panda",
        help="Robot name, e.g. Panda",
    )
    p.add_argument(
        "--device-id",
        type=int,
        default=-1,
        help="GPU device id for offscreen render (-1 = infer/default)",
    )
    p.add_argument(
        "--flip-vertical",
        action="store_true",
        help="Flip frames vertically if output is upside down",
    )
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def maybe_fix_image(img: np.ndarray, flip_vertical: bool) -> np.ndarray:
    arr = np.asarray(img)

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if flip_vertical:
        arr = arr[::-1]

    return arr


def make_env(args: argparse.Namespace):
    import robosuite as suite

    env_name = "NutAssemblySquare"

    env = suite.make(
        env_name=env_name,
        robots=args.robot,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=args.cameras,
        camera_heights=args.height,
        camera_widths=args.width,
        render_gpu_device_id=args.device_id,
        reward_shaping=False,
        control_freq=20,
        ignore_done=True,
    )
    return env, env_name


def save_frame_set(
    obs: Dict[str, np.ndarray],
    cameras: List[str],
    outdir: Path,
    flip_vertical: bool,
) -> None:
    for cam in cameras:
        key = f"{cam}_image"
        if key not in obs:
            available = ", ".join(sorted(obs.keys()))
            raise KeyError(f"Missing obs key '{key}'. Available keys: {available}")

        cam_dir = outdir / cam
        ensure_dir(cam_dir)

        img = maybe_fix_image(obs[key], flip_vertical=flip_vertical)
        imageio.imwrite(cam_dir / "000000.png", img)


def main() -> int:
    args = parse_args()
    ensure_dir(args.outdir)

    env, env_name = make_env(args)

    try:
        obs = env.reset()

        xml_str = env.sim.model.get_xml()

        with open("debug_scene.xml", "w") as f:
            f.write(xml_str)

        print("Saved full XML to debug_scene.xml")
        print(f"Created env: {env_name}")
        print(f"Saving cameras: {args.cameras}")
        print(f"Output dir: {args.outdir}")

        save_frame_set(obs, args.cameras, args.outdir, args.flip_vertical)

        print("Done.")
        for cam in args.cameras:
            print(f"  {cam}: {args.outdir / cam / '000000.png'}")

    finally:
        try:
            env.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())