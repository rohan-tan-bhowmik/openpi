#!/usr/bin/env python3
"""
Visualize RoboMimic third-person and wrist-camera views side by side.

Examples
--------
# Show demo 0 interactively
python visualize_robomimic_views.py /path/to/dataset.hdf5

# Show a specific demo
python visualize_robomimic_views.py /path/to/dataset.hdf5 --demo demo_12

# Save to mp4 instead of showing
python visualize_robomimic_views.py /path/to/dataset.hdf5 --save out.mp4

# Use different observation keys
python visualize_robomimic_views.py /path/to/dataset.hdf5 \
    --third-key agentview_image \
    --wrist-key robot0_eye_in_hand_image
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("hdf5_path", type=Path, help="Path to RoboMimic .hdf5 dataset")
    p.add_argument("--demo", type=str, default=None, help="Demo name, e.g. demo_0")
    p.add_argument(
        "--third-key",
        type=str,
        default="agentview_image",
        help="Observation key for 3rd-person view",
    )
    p.add_argument(
        "--wrist-key",
        type=str,
        default="robot0_eye_in_hand_image",
        help="Observation key for wrist view",
    )
    p.add_argument("--fps", type=float, default=20.0, help="Playback / save FPS")
    p.add_argument("--save", type=Path, default=None, help="Optional mp4 output path")
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on number of frames to play / save",
    )
    return p.parse_args()


def list_demos(f: h5py.File) -> list[str]:
    if "data" not in f:
        raise KeyError("Expected top-level group 'data' in RoboMimic hdf5 file.")
    demos = sorted(list(f["data"].keys()))
    if not demos:
        raise ValueError("No demos found under /data")
    return demos


def find_obs_group(demo_group: h5py.Group) -> h5py.Group:
    if "obs" in demo_group:
        return demo_group["obs"]
    raise KeyError("Expected demo group to contain /obs")


def to_uint8_rgb(frames: np.ndarray) -> np.ndarray:
    arr = np.asarray(frames)

    if arr.ndim != 4:
        raise ValueError(f"Expected frames with shape (T,H,W,C), got {arr.shape}")

    # CHW -> HWC if needed
    if arr.shape[-1] not in (1, 3, 4) and arr.shape[1] in (1, 3, 4):
        arr = np.transpose(arr, (0, 2, 3, 1))

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]

    if arr.dtype == np.uint8:
        return arr

    arr = arr.astype(np.float32)
    # Heuristic: if floats are in [0,1], scale up
    if arr.max() <= 1.0:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / float(h)
    target_w = max(1, int(round(w * scale)))
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

def upscale(img: np.ndarray, scale: float = 2.0) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(
        img,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_CUBIC,  # better for upscaling
    )


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]

    # scale text + box relative to image size
    font_scale = max(0.5, h / 300)
    thickness = max(1, int(h / 200))

    box_h = int(40 * font_scale)
    box_w = int(260 * font_scale)

    cv2.rectangle(out, (0, 0), (box_w, box_h), (0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        text,
        (int(10 * font_scale), int(25 * font_scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return out


def make_side_by_side(third: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    # 🔥 STEP 1: UPSCALE FIRST
    third = upscale(third, scale=2.0)
    wrist = upscale(wrist, scale=2.0)

    # 🔥 STEP 2: match heights AFTER upscale
    target_h = max(third.shape[0], wrist.shape[0])
    third = resize_to_height(third, target_h)
    wrist = resize_to_height(wrist, target_h)

    # 🔥 STEP 3: add labels AFTER scaling
    third = add_label(third, "3rd person")
    wrist = add_label(wrist, "wrist")

    pad = np.full((target_h, 16, 3), 30, dtype=np.uint8)
    return np.concatenate([third, pad, wrist], axis=1)


def get_demo_group(f: h5py.File, demo_name: str | None) -> tuple[str, h5py.Group]:
    demos = list_demos(f)
    if demo_name is None:
        demo_name = demos[0]
    if demo_name not in f["data"]:
        raise KeyError(f"Demo '{demo_name}' not found. Available examples: {demos[:10]}")
    return demo_name, f["data"][demo_name]


def load_views(
    demo_group: h5py.Group,
    third_key: str,
    wrist_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    obs = find_obs_group(demo_group)

    if third_key not in obs:
        raise KeyError(f"Missing obs key '{third_key}'. Available keys: {list(obs.keys())}")
    if wrist_key not in obs:
        raise KeyError(f"Missing obs key '{wrist_key}'. Available keys: {list(obs.keys())}")

    third = to_uint8_rgb(obs[third_key][()])
    wrist = to_uint8_rgb(obs[wrist_key][()])

    n = min(len(third), len(wrist))
    return third[:n], wrist[:n]


def write_video(frames: list[np.ndarray], out_path: Path, fps: float) -> None:
    if not frames:
        raise ValueError("No frames to write.")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {out_path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

def save_pngs(frames: list[np.ndarray], demo_name: str) -> None:
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir / f"{demo_name}_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, frame in enumerate(frames):
        path = out_dir / f"{i:06d}.png"
        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    print(f"Saved {len(frames)} PNGs to {out_dir}")

def interactive_play(frames: list[np.ndarray], fps: float, title: str) -> None:
    delay_ms = max(1, int(round(1000.0 / fps)))
    paused = False
    i = 0

    while True:
        if not paused:
            frame = frames[i]
            i = min(i + 1, len(frames) - 1)

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(title, bgr)
        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("r"):
            i = 0
            paused = False
        elif key == ord("n"):
            i = min(i + 1, len(frames) - 1)
            paused = True
            frame = frames[i]
        elif key == ord("p"):
            i = max(i - 1, 0)
            paused = True
            frame = frames[i]

        if i == len(frames) - 1 and not paused:
            paused = True

    cv2.destroyAllWindows()


def main() -> int:
    args = parse_args()

    if not args.hdf5_path.exists():
        print(f"File not found: {args.hdf5_path}", file=sys.stderr)
        return 1

    with h5py.File(args.hdf5_path, "r") as f:
        demo_name, demo_group = get_demo_group(f, args.demo)
        third, wrist = load_views(demo_group, args.third_key, args.wrist_key)

    if args.max_frames is not None:
        third = third[: args.max_frames]
        wrist = wrist[: args.max_frames]

    frames = [make_side_by_side(a, b) for a, b in zip(third, wrist)]

    print(f"Demo: {demo_name}")
    print(f"Frames: {len(frames)}")
    print(f"Third-person key: {args.third_key}")
    print(f"Wrist key: {args.wrist_key}")

    if args.save is not None:
        # keep video option
        write_video(frames, args.save, args.fps)
        print(f"Saved video to {args.save}")
    else:
        # save PNGs by default
        save_pngs(frames, demo_name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())