from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


MODEL_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AltFreezing on one image.")
    parser.add_argument("--image", type=Path, required=True, help="Path to the input image.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional checkpoint file. Defaults to models/altfreezing/checkpoints/*.pth.",
    )
    parser.add_argument(
        "--config",
        default="i3d_ori.yaml",
        help="Config filename inside repo/AltFreezing-main/setting/.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=32,
        help="How many identical frames to write into the temporary clip.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=8.0,
        help="Frame rate for the temporary clip.",
    )
    return parser.parse_args()


def fail(message: str, exit_code: int = 1) -> "NoReturn":
    print(f"ERROR:{message}", file=sys.stderr)
    raise SystemExit(exit_code)


def build_video_writer(cv2_module, output_path: Path, width: int, height: int, fps: float):
    for codec in ("MJPG", "XVID", "mp4v"):
        writer = cv2_module.VideoWriter(
            str(output_path),
            cv2_module.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer
    fail(f"Could not create a temporary video file at {output_path}.")


def main() -> int:
    args = parse_args()
    image_path = args.image.resolve()
    if not image_path.is_file():
        fail(f"Input image not found: {image_path}")

    try:
        import cv2
    except Exception as exc:
        fail(f"Missing AltFreezing runtime dependency: {exc}")

    frame = cv2.imread(str(image_path))
    if frame is None:
        fail(f"Could not read image: {image_path}")

    print(
        "AltFreezing note: the image is duplicated into a short clip because the original model is video-based."
    )

    with tempfile.TemporaryDirectory(prefix="altfreezing_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        video_path = temp_dir_path / "image_clip.avi"
        writer = build_video_writer(cv2, video_path, frame.shape[1], frame.shape[0], args.fps)
        try:
            for _ in range(max(args.frames, 1)):
                writer.write(frame)
        finally:
            writer.release()

        command = [
            sys.executable,
            str(MODEL_ROOT / "run_video.py"),
            "--video",
            str(video_path),
            "--config",
            args.config,
            "--skip-output",
        ]
        if args.checkpoint:
            command.extend(["--checkpoint", str(args.checkpoint.resolve())])

        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.stdout:
            print(completed.stdout.strip())
        if completed.stderr:
            print(completed.stderr.strip(), file=sys.stderr)
        return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
