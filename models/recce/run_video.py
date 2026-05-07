from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_image import (
    DEFAULT_CHECKPOINT,
    RECCE_IMAGE_SIZE,
    fail,
    fakeness_from_logits,
    load_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RECCE on sampled frames from one video.")
    parser.add_argument("--video", type=Path, required=True, help="Path to the input video.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Optional RECCE detector checkpoint.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=32,
        help="How many frames to sample from the video.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="How many sampled frames to score at once.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference instead of CUDA.",
    )
    return parser.parse_args()


def sample_video_frames(cv2_module, np_module, video_path: Path, requested_count: int) -> list[tuple[int, object]]:
    requested_count = max(requested_count, 1)
    capture = cv2_module.VideoCapture(str(video_path))
    if not capture.isOpened():
        fail(f"Could not open video: {video_path}")

    frames = []
    total_frames = int(capture.get(cv2_module.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames > 0:
        sample_count = min(requested_count, total_frames)
        step = total_frames / sample_count
        frame_indices = sorted({min(int(index * step), total_frames - 1) for index in range(sample_count)})
        for frame_index in frame_indices:
            capture.set(cv2_module.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append((frame_index, frame))
    else:
        frame_index = 0
        while len(frames) < requested_count:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frames.append((frame_index, frame))
            frame_index += 1

    capture.release()
    return frames


def read_video_info(cv2_module, video_path: Path) -> dict[str, object]:
    capture = cv2_module.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {"total_frames": None, "fps": None, "width": None, "height": None}

    total_frames = int(capture.get(cv2_module.CAP_PROP_FRAME_COUNT) or 0) or None
    fps = float(capture.get(cv2_module.CAP_PROP_FPS) or 0.0) or None
    width = int(capture.get(cv2_module.CAP_PROP_FRAME_WIDTH) or 0) or None
    height = int(capture.get(cv2_module.CAP_PROP_FRAME_HEIGHT) or 0) or None
    capture.release()
    return {
        "total_frames": total_frames,
        "fps": None if fps is None else round(fps, 3),
        "width": width,
        "height": height,
    }


def frame_timestamp(frame_index: int, fps: float | None) -> float | None:
    if not fps or fps <= 0:
        return None
    return frame_index / fps


def build_details(
    sampled_frames: list[tuple[int, object]],
    scores: list[float],
    video: dict[str, object],
    requested_frames: int,
) -> dict[str, object]:
    fps = video.get("fps") if isinstance(video.get("fps"), (int, float)) else None
    total_frames = video.get("total_frames") if isinstance(video.get("total_frames"), int) else None
    ranked = sorted(zip(sampled_frames, scores), key=lambda item: item[1], reverse=True)
    top_frames = []
    for (frame_index, _), frame_score in ranked[:5]:
        seconds = frame_timestamp(frame_index, fps)
        top_frames.append(
            {
                "frame": frame_index,
                "time_seconds": None if seconds is None else round(seconds, 3),
                "score": round(float(frame_score), 6),
                "reason": "The wrapper reports this frame because it had the highest RECCE score among the sampled frames.",
            }
        )

    suspicious_windows = []
    if top_frames:
        peak = top_frames[0]
        peak_frame = int(peak["frame"])
        radius = 5
        start_frame = max(0, peak_frame - radius)
        end_limit = total_frames - 1 if total_frames else peak_frame + radius
        end_frame = min(end_limit, peak_frame + radius)
        start_time = frame_timestamp(start_frame, fps)
        end_time = frame_timestamp(end_frame, fps)
        suspicious_windows.append(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time_seconds": None if start_time is None else round(start_time, 3),
                "end_time_seconds": None if end_time is None else round(end_time, 3),
                "peak_frame": peak_frame,
                "peak_time_seconds": peak.get("time_seconds"),
                "peak_score": peak.get("score"),
            }
        )

    return {
        "video": video,
        "selection": {
            "mode": "uniform",
            "requested_frames": requested_frames,
            "attempted_frames": requested_frames,
            "decoded_frames": len(sampled_frames),
            "scored_frames": len(scores),
        },
        "top_frames": top_frames,
        "suspicious_windows": suspicious_windows,
    }


def main() -> int:
    args = parse_args()
    video_path = args.video.resolve()
    if not video_path.is_file():
        fail(f"Input video not found: {video_path}")
    if args.frames < 1:
        fail("--frames must be at least 1.")
    if args.batch_size < 1:
        fail("--batch-size must be at least 1.")

    try:
        import cv2
        import numpy as np
        import torch
        from PIL import Image
        from torchvision import transforms
    except Exception as exc:
        fail(f"Missing RECCE runtime dependency: {exc}", unavailable=True)

    video_info = read_video_info(cv2, video_path)
    sampled_frames = sample_video_frames(cv2, np, video_path, args.frames)
    if not sampled_frames:
        fail(f"RECCE could not decode any frames from {video_path}.")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_model(torch, args.checkpoint.resolve(), device)
    transform = transforms.Compose(
        [
            transforms.Resize((RECCE_IMAGE_SIZE, RECCE_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Lambda(lambda tensor: tensor * 2.0 - 1.0),
        ]
    )
    tensors = [
        transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        for _, frame in sampled_frames
    ]

    scores = []
    with torch.no_grad():
        for start in range(0, len(tensors), args.batch_size):
            batch = torch.stack(tensors[start : start + args.batch_size]).to(device)
            logits = model(batch)
            if logits.dim() == 1:
                scores.append(fakeness_from_logits(torch, logits))
            else:
                scores.extend(
                    fakeness_from_logits(torch, logits[index : index + 1])
                    for index in range(logits.shape[0])
                )

    if not scores:
        fail(f"RECCE did not produce any predictions for {video_path}.")

    score = float(np.mean(scores))
    details = build_details(sampled_frames, scores, video_info, max(args.frames, 1))
    print(f"RECCE video fakeness: {score:.4f}")
    print(f"Frames scored: {len(scores)}")
    print("DETAIL_JSON:" + json.dumps(details, separators=(",", ":")))
    print(f"SCORE:{score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
