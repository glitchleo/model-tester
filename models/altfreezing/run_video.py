from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


MODEL_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = MODEL_ROOT.parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AltFreezing on one video.")
    parser.add_argument("--video", type=Path, required=True, help="Path to the input video.")
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
        "--max-frame",
        type=int,
        default=400,
        help="Maximum number of frames to sample from the input video.",
    )
    parser.add_argument(
        "--optimal-threshold",
        type=float,
        default=0.04,
        help="Threshold used only when writing the annotated output video.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=WORKSPACE_ROOT / "outputs" / "altfreezing",
        help="Directory for AltFreezing output videos.",
    )
    parser.add_argument(
        "--skip-output",
        action="store_true",
        help="Skip writing the annotated output video.",
    )
    return parser.parse_args()


def fail(message: str, *, unavailable: bool = False, exit_code: int = 1) -> "NoReturn":
    prefix = "UNAVAILABLE" if unavailable else "ERROR"
    stream = sys.stdout if unavailable else sys.stderr
    print(f"{prefix}:{message}", file=stream)
    raise SystemExit(0 if unavailable else exit_code)


def find_repo_root(repo_dir: Path, marker: str) -> Path:
    if not repo_dir.exists():
        fail(f"Missing repo directory: {repo_dir}", unavailable=True)
    candidates = [repo_dir]
    candidates.extend(sorted(path for path in repo_dir.iterdir() if path.is_dir()))
    for candidate in candidates:
        if (candidate / marker).exists():
            return candidate
    fail(f"Could not find the AltFreezing repo root under {repo_dir}", unavailable=True)


def resolve_checkpoint_path(checkpoints_dir: Path, provided: Path | None) -> Path:
    if provided:
        resolved = provided.resolve()
        if not resolved.is_file():
            fail(f"Checkpoint file not found: {resolved}")
        return resolved

    candidates = sorted(checkpoints_dir.glob("*.pth"))
    if not candidates:
        fail(
            "No AltFreezing checkpoint file was found in models/altfreezing/checkpoints.",
            unavailable=True,
        )
    return candidates[0]


def torch_load_compat(torch_module, path, map_location=None) -> object:
    try:
        return torch_module.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch_module.load(path, map_location=map_location)


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


def frame_window(
    frame_index: int,
    fps: float | None,
    total_frames: int | None,
    radius: int,
) -> dict[str, object]:
    start_frame = max(0, frame_index - radius)
    end_limit = total_frames - 1 if total_frames else frame_index + radius
    end_frame = min(end_limit, frame_index + radius)
    start_time = frame_timestamp(start_frame, fps)
    end_time = frame_timestamp(end_frame, fps)
    peak_time = frame_timestamp(frame_index, fps)
    return {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_time_seconds": None if start_time is None else round(start_time, 3),
        "end_time_seconds": None if end_time is None else round(end_time, 3),
        "peak_frame": frame_index,
        "peak_time_seconds": None if peak_time is None else round(peak_time, 3),
    }


def serializable_box(box: object) -> list[int] | None:
    if box is None:
        return None
    values = box.tolist() if hasattr(box, "tolist") else box
    return [int(value) for value in values]


def build_details(
    np_module,
    args: argparse.Namespace,
    video_info: dict[str, object],
    frame_res: dict[int, list[float]],
    frame_boxes: dict[int, object],
    decoded_frame_count: int,
    clip_count: int,
    clip_size: int,
) -> dict[str, object]:
    fps = video_info.get("fps")
    total_frames = video_info.get("total_frames")
    frame_scores = []
    for frame_index, scores in sorted(frame_res.items()):
        frame_score = float(np_module.mean(scores))
        timestamp = frame_timestamp(frame_index, fps if isinstance(fps, (int, float)) else None)
        box = frame_boxes.get(frame_index)
        frame_scores.append(
            {
                "frame": frame_index,
                "time_seconds": None if timestamp is None else round(timestamp, 3),
                "score": round(frame_score, 6),
                "clip_count": len(scores),
                "box": serializable_box(box),
                "reason": "mean AltFreezing score from temporal clips containing this tracked face frame",
            }
        )

    top_frames = sorted(frame_scores, key=lambda item: item["score"], reverse=True)[:5]
    radius = max(1, clip_size // 2)
    suspicious_windows = []
    for frame in top_frames[:3]:
        window = frame_window(
            int(frame["frame"]),
            fps if isinstance(fps, (int, float)) else None,
            total_frames if isinstance(total_frames, int) else None,
            radius,
        )
        window.update(
            {
                "peak_score": frame["score"],
                "clip_count_at_peak": frame["clip_count"],
                "reason": "this window is centered on one of the highest AltFreezing temporal clip scores",
            }
        )
        suspicious_windows.append(window)

    return {
        "model": "altfreezing",
        "selection": {
            "mode": "native_video",
            "max_frame": args.max_frame,
            "decoded_frames": decoded_frame_count,
            "scored_frames": len(frame_scores),
            "clips_scored": clip_count,
            "clip_size": clip_size,
        },
        "video": video_info,
        "top_frames": top_frames,
        "suspicious_windows": suspicious_windows,
        "frames": frame_scores,
        "explanation": (
            "Scores come from temporal face-track clips. The frame examples are high-scoring frames that appeared "
            "inside suspicious motion/appearance clips, not pixel-level attributions."
        ),
    }


def format_time(seconds: float | None) -> str:
    if seconds is None:
        return "unknown time"
    minutes = int(seconds // 60)
    remaining = seconds - minutes * 60
    return f"{minutes:02d}:{remaining:05.2f}"


def main() -> int:
    args = parse_args()
    video_path = args.video.resolve()
    if not video_path.is_file():
        fail(f"Input video not found: {video_path}")

    checkpoint_path = resolve_checkpoint_path(MODEL_ROOT / "checkpoints", args.checkpoint)
    repo_root = find_repo_root(MODEL_ROOT / "repo", "config.py")
    sys.path.insert(0, str(repo_root))
    cache_dir = MODEL_ROOT / ".torch-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(cache_dir)

    try:
        import cv2
        import numpy as np
        import torch
        from tqdm import tqdm

        from config import config as cfg
        from test_tools.common import detect_all, grab_all_frames
        from test_tools.ct.operations import find_longest, multiple_tracking
        from test_tools.faster_crop_align_xray import FasterCropAlignXRay
        from test_tools.supply_writer import SupplyWriter
        from test_tools.utils import get_crop_box
        from utils.plugin_loader import PluginLoader
    except Exception as exc:
        fail(f"Missing AltFreezing runtime dependency: {exc}")

    video_info = read_video_info(cv2, video_path)

    if not torch.cuda.is_available():
        fail(
            "AltFreezing needs a CUDA-enabled environment because its bundled video detector stack is GPU-bound."
        )

    cfg.init_with_yaml()
    cfg.update_with_yaml(args.config)
    cfg.freeze()

    device = torch.device("cuda")
    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255], device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255], device=device).view(1, 3, 1, 1, 1)

    classifier = PluginLoader.get_classifier(cfg.classifier_type)()
    classifier.to(device)
    classifier.eval()
    classifier.load(str(checkpoint_path))

    crop_align_func = FasterCropAlignXRay(cfg.imsize)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{video_path.stem}.avi"

    cache_file = video_path.with_name(f"{video_path.stem}_{args.max_frame}.pth")
    if cache_file.exists():
        detect_res, all_lm68 = torch_load_compat(torch, cache_file, map_location="cpu")
        frames = grab_all_frames(str(video_path), max_size=args.max_frame, cvt=True)
    else:
        detect_res, all_lm68, frames = detect_all(
            str(video_path),
            return_frames=True,
            max_size=args.max_frame,
        )
        torch.save((detect_res, all_lm68), cache_file)

    if not frames:
        fail(f"AltFreezing could not decode any frames from {video_path}.")
    if len(all_lm68) != len(detect_res):
        fail("Face detection output is inconsistent with the landmark results.")

    shape = frames[0].shape[:2]
    merged_detect_res = []
    for faces, faces_lm68 in zip(detect_res, all_lm68):
        merged_faces = []
        for (box, lm5, score), face_lm68 in zip(faces, faces_lm68):
            merged_faces.append((box, lm5, face_lm68, score))
        merged_detect_res.append(merged_faces)
    detect_res = merged_detect_res

    tracks = multiple_tracking(detect_res)
    track_ranges = [(0, len(detect_res))] * len(tracks)
    if len(tracks) == 0:
        track_ranges, tracks = find_longest(detect_res)
    if len(tracks) == 0:
        fail(f"AltFreezing could not track a face in {video_path}.")

    data_storage = {}
    frame_boxes = {}
    super_clips = []

    for track_index, ((start, end), track) in enumerate(zip(track_ranges, tracks)):
        super_clips.append(len(track))
        for item_index, (face, frame_index) in enumerate(zip(track, range(start, end))):
            box, lm5, lm68 = face[:3]
            big_box = get_crop_box(shape, box, scale=0.5)
            top_left = big_box[:2][None, :]
            new_lm5 = lm5 - top_left
            new_lm68 = lm68 - top_left
            new_box = (box.reshape(2, 2) - top_left).reshape(-1)
            info = (new_box, new_lm5, new_lm68, big_box)

            x1, y1, x2, y2 = big_box
            cropped = frames[frame_index][y1:y2, x1:x2]
            base_key = f"{track_index}_{item_index}_"
            data_storage[f"{base_key}img"] = cropped
            data_storage[f"{base_key}ldm"] = info
            data_storage[f"{base_key}idx"] = frame_index
            frame_boxes[frame_index] = np.rint(box).astype(int)

    clips_for_video = []
    clip_size = cfg.clip_size
    pad_length = clip_size - 1

    for super_clip_index, super_clip_size in enumerate(super_clips):
        inner_index = list(range(super_clip_size))
        if super_clip_size < clip_size:
            post_module = inner_index[1:-1][::-1] + inner_index
            if not post_module:
                post_module = inner_index
            post_module = (post_module * (pad_length // len(post_module) + 1))[:pad_length]

            pre_module = inner_index + inner_index[1:-1][::-1]
            if not pre_module:
                pre_module = inner_index
            pre_module = (pre_module * (pad_length // len(pre_module) + 1))[-pad_length:]
            inner_index = pre_module + inner_index + post_module

        clip_count = len(inner_index)
        frame_range = [
            inner_index[index : index + clip_size]
            for index in range(clip_count)
            if index + clip_size <= clip_count
        ]
        for indices in frame_range:
            clips_for_video.append([(super_clip_index, value) for value in indices])

    preds = []
    frame_res = {}

    for clip in tqdm(clips_for_video, desc="AltFreezing"):
        images = [data_storage[f"{i}_{j}_img"] for i, j in clip]
        landmarks = [data_storage[f"{i}_{j}_ldm"] for i, j in clip]
        frame_ids = [data_storage[f"{i}_{j}_idx"] for i, j in clip]

        _, images_align = crop_align_func(landmarks, images)
        batch = torch.as_tensor(images_align, dtype=torch.float32, device=device).permute(3, 0, 1, 2)
        batch = batch.unsqueeze(0).sub(mean).div(std)

        with torch.no_grad():
            output = classifier(batch)
        pred = float(torch.sigmoid(output["final_output"]).mean().item())
        preds.append(pred)
        for frame_id in frame_ids:
            frame_res.setdefault(frame_id, []).append(pred)

    if not preds:
        fail(f"AltFreezing did not produce any predictions for {video_path}.")

    score = float(np.mean(preds))
    details = build_details(
        np,
        args,
        video_info,
        frame_res,
        frame_boxes,
        len(frames),
        len(preds),
        clip_size,
    )
    top_frame = details["top_frames"][0] if details["top_frames"] else None
    print(f"AltFreezing fakeness: {score:.4f}")
    if top_frame:
        print(
            "Most suspicious frame: "
            f"{top_frame['frame']} at {format_time(top_frame['time_seconds'])}, "
            f"score {top_frame['score']:.4f}, clips {top_frame['clip_count']}"
        )
    print(f"DETAIL_JSON:{json.dumps(details, separators=(',', ':'))}")
    print(f"SCORE:{score:.6f}")

    if not args.skip_output:
        boxes = []
        scores = []
        for frame_index in range(len(frames)):
            if frame_index in frame_res:
                scores.append(float(np.mean(frame_res[frame_index])))
                boxes.append(frame_boxes[frame_index])
            else:
                scores.append(None)
                boxes.append(None)
        SupplyWriter(str(video_path), str(out_file), args.optimal_threshold).run(frames, scores, boxes)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
