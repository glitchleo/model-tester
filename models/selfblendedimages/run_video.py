from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from run_image import (
    MODEL_ROOT,
    fail,
    find_repo_root,
    load_retinaface_model,
    resolve_weight_path,
    torch_load_compat,
)


@dataclass(frozen=True)
class VideoInfo:
    total_frames: int
    fps: float
    width: int
    height: int


@dataclass(frozen=True)
class FrameScore:
    frame_index: int
    time_seconds: float | None
    score: float | None
    face_count: int
    pass_name: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SelfBlendedImages on sampled frames from one video.")
    parser.add_argument("--video", type=Path, required=True, help="Path to the input video.")
    parser.add_argument(
        "--weight",
        type=Path,
        help="Optional checkpoint file. Defaults to models/selfblendedimages/weights/*.tar.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=32,
        help="Maximum number of frames to analyze in limited modes.",
    )
    parser.add_argument(
        "--frame-mode",
        choices=["adaptive", "uniform", "all"],
        default="uniform",
        help=(
            "Frame selection strategy. 'adaptive' does a coarse interval pass and then samples near the "
            "highest-scoring moments; 'uniform' samples evenly; 'all' scores every decoded frame."
        ),
    )
    parser.add_argument(
        "--coarse-interval",
        type=int,
        default=20,
        help="Frame interval used by the first pass in adaptive mode.",
    )
    parser.add_argument(
        "--refine-window",
        type=int,
        default=8,
        help="How many neighboring frame positions on each side of a hotspot may be refined.",
    )
    parser.add_argument(
        "--hotspots",
        type=int,
        default=3,
        help="How many high-scoring coarse frames to refine around.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Maximum number of detected face crops to score per CUDA/CPU inference batch.",
    )
    parser.add_argument(
        "--face-detector-max-size",
        type=int,
        default=960,
        help="Cap the longest side used by the RetinaFace detector. Use 0 to keep the source video size.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference instead of CUDA.",
    )
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be at least 1.")
    if args.coarse_interval < 1:
        parser.error("--coarse-interval must be at least 1.")
    if args.refine_window < 0:
        parser.error("--refine-window must be 0 or greater.")
    if args.hotspots < 1:
        parser.error("--hotspots must be at least 1.")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")
    if args.face_detector_max_size < 0:
        parser.error("--face-detector-max-size must be 0 or greater.")
    return args


def read_video_info(cv2_module, video_path: Path) -> VideoInfo:
    capture = cv2_module.VideoCapture(str(video_path))
    if not capture.isOpened():
        fail(f"Could not open video: {video_path}")

    total_frames = int(capture.get(cv2_module.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2_module.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2_module.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2_module.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return VideoInfo(total_frames=total_frames, fps=fps, width=width, height=height)


def unique_sorted_indices(indices: Iterable[int | float], total_frames: int) -> list[int]:
    if total_frames <= 0:
        return []
    return sorted({min(max(int(round(index)), 0), total_frames - 1) for index in indices})


def uniform_indices(np_module, total_frames: int, requested_count: int) -> list[int]:
    if total_frames <= 0:
        return []
    sample_count = min(max(requested_count, 1), total_frames)
    step = total_frames / sample_count
    return sorted({min(int(index * step), total_frames - 1) for index in range(sample_count)})


def anchor_indices(total_frames: int) -> list[int]:
    last_frame = total_frames - 1
    return unique_sorted_indices([0, last_frame * 0.25, last_frame * 0.5, last_frame * 0.75, last_frame], total_frames)


def thin_indices(np_module, indices: list[int], requested_count: int) -> list[int]:
    if len(indices) <= requested_count:
        return indices
    positions = uniform_indices(np_module, len(indices), requested_count)
    return [indices[position] for position in positions]


def adaptive_coarse_indices(
    np_module,
    total_frames: int,
    frame_limit: int,
    coarse_interval: int,
) -> list[int]:
    frame_limit = min(frame_limit, total_frames)
    anchors = anchor_indices(total_frames)
    if frame_limit <= len(anchors):
        return thin_indices(np_module, anchors, frame_limit)

    interval_indices = list(range(0, total_frames, coarse_interval))
    if interval_indices[-1] != total_frames - 1:
        interval_indices.append(total_frames - 1)

    candidates = unique_sorted_indices([*anchors, *interval_indices], total_frames)
    if frame_limit <= 10:
        coarse_budget = frame_limit
    else:
        coarse_budget = max(len(anchors), min(len(candidates), frame_limit // 2))
    return thin_indices(np_module, candidates, coarse_budget)


def adaptive_refine_indices(
    total_frames: int,
    frame_scores: list[FrameScore],
    attempted_indices: set[int],
    remaining_budget: int,
    refine_window: int,
    hotspot_count: int,
) -> list[int]:
    if total_frames <= 0 or remaining_budget <= 0 or refine_window <= 0:
        return []

    scored_frames = [frame_score for frame_score in frame_scores if frame_score.score is not None]
    hotspots = sorted(scored_frames, key=lambda item: item.score or 0.0, reverse=True)[:hotspot_count]
    if not hotspots:
        return []

    candidate_set = set()
    hotspot_positions = [(rank, item.frame_index) for rank, item in enumerate(hotspots)]
    for _, frame_index in hotspot_positions:
        start_frame = max(0, frame_index - refine_window)
        end_frame = min(total_frames - 1, frame_index + refine_window)
        candidate_set.update(range(start_frame, end_frame + 1))

    candidate_set.difference_update(attempted_indices)

    def priority(frame_index: int) -> tuple[int, int, int]:
        rank, hotspot_frame = min(
            hotspot_positions,
            key=lambda item: (abs(frame_index - item[1]), item[0]),
        )
        return (abs(frame_index - hotspot_frame), rank, frame_index)

    return sorted(sorted(candidate_set, key=priority)[:remaining_budget])


def iter_frames_by_indices(cv2_module, video_path: Path, frame_indices: Iterable[int]) -> Iterator[tuple[int, object]]:
    capture = cv2_module.VideoCapture(str(video_path))
    if not capture.isOpened():
        fail(f"Could not open video: {video_path}")

    current_frame = -1
    try:
        for frame_index in sorted(frame_indices):
            if frame_index != current_frame + 1:
                capture.set(cv2_module.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            current_frame = frame_index
            if ok and frame is not None:
                yield frame_index, frame
    finally:
        capture.release()


def iter_sequential_frames(cv2_module, video_path: Path, limit: int | None = None) -> Iterator[tuple[int, object]]:
    capture = cv2_module.VideoCapture(str(video_path))
    if not capture.isOpened():
        fail(f"Could not open video: {video_path}")

    frame_index = 0
    try:
        while limit is None or frame_index < limit:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            yield frame_index, frame
            frame_index += 1
    finally:
        capture.release()


def frame_timestamp(frame_index: int, fps: float) -> float | None:
    if fps <= 0:
        return None
    return frame_index / fps


def is_cuda_oom(torch_module, exc: Exception) -> bool:
    oom_type = getattr(torch_module, "OutOfMemoryError", None)
    return (oom_type is not None and isinstance(exc, oom_type)) or "out of memory" in str(exc).lower()


def clear_cuda_cache(torch_module, device) -> None:
    if getattr(device, "type", None) == "cuda":
        torch_module.cuda.empty_cache()


def extract_faces_from_frame(cv2_module, np_module, frame, face_detector, crop_face_func) -> list[object]:
    faces = face_detector.predict_jsons(frame)
    cropped_faces = []
    for face in faces:
        bbox_values = face.get("bbox")
        if not bbox_values or len(bbox_values) != 4:
            continue

        x0, y0, x1, y1 = bbox_values
        if x1 <= x0 or y1 <= y0:
            continue

        bbox = np_module.array([[x0, y0], [x1, y1]])
        try:
            cropped = crop_face_func(
                frame,
                None,
                bbox,
                False,
                crop_by_bbox=True,
                only_img=True,
                phase="test",
            )
        except Exception:
            continue
        if cropped.size == 0:
            continue

        cropped_faces.append(cv2_module.resize(cropped, dsize=(380, 380)).transpose((2, 0, 1)))
    return cropped_faces


def predict_face_scores(torch_module, np_module, detector, device, faces: list[object], batch_size: int) -> list[float]:
    scores = []
    for start_index in range(0, len(faces), batch_size):
        face_batch = faces[start_index : start_index + batch_size]
        try:
            with torch_module.no_grad():
                batch = torch_module.tensor(np_module.stack(face_batch), device=device).float() / 255.0
                batch_scores = detector(batch).softmax(1)[:, 1].detach().cpu().numpy().tolist()
        except Exception as exc:
            clear_cuda_cache(torch_module, device)
            if is_cuda_oom(torch_module, exc) and len(face_batch) > 1:
                scores.extend(predict_face_scores(torch_module, np_module, detector, device, face_batch, 1))
                continue
            if is_cuda_oom(torch_module, exc):
                fail(
                    "SelfBlendedImages ran out of CUDA memory while scoring one face crop. "
                    "Try --cpu or lower --face-detector-max-size."
                )
            raise
        finally:
            if "batch" in locals():
                del batch
        clear_cuda_cache(torch_module, device)
        scores.extend(float(score) for score in batch_scores)
    return scores


def score_video_frames(
    cv2_module,
    np_module,
    torch_module,
    frames: Iterable[tuple[int, object]],
    video_info: VideoInfo,
    detector,
    face_detector,
    crop_face_func,
    device,
    batch_size: int,
    pass_name: str,
) -> list[FrameScore]:
    frame_scores = []
    for frame_index, frame in frames:
        try:
            rgb_frame = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2RGB)
            face_list = extract_faces_from_frame(cv2_module, np_module, rgb_frame, face_detector, crop_face_func)
        except Exception as exc:
            clear_cuda_cache(torch_module, device)
            if is_cuda_oom(torch_module, exc):
                fail(
                    "SelfBlendedImages ran out of CUDA memory while detecting faces. "
                    "Try --face-detector-max-size 640, --batch-size 1, or --cpu."
                )
            raise

        if not face_list:
            frame_scores.append(
                FrameScore(
                    frame_index=frame_index,
                    time_seconds=frame_timestamp(frame_index, video_info.fps),
                    score=None,
                    face_count=0,
                    pass_name=pass_name,
                    reason="no face detected in this frame",
                )
            )
            continue

        scores = predict_face_scores(torch_module, np_module, detector, device, face_list, batch_size)
        frame_scores.append(
            FrameScore(
                frame_index=frame_index,
                time_seconds=frame_timestamp(frame_index, video_info.fps),
                score=max(scores) if scores else None,
                face_count=len(scores),
                pass_name=pass_name,
                reason="highest SelfBlendedImages face-crop fakeness score in this frame",
            )
        )
    return frame_scores


def serializable_frame(frame_score: FrameScore) -> dict[str, object]:
    return {
        "frame": frame_score.frame_index,
        "time_seconds": None if frame_score.time_seconds is None else round(frame_score.time_seconds, 3),
        "score": None if frame_score.score is None else round(frame_score.score, 6),
        "face_count": frame_score.face_count,
        "pass": frame_score.pass_name,
        "reason": frame_score.reason,
    }


def suspicious_windows(
    frame_scores: list[FrameScore],
    video_info: VideoInfo,
    refine_window: int,
    window_count: int,
) -> list[dict[str, object]]:
    scored_frames = [frame_score for frame_score in frame_scores if frame_score.score is not None]
    top_frames = sorted(scored_frames, key=lambda item: item.score or 0.0, reverse=True)[:window_count]
    raw_windows = []
    radius = max(1, refine_window)
    for frame_score in top_frames:
        if video_info.total_frames > 0:
            start_frame = max(0, frame_score.frame_index - radius)
            end_frame = min(video_info.total_frames - 1, frame_score.frame_index + radius)
        else:
            start_frame = max(0, frame_score.frame_index - radius)
            end_frame = frame_score.frame_index + radius
        raw_windows.append(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "peak_frame": frame_score.frame_index,
                "peak_score": frame_score.score,
                "face_count": frame_score.face_count,
                "reason": "this window is centered on one of the highest per-frame max-face scores",
            }
        )

    merged = []
    for window in sorted(raw_windows, key=lambda item: item["start_frame"]):
        if merged and window["start_frame"] <= merged[-1]["end_frame"]:
            previous = merged[-1]
            previous["end_frame"] = max(previous["end_frame"], window["end_frame"])
            if (window["peak_score"] or 0.0) > (previous["peak_score"] or 0.0):
                previous["peak_frame"] = window["peak_frame"]
                previous["peak_score"] = window["peak_score"]
                previous["face_count"] = window["face_count"]
        else:
            merged.append(window)

    formatted = []
    for window in sorted(merged, key=lambda item: item["peak_score"] or 0.0, reverse=True)[:window_count]:
        start_time = frame_timestamp(window["start_frame"], video_info.fps)
        end_time = frame_timestamp(window["end_frame"], video_info.fps)
        formatted.append(
            {
                "start_frame": window["start_frame"],
                "end_frame": window["end_frame"],
                "start_time_seconds": None if start_time is None else round(start_time, 3),
                "end_time_seconds": None if end_time is None else round(end_time, 3),
                "peak_frame": window["peak_frame"],
                "peak_time_seconds": None
                if frame_timestamp(window["peak_frame"], video_info.fps) is None
                else round(frame_timestamp(window["peak_frame"], video_info.fps), 3),
                "peak_score": round(float(window["peak_score"]), 6),
                "face_count_at_peak": window["face_count"],
                "reason": window["reason"],
            }
        )
    return formatted


def build_details(
    args: argparse.Namespace,
    video_info: VideoInfo,
    frame_scores: list[FrameScore],
    attempted_indices: list[int],
    coarse_count: int,
    refined_count: int,
    fallback_reason: str | None = None,
) -> dict[str, object]:
    scored_frames = [frame_score for frame_score in frame_scores if frame_score.score is not None]
    top_frames = sorted(scored_frames, key=lambda item: item.score or 0.0, reverse=True)[:5]
    return {
        "model": "selfblendedimages",
        "selection": {
            "mode": args.frame_mode,
            "frame_limit": None if args.frame_mode == "all" else args.frames,
            "coarse_interval": args.coarse_interval,
            "refine_window": args.refine_window,
            "hotspots": args.hotspots,
            "attempted_frames": len(attempted_indices),
            "decoded_frames": len(frame_scores),
            "scored_frames": len(scored_frames),
            "coarse_frames": coarse_count,
            "refined_frames": refined_count,
            "fallback_reason": fallback_reason,
        },
        "video": {
            "total_frames": video_info.total_frames or None,
            "fps": round(video_info.fps, 3) if video_info.fps > 0 else None,
            "width": video_info.width or None,
            "height": video_info.height or None,
        },
        "top_frames": [serializable_frame(frame_score) for frame_score in top_frames],
        "suspicious_windows": suspicious_windows(frame_scores, video_info, args.refine_window, args.hotspots),
        "frames": [serializable_frame(frame_score) for frame_score in sorted(frame_scores, key=lambda item: item.frame_index)],
        "explanation": (
            "Scores are per-frame maxima across detected face crops. SelfBlendedImages does not provide pixel-level "
            "attribution here, so the explanation localizes suspicious times by high model scores on detected faces."
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

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    repo_root = find_repo_root(MODEL_ROOT / "repo", "src/inference/model.py")
    weight_path = resolve_weight_path(MODEL_ROOT / "weights", args.weight)
    inference_dir = repo_root / "src" / "inference"
    sys.path.insert(0, str(inference_dir))
    cache_dir = MODEL_ROOT / ".torch-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(cache_dir)

    try:
        import cv2
        import numpy as np
        import torch
        from efficientnet_pytorch import EfficientNet
        import retinaface.pre_trained_models as retina_models
        from preprocess import crop_face
    except Exception as exc:
        fail(f"Missing SelfBlendedImages runtime dependency: {exc}")

    video_info = read_video_info(cv2, video_path)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = torch_load_compat(torch, weight_path, device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        fail(f"Checkpoint format is not supported: {weight_path}")

    detector = torch.nn.Module()
    detector.net = EfficientNet.from_name("efficientnet-b4", num_classes=2)
    detector.forward = lambda x: detector.net(x)
    detector = detector.to(device)
    detector.load_state_dict(checkpoint["model"])
    detector.eval()

    source_max_size = max(video_info.width, video_info.height) or 960
    if args.face_detector_max_size > 0:
        max_size = min(source_max_size, args.face_detector_max_size)
    else:
        max_size = source_max_size
    face_detector = load_retinaface_model(
        retina_models,
        torch,
        cache_dir,
        max_size=max_size,
        device=device,
    )
    face_detector.eval()

    frame_scores: list[FrameScore]
    attempted_indices: list[int]
    coarse_count = 0
    refined_count = 0
    fallback_reason = None

    if video_info.total_frames > 0:
        if args.frame_mode == "all":
            selected_indices = list(range(video_info.total_frames))
            frame_scores = score_video_frames(
                cv2,
                np,
                torch,
                iter_frames_by_indices(cv2, video_path, selected_indices),
                video_info,
                detector,
                face_detector,
                crop_face,
                device,
                args.batch_size,
                "all",
            )
            attempted_indices = selected_indices
            coarse_count = len(selected_indices)
        elif args.frame_mode == "uniform":
            selected_indices = uniform_indices(np, video_info.total_frames, args.frames)
            frame_scores = score_video_frames(
                cv2,
                np,
                torch,
                iter_frames_by_indices(cv2, video_path, selected_indices),
                video_info,
                detector,
                face_detector,
                crop_face,
                device,
                args.batch_size,
                "uniform",
            )
            attempted_indices = selected_indices
            coarse_count = len(selected_indices)
        else:
            coarse_indices = adaptive_coarse_indices(
                np,
                video_info.total_frames,
                args.frames,
                args.coarse_interval,
            )
            coarse_scores = score_video_frames(
                cv2,
                np,
                torch,
                iter_frames_by_indices(cv2, video_path, coarse_indices),
                video_info,
                detector,
                face_detector,
                crop_face,
                device,
                args.batch_size,
                "coarse",
            )
            attempted_set = set(coarse_indices)
            remaining_budget = max(0, min(args.frames, video_info.total_frames) - len(attempted_set))
            refine_indices = adaptive_refine_indices(
                video_info.total_frames,
                coarse_scores,
                attempted_set,
                remaining_budget,
                args.refine_window,
                args.hotspots,
            )
            if len(refine_indices) < remaining_budget:
                fill_count = remaining_budget - len(refine_indices)
                fill_indices = [
                    frame_index
                    for frame_index in uniform_indices(np, video_info.total_frames, args.frames)
                    if frame_index not in attempted_set and frame_index not in refine_indices
                ][:fill_count]
                if fill_indices and not any(frame_score.score is not None for frame_score in coarse_scores):
                    fallback_reason = "No face was scored in the coarse pass; remaining budget used uniform frames."
                refine_indices = sorted([*refine_indices, *fill_indices])
            refined_scores = score_video_frames(
                cv2,
                np,
                torch,
                iter_frames_by_indices(cv2, video_path, refine_indices),
                video_info,
                detector,
                face_detector,
                crop_face,
                device,
                args.batch_size,
                "refined",
            )
            frame_scores = coarse_scores + refined_scores
            attempted_indices = sorted([*coarse_indices, *refine_indices])
            coarse_count = len(coarse_indices)
            refined_count = len(refine_indices)
    else:
        fallback_reason = "OpenCV did not report a reliable total frame count; sequential limited sampling was used."
        frame_limit = None if args.frame_mode == "all" else args.frames
        frame_scores = score_video_frames(
            cv2,
            np,
            torch,
            iter_sequential_frames(cv2, video_path, frame_limit),
            video_info,
            detector,
            face_detector,
            crop_face,
            device,
            args.batch_size,
            "sequential",
        )
        attempted_indices = [frame_score.frame_index for frame_score in frame_scores]
        coarse_count = len(attempted_indices)

    if not frame_scores:
        fail(f"SelfBlendedImages could not decode any selected frames from {video_path}.")

    scored_frame_scores = [frame_score for frame_score in frame_scores if frame_score.score is not None]
    if not scored_frame_scores:
        fail(f"No faces were detected in sampled frames from {video_path}.")

    per_frame_scores = [float(frame_score.score) for frame_score in scored_frame_scores]
    if not per_frame_scores:
        fail(f"SelfBlendedImages did not produce any predictions for {video_path}.")

    score = float(np.mean(per_frame_scores))
    details = build_details(args, video_info, frame_scores, attempted_indices, coarse_count, refined_count, fallback_reason)
    top_frame = details["top_frames"][0] if details["top_frames"] else None
    print(f"SelfBlendedImages video fakeness: {score:.4f}")
    print(
        "Frame selection: "
        f"{args.frame_mode}, attempted {len(attempted_indices)}, decoded {len(frame_scores)}, "
        f"scored {len(per_frame_scores)}"
    )
    if top_frame:
        print(
            "Most suspicious frame: "
            f"{top_frame['frame']} at {format_time(top_frame['time_seconds'])}, "
            f"score {top_frame['score']:.4f}, faces {top_frame['face_count']}"
        )
    print(f"DETAIL_JSON:{json.dumps(details, separators=(',', ':'))}")
    print(f"SCORE:{score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
