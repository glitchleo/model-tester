from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
RUNNER_PYTHON = REPO_PYTHON if REPO_PYTHON.is_file() else Path(sys.executable)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
VIDEO_LIST_SUFFIXES = {".lst", ".txt"}
MODEL_NAMES = ["altfreezing", "effort", "f3net", "recce", "selfblendedimages", "ucf"]
MIN_WEIGHT_BYTES = 1024 * 1024
VIDEO_PRESETS = {
    "quick": 8,
    "balanced": 32,
    "thorough": 96,
}
ALTFREEZING_DEFAULT_MAX_FRAME = 400
MODEL_LABELS = {
    "altfreezing": "AltFreezing",
    "effort": "EFFORT",
    "f3net": "F3Net",
    "recce": "RECCE",
    "selfblendedimages": "SelfBlendedImages",
    "ucf": "UCF",
}
RUNNERS = {
    "image": {
        "altfreezing": ROOT / "models" / "altfreezing" / "run_image.py",
        "effort": ROOT / "models" / "effort" / "run_image.py",
        "f3net": ROOT / "models" / "f3net" / "run_image.py",
        "recce": ROOT / "models" / "recce" / "run_image.py",
        "selfblendedimages": ROOT / "models" / "selfblendedimages" / "run_image.py",
        "ucf": ROOT / "models" / "ucf" / "run_image.py",
    },
    "video": {
        "altfreezing": ROOT / "models" / "altfreezing" / "run_video.py",
        "effort": ROOT / "models" / "effort" / "run_video.py",
        "f3net": ROOT / "models" / "f3net" / "run_video.py",
        "recce": ROOT / "models" / "recce" / "run_video.py",
        "selfblendedimages": ROOT / "models" / "selfblendedimages" / "run_video.py",
        "ucf": ROOT / "models" / "ucf" / "run_video.py",
    },
}
SUCCESS_NOTES = {
    ("image", "altfreezing"): "single image duplicated into a short video clip because the original model is video-based",
    ("image", "effort"): "image resized and normalized for the EFFORT CLIP-L14 detector",
    ("image", "f3net"): "image resized and scored with the F3Net FAD detector",
    ("image", "recce"): "image resized and scored with the RECCE reconstruction-classification detector",
    ("image", "ucf"): "image resized and scored with the UCF shared-feature detector",
    ("video", "altfreezing"): "native video model using temporal face-track clips",
    ("video", "effort"): "video sampled into frames and averaged because EFFORT is image-based",
    ("video", "f3net"): "video sampled into frames and averaged because F3Net is image-based",
    ("video", "recce"): "video sampled into frames and averaged because RECCE is image-based",
    ("video", "selfblendedimages"): "video sampled into frames and averaged because SelfBlendedImages is image-based",
    ("video", "ucf"): "video sampled into frames and averaged because UCF is image-based",
}
SCORE_PATTERN = re.compile(r"^SCORE:(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$", re.MULTILINE)
UNAVAILABLE_PATTERN = re.compile(r"^UNAVAILABLE:(?P<reason>.+)$", re.MULTILINE)
DETAIL_PATTERN = re.compile(r"^DETAIL_JSON:(?P<value>\{.*\})$", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score one image or video with one model or all configured models.",
    )
    parser.add_argument("media", type=Path, help="Path to an input image/video, a video folder, or a .txt/.lst video list.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When media is a folder, include videos from subfolders too.",
    )
    parser.add_argument(
        "--model",
        choices=["available", "both", "all", *MODEL_NAMES],
        default="available",
        help=(
            "Which model(s) to run. Defaults to all models with local assets available. "
            "Use all to try every configured model, including unavailable ones."
        ),
    )
    parser.add_argument(
        "--altfreezing-checkpoint",
        type=Path,
        help="Optional override for the AltFreezing checkpoint.",
    )
    parser.add_argument(
        "--effort-checkpoint",
        type=Path,
        help="Optional override for the EFFORT checkpoint.",
    )
    parser.add_argument(
        "--effort-clip-model",
        type=Path,
        help="Optional override for the local CLIP ViT-L/14 config directory.",
    )
    parser.add_argument(
        "--f3net-checkpoint",
        type=Path,
        help="Optional override for the F3Net detector checkpoint.",
    )
    parser.add_argument(
        "--recce-checkpoint",
        type=Path,
        help="Optional override for the RECCE detector checkpoint.",
    )
    parser.add_argument(
        "--ucf-checkpoint",
        type=Path,
        help="Optional override for the UCF detector checkpoint.",
    )
    parser.add_argument(
        "--selfblendedimages-weight",
        type=Path,
        help="Optional override for the SelfBlendedImages weight file.",
    )
    parser.add_argument(
        "--video-frames",
        type=int,
        help=(
            "Optional frame count for frame-based video scoring. F3Net and UCF sample this many frames uniformly; "
            "SelfBlendedImages uses it as the adaptive/uniform frame limit. AltFreezing ignores this and uses "
            "its native video path."
        ),
    )
    parser.add_argument(
        "--altfreezing-max-frame",
        type=int,
        default=ALTFREEZING_DEFAULT_MAX_FRAME,
        help=(
            "Maximum leading video frames decoded by AltFreezing's native runner. Defaults to the upstream demo "
            f"setting of {ALTFREEZING_DEFAULT_MAX_FRAME}; this is separate from --video-frames."
        ),
    )
    parser.add_argument(
        "--video-preset",
        choices=list(VIDEO_PRESETS),
        default="quick",
        help=(
            "Frame-based video test budget used when --video-frames is not provided. Defaults to quick. "
            "AltFreezing uses --altfreezing-max-frame instead."
        ),
    )
    parser.add_argument(
        "--video-frame-mode",
        choices=["adaptive", "uniform", "all"],
        default="uniform",
        help=(
            "SelfBlendedImages video frame strategy. Uniform uses the same fixed interval rule as the other "
            "frame-based models; adaptive does a coarse interval pass and then refines around the highest-scoring moments."
        ),
    )
    parser.add_argument(
        "--video-frame-interval",
        type=int,
        default=20,
        help="SelfBlendedImages adaptive coarse-pass interval in frames.",
    )
    parser.add_argument(
        "--video-refine-window",
        type=int,
        default=8,
        help="SelfBlendedImages adaptive refinement radius around each hotspot, in frames.",
    )
    parser.add_argument(
        "--video-hotspots",
        type=int,
        default=3,
        help="How many SelfBlendedImages coarse-pass hotspots to refine around.",
    )
    parser.add_argument(
        "--video-batch-size",
        type=int,
        default=2,
        help="SelfBlendedImages face-crop inference batch size for video scoring.",
    )
    parser.add_argument(
        "--selfblendedimages-face-max-size",
        type=int,
        default=960,
        help="Cap SelfBlendedImages RetinaFace detector size for videos. Use 0 to keep the source video size.",
    )
    parser.add_argument(
        "--write-altfreezing-output",
        action="store_true",
        help="When scoring a video with AltFreezing, also write its annotated output video.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of a table.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also print each runner's raw stdout and stderr.",
    )
    args = parser.parse_args()
    if args.video_frames is not None and args.video_frames < 1:
        parser.error("--video-frames must be at least 1.")
    if args.altfreezing_max_frame < 1:
        parser.error("--altfreezing-max-frame must be at least 1.")
    if args.video_frame_interval < 1:
        parser.error("--video-frame-interval must be at least 1.")
    if args.video_refine_window < 0:
        parser.error("--video-refine-window must be 0 or greater.")
    if args.video_hotspots < 1:
        parser.error("--video-hotspots must be at least 1.")
    if args.video_batch_size < 1:
        parser.error("--video-batch-size must be at least 1.")
    if args.selfblendedimages_face_max_size < 0:
        parser.error("--selfblendedimages-face-max-size must be 0 or greater.")
    return args


def infer_media_kind(media_path: Path) -> str | None:
    suffix = media_path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    return None


def collect_video_paths(media_path: Path, recursive: bool) -> tuple[list[Path], Path]:
    if media_path.is_dir():
        iterator = media_path.rglob("*") if recursive else media_path.iterdir()
        videos = sorted(
            path.resolve()
            for path in iterator
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        )
        return videos, media_path.resolve()

    if media_path.is_file() and media_path.suffix.lower() in VIDEO_LIST_SUFFIXES:
        videos = []
        base_dir = media_path.resolve().parent
        for raw_line in media_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip().lstrip("\ufeff")
            if not line or line.startswith("#"):
                continue
            candidate = Path(line)
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            candidate = candidate.resolve()
            if candidate.is_file() and candidate.suffix.lower() in VIDEO_SUFFIXES:
                videos.append(candidate)
        return videos, base_dir

    return [], media_path.resolve().parent


def clean_cell(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def score_cell(result: dict[str, object] | None) -> str:
    if not result:
        return "-"
    if result["status"] == "ok" and isinstance(result.get("score"), (int, float)):
        return f"{float(result['score']):.6f}"
    note = clean_cell(result.get("note"))
    if result["status"] == "unavailable":
        return "NA" if not note else f"NA:{note}"
    return "ERR" if not note else f"ERR:{note}"


def relative_video_label(video_path: Path, base_dir: Path) -> str:
    try:
        return str(video_path.relative_to(base_dir))
    except ValueError:
        return str(video_path)


def format_batch_report(
    rows: list[dict[str, object]],
    model_names: list[str],
) -> str:
    headers = ["video", "total_frames", "final_score", "status", *model_names]
    lines = ["\t".join(headers)]
    for row in rows:
        cells = [
            clean_cell(row.get("video")),
            clean_cell(row.get("total_frames", "-")),
            clean_cell(row.get("final_score", "-")),
            clean_cell(row.get("status", "-")),
        ]
        model_scores = row.get("model_scores")
        if not isinstance(model_scores, dict):
            model_scores = {}
        cells.extend(clean_cell(model_scores.get(model_name, "-")) for model_name in model_names)
        lines.append("\t".join(cells))
    return "\n".join(lines)


def score_video_batch(args: argparse.Namespace, media_path: Path) -> int:
    video_paths, base_dir = collect_video_paths(media_path, args.recursive)
    if not video_paths:
        print(f"No video files found in {media_path}.", file=sys.stderr)
        return 1

    model_names = selected_model_names(args.model, "video", args)
    if not model_names:
        print("No available video models were found.", file=sys.stderr)
        print("Run python .\\check_setup.py to see which weights or dependencies are missing.", file=sys.stderr)
        return 1

    rows = []
    had_success = False
    had_error = False
    for video_path in video_paths:
        results = [run_model(model_name, "video", args, video_path) for model_name in model_names]
        ok_results = [
            result
            for result in results
            if result["status"] == "ok" and isinstance(result.get("score"), (int, float))
        ]
        had_success = had_success or bool(ok_results)
        had_error = had_error or any(result["status"] == "error" for result in results)
        final_score = (
            sum(float(result["score"]) for result in ok_results) / len(ok_results)
            if ok_results
            else None
        )
        video = next(
            (
                video_info
                for video_info in (video_metadata(result.get("details")) for result in results)
                if video_info is not None
            ),
            None,
        )
        rows.append(
            {
                "video": relative_video_label(video_path, base_dir),
                "total_frames": "-" if not video else video.get("total_frames", "-"),
                "final_score": "-" if final_score is None else f"{final_score:.6f}",
                "status": "no_score" if final_score is None else score_label(final_score),
                "model_scores": {
                    str(result["model"]): score_cell(result)
                    for result in results
                },
            }
        )

    print(format_batch_report(rows, model_names))
    return 0 if had_success and not had_error else 1


def large_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > MIN_WEIGHT_BYTES


def repo_has_marker(repo_dir: Path, marker: str) -> bool:
    if not repo_dir.exists():
        return False
    candidates = [repo_dir]
    candidates.extend(sorted(path for path in repo_dir.iterdir() if path.is_dir()))
    return any((candidate / marker).exists() for candidate in candidates)


def torch_cuda_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def selfblended_weight_available(weights_dir: Path, provided: Path | None) -> bool:
    if provided:
        return provided.resolve().is_file()

    for candidate in sorted(weights_dir.glob("*.tar")):
        if not large_file(candidate):
            continue
        if not zipfile.is_zipfile(candidate):
            continue
        try:
            with zipfile.ZipFile(candidate) as archive:
                if any(name.endswith("data.pkl") for name in archive.namelist()):
                    return True
        except zipfile.BadZipFile:
            continue

    extracted = weights_dir / "FFc23_extracted"
    return (extracted / "archive" / "data.pkl").is_file()


def model_unavailable_reason(model_name: str, media_kind: str, args: argparse.Namespace) -> str | None:
    runner = RUNNERS.get(media_kind, {}).get(model_name)
    if runner is None or not runner.is_file():
        return f"missing {media_kind} runner"

    if model_name == "altfreezing":
        checkpoint_ok = (
            args.altfreezing_checkpoint.resolve().is_file()
            if args.altfreezing_checkpoint
            else any(large_file(path) for path in (ROOT / "models" / "altfreezing" / "checkpoints").glob("*.pth"))
        )
        aux_dir = ROOT / "auxillary"
        aux_ok = all(
            (aux_dir / name).is_file()
            for name in ("mobilenet0.25_Final.pth", "mobilenet_224_model_best_gdconv_external.pth")
        )
        missing = []
        if not checkpoint_ok:
            missing.append("checkpoint in models/altfreezing/checkpoints")
        if not aux_ok:
            missing.append("auxiliary face weights in auxillary")
        if not torch_cuda_available():
            missing.append("CUDA runtime")
        if not repo_has_marker(ROOT / "models" / "altfreezing" / "repo", "demo.py"):
            missing.append("AltFreezing source repo")
        return None if not missing else "missing " + ", ".join(missing)

    if model_name == "effort":
        model_root = ROOT / "models" / "effort"
        checkpoint = (
            args.effort_checkpoint.resolve()
            if args.effort_checkpoint
            else model_root / "weights" / "effort_clip_L14_trainOn_FaceForensic.pth"
        )
        clip_model = (
            args.effort_clip_model.resolve()
            if args.effort_clip_model
            else model_root / "pretrained" / "clip-vit-large-patch14"
        )
        missing = []
        if not large_file(checkpoint):
            missing.append("EFFORT checkpoint")
        if not (clip_model / "config.json").is_file():
            missing.append("CLIP ViT-L/14 config")
        if not repo_has_marker(model_root / "repo", "DeepfakeBench/training/demo.py"):
            missing.append("EFFORT source repo")
        return None if not missing else "missing " + ", ".join(missing)

    if model_name == "f3net":
        weights_dir = ROOT / "models" / "f3net" / "weights"
        backbone_ok = (weights_dir / "xception-b5690688.pth").is_file()
        detector_ok = (
            args.f3net_checkpoint.resolve().is_file()
            if args.f3net_checkpoint
            else any(
                large_file(path)
                for path in weights_dir.glob("*.pth")
                if path.name != "xception-b5690688.pth"
            )
        )
        missing = []
        if not backbone_ok:
            missing.append("Xception backbone")
        if not detector_ok:
            missing.append("F3Net detector checkpoint")
        if not repo_has_marker(ROOT / "models" / "f3net" / "repo", "models.py"):
            missing.append("F3Net source repo")
        return None if not missing else "missing " + ", ".join(missing)

    if model_name == "recce":
        model_root = ROOT / "models" / "recce"
        checkpoint = (
            args.recce_checkpoint.resolve()
            if args.recce_checkpoint
            else model_root / "weights" / "recce_best.pth"
        )
        checkpoint_ok = large_file(checkpoint)
        repo_ok = repo_has_marker(model_root / "repo", "model/network/Recce.py")
        missing = []
        if not checkpoint_ok:
            missing.append("RECCE checkpoint")
        if not repo_ok:
            missing.append("RECCE source repo")
        return None if not missing else "missing " + ", ".join(missing)

    if model_name == "selfblendedimages":
        weight_ok = selfblended_weight_available(
            ROOT / "models" / "selfblendedimages" / "weights",
            args.selfblendedimages_weight,
        )
        repo_ok = repo_has_marker(
            ROOT / "models" / "selfblendedimages" / "repo",
            "src/inference/model.py",
        )
        missing = []
        if not weight_ok:
            missing.append("SelfBlendedImages FFc23 checkpoint")
        if not repo_ok:
            missing.append("SelfBlendedImages source repo")
        return None if not missing else "missing " + ", ".join(missing)

    if model_name == "ucf":
        checkpoint = args.ucf_checkpoint.resolve() if args.ucf_checkpoint else ROOT / "models" / "ucf" / "ucf_best.pth"
        checkpoint_ok = large_file(checkpoint)
        repo_ok = repo_has_marker(ROOT / "models" / "f3net" / "repo", "xception.py")
        missing = []
        if not checkpoint_ok:
            missing.append("UCF checkpoint")
        if not repo_ok:
            missing.append("shared Xception source")
        return None if not missing else "missing " + ", ".join(missing)

    return "unknown model"


def model_is_available(model_name: str, media_kind: str, args: argparse.Namespace) -> bool:
    return model_unavailable_reason(model_name, media_kind, args) is None


def available_model_names(media_kind: str, args: argparse.Namespace) -> list[str]:
    return [
        model_name
        for model_name in MODEL_NAMES
        if model_is_available(model_name, media_kind, args)
    ]


def selected_model_names(selection: str, media_kind: str, args: argparse.Namespace) -> list[str]:
    if selection in {"available", "both"}:
        return available_model_names(media_kind, args)
    if selection == "all":
        return MODEL_NAMES
    return [selection]


def skipped_model_notes(selection: str, media_kind: str, args: argparse.Namespace) -> list[dict[str, str]]:
    if selection not in {"available", "both"}:
        return []
    skipped = []
    for model_name in MODEL_NAMES:
        reason = model_unavailable_reason(model_name, media_kind, args)
        if reason is not None:
            skipped.append({"model": model_name, "note": reason})
    return skipped


def video_frame_limit(args: argparse.Namespace) -> int:
    return args.video_frames if args.video_frames is not None else VIDEO_PRESETS[args.video_preset]


def build_command(model_name: str, media_kind: str, args: argparse.Namespace, media_path: Path) -> list[str]:
    input_flag = "--image" if media_kind == "image" else "--video"
    command = [str(RUNNER_PYTHON), str(RUNNERS[media_kind][model_name]), input_flag, str(media_path)]
    if model_name == "altfreezing" and args.altfreezing_checkpoint:
        command.extend(["--checkpoint", str(args.altfreezing_checkpoint.resolve())])
    if model_name == "effort" and args.effort_checkpoint:
        command.extend(["--checkpoint", str(args.effort_checkpoint.resolve())])
    if model_name == "effort" and args.effort_clip_model:
        command.extend(["--clip-model", str(args.effort_clip_model.resolve())])
    if model_name == "f3net" and args.f3net_checkpoint:
        command.extend(["--checkpoint", str(args.f3net_checkpoint.resolve())])
    if model_name == "recce" and args.recce_checkpoint:
        command.extend(["--checkpoint", str(args.recce_checkpoint.resolve())])
    if model_name == "ucf" and args.ucf_checkpoint:
        command.extend(["--checkpoint", str(args.ucf_checkpoint.resolve())])
    if model_name == "selfblendedimages" and args.selfblendedimages_weight:
        command.extend(["--weight", str(args.selfblendedimages_weight.resolve())])
    if media_kind == "video":
        if model_name == "altfreezing":
            command.extend(["--max-frame", str(args.altfreezing_max_frame)])
        else:
            command.extend(["--frames", str(video_frame_limit(args))])
    if media_kind == "video" and model_name == "selfblendedimages":
        command.extend(
            [
                "--frame-mode",
                args.video_frame_mode,
                "--coarse-interval",
                str(args.video_frame_interval),
                "--refine-window",
                str(args.video_refine_window),
                "--hotspots",
                str(args.video_hotspots),
                "--batch-size",
                str(args.video_batch_size),
                "--face-detector-max-size",
                str(args.selfblendedimages_face_max_size),
            ]
        )
    if media_kind == "video" and model_name == "altfreezing" and not args.write_altfreezing_output:
        command.append("--skip-output")
    return command


def last_non_empty_line(*chunks: str) -> str:
    for chunk in chunks:
        lines = [line.strip() for line in chunk.splitlines() if line.strip()]
        if lines:
            return lines[-1]
    return ""


def parse_detail_payload(stdout: str) -> dict[str, object] | None:
    detail_match = DETAIL_PATTERN.search(stdout)
    if not detail_match:
        return None
    try:
        payload = json.loads(detail_match.group("value"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def summarize_details(details: dict[str, object] | None) -> str:
    if not details:
        return ""
    selection = details.get("selection")
    top_frames = details.get("top_frames")
    if not isinstance(selection, dict):
        return ""

    mode = selection.get("mode", "video")
    scored = selection.get("scored_frames", 0)
    decoded = selection.get("decoded_frames", 0)
    attempted = selection.get("attempted_frames", 0)
    if mode == "native_video":
        clips = selection.get("clips_scored", 0)
        max_frame = selection.get("max_frame", "unknown")
        note = f"native video scan; decoded {decoded} frames with max-frame {max_frame}; scored {scored} tracked frames from {clips} clips"
        if isinstance(top_frames, list) and top_frames:
            top_frame = top_frames[0]
            if isinstance(top_frame, dict):
                frame = top_frame.get("frame")
                time_seconds = top_frame.get("time_seconds")
                score = top_frame.get("score")
                if frame is not None and score is not None:
                    if isinstance(time_seconds, (int, float)):
                        note += f"; peak frame {frame} at {float(time_seconds):.2f}s scored {float(score):.4f}"
                    else:
                        note += f"; peak frame {frame} scored {float(score):.4f}"
        return note

    note = f"{mode} frame scan; scored {scored}/{decoded} decoded frames ({attempted} attempted)"
    if isinstance(top_frames, list) and top_frames:
        top_frame = top_frames[0]
        if isinstance(top_frame, dict):
            frame = top_frame.get("frame")
            time_seconds = top_frame.get("time_seconds")
            score = top_frame.get("score")
            if frame is not None and score is not None:
                if isinstance(time_seconds, (int, float)):
                    note += f"; peak frame {frame} at {float(time_seconds):.2f}s scored {float(score):.4f}"
                else:
                    note += f"; peak frame {frame} scored {float(score):.4f}"
    return note


def format_seconds(seconds: object) -> str:
    if not isinstance(seconds, (int, float)):
        return "unknown time"
    minutes = int(float(seconds) // 60)
    remaining = float(seconds) - minutes * 60
    return f"{minutes:02d}:{remaining:05.2f}"


def score_label(score: float) -> str:
    if score >= 0.7:
        return "high suspicion of AI alteration"
    if score >= 0.5:
        return "suspicious"
    if score >= 0.35:
        return "uncertain / mixed signal"
    return "low suspicion"


def first_dict(value: object) -> dict[str, object] | None:
    if not isinstance(value, list) or not value:
        return None
    first_value = value[0]
    return first_value if isinstance(first_value, dict) else None


def numeric_value(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def format_frame_time(frame: object, seconds: object) -> str:
    if frame is None:
        return format_seconds(seconds)
    return f"frame {frame} ({format_seconds(seconds)})"


def format_window(window: dict[str, object] | None) -> str | None:
    if not window:
        return None
    start_frame = window.get("start_frame")
    end_frame = window.get("end_frame")
    start_time = window.get("start_time_seconds")
    end_time = window.get("end_time_seconds")
    if start_frame is None or end_frame is None:
        return None
    return (
        f"frames {start_frame}-{end_frame} "
        f"({format_seconds(start_time)} to {format_seconds(end_time)})"
    )


def video_metadata(details: dict[str, object] | None) -> dict[str, object] | None:
    if not isinstance(details, dict):
        return None
    video = details.get("video")
    return video if isinstance(video, dict) else None


def format_video_summary(video: dict[str, object] | None) -> str:
    if not video:
        return "Video metadata was not reported by the model runner."

    total_frames = video.get("total_frames")
    fps = video.get("fps")
    width = video.get("width")
    height = video.get("height")

    parts = []
    if total_frames is not None:
        parts.append(f"{total_frames} total frames")
    if fps is not None:
        parts.append(f"{float(fps):.3f} fps")
    if width is not None and height is not None:
        parts.append(f"{width}x{height}")
    return "Video metadata: " + ", ".join(parts) + "." if parts else "Video metadata was not reported by the model runner."


def explain_selection(details: dict[str, object], model: str) -> str:
    selection = details.get("selection")
    if not isinstance(selection, dict):
        return "The model scored the video and returned frame-level evidence."

    video_summary = format_video_summary(video_metadata(details))
    mode = selection.get("mode")
    attempted = selection.get("attempted_frames")
    decoded = selection.get("decoded_frames")
    scored = selection.get("scored_frames")

    if mode == "adaptive":
        coarse = selection.get("coarse_frames")
        refined = selection.get("refined_frames")
        interval = selection.get("coarse_interval")
        window = selection.get("refine_window")
        return (
            "SelfBlendedImages used adaptive sampling: it first checked broad interval frames "
            f"every {interval} frames, then used the remaining budget to inspect nearby frames around the "
            f"highest-scoring moments. It attempted {attempted} frames, decoded {decoded}, scored faces in "
            f"{scored}, with {coarse} coarse frames and {refined} refined frames. The refinement radius was "
            f"{window} frames on each side of a hotspot. {video_summary}"
        )

    if mode == "uniform":
        if model in {"effort", "f3net", "recce", "ucf"}:
            label = MODEL_LABELS.get(model, model)
            return (
                f"{label} is image-based, so the wrapper sampled frames uniformly across the video and averaged "
                f"their frame scores. It attempted {attempted} frames, decoded {decoded}, and scored {scored}. "
                f"{video_summary}"
            )
        return (
            "SelfBlendedImages used uniform sampling, spreading the frame budget across the whole video. "
            f"It attempted {attempted} frames, decoded {decoded}, and found scorable faces in {scored}. "
            f"{video_summary}"
        )

    if mode == "all":
        return (
            "SelfBlendedImages checked every decoded frame. "
            f"It attempted {attempted} frames, decoded {decoded}, and found scorable faces in {scored}. "
            f"{video_summary}"
        )

    if mode == "native_video":
        clips = selection.get("clips_scored")
        clip_size = selection.get("clip_size")
        max_frame = selection.get("max_frame")
        return (
            "AltFreezing used its native temporal video path, tracking faces and scoring short clips over time. "
            f"It decoded {decoded} frames from the video using a max-frame limit of {max_frame}, scored "
            f"{scored} tracked face frames, and scored {clips} temporal clips with clip size {clip_size}. "
            f"The frame examples come from the highest clip responses. {video_summary}"
        )

    return (
        f"The model used {mode} mode, attempted {attempted} frames, decoded {decoded}, "
        f"and scored {scored} frames with detected evidence. {video_summary}"
    )


def build_human_explanation(result: dict[str, object]) -> dict[str, object] | None:
    if result["status"] != "ok" or not isinstance(result.get("score"), (int, float)):
        return None

    model = str(result["model"])
    label = MODEL_LABELS.get(model, model)
    score = float(result["score"])
    details = result.get("details")
    verdict = score_label(score)
    explanation: dict[str, object] = {
        "headline": f"{label} final score is {score:.4f}, which is interpreted as {verdict}.",
        "verdict": verdict,
        "final_score": round(score, 6),
        "method": str(result.get("note") or SUCCESS_NOTES.get((result.get("media"), model), "")),
        "strongest_evidence": None,
        "suspicious_timeframe": None,
        "why_this_matters": (
            "Higher frame or clip scores mean the model saw stronger signs of manipulation in the detected face "
            "content at that moment."
        ),
        "limitation": (
            "This is model evidence, not proof by itself. These wrappers localize suspicious times from model "
            "scores; they do not produce pixel-level forensic attribution."
        ),
    }

    if not isinstance(details, dict):
        return explanation

    video = video_metadata(details)
    if video:
        explanation["video"] = video
        explanation["video_summary"] = format_video_summary(video)

    explanation["method"] = explain_selection(details, model)
    top_frame = first_dict(details.get("top_frames"))
    top_window = first_dict(details.get("suspicious_windows"))
    window_text = format_window(top_window)

    if top_frame:
        frame = top_frame.get("frame")
        time_seconds = top_frame.get("time_seconds")
        frame_score = numeric_value(top_frame.get("score"))
        face_count = top_frame.get("face_count")
        clip_count = top_frame.get("clip_count")
        reason = top_frame.get("reason")

        if model == "selfblendedimages":
            evidence_source = (
                f"{face_count} detected face crop(s)" if face_count is not None else "detected face crops"
            )
            model_reason = (
                "SelfBlendedImages is image-based, so for each selected video frame it detects faces, crops them, "
                "scores each crop, and uses the highest face score as that frame's suspiciousness."
            )
        elif model == "altfreezing":
            evidence_source = f"{clip_count} temporal clips" if clip_count is not None else "temporal clips"
            model_reason = (
                "AltFreezing is video-based, so it looks at temporal face-track clips and highlights frames that "
                "appear inside the strongest suspicious clips. Its evidence is based on tracked facial motion and "
                "appearance over a short time window, not on a single isolated image crop."
            )
        elif model == "effort":
            evidence_source = "sampled frame evidence"
            model_reason = (
                "EFFORT uses a CLIP-L14 vision backbone with orthogonal residual attention layers, so this wrapper "
                "highlights the sampled frame with the strongest fake-class response."
            )
        elif model == "recce":
            evidence_source = "sampled frame evidence"
            model_reason = (
                "RECCE combines reconstruction and classification signals, so this wrapper highlights the sampled "
                "frame with the strongest fake-class response."
            )
        else:
            evidence_source = "sampled frame evidence"
            model_reason = "This model highlights the sampled frame with the strongest score."

        score_text = "unknown" if frame_score is None else f"{frame_score:.4f}"
        explanation["strongest_evidence"] = (
            f"The strongest example was {format_frame_time(frame, time_seconds)} with a frame score of "
            f"{score_text}, based on {evidence_source}."
        )
        explanation["why_this_matters"] = f"{model_reason} {reason}" if isinstance(reason, str) else model_reason
        explanation["peak_frame"] = frame
        explanation["peak_time_seconds"] = time_seconds
        explanation["peak_frame_score"] = None if frame_score is None else round(frame_score, 6)

    if window_text:
        peak_score = numeric_value(top_window.get("peak_score")) if top_window else None
        score_text = "" if peak_score is None else f" with peak score {peak_score:.4f}"
        explanation["suspicious_timeframe"] = f"The most suspicious localized timeframe is {window_text}{score_text}."

    fallback = None
    selection = details.get("selection")
    if isinstance(selection, dict):
        fallback = selection.get("fallback_reason")
    if isinstance(fallback, str) and fallback:
        explanation["sampling_note"] = fallback

    return explanation


def reasoning_example(result: dict[str, object]) -> dict[str, object]:
    model = str(result["model"])
    details = result.get("details")
    score = result.get("score")
    label = MODEL_LABELS.get(model, model)
    example: dict[str, object] = {
        "model": model,
        "model_label": label,
        "score": score,
        "example": str(result.get("note") or "The model produced a score, but no frame-level reasoning was emitted."),
    }
    if not isinstance(details, dict):
        return example

    top_frames = details.get("top_frames")
    if not isinstance(top_frames, list) or not top_frames:
        explanation = details.get("explanation")
        if isinstance(explanation, str):
            example["example"] = explanation
        return example

    top_frame = top_frames[0]
    if not isinstance(top_frame, dict):
        return example

    frame = top_frame.get("frame")
    time_seconds = top_frame.get("time_seconds")
    frame_score = top_frame.get("score")
    face_count = top_frame.get("face_count")
    clip_count = top_frame.get("clip_count")
    reason = top_frame.get("reason")

    if model == "altfreezing":
        support = f"{clip_count} temporal clips" if clip_count is not None else "temporal clips"
        why = "its temporal face-track clips had the strongest fakeness response there"
    elif model == "selfblendedimages":
        support = f"{face_count} detected face crop(s)" if face_count is not None else "detected face crops"
        why = "the detected face crop had the strongest frame-level fakeness response there"
    elif model == "effort":
        support = "sampled frame evidence"
        why = "the EFFORT CLIP-L14 detector had the strongest fake-class response there"
    elif model == "recce":
        support = "sampled frame evidence"
        why = "the RECCE reconstruction-classification detector had the strongest fake-class response there"
    else:
        support = "sampled frame evidence"
        why = "this was one of the highest-scoring sampled frames"

    if isinstance(reason, str) and reason:
        why = reason

    if frame is not None and frame_score is not None:
        why_text = str(why).rstrip(".")
        example["example"] = (
            f"Peak evidence at frame {frame} ({format_seconds(time_seconds)}), "
            f"frame score {float(frame_score):.4f}, based on {support}; {why_text}."
        )
        example["frame"] = frame
        example["time_seconds"] = time_seconds
        example["frame_score"] = frame_score

    windows = details.get("suspicious_windows")
    if isinstance(windows, list) and windows:
        example["suspicious_window"] = windows[0]
    return example


def build_summary(
    results: list[dict[str, object]],
    skipped_models: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    skipped_models = skipped_models or []
    ok_results = [
        result
        for result in results
        if result["status"] == "ok" and isinstance(result.get("score"), (int, float))
    ]
    unavailable = [
        {
            "model": result["model"],
            "note": result["note"],
        }
        for result in results
        if result["status"] == "unavailable"
    ]
    errors = [
        {
            "model": result["model"],
            "note": result["note"],
        }
        for result in results
        if result["status"] == "error"
    ]

    if not ok_results:
        return {
            "status": "no_score",
            "final_score": None,
            "score_method": "no model returned a score",
            "models_scored": [],
            "video": None,
            "reasoning_examples": [],
            "human_explanation": "No final explanation is available because no model returned a usable score.",
            "model_explanations": [],
            "unavailable_models": unavailable,
            "error_models": errors,
            "skipped_models": skipped_models,
        }

    final_score = sum(float(result["score"]) for result in ok_results) / len(ok_results)
    model_explanations = [
        explanation
        for explanation in (build_human_explanation(result) for result in ok_results)
        if explanation is not None
    ]
    video = next(
        (
            video_info
            for video_info in (video_metadata(result.get("details")) for result in ok_results)
            if video_info is not None
        ),
        None,
    )
    return {
        "status": score_label(final_score),
        "final_score": round(final_score, 6),
        "score_method": "simple mean of successful model fakeness scores",
        "models_scored": [result["model"] for result in ok_results],
        "video": video,
        "model_scores": {
            str(result["model"]): round(float(result["score"]), 6)
            for result in ok_results
        },
        "reasoning_examples": [reasoning_example(result) for result in ok_results],
        "human_explanation": (
            model_explanations[0]["headline"]
            if len(model_explanations) == 1
            else f"{len(ok_results)} models returned scores. The final score is {final_score:.4f}, "
            f"interpreted as {score_label(final_score)}."
        ),
        "model_explanations": model_explanations,
        "unavailable_models": unavailable,
        "error_models": errors,
        "skipped_models": skipped_models,
    }


def format_final_report(summary: dict[str, object]) -> str:
    if summary["final_score"] is None:
        lines = ["", "FINAL SUMMARY", "No final score: no model returned a usable score."]
        unavailable = summary.get("unavailable_models")
        if isinstance(unavailable, list) and unavailable:
            lines.append("Unavailable models:")
            for item in unavailable:
                if isinstance(item, dict):
                    lines.append(f"- {item.get('model')}: {item.get('note')}")
        errors = summary.get("error_models")
        if isinstance(errors, list) and errors:
            lines.append("Models with errors:")
            for item in errors:
                if isinstance(item, dict):
                    lines.append(f"- {item.get('model')}: {item.get('note')}")
        return "\n".join(lines)

    lines = [
        "",
        "FINAL SUMMARY",
        f"Final score: {float(summary['final_score']):.6f}",
        f"Interpretation: {summary['status']}",
        f"Method: {summary['score_method']}",
        format_video_summary(summary.get("video") if isinstance(summary.get("video"), dict) else None),
        "Reasoning examples:",
    ]
    examples = summary.get("reasoning_examples")
    if isinstance(examples, list):
        for example in examples:
            if not isinstance(example, dict):
                continue
            label = example.get("model_label") or example.get("model")
            score = example.get("score")
            score_text = "-" if score is None else f"{float(score):.6f}"
            lines.append(f"- {label} score {score_text}: {example.get('example')}")

    model_explanations = summary.get("model_explanations")
    if isinstance(model_explanations, list) and model_explanations:
        lines.append("Detailed explanation:")
        for explanation in model_explanations:
            if not isinstance(explanation, dict):
                continue
            headline = explanation.get("headline")
            method = explanation.get("method")
            strongest = explanation.get("strongest_evidence")
            timeframe = explanation.get("suspicious_timeframe")
            why = explanation.get("why_this_matters")
            limitation = explanation.get("limitation")
            if headline:
                lines.append(f"- {headline}")
            if method:
                lines.append(f"  Method: {method}")
            if strongest:
                lines.append(f"  Strongest evidence: {strongest}")
            if timeframe:
                lines.append(f"  Suspicious timeframe: {timeframe}")
            if why:
                lines.append(f"  Why: {why}")
            if limitation:
                lines.append(f"  Limitation: {limitation}")

    unavailable = summary.get("unavailable_models")
    if isinstance(unavailable, list) and unavailable:
        names = ", ".join(str(item.get("model")) for item in unavailable if isinstance(item, dict))
        lines.append(f"Unavailable models skipped: {names}")

    skipped = summary.get("skipped_models")
    if isinstance(skipped, list) and skipped:
        lines.append("Models not run because they are not available locally:")
        for item in skipped:
            if isinstance(item, dict):
                lines.append(f"- {item.get('model')}: {item.get('note')}")

    errors = summary.get("error_models")
    if isinstance(errors, list) and errors:
        names = ", ".join(str(item.get("model")) for item in errors if isinstance(item, dict))
        lines.append(f"Models with errors: {names}")

    return "\n".join(lines)


def run_model(model_name: str, media_kind: str, args: argparse.Namespace, media_path: Path) -> dict[str, object]:
    command = build_command(model_name, media_kind, args, media_path)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    score_match = SCORE_PATTERN.search(stdout)
    unavailable_match = UNAVAILABLE_PATTERN.search(stdout)
    details = parse_detail_payload(stdout)

    result: dict[str, object] = {
        "model": model_name,
        "media": media_kind,
        "command": command,
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "score": None,
        "status": "error",
        "note": "",
        "details": details,
    }

    if score_match:
        result["status"] = "ok"
        result["score"] = float(score_match.group("value"))
        result["note"] = summarize_details(details) or SUCCESS_NOTES.get((media_kind, model_name), "")
        return result

    if unavailable_match:
        result["status"] = "unavailable"
        result["note"] = unavailable_match.group("reason").strip()
        return result

    note = last_non_empty_line(stderr, stdout)
    if note:
        result["note"] = note
    return result


def format_table(results: list[dict[str, object]]) -> str:
    rows = []
    for result in results:
        score = result["score"]
        rows.append(
            [
                str(result["model"]),
                str(result["status"]),
                "-" if score is None else f"{float(score):.6f}",
                str(result["note"]),
            ]
        )

    headers = ["MODEL", "STATUS", "SCORE", "NOTE"]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    lines = []
    lines.append("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    media_path = args.media.resolve()
    if media_path.is_dir() or (media_path.is_file() and media_path.suffix.lower() in VIDEO_LIST_SUFFIXES):
        return score_video_batch(args, media_path)

    if not media_path.is_file():
        print(f"Input media not found: {media_path}", file=sys.stderr)
        return 2

    media_kind = infer_media_kind(media_path)
    if media_kind is None:
        supported = ", ".join(sorted(IMAGE_SUFFIXES | VIDEO_SUFFIXES))
        print(f"Unsupported media type for {media_path}. Supported extensions: {supported}", file=sys.stderr)
        return 2

    selected_models = selected_model_names(args.model, media_kind, args)
    skipped_models = skipped_model_notes(args.model, media_kind, args)
    if not selected_models:
        print("No available models were found for this media type.", file=sys.stderr)
        print("Run python .\\check_setup.py to see which weights or dependencies are missing.", file=sys.stderr)
        return 1

    results = [run_model(model_name, media_kind, args, media_path) for model_name in selected_models]
    summary = build_summary(results, skipped_models)

    if args.json:
        payload = []
        for result in results:
            item = {
                "model": result["model"],
                "media": result["media"],
                "status": result["status"],
                "score": result["score"],
                "note": result["note"],
            }
            if result.get("details") is not None:
                item["details"] = result["details"]
            human_explanation = build_human_explanation(result)
            if human_explanation is not None:
                item["human_explanation"] = human_explanation
            payload.append(item)
        print(json.dumps({"summary": summary, "results": payload}, indent=2))
        print(format_final_report(summary), file=sys.stderr)
    else:
        print(format_table(results))
        print(format_final_report(summary))

    if args.verbose:
        for result in results:
            print()
            print(f"[{result['model']}] command")
            print(" ".join(str(part) for part in result["command"]))
            if result["stdout"]:
                print(f"[{result['model']}] stdout")
                print(str(result["stdout"]))
            if result["stderr"]:
                print(f"[{result['model']}] stderr")
                print(str(result["stderr"]))

    success_count = sum(1 for result in results if result["status"] == "ok")
    error_count = sum(1 for result in results if result["status"] == "error")
    return 0 if success_count > 0 and error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
