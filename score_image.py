from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
RUNNER_PYTHON = REPO_PYTHON if REPO_PYTHON.is_file() else Path(sys.executable)

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
VIDEO_LIST_SUFFIXES = {".lst", ".txt"}
MIN_WEIGHT_BYTES = 1024 * 1024
VIDEO_PRESETS = {"quick": 8, "balanced": 32, "thorough": 96}
ALTFREEZING_DEFAULT_MAX_FRAME = 400

SCORE_PATTERN = re.compile(r"^SCORE:(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$", re.MULTILINE)
UNAVAILABLE_PATTERN = re.compile(r"^UNAVAILABLE:(?P<reason>.+)$", re.MULTILINE)
DETAIL_PATTERN = re.compile(r"^DETAIL_JSON:(?P<value>\{.*\})$", re.MULTILINE)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    label: str
    image_runner: Path
    video_runner: Path
    image_note: str
    video_note: str

    def runner(self, media_kind: str) -> Path:
        return self.image_runner if media_kind == "image" else self.video_runner

    def success_note(self, media_kind: str) -> str:
        return self.image_note if media_kind == "image" else self.video_note


MODEL_SPECS = {
    spec.name: spec
    for spec in (
        ModelSpec(
            "altfreezing",
            "AltFreezing",
            ROOT / "models" / "altfreezing" / "run_image.py",
            ROOT / "models" / "altfreezing" / "run_video.py",
            "single image duplicated into a short video clip because the original model is video-based",
            "native video model using temporal face-track clips",
        ),
        ModelSpec(
            "effort",
            "EFFORT",
            ROOT / "models" / "effort" / "run_image.py",
            ROOT / "models" / "effort" / "run_video.py",
            "image resized and normalized for the EFFORT CLIP-L14 detector",
            "video sampled into frames and averaged because EFFORT is image-based",
        ),
        ModelSpec(
            "f3net",
            "F3Net",
            ROOT / "models" / "f3net" / "run_image.py",
            ROOT / "models" / "f3net" / "run_video.py",
            "image resized and scored with the F3Net FAD detector",
            "video sampled into frames and averaged because F3Net is image-based",
        ),
        ModelSpec(
            "recce",
            "RECCE",
            ROOT / "models" / "recce" / "run_image.py",
            ROOT / "models" / "recce" / "run_video.py",
            "image resized and scored with the RECCE reconstruction-classification detector",
            "video sampled into frames and averaged because RECCE is image-based",
        ),
        ModelSpec(
            "selfblendedimages",
            "SelfBlendedImages",
            ROOT / "models" / "selfblendedimages" / "run_image.py",
            ROOT / "models" / "selfblendedimages" / "run_video.py",
            "image scored with the SelfBlendedImages face-crop detector",
            "video sampled into frames and averaged because SelfBlendedImages is image-based",
        ),
        ModelSpec(
            "ucf",
            "UCF",
            ROOT / "models" / "ucf" / "run_image.py",
            ROOT / "models" / "ucf" / "run_video.py",
            "image resized and scored with the UCF shared-feature detector",
            "video sampled into frames and averaged because UCF is image-based",
        ),
    )
}
MODEL_NAMES = list(MODEL_SPECS)


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
    parser.add_argument("--altfreezing-checkpoint", type=Path, help="Optional AltFreezing checkpoint override.")
    parser.add_argument("--effort-checkpoint", type=Path, help="Optional EFFORT checkpoint override.")
    parser.add_argument("--effort-clip-model", type=Path, help="Optional local CLIP ViT-L/14 config directory.")
    parser.add_argument("--f3net-checkpoint", type=Path, help="Optional F3Net detector checkpoint override.")
    parser.add_argument("--recce-checkpoint", type=Path, help="Optional RECCE detector checkpoint override.")
    parser.add_argument("--ucf-checkpoint", type=Path, help="Optional UCF detector checkpoint override.")
    parser.add_argument("--selfblendedimages-weight", type=Path, help="Optional SelfBlendedImages weight override.")
    parser.add_argument(
        "--video-frames",
        type=int,
        help=(
            "Optional frame count for frame-based video scoring. "
            "AltFreezing uses --altfreezing-max-frame instead."
        ),
    )
    parser.add_argument(
        "--altfreezing-max-frame",
        type=int,
        default=ALTFREEZING_DEFAULT_MAX_FRAME,
        help=f"Maximum leading video frames decoded by AltFreezing. Defaults to {ALTFREEZING_DEFAULT_MAX_FRAME}.",
    )
    parser.add_argument(
        "--video-preset",
        choices=list(VIDEO_PRESETS),
        default="quick",
        help="Frame-based video budget used when --video-frames is omitted.",
    )
    parser.add_argument(
        "--video-frame-mode",
        choices=["adaptive", "uniform", "all"],
        default="uniform",
        help="SelfBlendedImages video frame strategy.",
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
        help="Cap SelfBlendedImages RetinaFace detector size for videos. Use 0 to keep the source size.",
    )
    parser.add_argument(
        "--write-altfreezing-output",
        action="store_true",
        help="When scoring a video with AltFreezing, also write its annotated output video.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    parser.add_argument("--verbose", action="store_true", help="Also print each runner's raw stdout and stderr.")

    args = parser.parse_args()
    validations = [
        (args.video_frames, "--video-frames", 1),
        (args.altfreezing_max_frame, "--altfreezing-max-frame", 1),
        (args.video_frame_interval, "--video-frame-interval", 1),
        (args.video_refine_window, "--video-refine-window", 0),
        (args.video_hotspots, "--video-hotspots", 1),
        (args.video_batch_size, "--video-batch-size", 1),
        (args.selfblendedimages_face_max_size, "--selfblendedimages-face-max-size", 0),
    ]
    for value, flag, minimum in validations:
        if value is not None and value < minimum:
            parser.error(f"{flag} must be at least {minimum}.")
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


def format_batch_report(rows: list[dict[str, object]], model_names: list[str]) -> str:
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
        if not large_file(candidate) or not zipfile.is_zipfile(candidate):
            continue
        try:
            with zipfile.ZipFile(candidate) as archive:
                if any(name.endswith("data.pkl") for name in archive.namelist()):
                    return True
        except zipfile.BadZipFile:
            continue

    return (weights_dir / "FFc23_extracted" / "archive" / "data.pkl").is_file()


def missing_reason(items: list[str]) -> str | None:
    return None if not items else "missing " + ", ".join(items)


def altfreezing_missing_items(args: argparse.Namespace) -> list[str]:
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
    return missing


def effort_missing_items(args: argparse.Namespace) -> list[str]:
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
    return missing


def f3net_missing_items(args: argparse.Namespace) -> list[str]:
    weights_dir = ROOT / "models" / "f3net" / "weights"
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
    if not (weights_dir / "xception-b5690688.pth").is_file():
        missing.append("Xception backbone")
    if not detector_ok:
        missing.append("F3Net detector checkpoint")
    if not repo_has_marker(ROOT / "models" / "f3net" / "repo", "models.py"):
        missing.append("F3Net source repo")
    return missing


def recce_missing_items(args: argparse.Namespace) -> list[str]:
    model_root = ROOT / "models" / "recce"
    checkpoint = (
        args.recce_checkpoint.resolve()
        if args.recce_checkpoint
        else model_root / "weights" / "recce_best.pth"
    )

    missing = []
    if not large_file(checkpoint):
        missing.append("RECCE checkpoint")
    if not repo_has_marker(model_root / "repo", "model/network/Recce.py"):
        missing.append("RECCE source repo")
    return missing


def selfblendedimages_missing_items(args: argparse.Namespace) -> list[str]:
    missing = []
    if not selfblended_weight_available(
        ROOT / "models" / "selfblendedimages" / "weights",
        args.selfblendedimages_weight,
    ):
        missing.append("SelfBlendedImages FFc23 checkpoint")
    if not repo_has_marker(ROOT / "models" / "selfblendedimages" / "repo", "src/inference/model.py"):
        missing.append("SelfBlendedImages source repo")
    return missing


def ucf_missing_items(args: argparse.Namespace) -> list[str]:
    checkpoint = args.ucf_checkpoint.resolve() if args.ucf_checkpoint else ROOT / "models" / "ucf" / "ucf_best.pth"
    missing = []
    if not large_file(checkpoint):
        missing.append("UCF checkpoint")
    if not repo_has_marker(ROOT / "models" / "f3net" / "repo", "xception.py"):
        missing.append("shared Xception source")
    return missing


def model_unavailable_reason(model_name: str, media_kind: str, args: argparse.Namespace) -> str | None:
    spec = MODEL_SPECS.get(model_name)
    if spec is None:
        return "unknown model"
    if not spec.runner(media_kind).is_file():
        return f"missing {media_kind} runner"

    checks = {
        "altfreezing": altfreezing_missing_items,
        "effort": effort_missing_items,
        "f3net": f3net_missing_items,
        "recce": recce_missing_items,
        "selfblendedimages": selfblendedimages_missing_items,
        "ucf": ucf_missing_items,
    }
    return missing_reason(checks[model_name](args))


def available_model_names(media_kind: str, args: argparse.Namespace) -> list[str]:
    return [
        model_name
        for model_name in MODEL_NAMES
        if model_unavailable_reason(model_name, media_kind, args) is None
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
        if reason:
            skipped.append({"model": model_name, "note": reason})
    return skipped


def checkpoint_override(model_name: str, args: argparse.Namespace) -> Path | None:
    return {
        "altfreezing": args.altfreezing_checkpoint,
        "effort": args.effort_checkpoint,
        "f3net": args.f3net_checkpoint,
        "recce": args.recce_checkpoint,
        "ucf": args.ucf_checkpoint,
    }.get(model_name)


def video_frame_limit(args: argparse.Namespace) -> int:
    return args.video_frames if args.video_frames is not None else VIDEO_PRESETS[args.video_preset]


def build_command(model_name: str, media_kind: str, args: argparse.Namespace, media_path: Path) -> list[str]:
    spec = MODEL_SPECS[model_name]
    command = [str(RUNNER_PYTHON), str(spec.runner(media_kind)), f"--{media_kind}", str(media_path)]

    checkpoint = checkpoint_override(model_name, args)
    if checkpoint:
        command.extend(["--checkpoint", str(checkpoint.resolve())])
    if model_name == "effort" and args.effort_clip_model:
        command.extend(["--clip-model", str(args.effort_clip_model.resolve())])
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
    match = DETAIL_PATTERN.search(stdout)
    if not match:
        return None
    try:
        payload = json.loads(match.group("value"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


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
    return value[0] if isinstance(value[0], dict) else None


def numeric_value(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def format_peak_frame(frame_info: dict[str, object] | None) -> str:
    if not frame_info:
        return ""
    frame = frame_info.get("frame")
    score = numeric_value(frame_info.get("score"))
    seconds = frame_info.get("time_seconds")
    if frame is None or score is None:
        return ""
    if isinstance(seconds, (int, float)):
        return f"; peak frame {frame} at {float(seconds):.2f}s scored {score:.4f}"
    return f"; peak frame {frame} scored {score:.4f}"


def summarize_details(details: dict[str, object] | None) -> str:
    if not details:
        return ""

    selection = details.get("selection")
    if not isinstance(selection, dict):
        return ""

    top_frame = first_dict(details.get("top_frames"))
    mode = selection.get("mode", "video")
    scored = selection.get("scored_frames", 0)
    decoded = selection.get("decoded_frames", 0)
    attempted = selection.get("attempted_frames", 0)

    if mode == "native_video":
        clips = selection.get("clips_scored", 0)
        max_frame = selection.get("max_frame", "unknown")
        base = f"native video scan; decoded {decoded} frames with max-frame {max_frame}; scored {scored} tracked frames from {clips} clips"
    else:
        base = f"{mode} frame scan; scored {scored}/{decoded} decoded frames ({attempted} attempted)"
    return base + format_peak_frame(top_frame)


def video_metadata(details: dict[str, object] | None) -> dict[str, object] | None:
    if not isinstance(details, dict):
        return None
    video = details.get("video")
    return video if isinstance(video, dict) else None


def format_video_summary(video: dict[str, object] | None) -> str:
    if not video:
        return "Video metadata was not reported by the model runner."

    fields = []
    if video.get("total_frames") is not None:
        fields.append(f"{video['total_frames']} total frames")
    if video.get("fps") is not None:
        fields.append(f"{float(video['fps']):.3f} fps")
    if video.get("width") is not None and video.get("height") is not None:
        fields.append(f"{video['width']}x{video['height']}")
    return "Video metadata: " + ", ".join(fields) + "." if fields else "Video metadata was not reported by the model runner."


def format_frame_time(frame: object, seconds: object) -> str:
    if frame is None:
        return format_seconds(seconds)
    return f"frame {frame} ({format_seconds(seconds)})"


def format_window(window: dict[str, object] | None) -> str | None:
    if not window:
        return None
    start_frame = window.get("start_frame")
    end_frame = window.get("end_frame")
    if start_frame is None or end_frame is None:
        return None
    return (
        f"frames {start_frame}-{end_frame} "
        f"({format_seconds(window.get('start_time_seconds'))} to {format_seconds(window.get('end_time_seconds'))})"
    )


def explain_selection(details: dict[str, object], model_name: str) -> str:
    selection = details.get("selection")
    if not isinstance(selection, dict):
        return "The model scored the media and returned evidence."

    mode = selection.get("mode")
    attempted = selection.get("attempted_frames")
    decoded = selection.get("decoded_frames")
    scored = selection.get("scored_frames")
    video_summary = format_video_summary(video_metadata(details))

    if mode == "adaptive":
        return (
            "SelfBlendedImages used adaptive sampling: it checked broad interval frames first, then inspected "
            f"nearby frames around the strongest moments. It attempted {attempted} frames, decoded {decoded}, "
            f"and scored faces in {scored}. {video_summary}"
        )
    if mode == "uniform":
        label = MODEL_SPECS[model_name].label
        return (
            f"{label} is image-based, so this wrapper sampled frames uniformly across the video and averaged "
            f"their scores. It attempted {attempted} frames, decoded {decoded}, and scored {scored}. {video_summary}"
        )
    if mode == "all":
        return (
            "SelfBlendedImages checked every decoded frame. "
            f"It attempted {attempted} frames, decoded {decoded}, and scored faces in {scored}. {video_summary}"
        )
    if mode == "native_video":
        return (
            "AltFreezing used its native video path: it tracked faces and scored short clips over time. "
            f"It decoded {decoded} frames, scored {scored} tracked face frames, and scored "
            f"{selection.get('clips_scored')} clips. {video_summary}"
        )
    return f"The model used {mode} mode, attempted {attempted} frames, decoded {decoded}, and scored {scored}."


def evidence_source(model_name: str, top_frame: dict[str, object]) -> tuple[str, str]:
    reason = top_frame.get("reason")
    if model_name == "altfreezing":
        clips = top_frame.get("clip_count")
        source = f"{clips} temporal clips" if clips is not None else "temporal clips"
        default_reason = "AltFreezing bases this evidence on face-track motion and appearance over time."
    elif model_name == "selfblendedimages":
        faces = top_frame.get("face_count")
        source = f"{faces} detected face crop(s)" if faces is not None else "detected face crops"
        default_reason = "SelfBlendedImages detects faces, crops them, and uses the strongest face score for each frame."
    elif model_name == "effort":
        source = "sampled frame evidence"
        default_reason = "EFFORT uses a CLIP-L14 detector and reports the strongest fake-class frame response."
    elif model_name == "recce":
        source = "sampled frame evidence"
        default_reason = "RECCE combines reconstruction and classification signals and reports the strongest frame response."
    else:
        source = "sampled frame evidence"
        default_reason = "This frame had one of the strongest scores among the sampled frames."
    return source, reason if isinstance(reason, str) and reason else default_reason


def build_human_explanation(result: dict[str, object]) -> dict[str, object] | None:
    if result["status"] != "ok" or not isinstance(result.get("score"), (int, float)):
        return None

    model_name = str(result["model"])
    label = MODEL_SPECS[model_name].label
    score = float(result["score"])
    details = result.get("details")
    verdict = score_label(score)
    explanation: dict[str, object] = {
        "headline": f"{label} final score is {score:.4f}, which is interpreted as {verdict}.",
        "verdict": verdict,
        "final_score": round(score, 6),
        "method": str(result.get("note") or MODEL_SPECS[model_name].success_note(str(result.get("media")))),
        "strongest_evidence": None,
        "suspicious_timeframe": None,
        "why_this_matters": "Higher scores mean the model saw stronger signs of manipulation.",
        "limitation": "This is model evidence, not proof by itself.",
    }

    if not isinstance(details, dict):
        return explanation

    video = video_metadata(details)
    if video:
        explanation["video"] = video
        explanation["video_summary"] = format_video_summary(video)
    explanation["method"] = explain_selection(details, model_name)

    top_frame = first_dict(details.get("top_frames"))
    if top_frame:
        frame_score = numeric_value(top_frame.get("score"))
        source, reason = evidence_source(model_name, top_frame)
        score_text = "unknown" if frame_score is None else f"{frame_score:.4f}"
        explanation["strongest_evidence"] = (
            f"The strongest example was {format_frame_time(top_frame.get('frame'), top_frame.get('time_seconds'))} "
            f"with a frame score of {score_text}, based on {source}."
        )
        explanation["why_this_matters"] = reason
        explanation["peak_frame"] = top_frame.get("frame")
        explanation["peak_time_seconds"] = top_frame.get("time_seconds")
        explanation["peak_frame_score"] = None if frame_score is None else round(frame_score, 6)

    top_window = first_dict(details.get("suspicious_windows"))
    window_text = format_window(top_window)
    if window_text:
        peak_score = numeric_value(top_window.get("peak_score")) if top_window else None
        score_text = "" if peak_score is None else f" with peak score {peak_score:.4f}"
        explanation["suspicious_timeframe"] = f"The most suspicious localized timeframe is {window_text}{score_text}."

    selection = details.get("selection")
    if isinstance(selection, dict) and isinstance(selection.get("fallback_reason"), str):
        explanation["sampling_note"] = selection["fallback_reason"]
    return explanation


def reasoning_example(result: dict[str, object]) -> dict[str, object]:
    model_name = str(result["model"])
    details = result.get("details")
    example: dict[str, object] = {
        "model": model_name,
        "model_label": MODEL_SPECS[model_name].label,
        "score": result.get("score"),
        "example": str(result.get("note") or "The model produced a score, but no frame-level evidence was emitted."),
    }

    if not isinstance(details, dict):
        return example

    top_frame = first_dict(details.get("top_frames"))
    if not top_frame:
        if isinstance(details.get("explanation"), str):
            example["example"] = details["explanation"]
        return example

    frame = top_frame.get("frame")
    score = numeric_value(top_frame.get("score"))
    if frame is not None and score is not None:
        source, reason = evidence_source(model_name, top_frame)
        example["example"] = (
            f"Peak evidence at frame {frame} ({format_seconds(top_frame.get('time_seconds'))}), "
            f"frame score {score:.4f}, based on {source}; {reason.rstrip('.')}."
        )
        example["frame"] = frame
        example["time_seconds"] = top_frame.get("time_seconds")
        example["frame_score"] = score

    window = first_dict(details.get("suspicious_windows"))
    if window:
        example["suspicious_window"] = window
    return example


def result_notes(results: list[dict[str, object]], status: str) -> list[dict[str, object]]:
    return [
        {"model": result["model"], "note": result["note"]}
        for result in results
        if result["status"] == status
    ]


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
    unavailable = result_notes(results, "unavailable")
    errors = result_notes(results, "error")

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


def append_notes(lines: list[str], title: str, notes: object) -> None:
    if not isinstance(notes, list) or not notes:
        return
    lines.append(title)
    for item in notes:
        if isinstance(item, dict):
            lines.append(f"- {item.get('model')}: {item.get('note')}")


def format_final_report(summary: dict[str, object]) -> str:
    if summary["final_score"] is None:
        lines = ["", "FINAL SUMMARY", "No final score: no model returned a usable score."]
        append_notes(lines, "Unavailable models:", summary.get("unavailable_models"))
        append_notes(lines, "Models with errors:", summary.get("error_models"))
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
            if isinstance(example, dict):
                score = example.get("score")
                score_text = "-" if score is None else f"{float(score):.6f}"
                lines.append(f"- {example.get('model_label') or example.get('model')} score {score_text}: {example.get('example')}")

    model_explanations = summary.get("model_explanations")
    if isinstance(model_explanations, list) and model_explanations:
        lines.append("Detailed explanation:")
        for explanation in model_explanations:
            if not isinstance(explanation, dict):
                continue
            for label, key in (
                ("", "headline"),
                ("  Method: ", "method"),
                ("  Strongest evidence: ", "strongest_evidence"),
                ("  Suspicious timeframe: ", "suspicious_timeframe"),
                ("  Why: ", "why_this_matters"),
                ("  Limitation: ", "limitation"),
            ):
                value = explanation.get(key)
                if value:
                    prefix = "- " if key == "headline" else label
                    lines.append(f"{prefix}{value}")

    append_notes(lines, "Unavailable models skipped:", summary.get("unavailable_models"))
    append_notes(lines, "Models not run because they are not available locally:", summary.get("skipped_models"))
    append_notes(lines, "Models with errors:", summary.get("error_models"))
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

    score_match = SCORE_PATTERN.search(stdout)
    if score_match:
        result["status"] = "ok"
        result["score"] = float(score_match.group("value"))
        result["note"] = summarize_details(details) or MODEL_SPECS[model_name].success_note(media_kind)
        return result

    unavailable_match = UNAVAILABLE_PATTERN.search(stdout)
    if unavailable_match:
        result["status"] = "unavailable"
        result["note"] = unavailable_match.group("reason").strip()
        return result

    result["note"] = last_non_empty_line(stderr, stdout)
    return result


def format_table(results: list[dict[str, object]]) -> str:
    rows = [
        [
            str(result["model"]),
            str(result["status"]),
            "-" if result["score"] is None else f"{float(result['score']):.6f}",
            str(result["note"]),
        ]
        for result in results
    ]
    headers = ["MODEL", "STATUS", "SCORE", "NOTE"]
    widths = [
        max(len(row[index]) for row in [headers, *rows])
        for index in range(len(headers))
    ]

    lines = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) for row in rows)
    return "\n".join(lines)


def result_payload(result: dict[str, object]) -> dict[str, object]:
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
    return item


def print_verbose_results(results: list[dict[str, object]]) -> None:
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
        print(json.dumps({"summary": summary, "results": [result_payload(result) for result in results]}, indent=2))
        print(format_final_report(summary), file=sys.stderr)
    else:
        print(format_table(results))
        print(format_final_report(summary))

    if args.verbose:
        print_verbose_results(results)

    success_count = sum(1 for result in results if result["status"] == "ok")
    error_count = sum(1 for result in results if result["status"] == "error")
    return 0 if success_count > 0 and error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
