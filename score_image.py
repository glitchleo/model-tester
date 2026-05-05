from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
MODEL_NAMES = ["altfreezing", "f3net", "selfblendedimages"]
READY_MODEL_NAMES = ["altfreezing", "selfblendedimages"]
MODEL_LABELS = {
    "altfreezing": "AltFreezing",
    "f3net": "F3Net",
    "selfblendedimages": "SelfBlendedImages",
}
RUNNERS = {
    "image": {
        "altfreezing": ROOT / "models" / "altfreezing" / "run_image.py",
        "f3net": ROOT / "models" / "f3net" / "run_image.py",
        "selfblendedimages": ROOT / "models" / "selfblendedimages" / "run_image.py",
    },
    "video": {
        "altfreezing": ROOT / "models" / "altfreezing" / "run_video.py",
        "f3net": ROOT / "models" / "f3net" / "run_video.py",
        "selfblendedimages": ROOT / "models" / "selfblendedimages" / "run_video.py",
    },
}
SUCCESS_NOTES = {
    ("image", "altfreezing"): "single image duplicated into a short video clip because the original model is video-based",
    ("video", "altfreezing"): "native video model using temporal face-track clips",
    ("video", "f3net"): "video sampled into frames and averaged because F3Net is image-based",
    ("video", "selfblendedimages"): "video sampled into frames and averaged because SelfBlendedImages is image-based",
}
SCORE_PATTERN = re.compile(r"^SCORE:(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$", re.MULTILINE)
UNAVAILABLE_PATTERN = re.compile(r"^UNAVAILABLE:(?P<reason>.+)$", re.MULTILINE)
DETAIL_PATTERN = re.compile(r"^DETAIL_JSON:(?P<value>\{.*\})$", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score one image or video with one model or all configured models.",
    )
    parser.add_argument("media", type=Path, help="Path to the input image or video.")
    parser.add_argument(
        "--model",
        choices=["both", "all", *MODEL_NAMES],
        default="both",
        help="Which model(s) to run. Defaults to both ready models: AltFreezing and SelfBlendedImages.",
    )
    parser.add_argument(
        "--altfreezing-checkpoint",
        type=Path,
        help="Optional override for the AltFreezing checkpoint.",
    )
    parser.add_argument(
        "--f3net-checkpoint",
        type=Path,
        help="Optional override for the F3Net detector checkpoint.",
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
            "Optional frame count for video scoring. For AltFreezing this is --max-frame; "
            "for F3Net this is uniform sampled frames; for SelfBlendedImages this is the adaptive/uniform frame limit."
        ),
    )
    parser.add_argument(
        "--video-frame-mode",
        choices=["adaptive", "uniform", "all"],
        default="adaptive",
        help=(
            "SelfBlendedImages video frame strategy. Adaptive does a coarse interval pass and then refines "
            "around the highest-scoring moments."
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


def selected_model_names(selection: str) -> list[str]:
    if selection == "both":
        return READY_MODEL_NAMES
    if selection == "all":
        return MODEL_NAMES
    return [selection]


def build_command(model_name: str, media_kind: str, args: argparse.Namespace, media_path: Path) -> list[str]:
    input_flag = "--image" if media_kind == "image" else "--video"
    command = [sys.executable, str(RUNNERS[media_kind][model_name]), input_flag, str(media_path)]
    if model_name == "altfreezing" and args.altfreezing_checkpoint:
        command.extend(["--checkpoint", str(args.altfreezing_checkpoint.resolve())])
    if model_name == "f3net" and args.f3net_checkpoint:
        command.extend(["--checkpoint", str(args.f3net_checkpoint.resolve())])
    if model_name == "selfblendedimages" and args.selfblendedimages_weight:
        command.extend(["--weight", str(args.selfblendedimages_weight.resolve())])
    if media_kind == "video" and args.video_frames is not None:
        frame_flag = "--max-frame" if model_name == "altfreezing" else "--frames"
        command.extend([frame_flag, str(args.video_frames)])
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


def explain_selection(details: dict[str, object]) -> str:
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

    explanation["method"] = explain_selection(details)
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
    else:
        support = "sampled frame evidence"
        why = "this was one of the highest-scoring sampled frames"

    if isinstance(reason, str) and reason:
        why = reason

    if frame is not None and frame_score is not None:
        example["example"] = (
            f"Peak evidence at frame {frame} ({format_seconds(time_seconds)}), "
            f"frame score {float(frame_score):.4f}, based on {support}; {why}."
        )
        example["frame"] = frame
        example["time_seconds"] = time_seconds
        example["frame_score"] = frame_score

    windows = details.get("suspicious_windows")
    if isinstance(windows, list) and windows:
        example["suspicious_window"] = windows[0]
    return example


def build_summary(results: list[dict[str, object]]) -> dict[str, object]:
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
    }


def format_final_report(summary: dict[str, object]) -> str:
    if summary["final_score"] is None:
        return "\nFINAL SUMMARY\nNo final score: no model returned a usable score."

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
    if not media_path.is_file():
        print(f"Input media not found: {media_path}", file=sys.stderr)
        return 2

    media_kind = infer_media_kind(media_path)
    if media_kind is None:
        supported = ", ".join(sorted(IMAGE_SUFFIXES | VIDEO_SUFFIXES))
        print(f"Unsupported media type for {media_path}. Supported extensions: {supported}", file=sys.stderr)
        return 2

    selected_models = selected_model_names(args.model)
    results = [run_model(model_name, media_kind, args, media_path) for model_name in selected_models]
    summary = build_summary(results)

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
