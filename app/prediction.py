from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import score_image

from .preprocessing import create_preprocessed_images


ROOT = score_image.ROOT
IMAGE_SUFFIXES = score_image.IMAGE_SUFFIXES
VIDEO_SUFFIXES = score_image.VIDEO_SUFFIXES

MODEL_DISPLAY_NAMES = {
    "altfreezing": "AltFreezing",
    "effort": "EFFORT",
    "f3net": "F3Net",
    "recce": "RECCE",
    "selfblendedimages": "SBI",
    "ucf": "UCF",
}

MODEL_ALIASES = {
    "alt-freezing": "altfreezing",
    "alt_freezing": "altfreezing",
    "effort": "effort",
    "f3": "f3net",
    "f3net": "f3net",
    "recce": "recce",
    "sbi": "selfblendedimages",
    "selfblended": "selfblendedimages",
    "self-blended-images": "selfblendedimages",
    "self_blended_images": "selfblendedimages",
    "selfblendedimages": "selfblendedimages",
    "ucf": "ucf",
    "available": "available",
    "all": "all",
    "both": "available",
}

DEFAULT_MODEL_SELECTION = os.getenv("MODEL_TESTER_DEFAULT_MODEL", "available")


def normalize_model_selection(model: str | None) -> str:
    selection = (model or DEFAULT_MODEL_SELECTION or "available").strip().lower()
    normalized = MODEL_ALIASES.get(selection, selection)
    valid = {"available", "all", *score_image.MODEL_NAMES}
    if normalized not in valid:
        options = ", ".join(sorted(valid | {"sbi"}))
        raise ValueError(f"Unknown model '{model}'. Valid options: {options}.")
    return normalized


def _path_or_none(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value)


def build_score_args(
    *,
    model: str,
    video_frames: int | None = None,
    video_preset: str = "quick",
    video_frame_mode: str = "uniform",
    video_frame_interval: int = 20,
    video_refine_window: int = 8,
    video_hotspots: int = 3,
    video_batch_size: int = 2,
    selfblendedimages_face_max_size: int = 960,
    altfreezing_max_frame: int = score_image.ALTFREEZING_DEFAULT_MAX_FRAME,
    write_altfreezing_output: bool = False,
    altfreezing_checkpoint: str | Path | None = None,
    effort_checkpoint: str | Path | None = None,
    effort_clip_model: str | Path | None = None,
    f3net_checkpoint: str | Path | None = None,
    recce_checkpoint: str | Path | None = None,
    ucf_checkpoint: str | Path | None = None,
    selfblendedimages_weight: str | Path | None = None,
) -> SimpleNamespace:
    if video_frames is not None and video_frames < 1:
        raise ValueError("video_frames must be at least 1.")
    if video_preset not in score_image.VIDEO_PRESETS:
        choices = ", ".join(score_image.VIDEO_PRESETS)
        raise ValueError(f"video_preset must be one of: {choices}.")
    if video_frame_mode not in {"adaptive", "uniform", "all"}:
        raise ValueError("video_frame_mode must be adaptive, uniform, or all.")
    if video_frame_interval < 1:
        raise ValueError("video_frame_interval must be at least 1.")
    if video_refine_window < 0:
        raise ValueError("video_refine_window must be 0 or greater.")
    if video_hotspots < 1:
        raise ValueError("video_hotspots must be at least 1.")
    if video_batch_size < 1:
        raise ValueError("video_batch_size must be at least 1.")
    if selfblendedimages_face_max_size < 0:
        raise ValueError("selfblendedimages_face_max_size must be 0 or greater.")
    if altfreezing_max_frame < 1:
        raise ValueError("altfreezing_max_frame must be at least 1.")

    return SimpleNamespace(
        recursive=False,
        model=model,
        altfreezing_checkpoint=_path_or_none(altfreezing_checkpoint),
        effort_checkpoint=_path_or_none(effort_checkpoint),
        effort_clip_model=_path_or_none(effort_clip_model),
        f3net_checkpoint=_path_or_none(f3net_checkpoint),
        recce_checkpoint=_path_or_none(recce_checkpoint),
        ucf_checkpoint=_path_or_none(ucf_checkpoint),
        selfblendedimages_weight=_path_or_none(selfblendedimages_weight),
        video_frames=video_frames,
        altfreezing_max_frame=altfreezing_max_frame,
        video_preset=video_preset,
        video_frame_mode=video_frame_mode,
        video_frame_interval=video_frame_interval,
        video_refine_window=video_refine_window,
        video_hotspots=video_hotspots,
        video_batch_size=video_batch_size,
        selfblendedimages_face_max_size=selfblendedimages_face_max_size,
        write_altfreezing_output=write_altfreezing_output,
        json=True,
        verbose=False,
    )


def status_from_score(fake_score: float) -> str:
    if fake_score >= 0.5:
        return "suspicious"
    if fake_score >= 0.35:
        return "uncertain"
    return "likely_real"


def _round_score(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def standardize_model_result(
    raw_result: dict[str, Any],
    input_type: str,
    processing_time: float,
    *,
    include_details: bool = False,
) -> dict[str, Any]:
    model_id = str(raw_result["model"])
    raw_status = str(raw_result.get("status", "error"))
    fake_score = (
        float(raw_result["score"])
        if raw_status == "ok" and isinstance(raw_result.get("score"), (int, float))
        else None
    )
    real_score = None if fake_score is None else max(0.0, min(1.0, 1.0 - fake_score))
    status = status_from_score(fake_score) if fake_score is not None else raw_status

    item: dict[str, Any] = {
        "model_name": MODEL_DISPLAY_NAMES.get(model_id, model_id),
        "model_id": model_id,
        "input_type": input_type,
        "fake_score": _round_score(fake_score),
        "real_score": _round_score(real_score),
        "status": status,
        "processing_time": round(processing_time, 3),
    }

    note = raw_result.get("note")
    if note:
        item["message"] = str(note)
    if include_details and raw_result.get("details") is not None:
        item["details"] = raw_result["details"]
    return item


def _combined_prediction(
    results: list[dict[str, Any]],
    input_type: str,
    processing_time: float,
) -> dict[str, Any]:
    scored = [
        result
        for result in results
        if isinstance(result.get("fake_score"), (int, float))
    ]
    if not scored:
        return {
            "model_name": "combined",
            "input_type": input_type,
            "fake_score": None,
            "real_score": None,
            "status": "no_score",
            "processing_time": round(processing_time, 3),
            "models_scored": [],
        }

    fake_score = sum(float(result["fake_score"]) for result in scored) / len(scored)
    return {
        "model_name": "combined",
        "input_type": input_type,
        "fake_score": _round_score(fake_score),
        "real_score": _round_score(1.0 - fake_score),
        "status": status_from_score(fake_score),
        "processing_time": round(processing_time, 3),
        "models_scored": [str(result["model_id"]) for result in scored],
    }


def analyze_media(
    media_path: str | Path,
    *,
    model: str | None = None,
    include_details: bool = False,
    **score_options: Any,
) -> dict[str, Any]:
    path = Path(media_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input media not found: {path}")

    input_type = score_image.infer_media_kind(path)
    if input_type is None:
        supported = ", ".join(sorted(IMAGE_SUFFIXES | VIDEO_SUFFIXES))
        raise ValueError(f"Unsupported media type for {path}. Supported extensions: {supported}.")

    selection = normalize_model_selection(model)
    args = build_score_args(model=selection, **score_options)
    selected_models = score_image.selected_model_names(selection, input_type, args)
    skipped_models = score_image.skipped_model_notes(selection, input_type, args)
    if not selected_models:
        summary = {
            "model_name": "combined",
            "input_type": input_type,
            "fake_score": None,
            "real_score": None,
            "status": "no_score",
            "processing_time": 0.0,
            "models_scored": [],
        }
        return {
            "input_path": str(path),
            "input_type": input_type,
            "model_selection": selection,
            "summary": summary,
            "results": [],
            "skipped_models": skipped_models,
        }

    started = time.perf_counter()
    standardized_results = []
    for model_name in selected_models:
        model_started = time.perf_counter()
        raw_result = score_image.run_model(model_name, input_type, args, path)
        elapsed = time.perf_counter() - model_started
        standardized_results.append(
            standardize_model_result(
                raw_result,
                input_type,
                elapsed,
                include_details=include_details,
            )
        )

    total_elapsed = time.perf_counter() - started
    summary = _combined_prediction(standardized_results, input_type, total_elapsed)
    return {
        "input_path": str(path),
        "input_type": input_type,
        "model_selection": selection,
        "summary": summary,
        "results": standardized_results,
        "skipped_models": skipped_models,
    }


def analyze_image_preprocessing_pipelines(
    media_path: str | Path,
    *,
    model: str | None = None,
    include_details: bool = False,
    **score_options: Any,
) -> dict[str, Any]:
    path = Path(media_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")

    input_type = score_image.infer_media_kind(path)
    if input_type != "image":
        supported = ", ".join(sorted(IMAGE_SUFFIXES))
        raise ValueError(f"Pipeline comparison only supports images. Supported extensions: {supported}.")

    selection = normalize_model_selection(model)
    args = build_score_args(model=selection, **score_options)
    selected_models = score_image.selected_model_names(selection, "image", args)
    skipped_models = score_image.skipped_model_notes(selection, "image", args)
    if not selected_models:
        summary = {
            "model_name": "combined",
            "input_type": "image",
            "fake_score": None,
            "real_score": None,
            "status": "no_score",
            "processing_time": 0.0,
            "models_scored": [],
            "pipelines_scored": [],
        }
        return {
            "analysis_type": "preprocessing_comparison",
            "input_path": str(path),
            "input_type": "image",
            "model_selection": selection,
            "summary": summary,
            "pipelines": [],
            "models": [],
            "results": [],
            "skipped_models": skipped_models,
        }

    started = time.perf_counter()
    variant_dir = ROOT / "outputs" / "preprocessed" / uuid.uuid4().hex
    variants = create_preprocessed_images(path, variant_dir)
    pipelines = []
    flattened_results = []

    for variant in variants:
        pipeline_started = time.perf_counter()
        pipeline_results = []
        variant_path = Path(variant["image_path"])

        for model_name in selected_models:
            model_started = time.perf_counter()
            raw_result = score_image.run_model(model_name, "image", args, variant_path)
            elapsed = time.perf_counter() - model_started
            result = standardize_model_result(
                raw_result,
                "image",
                elapsed,
                include_details=include_details,
            )
            result["pipeline_id"] = variant["pipeline_id"]
            result["pipeline_name"] = variant["pipeline_name"]
            pipeline_results.append(result)
            flattened_results.append(result)

        pipeline_elapsed = time.perf_counter() - pipeline_started
        pipelines.append(
            {
                **variant,
                "summary": _combined_prediction(pipeline_results, "image", pipeline_elapsed),
                "results": pipeline_results,
            }
        )

    total_elapsed = time.perf_counter() - started
    summary = _combined_prediction(flattened_results, "image", total_elapsed)
    summary["pipelines_scored"] = [
        pipeline["pipeline_id"]
        for pipeline in pipelines
        if isinstance(pipeline["summary"].get("fake_score"), (int, float))
    ]
    summary["pipelines_run"] = len(pipelines)

    return {
        "analysis_type": "preprocessing_comparison",
        "input_path": str(path),
        "input_type": "image",
        "model_selection": selection,
        "summary": summary,
        "pipelines": pipelines,
        "models": [
            {
                "model_id": model_name,
                "model_name": MODEL_DISPLAY_NAMES.get(model_name, model_name),
            }
            for model_name in selected_models
        ],
        "results": flattened_results,
        "skipped_models": skipped_models,
        "preprocessed_dir": str(variant_dir),
    }


def cli_payload(analysis: dict[str, Any], *, prefer_single_result: bool = True) -> dict[str, Any]:
    results = analysis.get("results")
    if prefer_single_result and isinstance(results, list) and len(results) == 1:
        return results[0]
    return {
        "summary": analysis["summary"],
        "results": results,
        "skipped_models": analysis.get("skipped_models", []),
    }


def model_availability(input_type: str | None = None) -> dict[str, Any]:
    if input_type is not None and input_type not in {"image", "video"}:
        raise ValueError("input_type must be image or video.")

    media_types = [input_type] if input_type else ["image", "video"]
    args = build_score_args(model="available")
    items = []
    for media_kind in media_types:
        available = set(score_image.available_model_names(media_kind, args))
        skipped = {
            item["model"]: item["note"]
            for item in score_image.skipped_model_notes("available", media_kind, args)
        }
        for model_id in score_image.MODEL_NAMES:
            items.append(
                {
                    "model_name": MODEL_DISPLAY_NAMES.get(model_id, model_id),
                    "model_id": model_id,
                    "input_type": media_kind,
                    "available": model_id in available,
                    "message": skipped.get(model_id, "ready"),
                }
            )

    return {"models": items}


def analysis_has_score(analysis: dict[str, Any]) -> bool:
    results = analysis.get("results")
    if not isinstance(results, list):
        return False
    return any(isinstance(result.get("fake_score"), (int, float)) for result in results)
