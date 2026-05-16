from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .prediction import IMAGE_SUFFIXES, ROOT, VIDEO_SUFFIXES, analyze_media, model_availability


app = FastAPI(
    title="Deepfake Model Tester API",
    version="0.1.0",
    description="Upload an image or video and run the standardized model prediction pipeline.",
)

UPLOAD_DIR = ROOT / "outputs" / "uploads"
RESULT_DIR = ROOT / "outputs" / "api_results"
RESULTS: dict[str, dict[str, Any]] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_filename(filename: str | None) -> str:
    name = Path(filename or "upload").name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or "upload"


async def _save_upload(file: UploadFile, allowed_suffixes: set[str]) -> Path:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed_suffixes:
        supported = ", ".join(sorted(allowed_suffixes))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported upload type '{suffix or 'none'}'. Supported extensions: {supported}.",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    output_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{_safe_filename(file.filename)}"
    with output_path.open("wb") as handle:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    return output_path


def _store_result(
    *,
    upload_path: Path,
    original_filename: str | None,
    analysis: dict[str, Any],
) -> dict[str, Any]:
    result_id = uuid.uuid4().hex
    record = {
        "id": result_id,
        "created_at": _now(),
        "filename": original_filename,
        "stored_path": str(upload_path),
        **analysis,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / f"{result_id}.json").write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )
    RESULTS[result_id] = record
    return record


async def _analyze_upload(
    *,
    file: UploadFile,
    allowed_suffixes: set[str],
    model: str,
    video_frames: int | None,
    video_preset: str,
    include_details: bool,
) -> dict[str, Any]:
    upload_path = await _save_upload(file, allowed_suffixes)
    try:
        analysis = analyze_media(
            upload_path,
            model=model,
            video_frames=video_frames,
            video_preset=video_preset,
            include_details=include_details,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _store_result(
        upload_path=upload_path,
        original_filename=file.filename,
        analysis=analysis,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def models(input_type: str | None = None) -> dict[str, Any]:
    try:
        return model_availability(input_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    model: str = Form("available"),
    include_details: bool = Form(False),
) -> dict[str, Any]:
    return await _analyze_upload(
        file=file,
        allowed_suffixes=IMAGE_SUFFIXES,
        model=model,
        video_frames=None,
        video_preset="quick",
        include_details=include_details,
    )


@app.post("/analyze-video")
async def analyze_video(
    file: UploadFile = File(...),
    model: str = Form("available"),
    video_frames: int | None = Form(None),
    video_preset: str = Form("quick"),
    include_details: bool = Form(False),
) -> dict[str, Any]:
    return await _analyze_upload(
        file=file,
        allowed_suffixes=VIDEO_SUFFIXES,
        model=model,
        video_frames=video_frames,
        video_preset=video_preset,
        include_details=include_details,
    )


@app.get("/result/{result_id}")
def get_result(result_id: str) -> dict[str, Any]:
    if result_id in RESULTS:
        return RESULTS[result_id]

    result_path = RESULT_DIR / f"{result_id}.json"
    if result_path.is_file():
        record = json.loads(result_path.read_text(encoding="utf-8"))
        RESULTS[result_id] = record
        return record

    raise HTTPException(status_code=404, detail=f"Result not found: {result_id}")
