from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


@dataclass(frozen=True)
class PipelineSpec:
    pipeline_id: str
    pipeline_name: str
    preprocessing: str
    purpose: str
    filename: str
    steps: tuple[Callable[[np.ndarray], np.ndarray], ...]


def _read_image(path: Path) -> np.ndarray:
    buffer = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def _write_png(path: Path, image: np.ndarray) -> None:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Could not encode preprocessed image: {path}")
    buffer.tofile(str(path))


def _resize(image: np.ndarray, max_side: int = 1024) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= 0:
        raise ValueError("Image has invalid dimensions.")
    scale = min(1.0, max_side / float(longest))
    if scale == 1.0:
        return image.copy()
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _normalize(image: np.ndarray) -> np.ndarray:
    normalized_channels = []
    for channel in cv2.split(image):
        low, high = np.percentile(channel, (1, 99))
        if high <= low:
            normalized_channels.append(channel.copy())
            continue
        stretched = (channel.astype(np.float32) - low) * (255.0 / (high - low))
        normalized_channels.append(np.clip(stretched, 0, 255).astype(np.uint8))
    return cv2.merge(normalized_channels)


def _contrast_enhance(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge((enhanced, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def _denoise(image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)


def _augment(image: np.ndarray) -> np.ndarray:
    flipped = cv2.flip(image, 1)
    sharpen_kernel = np.array(
        [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        dtype=np.float32,
    )
    sharpened = cv2.filter2D(flipped, -1, sharpen_kernel)
    return cv2.convertScaleAbs(sharpened, alpha=1.03, beta=2)


IMAGE_PIPELINES: tuple[PipelineSpec, ...] = (
    PipelineSpec(
        pipeline_id="p0",
        pipeline_name="P0: Baseline",
        preprocessing="Resize only",
        purpose="Minimum preprocessing reference",
        filename="p0_resize_only.png",
        steps=(_resize,),
    ),
    PipelineSpec(
        pipeline_id="p1",
        pipeline_name="P1",
        preprocessing="Resize + normalization",
        purpose="Standard input preparation",
        filename="p1_normalize.png",
        steps=(_resize, _normalize),
    ),
    PipelineSpec(
        pipeline_id="p2",
        pipeline_name="P2",
        preprocessing="Resize + normalization + contrast enhancement",
        purpose="Improve visibility of details",
        filename="p2_contrast.png",
        steps=(_resize, _normalize, _contrast_enhance),
    ),
    PipelineSpec(
        pipeline_id="p4",
        pipeline_name="P4",
        preprocessing="Resize + normalization + augmentation",
        purpose="Improve generalization",
        filename="p4_augmentation.png",
        steps=(_resize, _normalize, _augment),
    ),
    PipelineSpec(
        pipeline_id="p5",
        pipeline_name="P5: Full pipeline",
        preprocessing="Resize + normalization + contrast enhancement + denoising + augmentation",
        purpose="Combine the most promising techniques",
        filename="p5_full_pipeline.png",
        steps=(_resize, _normalize, _contrast_enhance, _denoise, _augment),
    ),
)


def create_preprocessed_images(input_path: str | Path, output_dir: str | Path) -> list[dict[str, str]]:
    source_path = Path(input_path).resolve()
    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    source_image = _read_image(source_path)
    variants = []
    for spec in IMAGE_PIPELINES:
        image = source_image.copy()
        for step in spec.steps:
            image = step(image)

        variant_path = target_dir / spec.filename
        _write_png(variant_path, image)
        variants.append(
            {
                "pipeline_id": spec.pipeline_id,
                "pipeline_name": spec.pipeline_name,
                "preprocessing": spec.preprocessing,
                "purpose": spec.purpose,
                "image_path": str(variant_path),
            }
        )

    return variants
