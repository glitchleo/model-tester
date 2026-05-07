from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn

from effort_detector import CLIP_IMAGE_SIZE, CLIP_MEAN, CLIP_STD, EffortDetector


MODEL_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = MODEL_ROOT / "weights" / "effort_clip_L14_trainOn_FaceForensic.pth"
DEFAULT_CLIP_MODEL = MODEL_ROOT / "pretrained" / "clip-vit-large-patch14"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EFFORT on one image.")
    parser.add_argument("--image", type=Path, required=True, help="Path to the input image.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Optional EFFORT detector checkpoint.",
    )
    parser.add_argument(
        "--clip-model",
        type=Path,
        default=DEFAULT_CLIP_MODEL,
        help="Local openai/clip-vit-large-patch14 directory containing config.json.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference instead of CUDA.",
    )
    return parser.parse_args()


def fail(message: str, *, unavailable: bool = False, exit_code: int = 1) -> NoReturn:
    prefix = "UNAVAILABLE" if unavailable else "ERROR"
    stream = sys.stdout if unavailable else sys.stderr
    print(f"{prefix}:{message}", file=stream)
    raise SystemExit(0 if unavailable else exit_code)


def torch_load_compat(torch_module, path: Path, device) -> object:
    try:
        return torch_module.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch_module.load(path, map_location=device)


def normalize_state_dict(raw_state_dict: dict) -> dict:
    normalized = {}
    for key, value in raw_state_dict.items():
        clean_key = key
        for prefix in ("module.", "model.", "network."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        normalized[clean_key] = value
    return normalized


def extract_state_dict(checkpoint: object) -> dict:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "net"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                return candidate
        return checkpoint
    fail("Checkpoint does not contain a valid state dict.")


def load_model(torch_module, checkpoint_path: Path, clip_model_dir: Path, device):
    if not checkpoint_path.is_file():
        fail(f"EFFORT checkpoint not found: {checkpoint_path}", unavailable=True)
    if not (clip_model_dir / "config.json").is_file():
        fail(f"CLIP config not found: {clip_model_dir / 'config.json'}", unavailable=True)

    model = EffortDetector(clip_model_dir)
    checkpoint = torch_load_compat(torch_module, checkpoint_path, device)
    state_dict = normalize_state_dict(extract_state_dict(checkpoint))
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        fail(
            "EFFORT checkpoint did not match the local model "
            f"(missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys})."
        )
    model = model.to(device)
    model.eval()
    return model


def fakeness_from_prob(prob_tensor) -> float:
    return float(prob_tensor.reshape(-1).mean().item())


def main() -> int:
    args = parse_args()
    image_path = args.image.resolve()
    if not image_path.is_file():
        fail(f"Input image not found: {image_path}")

    checkpoint_path = args.checkpoint.resolve()
    clip_model_dir = args.clip_model.resolve()

    try:
        import torch
        from PIL import Image
        from torchvision import transforms
    except Exception as exc:
        fail(f"Missing EFFORT runtime dependency: {exc}", unavailable=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_model(torch, checkpoint_path, clip_model_dir, device)

    transform = transforms.Compose(
        [
            transforms.Resize((CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model({"image": tensor}, inference=True)
        score = fakeness_from_prob(predictions["prob"])

    print(f"EFFORT fakeness: {score:.4f}")
    print(f"SCORE:{score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
