from __future__ import annotations

import argparse
import importlib
import sys
import warnings
from functools import partial
from pathlib import Path
from typing import NoReturn


MODEL_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = MODEL_ROOT / "weights" / "recce_best.pth"
RECCE_IMAGE_SIZE = 299


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RECCE on one image.")
    parser.add_argument("--image", type=Path, required=True, help="Path to the input image.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Optional RECCE detector checkpoint.",
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


def find_repo_root(repo_dir: Path, marker: str) -> Path:
    if not repo_dir.exists():
        fail(f"Missing repo directory: {repo_dir}", unavailable=True)
    candidates = [repo_dir]
    candidates.extend(sorted(path for path in repo_dir.iterdir() if path.is_dir()))
    for candidate in candidates:
        if (candidate / marker).exists():
            return candidate
    fail(f"Could not find {marker} under {repo_dir}", unavailable=True)


def torch_load_compat(torch_module, path: Path, device) -> object:
    try:
        return torch_module.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch_module.load(path, map_location=device)


def extract_state_dict(checkpoint: object) -> dict:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "net"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                return candidate
        return checkpoint
    fail("Checkpoint does not contain a valid state dict.")


def normalize_recce_state_dict(raw_state_dict: dict) -> dict:
    normalized = {}
    for key, value in raw_state_dict.items():
        clean_key = key
        for prefix in ("module.", "network."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        if clean_key.startswith("model."):
            clean_key = clean_key[len("model.") :]
        elif clean_key.startswith("backbone."):
            continue
        normalized[clean_key] = value
    return normalized


def infer_num_classes(state_dict: dict) -> int:
    fc_weight = state_dict.get("fc.weight")
    if hasattr(fc_weight, "shape") and len(fc_weight.shape) == 2:
        return int(fc_weight.shape[0])
    return 2


def load_recce_class(repo_root: Path):
    sys.path.insert(0, str(repo_root))
    try:
        recce_module = importlib.import_module("model.network.Recce")
    except Exception as exc:
        fail(f"Could not import RECCE source repo: {exc}", unavailable=True)
    recce_module.encoder_params["xception"]["init_op"] = partial(recce_module.xception, pretrained=False)
    return recce_module.Recce


def load_model(torch_module, checkpoint_path: Path, device):
    if not checkpoint_path.is_file():
        fail(f"RECCE checkpoint not found: {checkpoint_path}", unavailable=True)
    repo_root = find_repo_root(MODEL_ROOT / "repo", "model/network/Recce.py")
    Recce = load_recce_class(repo_root)
    checkpoint = torch_load_compat(torch_module, checkpoint_path, device)
    state_dict = normalize_recce_state_dict(extract_state_dict(checkpoint))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = Recce(num_classes=infer_num_classes(state_dict))
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        fail(
            "RECCE checkpoint did not match the local model "
            f"(missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys})."
        )
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def fakeness_from_logits(torch_module, logits) -> float:
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    else:
        logits = logits.reshape(logits.size(0), -1)
    if logits.size(1) == 1:
        scores = torch_module.sigmoid(logits[:, 0])
    else:
        scores = torch_module.softmax(logits, dim=1)[:, 1]
    return float(scores.mean().item())


def main() -> int:
    args = parse_args()
    image_path = args.image.resolve()
    if not image_path.is_file():
        fail(f"Input image not found: {image_path}")

    try:
        import torch
        from PIL import Image
        from torchvision import transforms
    except Exception as exc:
        fail(f"Missing RECCE runtime dependency: {exc}", unavailable=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_model(torch, args.checkpoint.resolve(), device)

    transform = transforms.Compose(
        [
            transforms.Resize((RECCE_IMAGE_SIZE, RECCE_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Lambda(lambda tensor: tensor * 2.0 - 1.0),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        score = fakeness_from_logits(torch, logits)

    print(f"RECCE fakeness: {score:.4f}")
    print(f"SCORE:{score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
