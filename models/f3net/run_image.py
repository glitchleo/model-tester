from __future__ import annotations

import argparse
import sys
from pathlib import Path


MODEL_ROOT = Path(__file__).resolve().parent
BACKBONE_NAME = "xception-b5690688.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run F3Net on one image.")
    parser.add_argument("--image", type=Path, required=True, help="Path to the input image.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional trained F3Net detector checkpoint.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference instead of CUDA.",
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
    fail(f"Could not find the F3Net repo root under {repo_dir}", unavailable=True)


def resolve_detector_checkpoint(weights_dir: Path, provided: Path | None) -> Path:
    if provided:
        resolved = provided.resolve()
        if not resolved.is_file():
            fail(f"Checkpoint file not found: {resolved}")
        return resolved

    candidates = sorted(path for path in weights_dir.glob("*.pth") if path.name != BACKBONE_NAME)
    if not candidates:
        fail(
            "F3Net only has the Xception backbone right now. Add a trained F3Net detector checkpoint to models/f3net/weights first.",
            unavailable=True,
        )
    return candidates[0]


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
        for key in ("state_dict", "model"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                return candidate
        return checkpoint
    fail("Checkpoint does not contain a valid state dict.")


def infer_mode(state_dict: dict) -> str:
    keys = state_dict.keys()
    has_fad = any(key.startswith("FAD_head.") or key.startswith("FAD_xcep.") for key in keys)
    has_lfs = any(key.startswith("LFS_head.") or key.startswith("LFS_xcep.") for key in keys)
    has_original = any(key.startswith("xcep.") for key in keys)

    if has_fad and has_lfs:
        return "Both"
    if has_fad:
        return "FAD"
    if has_lfs:
        return "LFS"
    if has_original:
        return "Original"

    fail(
        "Could not infer the F3Net mode from the checkpoint. Make sure this is a trained F3Net detector checkpoint and not only the Xception backbone.",
        unavailable=True,
    )


def torch_load_compat(torch_module, path: Path, device) -> object:
    try:
        return torch_module.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch_module.load(path, map_location=device)


def main() -> int:
    args = parse_args()
    image_path = args.image.resolve()
    if not image_path.is_file():
        fail(f"Input image not found: {image_path}")

    weights_dir = MODEL_ROOT / "weights"
    backbone_path = weights_dir / BACKBONE_NAME
    if not backbone_path.is_file():
        fail(
            f"Missing the required Xception backbone at {backbone_path}.",
            unavailable=True,
        )

    checkpoint_path = resolve_detector_checkpoint(weights_dir, args.checkpoint)
    repo_root = find_repo_root(MODEL_ROOT / "repo", "models.py")
    sys.path.insert(0, str(repo_root))

    try:
        import torch
        from PIL import Image
        from torchvision import transforms
        import models as f3net_models
    except Exception as exc:
        fail(f"Missing F3Net runtime dependency: {exc}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = torch_load_compat(torch, checkpoint_path, device)
    state_dict = normalize_state_dict(extract_state_dict(checkpoint))
    mode = infer_mode(state_dict)

    original_get_xcep_state_dict = f3net_models.get_xcep_state_dict

    def patched_get_xcep_state_dict(pretrained_path: str | None = None):
        return original_get_xcep_state_dict(str(backbone_path))

    f3net_models.get_xcep_state_dict = patched_get_xcep_state_dict

    model = f3net_models.F3Net(mode=mode).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Lambda(lambda tensor: tensor * 2.0 - 1.0),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        _, logits = model(tensor)
        score = float(torch.sigmoid(logits.squeeze()).item())

    print(f"F3Net ({mode}) fakeness: {score:.4f}")
    print(f"SCORE:{score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
