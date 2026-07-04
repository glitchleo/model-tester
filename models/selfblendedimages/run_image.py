from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path


MODEL_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SelfBlendedImages on one image.")
    parser.add_argument("--image", type=Path, required=True, help="Path to the input image.")
    parser.add_argument(
        "--weight",
        type=Path,
        help="Optional checkpoint file. Defaults to models/selfblendedimages/weights/*.tar.",
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
    fail(f"Could not find the SelfBlendedImages repo root under {repo_dir}", unavailable=True)


def rebuild_checkpoint_archive(extracted_dir: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for file_path in sorted(extracted_dir.rglob("*")):
            if not file_path.is_file():
                continue
            arcname = file_path.relative_to(extracted_dir).as_posix()
            info = zipfile.ZipInfo(arcname, date_time=(2024, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            with file_path.open("rb") as handle:
                archive.writestr(info, handle.read())
    return output_path


def is_valid_checkpoint_archive(path: Path) -> bool:
    if not path.is_file():
        return False
    if not zipfile.is_zipfile(path):
        return False
    try:
        import torch

        torch_load_compat(torch, path, "cpu")
        return True
    except Exception:
        return False


def resolve_weight_path(weights_dir: Path, provided: Path | None) -> Path:
    if provided:
        resolved = provided.resolve()
        if not resolved.is_file():
            fail(f"Weight file not found: {resolved}")
        return resolved

    tar_candidates = sorted(weights_dir.glob("*.tar"))
    for candidate in tar_candidates:
        if is_valid_checkpoint_archive(candidate):
            return candidate

    extracted_dir = weights_dir / "FFc23_extracted"
    if extracted_dir.is_dir():
        repaired = weights_dir / "FFc23_repacked.tar"
        if is_valid_checkpoint_archive(repaired):
            return repaired
        return rebuild_checkpoint_archive(extracted_dir, repaired)

    fail(
        "No SelfBlendedImages checkpoint file was found in models/selfblendedimages/weights.",
        unavailable=True,
    )


def torch_load_compat(torch_module, path: Path, device) -> object:
    try:
        return torch_module.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch_module.load(path, map_location=device)


def load_retinaface_model(retina_models, torch_module, cache_dir: Path, max_size: int, device):
    model_name = "resnet50_2020-07-20"
    model_info = retina_models.models[model_name]
    model = model_info.model(max_size=max_size, device=device)
    checkpoint_path = cache_dir / "hub" / "checkpoints" / Path(model_info.url).name.replace(
        "-f168fae3c.zip",
        ".pth",
    )

    if checkpoint_path.is_file():
        state_dict = torch_load_compat(torch_module, checkpoint_path, "cpu")
    else:
        state_dict = torch_module.utils.model_zoo.load_url(
            model_info.url,
            progress=True,
            map_location="cpu",
        )

    model.load_state_dict(state_dict)
    return model


def main() -> int:
    args = parse_args()
    image_path = args.image.resolve()
    if not image_path.is_file():
        fail(f"Input image not found: {image_path}")

    repo_root = find_repo_root(MODEL_ROOT / "repo", "src/inference/model.py")
    weight_path = resolve_weight_path(MODEL_ROOT / "weights", args.weight)
    inference_dir = repo_root / "src" / "inference"
    sys.path.insert(0, str(inference_dir))
    cache_dir = MODEL_ROOT / ".torch-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(cache_dir)

    try:
        import cv2
        import torch
        from efficientnet_pytorch import EfficientNet
        import retinaface.pre_trained_models as retina_models
        from preprocess import extract_face
    except Exception as exc:
        fail(f"Missing SelfBlendedImages runtime dependency: {exc}")

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

    frame = cv2.imread(str(image_path))
    if frame is None:
        fail(f"Could not read image: {image_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_detector = load_retinaface_model(
        retina_models,
        torch,
        cache_dir,
        max_size=max(frame.shape[:2]),
        device=device,
    )
    face_detector.eval()

    face_list = extract_face(frame, face_detector)
    if not face_list:
        fail(f"No face was detected in {image_path}.")

    with torch.no_grad():
        batch = torch.tensor(face_list, device=device).float() / 255.0
        scores = detector(batch).softmax(1)[:, 1].detach().cpu().numpy().tolist()

    score = float(max(scores))
    print(f"SelfBlendedImages fakeness: {score:.4f}")
    print(f"SCORE:{score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
