from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_VENV = ROOT / ".venv"
F3NET_BACKBONE = "xception-b5690688.pth"
EFFORT_CHECKPOINT = "effort_clip_L14_trainOn_FaceForensic.pth"
EFFORT_CLIP_DIR = "clip-vit-large-patch14"
RECCE_CHECKPOINT = "recce_best.pth"


@dataclass(frozen=True)
class Check:
    status: str
    area: str
    detail: str
    fix: str = ""


IMPORT_GROUPS = {
    "core runtime": [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("cv2", "opencv-python"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
        ("tqdm", "tqdm"),
    ],
    "backend API": [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("multipart", "python-multipart"),
    ],
    "AltFreezing runtime": [
        ("albumentations", "albumentations"),
        ("contexttimer", "contexttimer"),
        ("einops", "einops"),
        ("ffmpeg", "ffmpeg-python"),
        ("filterpy", "filterpy"),
        ("fvcore", "fvcore"),
        ("lmdb", "lmdb"),
        ("opencv_transforms", "opencv-transforms"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("simplejson", "simplejson"),
        ("sklearn", "scikit-learn"),
        ("tabulate", "tabulate"),
        ("tensorboardX", "tensorboardX"),
        ("termcolor", "termcolor"),
        ("turbojpeg", "PyTurboJPEG"),
        ("yaml", "PyYAML"),
        ("yacs", "yacs"),
    ],
    "SelfBlendedImages runtime": [
        ("efficientnet_pytorch", "efficientnet-pytorch"),
        ("retinaface", "retinaface-pytorch"),
    ],
    "EFFORT runtime": [
        ("transformers", "transformers"),
    ],
    "RECCE runtime": [
        ("timm", "timm"),
    ],
}
MODEL_IMPORT_GROUPS = {
    "all": list(IMPORT_GROUPS),
    "altfreezing": ["core runtime", "AltFreezing runtime"],
    "effort": ["core runtime", "EFFORT runtime"],
    "f3net": ["core runtime"],
    "recce": ["core runtime", "RECCE runtime"],
    "selfblendedimages": ["core runtime", "SelfBlendedImages runtime"],
    "ucf": ["core runtime"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the local model-tester setup.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="After setup checks pass, run score_image.py on a suitable sample input.",
    )
    parser.add_argument(
        "--model",
        choices=["all", "altfreezing", "effort", "f3net", "recce", "selfblendedimages", "ucf"],
        default="all",
        help="Model to validate and use for --smoke. Defaults to all.",
    )
    return parser.parse_args()


def path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def find_repo_root(repo_dir: Path, marker: str) -> Path | None:
    if not repo_dir.exists():
        return None
    candidates = [repo_dir]
    candidates.extend(sorted(path for path in repo_dir.iterdir() if path.is_dir()))
    for candidate in candidates:
        if (candidate / marker).exists():
            return candidate
    return None


def selfblended_weight_available(weights_dir: Path) -> bool:
    for candidate in sorted(weights_dir.glob("*.tar")):
        if candidate.stat().st_size < 1024 * 1024:
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


def check_python() -> list[Check]:
    checks: list[Check] = []
    version = ".".join(str(part) for part in sys.version_info[:3])
    if sys.version_info < (3, 10):
        checks.append(
            Check(
                "FAIL",
                "Python",
                f"Running Python {version}; this workspace is verified with Python 3.11.",
                "Create the venv with py -3.11 -m venv .venv.",
            )
        )
    else:
        checks.append(Check("OK", "Python", f"Running Python {version}."))

    executable = Path(sys.executable).resolve()
    if REPO_VENV.exists() and path_is_relative_to(executable, REPO_VENV.resolve()):
        checks.append(Check("OK", "Interpreter", f"Using repo venv: {executable}"))
    elif REPO_VENV.exists():
        checks.append(
            Check(
                "WARN",
                "Interpreter",
                f"Using {executable}, not the repo .venv.",
                r"Activate it with .\.venv\Scripts\Activate.ps1 or run .\.venv\Scripts\python.exe.",
            )
        )
    else:
        checks.append(
            Check(
                "WARN",
                "Interpreter",
                "No .venv folder was found.",
                "Create it with py -3.11 -m venv .venv, then install requirements.",
            )
        )
    return checks


def check_imports(model: str) -> list[Check]:
    checks: list[Check] = []
    for group_name in MODEL_IMPORT_GROUPS[model]:
        modules = IMPORT_GROUPS[group_name]
        missing = [package for module, package in modules if not module_available(module)]
        if missing:
            checks.append(
                Check(
                    "FAIL",
                    group_name,
                    "Missing: " + ", ".join(missing),
                    "Install with python -m pip install -r requirements.txt inside the active venv.",
                )
            )
        else:
            checks.append(Check("OK", group_name, "All expected imports are available."))
    return checks


def check_torch(model: str) -> list[Check]:
    if not module_available("torch"):
        return []

    checks: list[Check] = []
    try:
        import torch
    except Exception as exc:
        return [Check("FAIL", "PyTorch", f"PyTorch import failed: {exc}")]

    torch_version = getattr(torch, "__version__", "unknown")
    cuda_version = getattr(torch.version, "cuda", None) or "none"
    checks.append(Check("OK", "PyTorch", f"torch {torch_version}; CUDA runtime {cuda_version}."))

    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        checks.append(Check("OK", "CUDA", f"CUDA is available: {device}."))
    elif model in {"all", "altfreezing"}:
        detail = (
            "CUDA is not available. SelfBlendedImages, F3Net, and UCF can run on CPU, "
            "but AltFreezing needs CUDA here."
        )
        checks.append(
            Check(
                "WARN",
                "CUDA",
                detail,
                "Install a CUDA-enabled PyTorch build and NVIDIA driver if you need AltFreezing.",
            )
        )
    else:
        checks.append(Check("OK", "CUDA", f"CUDA is not available; {model} can run on CPU."))
    return checks


def check_repos(model: str) -> list[Check]:
    markers = [
        ("altfreezing", "AltFreezing repo", ROOT / "models" / "altfreezing" / "repo", "demo.py"),
        (
            "effort",
            "EFFORT repo",
            ROOT / "models" / "effort" / "repo",
            "DeepfakeBench/training/demo.py",
        ),
        ("f3net", "F3Net repo", ROOT / "models" / "f3net" / "repo", "models.py"),
        ("recce", "RECCE repo", ROOT / "models" / "recce" / "repo", "model/network/Recce.py"),
        ("ucf", "UCF shared Xception source", ROOT / "models" / "f3net" / "repo", "xception.py"),
        (
            "selfblendedimages",
            "SelfBlendedImages repo",
            ROOT / "models" / "selfblendedimages" / "repo",
            "src/inference/model.py",
        ),
    ]

    checks: list[Check] = []
    for model_name, area, repo_dir, marker in markers:
        if model not in {"all", model_name}:
            continue
        repo_root = find_repo_root(repo_dir, marker)
        if repo_root is None:
            checks.append(
                Check(
                    "FAIL",
                    area,
                    f"Could not find marker {marker} under {repo_dir}.",
                    "Place the upstream repository folder inside the matching models/<name>/repo/ directory.",
                )
            )
        else:
            checks.append(Check("OK", area, f"Found {repo_root}."))
    return checks


def check_assets(model: str) -> list[Check]:
    checks: list[Check] = []

    if model in {"all", "altfreezing"}:
        alt_checkpoints = [
            path
            for path in (ROOT / "models" / "altfreezing" / "checkpoints").glob("*.pth")
            if path.stat().st_size > 1024 * 1024
        ]
        if alt_checkpoints:
            checks.append(Check("OK", "AltFreezing checkpoint", f"Found {alt_checkpoints[0].name}."))
        else:
            checks.append(
                Check(
                    "FAIL",
                    "AltFreezing checkpoint",
                    "No usable .pth checkpoint found in models/altfreezing/checkpoints.",
                    "Put model.pth or another AltFreezing checkpoint in that folder.",
                )
            )

        aux_dir = ROOT / "auxillary"
        aux_files = [
            aux_dir / "mobilenet0.25_Final.pth",
            aux_dir / "mobilenet_224_model_best_gdconv_external.pth",
        ]
        missing_aux = [path.name for path in aux_files if not path.is_file()]
        if missing_aux:
            checks.append(
                Check(
                    "FAIL",
                    "AltFreezing auxiliary weights",
                    "Missing: " + ", ".join(missing_aux),
                    "Restore the face detection auxiliary weights in auxillary/.",
                )
            )
        else:
            checks.append(Check("OK", "AltFreezing auxiliary weights", "Both auxiliary weights are present."))

    if model in {"all", "effort"}:
        effort_root = ROOT / "models" / "effort"
        effort_checkpoint = effort_root / "weights" / EFFORT_CHECKPOINT
        if effort_checkpoint.is_file() and effort_checkpoint.stat().st_size > 1024 * 1024:
            checks.append(Check("OK", "EFFORT checkpoint", f"Found {EFFORT_CHECKPOINT}."))
        else:
            checks.append(
                Check(
                    "FAIL",
                    "EFFORT checkpoint",
                    f"No usable {EFFORT_CHECKPOINT} was found.",
                    "Put it in models/effort/weights/.",
                )
            )

        clip_dir = effort_root / "pretrained" / EFFORT_CLIP_DIR
        clip_config = clip_dir / "config.json"
        clip_weights = clip_dir / "model.safetensors"
        if clip_config.is_file():
            checks.append(Check("OK", "EFFORT CLIP config", f"Found {clip_config.name}."))
        else:
            checks.append(
                Check(
                    "FAIL",
                    "EFFORT CLIP config",
                    f"Missing {clip_config}.",
                    "Put the openai/clip-vit-large-patch14 config in models/effort/pretrained/clip-vit-large-patch14/.",
                )
            )
        if clip_weights.is_file() and clip_weights.stat().st_size > 1024 * 1024:
            checks.append(Check("OK", "EFFORT CLIP weights", f"Found {clip_weights.name}."))
        else:
            checks.append(
                Check(
                    "WARN",
                    "EFFORT CLIP weights",
                    "model.safetensors was not found. The local runner can load the full EFFORT checkpoint, but the upstream repo expects the CLIP file.",
                    "Put model.safetensors in models/effort/pretrained/clip-vit-large-patch14/ if you want the full local upstream setup.",
                )
            )

    if model in {"all", "f3net"}:
        f3net_weights = ROOT / "models" / "f3net" / "weights"
        if (f3net_weights / F3NET_BACKBONE).is_file():
            checks.append(Check("OK", "F3Net Xception backbone", f"Found {F3NET_BACKBONE}."))
        else:
            checks.append(
                Check(
                    "WARN",
                    "F3Net Xception backbone",
                    f"Missing {F3NET_BACKBONE}.",
                    "Put the Xception backbone in models/f3net/weights/ if you use upstream-style checkpoints.",
                )
            )

        detector_candidates = [
            path
            for path in f3net_weights.glob("*.pth")
            if path.name != F3NET_BACKBONE and path.stat().st_size > 1024 * 1024
        ]
        if detector_candidates:
            checks.append(Check("OK", "F3Net detector checkpoint", f"Found {detector_candidates[0].name}."))
        else:
            checks.append(
                Check(
                    "FAIL",
                    "F3Net detector checkpoint",
                    "No trained F3Net detector checkpoint was found.",
                    "Add a trained F3Net detector checkpoint to models/f3net/weights/.",
                )
            )

    if model in {"all", "recce"}:
        recce_checkpoint = ROOT / "models" / "recce" / "weights" / RECCE_CHECKPOINT
        if recce_checkpoint.is_file() and recce_checkpoint.stat().st_size > 1024 * 1024:
            checks.append(Check("OK", "RECCE checkpoint", f"Found {RECCE_CHECKPOINT}."))
        else:
            checks.append(
                Check(
                    "FAIL",
                    "RECCE checkpoint",
                    f"No usable {RECCE_CHECKPOINT} was found.",
                    "Put it in models/recce/weights/.",
                )
            )

    if model in {"all", "ucf"}:
        ucf_checkpoint = ROOT / "models" / "ucf" / "ucf_best.pth"
        if ucf_checkpoint.is_file() and ucf_checkpoint.stat().st_size > 1024 * 1024:
            checks.append(Check("OK", "UCF detector checkpoint", f"Found {ucf_checkpoint.name}."))
        else:
            checks.append(
                Check(
                    "FAIL",
                    "UCF detector checkpoint",
                    "No usable UCF checkpoint was found.",
                    "Put ucf_best.pth in models/ucf/ or pass --ucf-checkpoint to score_image.py.",
                )
            )

    if model in {"all", "selfblendedimages"}:
        sbi_weights = ROOT / "models" / "selfblendedimages" / "weights"
        if selfblended_weight_available(sbi_weights):
            checks.append(Check("OK", "SelfBlendedImages weights", "Found a usable checkpoint archive or extracted checkpoint."))
        else:
            checks.append(
                Check(
                    "FAIL",
                    "SelfBlendedImages weights",
                    "No usable FFc23 checkpoint was found.",
                    "Put a .tar checkpoint in models/selfblendedimages/weights/ or restore FFc23_extracted/.",
                )
            )

    sample_images = [
        path
        for path in (ROOT / "data" / "samples").glob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]
    sample_videos = [
        path
        for path in (ROOT / "data" / "samples").glob("*")
        if path.suffix.lower() in {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
    ]
    if sample_images:
        checks.append(Check("OK", "Sample images", f"Found {len(sample_images)} sample image(s)."))
    else:
        checks.append(
            Check(
                "WARN",
                "Sample images",
                "No sample images found in data/samples.",
                "Add at least one .jpg or .png before running the example commands.",
            )
        )
    if sample_videos:
        checks.append(Check("OK", "Sample videos", f"Found {len(sample_videos)} sample video(s)."))
    else:
        checks.append(
            Check(
                "WARN",
                "Sample videos",
                "No sample videos found in data/samples.",
                "Add at least one .mp4 before running the video example commands.",
            )
        )

    return checks


def print_checks(checks: list[Check]) -> None:
    status_width = max(len(check.status) for check in checks + [Check("STATUS", "", "")])
    area_width = max(len(check.area) for check in checks + [Check("", "AREA", "")])
    print("Setup check")
    print(f"Workspace: {ROOT}")
    print()
    print(f"{'STATUS'.ljust(status_width)}  {'AREA'.ljust(area_width)}  DETAIL")
    print(f"{'-' * status_width}  {'-' * area_width}  {'-' * 60}")
    for check in checks:
        detail = check.detail if not check.fix else f"{check.detail} Fix: {check.fix}"
        print(f"{check.status.ljust(status_width)}  {check.area.ljust(area_width)}  {detail}")


def synthetic_smoke_image() -> Path:
    sample = Path(tempfile.gettempdir()) / "model_tester_smoke.png"
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (256, 256), (112, 128, 144))
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 24, 232, 232), outline=(224, 224, 224), width=4)
    draw.line((24, 232, 232, 24), fill=(64, 96, 160), width=5)
    image.save(sample)
    return sample


def first_sample_image() -> Path | None:
    for suffix in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        sample = next((ROOT / "data" / "samples").glob(suffix), None)
        if sample is not None:
            return sample
    return None


def first_sample_video() -> Path | None:
    for suffix in ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm", "*.m4v", "*.mpeg", "*.mpg"):
        sample = next((ROOT / "data" / "samples").glob(suffix), None)
        if sample is not None:
            return sample
    return None


def run_smoke(model: str) -> int:
    if model == "altfreezing":
        sample = first_sample_video()
        if sample is None:
            print(
                "AltFreezing smoke tests require a sample video in data/samples.",
                file=sys.stderr,
            )
            return 1
    else:
        sample = first_sample_image()
        if sample is None:
            sample = synthetic_smoke_image()

    command = [sys.executable, str(ROOT / "score_image.py"), str(sample), "--model", model]
    print()
    print("Smoke test")
    print(" ".join(str(part) for part in command))
    sys.stdout.flush()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.returncode


def main() -> int:
    args = parse_args()
    checks = []
    checks.extend(check_python())
    checks.extend(check_imports(args.model))
    checks.extend(check_torch(args.model))
    checks.extend(check_repos(args.model))
    checks.extend(check_assets(args.model))
    print_checks(checks)
    sys.stdout.flush()

    failed = [check for check in checks if check.status == "FAIL"]
    if failed:
        return 1

    if args.smoke:
        return run_smoke(args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
