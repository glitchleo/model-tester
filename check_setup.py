from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO_VENV = ROOT / ".venv"
F3NET_BACKBONE = "xception-b5690688.pth"


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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the local model-tester setup.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="After setup checks pass, run score_image.py on data/samples/1.png.",
    )
    parser.add_argument(
        "--model",
        choices=["all", "altfreezing", "f3net", "selfblendedimages"],
        default="all",
        help="Model to use for --smoke. Defaults to all.",
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


def check_imports() -> list[Check]:
    checks: list[Check] = []
    for group_name, modules in IMPORT_GROUPS.items():
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


def check_torch() -> list[Check]:
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
    else:
        checks.append(
            Check(
                "WARN",
                "CUDA",
                "CUDA is not available. SelfBlendedImages can run on CPU, but AltFreezing needs CUDA here.",
                "Install a CUDA-enabled PyTorch build and NVIDIA driver if you need AltFreezing.",
            )
        )
    return checks


def check_repos() -> list[Check]:
    markers = [
        ("AltFreezing repo", ROOT / "models" / "altfreezing" / "repo", "demo.py"),
        ("F3Net repo", ROOT / "models" / "f3net" / "repo", "models.py"),
        (
            "SelfBlendedImages repo",
            ROOT / "models" / "selfblendedimages" / "repo",
            "src/inference/model.py",
        ),
    ]

    checks: list[Check] = []
    for area, repo_dir, marker in markers:
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


def check_assets() -> list[Check]:
    checks: list[Check] = []

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

    f3net_weights = ROOT / "models" / "f3net" / "weights"
    if (f3net_weights / F3NET_BACKBONE).is_file():
        checks.append(Check("OK", "F3Net backbone", f"Found {F3NET_BACKBONE}."))
    else:
        checks.append(
            Check(
                "FAIL",
                "F3Net backbone",
                f"Missing {F3NET_BACKBONE}.",
                "Put the Xception backbone in models/f3net/weights/.",
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
                "WARN",
                "F3Net detector checkpoint",
                "Only the Xception backbone is present; F3Net scoring will be reported as unavailable.",
                "Add a trained F3Net detector checkpoint to models/f3net/weights/ when you want F3Net results.",
            )
        )

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


def run_smoke(model: str) -> int:
    sample = ROOT / "data" / "samples" / "1.png"
    if not sample.is_file():
        print()
        print(f"Smoke test skipped: sample image not found at {sample}")
        return 1

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
    checks.extend(check_imports())
    checks.extend(check_torch())
    checks.extend(check_repos())
    checks.extend(check_assets())
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
