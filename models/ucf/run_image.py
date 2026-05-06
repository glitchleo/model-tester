from __future__ import annotations

import argparse
import sys
from pathlib import Path


MODEL_ROOT = Path(__file__).resolve().parent
ROOT = MODEL_ROOT.parents[1]
DEFAULT_CHECKPOINT = MODEL_ROOT / "ucf_best.pth"
UCF_IMAGE_SIZE = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UCF on one image.")
    parser.add_argument("--image", type=Path, required=True, help="Path to the input image.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Optional UCF detector checkpoint.",
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


def normalize_state_dict(raw_state_dict: dict) -> dict:
    normalized = {}
    for key, value in raw_state_dict.items():
        clean_key = key
        for prefix in ("module.", "model.", "network."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        normalized[clean_key] = value
    return normalized


def build_ucf_model(torch_module, nn_module, xception_module, state_dict: dict):
    class UCFXception(xception_module.Xception):
        def __init__(self) -> None:
            super().__init__(num_classes=2)
            if hasattr(self, "fc"):
                delattr(self, "fc")
            self.last_linear = nn_module.Linear(2048, 2)
            self.adjust_channel = nn_module.Sequential(
                nn_module.Conv2d(2048, 512, kernel_size=1, stride=1),
                nn_module.BatchNorm2d(512),
                nn_module.ReLU(inplace=False),
            )

        def features(self, input_tensor):
            features = super().features(input_tensor)
            return self.adjust_channel(features)

    class Conv2d1x1(nn_module.Module):
        def __init__(self, in_f: int, hidden_dim: int, out_f: int) -> None:
            super().__init__()
            self.conv2d = nn_module.Sequential(
                nn_module.Conv2d(in_f, hidden_dim, 1, 1),
                nn_module.LeakyReLU(inplace=True),
                nn_module.Conv2d(hidden_dim, hidden_dim, 1, 1),
                nn_module.LeakyReLU(inplace=True),
                nn_module.Conv2d(hidden_dim, out_f, 1, 1),
            )

        def forward(self, tensor):
            return self.conv2d(tensor)

    class Head(nn_module.Module):
        def __init__(self, in_f: int, hidden_dim: int, out_f: int) -> None:
            super().__init__()
            self.do = nn_module.Dropout(0.2)
            self.pool = nn_module.AdaptiveAvgPool2d(1)
            self.mlp = nn_module.Sequential(
                nn_module.Linear(in_f, hidden_dim),
                nn_module.LeakyReLU(inplace=True),
                nn_module.Linear(hidden_dim, out_f),
            )

        def forward(self, tensor):
            batch_size = tensor.size(0)
            feat = self.pool(tensor).view(batch_size, -1)
            logits = self.mlp(feat)
            logits = self.do(logits)
            return logits, feat

    class UCFInferenceModel(nn_module.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder_f = UCFXception()
            self.block_sha = Conv2d1x1(in_f=512, hidden_dim=256, out_f=256)
            self.head_sha = Head(in_f=256, hidden_dim=512, out_f=2)

        def forward(self, tensor):
            features = self.encoder_f.features(tensor)
            shared_features = self.block_sha(features)
            logits, _ = self.head_sha(shared_features)
            return logits

    model = UCFInferenceModel()
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        fail(f"UCF checkpoint is missing inference keys: {incompatible.missing_keys}")

    allowed_unexpected = ("encoder_c.", "con_gan.", "head_spe.", "block_spe.")
    unexpected = [
        key for key in incompatible.unexpected_keys
        if not key.startswith(allowed_unexpected)
    ]
    if unexpected:
        fail(f"UCF checkpoint has unexpected keys for this wrapper: {unexpected[:20]}")
    return model


def fakeness_from_logits(torch_module, logits) -> float:
    return float(torch_module.softmax(logits.reshape(logits.size(0), -1), dim=1)[:, 1].mean().item())


def load_model(torch_module, nn_module, xception_module, checkpoint_path: Path, device):
    checkpoint = torch_load_compat(torch_module, checkpoint_path, device)
    state_dict = normalize_state_dict(extract_state_dict(checkpoint))
    model = build_ucf_model(torch_module, nn_module, xception_module, state_dict)
    model = model.to(device)
    model.eval()
    return model


def main() -> int:
    args = parse_args()
    image_path = args.image.resolve()
    if not image_path.is_file():
        fail(f"Input image not found: {image_path}")

    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        fail(f"UCF checkpoint not found: {checkpoint_path}", unavailable=True)

    repo_root = find_repo_root(ROOT / "models" / "f3net" / "repo", "xception.py")
    sys.path.insert(0, str(repo_root))

    try:
        import torch
        from torch import nn
        from PIL import Image
        from torchvision import transforms
        import xception as xception_module
    except Exception as exc:
        fail(f"Missing UCF runtime dependency: {exc}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_model(torch, nn, xception_module, checkpoint_path, device)

    transform = transforms.Compose(
        [
            transforms.Resize((UCF_IMAGE_SIZE, UCF_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Lambda(lambda tensor: tensor * 2.0 - 1.0),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        score = fakeness_from_logits(torch, logits)

    print(f"UCF fakeness: {score:.4f}")
    print(f"SCORE:{score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
