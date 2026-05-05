from __future__ import annotations

import argparse
import sys
from pathlib import Path

from run_image import (
    BACKBONE_NAME,
    MODEL_ROOT,
    extract_state_dict,
    fail,
    find_repo_root,
    infer_mode,
    normalize_state_dict,
    resolve_detector_checkpoint,
    torch_load_compat,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run F3Net on sampled frames from one video.")
    parser.add_argument("--video", type=Path, required=True, help="Path to the input video.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Optional trained F3Net detector checkpoint.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=32,
        help="How many frames to sample from the video.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference instead of CUDA.",
    )
    return parser.parse_args()


def sample_video_frames(cv2_module, np_module, video_path: Path, requested_count: int) -> list[tuple[int, object]]:
    requested_count = max(requested_count, 1)
    capture = cv2_module.VideoCapture(str(video_path))
    if not capture.isOpened():
        fail(f"Could not open video: {video_path}")

    frames = []
    total_frames = int(capture.get(cv2_module.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames > 0:
        sample_count = min(requested_count, total_frames)
        frame_indices = sorted(set(int(index) for index in np_module.linspace(0, total_frames - 1, sample_count)))
        for frame_index in frame_indices:
            capture.set(cv2_module.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append((frame_index, frame))
    else:
        frame_index = 0
        while len(frames) < requested_count:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frames.append((frame_index, frame))
            frame_index += 1

    capture.release()
    return frames


def main() -> int:
    args = parse_args()
    video_path = args.video.resolve()
    if not video_path.is_file():
        fail(f"Input video not found: {video_path}")

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
        import cv2
        import numpy as np
        import torch
        from PIL import Image
        from torchvision import transforms
        import models as f3net_models
    except Exception as exc:
        fail(f"Missing F3Net runtime dependency: {exc}")

    sampled_frames = sample_video_frames(cv2, np, video_path, args.frames)
    if not sampled_frames:
        fail(f"F3Net could not decode any frames from {video_path}.")

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
    batch = torch.stack(
        [
            transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            for _, frame in sampled_frames
        ]
    ).to(device)

    with torch.no_grad():
        _, logits = model(batch)
        scores = torch.sigmoid(logits.reshape(-1)).detach().cpu().numpy().tolist()

    if not scores:
        fail(f"F3Net did not produce any predictions for {video_path}.")

    score = float(np.mean(scores))
    print(f"F3Net ({mode}) video fakeness: {score:.4f}")
    print(f"Frames scored: {len(scores)}")
    print(f"SCORE:{score:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
