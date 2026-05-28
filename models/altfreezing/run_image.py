from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report AltFreezing image inputs as unsupported.")
    parser.add_argument("--image", type=Path, required=True, help="Path to the input image.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Accepted for compatibility, but image scoring is not supported.",
    )
    parser.add_argument(
        "--config",
        default="i3d_ori.yaml",
        help="Accepted for compatibility, but image scoring is not supported.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=32,
        help="Accepted for compatibility, but image scoring is not supported.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=8.0,
        help="Accepted for compatibility, but image scoring is not supported.",
    )
    return parser.parse_args()


def fail(message: str, exit_code: int = 1) -> NoReturn:
    print(f"ERROR:{message}", file=sys.stderr)
    raise SystemExit(exit_code)


def main() -> int:
    args = parse_args()
    image_path = args.image.resolve()
    if not image_path.is_file():
        fail(f"Input image not found: {image_path}")

    print("UNAVAILABLE:AltFreezing supports video inputs only.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
