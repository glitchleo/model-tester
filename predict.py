from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.prediction import analyze_media, analysis_has_score, cli_payload


def parse_args(default_model: str = "available") -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one standardized deepfake model prediction.",
    )
    parser.add_argument(
        "--input",
        dest="input_path",
        type=Path,
        required=True,
        help="Path to an input image or video.",
    )
    parser.add_argument(
        "--model",
        default=default_model,
        help="Model to run: f3net, sbi, recce, ucf, effort, altfreezing, available, or all.",
    )
    parser.add_argument(
        "--video-frames",
        type=int,
        help="Frame count for frame-based video models.",
    )
    parser.add_argument(
        "--video-preset",
        choices=["quick", "balanced", "thorough"],
        default="quick",
        help="Frame budget preset used when --video-frames is not set.",
    )
    parser.add_argument(
        "--include-details",
        action="store_true",
        help="Include frame metadata/details when a video runner provides them.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Always print the combined analysis object, even when one model runs.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON.",
    )
    return parser.parse_args()


def main(default_model: str = "available") -> int:
    args = parse_args(default_model)
    try:
        analysis = analyze_media(
            args.input_path,
            model=args.model,
            video_frames=args.video_frames,
            video_preset=args.video_preset,
            include_details=args.include_details,
        )
    except FileNotFoundError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 2
    except ValueError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 2

    payload = cli_payload(analysis, prefer_single_result=not args.summary)
    print(json.dumps(payload, indent=2 if args.pretty else None))
    return 0 if analysis_has_score(analysis) else 1


if __name__ == "__main__":
    raise SystemExit(main())

