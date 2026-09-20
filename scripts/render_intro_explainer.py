#!/usr/bin/env python3
"""Render the shareable Manim intro explainer to docs/assets.

    python scripts/render_intro_explainer.py
    python scripts/render_intro_explainer.py --quality l   # fast preview
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import manim as _manim
except ImportError:
    _manim = None

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO / "docs" / "assets" / "intro-explainer.mp4"
SCENE_FILE = REPO / "scripts" / "intro_explainer.py"
MEDIA_DIR = REPO / "media"

QUALITY_DIRS = {
    "l": "480p15",
    "m": "720p30",
    "h": "1080p60",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the shareable intro explainer MP4.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination MP4")
    parser.add_argument(
        "--quality",
        choices=("l", "m", "h"),
        default="m",
        help="Manim quality: l=480p, m=720p (default), h=1080p",
    )
    args = parser.parse_args()

    if _manim is None:
        print("Manim is required. Install with: pip install 'manim>=0.18,<0.20'", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "manim",
        "-q" + args.quality,
        "-o",
        "intro-explainer.mp4",
        "--media_dir",
        str(MEDIA_DIR),
        "--disable_caching",
        str(SCENE_FILE),
        "IntroExplainer",
    ]
    print(" ".join(command))
    subprocess.run(command, check=True, cwd=REPO)

    produced = MEDIA_DIR / "videos" / "intro_explainer" / QUALITY_DIRS[args.quality] / "intro-explainer.mp4"
    if not produced.exists():
        matches = list((MEDIA_DIR / "videos").rglob("intro-explainer.mp4"))
        if not matches:
            raise FileNotFoundError(f"Manim finished but no MP4 was found under {MEDIA_DIR}")
        produced = matches[0]

    shutil.copy2(produced, args.output)
    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
