#!/usr/bin/env python3
"""Generate M-100 VinDr-CXR chest abnormality detection dataset.

Usage:
    python examples/generate.py --num-samples 3        # local smoke test
    python examples/generate.py --num-samples 800      # full EC2 run
    python examples/generate.py                        # default cap = 800

Bootstrap on EC2 calls this without --num-samples, so the argparse default
(800) controls the EC2-side run length. Default is intentionally NOT 2 — the
bootstrap 20-cap bug triggers on default<=2 (see HARNESS.md Part 11 / pitfalls).
"""
import os
os.environ.setdefault("PYTHONUNBUFFERED", "1")
import sys

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import TaskConfig, TaskPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate M-100 VinDr-CXR chest abnormality detection dataset",
    )
    parser.add_argument(
        "--num-samples", type=int, default=800,
        help="Cap on number of CXR images to render (default: 800).",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--generator", type=str, default=None)
    parser.add_argument("--output", type=str, default="data/questions")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--num-frames", type=int, default=60)
    parser.add_argument("--frame-size", type=int, default=512)
    parser.add_argument("--bbox-alpha", type=float, default=0.35)
    parser.add_argument("--s3-bucket", type=str, default="med-vr-datasets")
    parser.add_argument("--s3-prefix", type=str, default="M-100/VinDrCXR/")
    parser.add_argument(
        "--max-samples", type=int, default=800,
        help="Hard cap (mirrors --num-samples; kept for backward compat).",
    )
    args = parser.parse_args()

    print("Generating M-100 VinDr-CXR chest abnormality detection dataset...",
          flush=True)

    kwargs = dict(
        num_samples=args.num_samples,
        output_dir=Path(args.output),
        fps=args.fps,
        num_frames=args.num_frames,
        frame_size=args.frame_size,
        bbox_alpha=args.bbox_alpha,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
        max_samples=args.max_samples,
    )
    if args.generator is not None:
        kwargs["generator"] = args.generator

    config = TaskConfig(**kwargs)

    pipeline = TaskPipeline(config)
    samples = pipeline.run()

    layout = (
        f"{config.output_dir}/{config.generator}/{config.domain}_task/"
        if config.generator
        else f"{config.output_dir}/{config.domain}_task/"
    )
    print(f"Wrote {len(samples)} samples to {layout}", flush=True)


if __name__ == "__main__":
    main()
