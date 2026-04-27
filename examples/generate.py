#!/usr/bin/env python3
"""Generate M-100 INSPECT pulmonary embolism multimodal dataset.

Usage:
    python examples/generate.py --num-samples 3
    python examples/generate.py --num-samples 300 --output data/questions
    python examples/generate.py            # uses default cap (300)

Bootstrap on EC2 calls this with --output /tmp/out (no --num-samples), so
the default cap = 300 controls the EC2-side run length.

Note: GitHub repo name still contains "vindrcxr" (do not rename — the website
maps by repo prefix). The actual produced ``domain`` is ``inspect_pe_detection``,
which appears as the on-disk task folder name and in the metadata.
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

from src.pipeline import TaskPipeline, TaskConfig


def main():
    parser = argparse.ArgumentParser(
        description="Generate M-100 INSPECT PE multimodal dataset"
    )
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Cap on number of CT volumes to process (overrides --max-samples).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--generator", type=str, default=None)
    parser.add_argument("--output", type=str, default="data/questions")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=28)
    parser.add_argument("--frame-height", type=int, default=512)
    parser.add_argument("--ehr-panel-width", type=int, default=360)
    parser.add_argument("--window-level", type=int, default=100)
    parser.add_argument("--window-width", type=int, default=700)
    parser.add_argument("--s3-bucket", type=str, default="med-vr-datasets")
    parser.add_argument(
        "--s3-prefix", type=str,
        default="M-100/inspect_pe/inspectamultimodaldatasetforpulmonaryembolismdiagnosisandprog-3/full/",
    )
    parser.add_argument("--max-samples", type=int, default=300,
                        help="EC2-default cap (300). Override with --num-samples for local smoke tests.")
    args = parser.parse_args()

    print("Generating M-100 INSPECT pulmonary embolism multimodal dataset...", flush=True)

    kwargs = dict(
        num_samples=args.num_samples,
        output_dir=Path(args.output),
        seed=args.seed,
        start_index=args.start_index,
        fps=args.fps,
        num_frames=args.num_frames,
        frame_height=args.frame_height,
        ehr_panel_width=args.ehr_panel_width,
        window_level=args.window_level,
        window_width=args.window_width,
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
