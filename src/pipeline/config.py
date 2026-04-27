"""Pipeline configuration for M-100 VinDr-CXR chest abnormality detection.

VinDr-CXR (Vietnamese chest X-ray) is a public dataset of frontal chest X-rays
with bounding-box annotations for 14 thoracic abnormality classes plus a
"No finding" class. This pipeline renders one VBVR sample per CXR with
annotated bounding boxes (class-coloured) overlaid on the final frame.

Raw layout under ``s3://med-vr-datasets/M-100/VinDrCXR/`` (HF mirror):
    train/<image_id>.png         # 512x512 PNG chest X-rays
    train.csv                    # image_id, class_name, class_id, rad_id, x_min, y_min, x_max, y_max
    test/<image_id>.png          # (no labels)

We use the train split (it has bbox labels). Multiple radiologists can box the
same finding on one image — we union all bboxes per (image, class).
"""
from pathlib import Path

from pydantic import Field

from core.pipeline import PipelineConfig


class TaskConfig(PipelineConfig):
    """VinDr-CXR chest abnormality detection — frame + bbox-overlay rendering."""

    domain: str = Field(default="vindrcxr_chest_abnormality_detection")

    # Empty generator name → flat layout under data/questions/<domain>_task/<task_id>/
    # which matches website / production S3 layout.
    generator: str = Field(default="")

    # Source layout in s3://med-vr-datasets/M-100/VinDrCXR/
    s3_bucket: str = Field(default="med-vr-datasets")
    s3_prefix: str = Field(default="M-100/VinDrCXR/")
    raw_dir: Path = Field(default=Path("raw"))

    # Video: synthesise 60-frame clips (≥60 to satisfy harness minimum) per sample.
    fps: int = Field(default=12)
    num_frames: int = Field(default=60)

    # Frame output size (square, padded if needed).
    frame_size: int = Field(default=512)

    # Bbox overlay opacity for the bbox-reveal video.
    bbox_alpha: float = Field(default=0.35)

    # Default cap for EC2 runs (overridable via --num-samples).
    max_samples: int = Field(default=800, ge=1)
