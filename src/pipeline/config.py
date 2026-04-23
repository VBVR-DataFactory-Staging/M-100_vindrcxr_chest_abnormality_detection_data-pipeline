"""Pipeline configuration for M-100 VinDr-CXR chest abnormality detection."""
from pathlib import Path
from pydantic import Field
from core.pipeline import PipelineConfig


class TaskConfig(PipelineConfig):
    """Dataset + rendering settings for VinDr-CXR chest X-ray bbox detection."""

    domain: str = Field(default="vindrcxr_chest_abnormality_detection")
    generator: str = Field(default="")

    # Raw data location (populated by onramp pull-to-s3 → S3, cached locally by downloader)
    s3_bucket: str = Field(default="med-vr-datasets")
    s3_prefix: str = Field(default="M-100/VinDrCXR/")
    raw_dir: Path = Field(default=Path("raw"))

    # Rendering
    fps: int = Field(default=8, ge=1)
    num_frames: int = Field(default=20, ge=2)
    width: int = Field(default=640, ge=64)
    height: int = Field(default=640, ge=64)

    # Cap at 2000 samples by default per the M-100 spec.
    max_samples: int = Field(default=2000, ge=1)
