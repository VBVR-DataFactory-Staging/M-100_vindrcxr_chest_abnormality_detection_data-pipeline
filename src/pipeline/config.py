"""Pipeline configuration for M-043 (ddr_lesion_bbox_detection)."""
from pathlib import Path
from pydantic import Field
from core.pipeline import PipelineConfig


class TaskConfig(PipelineConfig):
    """Configuration for M-043 pipeline.

    Inherited from PipelineConfig:
        num_samples: Optional[int]  # Max samples (None = all)
        domain: str
        output_dir: Path
        split: str
    """

    domain: str = Field(default="vindrcxr_chest_abnormality_detection")

    s3_bucket: str = Field(
        default="med-vr-datasets",
        description="S3 bucket containing the raw M-043 data",
    )
    s3_prefix: str = Field(
        default="M-100_VinDr-CXR/raw/",
        description="S3 key prefix for the dataset raw data",
    )
    fps: int = Field(
        default=3,
        description="Frames per second for the generated videos",
    )
    raw_dir: Path = Field(
        default=Path("raw"),
        description="Local directory for downloaded raw data",
    )
    task_prompt: str = Field(
        default="This chest X-ray. Detect and draw bounding boxes around thoracic abnormalities from VinDr-CXR's 22 disease classes (Aortic enlargement, Cardiomegaly, Pleural effusion, etc.).",
        description="The task instruction shown to the reasoning model.",
    )
