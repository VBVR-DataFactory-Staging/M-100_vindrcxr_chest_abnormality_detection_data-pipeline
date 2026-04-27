"""Pipeline configuration for M-100 INSPECT pulmonary embolism multimodal dataset.

INSPECT (Stanford AIMI) is a multimodal PE dataset:
    - CTPA volumes:     s3://med-vr-datasets/M-100/inspect_pe/.../full/CTPA/PE<id>.nii.gz
    - EHR/labels TSVs:  s3://.../full/labels_20250611.tsv
    - Impressions TSV:  s3://.../full/impressions_20250611.tsv
    - Mapping TSV:      s3://.../full/study_mapping_20250611.tsv  (impression_id ↔ image_id)

We mirror M-60 RadFusion (the sister PE multimodal dataset) for video construction.

NOTE: GitHub repo name remains ``M-100_vindrcxr_chest_abnormality_detection_data-pipeline``
(the website maps by repo prefix, so we do not rename), but the actual ``domain``
(used for the on-disk task folder name) is the real INSPECT-style name.
"""
from pathlib import Path

from pydantic import Field

from core.pipeline import PipelineConfig


class TaskConfig(PipelineConfig):
    """Dataset + rendering settings for INSPECT PE multimodal pipeline."""

    domain: str = Field(default="inspect_pe_detection")

    # Empty generator name → flat layout under data/questions/<domain>_task/<task_id>/
    # which matches website / production S3 layout (avoids 7e/7m double-nesting).
    generator: str = Field(default="")

    # Source layout in s3://med-vr-datasets/M-100/
    s3_bucket: str = Field(default="med-vr-datasets")
    s3_prefix: str = Field(
        default="M-100/inspect_pe/inspectamultimodaldatasetforpulmonaryembolismdiagnosisandprog-3/full/"
    )
    raw_dir: Path = Field(default=Path("raw"))

    # CT windowing — mediastinal window highlights PE in pulmonary arteries.
    window_level: int = Field(default=100)
    window_width: int = Field(default=700)

    # Video: axial sweep through the pulmonary-artery region.
    fps: int = Field(default=8)
    num_frames: int = Field(default=28)

    # Frame output size.
    frame_height: int = Field(default=512)
    ehr_panel_width: int = Field(default=360)

    # PE highlight (red, 40% opacity) on annotated video / final frame.
    pe_alpha: float = Field(default=0.40)

    # Default cap (overridden by --num-samples on the CLI).
    max_samples: int = Field(default=300, ge=1)
