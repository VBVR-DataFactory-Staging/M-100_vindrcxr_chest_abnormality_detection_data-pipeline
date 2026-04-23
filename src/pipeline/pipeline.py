"""M-100 VinDr-CXR chest abnormality detection pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from core.pipeline import BasePipeline, SampleProcessor, TaskSample
from core.download import run_download

from .config import TaskConfig
from .transforms import build_sample_videos, read_yolo_labels, PROMPT_CLASSES


PROMPT = (
    "Identify all abnormalities in the final chest X-ray. 14 classes each with "
    "a distinct color bounding box: "
    + ", ".join(PROMPT_CLASSES)
    + ". Answer which classes are present."
)


class TaskPipeline(BasePipeline):
    """VinDr-CXR bbox detection pipeline: CXR + YOLO labels -> first/last/gt videos."""

    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.task_config = config

    def download(self) -> Iterator[dict]:
        yield from run_download(self.task_config)

    def process_sample(self, raw_sample: dict, idx: int) -> Optional[TaskSample]:
        cfg = self.task_config
        global_idx = int(getattr(cfg, "start_index", 0)) + idx
        task_id = f"{cfg.domain}_{global_idx:08d}"

        image_path = raw_sample.get("image_path")
        label_path = raw_sample.get("label_path")
        if not image_path:
            return None

        labels = read_yolo_labels(label_path)

        tmp_out = Path("_tmp") / task_id
        try:
            media = build_sample_videos(
                image_path=image_path,
                labels=labels,
                task_id=task_id,
                out_dir=tmp_out,
                width=cfg.width,
                height=cfg.height,
                fps=cfg.fps,
                num_frames=cfg.num_frames,
            )
        except FileNotFoundError as exc:
            print(f"  [skip] {task_id}: {exc}", flush=True)
            return None

        return SampleProcessor.build_sample(
            task_id=task_id,
            domain=cfg.domain,
            first_image=media["first_frame"],
            prompt=PROMPT,
            final_image=media["final_frame"],
            first_video=media["first_video"],
            last_video=media["last_video"],
            ground_truth_video=media["ground_truth_video"],
            metadata={
                "dataset": raw_sample.get("dataset", "VinDr-CXR"),
                "source_id": raw_sample.get("source_id"),
                "split": raw_sample.get("split"),
                "num_boxes": media["num_boxes"],
                "present_classes": media["present_classes"],
                "boxes": media["boxes"],
                "video_spec": {
                    "fps": cfg.fps,
                    "num_frames": cfg.num_frames,
                    "width": cfg.width,
                    "height": cfg.height,
                },
            },
        )
