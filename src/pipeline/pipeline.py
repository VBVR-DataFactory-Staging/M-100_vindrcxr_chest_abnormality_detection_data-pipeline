"""Pipeline for M-100 INSPECT pulmonary embolism multimodal diagnosis.

Mirrors M-60 RadFusion (sister PE multimodal dataset). Per-study workflow:

  1. Stream one CTPA NIfTI volume from S3 (delete after load).
  2. Pick an axial sweep through the central thorax (~pulmonary arteries).
  3. Render an EHR/clinical side panel from the impression text + label flags.
  4. Compose CT-slice + side-panel frames (clean + annotated variants).
  5. Write first / last / ground_truth mp4s.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

from core.pipeline import BasePipeline, SampleProcessor, TaskSample

from .config import TaskConfig
from .transforms import (
    colorize_ct_slice,
    compose_frame,
    pe_heuristic_mask,
    pick_sweep_indices,
    render_ehr_panel,
    resize_square,
    write_mp4,
)
from src.download.downloader import fetch_ct_volume, run_download


PROMPT = (
    "In the final CT slice, does the combination of pulmonary artery findings "
    "suggest acute pulmonary embolism (PE)? Consider the full multimodal "
    "context: contrast-enhanced CT axial sweep + the clinical impression "
    "summary in the side panel."
)


TMP_DIR = Path("_tmp")
CT_CACHE_DIR = Path("raw/_ct_cache")


class TaskPipeline(BasePipeline):
    """INSPECT PE multimodal pipeline."""

    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.task_config = config

    # ── 1) Download ──────────────────────────────────────────────────────
    def download(self) -> Iterator[dict]:
        # Honour max_samples set in config (overridable via --num-samples).
        cap = getattr(self.task_config, "max_samples", None)
        ns = getattr(self.task_config, "num_samples", None)
        limit = ns if ns is not None else cap
        n = 0
        for s in run_download(self.task_config):
            yield s
            n += 1
            if limit is not None and n >= limit:
                break

    # ── 2) Process one study ─────────────────────────────────────────────
    def process_sample(self, raw_sample: dict, idx: int) -> Optional[TaskSample]:
        cfg = self.task_config
        impr_id = int(raw_sample.get("impression_id", -1))
        label = int(raw_sample.get("label", 0))
        global_idx = int(getattr(cfg, "start_index", 0)) + idx
        task_id = f"{cfg.domain}_{global_idx:05d}"

        volume = fetch_ct_volume(
            raw_sample["s3_bucket"],
            raw_sample["s3_key"],
            CT_CACHE_DIR,
        )
        if volume is None or volume.ndim != 3:
            return None

        # INSPECT volumes are int16 HU, typically (Z, 512, 512).
        if volume.shape[0] < 8 or volume.shape[1] < 64 or volume.shape[2] < 64:
            return None

        sweep_idx = pick_sweep_indices(volume.shape[0], cfg.num_frames)
        if not sweep_idx:
            return None

        panel_clean = render_ehr_panel(
            raw=raw_sample,
            width=cfg.ehr_panel_width,
            height=cfg.frame_height,
            title="Clinical context",
            reveal=False,
        )
        panel_reveal = render_ehr_panel(
            raw=raw_sample,
            width=cfg.ehr_panel_width,
            height=cfg.frame_height,
            title="Clinical context",
            reveal=True,
        )

        clean_frames: List[np.ndarray] = []
        annotated_frames: List[np.ndarray] = []

        for s_i in sweep_idx:
            hu = volume[s_i].astype(np.int16)

            clean_rgb = colorize_ct_slice(hu, cfg.window_level, cfg.window_width, pe_mask=None)
            clean_rgb = resize_square(clean_rgb, cfg.frame_height)
            clean_frames.append(compose_frame(clean_rgb, panel_clean))

            mask = pe_heuristic_mask(hu) if label == 1 else None
            ann_rgb = colorize_ct_slice(
                hu, cfg.window_level, cfg.window_width,
                pe_mask=mask, alpha=cfg.pe_alpha,
            )
            ann_rgb = resize_square(ann_rgb, cfg.frame_height)
            annotated_frames.append(compose_frame(ann_rgb, panel_reveal))

        if not clean_frames or not annotated_frames:
            return None

        # Free the volume before writing videos (bound peak RAM).
        del volume

        TMP_DIR.mkdir(parents=True, exist_ok=True)
        tmp = TMP_DIR / task_id
        tmp.mkdir(parents=True, exist_ok=True)

        fps = cfg.fps
        first_video = tmp / "first_video.mp4"
        last_video = tmp / "last_video.mp4"
        gt_video = tmp / "ground_truth.mp4"

        write_mp4(clean_frames, first_video, fps)
        write_mp4(annotated_frames, last_video, fps)
        write_mp4(annotated_frames, gt_video, fps)

        first_rgb = clean_frames[len(clean_frames) // 2]
        final_rgb = annotated_frames[-1]

        metadata = {
            "task_id": task_id,
            "source_dataset": "INSPECT",
            "impression_id": impr_id,
            "image_id": raw_sample.get("image_id"),
            "label": label,
            "pe_acute": int(raw_sample.get("pe_acute", 0)),
            "pe_subseg": int(raw_sample.get("pe_subseg", 0)),
            "mortality_12m": raw_sample.get("mortality_12m"),
            "person_id": raw_sample.get("person_id"),
            "num_slices": int(len(sweep_idx)),
            "fps": fps,
            "window_level": cfg.window_level,
            "window_width": cfg.window_width,
        }

        return SampleProcessor.build_sample(
            task_id=task_id,
            domain=cfg.domain,
            first_image=first_rgb,
            prompt=PROMPT,
            final_image=final_rgb,
            first_video=str(first_video),
            last_video=str(last_video),
            ground_truth_video=str(gt_video),
            metadata=metadata,
        )

    # ── 3) Run (with tmp cleanup) ────────────────────────────────────────
    def run(self) -> List[TaskSample]:
        try:
            samples = super().run()
        finally:
            if TMP_DIR.exists():
                shutil.rmtree(TMP_DIR, ignore_errors=True)
            if CT_CACHE_DIR.exists():
                shutil.rmtree(CT_CACHE_DIR, ignore_errors=True)
        return samples
