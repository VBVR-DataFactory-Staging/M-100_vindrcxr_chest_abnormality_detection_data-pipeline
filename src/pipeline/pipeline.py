"""Pipeline for M-100 VinDr-CXR chest abnormality detection.

Per-sample workflow (one frontal CXR -> one VBVR sample):

  1. Sync ``s3://med-vr-datasets/M-100/VinDrCXR/`` -> ``raw/`` once via
     ``aws s3 sync`` (subprocess, NOT public-HTTP — bucket is private).
  2. Parse ``train.csv`` once and group bboxes per image_id, dropping rows
     with class_name == "No finding" (their boxes are bookkeeping).
  3. For each image_id (sorted, deterministic) up to ``num_samples``:
       a. Load PNG, letterbox to 512x512, scale bboxes accordingly.
       b. Render first_frame.png (raw CXR) and final_frame.png (CXR + bboxes).
       c. Encode 3 mp4s, each >=60 frames at 12 fps:
            - first_video.mp4  : zoom-in fade-in on the raw CXR
            - last_video.mp4   : bbox reveal (boxes appear one by one)
            - ground_truth.mp4 : full walkthrough (clean -> all -> per-box)
       d. Emit metadata listing classes + bbox counts.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from core.pipeline import BasePipeline, SampleProcessor, TaskSample

from .config import TaskConfig
from .transforms import (
    CLASS_COLORS,
    build_bbox_reveal_frames,
    build_walkthrough_frames,
    build_zoom_fadein_frames,
    draw_bbox_overlay,
    make_video,
    resize_pad_square,
    scale_bbox,
)

# Real-time stdout for EC2 log tailing — bootstrap.sh greps these prints.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:  # pragma: no cover
    pass


PROMPT = (
    "This is a frontal chest X-ray (PA / AP view). Identify and localize the "
    "thoracic abnormalities present in the image. The 14 candidate classes are: "
    "Aortic enlargement, Atelectasis, Calcification, Cardiomegaly, "
    "Consolidation, ILD, Infiltration, Lung Opacity, Nodule/Mass, "
    "Other lesion, Pleural effusion, Pleural thickening, Pneumothorax, "
    "Pulmonary fibrosis. If no abnormality is present, answer 'No finding'. "
    "For each finding present, give its class name and an approximate bounding "
    "box on the image. The final frame shows the radiologist-consensus boxes; "
    "your task is to reproduce that annotation from the raw view."
)


TMP_DIR = Path("_tmp")
NO_FINDING_LABEL = "No finding"


class TaskPipeline(BasePipeline):
    """VinDr-CXR detection pipeline."""

    def __init__(self, config: Optional[TaskConfig] = None):
        super().__init__(config or TaskConfig())
        self.task_config: TaskConfig = self.config  # type: ignore[assignment]
        self.raw_dir = Path(self.task_config.raw_dir)

    # ── 1) Raw sync ──────────────────────────────────────────────────────

    def _ensure_raw(self) -> None:
        """Ensure raw data is on local disk via private-bucket ``aws s3 sync``.

        Public HTTP access on med-vr-datasets gives 403/400, so we MUST use
        the AWS CLI (which uses the EC2 instance role). Skips if raw_dir is
        non-empty (already synced on a previous run).
        """
        if self.raw_dir.exists() and any(self.raw_dir.iterdir()):
            print(f"[_ensure_raw] {self.raw_dir} already populated, skipping sync",
                  flush=True)
            return
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        s3_uri = (
            f"s3://{self.task_config.s3_bucket}/"
            f"{self.task_config.s3_prefix.rstrip('/')}/"
        )
        print(f"[_ensure_raw] aws s3 sync {s3_uri} {self.raw_dir}/", flush=True)
        try:
            subprocess.run(
                ["aws", "s3", "sync", s3_uri, str(self.raw_dir) + "/",
                 "--no-progress"],
                check=True,
            )
            print("[_ensure_raw] sync done", flush=True)
        except subprocess.CalledProcessError as e:
            print(f"[FATAL] aws s3 sync failed: {e}", flush=True)
            raise

    # ── 2) Annotation index ──────────────────────────────────────────────

    def _find_annotation_csv(self) -> Path:
        """Locate the bbox CSV inside ``raw/`` (HF mirrors vary in subdir name).

        Tries train.csv, train_meta.csv, then any *.csv found that has the
        expected columns.
        """
        candidates = [
            self.raw_dir / "train.csv",
            self.raw_dir / "train_meta.csv",
            self.raw_dir / "annotations.csv",
        ]
        for c in candidates:
            if c.exists():
                return c
        for path in self.raw_dir.rglob("*.csv"):
            try:
                with open(path, newline="", encoding="utf-8") as f:
                    header = next(csv.reader(f))
                if "class_name" in header and "image_id" in header:
                    return path
            except (StopIteration, OSError):
                continue
        raise FileNotFoundError(
            f"Could not locate VinDr-CXR train CSV under {self.raw_dir}"
        )

    def _find_image(self, image_id: str) -> Optional[Path]:
        """Locate ``<image_id>.png`` (or .jpg) under raw/, scanning train/ first."""
        for ext in ("png", "jpg", "jpeg"):
            p = self.raw_dir / "train" / f"{image_id}.{ext}"
            if p.exists():
                return p
            p = self.raw_dir / f"{image_id}.{ext}"
            if p.exists():
                return p
        # Fallback: walk raw_dir (slow but robust to mirror layouts).
        for ext in ("png", "jpg", "jpeg"):
            for p in self.raw_dir.rglob(f"{image_id}.{ext}"):
                return p
        return None

    def _load_annotations(self) -> Dict[str, List[dict]]:
        """Parse the bbox CSV and group rows by image_id.

        Returns ``{image_id: [{class_name, x_min, y_min, x_max, y_max}, ...]}``.
        Drops rows with class_name == "No finding" (boxes there are sentinel
        zeros). Drops rows with missing or non-numeric coordinates.
        """
        csv_path = self._find_annotation_csv()
        print(f"[annotations] reading {csv_path}", flush=True)
        per_image: Dict[str, List[dict]] = defaultdict(list)
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row.get("image_id")
                cls = row.get("class_name") or row.get("class")
                if not image_id or not cls:
                    continue
                if cls.strip() == NO_FINDING_LABEL:
                    # Mark the image as having no findings via empty list —
                    # we still emit it (negative example).
                    per_image.setdefault(image_id, [])
                    continue
                try:
                    x1 = float(row["x_min"])
                    y1 = float(row["y_min"])
                    x2 = float(row["x_max"])
                    y2 = float(row["y_max"])
                except (KeyError, ValueError, TypeError):
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue
                per_image[image_id].append({
                    "class_name": cls.strip(),
                    "x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2,
                })
        print(f"[annotations] {len(per_image)} unique images indexed",
              flush=True)
        return per_image

    # ── 3) Download iterator ─────────────────────────────────────────────

    def download(self) -> Iterator[dict]:
        """Yield one raw record per CXR image, deterministic order, capped."""
        self._ensure_raw()
        per_image = self._load_annotations()
        cap = self.task_config.num_samples or self.task_config.max_samples
        emitted = 0
        # Stable order — sort by image_id so reruns produce the same first-N.
        for image_id in sorted(per_image.keys()):
            if cap is not None and emitted >= cap:
                break
            img_path = self._find_image(image_id)
            if img_path is None:
                # The HF 512px mirror sometimes lacks images for some IDs —
                # skip them rather than fail the whole run.
                continue
            yield {
                "image_id": image_id,
                "image_path": img_path,
                "boxes": per_image[image_id],
            }
            emitted += 1

    # ── 4) Sample renderer ───────────────────────────────────────────────

    def process_sample(self, raw_sample: dict, idx: int) -> Optional[TaskSample]:
        cfg = self.task_config
        image_id: str = raw_sample["image_id"]
        img_path: Path = raw_sample["image_path"]

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [skip] {image_id}: cv2.imread returned None", flush=True)
            return None
        src_h, src_w = img.shape[:2]
        base = resize_pad_square(img, target=cfg.frame_size)

        boxes_scaled: List[Tuple[int, int, int, int, str]] = []
        for b in raw_sample["boxes"]:
            x1, y1, x2, y2 = scale_bbox(
                (b["x_min"], b["y_min"], b["x_max"], b["y_max"]),
                src_w, src_h, cfg.frame_size,
            )
            x1 = max(0, min(cfg.frame_size - 1, x1))
            y1 = max(0, min(cfg.frame_size - 1, y1))
            x2 = max(0, min(cfg.frame_size - 1, x2))
            y2 = max(0, min(cfg.frame_size - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes_scaled.append((x1, y1, x2, y2, b["class_name"]))

        # Build the three video clips (each >=60 frames).
        zoom_frames = build_zoom_fadein_frames(base, cfg.num_frames)
        reveal_frames = build_bbox_reveal_frames(
            base, boxes_scaled, cfg.num_frames, alpha=cfg.bbox_alpha,
        )
        walk_frames = build_walkthrough_frames(
            base, boxes_scaled, cfg.num_frames, alpha=cfg.bbox_alpha,
        )

        # Persist mp4s to a tmp dir (OutputWriter copies them into the sample dir).
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        sid = f"{cfg.domain}_{idx:05d}"
        tmp = TMP_DIR / sid
        tmp.mkdir(parents=True, exist_ok=True)

        first_video = tmp / "first_video.mp4"
        last_video = tmp / "last_video.mp4"
        gt_video = tmp / "ground_truth.mp4"
        make_video(zoom_frames, first_video, cfg.fps)
        make_video(reveal_frames, last_video, cfg.fps)
        make_video(walk_frames, gt_video, cfg.fps)

        # Static frames: raw CXR + fully-annotated CXR.
        first_frame = base
        final_frame = draw_bbox_overlay(base, boxes_scaled, alpha=cfg.bbox_alpha)

        # Per-class counts for metadata.
        cls_counts: Dict[str, int] = defaultdict(int)
        for *_, cls in boxes_scaled:
            cls_counts[cls] += 1

        metadata = {
            "task_id": sid,
            "source_dataset": "VinDr-CXR",
            "image_id": image_id,
            "src_width": int(src_w),
            "src_height": int(src_h),
            "frame_size": int(cfg.frame_size),
            "num_boxes": len(boxes_scaled),
            "classes_present": sorted(cls_counts.keys()),
            "class_counts": dict(cls_counts),
            "no_finding": len(boxes_scaled) == 0,
            "fps": cfg.fps,
            "frames_per_video": cfg.num_frames,
            "boxes": [
                {"class_name": c, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
                for (x1, y1, x2, y2, c) in boxes_scaled
            ],
            "color_legend": {k: list(v) for k, v in CLASS_COLORS.items()},
        }

        if idx % 25 == 0:
            print(
                f"  [{idx:05d}] image_id={image_id} boxes={len(boxes_scaled)} "
                f"classes={sorted(cls_counts.keys()) or ['No finding']}",
                flush=True,
            )

        return SampleProcessor.build_sample(
            task_id=sid,
            domain=cfg.domain,
            first_image=cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB),
            prompt=PROMPT,
            final_image=cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB),
            first_video=str(first_video),
            last_video=str(last_video),
            ground_truth_video=str(gt_video),
            metadata=metadata,
        )

    # ── 5) Run wrapper (cleans tmp dir afterwards) ───────────────────────

    def run(self) -> List[TaskSample]:
        try:
            samples = super().run()
        finally:
            # Best-effort cleanup so we don't leave intermediate mp4s around.
            import shutil
            if TMP_DIR.exists():
                shutil.rmtree(TMP_DIR, ignore_errors=True)
        # Write a global manifest of class counts to help downstream debugging.
        try:
            out_dir = Path(self.task_config.output_dir) / f"{self.task_config.domain}_task"
            agg: Dict[str, int] = defaultdict(int)
            for s in samples:
                for k, v in (s.metadata or {}).get("class_counts", {}).items():
                    agg[k] += v
            (out_dir / "_class_distribution.json").write_text(
                json.dumps(dict(agg), indent=2) + "\n"
            )
        except Exception as e:  # pragma: no cover
            print(f"[run] could not write class distribution: {e}", flush=True)
        print(f"[run] total emitted = {len(samples)}", flush=True)
        return samples
