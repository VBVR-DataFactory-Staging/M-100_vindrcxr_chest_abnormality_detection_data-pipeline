"""VinDr-CXR downloader: list (image, label) pairs from the S3 raw mirror.

Raw data layout (produced by the onramp pull-to-s3 run):
    s3://med-vr-datasets/M-100/VinDrCXR/
        data.yaml
        images/train/<stem>.png
        images/val/<stem>.png
        labels/train/<stem>.txt     # YOLO format: cls cx cy w h  (normalized)
        labels/val/<stem>.txt

This downloader streams (image_path, label_path, split, stem) tuples — it does
**not** load the pixel data here. The pipeline opens each image lazily so we
can iterate thousands of studies without blowing RAM.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import boto3
from botocore.config import Config as BotoConfig


class TaskDownloader:
    """Mirror s3://med-vr-datasets/M-100/VinDrCXR/ locally, then list study pairs."""

    def __init__(self, config):
        self.config = config
        self.raw_dir = Path(getattr(config, "raw_dir", "raw"))
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.bucket = config.s3_bucket
        self.prefix = config.s3_prefix.rstrip("/") + "/"
        self._s3 = boto3.client(
            "s3",
            config=BotoConfig(retries={"max_attempts": 10, "mode": "adaptive"}),
        )

    # ---------- S3 sync ----------

    def _needs_sync(self) -> bool:
        # Sync only if we have no images yet.
        for split in ("train", "val"):
            d = self.raw_dir / "images" / split
            if d.exists() and any(d.glob("*.png")):
                return False
        return True

    def _sync_from_s3(self) -> None:
        print(
            f"Syncing s3://{self.bucket}/{self.prefix} -> {self.raw_dir}/",
            flush=True,
        )
        paginator = self._s3.get_paginator("list_objects_v2")
        downloaded = 0
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []) or []:
                key = obj["Key"]
                rel = key[len(self.prefix):] if key.startswith(self.prefix) else key
                if not rel:
                    continue
                dst = self.raw_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists() and dst.stat().st_size == obj["Size"]:
                    continue
                self._s3.download_file(self.bucket, key, str(dst))
                downloaded += 1
                if downloaded % 500 == 0:
                    print(f"  synced {downloaded} files...", flush=True)
        print(f"Sync done ({downloaded} new files).", flush=True)

    # ---------- iteration ----------

    def _iter_pairs(self) -> Iterator[dict]:
        img_dir = self.raw_dir / "images"
        lbl_dir = self.raw_dir / "labels"
        if not img_dir.exists():
            return
        # Deterministic ordering: split + stem alphabetical.
        for split in ("train", "val"):
            s_img = img_dir / split
            s_lbl = lbl_dir / split
            if not s_img.exists():
                continue
            for img_path in sorted(s_img.glob("*.png")):
                stem = img_path.stem
                # Skip augmented duplicates (_aug.png) — we want distinct studies.
                if stem.endswith("_aug"):
                    continue
                lbl_path = s_lbl / f"{stem}.txt"
                yield {
                    "dataset": "VinDr-CXR",
                    "source_id": stem,
                    "split": split,
                    "image_path": str(img_path),
                    "label_path": str(lbl_path) if lbl_path.exists() else None,
                }

    def download(self, limit: Optional[int] = None) -> Iterator[dict]:
        if self._needs_sync():
            self._sync_from_s3()
        else:
            print(f"Raw already present at {self.raw_dir}, skipping S3 sync.", flush=True)

        cap = getattr(self.config, "max_samples", None)
        if limit is None:
            limit = cap
        elif cap is not None:
            limit = min(limit, cap)

        yielded = 0
        for row in self._iter_pairs():
            if limit is not None and yielded >= limit:
                return
            yield row
            yielded += 1
        print(f"Downloader yielded {yielded} VinDr-CXR studies.", flush=True)


def create_downloader(config) -> TaskDownloader:
    """Factory called by ``core.download.run_download()``."""
    return TaskDownloader(config)
