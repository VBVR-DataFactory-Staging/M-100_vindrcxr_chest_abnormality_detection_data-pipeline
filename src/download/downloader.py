"""Vestigial download module for M-100 VinDr-CXR.

The real raw-data fetching lives in ``src.pipeline.pipeline.TaskPipeline._ensure_raw``
which uses ``aws s3 sync`` directly (the med-vr-datasets bucket is private, so
public-HTTP helpers in ``core.download`` would 400). This module is kept only
to satisfy ``core.download.run_download``'s lazy import of
``src.download.create_downloader`` — callers of TaskPipeline.run() never hit it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional


class TaskDownloader:
    """Pass-through downloader. The real sync happens inside the pipeline."""

    def __init__(self, config):
        self.config = config
        self.raw_dir = Path(getattr(config, "raw_dir", "raw"))

    def download(self, limit: Optional[int] = None) -> Iterator[dict]:
        # Empty iterator — TaskPipeline.download() ignores this class entirely
        # and yields raw samples from its own annotation index.
        if False:
            yield {}


def create_downloader(config) -> TaskDownloader:
    return TaskDownloader(config)
