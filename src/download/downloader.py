"""INSPECT M-100 downloader.

INSPECT layout under
``s3://med-vr-datasets/M-100/inspect_pe/inspectamultimodaldatasetforpulmonaryembolismdiagnosisandprog-3/full/``::

    labels_20250611.tsv          # impression_id → pe_positive, mortality, PE acute, ...
    impressions_20250611.tsv     # impression_id → free-text radiology impression
    study_mapping_20250611.tsv   # impression_id → image_id (PE<hex>) + person_id + dates
    CTPA/
        PE<hex>.nii.gz           # ~3370 CT volumes (NIfTI, int16 HU)
    EHR/
        README.md                # OMOP STARR pointer (no in-bundle EHR rows)

We can't pull every CT (309 GB). Instead:

  1. Pull the small TSVs once into ``raw/``.
  2. Build a ``image_id → s3_key`` lookup over CTPA/.
  3. Yield one raw sample per labels row that has a matching CT, pulling the
     volume on demand and deleting it after use.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import boto3
import pandas as pd


_TSV_NAMES = (
    "labels_20250611.tsv",
    "impressions_20250611.tsv",
    "study_mapping_20250611.tsv",
)


def _ensure_tsvs(bucket: str, prefix: str, raw_dir: Path) -> dict:
    """Download the small TSV metadata files if not cached; return DataFrames."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    out: dict = {}
    for name in _TSV_NAMES:
        local = raw_dir / name
        if not local.exists() or local.stat().st_size == 0:
            key = f"{prefix}{name}"
            print(f"  [download] s3://{bucket}/{key} -> {local}", flush=True)
            s3.download_file(bucket, key, str(local))
        out[name] = pd.read_csv(local, sep="\t")
    return out


def _build_image_index(bucket: str, prefix: str, raw_dir: Path) -> dict:
    """Return {image_id(str) -> s3_key(str)} mapping for all CTPA/*.nii.gz files.

    Cached to ``raw/ct_index.csv`` so we don't re-list 3000+ keys every run.
    """
    cache = raw_dir / "ct_index.csv"
    if cache.exists() and cache.stat().st_size > 0:
        df = pd.read_csv(cache)
        return dict(zip(df["image_id"].astype(str), df["s3_key"].astype(str)))

    s3 = boto3.client("s3")
    image_to_key: dict = {}
    sub_prefix = f"{prefix}CTPA/"
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=sub_prefix)
    for page in pages:
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            stem = key.rsplit("/", 1)[-1]
            if not stem.endswith(".nii.gz"):
                continue
            image_id = stem[: -len(".nii.gz")]
            image_to_key[image_id] = key

    rows = [{"image_id": i, "s3_key": k} for i, k in sorted(image_to_key.items())]
    pd.DataFrame(rows).to_csv(cache, index=False)
    print(f"  [index] built ct_index.csv with {len(image_to_key)} CT volumes", flush=True)
    return image_to_key


def _safe_bool(val) -> Optional[bool]:
    """Map TSV truthy strings ('TRUE'/'FALSE'/'Censored'/blank) → bool/None."""
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    s = str(val).strip().upper()
    if s in ("TRUE", "1", "T", "Y", "YES"):
        return True
    if s in ("FALSE", "0", "F", "N", "NO"):
        return False
    return None  # 'CENSORED' / NaN / unknown


def run_download(config) -> Iterator[dict]:
    """Yield one raw sample per INSPECT study (where a CT volume exists).

    Each raw sample dict::

        {
            "impression_id": int,
            "image_id":      str,        # PE<hex>
            "label":         int,        # 1 = PE positive, 0 = negative
            "pe_acute":      int,
            "pe_subseg":     int,
            "mortality_12m": Optional[bool],
            "impression":    str,        # radiology free text
            "s3_bucket":     str,
            "s3_key":        str,        # CTPA/PE<hex>.nii.gz key
            "person_id":     str,
            "procedure_dt":  str,
        }
    """
    bucket = config.s3_bucket
    prefix = config.s3_prefix
    raw_dir = Path(config.raw_dir)

    tsvs = _ensure_tsvs(bucket, prefix, raw_dir)
    labels = tsvs["labels_20250611.tsv"]
    impressions = tsvs["impressions_20250611.tsv"]
    mapping = tsvs["study_mapping_20250611.tsv"]

    image_index = _build_image_index(bucket, prefix, raw_dir)

    impr_text = dict(
        zip(impressions["impression_id"].astype(int), impressions["impressions"].astype(str))
    )
    map_by_impr = mapping.set_index("impression_id", drop=False)

    labels_sorted = labels.sort_values("impression_id").reset_index(drop=True)
    for _, row in labels_sorted.iterrows():
        impr_id = int(row["impression_id"])
        if impr_id not in map_by_impr.index:
            continue
        m = map_by_impr.loc[impr_id]
        if hasattr(m, "iloc") and not isinstance(m, pd.Series):
            m = m.iloc[0]
        image_id = str(m["image_id"])
        s3_key = image_index.get(image_id)
        if not s3_key:
            continue

        # PE label: prefer pe_positive (binary 0/1), fall back to pe_positive_nlp.
        label_raw = row.get("pe_positive")
        try:
            label = int(label_raw)
        except (TypeError, ValueError):
            label = 1 if _safe_bool(row.get("pe_positive_nlp")) else 0

        try:
            pe_acute = int(row.get("pe_acute") or 0)
        except (TypeError, ValueError):
            pe_acute = 0
        try:
            pe_sub = int(row.get("pe_subsegmentalonly") or 0)
        except (TypeError, ValueError):
            pe_sub = 0

        yield {
            "impression_id": impr_id,
            "image_id": image_id,
            "label": label,
            "pe_acute": pe_acute,
            "pe_subseg": pe_sub,
            "mortality_12m": _safe_bool(row.get("12_month_mortality")),
            "impression": impr_text.get(impr_id, ""),
            "s3_bucket": bucket,
            "s3_key": s3_key,
            "person_id": str(m.get("person_id", "")),
            "procedure_dt": str(m.get("procedure_DATETIME", "")),
        }


def create_downloader(config) -> "TaskDownloader":
    return TaskDownloader(config)


class TaskDownloader:
    """Streams INSPECT raw samples from S3 one study at a time."""

    def __init__(self, config):
        self.config = config
        self.raw_dir = Path(config.raw_dir)

    def download(self, limit: Optional[int] = None) -> Iterator[dict]:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for sample in run_download(self.config):
            yield sample
            count += 1
            if limit is not None and count >= limit:
                break


def fetch_ct_volume(bucket: str, key: str, cache_dir: Path):
    """Download one CT NIfTI volume, load as numpy (Z, H, W) int16, then delete file.

    Returns ``None`` if download or decode fails.
    """
    import nibabel as nib  # lazy import — only needed when actually fetching CTs
    import numpy as np

    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / Path(key).name
    s3 = boto3.client("s3")
    try:
        s3.download_file(bucket, key, str(local))
    except Exception as e:
        print(f"  [download] failed {key}: {e}", flush=True)
        return None

    arr = None
    try:
        img = nib.load(str(local))
        data = img.get_fdata(caching="unchanged")
        # NIfTI conv: data is (X, Y, Z). Convert to (Z, H, W) where Z=axial slice.
        if data.ndim == 3:
            arr = np.transpose(data, (2, 1, 0)).astype(np.int16)
        elif data.ndim == 4:
            arr = np.transpose(data[..., 0], (2, 1, 0)).astype(np.int16)
    except Exception as e:
        print(f"  [load] failed {local}: {e}", flush=True)
        arr = None
    finally:
        try:
            local.unlink(missing_ok=True)
        except Exception:
            pass
    return arr
