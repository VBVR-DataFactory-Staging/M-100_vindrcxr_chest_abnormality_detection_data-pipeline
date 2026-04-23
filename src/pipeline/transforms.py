"""Rendering utilities for M-100 VinDr-CXR bbox detection videos.

Builds three clips per sample:
  - first_video: 20-frame slow-zoom of the chest X-ray, no overlay
  - last_video:  same zoom stack + per-class colored bbox + class label text
  - ground_truth: raw -> (fade in bboxes) -> fully annotated

Plus first_frame.png (raw) and final_frame.png (annotated) keyframes.

YOLO label format on disk (normalized [0,1]): "cls cx cy w h".
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


# 22 VinDr-CXR classes (matches data.yaml in Benxelua/vindr-png-yolo-rescale).
VINDR_CLASSES: List[str] = [
    "Aortic enlargement",     # 0
    "Atelectasis",            # 1
    "Calcification",          # 2
    "Cardiomegaly",           # 3
    "Clavicle fracture",      # 4
    "Consolidation",          # 5
    "Edema",                  # 6
    "Emphysema",              # 7
    "Enlarged PA",            # 8
    "ILD",                    # 9
    "Infiltration",           # 10
    "Lung Opacity",           # 11
    "Lung cavity",            # 12
    "Lung cyst",              # 13
    "Mediastinal shift",      # 14
    "Nodule/Mass",            # 15
    "Other lesion",           # 16
    "Pleural effusion",       # 17
    "Pleural thickening",     # 18
    "Pneumothorax",           # 19
    "Pulmonary fibrosis",     # 20
    "Rib fracture",           # 21
]

# 14 classes requested by the M-100 spec — used for the prompt + overlay palette.
PROMPT_CLASSES: List[str] = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification",
    "Cardiomegaly",
    "Consolidation",
    "ILD",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
]


def _distinct_bgr_palette(n: int) -> List[Tuple[int, int, int]]:
    """Distinct hues spread evenly on HSV, converted to OpenCV BGR tuples."""
    palette = []
    for i in range(n):
        hue = int(round(179 * i / max(1, n)))
        hsv = np.uint8([[[hue, 220, 245]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        palette.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return palette


# Colors keyed by *class name* so the same disease always gets the same hue.
CLASS_COLORS = dict(zip(VINDR_CLASSES, _distinct_bgr_palette(len(VINDR_CLASSES))))


# -----------------------------------------------------------------------------
# Label I/O
# -----------------------------------------------------------------------------

def read_yolo_labels(label_path: str | None) -> List[Tuple[int, float, float, float, float]]:
    """Read a YOLO-format label file. Returns [(cls, cx, cy, w, h), ...] normalized.

    Empty or missing file -> empty list (meaning "no findings").
    """
    if not label_path:
        return []
    p = Path(label_path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(x) for x in parts[1:5])
        except ValueError:
            continue
        out.append((cls, cx, cy, w, h))
    return out


# -----------------------------------------------------------------------------
# Image / drawing helpers
# -----------------------------------------------------------------------------

def _load_bgr(image_path: str, size: Tuple[int, int]) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not open image: {image_path}")
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _yolo_to_xyxy(
    cx: float, cy: float, w: float, h: float, W: int, H: int,
) -> Tuple[int, int, int, int]:
    x1 = int(round((cx - w / 2) * W))
    y1 = int(round((cy - h / 2) * H))
    x2 = int(round((cx + w / 2) * W))
    y2 = int(round((cy + h / 2) * H))
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    return x1, y1, x2, y2


def draw_bboxes(
    frame_bgr: np.ndarray,
    labels: Sequence[Tuple[int, float, float, float, float]],
    *,
    alpha: float = 1.0,
    thickness: int = 2,
    draw_text: bool = True,
) -> np.ndarray:
    """Draw per-class colored bboxes + label text onto a BGR frame."""
    H, W = frame_bgr.shape[:2]
    if not labels:
        return frame_bgr.copy()

    overlay = frame_bgr.copy()
    for cls, cx, cy, w, h in labels:
        if not (0 <= cls < len(VINDR_CLASSES)):
            continue
        name = VINDR_CLASSES[cls]
        color = CLASS_COLORS.get(name, (255, 255, 255))
        x1, y1, x2, y2 = _yolo_to_xyxy(cx, cy, w, h, W, H)
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        if draw_text:
            label = name
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            pad = 3
            bx1, by1 = x1, max(0, y1 - th - 2 * pad)
            bx2, by2 = x1 + tw + 2 * pad, y1
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, -1)
            cv2.putText(
                overlay, label, (bx1 + pad, by2 - pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA,
            )

    if alpha < 1.0:
        return cv2.addWeighted(overlay, alpha, frame_bgr, 1.0 - alpha, 0)
    return overlay


def draw_banner(frame_bgr: np.ndarray, text: str) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    bh = max(30, h // 16)
    cv2.rectangle(out, (0, h - bh), (w, h), (0, 0, 0), -1)
    cv2.putText(
        out, text, (12, h - bh // 3),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return out


def zoom_crop(frame_bgr: np.ndarray, progress: float) -> np.ndarray:
    """Slow zoom-in (1.0 -> 1.12x) keeping output size constant."""
    zoom = 1.0 + 0.12 * float(np.clip(progress, 0.0, 1.0))
    h, w = frame_bgr.shape[:2]
    new_w, new_h = int(w / zoom), int(h / zoom)
    x0 = (w - new_w) // 2
    y0 = (h - new_h) // 2
    crop = frame_bgr[y0:y0 + new_h, x0:x0 + new_w]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


# -----------------------------------------------------------------------------
# Video writer (ffmpeg libx264 with mp4v fallback)
# -----------------------------------------------------------------------------

def write_mp4(frames: List[np.ndarray], out_path: Path, fps: int) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("write_mp4: empty frames list")

    h, w = frames[0].shape[:2]
    tmp_dir = Path(tempfile.mkdtemp(prefix="vbvr_tmp_mp4_"))
    try:
        for i, f in enumerate(frames):
            cv2.imwrite(str(tmp_dir / f"f_{i:05d}.png"), f)
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", str(tmp_dir / "f_%05d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            str(out_path),
        ]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            return out_path
    except FileNotFoundError:
        pass
    finally:
        for p in tmp_dir.glob("*"):
            try:
                p.unlink()
            except OSError:
                pass
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    try:
        for f in frames:
            vw.write(f)
    finally:
        vw.release()
    return out_path


# -----------------------------------------------------------------------------
# Sample builder
# -----------------------------------------------------------------------------

def build_sample_videos(
    image_path: str,
    labels: Sequence[Tuple[int, float, float, float, float]],
    task_id: str,
    out_dir: Path,
    *,
    width: int,
    height: int,
    fps: int,
    num_frames: int,
) -> dict:
    """Render three clips + keyframes for one CXR study."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    size = (int(width), int(height))
    raw_bgr = _load_bgr(image_path, size)
    ann_bgr = draw_bboxes(raw_bgr, labels)

    banner_raw = "Chest X-ray (raw)"
    banner_anno = "Annotated (per-class bbox + label)"

    first_frames, last_frames, gt_frames = [], [], []
    fade_end = max(1, num_frames // 2)
    for i in range(num_frames):
        prog = i / max(1, num_frames - 1)
        f1 = zoom_crop(raw_bgr, prog)
        first_frames.append(draw_banner(f1, banner_raw))
        f2 = zoom_crop(ann_bgr, prog)
        last_frames.append(draw_banner(f2, banner_anno))
        if i <= fade_end:
            a = i / fade_end
            blended = cv2.addWeighted(raw_bgr, 1.0 - a, ann_bgr, a, 0)
            lab = "Bbox reveal"
        else:
            blended = ann_bgr
            lab = banner_anno
        f3 = zoom_crop(blended, prog)
        gt_frames.append(draw_banner(f3, lab))

    first_video = write_mp4(first_frames, out_dir / f"{task_id}_first.mp4", fps)
    last_video = write_mp4(last_frames, out_dir / f"{task_id}_last.mp4", fps)
    gt_video = write_mp4(gt_frames, out_dir / f"{task_id}_gt.mp4", fps)

    first_frame_pil = Image.fromarray(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB))
    final_frame_pil = Image.fromarray(cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB))

    H, W = raw_bgr.shape[:2]
    boxes = []
    present = set()
    for cls, cx, cy, w, h in labels:
        if not (0 <= cls < len(VINDR_CLASSES)):
            continue
        name = VINDR_CLASSES[cls]
        present.add(name)
        x1, y1, x2, y2 = _yolo_to_xyxy(cx, cy, w, h, W, H)
        boxes.append({
            "class_id": cls,
            "class_name": name,
            "bbox_xyxy_px": [x1, y1, x2, y2],
            "bbox_yolo_norm": [round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)],
        })

    return {
        "first_video": str(first_video),
        "last_video": str(last_video),
        "ground_truth_video": str(gt_video),
        "first_frame": first_frame_pil,
        "final_frame": final_frame_pil,
        "boxes": boxes,
        "present_classes": sorted(present),
        "num_boxes": len(boxes),
    }
