"""Video + overlay primitives for VinDr-CXR (Linux-safe ffmpeg rendering).

CXR-specific notes:
- DO NOT horizontal-flip — left/right asymmetry is diagnostic (heart on left,
  cardiomegaly only counts if the cardiac silhouette is enlarged on its real side).
- Greyscale CXRs are loaded as 3-channel BGR for consistent overlay code.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


# 14 VinDr-CXR abnormality classes plus "No finding". Each gets a distinct
# colour for the bbox-overlay video. Colours are BGR (cv2 convention).
# Cardiomegaly red; pleural effusion blue (per task spec); rest distinguishable.
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "Aortic enlargement":      (0, 215, 255),   # gold
    "Atelectasis":             (180, 105, 255), # hot pink
    "Calcification":           (147, 20, 255),  # deep pink
    "Cardiomegaly":            (0, 0, 255),     # red
    "Consolidation":           (0, 165, 255),   # orange
    "ILD":                     (255, 255, 0),   # cyan
    "Infiltration":            (50, 205, 50),   # lime
    "Lung Opacity":            (140, 230, 240), # khaki
    "Nodule/Mass":             (203, 192, 255), # pink
    "Other lesion":            (200, 200, 200), # grey
    "Pleural effusion":        (255, 0, 0),     # blue (per task spec)
    "Pleural thickening":      (255, 191, 0),   # deep sky blue
    "Pneumothorax":            (0, 255, 255),   # yellow
    "Pulmonary fibrosis":      (130, 0, 75),    # indigo
    "No finding":              (100, 100, 100), # dim grey
}


def resize_pad_square(img: np.ndarray, target: int = 512) -> np.ndarray:
    """Letterbox an image to a *target*x*target* square (zero-pad bottom/right).

    Returns the padded image. Use :func:`scale_bbox` to map original-coordinate
    bounding boxes into the padded image.
    """
    h, w = img.shape[:2]
    scale = target / max(h, w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    out = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    pad_h = target - nh
    pad_w = target - nw
    return cv2.copyMakeBorder(out, 0, pad_h, 0, pad_w,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def scale_bbox(
    box: Tuple[float, float, float, float],
    src_w: int, src_h: int, target: int,
) -> Tuple[int, int, int, int]:
    """Scale a (x_min, y_min, x_max, y_max) box from the original image to the
    letterboxed *target* square. Returns integer pixel coords."""
    scale = target / max(src_w, src_h)
    x1, y1, x2, y2 = box
    return (
        int(round(x1 * scale)),
        int(round(y1 * scale)),
        int(round(x2 * scale)),
        int(round(y2 * scale)),
    )


def draw_bbox_overlay(
    image: np.ndarray,
    boxes: Sequence[Tuple[int, int, int, int, str]],
    alpha: float = 0.35,
) -> np.ndarray:
    """Draw labelled, class-coloured bounding boxes on a copy of *image*.

    Each entry in *boxes* is (x1, y1, x2, y2, class_name). Boxes get a 3-px
    border + a translucent fill (alpha) + a top label ribbon with the class name.
    """
    out = image.copy()
    if not boxes:
        return out

    # Translucent fill layer (drawn first, blended below).
    fill = out.copy()
    for x1, y1, x2, y2, label in boxes:
        color = CLASS_COLORS.get(label, (200, 200, 200))
        cv2.rectangle(fill, (x1, y1), (x2, y2), color, thickness=-1)
    out = cv2.addWeighted(out, 1.0 - alpha, fill, alpha, 0)

    # Solid borders + label ribbons (always opaque so the box is readable).
    font = cv2.FONT_HERSHEY_SIMPLEX
    for x1, y1, x2, y2, label in boxes:
        color = CLASS_COLORS.get(label, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        (tw, th), _ = cv2.getTextSize(label, font, 0.55, 1)
        ribbon_top = max(0, y1 - th - 6)
        cv2.rectangle(out, (x1, ribbon_top), (x1 + tw + 8, y1), color, -1)
        cv2.putText(out, label, (x1 + 4, y1 - 4), font, 0.55, (255, 255, 255),
                    1, cv2.LINE_AA)
    return out


def make_video(frames: List[np.ndarray], out_path: Path, fps: int = 12) -> None:
    """Encode a list of BGR frames to an H.264 mp4 via ffmpeg pipe.

    Uses libx264 + yuv420p so the result plays in browsers (vm-dataset.com).
    Ensures even dimensions (libx264 requires width%2==0, height%2==0).
    """
    if not frames:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    w2, h2 = w - (w % 2), h - (h % 2)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-vf", f"scale={w2}:{h2}",
        str(out_path),
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        if f.shape[:2] != (h, w):
            f = cv2.resize(f, (w, h))
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()


# ──────────────────────────────────────────────────────────────────────────
#  Per-sample frame builders
# ──────────────────────────────────────────────────────────────────────────

def _zoom_frame(img: np.ndarray, zoom: float) -> np.ndarray:
    """Centre-zoom an image by *zoom* (1.0 = identity, >1.0 = zoom in).

    Used by the zoom-in fade-in video to focus the radiologist's eye onto the
    chest cavity. We never flip horizontally.
    """
    h, w = img.shape[:2]
    if zoom <= 1.0:
        return img
    new_w = int(round(w / zoom))
    new_h = int(round(h / zoom))
    x0 = (w - new_w) // 2
    y0 = (h - new_h) // 2
    crop = img[y0:y0 + new_h, x0:x0 + new_w]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def build_zoom_fadein_frames(
    base: np.ndarray, num_frames: int,
) -> List[np.ndarray]:
    """Construct a fade-in + slow zoom-in clip on the raw CXR.

    From a black frame, fade up to the original brightness over the first
    third while zooming from 1.0 -> 1.15. Hold the zoomed image for the
    remainder. Pure spatial focus — no flips.
    """
    frames: List[np.ndarray] = []
    fade_n = max(6, num_frames // 3)
    hold_n = num_frames - fade_n
    for i in range(fade_n):
        t = (i + 1) / float(fade_n)            # 0..1
        zoom = 1.0 + 0.15 * t
        frame = _zoom_frame(base, zoom)
        # Linear fade-in from black.
        frames.append(np.clip(frame.astype(np.float32) * t, 0, 255).astype(np.uint8))
    final_zoom = _zoom_frame(base, 1.15)
    for _ in range(hold_n):
        frames.append(final_zoom.copy())
    return frames


def build_bbox_reveal_frames(
    base: np.ndarray,
    boxes: Sequence[Tuple[int, int, int, int, str]],
    num_frames: int,
    alpha: float = 0.35,
) -> List[np.ndarray]:
    """Construct a clip that gradually reveals each bounding box one-by-one.

    The CXR itself is shown for the first ~10 frames (clean), then bboxes are
    introduced incrementally — frame N reveals the first ceil(N * K / total)
    boxes. Final frames hold the fully-annotated image.
    """
    frames: List[np.ndarray] = []
    n_boxes = len(boxes)
    intro_n = min(10, num_frames // 6)
    reveal_n = max(num_frames - intro_n - 10, 1)

    # Intro: clean CXR (gives the model a chance to "see" it first).
    for _ in range(intro_n):
        frames.append(base.copy())

    # Reveal: incrementally show boxes.
    if n_boxes == 0:
        # No findings — hold the clean image for the remaining duration.
        while len(frames) < num_frames:
            frames.append(base.copy())
        return frames

    for i in range(reveal_n):
        progress = (i + 1) / float(reveal_n)
        k = max(1, int(np.ceil(progress * n_boxes)))
        frame = draw_bbox_overlay(base, boxes[:k], alpha=alpha)
        frames.append(frame)

    # Hold the final fully-annotated frame.
    final = draw_bbox_overlay(base, boxes, alpha=alpha)
    while len(frames) < num_frames:
        frames.append(final.copy())
    return frames


def build_walkthrough_frames(
    base: np.ndarray,
    boxes: Sequence[Tuple[int, int, int, int, str]],
    num_frames: int,
    alpha: float = 0.35,
) -> List[np.ndarray]:
    """Construct a full-walkthrough clip: clean -> all-boxes -> per-box highlight.

    Roughly equal thirds:
      1. Clean CXR (orient the viewer).
      2. All bboxes overlaid (full annotation).
      3. Per-box cycle — each finding is highlighted alone, then the full
         overlay returns at the end.
    """
    frames: List[np.ndarray] = []
    third = max(num_frames // 3, 4)

    # Part 1 — clean.
    for _ in range(third):
        frames.append(base.copy())

    # Part 2 — full overlay.
    full = draw_bbox_overlay(base, boxes, alpha=alpha)
    for _ in range(third):
        frames.append(full.copy())

    # Part 3 — per-box cycle (or hold full if no boxes).
    remaining = num_frames - 2 * third
    if not boxes or remaining <= 0:
        while len(frames) < num_frames:
            frames.append(full.copy())
        return frames

    per_box = max(remaining // max(len(boxes), 1), 2)
    for box in boxes:
        single = draw_bbox_overlay(base, [box], alpha=alpha)
        for _ in range(per_box):
            if len(frames) >= num_frames:
                break
            frames.append(single.copy())
        if len(frames) >= num_frames:
            break

    while len(frames) < num_frames:
        frames.append(full.copy())
    return frames
