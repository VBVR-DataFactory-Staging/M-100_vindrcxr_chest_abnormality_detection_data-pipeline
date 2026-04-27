"""Transforms for M-100 INSPECT pulmonary embolism multimodal pipeline.

Pipeline = CT volume + radiology impression text → axial sweep video (+ side panel).

Rendering contract:

    frame = [ CT_axial_slice (square)  ||  Clinical text panel ]

    first_video        = axial sweep through the PE region, NO highlight.
    last_video         = same sweep with red mask (40% opacity) on PE-likely
                         regions (PE-positive only) + reveal banner in side panel.
    ground_truth_video = same as last_video.

PE-region detection is heuristic — INSPECT does not ship per-voxel masks.
We use HU thresholds that pick out contrast-enhanced pulmonary arteries inside
a central-chest ROI, and visualise them in red on PE-positive studies.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ──────────────────────────────────────────────────────────────────────────────
#  CT windowing
# ──────────────────────────────────────────────────────────────────────────────

def window_slice(hu: np.ndarray, wl: int, ww: int) -> np.ndarray:
    """Apply HU window → uint8 grayscale."""
    vmin = wl - ww / 2.0
    vmax = wl + ww / 2.0
    img = np.clip((hu.astype(np.float32) - vmin) / (vmax - vmin) * 255.0, 0, 255)
    return img.astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
#  Axial-sweep slice selection
# ──────────────────────────────────────────────────────────────────────────────

def pick_sweep_indices(num_slices: int, num_frames: int) -> List[int]:
    """Pick N evenly-spaced slices from the central ~60% of the volume."""
    if num_slices <= 0:
        return []
    lo = int(num_slices * 0.20)
    hi = max(lo + 1, int(num_slices * 0.80))
    hi = min(hi, num_slices)
    if hi - lo < num_frames:
        lo, hi = 0, num_slices
    idxs = np.linspace(lo, hi - 1, num=min(num_frames, hi - lo)).round().astype(int)
    seen = set()
    out: List[int] = []
    for i in idxs:
        ii = int(i)
        if ii in seen:
            continue
        seen.add(ii)
        out.append(ii)
    return out


# ──────────────────────────────────────────────────────────────────────────────
#  PE mask heuristic (INSPECT has no per-voxel labels)
# ──────────────────────────────────────────────────────────────────────────────

def pe_heuristic_mask(hu_slice: np.ndarray) -> np.ndarray:
    """Heuristic pulmonary-artery highlight: bright (contrast-enhanced)
    voxels inside a central-chest circular ROI. Returns a bool mask.
    """
    h, w = hu_slice.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    radius = min(h, w) * 0.30
    roi = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (radius ** 2)
    # PE-protocol CT: contrast-enhanced arteries sit around +80..+400 HU.
    bright = (hu_slice >= 80) & (hu_slice <= 400)
    return roi & bright


def colorize_ct_slice(
    hu: np.ndarray,
    wl: int,
    ww: int,
    pe_mask: Optional[np.ndarray] = None,
    alpha: float = 0.40,
) -> np.ndarray:
    """Render CT slice as RGB; overlay PE-likely regions in red (optional)."""
    gray = window_slice(hu, wl, ww)
    rgb = np.stack([gray, gray, gray], axis=-1).astype(np.uint8)

    # Faint blue tint on normal vasculature.
    vessel_mask = (hu >= 60) & (hu < 80)
    if vessel_mask.any():
        blue = np.array([0, 100, 255], dtype=np.float32)
        base = rgb[vessel_mask].astype(np.float32)
        rgb[vessel_mask] = (base * 0.85 + blue * 0.15).astype(np.uint8)

    if pe_mask is not None and pe_mask.any():
        red = np.array([255, 0, 0], dtype=np.float32)
        base = rgb[pe_mask].astype(np.float32)
        rgb[pe_mask] = (base * (1 - alpha) + red * alpha).astype(np.uint8)

    return rgb


# ──────────────────────────────────────────────────────────────────────────────
#  Side panel (impression text + label flags)
# ──────────────────────────────────────────────────────────────────────────────

_FONT_CACHE: dict = {}


def _load_font(size: int) -> ImageFont.ImageFont:
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                font = ImageFont.truetype(path, size)
                _FONT_CACHE[size] = font
                return font
            except Exception:
                continue
    font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


_NUM_PREFIX = re.compile(r"^\s*\d+\s*[.)]\s*")


def _summarize_impression(text: str, max_chars: int = 380) -> str:
    """Trim very long radiology impressions for side-panel display."""
    if not text:
        return "(no impression text)"
    t = text.strip().strip('"')
    # Keep first 2-3 numbered findings.
    parts = re.split(r"\s*\d+\s*[.)]\s*", t)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) >= 2:
        joined = " · ".join(parts[:3])
    else:
        joined = t
    joined = re.sub(r"\s+", " ", joined)
    if len(joined) > max_chars:
        joined = joined[: max_chars - 1] + "…"
    return joined


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> List[str]:
    """Greedy word-wrap for the impression paragraph."""
    words = text.split()
    if not words:
        return []
    lines: List[str] = []
    line = words[0]
    for w in words[1:]:
        cand = line + " " + w
        bbox = draw.textbbox((0, 0), cand, font=font)
        if bbox[2] - bbox[0] <= max_width:
            line = cand
        else:
            lines.append(line)
            line = w
    lines.append(line)
    return lines


def render_ehr_panel(
    raw: dict,
    width: int,
    height: int,
    title: str,
    reveal: bool = False,
) -> np.ndarray:
    """Render a clinical side panel (dark bg, white text) as uint8 RGB."""
    img = Image.new("RGB", (width, height), (18, 20, 26))
    draw = ImageDraw.Draw(img)

    font_title = _load_font(20)
    font_label = _load_font(14)
    font_value = _load_font(14)
    font_body = _load_font(13)
    font_small = _load_font(12)

    pad = 16
    y = pad
    draw.text((pad, y), title, font=font_title, fill=(220, 230, 240))
    y += 28
    draw.line((pad, y, width - pad, y), fill=(80, 90, 110), width=1)
    y += 10

    label_col = pad
    value_col = pad + 130

    rows: List[Tuple[str, str]] = [
        ("Patient", f"id={raw.get('person_id', '?') or '?'}"),
        ("Image ID", str(raw.get("image_id", "?"))),
        ("Procedure", str(raw.get("procedure_dt", "") or "")[:10] or "—"),
        ("", ""),
        ("— Impression —", ""),
    ]
    for label, value in rows:
        if not label and not value:
            y += 6
            continue
        if label.startswith("—"):
            draw.text((label_col, y), label, font=font_label, fill=(140, 180, 220))
        else:
            draw.text((label_col, y), label, font=font_label, fill=(180, 190, 200))
            draw.text((value_col, y), value, font=font_value, fill=(240, 245, 250))
        y += 18

    body = _summarize_impression(raw.get("impression", ""))
    body_lines = _wrap_text(draw, body, font_body, width - 2 * pad)
    # Limit how many lines we draw so we leave room for footer + banner.
    max_body_lines = max(3, (height - y - 90) // 16)
    for line in body_lines[:max_body_lines]:
        draw.text((pad, y), line, font=font_body, fill=(220, 230, 240))
        y += 16

    # Footer
    y_foot = height - pad - 40
    draw.line((pad, y_foot, width - pad, y_foot), fill=(80, 90, 110), width=1)
    draw.text(
        (pad, y_foot + 4),
        "INSPECT  (Stanford AIMI, 2023)",
        font=font_small,
        fill=(140, 150, 170),
    )
    draw.text(
        (pad, y_foot + 18),
        "CTPA + impression → PE yes/no",
        font=font_small,
        fill=(140, 150, 170),
    )

    if reveal:
        label = int(raw.get("label", 0))
        pe_acute = int(raw.get("pe_acute", 0))
        pe_sub = int(raw.get("pe_subseg", 0))
        if label == 1:
            banner = (200, 30, 30)
            tag = "acute" if pe_acute else ("subsegmental" if pe_sub else "PE")
            text = f"PE POSITIVE  ({tag})"
        else:
            banner = (30, 150, 80)
            text = "PE NEGATIVE"
        bh = 30
        draw.rectangle((0, height - bh, width, height), fill=banner)
        draw.text((pad, height - bh + 7), text, font=font_label, fill=(250, 250, 255))

    return np.array(img, dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
#  Frame composition + video writing
# ──────────────────────────────────────────────────────────────────────────────

def compose_frame(ct_rgb: np.ndarray, panel_rgb: np.ndarray) -> np.ndarray:
    """Concatenate CT slice and side panel side by side (same height)."""
    h = ct_rgb.shape[0]
    if panel_rgb.shape[0] != h:
        pil = Image.fromarray(panel_rgb)
        pil = pil.resize((panel_rgb.shape[1], h), Image.Resampling.LANCZOS)
        panel_rgb = np.array(pil)
    out = np.concatenate([ct_rgb, panel_rgb], axis=1)
    H, W = out.shape[:2]
    if W % 2:
        out = out[:, :-1]
    if H % 2:
        out = out[:-1, :]
    return out


def write_mp4(frames: List[np.ndarray], out_path: Path, fps: int) -> None:
    """Write RGB frames to an H.264 mp4 via ffmpeg piping."""
    if not frames:
        return
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    w2 = w - (w % 2)
    h2 = h - (h % 2)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-vf", f"scale={w2}:{h2}",
        str(out_path),
    ]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for f in frames:
            if f.shape[:2] != (h, w):
                pil = Image.fromarray(f)
                pil = pil.resize((w, h), Image.Resampling.LANCZOS)
                f = np.array(pil)
            p.stdin.write(f.astype(np.uint8).tobytes())
    finally:
        p.stdin.close()
        p.wait()


def resize_square(rgb: np.ndarray, size: int) -> np.ndarray:
    """Resize an RGB frame to size x size via PIL LANCZOS."""
    pil = Image.fromarray(rgb)
    pil = pil.resize((size, size), Image.Resampling.LANCZOS)
    return np.array(pil)
