"""M-041 / M-041 / M-041: DDR (Diabetic Retinopathy) dataset pipelines.

M-041: lesion pixel segmentation (4 lesion types: EX hard exudate,
       HE hemorrhage, MA microaneurysm, SE soft exudate)
M-041: lesion bbox detection (same lesion types, VOC XML)
M-041: DR severity grading (0=none, 1=mild, 2=moderate, 3=severe, 4=PDR)

All three are Case D (independent fundus images). Loop 8 samples at 8 fps.
"""
from __future__ import annotations
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from common import (
    DATA_ROOT, write_task, COLORS, fit_square, draw_bbox,
)

ROOT = DATA_ROOT / "_extracted" / "13_DDR_full" / "DDR-dataset"
LESION_TYPES = ["EX", "HE", "MA", "SE"]
LESION_FULLNAMES = {
    "EX": "hard exudate",
    "HE": "hemorrhage",
    "MA": "microaneurysm",
    "SE": "soft exudate (cotton-wool spot)",
}
LESION_COLORS = {
    "EX": COLORS["yellow"],
    "HE": COLORS["red"],
    "MA": COLORS["magenta"],
    "SE": COLORS["cyan"],
}
DR_GRADES = {
    0: "no DR",
    1: "mild non-proliferative DR",
    2: "moderate non-proliferative DR",
    3: "severe non-proliferative DR",
    4: "proliferative DR",
}

# -------------------- M-041 lesion seg --------------------

PROMPT_043 = (
    "This is a color fundus photograph from the DDR diabetic retinopathy "
    "dataset. Segment the four types of retinal lesions: hard exudates "
    "(yellow), hemorrhages (red), microaneurysms (magenta), and soft exudates "
    "(cyan). Overlay each lesion with its assigned color and contour. The "
    "ground_truth video keeps only the fundus images containing lesions."
)


def load_lesion_seg(stem: str):
    img = cv2.imread(str(ROOT / "lesion_segmentation" / "train" / "image" / f"{stem}.jpg"),
                     cv2.IMREAD_COLOR)
    masks = {}
    for lt in LESION_TYPES:
        p = ROOT / "lesion_segmentation" / "train" / "label" / lt / f"{stem}.tif"
        if p.exists():
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is not None and (m > 0).any():
                masks[lt] = (m > 0).astype(np.uint8)
    return img, masks


def overlay_lesions(img_bgr, masks):
    out = img_bgr.copy()
    # Dilate tiny lesions (esp. MA microaneurysms) so they are visible
    # after the display downsample.
    dilate_kernel = np.ones((3, 3), np.uint8)
    for lt, mask in masks.items():
        color = LESION_COLORS[lt]
        m_uint8 = (mask > 0).astype(np.uint8)
        # MA are pinpoint — enlarge them more
        iterations = 3 if lt == "MA" else 1
        m_dilated = cv2.dilate(m_uint8, dilate_kernel, iterations=iterations)
        layer = out.copy()
        layer[m_dilated > 0] = color
        out = cv2.addWeighted(layer, 0.55, out, 0.45, 0)
        cnts, _ = cv2.findContours(m_dilated * 255, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color, 2)
    return out


def run_m043():
    stems = ["007-6606-400", "007-6475-400", "007-6160-300", "007-5885-300",
             "007-6894-400", "007-4885-300", "007-5169-300", "007-5510-300"]
    samples = [load_lesion_seg(s) for s in stems]
    for task_idx in range(2):
        ordered = samples[task_idx * 4:(task_idx + 1) * 4] or samples[:4]
        loop = (ordered * 2)[:8]
        first_frames, last_frames, gt_frames, flags = [], [], [], []
        for img, masks in loop:
            sq = fit_square(img, 512)
            sq_masks = {lt: fit_square(m, 512, is_mask=True) for lt, m in masks.items()}
            ann = overlay_lesions(sq, sq_masks)
            first_frames.append(sq)
            last_frames.append(ann)
            has = any(m.any() for m in sq_masks.values())
            flags.append(has)
            if has:
                gt_frames.append(ann)
        if not gt_frames:
            gt_frames = last_frames[:5]
        meta = {
            "task": "4-class retinal lesion segmentation",
            "dataset": "DDR lesion_segmentation",
            "lesion_types": LESION_TYPES,
            "colors": {k: k for k in LESION_COLORS},
            "fps_source": "manual (case D circular loop)",
            "source_split": "train",
        }
        d = write_task("M-042", "ddr_lesion_segmentation", task_idx,
                       first_frames[0], last_frames[0],
                       first_frames, last_frames, gt_frames,
                       PROMPT_043, meta, 8)
        print(f"  wrote {d}")


# -------------------- M-041 lesion detection (bbox) --------------------

PROMPT_044 = (
    "This is a color fundus photograph from the DDR diabetic retinopathy "
    "detection subset. Detect and localize every retinal lesion with a "
    "bounding box. Classify each lesion as: EX (hard exudate, yellow), HE "
    "(hemorrhage, red), MA (microaneurysm, magenta), or SE (soft exudate, "
    "cyan). The ground_truth video keeps only frames containing bounding boxes."
)


def parse_voc_xml(xml_path: Path):
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().upper()
        b = obj.find("bndbox")
        x1 = int(b.find("xmin").text)
        y1 = int(b.find("ymin").text)
        x2 = int(b.find("xmax").text)
        y2 = int(b.find("ymax").text)
        boxes.append((name, x1, y1, x2, y2))
    return boxes


def run_m044():
    stems = ["007-1774-100", "007-1782-100"]
    pairs = []
    for s in stems:
        img_p = ROOT / "lesion_segmentation" / "train" / "image" / f"{s}.jpg"
        xml_p = ROOT / "lesion_detection" / "train" / f"{s}.xml"
        img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        boxes = parse_voc_xml(xml_p) if xml_p.exists() else []
        pairs.append((img, boxes))

    for task_idx in range(2):
        ordered = pairs[task_idx:] + pairs[:task_idx]
        loop = (ordered * 4)[:8]
        first_frames, last_frames, gt_frames, flags = [], [], [], []
        for img, boxes in loop:
            # fit to 512 while tracking scale
            h, w = img.shape[:2]
            scale = 512 / max(h, w)
            resized = cv2.resize(img, (int(w * scale), int(h * scale)))
            sq = fit_square(resized, 512)
            pad_x = (512 - resized.shape[1]) // 2
            pad_y = (512 - resized.shape[0]) // 2
            scaled_boxes = [
                (x1 * scale + pad_x, y1 * scale + pad_y,
                 x2 * scale + pad_x, y2 * scale + pad_y, name)
                for name, x1, y1, x2, y2 in boxes
            ]
            ann = sq.copy()
            for x1, y1, x2, y2, name in scaled_boxes:
                color = LESION_COLORS.get(name, COLORS["white"])
                cv2.rectangle(ann, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
                cv2.putText(ann, name, (int(x1), max(10, int(y1) - 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
            first_frames.append(sq)
            last_frames.append(ann)
            has = len(scaled_boxes) > 0
            flags.append(has)
            if has:
                gt_frames.append(ann)
        if not gt_frames:
            gt_frames = last_frames[:5]
        meta = {
            "task": "retinal lesion bounding-box detection (4 classes)",
            "dataset": "DDR lesion_detection",
            "fps_source": "manual (case D circular loop)",
            "source_split": "train",
        }
        d = write_task("M-043", "ddr_lesion_bbox_detection", task_idx,
                       first_frames[0], last_frames[0],
                       first_frames, last_frames, gt_frames,
                       PROMPT_044, meta, 8)
        print(f"  wrote {d}")


# -------------------- M-041 DR grading --------------------

PROMPT_045 = (
    "This is a color fundus photograph from the DDR dataset. Classify the "
    "severity of diabetic retinopathy on the international 5-grade scale: "
    "0 = no DR, 1 = mild NPDR, 2 = moderate NPDR, 3 = severe NPDR, 4 = "
    "proliferative DR. Overlay the predicted grade and a severity bar on "
    "every frame."
)


def annotate_grade(img_bgr, grade: int):
    out = img_bgr.copy()
    h, w = out.shape[:2]
    label = f"DR grade: {grade} ({DR_GRADES[grade]})"
    cv2.rectangle(out, (10, 10), (w - 10, 48), (0, 0, 0), -1)
    cv2.putText(out, label, (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["yellow"], 1, cv2.LINE_AA)
    # Severity bar at bottom
    cv2.rectangle(out, (18, h - 30), (w - 18, h - 14), (40, 40, 40), -1)
    frac = grade / 4.0
    color = COLORS["green"] if grade == 0 else (
        COLORS["yellow"] if grade <= 2 else COLORS["red"])
    cv2.rectangle(out, (18, h - 30), (18 + int((w - 36) * frac), h - 14), color, -1)
    return out


def run_m045():
    samples = [
        ("007-0004-000.jpg", 0),
        ("007-0033-000.jpg", 1),
        ("007-0045-000.jpg", 2),
        ("007-1723-000.jpg", 3),
        ("007-3122-100.jpg", 4),
    ]
    loaded = []
    for name, grade in samples:
        img = cv2.imread(str(ROOT / "DR_grading" / "train" / name), cv2.IMREAD_COLOR)
        loaded.append((img, grade))

    for task_idx in range(2):
        ordered = loaded[task_idx:] + loaded[:task_idx]
        first_frames, last_frames, gt_frames, flags = [], [], [], []
        # each sample gets 2 frames so loop becomes 10
        for img, grade in ordered * 2:
            sq = fit_square(img, 512)
            ann = annotate_grade(sq, grade)
            first_frames.append(sq)
            last_frames.append(ann)
            has = grade > 0
            flags.append(has)
            if has:
                gt_frames.append(ann)
        if not gt_frames:
            gt_frames = last_frames[:5]
        meta = {
            "task": "diabetic retinopathy 5-grade severity classification",
            "dataset": "DDR DR_grading",
            "grades_used": list(DR_GRADES.keys()),
            "fps_source": "manual (case D circular loop)",
            "source_split": "train",
        }
        d = write_task("M-044", "ddr_dr_severity_grading", task_idx,
                       first_frames[0], last_frames[0],
                       first_frames, last_frames, gt_frames,
                       PROMPT_045, meta, 6)
        print(f"  wrote {d}")


def main():
    print("=== M-041 DDR lesion segmentation ===")
    run_m043()
    print("=== M-041 DDR lesion bbox detection ===")
    run_m044()
    print("=== M-041 DDR severity grading ===")
    run_m045()


if __name__ == "__main__":
    main()
