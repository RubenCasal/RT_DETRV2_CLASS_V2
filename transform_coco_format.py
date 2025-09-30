#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import json

# === CONFIG ===
SRC  = Path("/home/rcasal/Desktop/projects/PtFue/detvr2_training/RT_DETRV2_CLASS_Refactor/pedestrian_dataset2").resolve()
DEST = Path("/home/rcasal/Desktop/projects/PtFue/detvr2_training/RT_DETRV2_CLASS_Refactor/pedestrian_dataset(COCO)").resolve()
CLASS_NAMES: Optional[List[str]] = None      # p.ej.: ["pedestrian"] si quieres fijarlas
IMG_EXTS: Set[str] = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}
SPLITS = ("train","val")
SKIP_IF_NO_LABEL = True

# === Utils ===
def parse_yolo_line(ln: str):
    p = ln.strip().split()
    if len(p) < 5: return None
    try: return int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4])
    except: return None

def yolo_bbox_to_coco_abs(cx, cy, w, h, W, H):
    x = (cx - w/2.0)*W; y = (cy - h/2.0)*H
    bw = w*W; bh = h*H
    # clip
    if x < 0: bw += x; x = 0
    if y < 0: bh += y; y = 0
    if x + bw > W: bw = W - x
    if y + bh > H: bh = H - y
    return float(max(0,x)), float(max(0,y)), float(max(0,bw)), float(max(0,bh))

def list_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])

def label_path_for(im: Path, img_root: Path, lbl_root: Path) -> Path:
    return (lbl_root / im.relative_to(img_root)).with_suffix(".txt")

# === COCO por split ===
def build_categories(class_names: Optional[List[str]], classes_seen: Set[int]) -> List[Dict]:
    if class_names is None:
        max_id = max(classes_seen) if classes_seen else -1
        class_names = [f"class_{i}" for i in range(max_id + 1)]
    return [{"id": i, "name": n} for i, n in enumerate(class_names)]

def collect_classes_seen(img_root: Path, lbl_root: Path) -> Set[int]:
    seen = set()
    for im in list_images(img_root):
        lb = label_path_for(im, img_root, lbl_root)
        if not lb.exists(): 
            if SKIP_IF_NO_LABEL: continue
            else:                continue
        for ln in lb.read_text(encoding="utf-8", errors="ignore").splitlines():
            r = parse_yolo_line(ln)
            if r: seen.add(r[0])
    return seen

def make_coco_split(split: str, class_names: Optional[List[str]]):
    img_root = SRC/"images"/split
    lbl_root = SRC/"labels"/split
    if not img_root.exists() or not lbl_root.exists():
        raise SystemExit(f"[ERR] faltan carpetas para split '{split}'")

    from PIL import Image

    # 1) inferir categorías si no se fijan
    classes_seen = collect_classes_seen(img_root, lbl_root)
    cats = build_categories(class_names, classes_seen)
    valid_cat_ids = {c["id"] for c in cats}

    # 2) construir COCO
    images, annotations = [], []
    img_id, ann_id = 1, 1
    imgs = list_images(img_root)
    if not imgs: raise SystemExit(f"[ERR] sin imágenes en {img_root}")

    for im in imgs:
        rel_file = im.relative_to(img_root).as_posix()
        with Image.open(im) as I: W, H = I.size
        images.append({"id": img_id, "file_name": rel_file, "width": W, "height": H})

        lb = label_path_for(im, img_root, lbl_root)
        if lb.exists():
            for ln in lb.read_text(encoding="utf-8", errors="ignore").splitlines():
                r = parse_yolo_line(ln)
                if not r: continue
                cid, cx, cy, bw, bh = r
                if cid not in valid_cat_ids: 
                    # si CLASS_NAMES fija y hay ids fuera de rango -> ignora
                    continue
                x,y,w,h = yolo_bbox_to_coco_abs(cx,cy,bw,bh,W,H)
                if w <= 0 or h <= 0: 
                    continue
                annotations.append({
                    "id": ann_id, "image_id": img_id, "category_id": cid,
                    "bbox": [x,y,w,h], "area": w*h, "iscrowd": 0
                })
                ann_id += 1
        elif not SKIP_IF_NO_LABEL:
            # nada que añadir, pero la imagen queda en COCO sin anns
            pass
        else:
            # saltada si SKIP_IF_NO_LABEL True (no llegas aquí porque no la añadimos)
            pass

        img_id += 1

    out = {"images": images, "annotations": annotations, "categories": cats}
    (DEST/"annotations").mkdir(parents=True, exist_ok=True)
    out_path = DEST/"annotations"/f"{split}.json"
    out_path.write_text(json.dumps(out), encoding="utf-8")
    print(f"[OK] {split}: imgs={len(images)} anns={len(annotations)} → {out_path}")

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        make_coco_split(s, CLASS_NAMES)

if __name__ == "__main__":
    main()
