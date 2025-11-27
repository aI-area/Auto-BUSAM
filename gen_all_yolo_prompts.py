#!/usr/bin/env python
"""
gen_all_yolo_prompts.py

Run YOLOv8 on ALL original BUSI images (benign + malignant) and produce:

  all_prompts.json:
    {
      "benign (10).png": [x0, y0, x1, y1],
      ...
    }

Boxes are:
- single, highest-confidence detection per image (if any)
- normalized XYXY in [0,1]
- fallback [0,0,1,1] if no detection

Usage:
CUDA_VISIBLE_DEVICES=2 python gen_all_yolo_prompts.py \
  --data_root /home/hussain-hu/Auto-BUSAMpg/data \
  --weights /home/hussain-hu/Auto-BUSAMpg/runs/detect/busi_yolo/weights/best.pt \
  --out_json /home/hussain-hu/Auto-BUSAMpg/all_prompts.json \
  --imgsz 256 \
  --conf 0.25 \
  --iou 0.45 \
  --device cuda:0
"""

import os
import json
import argparse
from typing import List, Dict, Tuple

import torch
from PIL import Image
import numpy as np
from ultralytics import YOLO


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--device", type=str, default="cuda:0")
    return ap.parse_args()


def list_busi_images(root: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    paths = []
    for cls in ["benign", "malignant"]:
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fn in sorted(os.listdir(cls_dir)):
            if not fn.lower().endswith(exts):
                continue
            if "mask" in fn.lower():
                continue
            paths.append(os.path.join(cls_dir, fn))
    return paths


def get_size(path: str) -> Tuple[int, int]:
    with Image.open(path) as im:
        w, h = im.size
    return w, h


def main():
    args = parse_args()

    print("=== YOLO Prompt Generation: START ===")
    print(f"Data root : {args.data_root}")
    print(f"Weights   : {args.weights}")
    print(f"Out JSON  : {args.out_json}")
    print("--------------------------------------")

    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(f"data_root does not exist: {args.data_root}")
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"YOLO weights not found: {args.weights}")

    device = torch.device(args.device if "cuda" in args.device or args.device == "cpu"
                          else f"cuda:{args.device}")

    model = YOLO(args.weights)
    model.to(device)

    img_paths = list_busi_images(args.data_root)
    n = len(img_paths)
    print(f"Found {n} images (benign + malignant).")

    prompts: Dict[str, List[float]] = {}
    num_det = 0
    num_fb = 0

    for i, p in enumerate(img_paths, start=1):
        base = os.path.basename(p)
        w, h = get_size(p)

        results = model(
            p,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )

        res = results[0]
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes
            confs = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            best_idx = int(confs.argmax())
            x0, y0, x1, y1 = xyxy[best_idx]

            # clip
            x0 = float(np.clip(x0, 0, w - 1))
            x1 = float(np.clip(x1, 0, w - 1))
            y0 = float(np.clip(y0, 0, h - 1))
            y1 = float(np.clip(y1, 0, h - 1))

            nx0, ny0 = x0 / w, y0 / h
            nx1, ny1 = x1 / w, y1 / h

            prompts[base] = [nx0, ny0, nx1, ny1]
            num_det += 1
        else:
            prompts[base] = [0.0, 0.0, 1.0, 1.0]
            num_fb += 1

        if i % 50 == 0 or i == n:
            print(f"[{i}/{n}] processed. Detections: {num_det}, Fallbacks: {num_fb}")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(prompts, f, indent=2)

    print("--------------------------------------")
    print(f"Total images        : {n}")
    print(f"With YOLO detection : {num_det}")
    print(f"Fallback full boxes : {num_fb}")
    print(f"Saved prompts JSON  : {args.out_json}")
    print("=== YOLO Prompt Generation: DONE ===")


if __name__ == "__main__":
    main()
