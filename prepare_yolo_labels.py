# -*- coding: utf-8 -*-
##!/usr/bin/env python
#"""
#prepare_yolo_labels.py
#
#Build YOLOv8-ready dataset from BUSI masks:
#
#- Use only SINGLE-tumor cases (exactly one *_mask per image).
#- Resize images to 256x256.
#- Split 80/20 into train/val.
#- For val: save only the original image + label.
#- For train: save original + 5 augmented versions
#  using combinations of:
#    1) normalization
#    2) cropping
#    3) flipping
#    4) noise addition
#    5) blurring
#    6) contrast adjustment
#
#Resulting structure:
#  root/
#    images/train/*.png
#    images/val/*.png
#    labels/train/*.txt
#    labels/val/*.txt
#
#Each .txt file in YOLO format: "0 cx cy w h"
#
#Usage:
#  python prepare_yolo_labels.py --root_dir /home/hussain-hu/Auto-BUSAMpg/data
#"""
#
#import os
#import random
#import argparse
#from typing import List, Tuple
#
#import numpy as np
#from PIL import Image, ImageEnhance, ImageFilter
#import torchvision.transforms.functional as TF
#
#
## ----------------- Core helpers ----------------- #
#
#def mask_to_yolo(mask_paths: List[str], img_width: int, img_height: int) -> str:
#    """
#    Merge multiple masks into one bounding box and convert to YOLO format.
#    Returns an empty string if no lesion is present.
#    """
#    combined = np.zeros((img_height, img_width), dtype=np.uint8)
#
#    for m_path in mask_paths:
#        mask_img = Image.open(m_path).convert("L").resize((img_width, img_height), Image.NEAREST)
#        mask = np.array(mask_img)
#        combined = np.maximum(combined, mask)
#
#    if combined.max() == 0:
#        return ""  # no lesion
#
#    rows, cols = np.where(combined > 0)
#    y_min, y_max = rows.min(), rows.max()
#    x_min, x_max = cols.min(), cols.max()
#
#    cx = (x_min + x_max) / 2.0 / img_width
#    cy = (y_min + y_max) / 2.0 / img_height
#    bw = (x_max - x_min) / float(img_width)
#    bh = (y_max - y_min) / float(img_height)
#
#    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
#
#
#def apply_augmentation(img: Image.Image, label: List[float], technique: str) -> Tuple[Image.Image, List[float]]:
#    """
#    Apply a single augmentation technique and adjust label if geometric.
#
#    label = [cx, cy, bw, bh] in normalized coordinates.
#    """
#    cx, cy, bw, bh = label
#    w, h = img.width, img.height
#
#    if technique == "normalization":
#        # Intensity normalization in [0, 255] range
#        arr = np.array(img).astype(np.float32)
#        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-5) * 255.0
#        img = Image.fromarray(arr.astype(np.uint8))
#        # label unchanged
#
#    elif technique == "cropping":
#        # Random crop (80–95% of original size), then resize back to 256x256
#        crop_scale = random.uniform(0.8, 0.95)
#        crop_w, crop_h = int(w * crop_scale), int(h * crop_scale)
#
#        max_left = max(0, w - crop_w)
#        max_top = max(0, h - crop_h)
#        left = random.randint(0, max_left) if max_left > 0 else 0
#        top = random.randint(0, max_top) if max_top > 0 else 0
#
#        # Convert normalized center/size to pixel coords
#        box_px_w = bw * w
#        box_px_h = bh * h
#        box_px_cx = cx * w
#        box_px_cy = cy * h
#
#        # Shift center into cropped coordinate system
#        box_px_cx_cropped = box_px_cx - left
#        box_px_cy_cropped = box_px_cy - top
#
#        # If center goes outside the crop, this aug is invalid
#        if not (0 < box_px_cx_cropped < crop_w and 0 < box_px_cy_cropped < crop_h):
#            return img, label  # keep image as-is & label unchanged (safer than dropping)
#
#        # Resize image to 256x256
#        img = TF.crop(img, top, left, crop_h, crop_w)
#        img = img.resize((256, 256), Image.BILINEAR)
#        new_w, new_h = img.width, img.height
#
#        # Scale box center/size to new image
#        cx_new = box_px_cx_cropped * (new_w / crop_w) / new_w
#        cy_new = box_px_cy_cropped * (new_h / crop_h) / new_h
#        bw_new = box_px_w * (new_w / w) / new_w
#        bh_new = box_px_h * (new_h / h) / new_h
#
#        label = [cx_new, cy_new, bw_new, bh_new]
#        w, h = new_w, new_h  # update
#
#    elif technique == "flipping":
#        # Horizontal flip
#        img = TF.hflip(img)
#        cx = 1.0 - cx
#        label = [cx, cy, bw, bh]
#
#    elif technique == "noise addition":
#        arr = np.array(img).astype(np.float32)
#        sigma = random.uniform(5, 15)
#        noise = np.random.normal(0, sigma, size=arr.shape)
#        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
#        img = Image.fromarray(arr)
#        # label unchanged
#
#    elif technique == "blurring":
#        radius = random.uniform(0.5, 1.5)
#        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
#        # label unchanged
#
#    elif technique == "contrast adjustment":
#        factor = random.uniform(0.8, 1.2)
#        img = ImageEnhance.Contrast(img).enhance(factor)
#        # label unchanged
#
#    # Clip label to [0,1] and ensure positive size
#    cx, cy, bw, bh = label
#    cx = min(max(cx, 0.0), 1.0)
#    cy = min(max(cy, 0.0), 1.0)
#    bw = max(min(bw, 1.0), 1e-6)
#    bh = max(min(bh, 1.0), 1e-6)
#
#    return img, [cx, cy, bw, bh]
#
#
#def collect_single_tumor_images(root_dir: str, categories: List[str]) -> List[Tuple[str, List[str]]]:
#    """
#    Scan benign/malignant folders and collect only images with exactly one *_mask.
#    Returns list of (image_path, [mask_paths]).
#    """
#    all_images = []
#
#    for cat in categories:
#        cat_dir = os.path.join(root_dir, cat)
#        if not os.path.isdir(cat_dir):
#            continue
#
#        files = sorted(os.listdir(cat_dir))
#        img_to_masks = {}
#
#        # First pass: register base names
#        for f in files:
#            if f.lower().endswith(".png") and "_mask" not in f.lower():
#                base = f.rsplit(".png", 1)[0]
#                img_to_masks[base] = []
#
#        # Second pass: attach masks
#        for f in files:
#            if "_mask" in f.lower():
#                base = f.rsplit("_mask", 1)[0]
#                if base in img_to_masks:
#                    img_to_masks[base].append(os.path.join(cat_dir, f))
#
#        # Keep only single-tumor
#        for base, msks in img_to_masks.items():
#            if len(msks) == 1:
#                img_path = os.path.join(cat_dir, f"{base}.png")
#                all_images.append((img_path, msks))
#
#    return all_images
#
#
#def prepare_yolo_labels(root_dir: str, categories: List[str] = ("benign", "malignant")):
#    """
#    Build YOLO train/val split with augmentations.
#    """
#    # Output dirs
#    os.makedirs(os.path.join(root_dir, "images/train"), exist_ok=True)
#    os.makedirs(os.path.join(root_dir, "images/val"), exist_ok=True)
#    os.makedirs(os.path.join(root_dir, "labels/train"), exist_ok=True)
#    os.makedirs(os.path.join(root_dir, "labels/val"), exist_ok=True)
#
#    all_images = collect_single_tumor_images(root_dir, list(categories))
#    print(f"Total single-tumor images found: {len(all_images)}")
#
#    # Shuffle and split 80/20
#    random.seed(42)
#    np.random.seed(42)
#    random.shuffle(all_images)
#
#    split_idx = int(0.8 * len(all_images))
#    train_images = all_images[:split_idx]
#    val_images = all_images[split_idx:]
#
#    print(f"Train images: {len(train_images)}")
#    print(f"Val images  : {len(val_images)}")
#
#    techniques = [
#        "normalization",
#        "cropping",
#        "flipping",
#        "noise addition",
#        "blurring",
#        "contrast adjustment",
#    ]
#
#    def process_split(images, split_type: str):
#        for img_path, msks in images:
#            orig_img = Image.open(img_path).convert("RGB").resize((256, 256), Image.BILINEAR)
#            label_str = mask_to_yolo(msks, orig_img.width, orig_img.height)
#            if not label_str:
#                continue  # skip no-lesion
#
#            base_name = os.path.basename(img_path).rsplit(".png", 1)[0]
#            cx, cy, bw, bh = map(float, label_str.strip().split()[1:])
#            base_label = [cx, cy, bw, bh]
#
#            if split_type == "val":
#                # Only original
#                label_path = os.path.join(root_dir, "labels", "val", f"{base_name}.txt")
#                with open(label_path, "w") as f:
#                    f.write(label_str)
#                img_dest = os.path.join(root_dir, "images", "val", f"{base_name}.png")
#                orig_img.save(img_dest)
#            else:
#                # Train: original + 5 augmented
#                # Original
#                label_path = os.path.join(root_dir, "labels", "train", f"{base_name}_orig.txt")
#                with open(label_path, "w") as f:
#                    f.write(label_str)
#                img_dest = os.path.join(root_dir, "images", "train", f"{base_name}_orig.png")
#                orig_img.save(img_dest)
#
#                # 5 augmented samples
#                aug_id = 0
#                for _ in range(5):
#                    img_aug = orig_img.copy()
#                    label_aug = base_label.copy()
#
#                    # Apply 2 random techniques per aug (paper: mixed aug)
#                    selected = random.sample(techniques, 2)
#                    for tech in selected:
#                        img_aug, label_aug = apply_augmentation(img_aug, label_aug, tech)
#
#                    cx_a, cy_a, bw_a, bh_a = label_aug
#                    # Filter obviously invalid boxes
#                    if bw_a <= 0 or bh_a <= 0 or not (0 <= cx_a <= 1 and 0 <= cy_a <= 1):
#                        continue
#
#                    new_label_str = f"0 {cx_a:.6f} {cy_a:.6f} {bw_a:.6f} {bh_a:.6f}\n"
#                    aug_base = f"{base_name}_aug{aug_id}"
#                    aug_id += 1
#
#                    lbl_path = os.path.join(root_dir, "labels", "train", f"{aug_base}.txt")
#                    img_path_out = os.path.join(root_dir, "images", "train", f"{aug_base}.png")
#
#                    with open(lbl_path, "w") as f:
#                        f.write(new_label_str)
#                    img_aug.save(img_path_out)
#
#    process_split(train_images, "train")
#    process_split(val_images, "val")
#    print("YOLO dataset construction complete.")
#
#
#def parse_args():
#    ap = argparse.ArgumentParser()
#    ap.add_argument("--root_dir", type=str, required=True,
#                    help="BUSI root directory (contains benign/, malignant/)")
#    return ap.parse_args()
#
#
#if __name__ == "__main__":
#    args = parse_args()
#    prepare_yolo_labels(args.root_dir)





import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import List
import random
import torchvision.transforms.functional as TF

def mask_to_yolo(mask_paths: List[str], img_width: int, img_height: int) -> str:
    """Merge multiple masks into one bounding box and convert to YOLO format."""
    combined_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    for m_path in mask_paths:
        mask_img = Image.open(m_path).convert("L").resize((img_width, img_height), Image.NEAREST)
        mask = np.array(mask_img)
        combined_mask = np.maximum(combined_mask, mask)
    if np.max(combined_mask) == 0:
        return ""  # No lesion
    rows, cols = np.where(combined_mask > 0)
    x_min, x_max = np.min(cols), np.max(cols)
    y_min, y_max = np.min(rows), np.max(rows)
    cx = (x_min + x_max) / 2 / img_width
    cy = (y_min + y_max) / 2 / img_height
    bw = (x_max - x_min) / img_width
    bh = (y_max - y_min) / img_height
    return f"0 {cx} {cy} {bw} {bh}\n"  # Class 0: tumor

def apply_augmentation(img: Image.Image, label: list, technique: str) -> tuple:
    """Apply a single augmentation technique and adjust label if geometric."""
    if technique == "normalization":
        # Min-max normalization (intensity)
        img_arr = np.array(img)
        img_arr = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min() + 1e-5) * 255
        img = Image.fromarray(img_arr.astype(np.uint8))
        # Label unchanged
    elif technique == "cropping":
        # Random crop to 80-95% size, centered-ish
        crop_scale = random.uniform(0.8, 0.95)
        crop_w, crop_h = int(img.width * crop_scale), int(img.height * crop_scale)
        left = random.randint(0, img.width - crop_w)
        top = random.randint(0, img.height - crop_h)
        img = TF.crop(img, top, left, crop_h, crop_w)
        img = img.resize((256, 256))  # Resize back
        # Adjust box: [cx, cy, bw, bh]
        old_w, old_h = 256, 256
        label[0] = (label[0] * old_w - left) / crop_w
        label[1] = (label[1] * old_h - top) / crop_h
        label[2] /= crop_scale
        label[3] /= crop_scale
        # Clip to [0,1]
        label = [max(0, min(1, x)) for x in label]
    elif technique == "flipping":
        # Horizontal flip
        img = TF.hflip(img)
        # Adjust cx
        label[0] = 1 - label[0]
    elif technique == "noise addition":
        # Gaussian noise
        img_arr = np.array(img)
        noise = np.random.normal(0, random.uniform(5, 15), img_arr.shape)
        img_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr)
        # Label unchanged
    elif technique == "blurring":
        # Gaussian blur
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        # Label unchanged
    elif technique == "contrast adjustment":
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
        # Label unchanged
    return img, label

def prepare_yolo_labels(root_dir: str, categories: List[str] = ["benign", "malignant"]):
    """Scan dataset and create YOLO labels, splitting train/val (80/20). Apply six-fold aug to train."""
    os.makedirs(os.path.join(root_dir, "labels/train"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "labels/val"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "images/train"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "images/val"), exist_ok=True)
    
    all_images = []
    for cat in categories:
        cat_dir = os.path.join(root_dir, cat)  # Direct to benign/malignant
        if not os.path.isdir(cat_dir):
            continue
        files = os.listdir(cat_dir)
        img_to_masks = {}
        for f in files:
            if f.endswith(".png") and "_mask" not in f:
                base = f.split(".png")[0]
                img_to_masks[base] = []
            elif "_mask" in f:
                base = f.split("_mask")[0]
                if base in img_to_masks:
                    img_to_masks[base].append(os.path.join(cat_dir, f))
        for base, msks in img_to_masks.items():
            if len(msks) == 1:  # Single-tumor only, as per paper
                img_path = os.path.join(cat_dir, f"{base}.png")
                all_images.append((img_path, msks))
    
    # Shuffle and split 80/20
    np.random.seed(42)
    np.random.shuffle(all_images)
    split_idx = int(0.8 * len(all_images))
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    techniques = ["normalization", "cropping", "flipping", "noise addition", "blurring", "contrast adjustment"]

    def process_split(images, split_type):
        for img_path, msks in images:
            orig_img = Image.open(img_path).convert("RGB").resize((256, 256))
            label_str = mask_to_yolo(msks, orig_img.width, orig_img.height)
            if not label_str:
                continue  # Skip no-lesion
            base_name = os.path.basename(img_path).replace(".png", "")
            if split_type == "val":
                # Val: original only
                label_path = os.path.join(root_dir, f"labels/{split_type}", f"{base_name}.txt")
                with open(label_path, "w") as f:
                    f.write(label_str)
                img_dest = os.path.join(root_dir, f"images/{split_type}", os.path.basename(img_path))
                orig_img.save(img_dest)  # Save resized
            else:
                # Train: original + 5 augmented
                # Original
                label_path = os.path.join(root_dir, f"labels/train", f"{base_name}_orig.txt")
                with open(label_path, "w") as f:
                    f.write(label_str)
                img_dest = os.path.join(root_dir, f"images/train", f"{base_name}_orig.png")
                orig_img.save(img_dest)
                # 5 augmented
                for aug_idx in range(5):
                    img = orig_img.copy()
                    label = list(map(float, label_str.strip().split()[1:]))  # [cx, cy, bw, bh]
                    # Select two random techniques
                    selected = random.sample(techniques, 2)
                    for tech in selected:
                        img, label = apply_augmentation(img, label, tech)
                    # Skip if box invalid (e.g., cropped out)
                    if any(x <= 0 for x in label[2:]) or any(x > 1 or x < 0 for x in label):
                        continue
                    new_label_str = f"0 {' '.join(map(str, label))}\n"
                    aug_base = f"{base_name}_aug{aug_idx}"
                    label_path = os.path.join(root_dir, f"labels/train", f"{aug_base}.txt")
                    with open(label_path, "w") as f:
                        f.write(new_label_str)
                    img_dest = os.path.join(root_dir, f"images/train", f"{aug_base}.png")
                    img.save(img_dest)

    process_split(train_images, "train")
    process_split(val_images, "val")

if __name__ == "__main__":
    prepare_yolo_labels("/home/hussain-hu/Auto-BUSAMpg/data/")