#import os
#from typing import Optional, Sequence, List
#import torch
#from torch.utils.data import Dataset
#from PIL import Image
#import numpy as np
#from torchvision import transforms as T
#from torchvision.transforms import InterpolationMode
#from torchvision.transforms import functional as TF
#
#class BusDataset(Dataset):
#    """
#    BUSI dataset loader.
#    - Guarantees every (image, mask) is [C,H,W] = [3,256,256] and [1,256,256]
#    - Separate train/val instances via `is_train` and optional `indices`
#    - Excludes multi-tumor images (len(msks) >1) for single ROI focus (paper Section 3.3)
#    - Applies online augmentations for train (contrast, blur, noise, flip) as per paper techniques for low-contrast robustness (page 2 refs: contrast enhancement, noise removal)
#    """
#    def __init__(
#        self,
#        root_dir: str,
#        categories: Sequence[str] = ("benign", "malignant"),
#        is_train: bool = True,
#        indices: Optional[Sequence[int]] = None,
#        img_transform: Optional[callable] = None,
#        mask_transform: Optional[callable] = None
#    ):
#        super().__init__()
#        self.root_dir = root_dir
#        self.categories = tuple(categories)
#        self.is_train = bool(is_train)
#        # Discover files (sorted for deterministic indexing)
#        all_imgs: List[str] = []
#        all_msks: List[List[str]] = []
#        for cat in self.categories:
#            cat_dir = os.path.join(self.root_dir, cat)
#            if not os.path.isdir(cat_dir):
#                print(f"Warning: Category directory not found: {cat_dir}")
#                continue
#            files = sorted(os.listdir(cat_dir))
#            img_to_msks = {}
#            for f in files:
#                if f.endswith(".png") and "_mask" not in f:
#                    base = f.split(".png")[0]
#                    img_to_msks[base] = []
#                elif "_mask" in f:
#                    base = f.split("_mask")[0]
#                    if base in img_to_msks:
#                        img_to_msks[base].append(f)
#            multi_count = sum(1 for msks in img_to_msks.values() if len(msks) > 1)
#            if multi_count > 0:
#                print(f"Found {multi_count} multi-tumor images in '{cat}' - excluding.")
#            for base, msk_files in img_to_msks.items():
#                if msk_files and len(msk_files) == 1:  # Exclude multi-tumor (len >1)
#                    img_path = os.path.join(cat_dir, f"{base}.png")
#                    msk_paths = [os.path.join(cat_dir, m) for m in msk_files]
#                    all_imgs.append(img_path)
#                    all_msks.append(msk_paths)
#        # Optional subset by indices (e.g., for KFold)
#        if indices is not None:
#            sel_imgs = [all_imgs[i] for i in indices]
#            sel_msks = [all_msks[i] for i in indices]
#            self.image_paths, self.mask_paths = sel_imgs, sel_msks
#        else:
#            self.image_paths, self.mask_paths = all_imgs, all_msks
#        # --------- Transforms (fixed output size, with online augs for train as per paper techniques) ---------
#        base_resize_img = T.Resize((256, 256), interpolation=InterpolationMode.BILINEAR)
#        base_resize_msk = T.Resize((256, 256), interpolation=InterpolationMode.NEAREST)
#        # Paper techniques: contrast enhancement, noise removal (as blur/noise aug), flip for symmetry
#        aug_tf = T.Compose([
#            T.RandomHorizontalFlip(p=0.5),
#            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),  # Noise removal as blur
#            T.ColorJitter(contrast=0.2),  # Contrast enhancement
#            T.Lambda(lambda tensor: tensor + torch.randn_like(tensor) * 0.02),  # Speckle noise simulation
#        ]) if self.is_train else T.Compose([])
#        # If no custom, use defaults
#        self.img_tf = img_transform if img_transform else T.Compose([
#            T.ToTensor(),
#            aug_tf,
#            base_resize_img,
#            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#        ])
#        self.msk_tf = mask_transform if mask_transform else T.Compose([
#            base_resize_msk,
#            T.ToTensor(),  # -> [1,256,256], 0..1
#        ])
#    def __len__(self) -> int:
#        return len(self.image_paths)
#    def __getitem__(self, idx: int):
#        img = Image.open(self.image_paths[idx]).convert("RGB")
#        # Merge multi-masks if >1 (but excluded above, so len=1)
#        msk_paths = self.mask_paths[idx]
#        msk_np = np.zeros((img.height, img.width), dtype=np.uint8)
#        for m_path in msk_paths:
#            msk_single = np.array(Image.open(m_path).convert("L"))
#            msk_np = np.maximum(msk_np, msk_single)
#        msk = Image.fromarray(msk_np)
#        img = self.img_tf(img)  # [3,256,256]
#        msk = self.msk_tf(msk)  # [1,256,256]
#        msk = (msk > 0.5).float()  # binarize
#        return img, msk, self.image_paths[idx]  # Return image path for YOLO JSON lookup
#    


import os
from typing import Optional, Sequence, List
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
import random

class MinMaxNormalize(object):
    def __call__(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val - min_val > 1e-5:
            tensor = (tensor - min_val) / (max_val - min_val)
        return tensor

class BusDataset(Dataset):
    """
    BUSI dataset loader.
    - Guarantees every (image, mask) is [C,H,W] = [3,256,256] and [1,256,256]
    - Separate train/val instances via `is_train` and optional `indices`
    - Excludes multi-tumor images (len(msks) >1) for single ROI focus (paper Section 3.3)
    - Applies online 6-fold augmentations for train (normalization, cropping, flipping, noise, blurring, contrast) as per paper techniques for low-contrast robustness (page 2 refs: contrast enhancement, noise removal)
    """
    def __init__(
        self,
        root_dir: str,
        categories: Sequence[str] = ("benign", "malignant"),
        is_train: bool = True,
        indices: Optional[Sequence[int]] = None,
        img_transform: Optional[callable] = None,
        mask_transform: Optional[callable] = None
    ):
        super().__init__()
        self.root_dir = root_dir
        self.categories = tuple(categories)
        self.is_train = bool(is_train)
        # Discover files (sorted for deterministic indexing)
        all_imgs: List[str] = []
        all_msks: List[List[str]] = []
        for cat in self.categories:
            cat_dir = os.path.join(self.root_dir, cat)
            if not os.path.isdir(cat_dir):
                print(f"Warning: Category directory not found: {cat_dir}")
                continue
            files = sorted(os.listdir(cat_dir))
            img_to_msks = {}
            for f in files:
                if f.endswith(".png") and "_mask" not in f:
                    base = f.split(".png")[0]
                    img_to_msks[base] = []
                elif "_mask" in f:
                    base = f.split("_mask")[0]
                    if base in img_to_msks:
                        img_to_msks[base].append(f)
            multi_count = sum(1 for msks in img_to_msks.values() if len(msks) > 1)
            if multi_count > 0:
                print(f"Found {multi_count} multi-tumor images in '{cat}' - excluding.")
            for base, msk_files in img_to_msks.items():
                if msk_files and len(msk_files) == 1: # Exclude multi-tumor (len >1)
                    img_path = os.path.join(cat_dir, f"{base}.png")
                    msk_paths = [os.path.join(cat_dir, m) for m in msk_files]
                    all_imgs.append(img_path)
                    all_msks.append(msk_paths)
        # Optional subset by indices (e.g., for KFold)
        if indices is not None:
            sel_imgs = [all_imgs[i] for i in indices]
            sel_msks = [all_msks[i] for i in indices]
            self.image_paths, self.mask_paths = sel_imgs, sel_msks
        else:
            self.image_paths, self.mask_paths = all_imgs, all_msks
        # --------- Transforms (fixed output size, with online augs for train as per paper techniques) ---------
        base_resize_img = T.Resize((256, 256), interpolation=InterpolationMode.BILINEAR)
        base_resize_msk = T.Resize((256, 256), interpolation=InterpolationMode.NEAREST)
        # Paper techniques: normalization, cropping, flipping, noise addition, blurring, contrast adjustment
        aug_tf = T.Compose([
            T.Lambda(lambda img: TF.crop(img, random.randint(0, int(img.height * 0.05)), random.randint(0, int(img.width * 0.05)), int(img.height * random.uniform(0.8, 0.95)), int(img.width * random.uniform(0.8, 0.95)))) if random.random() < 0.5 else T.Lambda(lambda x: x),  # Random cropping with resize back in Compose
            T.RandomHorizontalFlip(p=0.5),
            T.Lambda(lambda tensor: tensor + torch.randn_like(tensor) * random.uniform(0.02, 0.06)) if random.random() < 0.5 else T.Lambda(lambda x: x),  # Noise addition
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)) if random.random() < 0.5 else T.Lambda(lambda x: x),  # Blurring
            T.ColorJitter(contrast=0.2) if random.random() < 0.5 else T.Lambda(lambda x: x),  # Contrast
            MinMaxNormalize() if random.random() < 0.5 else T.Lambda(lambda x: x),  # Normalization
        ]) if self.is_train else T.Compose([])  # No augmentation for validation
        # If no custom, use defaults
        self.img_tf = img_transform if img_transform else T.Compose([
            T.ToTensor(),
            aug_tf,
            base_resize_img,
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize to standard ImageNet mean/std
        ])
        self.msk_tf = mask_transform if mask_transform else T.Compose([
            base_resize_msk,
            T.ToTensor(), # -> [1,256,256], 0..1
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        # Merge multi-masks if >1 (but excluded above, so len=1)
        msk_paths = self.mask_paths[idx]
        msk_np = np.zeros((img.height, img.width), dtype=np.uint8)
        for m_path in msk_paths:
            msk_single_img = Image.open(m_path).convert("L").resize((img.width, img.height), Image.NEAREST)
            msk_single = np.array(msk_single_img)
            msk_np = np.maximum(msk_np, msk_single)
        msk = Image.fromarray(msk_np)
        img = self.img_tf(img) # [3,256,256]
        msk = self.msk_tf(msk) # [1,256,256]
        msk = (msk > 0.5).float() # binarize
        return img, msk, self.image_paths[idx] # Return image path for YOLO JSON lookup
