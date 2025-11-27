
from __future__ import annotations
import os
import json
import argparse
from typing import Tuple, Optional, List
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import SGD
from torchvision import transforms as T
from sklearn.model_selection import KFold
from torch.amp import autocast, GradScaler
import time

# Removed: from thop import profile, clever_format

from dataset import BusDataset
from loss import CombinedLoss
from model import AutoBUSAM


try:
    from dataset import BusDataset
    from loss import CombinedLoss
    from model import AutoBUSAM
    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.data import Subset
except ImportError:
    print("Warning: Using dummy classes for BusDataset, CombinedLoss, and AutoBUSAM.")

    class BusDataset:
        def __init__(self, root, categories, img_transform=None, mask_transform=None, is_train=False, indices=None):
            self._length = 100
            if isinstance(indices, list):
                self._length = len(indices)
            elif not os.path.isdir(root):
                self._length = 100

        def __len__(self):
            return self._length

        def __getitem__(self, idx):
            image = torch.randn(3, 256, 256)
            mask = (torch.rand(1, 256, 256) > 0.5).float()
            path = f"img_{idx}.png"
            return image, mask, path

    class CombinedLoss(torch.nn.Module):
        def __init__(self, alpha=0.7):
            super().__init__()
            self.bce = torch.nn.BCEWithLogitsLoss()

        def forward(self, logits, target):
            return self.bce(logits, target)

        def dice_loss(self, logits, target):
            return torch.abs(logits - target).mean()

    class AutoBUSAM(torch.nn.Module):
        def __init__(self, sam_ckpt, lora_rank=4, lora_alpha=4):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 1, kernel_size=3, padding=1)

        def forward(self, images, boxes_norm):
            return self.conv(images)

        def get_learnable_parameters(self):
            return list(self.parameters())

    class DistributedSampler:
        def __init__(self, dataset, num_replicas, rank, shuffle=True):
            pass

        def set_epoch(self, epoch):
            pass

    class Subset:
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]

    def try_build_bare(root, categories):
        return BusDataset(root, categories=categories)

    def batch_boxes_from_masks(masks: torch.Tensor) -> torch.Tensor:
        batch_size = masks.size(0)
        return torch.tensor([[0.0, 0.0, 1.0, 1.0]] * batch_size, dtype=torch.float32, device=masks.device)
# --------------------------------------------------------------------------


# --------- Args / Dist ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Auto-BUSAM train")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--sam_ckpt", type=str, required=True)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.004)
    p.add_argument("--splits", type=int, default=4)
    p.add_argument("--lora_rank", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=8, help="LoRA alpha scaling parameter")
    p.add_argument("--alpha", type=float, default=0.7, help="Weight for BCE loss in CombinedLoss")
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--min_lr_factor", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    p.set_defaults(pin_memory=True)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_only", action="store_true", help="Run evaluation only, skipping training")
    p.add_argument("--fold_to_run", type=int, default=None, help="Run only a specific fold (1-based index)")
    p.add_argument("--standalone", action="store_true")
    p.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--outdir", type=str, default=".")
    p.add_argument("--train_prompts_json", type=str, required=True,
                   help="Path to JSON {img_path: [x0,y0,x1,y1]} for training")
    p.add_argument("--val_prompts_json", type=str, required=True,
                   help="Path to JSON {x0,y0,x1,y1]} for validation")
    p.add_argument("--debug_shapes", action="store_true",
                   help="Enable debug printing of tensor shapes for the first batch.")
    return p.parse_args()


# Initialize Distributed Data Parallel
def init_distributed(standalone: bool) -> Tuple[int, int, int]:
    if standalone:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        rank = int(os.environ.get("RANK", "0"))
        local = int(os.environ.get("LOCAL_RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))
    else:
        rank, local, world = 0, 0, 1
    return rank, world, local


def is_main(rank: int) -> bool:
    return rank == 0


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(local: int) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        return torch.device("cuda", local)
    return torch.device("cpu")


# Learning Rate Scheduler
def linear_warmup_then_decay(epoch: int, total_epochs: int, warmup_epochs: int, min_factor: float) -> float:
    if epoch < warmup_epochs:
        return max(1e-8, (epoch + 1) / max(1, warmup_epochs))
    t = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
    return float((1.0 - t) + t * min_factor)


def make_scheduler(opt: torch.optim.Optimizer, total: int, warm: int, minf: float):
    def lr_lambda(ep):
        return linear_warmup_then_decay(ep, total, warm, minf)

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


# Dataset Builders (no augmentations)
def build_img_tfms():
    return T.Compose([
        T.Resize((256, 256), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
    ])


def build_mask_tfms():
    return T.Compose([
        T.Resize((256, 256), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(),
    ])


def build_reference_dataset(root_dir, categories, img_tf, mask_tf):
    return BusDataset(root_dir, categories, is_train=False, img_transform=img_tf, mask_transform=mask_tf)


def build_split_datasets(root_dir, categories, tr_idx, va_idx):
    img_tf = build_img_tfms()
    mask_tf = build_mask_tfms()
    train_set = BusDataset(root_dir, categories, is_train=True, indices=tr_idx,
                           img_transform=img_tf, mask_transform=mask_tf)
    val_set = BusDataset(root_dir, categories, is_train=False, indices=va_idx,
                         img_transform=img_tf, mask_transform=mask_tf)
    return train_set, val_set


# Helper to get boxes from JSON (robust handling)
def get_boxes_from_json(paths, json_dict):
    boxes = []
    for p in paths:
        filename = os.path.basename(p)
        box = json_dict.get(filename, [0.0, 0.0, 1.0, 1.0])

        if isinstance(box, list) and len(box) > 0 and isinstance(box[0], list):
            box = box[0]

        if not (isinstance(box, list) and len(box) == 4):
            box = [0.0, 0.0, 1.0, 1.0]

        boxes.append(box)

    return torch.tensor(boxes, dtype=torch.float32)


# Training Loop
def train_one_epoch(model, loader, optimizer, scaler, criterion, device, precision,
                    accum_steps, train_json, ep, total_epochs, args):
    model.train()
    total_loss, total_dice = 0.0, 0.0
    num_batches = 0
    optimizer.zero_grad()
    dtype = torch.float16 if precision == "fp16" else (
        torch.bfloat16 if precision == "bf16" else torch.float32
    )

    current_rank = dist.get_rank() if dist.is_initialized() else 0

    for batch_idx, (imgs, msks, paths) in enumerate(loader):
        imgs, msks = imgs.to(device), msks.to(device)
        boxes_norm = get_boxes_from_json(paths, train_json).to(device)

        if args.debug_shapes and batch_idx == 0 and is_main(current_rank):
            print("\n--- DEBUG: Train First Batch Shapes ---")
            print(f"| Batch Size: {imgs.shape[0]}")
            print(f"| Image Input Shape (B, C, H, W): {imgs.shape}")
            print(f"| Mask Target Shape (B, 1, H, W): {msks.shape}")
            print(f"| Box Prompt Shape (B, 4): {boxes_norm.shape}")
            print(f"| Logits dtype (Pre-autocast): {imgs.dtype}")
            print("---------------------------------------")

        with autocast("cuda", dtype=dtype, enabled=precision != "fp32"):
            logits = model(imgs, boxes_norm=boxes_norm)
            loss = criterion(logits, msks)

        if args.debug_shapes and batch_idx == 0 and is_main(current_rank):
            print(f"| Logits Output Shape (B, 1, H, W): {logits.shape}")
            print(f"| Logits Output dtype (Post-autocast): {logits.dtype}")
            print(f"| Loss Value (First Batch): {loss.item():.4f}")
            print("---------------------------------------")

        scaler.scale(loss / accum_steps).backward()

        if (batch_idx + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()
        try:
            dice = 1 - criterion.dice_loss(logits, msks).item()
        except AttributeError:
            dice = 0.0
        total_dice += dice
        num_batches += 1

    return total_loss / num_batches, total_dice / num_batches


# Validation Loop with Metrics
def validate(model, loader, criterion, device, precision, val_json, args):
    model.eval()
    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0
    total_precision, total_recall, total_specificity = 0.0, 0.0, 0.0
    num_batches = 0
    dtype = torch.float16 if precision == "fp16" else (
        torch.bfloat16 if precision == "bf16" else torch.float32
    )

    current_rank = dist.get_rank() if dist.is_initialized() else 0

    with torch.no_grad():
        for batch_idx, (imgs, msks, paths) in enumerate(loader):
            imgs, msks = imgs.to(device), msks.to(device)
            boxes_norm = get_boxes_from_json(paths, val_json).to(device)

            if args.debug_shapes and batch_idx == 0 and is_main(current_rank):
                print("\n--- DEBUG: Validation First Batch Shapes ---")
                print(f"| Batch Size: {imgs.shape[0]}")
                print(f"| Image Input Shape (B, C, H, W): {imgs.shape}")
                print(f"| Mask Target Shape (B, 1, H, W): {msks.shape}")
                print(f"| Box Prompt Shape (B, 4): {boxes_norm.shape}")
                print("------------------------------------------")

            with autocast("cuda", dtype=dtype, enabled=precision != "fp32"):
                logits = model(imgs, boxes_norm=boxes_norm)
                loss = criterion(logits, msks)

            if args.debug_shapes and batch_idx == 0 and is_main(current_rank):
                print(f"| Logits Output Shape (B, 1, H, W): {logits.shape}")
                print(f"| Logits Output dtype: {logits.dtype}")
                print(f"| Loss Value (First Batch): {loss.item():.4f}")
                print("------------------------------------------")

            total_loss += loss.item()
            preds = torch.sigmoid(logits) > 0.5

            tp = (preds * msks).sum(dim=(2, 3))
            fp = (preds * (1 - msks)).sum(dim=(2, 3))
            fn = ((~preds) * msks).sum(dim=(2, 3))
            tn = ((~preds) * (1 - msks)).sum(dim=(2, 3))

            dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
            iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
            precision = (tp + 1e-6) / (tp + fp + 1e-6)
            recall = (tp + 1e-6) / (tp + fn + 1e-6)
            specificity = (tn + 1e-6) / (tn + fp + 1e-6)

            total_dice += dice.mean().item()
            total_iou += iou.mean().item()
            total_precision += precision.mean().item()
            total_recall += recall.mean().item()
            total_specificity += specificity.mean().item()
            num_batches += 1

    return (
        total_loss / num_batches,
        total_dice / num_batches,
        total_iou / num_batches,
        total_precision / num_batches,
        total_recall / num_batches,
        total_specificity / num_batches,
    )


def count_parameters(model):
    """Counts total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def main():
    # START GLOBAL TIMING
    global_start_time = time.time()

    args = parse_args()
    rank, world, local = init_distributed(args.standalone)
    device = get_device(local)

    if is_main(rank):
        print(f"DDP: world={world} rank={rank} local_rank={local}")
        print(f"Device: {device}")

    if not os.path.exists(args.sam_ckpt):
        raise FileNotFoundError(f"SAM checkpoint not found: {args.sam_ckpt}")
    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(f"Data dir not found: {args.data_root}")

    # ?? Ensure output directory exists (fixes your earlier RuntimeError)
    if is_main(rank):
        os.makedirs(args.outdir, exist_ok=True)

    # Load YOLO boxes JSON
    train_json = json.load(open(args.train_prompts_json))
    val_json = json.load(open(args.val_prompts_json))

    torch.backends.cudnn.benchmark = True
    set_seed(args.seed)

    # Reference dataset
    img_tf = build_img_tfms()
    mask_tf = build_mask_tfms()
    ref_ds = BusDataset(
        args.data_root,
        ("benign", "malignant"),
        is_train=False,
        img_transform=img_tf,
        mask_transform=mask_tf,
    )
    kf = KFold(n_splits=args.splits, shuffle=True, random_state=args.seed)

    fold_best_losses, fold_best_dices = [], []
    fold_best_ious, fold_best_precisions = [], []
    fold_best_recalls, fold_best_specificities = [], []

    total_time_seconds = 0

    # DEBUG: Print general configuration once
    if is_main(rank):
        print("\n--- GENERAL CONFIGURATION CHECK ---")
        print(f"| Epochs: {args.epochs}, Folds: {args.splits}, Batch Size (per device): {args.batch_size}")
        print(f"| Accumulation Steps: {args.accum_steps}, Effective Global Batch Size: {args.batch_size * args.accum_steps * world}")
        print(f"| Precision: {args.precision}, LoRA Rank/Alpha: {args.lora_rank}/{args.lora_alpha}")
        print(f"| Debug Shapes Enabled: {args.debug_shapes}")
        print("-----------------------------------")

    # Flag to track if parameters were computed (only done once)
    computed_params = False

    for fold, (tr_idx, va_idx) in enumerate(kf.split(ref_ds), start=1):
        if args.fold_to_run is not None and fold != args.fold_to_run:
            continue

        # START FOLD TIMING
        fold_start_time = time.time()

        train_set = BusDataset(
            args.data_root,
            ("benign", "malignant"),
            is_train=True,
            indices=tr_idx,
            img_transform=img_tf,
            mask_transform=mask_tf,
        )
        val_set = BusDataset(
            args.data_root,
            ("benign", "malignant"),
            is_train=False,
            indices=va_idx,
            img_transform=img_tf,
            mask_transform=mask_tf,
        )

        train_sampler = DistributedSampler(train_set, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
        val_sampler = DistributedSampler(val_set, num_replicas=world, rank=rank, shuffle=False) if world > 1 else None

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )

        base = AutoBUSAM(args.sam_ckpt, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha)
        base = base.to(device)

        # COMPUTATIONAL COST CALCULATION (Main process only - run once)
        if is_main(rank) and not computed_params:
            total_params, trainable_params = count_parameters(base)
            computed_params = True

            print(f"\n{'='*15} Fold {fold}/{args.splits} Computational Metrics {'='*15}")
            print(f"| Total Parameters: {total_params:,}")
            print(f"| Trainable Parameters (LoRA): {trainable_params:,}")
            print("| Single Forward Pass FLOPs: N/A (Calculation disabled due to instability)")
            print(f"{'='*58}")

        if world > 1:
            model = torch.nn.parallel.DistributedDataParallel(
                base, device_ids=[local], output_device=local, find_unused_parameters=True
            )
        else:
            model = base

        optimizer = SGD(base.get_learnable_parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0)
        scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs, args.min_lr_factor)

        criterion = CombinedLoss(alpha=args.alpha)

        scaler = GradScaler('cuda', enabled=(args.precision != "fp32" and device.type == "cuda"))

        best_dice, best_loss = -1.0, float("inf")
        best_iou, best_precision, best_recall, best_specificity = -1.0, -1.0, -1.0, -1.0
        best_epoch = -1

        if not args.eval_only:
            for ep in range(args.epochs):
                # START EPOCH TIMING
                start_epoch_time = time.time()

                if train_sampler:
                    train_sampler.set_epoch(ep)

                tr_loss, tr_dice = train_one_epoch(
                    model, train_loader, optimizer, scaler, criterion,
                    device, args.precision, args.accum_steps,
                    train_json, ep, args.epochs, args
                )
                va_loss, va_dice, va_iou, va_precision, va_recall, va_specificity = validate(
                    model, val_loader, criterion, device, args.precision, val_json, args
                )
                scheduler.step()

                epoch_time = time.time() - start_epoch_time  # END EPOCH TIMING

                if is_main(rank):
                    print(
                        f"Epoch {ep + 1}/{args.epochs} | "
                        f"TrainLoss: {tr_loss:.4f} | TrainDice: {tr_dice:.4f} | "
                        f"ValLoss: {va_loss:.4f} | ValDice: {va_dice:.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.5f} | Time: {epoch_time:.2f}s"
                    )

                if va_dice > best_dice + 1e-6:
                    best_dice, best_loss = va_dice, va_loss
                    best_iou, best_precision, best_recall, best_specificity = (
                        va_iou, va_precision, va_recall, va_specificity
                    )
                    best_epoch = ep + 1

                    if is_main(rank):
                        ckpt = os.path.join(args.outdir, f"autobusam_fold{fold}_best.pth")
                        state_to_save = base.state_dict() if world == 1 else model.module.state_dict()
                        torch.save(state_to_save, ckpt)
                        print(f"  -> New best VaDice: {best_dice:.4f} @ Epoch {best_epoch}. Checkpoint saved.")

        # END FOLD TIMING
        fold_end_time = time.time()
        fold_time = fold_end_time - fold_start_time
        total_time_seconds += fold_time

        if is_main(rank):
            print(f"\n--- Fold {fold} Summary ---")
            print(f"Best ValDice: {best_dice:.4f} @ Epoch {best_epoch}")
            print(f"Fold Training Time: {fold_time:.2f} seconds")

        else:
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

        fold_best_losses.append(best_loss)
        fold_best_dices.append(best_dice)
        fold_best_ious.append(best_iou)
        fold_best_precisions.append(best_precision)
        fold_best_recalls.append(best_recall)
        fold_best_specificities.append(best_specificity)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()  # Ensure all ranks sync before next fold/exit

    # Final summary on main process
    if is_main(rank):
        total_time_minutes = total_time_seconds / 60

        print("\n" + "=" * 70)
        print("                 FINAL COMPUTATIONAL & TIMING SUMMARY")
        print("=" * 70)

        if computed_params:
            print(f"Total Model Parameters (including frozen): {total_params:,}")
            print(f"Trainable Parameters (LoRA): {trainable_params:,}")

        print("Computational metrics (FLOPs) omitted due to instability.")
        print(f"Total Training Time (All Folds): {total_time_seconds:.2f} seconds ({total_time_minutes:.2f} minutes)")
        print("=" * 70)

        print("\n--- CV done ---")
        print("Best ValLoss per fold:", [f"{x:.4f}" for x in fold_best_losses])
        print("Best ValDice per fold:", [f"{x:.4f}" for x in fold_best_dices])
        print("Best ValIoU per fold:", [f"{x:.4f}" for x in fold_best_ious])
        print("Best ValPrecision per fold:", [f"{x:.4f}" for x in fold_best_precisions])
        print("Best ValRecall per fold:", [f"{x:.4f}" for x in fold_best_recalls])
        print("Best ValSpecificity per fold:", [f"{x:.4f}" for x in fold_best_specificities])

        if len(fold_best_dices) > 1:
            print("\n--- Averages ---")
            print(f"Avg ValLoss: {float(np.mean(fold_best_losses)):.4f}")
            print(f"Avg ValDice: {float(np.mean(fold_best_dices)):.4f}")
            print(f"Avg ValIoU: {float(np.mean(fold_best_ious)):.4f}")
            print(f"Avg ValPrecision: {float(np.mean(fold_best_precisions)):.4f}")
            print(f"Avg ValRecall: {float(np.mean(fold_best_recalls)):.4f}")
            print(f"Avg ValSpec: {float(np.mean(fold_best_specificities)):.4f}")
        elif len(fold_best_dices) == 1:
            print("\n--- Averages (single fold) ---")
            print(f"Avg ValLoss: {fold_best_losses[0]:.4f}")
            print(f"Avg ValDice: {fold_best_dices[0]:.4f}")
            print(f"Avg ValIoU: {fold_best_ious[0]:.4f}")
            print(f"Avg ValPrecision: {fold_best_precisions[0]:.4f}")
            print(f"Avg ValRecall: {fold_best_recalls[0]:.4f}")
            print(f"Avg ValSpec: {fold_best_specificities[0]:.4f}")

    # The final cleanup of the process group
    if dist.is_available() and dist.is_initialized() and args.standalone:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

