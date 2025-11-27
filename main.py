import os
import argparse
import subprocess
from typing import List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train Auto-BUSAM on all folds (wrapper around train.py)")

    # Core paths
    p.add_argument("--data_root", type=str, required=True,
                   help="Root of BUSI data (same as train.py)")
    p.add_argument("--sam_ckpt", type=str, required=True,
                   help="Path to sam_vit_b_01ec64.pth")
    p.add_argument("--prompts_dir", type=str, required=True,
                   help="Directory containing train_prompts_foldK.json / val_prompts_foldK.json")
    p.add_argument("--outdir_base", type=str, required=True,
                   help="Base directory for checkpoints; fold suffix will be added")

    # Training hyper-params (forwarded to train.py)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--accum_steps", type=int, default=4)
    p.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--splits", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora_rank", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.004)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--min_lr_factor", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.7)

    # Device for CUDA_VISIBLE_DEVICES
    p.add_argument("--cuda_device", type=str, default="0",
                   help="Value for CUDA_VISIBLE_DEVICES (e.g., '0', '1', '2')")

    # Optional flags forwarded to train.py
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--no_pin_memory", action="store_true",
                   help="Disable pin_memory in train.py")
    p.add_argument("--debug_shapes", action="store_true",
                   help="Enable --debug_shapes in train.py")

    return p.parse_args()


def build_cmd_for_fold(args: argparse.Namespace, fold: int) -> List[str]:
    """Build the train.py command for a given fold as a list of args."""
    train_json = os.path.join(args.prompts_dir, f"train_prompts_fold{fold}.json")
    val_json = os.path.join(args.prompts_dir, f"val_prompts_fold{fold}.json")
    outdir = f"{args.outdir_base}_fold{fold}"

    cmd = [
        "python", "train.py",
        "--data_root", args.data_root,
        "--sam_ckpt", args.sam_ckpt,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--accum_steps", str(args.accum_steps),
        "--precision", args.precision,
        "--splits", str(args.splits),
        "--fold_to_run", str(fold),
        "--seed", str(args.seed),
        "--outdir", outdir,
        "--train_prompts_json", train_json,
        "--val_prompts_json", val_json,
        "--lora_rank", str(args.lora_rank),
        "--lora_alpha", str(args.lora_alpha),
        "--lr", str(args.lr),
        "--warmup_epochs", str(args.warmup_epochs),
        "--min_lr_factor", str(args.min_lr_factor),
        "--alpha", str(args.alpha),
        "--num_workers", str(args.num_workers),
    ]

    if args.no_pin_memory:
        cmd.append("--no_pin_memory")
    else:
        cmd.append("--pin_memory")

    if args.debug_shapes:
        cmd.append("--debug_shapes")

    return cmd


def main():
    args = parse_args()

    # Basic checks
    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(f"data_root not found: {args.data_root}")
    if not os.path.isfile(args.sam_ckpt):
        raise FileNotFoundError(f"SAM checkpoint not found: {args.sam_ckpt}")
    if not os.path.isdir(args.prompts_dir):
        raise FileNotFoundError(f"prompts_dir not found: {args.prompts_dir}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    # Loop over folds
    for fold in range(1, args.splits + 1):
        cmd = build_cmd_for_fold(args, fold)

        # Run train.py for this fold
        result = subprocess.run(cmd, env=env)

        if result.returncode != 0:
            break

    print("\n=== Done running all requested folds ===")


if __name__ == "__main__":
    main()
