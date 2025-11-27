# Auto-BUSAM
This repository contains the implementation of the following paper:
> **Auto-BUSAM: Low-contrast Breast Ultrasound Image Segmentation with Fine-tuned SAM and Auto Prompting**<br>

## Overview
<img src="figures/overview.png" />  
 Our method incorporates a learnable Prompt Generator module, which automatically generates bounding box prompts that guide SAM's focus to relevant regions within ultrasound images, minimizing the reliance on expert knowledge and reducing manual effort.  To further enhance SAM's segmentation capabilities,  a Low-Rank Adaptation (LoRA) module is specifically introduced to address challenges such as low contrast, variable tumor shapes, and blurred boundaries by filtering out noise and focusing on critical image features. Auto-BUSAM consistently outperforms leading segmentation models, which underscores Auto-BUSAM's potential to advance automated breast ultrasound image segmentation. 


## Compare with other methods
<img src="figures\comparison.png" />  
## Dataset

We use two datasets for training and evaluation:
[BUSI](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) &
[Dataset B](https://helward.mmu.ac.uk/STAFF/m.yap/dataset.php)

### Dataset Structure

Download the datasets from the provided sources and place them in `/path/data` with the following structure:

```text
data/
├── benign/
│   ├── benign (1).png          # Original image
│   └── benign (1)_mask.png     # Binary mask
└── malignant/
    ├── malignant (1).png
    └── malignant (1)_mask.png

## Dataset Loader
The dataset loader (dataset.py) automatically excludes multi-tumor images by checking connected components in masks, ensuring single-ROI focus (as per the paper's Section 3.3). For YOLOv8 prompt generation, labels are derived from the masks during preparation.


## Steps for Training and Model Fine-Tuning
## Run the preparation script to convert masks to YOLO-format labels (.txt files with normalized xywh for class 0 "tumor") and split into train/val (80/20 ratio).

python prepare_yolo_labels.py --root_dir /path/data

Output: This creates the following folders under /path/data/:


images/
├── train/
├── val/
labels/
├── train/
└── val/


## Step 2: Create busi.yaml
Save the following YAML configuration in the home directory as busi.yaml for YOLO training.


path: /path/data
train: images/train
val: images/val
nc: 1
names: [tumor]

## Step 3: Train YOLO Model
Train YOLOv8 on the prepared BUSI dataset to detect tumors and generate bounding box prompts.

python train_yolo_busi.py

## Step 4: Generate All Prompts
Generate bounding box prompts using the trained YOLO model.


python gen_all_yolo_prompts.py \
  --data_root /path/data \
  --weights /path/runs/detect/busi_yolo/weights/best.pt \
  --out_json /path/all_prompts.json \
  --imgsz 256 \
  --conf 0.25 \
  --iou 0.45

## Fine-Tuning SAM Model with LoRA
Step 5: Run Fine-Tuning
Fine-tune the SAM model using LoRA with the generated prompts.

python main.py \
  --data_root /path/data \
  --sam_ckpt /path/sam_vit_b_01ec64.pth \
  --epochs 150 \
  --batch_size 8 \
  --accum_steps 4 \
  --precision fp16 \
  --splits 4 \
  --seed 42 \
  --outdir_base /path/checkpoints/autobusam \
  --prompts_dir /path/to/prompts \
  --lora_rank 4



## Acknowledgement
We appreciate the developers of [Segment Anything Model](https://github.com/facebookresearch/segment-anything) and [YOLO](https://github.com/ultralytics/ultralytics). The code of Auto-BUSAM is built upon [BLO-SAM](https://github.com/importZL/BLO-SAM/tree/master?tab=readme-ov-file) and [SAM LoRA](https://github.com/JamesQFreeman/Sam_LoRA), and we express our gratitude to these awesome projects.

