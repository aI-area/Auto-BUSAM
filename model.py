import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
from sam_lora_mask_decoder import LoRA_Sam

class AutoBUSAM(nn.Module):
    """
    - Builds SAM ViT-B
    - Applies LoRA to the mask decoder attention + unfreezes decoder heads
    - forward(img_256, boxes_norm=None) returns **RAW LOGITS** at 256x256
    - Removed PromptGenerator since using YOLO only
    """
    def __init__(self, sam_checkpoint, lora_rank=4, lora_alpha=4):
        super().__init__()
        sam_builder = sam_model_registry["vit_b"]
        sam, _ = sam_builder(image_size=1024, num_classes=1, checkpoint=sam_checkpoint)
        
        self.lora = LoRA_Sam(sam, r=lora_rank, lora_alpha=lora_alpha)
        self.sam = self.lora.sam_model

    def forward(self, img_256: torch.Tensor, boxes_norm: torch.Tensor | None = None):
        """
        Args:
          img_256: [B,3,256,256] in [0,1] range (e.g., from dataset, possibly repeated grayscale to RGB)
          boxes_norm: [B,4] xyxy in [0,1] (from YOLO). If None, fallback to full image.
        Returns:
          logits_256: [B,1,256,256] RAW LOGITS (no sigmoid)
        """
        B = img_256.shape[0]
        # Upsample to SAM's native 1024
        img_1024 = F.interpolate(img_256, (1024, 1024), mode="bilinear", align_corners=False)
        # Apply SAM preprocessing: assume input [0,1], convert to [0,255], then normalize
        img_1024 = img_1024 * 255.0
        device = img_1024.device
        pixel_mean = self.sam.pixel_mean.to(device)
        pixel_std = self.sam.pixel_std.to(device)
        img_1024 = (img_1024 - pixel_mean) / pixel_std
        # ---- Choose prompts ----
        if boxes_norm is None:
            # Fallback: full image box
            boxes_norm = img_256.new_tensor([0.0, 0.0, 1.0, 1.0]).expand(B, 4)
        boxes_1024 = boxes_norm * 1024.0  # SAM expects pixel coords
        boxes = boxes_1024.unsqueeze(1)  # [B,1,4]
        # ---- Frozen encoders ----
        with torch.no_grad():
            img_emb = self.sam.image_encoder(img_1024)  # [B,C,64,64]
            sp, dp = self.sam.prompt_encoder(points=None, boxes=boxes, masks=None)
        # ---- Trainable decoder ----
        low_res_logits, _ = self.sam.mask_decoder(
            image_embeddings=img_emb,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sp,
            dense_prompt_embeddings=dp,
            multimask_output=False,
        )  # low_res_logits: [B,1,256,256] RAW LOGITS
        # SAM low_res is already 256x256, no need for interpolate
        return low_res_logits  # RAW logits

    def get_learnable_parameters(self):
        # LoRA + decoder heads + decoder upscaling (no PG)
        yield from self.lora.get_learnable_parameters()


