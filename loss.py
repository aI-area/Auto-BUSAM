
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, epsilon=1e-6):
        inputs = torch.sigmoid(inputs)
        intersection = torch.sum(inputs * targets, dim=(2, 3))
        union = torch.sum(inputs, dim=(2, 3)) + torch.sum(targets, dim=(2, 3))
        dice_score = (2. * intersection + epsilon) / (union + epsilon)
        return 1 - dice_score.mean()

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7, box_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.box_weight = box_weight  # For prompt gen
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.box_loss = nn.MSELoss()  # For boxes if GT provided

    def forward(self, inputs, targets, pred_boxes=None, gt_boxes=None):
        bce = self.bce_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        seg_loss = self.alpha * bce + (1 - self.alpha) * dice
        if pred_boxes is not None and gt_boxes is not None:
            box_l = self.box_loss(pred_boxes, gt_boxes)
            return seg_loss + self.box_weight * box_l
        return seg_loss