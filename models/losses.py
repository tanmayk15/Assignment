"""
Loss Functions
Focal Loss for classification, Smooth L1 for regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [N, C] class logits
            targets: [N] class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss for bounding box regression
    
    L(x) = 0.5 * x^2           if |x| < 1
           |x| - 0.5           otherwise
    """
    
    def __init__(self, beta=1.0, reduction='mean'):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, inputs, targets, weights=None):
        """
        Args:
            inputs: [N, 4] predicted box deltas
            targets: [N, 4] target box deltas
            weights: [N] optional weights for each sample
        """
        diff = torch.abs(inputs - targets)
        
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        
        if weights is not None:
            loss = loss * weights.unsqueeze(1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def compute_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes
    
    Args:
        boxes1: [N, 4] (x1, y1, x2, y2)
        boxes2: [M, 4] (x1, y1, x2, y2)
        
    Returns:
        iou: [N, M] IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    union = area1[:, None] + area2 - inter
    
    iou = inter / union
    return iou


def encode_boxes(reference_boxes, gt_boxes):
    """
    Encode ground truth boxes relative to reference boxes
    
    Returns: [N, 4] encoded deltas (dx, dy, dw, dh)
    """
    # Reference boxes
    ref_widths = reference_boxes[:, 2] - reference_boxes[:, 0]
    ref_heights = reference_boxes[:, 3] - reference_boxes[:, 1]
    ref_ctr_x = reference_boxes[:, 0] + 0.5 * ref_widths
    ref_ctr_y = reference_boxes[:, 1] + 0.5 * ref_heights
    
    # Ground truth boxes
    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights
    
    # Encode with numerical stability
    eps = 1e-6
    targets_dx = (gt_ctr_x - ref_ctr_x) / (ref_widths + eps)
    targets_dy = (gt_ctr_y - ref_ctr_y) / (ref_heights + eps)
    targets_dw = torch.log((gt_widths + eps) / (ref_widths + eps))
    targets_dh = torch.log((gt_heights + eps) / (ref_heights + eps))
    
    targets = torch.stack([targets_dx, targets_dy, targets_dw, targets_dh], dim=1)
    return targets


if __name__ == "__main__":
    # Test Focal Loss
    focal_loss = FocalLoss()
    inputs = torch.randn(100, 5)  # 100 samples, 5 classes
    targets = torch.randint(0, 5, (100,))
    loss = focal_loss(inputs, targets)
    print(f"Focal Loss: {loss.item():.4f}")
    
    # Test Smooth L1 Loss
    smooth_l1 = SmoothL1Loss()
    pred_boxes = torch.randn(100, 4)
    target_boxes = torch.randn(100, 4)
    loss = smooth_l1(pred_boxes, target_boxes)
    print(f"Smooth L1 Loss: {loss.item():.4f}")
    
    # Test IoU
    boxes1 = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
    boxes2 = torch.tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
    iou = compute_iou(boxes1, boxes2)
    print(f"IoU matrix:\n{iou}")
