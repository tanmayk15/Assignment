"""
Utility functions for bounding box operations
"""

import torch
import numpy as np
from typing import Union, Tuple


def box_area(boxes: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Compute area of boxes
    
    Args:
        boxes: [N, 4] in (x1, y1, x2, y2) format
        
    Returns:
        areas: [N] box areas
    """
    if isinstance(boxes, torch.Tensor):
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    else:
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU between two sets of boxes
    
    Args:
        boxes1: [N, 4] in (x1, y1, x2, y2) format
        boxes2: [M, 4] in (x1, y1, x2, y2) format
        
    Returns:
        iou: [N, M] IoU matrix
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    union = area1[:, None] + area2 - inter
    
    iou = inter / (union + 1e-6)
    return iou


def clip_boxes_to_image(boxes: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """
    Clip boxes to image boundaries
    
    Args:
        boxes: [N, 4] in (x1, y1, x2, y2) format
        size: (height, width) of image
        
    Returns:
        clipped_boxes: [N, 4]
    """
    height, width = size
    clipped_boxes = boxes.clone()
    clipped_boxes[:, 0].clamp_(min=0, max=width)  # x1
    clipped_boxes[:, 1].clamp_(min=0, max=height)  # y1
    clipped_boxes[:, 2].clamp_(min=0, max=width)  # x2
    clipped_boxes[:, 3].clamp_(min=0, max=height)  # y2
    return clipped_boxes


def remove_small_boxes(boxes: torch.Tensor, min_size: float) -> torch.Tensor:
    """
    Remove boxes with area smaller than min_size
    
    Args:
        boxes: [N, 4] in (x1, y1, x2, y2) format
        min_size: minimum box size
        
    Returns:
        keep: [K] indices of boxes to keep
    """
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    return torch.where(keep)[0]


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2) format
    
    Args:
        boxes: [N, 4] in (cx, cy, w, h) format
        
    Returns:
        boxes: [N, 4] in (x1, y1, x2, y2) format
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from (x1, y1, x2, y2) to (cx, cy, w, h) format
    
    Args:
        boxes: [N, 4] in (x1, y1, x2, y2) format
        
    Returns:
        boxes: [N, 4] in (cx, cy, w, h) format
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    idxs: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """
    Performs non-maximum suppression in a batched fashion.
    
    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.
    
    Args:
        boxes: [N, 4] boxes in (x1, y1, x2, y2) format
        scores: [N] confidence scores
        idxs: [N] category indices
        iou_threshold: IoU threshold for NMS
        
    Returns:
        keep: [K] indices of boxes to keep after NMS
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    
    # Strategy: offset boxes by category to perform NMS separately
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    
    from torchvision.ops import nms
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep


if __name__ == "__main__":
    # Test box operations
    boxes1 = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]])
    boxes2 = torch.tensor([[0, 0, 10, 10], [10, 10, 20, 20]])
    
    print("Box IoU:")
    iou = box_iou(boxes1, boxes2)
    print(iou)
    
    print("\nBox area:")
    area = box_area(boxes1)
    print(area)
    
    print("\nClip boxes:")
    clipped = clip_boxes_to_image(boxes1, size=(8, 8))
    print(clipped)
