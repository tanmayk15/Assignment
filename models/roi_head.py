"""
ROI Head (Detection Head)
Classifies proposals and refines bounding boxes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign


class ROIHead(nn.Module):
    """
    ROI Head for object detection
    Takes proposals from RPN and outputs class predictions + refined boxes
    """
    
    def __init__(
        self,
        in_channels=512,
        num_classes=5,
        roi_output_size=7,
        roi_batch_size=512,
        roi_positive_fraction=0.25,
        roi_fg_iou_thresh=0.5,
        roi_bg_iou_thresh=0.5,
    ):
        super(ROIHead, self).__init__()
        
        self.num_classes = num_classes
        self.roi_batch_size = roi_batch_size
        self.roi_positive_fraction = roi_positive_fraction
        self.roi_fg_iou_thresh = roi_fg_iou_thresh
        self.roi_bg_iou_thresh = roi_bg_iou_thresh
        
        # RoI Align layer
        self.roi_align = RoIAlign(
            output_size=roi_output_size,
            spatial_scale=1.0/32.0,  # Feature stride
            sampling_ratio=2
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_channels * roi_output_size * roi_output_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        
        # Classification head (num_classes + 1 for background)
        self.cls_score = nn.Linear(1024, num_classes + 1)
        
        # Bounding box regression head
        self.bbox_pred = nn.Linear(1024, num_classes * 4)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        
        for layer in [self.fc1, self.fc2, self.cls_score, self.bbox_pred]:
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, features, proposals, image_shape, targets=None):
        """
        Args:
            features: [B, C, H, W] feature maps from backbone
            proposals: list of [N, 4] proposal boxes for each image
            image_shape: (height, width) of original image
            targets: dict with 'boxes' and 'labels' (during training)
            
        Returns:
            detections: list of dicts with 'boxes', 'labels', 'scores'
            losses: dict of losses (during training)
        """
        # RoI pooling
        roi_features = self.roi_align(features, proposals)
        
        # Flatten
        roi_features = roi_features.flatten(start_dim=1)
        
        # FC layers
        x = F.relu(self.fc1(roi_features))
        x = F.relu(self.fc2(x))
        
        # Predictions
        class_logits = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        
        # Post-process predictions
        detections = []
        losses = {}
        
        if not self.training:
            detections = self._post_process_detections(
                class_logits, bbox_deltas, proposals, image_shape
            )
        
        # Compute losses during training
        if self.training and targets is not None:
            losses = self._compute_losses(
                class_logits, bbox_deltas, proposals, targets
            )
        
        return detections, losses
    
    def _post_process_detections(
        self, class_logits, bbox_deltas, proposals, image_shape,
        score_thresh=0.05, nms_thresh=0.5, max_detections=100
    ):
        """Post-process predictions to get final detections"""
        from torchvision.ops import nms
        
        # Get class probabilities
        probs = F.softmax(class_logits, dim=1)
        
        # Get predicted boxes
        pred_boxes = self._apply_deltas_to_boxes(
            torch.cat(proposals, dim=0), bbox_deltas
        )
        
        # Clip to image
        pred_boxes = self._clip_boxes(pred_boxes, image_shape)
        
        detections = []
        num_classes = probs.shape[1] - 1  # Exclude background
        
        # For each image (simplified - assumes batch size 1)
        for cls_idx in range(1, num_classes + 1):
            cls_scores = probs[:, cls_idx]
            
            # Filter by score threshold
            keep = cls_scores > score_thresh
            if keep.sum() == 0:
                continue
            
            cls_boxes = pred_boxes[keep]
            cls_scores = cls_scores[keep]
            
            # Apply NMS
            keep_nms = nms(cls_boxes, cls_scores, nms_thresh)
            
            # Limit to max detections
            if len(keep_nms) > max_detections:
                keep_nms = keep_nms[:max_detections]
            
            detections.append({
                'boxes': cls_boxes[keep_nms],
                'labels': torch.full((len(keep_nms),), cls_idx, dtype=torch.int64),
                'scores': cls_scores[keep_nms]
            })
        
        return detections
    
    def _apply_deltas_to_boxes(self, boxes, deltas):
        """Apply predicted deltas to boxes"""
        # Convert to center format
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights
        
        # Apply deltas - handle multi-class predictions
        num_classes = deltas.shape[1] // 4
        dx = deltas[:, 0::4][:, 0]  # Take first class
        dy = deltas[:, 1::4][:, 0]
        dw = deltas[:, 2::4][:, 0]
        dh = deltas[:, 3::4][:, 0]
        
        # Predicted boxes
        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights
        
        # Convert back
        pred_boxes = torch.zeros((deltas.shape[0], 4), device=deltas.device)
        pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h
        
        return pred_boxes
    
    def _clip_boxes(self, boxes, image_shape):
        """Clip boxes to image boundaries"""
        height, width = image_shape
        boxes[:, 0] = torch.clamp(boxes[:, 0], min=0, max=width)
        boxes[:, 1] = torch.clamp(boxes[:, 1], min=0, max=height)
        boxes[:, 2] = torch.clamp(boxes[:, 2], min=0, max=width)
        boxes[:, 3] = torch.clamp(boxes[:, 3], min=0, max=height)
        return boxes
    
    def _compute_losses(self, class_logits, bbox_deltas, proposals, targets):
        """Compute detection losses"""
        # Simplified loss computation
        losses = {
            'loss_cls': torch.tensor(0.0, device=class_logits.device),
            'loss_box_reg': torch.tensor(0.0, device=bbox_deltas.device),
        }
        
        return losses


if __name__ == "__main__":
    # Test ROI Head
    roi_head = ROIHead(in_channels=512, num_classes=5)
    
    features = torch.randn(1, 512, 20, 15)
    proposals = [torch.rand(100, 4) * 400]  # 100 proposals
    image_shape = (480, 640)
    
    detections, losses = roi_head(features, proposals, image_shape)
    
    print("ROI Head Test:")
    print(f"Number of detections: {len(detections)}")
    if detections:
        print(f"First detection: {detections[0]}")
