"""
Region Proposal Network (RPN)
Generates object proposals from feature maps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RPN(nn.Module):
    """
    Region Proposal Network
    Generates candidate object proposals with objectness scores
    """
    
    def __init__(
        self, 
        in_channels=512,
        anchor_scales=[64, 128, 256],
        anchor_ratios=[0.5, 1.0, 2.0],
        rpn_batch_size=256,
        rpn_positive_fraction=0.5,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_nms_thresh=0.7,
    ):
        super(RPN, self).__init__()
        
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.num_anchors = len(anchor_scales) * len(anchor_ratios)
        
        self.rpn_batch_size = rpn_batch_size
        self.rpn_positive_fraction = rpn_positive_fraction
        self.rpn_fg_iou_thresh = rpn_fg_iou_thresh
        self.rpn_bg_iou_thresh = rpn_bg_iou_thresh
        self.rpn_nms_thresh = rpn_nms_thresh
        
        # 3x3 convolution
        self.conv = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)
        
        # Classification: objectness (object vs background)
        self.cls_logits = nn.Conv2d(512, self.num_anchors * 2, kernel_size=1)
        
        # Regression: bounding box deltas (x, y, w, h)
        self.bbox_pred = nn.Conv2d(512, self.num_anchors * 4, kernel_size=1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, features, image_shape, targets=None):
        """
        Args:
            features: [B, C, H, W] feature maps from backbone
            image_shape: (height, width) of original image
            targets: dict with 'boxes' and 'labels' (during training)
            
        Returns:
            proposals: list of [N, 4] proposal boxes for each image
            losses: dict of losses (during training)
        """
        batch_size = features.shape[0]
        feature_height, feature_width = features.shape[2:]
        
        # Shared conv
        x = F.relu(self.conv(features))
        
        # Objectness scores: [B, num_anchors*2, H, W]
        objectness = self.cls_logits(x)
        
        # Bbox deltas: [B, num_anchors*4, H, W]
        bbox_deltas = self.bbox_pred(x)
        
        # Generate anchors
        anchors = self._generate_anchors(
            feature_height, feature_width, 
            image_shape, features.device
        )
        
        # Reshape outputs
        objectness = objectness.permute(0, 2, 3, 1).contiguous()
        objectness = objectness.view(batch_size, -1, 2)
        
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)
        
        # Generate proposals
        proposals = []
        losses = {}
        
        for i in range(batch_size):
            # Apply deltas to anchors
            props = self._apply_deltas_to_anchors(
                anchors, bbox_deltas[i]
            )
            
            # Clip to image boundaries
            props = self._clip_boxes(props, image_shape)
            
            # Apply NMS
            objectness_scores = F.softmax(objectness[i], dim=1)[:, 1]
            keep = self._nms(props, objectness_scores, self.rpn_nms_thresh)
            
            proposals.append(props[keep])
        
        # Compute losses during training
        if self.training and targets is not None:
            losses = self._compute_losses(
                objectness, bbox_deltas, anchors, targets
            )
        
        return proposals, losses
    
    def _generate_anchors(self, feature_h, feature_w, image_shape, device):
        """Generate anchor boxes at each spatial location"""
        stride = 32  # Feature stride
        
        # Generate anchor scales and ratios
        anchors_list = []
        for scale in self.anchor_scales:
            for ratio in self.anchor_ratios:
                # Anchor dimensions with numerical stability
                ratio_tensor = torch.tensor(ratio).clamp(min=1e-6)
                w = scale * torch.sqrt(ratio_tensor)
                h = scale / torch.sqrt(ratio_tensor)
                anchors_list.append([-w/2, -h/2, w/2, h/2])
        
        base_anchors = torch.tensor(anchors_list, dtype=torch.float32, device=device)
        
        # Generate grid
        shifts_x = torch.arange(0, feature_w, dtype=torch.float32, device=device) * stride
        shifts_y = torch.arange(0, feature_h, dtype=torch.float32, device=device) * stride
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        
        shifts = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=2)
        shifts = shifts.reshape(-1, 4)
        
        # Add shifts to base anchors
        anchors = base_anchors.view(1, -1, 4) + shifts.view(-1, 1, 4)
        anchors = anchors.reshape(-1, 4)
        
        return anchors
    
    def _apply_deltas_to_anchors(self, anchors, deltas):
        """Apply predicted deltas to anchors to get proposals"""
        # Convert anchors to (x, y, w, h)
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        ctr_x = anchors[:, 0] + 0.5 * widths
        ctr_y = anchors[:, 1] + 0.5 * heights
        
        # Apply deltas
        dx = deltas[:, 0]
        dy = deltas[:, 1]
        dw = deltas[:, 2]
        dh = deltas[:, 3]
        
        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights
        
        # Convert back to (x1, y1, x2, y2)
        pred_boxes = torch.zeros_like(deltas)
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
    
    def _nms(self, boxes, scores, threshold):
        """Non-Maximum Suppression"""
        from torchvision.ops import nms
        return nms(boxes, scores, threshold)
    
    def _compute_losses(self, objectness, bbox_deltas, anchors, targets):
        """Compute RPN losses"""
        # This is a simplified version - full implementation would include:
        # 1. Match anchors to ground truth boxes
        # 2. Sample positive and negative anchors
        # 3. Compute focal loss for classification
        # 4. Compute smooth L1 loss for regression
        
        losses = {
            'loss_rpn_cls': torch.tensor(0.0, device=objectness.device),
            'loss_rpn_reg': torch.tensor(0.0, device=bbox_deltas.device),
        }
        
        return losses


if __name__ == "__main__":
    # Test RPN
    rpn = RPN(in_channels=512)
    features = torch.randn(2, 512, 20, 15)
    image_shape = (480, 640)
    
    proposals, losses = rpn(features, image_shape)
    
    print("RPN Test:")
    print(f"Number of proposals per image: {[p.shape[0] for p in proposals]}")
    print(f"Proposal shape: {proposals[0].shape}")
