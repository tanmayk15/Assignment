"""
Complete Faster R-CNN Model
Combines backbone, RPN, and ROI head
"""

import torch
import torch.nn as nn
from .backbone import build_backbone
from .rpn import RPN
from .roi_head import ROIHead
from .losses import FocalLoss, SmoothL1Loss


class FasterRCNN(nn.Module):
    """
    Faster R-CNN Object Detector
    Trained from scratch (no pre-trained weights)
    """
    
    def __init__(self, config):
        super(FasterRCNN, self).__init__()
        
        self.config = config
        model_cfg = config['model']
        
        # Backbone
        self.backbone = build_backbone(model_cfg['backbone'])
        
        # RPN
        rpn_cfg = model_cfg['rpn']
        self.rpn = RPN(
            in_channels=self.backbone.out_channels(),
            anchor_scales=rpn_cfg['anchor_scales'],
            anchor_ratios=rpn_cfg['anchor_ratios'],
            rpn_batch_size=rpn_cfg['rpn_batch_size'],
            rpn_positive_fraction=rpn_cfg['rpn_positive_fraction'],
            rpn_fg_iou_thresh=rpn_cfg['rpn_fg_iou_thresh'],
            rpn_bg_iou_thresh=rpn_cfg['rpn_bg_iou_thresh'],
            rpn_nms_thresh=rpn_cfg['rpn_nms_thresh'],
        )
        
        # ROI Head
        roi_cfg = model_cfg['roi_head']
        self.roi_head = ROIHead(
            in_channels=self.backbone.out_channels(),
            num_classes=model_cfg['num_classes'],
            roi_output_size=roi_cfg['roi_output_size'],
            roi_batch_size=roi_cfg['roi_batch_size'],
            roi_positive_fraction=roi_cfg['roi_positive_fraction'],
            roi_fg_iou_thresh=roi_cfg['roi_fg_iou_thresh'],
            roi_bg_iou_thresh=roi_cfg['roi_bg_iou_thresh'],
        )
        
        # Loss functions
        loss_cfg = model_cfg['loss']
        self.focal_loss = FocalLoss(
            alpha=loss_cfg['focal_loss_alpha'],
            gamma=loss_cfg['focal_loss_gamma']
        )
        self.smooth_l1_loss = SmoothL1Loss()
        
        # Loss weights
        self.loss_weights = {
            'rpn_cls': loss_cfg['rpn_cls_loss_weight'],
            'rpn_reg': loss_cfg['rpn_reg_loss_weight'],
            'det_cls': loss_cfg['det_cls_loss_weight'],
            'det_reg': loss_cfg['det_reg_loss_weight'],
        }
        
        # Image size
        self.image_size = tuple(config['dataset']['image_size'])
    
    def forward(self, images, targets=None):
        """
        Args:
            images: [B, 3, H, W] input images
            targets: list of dicts with 'boxes' and 'labels' (during training)
            
        Returns:
            detections: list of dicts with 'boxes', 'labels', 'scores'
            losses: dict of losses (during training)
        """
        # Extract features
        features = self.backbone(images)
        
        # RPN
        proposals, rpn_losses = self.rpn(
            features, self.image_size, targets
        )
        
        # ROI Head
        detections, roi_losses = self.roi_head(
            features, proposals, self.image_size, targets
        )
        
        # Combine losses
        losses = {}
        if self.training:
            losses.update(rpn_losses)
            losses.update(roi_losses)
            
            # Weight losses
            total_loss = (
                self.loss_weights['rpn_cls'] * losses.get('loss_rpn_cls', 0) +
                self.loss_weights['rpn_reg'] * losses.get('loss_rpn_reg', 0) +
                self.loss_weights['det_cls'] * losses.get('loss_cls', 0) +
                self.loss_weights['det_reg'] * losses.get('loss_box_reg', 0)
            )
            losses['total_loss'] = total_loss
        
        return detections, losses
    
    def predict(self, images, confidence_threshold=0.5):
        """
        Inference mode prediction
        
        Args:
            images: [B, 3, H, W] or single image [3, H, W]
            confidence_threshold: minimum confidence score
            
        Returns:
            detections: list of dicts with 'boxes', 'labels', 'scores'
        """
        self.eval()
        
        # Handle single image
        if images.dim() == 3:
            images = images.unsqueeze(0)
        
        with torch.no_grad():
            detections, _ = self.forward(images)
        
        # Filter by confidence
        filtered_detections = []
        for det in detections:
            if isinstance(det, dict):
                mask = det['scores'] >= confidence_threshold
                filtered_det = {
                    'boxes': det['boxes'][mask],
                    'labels': det['labels'][mask],
                    'scores': det['scores'][mask],
                }
                filtered_detections.append(filtered_det)
            else:
                filtered_detections.append(det)
        
        return filtered_detections
    
    def get_model_size(self):
        """Get model size in MB"""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


def build_model(config):
    """
    Factory function to build Faster R-CNN model
    
    Args:
        config: dict or yaml config
    """
    return FasterRCNN(config)


if __name__ == "__main__":
    import yaml
    
    # Load config
    with open('../configs/resnet18_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Build model
    model = build_model(config)
    
    print("=" * 60)
    print("Faster R-CNN Model")
    print("=" * 60)
    
    # Test forward pass
    images = torch.randn(2, 3, 640, 480)
    detections, losses = model(images)
    
    print(f"\nInput shape: {images.shape}")
    print(f"Number of detections: {len(detections)}")
    
    # Model info
    params = model.count_parameters()
    size = model.get_model_size()
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {params['total']:,}")
    print(f"  Trainable parameters: {params['trainable']:,}")
    print(f"  Model size: {size:.2f} MB")
