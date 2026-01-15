"""
Custom VLM Architecture for PCB Inspection
Combines vision encoder, fusion module, and localization head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import json


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for spatial awareness"""
    
    def __init__(self, dim: int, max_h: int = 100, max_w: int = 100):
        super().__init__()
        self.dim = dim
        
        # Create positional encoding
        pe = torch.zeros(max_h, max_w, dim)
        
        for i in range(max_h):
            for j in range(max_w):
                for k in range(0, dim, 4):
                    pe[i, j, k] = torch.sin(i / (10000 ** (k / dim)))
                    pe[i, j, k + 1] = torch.cos(i / (10000 ** (k / dim)))
                    pe[i, j, k + 2] = torch.sin(j / (10000 ** ((k + 2) / dim)))
                    pe[i, j, k + 3] = torch.cos(j / (10000 ** ((k + 2) / dim)))
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, N, D]
            coords: Coordinates [B, N, 2] (normalized 0-1)
        Returns:
            Features with positional encoding [B, N, D]
        """
        B, N, D = x.shape
        h, w = self.pe.shape[:2]
        
        # Convert normalized coords to indices
        h_idx = (coords[:, :, 0] * (h - 1)).long()
        w_idx = (coords[:, :, 1] * (w - 1)).long()
        
        # Gather positional encodings
        pos_enc = self.pe[h_idx, w_idx]
        
        return x + pos_enc


class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale features"""
    
    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Lateral connections (reduce channel dimension)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels
        ])
        
        # Output convs (smooth features)
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels
        ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: Multi-scale features from backbone
        Returns:
            FPN features at multiple scales
        """
        # Lateral connections
        laterals = [
            lateral_conv(features[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            # Upsample and add
            laterals[i - 1] += F.interpolate(
                laterals[i], 
                size=laterals[i - 1].shape[2:],
                mode='nearest'
            )
        
        # Output convs
        fpn_features = [
            fpn_conv(laterals[i])
            for i, fpn_conv in enumerate(self.fpn_convs)
        ]
        
        return fpn_features


class DefectAwareAttention(nn.Module):
    """Multi-head attention with defect-aware bias"""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self, 
        x: torch.Tensor, 
        defect_prior: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input features [B, N, D]
            defect_prior: Optional defect heatmap [B, N, 1]
        Returns:
            Attended features [B, N, D]
        """
        B, N, D = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply defect prior if available
        if defect_prior is not None:
            attn = attn + defect_prior.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        
        return out


class SpatialCrossAttention(nn.Module):
    """Cross-attention between vision and language with spatial grounding"""
    
    def __init__(self, visual_dim: int, text_dim: int, hidden_dim: int = 768, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Projections
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Spatial encoding
        self.spatial_encoding = PositionalEncoding2D(hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self, 
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        spatial_coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            visual_features: [B, N_v, D_v]
            text_features: [B, N_t, D_t]
            spatial_coords: [B, N_v, 2] normalized coordinates
        Returns:
            attended_features: [B, N_t, D]
            attention_weights: [B, num_heads, N_t, N_v]
        """
        # Project features
        V = self.visual_proj(visual_features)
        T = self.text_proj(text_features)
        
        # Add spatial encoding to visual features
        V = self.spatial_encoding(V, spatial_coords)
        
        # Cross-attention: text queries visual features
        attended, attn_weights = self.cross_attn(query=T, key=V, value=V)
        
        # Residual connection and normalization
        attended = self.norm(attended + T)
        attended = self.out_proj(attended)
        
        return attended, attn_weights


class LocalizationHead(nn.Module):
    """Detection head for precise defect localization"""
    
    def __init__(
        self, 
        feature_dim: int = 768, 
        num_classes: int = 10,
        roi_size: int = 7
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.roi_size = roi_size
        
        # RoI pooling
        self.roi_align = None  # Would use torchvision.ops.RoIAlign in practice
        
        # Flatten dimension after RoI pooling
        flatten_dim = feature_dim * roi_size * roi_size
        
        # Box regression branch
        self.box_head = nn.Sequential(
            nn.Linear(flatten_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 4)  # [x, y, w, h]
        )
        
        # Classification branch
        self.cls_head = nn.Sequential(
            nn.Linear(flatten_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Confidence branch
        self.conf_head = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        features: torch.Tensor, 
        proposals: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: Feature maps [B, C, H, W]
            proposals: Proposed regions [N, 5] (batch_idx, x1, y1, x2, y2)
        Returns:
            boxes: Refined boxes [N, 4]
            classes: Class logits [N, num_classes]
            confidences: Confidence scores [N, 1]
        """
        # RoI pooling (simplified - would use actual RoIAlign)
        B, C, H, W = features.shape
        pooled = torch.randn(proposals.shape[0], C, self.roi_size, self.roi_size).to(features.device)
        pooled_flat = pooled.flatten(1)
        
        # Predictions
        boxes = self.box_head(pooled_flat)
        classes = self.cls_head(pooled_flat)
        confidences = self.conf_head(pooled_flat)
        
        return boxes, classes, confidences


class CustomPCBVLM(nn.Module):
    """
    Custom Vision-Language Model for PCB Inspection
    Integrates modified vision encoder, cross-attention fusion, and localization head
    """
    
    def __init__(
        self,
        vision_encoder: str = 'resnet50',
        language_model: str = 'gpt2',
        hidden_dim: int = 768,
        num_defect_classes: int = 10,
        max_text_len: int = 512
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_defect_classes = num_defect_classes
        
        # Vision Encoder (simplified - would use actual backbone)
        self.vision_encoder = self._build_vision_encoder(vision_encoder)
        
        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(
            in_channels=[256, 512, 1024, 2048],
            out_channels=256
        )
        
        # Defect-aware attention
        self.defect_attention = DefectAwareAttention(dim=hidden_dim)
        
        # Language Encoder (simplified - would use actual LM)
        self.text_encoder = nn.Embedding(50000, hidden_dim)
        self.text_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8),
            num_layers=6
        )
        
        # Cross-modal fusion
        self.cross_attention = SpatialCrossAttention(
            visual_dim=256,
            text_dim=hidden_dim,
            hidden_dim=hidden_dim
        )
        
        # Localization head
        self.localization_head = LocalizationHead(
            feature_dim=hidden_dim,
            num_classes=num_defect_classes
        )
        
        # Language generation head
        self.language_head = nn.Linear(hidden_dim, 50000)  # vocab size
        
        # Confidence calibration
        self.confidence_calibrator = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def _build_vision_encoder(self, encoder_name: str) -> nn.Module:
        """Build vision encoder (simplified)"""
        # In practice, would use timm or torchvision models
        return nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
    
    def encode_image(self, image: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Encode image with multi-scale features
        Args:
            image: [B, 3, H, W]
        Returns:
            features: [B, N, D]
            multi_scale_features: List of feature maps
        """
        # Extract features (simplified)
        B, C, H, W = image.shape
        
        # Multi-scale features
        multi_scale = [
            torch.randn(B, 256, H//4, W//4).to(image.device),
            torch.randn(B, 512, H//8, W//8).to(image.device),
            torch.randn(B, 1024, H//16, W//16).to(image.device),
            torch.randn(B, 2048, H//32, W//32).to(image.device)
        ]
        
        # Apply FPN
        fpn_features = self.fpn(multi_scale)
        
        # Flatten spatial dimensions
        features = fpn_features[0].flatten(2).transpose(1, 2)  # [B, N, D]
        
        return features, fpn_features
    
    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Encode text query
        Args:
            text_tokens: [B, L] token indices
        Returns:
            text_features: [B, L, D]
        """
        # Embed tokens
        embedded = self.text_encoder(text_tokens)
        
        # Transform
        text_features = self.text_transformer(embedded)
        
        return text_features
    
    def forward(
        self,
        image: torch.Tensor,
        text_tokens: torch.Tensor,
        return_localization: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        Args:
            image: [B, 3, H, W]
            text_tokens: [B, L]
            return_localization: Whether to return bounding boxes
        Returns:
            Dictionary with predictions
        """
        B = image.shape[0]
        
        # Encode modalities
        visual_features, fpn_features = self.encode_image(image)
        text_features = self.encode_text(text_tokens)
        
        # Generate spatial coordinates
        N = visual_features.shape[1]
        h_size = int(N ** 0.5)
        coords = torch.stack([
            torch.arange(h_size).repeat(h_size).float() / h_size,
            torch.arange(h_size).repeat_interleave(h_size).float() / h_size
        ], dim=1).unsqueeze(0).expand(B, -1, -1).to(image.device)
        
        # Cross-modal fusion
        fused_features, attn_weights = self.cross_attention(
            visual_features, text_features, coords
        )
        
        # Language generation
        language_logits = self.language_head(fused_features)
        
        # Confidence score
        confidence = self.confidence_calibrator(fused_features.mean(dim=1))
        
        outputs = {
            'language_logits': language_logits,
            'confidence': confidence,
            'attention_weights': attn_weights
        }
        
        # Localization (if requested)
        if return_localization:
            # Generate proposals (simplified)
            proposals = torch.randn(100, 5).to(image.device)
            
            boxes, classes, box_confidences = self.localization_head(
                fpn_features[0], proposals
            )
            
            outputs.update({
                'boxes': boxes,
                'classes': classes,
                'box_confidences': box_confidences
            })
        
        return outputs
    
    def generate(
        self,
        image: torch.Tensor,
        question: str,
        max_length: int = 100
    ) -> Dict:
        """
        Generate structured response
        Args:
            image: [1, 3, H, W]
            question: Natural language query
            max_length: Maximum response length
        Returns:
            Structured response with answer and locations
        """
        # Tokenize question (simplified)
        text_tokens = torch.randint(0, 50000, (1, 20)).to(image.device)
        
        # Forward pass
        outputs = self.forward(image, text_tokens, return_localization=True)
        
        # Generate response (simplified - would use actual generation)
        response = {
            'answer': 'Found 3 solder bridge defects',
            'count': 3,
            'locations': [
                {
                    'bbox': [120, 340, 145, 365],
                    'confidence': 0.95,
                    'type': 'solder_bridge'
                },
                {
                    'bbox': [200, 150, 225, 175],
                    'confidence': 0.89,
                    'type': 'solder_bridge'
                },
                {
                    'bbox': [450, 280, 475, 305],
                    'confidence': 0.92,
                    'type': 'solder_bridge'
                }
            ],
            'confidence': outputs['confidence'].item()
        }
        
        return response


def main():
    """Test custom VLM architecture"""
    print("=" * 60)
    print("Custom PCB VLM Architecture Test")
    print("=" * 60)
    
    # Create model
    model = CustomPCBVLM(
        vision_encoder='resnet50',
        language_model='gpt2',
        hidden_dim=768,
        num_defect_classes=10
    )
    
    print(f"\n✓ Model created successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 2
    image = torch.randn(batch_size, 3, 1024, 1024)
    text_tokens = torch.randint(0, 50000, (batch_size, 20))
    
    print(f"\n✓ Testing forward pass...")
    outputs = model(image, text_tokens)
    
    print(f"  Language logits shape: {outputs['language_logits'].shape}")
    print(f"  Confidence shape: {outputs['confidence'].shape}")
    print(f"  Boxes shape: {outputs['boxes'].shape}")
    print(f"  Classes shape: {outputs['classes'].shape}")
    
    # Test generation
    print(f"\n✓ Testing generation...")
    response = model.generate(image[:1], "How many defects are there?")
    print(f"  Response: {json.dumps(response, indent=2)}")
    
    print("\n" + "=" * 60)
    print("Architecture test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()