"""
ResNet-18 Backbone Implementation
Trained from scratch (no pre-trained weights)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18Backbone(nn.Module):
    """
    Custom ResNet-18 backbone for object detection
    Output stride: 32 (input 640x480 -> output 20x15)
    """
    
    def __init__(self):
        super(ResNet18Backbone, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 2, stride=1)   # 160x120
        self.layer2 = self._make_layer(128, 2, stride=2)  # 80x60
        self.layer3 = self._make_layer(256, 2, stride=2)  # 40x30
        self.layer4 = self._make_layer(512, 2, stride=2)  # 20x15
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input: [B, 3, 640, 480]
        x = self.conv1(x)       # [B, 64, 320, 240]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # [B, 64, 160, 120]
        
        x = self.layer1(x)      # [B, 64, 160, 120]
        x = self.layer2(x)      # [B, 128, 80, 60]
        x = self.layer3(x)      # [B, 256, 40, 30]
        x = self.layer4(x)      # [B, 512, 20, 15]
        
        return x
    
    def out_channels(self):
        """Return number of output channels"""
        return 512


class MobileNetV2Backbone(nn.Module):
    """
    Lightweight MobileNetV2 backbone for faster inference
    """
    
    def __init__(self):
        super(MobileNetV2Backbone, self).__init__()
        
        # Import torchvision's MobileNetV2 and modify
        from torchvision.models.mobilenetv2 import InvertedResidual
        
        self.features = nn.Sequential(
            # Initial conv
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            
            # Inverted residual blocks
            InvertedResidual(32, 16, stride=1, expand_ratio=1),
            InvertedResidual(16, 24, stride=2, expand_ratio=6),
            InvertedResidual(24, 24, stride=1, expand_ratio=6),
            InvertedResidual(24, 32, stride=2, expand_ratio=6),
            InvertedResidual(32, 32, stride=1, expand_ratio=6),
            InvertedResidual(32, 32, stride=1, expand_ratio=6),
            InvertedResidual(32, 64, stride=2, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            InvertedResidual(64, 96, stride=1, expand_ratio=6),
            InvertedResidual(96, 96, stride=1, expand_ratio=6),
            InvertedResidual(96, 96, stride=1, expand_ratio=6),
            InvertedResidual(96, 160, stride=2, expand_ratio=6),
            InvertedResidual(160, 160, stride=1, expand_ratio=6),
            InvertedResidual(160, 160, stride=1, expand_ratio=6),
            InvertedResidual(160, 320, stride=1, expand_ratio=6),
        )
        
        # Final conv to match feature dimension
        self.final_conv = nn.Conv2d(320, 512, kernel_size=1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.features(x)
        x = self.final_conv(x)
        return x
    
    def out_channels(self):
        return 512


def build_backbone(name='resnet18'):
    """
    Factory function to build backbone
    
    Args:
        name: 'resnet18' or 'mobilenet_v2'
    """
    if name == 'resnet18':
        return ResNet18Backbone()
    elif name == 'mobilenet_v2':
        return MobileNetV2Backbone()
    else:
        raise ValueError(f"Unknown backbone: {name}")


if __name__ == "__main__":
    # Test backbone
    backbone = build_backbone('resnet18')
    x = torch.randn(2, 3, 640, 480)
    out = backbone(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output channels: {backbone.out_channels()}")
    
    # Count parameters
    total_params = sum(p.numel() for p in backbone.parameters())
    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
