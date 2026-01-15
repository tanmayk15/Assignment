"""
Create demo video with annotations
"""

import argparse
import yaml
import torch
import cv2
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model
from engine import ObjectDetector


def parse_args():
    parser = argparse.ArgumentParser(description='Create demo video')
    parser.add_argument('--input', type=str, required=True,
                       help='Input video path')
    parser.add_argument('--output', type=str, required=True,
                       help='Output video path')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Model checkpoint')
    parser.add_argument('--config', type=str, default='configs/resnet18_config.yaml',
                       help='Config file')
    parser.add_argument('--show_fps', action='store_true',
                       help='Show FPS counter')
    parser.add_argument('--show_confidence', action='store_true',
                       help='Show confidence scores')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Build model
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    detector = ObjectDetector(model, config['dataset']['classes'], config, device)
    
    # Process video
    print(f"Processing {args.input}...")
    
    cap = cv2.VideoCapture(args.input)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictions = detector.predict(frame_rgb)
        
        # Visualize
        annotated = detector.visualize(frame_rgb, predictions)
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        
        out.write(annotated_bgr)
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    out.release()
    
    print(f"✓ Demo video saved to {args.output}")


if __name__ == '__main__':
    main()
