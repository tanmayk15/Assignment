"""
Video Inference Script
Run inference on video files
"""

import argparse
import yaml
import torch
import cv2
import time

from models import build_model
from engine import ObjectDetector


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on video')
    parser.add_argument('--config', type=str, default='configs/resnet18_config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video')
    parser.add_argument('--output', type=str, default='output.mp4',
                       help='Path to save output video')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold for detections')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--fps', type=int, default=None,
                       help='Output video FPS (default: same as input)')
    parser.add_argument('--skip_frames', type=int, default=1,
                       help='Process every Nth frame (1 = all frames)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build model
    print("Building model...")
    model = build_model(config)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Create detector
    detector = ObjectDetector(
        model=model,
        classes=config['dataset']['classes'],
        config=config,
        device=device,
    )
    
    print("Model loaded successfully!")
    
    # Open video
    cap = cv2.VideoCapture(args.video)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_fps = args.fps or input_fps
    
    print(f"\nVideo info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {input_fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Output FPS: {output_fps}")
    print()
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, output_fps, (width, height))
    
    # Process video
    frame_count = 0
    processed_count = 0
    total_inference_time = 0
    
    print("Processing video...")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames if specified
            if frame_count % args.skip_frames != 0:
                out.write(frame)
                continue
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference
            predictions = detector.predict(
                frame_rgb,
                confidence_threshold=args.confidence_threshold
            )
            
            processed_count += 1
            total_inference_time += predictions['inference_time']
            
            # Visualize on frame
            annotated = detector.visualize(
                frame_rgb,
                predictions,
                save_path=None,
                show=False,
            )
            
            # Convert back to BGR
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            
            # Write frame
            out.write(annotated_bgr)
            
            # Print progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                avg_fps = processed_count / total_inference_time if total_inference_time > 0 else 0
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}), "
                      f"Avg FPS: {avg_fps:.1f}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Release resources
        cap.release()
        out.release()
        
        # Print summary
        print("\n" + "=" * 60)
        print("Video processing complete!")
        print("=" * 60)
        print(f"Processed frames: {processed_count}")
        print(f"Total time: {total_inference_time:.2f}s")
        if processed_count > 0:
            avg_time = total_inference_time / processed_count
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0
            print(f"Average inference time: {avg_time:.3f}s")
            print(f"Average FPS: {avg_fps:.1f}")
        print(f"\nOutput saved to {args.output}")


if __name__ == '__main__':
    main()
