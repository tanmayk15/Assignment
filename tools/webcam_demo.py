"""
Webcam Demo Script
Run real-time inference on webcam feed
"""

import argparse
import yaml
import torch
import cv2
import time

from models import build_model
from engine import ObjectDetector


def parse_args():
    parser = argparse.ArgumentParser(description='Run real-time inference on webcam')
    parser.add_argument('--config', type=str, default='configs/resnet18_config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold for detections')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--camera_id', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--display_size', type=int, nargs=2, default=[1280, 720],
                       help='Display window size (width height)')
    
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
    print(f"Classes: {config['dataset']['classes']}")
    print()
    
    # Open webcam
    cap = cv2.VideoCapture(args.camera_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera_id}")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Starting webcam demo...")
    print("Press 'q' to quit, 's' to save screenshot")
    print()
    
    # FPS calculation
    fps_history = []
    frame_count = 0
    screenshot_count = 0
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to grab frame")
                break
            
            frame_count += 1
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference
            start_time = time.time()
            predictions = detector.predict(
                frame_rgb,
                confidence_threshold=args.confidence_threshold
            )
            inference_time = time.time() - start_time
            
            # Calculate FPS
            fps = 1.0 / inference_time if inference_time > 0 else 0
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)
            
            # Visualize
            annotated = detector.visualize(
                frame_rgb,
                predictions,
                save_path=None,
                show=False,
            )
            
            # Convert back to BGR
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            
            # Add FPS counter
            cv2.putText(
                annotated_bgr,
                f"FPS: {avg_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )
            
            # Add detection count
            num_detections = len(predictions['boxes'])
            cv2.putText(
                annotated_bgr,
                f"Detections: {num_detections}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )
            
            # Resize for display
            display_frame = cv2.resize(
                annotated_bgr,
                tuple(args.display_size)
            )
            
            # Show frame
            cv2.imshow('Object Detection Demo', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f'screenshot_{screenshot_count}.jpg'
                cv2.imwrite(filename, annotated_bgr)
                print(f"Saved screenshot: {filename}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        print("\n" + "=" * 60)
        print("Webcam demo complete!")
        print("=" * 60)
        print(f"Total frames processed: {frame_count}")
        if fps_history:
            print(f"Average FPS: {sum(fps_history) / len(fps_history):.1f}")
        print(f"Screenshots saved: {screenshot_count}")


if __name__ == '__main__':
    main()
