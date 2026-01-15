"""
Inference Script
Run inference on a single image or directory of images
"""

import argparse
import yaml
import torch
import os
import glob

from models import build_model
from engine import ObjectDetector


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on images')
    parser.add_argument('--config', type=str, default='configs/resnet18_config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to input image')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='Path to directory of images')
    parser.add_argument('--output', type=str, default='output.jpg',
                       help='Path to save output image')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Path to save output images (for directory input)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Confidence threshold for detections')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--show', action='store_true',
                       help='Display the output image')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Check inputs
    if args.image is None and args.image_dir is None:
        print("Error: Must specify either --image or --image_dir")
        return
    
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
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Best mAP: {checkpoint.get('best_map', 'N/A'):.3f}")
    else:
        model.load_state_dict(checkpoint)
    
    # Create detector
    detector = ObjectDetector(
        model=model,
        classes=config['dataset']['classes'],
        config=config,
        device=device,
    )
    
    print(f"Model loaded successfully!")
    print(f"Classes: {config['dataset']['classes']}")
    print()
    
    # Single image inference
    if args.image:
        print(f"Running inference on {args.image}...")
        
        # Predict
        predictions = detector.predict(
            args.image,
            confidence_threshold=args.confidence_threshold
        )
        
        # Print results
        print(f"Found {len(predictions['boxes'])} detections:")
        for box, label, score in zip(predictions['boxes'], 
                                      predictions['labels'], 
                                      predictions['scores']):
            class_idx = label.item() - 1 if label > 0 else 0
            if class_idx < len(config['dataset']['classes']):
                class_name = config['dataset']['classes'][class_idx]
            else:
                class_name = 'unknown'
            print(f"  {class_name}: {score:.3f} at {box.tolist()}")
        
        print(f"Inference time: {predictions['inference_time']:.3f}s")
        print(f"FPS: {predictions['fps']:.1f}")
        
        # Visualize
        detector.visualize(
            args.image,
            predictions,
            save_path=args.output,
            show=args.show,
        )
        
        print(f"\nOutput saved to {args.output}")
    
    # Directory inference
    elif args.image_dir:
        print(f"Running inference on images in {args.image_dir}...")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Get all images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
        
        print(f"Found {len(image_paths)} images")
        
        # Process each image
        total_time = 0
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            print(f"Processing {filename}...")
            
            # Predict
            predictions = detector.predict(
                img_path,
                confidence_threshold=args.confidence_threshold
            )
            
            total_time += predictions['inference_time']
            
            # Save visualization
            output_path = os.path.join(args.output_dir, f'det_{filename}')
            detector.visualize(
                img_path,
                predictions,
                save_path=output_path,
                show=False,
            )
        
        avg_time = total_time / len(image_paths)
        avg_fps = 1.0 / avg_time
        
        print(f"\nProcessed {len(image_paths)} images")
        print(f"Average inference time: {avg_time:.3f}s")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Outputs saved to {args.output_dir}")


if __name__ == '__main__':
    main()
