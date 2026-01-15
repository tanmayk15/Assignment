"""
Evaluation Script
Evaluate trained model on test set
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from models import build_model
from data import VOCDetectionDataset, collate_fn
from engine import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate object detection model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--split', type=str, default='test',
                       help='Dataset split to evaluate (test/val)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--confidence_threshold', type=float, default=0.05,
                       help='Confidence threshold for detections')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update data directory
    config['dataset']['data_dir'] = args.data_dir
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    print(f"\nLoading {args.split} dataset...")
    dataset = VOCDetectionDataset(
        data_dir=args.data_dir,
        split=args.split,
        classes=config['dataset']['classes'],
        config=config,
    )
    
    print(f"Dataset: {len(dataset)} samples")
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # Build model
    print("\nBuilding model...")
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
    
    model.to(device)
    model.eval()
    
    # Print model info
    params = model.count_parameters()
    size = model.get_model_size()
    print(f"\nModel: {config['model']['name']}")
    print(f"Backbone: {config['model']['backbone']}")
    print(f"Total parameters: {params['total']:,}")
    print(f"Model size: {size:.2f} MB")
    
    # Create evaluator
    evaluator = Evaluator(
        classes=config['dataset']['classes'],
        iou_thresholds=config['evaluation']['iou_thresholds'],
        device=device,
    )
    
    # Run evaluation
    print("\n" + "=" * 60)
    print("Starting evaluation...")
    print("=" * 60)
    
    results = evaluator.evaluate(
        model=model,
        data_loader=data_loader,
        confidence_threshold=args.confidence_threshold,
    )
    
    # Save results
    import json
    import os
    
    output_dir = os.path.dirname(args.checkpoint)
    results_file = os.path.join(output_dir, f'eval_results_{args.split}.json')
    
    # Convert numpy types to Python types for JSON serialization
    results_json = {}
    for key, value in results.items():
        if isinstance(value, dict):
            results_json[key] = {k: float(v) for k, v in value.items()}
        else:
            results_json[key] = float(value)
    
    with open(results_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
