"""
ONNX Export Script
Convert PyTorch model to ONNX format for deployment
"""

import argparse
import yaml
import torch
import onnx
import onnxsim

from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Export model to ONNX')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='detector.onnx',
                       help='Output ONNX file path')
    parser.add_argument('--simplify', action='store_true',
                       help='Simplify ONNX model')
    parser.add_argument('--opset_version', type=int, default=13,
                       help='ONNX opset version')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Build model
    print("Building model...")
    model = build_model(config)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Create dummy input
    image_size = tuple(config['dataset']['image_size'])
    dummy_input = torch.randn(1, 3, image_size[0], image_size[1])
    
    print(f"\nExporting to ONNX...")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Opset version: {args.opset_version}")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=args.opset_version,
        do_constant_folding=True,
        input_names=['image'],
        output_names=['boxes', 'scores', 'labels'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'boxes': {0: 'num_detections'},
            'scores': {0: 'num_detections'},
            'labels': {0: 'num_detections'},
        }
    )
    
    print(f"✓ Model exported to {args.output}")
    
    # Simplify if requested
    if args.simplify:
        print("\nSimplifying ONNX model...")
        model_onnx = onnx.load(args.output)
        model_simplified, check = onnxsim.simplify(model_onnx)
        
        if check:
            simplified_path = args.output.replace('.onnx', '_simplified.onnx')
            onnx.save(model_simplified, simplified_path)
            print(f"✓ Simplified model saved to {simplified_path}")
        else:
            print("✗ Simplification failed")
    
    # Verify ONNX model
    print("\nVerifying ONNX model...")
    model_onnx = onnx.load(args.output)
    onnx.checker.check_model(model_onnx)
    print("✓ ONNX model is valid")
    
    # Print model info
    print("\nModel Information:")
    print(f"  IR version: {model_onnx.ir_version}")
    print(f"  Producer: {model_onnx.producer_name}")
    print(f"  Graph nodes: {len(model_onnx.graph.node)}")
    
    print("\nInputs:")
    for input in model_onnx.graph.input:
        print(f"  {input.name}: {[d.dim_value for d in input.type.tensor_type.shape.dim]}")
    
    print("\nOutputs:")
    for output in model_onnx.graph.output:
        print(f"  {output.name}: {[d.dim_value for d in output.type.tensor_type.shape.dim]}")
    
    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
