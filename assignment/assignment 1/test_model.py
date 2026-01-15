"""
Quick test script to verify the model works
"""
import torch
import yaml
from models import build_model

print("="*60)
print("TESTING OBJECT DETECTION MODEL")
print("="*60)

# Load configuration
print("\n1. Loading configuration...")
with open('configs/resnet18_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print("   ✓ Config loaded")

# Build model
print("\n2. Building Faster R-CNN model...")
model = build_model(config)
print("   ✓ Model built successfully")

# Print model info
print("\n3. Model Information:")
params = model.count_parameters()
size = model.get_model_size()
print(f"   - Total parameters: {params['total']:,}")
print(f"   - Trainable parameters: {params['trainable']:,}")
print(f"   - Model size: {size:.1f} MB")

# Test inference
print("\n4. Testing inference...")
model.eval()
dummy_image = torch.randn(1, 3, 640, 480)  # Batch of 1 image

with torch.no_grad():
    detections, _ = model(dummy_image)

print(f"   ✓ Inference successful!")
print(f"   - Input shape: {dummy_image.shape}")
print(f"   - Output detections: {len(detections)} image(s)")

print("\n" + "="*60)
print("✓ ALL TESTS PASSED - MODEL IS WORKING!")
print("="*60)
print("\nWhat you can do next:")
print("1. Train: python tools/train.py --config configs/resnet18_config.yaml --data_dir dataset/ --output_dir outputs/")
print("2. Evaluate: python tools/eval.py --config configs/resnet18_config.yaml --checkpoint model.pth --data_dir dataset/")
print("3. Inference: python tools/inference.py --config configs/resnet18_config.yaml --checkpoint model.pth --image_path image.jpg")
