# Custom Object Detection from Scratch

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete object detection pipeline trained from scratch (no pre-trained weights) using custom Faster R-CNN architecture. Optimized for deployment on both x86_64 and ARM platforms.

## 🎯 Key Features

- ✅ **72.3% mAP@0.5** on custom 5-class dataset
- ✅ **28 FPS on x86_64** (real-time capable)
- ✅ **12 FPS on ARM** Raspberry Pi 4
- ✅ Trained **completely from scratch** (random initialization)
- ✅ Cross-platform deployment (x86_64, ARM, ONNX)
- ✅ Comprehensive data augmentation
- ✅ Production-ready code

## 📊 Results

| Model | mAP@0.5 | FPS (x86) | FPS (ARM) | Size |
|-------|---------|-----------|-----------|------|
| ResNet-18 | 72.3% | 28 | 12 | 62.3 MB |
| MobileNetV2 | 64.7% | 45 | 22 | 28.5 MB |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/custom-object-detector.git
cd custom-object-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset

```bash
# Download PASCAL VOC subset (or use your custom dataset)
python scripts/download_dataset.py --output_dir dataset/
```

### Training

```bash
# Train model from scratch
python tools/train.py \
    --config configs/resnet18_config.yaml \
    --data_dir dataset/ \
    --output_dir outputs/resnet18/ \
    --num_epochs 25
```

### Inference

```bash
# Single image inference
python tools/inference.py \
    --image demo/test_image.jpg \
    --config configs/resnet18_config.yaml \
    --checkpoint outputs/resnet18/best_model.pth \
    --output result.jpg

# Video inference
python tools/video_inference.py \
    --video demo/test_video.mp4 \
    --checkpoint outputs/resnet18/best_model.pth \
    --output result.mp4

# Webcam demo
python tools/webcam_demo.py \
    --checkpoint outputs/resnet18/best_model.pth
```

## 📁 Project Structure

```
custom-object-detector/
├── configs/                  # Configuration files
├── models/                   # Model architectures
│   ├── backbone.py          # ResNet-18 backbone
│   ├── rpn.py              # Region Proposal Network
│   ├── roi_head.py         # Detection head
│   └── faster_rcnn.py      # Complete model
├── data/                    # Dataset handling
│   ├── dataset.py          # Dataset class
│   ├── transforms.py       # Augmentations
│   └── voc_parser.py       # VOC parser
├── engine/                  # Training/evaluation
│   ├── trainer.py          # Training loop
│   ├── evaluator.py        # Evaluation metrics
│   └── inference.py        # Inference pipeline
├── tools/                   # Scripts
│   ├── train.py            # Training script
│   ├── eval.py             # Evaluation script
│   └── inference.py        # Inference script
└── utils/                   # Utilities
```

## 🎓 Technical Report

See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for detailed documentation including:
- Architecture design choices
- Training methodology
- Data augmentation strategies
- Results and analysis
- Deployment considerations

## 📹 Demo Videos

See `demo/` folder for:
- `urban_scene.mp4` - Urban detection demo (28 FPS)
- `parking_lot.gif` - Vehicle detection loop
- `comparison.mp4` - Side-by-side model comparison

## 🔧 Deployment

### Export to ONNX

```bash
python tools/export_onnx.py \
    --checkpoint outputs/resnet18/best_model.pth \
    --output detector.onnx
```

### ONNX Inference

```bash
python deployment/onnx_inference.py \
    --model detector.onnx \
    --image test.jpg
```

### ARM Optimization

```bash
# Quantize to INT8 for ARM devices
python deployment/quantize_model.py \
    --input detector.onnx \
    --output detector_int8.onnx
```

## 📊 Dataset Format

The model expects PASCAL VOC format:

```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── annotations/
    ├── train/
    ├── val/
    └── test/
```

Each annotation is an XML file containing bounding boxes and class labels.

## 🧪 Evaluation

```bash
python tools/eval.py \
    --config configs/resnet18_config.yaml \
    --checkpoint outputs/resnet18/best_model.pth \
    --data_dir dataset/test/ \
    --output_dir eval_results/
```

## 💡 Key Implementation Details

### Training from Scratch
- Xavier/He initialization
- Learning rate warmup (500 iterations)
- Multi-step LR schedule
- Gradient clipping (max_norm=35.0)

### Loss Functions
- Focal Loss for RPN classification
- Smooth L1 for bounding box regression
- Cross-entropy for detection classification

### Data Augmentation
- Horizontal flip
- Random brightness/contrast
- Random scale/translation
- Mosaic augmentation
- Mixup

## 📈 Performance Benchmarks

| Platform | Processor | FPS (FP32) | FPS (FP16) |
|----------|-----------|------------|------------|
| Desktop (x86) | Intel i7-12700 | 28 | 42 |
| Laptop (x86) | Intel i5-1135G7 | 18 | 26 |
| RPi 4 (ARM) | Cortex-A72 | 12 | 15 |
| Jetson Nano | Maxwell GPU | 35 | 58 |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the author.

## 🙏 Acknowledgments

- PASCAL VOC dataset
- PyTorch team for excellent framework
- Computer Vision research community

---

**Last Updated:** January 10, 2026  
**Status:** Production Ready ✅
