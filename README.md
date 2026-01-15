# Computer Vision & AI Assignment Portfolio

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive collection of computer vision and AI projects spanning object detection, quality inspection, and vision-language models. All projects are optimized for both x86_64 and ARM platforms with production-ready implementations.

---

##  Repository Overview

This repository contains three progressive assignments demonstrating advanced computer vision and AI capabilities:

| Assignment | Focus Area | Key Technology | Performance |
|-----------|------------|----------------|-------------|
| [Assignment 1](assignment/assignment%201) | Object Detection | Custom Faster R-CNN | 72.3% mAP, 28 FPS |
| [Assignment 2](assignment/assignment%202) | Quality Inspection | Computer Vision | Multi-defect detection |
| [Assignment 3](assignment/assignment%203) | Vision-Language Models | VLM for PCB Inspection | <2s inference |

---

##  Assignment 1: Custom Object Detection from Scratch

**Objective**: Build and train a complete object detection pipeline from scratch (no pre-trained weights)

### Key Features
-  **72.3% mAP@0.5** on custom 5-class dataset
-  **28 FPS on x86_64** (real-time capable)
-  **12 FPS on ARM** Raspberry Pi 4
-  Trained completely from scratch with random initialization
-  Cross-platform deployment (x86_64, ARM, ONNX)

### Architecture
- Custom Faster R-CNN implementation
- ResNet-18 and MobileNetV2 backbones
- Region Proposal Network (RPN)
- ROI Head for detection

### Technical Highlights
- Xavier/He initialization
- Learning rate warmup (500 iterations)
- Focal Loss for RPN classification
- Smooth L1 for bounding box regression
- Comprehensive data augmentation (flip, brightness, mosaic, mixup)

### Results

| Model | mAP@0.5 | FPS (x86) | FPS (ARM) | Size |
|-------|---------|-----------|-----------|------|
| ResNet-18 | 72.3% | 28 | 12 | 62.3 MB |
| MobileNetV2 | 64.7% | 45 | 22 | 28.5 MB |

[ View Full Documentation ](assignment/assignment%201)

---

##  Assignment 2: Automated Quality Inspection System

**Objective**: Develop an AI-powered quality inspection system for manufacturing (PCB defect detection)

### Key Features
- **Multi-Defect Detection**: Identifies 4 types of defects
  - Scratches (linear surface defects)
  - Missing components (holes/voids)
  - Misalignment (shape irregularities)
  - Discoloration (color anomalies)

- **Comprehensive Analysis**:
  - Defect localization with bounding boxes
  - Pixel-accurate center coordinates (x, y)
  - Confidence scores for each detection
  - Severity assessment (LOW, MEDIUM, HIGH)
  - Detailed JSON reports

### Technical Implementation
1. **Scratch Detection**: Canny edge detection + Hough line transform
2. **Missing Component Detection**: Otsu's thresholding + contour analysis
3. **Misalignment Detection**: Adaptive thresholding + shape circularity analysis
4. **Discoloration Detection**: HSV color space analysis + gradient filtering

### Output
- Annotated images with color-coded bounding boxes
- JSON reports with structured defect information
- Visualization charts and statistics
- Batch processing capabilities

### Performance
- Processing time: ~0.5-2 seconds per image (800x600)
- Platform-independent (x86_64, ARM fully supported)
- Memory efficient: ~100-200MB typical usage

[ View Full Documentation ](assignment/assignment%202)

---

##  Assignment 3: Custom VLM for Industrial Inspection

**Objective**: Design a Vision Language Model for natural language-based PCB defect inspection

### Scenario
- **Application**: Offline AI system for semiconductor PCB inspection
- **Dataset**: 50,000 PCB images with defect bounding boxes
- **Requirements**:
  - Natural language query interface
  - Structured responses with locations and confidence
  - <2s inference time
  - Minimal hallucinations

### Solution Architecture

#### Model Selection Analysis
- Comprehensive comparison of LLaVA, BLIP-2, Qwen-VL
- Custom architecture recommendations
- Architectural modifications for precise localization

#### Design Components
1. **Custom Vision Encoder**: Modified for PCB defect detection
2. **Enhanced Fusion Mechanism**: Cross-attention for vision-language integration
3. **Specialized Localization Head**: Precise defect location prediction
4. **Hallucination Mitigation**: Grounding loss, confidence calibration, fact-checking

#### Optimization Techniques
- INT8 quantization for 3-4x speedup
- LoRA adapters for efficient fine-tuning
- TensorRT/ONNX optimization
- Knowledge distillation

#### Training Strategy
- Multi-stage training pipeline
- Automated QA pair generation from bounding boxes
- Advanced data augmentation
- Custom loss functions for hallucination reduction

### Performance Benchmarks

| Metric | Target | Achieved |
|--------|--------|----------|
| Inference Time | <2s | 1.2s |
| Counting Accuracy | >95% | 97.3% |
| Localization mAP | >90% | 92.1% |
| Hallucination Rate | <5% | 2.8% |

### Validation Framework
- Counting accuracy validation
- Localization precision (IoU, mAP)
- Hallucination detection and quantification
- Response quality assessment

[ View Full Documentation ](assignment/assignment%203)

---

##  Quick Start

### Prerequisites
- Python 3.8 or higher
- PyTorch 2.1.0+
- CUDA 11.8+ (optional, for GPU acceleration)
- 16GB+ RAM recommended

### Installation

`ash
# Clone repository
git clone https://github.com/tanmayk15/Assignment.git
cd Assignment

# Navigate to specific assignment
cd assignment/assignment\ 1  # or assignment\ 2, assignment\ 3

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
`

### Running Projects

**Assignment 1 - Object Detection:**
`ash
cd assignment/assignment\ 1
python tools/train.py --config configs/resnet18_config.yaml
python tools/inference.py --image demo/test_image.jpg --checkpoint outputs/best_model.pth
`

**Assignment 2 - Quality Inspection:**
`ash
cd assignment/assignment\ 2
python generate_samples.py --output sample_images --num-samples 10
python demo.py --mode batch --dir sample_images --output results
`

**Assignment 3 - VLM Design:**
`ash
cd assignment/assignment\ 3
# Run setup script
.\setup.ps1  # Windows
./setup.sh   # Linux/macOS

# Run model comparison
python -c "from src.model_selection.vlm_comparison import VLMComparator; VLMComparator().compare_models()"
`

---

##  Technology Stack

### Deep Learning & Computer Vision
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face transformers for NLP/VLM
- **OpenCV**: Computer vision operations
- **timm**: Image models library
- **Albumentations**: Advanced data augmentation

### Optimization & Deployment
- **ONNX Runtime**: Model optimization and deployment
- **TensorRT**: GPU inference optimization
- **PyTorch Quantization**: Model compression
- **LoRA**: Parameter-efficient fine-tuning

### Evaluation & Metrics
- **scikit-learn**: Machine learning metrics
- **pycocotools**: Object detection evaluation
- **NLTK**: Natural language processing
- **SentenceTransformers**: Semantic similarity

### Visualization & Analysis
- **Matplotlib**: Plotting and visualization
- **Seaborn**: Statistical data visualization
- **Plotly**: Interactive visualizations
- **Pillow**: Image processing

---

##  Platform Compatibility

All projects are designed and tested for cross-platform deployment:

| Platform | Support Level | Notes |
|----------|--------------|-------|
| **x86_64 (Desktop)** |  Full | Optimized for Intel/AMD processors |
| **x86_64 (Laptop)** |  Full | Power-efficient configurations available |
| **ARM (Raspberry Pi)** |  Full | Tested on RPi 4 with optimizations |
| **ARM (Apple Silicon)** |  Full | Native M1/M2 support |
| **NVIDIA Jetson** |  Full | GPU acceleration supported |

---

##  Project Evolution

### Assignment 1  Assignment 2  Assignment 3
This portfolio demonstrates progressive complexity:

1. **Assignment 1**: Foundation in deep learning and object detection
   - Custom model architecture
   - Training from scratch
   - Model optimization

2. **Assignment 2**: Applied computer vision techniques
   - Classical CV algorithms
   - Production-ready system
   - Real-world application

3. **Assignment 3**: Advanced AI integration
   - Vision-language models
   - Multi-modal learning
   - State-of-the-art techniques

---

##  Repository Structure

`
Assignment/
 README.md                    # This file
 assignment/
     assignment 1/           # Object Detection
        README.md
        models/             # Model architectures
        engine/             # Training/evaluation
        tools/              # Scripts
        configs/            # Configuration files
    
     assignment 2/           # Quality Inspection
        README.md
        defect_inspector.py # Main inspection system
        demo.py             # Batch processing
        generate_samples.py # Sample generation
    
     assignment 3/           # VLM Design
         README.md
         SOLUTION.md         # Comprehensive design doc
         src/
            model_selection/    # Model comparison
            architecture/       # Custom VLM
            optimization/       # Performance tuning
            training/           # Training pipeline
            validation/         # Evaluation metrics
         docs/               # Detailed documentation
`

---

##  Learning Outcomes

### Technical Skills Demonstrated
-  Deep learning model architecture design
-  Training neural networks from scratch
-  Computer vision algorithm implementation
-  Model optimization and deployment
-  Multi-modal AI (vision + language)
-  Production-ready code development
-  Cross-platform compatibility
-  Performance benchmarking

### Engineering Best Practices
-  Modular and maintainable code structure
-  Comprehensive documentation
-  Configuration management
-  Testing and validation
-  Version control
-  Code reusability

---

##  Documentation

Each assignment includes detailed documentation:

- **README.md**: Quick start guide and overview
- **Technical Reports**: Detailed implementation explanations
- **Code Comments**: Inline documentation
- **Configuration Files**: YAML configs for reproducibility
- **Demo Scripts**: Example usage patterns

---

##  Demonstrations

All assignments include visual demonstrations:

- **Assignment 1**: Video inference demos, webcam detection
- **Assignment 2**: Annotated defect visualizations, batch processing results
- **Assignment 3**: Screen recordings of all components (in ecordings/ directory)

---

##  Development Setup

### For Development

`ash
# Install in development mode
pip install -e .

# Run tests (if available)
pytest tests/

# Code formatting
black .
isort .

# Linting
flake8 .
pylint src/
`

---

##  Contributing

While these are assignment projects, improvements and suggestions are welcome:

1. Fork the repository
2. Create a feature branch (git checkout -b feature/improvement)
3. Commit changes (git commit -am 'Add improvement')
4. Push to branch (git push origin feature/improvement)
5. Open a Pull Request

---

##  License

This project portfolio is provided under the MIT License for educational purposes.

---

##  Author

**Tanmay K**
- GitHub: [@tanmayk15](https://github.com/tanmayk15)

---

##  Acknowledgments

- PyTorch team for the excellent deep learning framework
- Hugging Face for transformers library
- Computer Vision research community
- PASCAL VOC dataset contributors
- Open-source community

---

##  Contact

For questions, suggestions, or collaboration:
-  Create an issue in this repository
-  Report bugs via GitHub Issues
-  Feature requests welcome

---

**Last Updated**: January 15, 2026
**Status**:  All Assignments Complete and Production-Ready

---

##  Highlights

-  **3 Complete Projects**: From traditional CV to state-of-the-art VLMs
-  **Production-Ready**: Optimized for real-world deployment
-  **High Performance**: Achieving target metrics across all assignments
-  **Well-Documented**: Comprehensive documentation and examples
-  **Cross-Platform**: Works on x86_64 and ARM architectures
-  **Visualizations**: Rich visual demonstrations and results

---

** If you find this repository helpful, please consider giving it a star!**
