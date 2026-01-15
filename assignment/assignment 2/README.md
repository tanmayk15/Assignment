# Automated Quality Inspection System for Manufacturing

A computer vision-based automated quality inspection system for detecting and classifying defects in manufactured items (PCBs - Printed Circuit Boards).

## 🎯 Features

- **Multi-Defect Detection**: Identifies 4 types of defects:
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

- **Visualization**: 
  - Annotated images with color-coded bounding boxes
  - Comparison views (original vs annotated)
  - Summary statistics and charts

## 📋 Requirements

- Python 3.8 or higher
- Compatible with x86_64 and ARM platforms
- See `requirements.txt` for dependencies

## 🚀 Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd assignment-2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
assignment-2/
├── defect_inspector.py      # Main inspection system
├── generate_samples.py      # Generate synthetic defect samples
├── demo.py                  # Batch processing and demos
├── requirements.txt         # Python dependencies
├── sample_images/           # Sample PCB images (generated)
├── results/                 # Output directory (generated)
└── README.md               # This file
```

## 🎮 Usage

### 1. Generate Sample Images

First, generate synthetic PCB images with various defects:

```bash
python generate_samples.py --output sample_images --num-samples 10
```

This creates:
- 1 perfect (defect-free) PCB
- 10 PCBs with mixed defects
- 4 PCBs with single defect types (for validation)

### 2. Run Inspection on Single Image

Inspect a single image:

```bash
python defect_inspector.py --image sample_images/pcb_defect_01.jpg --output results --visualize
```

Parameters:
- `--image`: Path to input image (required)
- `--output`: Output directory (default: 'output')
- `--sensitivity`: Detection sensitivity 0-1 (default: 0.5)
- `--visualize`: Display visualization

### 3. Batch Processing

Process all images in a directory:

```bash
python demo.py --mode batch --dir sample_images --output results
```

This will:
- Process all images in the directory
- Generate annotated images for each
- Create individual JSON reports
- Generate a summary report with statistics
- Create visualization charts

### 4. Single Image Demo

Test on a specific image:

```bash
python demo.py --mode single --image sample_images/pcb_perfect.jpg --output results
```

## 📊 Output Format

### JSON Report Structure

```json
{
  "image_path": "sample_images/pcb_defect_01.jpg",
  "timestamp": "2026-01-12T10:30:00",
  "total_defects": 3,
  "quality_status": "FAIL",
  "defects": [
    {
      "type": "scratch",
      "center": [245, 178],
      "bbox": [200, 150, 90, 56],
      "severity": "MEDIUM",
      "confidence": 0.875,
      "area": 94.3,
      "description": "Linear scratch detected, length: 94px"
    }
  ]
}
```

### Defect Properties

Each detected defect includes:
- `type`: Defect classification (scratch, missing_component, misalignment, discoloration)
- `center`: (x, y) pixel coordinates of defect center
- `bbox`: Bounding box as (x, y, width, height)
- `severity`: Assessment level (LOW, MEDIUM, HIGH)
- `confidence`: Detection confidence score (0-1)
- `area`: Defect size metric
- `description`: Human-readable description

## 🎨 Visualization

The system generates:

1. **Annotated Images**: Original images with:
   - Color-coded bounding boxes per defect type
   - Center point markers
   - Coordinate labels
   - Status banner (PASS/FAIL)

2. **Comparison Views**: Side-by-side original and annotated

3. **Summary Charts**: 
   - Pass/Fail distribution
   - Defect type breakdown
   - Severity distribution
   - Overall statistics

### Defect Color Codes

- 🔴 **Scratch**: Red
- 🔵 **Missing Component**: Blue  
- 🟡 **Misalignment**: Yellow
- 🟠 **Discoloration**: Orange

## 🧪 Testing

### Quick Test

1. Generate samples:
```bash
python generate_samples.py
```

2. Run batch inspection:
```bash
python demo.py --mode batch
```

3. Check results in `results/` directory

### Custom Images

To test with your own images:

```bash
python defect_inspector.py --image your_image.jpg --output my_results
```

## 🔧 Advanced Usage

### Adjust Detection Sensitivity

For more sensitive detection (detects smaller defects):
```bash
python defect_inspector.py --image sample.jpg --sensitivity 0.8
```

For less sensitive detection (only major defects):
```bash
python defect_inspector.py --image sample.jpg --sensitivity 0.3
```

### Python API

Use the inspector programmatically:

```python
from defect_inspector import DefectInspector

# Initialize
inspector = DefectInspector(sensitivity=0.5)

# Analyze image
report = inspector.analyze_image('path/to/image.jpg')

# Visualize
inspector.visualize_defects('path/to/image.jpg', report, 'output.jpg')

# Save report
inspector.save_report(report, 'report.json')
```

## 📦 Dependencies

Main libraries:
- OpenCV (cv2): Image processing and computer vision
- NumPy: Numerical computations
- Matplotlib: Visualization
- Pillow: Image handling
- scikit-image: Advanced image processing

See `requirements.txt` for complete list with versions.

## 🖥️ Platform Compatibility

- **x86_64**: Full support (Windows, Linux, macOS)
- **ARM**: Full support (Raspberry Pi, Apple Silicon, etc.)
- All dependencies are platform-independent

## 📸 Sample Images

The `sample_images/` directory contains:

- `pcb_perfect.jpg`: Defect-free reference
- `pcb_defect_XX_*.jpg`: Various defect combinations
- `pcb_scratch_only.jpg`: Isolated scratch defects
- `pcb_missing_only.jpg`: Isolated missing component defects
- `pcb_misalign_only.jpg`: Isolated misalignment defects
- `pcb_discolor_only.jpg`: Isolated discoloration defects

All images include visible defects suitable for testing the inspection system.

## 🎥 Screen Recordings

Screen recordings demonstrating the system are available in the repository:
- `demo_single_inspection.mp4`: Single image inspection
- `demo_batch_processing.mp4`: Batch processing workflow
- `demo_visualization.mp4`: Visualization features

## 🔬 Technical Details

### Detection Algorithms

1. **Scratch Detection**:
   - Canny edge detection
   - Hough line transform
   - Length-based filtering

2. **Missing Component Detection**:
   - Otsu's thresholding
   - Morphological operations
   - Contour analysis

3. **Misalignment Detection**:
   - Adaptive thresholding
   - Shape circularity analysis
   - Aspect ratio evaluation

4. **Discoloration Detection**:
   - HSV color space analysis
   - Color range segmentation
   - Gradient-based filtering

### Performance

- Processing time: ~0.5-2 seconds per image (800x600)
- Accuracy: Depends on image quality and sensitivity settings
- Memory: ~100-200MB typical usage

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is provided as-is for educational and demonstration purposes.

## 👤 Author

Assignment 2 - Automated Quality Inspection System

## 📞 Support

For questions or issues, please open an issue in the GitHub repository.

---

**Note**: This is a prototype system designed for demonstration. For production use, additional calibration and training on real-world data would be recommended.
