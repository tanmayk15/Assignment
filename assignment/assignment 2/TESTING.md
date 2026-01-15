# Quality Inspection System - Testing Guide

## Quick Test Instructions

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Demo

The easiest way to test the system:

```bash
python quickstart.py
```

This will:
1. Generate sample PCB images with defects
2. Run batch inspection on all samples
3. Create annotated images and reports
4. Generate summary statistics

### 3. Check Results

After running quickstart:

- **Sample Images**: `sample_images/` directory
- **Annotated Results**: `results/` directory
- **Summary Report**: `results/summary_report.json`
- **Visualization**: `results/summary_visualization.png`

### 4. Test Individual Commands

#### Generate Samples
```bash
python generate_samples.py --output sample_images --num-samples 10
```

#### Inspect Single Image
```bash
python defect_inspector.py --image sample_images/pcb_defect_01_scratch_missing.jpg --output results
```

#### Batch Processing
```bash
python demo.py --mode batch --dir sample_images --output results
```

### 5. Understanding the Output

Each inspection produces:

1. **Annotated Image** (`*_annotated.jpg`):
   - Visual representation with bounding boxes
   - Color-coded by defect type
   - Shows center coordinates

2. **JSON Report** (`*_report.json`):
   ```json
   {
     "quality_status": "PASS/FAIL",
     "total_defects": 3,
     "defects": [
       {
         "type": "scratch",
         "center": [x, y],
         "bbox": [x, y, w, h],
         "severity": "MEDIUM",
         "confidence": 0.875
       }
     ]
   }
   ```

3. **Summary Report** (batch mode):
   - Overall pass/fail statistics
   - Defect type distribution
   - Severity breakdown

### 6. Expected Results

For a typical defective PCB:
- ✓ Detects 2-5 defects per image
- ✓ Confidence scores: 0.6-0.95
- ✓ Accurate center coordinates
- ✓ Proper severity classification

For perfect PCB:
- ✓ Zero defects detected
- ✓ Status: PASS

### 7. Screen Recording Checklist

When recording demos, show:

1. ✅ Running quickstart.py
2. ✅ Generated sample images
3. ✅ Annotated output images
4. ✅ JSON report content
5. ✅ Summary visualization
6. ✅ Single image inspection
7. ✅ Batch processing workflow

### 8. Troubleshooting

**Issue**: No defects detected
- Solution: Increase sensitivity `--sensitivity 0.8`

**Issue**: Too many false positives
- Solution: Decrease sensitivity `--sensitivity 0.3`

**Issue**: Import errors
- Solution: Reinstall dependencies `pip install -r requirements.txt`

### 9. Performance Benchmarks

Expected processing times:
- Image generation: ~0.1s per image
- Single inspection: ~0.5-2s per image
- Batch (10 images): ~10-20s total

### 10. Validation Tests

Run these to verify system works:

```bash
# Test 1: Perfect PCB (should detect 0 defects)
python defect_inspector.py --image sample_images/pcb_perfect.jpg --output test_results

# Test 2: Scratch only (should detect scratches)
python defect_inspector.py --image sample_images/pcb_scratch_only.jpg --output test_results

# Test 3: Batch processing
python demo.py --mode batch --dir sample_images --output test_results
```

### 11. API Usage Example

```python
from defect_inspector import DefectInspector

# Create inspector
inspector = DefectInspector(sensitivity=0.5)

# Analyze image
report = inspector.analyze_image('image.jpg')

# Access results
print(f"Status: {report['quality_status']}")
print(f"Defects: {report['total_defects']}")

for defect in report['defects']:
    print(f"Type: {defect['type']}")
    print(f"Position: {defect['center']}")
    print(f"Severity: {defect['severity']}")
```

### 12. Custom Image Testing

To test with your own images:

1. Place image in any directory
2. Run: `python defect_inspector.py --image /path/to/image.jpg --output my_results`
3. Check `my_results/` for output

### 13. Batch Custom Images

1. Place all images in a folder (e.g., `my_images/`)
2. Run: `python demo.py --mode batch --dir my_images --output my_results`
3. Check `my_results/` for all outputs and summary

---

## Submission Checklist

- ✅ All source code files
- ✅ requirements.txt
- ✅ README.md with instructions
- ✅ Sample images (defective + perfect)
- ✅ Screen recordings
- ✅ GitHub repository link
- ✅ Test results/output examples
