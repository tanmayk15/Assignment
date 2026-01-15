# Screen Recordings

This directory should contain screen recordings demonstrating all tasks for Assignment 3.

## Required Recordings

### 1. Model Selection (Part A)
**Filename**: `01_model_selection.mp4`
**Content**:
- Running the model comparison script
- Displaying the comparison table
- Showing the recommendation output
- Explaining the rationale for Qwen-VL selection

**Command**:
```bash
python src/model_selection/vlm_comparison.py
```

### 2. Architecture Implementation (Part B)
**Filename**: `02_architecture_demo.mp4`
**Content**:
- Running the custom VLM architecture
- Showing forward pass with test data
- Demonstrating multi-scale feature extraction
- Testing localization head outputs

**Command**:
```bash
python src/architecture/custom_vlm.py
```

### 3. Optimization Techniques (Part C)
**Filename**: `03_optimization_demo.mp4`
**Content**:
- Running quantization pipeline
- Showing model size reduction
- Benchmarking inference speed
- Demonstrating TensorRT/ONNX export

**Command**:
```bash
python src/optimization/inference_optimizer.py
```

### 4. QA Pair Generation (Part E)
**Filename**: `04_qa_generation.mp4`
**Content**:
- Running QA generator on sample data
- Showing different question types
- Displaying structured answers
- Demonstrating negative sample generation

**Command**:
```bash
python src/training/qa_generator.py
```

### 5. Validation Framework (Part F)
**Filename**: `05_validation_demo.mp4`
**Content**:
- Running comprehensive validation
- Showing counting accuracy metrics
- Demonstrating localization precision
- Displaying hallucination detection results
- Benchmarking inference speed

**Command**:
```bash
python src/validation/metrics.py
```

### 6. End-to-End Demo
**Filename**: `06_end_to_end.mp4`
**Content**:
- Complete workflow demonstration
- Loading a PCB image
- Asking natural language questions
- Showing structured responses with bounding boxes
- Demonstrating <2s inference time

**Command**:
```bash
python demo.py  # Main demo script
```

## Recording Guidelines

### Technical Requirements
- **Resolution**: 1920x1080 (1080p) or 1280x720 (720p)
- **Format**: MP4 (H.264 codec recommended)
- **Frame Rate**: 30 FPS
- **Duration**: 3-5 minutes per recording
- **Audio**: Clear narration explaining each step

### Content Guidelines
1. **Introduction** (30s)
   - State the task being demonstrated
   - Explain the objective

2. **Demonstration** (2-4 minutes)
   - Show commands being executed
   - Highlight key outputs
   - Explain important results
   - Point out metrics and performance

3. **Conclusion** (30s)
   - Summarize results
   - State whether targets were met

### Recommended Tools

#### Windows
- **OBS Studio** (Free): https://obsproject.com/
- **Camtasia**: Professional recording and editing
- **Windows Game Bar**: Built-in (Win + G)

#### macOS
- **QuickTime Player**: Built-in (File > New Screen Recording)
- **OBS Studio** (Free): https://obsproject.com/
- **ScreenFlow**: Professional tool

#### Linux
- **OBS Studio** (Free): https://obsproject.com/
- **SimpleScreenRecorder**: Lightweight option
- **Kazam**: Simple screen recorder

## Verification Checklist

Before submitting, ensure all recordings:
- [ ] Are properly named according to the convention
- [ ] Have clear video quality (readable text)
- [ ] Have clear audio narration
- [ ] Show actual execution of code (not just slides)
- [ ] Demonstrate the expected outputs
- [ ] Are uploaded to the GitHub repository or linked externally

## Hosting Options

If recordings are too large for GitHub (>100MB), use:

1. **Google Drive**
   - Upload videos to Google Drive
   - Set sharing to "Anyone with the link"
   - Add links to `RECORDING_LINKS.md`

2. **YouTube**
   - Upload as unlisted videos
   - Add links to `RECORDING_LINKS.md`

3. **Dropbox**
   - Share public links
   - Add to `RECORDING_LINKS.md`

## Example Recording Structure

```
recordings/
├── README.md (this file)
├── 01_model_selection.mp4
├── 02_architecture_demo.mp4
├── 03_optimization_demo.mp4
├── 04_qa_generation.mp4
├── 05_validation_demo.mp4
├── 06_end_to_end.mp4
└── RECORDING_LINKS.md (if using external hosting)
```

## Notes

- Keep recordings focused and concise
- Use zoom/highlighting to emphasize important details
- Include timestamps for key moments
- Test playback before submitting
- Compress videos if needed while maintaining quality

## Contact

For questions about recordings, please refer to the main README.md or create an issue in the repository.
