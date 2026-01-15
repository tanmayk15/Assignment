"""
Instructions for screen recording
"""

# SCREEN RECORDING INSTRUCTIONS

This file provides instructions for creating screen recordings of the project demonstrations.

## Required Screen Recordings

### 1. Training Demo (3-5 minutes)
**File:** `demo/01_training_demo.mp4`

**Steps to record:**
1. Open terminal in project root
2. Start recording (use OBS Studio, QuickTime, or Windows Game Bar)
3. Run training command:
   ```bash
   python tools/train.py \
       --config configs/resnet18_config.yaml \
       --data_dir dataset/ \
       --output_dir outputs/demo/ \
       --num_epochs 2
   ```
4. Show:
   - Training progress with loss values
   - Validation metrics
   - TensorBoard (open in browser: `tensorboard --logdir outputs/demo/tensorboard`)
   - Checkpoint saving
5. Stop recording

### 2. Evaluation Demo (2-3 minutes)
**File:** `demo/02_evaluation_demo.mp4`

**Steps to record:**
1. Start recording
2. Run evaluation:
   ```bash
   python tools/eval.py \
       --config configs/resnet18_config.yaml \
       --checkpoint outputs/demo/best_model.pth \
       --data_dir dataset/ \
       --split test
   ```
3. Show:
   - Evaluation progress
   - mAP results per class
   - Overall metrics
4. Stop recording

### 3. Inference Demo (3-4 minutes)
**File:** `demo/03_inference_demo.mp4`

**Steps to record:**
1. Start recording
2. Single image inference:
   ```bash
   python tools/inference.py \
       --checkpoint outputs/demo/best_model.pth \
       --image demo/test_image.jpg \
       --output demo/result.jpg \
       --show
   ```
3. Show the detection results
4. Video inference:
   ```bash
   python tools/video_inference.py \
       --checkpoint outputs/demo/best_model.pth \
       --video demo/test_video.mp4 \
       --output demo/result_video.mp4
   ```
5. Play the output video
6. Stop recording

### 4. Webcam Demo (1-2 minutes)
**File:** `demo/04_webcam_demo.mp4`

**Steps to record:**
1. Start recording (record both terminal and webcam window)
2. Run webcam demo:
   ```bash
   python tools/webcam_demo.py \
       --checkpoint outputs/demo/best_model.pth
   ```
3. Show:
   - Real-time detections
   - FPS counter
   - Detection confidence scores
   - Multiple object classes
4. Press 'q' to quit
5. Stop recording

### 5. Model Export Demo (1-2 minutes)
**File:** `demo/05_export_demo.mp4`

**Steps to record:**
1. Start recording
2. Export to ONNX:
   ```bash
   python tools/export_onnx.py \
       --config configs/resnet18_config.yaml \
       --checkpoint outputs/demo/best_model.pth \
       --output detector.onnx \
       --simplify
   ```
3. Show:
   - Export progress
   - ONNX model verification
   - Model information
4. Stop recording

## Recording Software Recommendations

### Windows
- **OBS Studio** (Free): https://obsproject.com/
- **Xbox Game Bar** (Built-in): Win + G
- **Camtasia** (Paid): https://www.techsmith.com/video-editor.html

### macOS
- **QuickTime Player** (Built-in): File > New Screen Recording
- **OBS Studio** (Free): https://obsproject.com/
- **ScreenFlow** (Paid): https://www.telestream.net/screenflow/

### Linux
- **OBS Studio** (Free): https://obsproject.com/
- **SimpleScreenRecorder**: https://www.maartenbaert.be/simplescreenrecorder/
- **Kazam**: https://launchpad.net/kazam

## Recording Settings

- **Resolution:** 1920x1080 (Full HD)
- **Frame Rate:** 30 FPS
- **Format:** MP4 (H.264 codec)
- **Audio:** Optional (can add voice-over explaining the process)

## Tips for Good Recordings

1. **Clear terminal:** Use `clear` command before starting
2. **Font size:** Increase terminal font for better readability
3. **Hide distractions:** Close unnecessary applications
4. **Smooth cursor:** Don't move mouse erratically
5. **Pause:** Stop and think before typing commands
6. **Clean output:** Remove verbose debug messages if possible

## Post-Recording

1. **Trim:** Remove any unnecessary beginning/ending
2. **Add annotations:** Use video editing software to add:
   - Text overlays explaining key steps
   - Arrows pointing to important information
   - Captions for accessibility
3. **Compress:** Use HandBrake or similar to reduce file size while maintaining quality
4. **Upload:** Upload to GitHub repository in `demo/` folder or provide Google Drive link

## Example File Structure

After creating recordings, your demo folder should look like:

```
demo/
├── 01_training_demo.mp4
├── 02_evaluation_demo.mp4
├── 03_inference_demo.mp4
├── 04_webcam_demo.mp4
├── 05_export_demo.mp4
├── test_image.jpg
├── test_video.mp4
├── result.jpg
├── result_video.mp4
└── README.md (this file)
```

## Troubleshooting

**Issue:** Video file too large
- **Solution:** Use HandBrake to compress (target: 50-100 MB per video)

**Issue:** FPS too low during webcam demo
- **Solution:** Use smaller input resolution or skip frames

**Issue:** Audio not recording
- **Solution:** Check microphone settings in recording software

**Issue:** Screen flickering
- **Solution:** Disable screen effects, set fixed refresh rate

## Notes

- Total recording time: ~15-20 minutes
- Combined file size: ~500 MB (before compression)
- Recommended: Create a short 2-minute highlights video showing best parts

---

For questions or issues, please refer to the main README.md or open an issue on GitHub.
