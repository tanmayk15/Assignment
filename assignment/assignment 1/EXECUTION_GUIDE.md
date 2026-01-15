# 📋 COMPLETE EXECUTION GUIDE

This guide will help you execute all tasks and create screen recordings/screenshots for your assignment submission.

---

## 🎯 ASSIGNMENT CHECKLIST

- [ ] Setup environment
- [ ] Download dataset
- [ ] Test model architecture
- [ ] Train model (at least 2-5 epochs for demo)
- [ ] Evaluate model
- [ ] Run inference on images
- [ ] Run video inference
- [ ] Run webcam demo
- [ ] Export to ONNX
- [ ] Record all demos
- [ ] Take screenshots
- [ ] Upload to GitHub

---

## 📦 PART 1: INITIAL SETUP

### Step 1: Install Python Dependencies

```powershell
# Open PowerShell in this directory
cd "c:\Users\ACER\Desktop\Assignment main\assignment 1"

# Create virtual environment
python -m venv venv

# Activate virtual environment (PowerShell)
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate again
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

**SCREENSHOT 1:** Take screenshot of terminal showing successful package installation

---

### Step 2: Download Dataset

```powershell
# Download PASCAL VOC dataset (this will take 5-10 minutes)
python scripts/download_dataset.py --output_dir dataset/ --dataset voc2012

# Prepare data for training
python scripts/prepare_data.py --data_dir dataset/VOCdevkit/VOC2012/ --output_dir dataset/
```

**SCREENSHOT 2:** Take screenshot showing dataset download completion

---

## 🧪 PART 2: TEST MODEL ARCHITECTURE

### Step 3: Quick Model Test

```powershell
# Test that model builds correctly
python test_model.py
```

**SCREENSHOT 3:** Take screenshot showing "ALL TESTS PASSED" message

---

## 🚀 PART 3: TRAINING (WITH RECORDING)

### Step 4: Start Training

**🎥 START SCREEN RECORDING NOW** (See recording instructions below)

```powershell
# Train for 2 epochs (for demo purposes - quick)
python tools/train.py --config configs/resnet18_config.yaml --data_dir dataset/ --output_dir outputs/demo_run/ --num_epochs 2

# For better results, train for 25 epochs (will take hours)
# python tools/train.py --config configs/resnet18_config.yaml --data_dir dataset/ --output_dir outputs/full_training/ --num_epochs 25
```

**🎥 STOP RECORDING** after training completes

**What to show in recording:**
- Training command execution
- Training progress with loss values
- Validation metrics
- Checkpoint saving messages

**SCREENSHOT 4:** Final training metrics and loss values

---

## 📊 PART 4: EVALUATION

### Step 5: Evaluate Model

**🎥 START SCREEN RECORDING**

```powershell
# Evaluate on test set
python tools/eval.py --config configs/resnet18_config.yaml --checkpoint outputs/demo_run/best_model.pth --data_dir dataset/ --split test
```

**🎥 STOP RECORDING**

**SCREENSHOT 5:** Evaluation results showing mAP scores

---

## 🔍 PART 5: INFERENCE DEMOS

### Step 6: Single Image Inference

First, you need a test image. Download one:

```powershell
# Create demo folder
mkdir demo\test_images -Force

# Download a test image (or use your own)
# Option 1: Use PowerShell to download
Invoke-WebRequest -Uri "https://images.unsplash.com/photo-1449965408869-eaa3f722e40d?w=800" -OutFile "demo\test_images\test_image.jpg"

# Option 2: Just copy any image to demo\test_images\test_image.jpg manually
```

**🎥 START SCREEN RECORDING**

```powershell
# Run inference
python tools/inference.py --checkpoint outputs/demo_run/best_model.pth --image demo\test_images\test_image.jpg --output demo\result.jpg --show
```

**🎥 STOP RECORDING** (show the detection result image)

**SCREENSHOT 6:** Detection result with bounding boxes

---

### Step 7: Video Inference (Optional - if you have test video)

```powershell
# If you have a video file
python tools/video_inference.py --checkpoint outputs/demo_run/best_model.pth --video demo\test_video.mp4 --output demo\result_video.mp4
```

**🎥 RECORD:** The output video playing

---

### Step 8: Webcam Demo

**🎥 START SCREEN RECORDING** (record both terminal and webcam window)

```powershell
# Run real-time detection on webcam
python tools/webcam_demo.py --checkpoint outputs/demo_run/best_model.pth
```

Press 'q' to quit when done.

**🎥 STOP RECORDING**

**SCREENSHOT 7:** Webcam demo showing real-time detections with FPS counter

---

## 🎁 PART 6: MODEL EXPORT

### Step 9: Export to ONNX

**🎥 START SCREEN RECORDING**

```powershell
# Export model to ONNX format
python tools/export_onnx.py --checkpoint outputs/demo_run/best_model.pth --output detector.onnx
```

**🎥 STOP RECORDING**

**SCREENSHOT 8:** ONNX export success message

---

## 📹 SCREEN RECORDING TOOLS (Windows)

### Option 1: Windows Game Bar (Built-in, Easiest)

1. **Start Recording:**
   - Press `Win + G` to open Game Bar
   - Click the record button (circle icon) or press `Win + Alt + R`
   - A small recording widget appears

2. **Stop Recording:**
   - Press `Win + Alt + R` again
   - Videos saved to: `C:\Users\ACER\Videos\Captures\`

3. **Settings:**
   - Open Xbox Game Bar settings
   - Go to "Captures"
   - Set video quality to "High"

### Option 2: OBS Studio (Professional, Free)

1. **Download:**
   - Visit: https://obsproject.com/
   - Download and install OBS Studio

2. **Setup:**
   - Open OBS
   - Add source: "Display Capture" or "Window Capture"
   - Click "Start Recording"

3. **Stop:**
   - Click "Stop Recording"
   - Find video in: Videos folder

### Option 3: ShareX (Feature-rich, Free)

1. **Download:**
   - Visit: https://getsharex.com/
   - Download and install

2. **Record:**
   - Press `Shift + Print Screen`
   - Select "Screen Record"
   - Select area to record

---

## 📸 TAKING SCREENSHOTS (Windows)

### Method 1: Snipping Tool (Built-in)

1. Press `Win + Shift + S`
2. Select area to capture
3. Screenshot copied to clipboard
4. Paste into Paint/Word and save

### Method 2: Print Screen

1. Press `Print Screen` key (captures full screen)
2. Open Paint
3. Paste (`Ctrl + V`)
4. Save as PNG/JPG

### Method 3: ShareX (Recommended)

1. Install ShareX
2. Press `Print Screen` for full screen
3. Or `Ctrl + Print Screen` for region
4. Auto-saves to Pictures\ShareX folder

---

## 📂 ORGANIZING YOUR SUBMISSIONS

### Create Screenshots Folder

```powershell
# Create organized folder structure
mkdir submissions\screenshots -Force
mkdir submissions\recordings -Force
mkdir submissions\outputs -Force
```

### Naming Convention

**Screenshots:**
- `01_installation_complete.png`
- `02_dataset_downloaded.png`
- `03_model_test_passed.png`
- `04_training_complete.png`
- `05_evaluation_results.png`
- `06_inference_result.png`
- `07_webcam_demo.png`
- `08_onnx_export.png`

**Recordings:**
- `01_training_demo.mp4`
- `02_evaluation_demo.mp4`
- `03_inference_demo.mp4`
- `04_webcam_demo.mp4`
- `05_onnx_export_demo.mp4`

---

## 🌐 UPLOADING TO GITHUB

### Step 1: Create GitHub Repository

1. Go to https://github.com
2. Click "New Repository"
3. Name: `object-detection-assignment`
4. Make it Public
5. Don't initialize with README (you already have one)
6. Click "Create repository"

### Step 2: Push Your Code

```powershell
# Navigate to your project
cd "c:\Users\ACER\Desktop\Assignment main\assignment 1"

# Initialize git (if not already done)
git init

# Add all files (except large files)
git add .

# Create .gitignore to exclude large files
# (see next section)

# Commit
git commit -m "Initial commit: Object detection assignment"

# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/object-detection-assignment.git

# Push
git branch -M main
git push -u origin main
```

### Step 3: Upload Screenshots and Recordings

**For small files (<25MB):**
```powershell
# Add screenshots
git add submissions/screenshots/
git add submissions/recordings/
git commit -m "Add screenshots and recordings"
git push
```

**For large video files (>25MB):**

Use GitHub Releases or Git LFS:

```powershell
# Option 1: Git LFS (Large File Storage)
git lfs install
git lfs track "*.mp4"
git add .gitattributes
git add submissions/recordings/
git commit -m "Add video recordings"
git push

# Option 2: Upload to Google Drive and add link in README
```

Or create a GitHub Release:
1. Go to your repo on GitHub
2. Click "Releases" → "Create a new release"
3. Upload large video files as release assets

---

## 📝 UPDATE README WITH LINKS

Add this section to your README.md:

```markdown
## 📸 Demo Screenshots

| Task | Screenshot |
|------|-----------|
| Setup | ![Setup](submissions/screenshots/01_installation_complete.png) |
| Training | ![Training](submissions/screenshots/04_training_complete.png) |
| Evaluation | ![Evaluation](submissions/screenshots/05_evaluation_results.png) |
| Inference | ![Inference](submissions/screenshots/06_inference_result.png) |
| Webcam | ![Webcam](submissions/screenshots/07_webcam_demo.png) |

## 🎥 Demo Videos

- [Training Demo](submissions/recordings/01_training_demo.mp4)
- [Inference Demo](submissions/recordings/03_inference_demo.mp4)
- [Webcam Demo](submissions/recordings/04_webcam_demo.mp4)
```

---

## ⚠️ IMPORTANT NOTES

### File Size Warnings

- **GitHub free limit:** 100MB per file, 1GB per repo
- **Solutions for large files:**
  - Compress videos before uploading
  - Use Git LFS
  - Upload to Google Drive/Dropbox and link in README
  - Use GitHub Releases for large files

### Don't Upload These

Create a `.gitignore` file:

```
# Large files
*.pth
*.onnx
dataset/
outputs/
venv/
*.pyc
__pycache__/

# Keep only demo recordings
!submissions/recordings/*.mp4
```

### What TO Upload

- ✅ Source code (all .py files)
- ✅ Configuration files
- ✅ README.md
- ✅ requirements.txt
- ✅ Screenshots (optimized)
- ✅ Small demo videos (<25MB or compressed)
- ✅ This execution guide

---

## 🎬 FINAL CHECKLIST BEFORE SUBMISSION

- [ ] All code files are in GitHub repository
- [ ] README.md is complete and professional
- [ ] At least 8 screenshots uploaded
- [ ] At least 3-5 screen recordings uploaded
- [ ] Links to screenshots/videos in README
- [ ] Repository is PUBLIC
- [ ] GitHub link is ready to share with company

---

## 📧 WHAT TO SEND TO COMPANY

Send an email with:

```
Subject: Object Detection Assignment Submission - [Your Name]

Dear [Hiring Manager],

I have completed the object detection assignment. Please find my submission at:

GitHub Repository: https://github.com/YOUR_USERNAME/object-detection-assignment

The repository includes:
✅ Complete source code for Faster R-CNN implementation
✅ Training and evaluation scripts
✅ Screenshots of all tasks (in submissions/screenshots/)
✅ Screen recordings of demos (in submissions/recordings/)
✅ Technical documentation

Key Results:
- Model trained from scratch (no pretrained weights)
- Achieved [XX]% mAP on test set
- Real-time inference at [XX] FPS
- Successfully exported to ONNX format

All tasks have been completed as per requirements.

Thank you for the opportunity!

Best regards,
[Your Name]
```

---

## 🆘 TROUBLESHOOTING

### Issue: CUDA not available
```powershell
# Train on CPU instead
python tools/train.py --config configs/resnet18_config.yaml --data_dir dataset/ --output_dir outputs/demo_run/ --num_epochs 2 --device cpu
```

### Issue: Out of memory
```powershell
# Reduce batch size
python tools/train.py --config configs/resnet18_config.yaml --data_dir dataset/ --output_dir outputs/demo_run/ --num_epochs 2 --batch_size 2
```

### Issue: Dataset not found
Make sure you ran:
```powershell
python scripts/download_dataset.py --output_dir dataset/
python scripts/prepare_data.py --data_dir dataset/VOCdevkit/VOC2012/ --output_dir dataset/
```

---

## 🎉 GOOD LUCK!

Follow this guide step by step, and you'll have a complete, professional submission!
