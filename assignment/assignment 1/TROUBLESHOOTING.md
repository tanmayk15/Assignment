# 🔧 TROUBLESHOOTING GUIDE

Common issues and solutions when running the assignment.

---

## Installation Issues

### Issue: "pip not recognized"
**Solution:**
```powershell
# Add Python to PATH, or use full path:
python -m pip install -r requirements.txt
```

### Issue: "Execution policy error" when activating venv
**Error:** `cannot be loaded because running scripts is disabled`

**Solution:**
```powershell
# Run as Administrator (temporarily for this session only):
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate again:
.\venv\Scripts\Activate.ps1
```

### Issue: PyTorch installation fails
**Solution:**
```powershell
# Install CPU version explicitly:
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
```

---

## Dataset Issues

### Issue: Dataset download is very slow
**Solution:**
- Use a download manager
- Or download manually from: http://host.robots.ox.ac.uk/pascal/VOC/
- Extract to `dataset/` folder

### Issue: "Dataset not found" error
**Check:**
```powershell
# Verify directory structure:
dir dataset\

# Should have:
# dataset\images\train\
# dataset\images\val\
# dataset\images\test\
# dataset\annotations\train\
# dataset\annotations\val\
# dataset\annotations\test\
```

**Fix:**
```powershell
# Re-run data preparation:
python scripts/prepare_data.py --data_dir dataset/VOCdevkit/VOC2012/ --output_dir dataset/
```

### Issue: "No images found"
**Solution:**
- Check that images are in correct folders
- Verify file extensions (.jpg, .jpeg, .png)
- Check annotations folder has corresponding .xml files

---

## Training Issues

### Issue: CUDA out of memory
**Solution:**
```powershell
# Option 1: Reduce batch size
python tools/train.py --config configs/resnet18_config.yaml --data_dir dataset/ --output_dir outputs/ --batch_size 2

# Option 2: Train on CPU
python tools/train.py --config configs/resnet18_config.yaml --data_dir dataset/ --output_dir outputs/ --device cpu
```

### Issue: Training is too slow on CPU
**Expected:**
- CPU training: ~30-60 minutes per epoch
- GPU training: ~3-5 minutes per epoch

**Solution:**
- For demo purposes, use just 2 epochs
- If you have GPU, make sure CUDA is properly installed
- Consider using Google Colab (free GPU)

### Issue: "No module named 'models'"
**Solution:**
```powershell
# Make sure you're in the project root directory
cd "c:\Users\ACER\Desktop\Assignment main\assignment 1"

# And virtual environment is activated
.\venv\Scripts\Activate.ps1
```

### Issue: Loss is NaN or not decreasing
**Possible causes:**
- Learning rate too high
- Dataset issues
- Batch size too small

**Solution:**
- Check dataset is correctly loaded
- Try reducing learning rate in config
- Increase batch size if possible

---

## Inference Issues

### Issue: "Checkpoint not found"
**Solution:**
```powershell
# Check if model was saved during training
dir outputs\demo\

# Should see best_model.pth or checkpoint_epoch_XX.pth
# Use the correct path:
python tools/inference.py --checkpoint outputs\demo\best_model.pth --image test.jpg
```

### Issue: "Image not found"
**Solution:**
```powershell
# Use absolute path or verify relative path:
python tools/inference.py --checkpoint outputs\demo\best_model.pth --image "c:\Users\ACER\Desktop\test.jpg"
```

### Issue: No detections / bounding boxes
**Possible reasons:**
- Model not trained enough (only 2 epochs for demo)
- Confidence threshold too high
- Objects not in training classes

**Solution:**
- Train for more epochs (25+ for good results)
- Lower confidence threshold in config
- Use images with objects from PASCAL VOC classes

---

## Webcam Issues

### Issue: "Cannot access webcam"
**Solution:**
```powershell
# Check if camera is being used by another app
# Close Zoom, Skype, Teams, etc.

# Try specifying camera index:
python tools/webcam_demo.py --checkpoint outputs\demo\best_model.pth --camera 0

# Or try camera 1:
python tools/webcam_demo.py --checkpoint outputs\demo\best_model.pth --camera 1
```

### Issue: Very low FPS (< 5 FPS)
**Solution:**
- Expected on CPU: 8-15 FPS
- Try reducing input size in config
- Close other applications
- Use MobileNet config for faster inference:
```powershell
python tools/webcam_demo.py --checkpoint outputs\demo\mobilenet_best.pth --config configs/mobilenet_config.yaml
```

---

## Recording Issues

### Issue: Game Bar won't record terminal
**Solution:**
- Game Bar is designed for games
- Use OBS Studio instead
- Or use ShareX (better for terminal recording)

### Issue: Video files are too large for GitHub
**Solutions:**

**Option 1: Compress videos**
```powershell
# Using ffmpeg (install from ffmpeg.org):
ffmpeg -i input.mp4 -vcodec libx264 -crf 28 output.mp4
```

**Option 2: Upload to external hosting**
- YouTube (unlisted)
- Google Drive
- Dropbox
- Then link in README

**Option 3: Git LFS**
```powershell
git lfs install
git lfs track "*.mp4"
git add .gitattributes
```

### Issue: Screenshots are not clear
**Solution:**
- Use PNG format (not JPG)
- Don't compress too much
- Take full screen screenshots
- Make sure text is readable

---

## GitHub Issues

### Issue: "git not recognized"
**Solution:**
```powershell
# Install Git from: https://git-scm.com/download/win
# Then restart PowerShell
```

### Issue: Push rejected - file too large
**Solution:**
```powershell
# Remove large file from tracking:
git rm --cached large_file.pth

# Add to .gitignore:
echo "*.pth" >> .gitignore

# Commit and push again:
git add .gitignore
git commit -m "Remove large files"
git push
```

### Issue: Authentication failed
**Solution:**
```powershell
# Use Personal Access Token instead of password
# Generate token at: https://github.com/settings/tokens
# Use token as password when pushing
```

---

## Model Export Issues

### Issue: ONNX export fails
**Solution:**
```powershell
# Make sure ONNX is installed:
pip install onnx onnxruntime

# Try with CPU device:
python tools/export_onnx.py --checkpoint outputs\demo\best_model.pth --output detector.onnx --device cpu
```

---

## Python Environment Issues

### Issue: Wrong Python version
**Check version:**
```powershell
python --version
# Should be 3.8 or higher
```

**Solution:**
- Install Python 3.8+ from python.org
- Or use Anaconda

### Issue: Packages not found after installation
**Solution:**
```powershell
# Make sure venv is activated (should see (venv) in prompt)
# If not:
.\venv\Scripts\Activate.ps1

# Reinstall:
pip install -r requirements.txt
```

---

## Performance Issues

### Low mAP (< 50%)
**Reasons:**
- Only trained for 2 epochs (for demo)
- Small dataset
- Need more training

**Expected:**
- 2 epochs: ~30-40% mAP (for demo only)
- 10 epochs: ~55-65% mAP
- 25+ epochs: ~70-75% mAP

### Slow training
**Tips:**
- Close other applications
- Use GPU if available
- Reduce batch size
- Use MobileNet instead of ResNet-18

---

## Getting Help

### Before asking:
1. ✅ Check this troubleshooting guide
2. ✅ Read error messages carefully
3. ✅ Verify file paths are correct
4. ✅ Make sure venv is activated
5. ✅ Check you're in correct directory

### Where to get help:
- GitHub Issues (for code problems)
- PyTorch forums (for PyTorch issues)
- Stack Overflow (for general Python questions)

### When reporting issues:
- Include full error message
- Mention your OS and Python version
- Show the command you ran
- Describe what you expected vs what happened

---

## Quick Fixes Summary

| Issue | Quick Fix |
|-------|-----------|
| CUDA out of memory | `--batch_size 2 --device cpu` |
| Slow training | Use 2 epochs for demo, or try MobileNet |
| Can't activate venv | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| Git file too large | Add to `.gitignore`, use Git LFS, or external hosting |
| No detections | Train more epochs, lower threshold |
| Webcam not working | Close other apps, try `--camera 1` |
| Module not found | Activate venv, `pip install -r requirements.txt` |

---

## Still Having Issues?

1. **Try the quick start script:**
   ```powershell
   .\quick_start.ps1
   ```

2. **Re-read the EXECUTION_GUIDE.md**

3. **Check you completed all prerequisites**

4. **Try on a different machine** (if available)

5. **Use Google Colab** (free GPU for training)

---

## Emergency Mode: Google Colab

If nothing works locally, use Google Colab (free):

1. Go to: https://colab.research.google.com/
2. Upload your project files
3. Run in Colab notebook:
```python
!pip install -r requirements.txt
!python tools/train.py --config configs/resnet18_config.yaml --data_dir dataset/ --output_dir outputs/demo/ --num_epochs 2
```

4. Download results and screenshots from Colab

---

**Remember:** For the assignment demo, 2 epochs is acceptable! The company wants to see that you can run the code, not achieve perfect accuracy.
