# GitHub Repository Setup Guide

## Quick Setup

### 1. Initialize Git Repository

```bash
cd "assignment 3"
git init
git add .
git commit -m "Initial commit: Custom VLM Design for PCB Inspection"
```

### 2. Create GitHub Repository

1. Go to [GitHub](https://github.com/new)
2. Create a new repository named: `custom-vlm-pcb-inspection`
3. Keep it public
4. Don't initialize with README (we already have one)

### 3. Push to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/custom-vlm-pcb-inspection.git
git branch -M main
git push -u origin main
```

## Repository Structure

Your GitHub repository will contain:

```
custom-vlm-pcb-inspection/
├── README.md                    # Main documentation
├── SOLUTION.md                  # Comprehensive solution document
├── requirements.txt             # Python dependencies
├── setup.sh                     # Unix/Linux setup script
├── setup.ps1                    # Windows setup script
├── demo.py                      # End-to-end demonstration
├── run_all.py                   # Run all demonstrations
├── src/                         # Source code
│   ├── model_selection/
│   │   └── vlm_comparison.py
│   ├── architecture/
│   │   └── custom_vlm.py
│   ├── optimization/
│   │   └── inference_optimizer.py
│   ├── training/
│   │   └── qa_generator.py
│   └── validation/
│       └── metrics.py
└── recordings/                  # Screen recordings
    ├── README.md
    └── [video files or links]
```

## Adding Screen Recordings

### Option 1: Direct Upload (if <100MB each)

```bash
git add recordings/*.mp4
git commit -m "Add screen recordings"
git push
```

### Option 2: External Hosting (recommended for large files)

1. Upload videos to Google Drive, YouTube, or Dropbox
2. Create `recordings/RECORDING_LINKS.md`:

```markdown
# Screen Recording Links

## Part A: Model Selection
- **YouTube**: https://youtu.be/YOUR_VIDEO_ID
- **Duration**: 4:32

## Part B: Architecture Implementation
- **YouTube**: https://youtu.be/YOUR_VIDEO_ID
- **Duration**: 5:15

[... etc ...]
```

3. Commit and push:

```bash
git add recordings/RECORDING_LINKS.md
git commit -m "Add screen recording links"
git push
```

## GitHub Repository Best Practices

### 1. Add a .gitignore

Create `.gitignore`:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
*.egg-info/
dist/
build/

# Data
data/
*.h5
*.pkl
*.pth
*.onnx
*.trt

# Outputs
outputs/
logs/
models/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Large files (if not using LFS)
*.mp4
*.avi
*.mov
```

### 2. Add GitHub Actions (Optional)

Create `.github/workflows/test.yml`:

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run tests
      run: |
        python run_all.py
```

### 3. Add License

Create `LICENSE`:

```
MIT License

Copyright (c) 2026 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

## Final Checklist

Before submission, ensure:

- [ ] All code is committed and pushed
- [ ] README.md is comprehensive
- [ ] SOLUTION.md contains all answers
- [ ] requirements.txt is complete
- [ ] Setup scripts work on both platforms
- [ ] All demonstrations run successfully
- [ ] Screen recordings are accessible
- [ ] Repository is public
- [ ] GitHub URL is documented

## Example GitHub URL

Your submission should include:

```
GitHub Repository: https://github.com/YOUR_USERNAME/custom-vlm-pcb-inspection
```

## Viewing Instructions

To clone and run the project:

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/custom-vlm-pcb-inspection.git
cd custom-vlm-pcb-inspection

# Setup (Windows)
.\setup.ps1

# Setup (Unix/Linux/macOS)
chmod +x setup.sh
./setup.sh

# Run all demonstrations
python run_all.py

# Or run individual components
python src/model_selection/vlm_comparison.py
python src/architecture/custom_vlm.py
python src/optimization/inference_optimizer.py
python src/training/qa_generator.py
python src/validation/metrics.py
python demo.py
```

## Support

For issues or questions:
1. Check README.md documentation
2. Review SOLUTION.md for detailed explanations
3. Run individual scripts to isolate problems
4. Check GitHub Issues for similar problems

---

**Note**: Replace `YOUR_USERNAME` with your actual GitHub username in all commands.
