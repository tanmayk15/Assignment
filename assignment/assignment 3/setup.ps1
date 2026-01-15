# Setup script for Windows (PowerShell)
# Custom VLM Design for Industrial Quality Inspection

Write-Host "=" -NoNewline; Write-Host ("=" * 69)
Write-Host "Custom VLM Setup for PCB Inspection - Windows"
Write-Host "=" -NoNewline; Write-Host ("=" * 69)

# Check Python version
Write-Host "`n[1/6] Checking Python version..."
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ $pythonVersion"
} else {
    Write-Host "  ✗ Python not found. Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "`n[2/6] Creating virtual environment..."
if (Test-Path "venv") {
    Write-Host "  Virtual environment already exists. Skipping..."
} else {
    python -m venv venv
    Write-Host "  ✓ Virtual environment created"
}

# Activate virtual environment
Write-Host "`n[3/6] Activating virtual environment..."
& ".\venv\Scripts\Activate.ps1"
Write-Host "  ✓ Virtual environment activated"

# Upgrade pip
Write-Host "`n[4/6] Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel
Write-Host "  ✓ Pip upgraded"

# Install dependencies
Write-Host "`n[5/6] Installing dependencies..."
pip install -r requirements.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Dependencies installed successfully"
} else {
    Write-Host "  ✗ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Verify installation
Write-Host "`n[6/6] Verifying installation..."
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'  Transformers: {transformers.__version__}')"
python -c "import cv2; print(f'  OpenCV: {cv2.__version__}')"
Write-Host "  ✓ All core packages verified"

# Create necessary directories
Write-Host "`nCreating project directories..."
$dirs = @("data", "models", "outputs", "logs", "recordings")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "  ✓ Created $dir/"
    }
}

# Check for GPU support
Write-Host "`nChecking GPU support..."
$gpuCheck = python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>&1
Write-Host "  $gpuCheck"

Write-Host "`n" -NoNewline
Write-Host "=" -NoNewline; Write-Host ("=" * 69)
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "=" -NoNewline; Write-Host ("=" * 69)

Write-Host "`nNext steps:"
Write-Host "  1. Activate the environment: .\venv\Scripts\Activate.ps1"
Write-Host "  2. Run model selection: python src\model_selection\vlm_comparison.py"
Write-Host "  3. Test architecture: python src\architecture\custom_vlm.py"
Write-Host "  4. Run optimization: python src\optimization\inference_optimizer.py"
Write-Host "  5. Generate QA pairs: python src\training\qa_generator.py"
Write-Host "  6. Run validation: python src\validation\metrics.py"

Write-Host "`nFor Jupyter notebooks:"
Write-Host "  jupyter notebook notebooks/"

Write-Host "`n" -NoNewline
