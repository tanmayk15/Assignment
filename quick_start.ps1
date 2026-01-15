# QUICK START SCRIPT
# Run this script to set up and test everything quickly

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "OBJECT DETECTION ASSIGNMENT - QUICK START" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "[1/7] Checking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Python found: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "  ✗ Python not found! Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Create virtual environment if not exists
Write-Host ""
Write-Host "[2/7] Setting up virtual environment..." -ForegroundColor Yellow
if (!(Test-Path "venv")) {
    python -m venv venv
    Write-Host "  ✓ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "  ✓ Virtual environment already exists" -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "[3/7] Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"
Write-Host "  ✓ Virtual environment activated" -ForegroundColor Green

# Install requirements
Write-Host ""
Write-Host "[4/7] Installing dependencies..." -ForegroundColor Yellow
Write-Host "  (This may take a few minutes)" -ForegroundColor Gray
pip install -r requirements.txt --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "  ✗ Error installing dependencies" -ForegroundColor Red
    exit 1
}

# Test model
Write-Host ""
Write-Host "[5/7] Testing model architecture..." -ForegroundColor Yellow
python test_model.py
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Model test passed" -ForegroundColor Green
} else {
    Write-Host "  ✗ Model test failed" -ForegroundColor Red
    exit 1
}

# Create necessary folders
Write-Host ""
Write-Host "[6/7] Creating output directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "outputs" | Out-Null
New-Item -ItemType Directory -Force -Path "demo/test_images" | Out-Null
New-Item -ItemType Directory -Force -Path "submissions/screenshots" | Out-Null
New-Item -ItemType Directory -Force -Path "submissions/recordings" | Out-Null
Write-Host "  ✓ Directories created" -ForegroundColor Green

# Check dataset
Write-Host ""
Write-Host "[7/7] Checking dataset..." -ForegroundColor Yellow
if (Test-Path "dataset/images") {
    Write-Host "  ✓ Dataset found" -ForegroundColor Green
} else {
    Write-Host "  ⚠ Dataset not found" -ForegroundColor Yellow
    Write-Host "  Run this to download:" -ForegroundColor Gray
    Write-Host "  python scripts/download_dataset.py --output_dir dataset/" -ForegroundColor Gray
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "✓ SETUP COMPLETE!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "1. Download dataset: python scripts/download_dataset.py --output_dir dataset/" -ForegroundColor Gray
Write-Host "2. Prepare data: python scripts/prepare_data.py --data_dir dataset/VOCdevkit/VOC2012/ --output_dir dataset/" -ForegroundColor Gray
Write-Host "3. Train model: python tools/train.py --config configs/resnet18_config.yaml --data_dir dataset/ --output_dir outputs/demo/ --num_epochs 2" -ForegroundColor Gray
Write-Host ""
Write-Host "For complete instructions, see: EXECUTION_GUIDE.md" -ForegroundColor Cyan
