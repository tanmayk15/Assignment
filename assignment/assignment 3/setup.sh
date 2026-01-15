#!/bin/bash
# Setup script for Unix/Linux/macOS
# Custom VLM Design for Industrial Quality Inspection

echo "======================================================================"
echo "Custom VLM Setup for PCB Inspection - Unix/Linux/macOS"
echo "======================================================================"

# Check Python version
echo -e "\n[1/6] Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
    PIP_CMD=pip3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
    PIP_CMD=pip
else
    echo "  ✗ Python not found. Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo "  ✓ $PYTHON_VERSION"

# Create virtual environment
echo -e "\n[2/6] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "  Virtual environment already exists. Skipping..."
else
    $PYTHON_CMD -m venv venv
    echo "  ✓ Virtual environment created"
fi

# Activate virtual environment
echo -e "\n[3/6] Activating virtual environment..."
source venv/bin/activate
echo "  ✓ Virtual environment activated"

# Upgrade pip
echo -e "\n[4/6] Upgrading pip..."
$PIP_CMD install --upgrade pip setuptools wheel
echo "  ✓ Pip upgraded"

# Install dependencies
echo -e "\n[5/6] Installing dependencies..."
$PIP_CMD install -r requirements.txt
if [ $? -eq 0 ]; then
    echo "  ✓ Dependencies installed successfully"
else
    echo "  ✗ Failed to install dependencies"
    exit 1
fi

# Verify installation
echo -e "\n[6/6] Verifying installation..."
$PYTHON_CMD -c "import torch; print(f'  PyTorch: {torch.__version__}')"
$PYTHON_CMD -c "import transformers; print(f'  Transformers: {transformers.__version__}')"
$PYTHON_CMD -c "import cv2; print(f'  OpenCV: {cv2.__version__}')"
echo "  ✓ All core packages verified"

# Create necessary directories
echo -e "\nCreating project directories..."
for dir in data models outputs logs recordings; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "  ✓ Created $dir/"
    fi
done

# Check for GPU support
echo -e "\nChecking GPU support..."
GPU_CHECK=$($PYTHON_CMD -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>&1)
echo "  $GPU_CHECK"

echo -e "\n======================================================================"
echo "Setup Complete!"
echo "======================================================================"

echo -e "\nNext steps:"
echo "  1. Activate the environment: source venv/bin/activate"
echo "  2. Run model selection: python src/model_selection/vlm_comparison.py"
echo "  3. Test architecture: python src/architecture/custom_vlm.py"
echo "  4. Run optimization: python src/optimization/inference_optimizer.py"
echo "  5. Generate QA pairs: python src/training/qa_generator.py"
echo "  6. Run validation: python src/validation/metrics.py"

echo -e "\nFor Jupyter notebooks:"
echo "  jupyter notebook notebooks/"

echo ""
