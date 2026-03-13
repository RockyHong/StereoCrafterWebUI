#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "============================================"
echo " StereoCrafter WebUI - Launcher"
echo "============================================"
echo

# ── Step 1/6: Check Python ───────────────────
echo "[1/6] Checking Python..."
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "       [FAILED] Python not found. Please install Python 3.8+."
    exit 1
fi
echo "       [OK] $($PYTHON --version)"
echo

# ── Step 2/6: Create venv ────────────────────
echo "[2/6] Checking virtual environment..."
if [ -d "venv" ]; then
    echo "       [OK] venv already exists"
else
    echo "       Creating venv..."
    $PYTHON -m venv venv
    echo "       [OK] venv created"
fi
echo

# ── Activate venv ─────────────────────────────
source venv/bin/activate

# ── Step 3/6: Install PyTorch with CUDA ──────
echo "[3/6] Checking PyTorch..."
if [ -f "venv/.torch_installed" ]; then
    echo "       [OK] PyTorch already installed"
else
    echo "       Detecting GPU and CUDA version..."
    CUDA_TAG=$(python detect_cuda.py)
    echo "       Detected: $CUDA_TAG"
    echo
    if [ "$CUDA_TAG" = "cpu" ]; then
        echo "       [WARNING] No NVIDIA GPU detected. Installing CPU-only PyTorch."
        pip install torch torchvision
    else
        echo "       Installing PyTorch for $CUDA_TAG..."
        echo
        pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
        echo
        echo "       Installing xformers..."
        pip install xformers || echo "       [WARNING] xformers failed. Optional, continuing..."
    fi
    touch "venv/.torch_installed"
    echo
    echo "       [OK] PyTorch installed"
fi
echo

# ── Step 4/6: Install other dependencies ─────
echo "[4/6] Checking dependencies..."
if [ -f "venv/.webui_installed" ]; then
    echo "       [OK] Dependencies already installed"
else
    echo "       Installing remaining packages..."
    echo
    pip install -r requirements_webui.txt
    touch "venv/.webui_installed"
    echo
    echo "       [OK] Dependencies installed"
fi
echo

# ── Step 5/6: Compile Forward-Warp ───────────
echo "[5/6] Checking Forward-Warp CUDA module..."
if [ -f "venv/.forwardwarp_installed" ]; then
    echo "       [OK] Forward-Warp already compiled"
else
    echo "       Step 1: Compiling CUDA kernel..."
    echo
    pushd dependency/Forward-Warp/Forward_Warp/cuda
    python setup.py install
    popd
    echo
    echo "       Step 2: Installing Python wrapper..."
    pushd dependency/Forward-Warp
    pip install .
    popd
    touch "venv/.forwardwarp_installed"
    echo
    echo "       [OK] Forward-Warp compiled"
fi
echo

# ── Step 6/6: Launch ─────────────────────────
echo "[6/6] Launching StereoCrafter WebUI..."
echo "       Keep this terminal open while using the app."
echo "       Press Ctrl+C to stop the server."
echo
echo "============================================"
echo
python app.py
