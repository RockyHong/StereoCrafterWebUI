@echo off
title StereoCrafter WebUI
cd /d "%~dp0"

echo ============================================
echo  StereoCrafter WebUI - Launcher
echo ============================================
echo.

:: ── Step 1/6: Check Python ───────────────────
echo [1/6] Checking Python...
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo        [FAILED] Python not found.
    echo        Please install Python 3.8+ and add it to PATH.
    goto :end
)
for /f "tokens=*" %%i in ('python --version 2^>^&1') do echo        [OK] %%i
echo.

:: ── Step 2/6: Create venv ────────────────────
echo [2/6] Checking virtual environment...
if exist "venv\Scripts\activate.bat" (
    echo        [OK] venv already exists
    echo.
    goto :activate
)
echo        Creating venv...
python -m venv venv
if not exist "venv\Scripts\activate.bat" (
    echo        [FAILED] Could not create virtual environment.
    goto :end
)
echo        [OK] venv created
echo.

:: ── Activate venv ─────────────────────────────
:activate
call venv\Scripts\activate.bat

:: ── Step 3/6: Install PyTorch with CUDA ──────
echo [3/6] Checking PyTorch...
if exist "venv\.torch_installed" (
    echo        [OK] PyTorch already installed
    echo.
    goto :checkdeps
)
echo        Detecting GPU and CUDA version...
for /f "tokens=*" %%i in ('python detect_cuda.py') do set CUDA_TAG=%%i
echo        Detected: %CUDA_TAG%
echo.
if "%CUDA_TAG%"=="cpu" (
    echo        [WARNING] No NVIDIA GPU detected. Installing CPU-only PyTorch.
    echo        Conversion will be very slow without a GPU.
    echo.
    pip install torch torchvision
    goto :torch_done
)
echo        Installing PyTorch for %CUDA_TAG%...
echo.
pip install torch torchvision --index-url https://download.pytorch.org/whl/%CUDA_TAG%
if %errorlevel% neq 0 (
    echo.
    echo        [FAILED] PyTorch install failed. Check errors above.
    goto :end
)
echo.
echo        Installing xformers...
pip install xformers 2>&1
if %errorlevel% neq 0 (
    echo        [WARNING] xformers install failed. This is optional, continuing...
    echo.
)
:torch_done
echo. > "venv\.torch_installed"
echo        [OK] PyTorch installed
echo.

:: ── Step 4/6: Install other dependencies ─────
:checkdeps
echo [4/6] Checking dependencies...
if exist "venv\.webui_installed" (
    echo        [OK] Dependencies already installed
    echo.
    goto :checkwarp
)
echo        Installing remaining packages...
echo.
pip install -r requirements_webui.txt
if %errorlevel% neq 0 (
    echo.
    echo        [FAILED] pip install failed. Check the errors above.
    goto :end
)
echo. > "venv\.webui_installed"
echo.
echo        [OK] Dependencies installed
echo.

:: ── Step 5/6: Compile Forward-Warp ───────────
:checkwarp
echo [5/6] Checking Forward-Warp CUDA module...
if exist "venv\.forwardwarp_installed" (
    echo        [OK] Forward-Warp already compiled
    echo.
    goto :launch
)
echo        Step 1: Compiling CUDA kernel...
echo.
cd /d "%~dp0\dependency\Forward-Warp\Forward_Warp\cuda"
python setup.py install 2>&1
cd /d "%~dp0"
if %errorlevel% neq 0 (
    echo.
    echo        [WARNING] CUDA kernel compilation failed.
    echo        Make sure CUDA toolkit is installed and matches your GPU.
    echo        The UI will launch, but conversion will not work.
    echo.
    goto :launch
)
echo.
echo        Step 2: Installing Python wrapper...
cd /d "%~dp0\dependency\Forward-Warp"
pip install . 2>&1
cd /d "%~dp0"
if %errorlevel% neq 0 (
    echo.
    echo        [WARNING] Forward-Warp Python package install failed.
    echo.
    goto :launch
)
echo. > "venv\.forwardwarp_installed"
echo.
echo        [OK] Forward-Warp compiled
echo.

:: ── Step 6/6: Launch ─────────────────────────
:launch
echo [6/6] Launching StereoCrafter WebUI...
echo        Keep this window open while using the app.
echo        Press Ctrl+C to stop the server.
echo.
echo ============================================
echo.
python app.py

:end
echo.
echo ============================================
if %errorlevel% neq 0 (
    echo  Something went wrong. Check the log above.
) else (
    echo  Server stopped.
)
echo  Press any key to close this window.
echo ============================================
pause >nul
