@echo off
REM Video Extender Pro - Windows Installation Script

echo ============================================================
echo Video Extender Pro - Installation Script
echo ============================================================

REM Check Python
echo.
echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python 3.8 or higher.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

python --version
echo Python found!

REM Check FFmpeg
echo.
echo Checking FFmpeg installation...
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: FFmpeg not found.
    echo FFmpeg is required for audio processing.
    echo Download from: https://ffmpeg.org/download.html
    echo.
    set /p continue="Continue without FFmpeg? (y/n): "
    if /i not "%continue%"=="y" exit /b 1
) else (
    echo FFmpeg found!
)

REM Create virtual environment
echo.
echo Setting up virtual environment...
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created!
) else (
    echo Virtual environment already exists!
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Detect GPU
echo.
echo Detecting GPU support...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo NVIDIA GPU detected - installing PyTorch with CUDA...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
) else (
    echo No NVIDIA GPU detected - installing CPU version...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

REM Install other dependencies
echo.
echo Installing other dependencies...
pip install opencv-python-headless numpy streamlit einops pillow scipy imageio imageio-ffmpeg librosa soundfile tqdm

REM Run test
echo.
echo Running installation test...
python test_installation.py

REM Success
echo.
echo ============================================================
echo Installation Complete!
echo ============================================================
echo.
echo To start the application:
echo   1. Activate the virtual environment:
echo      venv\Scripts\activate.bat
echo   2. Run the application:
echo      streamlit run app.py
echo.
echo The application will open in your browser at http://localhost:8501
echo.
pause
