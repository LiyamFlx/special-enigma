# Installation Troubleshooting Guide

## Common Installation Errors and Solutions

### Error: "Could not find a version that satisfies the requirement..."

**Cause:** Specific package versions may not be available for your Python version or platform.

**Solutions:**

#### Option 1: Use Flexible Requirements
```bash
pip install -r requirements-flexible.txt
```

#### Option 2: Install PyTorch Separately First
PyTorch often causes issues - install it separately:

**For CPU only:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-flexible.txt
```

**For NVIDIA GPU (CUDA 11.8):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-flexible.txt
```

**For NVIDIA GPU (CUDA 12.1):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-flexible.txt
```

#### Option 3: Install Without Exact Versions
```bash
pip install opencv-python-headless numpy streamlit
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install einops pillow scipy imageio imageio-ffmpeg librosa soundfile tqdm
```

---

### Error: "externally-managed-environment" (PEP 668)

**Cause:** Your system Python is protected by the OS.

**Solution:** Use a virtual environment (REQUIRED):

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate.bat  # Windows

# Now install
pip install -r requirements.txt
```

---

### Error: Building wheels for packages (scipy, numpy, etc.)

**Cause:** Pre-built wheels not available for your platform.

**Solutions:**

#### macOS:
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install with Homebrew Python (recommended)
brew install python@3.11
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Linux:
```bash
# Install build dependencies
sudo apt-get update
sudo apt-get install python3-dev build-essential libatlas-base-dev gfortran

# Then install requirements
pip install -r requirements.txt
```

#### Windows:
```bash
# Use Python from python.org (not Microsoft Store)
# Download from: https://www.python.org/downloads/

# Or install Visual C++ Build Tools
# https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

---

### Error: "No module named 'torch'" when running app

**Cause:** PyTorch not installed or wrong environment.

**Solution:**

```bash
# Make sure you're in the virtual environment
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate.bat  # Windows

# Verify PyTorch installation
python -c "import torch; print(torch.__version__)"

# If it fails, install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

### Error: FFmpeg not found (audio processing fails)

**Cause:** FFmpeg not installed on system.

**Solutions:**

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**Windows:**
1. Download FFmpeg from https://ffmpeg.org/download.html
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to PATH environment variable
4. Restart terminal/IDE

**Verify:**
```bash
ffmpeg -version
```

---

### Error: "ImportError: cannot import name 'extend_video_generative'"

**Cause:** Running from wrong directory or module not found.

**Solution:**

```bash
# Make sure you're in the project directory
cd /path/to/special-enigma

# Verify files exist
ls -la *.py

# Should see: app.py, video_generator.py, audio_processor.py

# Run from project root
streamlit run app.py
```

---

## Minimal Installation (No AI features)

If you just want to test the basic app without AI:

```bash
pip install opencv-python-headless numpy streamlit imageio imageio-ffmpeg
```

Then comment out AI imports in `app.py`:
```python
# from video_generator import extend_video_generative
# from audio_processor import process_video_with_audio
```

---

## Step-by-Step Clean Installation

1. **Remove old environment:**
```bash
rm -rf venv
rm -rf __pycache__
```

2. **Create fresh virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
```

3. **Upgrade pip:**
```bash
pip install --upgrade pip
```

4. **Install PyTorch first:**
```bash
# CPU version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

5. **Install remaining packages:**
```bash
pip install opencv-python-headless numpy streamlit einops pillow scipy imageio imageio-ffmpeg librosa soundfile tqdm
```

6. **Test installation:**
```bash
python test_installation.py
```

7. **Run app:**
```bash
streamlit run app.py
```

---

## System Requirements Check

### Minimum Requirements:
- Python 3.8 or higher
- 8GB RAM
- 2GB free disk space

### Verify Python version:
```bash
python3 --version
# Should show Python 3.8.x or higher
```

### Verify pip version:
```bash
pip --version
# Should show pip 21.0 or higher
```

---

## Platform-Specific Notes

### macOS Apple Silicon (M1/M2/M3)
PyTorch has native ARM support:
```bash
pip install torch torchvision
```

### Windows
Use PowerShell or Command Prompt (not Git Bash for activation):
```powershell
python -m venv venv
venv\Scripts\activate.bat
```

### Linux
May need additional libraries:
```bash
sudo apt-get install libsndfile1 libsndfile1-dev
```

---

## Still Having Issues?

1. **Check Python version compatibility:**
   - Use Python 3.8, 3.9, 3.10, or 3.11
   - Avoid Python 3.12+ (some packages not yet compatible)

2. **Try using conda instead of pip:**
```bash
conda create -n video-extender python=3.10
conda activate video-extender
pip install -r requirements-flexible.txt
```

3. **Get detailed error information:**
```bash
pip install -r requirements.txt --verbose
```

4. **Check for conflicting packages:**
```bash
pip list
pip check
```

---

## Quick Test

After installation, run this to verify everything works:

```python
python3 << 'EOF'
import cv2
import numpy as np
import torch
import streamlit
from video_generator import VideoExtrapolator
from audio_processor import extend_audio_smooth
print("✅ All imports successful!")
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
EOF
```

---

## Contact

If you're still experiencing issues, please provide:
1. Your operating system and version
2. Python version (`python3 --version`)
3. Complete error message
4. Output of `pip list`
