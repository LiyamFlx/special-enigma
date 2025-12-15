#!/bin/bash

# Video Extender Pro - Installation Script
# Supports macOS, Linux, and Windows (via Git Bash/WSL)

set -e  # Exit on error

echo "============================================================"
echo "Video Extender Pro - Installation Script"
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}Checking Python installation...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✅ Python found: $PYTHON_VERSION${NC}"
    PYTHON_CMD=python3
    PIP_CMD=pip3
elif command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✅ Python found: $PYTHON_VERSION${NC}"
    PYTHON_CMD=python
    PIP_CMD=pip
else
    echo -e "${RED}❌ Python not found. Please install Python 3.8 or higher.${NC}"
    exit 1
fi

# Check FFmpeg
echo -e "\n${YELLOW}Checking FFmpeg installation...${NC}"
if command -v ffmpeg &> /dev/null; then
    FFMPEG_VERSION=$(ffmpeg -version 2>&1 | head -n 1)
    echo -e "${GREEN}✅ FFmpeg found: $FFMPEG_VERSION${NC}"
else
    echo -e "${YELLOW}⚠️  FFmpeg not found.${NC}"
    echo "FFmpeg is required for audio processing."
    echo "Install instructions:"
    echo "  - macOS: brew install ffmpeg"
    echo "  - Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "  - Windows: Download from https://ffmpeg.org/download.html"
    read -p "Continue without FFmpeg? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create virtual environment (recommended)
echo -e "\n${YELLOW}Setting up virtual environment...${NC}"
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${GREEN}✅ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
$PYTHON_CMD -m pip install --upgrade pip

# Detect GPU support
echo -e "\n${YELLOW}Detecting GPU support...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✅ NVIDIA GPU detected${NC}"
    GPU_AVAILABLE=true
else
    echo -e "${YELLOW}⚠️  No NVIDIA GPU detected - will use CPU${NC}"
    GPU_AVAILABLE=false
fi

# Install PyTorch
echo -e "\n${YELLOW}Installing PyTorch...${NC}"
if [ "$GPU_AVAILABLE" = true ]; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing PyTorch (CPU version)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# Install other dependencies
echo -e "\n${YELLOW}Installing other dependencies...${NC}"
pip install opencv-python-headless numpy streamlit einops pillow scipy imageio imageio-ffmpeg librosa soundfile tqdm

# Run installation test
echo -e "\n${YELLOW}Running installation test...${NC}"
$PYTHON_CMD test_installation.py

# Success message
echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}Installation Complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "To start the application:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo "  2. Run the application:"
echo "     streamlit run app.py"
echo ""
echo "The application will open in your browser at http://localhost:8501"
