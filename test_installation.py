#!/usr/bin/env python3
"""
Installation and functionality test script for Video Extender Pro
"""

import sys
import subprocess

def test_imports():
    """Test if all required packages are installed"""
    print("Testing package imports...")

    required_packages = {
        'cv2': 'opencv-python-headless',
        'numpy': 'numpy',
        'streamlit': 'streamlit',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'librosa': 'librosa',
        'soundfile': 'soundfile',
        'scipy': 'scipy',
        'imageio': 'imageio',
    }

    failed = []
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT FOUND")
            failed.append(package)

    return failed

def test_ffmpeg():
    """Test if FFmpeg is installed"""
    print("\nTesting FFmpeg installation...")
    try:
        result = subprocess.run(['ffmpeg', '-version'],
                              capture_output=True,
                              text=True,
                              timeout=5)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"✅ FFmpeg installed: {version}")
            return True
        else:
            print("❌ FFmpeg not working properly")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("❌ FFmpeg not found")
        print("   Install: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)")
        return False

def test_torch_cuda():
    """Test PyTorch CUDA availability"""
    print("\nTesting GPU acceleration...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
        else:
            print("⚠️  CUDA not available - will use CPU (slower)")
            print("   For GPU support, install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")

def test_video_generator():
    """Test if video_generator module loads"""
    print("\nTesting video_generator module...")
    try:
        from video_generator import MotionAwareExtender, VideoExtrapolator
        print("✅ video_generator module loaded successfully")

        # Test model initialization
        extender = MotionAwareExtender()
        print(f"✅ Model initialized on device: {extender.device}")
        return True
    except Exception as e:
        print(f"❌ video_generator test failed: {e}")
        return False

def test_audio_processor():
    """Test if audio_processor module loads"""
    print("\nTesting audio_processor module...")
    try:
        from audio_processor import extend_audio_smooth, process_video_with_audio
        print("✅ audio_processor module loaded successfully")
        return True
    except Exception as e:
        print(f"❌ audio_processor test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Video Extender Pro - Installation Test")
    print("=" * 60)

    # Test imports
    failed_packages = test_imports()

    # Test FFmpeg
    ffmpeg_ok = test_ffmpeg()

    # Test CUDA
    test_torch_cuda()

    # Test custom modules
    video_gen_ok = test_video_generator()
    audio_ok = test_audio_processor()

    # Summary
    print("\n" + "=" * 60)
    print("INSTALLATION TEST SUMMARY")
    print("=" * 60)

    if not failed_packages and ffmpeg_ok and video_gen_ok and audio_ok:
        print("✅ All tests passed! Installation is complete.")
        print("\nTo start the application, run:")
        print("   streamlit run app.py")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
        if failed_packages:
            print(f"\nMissing packages: {', '.join(failed_packages)}")
            print("Install with: pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
