# Quick Start Guide - Video Extender Pro

## Installation (3 minutes)

### Automated Installation (Recommended)

**macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```cmd
setup.bat
```

### Manual Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Install FFmpeg:**
- macOS: `brew install ffmpeg`
- Ubuntu: `sudo apt-get install ffmpeg`
- Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## Running the Application

```bash
streamlit run app.py
```

Opens automatically at: `http://localhost:8501`

## Usage (30 seconds)

1. **Upload video** (MP4, MOV, or AVI)
2. **Choose extension:**
   - 5 seconds → Social media clips
   - 10 seconds → Professional content
3. **Select quality:**
   - Fast → Quick preview
   - Balanced → Recommended
   - Quality → Best results
4. **Click "Extend Video with AI"**
5. **Download** your extended video!

## Examples

### Social Media Clip (5s extension)
**Input:** 3-second TikTok clip
**Output:** 8-second extended version with smooth motion
**Time:** ~30 seconds processing

### Professional B-Roll (10s extension)
**Input:** 15-second footage
**Output:** 25-second extended video with audio
**Time:** ~90 seconds processing

## Tips for Best Results

✅ **DO:**
- Use videos with smooth, consistent motion
- Keep lighting consistent
- Enable audio processing for complete sync
- Use "Balanced" mode for best speed/quality trade-off

❌ **DON'T:**
- Use videos with abrupt scene cuts
- Extend very dark or grainy footage
- Expect perfect results with chaotic motion

## Troubleshooting

**Problem:** "Module not found" errors
**Solution:** Run `pip install -r requirements.txt`

**Problem:** Slow processing
**Solution:**
- Use "Fast" mode
- Disable audio processing
- Ensure GPU is available (NVIDIA)

**Problem:** Audio issues
**Solution:** Make sure FFmpeg is installed

## Next Steps

- Read the full [README.md](README.md) for technical details
- Check out the [OPTIMIZATION_ROADMAP.md](OPTIMIZATION_ROADMAP.md) for advanced features
- Experiment with different quality modes and extension durations

---

**Need Help?** Check the detailed README.md or create an issue.
