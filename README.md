# Video Extender Pro - AI Generative Extension

Transform your short video clips into extended content using cutting-edge AI technology. Perfect for social media creators and professional video producers.

## Features

### 🚀 AI-Powered Video Extension
- **Generative Frame Prediction**: Neural network-based frame extrapolation using ConvLSTM architecture
- **Motion-Aware Processing**: Intelligent motion analysis for smooth, realistic extensions
- **Flexible Duration**: Extend videos by 5 seconds (social media) or 10 seconds (professional content)

### 🎵 Complete Audio Processing
- **Audio Time-Stretching**: Phase vocoder technology for high-quality audio extension
- **Automatic Synchronization**: Keep audio perfectly synced with extended video
- **Smart Loop & Crossfade**: Seamless audio transitions

### ⚡ Quality Modes
- **Fast**: Quick processing with optical flow (perfect for previews)
- **Balanced**: AI-powered extension with optimized performance
- **Quality**: Maximum AI quality for professional results

## Installation

### 1. Clone or Download the Repository
```bash
git clone <repository-url>
cd special-enigma
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: PyTorch installation may require specific commands based on your system:
- **CUDA GPU** (recommended for best performance):
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```
- **CPU only**:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```

### 3. Install FFmpeg (Required for Audio Processing)
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

## Usage

### Start the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Video Extender

1. **Upload Your Video**
   - Supported formats: MP4, MOV, AVI
   - Recommended: 5-60 seconds duration

2. **Select Extension Duration**
   - **5 seconds**: Ideal for social media clips (TikTok, Instagram Reels)
   - **10 seconds**: Perfect for professional content, presentations, B-roll

3. **Choose Quality Mode**
   - **Fast**: 2-3x faster, uses optical flow extrapolation
   - **Balanced**: AI-powered, best speed/quality trade-off (recommended)
   - **Quality**: Highest AI quality, slower processing

4. **Enable/Disable Audio Processing**
   - Check "Process Audio" to extend and sync audio track
   - Uncheck for faster processing if audio isn't needed

5. **Click "Extend Video with AI"**
   - Watch real-time progress
   - View statistics and comparison
   - Download your extended video

## Technical Details

### Video Processing Pipeline

```
Input Video → Frame Extraction → AI Extension → Audio Processing → Output
                                       ↓
                    ┌─────────────────────────────┐
                    │  Neural Frame Prediction     │
                    │  - ConvLSTM temporal model   │
                    │  - Optical flow guidance     │
                    │  - Motion-aware blending     │
                    └─────────────────────────────┘
```

### Key Components

#### 1. **video_generator.py**
- `VideoExtrapolator`: Neural network for frame prediction
- `MotionAwareExtender`: Hybrid approach combining AI and optical flow
- `interpolate_frames`: High-quality frame interpolation

#### 2. **audio_processor.py**
- `extend_audio_phase_vocoder`: Time-stretching using librosa
- `extend_audio_smooth`: Crossfade-based looping
- `process_video_with_audio`: Complete audio pipeline

#### 3. **app.py**
- Streamlit UI
- Video processing orchestration
- Progress tracking and visualization

## Architecture

### Neural Network Model

The `VideoExtrapolator` uses a ConvLSTM-based architecture:
- **Encoder**: Spatial feature extraction (Conv2D layers)
- **ConvLSTM**: Temporal dynamics modeling
- **Decoder**: Frame generation with upsampling

### Motion Estimation
- **DIS Optical Flow**: Dense Inverse Search for accurate motion vectors
- **Bidirectional Flow**: Forward and backward flow for occlusion handling
- **Multi-scale Processing**: Handle both small and large motions

### Audio Extension
- **Phase Vocoder**: Preserves pitch while stretching time
- **Crossfade Looping**: Seamless transitions for extended sections
- **FFmpeg Integration**: Professional-grade audio/video muxing

## Performance Optimization

### GPU Acceleration
The application automatically uses GPU (CUDA) if available:
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### Memory Management
- Streaming pipeline for large videos
- Automatic resolution downscaling for >1080p
- Frame batching for efficient processing

### Quality vs Speed Trade-offs
| Mode | Neural Prediction | Processing Speed | Quality | Best For |
|------|------------------|------------------|---------|----------|
| Fast | No | 3-5x real-time | Good | Previews, social media |
| Balanced | Yes | 1-2x real-time | Very Good | General use |
| Quality | Yes + refinement | 0.5-1x real-time | Excellent | Professional work |

## Use Cases

### Social Media (5s Extension)
- Extend TikTok/Reels to meet minimum duration
- Create seamless loops
- Add breathing room to short clips

### Professional Content (10s Extension)
- Extend B-roll footage
- Create presentation intros/outros
- Generate additional content from limited footage

## Troubleshooting

### Issue: "Cannot import torch"
**Solution**: Install PyTorch:
```bash
pip install torch torchvision
```

### Issue: "Audio extraction failed"
**Solution**: Install FFmpeg (see Installation section)

### Issue: Slow processing
**Solutions**:
- Use "Fast" quality mode
- Ensure GPU is available (check with `torch.cuda.is_available()`)
- Process shorter videos
- Disable audio processing

### Issue: Poor extension quality
**Solutions**:
- Use "Quality" mode
- Ensure input video has consistent motion
- Avoid videos with scene cuts
- Use well-lit footage

## Advanced Configuration

### Modify Extension Parameters
Edit `video_generator.py`:
```python
# Adjust neural network capacity
model = VideoExtrapolator(hidden_channels=64)  # Increase for better quality

# Modify optical flow settings
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
dis.setGradientDescentIterations(12)  # More iterations = better accuracy
```

### Custom Extension Durations
Edit `app.py`:
```python
extension_mode = st.radio(
    "Extension Duration",
    ["5 seconds", "10 seconds", "15 seconds"],  # Add custom durations
    ...
)
```

## Limitations

- **Scene Changes**: Cannot extend across scene cuts (results in artifacts)
- **Complex Motion**: Very fast or chaotic motion may not extend perfectly
- **Resolution**: Limited to 1080p max for performance
- **Video Length**: Input videos should be 5-60 seconds for best results

## Future Enhancements

- [ ] Integration with Stable Video Diffusion for higher quality
- [ ] Real-time preview during extension
- [ ] Batch processing for multiple videos
- [ ] Custom extension points (extend from middle, not just end)
- [ ] Enhanced audio synthesis using AI models
- [ ] Timeline-based editing interface

## Technical Requirements

- **Python**: 3.8+
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Storage**: 2GB free space for dependencies

## License

This project is for educational and research purposes.

## Credits

Built using:
- **PyTorch**: Neural network framework
- **OpenCV**: Computer vision and optical flow
- **Librosa**: Audio processing
- **Streamlit**: Web interface
- **FFmpeg**: Audio/video encoding

---

**Made with AI-powered video generation technology**
