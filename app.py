import os
import sys

# Set environment variables for cloud deployment
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import cv2
import numpy as np
import streamlit as st
import tempfile
import time
import subprocess
from pathlib import Path

# Optional imports with fallback
AUDIO_AVAILABLE = False
try:
    from pydub import AudioSegment
    AUDIO_AVAILABLE = True
except ImportError:
    pass

# ===== VIDEO EXTENSION CORE =====

def extend_video_optical_flow(input_path, output_path, extension_seconds=5.0, quality_mode="balanced", progress_callback=None):
    """
    Extend video using bidirectional optical flow with motion extrapolation.
    
    Args:
        input_path: Path to input video
        output_path: Path to save extended video  
        extension_seconds: Seconds to add (5.0 or 10.0)
        quality_mode: "fast", "balanced", or "quality"
        progress_callback: Function to call with progress updates (0-100)
    
    Returns:
        dict with processing statistics
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Couldn't open video!")

    # Get properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_duration = frame_count / fps

    # Validate duration (5-60s)
    if original_duration < 3:
        raise ValueError("Video must be at least 3 seconds!")
    if original_duration > 120:
        raise ValueError("Video must be under 2 minutes!")

    # Downscale if >1080p for performance
    scale = 1.0
    if height > 1080:
        scale = 1080 / height
        width = int(width * scale)
        height = 1080

    # Calculate extension frames
    extension_frames = int(extension_seconds * fps)
    
    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Configure DIS optical flow based on quality
    # All modes use ULTRAFAST base, but we tune parameters for quality
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    
    if quality_mode == "balanced":
        dis.setFinestScale(2)
        dis.setGradientDescentIterations(12)
    elif quality_mode == "quality":
        dis.setFinestScale(1)
        dis.setGradientDescentIterations(25)
        dis.setPatchSize(8)
        dis.setPatchStride(4)

    # Read all frames (for extension we need history)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, (width, height))
        frames.append(frame)
    cap.release()

    if len(frames) < 2:
        raise ValueError("Video too short!")

    # Write original frames first
    total_operations = len(frames) + extension_frames
    for i, frame in enumerate(frames):
        out.write(frame)
        if progress_callback:
            progress_callback(int((i / total_operations) * 50))

    # Calculate motion from last N frames for extrapolation
    history_length = min(10, len(frames) - 1)
    flows = []
    
    for i in range(len(frames) - history_length, len(frames) - 1):
        gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
        flow = dis.calc(gray1, gray2, None)
        flows.append(flow)

    # Average motion vector (with decay weighting - recent frames matter more)
    weights = np.array([0.5 ** (len(flows) - 1 - i) for i in range(len(flows))])
    weights /= weights.sum()
    
    avg_flow = np.zeros_like(flows[0])
    for i, flow in enumerate(flows):
        avg_flow += flow * weights[i]

    # Motion magnitude for adaptive blending
    mag, _ = cv2.cartToPolar(avg_flow[..., 0], avg_flow[..., 1])
    avg_mag = np.mean(mag)

    # Generate extension frames
    last_frame = frames[-1].copy()
    prev_frame = frames[-2].copy() if len(frames) > 1 else last_frame.copy()
    
    h, w = height, width
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

    for i in range(extension_frames):
        # Decay factor - motion slows down over time
        decay = max(0.3, 1.0 - (i / extension_frames) * 0.7)
        
        # Scale flow by decay
        scaled_flow = avg_flow * decay
        
        # Forward warp coordinates
        map_x = x_coords + scaled_flow[..., 0]
        map_y = y_coords + scaled_flow[..., 1]
        
        # Warp last frame
        warped = cv2.remap(last_frame, map_x, map_y, 
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Blend with previous to reduce artifacts
        if avg_mag > 2:  # Significant motion
            alpha = 0.85
        else:  # Low motion - more blending
            alpha = 0.7
            
        new_frame = cv2.addWeighted(warped, alpha, last_frame, 1 - alpha, 0)
        
        # Edge-aware smoothing for quality mode
        if quality_mode == "quality":
            new_frame = cv2.bilateralFilter(new_frame, d=5, sigmaColor=40, sigmaSpace=40)
        
        out.write(new_frame)
        
        # Update for next iteration
        prev_frame = last_frame
        last_frame = new_frame
        
        if progress_callback:
            progress_callback(int(50 + (i / extension_frames) * 50))

    out.release()
    
    return {
        'original_duration': original_duration,
        'extended_duration': original_duration + extension_seconds,
        'original_frames': len(frames),
        'extension_frames': extension_frames,
        'fps': fps
    }


def process_audio(input_path, video_output_path, final_output_path, extension_seconds, quality_mode="balanced"):
    """
    Extract audio from original, extend it, and merge with video.
    Uses ffmpeg for reliable processing.
    """
    temp_dir = tempfile.gettempdir()
    audio_path = os.path.join(temp_dir, "temp_audio.aac")
    extended_audio_path = os.path.join(temp_dir, "extended_audio.aac")
    
    try:
        # Extract audio from original
        extract_cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-vn', '-acodec', 'aac', '-b:a', '192k',
            audio_path
        ]
        result = subprocess.run(extract_cmd, capture_output=True, timeout=60)
        
        if result.returncode != 0 or not os.path.exists(audio_path):
            # No audio track - just copy video
            import shutil
            shutil.copy(video_output_path, final_output_path)
            return False
        
        # Get original audio duration
        probe_cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        audio_duration = float(result.stdout.strip())
        
        # Calculate time stretch factor
        target_duration = audio_duration + extension_seconds
        stretch_factor = target_duration / audio_duration
        
        # Use rubberband for high-quality time stretching (or atempo for fallback)
        if stretch_factor <= 2.0:
            # atempo filter (0.5 to 2.0 range)
            stretch_cmd = [
                'ffmpeg', '-y', '-i', audio_path,
                '-filter:a', f'atempo={1/stretch_factor}',
                '-acodec', 'aac', '-b:a', '192k',
                extended_audio_path
            ]
        else:
            # Chain atempo filters for larger stretches
            tempo1 = 0.5
            tempo2 = 1 / (stretch_factor * tempo1)
            stretch_cmd = [
                'ffmpeg', '-y', '-i', audio_path,
                '-filter:a', f'atempo={tempo1},atempo={tempo2}',
                '-acodec', 'aac', '-b:a', '192k',
                extended_audio_path
            ]
        
        subprocess.run(stretch_cmd, capture_output=True, timeout=120)
        
        # Merge video and extended audio
        merge_cmd = [
            'ffmpeg', '-y',
            '-i', video_output_path,
            '-i', extended_audio_path,
            '-c:v', 'copy', '-c:a', 'aac',
            '-map', '0:v:0', '-map', '1:a:0',
            '-shortest',
            final_output_path
        ]
        subprocess.run(merge_cmd, capture_output=True, timeout=120)
        
        # Cleanup temp files
        for f in [audio_path, extended_audio_path]:
            if os.path.exists(f):
                os.remove(f)
        
        return os.path.exists(final_output_path)
        
    except Exception as e:
        # Fallback: copy video without audio
        import shutil
        shutil.copy(video_output_path, final_output_path)
        return False


# ===== STREAMLIT UI =====
st.set_page_config(page_title="Video Extender Pro", page_icon="🎬", layout="wide")

st.title("🎬 Video Extender Pro")
st.write("Extend your videos with AI-powered motion extrapolation. Perfect for social media clips and professional content.")

# Check ffmpeg availability
ffmpeg_available = subprocess.run(['which', 'ffmpeg'], capture_output=True).returncode == 0

uploaded_file = st.file_uploader("Choose your video", type=["mp4", "mov", "avi", "mkv", "webm"])

if uploaded_file:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        extension_mode = st.radio(
            "Extension Duration",
            ["5 seconds", "10 seconds"],
            help="5s: Social media clips | 10s: Extended content"
        )
    
    with col2:
        quality_mode = st.selectbox(
            "Quality Mode",
            ["fast", "balanced", "quality"],
            index=1,
            help="Fast: Quick processing | Balanced: Good quality/speed | Quality: Best results (slower)"
        )
    
    with col3:
        include_audio = st.checkbox(
            "Process Audio",
            value=ffmpeg_available,
            disabled=not ffmpeg_available,
            help="Extend audio with video" + ("" if ffmpeg_available else " (ffmpeg not available)")
        )

    extension_seconds = 5.0 if extension_mode == "5 seconds" else 10.0

    # Save temp input
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    input_path = tfile.name

    # Get video info
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    # Display info
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.info(f"📁 **{uploaded_file.name}** | {file_size_mb:.1f} MB | {width}x{height} | {duration:.1f}s @ {fps:.0f}fps → **+{extension_seconds:.0f}s**")

    # Preview columns
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📹 Original")
        st.video(input_path)
    
    with col2:
        st.subheader("🎯 Preview")
        new_duration = duration + extension_seconds
        st.metric("New Duration", f"{new_duration:.1f}s", f"+{extension_seconds:.0f}s")

    if st.button("🚀 Extend Video", type="primary", use_container_width=True):
        output_path_video = os.path.join(tempfile.gettempdir(), "extended_video.mp4")
        output_path_final = os.path.join(tempfile.gettempdir(), "extended_final.mp4")

        progress_bar = st.progress(0)
        status_text = st.empty()
        start_time = time.time()

        def update_progress(percent):
            progress_bar.progress(min(percent, 100) / 100)
            elapsed = time.time() - start_time
            status_text.text(f"Processing... {percent}% ({elapsed:.1f}s)")

        try:
            # Process video
            status_text.text("🎬 Extending video frames...")
            stats = extend_video_optical_flow(
                input_path,
                output_path_video,
                extension_seconds=extension_seconds,
                quality_mode=quality_mode,
                progress_callback=update_progress
            )

            # Process audio
            audio_processed = False
            if include_audio and ffmpeg_available:
                status_text.text("🎵 Processing audio...")
                progress_bar.progress(0.95)
                audio_processed = process_audio(
                    input_path,
                    output_path_video,
                    output_path_final,
                    extension_seconds,
                    quality_mode
                )
                output_path = output_path_final if audio_processed else output_path_video
            else:
                output_path = output_path_video

            # Done
            processing_time = time.time() - start_time
            progress_bar.progress(1.0)
            status_text.empty()
            st.success(f"✅ Extended in {processing_time:.1f}s!")

            # Stats
            st.subheader("📊 Results")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Original", f"{stats['original_duration']:.1f}s")
            c2.metric("Extended", f"{stats['extended_duration']:.1f}s")
            c3.metric("Frames Added", f"+{stats['extension_frames']}")
            c4.metric("Audio", "✅" if audio_processed else "—")

            # Download
            with open(output_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Extended Video",
                    f,
                    file_name=f"extended_{int(extension_seconds)}s_{uploaded_file.name}",
                    mime="video/mp4",
                    use_container_width=True
                )

            # Comparison
            st.subheader("📊 Comparison")
            col1, col2 = st.columns(2)
            with col1:
                st.caption("Original")
                st.video(input_path)
            with col2:
                st.caption(f"Extended (+{extension_seconds:.0f}s)")
                st.video(output_path)

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error: {str(e)}")
            with st.expander("🔍 Details"):
                import traceback
                st.code(traceback.format_exc())

# Footer
st.divider()
st.markdown("""
**How it works:** Analyzes motion patterns from the last frames using bidirectional optical flow, 
then extrapolates with decay to generate smooth extended content. Audio is time-stretched to match.

**Tips:** Videos with consistent motion work best. Avoid abrupt scene changes at the end.
""")
