import os
import sys

# Set environment variables for cloud deployment
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Import with error handling
try:
    import cv2
    import numpy as np
    import streamlit as st
    import tempfile
    import time
    from video_generator import extend_video_generative
    from audio_processor import process_video_with_audio
except ImportError as e:
    print(f"Import Error: {e}", file=sys.stderr)
    import streamlit as st
    st.error(f"Failed to import required modules: {e}")
    st.stop()

def extend_video(input_path, output_path, quality_mode="balanced", progress_callback=None):
    """
    Extended video using advanced optical flow interpolation with quality improvements.
    
    Args:
        input_path: Path to input video
        output_path: Path to save extended video
        quality_mode: "fast", "balanced", or "quality"
        progress_callback: Function to call with progress updates (0-100)
    """
    # Load video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Couldn't open video!")

    # Get properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Downscale if >1080p
    if height > 1080:
        scale = 1080 / height
        width = int(width * scale)
        height = 1080

    # Check duration (5-60s)
    duration = frame_count / fps
    if duration < 5 or duration > 60:
        raise ValueError("Video must be 5-60 seconds!")

    # Output writer (MP4, same fps/res, no audio)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize DIS optical flow based on quality mode
    if quality_mode == "fast":
        dis_forward = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        dis_backward = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    elif quality_mode == "quality":
        dis_forward = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        dis_backward = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    else:  # balanced
        dis_forward = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        dis_backward = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("Empty video!")
    
    # Resize if needed
    if prev_frame.shape[:2] != (height, width):
        prev_frame = cv2.resize(prev_frame, (width, height))
    out.write(prev_frame)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Process frame by frame
    frame_idx = 1
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        # Resize if needed
        if curr_frame.shape[:2] != (height, width):
            curr_frame = cv2.resize(curr_frame, (width, height))
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # === IMPROVEMENT 1: Bidirectional Optical Flow (DIS) ===
        # Forward flow: prev -> curr
        flow_forward = dis_forward.calc(prev_gray, curr_gray, None)
        
        # Backward flow: curr -> prev
        flow_backward = dis_backward.calc(curr_gray, prev_gray, None)

        # === IMPROVEMENT 2: Occlusion Detection ===
        h, w = height, width
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Forward warp coordinates
        forward_x = x_coords + flow_forward[:, :, 0]
        forward_y = y_coords + flow_forward[:, :, 1]
        
        # Warp backward flow to previous frame
        warped_backward = cv2.remap(flow_backward, forward_x, forward_y, 
                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        # Forward-backward consistency check
        consistency_error = np.sqrt(
            (flow_forward[:, :, 0] + warped_backward[:, :, 0]) ** 2 +
            (flow_forward[:, :, 1] + warped_backward[:, :, 1]) ** 2
        )
        
        # Occlusion mask (areas with high inconsistency)
        occlusion_threshold = 1.5 if quality_mode == "quality" else 2.0
        occlusion_mask = consistency_error > occlusion_threshold

        # === IMPROVEMENT 3: Bidirectional Interpolation ===
        alpha = 0.5  # Midpoint
        
        # Forward interpolation (from prev_frame)
        interp_flow_forward = alpha * flow_forward
        map_x_forward = x_coords + interp_flow_forward[:, :, 0]
        map_y_forward = y_coords + interp_flow_forward[:, :, 1]
        forward_interp = cv2.remap(prev_frame, map_x_forward, map_y_forward, 
                                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Backward interpolation (from curr_frame)
        interp_flow_backward = (1 - alpha) * flow_backward
        map_x_backward = x_coords + interp_flow_backward[:, :, 0]
        map_y_backward = y_coords + interp_flow_backward[:, :, 1]
        backward_interp = cv2.remap(curr_frame, map_x_backward, map_y_backward, 
                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Blend both directions
        interpolated = cv2.addWeighted(forward_interp, 0.5, backward_interp, 0.5, 0)

        # === IMPROVEMENT 4: Motion-Adaptive Blending ===
        mag, _ = cv2.cartToPolar(flow_forward[..., 0], flow_forward[..., 1])
        avg_mag = np.mean(mag)
        
        # Normalize motion magnitude to 0-1 for blending weight
        motion_weight = np.clip(mag / 20.0, 0, 1)
        motion_weight = np.stack([motion_weight] * 3, axis=2)  # Convert to 3 channels
        
        # Simple blend for comparison
        simple_blend = cv2.addWeighted(prev_frame, 0.5, curr_frame, 0.5, 0)
        
        # Adaptive blend: high motion = trust interpolation, low motion = trust blend
        if avg_mag < 3:  # Very low motion
            interpolated = cv2.addWeighted(interpolated, 0.3, simple_blend, 0.7, 0)
        else:
            interpolated = (motion_weight * interpolated + 
                           (1 - motion_weight) * simple_blend).astype(np.uint8)

        # === IMPROVEMENT 5: Handle Occlusions ===
        # For occluded regions, use simple blending
        occlusion_mask_3ch = np.stack([occlusion_mask] * 3, axis=2)
        interpolated = np.where(occlusion_mask_3ch, simple_blend, interpolated)

        # === IMPROVEMENT 6: Edge-Aware Smoothing ===
        if quality_mode == "quality":
            # Bilateral filter preserves edges while smoothing
            interpolated = cv2.bilateralFilter(interpolated, d=5, sigmaColor=50, sigmaSpace=50)
        else:
            # Lighter temporal smoothing for faster modes
            interpolated = cv2.addWeighted(interpolated, 0.85, prev_frame, 0.15, 0)

        # Write interpolated + current frame
        out.write(interpolated)
        out.write(curr_frame)

        # Update for next iteration
        prev_frame = curr_frame
        prev_gray = curr_gray
        
        # Progress callback
        frame_idx += 1
        if progress_callback:
            progress = int((frame_idx / frame_count) * 100)
            progress_callback(progress)

    # Cleanup
    cap.release()
    out.release()


# ===== STREAMLIT UI =====
st.set_page_config(page_title="Video Extender Pro", page_icon="🎬", layout="wide")

st.title("🎬 Video Extender Pro - AI Generative Extension")
st.write("Extend your videos with AI-powered frame generation! Perfect for social media clips and professional content.")

# Main controls
uploaded_file = st.file_uploader("Choose your video", type=["mp4", "mov", "avi"])

if uploaded_file:
    col1, col2, col3 = st.columns(3)
    with col1:
        extension_mode = st.radio(
            "Extension Duration",
            ["5 seconds", "10 seconds"],
            help="5s: Perfect for social media clips\n10s: Extended professional content"
        )
    with col2:
        quality_mode = st.selectbox(
            "Quality Mode",
            ["fast", "balanced", "quality"],
            index=1,
            help="Fast: Quick processing (optical flow)\nBalanced: AI-powered with good speed\nQuality: Best AI results (slower)"
        )
    with col3:
        include_audio = st.checkbox(
            "Process Audio",
            value=True,
            help="Extend and synchronize audio track"
        )

    # Parse extension duration
    extension_seconds = 5.0 if extension_mode == "5 seconds" else 10.0

    # Display file info
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.info(f"📁 **File:** {uploaded_file.name} ({file_size_mb:.1f} MB) | **Extension:** +{extension_seconds:.0f}s")

    # Save temp input
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    input_path = tfile.name

    # Show original video preview
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📹 Original Video")
        st.video(input_path)

        # Get video info
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()

        st.caption(f"Duration: {duration:.1f}s | FPS: {fps:.0f} | Frames: {frame_count}")

    with col2:
        st.subheader("🎯 Extended Preview")
        st.info(f"**New Duration:** {duration + extension_seconds:.1f}s\n\n**Added Content:** {extension_seconds:.0f}s of AI-generated frames")

    if st.button("🚀 Extend Video with AI", type="primary", use_container_width=True):
        output_path_no_audio = os.path.join(tempfile.gettempdir(), "extended_video_no_audio.mp4")
        output_path_final = os.path.join(tempfile.gettempdir(), "extended_video_final.mp4")

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        start_time = time.time()

        def update_progress(percent):
            progress_bar.progress(percent / 100)
            elapsed = time.time() - start_time
            status_text.text(f"Processing... {percent}% complete ({elapsed:.1f}s elapsed)")

        try:
            # Process video with generative extension
            status_text.text("🎬 Generating extended frames with AI...")
            stats = extend_video_generative(
                input_path,
                output_path_no_audio,
                extension_seconds=extension_seconds,
                quality_mode=quality_mode,
                progress_callback=update_progress
            )

            # Process audio if requested
            if include_audio:
                status_text.text("🎵 Extending audio track...")
                progress_bar.progress(95)
                audio_processed = process_video_with_audio(
                    input_path,
                    output_path_no_audio,
                    output_path_final,
                    extension_seconds,
                    quality_mode
                )
                output_path = output_path_final
            else:
                output_path = output_path_no_audio
                audio_processed = False

            # Success message
            processing_time = time.time() - start_time
            progress_bar.progress(100)
            status_text.empty()
            st.success(f"✅ Video extended successfully in {processing_time:.1f}s!")

            # Stats
            st.subheader("📊 Processing Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Original Duration", f"{stats['original_duration']:.1f}s")
            with col2:
                st.metric("Extended Duration", f"{stats['extended_duration']:.1f}s")
            with col3:
                st.metric("Frames Added", f"+{stats['extension_frames']}")
            with col4:
                st.metric("Audio", "✅ Extended" if audio_processed else "⚠️ Not processed")

            # Download button
            with open(output_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Extended Video",
                    f,
                    file_name=f"extended_{int(extension_seconds)}s_{uploaded_file.name}",
                    mime="video/mp4",
                    use_container_width=True
                )

            # Side-by-side comparison
            st.subheader("📊 Before & After Comparison")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original Video**")
                st.video(input_path)
            with col2:
                st.write(f"**Extended (+{extension_seconds:.0f}s)**")
                st.video(output_path)

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error: {str(e)}")

            # Show detailed error in expander
            with st.expander("🔍 Error Details"):
                import traceback
                st.code(traceback.format_exc())

# Footer
st.divider()
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🎯 How it works
    This app uses **AI-powered generative video extension**:
    - **Neural frame prediction** using ConvLSTM architecture
    - **Motion-aware extrapolation** for smooth continuity
    - **Optical flow guidance** for realistic motion
    - **Audio time-stretching** with phase vocoder technology
    - **Smart quality modes** (fast/balanced/quality)
    """)

with col2:
    st.markdown("""
    ### 💡 Use Cases
    **5-Second Extension:**
    - Social media clips (TikTok, Instagram Reels)
    - Quick video loops
    - Short promotional content

    **10-Second Extension:**
    - Professional presentations
    - Extended B-roll footage
    - Longer form content creation
    """)

st.markdown("""
**Tips for best results:**
- Videos with consistent motion work best
- Avoid abrupt scene changes
- Well-lit footage produces better AI predictions
- Enable audio processing for complete synchronization
""")
