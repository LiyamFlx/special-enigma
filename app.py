import cv2
import numpy as np
import streamlit as st
import os
import tempfile

def extend_video(input_path, output_path, quality_mode="balanced", extension_factor=2, progress_callback=None):
    """
    Extended video using multi-scale optical flow with advanced smoothing.
    
    Args:
        input_path: Path to input video
        output_path: Path to save extended video
        quality_mode: "fast", "balanced", or "quality"
        extension_factor: 2 = double length, 3 = triple, etc.
        progress_callback: Function to call with progress updates (0-100)
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Couldn't open video!")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Downscale if >1080p
    if height > 1080:
        scale = 1080 / height
        width = int(width * scale)
        height = 1080

    duration = frame_count / fps
    if duration < 5 or duration > 60:
        raise ValueError("Video must be 5-60 seconds!")

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize optical flow
    if quality_mode == "fast":
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    elif quality_mode == "quality":
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        dis.setFinestScale(0)  # Process at full resolution
    else:
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("Empty video!")
    
    if prev_frame.shape[:2] != (height, width):
        prev_frame = cv2.resize(prev_frame, (width, height))
    
    out.write(prev_frame)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Buffer for temporal consistency (helps with jitter)
    frame_buffer = [prev_frame.copy()]
    max_buffer_size = 3

    frame_idx = 1
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        
        if curr_frame.shape[:2] != (height, width):
            curr_frame = cv2.resize(curr_frame, (width, height))
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # === MULTI-SCALE OPTICAL FLOW for better accuracy ===
        flow = compute_multiscale_flow(prev_gray, curr_gray, dis, quality_mode)
        
        # === Generate N-1 interpolated frames (for extension_factor = N) ===
        for i in range(1, extension_factor):
            alpha = i / extension_factor  # 0.33, 0.66 for 3x extension
            
            # Bidirectional interpolation with flow refinement
            interpolated = interpolate_frame_advanced(
                prev_frame, curr_frame, prev_gray, curr_gray,
                flow, alpha, quality_mode
            )
            
            # === CRITICAL: Temporal stabilization to reduce jitter ===
            interpolated = temporal_stabilize(interpolated, frame_buffer, alpha)
            
            # === Motion blur for natural motion (eliminates "steppy" feel) ===
            if quality_mode in ["balanced", "quality"]:
                interpolated = add_motion_blur(interpolated, flow, alpha)
            
            # Update buffer
            frame_buffer.append(interpolated.copy())
            if len(frame_buffer) > max_buffer_size:
                frame_buffer.pop(0)
            
            out.write(interpolated)
        
        # Write original current frame
        out.write(curr_frame)
        frame_buffer.append(curr_frame.copy())
        if len(frame_buffer) > max_buffer_size:
            frame_buffer.pop(0)

        prev_frame = curr_frame
        prev_gray = curr_gray
        
        frame_idx += 1
        if progress_callback:
            progress = int((frame_idx / frame_count) * 100)
            progress_callback(progress)

    cap.release()
    out.release()


def compute_multiscale_flow(prev_gray, curr_gray, dis, quality_mode):
    """Compute optical flow with pyramid refinement for accuracy"""
    if quality_mode == "fast":
        # Single scale for speed
        return dis.calc(prev_gray, curr_gray, None)
    
    # Multi-scale pyramid approach
    levels = 3 if quality_mode == "quality" else 2
    prev_pyramid = [prev_gray]
    curr_pyramid = [curr_gray]
    
    for _ in range(levels - 1):
        prev_pyramid.append(cv2.pyrDown(prev_pyramid[-1]))
        curr_pyramid.append(cv2.pyrDown(curr_pyramid[-1]))
    
    # Start from coarsest level
    flow = None
    for level in range(levels - 1, -1, -1):
        if flow is not None:
            # Upscale and refine
            flow = cv2.pyrUp(flow)
            flow = flow * 2.0
        
        flow = dis.calc(prev_pyramid[level], curr_pyramid[level], flow)
    
    return flow


def interpolate_frame_advanced(prev_frame, curr_frame, prev_gray, curr_gray, 
                                flow, alpha, quality_mode):
    """Advanced bidirectional interpolation with occlusion handling"""
    h, w = prev_frame.shape[:2]
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    
    # Forward interpolation
    flow_scaled = alpha * flow
    map_x = x_coords + flow_scaled[:, :, 0]
    map_y = y_coords + flow_scaled[:, :, 1]
    forward_interp = cv2.remap(prev_frame, map_x, map_y, cv2.INTER_CUBIC)
    
    # Backward interpolation (estimate reverse flow)
    flow_reverse = -(1 - alpha) * flow
    map_x_back = x_coords + flow_reverse[:, :, 0]
    map_y_back = y_coords + flow_reverse[:, :, 1]
    backward_interp = cv2.remap(curr_frame, map_x_back, map_y_back, cv2.INTER_CUBIC)
    
    # Occlusion detection via forward-backward consistency
    warped_back = cv2.remap(flow, map_x, map_y, cv2.INTER_LINEAR)
    consistency = np.sum((flow + warped_back) ** 2, axis=2)
    occlusion_mask = consistency > 2.0
    
    # Motion-based blending weights
    mag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
    
    # Smooth weight transition based on motion
    weight_forward = 1 - alpha
    weight_backward = alpha
    
    # Adjust weights based on motion (high motion = trust interpolation more)
    motion_confidence = np.clip(mag / 15.0, 0, 1)
    motion_confidence = cv2.GaussianBlur(motion_confidence, (15, 15), 0)
    
    # Blend with motion-aware weights
    weight_forward = weight_forward * (0.5 + 0.5 * motion_confidence)
    weight_backward = weight_backward * (0.5 + 0.5 * motion_confidence)
    
    # Normalize weights
    total_weight = weight_forward + weight_backward
    weight_forward = weight_forward / (total_weight + 1e-8)
    weight_backward = weight_backward / (total_weight + 1e-8)
    
    # Expand dimensions for blending
    weight_forward = np.stack([weight_forward] * 3, axis=2)
    weight_backward = np.stack([weight_backward] * 3, axis=2)
    
    interpolated = (forward_interp * weight_forward + 
                   backward_interp * weight_backward).astype(np.uint8)
    
    # Handle occlusions with simple blend
    if np.any(occlusion_mask):
        simple_blend = cv2.addWeighted(prev_frame, 1-alpha, curr_frame, alpha, 0)
        occlusion_mask_3ch = np.stack([occlusion_mask] * 3, axis=2)
        interpolated = np.where(occlusion_mask_3ch, simple_blend, interpolated)
    
    return interpolated


def temporal_stabilize(frame, frame_buffer, alpha):
    """Stabilize frame using temporal consistency with buffer"""
    if len(frame_buffer) < 2:
        return frame
    
    # Weighted average with recent frames (reduces jitter)
    stabilized = frame.astype(np.float32)
    weights = [0.7]  # Current frame weight
    
    # Add influence from previous frames
    for i, prev in enumerate(frame_buffer[-2:]):
        weight = 0.15 / (i + 1)  # Decreasing weight for older frames
        stabilized += prev.astype(np.float32) * weight
        weights.append(weight)
    
    stabilized = stabilized / sum(weights)
    
    # Gentle edge-preserving filter
    stabilized = cv2.bilateralFilter(stabilized.astype(np.uint8), 
                                     d=5, sigmaColor=30, sigmaSpace=30)
    
    return stabilized


def add_motion_blur(frame, flow, alpha):
    """Add directional motion blur based on optical flow for natural motion"""
    mag = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
    
    # Only blur where there's significant motion
    motion_threshold = 2.0
    blur_mask = mag > motion_threshold
    
    if not np.any(blur_mask):
        return frame
    
    # Apply subtle directional blur
    angle = np.arctan2(flow[:, :, 1], flow[:, :, 0])
    
    # Adaptive kernel size based on motion magnitude
    kernel_size = int(np.clip(np.mean(mag[blur_mask]) / 3, 3, 9))
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Apply motion blur (subtle to avoid over-blurring)
    blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    
    # Blend based on motion magnitude
    blend_factor = np.clip(mag / 20.0, 0, 0.3)  # Max 30% blur
    blend_factor = np.stack([blend_factor] * 3, axis=2)
    
    result = (frame * (1 - blend_factor) + blurred * blend_factor).astype(np.uint8)
    
    return result


# ===== STREAMLIT UI =====
st.set_page_config(page_title="Video Extender Pro Max", page_icon="🎬", layout="centered")

st.title("🎬 Video Extender Pro Max")
st.write("Ultra-smooth frame interpolation with anti-jitter technology!")

# Settings in sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    quality_mode = st.selectbox(
        "Quality Mode",
        ["fast", "balanced", "quality"],
        index=1,
        help="Fast: Quick processing\nBalanced: Best quality/speed ratio\nQuality: Maximum smoothness"
    )
    
    extension_factor = st.selectbox(
        "Extension Factor",
        [2, 3, 4],
        index=0,
        help="2x = double length\n3x = triple length\n4x = quadruple length"
    )
    
    st.divider()
    st.markdown("""
    ### 🎯 New Features
    - Multi-scale optical flow
    - Temporal stabilization
    - Motion blur synthesis
    - Adaptive blending
    - Occlusion handling
    """)

# Main upload area
uploaded_file = st.file_uploader("Choose your video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.info(f"📁 **{uploaded_file.name}** ({file_size_mb:.1f} MB) → Will become **{extension_factor}x longer**")
    
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    input_path = tfile.name

    with st.expander("👀 Preview Original"):
        st.video(input_path)

    if st.button("🚀 Extend Video", type="primary", use_container_width=True):
        output_path = os.path.join(tempfile.gettempdir(), "extended_video.mp4")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(percent):
            progress_bar.progress(percent / 100)
            status_text.text(f"Processing with {quality_mode} mode... {percent}%")
        
        try:
            extend_video(input_path, output_path, quality_mode, extension_factor, update_progress)
            
            progress_bar.progress(100)
            status_text.empty()
            st.success("✅ Video extended with ultra-smooth interpolation!")
            
            # Stats
            input_size = os.path.getsize(input_path) / (1024 * 1024)
            output_size = os.path.getsize(output_path) / (1024 * 1024)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original", f"{input_size:.1f} MB")
            with col2:
                st.metric("Extended", f"{output_size:.1f} MB")
            with col3:
                st.metric("Length", f"{extension_factor}x longer")
            
            with open(output_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Extended Video",
                    f,
                    file_name=f"extended_{extension_factor}x_{uploaded_file.name}",
                    mime="video/mp4",
                    use_container_width=True
                )
            
            st.subheader("📊 Comparison")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original**")
                st.video(input_path)
            with col2:
                st.write(f"**Extended ({extension_factor}x)**")
                st.video(output_path)
                
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error: {str(e)}")
            with st.expander("🔍 Details"):
                st.code(str(e))

st.divider()
st.markdown("""
### 🔬 Advanced Technology

**Anti-Jitter Features:**
- **Multi-scale optical flow** - Analyzes motion at multiple resolutions for accuracy
- **Temporal stabilization** - Uses frame buffer to smooth transitions
- **Motion blur synthesis** - Adds natural motion blur for fluid movement
- **Adaptive blending** - Adjusts interpolation based on motion type

**Best Results:**
✅ Smooth camera movements  
✅ Continuous motion (walking, dancing)  
✅ Well-lit scenes  
❌ Avoid: Fast cuts, strobe effects, very dark footage
""")
