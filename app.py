import cv2
import numpy as np
import streamlit as st
import os
import tempfile

def extend_video(input_path, output_path):
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
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        # Resize if needed
        if curr_frame.shape[:2] != (height, width):
            curr_frame = cv2.resize(curr_frame, (width, height))
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # 1. Frame Analysis: Compute optical flow (Farneback for dense motion)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 
                                            pyr_scale=0.5, levels=3, winsize=15, 
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        # Detect average motion magnitude (for low-motion micro-loops/blending)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_mag = np.mean(mag)
        is_low_motion = avg_mag < 5  # Threshold for "low motion" (tune as needed)

        # 2. Temporal Extension: Generate interpolated frame (hybrid interp + blend for loops)
        alpha = 0.5  # Midpoint for 2x extension
        h, w = height, width
        interp_flow = alpha * flow
        # Create remap coordinates (approx backward warp from curr_frame)
        map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1)) + interp_flow[:, :, 0]
        map_y = np.repeat(np.arange(h, dtype=np.float32)[:, np.newaxis], w, axis=1) + interp_flow[:, :, 1]
        interpolated = cv2.remap(curr_frame, map_x, map_y, cv2.INTER_LINEAR)

        # If low motion, blend with duplicate for micro-loop effect (avoids repetition patterns)
        if is_low_motion:
            blended = cv2.addWeighted(prev_frame, 0.5, curr_frame, 0.5, 0)  # Cross-fade blend
            interpolated = cv2.addWeighted(interpolated, 0.7, blended, 0.3, 0)  # Mix interp with blend

        # 3. Post-Processing: Basic temporal smoothing (blend edges)
        smoothed = cv2.addWeighted(interpolated, 0.8, prev_frame, 0.2, 0)  # Light blend with prev for damping

        # Write interpolated + current
        out.write(smoothed)
        out.write(curr_frame)

        # Update prev
        prev_frame = curr_frame
        prev_gray = curr_gray

    # Cleanup
    cap.release()
    out.release()

# Streamlit UI
st.title("Spicy Video Extender MVP")
st.write("Upload a short clip (5-60s, MP4/MOV) and let's make it last longer... 😉")

uploaded_file = st.file_uploader("Choose your video", type=["mp4", "mov"])

if uploaded_file is not None:
    # Save temp input
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    input_path = tfile.name

    if st.button("Extend Video"):
        with st.spinner("Extending that provocative clip... Hold tight!"):
            output_path = os.path.join(tempfile.gettempdir(), "extended_video.mp4")
            try:
                extend_video(input_path, output_path)
                st.success("Done! Your extended video is ready.")
                with open(output_path, "rb") as f:
                    st.download_button("Download Extended Video", f, file_name="extended_video.mp4")
                # Preview (small)
                st.video(output_path)
            except Exception as e:
                st.error(f"Oops! Something went wrong: {e}")
