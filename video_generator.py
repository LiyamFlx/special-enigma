import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell for video frame prediction"""
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding
        )

    def forward(self, x, hidden_state):
        h, c = hidden_state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        i, f, o, g = torch.split(gates, self.hidden_channels, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class VideoExtrapolator(nn.Module):
    """Lightweight model for extending video clips beyond boundaries"""
    def __init__(self, hidden_channels=64):
        super().__init__()
        self.hidden_channels = hidden_channels

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # ConvLSTM for temporal modeling
        self.conv_lstm = ConvLSTMCell(64, hidden_channels)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input_sequence, num_predictions=1):
        """
        Args:
            input_sequence: [batch, time, channels, height, width]
            num_predictions: number of future frames to predict
        Returns:
            predicted frames: [batch, num_predictions, channels, height, width]
        """
        batch_size, seq_len, c, h, w = input_sequence.shape

        # Encode input sequence
        encoded_sequence = []
        for t in range(seq_len):
            encoded = self.encoder(input_sequence[:, t])
            encoded_sequence.append(encoded)

        # Initialize hidden states
        _, c_enc, h_enc, w_enc = encoded_sequence[0].shape
        h_state = torch.zeros(batch_size, self.hidden_channels, h_enc, w_enc).to(input_sequence.device)
        c_state = torch.zeros(batch_size, self.hidden_channels, h_enc, w_enc).to(input_sequence.device)

        # Process input sequence
        for t in range(seq_len):
            h_state, c_state = self.conv_lstm(encoded_sequence[t], (h_state, c_state))

        # Generate predictions
        predictions = []
        for _ in range(num_predictions):
            # Decode current state to frame
            pred_frame = self.decoder(h_state)
            predictions.append(pred_frame)

            # Update state for next prediction
            encoded_pred = self.encoder(pred_frame)
            h_state, c_state = self.conv_lstm(encoded_pred, (h_state, c_state))

        predictions = torch.stack(predictions, dim=1)
        return predictions


class MotionAwareExtender:
    """Motion-aware video extension using optical flow and frame prediction"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = VideoExtrapolator(hidden_channels=64).to(device)
        self.model.eval()  # Use in inference mode

    def preprocess_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Convert OpenCV frames to torch tensor"""
        # frames: list of [H, W, C] BGR images
        processed = []
        for frame in frames:
            # BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Normalize to [0, 1]
            frame_norm = frame_rgb.astype(np.float32) / 255.0
            # HWC to CHW
            frame_tensor = torch.from_numpy(frame_norm).permute(2, 0, 1)
            processed.append(frame_tensor)

        # Stack to [T, C, H, W]
        batch = torch.stack(processed, dim=0)
        # Add batch dimension [1, T, C, H, W]
        batch = batch.unsqueeze(0)
        return batch.to(self.device)

    def postprocess_frames(self, tensor: torch.Tensor) -> List[np.ndarray]:
        """Convert torch tensor to OpenCV frames"""
        # tensor: [1, T, C, H, W]
        tensor = tensor.squeeze(0).cpu()  # [T, C, H, W]
        frames = []
        for t in range(tensor.shape[0]):
            frame = tensor[t].permute(1, 2, 0).numpy()  # CHW to HWC
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frames.append(frame_bgr)
        return frames

    def extend_with_optical_flow(self, frames: List[np.ndarray], num_new_frames: int) -> List[np.ndarray]:
        """Extend using optical flow extrapolation (fast fallback method)"""
        if len(frames) < 2:
            # Just repeat last frame
            return [frames[-1].copy() for _ in range(num_new_frames)]

        # Compute optical flow from second-to-last to last frame
        gray1 = cv2.cvtColor(frames[-2], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        extended_frames = []
        last_frame = frames[-1].copy()
        h, w = last_frame.shape[:2]

        for i in range(1, num_new_frames + 1):
            # Create coordinate grid
            y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

            # Apply accumulated flow
            map_x = x_coords + flow[:, :, 0] * i
            map_y = y_coords + flow[:, :, 1] * i

            # Warp frame
            warped = cv2.remap(
                last_frame, map_x, map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )

            # Blend with last frame to reduce artifacts
            alpha = 1.0 / (1.0 + i * 0.3)  # Gradually fade
            blended = cv2.addWeighted(warped, alpha, last_frame, 1 - alpha, 0)

            extended_frames.append(blended)

        return extended_frames

    def extend_forward(self, frames: List[np.ndarray], num_new_frames: int,
                      use_neural: bool = True) -> List[np.ndarray]:
        """
        Extend video forward (generate frames after last frame)

        Args:
            frames: List of input frames (last 5-10 frames recommended)
            num_new_frames: Number of frames to generate
            use_neural: Use neural prediction (slower) vs optical flow (faster)
        """
        if not use_neural or len(frames) < 3:
            return self.extend_with_optical_flow(frames, num_new_frames)

        try:
            with torch.no_grad():
                # Use last 5 frames as context
                context_frames = frames[-5:] if len(frames) >= 5 else frames

                # Preprocess
                input_tensor = self.preprocess_frames(context_frames)

                # Generate predictions
                predictions = self.model(input_tensor, num_predictions=num_new_frames)

                # Postprocess
                new_frames = self.postprocess_frames(predictions)

                return new_frames
        except Exception as e:
            print(f"Neural extension failed: {e}, falling back to optical flow")
            return self.extend_with_optical_flow(frames, num_new_frames)

    def extend_backward(self, frames: List[np.ndarray], num_new_frames: int) -> List[np.ndarray]:
        """
        Extend video backward (generate frames before first frame)
        Reverse the sequence, extend, then reverse back
        """
        reversed_frames = list(reversed(frames))
        extended = self.extend_forward(reversed_frames, num_new_frames, use_neural=False)
        return list(reversed(extended))


def interpolate_frames(frame1: np.ndarray, frame2: np.ndarray,
                       num_interpolations: int = 1) -> List[np.ndarray]:
    """
    Interpolate frames between two frames using optical flow

    Args:
        frame1: First frame
        frame2: Second frame
        num_interpolations: Number of frames to insert between
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute bidirectional optical flow
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow_forward = dis.calc(gray1, gray2, None)
    flow_backward = dis.calc(gray2, gray1, None)

    h, w = frame1.shape[:2]
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

    interpolated = []
    for i in range(1, num_interpolations + 1):
        alpha = i / (num_interpolations + 1)

        # Forward warp
        map_x_fwd = x_coords + alpha * flow_forward[:, :, 0]
        map_y_fwd = y_coords + alpha * flow_forward[:, :, 1]
        forward_warp = cv2.remap(frame1, map_x_fwd, map_y_fwd,
                                 cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Backward warp
        map_x_bwd = x_coords + (1 - alpha) * flow_backward[:, :, 0]
        map_y_bwd = y_coords + (1 - alpha) * flow_backward[:, :, 1]
        backward_warp = cv2.remap(frame2, map_x_bwd, map_y_bwd,
                                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Blend
        blended = cv2.addWeighted(forward_warp, 1 - alpha, backward_warp, alpha, 0)
        interpolated.append(blended)

    return interpolated


def extend_video_generative(input_path: str, output_path: str,
                            extension_seconds: float = 5.0,
                            quality_mode: str = "balanced",
                            progress_callback=None) -> dict:
    """
    Extend video using generative AI approach

    Args:
        input_path: Input video path
        output_path: Output video path
        extension_seconds: Duration to extend (5 or 10 seconds)
        quality_mode: "fast", "balanced", or "quality"
        progress_callback: Progress callback function

    Returns:
        dict with processing statistics
    """
    # Initialize
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Downscale if needed
    if height > 1080:
        scale = 1080 / height
        width = int(width * scale)
        height = 1080

    # Calculate extension parameters
    num_extension_frames = int(extension_seconds * fps)
    num_interpolations = 1 if quality_mode == "fast" else (2 if quality_mode == "quality" else 1)

    # Setup output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Read all frames
    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        all_frames.append(frame)
    cap.release()

    if len(all_frames) < 2:
        raise ValueError("Video too short")

    # Initialize extender
    extender = MotionAwareExtender()

    total_operations = len(all_frames) + num_extension_frames
    processed = 0

    # Write original frames with interpolation
    for i in range(len(all_frames)):
        out.write(all_frames[i])
        processed += 1

        if progress_callback:
            progress_callback(int((processed / total_operations) * 90))

        # Interpolate between frames
        if i < len(all_frames) - 1 and num_interpolations > 0:
            interpolated = interpolate_frames(
                all_frames[i],
                all_frames[i + 1],
                num_interpolations
            )
            for interp_frame in interpolated:
                out.write(interp_frame)

    # Extend forward using generative approach
    if progress_callback:
        progress_callback(90)

    context_frames = all_frames[-10:] if len(all_frames) >= 10 else all_frames
    use_neural = quality_mode in ["balanced", "quality"]

    extended_frames = extender.extend_forward(
        context_frames,
        num_extension_frames,
        use_neural=use_neural
    )

    for i, ext_frame in enumerate(extended_frames):
        out.write(ext_frame)
        processed += 1
        if progress_callback:
            progress = 90 + int((i / num_extension_frames) * 10)
            progress_callback(progress)

    out.release()

    stats = {
        'original_frames': len(all_frames),
        'extension_frames': num_extension_frames,
        'total_frames': len(all_frames) + num_extension_frames,
        'original_duration': len(all_frames) / fps,
        'extended_duration': (len(all_frames) + num_extension_frames) / fps,
        'extension_added': extension_seconds
    }

    return stats
