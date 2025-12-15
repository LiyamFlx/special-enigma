import numpy as np
import librosa
import soundfile as sf
import cv2
from scipy import signal
from typing import Optional, Tuple
import os
import tempfile


def extract_audio_from_video(video_path: str, output_audio_path: Optional[str] = None) -> Optional[str]:
    """
    Extract audio from video file

    Args:
        video_path: Path to video file
        output_audio_path: Path for output audio (if None, creates temp file)

    Returns:
        Path to extracted audio file, or None if no audio
    """
    if output_audio_path is None:
        output_audio_path = tempfile.mktemp(suffix='.wav')

    try:
        # Use OpenCV to check if video has audio
        cap = cv2.VideoCapture(video_path)
        cap.release()

        # Extract audio using ffmpeg via imageio
        import imageio_ffmpeg as ffmpeg
        import subprocess

        cmd = [
            ffmpeg.get_ffmpeg_exe(),
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM audio
            '-ar', '44100',  # Sample rate
            '-ac', '2',  # Stereo
            '-y',  # Overwrite
            output_audio_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if os.path.exists(output_audio_path) and os.path.getsize(output_audio_path) > 0:
            return output_audio_path
        else:
            return None

    except Exception as e:
        print(f"Audio extraction failed: {e}")
        return None


def extend_audio_smooth(audio: np.ndarray, sr: int, extension_seconds: float) -> np.ndarray:
    """
    Extend audio smoothly using looping with crossfade

    Args:
        audio: Audio samples [channels, samples] or [samples]
        sr: Sample rate
        extension_seconds: Duration to extend in seconds

    Returns:
        Extended audio
    """
    extension_samples = int(extension_seconds * sr)

    # Ensure audio is 2D [channels, samples]
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
        was_mono = True
    else:
        was_mono = False

    num_channels, num_samples = audio.shape

    # Use last 2 seconds for looping material
    loop_duration = min(2.0, num_samples / sr)
    loop_samples = int(loop_duration * sr)
    loop_material = audio[:, -loop_samples:]

    # Create extension by repeating and crossfading
    extended_part = []
    crossfade_samples = int(0.1 * sr)  # 100ms crossfade

    remaining = extension_samples
    while remaining > 0:
        chunk_size = min(loop_samples, remaining)
        chunk = loop_material[:, :chunk_size]

        if len(extended_part) > 0 and crossfade_samples > 0:
            # Crossfade with previous chunk
            fade_in = np.linspace(0, 1, crossfade_samples)
            fade_out = np.linspace(1, 0, crossfade_samples)

            # Apply crossfade
            overlap = min(crossfade_samples, chunk_size, extended_part[-1].shape[1])
            for ch in range(num_channels):
                extended_part[-1][ch, -overlap:] *= fade_out[:overlap]
                chunk[ch, :overlap] *= fade_in[:overlap]
                extended_part[-1][ch, -overlap:] += chunk[ch, :overlap]

            extended_part.append(chunk[:, overlap:])
        else:
            extended_part.append(chunk)

        remaining -= chunk_size

    # Concatenate
    if extended_part:
        extension = np.concatenate(extended_part, axis=1)
    else:
        extension = np.zeros((num_channels, extension_samples))

    # Combine original + extension
    result = np.concatenate([audio, extension], axis=1)

    # Convert back to mono if input was mono
    if was_mono:
        result = result[0]

    return result


def extend_audio_phase_vocoder(audio: np.ndarray, sr: int,
                                extension_seconds: float,
                                quality: str = "balanced") -> np.ndarray:
    """
    Extend audio using phase vocoder for time stretching

    Args:
        audio: Audio samples
        sr: Sample rate
        extension_seconds: Duration to extend
        quality: "fast", "balanced", or "quality"

    Returns:
        Extended audio
    """
    # Calculate target duration
    original_duration = len(audio) / sr
    target_duration = original_duration + extension_seconds

    # Time stretch ratio
    stretch_ratio = target_duration / original_duration

    # Use librosa for high-quality time stretching
    hop_length = {
        'fast': 1024,
        'balanced': 512,
        'quality': 256
    }.get(quality, 512)

    try:
        stretched = librosa.effects.time_stretch(audio, rate=1/stretch_ratio, hop_length=hop_length)
        return stretched
    except Exception as e:
        print(f"Phase vocoder failed: {e}, using smooth extension")
        return extend_audio_smooth(audio, sr, extension_seconds)


def merge_audio_video(video_path: str, audio_path: str, output_path: str):
    """
    Merge audio and video files

    Args:
        video_path: Path to video file
        audio_path: Path to audio file
        output_path: Path for output file
    """
    import imageio_ffmpeg as ffmpeg
    import subprocess

    cmd = [
        ffmpeg.get_ffmpeg_exe(),
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',  # Copy video codec
        '-c:a', 'aac',   # AAC audio codec
        '-strict', 'experimental',
        '-shortest',  # Match shortest stream
        '-y',  # Overwrite
        output_path
    ]

    subprocess.run(cmd, capture_output=True)


def process_video_with_audio(input_video: str, output_video_no_audio: str,
                             output_video_final: str, extension_seconds: float,
                             quality: str = "balanced") -> bool:
    """
    Complete audio processing pipeline for extended video

    Args:
        input_video: Original video with audio
        output_video_no_audio: Extended video without audio
        output_video_final: Final output with extended audio
        extension_seconds: How much to extend audio
        quality: Quality mode

    Returns:
        True if audio was processed, False if no audio in original
    """
    # Extract audio from original
    temp_audio_original = tempfile.mktemp(suffix='.wav')
    audio_path = extract_audio_from_video(input_video, temp_audio_original)

    if audio_path is None or not os.path.exists(audio_path):
        # No audio in original, just copy video
        import shutil
        shutil.copy(output_video_no_audio, output_video_final)
        return False

    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None, mono=False)

        # Extend audio
        if quality == "fast":
            extended_audio = extend_audio_smooth(audio, sr, extension_seconds)
        else:
            extended_audio = extend_audio_phase_vocoder(audio, sr, extension_seconds, quality)

        # Save extended audio
        temp_audio_extended = tempfile.mktemp(suffix='.wav')

        # Ensure proper shape for soundfile
        if extended_audio.ndim == 1:
            audio_to_save = extended_audio
        else:
            audio_to_save = extended_audio.T  # soundfile expects [samples, channels]

        sf.write(temp_audio_extended, audio_to_save, sr)

        # Merge with video
        merge_audio_video(output_video_no_audio, temp_audio_extended, output_video_final)

        # Cleanup temp files
        if os.path.exists(temp_audio_original):
            os.remove(temp_audio_original)
        if os.path.exists(temp_audio_extended):
            os.remove(temp_audio_extended)

        return True

    except Exception as e:
        print(f"Audio processing error: {e}")
        # Fallback: copy video without audio
        import shutil
        shutil.copy(output_video_no_audio, output_video_final)
        return False
