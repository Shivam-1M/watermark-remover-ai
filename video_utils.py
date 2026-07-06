"""
=============================================================================
video_utils.py — FFmpeg Video Processing Pipeline
=============================================================================
Utility functions for video manipulation using ffmpeg-python.

Functions:

    extract_audio()     — Separate the audio track from a video file.
    split_frames()      — Decompose a video into numbered PNG frames.
    reassemble_video()  — Recompose PNG frames into an MP4 with audio.
    get_video_fps()     — Probe the video's frame rate.

All functions use ffmpeg-python which wraps the ffmpeg CLI.
FFmpeg must be installed in the system PATH (handled by the Dockerfile).

Security Notes:
    - All file paths are constructed from UUID-based task directories.
    - No user-supplied strings are passed directly to ffmpeg commands.
    - TODO(security): Validate video content integrity (magic bytes) before
      processing to prevent malicious file exploitation.
=============================================================================
"""

import os
import glob
import logging
import subprocess

import ffmpeg

logger = logging.getLogger("watermark-app.video_utils")


def extract_audio(video_path: str, audio_path: str) -> None:
    """
    Extract the audio track from a video file.

    The audio is saved in AAC format for later remuxing with the
    processed video frames.

    If the video has no audio track, this function logs a warning
    and creates an empty placeholder file so downstream code can
    detect the absence.

    Args:
        video_path: Absolute path to the input video file.
        audio_path: Absolute path for the output audio file (.aac).

    Raises:
        RuntimeError: If ffmpeg fails unexpectedly.
    """
    try:
        # Probe the video to check if it has an audio stream
        probe = ffmpeg.probe(video_path)
        audio_streams = [
            s for s in probe.get("streams", [])
            if s.get("codec_type") == "audio"
        ]

        if not audio_streams:
            logger.warning("No audio track found in video: %s", video_path)
            # Create an empty marker file so reassembly knows to skip audio
            with open(audio_path + ".noaudio", "w") as f:
                f.write("no_audio")
            return

        (
            ffmpeg
            .input(video_path)
            .output(audio_path, acodec="aac", vn=None)  # Audio only, no video
            .overwrite_output()
            .run(quiet=True, capture_stderr=True)
        )
        logger.info("Extracted audio: %s", audio_path)

    except ffmpeg.Error as e:
        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        logger.error("FFmpeg audio extraction failed: %s", stderr)
        raise RuntimeError("Failed to extract audio from video.")


def split_frames(video_path: str, frames_dir: str) -> int:
    """
    Split a video into individual PNG frames.

    Frames are saved as sequentially numbered files:
        frame_00001.png, frame_00002.png, ...

    Args:
        video_path: Absolute path to the input video file.
        frames_dir: Absolute path to the output directory for frames.

    Returns:
        The total number of frames extracted.

    Raises:
        RuntimeError: If ffmpeg fails to split the video.
    """
    os.makedirs(frames_dir, exist_ok=True)

    # Output pattern: frame_00001.png, frame_00002.png, etc.
    output_pattern = os.path.join(frames_dir, "frame_%05d.png")

    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_pattern, format="image2")
            .overwrite_output()
            .run(quiet=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        logger.error("FFmpeg frame split failed: %s", stderr)
        raise RuntimeError("Failed to split video into frames.")

    # Count the extracted frames
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    frame_count = len(frame_files)
    logger.info("Split video into %d frames in: %s", frame_count, frames_dir)

    return frame_count


def get_video_fps(video_path: str) -> float:
    """
    Probe the video file to determine its frame rate (FPS).

    Args:
        video_path: Absolute path to the video file.

    Returns:
        The frame rate as a float (e.g., 29.97, 30.0, 60.0).

    Raises:
        RuntimeError: If probing fails or no video stream is found.
    """
    try:
        probe = ffmpeg.probe(video_path)
        video_streams = [
            s for s in probe.get("streams", [])
            if s.get("codec_type") == "video"
        ]

        if not video_streams:
            raise RuntimeError("No video stream found in file.")

        # Parse the frame rate string (e.g., "30000/1001" or "30/1")
        r_frame_rate = video_streams[0].get("r_frame_rate", "30/1")
        num, den = r_frame_rate.split("/")
        fps = float(num) / float(den)
        logger.info("Detected video FPS: %.2f", fps)
        return fps

    except ffmpeg.Error as e:
        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        logger.error("FFmpeg probe failed: %s", stderr)
        raise RuntimeError("Failed to determine video frame rate.")


def reassemble_video(
    frames_dir: str,
    audio_path: str,
    output_path: str,
    fps: float = 30.0,
    overlay_path: str | None = None,
) -> None:
    """
    Reassemble PNG frames into an MP4 video and remux the original audio.

    The output video uses H.264 (libx264) encoding with the 'yuv420p'
    pixel format for maximum compatibility. CRF 18 provides near-lossless
    quality; adjust for file size vs. quality tradeoffs.

    Args:
        frames_dir:  Directory containing numbered PNG frames.
        audio_path:  Path to the extracted audio file (.aac).
        output_path: Path for the final output .mp4 file.
        fps:         Frame rate for the output video.

    Raises:
        RuntimeError: If ffmpeg fails during reassembly.
    """
    # Input: sequentially numbered frame images
    frame_pattern = os.path.join(frames_dir, "frame_%05d.png")

    # Check if there's an audio track to include
    has_audio = os.path.isfile(audio_path) and not os.path.isfile(
        audio_path + ".noaudio"
    )

    try:
        # Base video stream
        video_input = ffmpeg.input(
            frame_pattern,
            framerate=fps,
            format="image2",
        )

        # Apply overlay filter if a valid overlay image is provided
        if overlay_path and os.path.isfile(overlay_path):
            overlay_input = ffmpeg.input(overlay_path)
            video_stream = ffmpeg.overlay(video_input, overlay_input)
            logger.info("Applying logo overlay filter via FFmpeg.")
        else:
            video_stream = video_input

        if has_audio:
            # --- Reassemble with audio ---
            audio_input = ffmpeg.input(audio_path)

            (
                ffmpeg
                .output(
                    video_stream,
                    audio_input,
                    output_path,
                    vcodec="libx264",     # H.264 encoder
                    pix_fmt="yuv420p",    # Widely compatible pixel format
                    crf=18,               # High quality (lower = better)
                    preset="medium",      # Encoding speed vs. compression
                    acodec="aac",         # Re-encode audio as AAC
                    shortest=None,        # End when shortest stream ends
                )
                .overwrite_output()
                .run(quiet=True, capture_stderr=True)
            )
        else:
            # --- Reassemble without audio ---
            logger.info("Reassembling video without audio track.")
            (
                ffmpeg
                .output(
                    video_stream,
                    output_path,
                    vcodec="libx264",
                    pix_fmt="yuv420p",
                    crf=18,
                    preset="medium",
                )
                .overwrite_output()
                .run(quiet=True, capture_stderr=True)
            )

        logger.info("Reassembled video saved to: %s", output_path)

    except ffmpeg.Error as e:
        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        logger.error("FFmpeg reassembly failed: %s", stderr)
        raise RuntimeError("Failed to reassemble video from frames.")
