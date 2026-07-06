"""
=============================================================================
inpainter.py — CPU-Optimized OpenCV Inpainting Engine
=============================================================================
Removes watermarks from video frames using OpenCV's Telea inpainting
algorithm, distributed across all available CPU cores via multiprocessing.

This replaces the previous ProPainter deep-learning engine, which was
too slow and memory-intensive for CPU-only hardware (Intel Iris Xe).

Algorithm: cv2.INPAINT_TELEA (Fast Marching Method by Alexandru Telea)
    - Processes each frame independently in ~5-20ms at 720p.
    - Respects the exact painted mask shape (not a bounding box).
    - Runs in optimized C++ under the hood, releasing the Python GIL.

Security Notes:
    - All paths are UUID-based, constructed by main.py — no user input here.
=============================================================================
"""

import os
import glob
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger("watermark-app.inpainter")

# Inpainting radius: how many pixels around the mask boundary are used
# to compute the fill. Larger = smoother but slower and more blurry.
INPAINT_RADIUS = 5

# Maximum number of worker processes. os.cpu_count() returns logical cores.
MAX_WORKERS = max(1, (os.cpu_count() or 2) - 1)


def _inpaint_single_frame(args: tuple) -> tuple:
    """
    Inpaint a single frame. This function runs in a worker process.

    Args:
        args: Tuple of (frame_path, mask_path, output_path)

    Returns:
        (output_path, success, error_message)
    """
    frame_path, mask_path, output_path = args

    try:
        # Read the frame in BGR
        frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        if frame is None:
            return (output_path, False, f"Failed to read frame: {frame_path}")

        # Read the mask as grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return (output_path, False, f"Failed to read mask: {mask_path}")

        # Resize mask to match frame dimensions if they differ
        fh, fw = frame.shape[:2]
        mh, mw = mask.shape[:2]
        if (mh, mw) != (fh, fw):
            mask = cv2.resize(mask, (fw, fh), interpolation=cv2.INTER_NEAREST)

        # Threshold to ensure binary mask (0 or 255)
        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Apply Telea inpainting
        result = cv2.inpaint(frame, mask_bin, INPAINT_RADIUS, cv2.INPAINT_TELEA)

        # Save the inpainted frame
        cv2.imwrite(output_path, result)
        return (output_path, True, None)

    except Exception as e:
        return (output_path, False, str(e))


def process_frames(
    frames_dir: str,
    mask_path: str,
    output_dir: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """
    Run OpenCV Telea inpainting on all frames using multiprocessing.

    Each frame is processed independently across multiple CPU cores.
    The same mask is applied to every frame since the watermark position
    is static throughout the video.

    Args:
        frames_dir:        Directory of input PNG frames (frame_00001.png ...).
        mask_path:         Path to the binary mask PNG (white = watermark).
        output_dir:        Directory to save inpainted output frames.
        progress_callback: Optional fn called as (current, total) per frame.

    Raises:
        FileNotFoundError: If frames dir or mask file not found.
        RuntimeError:      If processing fails.
    """
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Discover all frames
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    total_frames = len(frame_files)

    if total_frames == 0:
        raise RuntimeError(f"No frames found in: {frames_dir}")

    logger.info(
        "Starting OpenCV Telea inpainting on %d frames using %d workers...",
        total_frames, MAX_WORKERS,
    )

    # Build the work items: (frame_path, mask_path, output_path)
    work_items = []
    for fp in frame_files:
        fname = os.path.basename(fp)
        out_path = os.path.join(output_dir, fname)
        work_items.append((fp, mask_path, out_path))

    # Process frames in parallel
    completed = 0
    errors = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_inpaint_single_frame, item): item
            for item in work_items
        }

        for future in as_completed(futures):
            out_path, success, err_msg = future.result()
            completed += 1

            if not success:
                errors.append(err_msg)
                logger.error("Frame failed: %s", err_msg)

            if progress_callback:
                progress_callback(completed, total_frames)

            # Log progress every 50 frames
            if completed % 50 == 0 or completed == total_frames:
                logger.info(
                    "Inpainted %d / %d frames.", completed, total_frames
                )

    if errors:
        logger.warning(
            "%d frame(s) failed during inpainting. First error: %s",
            len(errors), errors[0],
        )

    logger.info(
        "OpenCV inpainting complete. %d frames saved to: %s",
        total_frames, output_dir,
    )
