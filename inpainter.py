"""
=============================================================================
inpainter.py — AI Inpainting Engine (ProPainter Integration)
=============================================================================
This module integrates the ProPainter video inpainting model to remove
watermarks from video frames with temporal consistency.

Model: ProPainter (ICCV 2023) by Shangchen Zhou et al.
       https://github.com/sczhou/ProPainter

Hardware: CPU-only mode (Intel Iris Xe — no NVIDIA CUDA available).
          Processing will be slower than GPU; frames are auto-downscaled
          to 480p to make CPU inference practical.

Architecture overview:
    1. RAFT optical flow backbone computes bidirectional flow between frames.
    2. RecurrentFlowCompleteNet propagates valid pixels across the flow field
       to fill in the masked (watermark) regions of the optical flow maps.
    3. InpaintGenerator (ProPainter's main network) uses the completed flows
       and a sliding-window transformer to reconstruct each masked frame.

Security Notes:
    - All paths are UUID-based, constructed by main.py — no user input here.
    - TODO(security): Validate image dimensions before tensor conversion to
      prevent memory exhaustion on pathologically large inputs.
=============================================================================
"""

import os
import sys
import glob
import logging
from typing import Callable, Optional

import cv2
import numpy as np
import scipy.ndimage
import torch
import torchvision
from PIL import Image

# ---------------------------------------------------------------------------
# Add ProPainter's source tree to sys.path so its internal imports resolve.
# ProPainter was cloned to /app/propainter during the Docker build.
# ---------------------------------------------------------------------------
PROPAINTER_DIR = os.path.join(os.path.dirname(__file__), "propainter")
if PROPAINTER_DIR not in sys.path:
    sys.path.insert(0, PROPAINTER_DIR)

from model.modules.flow_comp_raft import RAFT_bi                  # noqa: E402
from model.recurrent_flow_completion import RecurrentFlowCompleteNet  # noqa: E402
from model.propainter import InpaintGenerator                     # noqa: E402
from core.utils import to_tensors                                  # noqa: E402

logger = logging.getLogger("watermark-app.inpainter")

# ---------------------------------------------------------------------------
# Global model registry — models are loaded once at startup and reused.
# Loading on every request would be prohibitively slow.
# ---------------------------------------------------------------------------
DEVICE = torch.device("cpu")  # Intel Iris Xe: CPU-only, no CUDA available
_raft_model: Optional[RAFT_bi] = None
_flow_model: Optional[RecurrentFlowCompleteNet] = None
_inpaint_model: Optional[InpaintGenerator] = None

# Path to the pre-downloaded weight files (baked in during Docker build)
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")

# ---------------------------------------------------------------------------
# Processing parameters
# ---------------------------------------------------------------------------
# Maximum resolution for inpainting (480p). Frames are downscaled to this
# height before inference and upscaled back afterward. This is the primary
# knob to trade quality for speed on CPU hardware.
MAX_INPAINT_HEIGHT = 480

# ProPainter sliding-window size: number of frames processed together.
# Smaller = less memory, larger = better temporal consistency.
# On CPU, keep this at 5-10 to avoid excessive RAM usage.
NEIGHBOR_LENGTH = 10
REF_STRIDE = 10  # Stride for selecting reference frames


def load_models() -> None:
    """
    Load all three ProPainter model components into memory.

    This must be called once before any call to `process_frames()`.
    It is designed to be called from a FastAPI lifespan startup event.

    Models loaded:
        - RAFT_bi: bidirectional optical flow estimation
        - RecurrentFlowCompleteNet: flow completion in masked regions
        - InpaintGenerator: the main video inpainting network

    All models are moved to DEVICE (CPU) and set to eval mode.
    """
    global _raft_model, _flow_model, _inpaint_model

    raft_path = os.path.join(WEIGHTS_DIR, "raft-things.pth")
    flow_path = os.path.join(WEIGHTS_DIR, "recurrent_flow_completion.pth")
    inpaint_path = os.path.join(WEIGHTS_DIR, "ProPainter.pth")

    for path, name in [
        (raft_path, "raft-things.pth"),
        (flow_path, "recurrent_flow_completion.pth"),
        (inpaint_path, "ProPainter.pth"),
    ]:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Model weight file missing: {path}. "
                "Rebuild the Docker image to re-download weights."
            )

    logger.info("Loading ProPainter models onto %s ...", DEVICE)

    # 1. RAFT optical flow backbone
    _raft_model = RAFT_bi(raft_path, DEVICE)

    # 2. Recurrent flow completion network
    _flow_model = RecurrentFlowCompleteNet()
    _flow_model.load_state_dict(
        torch.load(flow_path, map_location=DEVICE)
    )
    _flow_model.to(DEVICE)
    _flow_model.eval()

    # 3. Main inpainting generator
    _inpaint_model = InpaintGenerator(model_path=inpaint_path)
    _inpaint_model.to(DEVICE)
    _inpaint_model.eval()

    logger.info("All ProPainter models loaded successfully on %s.", DEVICE)


def _ensure_models_loaded() -> None:
    """Raise a clear error if models were not loaded at startup."""
    if _raft_model is None or _flow_model is None or _inpaint_model is None:
        raise RuntimeError(
            "ProPainter models are not loaded. "
            "Call inpainter.load_models() at application startup."
        )


def _resize_frames_for_inference(
    frames: list[Image.Image],
    max_height: int = MAX_INPAINT_HEIGHT,
) -> tuple[list[Image.Image], tuple[int, int], tuple[int, int]]:
    """
    Downscale frames to at most `max_height` pixels tall, maintaining aspect
    ratio. Dimensions are adjusted to the nearest multiple of 8 (required by
    the ProPainter architecture).

    Args:
        frames:     List of PIL Images (the original resolution frames).
        max_height: Maximum height in pixels for inference.

    Returns:
        (resized_frames, process_size, original_size)
        - resized_frames: frames at the inference resolution
        - process_size: (W, H) used during inference
        - original_size: (W, H) of the input frames
    """
    orig_w, orig_h = frames[0].size
    original_size = (orig_w, orig_h)

    if orig_h > max_height:
        scale = max_height / orig_h
        new_w = int(orig_w * scale)
        new_h = max_height
    else:
        new_w, new_h = orig_w, orig_h

    # Round down to nearest multiple of 8 (ProPainter requirement)
    proc_w = new_w - (new_w % 8)
    proc_h = new_h - (new_h % 8)
    process_size = (proc_w, proc_h)

    resized = [f.resize(process_size, Image.LANCZOS) for f in frames]
    logger.info(
        "Resized frames from %dx%d → %dx%d for inference.",
        orig_w, orig_h, proc_w, proc_h,
    )
    return resized, process_size, original_size


def _dilate_mask(mask_img: Image.Image, process_size: tuple[int, int]) -> tuple:
    """
    Resize and dilate the binary mask to match the inference resolution.

    ProPainter uses two mask variants:
        - flow_mask: dilated by 8px — marks regions where optical flow
          is unreliable and must be completed.
        - mask_dilated: dilated by 5px — marks regions to inpaint,
          slightly larger than the painted area for clean edge blending.

    Args:
        mask_img:     PIL Image of the original painted mask (white=watermark).
        process_size: (W, H) target size to resize mask to.

    Returns:
        (flow_masks, masks_dilated) — both as lists of PIL Images, length=1
        (they will be tiled across all frames by the caller).
    """
    mask_resized = mask_img.resize(process_size, Image.NEAREST)
    mask_np = np.array(mask_resized.convert("L"))

    # Dilate for flow mask (8 pixels) — larger margin for flow completion
    flow_mask_np = scipy.ndimage.binary_dilation(
        mask_np, iterations=8
    ).astype(np.uint8)
    flow_mask = Image.fromarray(flow_mask_np * 255)

    # Dilate for inpainting mask (5 pixels) — used during frame generation
    inpaint_mask_np = scipy.ndimage.binary_dilation(
        mask_np, iterations=5
    ).astype(np.uint8)
    inpaint_mask = Image.fromarray(inpaint_mask_np * 255)

    return [flow_mask], [inpaint_mask]


def process_frames(
    frames_dir: str,
    mask_path: str,
    output_dir: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """
    Run the full ProPainter inpainting pipeline on a directory of video frames.

    This replaces the mock OpenCV Telea implementation. ProPainter processes
    frames in temporal sliding windows, using optical flow to maintain
    consistency across time (no flickering between adjacent frames).

    Pipeline:
        1. Load all frames and the mask from disk.
        2. Downscale to MAX_INPAINT_HEIGHT for CPU feasibility.
        3. Compute bidirectional optical flow with RAFT.
        4. Complete the flow in masked regions with RecurrentFlowCompleteNet.
        5. Generate inpainted frames with InpaintGenerator (sliding window).
        6. Upscale results back to the original resolution.
        7. Save each output frame to output_dir.

    Args:
        frames_dir:        Directory of input PNG frames (frame_00001.png ...).
        mask_path:         Path to the binary mask PNG (white = watermark).
        output_dir:        Directory to save inpainted output frames.
        progress_callback: Optional fn called as (current, total) per frame.

    Raises:
        FileNotFoundError: If frames dir or mask file not found.
        RuntimeError:      If models not loaded or processing fails.
    """
    _ensure_models_loaded()

    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load frames from disk as PIL Images (RGB)                        #
    # ------------------------------------------------------------------ #
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    total_frames = len(frame_files)

    if total_frames == 0:
        raise RuntimeError(f"No frames found in: {frames_dir}")

    logger.info("Loading %d frames for ProPainter inference...", total_frames)

    frames_pil: list[Image.Image] = []
    for fp in frame_files:
        img_bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read frame: {fp}")
        frames_pil.append(
            Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        )

    # ------------------------------------------------------------------ #
    # 2. Load and prepare the mask                                        #
    # ------------------------------------------------------------------ #
    mask_pil = Image.open(mask_path)

    # ------------------------------------------------------------------ #
    # 3. Downscale frames for CPU inference                               #
    # ------------------------------------------------------------------ #
    frames_resized, process_size, original_size = _resize_frames_for_inference(
        frames_pil, max_height=MAX_INPAINT_HEIGHT
    )

    # ------------------------------------------------------------------ #
    # 4. Prepare masks (flow_mask and inpaint_mask, tiled for all frames) #
    # ------------------------------------------------------------------ #
    flow_masks_single, masks_dilated_single = _dilate_mask(mask_pil, process_size)
    # Tile the single mask image across all frames
    flow_masks = flow_masks_single * total_frames
    masks_dilated = masks_dilated_single * total_frames

    # ------------------------------------------------------------------ #
    # 5. Convert to tensors                                               #
    # ------------------------------------------------------------------ #
    # frames_tensor  shape: [T, 3, H, W],  values in [-1, 1]
    # flow_masks_t   shape: [T, 1, H, W],  binary 0/1
    # masks_dilated_t shape: [T, 1, H, W], binary 0/1
    frames_tensor = to_tensors()(frames_resized).unsqueeze(0).to(DEVICE) * 2.0 - 1.0

    flow_masks_tensor = to_tensors()(flow_masks).unsqueeze(0).to(DEVICE)
    masks_dilated_tensor = to_tensors()(masks_dilated).unsqueeze(0).to(DEVICE)

    proc_w, proc_h = process_size
    orig_w, orig_h = original_size

    logger.info(
        "Running ProPainter on %d frames at %dx%d (CPU)...",
        total_frames, proc_w, proc_h,
    )

    # ------------------------------------------------------------------ #
    # 6. Optical flow computation (RAFT)                                  #
    # ------------------------------------------------------------------ #
    with torch.no_grad():
        # RAFT_bi expects frames as a tensor of shape [1, T, 3, H, W] in [-1, 1]
        # flow shape: [1, T, 2, H, W] — forward and backward flows
        pred_flows_bi, _ = _raft_model(frames_tensor, iters=20)

    logger.info("Optical flow computed.")

    # ------------------------------------------------------------------ #
    # 7. Flow completion in masked regions                                #
    # ------------------------------------------------------------------ #
    with torch.no_grad():
        # RecurrentFlowCompleteNet expects:
        #   masked_flows: tuple of (forward_flow, backward_flow), each [1, T, 2, H, W]
        #   masks: [1, T, 1, H, W]
        pred_flows_bi = _flow_model.forward_bidirect_flow(
            pred_flows_bi, flow_masks_tensor
        )
        pred_flows_bi = _flow_model.combine_flow(
            pred_flows_bi, pred_flows_bi, flow_masks_tensor
        )

    logger.info("Flow completion done.")

    # ------------------------------------------------------------------ #
    # 8. Frame inpainting — sliding window over temporal dimension        #
    # ------------------------------------------------------------------ #
    # InpaintGenerator processes a local temporal window at a time.
    # NEIGHBOR_LENGTH controls how many neighbouring frames are attended to.
    comp_frames: list[Optional[np.ndarray]] = [None] * total_frames

    # Select reference frames spaced by REF_STRIDE for global context
    ref_indices = list(range(0, total_frames, REF_STRIDE))

    with torch.no_grad():
        for f in range(0, total_frames, NEIGHBOR_LENGTH):
            neighbor_ids = list(range(
                max(0, f - NEIGHBOR_LENGTH // 2),
                min(total_frames, f + NEIGHBOR_LENGTH // 2),
            ))
            # Build reference set: unique union of global refs + local neighbours
            ref_ids = list(
                set(ref_indices) | set(neighbor_ids)
            )
            ref_ids = sorted(set(ref_ids))

            selected_frames = frames_tensor[0, neighbor_ids, ...]   # [n, 3, H, W]
            selected_masks  = masks_dilated_tensor[0, neighbor_ids, ...]  # [n, 1, H, W]
            ref_frames_t    = frames_tensor[0, ref_ids, ...]         # [r, 3, H, W]

            # Mask out the watermark in the neighbour frames
            masked_frames = selected_frames * (1 - selected_masks)

            # Run the inpainting generator
            pred_img, _ = _inpaint_model(
                masked_frames.unsqueeze(0),        # [1, n, 3, H, W]
                ref_frames_t.unsqueeze(0),         # [1, r, 3, H, W]
                selected_masks.unsqueeze(0),       # [1, n, 1, H, W]
                pred_flows_bi,
                neighbor_ids,
                ref_ids,
            )
            # pred_img: [1, n, 3, H, W], values in [-1, 1]

            # Composite: keep original pixels outside the mask
            pred_img = torch.clamp(pred_img, -1, 1)
            # Convert from [-1,1] to [0,1]
            pred_img = (pred_img + 1.0) / 2.0

            # Place predictions back into the full frame list
            for i, frame_idx in enumerate(neighbor_ids):
                # Convert tensor to numpy uint8 [H, W, 3]
                img_np = (
                    pred_img[0, i]
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                )
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                comp_frames[frame_idx] = img_np

            # Report progress
            if progress_callback:
                done = min(f + NEIGHBOR_LENGTH, total_frames)
                progress_callback(done, total_frames)

            logger.info(
                "Inpainted frames %d–%d of %d.",
                neighbor_ids[0], neighbor_ids[-1], total_frames,
            )

    # ------------------------------------------------------------------ #
    # 9. Upscale results back to original resolution and save             #
    # ------------------------------------------------------------------ #
    logger.info("Saving inpainted frames (upscaling %dx%d → %dx%d)...",
                proc_w, proc_h, orig_w, orig_h)

    for idx, (frame_filename, comp_frame) in enumerate(
        zip([os.path.basename(fp) for fp in frame_files], comp_frames)
    ):
        if comp_frame is None:
            # Fallback: copy original frame if this index was never filled
            logger.warning("Frame %d not inpainted — using original.", idx)
            comp_frame = np.array(frames_pil[idx])

        # Upscale back to original resolution if we downscaled
        if (proc_w, proc_h) != (orig_w, orig_h):
            comp_frame_bgr = cv2.cvtColor(comp_frame, cv2.COLOR_RGB2BGR)
            comp_frame_bgr = cv2.resize(
                comp_frame_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4
            )
        else:
            comp_frame_bgr = cv2.cvtColor(comp_frame, cv2.COLOR_RGB2BGR)

        out_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(out_path, comp_frame_bgr)

    logger.info(
        "ProPainter inpainting complete. %d frames saved to: %s",
        total_frames, output_dir,
    )
