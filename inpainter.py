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
          to 360p to make CPU inference practical and avoid OOM kills.

Architecture overview:
    1. RAFT optical flow backbone computes bidirectional flow in short clips.
    2. RecurrentFlowCompleteNet completes the flow in the masked watermark
       region so it can propagate valid pixels across the sequence.
    3. InpaintGenerator (sliding-window transformer) reconstructs each
       masked frame using temporal context from neighbouring frames.

Security Notes:
    - All paths are UUID-based, constructed by main.py — no user input here.
    - TODO(security): Validate image dimensions before tensor conversion to
      prevent memory exhaustion on pathologically large inputs.
=============================================================================
"""

import os
import sys
import gc
import glob
import logging
from typing import Callable, Optional

import cv2
import numpy as np
import scipy.ndimage
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Add ProPainter's source tree to sys.path so its internal imports resolve.
# ProPainter was cloned to /app/propainter during the Docker build.
# ---------------------------------------------------------------------------
PROPAINTER_DIR = os.path.join(os.path.dirname(__file__), "propainter")
if PROPAINTER_DIR not in sys.path:
    sys.path.insert(0, PROPAINTER_DIR)

from model.modules.flow_comp_raft import RAFT_bi                      # noqa: E402
from model.recurrent_flow_completion import RecurrentFlowCompleteNet  # noqa: E402
from model.propainter import InpaintGenerator                         # noqa: E402
from core.utils import to_tensors                                      # noqa: E402

logger = logging.getLogger("watermark-app.inpainter")

# ---------------------------------------------------------------------------
# Global model registry — models are loaded once at startup and reused.
# ---------------------------------------------------------------------------
DEVICE = torch.device("cpu")  # Intel Iris Xe: CPU-only, no CUDA available
_raft_model: Optional[RAFT_bi] = None
_flow_model: Optional[RecurrentFlowCompleteNet] = None
_inpaint_model: Optional[InpaintGenerator] = None

# Path to the pre-downloaded weight files (baked in during Docker build)
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")

# ---------------------------------------------------------------------------
# Processing parameters — tuned for CPU / low-RAM environment
# ---------------------------------------------------------------------------
# Reduce resolution to 360p to keep RAM well under Docker's limit.
MAX_INPAINT_HEIGHT = 360

# Number of frames fed to RAFT per chunk. Keep small (≤20) to avoid OOM.
RAFT_CLIP_LEN = 15

# ProPainter sliding window: frames inpainted together.
NEIGHBOR_LENGTH = 10
REF_STRIDE = 10


# =============================================================================
# Model loading
# =============================================================================

def load_models() -> None:
    """
    Load all three ProPainter model components into memory.

    Must be called once at application startup (FastAPI lifespan event).
    """
    global _raft_model, _flow_model, _inpaint_model

    raft_path = os.path.join(WEIGHTS_DIR, "raft-things.pth")
    flow_path = os.path.join(WEIGHTS_DIR, "recurrent_flow_completion.pth")
    inpaint_path = os.path.join(WEIGHTS_DIR, "ProPainter.pth")

    for path in [raft_path, flow_path, inpaint_path]:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Model weight file missing: {path}. "
                "Rebuild the Docker image to re-download weights."
            )

    logger.info("Loading ProPainter models onto %s ...", DEVICE)

    _raft_model = RAFT_bi(raft_path, DEVICE)

    _flow_model = RecurrentFlowCompleteNet()
    _flow_model.load_state_dict(torch.load(flow_path, map_location=DEVICE))
    _flow_model.to(DEVICE)
    _flow_model.eval()

    _inpaint_model = InpaintGenerator(model_path=inpaint_path)
    _inpaint_model.to(DEVICE)
    _inpaint_model.eval()

    logger.info("All ProPainter models loaded successfully on %s.", DEVICE)


def _ensure_models_loaded() -> None:
    if _raft_model is None or _flow_model is None or _inpaint_model is None:
        raise RuntimeError(
            "ProPainter models are not loaded. "
            "Call inpainter.load_models() at application startup."
        )


# =============================================================================
# Helper utilities
# =============================================================================

def _resize_frames(
    frames: list,
    max_height: int = MAX_INPAINT_HEIGHT,
) -> tuple:
    """Downscale frames to max_height, snapped to nearest multiple of 8."""
    orig_w, orig_h = frames[0].size
    if orig_h > max_height:
        scale = max_height / orig_h
        new_w = int(orig_w * scale)
        new_h = max_height
    else:
        new_w, new_h = orig_w, orig_h

    proc_w = new_w - (new_w % 8)
    proc_h = new_h - (new_h % 8)
    process_size = (proc_w, proc_h)
    resized = [f.resize(process_size, Image.LANCZOS) for f in frames]
    logger.info("Resized frames %dx%d → %dx%d for inference.", orig_w, orig_h, proc_w, proc_h)
    return resized, process_size, (orig_w, orig_h)


def _dilate_mask(mask_img: Image.Image, process_size: tuple) -> tuple:
    """
    Return (flow_masks, masks_dilated) as lists of length 1.
    Caller tiles them to match the total frame count.
    """
    mask_resized = mask_img.resize(process_size, Image.NEAREST)
    mask_np = np.array(mask_resized.convert("L"))

    flow_mask_np = scipy.ndimage.binary_dilation(mask_np, iterations=8).astype(np.uint8)
    inpaint_mask_np = scipy.ndimage.binary_dilation(mask_np, iterations=5).astype(np.uint8)

    return (
        [Image.fromarray(flow_mask_np * 255)],
        [Image.fromarray(inpaint_mask_np * 255)],
    )


def _compute_raft_flows_chunked(frames_t: torch.Tensor, total_frames: int) -> tuple:
    """
    Compute bidirectional RAFT optical flow in short temporal chunks to
    avoid OOM on CPU. Returns (gt_flows_f, gt_flows_b) each [1, T-1, 2, H, W].

    frames_t: [1, T, 3, H, W] in [-1, 1]
    """
    gt_flows_f_list, gt_flows_b_list = [], []

    with torch.no_grad():
        for f in range(0, total_frames, RAFT_CLIP_LEN):
            end_f = min(total_frames, f + RAFT_CLIP_LEN)
            # Give one frame of overlap so flow is continuous at boundaries
            if f == 0:
                clip = frames_t[:, f:end_f]
            else:
                clip = frames_t[:, f - 1:end_f]

            flows_f, flows_b = _raft_model(clip, iters=10)

            if f == 0:
                gt_flows_f_list.append(flows_f)
                gt_flows_b_list.append(flows_b)
            else:
                # Drop the overlap frame's flow so clips concatenate cleanly
                gt_flows_f_list.append(flows_f[:, 1:])
                gt_flows_b_list.append(flows_b[:, 1:])

            logger.info("RAFT: computed flow for frames %d-%d / %d", f, end_f - 1, total_frames)

    gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
    gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
    return gt_flows_f, gt_flows_b


# =============================================================================
# Main pipeline
# =============================================================================

def process_frames(
    frames_dir: str,
    mask_path: str,
    output_dir: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """
    Run the full ProPainter inpainting pipeline on a directory of video frames.

    Steps:
        1. Load all PNG frames from disk as PIL Images.
        2. Downscale to MAX_INPAINT_HEIGHT (360p) for CPU feasibility.
        3. Compute RAFT bidirectional flows in short clips (RAFT_CLIP_LEN).
        4. Complete the flow inside the masked watermark region.
        5. Propagate valid image pixels across the completed flow field.
        6. Run the sliding-window InpaintGenerator transformer.
        7. Upscale results back to original resolution and save to output_dir.
    """
    _ensure_models_loaded()

    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    if not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load frames
    # ------------------------------------------------------------------
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    total_frames = len(frame_files)
    if total_frames == 0:
        raise RuntimeError(f"No frames found in: {frames_dir}")

    logger.info("Loading %d frames for ProPainter inference...", total_frames)
    frames_pil = []
    for fp in frame_files:
        img_bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read frame: {fp}")
        frames_pil.append(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))

    # ------------------------------------------------------------------
    # 2. Downscale
    # ------------------------------------------------------------------
    frames_resized, process_size, original_size = _resize_frames(frames_pil)
    proc_w, proc_h = process_size
    orig_w, orig_h = original_size

    # ------------------------------------------------------------------
    # 3. Prepare masks (single mask tiled for every frame)
    # ------------------------------------------------------------------
    mask_pil = Image.open(mask_path)
    flow_masks_1, masks_dilated_1 = _dilate_mask(mask_pil, process_size)
    flow_masks     = flow_masks_1 * total_frames
    masks_dilated  = masks_dilated_1 * total_frames

    # ------------------------------------------------------------------
    # 4. Convert to tensors
    #    frames_t       : [1, T, 3, H, W] in [-1, 1]
    #    flow_masks_t   : [1, T, 1, H, W] in {0, 1}
    #    masks_dilated_t: [1, T, 1, H, W] in {0, 1}
    # ------------------------------------------------------------------
    frames_t        = to_tensors()(frames_resized).unsqueeze(0).to(DEVICE) * 2.0 - 1.0
    flow_masks_t    = to_tensors()(flow_masks).unsqueeze(0).to(DEVICE)
    masks_dilated_t = to_tensors()(masks_dilated).unsqueeze(0).to(DEVICE)

    logger.info("Running ProPainter on %d frames at %dx%d (CPU)...", total_frames, proc_w, proc_h)

    # ------------------------------------------------------------------
    # 5. RAFT optical flow — chunked to avoid OOM
    # ------------------------------------------------------------------
    gt_flows_f, gt_flows_b = _compute_raft_flows_chunked(frames_t, total_frames)
    gt_flows_bi = (gt_flows_f, gt_flows_b)
    logger.info("RAFT optical flow complete.")

    # ------------------------------------------------------------------
    # 6. Flow completion in masked regions
    # ------------------------------------------------------------------
    with torch.no_grad():
        pred_flows_bi, _ = _flow_model.forward_bidirect_flow(gt_flows_bi, flow_masks_t)
        pred_flows_bi = _flow_model.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks_t)
    logger.info("Flow completion done.")

    # ------------------------------------------------------------------
    # 7. Image propagation — fill mask region using warped valid pixels
    # ------------------------------------------------------------------
    with torch.no_grad():
        masked_frames = frames_t * (1.0 - masks_dilated_t)
        _, b_t, _, _, _ = masks_dilated_t.size()  # (1, T, 1, H, W) → b_t=T

        PROP_LEN = min(80, total_frames)
        if total_frames > PROP_LEN:
            updated_frames_list, updated_masks_list = [], []
            pad = 5
            for f in range(0, total_frames, PROP_LEN):
                s = max(0, f - pad)
                e = min(total_frames, f + PROP_LEN + pad)
                pad_s = f - s
                pad_e = e - min(total_frames, f + PROP_LEN)

                prop_imgs, updated_local_masks = _inpaint_model.img_propagation(
                    masked_frames[:, s:e],
                    (pred_flows_bi[0][:, s:e - 1], pred_flows_bi[1][:, s:e - 1]),
                    masks_dilated_t[:, s:e],
                    "nearest",
                )
                b, t, c, h, w = masks_dilated_t[:, s:e].size()
                upd_f = frames_t[:, s:e] * (1 - masks_dilated_t[:, s:e]) + \
                        prop_imgs.view(b, t, 3, h, w) * masks_dilated_t[:, s:e]
                upd_m = updated_local_masks.view(b, t, 1, h, w)

                trim_s = pad_s
                trim_e = t - pad_e if pad_e > 0 else t
                updated_frames_list.append(upd_f[:, trim_s:trim_e])
                updated_masks_list.append(upd_m[:, trim_s:trim_e])

            updated_frames = torch.cat(updated_frames_list, dim=1)
            updated_masks  = torch.cat(updated_masks_list, dim=1)
        else:
            prop_imgs, updated_local_masks = _inpaint_model.img_propagation(
                masked_frames, pred_flows_bi, masks_dilated_t, "nearest"
            )
            b, t, c, h, w = masks_dilated_t.size()
            updated_frames = frames_t * (1 - masks_dilated_t) + \
                             prop_imgs.view(b, t, 3, h, w) * masks_dilated_t
            updated_masks  = updated_local_masks.view(b, t, 1, h, w)

    logger.info("Image propagation done.")

    # ------------------------------------------------------------------
    # 8. Sliding-window transformer inpainting
    # ------------------------------------------------------------------
    comp_frames: list = [None] * total_frames
    ref_indices = list(range(0, total_frames, REF_STRIDE))
    neighbor_stride = NEIGHBOR_LENGTH // 2

    with torch.no_grad():
        for f in range(0, total_frames, neighbor_stride):
            neighbor_ids = list(range(
                max(0, f - neighbor_stride),
                min(total_frames, f + neighbor_stride),
            ))
            ref_ids = sorted(set(ref_indices) | set(neighbor_ids))

            # Gather tensors for this window
            sel_frames  = updated_frames[0, neighbor_ids]   # [n, 3, H, W]
            sel_masks   = updated_masks[0, neighbor_ids]    # [n, 1, H, W]
            ref_frames  = updated_frames[0, ref_ids]        # [r, 3, H, W]

            pred_img, _ = _inpaint_model(
                sel_frames.unsqueeze(0),   # [1, n, 3, H, W]
                ref_frames.unsqueeze(0),   # [1, r, 3, H, W]
                sel_masks.unsqueeze(0),    # [1, n, 1, H, W]
                pred_flows_bi,
                neighbor_ids,
                ref_ids,
            )
            # pred_img: [1, n, 3, H, W] in [-1, 1]
            pred_img = torch.clamp(pred_img, -1, 1)
            pred_img = (pred_img + 1.0) / 2.0  # → [0, 1]

            for i, fidx in enumerate(neighbor_ids):
                img_np = pred_img[0, i].permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                comp_frames[fidx] = img_np

            if progress_callback:
                progress_callback(min(f + neighbor_stride, total_frames), total_frames)

            logger.info("Inpainted frames %d–%d / %d.", neighbor_ids[0], neighbor_ids[-1], total_frames)

    # Free the large tensors before writing to disk
    del frames_t, flow_masks_t, masks_dilated_t, pred_flows_bi
    del updated_frames, updated_masks, gt_flows_bi
    gc.collect()

    # ------------------------------------------------------------------
    # 9. Upscale and save
    # ------------------------------------------------------------------
    logger.info("Saving frames (upscaling %dx%d → %dx%d)...", proc_w, proc_h, orig_w, orig_h)
    filenames = [os.path.basename(fp) for fp in frame_files]

    for idx, (fname, comp_frame) in enumerate(zip(filenames, comp_frames)):
        if comp_frame is None:
            logger.warning("Frame %d not inpainted — using original.", idx)
            comp_frame = np.array(frames_pil[idx])

        comp_bgr = cv2.cvtColor(comp_frame, cv2.COLOR_RGB2BGR)
        if (proc_w, proc_h) != (orig_w, orig_h):
            comp_bgr = cv2.resize(comp_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

        cv2.imwrite(os.path.join(output_dir, fname), comp_bgr)

    logger.info("ProPainter inpainting complete. %d frames saved to: %s", total_frames, output_dir)
