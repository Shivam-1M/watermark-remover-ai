"""
=============================================================================
main.py — FastAPI Application Entry Point
=============================================================================
Video Watermark Removal Application — OpenCV Inpainting Engine

Endpoints:
    POST   /upload           Upload a video file for processing
    GET    /frame/{task_id}  Retrieve the first frame of the uploaded video (deprecated)
    GET    /video/{task_id}  Stream the uploaded video for playback
    POST   /process          Submit a mask and start background inpainting
    GET    /status/{task_id} Poll the progress of the inpainting task
    GET    /download/{task_id} Download the final processed video

Security Notes:
    - File uploads are validated by extension allow-list and size limit.
    - Uploaded files are renamed to UUIDs to prevent path traversal.
    - The uploads directory is outside the web root static directory.
    - TODO(security): Add authentication/authorization for production use.
    - TODO(security): Add rate limiting to all API endpoints.
    - TODO(security): Implement CSRF protection if cookie-based auth is added.
    - TODO(security): Integrate malware scanning for uploaded files.
    - TODO(security): Add HTTPS/TLS termination via reverse proxy in production.
============================================================================="""

import os
import uuid
import logging
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    BackgroundTasks,
    HTTPException,
)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import video_utils
import inpainter

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("watermark-app")

# ---------------------------------------------------------------------------
# Application configuration
# ---------------------------------------------------------------------------
# Maximum upload size in bytes (default: 100 MB)
MAX_UPLOAD_SIZE = int(os.environ.get("MAX_UPLOAD_SIZE_MB", 100)) * 1024 * 1024

# Base directory for all uploaded/processed files
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Allow-list of accepted video file extensions
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# ---------------------------------------------------------------------------
# FastAPI app initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Video Watermark Remover",
    description="Self-contained video watermark removal using OpenCV inpainting.",
    version="3.0.0",
)

# ---------------------------------------------------------------------------
# Security headers middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_security_headers(request, call_next):
    """Inject security headers into every HTTP response."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # TODO(security): Add a strict CSP policy for production.
    # Content-Security-Policy is set via the HTML meta tag for now,
    # but should be moved to a server-side header in production.
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=()"
    )
    return response


# ---------------------------------------------------------------------------
# CORS configuration
# ---------------------------------------------------------------------------
# TODO(security): Restrict origins to specific trusted domains in production.
# Using same-origin only for now since frontend is served from the same host.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static file serving (frontend)
# ---------------------------------------------------------------------------
# Mount the static directory to serve the frontend HTML/CSS/JS files.
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------------------------
# In-memory task status store
# ---------------------------------------------------------------------------
# In production, replace this with Redis or a database.
# Structure: { task_id: { "status": str, "progress": int, "error": str|None } }
task_store: dict[str, dict] = {}


def _validate_video_extension(filename: str) -> str:
    """
    Validate that the uploaded file has an allowed video extension.

    Args:
        filename: The original filename from the upload.

    Returns:
        The lowercase file extension (e.g., '.mp4').

    Raises:
        HTTPException: If the extension is not in the allow-list.
    """
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )
    return ext


# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/")
async def root():
    """Serve the main frontend page."""
    return FileResponse("static/index.html")


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file for watermark removal.

    The file is validated by extension and size, then saved with a UUID
    filename to prevent path traversal attacks.

    Returns:
        JSON with the assigned task_id.
    """
    # --- Validate file extension ---
    ext = _validate_video_extension(file.filename or "unknown.bin")

    # --- Generate a unique task ID and create the task directory ---
    task_id = str(uuid.uuid4())
    task_dir = UPLOAD_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    # --- Read and validate file size ---
    video_path = task_dir / f"original{ext}"
    total_bytes = 0

    try:
        with open(video_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1 MB chunks
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_SIZE:
                    # Clean up the partial file
                    f.close()
                    video_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File too large. Maximum size is "
                            f"{MAX_UPLOAD_SIZE // (1024 * 1024)} MB."
                        ),
                    )
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("File upload failed for task %s: %s", task_id, str(e))
        raise HTTPException(status_code=500, detail="File upload failed.")

    logger.info(
        "Uploaded video for task %s (%d bytes, ext=%s)", task_id, total_bytes, ext
    )

    # --- Initialize task status ---
    task_store[task_id] = {
        "status": "uploaded",
        "progress": 0,
        "error": None,
        "video_path": str(video_path),
    }

    return JSONResponse(
        content={"task_id": task_id, "message": "Video uploaded successfully."}
    )


@app.get("/frame/{task_id}")
async def get_frame(task_id: str):
    """
    Return the first frame of the uploaded video as a PNG image.

    This frame is displayed on the frontend canvas so the user can
    paint the watermark mask over it.
    """
    # Sanitize task_id: must be a valid UUID to prevent path traversal
    try:
        uuid.UUID(task_id, version=4)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task ID format.")

    frame_path = UPLOAD_DIR / task_id / "first_frame.png"
    if not frame_path.is_file():
        raise HTTPException(status_code=404, detail="Frame not found.")

    # Resolve and verify the path is within the uploads directory
    resolved = frame_path.resolve()
    if not str(resolved).startswith(str(UPLOAD_DIR.resolve()) + os.sep):
        raise HTTPException(status_code=403, detail="Access denied.")

    return FileResponse(
        str(resolved),
        media_type="image/png",
        headers={
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store",
        },
    )


@app.get("/video/{task_id}")
async def get_video(task_id: str):
    """
    Stream the uploaded video file for playback in the mask editor.
    """
    # Sanitize task_id: must be a valid UUID to prevent path traversal
    try:
        uuid.UUID(task_id, version=4)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task ID format.")

    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found.")
        
    video_path = Path(task_store[task_id]["video_path"])
    
    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="Video not found.")

    # Resolve and verify the path is within the uploads directory
    resolved = video_path.resolve()
    if not str(resolved).startswith(str(UPLOAD_DIR.resolve()) + os.sep):
        raise HTTPException(status_code=403, detail="Access denied.")

    return FileResponse(
        str(resolved),
        media_type="video/mp4",
        headers={
            "X-Content-Type-Options": "nosniff",
            "Accept-Ranges": "bytes",
        },
    )


@app.post("/process")
async def process_video(
    background_tasks: BackgroundTasks,
    task_id: str = "",
    mask: UploadFile = File(...),
):
    """
    Start the watermark removal process.

    Accepts the task_id and a binary mask image (PNG). The mask should
    be a black-and-white image where white pixels indicate the watermark
    region to be inpainted.

    The processing runs as a background task so the endpoint returns
    immediately.
    """
    # --- Validate task_id ---
    try:
        uuid.UUID(task_id, version=4)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task ID format.")

    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found.")

    if task_store[task_id]["status"] == "processing":
        raise HTTPException(status_code=409, detail="Task is already processing.")

    # --- Save the mask image ---
    task_dir = UPLOAD_DIR / task_id

    # Validate path is within uploads directory
    resolved_dir = task_dir.resolve()
    if not str(resolved_dir).startswith(str(UPLOAD_DIR.resolve()) + os.sep):
        raise HTTPException(status_code=403, detail="Access denied.")

    mask_path = task_dir / "mask.png"

    # Validate mask file extension
    mask_ext = Path(mask.filename or "mask.png").suffix.lower()
    if mask_ext != ".png":
        raise HTTPException(
            status_code=400, detail="Mask must be a PNG image."
        )

    try:
        mask_data = await mask.read()
        # Validate mask size (max 10 MB)
        if len(mask_data) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Mask file too large.")

        # Validate PNG magic bytes
        if not mask_data[:8] == b"\x89PNG\r\n\x1a\n":
            raise HTTPException(
                status_code=400,
                detail="Invalid mask file. Must be a valid PNG image.",
            )

        with open(mask_path, "wb") as f:
            f.write(mask_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Mask save failed for task %s: %s", task_id, str(e))
        raise HTTPException(status_code=500, detail="Failed to save mask.")

    # --- Update task status and launch background processing ---
    task_store[task_id]["status"] = "processing"
    task_store[task_id]["progress"] = 0

    background_tasks.add_task(
        _run_processing_pipeline,
        task_id,
        task_store[task_id]["video_path"],
        str(mask_path),
        str(task_dir),
    )

    logger.info("Started processing pipeline for task %s", task_id)

    return JSONResponse(
        content={
            "task_id": task_id,
            "message": "Processing started.",
        }
    )


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    """
    Poll the current progress of a processing task.

    Returns:
        JSON with status ('uploaded', 'processing', 'complete', 'error')
        and progress (0-100).
    """
    # --- Validate task_id ---
    try:
        uuid.UUID(task_id, version=4)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task ID format.")

    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found.")

    task = task_store[task_id]
    return JSONResponse(
        content={
            "task_id": task_id,
            "status": task["status"],
            "progress": task["progress"],
            "error": task.get("error"),
        }
    )


@app.get("/download/{task_id}")
async def download_video(task_id: str):
    """
    Download the final processed video with the watermark removed.

    The video is served with Content-Disposition: attachment to force
    a download rather than inline playback.
    """
    # --- Validate task_id ---
    try:
        uuid.UUID(task_id, version=4)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task ID format.")

    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found.")

    if task_store[task_id]["status"] != "complete":
        raise HTTPException(
            status_code=400,
            detail="Processing is not yet complete.",
        )

    output_path = UPLOAD_DIR / task_id / "output.mp4"

    # Validate path is within uploads directory
    resolved = output_path.resolve()
    if not str(resolved).startswith(str(UPLOAD_DIR.resolve()) + os.sep):
        raise HTTPException(status_code=403, detail="Access denied.")

    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="Output video not found.")

    return FileResponse(
        str(resolved),
        media_type="video/mp4",
        filename=f"watermark_removed_{task_id[:8]}.mp4",
        headers={
            "Content-Disposition": f'attachment; filename="watermark_removed_{task_id[:8]}.mp4"',
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store",
        },
    )


# =============================================================================
# Background Processing Pipeline
# =============================================================================


def _run_processing_pipeline(
    task_id: str,
    video_path: str,
    mask_path: str,
    task_dir: str,
):
    """
    Execute the full watermark removal pipeline as a background task.

    Pipeline stages:
        1. Extract audio from the original video.
        2. Split the video into individual PNG frames.
        3. Run the inpainting model on each frame with the mask.
        4. Reassemble the inpainted frames into a video and remux audio.

    Progress is reported to the task_store so the frontend can poll it.
    """
    try:
        logger.info("[%s] Pipeline started.", task_id)

        # --- Stage 1: Extract audio ---
        task_store[task_id]["progress"] = 5
        audio_path = os.path.join(task_dir, "audio.aac")
        video_utils.extract_audio(video_path, audio_path)
        logger.info("[%s] Audio extracted.", task_id)

        # --- Stage 2: Split video into frames ---
        task_store[task_id]["progress"] = 10
        frames_dir = os.path.join(task_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        frame_count = video_utils.split_frames(video_path, frames_dir)
        logger.info("[%s] Extracted %d frames.", task_id, frame_count)

        # --- Stage 3: Inpaint each frame (AI processing) ---
        output_frames_dir = os.path.join(task_dir, "output_frames")
        os.makedirs(output_frames_dir, exist_ok=True)

        def progress_callback(current_frame: int, total_frames: int):
            """Update the task progress (10% to 85% range for inpainting)."""
            pct = 10 + int((current_frame / max(total_frames, 1)) * 75)
            task_store[task_id]["progress"] = min(pct, 85)

        inpainter.process_frames(
            frames_dir=frames_dir,
            mask_path=mask_path,
            output_dir=output_frames_dir,
            progress_callback=progress_callback,
        )
        logger.info("[%s] Inpainting complete.", task_id)

        # --- Stage 4: Reassemble video with audio ---
        task_store[task_id]["progress"] = 90
        output_path = os.path.join(task_dir, "output.mp4")

        # Get the original video's FPS for accurate reassembly
        fps = video_utils.get_video_fps(video_path)

        video_utils.reassemble_video(
            frames_dir=output_frames_dir,
            audio_path=audio_path,
            output_path=output_path,
            fps=fps,
        )
        logger.info("[%s] Video reassembled.", task_id)

        # --- Done ---
        task_store[task_id]["status"] = "complete"
        task_store[task_id]["progress"] = 100
        logger.info("[%s] Pipeline complete.", task_id)

    except Exception as e:
        logger.error("[%s] Pipeline failed: %s", task_id, str(e))
        task_store[task_id]["status"] = "error"
        # Return a generic error message to the client
        task_store[task_id]["error"] = "Processing failed. Please try again."
