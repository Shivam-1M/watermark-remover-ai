/**
 * ==========================================================================
 * app.js — WatermarkAI Frontend Application
 * ==========================================================================
 * Handles:
 *   1. Drag-and-drop / click-to-upload video files
 *   2. Canvas-based mask painting (brush tool with undo)
 *   3. Mask submission and background processing initiation
 *   4. Progress polling via /status endpoint
 *   5. Download link generation on completion
 *
 * Security:
 *   - All DOM manipulation uses safe APIs (createElement, textContent, etc.)
 *   - No innerHTML, outerHTML, or document.write usage
 *   - File type validation is performed client-side as a UX convenience;
 *     the server performs authoritative validation.
 *   - TODO(security): Add CSRF token handling if auth is implemented.
 * ==========================================================================
 */

"use strict";

/* -------------------------------------------------------------------------
 * DOM Element References
 * ------------------------------------------------------------------------- */
const elements = {
    // Sections
    sectionUpload:   document.getElementById("section-upload"),
    sectionMask:     document.getElementById("section-mask"),
    sectionStatus:   document.getElementById("section-status"),
    sectionDownload: document.getElementById("section-download"),

    // Upload
    uploadZone:        document.getElementById("upload-zone"),
    uploadZoneContent: document.getElementById("upload-zone-content"),
    uploadProgress:    document.getElementById("upload-progress"),
    uploadStatusText:  document.getElementById("upload-status-text"),
    fileInput:         document.getElementById("file-input"),

    // Canvas / Video
    canvasContainer: document.getElementById("canvas-container"),
    videoFrame:      document.getElementById("video-frame"),
    canvasMask:      document.getElementById("canvas-mask"),

    // Toolbar
    brushSize:      document.getElementById("brush-size"),
    brushSizeValue: document.getElementById("brush-size-value"),
    btnUndo:        document.getElementById("btn-undo"),
    btnClear:       document.getElementById("btn-clear"),

    // Actions
    btnProcess:  document.getElementById("btn-process"),
    btnDownload: document.getElementById("btn-download"),
    btnNew:      document.getElementById("btn-new"),

    // Progress
    progressBarFill: document.getElementById("progress-bar-fill"),
    progressText:    document.getElementById("progress-text"),
    progressDetail:  document.getElementById("progress-detail"),

    // Toast
    toastError:   document.getElementById("toast-error"),
    toastMessage: document.getElementById("toast-message"),
    toastClose:   document.getElementById("toast-close"),
};


/* -------------------------------------------------------------------------
 * Application State
 * ------------------------------------------------------------------------- */
const state = {
    taskId: null,               // Current task UUID from the server
    isDrawing: false,           // Whether the user is currently painting
    brushSize: 20,              // Current brush radius in pixels
    maskHistory: [],            // Array of ImageData snapshots for undo
    maxHistorySize: 30,         // Maximum number of undo steps
    pollInterval: null,         // setInterval ID for status polling
};

// Accepted video MIME types (client-side pre-validation)
const ACCEPTED_TYPES = new Set([
    "video/mp4",
    "video/avi",
    "video/x-msvideo",
    "video/quicktime",
    "video/x-matroska",
    "video/webm",
]);

// Accepted file extensions (fallback if MIME type is missing)
const ACCEPTED_EXTENSIONS = new Set([".mp4", ".avi", ".mov", ".mkv", ".webm"]);

// Maximum file size in bytes (100 MB)
const MAX_FILE_SIZE = 100 * 1024 * 1024;


/* -------------------------------------------------------------------------
 * Initialization
 * ------------------------------------------------------------------------- */
document.addEventListener("DOMContentLoaded", init);

function init() {
    setupUploadZone();
    setupToolbar();
    setupCanvasEvents();
    setupActionButtons();
    setupToast();
}


/* =========================================================================
 * 1. UPLOAD HANDLING
 * ========================================================================= */

function setupUploadZone() {
    const zone = elements.uploadZone;
    const input = elements.fileInput;

    // Click to open file dialog
    zone.addEventListener("click", () => input.click());

    // Keyboard accessibility
    zone.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            input.click();
        }
    });

    // File selected via dialog
    input.addEventListener("change", () => {
        if (input.files && input.files.length > 0) {
            handleFileUpload(input.files[0]);
        }
    });

    // --- Drag-and-drop events ---
    zone.addEventListener("dragenter", (e) => {
        e.preventDefault();
        zone.classList.add("upload-zone--dragover");
    });

    zone.addEventListener("dragover", (e) => {
        e.preventDefault();
        zone.classList.add("upload-zone--dragover");
    });

    zone.addEventListener("dragleave", (e) => {
        e.preventDefault();
        zone.classList.remove("upload-zone--dragover");
    });

    zone.addEventListener("drop", (e) => {
        e.preventDefault();
        zone.classList.remove("upload-zone--dragover");

        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });
}


/**
 * Validate and upload a video file to the backend.
 *
 * @param {File} file - The video file to upload.
 */
async function handleFileUpload(file) {
    // --- Client-side validation (UX only; server validates authoritatively) ---
    const ext = "." + file.name.split(".").pop().toLowerCase();

    if (!ACCEPTED_TYPES.has(file.type) && !ACCEPTED_EXTENSIONS.has(ext)) {
        showError("Unsupported file type. Please upload MP4, AVI, MOV, MKV, or WebM.");
        return;
    }

    if (file.size > MAX_FILE_SIZE) {
        showError("File is too large. Maximum size is 100 MB.");
        return;
    }

    // --- Show upload progress UI ---
    elements.uploadZoneContent.hidden = true;
    elements.uploadProgress.hidden = false;
    elements.uploadStatusText.textContent = "Uploading…";

    try {
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch("/upload", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || "Upload failed.");
        }

        const data = await response.json();
        state.taskId = data.task_id;

        elements.uploadStatusText.textContent = "Loading preview…";

        // Load the video into the player
        await loadVideoPlayer(state.taskId);

        // Transition to the mask editor step
        showSection("mask");

    } catch (err) {
        showError(err.message || "Failed to upload video.");
        // Reset upload zone
        elements.uploadZoneContent.hidden = false;
        elements.uploadProgress.hidden = true;
    }
}


/**
 * Fetch the uploaded video and load it into the video player.
 *
 * @param {string} taskId - The task UUID.
 */
function loadVideoPlayer(taskId) {
    return new Promise((resolve, reject) => {
        const video = elements.videoFrame;

        video.onloadedmetadata = () => {
            const maskCanvas = elements.canvasMask;

            // Set the mask canvas to match the true video dimensions
            maskCanvas.width = video.videoWidth;
            maskCanvas.height = video.videoHeight;

            // Initialize the mask canvas as fully transparent
            const maskCtx = maskCanvas.getContext("2d");
            maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);

            // Save initial empty state for undo
            saveMaskState();

            resolve();
        };

        video.onerror = () => {
            reject(new Error("Failed to load video preview."));
        };

        // Stream the video from the backend
        video.src = "/video/" + encodeURIComponent(taskId);
    });
}


/* =========================================================================
 * 2. CANVAS MASK PAINTING
 * ========================================================================= */

function setupCanvasEvents() {
    const canvas = elements.canvasMask;

    // --- Mouse events ---
    canvas.addEventListener("mousedown", startDrawing);
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mouseup", stopDrawing);
    canvas.addEventListener("mouseleave", stopDrawing);

    // --- Touch events (for mobile/tablet) ---
    canvas.addEventListener("touchstart", (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        startDrawing(touchToMouseEvent(touch, canvas));
    });

    canvas.addEventListener("touchmove", (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        draw(touchToMouseEvent(touch, canvas));
    });

    canvas.addEventListener("touchend", (e) => {
        e.preventDefault();
        stopDrawing();
    });
}


/**
 * Convert a Touch event to a mouse-like event with offsetX/offsetY.
 *
 * @param {Touch} touch  - The touch point.
 * @param {HTMLCanvasElement} canvas - The target canvas.
 * @returns {Object} An object with offsetX and offsetY properties.
 */
function touchToMouseEvent(touch, canvas) {
    const rect = canvas.getBoundingClientRect();
    return {
        offsetX: (touch.clientX - rect.left) * (canvas.width / rect.width),
        offsetY: (touch.clientY - rect.top) * (canvas.height / rect.height),
    };
}


/**
 * Begin a new brush stroke.
 *
 * @param {MouseEvent|Object} e - The mouse or simulated event.
 */
function startDrawing(e) {
    state.isDrawing = true;
    // Save state before this stroke for undo
    saveMaskState();
    draw(e);
}


/**
 * Continue the current brush stroke (paint white circles along the path).
 *
 * The mask uses a semi-transparent red overlay on the canvas so the user
 * can see where they've painted. The actual mask exported for processing
 * is a black-and-white image.
 *
 * @param {MouseEvent|Object} e - The mouse or simulated event.
 */
function draw(e) {
    if (!state.isDrawing) return;

    const canvas = elements.canvasMask;
    const ctx = canvas.getContext("2d");

    // Scale coordinates if the canvas display size differs from its
    // internal resolution (CSS scaling).
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    // Use offsetX/offsetY for mouse events, or the pre-calculated values
    // for touch events (which already account for scaling).
    let x, y;
    if (e instanceof MouseEvent) {
        x = e.offsetX * scaleX;
        y = e.offsetY * scaleY;
    } else {
        x = e.offsetX;
        y = e.offsetY;
    }

    // Draw a semi-transparent red circle (visual feedback for the user)
    ctx.globalCompositeOperation = "source-over";
    ctx.fillStyle = "rgba(255, 60, 80, 0.45)";
    ctx.beginPath();
    ctx.arc(x, y, state.brushSize / 2, 0, Math.PI * 2);
    ctx.fill();
}


/**
 * End the current brush stroke.
 */
function stopDrawing() {
    state.isDrawing = false;
}


/**
 * Save the current mask canvas state for undo functionality.
 */
function saveMaskState() {
    const canvas = elements.canvasMask;
    const ctx = canvas.getContext("2d");

    if (canvas.width === 0 || canvas.height === 0) return;

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    state.maskHistory.push(imageData);

    // Limit history size to prevent excessive memory usage
    if (state.maskHistory.length > state.maxHistorySize) {
        state.maskHistory.shift();
    }
}


/**
 * Undo the last brush stroke by restoring the previous canvas state.
 */
function undoMask() {
    if (state.maskHistory.length <= 1) return;  // Keep the initial empty state

    // Remove the current state
    state.maskHistory.pop();

    // Restore the previous state
    const previousState = state.maskHistory[state.maskHistory.length - 1];
    const ctx = elements.canvasMask.getContext("2d");
    ctx.putImageData(previousState, 0, 0);
}


/**
 * Clear all mask strokes, resetting the mask canvas to transparent.
 */
function clearMask() {
    const canvas = elements.canvasMask;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    state.maskHistory = [];
    saveMaskState();
}


/**
 * Export the painted mask as a black-and-white PNG blob.
 *
 * The visual red overlay is converted to a binary mask:
 *   - Any pixel with alpha > 0 becomes white (255, 255, 255)
 *   - All other pixels are black (0, 0, 0)
 *
 * @returns {Promise<Blob>} A PNG blob of the binary mask.
 */
function exportMaskAsBlob() {
    return new Promise((resolve, reject) => {
        const sourceCanvas = elements.canvasMask;
        const ctx = sourceCanvas.getContext("2d");
        const imageData = ctx.getImageData(
            0, 0, sourceCanvas.width, sourceCanvas.height
        );

        // Create a new offscreen canvas for the binary mask
        const maskCanvas = document.createElement("canvas");
        maskCanvas.width = sourceCanvas.width;
        maskCanvas.height = sourceCanvas.height;
        const maskCtx = maskCanvas.getContext("2d");

        // Fill with black background
        maskCtx.fillStyle = "#000000";
        maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);

        // Convert: any painted pixel (alpha > 0) → white
        const maskImageData = maskCtx.getImageData(
            0, 0, maskCanvas.width, maskCanvas.height
        );
        const src = imageData.data;
        const dst = maskImageData.data;

        for (let i = 0; i < src.length; i += 4) {
            if (src[i + 3] > 0) {
                // Pixel has paint → white in mask
                dst[i]     = 255;  // R
                dst[i + 1] = 255;  // G
                dst[i + 2] = 255;  // B
                dst[i + 3] = 255;  // A
            }
            // Otherwise remains black (already filled)
        }

        maskCtx.putImageData(maskImageData, 0, 0);

        maskCanvas.toBlob((blob) => {
            if (blob) {
                resolve(blob);
            } else {
                reject(new Error("Failed to export mask as PNG."));
            }
        }, "image/png");
    });
}


/* =========================================================================
 * 3. TOOLBAR CONTROLS
 * ========================================================================= */

function setupToolbar() {
    // Brush size slider
    elements.brushSize.addEventListener("input", () => {
        state.brushSize = parseInt(elements.brushSize.value, 10);
        elements.brushSizeValue.textContent = state.brushSize + "px";
    });

    // Undo button
    elements.btnUndo.addEventListener("click", undoMask);

    // Clear button
    elements.btnClear.addEventListener("click", clearMask);
}


/* =========================================================================
 * 4. PROCESS AND POLLING
 * ========================================================================= */

function setupActionButtons() {
    // Process button: submit the mask and start inpainting
    elements.btnProcess.addEventListener("click", startProcessing);

    // New video button: reset the app to the upload state
    elements.btnNew.addEventListener("click", resetApp);
}


/**
 * Submit the painted mask to the backend and initiate processing.
 */
async function startProcessing() {
    if (!state.taskId) {
        showError("No video uploaded. Please upload a video first.");
        return;
    }

    // Disable the button to prevent double submission
    elements.btnProcess.disabled = true;

    try {
        // Export the mask canvas as a binary PNG blob
        const maskBlob = await exportMaskAsBlob();

        // Check that the user actually painted something
        const hasContent = await maskHasContent();
        if (!hasContent) {
            showError("Please paint over the watermark area before processing.");
            elements.btnProcess.disabled = false;
            return;
        }

        // Submit the mask to the backend
        const formData = new FormData();
        formData.append("mask", maskBlob, "mask.png");

        const response = await fetch(
            "/process?task_id=" + encodeURIComponent(state.taskId),
            {
                method: "POST",
                body: formData,
            }
        );

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || "Failed to start processing.");
        }

        // Transition to the status/progress view
        showSection("status");

        // Begin polling for progress updates
        startPolling();

    } catch (err) {
        showError(err.message || "Failed to start processing.");
        elements.btnProcess.disabled = false;
    }
}


/**
 * Check if the user has actually painted any mask content.
 *
 * @returns {Promise<boolean>} True if the mask has painted pixels.
 */
function maskHasContent() {
    return new Promise((resolve) => {
        const canvas = elements.canvasMask;
        const ctx = canvas.getContext("2d");
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;

        for (let i = 3; i < data.length; i += 4) {
            if (data[i] > 0) {
                resolve(true);
                return;
            }
        }
        resolve(false);
    });
}


/**
 * Start polling the /status endpoint every 1.5 seconds.
 */
function startPolling() {
    // Clear any existing poll interval
    if (state.pollInterval) {
        clearInterval(state.pollInterval);
    }

    // Initial poll
    pollStatus();

    // Poll every 1.5 seconds
    state.pollInterval = setInterval(pollStatus, 1500);
}


/**
 * Fetch the current processing status from the backend.
 */
async function pollStatus() {
    if (!state.taskId) return;

    try {
        const response = await fetch(
            "/status/" + encodeURIComponent(state.taskId)
        );

        if (!response.ok) {
            throw new Error("Failed to fetch status.");
        }

        const data = await response.json();

        // Update progress bar
        const progress = Math.min(Math.max(data.progress || 0, 0), 100);
        elements.progressBarFill.style.width = progress + "%";
        elements.progressText.textContent = progress + "%";

        // Update status detail text
        if (data.status === "processing") {
            if (progress < 10) {
                elements.progressDetail.textContent = "Extracting frames and audio…";
            } else if (progress < 85) {
                elements.progressDetail.textContent = "Running AI inpainting on video frames…";
            } else {
                elements.progressDetail.textContent = "Reassembling video…";
            }
        }

        // Handle completion
        if (data.status === "complete") {
            clearInterval(state.pollInterval);
            state.pollInterval = null;

            // Set download link
            elements.btnDownload.href = "/download/" + encodeURIComponent(state.taskId);

            // Transition to download section
            showSection("download");
        }

        // Handle errors
        if (data.status === "error") {
            clearInterval(state.pollInterval);
            state.pollInterval = null;
            showError(data.error || "Processing failed. Please try again.");
            showSection("mask");
            elements.btnProcess.disabled = false;
        }

    } catch (err) {
        // Don't stop polling on transient network errors
        // (the server might be temporarily busy)
    }
}


/* =========================================================================
 * 5. UI HELPERS
 * ========================================================================= */

/**
 * Show a specific step section and hide all others.
 *
 * @param {"upload"|"mask"|"status"|"download"} sectionName
 */
function showSection(sectionName) {
    const sectionMap = {
        upload:   elements.sectionUpload,
        mask:     elements.sectionMask,
        status:   elements.sectionStatus,
        download: elements.sectionDownload,
    };

    for (const [name, el] of Object.entries(sectionMap)) {
        if (name === sectionName) {
            el.classList.remove("step-section--hidden");
            // Re-trigger the fade-in animation
            el.style.animation = "none";
            // Force reflow so the browser recognizes the animation reset
            void el.offsetHeight;
            el.style.animation = "";
        } else {
            el.classList.add("step-section--hidden");
        }
    }
}


/**
 * Show an error toast notification.
 *
 * @param {string} message - The error message to display.
 */
function showError(message) {
    elements.toastMessage.textContent = message;
    elements.toastError.hidden = false;

    // Auto-dismiss after 6 seconds
    setTimeout(() => {
        elements.toastError.hidden = true;
    }, 6000);
}


/**
 * Setup toast close button.
 */
function setupToast() {
    elements.toastClose.addEventListener("click", () => {
        elements.toastError.hidden = true;
    });
}


/**
 * Reset the entire application to the initial upload state.
 */
function resetApp() {
    // Clear polling
    if (state.pollInterval) {
        clearInterval(state.pollInterval);
        state.pollInterval = null;
    }

    // Reset state
    state.taskId = null;
    state.maskHistory = [];
    state.pollInterval = null;

    // Reset upload zone
    elements.uploadZoneContent.hidden = false;
    elements.uploadProgress.hidden = true;
    elements.fileInput.value = "";

    // Reset progress bar
    elements.progressBarFill.style.width = "0%";
    elements.progressText.textContent = "Initializing…";

    // Re-enable process button
    elements.btnProcess.disabled = false;

    // Show upload section
    showSection("upload");
}
