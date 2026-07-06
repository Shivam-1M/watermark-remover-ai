# 🎬 WatermarkAI — Video Watermark Remover

> **Video watermark removal — 100% local, self-contained, no paid APIs.**

A full-stack application that removes watermarks from videos using OpenCV inpainting. Upload a video, paint over the watermark on an interactive canvas, and download the clean result. Everything runs locally on your machine — no cloud services, no subscriptions.

---

## ✨ Features

- **Drag & Drop Upload** — Upload videos up to 100 MB (MP4, AVI, MOV, MKV, WebM)
- **Interactive Canvas Mask Editor** — Paint over watermarks with an adjustable brush, undo/clear support
- **OpenCV Telea Inpainting** — Fast, CPU-optimized frame-by-frame watermark removal
- **Multi-core Processing** — Distributes frame inpainting across all CPU cores via multiprocessing
- **Audio Preservation** — Automatically extracts and remuxes the original audio track
- **Real-time Progress** — Live progress bar that polls the backend during processing
- **Premium Dark UI** — Glassmorphism design with smooth gradients, animations, and responsive layout

---

## 📸 How It Works

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  1. Upload   │ ──▶ │  2. Paint    │ ──▶ │  3. Process  │ ──▶ │  4. Download │
│    Video     │     │    Mask      │     │  (Inpaint)   │     │    Result    │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

1. **Upload** a video file via drag-and-drop or file picker
2. **Paint** over the watermark area on the first frame using the brush tool
3. **Process** — the backend splits frames, runs inpainting, and reassembles the video
4. **Download** the watermark-free video with original audio intact

---

## 🏗️ Architecture

```
watermark/
├── Dockerfile              # Lightweight python:3.10-slim container
├── docker-compose.yml      # Compose config with volume mounts
├── .dockerignore            # Excludes dev files from build context
├── .gitignore               # Git ignore rules
├── requirements.txt         # Python dependencies (no PyTorch!)
│
├── main.py                  # FastAPI app — 5 REST endpoints
├── video_utils.py           # FFmpeg wrapper — audio/frame/video pipeline
├── inpainter.py             # OpenCV Telea inpainting + multiprocessing
│
├── uploads/                 # Runtime directory (gitignored)
│
└── static/
    ├── index.html           # Single-page frontend (semantic HTML5)
    ├── style.css            # Premium dark-mode design system
    └── app.js               # Vanilla JS — canvas, upload, polling logic
```

### Backend (Python / FastAPI)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | POST | Upload a video file, extract first frame |
| `/frame/{task_id}` | GET | Retrieve the first frame as PNG for the canvas |
| `/process` | POST | Submit a binary mask and start background processing |
| `/status/{task_id}` | GET | Poll processing progress (0–100%) |
| `/download/{task_id}` | GET | Download the final processed video |

### Processing Pipeline

```
Original Video
    │
    ├──▶ Extract Audio (AAC)
    ├──▶ Split into PNG Frames
    ├──▶ Inpaint Each Frame with Mask (OpenCV Telea, multi-core)
    │
    └──▶ Reassemble Frames + Remux Audio ──▶ Output.mp4
```

### Frontend (Vanilla JS + Canvas API)

- **No frameworks** — pure HTML5, CSS, and JavaScript
- **Canvas mask editor** — draws semi-transparent red overlay; exports as black/white PNG
- **Safe DOM APIs** — no `innerHTML`; uses `createElement`, `textContent` throughout

---

## 🚀 Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)

### Run

```bash
# Clone the repository
git clone https://github.com/Shivam-1M/watermark-remover-ai.git
cd watermark-remover-ai

# Build and start the container
docker-compose up --build

# Open in your browser
# http://localhost:8000
```

### Stop

```bash
docker-compose down
```

---

## ⚙️ Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `MAX_UPLOAD_SIZE_MB` | `100` | Maximum video upload size in MB |
| `UPLOAD_DIR` | `/app/uploads` | Directory for processed files |

These can be changed in `docker-compose.yml` under the `environment` section.

---

## 🤖 Inpainting Engine

The app uses **OpenCV's Telea Inpainting Algorithm** (Fast Marching Method):

- **Speed:** ~5-20ms per frame at 720p on a modern CPU.
- **Multi-core:** Frames are processed in parallel using `ProcessPoolExecutor` across all available CPU cores.
- **Precision:** Respects the exact painted mask shape (not a bounding box).
- **Lightweight:** No deep learning, no model weights, no GPU required.

> **Note:** Because each frame is inpainted independently (no temporal context), there may be slight visual inconsistency between frames in areas with highly dynamic backgrounds. For static or slowly-moving backgrounds (which is most watermark removal use cases), the results are excellent.

---

## 🔒 Security

| Measure | Implementation |
|---------|---------------|
| File validation | Extension allow-list + size limit (100 MB) + PNG magic byte check for masks |
| Path traversal | UUID-based filenames, resolved path boundary checks |
| HTTP headers | `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Permissions-Policy` |
| CSP | Content Security Policy restricting script/style/connect sources |
| CORS | Restricted to `localhost:8000` only |
| Downloads | `Content-Disposition: attachment` on all file responses |
| Frontend | No `innerHTML` / `document.write` — safe DOM APIs only |

### TODO (Production Hardening)

- [ ] Authentication and authorization
- [ ] Rate limiting on all API endpoints
- [ ] CSRF token handling
- [ ] HTTPS/TLS via reverse proxy
- [ ] Malware scanning for uploaded files
- [ ] Redis/DB-backed task store (replacing in-memory dict)

---

## 📋 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.10, FastAPI, Uvicorn |
| **Video Processing** | FFmpeg (via ffmpeg-python) |
| **Inpainting Engine** | OpenCV Telea (C++) + Python Multiprocessing |
| **Frontend** | HTML5, Vanilla CSS, Vanilla JavaScript, Canvas API |
| **Container** | Docker, python:3.10-slim |
| **Typography** | Inter (Google Fonts) |

---

## 📝 Development Log

### v3.0.0 — OpenCV Multiprocessing Engine (2026-07-06)

**Replaced ProPainter deep learning with CPU-optimized OpenCV inpainting**

- ✅ Switched from ProPainter (PyTorch transformer) to OpenCV Telea (Fast Marching Method).
- ✅ Added Python multiprocessing to distribute frame inpainting across all CPU cores.
- ✅ Removed ~2GB of PyTorch/CUDA/model weight dependencies from the Docker image.
- ✅ Build time reduced from ~12 minutes to ~1 minute.
- ✅ Processing speed: seconds instead of hours for a 240-frame video.
- ✅ No more OOM kills — OpenCV uses negligible memory per frame.

### v2.0.0 — ProPainter Integration (2026-07-06)

**Replaced mock OpenCV engine with real deep learning inpainting**

- ✅ Integrated ProPainter (ICCV 2023) sliding-window transformer architecture.
- ✅ Integrated RAFT optical flow and RecurrentFlowCompleteNet for temporal consistency.
- ⚠️ Too slow for CPU-only hardware — each RAFT chunk took ~90 seconds.
- ⚠️ OOM kills on 240+ frame videos even at 360p resolution.
- 🔄 Superseded by v3.0.0 which uses lightweight OpenCV instead.

### v1.0.0 — Initial Release (2026-07-06)

**Full-stack scaffold with mock AI pipeline**

- ✅ FastAPI backend with 5 REST endpoints
- ✅ FFmpeg video processing pipeline (extract audio, split frames, reassemble + remux)
- ✅ Drag-and-drop video upload with client + server validation
- ✅ Interactive HTML5 Canvas mask editor with brush tool, undo, and clear
- ✅ Real-time progress bar with `/status` polling
- ✅ Premium dark-mode UI — glassmorphism, ambient glows, violet/cyan palette
- ✅ Docker + Docker Compose with optional NVIDIA GPU pass-through
- ✅ WSL2-compatible volume mappings and file permissions
- ✅ Security hardening — CSP, safe DOM APIs, UUID filenames, path traversal checks
- ✅ Responsive design with reduced-motion accessibility support

**Bug Fixes:**
- 🐛 Fixed error toast appearing on page load (`hidden` attribute overridden by CSS `display: flex`)
- 🐛 Removed unsupported `frame-ancestors` directive from CSP `<meta>` tag
- 🐛 Made GPU pass-through optional in Docker Compose for environments without NVIDIA drivers

---

## 📄 License

This project is open source. Feel free to use, modify, and distribute.

---

<p align="center">
  Built with ◆ <strong>WatermarkAI</strong> — 100% local, zero cloud dependencies
</p>
