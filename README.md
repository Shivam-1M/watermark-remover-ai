# 🎬 WatermarkAI — Video Watermark Remover

> **AI-powered video watermark removal — 100% local, self-contained, no paid APIs.**

A full-stack application that removes watermarks from videos using AI inpainting. Upload a video, paint over the watermark on an interactive canvas, and download the clean result. Everything runs locally on your machine — no cloud services, no subscriptions.

---

## ✨ Features

- **Drag & Drop Upload** — Upload videos up to 100 MB (MP4, AVI, MOV, MKV, WebM)
- **Interactive Canvas Mask Editor** — Paint over watermarks with an adjustable brush, undo/clear support
- **AI Inpainting Engine** — Real ProPainter model (ICCV 2023) for temporal consistency
- **Audio Preservation** — Automatically extracts and remuxes the original audio track
- **Real-time Progress** — Live progress bar that polls the backend during processing
- **CPU-Optimized (Intel Iris Xe)** — Runs locally without NVIDIA GPU using automatic 480p downscaling
- **Premium Dark UI** — Glassmorphism design with smooth gradients, animations, and responsive layout

---

## 📸 How It Works

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  1. Upload   │ ──▶ │  2. Paint    │ ──▶ │  3. Process  │ ──▶ │  4. Download │
│    Video     │     │    Mask      │     │  (AI Magic)  │     │    Result    │
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
├── Dockerfile              # NVIDIA CUDA 11.8 container image
├── docker-compose.yml      # Compose config (optional GPU pass-through)
├── .dockerignore            # Excludes dev files from build context
├── .gitignore               # Git ignore rules
├── requirements.txt         # Python dependencies
│
├── main.py                  # FastAPI app — 5 REST endpoints
├── video_utils.py           # FFmpeg wrapper — audio/frame/video pipeline
├── inpainter.py             # AI inpainting engine (ProPainter integration)
│
├── propainter/              # Cloned ProPainter model source code
├── weights/                 # Pre-downloaded model checkpoints
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
    ├──▶ Inpaint Each Frame with Mask (AI)
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
- (Optional) [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU acceleration

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

## 🎮 GPU Acceleration (Optional)

The current Docker configuration runs purely on **CPU** due to the original host machine running Intel Iris Xe graphics. Inference will be slow, but it works cross-platform.

If you have an NVIDIA GPU, you can re-enable CUDA acceleration:

1. In `docker-compose.yml`, uncomment the `deploy` block.
2. In `Dockerfile`, change the base image from `python:3.10-slim` back to `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` and reinstall python via apt.
3. In `requirements.txt`, reinstall torch using the cu118 index url.
4. In `inpainter.py`, change `DEVICE = torch.device("cpu")` to `"cuda"`.

---

## 🤖 AI Model Integration

The app uses the **ProPainter** deep learning model by Shangchen Zhou et al. (ICCV 2023).

- **Optical Flow:** RAFT computes bidirectional flows.
- **Flow Completion:** Fills the flow field within the watermark region.
- **Inpainting:** A sliding-window transformer reconstructs the masked areas.

To keep memory and compute manageable on a CPU, videos are automatically downscaled to 480p before processing, and upscaled afterward. The model weights are automatically downloaded into the Docker image during `docker-compose build`.

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
| **AI Engine** | ProPainter (PyTorch) + RAFT |
| **Frontend** | HTML5, Vanilla CSS, Vanilla JavaScript, Canvas API |
| **Container** | Docker, NVIDIA CUDA 11.8 base image |
| **Typography** | Inter (Google Fonts) |

---

## 📝 Development Log

### v2.0.0 — ProPainter Integration (2026-07-06)

**Replaced mock OpenCV engine with real deep learning inpainting**

- ✅ Integrated ProPainter (ICCV 2023) sliding-window transformer architecture.
- ✅ Integrated RAFT optical flow and RecurrentFlowCompleteNet for temporal consistency.
- ✅ Added auto-downscaling to 480p for feasible CPU-bound processing.
- ✅ Loaded models into a FastAPI lifespan startup event to eliminate per-request loading latency.
- ✅ Baked model weights (~160MB) directly into the Docker image at build time.
- ✅ Dropped NVIDIA/CUDA dependencies in favour of `python:3.10-slim` to support Intel Iris Xe hardware natively.

### v1.0.0 — Initial Release (2026-07-06)

**Full-stack scaffold with mock AI pipeline**

- ✅ FastAPI backend with 5 REST endpoints
- ✅ FFmpeg video processing pipeline (extract audio, split frames, reassemble + remux)
- ✅ Mock inpainting engine using OpenCV Telea algorithm
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
