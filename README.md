# 🎬 WatermarkAI — Video Watermark Remover

> **AI-powered video watermark removal — 100% local, self-contained, no paid APIs.**

A full-stack application that removes watermarks from videos using AI inpainting. Upload a video, paint over the watermark on an interactive canvas, and download the clean result. Everything runs locally on your machine — no cloud services, no subscriptions.

---

## ✨ Features

- **Drag & Drop Upload** — Upload videos up to 100 MB (MP4, AVI, MOV, MKV, WebM)
- **Interactive Canvas Mask Editor** — Paint over watermarks with an adjustable brush, undo/clear support
- **AI Inpainting Engine** — Frame-by-frame watermark removal with a pluggable model architecture
- **Audio Preservation** — Automatically extracts and remuxes the original audio track
- **Real-time Progress** — Live progress bar that polls the backend during processing
- **GPU Acceleration Ready** — Docker configuration supports NVIDIA GPU pass-through via WSL2
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
├── inpainter.py             # AI inpainting engine (mock + integration points)
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

The app supports NVIDIA GPU pass-through for PyTorch-based model inference. This requires WSL2 with NVIDIA drivers.

### Setup

1. Install NVIDIA GPU drivers for WSL2 on the **Windows host**
2. Inside WSL2, install the NVIDIA Container Toolkit:
   ```bash
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```
3. Uncomment the GPU block in `docker-compose.yml`:
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: all
             capabilities: [gpu]
   ```

---

## 🤖 AI Model Integration

The current build uses **OpenCV's Telea inpainting** as a lightweight mock. The codebase is structured for drop-in replacement with a PyTorch model like [ProPainter](https://github.com/sczhou/ProPainter).

Search for `>>> INTEGRATION POINT` in `inpainter.py` to find the three injection sites:

| Location | Purpose |
|----------|---------|
| **Line ~30** | PyTorch / model imports |
| **Line ~40** | Model loading (module-level singleton) |
| **Line ~150** | Frame inference in `_inpaint_single_frame()` |

Each integration point includes complete example code for tensor conversions and inference calls.

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
| **AI Engine** | OpenCV (mock) / PyTorch + ProPainter (production) |
| **Frontend** | HTML5, Vanilla CSS, Vanilla JavaScript, Canvas API |
| **Container** | Docker, NVIDIA CUDA 11.8 base image |
| **Typography** | Inter (Google Fonts) |

---

## 📝 Development Log

### v1.0.0 — Initial Release (2025-07-06)

**Full-stack scaffold with mock AI pipeline**

- ✅ FastAPI backend with 5 REST endpoints
- ✅ FFmpeg video processing pipeline (extract audio, split frames, reassemble + remux)
- ✅ Mock inpainting engine using OpenCV Telea algorithm
- ✅ ProPainter integration points with example PyTorch code
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
