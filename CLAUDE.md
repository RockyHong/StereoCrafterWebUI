# CLAUDE.md - StereoCrafterWebUI

## Project Overview

StereoCrafter converts 2D monocular videos into stereoscopic 3D videos using a two-stage deep learning pipeline. This fork adds a **Gradio-based Web UI** so anyone can clone, run a script, and use it from the browser.

## Architecture

### Core Pipeline (upstream — do not modify)
1. **Stage 1 - Depth Splatting** (`depth_splatting_inference.py`): Depth estimation via DepthCrafter + forward warping to generate candidate right-eye view and occlusion mask.
2. **Stage 2 - Inpainting** (`inpainting_inference.py`): Diffusion-based stereo inpainting (StereoCrafter UNet) fills occluded regions and outputs final stereo video.

### Web UI (our additions)
- `app.py` — Gradio single-page app: model setup, video upload, conversion, download
- `run.bat` / `run.sh` — One-click launcher with lazy initialization (venv, pip, CUDA module)
- `requirements_webui.txt` — Our dependencies, extends upstream `requirements.txt` via `-r`

## Directory Structure

```
StereoCrafterWebUI/
├── CLAUDE.md                          # This file
├── app.py                             # Web UI entry point
├── run.bat                            # Windows launcher (double-click)
├── run.sh                             # Linux/Mac launcher
├── requirements_webui.txt             # WebUI deps (extends requirements.txt)
├── requirements.txt                   # Upstream deps (DO NOT MODIFY)
├── depth_splatting_inference.py        # Stage 1 (upstream)
├── inpainting_inference.py            # Stage 2 (upstream)
├── pipelines/
│   └── stereo_video_inpainting.py     # Custom diffusion pipeline (upstream)
├── dependency/
│   ├── DepthCrafter/                  # Depth estimation (git submodule)
│   └── Forward-Warp/                  # CUDA forward warping (git submodule)
├── weights/                           # Model checkpoints (not committed)
│   ├── stable-video-diffusion-img2vid-xt-1-1/  # Requires HF license acceptance
│   ├── DepthCrafter/                            # Public
│   └── StereoCrafter/                           # Public
├── source_video/                      # Sample inputs
└── outputs/                           # Generated results
```

## Key Technical Details

- **Python stack**: torch 2.0.1, diffusers 0.29.2, transformers 4.42.3, decord, opencv
- **GPU requirements**: CUDA 11.8+, 16GB+ VRAM (24GB+ for 2K video)
- **Custom CUDA module**: `Forward-Warp` compiled via `pip install .` in `dependency/Forward-Warp/`
- **Video I/O**: `decord` for reading, `opencv` / `mediapy` for writing
- **Windows path quirk**: `inpainting_inference.py` uses `split("/")` for path parsing — always pass forward-slash paths to it

## Model Weights

| Model | HuggingFace Repo | Auth Required |
|---|---|---|
| SVD | `stabilityai/stable-video-diffusion-img2vid-xt-1-1` | Yes — user must accept license on HF then provide token |
| DepthCrafter | `tencent/DepthCrafter` | No |
| StereoCrafter | `TencentARC/StereoCrafter` | No |

The Gradio UI has a "Model Setup" panel that auto-detects which models are present and can download missing ones (via `huggingface_hub.snapshot_download`).

## Launcher Flow (run.bat / run.sh)

1. Check Python is installed
2. Create `venv/` if it doesn't exist
3. Activate venv
4. Install `requirements_webui.txt` if first run (marker: `venv/.webui_installed`)
5. Compile Forward-Warp CUDA module if first run (marker: `venv/.forwardwarp_installed`)
6. Launch `python app.py` (opens browser automatically)

## Development Guidelines

- **Platform**: Windows 11, bash shell (Git Bash), CUDA GPU
- **Branch**: `webui` (main development branch for the UI)
- **Do not modify upstream files** (`requirements.txt`, `depth_splatting_inference.py`, `inpainting_inference.py`, `pipelines/`) — wrap them from `app.py`
- **Keep the UI minimal** — the goal is to minimize effort for the end user
- **Separate our deps**: use `requirements_webui.txt`, never pollute upstream `requirements.txt`

## Relevant Skills

- `python-pro` — Python patterns, async, type hints
- `frontend-design` — UI layout and design for the Gradio interface
- `debugging-wizard` — Investigating CUDA/GPU/model loading issues
