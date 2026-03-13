# StereoCrafter WebUI

A one-click web interface for [StereoCrafter](https://github.com/TencentARC/StereoCrafter) — convert 2D videos to stereoscopic 3D from your browser.

No command line needed. Just run the launcher, upload a video, and download your 3D result.

## Quick Start

**Windows:**
```
git clone --recursive https://github.com/user/StereoCrafterWebUI
cd StereoCrafterWebUI
run.bat
```

**Linux/Mac:**
```
git clone --recursive https://github.com/user/StereoCrafterWebUI
cd StereoCrafterWebUI
chmod +x run.sh && ./run.sh
```

That's it. The launcher handles everything automatically:
1. Creates a Python virtual environment
2. Installs dependencies (detects your GPU and installs the correct PyTorch)
3. Compiles the CUDA module
4. Opens the web UI in your browser

## Model Weights

On first launch, the UI shows a **Model Setup** panel to download the required models:

| Model | Size | Auth Required |
|---|---|---|
| [DepthCrafter](https://huggingface.co/tencent/DepthCrafter) | ~5 GB | No |
| [StereoCrafter](https://huggingface.co/TencentARC/StereoCrafter) | ~10 GB | No |
| [SVD img2vid](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1) | ~10 GB | Yes — accept license on HuggingFace first |

For SVD: visit the [model page](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1), accept the license, then paste your [HuggingFace token](https://huggingface.co/settings/tokens) into the UI.

## Requirements

- **GPU**: NVIDIA with 16GB+ VRAM (24GB+ for 2K video)
- **CUDA Toolkit**: Installed and on PATH ([download](https://developer.nvidia.com/cuda-toolkit))
- **Python**: 3.8+
- **OS**: Windows 10/11 or Linux

Tested on RTX 5090 (CUDA 12.8) and should work on GPUs from GTX 1080 to RTX 50-series.

## Output Formats

- **Side-by-Side** — for 3D displays, VR headsets
- **Anaglyph** — red/cyan, viewable with basic 3D glasses

## Parameters

Most users can leave defaults and just click **Convert to 3D**. For fine-tuning:

| Parameter | Default | What it does |
|---|---|---|
| Max Disparity | 20 | 3D depth strength. Higher = more pronounced 3D |
| Process Length | -1 | Frames to process (-1 = all) |
| Batch Size | 10 | Lower = less VRAM |
| Tile Number | 1 | Use 2 for 2K+, 4 for 4K resolution |

## Upstream

This is a WebUI wrapper around [StereoCrafter by Tencent ARC Lab](https://github.com/TencentARC/StereoCrafter). All credit for the core algorithm goes to the original authors.

If you use this in research, please cite the original paper:

```bibtex
@article{zhao2024stereocrafter,
  title={Stereocrafter: Diffusion-based generation of long and high-fidelity stereoscopic 3d from monocular videos},
  author={Zhao, Sijie and Hu, Wenbo and Cun, Xiaodong and Zhang, Yong and Li, Xiaoyu and Kong, Zhe and Gao, Xiangjun and Niu, Muyao and Shan, Ying},
  journal={arXiv preprint arXiv:2409.07447},
  year={2024}
}
```
