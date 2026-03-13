import os
import gc
import time

import gradio as gr
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

MODELS = {
    "SVD": {
        "path": os.path.join(WEIGHTS_DIR, "stable-video-diffusion-img2vid-xt-1-1"),
        "repo_id": "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
        "requires_auth": True,
    },
    "DepthCrafter": {
        "path": os.path.join(WEIGHTS_DIR, "DepthCrafter"),
        "repo_id": "tencent/DepthCrafter",
        "requires_auth": False,
    },
    "StereoCrafter": {
        "path": os.path.join(WEIGHTS_DIR, "StereoCrafter"),
        "repo_id": "TencentARC/StereoCrafter",
        "requires_auth": False,
    },
}

SVD_PATH = MODELS["SVD"]["path"]
DEPTHCRAFTER_PATH = MODELS["DepthCrafter"]["path"]
STEREOCRAFTER_PATH = MODELS["StereoCrafter"]["path"]


# ── Model management ──────────────────────────────────────────────────

def get_status_md():
    header = "| Model | Status | Needs HF Token |\n|---|---|---|\n"
    rows = []
    for name, info in MODELS.items():
        ready = os.path.isdir(info["path"])
        status = "Ready" if ready else "**Missing**"
        auth = "Yes" if info["requires_auth"] else "No"
        rows.append(f"| {name} | {status} | {auth} |")
    return header + "\n".join(rows)


def download_models(hf_token, progress=gr.Progress()):
    from huggingface_hub import snapshot_download

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    token = hf_token.strip() if hf_token else None

    missing = {n: m for n, m in MODELS.items() if not os.path.isdir(m["path"])}
    if not missing:
        gr.Info("All models already downloaded.")
        return get_status_md()

    for i, (name, info) in enumerate(missing.items()):
        if info["requires_auth"] and not token:
            raise gr.Error(
                f"{name} requires a HuggingFace token. "
                "1) Accept the license at huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1 "
                "2) Enter your HF token above."
            )
        progress(i / len(missing), desc=f"Downloading {name}...")
        snapshot_download(
            repo_id=info["repo_id"],
            local_dir=info["path"],
            token=token if info["requires_auth"] else None,
        )

    progress(1.0, desc="All models downloaded!")
    gr.Info("All models downloaded successfully!")
    return get_status_md()


# ── Conversion ────────────────────────────────────────────────────────

def resolve_video_path(file_path, file_upload):
    """Use typed path if provided, otherwise fall back to uploaded file."""
    if file_path and file_path.strip():
        path = file_path.strip()
        if not os.path.isfile(path):
            raise gr.Error(f"File not found: {path}")
        return path
    if file_upload is not None:
        return file_upload
    raise gr.Error("Please provide a video path or upload a file.")


def convert(file_path, file_upload, max_disp, process_length, batch_size,
            frames_chunk, overlap, tile_num, progress=gr.Progress()):
    input_video = resolve_video_path(file_path, file_upload)

    for name, info in MODELS.items():
        if not os.path.isdir(info["path"]):
            raise gr.Error(f"Missing model: {name}. Go to Model Setup to download it.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    process_length = int(process_length)
    tile_num = int(tile_num)

    video_basename = os.path.splitext(os.path.basename(input_video))[0]
    run_name = f"{video_basename}_{int(time.time())}"
    splatting_path = os.path.join(OUTPUT_DIR, f"{run_name}_splatting_results.mp4")

    # ── Stage 1: Depth Splatting ──
    progress(0.0, desc="Stage 1/2: Loading DepthCrafter...")
    from depth_splatting_inference import DepthCrafterDemo, DepthSplatting

    demo = DepthCrafterDemo(unet_path=DEPTHCRAFTER_PATH, pre_trained_path=SVD_PATH)

    progress(0.15, desc="Stage 1/2: Estimating depth...")
    video_depth, depth_vis = demo.infer(input_video, splatting_path, process_length)

    progress(0.35, desc="Stage 1/2: Forward warping...")
    DepthSplatting(input_video, splatting_path, video_depth, depth_vis,
                   max_disp, process_length, batch_size)

    del demo, video_depth, depth_vis
    gc.collect()
    torch.cuda.empty_cache()

    # ── Stage 2: Stereo Inpainting ──
    progress(0.45, desc="Stage 2/2: Loading StereoCrafter...")
    from inpainting_inference import main as inpainting_main

    # Forward slashes for the inpainting script's path splitting on Windows
    inpainting_main(
        pre_trained_path=SVD_PATH,
        unet_path=STEREOCRAFTER_PATH,
        input_video_path=splatting_path.replace("\\", "/"),
        save_dir=OUTPUT_DIR.replace("\\", "/"),
        frames_chunk=frames_chunk,
        overlap=overlap,
        tile_num=tile_num,
    )

    gc.collect()
    torch.cuda.empty_cache()
    progress(1.0, desc="Done!")

    # ── Collect results ──
    sbs_path = os.path.join(OUTPUT_DIR, f"{run_name}_inpainting_results_sbs.mp4")
    anaglyph_path = os.path.join(OUTPUT_DIR, f"{run_name}_inpainting_results_anaglyph.mp4")

    return (
        sbs_path if os.path.exists(sbs_path) else None,
        anaglyph_path if os.path.exists(anaglyph_path) else None,
    )


# ── Gradio UI ─────────────────────────────────────────────────────────

all_models_ready = all(os.path.isdir(m["path"]) for m in MODELS.values())

with gr.Blocks(title="StereoCrafter WebUI") as app:
    gr.Markdown("# StereoCrafter WebUI\nConvert 2D videos to stereoscopic 3D")

    # ── Model Setup (auto-open if models missing) ──
    with gr.Accordion("Model Setup", open=not all_models_ready):
        status_md = gr.Markdown(get_status_md())
        hf_token = gr.Textbox(
            label="HuggingFace Token (only needed for SVD model)",
            type="password",
            placeholder="hf_...",
            info="Accept license at huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1 then paste token from huggingface.co/settings/tokens",
        )
        download_btn = gr.Button("Download Missing Models")
        download_btn.click(fn=download_models, inputs=[hf_token], outputs=[status_md])

    # ── Conversion ──
    with gr.Row():
        with gr.Column():
            input_path = gr.Textbox(
                label="Video Path (no file copying)",
                placeholder=r"C:\Videos\my_video.mp4",
                info="Paste path directly — best for large files",
            )
            input_upload = gr.File(
                label="Or upload a file",
                file_types=["video"],
            )

            with gr.Accordion("Stage 1: Depth Splatting", open=False):
                max_disp = gr.Slider(1, 60, value=20, step=1,
                                     label="Max Disparity",
                                     info="Controls 3D depth strength")
                process_length = gr.Number(value=-1, precision=0,
                                           label="Process Length",
                                           info="Frames to process (-1 = all)")
                batch_size = gr.Slider(1, 50, value=10, step=1,
                                       label="Batch Size",
                                       info="Lower = less VRAM usage")

            with gr.Accordion("Stage 2: Inpainting", open=False):
                frames_chunk = gr.Slider(8, 50, value=23, step=1,
                                         label="Frames per Chunk")
                overlap = gr.Slider(1, 10, value=3, step=1,
                                    label="Chunk Overlap")
                tile_num = gr.Radio([1, 2, 4], value=1,
                                    label="Tile Number",
                                    info="Use 2 for 2K+, 4 for 4K")

            convert_btn = gr.Button("Convert to 3D", variant="primary", size="lg")

        with gr.Column():
            output_sbs = gr.Video(label="Side-by-Side Stereo")
            output_anaglyph = gr.Video(label="Anaglyph 3D")

    convert_btn.click(
        fn=convert,
        inputs=[input_path, input_upload, max_disp, process_length, batch_size,
                frames_chunk, overlap, tile_num],
        outputs=[output_sbs, output_anaglyph],
    )

if __name__ == "__main__":
    app.launch(inbrowser=True)
