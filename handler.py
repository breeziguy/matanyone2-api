import runpod
import base64
import tempfile
import os
import subprocess
import cv2
import numpy as np
import torch
from PIL import Image
from rembg import remove as rembg_remove
from torchvision.transforms.functional import to_tensor

import sys
sys.path.insert(0, "/app/MatAnyone2")

from matanyone2.utils.get_default_model import get_matanyone2_model
from matanyone2.inference.inference_core import InferenceCore
from matanyone2.utils.device import get_default_device, safe_autocast_decorator

CKPT_PATH = "/app/pretrained_models/matanyone2.pth"
device = get_default_device()

print("Loading MatAnyone2 model...")
matanyone2_model = get_matanyone2_model(CKPT_PATH, device)
print("Model loaded.")


def read_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def generate_mask(first_frame_np: np.ndarray) -> np.ndarray:
    """Auto-segment person using rembg. Swap for SAM2 later if needed."""
    result = rembg_remove(Image.fromarray(first_frame_np))
    alpha = np.array(result)[:, :, 3]
    return (alpha > 127).astype(np.uint8) * 255


@torch.inference_mode()
@safe_autocast_decorator()
def get_alpha_mattes(frames_np, mask_np, n_warmup=10):
    """Run MatAnyone2 and return per-frame alpha mattes (H,W) uint8."""
    processor = InferenceCore(matanyone2_model, cfg=matanyone2_model.cfg)
    mask_tensor = torch.from_numpy(mask_np).to(device)

    all_frames = [frames_np[0]] * n_warmup + frames_np
    phas = []

    for ti, frame in enumerate(all_frames):
        image = to_tensor(frame).float().to(device)
        if ti == 0:
            output_prob = processor.step(image, mask_tensor, objects=[1])
            output_prob = processor.step(image, first_frame_pred=True)
        elif ti <= n_warmup:
            output_prob = processor.step(image, first_frame_pred=True)
        else:
            output_prob = processor.step(image)

        mask_tensor = processor.output_prob_to_mask(output_prob)

        if ti > (n_warmup - 1):
            pha = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
            phas.append(pha)

    return phas


def write_transparent_mov(frames_np, phas, fps, output_path):
    """
    Combine original RGB frames + alpha mattes into a ProRes 4444 .mov
    with a real alpha channel — no green screen, original quality.
    """
    h, w = frames_np[0].shape[:2]

    # Pipe raw RGBA frames into ffmpeg → ProRes 4444 with alpha
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgba",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", "prores_ks",
        "-profile:v", "4444",   # ProRes 4444 — supports alpha
        "-pix_fmt", "yuva444p10le",
        "-vendor", "apl0",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    for frame_rgb, pha in zip(frames_np, phas):
        rgba = np.dstack([frame_rgb, pha])   # original color + alpha matte
        proc.stdin.write(rgba.tobytes())

    proc.stdin.close()
    proc.wait()


def handler(job):
    job_input = job["input"]
    video_b64 = job_input.get("video_base64")
    if not video_b64:
        return {"error": "Missing video_base64"}

    n_warmup = int(job_input.get("n_warmup", 10))

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.mov")
        output_path = os.path.join(tmpdir, "output.mov")

        with open(input_path, "wb") as f:
            f.write(base64.b64decode(video_b64))

        frames_np, fps = read_frames(input_path)
        mask_np = generate_mask(frames_np[0])
        phas = get_alpha_mattes(frames_np, mask_np, n_warmup=n_warmup)
        write_transparent_mov(frames_np, phas, fps, output_path)

        with open(output_path, "rb") as f:
            output_b64 = base64.b64encode(f.read()).decode("utf-8")

    return {"video_base64": output_b64, "format": "mov"}


runpod.serverless.start({"handler": handler})
