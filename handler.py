import runpod
import base64
import tempfile
import os
import cv2
import imageio
import numpy as np
import torch
import ffmpeg
from PIL import Image
from rembg import remove as rembg_remove
from torchvision.transforms.functional import to_tensor

import sys
sys.path.insert(0, "/app/MatAnyone2")

from matanyone2.utils.get_default_model import get_matanyone2_model
from matanyone2.inference.inference_core import InferenceCore
from matanyone2.utils.device import get_default_device, safe_autocast_decorator


def read_frames_from_video(video_path):
    """Read video frames using cv2. Returns (frames_np, fps, num_frames)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps, len(frames)

CKPT_PATH = "/app/pretrained_models/matanyone2.pth"
device = get_default_device()

print("Loading MatAnyone2 model...")
matanyone2_model = get_matanyone2_model(CKPT_PATH, device)
print("Model loaded.")


def generate_mask(first_frame_np: np.ndarray) -> np.ndarray:
    """Auto-segment person in first frame using rembg. Swap this function later for SAM2 or interactive draw."""
    pil_img = Image.fromarray(first_frame_np)
    result = rembg_remove(pil_img)
    alpha = np.array(result)[:, :, 3]
    mask = (alpha > 127).astype(np.uint8) * 255
    return mask


@torch.inference_mode()
@safe_autocast_decorator()
def run_matanyone2(frames_np, mask_np, n_warmup=10):
    processor = InferenceCore(matanyone2_model, cfg=matanyone2_model.cfg)
    mask_tensor = torch.from_numpy(mask_np).to(device)
    objects = [1]

    all_frames = [frames_np[0]] * n_warmup + frames_np
    fgrs, phas = [], []
    bgr = (np.array([120, 255, 155], dtype=np.float32) / 255).reshape((1, 1, 3))

    for ti, frame in enumerate(all_frames):
        image = to_tensor(frame).float().to(device)

        if ti == 0:
            output_prob = processor.step(image, mask_tensor, objects=objects)
            output_prob = processor.step(image, first_frame_pred=True)
        elif ti <= n_warmup:
            output_prob = processor.step(image, first_frame_pred=True)
        else:
            output_prob = processor.step(image)

        mask_tensor = processor.output_prob_to_mask(output_prob)
        pha = mask_tensor.unsqueeze(2).cpu().numpy()
        com = frame / 255.0 * pha + bgr * (1 - pha)

        if ti > (n_warmup - 1):
            fgrs.append((com * 255).astype(np.uint8))
            phas.append((pha * 255).astype(np.uint8))

    return fgrs, phas


def merge_to_transparent_webm(fgr_frames, pha_frames, fps, output_path):
    tmp_fgr = output_path.replace(".webm", "_fgr_tmp.mp4")
    tmp_pha = output_path.replace(".webm", "_pha_tmp.mp4")

    imageio.mimwrite(tmp_fgr, fgr_frames, fps=fps, quality=9)
    imageio.mimwrite(tmp_pha, [f[:, :, 0] for f in pha_frames], fps=fps, quality=9)

    (
        ffmpeg
        .output(
            ffmpeg.input(tmp_fgr),
            ffmpeg.input(tmp_pha),
            output_path,
            filter_complex="[0:v][1:v]alphamerge",
            vcodec="libvpx-vp9",
            pix_fmt="yuva420p",
            **{"b:v": "0", "crf": "10"},
        )
        .overwrite_output()
        .run(quiet=True)
    )

    os.remove(tmp_fgr)
    os.remove(tmp_pha)


def handler(job):
    job_input = job["input"]
    video_b64 = job_input.get("video_base64")
    if not video_b64:
        return {"error": "Missing video_base64"}

    n_warmup = int(job_input.get("n_warmup", 10))

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.mp4")
        output_path = os.path.join(tmpdir, "output.webm")

        with open(input_path, "wb") as f:
            f.write(base64.b64decode(video_b64))

        frames_np, fps, length = read_frames_from_video(input_path)

        mask_np = generate_mask(frames_np[0])
        fgr_frames, pha_frames = run_matanyone2(frames_np, mask_np, n_warmup=n_warmup)
        merge_to_transparent_webm(fgr_frames, pha_frames, fps, output_path)

        with open(output_path, "rb") as f:
            output_b64 = base64.b64encode(f.read()).decode("utf-8")

    return {"video_base64": output_b64, "format": "webm"}


runpod.serverless.start({"handler": handler})
