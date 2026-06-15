import runpod
import base64
import tempfile
import os
import subprocess
import cv2
import numpy as np
import torch
from PIL import Image

CKPT = "/app/weights/sam2.1_hiera_large.pt"
CFG  = "configs/sam2.1/sam2.1_hiera_l.yaml"

print("Loading SAM2 model...")
from sam2.build_sam import build_sam2_video_predictor
predictor = build_sam2_video_predictor(CFG, CKPT)
print("SAM2 ready.")


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


def get_person_center(frame_rgb):
    """Find person center using rembg — returns (x, y) in original frame coords."""
    from rembg import remove
    result = remove(Image.fromarray(frame_rgb))
    alpha = np.array(result)[:, :, 3]
    ys, xs = np.where(alpha > 127)
    if len(xs) == 0:
        # fallback: center of frame
        h, w = frame_rgb.shape[:2]
        return w // 2, h // 2
    return int(xs.mean()), int(ys.mean())


def run_sam2(frames, prompt_x, prompt_y, frames_dir):
    """Run SAM2 video predictor. Returns list of (H,W) uint8 alpha masks."""
    # Save frames as JPEG for SAM2 video predictor
    for i, frame in enumerate(frames):
        cv2.imwrite(
            os.path.join(frames_dir, f"{i:05d}.jpg"),
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        )

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=frames_dir)

        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=1,
            points=np.array([[prompt_x, prompt_y]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )

        masks = {}
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            mask = (mask_logits[0][0].cpu().numpy() > 0).astype(np.uint8) * 255
            masks[frame_idx] = mask

    return [masks[i] for i in range(len(frames))]


def write_transparent_mov(frames, alphas, fps, output_path):
    """Pipe RGBA frames to ffmpeg → ProRes 4444 .mov with alpha channel."""
    h, w = frames[0].shape[:2]
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", "prores_ks", "-profile:v", "4444",
        "-pix_fmt", "yuva444p10le", "-vendor", "apl0",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for frame_rgb, alpha in zip(frames, alphas):
        rgba = np.dstack([frame_rgb, alpha])
        proc.stdin.write(rgba.tobytes())
    proc.stdin.close()
    proc.wait()


def handler(job):
    job_input = job["input"]
    video_b64 = job_input.get("video_base64")
    if not video_b64:
        return {"error": "Missing video_base64"}

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path  = os.path.join(tmpdir, "input.mov")
        frames_dir  = os.path.join(tmpdir, "frames")
        output_path = os.path.join(tmpdir, "output.mov")
        os.makedirs(frames_dir)

        with open(input_path, "wb") as f:
            f.write(base64.b64decode(video_b64))

        frames, fps = read_frames(input_path)

        # Use provided prompt or auto-detect person center
        prompt_x = job_input.get("prompt_x")
        prompt_y = job_input.get("prompt_y")
        if prompt_x is None or prompt_y is None:
            prompt_x, prompt_y = get_person_center(frames[0])

        alphas = run_sam2(frames, prompt_x, prompt_y, frames_dir)
        write_transparent_mov(frames, alphas, fps, output_path)

        with open(output_path, "rb") as f:
            output_b64 = base64.b64encode(f.read()).decode("utf-8")

    return {"video_base64": output_b64, "format": "mov"}


runpod.serverless.start({"handler": handler})
