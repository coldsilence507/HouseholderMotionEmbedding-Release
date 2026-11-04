#!/usr/bin/env python3
"""
NoiseControl inference: generate a video by transferring motion from a reference video.

Usage:
  conda run -n <env> python infer.py \
    --ref_video   path/to/reference.mp4 \
    --first_frame path/to/first_frame.png \
    --prompt      "a photo of a dog running" \
    --output      output.mp4

Optional:
  --lora_path           path/to/pytorch_lora_weights.safetensors
  --model               THUDM/CogVideoX-5b-I2V
  --alpha               0.3   # 0 = full warp, 1 = pure random noise
  --guidance_scale      6.0
  --num_inference_steps 50
  --seed                42
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from torchvision.models.optical_flow import raft_large

# ── Path setup ────────────────────────────────────────────────────────────────

REPO_ROOT    = Path(__file__).parent
TRAINING_DIR = REPO_ROOT / "training" / "cogvideox"
sys.path.insert(0, str(TRAINING_DIR))

from model.HouseHolderTransform_flow import FlowHouseholder  # noqa: E402

FLOW_HOUSEHOLDER_PTH = TRAINING_DIR / "model" / "flow_householder_P2.pth"
DEFAULT_LORA_PATH = (
    REPO_ROOT
    / "ckpt/ckpt"
    / "experiment-lora-of-noise-downtemp-blendnorm-degrad-fix-cogvideox-lora"
      "__optimizer_adamw__steps_30000__lr-schedule_cosine_with_restarts"
      "__learning-rate_1e-4"
    / "checkpoint-30000"
    / "pytorch_lora_weights.safetensors"
)
DEFAULT_MODEL = "THUDM/CogVideoX-5b-I2V"

# CogVideoX-5b-I2V output resolution
HEIGHT, WIDTH, NUM_FRAMES = 480, 720, 49


# ── Noise helpers (ported from training/cogvideox/test_of.py) ─────────────────

def _split_into_n_sublists(t: torch.Tensor, n: int) -> list:
    """Split first dim of tensor into n contiguous groups."""
    length = len(t)
    cuts = [round(i * length / n) for i in range(n + 1)]
    return [t[cuts[i]:cuts[i + 1]] for i in range(n)]


def downtemp_noise(noise: torch.Tensor, mode: str = "blend_norm") -> torch.Tensor:
    """Reduce 49 temporal frames → 13 latent frames via grouped mean + normalisation.

    noise : [49, C, H, W]
    returns [13, C, H, W]
    """
    assert mode == "blend_norm", f"unsupported mode: {mode}"
    groups  = _split_into_n_sublists(noise, 13)
    avg     = torch.stack([g.float().mean(0) for g in groups])       # [13, C, H, W]
    normed  = torch.stack([
        x / x.std(1, keepdim=True).clamp(min=1e-6) for x in avg
    ])                                                                 # [13, C, H, W]
    return normed.to(noise.dtype)


def mix_new_noise(noise: torch.Tensor, alpha: float) -> torch.Tensor:
    """Variance-preserving blend of warped noise with fresh random noise.

    alpha=0 → keep warped noise unchanged
    alpha=1 → pure independent Gaussian
    """
    fresh = torch.randn_like(noise)
    return (noise * (1 - alpha) + fresh * alpha) / (alpha**2 + (1 - alpha)**2) ** 0.5


def downsample_flow(flow: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """flow : [B, F, H, W, 2] → [B, F, H', W', 2], rescaling u/v accordingly."""
    B, N, H, W, _ = flow.shape
    f = flow.permute(0, 1, 4, 2, 3).contiguous()          # [B, F, 2, H, W]
    f = F.interpolate(f, size=(2, out_h, out_w), mode="area")
    f[:, :, 0] *= out_w / float(W)
    f[:, :, 1] *= out_h / float(H)
    return f.permute(0, 1, 3, 4, 2).contiguous()          # [B, F, H', W', 2]


# ── Video I/O ─────────────────────────────────────────────────────────────────

def load_video_tensor(
    path: Path,
    num_frames: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """Load MP4 → [1, F, 3, H, W] float32 normalised to [-1, 1]."""
    cap    = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < num_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"Could not read any frames from {path}")

    # Pad short videos by repeating the last frame
    while len(frames) < num_frames:
        frames.append(frames[-1])
    frames = frames[:num_frames]

    t = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float()  # [F,3,H,W]
    t = F.interpolate(t, size=(height, width), mode="bilinear", align_corners=False)
    t = t / 127.5 - 1.0                                                  # [-1, 1]
    return t.unsqueeze(0).to(device)                                     # [1,F,3,H,W]


# ── Warped-noise extraction ───────────────────────────────────────────────────

def _flow_to_bgr(flow_hw: torch.Tensor) -> np.ndarray:
    """Convert [1, 2, H, W] flow tensor to BGR uint8 HSV colour image (full-res)."""
    dx = flow_hw[0, 0].cpu().float().numpy()
    dy = flow_hw[0, 1].cpu().float().numpy()
    mag = np.sqrt(dx ** 2 + dy ** 2)
    ang = np.arctan2(dy, dx)
    max_mag = np.percentile(mag, 98) + 1e-6
    h = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    s = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)
    v = np.full_like(h, 255)
    return cv2.cvtColor(np.stack([h, s, v], axis=-1), cv2.COLOR_HSV2BGR)


def save_flow_vis_video(
    flow_frames: list[np.ndarray],   # list of BGR uint8 arrays
    output_path: Path,
    fps: int = 8,
) -> None:
    """Write optical-flow colour frames to an MP4."""
    if not flow_frames:
        return
    H, W = flow_frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )
    for frame in flow_frames:
        writer.write(frame)
    writer.release()
    print(f"Flow vis → {output_path}")


@torch.no_grad()
def get_noise_from_video_flow(
    videos: torch.Tensor,           # [1, F, 3, H, W] in [-1, 1]
    raft_model: torch.nn.Module,
    householder: FlowHouseholder,
    device: torch.device,
    dtype: torch.dtype,
    return_flows: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[np.ndarray]]:
    """Extract optical flow from reference video and warp noise with FlowHouseholder.

    Returns: [1, 1, F, C, H//8, W//8]
    If return_flows=True: also returns list of 48 BGR flow-vis frames (H//2, W//2).
    """
    resize_flow = 4

    # Halve spatial resolution before running RAFT (matches training)
    rs = F.interpolate(
        videos, size=(3, videos.shape[-2] // 2, videos.shape[-1] // 2), mode="area"
    )  # [1, F, 3, H//2, W//2]

    all_noises = []
    flow_vis_frames: list[np.ndarray] = []

    for b in range(rs.shape[0]):
        prev  = rs[b][0]                          # [3, H//2, W//2]
        _, H, W = prev.shape
        noise = torch.randn(
            (1, 1, householder.C, H // resize_flow, W // resize_flow),
            device=device, dtype=dtype,
        )
        noises = [noise]

        for frame in rs[b][1:]:
            # RAFT: input range [-1, 1], returns list of flow fields (finest = last)
            flow_hw = raft_model(prev.unsqueeze(0), frame.unsqueeze(0))[-1]  # [1,2,H,W]
            dx, dy  = flow_hw[0].unsqueeze(-1)                               # each [H,W,1]
            prev    = frame

            if return_flows and b == 0:
                flow_vis_frames.append(_flow_to_bgr(flow_hw))

            flow    = torch.cat([dx, dy], dim=-1).view(1, 1, H, W, 2)
            ds_flow = downsample_flow(flow, H // resize_flow, W // resize_flow)
            noises.append(householder(noises[-1], ds_flow))

        all_noises.append(torch.cat(noises, dim=1))  # [1, F, C, H', W']

    result = torch.stack(all_noises).to(dtype=dtype)  # [1, 1, F, C, H', W']
    if return_flows:
        return result, flow_vis_frames
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="NoiseControl inference")
    p.add_argument("--ref_video",   required=True,  help="Reference video for motion extraction")
    p.add_argument("--first_frame", required=True,  help="First frame image for the new video")
    p.add_argument("--prompt",      required=True,  help="Text prompt")
    p.add_argument("--output",      default="output.mp4")
    p.add_argument("--model",       default=DEFAULT_MODEL)
    p.add_argument("--lora_path",   default=str(DEFAULT_LORA_PATH))
    p.add_argument("--alpha",       type=float, default=0.3,
                   help="Noise degradation: 0=full warp, 1=pure random (default: 0.3)")
    p.add_argument("--guidance_scale",      type=float, default=6.0)
    p.add_argument("--num_inference_steps", type=int,   default=50)
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device)
    dtype  = torch.bfloat16

    # ── 1. Load RAFT ──────────────────────────────────────────────────────
    print("[1/5] Loading RAFT...")
    raft = raft_large(pretrained=True, progress=False).to(device).eval()

    # ── 2. Load FlowHouseholder ───────────────────────────────────────────
    print("[2/5] Loading FlowHouseholder...")
    householder = FlowHouseholder(C=16, P=2, seed=42)
    householder.load_state_dict(
        torch.load(FLOW_HOUSEHOLDER_PTH, weights_only=True, map_location="cpu")
    )
    householder.eval().to(device)

    # ── 3. Warp noise from reference video ────────────────────────────────
    print("[3/5] Extracting flow and warping noise...")
    videos = load_video_tensor(Path(args.ref_video), NUM_FRAMES, HEIGHT, WIDTH, device)

    correlated, flow_vis = get_noise_from_video_flow(
        videos, raft, householder, device, dtype, return_flows=True
    )
    # correlated: [1, 1, 49, 16, 60, 90]

    flow_vis_path = Path(args.output).with_name(Path(args.output).stem + "_flow_vis.mp4")
    save_flow_vis_video(flow_vis, flow_vis_path)

    noise   = mix_new_noise(correlated, alpha=args.alpha)       # [1, 1, 49, 16, 60, 90]
    latents = downtemp_noise(noise[0].squeeze(0), "blend_norm") # [13, 16, 60, 90]
    latents = latents.unsqueeze(0).to(device, dtype=dtype)      # [1, 13, 16, 60, 90]

    del raft, householder, videos, correlated, noise
    torch.cuda.empty_cache()

    # ── 4. Load CogVideoX + LoRA ──────────────────────────────────────────
    print("[4/5] Loading CogVideoX pipeline...")
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload()

    lora_path = Path(args.lora_path)
    if lora_path.exists():
        print(f"         Loading LoRA: {lora_path}")
        pipe.load_lora_weights(str(lora_path))
    else:
        print(f"Warning: LoRA not found at {lora_path} — running base model only.")

    # ── 5. Generate ────────────────────────────────────────────────────────
    print("[5/5] Generating...")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    output = pipe(
        image=load_image(args.first_frame),
        prompt=args.prompt,
        latents=latents,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        generator=generator,
        use_dynamic_cfg=True,
    )

    export_to_video(output.frames[0], args.output, fps=8)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
