#!/usr/bin/env python3
"""
NoiseControl inference from pre-computed optical flow.

Like infer.py but takes a flow .npy file instead of a reference video,
skipping the RAFT step entirely.

Flow npy format:  (F-1, 2, H_half, W_half) float  — RAFT-style (dx, dy) in pixels
  e.g. (48, 2, 240, 360) for a 49-frame 480×720 video.

Usage:
  conda run -n videogen python infer_from_flow.py \
    --flows       path/to/flows_dxdy.npy \
    --first_frame path/to/first_frame.png \
    --prompt      "a black swan swimming" \
    --output      output.mp4
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from PIL import Image

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

HEIGHT, WIDTH, NUM_FRAMES = 480, 720, 49


# ── Noise helpers (identical to infer.py) ────────────────────────────────────

def _split_into_n_sublists(t: torch.Tensor, n: int) -> list:
    length = len(t)
    cuts = [round(i * length / n) for i in range(n + 1)]
    return [t[cuts[i]:cuts[i + 1]] for i in range(n)]


def downtemp_noise(noise: torch.Tensor, mode: str = "blend_norm") -> torch.Tensor:
    """49 temporal frames → 13 latent frames via grouped mean + normalisation."""
    assert mode == "blend_norm"
    groups = _split_into_n_sublists(noise, 13)
    avg    = torch.stack([g.float().mean(0) for g in groups])
    normed = torch.stack([x / x.std(1, keepdim=True).clamp(min=1e-6) for x in avg])
    return normed.to(noise.dtype)


def mix_new_noise(noise: torch.Tensor, alpha: float) -> torch.Tensor:
    fresh = torch.randn_like(noise)
    return (noise * (1 - alpha) + fresh * alpha) / (alpha ** 2 + (1 - alpha) ** 2) ** 0.5


def downsample_flow(flow: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """flow : [B, F, H, W, 2] → [B, F, H', W', 2]"""
    B, N, H, W, _ = flow.shape
    f = flow.permute(0, 1, 4, 2, 3).contiguous()           # [B, F, 2, H, W]
    f = F.interpolate(f, size=(2, out_h, out_w), mode="area")
    f[:, :, 0] *= out_w / float(W)
    f[:, :, 1] *= out_h / float(H)
    return f.permute(0, 1, 3, 4, 2).contiguous()            # [B, F, H', W', 2]


# ── First-frame helper ───────────────────────────────────────────────────────

def load_first_frame(path: str) -> Image.Image:
    """Accept an image file or an MP4 — returns a PIL Image of the first frame."""
    import cv2
    p = Path(path)
    if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        cap = cv2.VideoCapture(str(p))
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            raise ValueError(f"Could not read first frame from video: {p}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    return load_image(path)


# ── Core: warp noise from pre-computed flows ──────────────────────────────────

@torch.no_grad()
def get_noise_from_flow_npy(
    flows_np: np.ndarray,        # (F-1, 2, H_half, W_half) float — RAFT output
    householder: FlowHouseholder,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Warp noise with FlowHouseholder using pre-computed optical flow.

    flows_np : (F-1, 2, H_half, W_half) — frame-to-frame flows in pixel units
               at half-resolution (matches what infer.py extracts via RAFT).

    Returns : [1, 1, F, C, H_half//4, W_half//4]
    """
    resize_flow = 4
    num_flows, _, H, W = flows_np.shape          # H=240, W=360 for 480×720 video

    noise = torch.randn(
        (1, 1, householder.C, H // resize_flow, W // resize_flow),
        device=device, dtype=dtype,
    )  # [1, 1, C, 60, 90]
    noises = [noise]

    flows_t = torch.from_numpy(flows_np).to(device=device)  # float32 [F-1, 2, H, W]

    for i in range(num_flows):
        dx = flows_t[i, 0].unsqueeze(-1)   # [H, W, 1]
        dy = flows_t[i, 1].unsqueeze(-1)   # [H, W, 1]
        flow = torch.cat([dx, dy], dim=-1).view(1, 1, H, W, 2)  # float32 [1,1,H,W,2]
        ds_flow = downsample_flow(flow, H // resize_flow, W // resize_flow)  # float32
        noises.append(householder(noises[-1], ds_flow))

    all_noises = torch.cat(noises, dim=1)                        # [1, F, C, 60, 90]
    return all_noises.unsqueeze(0).to(dtype=dtype)               # [1, 1, F, C, 60, 90]


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="NoiseControl inference from pre-computed flow")
    p.add_argument("--flows",       required=True,  help="Path to flows_dxdy.npy  (F-1, 2, H_half, W_half)")
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

    # ── 1. Load FlowHouseholder ───────────────────────────────────────────
    print("[1/4] Loading FlowHouseholder...")
    householder = FlowHouseholder(C=16, P=2, seed=42)
    householder.load_state_dict(
        torch.load(FLOW_HOUSEHOLDER_PTH, weights_only=True, map_location="cpu")
    )
    householder.eval().to(device)

    # ── 2. Warp noise from pre-computed flows ─────────────────────────────
    print(f"[2/4] Warping noise from {args.flows} ...")
    flows_np = np.load(args.flows).astype(np.float32)   # (F-1, 2, H_half, W_half)
    print(f"      flows shape: {flows_np.shape}  (expects {NUM_FRAMES-1} flows)")

    correlated = get_noise_from_flow_npy(flows_np, householder, device, dtype)
    # correlated: [1, 1, F, 16, H//8, W//8]

    noise   = mix_new_noise(correlated, alpha=args.alpha)
    latents = downtemp_noise(noise[0].squeeze(0), "blend_norm")  # [13, 16, H', W']
    latents = latents.unsqueeze(0).to(device, dtype=dtype)        # [1, 13, 16, H', W']
    print(f"      latents shape: {latents.shape}")

    del householder, correlated, noise
    torch.cuda.empty_cache()

    # ── 3. Load CogVideoX + LoRA ──────────────────────────────────────────
    print("[3/4] Loading CogVideoX pipeline...")
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload()

    lora_path = Path(args.lora_path)
    if lora_path.exists():
        print(f"      Loading LoRA: {lora_path}")
        pipe.load_lora_weights(str(lora_path))
    else:
        print(f"Warning: LoRA not found at {lora_path} — running base model only.")

    # ── 4. Generate ────────────────────────────────────────────────────────
    print("[4/4] Generating...")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    output = pipe(
        image=load_first_frame(args.first_frame),
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
