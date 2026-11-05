#!/usr/bin/env bash
# NoiseControl example: motion_transfer_cyberpunk_robots  (alpha=0.5, seed=4)
# Motion transfer: DAVIS 'lab-coat' motion -> cyberpunk robots.
# Prompt: Cyberpunk rooftop at night, wet reflective ground, neon city skyline, three humanoid robots stand...
#
# Usage (from anywhere):  LORA_PATH=/path/to/pytorch_lora_weights.safetensors bash assets/motion_transfer_cyberpunk_robots/run.sh
set -euo pipefail
EXAMPLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$EXAMPLE/../.."   # repository root

LORA_PATH=${LORA_PATH:-checkpoints/pytorch_lora_weights.safetensors}
OUT_DIR=${OUT_DIR:-outputs}
mkdir -p "$OUT_DIR"

python infer.py \
    --ref_video   "$EXAMPLE/reference.mp4" \
    --first_frame "$EXAMPLE/first_frame.png" \
    --prompt      "$(cat "$EXAMPLE/prompt.txt")" \
    --lora_path   "$LORA_PATH" \
    --alpha       0.5 \
    --num_inference_steps 30 \
    --guidance_scale 6.0 \
    --seed        4 \
    --output      "$OUT_DIR/motion_transfer_cyberpunk_robots.mp4"
