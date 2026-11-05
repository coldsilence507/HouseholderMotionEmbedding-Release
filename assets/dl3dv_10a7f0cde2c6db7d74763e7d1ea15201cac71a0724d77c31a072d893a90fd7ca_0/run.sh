#!/usr/bin/env bash
# NoiseControl example: dl3dv_10a7f0cde2c6db7d74763e7d1ea15201cac71a0724d77c31a072d893a90fd7ca_0  (alpha=0.3, seed=42)
# Reference video -> RAFT flow -> Householder-warped noise -> CogVideoX I2V.
# Prompt: A peaceful park with a red-brown pathway, surrounded by trees with green and yellow leaves, sugge...
#
# Usage (from anywhere):  LORA_PATH=/path/to/pytorch_lora_weights.safetensors bash assets/dl3dv_10a7f0cde2c6db7d74763e7d1ea15201cac71a0724d77c31a072d893a90fd7ca_0/run.sh
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
    --alpha       0.3 \
    --num_inference_steps 30 \
    --guidance_scale 6.0 \
    --seed        42 \
    --output      "$OUT_DIR/dl3dv_10a7f0cde2c6db7d74763e7d1ea15201cac71a0724d77c31a072d893a90fd7ca_0.mp4"
