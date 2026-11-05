#!/usr/bin/env bash
# NoiseControl example: davis_goat  (alpha=0.3, seed=42)
# Reference video -> RAFT flow -> Householder-warped noise -> CogVideoX I2V.
# Prompt: a goat standing on top of a rocky hill
#
# Usage (from anywhere):  LORA_PATH=/path/to/pytorch_lora_weights.safetensors bash assets/davis_goat/run.sh
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
    --output      "$OUT_DIR/davis_goat.mp4"
