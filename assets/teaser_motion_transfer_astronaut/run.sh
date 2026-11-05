#!/usr/bin/env bash
# NoiseControl example: teaser_motion_transfer_astronaut  (alpha=0.8, seed=42)
# Motion transfer: DAVIS 'lucia' walking motion -> astronaut.
# Prompt: A astronaut is walking on the moon
#
# Usage (from anywhere):  LORA_PATH=/path/to/pytorch_lora_weights.safetensors bash assets/teaser_motion_transfer_astronaut/run.sh
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
    --alpha       0.8 \
    --num_inference_steps 30 \
    --guidance_scale 6.0 \
    --seed        42 \
    --output      "$OUT_DIR/teaser_motion_transfer_astronaut.mp4"
