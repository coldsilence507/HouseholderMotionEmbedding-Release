#!/usr/bin/env bash
# NoiseControl example: dl3dv_031f297d8735ded8424460ea168029293934d4468a1bc3b31cbe45b2aa08d599_2  (alpha=0.3, seed=42)
# Reference video -> RAFT flow -> Householder-warped noise -> CogVideoX I2V.
# Prompt: A macy's store, showcasing a variety of suitcases on metal racks. initially, the focus is on a se...
#
# Usage (from anywhere):  LORA_PATH=/path/to/pytorch_lora_weights.safetensors bash assets/dl3dv_031f297d8735ded8424460ea168029293934d4468a1bc3b31cbe45b2aa08d599_2/run.sh
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
    --output      "$OUT_DIR/dl3dv_031f297d8735ded8424460ea168029293934d4468a1bc3b31cbe45b2aa08d599_2.mp4"
