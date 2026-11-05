#!/usr/bin/env bash
# NoiseControl example: dl3dv_072425e7d9627a32db1ffe46d961361f68b0f009a924d18e2daf17114a242e1a_1  (alpha=0.3, seed=42)
# Reference video -> RAFT flow -> Householder-warped noise -> CogVideoX I2V.
# Prompt: A serene parking lot at what appears to be dawn or dusk, with the sun casting long shadows. the l...
#
# Usage (from anywhere):  LORA_PATH=/path/to/pytorch_lora_weights.safetensors bash assets/dl3dv_072425e7d9627a32db1ffe46d961361f68b0f009a924d18e2daf17114a242e1a_1/run.sh
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
    --output      "$OUT_DIR/dl3dv_072425e7d9627a32db1ffe46d961361f68b0f009a924d18e2daf17114a242e1a_1.mp4"
