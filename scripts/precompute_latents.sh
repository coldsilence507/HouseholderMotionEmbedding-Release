#!/bin/bash
# Step 1: encode RealCam-Vid clips with the CogVideoX VAE / T5 once and cache them under
# $DATA_ROOT/$CACHE_ROOT. Run for both splits: training reads the train cache, and
# validation during training reads the test cache.
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the RealCam-Vid root (contains RealCam-Vid_train.npz)}"
MODEL="${MODEL:-THUDM/CogVideoX-5b-I2V}"
CACHE_ROOT="${CACHE_ROOT:-latent_cache}"
GPU_IDS="${GPU_IDS:-0}"

for DATAFILE in RealCam-Vid_train.npz RealCam-Vid_test.npz; do
  accelerate launch --config_file accelerate_config/experiment.yaml --gpu_ids "$GPU_IDS" \
    training/cogvideox/cogvideox_realcam_precompute.py \
    --precompute \
    --pretrained_model_name_or_path "$MODEL" \
    --data_root "$DATA_ROOT" \
    --train_datafile "$DATAFILE" \
    --cache_root "$CACHE_ROOT" \
    --height_buckets 480 \
    --width_buckets 720 \
    --frame_buckets 49 \
    --max_num_frames 49 \
    --mixed_precision bf16 \
    --enable_slicing \
    --enable_tiling \
    --seed 42 \
    --output_dir ./ckpts/precompute
done
