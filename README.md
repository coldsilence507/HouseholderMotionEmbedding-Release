# [NeurIPS 2026 Spotlight] Householder Motion Embedding: Noise Modulation for Motion-Controllable Video Generation

Motion-controllable image-to-video generation by **transforming the initial noise** with optical flow.

NoiseControl takes a reference motion (a video, or a pre-computed optical-flow field) and turns it
into temporally correlated Gaussian noise. Each frame's noise is obtained from the previous one by a
**per-pixel, flow-conditioned Householder transform** (`FlowHouseholder`). The transform is
orthogonal, so the noise stays exactly Gaussian, and it is palindromic, so reversing the flow inverts it
(`Q(-f) = Q(f)^-1`). The warped noise is the initial latent of
[CogVideoX-5B-I2V](https://huggingface.co/THUDM/CogVideoX-5b-I2V), fine-tuned with a LoRA so that it
follows the motion encoded in the noise.

> [![Project Page](https://img.shields.io/badge/Project-Page-green?logo=googlechrome&logoColor=green)](https://coldsilence507.github.io/HouseholderMotionEmbedding/)
  [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-HouseholderMotionEmbedding-blue)](https://huggingface.co/anujeremylin/HouseholderMotionEmbedding/tree/main)

## Installation

```bash
git clone https://github.com/<user>/NoiseControl.git
cd NoiseControl
conda create -n noisecontrol python=3.12 -y && conda activate noisecontrol
pip install -r requirements.txt
```

Tested with Python 3.12, PyTorch 2.7 and CUDA 12.8. Install the PyTorch build for your CUDA version first if needed.

## Checkpoints

| File | Size | Where |
|---|---|---|
| `training/cogvideox/model/flow_householder_P2.pth` | 1 KB | included in this repo |
| NoiseControl LoRA for CogVideoX-5B-I2V (`pytorch_lora_weights.safetensors`) | 4.2 GB | [Download Here](https://huggingface.co/anujeremylin/HouseholderMotionEmbedding/blob/main/pytorch_lora_weights.safetensors)  |

The `FlowHouseholder` weights are **not trained**. They are a fixed, seeded initialisation that stays
frozen during LoRA training. They were drawn with a CUDA RNG, which is not reproducible on CPU, so
they are shipped as files. `P` is the number of Householder reflection pairs. The released LoRA uses `P=2`.

Pass the LoRA with `--lora_path`. If the file is not found, the scripts print a warning and run the
base model without it. The CogVideoX base model is downloaded from the Hugging Face Hub automatically.

## Inference

Output: a 49-frame, 480×720 video at 8 fps.

**Motion transfer from a reference video.** Optical flow is estimated with RAFT:

```bash
python infer.py \
    --ref_video   /path/to/reference.mp4 \
    --first_frame /path/to/first_frame.png \
    --prompt      "your text prompt here" \
    --output      output.mp4 \
    --alpha       0.3
```

This also writes `output_flow_vis.mp4` next to the output: a colour-coded visualisation of the estimated flow.

**Examples from the paper.** Every inference example shown in the paper and supplementary material has
its own folder under `assets/`:

```
assets/<example>/
  reference.mp4     # reference video (motion source)
  first_frame.png   # input first frame
  prompt.txt        # text prompt
  run.sh            # inference command with the paper settings
```

```bash
export LORA_PATH=/path/to/pytorch_lora_weights.safetensors   # default: checkpoints/pytorch_lora_weights.safetensors
bash assets/teaser_motion_transfer_astronaut/run.sh           # one example -> outputs/teaser_motion_transfer_astronaut.mp4
bash scripts/run_examples.sh 'davis_*'                        # all examples whose folder matches a glob
bash scripts/run_examples.sh                                  # all 35 examples
```

| Folder prefix | Examples | Reference video | `--alpha` |
|---|---|---|---|
| `teaser_` | camera motion (aerial estate), motion transfer (astronaut), local motion (penguin) | rendered camera-motion signal / DAVIS `lucia` / cut-and-drag signal | 0.3 / 0.8 / 0.6 |
| `motion_transfer_` | cyberpunk robots | DAVIS `lab-coat` | 0.5 (seed 4) |
| `davis_` | 8 DAVIS sequences | the original DAVIS clip | 0.3 |
| `dl3dv_` | 6 DL3DV-10K camera trajectories | the original DL3DV-10K clip | 0.3 |
| `mc_bench_` | 17 MC-Bench object / fine-grained drags | cut-and-drag motion signal rendered from the MC-Bench trajectory | 0.6 |

All examples use 30 sampling steps, guidance 6.0, and seed 42 unless the table says otherwise. Sampling is
stochastic across GPUs and library versions, so your outputs can differ from the paper's in detail but not in the motion.
For the paper's MC-Bench numbers, the flow outside the dragged object's mask was also set to zero. `infer.py` does
not do this, so slight background motion can appear in the MC-Bench examples.


**From a pre-computed flow field** (skips RAFT):

```bash
python infer_from_flow.py \
  --flows       path/to/flows_dxdy.npy \
  --first_frame path/to/first_frame.png \
  --prompt      "A brown bear walking on rocks" \
  --lora_path   checkpoints/pytorch_lora_weights.safetensors \
  --output      outputs/bear.mp4
```

`flows_dxdy.npy` has shape `(48, 2, 240, 360)`. It holds frame-to-frame `(dx, dy)` flow in pixels, at
half the output resolution. This is the same layout that
[Go-with-the-Flow](https://github.com/Eyeline-Labs/Go-with-the-Flow)'s `make_warped_noise.py` writes,
so you can drive generation with flow from any source, for example cut-and-drag animations.

| Option | Default | Meaning |
|---|---|---|
| `--alpha` | `0.3` | Noise degradation. `0` uses fully warped noise (strongest motion adherence). `1` uses pure random noise (no motion control). |
| `--num_inference_steps` | `50` | Sampling steps. `30` works well and is faster. |
| `--guidance_scale` | `6.0` | Classifier-free guidance, with dynamic CFG. |
| `--seed` | `42` | Seed of the diffusion sampler. |

Inference uses model CPU offloading and VAE tiling to reduce GPU memory.

## Training

The LoRA is trained on RealCam-Vid. For each training clip,
RAFT flow is computed between consecutive frames, noise is warped with `FlowHouseholder`, and the
result is blended with fresh noise at a random degradation level in `[0, 1)`. This warped noise
replaces the i.i.d. noise in the standard CogVideoX v-prediction objective.

**Data.** `DATA_ROOT` must contain `RealCam-Vid_train.npz` and `RealCam-Vid_test.npz`. These are
metadata lists whose entries have `video_path` (relative to `DATA_ROOT`), `long_caption`,
`camera_extrinsics`, `camera_intrinsics` and `align_factor`. First frames are extracted on first use
to a `first_frames/` folder next to the video folders.

**1. Cache VAE latents and T5 prompt embeddings** (once, for both the train and the test split):

```bash
DATA_ROOT=/path/to/RealCam-Vid GPU_IDS=0 bash scripts/precompute_latents.sh
```

**2. Train the LoRA.** These are the paper settings: 4 GPUs, batch size 2 per GPU, LoRA rank 2048, 30k steps, bf16:

```bash
DATA_ROOT=/path/to/RealCam-Vid GPU_IDS=0,1,2,3 bash scripts/train.sh
```

Run both scripts from the repository root. Checkpoints are written to `./ckpts/noisecontrol-of-noise-lora/checkpoint-*`. Training resumes from
the latest checkpoint automatically. Validation videos are logged to Weights & Biases, which defaults
to offline mode (`WANDB_MODE=offline`). Set `MODEL=/local/path/CogVideoX-5b-I2V` to use a local copy
of the base model.

## Repository layout

```
infer.py                      # reference video -> RAFT flow -> warped noise -> video
infer_from_flow.py            # flows_dxdy.npy  ->               warped noise -> video
scripts/
  precompute_latents.sh       # cache VAE latents + prompt embeddings
  train.sh                    # LoRA training launcher (accelerate)
accelerate_config/experiment.yaml
training/cogvideox/
  cogvideox_i2v_lora_of_noise.py   # LoRA training script
  cogvideox_realcam_precompute.py  # latent / prompt-embedding cache builder
  model/
    HouseHolderTransform_flow.py         # FlowHouseholder: flow-conditioned palindromic Householder transform
    HouseHolderTransform_flow_no_rev.py  # FlowHouseholder variant used by the training script
    HouseHolderTransform.py, HouseHolderTransform1.py  # Householder modules imported by the training scripts
    flow_householder_P*.pth              # frozen FlowHouseholder weights
  dataset/                    # RealCam-Vid loader
  utils/                      # args, optimizers, text encoding, RoPE helpers
```

## Acknowledgements

This code builds on [CogVideoX](https://github.com/THUDM/CogVideo) and its diffusers LoRA fine-tuning
scripts, [Go-with-the-Flow](https://github.com/Eyeline-Labs/Go-with-the-Flow) (noise degradation
and temporal down-sampling), [RAFT](https://github.com/princeton-vl/RAFT) via torchvision, and the
RealCam-Vid dataset. The example inputs come from [DAVIS](https://davischallenge.org/),
[DL3DV-10K](https://github.com/DL3DV-10K/Dataset) and MC-Bench, and remain under their original licenses.

## Citation

```bibtex
@article{lin2026hme,
  title     = {Householder Motion Embedding: Noise Modulation for
               Motion-Controllable Video Generation},
  author    = {Lin, Tao Jun and Ni, Jinhong and Zha, Ruyi and Wang, Yifu
               and Song, Xibin and Ji, Pan and Li, Hongdong},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE). The CogVideoX-5B-I2V base weights are distributed under their own
[license](https://huggingface.co/THUDM/CogVideoX-5b-I2V/blob/main/LICENSE).
