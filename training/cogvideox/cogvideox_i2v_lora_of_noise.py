# Copyright 2024 The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc, time
import logging
import math, json
import os, pdb
import random
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import diffusers
import torch
import transformers
import wandb
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
    DeepSpeedPlugin,
)
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.optimization import get_scheduler
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_unet_state_dict_to_peft, export_to_video, load_image, load_video
from diffusers.utils.torch_utils import is_compiled_module

from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5EncoderModel

from model.HouseHolderTransform1 import HouseholderTransform, canonicalize_plucker
from model.HouseHolderTransform_flow_no_rev import FlowHouseholder

import rp 
# rp.git_import('CommonSource')
# import rp.git.CommonSource.noise_warp as nw
from torchvision.models.optical_flow import raft_large
from torchvision import transforms

from dataset.i2v_camera_dataset_realcam import I2VDatasetWithResize
from typing_extensions import override

from utils.args import get_args
from utils.text_encoder import compute_prompt_embeddings  # isort:skip
from utils.utils import (
    get_gradient_norm,
    get_optimizer,
    prepare_rotary_positional_embeddings,
    print_memory,
    reset_memory,
    unwrap_model,
    unload_model,
    free_memory
)


import torch.nn.functional as F
from einops import rearrange

LOG_NAME = "CogVideoX_Finetune"
LOG_LEVEL = "INFO"

logger = get_logger(LOG_NAME, LOG_LEVEL)

def downsamp_mean(x, l=13):
    return torch.stack([rp.mean(u) for u in rp.split_into_n_sublists(x, l)])

def normalized_noises(noises):
    #Noises is in TCHW form
    return torch.stack([x / x.std(1, keepdim=True) for x in noises])

def downtemp_noise(noise, noise_downtemp_interp):
    assert noise_downtemp_interp in {'nearest', 'blend', 'blend_norm', 'randn'}, noise_downtemp_interp
    if   noise_downtemp_interp == 'nearest'    : return                  rp.resize_list(noise, 13)
    elif noise_downtemp_interp == 'blend'      : return                   downsamp_mean(noise, 13)
    elif noise_downtemp_interp == 'blend_norm' : return normalized_noises(downsamp_mean(noise, 13))
    elif noise_downtemp_interp == 'randn'      : return torch.randn_like(rp.resize_list(noise, 13)) #Basically no warped noise, just r
    else: assert False, 'impossible'

def blend_noise(noise_background, noise_foreground, alpha):
    """ Variance-preserving blend """
    return (noise_foreground * alpha + noise_background * (1-alpha))/(alpha ** 2 + (1-alpha) ** 2)**.5

def mix_new_noise(noise, alpha):
    return blend_noise(noise, torch.randn_like(noise), alpha)

def downsample_flow(flow, out_h, out_w, in_norm="pixels", mask=None):
    B, N, H, W, _ = flow.shape
    f = flow.permute(0,1,4,2,3).contiguous() # [B,F,2,H,W]
    f_interp = F.interpolate(f, size=(2, out_h, out_w), mode='area')        # [B,2,H',W']
    sx = out_w / float(W)
    sy = out_h / float(H)
    f_interp[:, :, 0] *= sx   # u
    f_interp[:, :, 1] *= sy   # v
    out = f_interp.permute(0,1,3,4,2).contiguous()    # [B,H',W',2]
    return out

def get_noise_from_video_flow(videos, raft_model, householder, device, dtype):
    resize_flow = 4
    rs_videos = F.interpolate(videos, size=(3,videos.shape[-2]//2,videos.shape[-1]//2), mode='area') # resize to 1/2 resolution [1,49,3, 240,360]
    all_noises = []
    for b_idx in range(rs_videos.shape[0]):
        prev_video_frame = rs_videos[b_idx][0]
        # input to raft_model should be in range [-1,1]
        _, H, W = prev_video_frame.shape
        noise = torch.randn((1,1,householder.C,H//resize_flow, W//resize_flow), device=device, dtype=dtype) # [B,1,H,W,C]
        noises = [noise]
        for f_idx, video_frame in enumerate(rs_videos[b_idx][1:]):
            dx, dy = raft_model(prev_video_frame.unsqueeze(0), video_frame.unsqueeze(0))[-1][0].unsqueeze(-1)
            prev_video_frame = video_frame
            flow = torch.cat([dx, dy], dim=-1).view(1, 1, H, W, 2) # [1, 1, H, W, 2]
            ds_flow = downsample_flow(flow, out_h=flow.shape[2]//resize_flow, out_w=flow.shape[3]//resize_flow)
            transformed_noise = householder(noises[-1], ds_flow)
            noises.append(transformed_noise)

        noises = torch.cat(noises, dim=1)
        noises = mix_new_noise(noises, alpha=random.random())
        noises = downtemp_noise(noises[0], "blend_norm")
        all_noises.append(noises)
    all_noises = torch.stack(all_noises).to(dtype=dtype, non_blocking=True)
    return all_noises


def log_validation(
    accelerator: Accelerator,
    pipe: CogVideoXImageToVideoPipeline,
    args: Dict[str, Any],
    pipeline_args: Dict[str, Any],
    video_gt,
    trainning_epoch,
    trainning_step,
    is_final_validation: bool = False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_videos} videos with prompt: {pipeline_args['prompt']}."
    )

    pipe = pipe.to(accelerator.device)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    videos = []
    for _ in range(args.num_validation_videos):
        video = pipe(**pipeline_args, generator=generator, output_type="np").frames[0]
        videos.append(video)
    
    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "wandb":
            video_filenames = []
            for i, video in enumerate(videos):
                prompt = (
                    pipeline_args["prompt"][:25]
                    .replace(" ", "_")
                    .replace(" ", "_")
                    .replace("'", "_")
                    .replace('"', "_")
                    .replace("/", "_")
                )
                filename = os.path.join(args.output_dir, f"ep{trainning_epoch:05d}_st{trainning_step:05d}_{phase_name}_video_{i}_{prompt}.mp4")
                export_to_video(video, filename, fps=8)
                video_filenames.append(filename)
                
            if is_final_validation:
                
                gt_dir = os.path.join(args.output_dir, "gt_video")
                os.makedirs(gt_dir, exist_ok=True)
                gt_filename = os.path.join(gt_dir, f"{phase_name}_gt_video_{i}_{prompt}.mp4")
                
                video_gt = video_gt.unsqueeze(0).permute(0,2,1,3,4)
                video_gt_np = pipe.video_processor.postprocess_video(video=video_gt, output_type="np")
                export_to_video(video_gt_np[0], gt_filename, fps=8)

                tracker.log(
                    {
                        phase_name: [
                            wandb.Video(filename, caption=f"{i}: {pipeline_args['prompt']}")
                            for i, filename in enumerate(video_filenames)
                        ],
                        "gt_video": [
                            wandb.Video(gt_filename, caption="ground_truth")
                        ]
                    }
                )
            else: 
                tracker.log(
                    {
                        phase_name: [
                            wandb.Video(filename, caption=f"{i}: {pipeline_args['prompt']}")
                            for i, filename in enumerate(video_filenames)
                        ]
                    }
                )

    return videos

def run_validation(
    args: Dict[str, Any],
    accelerator: Accelerator,
    transformer,
    householder_transform: HouseholderTransform,
    raft_model,
    scheduler,
    model_config: Dict[str, Any],
    weight_dtype: torch.dtype,
    trainning_epoch,
    trainning_step,
) -> None:
    accelerator.print("===== Memory before validation =====")
    print_memory(accelerator.device)
    torch.cuda.synchronize(accelerator.device)

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=unwrap_model(accelerator, transformer),
        # scheduler=scheduler,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )

    if args.enable_slicing:
        pipe.vae.enable_slicing()
    if args.enable_tiling:
        pipe.vae.enable_tiling()
    if args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()

    val_dataset_init_kwargs = {
        "data_root": Path(args.data_root),
        "cache_root": Path(f"{args.data_root}/{args.cache_root}"),
        "metadata_path": Path(f"{args.data_root}/{args.val_datafile}"),
        "enable_align_factor": True,
        "precompute": args.precompute,
        "accelerator": accelerator,
        "train_resolution": [49, 480, 720],
        "max_num_frames": args.max_num_frames,
        "height": args.height_buckets[0],
        "width": args.width_buckets[0],
        "keep_aspect_ratio": True,
        "use_precompute_video_latents": True
    }
    val_dataset = I2VDatasetWithResize(**val_dataset_init_kwargs)
    
    count=0
    
    progress_bar = tqdm(
        range(0, 10),
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    
    for val_data in val_dataset:
        #image here is the image path
        validation_image = val_data['image_path']
        validation_video = val_data['video_path']
        validation_prompt = val_data['prompt']
        videos = val_data['video'].to(accelerator.device, non_blocking=True)  # [F,C,H,W]
        # validation_noise = val_data['noise']
        correlated_noise = get_noise_from_video_flow(videos.unsqueeze(0), raft_model, householder_transform, device=accelerator.device, dtype=weight_dtype)
        # validation_noise = correlated_noise[:, ::4]
        validation_noise = correlated_noise

        # plucker_emb = val_data["plucker_embedding"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
        
        # rand_noise = torch.randn(1, 1, 16, 60, 90, device=accelerator.device, dtype=weight_dtype)
        # rand_noise = rand_noise.repeat(1, 49, 1, 1, 1) # [B, F, C, H, W]
        # householder_transform.to(accelerator.device)
        # plucker_emb = plucker_emb.unsqueeze(0)
        # m = plucker_emb[...,:3]
        # d = plucker_emb[...,3:]
        # d_hat, m_can = canonicalize_plucker(d, m)

        # transformed_noise = householder_transform(rand_noise, d_hat, m_can)
        # transformed_noise = householder_transform(rand_noise, plucker_emb) # [B,F,C,H,W]                
        # validation_noise = transformed_noise[:,::4] # [B,13,H,W,6]
        # validation_noise = nw.mix_new_noise(validation_noise, alpha=0.5) # degradation
        # validation_noise = mix_new_noise(validation_noise, alpha=0.5)
        
        
        pipeline_args = {
            "image": load_image(validation_image),
            "prompt": validation_prompt,
            "latents": validation_noise,
            "guidance_scale": args.guidance_scale,
            "use_dynamic_cfg": args.use_dynamic_cfg,
            "height": args.height,
            "width": args.width,
            "max_sequence_length": model_config.max_text_seq_length,
        }
        log_validation(
            pipe=pipe,
            args=args,
            accelerator=accelerator,
            pipeline_args=pipeline_args,
            video_gt=videos,
            trainning_epoch=trainning_epoch,
            trainning_step=trainning_step,
        )

        progress_bar.update(1)
        count+=1
        if count >= 10:
            break

    accelerator.print("===== Memory after validation =====")
    print_memory(accelerator.device)
    reset_memory(accelerator.device)

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(accelerator.device)


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )
    logging_dir = Path(args.output_dir, args.logging_dir)
    
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_process_group_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=args.nccl_timeout))
    zero2_plugin = DeepSpeedPlugin(hf_ds_config=args.deepspeed_config) if args.deepspeed_config is not None else None
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
        deepspeed_plugin=zero2_plugin,
    )
    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
        
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Prepare models and scheduler
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )

    load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
        revision=args.revision,
        variant=args.variant,
    )
    
    if args.ignore_learned_positional_embeddings:
        del transformer.patch_embed.pos_embedding
        transformer.patch_embed.use_learned_positional_embeddings = False
        transformer.config.use_learned_positional_embeddings = False

    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    
    householder_transform = FlowHouseholder(C=16, P=2, seed=args.seed)
    householder_transform.load_state_dict(torch.load("training/cogvideox/model/flow_householder_P2.pth", weights_only=True))

    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()
    
    # raft model for optical flow prediction, however it is not used in current version.
    raft_model = raft_large(pretrained=True, progress=False).to(accelerator.device)
    raft_model.eval()
        
    # We only train the additional adapter LoRA layers
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    householder_transform.requires_grad_(False)
    raft_model.requires_grad_(False)

    VAE_SCALING_FACTOR = vae.config.scaling_factor
    VAE_SCALE_FACTOR_SPATIAL = 2 ** (len(vae.config.block_out_channels) - 1)
    RoPE_BASE_HEIGHT = transformer.config.sample_height * VAE_SCALE_FACTOR_SPATIAL
    RoPE_BASE_WIDTH = transformer.config.sample_width * VAE_SCALE_FACTOR_SPATIAL


    weight_dtype = torch.float32

    if accelerator.state.deepspeed_plugin:
        # DeepSpeed is handling precision, use what's in the DeepSpeed config
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.bfloat16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
    

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    householder_transform.to(accelerator.device)
    raft_model.to(accelerator.device)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    

    # now we will add new LoRA weights to the attention layers
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights=True,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(transformer_lora_config)
    
    
    # def unwrap_model(model):
    #     model = accelerator.unwrap_model(model)
    #     model = model._orig_mod if is_compiled_module(model) else model
    #     return model
    
    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None

            for model in models:
                if isinstance(unwrap_model(accelerator, model), type(unwrap_model(accelerator, transformer))):
                    model = unwrap_model(accelerator, model)
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                else:
                    raise ValueError(f"Unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            CogVideoXImageToVideoPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        transformer_ = None

        # This is a bit of a hack but I don't know any other solution.
        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()

                if isinstance(unwrap_model(accelerator, model), type(unwrap_model(accelerator, transformer))):
                    transformer_ = unwrap_model(accelerator, model)
                else:
                    raise ValueError(f"Unexpected save model: {unwrap_model(accelerator, model).__class__}")
        else:
            transformer_ = CogVideoXTransformer3DModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="transformer"
            )
            transformer_.add_adapter(transformer_lora_config)

        lora_state_dict = CogVideoXImageToVideoPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params([transformer_])

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params([transformer], dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Optimization parameters
    transformer_parameters_with_lr = {
        "params": transformer_lora_parameters,
        "lr": args.learning_rate,
    }
    params_to_optimize = [transformer_parameters_with_lr]
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])

    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    optimizer = get_optimizer(
        params_to_optimize=params_to_optimize,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        beta3=args.beta3,
        epsilon=args.epsilon,
        weight_decay=args.weight_decay,
        prodigy_decouple=args.prodigy_decouple,
        prodigy_use_bias_correction=args.prodigy_use_bias_correction,
        prodigy_safeguard_warmup=args.prodigy_safeguard_warmup,
        use_8bit=args.use_8bit,
        use_4bit=args.use_4bit,
        use_torchao=args.use_torchao,
        use_deepspeed=use_deepspeed_optimizer,
        use_cpu_offload_optimizer=args.use_cpu_offload_optimizer,
        offload_gradients=args.offload_gradients,
    )
    
    dataset_init_kwargs = {
        "data_root": Path(args.data_root),
        "cache_root": Path(f"{args.data_root}/{args.cache_root}"),
        "metadata_path": Path(f"{args.data_root}/{args.train_datafile}"),
        "enable_align_factor": True,
        "precompute": args.precompute,
        # "vae": vae,
        # "tokenizer": tokenizer,
        # "text_encoder": text_encoder,
        # "accelerator": accelerator,
        "train_resolution": [49, 480, 720],
        "max_num_frames": args.max_num_frames,
        "height": args.height_buckets[0],
        "width": args.width_buckets[0],
        "keep_aspect_ratio": True,
    }
    
    if args.video_reshape_mode is None:
        # train_dataset = VideoDatasetWithResizing(**dataset_init_kwargs)
        train_dataset = I2VDatasetWithResize(**dataset_init_kwargs)
    else:
        raise NotImplementedError


    def collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {
            "encoded_videos": [], "encoded_images": [], "prompt_embeddings": [], "images": [], "plucker_embeddings": [], "prompts": [], "videos": []
        }

        for sample in samples:
            encoded_video = sample["encoded_video"]
            encoded_image = sample["encoded_image"]
            prompt_embedding = sample["prompt_embedding"]
            image = sample["image"]
            plucker_embedding = sample["plucker_embedding"]
            video = sample["video"]
            prompt = sample["prompt"]

            ret["encoded_videos"].append(encoded_video)
            ret["encoded_images"].append(encoded_image)
            ret["prompt_embeddings"].append(prompt_embedding)
            # comment out this if precompute
            ret["plucker_embeddings"].append(plucker_embedding)
            ret["images"].append(image)
            ret["videos"].append(video)
            ret["prompts"].append(prompt)

        ret["encoded_videos"] = torch.stack(ret["encoded_videos"])
        ret["encoded_images"] = torch.stack(ret["encoded_images"])
        ret["prompt_embeddings"] = torch.stack(ret["prompt_embeddings"])
        ret["images"] = torch.stack(ret["images"])
        # comment out this if precompute
        ret["plucker_embeddings"] = torch.stack(ret["plucker_embeddings"])
        ret["videos"] = torch.stack(ret["videos"])

        return ret
    
    # Precompute latent for video and prompt embedding
    if args.precompute:
        logger.info("Precomputing latent for video and prompt embedding ...")
        tmp_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            # collate_fn=collate_fn,
            batch_size=1,
            num_workers=0,
            pin_memory=args.pin_memory,
            shuffle=True
        )
        tmp_data_loader = accelerator.prepare_data_loader(tmp_data_loader)
        for _ in tqdm(tmp_data_loader):
            ...
        accelerator.wait_for_everyone()
        logger.info("Precomputing latent for video and prompt embedding ... Done")
        return

    unload_model(vae)
    unload_model(text_encoder)
    free_memory()
            
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
        # sampler=BucketSampler(train_dataset, batch_size=args.train_batch_size, shuffle=True),
        num_workers=args.dataloader_num_workers,
        pin_memory=args.pin_memory,
        shuffle=True,
    )
    
    
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
        
    if args.use_cpu_offload_optimizer:
        lr_scheduler = None
        accelerator.print(
            "CPU Offload Optimizer cannot be used with DeepSpeed or builtin PyTorch LR Schedulers. If "
            "you are training with those settings, they will be ignored."
        )
    else:
        if use_deepspeed_scheduler:
            from accelerate.utils import DummyScheduler

            lr_scheduler = DummyScheduler(
                name=args.lr_scheduler,
                optimizer=optimizer,
                total_num_steps=args.max_train_steps * accelerator.num_processes,
                num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            )
        else:
            lr_scheduler = get_scheduler(
                args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                num_training_steps=args.max_train_steps * accelerator.num_processes,
                num_cycles=args.lr_num_cycles,
                power=args.lr_power,
            )
            
    # Prepare everything with our `accelerator`.
    # remove householder_transform if using deepspeed
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
        tracker_name = args.tracker_name or "cogvideox-lora"
        accelerator.init_trackers(tracker_name, config=vars(args))

        accelerator.print("===== Memory before training =====")
        reset_memory(accelerator.device)
        print_memory(accelerator.device)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    accelerator.print("***** Running training *****")
    accelerator.print(f"  Num trainable parameters = {num_trainable_parameters}")
    accelerator.print(f"  Num examples = {len(train_dataset)}")
    accelerator.print(f"  Num batches each epoch = {len(train_dataloader)}")
    accelerator.print(f"  Num epochs = {args.num_train_epochs}")
    accelerator.print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    accelerator.print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    accelerator.print(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    accelerator.print(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if not args.resume_from_checkpoint:
        initial_global_step = 0
    else:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # For DeepSpeed training
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

    if args.load_tensors:
        del vae, text_encoder
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(accelerator.device)

    alphas_cumprod = scheduler.alphas_cumprod.to(accelerator.device, dtype=torch.float32)

    # if accelerator.is_main_process:
    #     run_validation(args, accelerator, transformer, householder_transform, raft_model, scheduler, model_config, weight_dtype, 0, 0)
    
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            logs = {}
            
            with accelerator.accumulate(models_to_accumulate):
                # images = batch["images"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                videos = batch["videos"].to(accelerator.device, non_blocking=True)
                # prompts = batch["prompt"]
                video_latents = batch["encoded_videos"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                image_latents = batch["encoded_images"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                prompt_embedding = batch["prompt_embeddings"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                plucker_emb = batch["plucker_embeddings"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                
                image_latent_dist = DiagonalGaussianDistribution(image_latents)
                latent_dist = DiagonalGaussianDistribution(video_latents)
                
                image_latents = image_latent_dist.sample() * VAE_SCALING_FACTOR
                image_latents = image_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                image_latents = image_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)
                
                video_latents = latent_dist.sample() * VAE_SCALING_FACTOR
                video_latents = video_latents.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
                video_latents = video_latents.to(memory_format=torch.contiguous_format, dtype=weight_dtype)

                padding_shape = (video_latents.shape[0], video_latents.shape[1] - 1, *video_latents.shape[2:])
                latent_padding = image_latents.new_zeros(padding_shape)
                image_latents = torch.cat([image_latents, latent_padding], dim=1)

                if random.random() < args.noised_image_dropout:
                    image_latents = torch.zeros_like(image_latents)

                batch_size, latent_frames, num_channels, height, width = video_latents.shape
                prompt_embedding = prompt_embedding.to(dtype=weight_dtype)
                # num_frames = args.frame_buckets[0]

                # print(f"{video_latents.shape=}, {image_latents.shape=}, {prompt_embedding.shape=}, {plucker_emb.shape=}")
                                
                #initialize temporal correlated noise per video:
                # noises = torch.randn((batch_size, 1, num_channels, height, width), device=accelerator.device)
                # noises = noises.repeat(1, num_frames, 1, 1, 1) # [B, F, C, H, W]
                
                # householder_transform.to(accelerator.device)
                # m = plucker_emb[...,:3]
                # d = plucker_emb[...,3:]
                # d_hat, m_can = canonicalize_plucker(d, m)
                correlated_noise = get_noise_from_video_flow(videos, raft_model, householder_transform, device=accelerator.device, dtype=weight_dtype)
                # downtemp_noise = correlated_noise[:, ::4]
                # transformed_noise = householder_transform(noises, plucker_emb) # [B,F,C,H,W]                
                # transformed_noise = householder_transform(noises, d_hat, m_can)
                # downtemp_noise = transformed_noise[:,::4] # [B,13,H,W,6]

                # noise = mix_new_noise(correlated_noise, alpha=random.random()) # degradation [0,1]   
       
                noise = correlated_noise

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (batch_size,),
                    dtype=torch.int64,
                    device=accelerator.device,
                )

                # Prepare rotary embeds
                image_rotary_emb = (
                    prepare_rotary_positional_embeddings(
                        height=height * VAE_SCALE_FACTOR_SPATIAL,
                        width=width * VAE_SCALE_FACTOR_SPATIAL,
                        num_frames=latent_frames,
                        vae_scale_factor_spatial=VAE_SCALE_FACTOR_SPATIAL,
                        patch_size=model_config.patch_size,
                        patch_size_t=model_config.patch_size_t if hasattr(model_config, "patch_size_t") else None,
                        attention_head_dim=model_config.attention_head_dim,
                        device=accelerator.device,
                        base_height=RoPE_BASE_HEIGHT,
                        base_width=RoPE_BASE_WIDTH,
                    )
                    if model_config.use_rotary_positional_embeddings
                    else None
                )
                
                noisy_video_latents = scheduler.add_noise(video_latents, noise, timesteps)
                noisy_model_input = torch.cat([noisy_video_latents, image_latents], dim=2)
                
                ofs_embed_dim = model_config.ofs_embed_dim if hasattr(model_config, "ofs_embed_dim") else None,
                ofs_emb = None if ofs_embed_dim is None else noisy_model_input.new_full((1,), fill_value=2.0)
                
                # tt = time.time()

                model_output = transformer(
                    hidden_states=noisy_model_input,
                    encoder_hidden_states=prompt_embedding,
                    timestep=timesteps,
                    ofs=ofs_emb,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0]
                
                model_pred = scheduler.get_velocity(model_output, noisy_video_latents, timesteps)

                weights = 1 / (1 - alphas_cumprod[timesteps])
                while len(weights.shape) < len(model_pred.shape):
                    weights = weights.unsqueeze(-1)

                target = video_latents

                loss = torch.mean(
                    (weights * (model_pred - target) ** 2).reshape(batch_size, -1),
                    dim=1,
                )
                loss = loss.mean()
                
                accelerator.backward(loss)
                if accelerator.sync_gradients and accelerator.distributed_type != DistributedType.DEEPSPEED:
                    gradient_norm_before_clip = get_gradient_norm(transformer.parameters())
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                    gradient_norm_after_clip = get_gradient_norm(transformer.parameters())
                    logs.update(
                        {
                            "gradient_norm_before_clip": gradient_norm_before_clip,
                            "gradient_norm_after_clip": gradient_norm_after_clip,
                        }
                    )

                if accelerator.state.deepspeed_plugin is None:
                    optimizer.step()
                    optimizer.zero_grad()

                if not args.use_cpu_offload_optimizer:
                    lr_scheduler.step()
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Checkpointing
                if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                should_run_validation = args.val_data_root is not None and (
                    args.validation_steps is not None and global_step % args.validation_steps == 0
                )
                if should_run_validation:
                    run_validation(args, accelerator, transformer, householder_transform, raft_model, scheduler, model_config, weight_dtype, epoch, global_step)

            last_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else args.learning_rate
            logs.update(
                {
                    "loss": loss.detach().item(),
                    "lr": last_lr,
                }
            )
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            should_run_validation = args.val_data_root is not None and (
                args.validation_epochs is not None and (epoch + 1) % args.validation_epochs == 0
            )
            if should_run_validation:
                run_validation(args, accelerator, transformer, householder_transform, raft_model, scheduler, model_config, weight_dtype, epoch, global_step)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        transformer = unwrap_model(accelerator, transformer)
        dtype = (
            torch.float16
            if args.mixed_precision == "fp16"
            else torch.bfloat16
            if args.mixed_precision == "bf16"
            else torch.float32
        )
        transformer = transformer.to(dtype)
        transformer_lora_layers = get_peft_model_state_dict(transformer)

        CogVideoXImageToVideoPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
        )

        # Cleanup trained models to save memory
        if args.load_tensors:
            del transformer
        else:
            del transformer, text_encoder, vae

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(accelerator.device)

        accelerator.print("===== Memory before testing =====")
        print_memory(accelerator.device)
        reset_memory(accelerator.device)

        # Final test inference
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)

        if args.enable_slicing:
            pipe.vae.enable_slicing()
        if args.enable_tiling:
            pipe.vae.enable_tiling()
        if args.enable_model_cpu_offload:
            pipe.enable_model_cpu_offload()

        # Load LoRA weights
        lora_scaling = args.lora_alpha / args.rank
        pipe.load_lora_weights(args.output_dir, adapter_name="cogvideox-lora")
        pipe.set_adapters(["cogvideox-lora"], [lora_scaling])

        # Run inference
        validation_outputs = []
        if args.val_data_root and args.num_validation_videos > 0:
            val_dataset_init_kwargs = {
                    "data_root": Path(args.data_root),
                    "cache_root": Path(f"{args.data_root}/{args.cache_root}"),
                    "metadata_path": Path(f"{args.data_root}/{args.val_datafile}"),
                    "enable_align_factor": True,
                    "precompute": args.precompute,
                    "accelerator": accelerator,
                    "train_resolution": [49, 480, 720],
                    "max_num_frames": args.max_num_frames,
                    "height": args.height_buckets[0],
                    "width": args.width_buckets[0],
                    "keep_aspect_ratio": True,
                }
            val_dataset = I2VDatasetWithResize(**val_dataset_init_kwargs)
            
            progress_bar = tqdm(
                range(0, len(val_dataset)),
                desc="Steps",
                # Only show the progress bar once on each machine.
                disable=not accelerator.is_local_main_process,
            )
            
            for val_data in val_dataset:
                #image here is the image path
                validation_image = (val_data['image'] + 1.) / 2.  # to [0,1]
                validation_video = val_data['video_path']
                validation_prompt = val_data['prompt']
                # validation_noise = val_data['noise']
                # plucker_emb = val_data["plucker_embedding"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)

                # rand_noise = torch.randn(1, 1, 16, 60, 90, device=accelerator.device, dtype=weight_dtype)
                # rand_noise = rand_noise.repeat(1, 49, 1, 1, 1) # [B, F, C, H, W]
                # householder_transform.to(accelerator.device)
                # plucker_emb = plucker_emb.unsqueeze(0)
                # m = plucker_emb[...,:3]
                # d = plucker_emb[...,3:]
                # d_hat, m_can = canonicalize_plucker(d, m)

                # transformed_noise = householder_transform(rand_noise, d_hat, m_can)
                # # transformed_noise = householder_transform(rand_noise, plucker_emb) # [B,F,C,H,W]                
                # validation_noise = transformed_noise[:,::4] # [B,13,H,W,6]
                # # validation_noise = nw.mix_new_noise(validation_noise, alpha=0.5) # degradation
                videos = val_data['video'].to(accelerator.device, non_blocking=True)  # [F,C,H,W]
                # validation_noise = val_data['noise']
                correlated_noise = get_noise_from_video_flow(videos.unsqueeze(0), raft_model, householder_transform, device=accelerator.device, dtype=weight_dtype)
                # validation_noise = correlated_noise[:, ::4]
                
                # validation_noise = mix_new_noise(correlated_noise, alpha=0.5)
                validation_noise = correlated_noise
                
                to_pil = transforms.ToPILImage()
                validation_image = to_pil(validation_image)

                pipeline_args = {
                    "image": load_image(validation_image),
                    "prompt": validation_prompt,
                    "latents": validation_noise,
                    "guidance_scale": args.guidance_scale,
                    "use_dynamic_cfg": args.use_dynamic_cfg,
                    "height": args.height,
                    "width": args.width,
                    "max_sequence_length": model_config.max_text_seq_length,
                }

                video = log_validation(
                    accelerator=accelerator,
                    pipe=pipe,
                    args=args,
                    pipeline_args=pipeline_args,
                    video_gt=videos,
                    trainning_epoch=epoch,
                    trainning_step=global_step,
                    is_final_validation=True,
                )
                
                validation_outputs.extend(video)
                progress_bar.update(1)


        accelerator.print("===== Memory after testing =====")
        print_memory(accelerator.device)
        reset_memory(accelerator.device)
        torch.cuda.synchronize(accelerator.device)

    accelerator.end_training()
        
        
if __name__ == "__main__":
    args = get_args()
    main(args)
