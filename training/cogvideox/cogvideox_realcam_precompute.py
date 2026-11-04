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
from diffusers.utils import convert_unet_state_dict_to_peft, export_to_video, load_image
from diffusers.utils.torch_utils import is_compiled_module

from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5EncoderModel

from model.HouseHolderTransform import HouseholderTransform

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

LOG_NAME = "CogVideoX_Finetune"
LOG_LEVEL = "INFO"

logger = get_logger(LOG_NAME, LOG_LEVEL)




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
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
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
    # transformer = CogVideoXTransformer3DModel.from_pretrained(
    #     args.pretrained_model_name_or_path,
    #     subfolder="transformer",
    #     torch_dtype=load_dtype,
    #     revision=args.revision,
    #     variant=args.variant,
    # )
    
    # if args.ignore_learned_positional_embeddings:
    #     del transformer.patch_embed.pos_embedding
    #     transformer.patch_embed.use_learned_positional_embeddings = False
    #     transformer.config.use_learned_positional_embeddings = False

    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    
    householder_transform = HouseholderTransform(C=16, k_reflections=4, learnable=False, seed=42)
    # householder_transform.load_state_dict(torch.load("/mnt/afse2/taojun/Projects/NoiseControl/training/cogvideox/model/householder_C16_k4.pth", weights_only=True))
    
    # scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()
    
    # raft model for optical flow prediction, however it is not used in current version.
    # raft_model = raft_large(pretrained=True, progress=False).to(accelerator.device)
    # raft_model.eval()
        
    # We only train the additional adapter LoRA layers
    text_encoder.requires_grad_(False)
    # transformer.requires_grad_(False)
    vae.requires_grad_(False)
    householder_transform.requires_grad_(False)
    # raft_model.requires_grad_(False)

    # VAE_SCALING_FACTOR = vae.config.scaling_factor
    # VAE_SCALE_FACTOR_SPATIAL = 2 ** (len(vae.config.block_out_channels) - 1)
    # RoPE_BASE_HEIGHT = transformer.config.sample_height * VAE_SCALE_FACTOR_SPATIAL
    # RoPE_BASE_WIDTH = transformer.config.sample_width * VAE_SCALE_FACTOR_SPATIAL
    
    # print(f"{VAE_SCALE_FACTOR_SPATIAL=}")
    # print(f"{transformer.config.sample_height=}")
    # print(f"{transformer.config.sample_width=}")

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
    # transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    householder_transform.to(accelerator.device, dtype=weight_dtype)
    # raft_model.to(accelerator.device)
    
    
    # def unwrap_model(model):
    #     model = accelerator.unwrap_model(model)
    #     model = model._orig_mod if is_compiled_module(model) else model
    #     return model
    
    # # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    # def save_model_hook(models, weights, output_dir):
    #     if accelerator.is_main_process:
    #         transformer_lora_layers_to_save = None

    #         for model in models:
    #             if isinstance(unwrap_model(accelerator, model), type(unwrap_model(accelerator, transformer))):
    #                 model = unwrap_model(accelerator, model)
    #                 transformer_lora_layers_to_save = get_peft_model_state_dict(model)
    #             else:
    #                 raise ValueError(f"Unexpected save model: {model.__class__}")

    #             # make sure to pop weight so that corresponding model is not saved again
    #             if weights:
    #                 weights.pop()

    #         CogVideoXImageToVideoPipeline.save_lora_weights(
    #             output_dir,
    #             transformer_lora_layers=transformer_lora_layers_to_save,
    #         )

    # def load_model_hook(models, input_dir):
    #     transformer_ = None

    #     # This is a bit of a hack but I don't know any other solution.
    #     if not accelerator.distributed_type == DistributedType.DEEPSPEED:
    #         while len(models) > 0:
    #             model = models.pop()

    #             if isinstance(unwrap_model(accelerator, model), type(unwrap_model(accelerator, transformer))):
    #                 transformer_ = unwrap_model(accelerator, model)
    #             else:
    #                 raise ValueError(f"Unexpected save model: {unwrap_model(accelerator, model).__class__}")
    #     else:
    #         transformer_ = CogVideoXTransformer3DModel.from_pretrained(
    #             args.pretrained_model_name_or_path, subfolder="transformer"
    #         )
    #         transformer_.add_adapter(transformer_lora_config)

    #     lora_state_dict = CogVideoXImageToVideoPipeline.lora_state_dict(input_dir)

    #     transformer_state_dict = {
    #         f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
    #     }
    #     transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
    #     incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
    #     if incompatible_keys is not None:
    #         # check only for unexpected keys
    #         unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
    #         if unexpected_keys:
    #             logger.warning(
    #                 f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
    #                 f" {unexpected_keys}. "
    #             )

    #     # Make sure the trainable params are in float32. This is again needed since the base models
    #     # are in `weight_dtype`. More details:
    #     # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
    #     if args.mixed_precision == "fp16":
    #         # only upcast trainable parameters (LoRA) into fp32
    #         cast_training_params([transformer_])

    # accelerator.register_save_state_pre_hook(save_model_hook)
    # accelerator.register_load_state_pre_hook(load_model_hook)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    # if args.mixed_precision == "fp16":
    #     # only upcast trainable parameters (LoRA) into fp32
    #     cast_training_params([transformer], dtype=torch.float32)

    
    dataset_init_kwargs = {
        "data_root": Path(args.data_root),
        "cache_root": Path(f"{args.data_root}/{args.cache_root}"),
        "metadata_path": Path(f"{args.data_root}/{args.train_datafile}"),
        "enable_align_factor": True,
        "precompute": args.precompute,
        "vae": vae,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "accelerator": accelerator,
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
            "encoded_videos": [], "prompt_embeddings": [], "images": [], "plucker_embeddings": [], "prompts": [], "videos": []
        }

        for sample in samples:
            encoded_video = sample["encoded_video"]
            prompt_embedding = sample["prompt_embedding"]
            image = sample["image"]
            plucker_embedding = sample["plucker_embedding"]
            video = sample["video"]
            prompt = sample["prompt"]

            ret["encoded_videos"].append(encoded_video)
            ret["prompt_embeddings"].append(prompt_embedding)
            # comment out this if precompute
            # ret["plucker_embeddings"].append(plucker_embedding)
            ret["images"].append(image)
            ret["videos"].append(video)
            ret["prompts"].append(prompt)

        ret["encoded_videos"] = torch.stack(ret["encoded_videos"])
        ret["prompt_embeddings"] = torch.stack(ret["prompt_embeddings"])
        ret["images"] = torch.stack(ret["images"])
        # comment out this if precompute
        #ret["plucker_embeddings"] = torch.stack(ret["plucker_embeddings"])
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
            pin_memory=args.pin_memory
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
                
    assert False, "passed all conditions."
        
        
if __name__ == "__main__":
    args = get_args()
    main(args)
