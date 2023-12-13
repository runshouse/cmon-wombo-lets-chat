import argparse
# import datetime
import inspect
import os
from omegaconf import OmegaConf

import torch
import shutil
import diffusers
from diffusers import AutoencoderKL, DDIMScheduler

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import re

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, \
    convert_ldm_vae_checkpoint
from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange, repeat

import csv, pdb, glob
from safetensors import safe_open
import math
from pathlib import Path
import xformers

def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)

    if args.context_length == 0:
        args.context_length = args.L
    if args.context_overlap == -1:
        args.context_overlap = args.context_length // 2

    time_str = "outputs"
    if args.cloudsave:
        savedir = f"/content/drive/MyDrive/AnimateDiff/outputs/{time_str}"
    else:
        savedir = f"{args.outputdir}/{time_str}"
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    inference_config = OmegaConf.load(args.inference_config)

    config = OmegaConf.load(args.config)
    samples = []

    sample_idx = args.scene_number
    
    print('Made it to line 54')
    for model_idx, (config_key, model_config) in enumerate(list(config.items())):

        motion_modules = model_config.motion_module
        motion_modules = [motion_modules] if isinstance(motion_modules, str) else list(motion_modules)
        for motion_module in motion_modules:

            ### >>> create validation pipeline >>> ###
            print("tokenizer")
            tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
            print("text encoder")
            text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
            print("vae")
            vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
            print("unet")
 
            try:
                unet = UNet3DConditionModel.from_pretrained_2d(
                    args.pretrained_model_path,
                    subfolder="unet",
                    unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)
                )
            except Exception as e:
                print("Error while loading U-Net model:", e)
          
            print('Made it to line 68')
            if is_xformers_available(): unet.enable_xformers_memory_efficient_attention()

            print("Using ", args.offload)
            
            pipeline = AnimationPipeline(
                vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
                scan_inversions=not args.disable_inversions, init_image_strength=args.init_strength
            ).to(args.offload)

            # enable memory savings
            pipe.enable_vae_slicing()
            pipe.enable_model_cpu_offload()
            pipe.enable_xformers_memory_efficient_attention()

            
            # 1. unet ckpt
            # 1.1 motion module
            motion_module_state_dict = torch.load(motion_module, map_location=args.offload)
            if "global_step" in motion_module_state_dict: func_args.update(
                {"global_step": motion_module_state_dict["global_step"]})
            missing, unexpected = pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
            assert len(unexpected) == 0

            # 1.2 T2I
            if model_config.path != "":
                if model_config.path.endswith(".ckpt"):
                    state_dict = torch.load(model_config.path)
                    pipeline.unet.load_state_dict(state_dict)

                elif model_config.path.endswith(".safetensors"):
                    state_dict = {}
                    with safe_open(model_config.path, framework="pt", device=args.offload) as f:
                        for key in f.keys():
                            state_dict[key] = f.get_tensor(key)

                    is_lora = all("lora" in k for k in state_dict.keys())
                    if not is_lora:
                        base_state_dict = state_dict
                    else:
                        base_state_dict = {}
                        with safe_open(model_config.base, framework="pt", device=args.offload) as f:
                            for key in f.keys():
                                base_state_dict[key] = f.get_tensor(key)

                                # vae
                    converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_state_dict, pipeline.vae.config)
                    pipeline.vae.load_state_dict(converted_vae_checkpoint)
                    # unet
                    converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_state_dict, pipeline.unet.config)
                    pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
                    # text_model
                    pipeline.text_encoder = convert_ldm_clip_checkpoint(base_state_dict)

                    # import pdb
                    # pdb.set_trace()
                    if is_lora:
                        pipeline = convert_lora(pipeline, state_dict, alpha=model_config.lora_alpha)

            if args.offload == 'cpu':
                pipeline.enable_sequential_cpu_offload()
            else:
                pipeline.to("cuda")

            
            ### <<< create validation pipeline <<< ###

            prompts = model_config.prompt
            n_prompts = list(model_config.n_prompt) * len(prompts) if len(
                model_config.n_prompt) == 1 else model_config.n_prompt
            init_image = model_config.init_image if hasattr(model_config, 'init_image') else None

            random_seeds = model_config.get("seed", [-1])
            random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
            random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds

            config[config_key].random_seed = []
            
            for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):
                print('Made it to line 154')
                # manually set random seed for reproduction
                if random_seed != -1:
                    torch.manual_seed(random_seed)
                else:
                    torch.seed()
                config[config_key].random_seed.append(torch.initial_seed())

                print(f"current seed: {torch.initial_seed()}")
                print(f"sampling {prompt} ...")
                print('Made it to line 164')
                try:
                    sample = pipeline(
                        prompt,
                        init_image=model_config.init_image,
                        negative_prompt=n_prompt,
                        num_inference_steps=model_config.steps,
                        guidance_scale=model_config.guidance_scale,
                        width=args.W,
                        height=args.H,
                        video_length=args.L,
                        temporal_context=args.context_length,
                        strides=args.context_stride + 1,
                        overlap=args.context_overlap,
                        fp16=not args.fp32,
                        init_image_strength=args.init_strength
                    ).videos
                    print('Made it to line 181')
                    samples.append(sample)
                    
                except Exception as e:
                    print(f"Error processing model {model_idx} with config {config_key}: {e}")
                    # Add any specific exception handling here if needed
                
                print('Made it to line 188')
                prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
                prompt = re.sub(r'[^\w\s-]', '', prompt)[:16]

                save_videos_grid(sample, f"{savedir}/{sample_idx}-{prompt}-{time_str}.gif")
                if args.cloudsave:
                    save_videos_grid(sample, f"/content/outputs/{time_str}/{sample_idx}-{prompt}-{time_str}.gif")
                print(f"saving original scale outputs to {savedir}/{sample_idx}-{prompt}-{time_str}.gif")
                print('Made it to line 189')
                sample_idx += 1

    samples = torch.concat(samples)
    save_videos_grid(samples, f"{savedir}/combined.gif", n_rows=4)
    if args.cloudsave:
        save_videos_grid(samples, f"/content/latest.gif", n_rows=4)
    OmegaConf.save(config, f"{savedir}/config.yaml")
    if init_image is not None:
        shutil.copy(init_image, f"{savedir}/init_image.jpg")


if __name__ == "__main__":
    print('attempting to parse arguments 1')
    parser = argparse.ArgumentParser()
    print('first stage arguments parsed')
    parser.add_argument("--pretrained_model_path", type=str, default="models/StableDiffusion", )
    parser.add_argument("--inference_config", type=str, default="configs/inference/inference.yaml")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--cloudsave", type=bool, default=False)
    parser.add_argument("--outputdir", type=str, default='AnimateDiff/outputs/')
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--disable_inversions", action="store_true",
                        help="do not scan for downloaded textual inversions")
    parser.add_argument("--context_length", type=int, default=0,
                        help="temporal transformer context length (0 for same as -L)")
    parser.add_argument("--context_stride", type=int, default=0,
                        help="max stride of motion is 2^context_stride")
    parser.add_argument("--context_overlap", type=int, default=-1,
                        help="overlap between chunks of context (-1 for half of context length)")
    parser.add_argument("--init_strength", type=float, default=0.5, help="sets the strength of influence for the init image")
    parser.add_argument("--scene_number", type=int, default=0, help="Starting scene number")
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)
    parser.add_argument("--offload", type=str, default='cuda')
    print('attempting to parse arguments 2')
    args = parser.parse_args()
    print('reached main function call')
    main(args)
