import argparse
import inspect
import os
from omegaconf import OmegaConf

import torch
import shutil
from diffusers import AutoencoderKL, DDIMScheduler, LCMScheduler, AutoPipelineForText2Image

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

    for model_idx, (config_key, model_config) in enumerate(list(config.items())):

        motion_modules = model_config.motion_module
        motion_modules = [motion_modules] if isinstance(motion_modules, str) else list(motion_modules)
        for motion_module in motion_modules:
            print('Made it to line 68')
            try:
                # Create the AutoPipelineForText2Image
                pipe = AutoPipelineForText2Image.from_pretrained(model_config.model_id, torch_dtype=torch.float16,
                                                                 variant="fp16")
                pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
                pipe.to("cuda")
                # Load and fuse LCM Lora
                pipe.load_lora_weights(model_config.adapter_id)
                pipe.fuse_lora()

                # Generate image using the provided prompt
                image = pipe(prompt=model_config.prompt, num_inference_steps=model_config.steps,
                             guidance_scale=model_config.guidance_scale)
                samples.append(image)

            except Exception as e:
                print(f"Error processing model {model_idx} with config {config_key}: {e}")

            prompt = "-".join((model_config.prompt.replace("/", "").split(" ")[:10]))
            prompt = re.sub(r'[^\w\s-]', '', prompt)[:16]

            save_videos_grid(image, f"{savedir}/{sample_idx}-{prompt}-{time_str}.gif")
            if args.cloudsave:
                save_videos_grid(image, f"/content/outputs/{time_str}/{sample_idx}-{prompt}-{time_str}.gif")
            print(f"saving original scale outputs to {savedir}/{sample_idx}-{prompt}-{time_str}.gif")
            sample_idx += 1

    samples = torch.cat(samples)
    save_videos_grid(samples, f"{savedir}/combined.gif", n_rows=4)
    if args.cloudsave:
        save_videos_grid(samples, f"/content/latest.gif", n_rows=4)
    OmegaConf.save(config, f"{savedir}/config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ... (rest of the argument parsing remains unchanged)

    args = parser.parse_args()
    main(args)
