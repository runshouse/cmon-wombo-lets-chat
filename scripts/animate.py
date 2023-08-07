import argparse
import os
import re
import torch
import shutil
from tqdm.auto import tqdm
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint
from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora
from diffusers.utils.import_utils import is_xformers_available
from pathlib import Path
from omegaconf import OmegaConf
from transformers import CLIPTextModel, CLIPTokenizer
from animatediff.models.unet import UNet3DConditionModel
from diffusers import AutoencoderKL, DDIMScheduler

def main(args):
    inference_config = OmegaConf.load(args.inference_config)
    config = OmegaConf.load(args.config)
    output_dir = "/content/drive/MyDrive/AnimateDiff/outputs/" if args.cloudsave else args.outputdir
    os.makedirs(output_dir, exist_ok=True)
    
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
    unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet",
                                                   unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))
    if is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()

    for model_key, model_config in config.items():
        for motion_module in model_config.motion_module:
            pipeline = create_animation_pipeline(text_encoder, tokenizer, unet, inference_config, motion_module)
            sample_idx = args.scene_number
            for prompt, n_prompt, random_seed in zip(model_config.prompt, model_config.n_prompt, model_config.get("seed", [-1])):
                torch.manual_seed(random_seed) if random_seed != -1 else torch.seed()
                sample = generate_sample(pipeline, prompt, n_prompt, model_config.steps, model_config.guidance_scale, args)
                save_sample(sample, prompt, sample_idx, output_dir, args.cloudsave)
                sample_idx += 1

def create_animation_pipeline(text_encoder, tokenizer, unet, inference_config, motion_module):
    pipeline = AnimationPipeline(
        vae=AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae"),
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
        scan_inversions=not args.disable_inversions,
        init_image_strength=args.init_strength
    ).to("cuda")
    motion_module_state_dict = torch.load(motion_module, map_location="cpu")
    missing, unexpected = pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
    return pipeline

def generate_sample(pipeline, prompt, n_prompt, steps, guidance_scale, args):
    return pipeline(
        prompt,
        init_image=None,
        negative_prompt=n_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=args.W,
        height=args.H,
        video_length=args.L,
        temporal_context=args.context_length,
        strides=args.context_stride + 1,
        overlap=args.context_overlap,
        fp16=not args.fp32,
        init_image_strength=args.init_strength
    ).videos

def save_sample(sample, prompt, sample_idx, output_dir, cloudsave):
    prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
    prompt = re.sub(r'[^\w\s-]', '', prompt)[:16]
    file_path = f"{output_dir}/{sample_idx}-{prompt}-outputs.gif"
    save_videos_grid(sample, file_path)
    if cloudsave:
        shutil.copy(file_path, f"/content/outputs/outputs/{sample_idx}-{prompt}-outputs.gif")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ... (rest of argument parsing)
    args = parser.parse_args()
    main(args)
