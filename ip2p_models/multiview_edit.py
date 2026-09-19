from diffusers import (
    DDIMScheduler,
    AutoencoderKL,
)
from transformers import (
    CLIPTextModel, 
    CLIPTokenizer
)
from models.ip2p_pipeline import InstructPix2PixPipeline
from models.ip2p_unet import UNet3DConditionModel
import configargparse as argparse
import torch
import torchvision
import numpy as np
from PIL import Image, ImageOps
import torch.nn.functional as F

from einops import rearrange; 
from PIL import Image

from einops import rearrange
from tqdm import tqdm
import math
import os

from pytorch_lightning import seed_everything
seed_everything(20211202)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16

DDIM_SOURCE = "CompVis/stable-diffusion-v1-4"
IP2P_SOURCE = "timbrooks/instruct-pix2pix"
tokenizer = CLIPTokenizer.from_pretrained(IP2P_SOURCE, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(IP2P_SOURCE, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(IP2P_SOURCE, subfolder="vae")
unet = UNet3DConditionModel.from_pretrained_2d(IP2P_SOURCE, subfolder="unet")

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)

vae = vae.to(device, dtype=torch.float32)
text_encoder = text_encoder.to(device, dtype=torch_dtype)
unet = unet.to(device, dtype=torch_dtype)
        
pipe = InstructPix2PixPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler.from_pretrained(DDIM_SOURCE, subfolder="scheduler"),
    )
unet.eval().set_attention_slice("auto")
vae.eval().enable_slicing()
text_encoder.eval()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dynerf")
    parser.add_argument("--scene_name", "--scene", type=str, default="cook_spinach")
    parser.add_argument("--prompt", type=str, default="What if it was painted by Van Gogh?")
    parser.add_argument("--resize", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--image_guidance_scale", type=float, default=1.5)
    return parser.parse_args()

args = parse_args()
if not 64 <= args.resize <= 512:
    raise ValueError("Use a resize between 64 and 512 for the T4 editor")
image_dir = f'./data/{args.dataset}/time0_{args.scene_name}/'

image_extensions = ('.png', '.jpg')
files = sorted(
    os.path.join(image_dir, name) for name in os.listdir(image_dir)
    if os.path.isfile(os.path.join(image_dir, name)) and name.lower().endswith(image_extensions)
)
sequence_length = len(files)
if not sequence_length:
    raise ValueError(f"No source images found in {image_dir}")

#sequence_length = args.sequence_length
prompt = args.prompt
guidance_scale = args.guidance_scale
image_guidance_scale = args.image_guidance_scale
diffusion_step = args.steps
num_train_timesteps = 1000
latents_type = 'noisy_latents' # 'noise', 'noisy_latents'

tag = prompt.split(' ')[-1].replace('?', '')

print(f'Loaded {len(files)} images from {image_dir}')

images = []
for file in files:
    image = Image.open(file).convert('RGB')
    width, height = image.size
    if args.resize is None:
        args.resize = max(width, height)
    factor = min(1, args.resize / max(width, height))
    width = max(64, int(width * factor) // 64 * 64)
    height = max(64, int(height * factor) // 64 * 64)
    image = image.resize((width, height), resample=Image.Resampling.LANCZOS)
    image = torch.from_numpy(np.array(image) / 255).permute(2, 0, 1).unsqueeze(0).to(torch_dtype).to(device)
    images.append(image)
images = torch.cat(images, dim=0) # (f, c, h, w)

dataset_length, _, H, W = images.shape
RH, RW = H // 8 * 8, W // 8 * 8

images = images.to(device, dtype=torch_dtype)
images = F.interpolate(images, size=(RH, RW), mode='bilinear', align_corners=False) # (f, c, h, w)
images_cond = images.clone().to(device, dtype=torch_dtype) # (f, c, h, w)

with torch.no_grad():
    image_latents = torch.cat([
        pipe.vae.encode(2 * image[None].float() - 1).latent_dist.mode()
        for image in images_cond
    ]).to(torch_dtype)
    latents = image_latents * pipe.vae.config.scaling_factor

latents = rearrange(latents, "(b f) c h w -> b c f h w", f=sequence_length) # (b, 4, f, h//4, w//4)
image_latents = rearrange(image_latents, "(b f) c h w -> b c f h w", f=sequence_length) # (b, 4, f, h//4, w//4)
uncond_image_latents = torch.zeros_like(image_latents)

with torch.no_grad():
    prompt_embeds = pipe._encode_prompt(
        prompt, device=device, num_images_per_prompt=1, do_classifier_free_guidance=True,
    )
text_encoder.to("cpu")

pipe.scheduler.set_timesteps(diffusion_step, device=device)

if latents_type == 'noise':
    latents = torch.randn_like(latents)
elif latents_type == 'noisy_latents':
    noise = torch.randn_like(latents) # (b, 4, f, h//4, w//4)
    latents = pipe.scheduler.add_noise(latents, noise, pipe.scheduler.timesteps[0])  
else:
    raise NotImplementedError
    
for i, t in tqdm(enumerate(pipe.scheduler.timesteps), total=len(pipe.scheduler.timesteps), desc="Inference"):
    with torch.no_grad():
        predictions = []
        for frame in range(sequence_length):
            # Keyframe attention sees the same anchor in every bounded pair.
            slots = [0] if frame == 0 else [0, frame]
            branches = []
            for branch in range(3):
                condition = image_latents[:, :, slots] if branch < 2 else uncond_image_latents[:, :, slots]
                model_input = torch.cat([latents[:, :, slots], condition], dim=1)
                prediction = pipe.unet(
                    model_input, t, prompt_embeds[branch:branch + 1],
                    return_dict=False,
                )[0]
                branches.append(prediction[:, :, -1:])
            text, image, unconditional = branches
            predictions.append(
                unconditional + guidance_scale * (text - image)
                + image_guidance_scale * (image - unconditional)
            )
        noise_pred = torch.cat(predictions, dim=2)
    
    # compute the previous noisy sample x_t -> x_t-1
    latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0] # [b, c, f, h//4, w//4]
    
latents = rearrange(latents, "b c f h w -> (b f) c h w")
latents = latents / pipe.vae.config.scaling_factor
with torch.no_grad():
    video = torch.cat([pipe.vae.decode(latent[None].float()).sample.cpu() for latent in latents])
    
video = (video / 2 + 0.5).clamp(0, 1) # (b*f, 3, h, w) [-1, 1] -> [0, 1]

save_dir = f"./data/{args.dataset}/{args.scene_name}/{prompt.split(' ')[-1].replace('?', '')}"
os.makedirs(save_dir, exist_ok=True)
for i in range(sequence_length):
    filename = f"edited_{prompt.split(' ')[-1].replace('?', '')}_{os.path.basename(files[i])}"
    save_path = os.path.join(save_dir, filename)
    torchvision.utils.save_image(video[i], save_path)


