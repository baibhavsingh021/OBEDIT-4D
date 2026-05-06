#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# ============================================================
# MODIFIED: SDS → VSD (Variational Score Distillation)
# Same VSD logic as fully_edit_sds.py, adapted for the
# refinement stage (sequence_length=1, resize=128, 800 iters,
# command-line prompt/guidance, auto-detect latest checkpoint).
#
# VSD gradient:
#   grad = w * (ε_pretrained(z_t | cI,cT) − ε_phi(z_t | cI,cT))
# LoRA update:
#   loss_phi = MSE(ε_phi_cfg(z_t), ε_noise)
# ============================================================
import sys
sys.stdout.isatty = lambda: False

import numpy as np
import random
import os
import math
import torch
import torch.nn as nn
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, ModelHiddenParams, OptimizationParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
import lpips
from utils.scene_utils import render_training_image
from time import time
import copy

to8b = lambda x: (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from diffusers import DDIMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from ip2p_models.models.ip2p_pipeline import InstructPix2PixPipeline
from ip2p_models.models.ip2p_unet import UNet3DConditionModel
import configargparse as argparse
import torch
import torchvision
import numpy as np
from PIL import Image, ImageOps
import torch.nn.functional as F
from PIL import Image
from einops import rearrange
from tqdm import tqdm
import math
import os
from pytorch_lightning import seed_everything


# ============================================================
# VSD LoRA Components  (identical to fully_edit_sds.py)
# ============================================================

class LoRALinear(nn.Module):
    """
    LoRA adapter for nn.Linear.
    Base weights: frozen.  lora_A, lora_B: trained in float32.
    Output = base(x) + scale * (x · A^T · B^T)  [cast to input dtype]
    """
    def __init__(self, linear: nn.Linear, rank: int = 4, alpha: float = 4.0):
        super().__init__()
        self.linear = linear
        self.linear.requires_grad_(False)
        in_f, out_f = linear.in_features, linear.out_features
        device = linear.weight.device          # inherit device from the frozen linear
        self.lora_A = nn.Parameter(torch.empty(rank, in_f, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.normal_(self.lora_B, std=1e-4)   # small non-zero → instant diversity
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        x32  = x.float()
        lora = (x32 @ self.lora_A.T @ self.lora_B.T) * self.scale
        return base + lora.to(base.dtype)


def inject_lora_into_unet(unet: nn.Module, rank: int = 4, alpha: float = 4.0):
    """
    Inject LoRA into Q and V projections of all attention modules.
    Returns list of trainable parameters (lora_A and lora_B tensors).
    """
    lora_params = []
    n_replaced  = 0
    for module in unet.modules():
        for attr in ('to_q', 'to_k', 'to_v'):
            layer = getattr(module, attr, None)
            if isinstance(layer, nn.Linear):
                lora = LoRALinear(layer, rank=rank, alpha=alpha)
                setattr(module, attr, lora)
                lora_params += [lora.lora_A, lora.lora_B]
                n_replaced  += 1
    print(f"[VSD] LoRA injected into {n_replaced} attention projections "
          f"(rank={rank}, alpha={alpha}). "
          f"Trainable params: {sum(p.numel() for p in lora_params):,}")
    return lora_params


# ============================================================
# Memory-efficient VAE helpers
# (batch encoding + CPU offload to stay within VRAM budget)
# ============================================================

def encode_1(ip2p, input, encode_batch_size=1):
    """Encode rendered images to latents with per-sample CPU offload."""
    latents_list = []
    for i in range(0, input.shape[0], encode_batch_size):
        batch  = input[i:i + encode_batch_size]
        latent = ip2p.vae.encode(2 * batch - 1).latent_dist.sample() * 0.18215
        latents_list.append(latent.cpu())
        del batch, latent
        torch.cuda.empty_cache()
    return torch.cat(latents_list, dim=0).to(device=input.device)


def encode_2(ip2p, input, encode_batch_size=1):
    """Encode conditioning images to latents with per-sample CPU offload."""
    latents_list = []
    for i in range(0, input.shape[0], encode_batch_size):
        batch  = input[i:i + encode_batch_size]
        latent = ip2p.vae.encode(2 * batch - 1).latent_dist.mode()
        latents_list.append(latent.cpu())
        del batch, latent
        torch.cuda.empty_cache()
    return torch.cat(latents_list, dim=0).to(device=input.device)


# ============================================================
# Core training loop
# ============================================================

def scene_reconstruction(dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
                          checkpoint_iterations, checkpoint, debug_from,
                          gaussians, scene, stage, tb_writer, train_iter, timer,
                          ip2p, unet_phi, lora_optimizer,
                          prompt, guidance_scale, image_guidance_scale):
    """
    VSD-based refinement stage.
    sequence_length=1 and a reduced resize keep VRAM usage minimal
    while the VSD gradient provides high-quality editing signal.
    """
    torch_dtype    = torch.float16
    device         = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sequence_length = 2           # temporal context for 3D UNet attention
    diffusion_step  = 20
    num_train_timesteps = 1000

    first_iter = 0
    gaussians.training_only3dgs_setup(opt)

    bg_color   = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end   = torch.cuda.Event(enable_timing=True)

    viewpoint_stack  = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    final_iter       = train_iter

    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter  += 1

    video_cams = scene.getVideoCameras()
    test_cams  = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    if not viewpoint_stack and not opt.dataloader:
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)

    batch_size = opt.batch_size
    print("data loading done")

    if opt.dataloader:
        viewpoint_stack = scene.getTrainCameras()
        if opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,
                                                sampler=sampler, num_workers=1, collate_fn=list)
            random_loader = False
        else:
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,
                                                shuffle=True, num_workers=1, collate_fn=list)
            random_loader = True
        loader = iter(viewpoint_stack_loader)

    if stage == "coarse" and opt.zerostamp_init:
        load_in_memory = True
        temp_list = get_stamp_list(viewpoint_stack, 0)
        viewpoint_stack = temp_list.copy()
    else:
        load_in_memory = False

    count = 0
    for iteration in range(first_iter, final_iter + 1):
        # ---- Dynamic scheduling based on training progress ----
        progress = iteration / final_iter  # 0.0 → 1.0
        
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, \
                    keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam is not None:
                    count += 1
                    viewpoint_index = (count) % len(video_cams)
                    if (count // (len(video_cams))) % 2 == 0:
                        viewpoint_index = viewpoint_index
                    else:
                        viewpoint_index = len(video_cams) - viewpoint_index - 1
                    viewpoint = video_cams[viewpoint_index]
                    custom_cam.time = viewpoint.time
                    net_image = render(custom_cam, gaussians, pipe, background,
                                       scaling_modifer, stage=stage,
                                       cam_type=scene.dataset_type)["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                print(e)
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # ---- sample viewpoint cameras ----
        if opt.dataloader and not load_in_memory:
            try:
                viewpoint_cams = []
                while len(viewpoint_cams) < sequence_length:
                    batch_cams = next(loader)
                    viewpoint_cams.extend(batch_cams)
                viewpoint_cams = viewpoint_cams[:sequence_length]
                assert len(viewpoint_cams) == sequence_length
            except StopIteration:
                print("reset dataloader into random dataloader.")
                if not random_loader:
                    viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=opt.batch_size,
                                                        shuffle=True, num_workers=1, collate_fn=list)
                    random_loader = True
                loader = iter(viewpoint_stack_loader)
                torch.cuda.empty_cache()
                continue
        else:
            idx = 0
            viewpoint_cams = []
            while idx < sequence_length:
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
                if not viewpoint_stack:
                    viewpoint_stack = temp_list.copy()
                viewpoint_cams.append(viewpoint_cam)
                idx += 1
            if len(viewpoint_cams) == 0:
                continue

        if (iteration - 1) == debug_from:
            pipe.debug = True

        # ---- render ----
        images, gt_images, radii_list, visibility_filter_list, viewspace_point_tensor_list = [], [], [], [], []
        for viewpoint_cam in viewpoint_cams:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background,
                                stage=stage, cam_type=scene.dataset_type)
            image, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg["render"], render_pkg["viewspace_points"],
                render_pkg["visibility_filter"], render_pkg["radii"])
            images.append(image.unsqueeze(0))
            if scene.dataset_type != "PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image = viewpoint_cam['image'].cuda()
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)

        radii             = torch.cat(radii_list, 0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        image_tensor      = torch.cat(images, 0)
        gt_image_tensor   = torch.cat(gt_images, 0)

        del images, gt_images, radii_list, visibility_filter_list
        torch.cuda.empty_cache()

        # ---- resize for VAE  (small for VRAM budget) ----
        dataset_length, C, H, W = image_tensor.shape
        args.resize = 512                   # memory-efficient resize
        factor      = args.resize / max(W, H)
        factor      = math.ceil(min(W, H) * factor / 64) * 64 / min(W, H)
        new_width   = int((W * factor) // 64) * 64
        new_height  = int((H * factor) // 64) * 64

        vae_input_images = F.interpolate(image_tensor, size=(new_height, new_width),
                                          mode='bilinear', align_corners=False
                                          ).to(device=device, dtype=torch_dtype)
        vae_input_images_cond = F.interpolate(gt_image_tensor, size=(new_height, new_width),
                                               mode='bilinear', align_corners=False
                                               ).to(device=device, dtype=torch_dtype)

        # ==============================================================
        # VSD: Variational Score Distillation  (refinement stage)
        #
        # Same as fully_edit_sds.py but sequence_length=1, resize=128.
        # ε_phi is trained to model the single-frame rendered distribution,
        # giving a sharp, low-noise gradient at every refinement step.
        # ==============================================================

        # 1. Encode to CLEAN latents (gradient path to Gaussians intact)
        clean_latents = encode_1(ip2p, vae_input_images)
        image_latents = encode_2(ip2p, vae_input_images_cond)

        del vae_input_images, vae_input_images_cond
        torch.cuda.empty_cache()

        clean_latents = rearrange(clean_latents, "(b f) c h w -> b c f h w",
                                   f=sequence_length).to(device=device, dtype=torch_dtype)
        image_latents = rearrange(image_latents, "(b f) c h w -> b c f h w",
                                   f=sequence_length).to(device=device, dtype=torch_dtype)
        uncond_image_latents = torch.zeros_like(image_latents)

        prompt_embeds = ip2p._encode_prompt(
            prompt, device=device, num_images_per_prompt=1,
            do_classifier_free_guidance=True)

        ip2p.scheduler.config.num_train_timesteps = num_train_timesteps
        ip2p.scheduler.set_timesteps(diffusion_step)
        
        # Save image_latents before stacking for CFG (for single-conditional phi training)
        image_latents_cond = image_latents.clone()

        # 2. Sample noise and timestep; form noisy latents z_t
        # Dynamic t sampling: high timesteps early (weak recon) → low timesteps late (strong recon)
        noise         = torch.randn_like(clean_latents)
        t_max         = int(1000 * (0.8 - 0.5 * progress))   # 800 → 300
        t_min         = int(1000 * (0.1 - 0.08 * progress))  # 100 → 20
        t             = torch.randint(t_min, t_max, [1], dtype=torch.long, device=device)
        noisy_latents = ip2p.scheduler.add_noise(clean_latents, noise, t)

        # 3. CFG conditioning stack
        image_cond_cat = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)

        # ---- 4a. IP2P score  (pretrained frozen) ----
        # Scale guidance down over time: high early (drive edit) → low late (lock in)
        effective_guidance       = guidance_scale       * (1.0 - 0.3 * progress)   # 7.5 → 5.25
        effective_image_guidance = image_guidance_scale * (1.0 + 0.5 * progress)   # 1.5 → 2.25
        
        latent_ip2p_input = torch.cat([noisy_latents] * 3)
        latent_ip2p_input = torch.cat([latent_ip2p_input, image_cond_cat], dim=1)
        with torch.no_grad():
            noise_pred_ip2p = ip2p.unet(latent_ip2p_input, t, prompt_embeds,
                                         None, None, False)[0]
            np_text, np_image, np_uncond = noise_pred_ip2p.chunk(3)
            noise_pred_ip2p = (np_uncond
                               + effective_guidance       * (np_text  - np_image)
                               + effective_image_guidance * (np_image - np_uncond))

        # ---- 4b. Phi (LoRA) score — single-conditional (NO CFG) ----
        # Phi learns the rendering distribution p_phi(ε | z_t, cI, cT), not a CFG composition.
        # Single-conditional input only → saves 3x memory vs 3x stacking.
        latent_phi_input = torch.cat([noisy_latents.detach(), image_latents_cond.detach()], dim=1)
        noise_pred_phi = unet_phi(latent_phi_input, t,
                                   prompt_embeds[1:2].detach(), None, None, False)[0]

        # ---- 5. VSD gradient ----
        alphas = ip2p.scheduler.alphas_cumprod.to(device)
        w      = (1 - alphas[t]).view(-1, 1, 1, 1)

        grad = w * (noise_pred_ip2p - noise_pred_phi.detach())
        grad = torch.nan_to_num(grad)

        target   = (noisy_latents - grad).detach().to(dtype=torch.float16)
        loss_vsd = 0.5 * F.mse_loss(noisy_latents, target, reduction="mean")
        loss_vsd = loss_vsd.to(dtype=torch.float16)

        # ---- 6. Reconstruction loss (grows over time to lock in edits) ----
        lambda_recon = 0.0 + 0.15 * progress  # 0.0 → 0.15
        loss_recon = l1_loss(image_tensor, gt_image_tensor)
        
        # ---- 7. Combined loss ----
        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        loss  = loss_vsd + lambda_recon * loss_recon
        # TV regularisation commented out for refinement stage (same as original)

        # ---- 7. Gaussian backward ----
        loss.backward()

        if torch.isnan(loss).any():
            print("loss is nan, end training, reexecv program now.")
            os.execv(sys.executable, [sys.executable] + sys.argv)

        # ---- 8. LoRA (phi) update ----
        #   Train phi to match the diffusion noise (DDPM objective).
        #   Reuses noise_pred_phi from section 4b (already in gradient tape).
        lora_optimizer.zero_grad()
        loss_phi = 0.5 * F.mse_loss(noise_pred_phi, noise.detach(), reduction="mean")
        loss_phi.backward()
        lora_optimizer.step()

        # ---- cleanup ----
        del (clean_latents, image_latents, image_latents_cond, uncond_image_latents,
             latent_ip2p_input, latent_phi_input, image_cond_cat,
             noise_pred_ip2p, noise_pred_phi,
             noise, grad, target, loss_vsd, loss_recon, loss_phi)
        torch.cuda.empty_cache()

        # ---- viewspace gradient accumulation ----
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(len(viewspace_point_tensor_list)):
            viewspace_point_tensor_grad += viewspace_point_tensor_list[idx].grad
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_       + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                          "point": f"{total_point}"})
                progress_bar.update(10)
                torch.cuda.empty_cache()
            if iteration == opt.iterations:
                progress_bar.close()

            timer.pause()
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save_refine(iteration, stage, prompt)
            if dataset.render_process:
                if ((iteration < 1000 and iteration % 10 == 9)
                        or (iteration < 3000 and iteration % 50 == 49)
                        or (iteration < 60000 and iteration % 100 == 99)):
                    render_training_image(
                        scene, gaussians, [test_cams[iteration % len(test_cams)]],
                        render, pipe, background, stage + "test", iteration,
                        timer.get_elapsed_time(), scene.dataset_type)
                    render_training_image(
                        scene, gaussians, [train_cams[iteration % len(train_cams)]],
                        render, pipe, background, stage + "train", iteration,
                        timer.get_elapsed_time(), scene.dataset_type)
            timer.start()

            # ---- densification disabled for refinement stage ----
            # Point cloud from prior stage already well-formed.
            # Densification adds new Gaussians with no history that get pushed
            # randomly by diffusion gradients → causes floaters and noise.
            # if iteration < opt.densify_until_iter:
            #     gaussians.max_radii2D[visibility_filter] = torch.max(
            #         gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            #     gaussians.add_densification_stats(
            #         viewspace_point_tensor_grad * 0.000001, visibility_filter)
            # 
            #     opacity_threshold = (opt.opacity_threshold_fine_init
            #                          - iteration * (opt.opacity_threshold_fine_init
            #                                         - opt.opacity_threshold_fine_after)
            #                          / opt.densify_until_iter)
            #     densify_threshold = (opt.densify_grad_threshold_fine_init
            #                          - iteration * (opt.densify_grad_threshold_fine_init
            #                                         - opt.densify_grad_threshold_after)
            #                          / opt.densify_until_iter)
            # 
            #     if (iteration > opt.densify_from_iter
            #             and iteration % opt.densification_interval == 0
            #             and gaussians.get_xyz.shape[0] < 30000):
            #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #         gaussians.densify(densify_threshold, opacity_threshold,
            #                           scene.cameras_extent, size_threshold, 5, 5,
            #                           scene.model_path, iteration, stage)
            #         torch.cuda.empty_cache()
            # 
            #     if (iteration > opt.pruning_from_iter
            #             and iteration % opt.pruning_interval == 0
            #             and gaussians.get_xyz.shape[0] > 10000):
            #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #         gaussians.prune(densify_threshold, opacity_threshold,
            #                         scene.cameras_extent, size_threshold)
            #         torch.cuda.empty_cache()
            # 
            #     if (iteration % opt.densification_interval == 0
            #             and gaussians.get_xyz.shape[0] < 30000
            #             and opt.add_point):
            #         gaussians.grow(5, 5, scene.model_path, iteration, stage)
            # 
            #     if iteration % opt.opacity_reset_interval == 0:
            #         print("reset opacity")
            #         gaussians.reset_opacity()

            if iteration < opt.iterations:
                # Gradient clipping to prevent exploding gradients from diffusion signal
                torch.nn.utils.clip_grad_norm_(
                    [p for group in gaussians.optimizer.param_groups for p in group['params']],
                    max_norm=1.0
                )
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + f"/chkpnt_{stage}_{iteration}.pth")


def training(dataset, hyper, opt, pipe, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, debug_from, expname,
             prompt, guidance_scale, image_guidance_scale):
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(dataset.sh_degree, hyper)
    dataset.model_path = args.model_path
    timer = Timer()
    scene = Scene(dataset, gaussians, load_coarse=None)
    gaussians.load_ply(args.ply_path)
    print(f"Loaded ply from {args.ply_path}")

    # ---- auto-detect latest iteration checkpoint ----
    point_cloud_dir = os.path.join(args.model_path, "point_cloud")
    iterations = []
    if os.path.exists(point_cloud_dir):
        for name in os.listdir(point_cloud_dir):
            if name.startswith("iteration_"):
                try:
                    iterations.append(int(name.split("_")[1]))
                except ValueError:
                    pass
    else:
        raise FileNotFoundError(f"Point cloud folder not found: {point_cloud_dir}")
    if not iterations:
        raise FileNotFoundError(f"No iteration folders found in {point_cloud_dir}")
    latest_iter = max(iterations)
    print(f"Loading latest checkpoint: iteration_{latest_iter}")
    gaussians.load_model(os.path.join(point_cloud_dir, f"iteration_{latest_iter}"))

    gaussians._deformation_table = torch.gt(
        torch.ones((gaussians.get_xyz.shape[0],), device="cuda"), 0)
    print("Loaded deformation field")
    timer.start()

    seed_everything(20211202)
    device      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16

    DDIM_SOURCE = "CompVis/stable-diffusion-v1-4"
    IP2P_SOURCE = "timbrooks/instruct-pix2pix"

    tokenizer    = CLIPTokenizer.from_pretrained(IP2P_SOURCE, subfolder="tokenizer")
    sys.stdout.isatty = lambda: False
    text_encoder = CLIPTextModel.from_pretrained(IP2P_SOURCE, subfolder="text_encoder")
    vae          = AutoencoderKL.from_pretrained(IP2P_SOURCE, subfolder="vae")
    unet         = UNet3DConditionModel.from_pretrained_2d(IP2P_SOURCE, subfolder="unet")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    vae          = vae.to(device, dtype=torch_dtype)
    text_encoder = text_encoder.to(device, dtype=torch_dtype)
    unet         = unet.to(device, dtype=torch_dtype)

    ip2p = InstructPix2PixPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler.from_pretrained(DDIM_SOURCE, subfolder="scheduler"),
    )
    print("Ready IP2P")

    # ---- VSD: phi model (frozen UNet + LoRA) ----
    print("[VSD] Creating phi model …")
    unet_phi    = copy.deepcopy(unet)
    unet_phi.requires_grad_(False)
    lora_params = inject_lora_into_unet(unet_phi, rank=4, alpha=4.0)
    # LoRA A/B remain float32 (LoRALinear handles mixed-precision cast)
    lora_optimizer = torch.optim.AdamW(lora_params, lr=1e-4, weight_decay=1e-2)
    print("[VSD] Phi model ready.")

    scene_reconstruction(
        dataset, opt, hyper, pipe, testing_iterations, saving_iterations,
        checkpoint_iterations, checkpoint, debug_from,
        gaussians, scene, "fine", tb_writer, 500, timer,
        ip2p, unet_phi, lora_optimizer,
        prompt, guidance_scale, image_guidance_scale)


def prepare_output_and_logger(expname):
    if not args.model_path:
        unique_str = expname
        args.model_path = os.path.join("./output/", unique_str)
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed,
                    testing_iterations, scene: Scene, renderFunc, renderArgs, stage, dataset_type):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/vsd_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test',
             'cameras': [scene.getTestCameras()[idx % len(scene.getTestCameras())]
                         for idx in range(10, 5000, 299)]},
            {'name': 'train',
             'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                         for idx in range(10, 5000, 299)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test   = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, stage=stage,
                                   cam_type=dataset_type, *renderArgs)["render"], 0.0, 1.0)
                    if dataset_type == "PanopticSports":
                        gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    try:
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(
                                stage + "/" + config['name'] + f"_view_{viewpoint.image_name}/render",
                                image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(
                                    stage + "/" + config['name'] + f"_view_{viewpoint.image_name}/ground_truth",
                                    gt_image[None], global_step=iteration)
                    except Exception:
                        pass
                    l1_test   += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image, mask=None).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test   /= len(config['cameras'])
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: "
                      f"L1 {l1_test} PSNR {psnr_test}")
                if tb_writer:
                    tb_writer.add_scalar(
                        stage + "/" + config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(
                        stage + "/" + config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram(f"{stage}/scene/opacity_histogram",
                                    scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(f'{stage}/total_points',
                                 scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate',
                                 scene.gaussians._deformation_table.sum()
                                 / scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_histogram(f"{stage}/scene/motion_histogram",
                                    scene.gaussians._deformation_accum.mean(dim=-1) / 100,
                                    iteration, max_bins=500)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip',                    type=str,   default="127.0.0.1")
    parser.add_argument('--port',                  type=int,   default=6009)
    parser.add_argument('--debug_from',            type=int,   default=-1)
    parser.add_argument('--detect_anomaly',        action='store_true', default=False)
    parser.add_argument("--test_iterations",       nargs="+",  type=int,
                        default=[3000, 5000])
    parser.add_argument("--save_iterations",       nargs="+",  type=int,
                        default=[100, 300, 500, 800, 1000, 1500, 2000, 3000, 5000])
    parser.add_argument("--quiet",                 action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+",  type=int, default=[])
    parser.add_argument("--start_checkpoint",      type=str,   default=None)
    parser.add_argument("--expname",               type=str,   default="")
    parser.add_argument("--configs",               type=str,   default="")
    parser.add_argument("--ply_path",              type=str,   default="")
    # VSD-compatible prompt / guidance arguments (same CLI as before)
    parser.add_argument("--prompt",                type=str,   default="")
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--image_guidance_scale', type=float, default=1.5)  

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args),
             args.test_iterations, args.save_iterations, args.checkpoint_iterations,
             args.start_checkpoint, args.debug_from, args.expname,
             args.prompt, args.guidance_scale, args.image_guidance_scale)

    print("\nEditing complete.")