"""Bounded view/time windows and detached diffusion targets for 4DGS refinement."""

from collections import OrderedDict
from copy import copy
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import trange

from gaussian_renderer import render
from ip2p_models.trajectory_editor import TrajectoryEditor
from utils.loss_utils import ssim
from utils.trajectory_transport import (
    CanonicalResidualMemory,
    TrajectoryTransport,
    project_trajectories,
)


class ViewTimeWindows:
    def __init__(self, cameras, views, times, stride, seed):
        if cameras.dataset_type not in ("dynerf", "MultipleView"):
            raise ValueError("Trajectory refinement requires calibrated dynerf/MultipleView data; "
                             "use --refinement_mode sds for other dataset layouts")
        if views < 2 or times < 2 or views * times > 6 or stride < 1:
            raise ValueError("Use at least two views and times, at most six slots, and positive stride")
        self.cameras = cameras
        self.views, self.times, self.stride = views, times, stride
        self.rng = random.Random(seed)
        self.tracks = {}
        centers = {}
        for index, ((rotation, translation), time) in enumerate(zip(
                cameras.dataset.image_poses, cameras.dataset.image_times)):
            key = tuple(np.round(np.concatenate([rotation.flatten(), translation.flatten()]), 6))
            track = self.tracks.setdefault(key, {})
            stamp = round(float(time), 6)
            if stamp in track:
                raise ValueError("Duplicate camera/time observation in calibrated dataset")
            track[stamp] = index
            centers[key] = -rotation @ translation
        self.keys = list(self.tracks)
        if len(self.keys) < views:
            raise ValueError("Not enough distinct calibrated views for trajectory refinement")
        self.stamps = sorted(set.intersection(*(set(track) for track in self.tracks.values())))
        if len(self.keys) < views or len(self.stamps) < times:
            raise ValueError("Not enough distinct synchronized views/times for trajectory refinement")
        self.neighbors = {
            key: sorted((other for other in self.keys if other != key),
                        key=lambda other: np.linalg.norm(centers[key] - centers[other]))
            for key in self.keys
        }

    def sample(self):
        anchor = self.rng.choice(self.keys)
        neighbors = self.neighbors[anchor]
        # Mix nearby baselines with global pairs so view groups remain connected.
        pool = neighbors[:max(self.views - 1, 4)] if self.rng.random() < 0.8 else neighbors
        views = [anchor] + self.rng.sample(pool, self.views - 1)
        stride = self.rng.randint(1, min(self.stride, (len(self.stamps) - 1) // (self.times - 1)))
        start = self.rng.randrange(len(self.stamps) - (self.times - 1) * stride)
        stamps = self.stamps[start:start + self.times * stride:stride]
        return [self.tracks[view][stamp] for stamp in stamps for view in views]

    def reference_indices(self):
        return [track[min(track)] for track in self.tracks.values()]


def resize_camera(camera, resolution):
    camera = copy(camera)
    height, width = camera.image_height, camera.image_width
    scale = min(1, resolution / max(height, width))
    size = (max(64, int(height * scale) // 64 * 64),
            max(64, int(width * scale) // 64 * 64))
    camera.original_image = F.interpolate(
        camera.original_image[None].float(), size=size, mode="bilinear", align_corners=False,
    )[0]
    camera.image_height, camera.image_width = size
    return camera


@torch.no_grad()
def lift_edit_mask(windows, reference, pipe, background, options):
    count = reference.get_xyz.shape[0]
    if not options.refine_mask_dir:
        return torch.ones(count, device=background.device)
    numerator = torch.zeros(count, device=background.device)
    denominator = torch.zeros_like(numerator)
    for index in windows.reference_indices():
        source = Path(windows.cameras.dataset.image_paths[index])
        folder = source.parent.parent if source.parent.name == "images" else source.parent
        if not folder.name.startswith("cam") or not folder.name[3:].isdigit():
            raise ValueError(f"Cannot associate a mask with camera folder {folder}")
        path = Path(options.refine_mask_dir) / f"mask_cam{int(folder.name[3:]):02d}.png"
        with Image.open(path) as image:
            mask = torch.from_numpy(np.array(image.convert("L"), copy=True)).to(background.device)
        camera = resize_camera(windows.cameras[index], options.refine_resolution)
        geometry = render(camera, reference, pipe, background, return_geometry=True)
        projection = project_trajectories(camera, geometry, options.refine_depth_tolerance)
        transport = TrajectoryTransport([projection], mask.shape)
        sample = transport.gather_frame(mask[None].float() / 255, 0)[:, 0]
        numerator += sample * projection.confidence
        denominator += projection.confidence
    membership = (numerator / denominator.clamp_min(1e-6)).clamp(0, 1)
    if not (membership > 0.1).any():
        raise ValueError("No visible edit-mask support on the frozen Gaussian trajectories")
    return membership


class ReferenceCache:
    def __init__(self, windows, reference, pipe, background, editor, membership, options):
        self.windows, self.reference = windows, reference
        self.pipe, self.background, self.editor = pipe, background, editor
        self.membership, self.options = membership, options
        self.entries = OrderedDict()

    @torch.no_grad()
    def get(self, index):
        if index in self.entries:
            self.entries.move_to_end(index)
            return self.entries[index]
        camera = resize_camera(self.windows.cameras[index], self.options.refine_resolution)
        original = camera.original_image.to(self.background.device)
        reference = render(camera, self.reference, self.pipe, self.background)["render"].clamp(0, 1)
        if self.options.refine_mask_dir:
            mask = render(
                camera, self.reference, self.pipe, torch.zeros_like(self.background),
                override_color=self.membership[:, None].expand(-1, 3).contiguous(),
            )["render"][:1].clamp(0, 1)
        else:
            mask = torch.ones_like(original[:1])
        reference = reference * mask + original * (1 - mask)
        conditioning = self.editor.encode(original[None])[0]
        reference_latent = self.editor.encode(reference[None])[0] * self.editor.scale
        entry = (camera, original.cpu(), reference.cpu(), mask.cpu(),
                 conditioning.cpu(), reference_latent.cpu())
        self.entries[index] = entry
        while len(self.entries) > self.options.refine_cache_size:
            self.entries.popitem(last=False)
        return entry


def add_trajectory_arguments(parser):
    group = parser.add_argument_group("Trajectory-consistent T4 refinement")
    group.add_argument("--refinement_mode", choices=("trajectory", "sds"), default="trajectory")
    group.add_argument("--refine_iterations", type=int, default=800)
    group.add_argument("--refine_views", type=int, default=2)
    group.add_argument("--refine_times", type=int, default=2)
    group.add_argument("--refine_time_stride", type=int, default=8)
    group.add_argument("--refine_resolution", type=int, default=512)
    group.add_argument("--refine_diffusion_steps", type=int, default=20)
    group.add_argument("--refine_target_interval", type=int, default=8)
    group.add_argument("--refine_cache_size", type=int, default=16)
    group.add_argument("--refine_strength", type=float, default=0.35)
    group.add_argument("--refine_consensus", type=float, default=0.35)
    group.add_argument("--refine_memory_weight", type=float, default=0.35)
    group.add_argument("--refine_noise_correlation", type=float, default=0.5)
    group.add_argument("--refine_reference_weight", type=float, default=0.2)
    group.add_argument("--refine_multiview_weight", type=float, default=0.05)
    group.add_argument("--refine_temporal_weight", type=float, default=0.1)
    group.add_argument("--refine_background_weight", type=float, default=1.0)
    group.add_argument("--refine_depth_tolerance", type=float, default=0.03)
    group.add_argument("--refine_mask_dir", default="", help="mask_camNN.png from mask_image_swap; empty edits the full scene")
    group.add_argument("--refine_seed", type=int, default=20211202)


def refine_trajectories(scene, gaussians, pipe, opt, dataset, ip2p, options, writer):
    if not 64 <= options.refine_resolution <= 512:
        raise ValueError("T4 trajectory refinement requires a resolution between 64 and 512")
    if min(options.refine_iterations, options.refine_target_interval, options.refine_cache_size,
           options.refine_diffusion_steps) < 1:
        raise ValueError("Iteration, diffusion-step, refresh and cache counts must be positive")
    if options.refine_diffusion_steps > ip2p.scheduler.config.num_train_timesteps:
        raise ValueError("Diffusion steps exceed the pretrained noise schedule")
    for value in (options.refine_strength, options.refine_consensus,
                  options.refine_reference_weight, options.refine_memory_weight,
                  options.refine_noise_correlation):
        if not 0 <= value <= 1:
            raise ValueError("Diffusion strength and mixing weights must be in [0, 1]")
    if options.refine_depth_tolerance <= 0 or min(
            options.refine_multiview_weight, options.refine_temporal_weight,
            options.refine_background_weight) < 0:
        raise ValueError("Depth tolerance must be positive and loss weights nonnegative")
    if pipe.compute_cov3D_python:
        raise ValueError("Trajectory refinement requires deformation-aware rasterizer covariance")
    device = gaussians.get_xyz.device
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                              dtype=torch.float32, device=device)
    windows = ViewTimeWindows(scene.getTrainCameras(), options.refine_views,
                             options.refine_times, options.refine_time_stride, options.refine_seed)
    # Preserve the canonical row IDs, geometry and HexPlane motion throughout editing.
    for tensor in (gaussians._xyz, gaussians._scaling, gaussians._rotation, gaussians._opacity):
        tensor.requires_grad_(False)
    gaussians._deformation.requires_grad_(False).eval()
    reference = copy(gaussians)
    reference._features_dc = gaussians._features_dc.detach().clone()
    reference._features_rest = gaussians._features_rest.detach().clone()
    optimizer = torch.optim.Adam([
        {"params": [gaussians._features_dc], "lr": opt.feature_lr},
        {"params": [gaussians._features_rest], "lr": opt.feature_lr / 20},
    ], eps=1e-15)
    editor = TrajectoryEditor(
        ip2p, options.prompt, options.guidance_scale, options.image_guidance_scale,
        options.refine_diffusion_steps, options.refine_consensus,
        options.refine_reference_weight, options.refine_memory_weight, options.refine_noise_correlation,
    )
    membership = lift_edit_mask(windows, reference, pipe, background, options)
    cache = ReferenceCache(windows, reference, pipe, background, editor, membership, options)
    memory = CanonicalResidualMemory(gaussians.get_xyz.shape[0], 4, device)
    generator = torch.Generator(device=device).manual_seed(options.refine_seed)
    first = 0
    if options.start_checkpoint:
        state = torch.load(options.start_checkpoint, map_location=device)
        if not isinstance(state, dict) or state.get("mode") != "trajectory":
            raise ValueError("Use a trajectory checkpoint or --refinement_mode sds for legacy checkpoints")
        if state["source"] != str(Path(options.ply_path).resolve()) or state["prompt"] != options.prompt:
            raise ValueError("Resume requires the original edited PLY and the same instruction")
        if not torch.equal(state["xyz"], gaussians.get_xyz):
            raise ValueError("Canonical Gaussian identities changed since the checkpoint")
        with torch.no_grad():
            gaussians._features_dc.copy_(state["dc"])
            gaussians._features_rest.copy_(state["rest"])
        optimizer.load_state_dict(state["optimizer"])
        memory.features.copy_(state["memory"])
        memory.confidence.copy_(state["confidence"])
        windows.rng.setstate(state["sampler"])
        generator.set_state(state["noise"].cpu())
        first = state["iteration"]
    for iteration in trange(first + 1, options.refine_iterations + 1, desc="4D trajectory refinement"):
        if (iteration - first - 1) % options.refine_target_interval == 0:
            entries = [cache.get(index) for index in windows.sample()]
            cameras = [entry[0] for entry in entries]
            originals, references, masks, conditioning, reference_latents = [
                torch.stack([entry[column] for entry in entries]).to(device)
                for column in range(1, 6)
            ]
            with torch.no_grad():
                projections, current = [], []
                for camera in cameras:
                    geometry = render(camera, gaussians, pipe, background, return_geometry=True)
                    projection = project_trajectories(camera, geometry, options.refine_depth_tolerance)
                    projection.confidence *= membership
                    projections.append(projection)
                    current.append(geometry["render"].clamp(0, 1))
                current = torch.stack(current)
                latents = editor.encode(current) * editor.scale
                latent_masks = F.interpolate(masks, size=latents.shape[-2:], mode="area")
                transport = TrajectoryTransport(projections, latents.shape[-2:])
                targets = editor.edit(
                    latents, conditioning, reference_latents, latent_masks, transport, memory,
                    generator, options.refine_strength,
                )
                targets = targets * masks + originals * (1 - masks)
                image_transport = TrajectoryTransport(projections, targets.shape[-2:])
                del geometry, transport, current, latents
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            # Peers are refreshed each optimizer step; only one renderer graph is live.
            peers = torch.stack([
                image_transport.gather_frame(
                    render(camera, gaussians, pipe, background)["render"] - originals[i], i,
                ) for i, camera in enumerate(cameras)
            ])
        total = torch.zeros((), device=device)
        for i, camera in enumerate(cameras):
            image = render(camera, gaussians, pipe, background)["render"]
            mask = masks[i]
            reconstruction = ((image - targets[i]).abs() * mask).sum() / (3 * mask.sum()).clamp_min(1)
            structure = 1 - ssim(image * mask + targets[i] * (1 - mask), targets[i])
            background_loss = ((image - originals[i]).abs() * (1 - mask)).sum()
            background_loss /= (3 * (1 - mask).sum()).clamp_min(1)
            anchor = ((image - references[i]).abs() * mask).sum() / (3 * mask.sum()).clamp_min(1)
            residual = image - originals[i]
            multiview = image_transport.consistency_loss(residual, i, peers, temporal=False)
            temporal = image_transport.consistency_loss(residual, i, peers, temporal=True)
            loss = ((1 - opt.lambda_dssim) * reconstruction + opt.lambda_dssim * structure
                    + options.refine_background_weight * background_loss
                    + options.refine_reference_weight * anchor
                    + options.refine_multiview_weight * multiview
                    + options.refine_temporal_weight * temporal) / len(cameras)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite trajectory reconstruction loss")
            loss.backward()
            total += loss.detach()
        torch.nn.utils.clip_grad_norm_(
            [gaussians._features_dc, gaussians._features_rest], 1.0, error_if_nonfinite=True,
        )
        optimizer.step()
        if writer:
            writer.add_scalar("trajectory/loss", total.item(), iteration)
            writer.add_scalar("trajectory/memory_coverage", (memory.confidence > 0).float().mean(), iteration)
        if iteration in options.save_iterations or iteration == options.refine_iterations:
            scene.save_refine(iteration, "fine", options.prompt)
        if iteration in options.checkpoint_iterations:
            torch.save({
                "mode": "trajectory", "iteration": iteration, "prompt": options.prompt,
                "source": str(Path(options.ply_path).resolve()), "xyz": gaussians.get_xyz.detach(),
                "dc": gaussians._features_dc.detach(), "rest": gaussians._features_rest.detach(),
                "optimizer": optimizer.state_dict(), "memory": memory.features,
                "confidence": memory.confidence, "sampler": windows.rng.getstate(),
                "noise": generator.get_state(),
            }, str(Path(scene.model_path) / f"chkpnt_trajectory_{iteration}.pth"))
    if writer:
        writer.close()
