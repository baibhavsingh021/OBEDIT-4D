"""Joint DDIM editing through visibility-gated Gaussian residual transport."""

import torch
from diffusers import DDIMScheduler

from utils.trajectory_transport import CanonicalResidualMemory, TrajectoryTransport


class TrajectoryEditor:
    def __init__(self, pipe, prompt, guidance_scale, image_guidance_scale,
                 steps=20, consensus=0.35, reference_weight=0.2, memory_weight=0.35,
                 noise_correlation=0.5):
        self.pipe = pipe
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, clip_sample=False)
        self.steps = steps
        self.consensus = consensus
        self.reference_weight = reference_weight
        self.memory_weight = memory_weight
        self.noise_correlation = noise_correlation
        self.guidance_scale = guidance_scale
        self.image_guidance_scale = image_guidance_scale
        self.device = next(pipe.unet.parameters()).device
        self.dtype = next(pipe.unet.parameters()).dtype
        self.scale = pipe.vae.config.scaling_factor
        if pipe.scheduler.config.prediction_type != "epsilon":
            raise ValueError("Trajectory DDIM requires an epsilon-prediction IP2P checkpoint")
        pipe.unet.requires_grad_(False).eval()
        pipe.vae.requires_grad_(False).eval()
        pipe.text_encoder.requires_grad_(False).eval()
        pipe.unet.set_attention_slice("auto")
        pipe.vae.enable_slicing()
        with torch.no_grad():
            self.embeddings = pipe._encode_prompt(
                prompt, device=self.device, num_images_per_prompt=1,
                do_classifier_free_guidance=True,
            ).to(dtype=self.dtype)
        pipe.text_encoder.to("cpu")

    @torch.no_grad()
    def encode(self, images):
        dtype = next(self.pipe.vae.parameters()).dtype
        return torch.cat([
            self.pipe.vae.encode(2 * image[None].to(self.device, dtype=dtype) - 1).latent_dist.mode()
            for image in images
        ]).to(dtype=self.dtype)

    @torch.no_grad()
    def decode(self, latents):
        dtype = next(self.pipe.vae.parameters()).dtype
        return torch.cat([
            (self.pipe.vae.decode(latent[None].to(dtype=dtype) / self.scale).sample / 2 + 0.5)
            .clamp(0, 1).float() for latent in latents
        ])

    @torch.no_grad()
    def predict_noise(self, latents, conditioning, timestep):
        predictions = []
        for latent, condition in zip(latents, conditioning):
            branches = []
            # CFG branches and frames run serially; only the small latent bank is joint.
            for branch in range(3):
                image = condition if branch < 2 else torch.zeros_like(condition)
                model_input = torch.cat([latent, image], dim=0)[None, :, None]
                branches.append(self.pipe.unet(
                    model_input, timestep, self.embeddings[branch:branch + 1],
                    return_dict=False,
                )[0][0, :, 0])
            text, image, unconditional = branches
            predictions.append(
                unconditional + self.guidance_scale * (text - image)
                + self.image_guidance_scale * (image - unconditional)
            )
        return torch.stack(predictions)

    @torch.no_grad()
    def edit(self, current, conditioning, reference, masks, transport: TrajectoryTransport,
             memory: CanonicalResidualMemory, generator, strength=0.35):
        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(self.steps, device=self.device)
        count = max(1, min(self.steps, round(self.steps * strength)))
        timesteps = scheduler.timesteps[-count:]
        original = conditioning * self.scale
        if strength == 0:
            return self.decode(current)
        noise = transport.shared_noise(current.shape[1], generator, self.noise_correlation).to(current)
        latents = scheduler.add_noise(current, noise, timesteps[:1])
        for index, timestep in enumerate(timesteps):
            epsilon = self.predict_noise(latents, conditioning, timestep).float()
            alpha = scheduler.alphas_cumprod[timestep].to(device=self.device, dtype=torch.float32)
            sigma = (1 - alpha).sqrt()
            clean = (latents.float() - sigma * epsilon) / alpha.sqrt()
            residual = clean - original.float()
            residual = transport.consensus(
                residual, memory.features, memory.confidence,
                self.consensus, self.memory_weight,
            )
            clean = original.float() + residual
            clean = clean.lerp(reference.float(), self.reference_weight)
            clean = masks * clean + (1 - masks) * original.float()
            corrected_epsilon = (latents.float() - alpha.sqrt() * clean) / sigma.clamp_min(1e-6)
            latents = scheduler.step(
                corrected_epsilon.to(latents.dtype), timestep, latents,
                eta=0.0, return_dict=False,
            )[0]
            if index + 1 < len(timesteps):
                background = scheduler.add_noise(original, noise, timesteps[index + 1:index + 2])
            else:
                background = original
            latents = masks * latents + (1 - masks) * background
            latents = latents.to(self.dtype)
        if not torch.isfinite(latents).all():
            raise FloatingPointError("Non-finite joint diffusion targets")
        memory.update(transport, (latents - original) * masks)
        return self.decode(latents)
