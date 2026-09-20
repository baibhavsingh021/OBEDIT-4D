"""SDXL img2img fallback adapter.

This adapter is deliberately pipeline-level. It does not expose fake UNet
hooks; cross-view coupling is implemented by the caller through references and
4DGS correspondence.
"""

import torch

from .base_adapter import BaseEditorAdapter, EditorCapabilities


class SDXLAdapter(BaseEditorAdapter):
    capabilities = EditorCapabilities(
        supports_references=True,
        supports_latent_hooks=False,
        supports_region_conditioning=False,
        supports_cpu_offload=True,
    )

    def __init__(self, model_path="stabilityai/stable-diffusion-xl-base-1.0",
                 device="cuda", enable_cpu_offload=True):
        super().__init__(device=device, dtype=torch.float16)
        self.model_path = model_path
        self.enable_cpu_offload = enable_cpu_offload
        self._pipeline = None

    def _load_model(self):
        if self._pipeline is not None:
            return
        from diffusers import StableDiffusionXLImg2ImgPipeline
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            self.model_path, torch_dtype=torch.float16, variant="fp16",
            use_safetensors=True,
        )
        if self.enable_cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self.device)
        pipe.enable_vae_tiling()
        pipe.enable_attention_slicing()
        self._pipeline = pipe

    def edit(self, image, instruction, target_mask=None, references=None,
             negative_prompt=None, **kwargs):
        self.validate_image(image)
        self._load_model()
        from PIL import Image
        import torchvision.transforms as transforms
        to_pil = transforms.ToPILImage()
        input_pil = to_pil(((image[0].detach().cpu().float() + 1.0) / 2.0).clamp(0, 1))
        # The base img2img API has no verified region hook here. The caller
        # restores the protected complement after generation.
        result = self._pipeline(
            prompt=instruction,
            negative_prompt=negative_prompt,
            image=input_pil,
            strength=kwargs.get("strength", 0.65),
            num_inference_steps=kwargs.get("num_inference_steps", 30),
            guidance_scale=kwargs.get("guidance_scale", 7.5),
        ).images[0]
        output = transforms.ToTensor()(result).unsqueeze(0).to(self.device, self.dtype)
        return output * 2.0 - 1.0
