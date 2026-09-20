"""Optional OmniGen adapter.

OmniGen is used through its published pipeline API. Its internal denoiser is
not treated as an SD/UNet, so geometry-aware latent hooks are disabled until a
backend-specific implementation is verified against the installed release.
"""

import torch

from .base_adapter import BaseEditorAdapter, EditorCapabilities


class OmniGenAdapter(BaseEditorAdapter):
    capabilities = EditorCapabilities(
        supports_references=True,
        supports_latent_hooks=False,
        supports_region_conditioning=False,
        supports_cpu_offload=True,
    )

    def __init__(self, model_path="BAAI/OmniGen-v1", device="cuda",
                 enable_cpu_offload=True, vae_tiling=True):
        super().__init__(device=device, dtype=torch.float16)
        self.model_path = model_path
        self.enable_cpu_offload = enable_cpu_offload
        self.vae_tiling = vae_tiling
        self._pipeline = None

    def _load_model(self):
        if self._pipeline is not None:
            return
        try:
            from diffusers import OmniGenPipeline
        except ImportError as exc:
            raise ImportError(
                "OmniGen requires a standalone newer diffusers environment; "
                "the legacy Python 3.7 4DGS environment is not sufficient."
            ) from exc
        pipe = OmniGenPipeline.from_pretrained(
            self.model_path, torch_dtype=torch.float16
        )
        if self.enable_cpu_offload:
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(self.device)
        if self.vae_tiling and hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        self._pipeline = pipe

    def edit(self, image, instruction, target_mask=None, references=None,
             negative_prompt=None, **kwargs):
        self.validate_image(image)
        if image.shape[0] != 1:
            raise ValueError("OmniGen T4 mode edits one view per call")
        self._load_model()
        from PIL import Image
        import torchvision.transforms as transforms

        to_pil = transforms.ToPILImage()
        images = [to_pil(((image[0].detach().cpu().float() + 1.0) / 2.0).clamp(0, 1))]
        for reference in references or []:
            if reference.ndim == 4:
                reference = reference[0]
            images.append(to_pil(((reference.detach().cpu().float() + 1.0) / 2.0).clamp(0, 1)))
        # The public OmniGen API has no verified latent mask hook here. The
        # caller restores the protected complement after generation.
        result = self._pipeline(
            prompt=instruction,
            images=images,
            num_inference_steps=kwargs.get("num_inference_steps", 30),
            guidance_scale=kwargs.get("guidance_scale", 7.5),
            height=image.shape[-2], width=image.shape[-1],
        ).images[0]
        output = transforms.ToTensor()(result).unsqueeze(0).to(self.device, self.dtype)
        return output * 2.0 - 1.0
