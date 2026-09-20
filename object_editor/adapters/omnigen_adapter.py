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

    def __init__(self, model_path="Shitao/OmniGen-v1", device="cuda",
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
            from OmniGen import OmniGenPipeline
        except ImportError as exc:
            raise ImportError(
                "OmniGen is missing. Install object_editor/requirements-colab.txt "
                "in the current runtime before constructing the editor."
            ) from exc
        self._pipeline = OmniGenPipeline.from_pretrained(
            self.model_path, use_fp16=True
        )

    def edit(self, image, instruction, target_mask=None, references=None,
             negative_prompt=None, **kwargs):
        self.validate_image(image)
        if image.shape[0] != 1:
            raise ValueError("OmniGen T4 mode edits one view per call")
        self._load_model()
        from PIL import Image
        import torchvision.transforms as transforms

        to_pil = transforms.ToPILImage()
        input_images = [to_pil(((image[0].detach().cpu().float() + 1.0) / 2.0).clamp(0, 1))]
        for reference in references or []:
            if reference.ndim == 4:
                reference = reference[0]
            input_images.append(to_pil(((reference.detach().cpu().float() + 1.0) / 2.0).clamp(0, 1)))
        placeholders = " ".join(
            "<img><|image_{}|></img>".format(index + 1)
            for index in range(len(input_images))
        )
        prompt = "{} Reference images: {}".format(instruction, placeholders)
        # The public OmniGen API has no verified latent mask hook here. The
        # caller restores the protected complement after generation.
        generated = self._pipeline(
            prompt=prompt,
            input_images=input_images,
            num_inference_steps=kwargs.get("num_inference_steps", 30),
            guidance_scale=kwargs.get("guidance_scale", 2.5),
            img_guidance_scale=kwargs.get("img_guidance_scale", 1.6),
            height=image.shape[-2], width=image.shape[-1],
            offload_model=self.enable_cpu_offload,
        )
        result = generated[0] if isinstance(generated, (list, tuple)) else generated.images[0]
        output = transforms.ToTensor()(result).unsqueeze(0).to(self.device, self.dtype)
        return output * 2.0 - 1.0
