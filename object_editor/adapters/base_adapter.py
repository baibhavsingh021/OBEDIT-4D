"""Stable adapter contract for image editors.

The 4DGS pipeline must not depend on a diffusion implementation's internal
UNet layout. Adapters therefore expose image editing as the required path and
only expose latent operations when the backend can prove that they are valid.
"""

import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch


@dataclass(frozen=True)
class EditorCapabilities:
    supports_references: bool = False
    supports_latent_hooks: bool = False
    supports_region_conditioning: bool = False
    supports_cpu_offload: bool = False


class BaseEditorAdapter(abc.ABC):
    """Backend-neutral editor interface.

    Images are ``(B, 3, H, W)`` float tensors in ``[-1, 1]``. Masks are
    ``(B, 1, H, W)`` float tensors in ``[0, 1]``. A backend may process only
    one view at a time; the caller owns cross-view batching and correspondence.
    """

    capabilities = EditorCapabilities()

    def __init__(self, device="cuda", dtype=torch.float16):
        if not str(device).startswith("cuda"):
            raise ValueError("object_editor requires a CUDA device")
        if dtype != torch.float16:
            raise ValueError("T4 mode requires torch.float16; bf16 is unsupported")
        self.device = torch.device(device)
        self.dtype = dtype

    @abc.abstractmethod
    def edit(self, image, instruction, target_mask=None, references=None,
             negative_prompt=None, **kwargs):
        """Return an edited image with the same shape and value range."""

    def encode_image(self, image):
        raise NotImplementedError("This adapter does not expose latent encoding")

    def decode_latent(self, latent):
        raise NotImplementedError("This adapter does not expose latent decoding")

    def sample(self, *args, **kwargs):
        raise NotImplementedError("This adapter does not expose latent sampling")

    def memory_cleanup(self):
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def validate_image(image, name="image"):
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise ValueError("{} must have shape (B,3,H,W)".format(name))
        if image.shape[1] != 3:
            raise ValueError("{} must have three RGB channels".format(name))
        if not torch.is_floating_point(image):
            raise TypeError("{} must be floating point".format(name))
