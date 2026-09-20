"""Legacy InstructPix2Pix adapter placeholder.

The existing ``ip2p_models`` path remains authoritative. This wrapper keeps
selection/configuration stable without importing the legacy model at package
import time.
"""

from .base_adapter import BaseEditorAdapter, EditorCapabilities


class IP2PAdapter(BaseEditorAdapter):
    capabilities = EditorCapabilities(supports_latent_hooks=True)

    def __init__(self, *args, **kwargs):
        super().__init__(device=kwargs.pop("device", "cuda"))
        self._args = args
        self._kwargs = kwargs
        self._editor = None

    def _load_model(self):
        if self._editor is None:
            from ip2p_models.ip2p import InstructPix2Pix
            self._editor = InstructPix2Pix(self.device)

    def edit(self, image, instruction, target_mask=None, references=None, **kwargs):
        self.validate_image(image)
        self._load_model()
        raise NotImplementedError(
            "Use the existing IP2P call with encoded text embeddings, or add a "
            "version-specific adapter mapping before selecting ip2p here."
        )
