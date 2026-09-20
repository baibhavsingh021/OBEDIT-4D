"""Image/mask bridge used by the existing 4DGS orchestration.

The caller supplies synchronized rendered images and masks reconciled through
4DGS. This module does not run Grounded-SAM or assume a foreground category.
"""

import torch

from .core import (CanonicalGaussianFeatureAnchoring,
                   GeometryAwareCrossViewCoupling,
                   ProtectedRegionLatentPreservation)


class ObjectEditorPipeline:
    def __init__(self, adapter, enable_cgfa=True, enable_gaxlc=True,
                 enable_prlp=True, coupling_strength=0.7):
        self.adapter = adapter
        self.anchor = CanonicalGaussianFeatureAnchoring(enable=enable_cgfa)
        self.coupling = None
        self.enable_gaxlc = enable_gaxlc
        self.enable_prlp = enable_prlp
        self.coupling_strength = coupling_strength

    def edit_views(self, images, target_masks, instruction, anchor_view=0,
                   references=None, **editor_kwargs):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape (V,3,H,W)")
        if len(target_masks) != images.shape[0]:
            raise ValueError("one reconciled mask is required per view")
        if not 0 <= anchor_view < images.shape[0]:
            raise IndexError("anchor_view is outside the image batch")
        for mask in target_masks:
            if mask.shape != images.shape[-2:]:
                raise ValueError("target masks must match image resolution")

        edited = [None] * images.shape[0]
        anchor_result = None
        order = [anchor_view] + [index for index in range(images.shape[0])
                     if index != anchor_view]
        for view_index in order:
            view_references = references or []
            if anchor_result is not None:
                view_references = [anchor_result] + list(view_references)
            result = self.adapter.edit(
                images[view_index:view_index + 1], instruction,
                target_mask=target_masks[view_index].unsqueeze(0),
                references=view_references, **editor_kwargs
            )
            if result.shape != images[view_index:view_index + 1].shape:
                raise ValueError("adapter returned an image with the wrong shape")
            edited[view_index] = result[0]
            if view_index == anchor_view:
                anchor_result = result[0].detach()

        proposals = torch.stack(edited, dim=0)
        if self.enable_gaxlc and self.coupling is not None:
            self.coupling.target_masks = {
                index: mask.to(proposals.device)
                for index, mask in enumerate(target_masks)
            }
            proposals = self.coupling.couple(proposals, anchor_index=anchor_view)

        # Final safety net. Generation-time preservation is backend-specific;
        # unsupported adapters must still never alter the protected complement.
        if self.enable_prlp:
            mask = torch.stack(target_masks).to(proposals.device, proposals.dtype).unsqueeze(1)
            proposals = mask * proposals + (1.0 - mask) * images.to(proposals.device)
        return proposals

    def set_correspondence(self, correspondence_maps):
        """Set renderer-produced latent/image maps before editing."""
        self.coupling = GeometryAwareCrossViewCoupling(
            correspondence_maps, {}, strength=self.coupling_strength,
            enable=self.enable_gaxlc,
        )
