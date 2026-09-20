"""Geometry-aware coupling of image-space edit proposals.

The coupling operates on *clean edit proposals*, not scheduler noise. A
correspondence map must be produced by the 4DGS renderer/camera path; this
prevents invalid pixel matching when depth or occlusion is unknown.
"""

import torch


class GeometryAwareCrossViewCoupling:
    def __init__(self, correspondence_maps, target_masks, strength=0.7, enable=True):
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        self.disabled = not enable
        self.maps = correspondence_maps
        self.target_masks = target_masks
        self.strength = strength

    def couple(self, proposals, anchor_index=0, confidence=None):
        if self.disabled:
            return proposals
        if proposals.ndim != 4:
            raise ValueError("proposals must have shape (V,C,H,W)")
        if not 0 <= anchor_index < proposals.shape[0]:
            raise IndexError("anchor_index is outside the proposal batch")
        output = proposals.clone()
        anchor = proposals[anchor_index]
        for view_index in range(proposals.shape[0]):
            if view_index == anchor_index:
                continue
            key = (anchor_index, view_index)
            if key not in self.maps:
                raise KeyError("missing 4DGS correspondence map {}".format(key))
            source_yx, valid = self.maps[key]
            if source_yx.shape[:2] != proposals.shape[-2:] or valid.shape != source_yx.shape[:2]:
                raise ValueError("correspondence map must match latent proposal resolution")
            mask = self.target_masks[view_index].to(proposals.device)
            if mask.shape != valid.shape:
                raise ValueError("target mask and correspondence resolution differ")
            sampled = anchor[:, source_yx[..., 0], source_yx[..., 1]]
            weight = self.strength * valid.to(proposals.dtype) * mask.to(proposals.dtype)
            if confidence is not None:
                weight = weight * confidence[view_index].to(proposals.device)
            output[view_index] = output[view_index] * (1 - weight.unsqueeze(0)) + sampled * weight.unsqueeze(0)
        return output
