"""Canonical Gaussian appearance anchoring.

The renderer supplies a latent-resolution correspondence and visibility map.
Features are transported only where a Gaussian is visible in both views; no
unverified scatter over projected Gaussian centers is used.
"""

import torch


class CanonicalGaussianFeatureAnchoring:
    def __init__(self, enable=True):
        self.disabled = not enable
        self._canonical = None

    def set_canonical_features(self, features):
        if self.disabled:
            return
        if features.ndim != 3:
            raise ValueError("canonical features must have shape (C,H,W)")
        self._canonical = features.detach()

    def project_features_to_view(self, source_yx, valid, output_shape=None):
        if self.disabled:
            return None
        if self._canonical is None:
            raise RuntimeError("canonical features have not been set")
        if source_yx.ndim != 3 or source_yx.shape[-1] != 2:
            raise ValueError("source_yx must have shape (H,W,2) in y,x order")
        if valid.shape != source_yx.shape[:2]:
            raise ValueError("valid correspondence map shape mismatch")
        height, width = source_yx.shape[:2]
        if output_shape is not None and tuple(output_shape) != (height, width):
            raise ValueError("output_shape disagrees with correspondence map")
        source_y = source_yx[..., 0].clamp(0, self._canonical.shape[-2] - 1)
        source_x = source_yx[..., 1].clamp(0, self._canonical.shape[-1] - 1)
        projected = self._canonical[:, source_y, source_x]
        return projected * valid.to(projected.dtype).unsqueeze(0)
