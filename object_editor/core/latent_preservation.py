"""Protected-region preservation using the active scheduler's forward process."""

import torch


class ProtectedRegionLatentPreservation:
    def __init__(self, target_masks, enable=True):
        self.disabled = not enable
        self.target_masks = target_masks

    def preserve(self, predicted_latents, original_noised_latents, view_indices=None):
        if self.disabled:
            return predicted_latents
        if predicted_latents.shape != original_noised_latents.shape:
            raise ValueError("predicted and original noised latents must have identical shapes")
        indices = list(range(predicted_latents.shape[0])) if view_indices is None else list(view_indices)
        if len(indices) != predicted_latents.shape[0]:
            raise ValueError("view_indices must match latent batch size")
        result = predicted_latents.clone()
        for batch_index, view_index in enumerate(indices):
            mask = self.target_masks[view_index].to(result.device, result.dtype)
            if mask.shape != result.shape[-2:]:
                raise ValueError("target mask must already be at latent resolution")
            mask = mask.unsqueeze(0)
            result[batch_index] = mask * result[batch_index] + (1 - mask) * original_noised_latents[batch_index]
        return result

    @staticmethod
    def forward_original(scheduler, clean_latents, noise, timestep):
        """Use the scheduler's exact parameterization, rather than DDPM assumptions."""
        return scheduler.add_noise(clean_latents, noise, timestep)
