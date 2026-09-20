"""Trajectory and protected-region terms for the existing refinement stage."""

import torch
import torch.nn.functional as F


class TrajectoryAwareSDS:
    def __init__(self, trajectory_provider=None, target_indices=None,
                 lambda_trajectory=0.1, lambda_protection=1.0, enable=True):
        self.trajectory_provider = trajectory_provider
        self.target_indices = target_indices
        self.lambda_trajectory = lambda_trajectory
        self.lambda_protection = lambda_protection
        self.disabled = not enable
        self._canonical_target = None
        self._protected_reference = None

    def set_references(self, canonical_target, protected_positions=None):
        self._canonical_target = canonical_target.detach().clone()
        if protected_positions is not None:
            self._protected_reference = protected_positions.detach().clone()

    def regularization(self, current_positions, timestep, protected_positions=None):
        device = current_positions.device
        total = torch.zeros((), device=device, dtype=current_positions.dtype)
        terms = {"trajectory": total, "protection": total}
        if self.disabled:
            return total, terms
        if self.trajectory_provider is not None and self._canonical_target is not None:
            expected = self.trajectory_provider(self._canonical_target, timestep)
            if expected.shape != current_positions.shape:
                raise ValueError("trajectory provider returned an incompatible position shape")
            terms["trajectory"] = F.mse_loss(current_positions, expected)
            total = total + self.lambda_trajectory * terms["trajectory"]
        reference = protected_positions if protected_positions is not None else self._protected_reference
        if reference is not None:
            if reference.shape != current_positions.shape:
                raise ValueError("protected position reference shape mismatch")
            terms["protection"] = F.mse_loss(current_positions, reference)
            total = total + self.lambda_protection * terms["protection"]
        return total, terms

    def add_to_loss(self, base_loss, current_positions, timestep, protected_positions=None):
        regularizer, terms = self.regularization(current_positions, timestep, protected_positions)
        total = base_loss + regularizer
        terms["base"] = base_loss.detach()
        terms["total"] = total.detach()
        return total, terms
