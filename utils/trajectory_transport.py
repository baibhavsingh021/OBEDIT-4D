"""Sparse feature transport along frozen Gaussian trajectories."""

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class TrajectoryProjection:
    grid: Tensor
    confidence: Tensor
    direction: Tensor
    time: float


@torch.no_grad()
def project_trajectories(camera, geometry, depth_tolerance=0.03):
    points = geometry["means3D"].float()
    homogeneous = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)
    clip = homogeneous @ camera.full_proj_transform.to(points)
    z = (homogeneous @ camera.world_view_transform.to(points))[:, 2]
    grid = clip[:, :2] / clip[:, 3:].clamp_min(1e-6)
    finite = torch.isfinite(grid).all(dim=-1) & torch.isfinite(z)
    inside = finite & (grid.abs() < 1).all(dim=-1) & (z > camera.znear)
    grid = torch.nan_to_num(grid, nan=2.0, posinf=2.0, neginf=-2.0).clamp(-2, 2)

    def sample(image):
        return F.grid_sample(
            image[None].float(), grid[None, :, None], align_corners=False,
            padding_mode="zeros",
        )[0, 0, :, 0]

    depth = sample(geometry["surface_depth"])
    alpha = sample(geometry["alpha"])
    variance = sample(geometry["depth_variance"])
    tolerance = depth.abs().clamp_min(1e-3) * depth_tolerance
    depth_error = (z - depth).abs() / tolerance
    # Reject mixed surfaces and occlusions rather than blending their features.
    visible = inside & geometry["visibility_filter"] & (alpha > 0.5)
    visible &= (depth_error < 2) & (variance < (2 * tolerance).square())
    confidence = torch.where(visible, torch.exp(-0.5 * depth_error.square()), 0.0)
    confidence *= alpha * geometry["opacity"].flatten().float()
    confidence = torch.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0)
    direction = F.normalize(camera.camera_center.to(points)[None] - points, dim=-1)
    return TrajectoryProjection(grid, confidence, direction, float(camera.time))


class TrajectoryTransport:
    """O(frames * Gaussians) gather/splat; no pixel-by-pixel attention matrix."""

    def __init__(self, projections: List[TrajectoryProjection], size: Tuple[int, int]):
        if not projections:
            raise ValueError("At least one calibrated projection is required")
        self.projections = projections
        self.height, self.width = size
        self.weights = torch.stack([p.confidence for p in projections])
        self.stencils = [self._stencil(p.grid) for p in projections]

    def _stencil(self, grid):
        x = (grid[:, 0] + 1) * self.width / 2 - 0.5
        y = (grid[:, 1] + 1) * self.height / 2 - 0.5
        x0, y0 = x.floor(), y.floor()
        indices, weights = [], []
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            ix, iy = x0 + dx, y0 + dy
            valid = (ix >= 0) & (ix < self.width) & (iy >= 0) & (iy < self.height)
            indices.append((iy.clamp(0, self.height - 1) * self.width
                            + ix.clamp(0, self.width - 1)).long())
            weights.append((1 - (x - ix).abs()) * (1 - (y - iy).abs()) * valid)
        return torch.stack(indices), torch.stack(weights)

    def gather_frame(self, image: Tensor, index: int) -> Tensor:
        projection = self.projections[index]
        return F.grid_sample(
            image[None].float(), projection.grid[None, :, None],
            align_corners=False, padding_mode="zeros",
        )[0, :, :, 0].transpose(0, 1)

    def gather(self, images: Tensor) -> Tensor:
        return torch.stack([self.gather_frame(image, i) for i, image in enumerate(images)])

    def splat_frame(self, values: Tensor, confidence: Tensor, index: int):
        indices, bilinear = self.stencils[index]
        weights = bilinear * confidence[None]
        output = values.new_zeros(values.shape[1], self.height * self.width)
        mass = values.new_zeros(1, self.height * self.width)
        for corner in range(4):
            output.scatter_add_(
                1, indices[corner][None].expand(values.shape[1], -1),
                values.transpose(0, 1) * weights[corner][None],
            )
            mass.scatter_add_(1, indices[corner][None], weights[corner][None])
        output = output / mass.clamp_min(1e-6)
        return output.reshape(-1, self.height, self.width), mass.reshape(1, self.height, self.width)

    def consensus(self, residuals: Tensor, memory: Tensor, memory_weight: Tensor,
                  strength: float, memory_strength: float) -> Tensor:
        features = self.gather(residuals)
        outputs = []
        for i, projection in enumerate(self.projections):
            numerator = torch.zeros_like(features[i])
            denominator = torch.zeros_like(self.weights[i])
            for j, other in enumerate(self.projections):
                if i == j:
                    continue
                angle = (projection.direction * other.direction).sum(-1).clamp(-1, 1)
                # Camera/time priors attend only to the same canonical Gaussian.
                weight = self.weights[j] * torch.exp(2 * (angle - 1))
                weight *= 1 / (1 + abs(projection.time - other.time))
                disagreement = (features[j] - features[i]).square().mean(-1)
                weight *= (1 + disagreement).rsqrt()
                numerator += features[j] * weight[:, None]
                denominator += weight
            weight = memory_strength * memory_weight
            numerator += memory * weight[:, None]
            denominator += weight
            target = numerator / denominator[:, None].clamp_min(1e-6)
            correction = target - features[i]
            valid = self.weights[i] * denominator.clamp(0, 1)
            delta, mass = self.splat_frame(correction, valid, i)
            outputs.append(residuals[i].float() + strength * mass.clamp(0, 1) * delta)
        return torch.stack(outputs).to(residuals.dtype)

    @torch.no_grad()
    def shared_noise(self, channels: int, generator: torch.Generator, correlation=0.5):
        count = self.weights.shape[1]
        canonical = torch.randn(
            count, channels, device=self.weights.device, generator=generator,
        )
        frames = []
        for i in range(len(self.projections)):
            indices, bilinear = self.stencils[i]
            weights = bilinear * self.weights[i][None]
            shared = canonical.new_zeros(channels, self.height * self.width)
            variance = canonical.new_zeros(1, self.height * self.width)
            for corner in range(4):
                shared.scatter_add_(
                    1, indices[corner][None].expand(channels, -1),
                    canonical.T * weights[corner][None],
                )
                variance.scatter_add_(1, indices[corner][None], weights[corner][None].square())
            # Normalize variance, not total weight, to retain unit Gaussian noise.
            shared /= variance.clamp_min(1e-8).sqrt()
            rho = correlation * (variance > 1e-8).float()
            independent = torch.randn(shared.shape, device=shared.device, generator=generator)
            noise = rho.sqrt() * shared + (1 - rho).sqrt() * independent
            frames.append(noise.reshape(channels, self.height, self.width))
        return torch.stack(frames)

    def consistency_loss(self, residual: Tensor, index: int, peers: Tensor, temporal: bool):
        current = self.gather_frame(residual, index)
        numerator = current.sum() * 0
        denominator = current.new_zeros(())
        for j, projection in enumerate(self.projections):
            is_temporal = abs(projection.time - self.projections[index].time) > 1e-6
            if j == index or is_temporal != temporal:
                continue
            weight = self.weights[index] * self.weights[j]
            angle = (projection.direction * self.projections[index].direction).sum(-1).clamp(-1, 1)
            weight = weight * torch.exp(2 * (angle - 1))
            weight = weight / (1 + abs(projection.time - self.projections[index].time))
            error = ((current - peers[j]).square() + 1e-6).sqrt().mean(-1) - 1e-3
            numerator = numerator + (weight * error).sum()
            denominator = denominator + weight.sum()
        return numerator / denominator.clamp_min(1)


class CanonicalResidualMemory:
    """Bounded EMA of edit residuals keyed by immutable Gaussian row IDs."""

    def __init__(self, count: int, channels: int, device):
        self.features = torch.zeros(count, channels, device=device)
        self.confidence = torch.zeros(count, device=device)

    @torch.no_grad()
    def update(self, transport: TrajectoryTransport, residuals: Tensor, decay=0.95):
        samples = transport.gather(residuals)
        weights = transport.weights
        mass = weights.sum(0)
        mean = (samples * weights[:, :, None]).sum(0) / mass[:, None].clamp_min(1e-6)
        variance = ((samples - mean[None]).square().mean(-1) * weights).sum(0)
        confidence = mass.clamp(0, 1) / (1 + variance / mass.clamp_min(1e-6))
        rate = torch.where(self.confidence > 0, 1 - decay, 1.0) * confidence
        self.features.lerp_(mean, rate[:, None])
        self.confidence = torch.maximum(self.confidence, confidence)
