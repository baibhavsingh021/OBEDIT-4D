"""Geometry contracts for canonical Gaussian correspondence."""

import torch


def project_gaussians_to_view(gaussian_positions, camera, image_width, image_height):
    if gaussian_positions.ndim != 2 or gaussian_positions.shape[1] != 3:
        raise ValueError("gaussian_positions must have shape (N, 3)")
    ones = torch.ones((gaussian_positions.shape[0], 1), device=gaussian_positions.device,
                      dtype=gaussian_positions.dtype)
    world_h = torch.cat((gaussian_positions, ones), dim=1)
    view = camera.world_view_transform.to(gaussian_positions.device)
    projection = camera.full_proj_transform.to(gaussian_positions.device)
    camera_h = torch.matmul(world_h, view.t())
    screen_h = torch.matmul(camera_h, projection.t())
    w = screen_h[:, 3]
    safe_w = torch.where(w.abs() > 1e-8, w, torch.ones_like(w))
    ndc = screen_h[:, :3] / safe_w[:, None]
    pixels = torch.stack((
        ((ndc[:, 0] + 1) * 0.5 * image_width),
        ((1 - ndc[:, 1]) * 0.5 * image_height),
    ), dim=1)
    pixels_int = pixels.floor().long()
    valid = (camera_h[:, 2] > 0) & (w > 0) & (pixels_int[:, 0] >= 0) & \
        (pixels_int[:, 0] < image_width) & (pixels_int[:, 1] >= 0) & \
        (pixels_int[:, 1] < image_height)
    return pixels_int, valid


def compute_cross_view_correspondence(gaussian_positions, camera_a, camera_b,
                                      image_width, image_height):
    pixels_a, valid_a = project_gaussians_to_view(
        gaussian_positions, camera_a, image_width, image_height)
    pixels_b, valid_b = project_gaussians_to_view(
        gaussian_positions, camera_b, image_width, image_height)
    valid = valid_a & valid_b
    return pixels_a, pixels_b, valid
