"""Executable shape and device contracts for unvalidated integration code."""

import torch


def assert_tensor(tensor, name, expected_dims=None, expected_shape=None,
                  expected_dtype=None, expected_device=None):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("{} must be a torch.Tensor".format(name))
    if expected_dims is not None and tensor.ndim != expected_dims:
        raise ValueError("{} expected {} dimensions, got {} ({})".format(
            name, expected_dims, tensor.ndim, tuple(tensor.shape)))
    if expected_shape is not None:
        if len(expected_shape) != tensor.ndim:
            raise ValueError("{} expected shape template {}".format(name, expected_shape))
        for actual, expected in zip(tensor.shape, expected_shape):
            if expected is not None and actual != expected:
                raise ValueError("{} shape mismatch: {} vs {}".format(
                    name, tuple(tensor.shape), expected_shape))
    if expected_dtype is not None and tensor.dtype != expected_dtype:
        raise TypeError("{} expected {}, got {}".format(name, expected_dtype, tensor.dtype))
    if expected_device is not None and tensor.device.type != expected_device:
        raise ValueError("{} expected device {}, got {}".format(
            name, expected_device, tensor.device))
    return tensor


def assert_mask_valid(mask, name="mask", height=None, width=None):
    assert_tensor(mask, name, expected_dims=2)
    if height is not None and mask.shape[0] != height:
        raise ValueError("{} has height {}, expected {}".format(name, mask.shape[0], height))
    if width is not None and mask.shape[1] != width:
        raise ValueError("{} has width {}, expected {}".format(name, mask.shape[1], width))
    if not torch.is_floating_point(mask):
        raise TypeError("{} must be a floating mask in [0, 1]".format(name))
    if torch.any(mask < 0) or torch.any(mask > 1):
        raise ValueError("{} must be in [0, 1]".format(name))


def assert_camera_consistency(cameras, num_views):
    if len(cameras) != num_views:
        raise ValueError("expected {} cameras, got {}".format(num_views, len(cameras)))
    for index, camera in enumerate(cameras):
        for attribute in ("world_view_transform", "full_proj_transform",
                          "image_width", "image_height"):
            if not hasattr(camera, attribute):
                raise AttributeError("camera {} lacks {}".format(index, attribute))
