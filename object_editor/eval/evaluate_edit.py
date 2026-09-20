"""Small metric contracts; full dataset evaluation stays in project scripts."""

import torch


def masked_l1(predicted, reference, mask):
    if predicted.shape != reference.shape:
        raise ValueError("predicted and reference shapes differ")
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[0] != predicted.shape[0] or mask.shape[-2:] != predicted.shape[-2:]:
        raise ValueError("mask does not match image batch/spatial shape")
    error = (predicted - reference).abs() * mask.to(predicted.dtype)
    return error.sum() / mask.sum().clamp_min(1.0)


def protected_l1(edited, original, target_mask):
    if target_mask.ndim == 3:
        target_mask = target_mask.unsqueeze(1)
    return masked_l1(edited, original, 1.0 - target_mask)
