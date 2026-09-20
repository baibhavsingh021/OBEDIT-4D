"""Explicit T4 memory policy; no silent fallback to an unsuitable mode."""

import gc
import torch


class MemoryManager:
    def __init__(self, total_vram_gb=16.0, safety_margin_gb=2.0,
                 sequential_views=True):
        if total_vram_gb <= safety_margin_gb:
            raise ValueError("safety margin must be smaller than total VRAM")
        self.total_vram_gb = total_vram_gb
        self.safety_margin_gb = safety_margin_gb
        self.sequential_views = sequential_views

    @property
    def budget_gb(self):
        return self.total_vram_gb - self.safety_margin_gb

    def check(self, operation, estimated_gb):
        used = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
        if used + estimated_gb > self.budget_gb:
            raise RuntimeError(
                "T4 memory budget exceeded for {}: {:.2f} + {:.2f} > {:.2f} GB".format(
                    operation, used, estimated_gb, self.budget_gb))

    def clear(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
