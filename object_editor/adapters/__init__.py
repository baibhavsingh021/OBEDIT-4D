"""Diffusion editor adapters."""

from .base_adapter import BaseEditorAdapter, EditorCapabilities
from .ip2p_adapter import IP2PAdapter
from .omnigen_adapter import OmniGenAdapter
from .sdxl_adapter import SDXLAdapter

EDITOR_REGISTRY = {
    "omnigen": OmniGenAdapter,
    "sdxl": SDXLAdapter,
    "ip2p": IP2PAdapter,
}


def get_editor(name):
    if name not in EDITOR_REGISTRY:
        raise ValueError(
            "Unknown editor '{}'. Available: {}".format(
                name, sorted(EDITOR_REGISTRY)
            )
        )
    return EDITOR_REGISTRY[name]
