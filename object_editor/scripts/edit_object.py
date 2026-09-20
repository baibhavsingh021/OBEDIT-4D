"""Standalone image/mask editor bridge for synchronized 4DGS renders.

The 4DGS process writes one synchronized image and one reconciled mask per
view into folders. This script edits those folders and writes the result back
in the same order. Gaussian fitting and temporal refinement remain owned by
the existing 4DGS entry points.
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as transforms

from object_editor.adapters import get_editor
from object_editor.core import EditType, PreservationMode, TargetSpec
from object_editor.pipeline import ObjectEditorPipeline


def build_parser():
    parser = argparse.ArgumentParser(description="Edit synchronized 4DGS images")
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--editor_model", choices=("omnigen", "sdxl", "ip2p"), default="omnigen")
    parser.add_argument("--editor_ckpt", default="BAAI/OmniGen-v1")
    parser.add_argument("--target_query", default="")
    parser.add_argument("--edit_instruction", required=True)
    parser.add_argument("--edit_type", choices=[item.value for item in EditType], default="appearance")
    parser.add_argument("--preservation_mode", choices=[item.value for item in PreservationMode], default="strict")
    parser.add_argument("--disable_cgfa", action="store_true")
    parser.add_argument("--disable_gaxlc", action="store_true")
    parser.add_argument("--disable_prlp", action="store_true")
    parser.add_argument("--disable_tasds", action="store_true")
    parser.add_argument("--coupling_strength", type=float, default=0.7)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--no_cpu_offload", action="store_true")
    parser.add_argument("--no_vae_tiling", action="store_true")
    return parser


def make_target_spec(args, manual_masks=None):
    spec = TargetSpec(
        text_query=args.target_query,
        instruction=args.edit_instruction,
        edit_type=EditType(args.edit_type),
        preservation=PreservationMode(args.preservation_mode),
        manual_masks=manual_masks,
    )
    spec.validate()
    return spec


def main(argv=None):
    args = build_parser().parse_args(argv)
    image_paths = sorted(Path(args.images_dir).glob("*.png"))
    mask_paths = sorted(Path(args.mask_dir).glob("*.png"))
    if not image_paths:
        raise FileNotFoundError("No PNG images found in {}".format(args.images_dir))
    if len(image_paths) != len(mask_paths):
        raise ValueError("images and masks must contain the same number of PNG files")
    to_tensor = transforms.ToTensor()
    images = torch.stack([
        to_tensor(Image.open(path).convert("RGB")) * 2.0 - 1.0
        for path in image_paths
    ])
    masks = [to_tensor(Image.open(path).convert("L")) for path in mask_paths]
    masks = [mask.squeeze(0).clamp(0.0, 1.0) for mask in masks]
    spec = make_target_spec(args, {
        index: mask.cpu().numpy() for index, mask in enumerate(masks)
    })
    editor_class = get_editor(args.editor_model)
    common = {"device": "cuda"}
    if args.editor_model in ("omnigen", "sdxl"):
        common["model_path"] = args.editor_ckpt
        common["enable_cpu_offload"] = not args.no_cpu_offload
    if args.editor_model == "omnigen":
        common["vae_tiling"] = not args.no_vae_tiling
    editor = editor_class(**common)
    pipeline = ObjectEditorPipeline(
        editor,
        enable_cgfa=not args.disable_cgfa,
        enable_gaxlc=not args.disable_gaxlc,
        enable_prlp=not args.disable_prlp,
        coupling_strength=args.coupling_strength,
    )
    edited = pipeline.edit_views(
        images, masks, spec.instruction,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )
    output_dir = Path(args.output_dir) / spec.get_run_name()
    output_dir.mkdir(parents=True, exist_ok=True)
    to_pil = transforms.ToPILImage()
    for path, image in zip(image_paths, edited):
        to_pil(((image.detach().cpu().float() + 1.0) / 2.0).clamp(0, 1)).save(
            output_dir / path.name
        )
    print("Saved {} edited views to {}".format(len(edited), output_dir))
    return output_dir


if __name__ == "__main__":
    main()
