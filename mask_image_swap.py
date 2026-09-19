"""
    Originals: ./data/{dataset}/time0_{scene_name}/original_time0_{i}.png
    Edits:     ./data/{dataset}/{scene_name}/{tag}/edited_{tag}_original_time0_{i}.png

    python mask_edit.py --dataset dynerf --scene_name cook_spinach \
        --prompt "What if it was wearing sunglasses?" \
        --guidance_scale 7.5 --image_guidance_scale 1.5

    -- Dependencies --
    pip install -q transformers accelerate opencv-python matplotlib pillow
    pip install git+https://github.com/facebookresearch/segment-anything-2.git
    wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
"""

import argparse
import gc
import os

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ----------------------------------------------------------------------------
# Fixed configuration
# ----------------------------------------------------------------------------

DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
SAM2_CONFIG_FILE = "sam2_hiera_l.yaml"
SAM2_CKPT_PATH = "sam2_hiera_large.pt"
MAX_SIDE = 800
DEFAULT_NUM_CAMS = 21


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Mask and composite multiview_edit.py outputs against time0 originals."
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--num_cams", type=int, default=DEFAULT_NUM_CAMS,
                         help="Number of cameras/images in the scene (default: 21).")
    parser.add_argument("--text_threshold", type=float, default=0.2)
    # Accepted for pipeline call-signature symmetry; unused here.
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--image_guidance_scale", type=float, default=None)
    return parser.parse_args()


def get_tag(prompt: str) -> str:
    return prompt.split(" ")[-1].replace("?", "")


# ----------------------------------------------------------------------------
# Image loading
# ----------------------------------------------------------------------------

def load_and_resize(img_path: str, max_side: int = MAX_SIDE) -> np.ndarray:
    image = cv2.imread(img_path)
    assert image is not None, f"Could not load image: {img_path}"
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_pairs(dataset: str, scene_name: str, tag: str, num_cams: int):
    """Load (original, edited, edited_path) triples for each camera index."""
    originals_dir = f"./data/{dataset}/time0_{scene_name}"
    edits_dir = f"./data/{dataset}/{scene_name}/{tag}"

    pairs = []
    for i in range(num_cams):
        orig_path = os.path.join(originals_dir, f"original_time0_{i}.png")
        edit_path = os.path.join(edits_dir, f"edited_{tag}_original_time0_{i}.png")

        if not os.path.exists(orig_path):
            print(f"⚠️  Skipping cam {i}: missing original {orig_path}")
            continue
        if not os.path.exists(edit_path):
            print(f"⚠️  Skipping cam {i}: missing edit {edit_path}")
            continue

        original = load_and_resize(orig_path)
        edited = load_and_resize(edit_path)
        edited = cv2.resize(edited, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_CUBIC)
        pairs.append((i, original, edited, edit_path))

    return pairs


# ----------------------------------------------------------------------------
# GroundingDINO detection
# ----------------------------------------------------------------------------

def load_dino():
    processor = AutoProcessor.from_pretrained(DINO_MODEL)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL).to("cuda").eval()
    return processor, dino


def detect(image: np.ndarray, processor, dino, text_prompt: str, text_threshold: float):
    """Run GroundingDINO on a single image, return boxes/scores/labels."""
    pil_image = Image.fromarray(image)
    inputs = processor(images=pil_image, text=text_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = dino(**inputs)

    H, W = image.shape[:2]
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        text_threshold=text_threshold,
        target_sizes=[(H, W)],
    )[0]

    return results["boxes"], results["scores"], results["labels"]


# ----------------------------------------------------------------------------
# SAM2 segmentation
# ----------------------------------------------------------------------------

def load_sam2_predictor():
    sam2 = build_sam2(
        config_file=SAM2_CONFIG_FILE,
        ckpt_path=SAM2_CKPT_PATH,
        device="cuda",
    )
    return SAM2ImagePredictor(sam2)


def predict_best_mask(predictor, image: np.ndarray, boxes_xyxy, scores) -> np.ndarray:
    """Run SAM2 on the highest-scoring DINO box, return mask of the detected object (1=object)."""
    predictor.set_image(image)

    idx = scores.argmax().item()
    x1, y1, x2, y2 = boxes_xyxy[idx].cpu().numpy()
    sam_box = np.array([[x1, y1, x2, y2]])

    masks, scores_sam, _ = predictor.predict(box=sam_box, multimask_output=True)
    return masks[scores_sam.argmax()]


# ----------------------------------------------------------------------------
# Compositing
# ----------------------------------------------------------------------------

def composite(original: np.ndarray, edited: np.ndarray, object_mask: np.ndarray) -> np.ndarray:
    """
    Keep the EDITED pixels where the detected object is, ORIGINAL pixels everywhere else.
    object_mask: 1 where the detected/edited object is, 0 elsewhere.
    """
    mask_3d = object_mask[:, :, np.newaxis].astype(bool)
    return np.where(mask_3d, edited, original)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    args = parse_args()
    tag = get_tag(args.prompt)

    print("------------------------------------------")
    print(f"  - dataset: {args.dataset}")
    print(f"  - scene: {args.scene_name}")
    print(f"  - tag (mask prompt): \"{tag}\"")
    print(f"  - num_cams: {args.num_cams}")
    print("------------------------------------------\n")

    pairs = load_pairs(args.dataset, args.scene_name, tag, args.num_cams)
    if not pairs:
        print("❌ No valid original/edited image pairs found. Exiting.")
        return

    print(f"Loaded {len(pairs)} original/edited pairs.\n")

    print("Loading GroundingDINO...")
    processor, dino = load_dino()

    # DINO needs a trailing period.
    text_prompt = f"{tag}."

    detections = []
    for i, original, edited, edit_path in pairs:
        boxes, scores, labels = detect(edited, processor, dino, text_prompt, args.text_threshold)
        print(f"Cam {i}: detected {labels} ({len(boxes)} box(es))")
        detections.append((i, original, edited, edit_path, boxes, scores))

    # Free DINO before loading SAM2.
    del dino
    torch.cuda.empty_cache()
    gc.collect()

    print("\nLoading SAM2...")
    predictor = load_sam2_predictor()

    saved = 0
    mask_dir = os.path.join(f"./data/{args.dataset}/{args.scene_name}/{tag}", "masks")
    os.makedirs(mask_dir, exist_ok=True)
    for i, original, edited, edit_path, boxes, scores in detections:
        if len(boxes) == 0:
            print(f"Cam {i}: no '{tag}' detected, preserving the original image.")
            mask = np.zeros(original.shape[:2], dtype=bool)
        else:
            mask = predict_best_mask(predictor, edited, boxes, scores)
        result = composite(original, edited, mask)

        Image.fromarray(mask.astype(np.uint8) * 255).save(os.path.join(mask_dir, f"mask_cam{i:02d}.png"))
        Image.fromarray(result.astype(np.uint8)).save(edit_path)
        saved += 1

    print(f"\n✅ Composited and saved {saved}/{len(pairs)} images back to "
          f"./data/{args.dataset}/{args.scene_name}/{tag}/")


if __name__ == "__main__":
    main()
