#!/bin/bash
set -euo pipefail
# ===================================================================
# ./run_instruct_4dgs.sh [dataset] [scene_name] [prompt] [guidance_scale] [image_guidance_scale]
# ===================================================================

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <dataset> <scene_name> <prompt> <guidance_scale> <image_guidance_scale>"
    exit 1
fi

DATASET="$1"
SCENE_NAME="$2"
PROMPT="$3"
GUIDANCE_SCALE="$4"
IMAGE_GUIDANCE_SCALE="$5"
EDIT_TAG="${PROMPT##* }"
EDIT_TAG="${EDIT_TAG//\?/}"

echo "------------------------------------------"
echo "  - dataset: ${DATASET}"
echo "  - scene: ${SCENE_NAME}"
echo "  - prompt: \"${PROMPT}\""
echo "------------------------------------------"
echo ""

echo "[1/5] Collect time0 images..."
python time0_collect.py --dataset "${DATASET}" --scene_name "${SCENE_NAME}"
echo ""
echo "[2/5] edit time0 images..."
python ./ip2p_models/multiview_edit.py \
    --dataset "${DATASET}" \
    --scene "${SCENE_NAME}" \
    --prompt "${PROMPT}" \
    --resize 512 \
    --steps 20 \
    --guidance_scale ${GUIDANCE_SCALE} \
    --image_guidance_scale ${IMAGE_GUIDANCE_SCALE}
echo "✅ Completed time0 image editing."
echo ""

echo "[3/5] Mask + composite edited images against originals..."
python mask_image_swap.py \
    --dataset "${DATASET}" \
    --scene_name "${SCENE_NAME}" \
    --prompt "${PROMPT}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --image_guidance_scale "${IMAGE_GUIDANCE_SCALE}"
if [ $? -ne 0 ]; then
    echo "❌ Masking step failed. Exiting."
    exit 1
fi
echo "✅ Completed masking and compositing."
echo ""

echo "[4/5] 3D editing"
# Dynamically find the highest iteration in point_cloud/
POINT_CLOUD_DIR="./output/${DATASET}/${SCENE_NAME}/point_cloud"
BEST_ITER_3D=$(ls -d "${POINT_CLOUD_DIR}"/iteration_* 2>/dev/null | sed 's/.*iteration_//' | sort -n | tail -1)
if [ -z "$BEST_ITER_3D" ]; then
    echo "❌ No iteration folders found in ${POINT_CLOUD_DIR}. Exiting."
    exit 1
fi
echo "   → Selected highest iteration: ${BEST_ITER_3D} from ${POINT_CLOUD_DIR}"
python edit_3d.py \
    --configs "./arguments/${DATASET}/${SCENE_NAME}.py" \
    --ply_path "${POINT_CLOUD_DIR}/iteration_${BEST_ITER_3D}/point_cloud.ply" \
    -s "./data/${DATASET}/${SCENE_NAME}" \
    --model_path "./output/${DATASET}/${SCENE_NAME}" \
    --dataset "${DATASET}" \
    --scene "${SCENE_NAME}" \
    --prompt "${PROMPT}"
echo "✅ Completed 3d editing."
echo ""

echo "[5/5] Gaussian trajectory refinement"
# Dynamically find the highest iteration in point_cloud_3dedit/<prompt>/
POINT_CLOUD_3DEDIT_DIR="./output/${DATASET}/${SCENE_NAME}/point_cloud_3dedit/${PROMPT}"
BEST_ITER_SDS=$(ls -d "${POINT_CLOUD_3DEDIT_DIR}"/iteration_* 2>/dev/null | sed 's/.*iteration_//' | sort -n | tail -1)
if [ -z "$BEST_ITER_SDS" ]; then
    echo "❌ No iteration folders found in ${POINT_CLOUD_3DEDIT_DIR}. Exiting."
    exit 1
fi
echo "   → Selected highest iteration: ${BEST_ITER_SDS} from ${POINT_CLOUD_3DEDIT_DIR}"
python refine_sds.py \
    --configs "./arguments/${DATASET}/${SCENE_NAME}.py" \
    --ply_path "${POINT_CLOUD_3DEDIT_DIR}/iteration_${BEST_ITER_SDS}/point_cloud.ply" \
    -s "./data/${DATASET}/${SCENE_NAME}" \
    --model_path "./output/${DATASET}/${SCENE_NAME}" \
    --prompt "${PROMPT}" \
    --refinement_mode trajectory \
    --refine_views 2 \
    --refine_times 2 \
    --refine_resolution 512 \
    --refine_target_interval 8 \
    --refine_mask_dir "./data/${DATASET}/${SCENE_NAME}/${EDIT_TAG}/masks" \
    --guidance_scale ${GUIDANCE_SCALE} \
    --image_guidance_scale ${IMAGE_GUIDANCE_SCALE}
echo "✅ Completed score refinement."
echo ""

echo "🎉 All pipeline steps have been executed."
