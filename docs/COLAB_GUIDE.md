# Colab Guide

This guide describes the current `object_editor` integration in one Colab
notebook and one runtime. It uses one modern PyTorch environment; do not
install the old pinned Diffusers requirements after the unified requirements.

The current integration is staged:

1. The 4DGS code trains/loads a scene and produces synchronized images
   and reconciled masks on disk.
2. The object editor edits those PNGs and writes edited PNGs.
3. The same runtime consumes the edited PNGs for canonical fitting and
   existing temporal refinement.
4. The existing renderer produces the final 4D output.

`object_editor/scripts/edit_object.py` does not currently load a 4DGS scene,
run Grounded-SAM, build renderer correspondence maps, or launch `edit_3d.py`
itself. Those are deliberate process boundaries, not automatic steps.

## 0. Colab prerequisites

Use a GPU runtime with a T4 or better and enough Google Drive space for the
scene, checkpoints, and rendered image folders. In Colab select **Runtime >
Change runtime type > T4 GPU**. Confirm that the runtime has CUDA before
installing anything. Do not install the standalone object-editor requirements
into the legacy 4DGS environment.

The workflow is designed for one notebook. Colab runtime resets delete
`/content`; save all durable files under Drive. The legacy IP2P fallback is
kept selectable but is not part of the default modern dependency path.

## 1. Mount Drive and choose paths

Run this first in the notebook.

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

from pathlib import Path
PROJECT_ROOT = Path('/content/drive/MyDrive/OBEDIT-4D')
PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
print(PROJECT_ROOT)
```

Expected result: Drive is mounted and `PROJECT_ROOT` points to a persistent
folder. Keep raw data, trained checkpoints, and edited outputs there.

## 2. Clone this repository

Use the repository branch containing the new editor. Do not clone a second
unrelated `4DGaussians` tree over the project root.

```bash
%cd /content
git clone --branch object_editor https://github.com/baibhavsingh021/OBEDIT-4D.git OBEDIT-4D
%cd /content/OBEDIT-4D
git submodule update --init --recursive
git status --short
git branch --show-current
```

Expected result: the branch is `object_editor`, `object_editor/`,
`docs/EDITOR_DESIGN.md`, and the modified `edit_3d.py` are present. The public
push may require the repository owner to publish the branch; if cloning the
branch fails, the branch must first be pushed or downloaded as an archive.

## 3. Install the unified single-runtime environment

Do not run `pip install -r requirements.txt` in this workflow. That file pins
old Diffusers/Transformers versions for the legacy IP2P path and conflicts with
OmniGen. Install the unified file instead:

```bash
%cd /content/OBEDIT-4D
pip install -r object_editor/requirements-colab.txt
pip install --no-deps "git+https://github.com/VectorSpaceLab/OmniGen.git"
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

Expected result: OmniGen, modern Diffusers dependencies, the rasterizer, and
`simple-knn` are installed in the same runtime. If pip replaces Torch, restart
the runtime once, remount Drive, return to `/content/OBEDIT-4D`, and continue
from the next cell. If a CUDA extension fails, stop and repair the Torch/CUDA
match before running a long edit.

## 4. Prepare scene data

Place a processed scene and a trained 4DGS checkpoint in persistent storage.
The expected legacy layout is approximately:

```text
/content/drive/MyDrive/OBEDIT-4D/
  data/dynerf/<scene>/
  output/dynerf/<scene>/point_cloud/...
```

For an already-trained scene, you need the source scene directory, the model
path, and the trained `point_cloud`/deformation files. If the scene is not
already trained, follow the upstream 4DGaussians/Instruct-4DGS reconstruction
steps first. That is a separate, potentially long training stage.

Optional path check:

```bash
%cd /content/OBEDIT-4D
find /content/drive/MyDrive/OBEDIT-4D/data -maxdepth 3 -type f | head
find /content/drive/MyDrive/OBEDIT-4D/output -maxdepth 5 -type f | head
```

Expected result: the commands list real scene files. Empty output means the
scene has not been copied or mounted at the expected path.

## 5. Produce synchronized input images and masks

The standalone editor requires:

```text
images_dir/*.png   # synchronized rendered RGB images, same ordering as masks
mask_dir/*.png     # one grayscale target mask per image, values 0..255
```

Masks must already be reconciled using the 4DGS geometry/visibility path. For
background editing, the mask must select the background and leave dynamic
foreground protected. For manual masks, use exactly one mask for every input
image and keep filenames aligned.

The current branch does not yet provide a Colab cell that automatically runs
Grounded-SAM and the 4DGS renderer to create these folders. You can use your
existing mask-generation/rendering workflow, then verify:

```bash
find /content/drive/MyDrive/OBEDIT-4D/inputs/images -name '*.png' | sort | head
find /content/drive/MyDrive/OBEDIT-4D/inputs/masks -name '*.png' | sort | head
```

The important invariant is equal file counts, matching sorted order, and
identical spatial resolution.

## 6. Run the editor in the same runtime

There is no second notebook or environment. Continue in the same runtime after
the input images and masks exist.

## 7. Run the standalone editor

Example for a strict appearance edit:

```bash
%cd /content/OBEDIT-4D
python -m object_editor.scripts.edit_object \
  --images_dir /content/drive/MyDrive/OBEDIT-4D/inputs/images \
  --mask_dir /content/drive/MyDrive/OBEDIT-4D/inputs/masks \
  --output_dir /content/drive/MyDrive/OBEDIT-4D/edited \
  --editor_model omnigen \
  --editor_ckpt Shitao/OmniGen-v1 \
  --target_query "the red water bottle" \
  --edit_instruction "make the water bottle metallic blue" \
  --edit_type appearance \
  --preservation_mode strict \
  --coupling_strength 0.7
```

Expected result: the first run downloads the selected checkpoint and creates
an output directory like:

```text
edited/edit_appearance_<hash>/
  view_000.png
  view_001.png
  ...
```

The script processes one view at a time, uses the first view as the appearance
reference for later views, and restores the protected complement using masks.
Runtime and memory usage depend strongly on resolution and view count.

For a manual-mask-only edit, omit or leave `--target_query` empty; the masks
satisfy target validation:

```bash
python -m object_editor.scripts.edit_object \
  --images_dir ... --mask_dir ... --output_dir ... \
  --editor_model omnigen \
  --edit_instruction "change the selected material to brushed steel" \
  --edit_type appearance
```

For a background edit, the masks must select background pixels:

```bash
python -m object_editor.scripts.edit_object \
  --images_dir ... --mask_dir ... --output_dir ... \
  --editor_model omnigen \
  --target_query "the kitchen background" \
  --edit_instruction "change it to a modern restaurant kitchen" \
  --edit_type background \
  --preservation_mode strict
```

## 8. Important current limitations

- OmniGen is installed from its official repository and imported as
  `OmniGen.OmniGenPipeline`; the adapter is lazy-loaded. Diagnose model/API
  failures before starting a long Gaussian run.
- `--disable_*` flags are configuration controls, but full CGFA/GAXLC behavior
  requires renderer-produced correspondence maps. The current folder entry
  point has no map-loading CLI, so GAXLC is inactive unless integrated by a
  caller through `ObjectEditorPipeline.set_correspondence(...)`.
- The current folder entry point does not launch `edit_3d.py` or TA-SDS. Use
  the existing 4DGS fitting/refinement workflow after writing edited PNGs.
- Do not claim final 4D consistency from independent folder images until the
  renderer correspondence handoff is connected and evaluated.
- IP2P remains a legacy selection path, but the new standalone CLI is not a
  drop-in replacement for the old text-embedding call signature.

## 9. Feed edited images back into the same runtime

The modified `edit_3d.py` resolves files by the deterministic run name and
several common camera filename formats. It now accepts the editor output
directly through `--edited_images_path`. A Drive-backed data link is useful for
scene loading:

```bash
%cd /content/OBEDIT-4D
rm -rf data
ln -s /content/drive/MyDrive/OBEDIT-4D/data data
```

For the most predictable handoff, name files using the camera image names
expected by the scene and place the editor output under the run directory that
the legacy script will resolve.

The existing Gaussian fitting command still depends on the scene's configured
arguments and trained PLY path. Adapt the old command for your scene, for
example:

```bash
%cd /content/OBEDIT-4D
python edit_3d.py \
  --configs arguments/dynerf/cook_spinach.py \
  --dataset dynerf \
  --scene cook_spinach \
  --prompt "make the target metallic blue" \
  --target_query "the red water bottle" \
  --edit_instruction "make the target metallic blue" \
  --edit_type appearance \
  --run_name edit_appearance_<hash> \
  --edited_images_path /content/drive/MyDrive/OBEDIT-4D/edited/edit_appearance_<hash> \
  --ply_path /content/drive/MyDrive/OBEDIT-4D/output/dynerf/cook_spinach/point_cloud/.../point_cloud.ply
```

Use the exact model/checkpoint arguments required by your existing scene setup.
Inspect the supplied `edited_images_path` before a long refinement run.

Expected result: a saved edited Gaussian point cloud under the configured
model output, followed by the existing temporal/refinement artifacts. The
trajectory transport hook is optional; without it, a warning indicates that a
nonzero timestep has fallen back to t=0 supervision.

## 10. Render the final result

Use the existing renderer with the saved edited point cloud, as in the old
README:

```bash
%cd /content/OBEDIT-4D
python render_edited4d.py \
  --configs arguments/dynerf/cook_spinach.py \
  --ply_path /content/drive/MyDrive/OBEDIT-4D/output/dynerf/cook_spinach/point_cloud_refine/<run_name>/iteration_800/point_cloud.ply \
  -s /content/drive/MyDrive/OBEDIT-4D/data/dynerf/cook_spinach \
  --model_path /content/drive/MyDrive/OBEDIT-4D/output/dynerf/cook_spinach
```

Expected result: rendered edited frames/video in the renderer's configured
output location. Inspect several views and timesteps, especially protected
foreground/background boundaries, disocclusions, and small target objects.

## 11. Recommended first run

Start with 2-4 low-resolution synchronized views, a large target, and
`--disable_gaxlc` only if no renderer correspondence is available. Confirm
that the editor writes valid images before attempting all views. Then run the
legacy Gaussian fitting and render three or more timesteps before increasing
resolution or refinement iterations.

The first successful editor run proves only the 2D image/mask handoff. The
final success criterion requires the subsequent 4DGS fit, temporal refinement,
and cross-view/timestep evaluation.
