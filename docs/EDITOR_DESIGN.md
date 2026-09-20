# Object Editor Design and Implementation Guide

## 1. Purpose

This document describes the changes made on the `object_editor` branch relative
to the original OBEDIT-4D pipeline. It is both the architecture record and the
implementation guide for the new editor path.

The original pipeline is still preserved. Its main path is:

1. Render synchronized views of a trained 4DGS scene.
2. Generate target masks with Grounded-SAM or provide masks manually.
3. Edit images with Coherent-IP2P/IP2P.
4. Blend the edited pixels with the original image.
5. Fit canonical Gaussians with the deformation field frozen.
6. Refine the result with the existing SDS/refinement code.

The new branch adds an adapter-based editor and 4DGS-aware contracts around
that flow. It does not delete the old entry points, checkpoints, output formats,
or IP2P implementation.

## 2. Changes from the base repository

### New package layout

```text
object_editor/
  adapters/       Backend-neutral editor interface, OmniGen, SDXL, IP2P
  core/           Target policy and 4D-aware mechanisms
  utils/          Tensor, camera, geometry, and T4 memory contracts
  eval/           Masked evaluation helpers
  configs/        Scenario and ablation configurations
  scripts/        Standalone image/mask editing entry point
  pipeline.py     Synchronized view editing bridge
  requirements-colab.txt
```

### New editor adapter contract

`BaseEditorAdapter` defines a small common interface. The required operation is
image editing:

```text
image:        (1, 3, H, W), float, [-1, 1]
target_mask:  (1, H, W), float, [0, 1]
references:   zero or more RGB reference tensors
output:       (1, 3, H, W), float, [-1, 1]
```

Latent sampling and latent hooks are optional capabilities. The code does not
pretend that every diffusion model has an SD/UNet interface.

### New backend selection

The available selectors are:

| Selector | Role | Interface |
|---|---|---|
| `omnigen` | Primary editor | Official `OmniGen.OmniGenPipeline` API |
| `sdxl` | Pipeline fallback | SDXL image-to-image API |
| `ip2p` | Legacy selector | Existing IP2P implementation boundary |

OmniGen uses the verified package import `from OmniGen import OmniGenPipeline`
and checkpoint `Shitao/OmniGen-v1`. It is called with `input_images`, image
placeholders, `img_guidance_scale`, and `offload_model`; it is not called as an
SDXL UNet.

### New target contract

`TargetSpec` separates the target from the edit instruction. It supports:

- open-vocabulary query, such as `the red water bottle`;
- manual per-view masks;
- reference images and reference annotations;
- single or multiple targets;
- background-only edits;
- edit types: appearance, replacement, removal, background, geometry, style;
- preservation modes: strict, relaxed, and free;
- protected attributes such as shape, pose, articulation, and identity.

The old prompt-token filename convention, `prompt.split(' ')[-1]`, is removed.
Run names are deterministic hashes of target query, instruction, and edit type,
or can be supplied explicitly with `--run_name`.

## 3. Single-runtime Colab architecture

The current implementation uses one modern Colab runtime:

```text
Drive data/checkpoint
        |
        v
4DGS renderer + mask handoff
        |
        | synchronized RGB PNGs + reconciled mask PNGs
        v
OmniGen object_editor
        |
        | edited RGB PNGs
        v
edit_3d.py canonical Gaussian fitting
        |
        v
existing temporal/SDS refinement
        |
        v
render_edited4d.py
```

Install `object_editor/requirements-colab.txt`, not the root
`requirements.txt`. The root file contains old Diffusers and Transformers pins
for the legacy IP2P path. The Colab requirements provide modern Torch,
Diffusers, OmniGen-compatible Transformers, the custom-renderer dependencies,
and packages directly imported by `edit_3d.py` such as `lpips`, `open3d`,
`plyfile`, `scipy`, and `scikit-learn`.

The official OmniGen source is installed with `--no-deps` so it cannot silently
replace the selected Torch/Diffusers versions.

The complete cell sequence is in
`notebooks/OBEDIT_4D_Colab.ipynb`; the detailed operational instructions are
in `docs/COLAB_GUIDE.md`.

## 4. Runtime data contract

Before running the editor, the 4DGS/mask stage must create:

```text
images_dir/
  view_000.png
  view_001.png
  ...
mask_dir/
  view_000.png
  view_001.png
  ...
```

The image and mask filenames must match exactly, counts must match, and every
mask must have the same height and width as its image. RGB images are converted
to `[-1, 1]`; grayscale masks are converted to `[0, 1]`.

The mask is the target region. Its complement is protected. Consequently, a
background edit must use a background mask so that dynamic foreground pixels
remain protected. A manual mask is accepted without a text query.

The current editor entry point intentionally does not invent scene renders or
masks from a checkpoint. Grounded-SAM, manual annotation, and renderer-based
mask reconciliation remain explicit inputs.

## 5. Implemented editing flow

`ObjectEditorPipeline.edit_views` performs the following operations:

1. Validate synchronized image/mask shapes.
2. Edit the selected anchor view first.
3. Keep the anchor result on CPU between calls.
4. Provide the anchor result as a reference image to later views when CGFA is
enabled.
5. Process one view at a time, which is the T4 memory mode.
6. Optionally apply renderer-produced cross-view correspondence coupling.
7. Restore the protected image complement as a final safety net.
8. Return edited views in the original input order.

The final compositing step does not replace generation-time protection; it is a
fallback because the public OmniGen API used here does not expose a verified
latent mask hook. This is why PRLP is implemented as a scheduler-aware core
operator but is not falsely claimed to run inside OmniGen denoising.

## 6. 4DGS-aware mechanisms

These mechanisms are independently disableable and are intended to make the
editing process, not only the final image blend, aware of 4DGS structure.

### CGFA: canonical Gaussian feature/appearance anchoring

Flag: `--disable_cgfa`

Independent views can invent different identities and details. A canonical
view is edited first and used as a reference for subsequent views. When a
renderer-produced map is available, canonical features can be transported as:

`F_v(x) = V_v(x) F_c(C_{c->v}(x))`

where `C` is a canonical correspondence and `V` is visibility confidence.
The code refuses to claim visibility from projected Gaussian centers alone.

In the current folder CLI, CGFA is realized as canonical appearance-reference
conditioning. Full feature-map transport requires the caller to provide the
correspondence map through the pipeline API.

### GAXLC: geometry-aware cross-view coupling

Flag: `--disable_gaxlc`; strength: `--coupling_strength`

Independent proposals can hallucinate incompatible surface details. For clean
image proposals `P_v`, the coupling operator is:

`P'_v = (1 - lambda W_v) P_v + lambda W_v P_c(C_{c->v})`

with `W_v = M_v V_v`. The implementation consumes `(source_yx, valid)` maps
from the 4DGS renderer and applies the operation only in the target region.
It does not mix arbitrary scheduler noise and does not use a fixed seed as a
substitute for geometry.

Important status: the folder CLI has no correspondence-map file format yet.
Therefore GAXLC is inactive unless a caller invokes
`ObjectEditorPipeline.set_correspondence(...)`. This is explicit in the guide
and prevents silently fabricating 3D consistency.

### PRLP: protected-region latent preservation

Flag: `--disable_prlp`

For a backend that exposes a valid denoising hook, the original protected
region is restored at the scheduler's own parameterization:

`z_t^orig = Scheduler.add_noise(z_0, epsilon, t)`

`z_t = M z_t^edit + (1 - M) z_t^orig`

This avoids assuming a particular DDPM schedule. For OmniGen in the current
adapter, only the final image-space safety composite is available because no
verified latent hook is exposed by the public pipeline API.

### TA-SDS: trajectory-aware refinement

Flag: `--disable_tasds`

The refinement objective can add:

`L = L_base + lambda_traj ||x_t - D(x_0,t)||^2 +
lambda_protect ||x_protected - x_protected^orig||^2`

where `D` is the frozen deformation provider. This preserves the existing
L1/SSIM and SDS behavior and adds trajectory/protected-position regularization;
it does not replace Gaussian fitting or unrestricted SDS with a 2D loss.

## 7. Existing-file changes

### `arguments/__init__.py`

Adds `EditorParams`, including backend selection, checkpoint, target fields,
preservation mode, ablation flags, memory controls, coupling strength, and run
name. Defaults are non-required so the original legacy command remains usable.
Validation rejects unsupported edit types, invalid coupling strengths, and
invalid batch sizes.

### `edit_3d.py`

The script now:

- uses deterministic run names;
- accepts `--edited_images_path` for direct editor handoff;
- resolves common numeric and zero-padded PNG names;
- removes dependence on scene-name camera dictionaries;
- exposes a generic camera accessor;
- skips NaN iterations and reduces optimizer learning rates instead of calling
  `os.execv` and restarting the process;
- exposes a timestep supervision helper that prefers available timestep data,
  then trajectory transport, then an explicit t=0 warning fallback.

The original frozen-deformation fitting and output conventions remain intact.

### `object_editor/scripts/edit_object.py`

This is the new image/mask entry point. It validates inputs, creates the
selected adapter, processes synchronized views sequentially, and writes:

```text
<output_dir>/edit_<edit_type>_<hash>/*.png
```

It does not automatically launch Gaussian fitting; that remains an explicit
`edit_3d.py` step so intermediate outputs can be inspected.

## 8. Ablations

| Configuration | CGFA | GAXLC | PRLP | TA-SDS | Meaning |
|---|---:|---:|---:|---:|---|
| Full supported path | on | on if maps exist | on where backend supports it | on in refinement | Maximum available consistency |
| No anchor | off | on if maps exist | on | on | Tests reference conditioning |
| No coupling | on | off | on | on | Tests geometry coupling |
| No protection | on | on if maps exist | off | on | Measures complement drift |
| No trajectory | on | on if maps exist | on | off | Measures temporal refinement drift |
| Legacy baseline | off | off | off | off | Existing IP2P-style comparison |

An ablation must record whether correspondence maps were available. A run
labelled "full" without maps is not a full GAXLC evaluation.

## 9. T4 policy and expected resource behavior

The supported T4 policy is:

- fp16 model execution;
- no bf16, FP8, or FlashAttention-2 requirement;
- one view per editor call;
- CPU/offloaded model execution where supported;
- CPU-resident intermediate view results;
- VAE tiling option retained for compatible backends;
- no optimizer or large-scale training for the diffusion editor.

Start with 2-4 views at moderate resolution. Verify the image/mask handoff,
then increase view count and resolution. Small targets, extreme viewpoint
changes, and disocclusions require renderer visibility and correspondence
support; they cannot be solved reliably by independent 2D edits.

## 10. Supported scope and limitations

Supported as a representation and workflow:

- appearance, material, color, and texture edits;
- replacement and removal with a suitable mask/reference;
- background-only edits with foreground protection;
- multiple targets through a reconciled combined mask;
- limited geometry edits with relaxed/free preservation.

Not guaranteed by the current CLI alone:

- automatic Grounded-SAM execution;
- automatic 4DGS rendering-to-mask reconciliation;
- full CGFA feature transport without correspondence maps;
- GAXLC without correspondence maps;
- generation-time PRLP inside OmniGen;
- automatic TA-SDS launch from `edit_object.py`;
- face identity, pose, physics, or category-specific guarantees.

The final criterion remains empirical: inspect multiple synchronized views and
timesteps, protected regions, boundaries, and disocclusions, then report both
edit quality and consistency metrics. No claim of final quality should be made
from the 2D editor output alone.
