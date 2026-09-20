# Object Editor Design

## Decision

`object_editor` is an adapter-based editing layer beside the existing 4DGS
scripts. `master`/`updated_Editor` and the current IP2P refinement remain
fallbacks. The preferred experimental backend is OmniGen through its published
pipeline API, in a standalone newer Diffusers environment. The Python 3.7 /
PyTorch 1.13 4DGS environment remains responsible for rendering, masks,
correspondence, Gaussian fitting, and refinement; the editor boundary is images
and masks on disk or tensors with the documented contract.

OmniGen is not treated as an SDXL UNet: its exact latent/conditioning internals
must be verified against the installed release. Therefore generation-time
latent hooks are disabled unless an adapter explicitly advertises them. SDXL
is a pipeline-level fallback and IP2P is a legacy selector.

## Problem formulation

For synchronized views `v` and times `t`, edit only target region `M(v,t)`
according to instruction `q`, while preserving complement `1-M`, target
attributes marked protected, and 4DGS correspondence. The edited images are
supervision for the existing frozen-deformation canonical Gaussian fit; the
existing SDS/refinement stage remains available.

## 4DGS-aware mechanisms

**CGFA, `disable_cgfa`.** Independent editors lose identity and detail. A
canonical edited feature map `F_c` is transported using renderer-produced
correspondence `C_{c->v}` and visibility `V`:

`F_v(x) = V_v(x) F_c(C_{c->v}(x))`.

This uses canonical Gaussian visibility and camera/deformation correspondence;
2D IP2P and post-hoc SDS have no such appearance transport. The implementation
requires a valid map and never treats all projected Gaussian centers as visible.

**GAXLC, `disable_gaxlc`.** Independent proposals hallucinate different
surface details. For clean proposals `P_v`, not arbitrary scheduler noise:

`P'_v = (1-lambda W_v)P_v + lambda W_v P_c(C_{c->v})`,

where `W_v = M_v V_v` and `lambda` is configurable. This is geometry-grounded
cross-view coupling before canonical fitting; fixed random seeds or attention
sharing do not establish 3D correspondence.

**PRLP, `disable_prlp`.** Diffusion can bleed changes across a boundary. Given
clean latent `z_0`, the active scheduler's exact forward process produces
`z_t^orig = Scheduler.add_noise(z_0, epsilon, t)`. At each supported denoising
boundary:

`z_t = M z_t^edit + (1-M) z_t^orig`.

This is valid for the scheduler actually in use; the adapter must provide the
boundary. Pixel compositing remains only a final safety net. OmniGen currently
reports no latent-hook support in this package, so PRLP is not falsely claimed
for that backend.

**TA-SDS, `disable_tasds`.** Refinement can move corresponding Gaussians
inconsistently. With frozen deformation provider `D`:

`L = L_base + lambda_traj ||x_t-D(x_0,t)||^2 +
 lambda_protect ||x_protected-x^orig_protected||^2`.

This regularizes the existing 4DGS refinement rather than replacing it or using
unrestricted SDS as the editor.

## Target and preservation

`TargetSpec` separates query/manual masks/reference annotations from the edit
instruction. Grounded-SAM remains the open-vocabulary resolver; manual masks
override it. The complement is protected, including dynamic foreground for
background-only edits. Category-specific parsers are optional and outside the
core contract. Appearance, material, replacement, removal, background and
limited geometry edits are represented; strict preservation is required for
appearance/removal and relaxed/free preservation is required for geometry.

## Ablations

| Run | CGFA | GAXLC | PRLP | TA-SDS |
|---|---:|---:|---:|---:|
| full | on | on | on where supported | on |
| no appearance anchor | off | on | on | on |
| no proposal coupling | on | off | on | on |
| no protected generation | on | on | off | on |
| baseline refinement | off | off | off | off |

## T4 and open risks

T4 mode is fp16, one view per call, sequential CPU offload, VAE tiling, cached
CPU references, and chunked view/time processing. No bf16, FlashAttention-2,
FP8, large-scale training, or model download is implied. Actual OmniGen peak
memory, pipeline version/API, reference-image semantics, license compliance,
and quality are unvalidated here. The 4DGS renderer must supply occlusion-aware
latent-resolution maps; projected center points are insufficient.
