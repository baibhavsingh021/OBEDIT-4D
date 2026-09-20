# object_editor

A focused 4DGS-aware editing layer for OBEDIT-4D. It keeps the existing
`edit_3d.py`, `edit_3d_mv.py`, IP2P, and SDS paths intact and provides adapter
contracts plus independently ablatable canonical anchoring, cross-view
coupling, protected-region preservation, and trajectory regularization.

Use `TargetSpec` for target selection and the instruction separately. The
Grounded-SAM/renderer integration must provide reconciled masks and
occlusion-aware correspondences. OmniGen is optional and must be installed in
a compatible standalone Diffusers environment; it is not imported by the
legacy 4DGS entry points.

See `docs/EDITOR_DESIGN.md` for equations, ablations, assumptions, and the T4
policy. The package is intentionally unvalidated because project execution,
installation, downloads, and inference were not performed during integration.
