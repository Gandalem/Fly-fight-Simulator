# Third-party attribution

`flyfight/embodiment/cached_controller.py` adapts the FlyGym 2.1.0
`flygym_demo.complex_terrain.hybrid_controller` and observation code.

- Copyright 2023-2026 The NeuroMechFly v2 Authors.
- Upstream: https://github.com/NeLy-EPFL/flygym
- License: Apache License 2.0, reproduced in `FlyGym-LICENSE.txt`.
- Modifications: cached body/geometry/DOF mappings, phase-knot caching, batched
  evaluation of the original spline coefficients, and allocation reduction.
  The original controller equations and physics timestep are retained.
