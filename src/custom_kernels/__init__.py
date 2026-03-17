"""Custom nested sampling kernels for BlackJAX-NS.

This module provides GPU-accelerated nested sampling kernels optimized for
gravitational wave parameter estimation.

Kernel variants:
- acceptance_walk_sampler: GPU-optimized improved kernel (pinned BlackJAX API)
- acceptance_walk_sampler_legacy: Original kernel (pinned BlackJAX API)
- acceptance_walk_sampler_newapi: For the current BlackJAX nested_sampling branch
"""

from .acceptance_walk import bilby_adaptive_de_sampler_unit_cube as acceptance_walk_sampler_legacy
from .acceptance_walk_improved import bilby_adaptive_de_sampler_unit_cube_improved as acceptance_walk_sampler
from .acceptance_walk_newAPI import acceptance_walk_sampler_newapi
# from .acceptance_walk_fastslow import bilby_adaptive_de_sampler_fast_slow, FastSlowConfig
from .unit_cube_wrappers import (
    create_unit_cube_functions,
    init_unit_cube_particles,
    transform_to_physical,
    # create_fast_slow_likelihood
)

__all__ = [
    "acceptance_walk_sampler",
    "acceptance_walk_sampler_legacy",
    "acceptance_walk_sampler_newapi",
    # "bilby_adaptive_de_sampler_fast_slow",
    # "FastSlowConfig",
    "create_unit_cube_functions",
    "init_unit_cube_particles",
    "transform_to_physical",
    # "create_fast_slow_likelihood"
]
