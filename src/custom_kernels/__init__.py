"""Custom nested sampling kernels for BlackJAX-NS.

This module provides GPU-accelerated nested sampling kernels optimized for 
gravitational wave parameter estimation.
"""

from .acceptance_walk import bilby_adaptive_de_sampler_unit_cube as acceptance_walk_sampler
from .unit_cube_wrappers import (
    create_unit_cube_functions,
    init_unit_cube_particles, 
    transform_to_physical
)

__all__ = [
    "acceptance_walk_sampler",
    "create_unit_cube_functions",
    "init_unit_cube_particles",
    "transform_to_physical"
]
