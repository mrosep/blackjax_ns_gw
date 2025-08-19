"""JAX-native implementation of Bilby FixedRWalk kernel for BlackJAX nested sampling (STATIC VERSION).

This module provides a high-performance differential evolution random walk kernel
that operates directly in physical parameter space, following the proven Bilby/dynesty
algorithm while leveraging BlackJAX's GPU-optimized architecture.

This is the STATIC version where the chain length (num_inner_steps) is fixed.
For the adaptive version that matches Bilby's 'acceptance-walk' behavior, see bilby_adaptive_rwalk.py.
"""

from typing import NamedTuple
from functools import partial

import jax
import jax.numpy as jnp

from blackjax.base import SamplingAlgorithm
from blackjax.ns.utils import repeat_kernel
from blackjax.ns.base import PartitionedState
from blackjax.ns.adaptive import build_kernel as build_adaptive_kernel
from blackjax.ns.adaptive import init as adaptive_init
from blackjax.ns.base import delete_fn as default_delete_fn


class DEInfo(NamedTuple):
    """Diagnostic information for a DE MCMC chain.
    
    When used with repeat_kernel, the fields of this tuple will be
    summed across all inner steps.
    """
    is_accepted: bool  # Will be summed to count total accepted steps
    evals: int         # Will be summed to count total evaluations


def de_rwalk_one_step_physical(
    rng_key: jax.Array,
    state: PartitionedState,
    logprior_fn: callable,
    loglikelihood_fn: callable,
    loglikelihood_0: float,
    params: dict,
):
    """Single differential evolution step in physical parameter space.
    
    Implements Bilby FixedRWalk proposal with mixed scaling: small steps
    (Gamma-distributed) for local exploration and large jumps for mode hopping.
    """
    # Split keys for independent random operations
    key_a, key_b, key_mix, key_gamma = jax.random.split(rng_key, 4)
    
    current_pos = state.position
    live_points = params['live_points']
    n_live = live_points.shape[0]
    
    # Robust live point selection - guaranteed distinct indices
    idx_a = jax.random.randint(key_a, (), 0, n_live)
    idx_b_raw = jax.random.randint(key_b, (), 0, n_live - 1)
    idx_b = jnp.where(idx_b_raw >= idx_a, idx_b_raw + 1, idx_b_raw)
    
    # Differential evolution vector
    point_a = live_points[idx_a]
    point_b = live_points[idx_b]
    delta = point_a - point_b
    
    # Mixed scaling - small steps vs large jumps
    is_small_step = jax.random.uniform(key_mix) < params['mix']
    gamma_small = params['scale'] * jax.random.gamma(key_gamma, 4.0) * 0.25
    gamma_large = 1.0
    gamma = jnp.where(is_small_step, gamma_small, gamma_large)
    
    # Physical space proposal
    pos_prop = current_pos + delta * gamma
    
    # Constraint evaluation following BlackJAX slice sampling pattern
    logp_prop = logprior_fn(pos_prop)
    logl_prop = loglikelihood_fn(pos_prop)
    
    constraints = jnp.array([
        jnp.isfinite(logp_prop),  # Prior constraint
        logl_prop > loglikelihood_0  # Likelihood constraint
    ])
    is_accepted = jnp.all(constraints)
    
    # Conditional state update
    final_pos = jnp.where(is_accepted, pos_prop, current_pos)
    final_logp = jnp.where(is_accepted, logp_prop, state.logprior)
    final_logl = jnp.where(is_accepted, logl_prop, state.loglikelihood)
    
    new_state = PartitionedState(final_pos, final_logp, final_logl)
    info = DEInfo(is_accepted=is_accepted, evals=1)
    
    return new_state, info


def update_inner_kernel_params_fn(ns_state, *args, **kwargs):
    """Provide live points and adaptive scaling to kernel.
    
    This signature is robust to the different call patterns in adaptive.init
    and adaptive.step by using *args to capture extra positional arguments.
    """
    live_particles = ns_state.particles
    n_dim = live_particles.shape[-1]
    return {
        "live_points": live_particles,
        "mix": 0.5,  # 50% probability of small vs large steps
        "scale": 2.38 / jnp.sqrt(2 * n_dim)  # Standard DE scaling
    }


def de_rwalk_sampler_physical_static(
    logprior_fn: callable,
    loglikelihood_fn: callable,
    num_inner_steps: int = 60,
    num_delete: int = 1,
) -> SamplingAlgorithm:
    """Create static differential evolution random walk sampler for nested sampling.
    
    Implements Bilby FixedRWalk algorithm in JAX-native form for GPU acceleration.
    Operates directly in physical parameter space with mixed proposal scaling.
    This is the STATIC version - chain length is fixed at num_inner_steps.
    
    Parameters
    ----------
    logprior_fn
        Log-prior density function operating on physical parameters.
    loglikelihood_fn
        Log-likelihood function operating on physical parameters.
    num_inner_steps
        Number of MCMC steps per particle at each NS iteration.
    num_delete
        Number of particles to replace at each NS iteration.
        
    Returns
    -------
    SamplingAlgorithm
        BlackJAX sampling algorithm with init and step functions.
    """
    # Apply repeat_kernel decorator exactly like BlackJAX NSS
    repeated_de_kernel = repeat_kernel(num_inner_steps)(de_rwalk_one_step_physical)
    
    delete_fn = partial(default_delete_fn, num_delete=num_delete)
    
    # Use adaptive kernel builder for live point access
    kernel = build_adaptive_kernel(
        logprior_fn,
        loglikelihood_fn,
        delete_fn,
        repeated_de_kernel,
        update_inner_kernel_params_fn,
    )
    
    # Standard adaptive initialization
    init_fn = partial(
        adaptive_init,
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        update_inner_kernel_params_fn=update_inner_kernel_params_fn,
    )
    
    return SamplingAlgorithm(init_fn, kernel)