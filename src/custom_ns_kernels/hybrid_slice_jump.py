"""JAX-native hybrid nested sampling kernel combining DE jumps with slice sampling.

This module provides a high-performance hybrid kernel that combines:
1. Differential evolution jumps between live points (from Bilby approach)
2. Mahalanobis-normalized slice sampling with ensemble-derived directions

The key innovation is using ensemble geometry for slice directions while maintaining
identical constraint satisfaction for both DE jumps and slice sampling steps.
"""

from typing import NamedTuple
from functools import partial

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.ss import SliceState, vertical_slice, horizontal_slice, default_stepper_fn
from blackjax.ns.adaptive import build_kernel as build_adaptive_kernel
from blackjax.ns.adaptive import init as adaptive_init
from blackjax.ns.base import PartitionedState, new_state_and_info
from blackjax.ns.base import delete_fn as default_delete_fn
from blackjax.ns.utils import repeat_kernel, get_first_row
from blackjax.smc.tuning.from_particles import particles_covariance_matrix
from blackjax.types import ArrayTree, PRNGKey


class HybridInfo(NamedTuple):
    """Diagnostic information for a hybrid DE-Slice step."""
    is_accepted: bool
    mode: int  # 0 for DE jump, 1 for slice sampling
    evals: int  # Number of likelihood evaluations


def update_hybrid_params_fn(ns_state, *args, **kwargs):
    """Update parameters for the hybrid kernel.
    
    Provides both the ensemble covariance matrix (for Mahalanobis normalization
    of slice directions) and the live points (for DE jumps and direction generation).
    
    Parameters
    ----------
    ns_state : NSState
        Current nested sampling state containing live particles
    info : NSInfo, optional
        Information from previous step (unused)
    **kwargs
        Additional keyword arguments (unused)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'live_points': Current live particles for DE and direction generation
        - 'cov': Covariance matrix for Mahalanobis normalization
    """
    cov_matrix = particles_covariance_matrix(ns_state.particles)
    return {
        "live_points": ns_state.particles,
        "cov": cov_matrix,
    }


def hybrid_one_step(
    rng_key: PRNGKey,
    state: PartitionedState,
    logprior_fn,
    loglikelihood_fn,
    loglikelihood_0: float,
    params: dict,
):
    """Single hybrid step: DE jump or slice sampling with identical constraints.
    
    Critical: Performs vertical_slice BEFORE branching to ensure both DE jumps
    and slice sampling use identical constraint checking (same logslice value).
    """
    key_vs, key_mode, key_ensemble, key_step = jax.random.split(rng_key, 4)
    
    # STEP 1: Perform vertical slice BEFORE branching (ensures identical constraints)
    slice_state = SliceState(state.position, state.logprior)
    intermediate_state, _ = vertical_slice(key_vs, slice_state)
    
    # STEP 2: Generate ensemble direction vector (common to both branches)
    live_points = params['live_points']
    n_live = live_points.shape[0]
    
    key_a, key_b = jax.random.split(key_ensemble)
    idx_a = jax.random.randint(key_a, (), 0, n_live)
    idx_b_raw = jax.random.randint(key_b, (), 0, n_live - 1)
    idx_b = jnp.where(idx_b_raw >= idx_a, idx_b_raw + 1, idx_b_raw)
    
    point_a = jax.tree.map(lambda x: x[idx_a], live_points)
    point_b = jax.tree.map(lambda x: x[idx_b], live_points)
    ensemble_direction = jax.tree.map(lambda a, b: a - b, point_a, point_b)
    
    # STEP 3: Decide which mode to use for this particle
    is_slice_mode = jax.random.uniform(key_mode) < params.get('p_slice', 0.5)
    
    # Get unravel function for PyTree operations
    _, unravel_fn = ravel_pytree(state.position)
    
    def de_jump_fn(step_key):
        """DE jump branch: use ensemble_direction directly as delta."""
        # DE proposal: current position + ensemble_direction (gamma=1.0)
        pos_prop = jax.tree.map(lambda c, d: c + d, state.position, ensemble_direction)
        
        # IDENTICAL CONSTRAINTS to slice sampling
        logp_prop = logprior_fn(pos_prop)
        logl_prop = loglikelihood_fn(pos_prop)
        
        constraints_met = jnp.all(jnp.array([
            jnp.isfinite(logp_prop),                    # Valid prior
            logl_prop > loglikelihood_0,                # NS likelihood constraint
            logp_prop >= intermediate_state.logslice    # Slice constraint (CRITICAL!)
        ]))
        
        # Conditional update
        final_pos = jax.tree.map(
            lambda prop, curr: jnp.where(constraints_met, prop, curr),
            pos_prop, state.position
        )
        final_logp = jnp.where(constraints_met, logp_prop, state.logprior)
        final_logl = jnp.where(constraints_met, logl_prop, state.loglikelihood)
        
        new_part_state = PartitionedState(final_pos, final_logp, final_logl)
        info = HybridInfo(is_accepted=constraints_met, mode=0, evals=1)
        return new_part_state, info
    
    def slice_sample_fn(step_key):
        """Slice sampling branch: normalize ensemble_direction with Mahalanobis."""
        cov_matrix = params['cov']
        
        # Mahalanobis normalization (BlackJAX ss.py style - inline)
        d_flat, _ = ravel_pytree(ensemble_direction)
        invcov = jnp.linalg.inv(cov_matrix)
        norm = jnp.sqrt(jnp.einsum("i,ij,j", d_flat, invcov, d_flat))
        d_norm_flat = d_flat / norm
        norm_direction = unravel_fn(d_norm_flat)
        
        # Call horizontal_slice directly (vertical already performed)
        constraint_fn = lambda x: jnp.array([loglikelihood_fn(x)])
        constraint = jnp.array([loglikelihood_0])
        strict = jnp.array([True])
        
        new_slice_state, slice_info = horizontal_slice(
            step_key, intermediate_state, norm_direction, default_stepper_fn,
            logprior_fn, constraint_fn, constraint, strict
        )
        
        # Convert slice output to PartitionedState format
        final_state, _ = new_state_and_info(
            position=new_slice_state.position,
            logprior=new_slice_state.logdensity,
            loglikelihood=slice_info.constraint[0],
            info=slice_info,
        )
        
        info = HybridInfo(is_accepted=True, mode=1, evals=slice_info.evals)
        return final_state, info
    
    # STEP 4: Branch execution with lax.cond (identical constraints guaranteed)
    return jax.lax.cond(is_slice_mode, slice_sample_fn, de_jump_fn, key_step)


def hybrid_slice_jump_sampler(
    logprior_fn,
    loglikelihood_fn,
    num_inner_steps: int = 60,
    p_slice: float = 0.5,
    num_delete: int = 1,
) -> SamplingAlgorithm:
    """Create hybrid slice-jump nested sampling kernel.
    
    Combines differential evolution jumps between live points with Mahalanobis-normalized
    slice sampling using ensemble-derived directions. Both modes use identical constraint
    checking to ensure proper nested sampling behavior.
    
    Parameters
    ----------
    logprior_fn : callable
        Log-prior density function operating on physical parameters
    loglikelihood_fn : callable  
        Log-likelihood function operating on physical parameters
    num_inner_steps : int, default=60
        Number of MCMC steps per particle at each NS iteration
    p_slice : float, default=0.5
        Probability of slice sampling vs DE jump (0.0 = pure DE, 1.0 = pure slice)
    num_delete : int, default=1
        Number of particles to replace at each NS iteration
        
    Returns
    -------
    SamplingAlgorithm
        BlackJAX sampling algorithm with init and step functions
    """
    
    # Create inner kernel with repeat wrapper
    @repeat_kernel(num_inner_steps)
    def inner_kernel(rng_key, state, logprior_fn, loglikelihood_fn, loglikelihood_0, params):
        # Add p_slice to params for the kernel
        kernel_params = params.copy()
        kernel_params['p_slice'] = p_slice
        return hybrid_one_step(
            rng_key, state, logprior_fn, loglikelihood_fn, loglikelihood_0, kernel_params
        )
    
    delete_fn = partial(default_delete_fn, num_delete=num_delete)
    
    # Build adaptive kernel using BlackJAX framework
    kernel = build_adaptive_kernel(
        logprior_fn,
        loglikelihood_fn,
        delete_fn,
        inner_kernel,
        update_hybrid_params_fn,
    )
    
    # Create initialization function
    init_fn = partial(
        adaptive_init,
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        update_inner_kernel_params_fn=update_hybrid_params_fn,
    )
    
    return SamplingAlgorithm(init_fn, kernel)