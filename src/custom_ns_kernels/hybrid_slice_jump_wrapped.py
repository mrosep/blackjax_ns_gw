"""JAX-native generalized hybrid nested sampling kernel with customizable slice directions.

This module provides a high-performance hybrid kernel that combines:
1. Differential evolution jumps between live points (from Bilby approach)
2. Customizable slice sampling with user-defined direction generation methods

The key innovation is allowing custom slice direction generation while maintaining
identical constraint satisfaction for both DE jumps and slice sampling steps.
Supports multiple built-in direction methods and custom user functions.
"""

from typing import NamedTuple, Callable, Union, Optional
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
from blackjax.ns.nss import sample_direction_from_covariance, compute_covariance_from_particles
from blackjax.ns.utils import repeat_kernel, get_first_row
from blackjax.smc.tuning.from_particles import particles_covariance_matrix, particles_as_rows
from blackjax.types import ArrayTree, PRNGKey


def create_wrapped_stepper(mask_tree, period_tree):
    """Factory returning stepper with wraparound logic."""
    def stepper_fn(position, direction, step_size):
        proposed = jax.tree.map(lambda pos, d: pos + step_size * d, position, direction)
        return jax.tree.map(
            lambda prop, mask, period: jnp.where(mask, jnp.mod(prop, period), prop),
            proposed, mask_tree, period_tree
        )
    return stepper_fn


def _calc_covmat_flat(x_flat, wraparound_flat, periods_flat):
    """Calculate wrapped covariance matrix for flat arrays."""
    n_dims, n_particles = x_flat.shape
    
    # Normalize for trigonometric functions
    x_normalized = x_flat / periods_flat[:, jnp.newaxis]
    
    # Calculate circular mean using trigonometric approach
    mean_sin = jnp.sum(jnp.sin(x_normalized * 2.0 * jnp.pi), axis=1)
    mean_cos = jnp.sum(jnp.cos(x_normalized * 2.0 * jnp.pi), axis=1)
    
    mean_angle = jnp.atan2(mean_sin, mean_cos)
    circ_mean_norm = jnp.mod(mean_angle / (2.0 * jnp.pi), 1.0)
    circ_mean = circ_mean_norm * periods_flat
    
    # Linear mean
    lin_mean = jnp.sum(x_flat, axis=1) / n_particles
    
    # Choose appropriate mean based on wraparound flag
    mu = jnp.where(wraparound_flat, circ_mean, lin_mean)

    # Calculate shortest-arc deviations for wrapped parameters
    dx = x_flat - mu[:, jnp.newaxis]
    dx_norm = dx / periods_flat[:, jnp.newaxis]
    dx_norm_short = dx_norm - jnp.round(dx_norm)
    dx_short = dx_norm_short * periods_flat[:, jnp.newaxis]
    
    dx_final = jnp.where(wraparound_flat[:, jnp.newaxis], dx_short, dx)

    # Final covariance matrix
    covmat = (dx_final @ dx_final.T) / (n_particles - 1.0)
    return covmat


def create_wrapped_covariance_adaptor(wraparound_mask, wraparound_periods):
    """Factory for wrapped covariance adaptation function.
    
    Creates a function that can be passed as adapt_direction_params_fn
    to compute wrapped covariance matrices from live points.
    """
    # Flatten the static config PyTrees once
    flat_mask, _ = ravel_pytree(wraparound_mask)
    flat_periods, _ = ravel_pytree(wraparound_periods)
    
    def adapt_direction_params_fn(state, info, inner_kernel_params=None):
        """Compute wrapped covariance from live points."""
        # Get single particle structure for unraveling
        single_particle = get_first_row(state.particles)
        _, unravel_fn = ravel_pytree(single_particle)
        
        # Convert particles to flat (n_dims, n_particles) array
        particles_flat = particles_as_rows(state.particles).T
        
        # Compute wrapped covariance matrix
        cov_matrix_flat = _calc_covmat_flat(particles_flat, flat_mask, flat_periods)
        
        # Convert back to PyTree structure
        cov_pytree = jax.vmap(unravel_fn)(cov_matrix_flat)
        
        return {"cov": cov_pytree}
    
    return adapt_direction_params_fn


class HybridInfo(NamedTuple):
    """Diagnostic information for hybrid DE-Slice step."""
    is_accepted: bool
    mode: int  # 0 for DE jump, 1 for slice sampling
    evals: int


def _direction_ensemble(rng_key: PRNGKey, params: dict) -> ArrayTree:
    """Generate Mahalanobis-normalized slice direction from two distinct live points."""
    live_points = params['live_points']
    n_live = jax.tree.flatten(jax.tree.map(lambda x: x.shape[0], live_points))[0][0]
    
    key_a, key_b = jax.random.split(rng_key)
    idx_a = jax.random.randint(key_a, (), 0, n_live)
    idx_b_raw = jax.random.randint(key_b, (), 0, n_live - 1)
    idx_b = jnp.where(idx_b_raw >= idx_a, idx_b_raw + 1, idx_b_raw)
    
    point_a = jax.tree.map(lambda x: x[idx_a], live_points)
    point_b = jax.tree.map(lambda x: x[idx_b], live_points)
    direction = jax.tree.map(lambda a, b: a - b, point_a, point_b)
    
    # Mahalanobis normalization like BlackJAX NSS
    cov = params["cov"]
    row = get_first_row(cov)
    _, unravel_fn = ravel_pytree(row)
    d_flat, _ = ravel_pytree(direction)
    
    # Convert PyTree cov to flat matrix and normalize
    from blackjax.smc.tuning.from_particles import particles_as_rows
    cov_flat = particles_as_rows(cov)
    invcov = jnp.linalg.inv(cov_flat)
    norm = jnp.sqrt(jnp.einsum("i,ij,j", d_flat, invcov, d_flat))
    d_norm_flat = d_flat / (norm + 1e-9)
    
    return unravel_fn(d_norm_flat)


def _direction_covariance(rng_key: PRNGKey, params: dict) -> ArrayTree:
    """Generate slice direction from ensemble covariance using BlackJAX NSS."""
    return sample_direction_from_covariance(rng_key, params)




def hybrid_one_step(
    rng_key: PRNGKey,
    state: PartitionedState,
    logprior_fn,
    loglikelihood_fn,
    loglikelihood_0: float,
    params: dict,
    p_slice: float,
    direction_fn: Callable,
    stepper_fn: Callable,
):
    """Single hybrid step: DE jump or slice sampling with identical constraints."""
    key_vs, key_mode, key_ensemble, key_step = jax.random.split(rng_key, 4)
    
    # STEP 1: Perform vertical slice BEFORE branching (ensures identical constraints)
    slice_state = SliceState(state.position, state.logprior)
    intermediate_state, _ = vertical_slice(key_vs, slice_state)
    
    # STEP 2: Decide which mode to use for this particle
    is_slice_mode = jax.random.uniform(key_mode) < p_slice
    
    # Get unravel function for PyTree operations
    _, unravel_fn = ravel_pytree(state.position)
    
    def de_jump_fn(de_key):
        """DE jump branch: use unnormalized ensemble direction as delta."""
        direction_key, _ = jax.random.split(de_key)
        
        # Generate UNNORMALIZED ensemble direction inline
        live_points = params['live_points']
        n_live = jax.tree.flatten(jax.tree.map(lambda x: x.shape[0], live_points))[0][0]
        
        key_a, key_b = jax.random.split(direction_key)
        idx_a = jax.random.randint(key_a, (), 0, n_live)
        idx_b_raw = jax.random.randint(key_b, (), 0, n_live - 1)
        idx_b = jnp.where(idx_b_raw >= idx_a, idx_b_raw + 1, idx_b_raw)
        
        point_a = jax.tree.map(lambda x: x[idx_a], live_points)
        point_b = jax.tree.map(lambda x: x[idx_b], live_points)
        ensemble_direction = jax.tree.map(lambda a, b: a - b, point_a, point_b)
        
        # DE proposal using stepper (step_size=1.0 for direct jump)
        pos_prop = stepper_fn(state.position, ensemble_direction, 1.0)
        
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
    
    def slice_sample_fn(slice_key):
        """Slice sampling branch: use pre-selected normalized direction function."""
        direction_key, step_key = jax.random.split(slice_key)
        
        # Generate normalized direction using pre-selected function
        norm_direction = direction_fn(direction_key, params)
        
        # Call horizontal_slice directly (vertical already performed)
        constraint_fn = lambda x: jnp.array([loglikelihood_fn(x)])
        constraint = jnp.array([loglikelihood_0])
        strict = jnp.array([True])
        
        new_slice_state, slice_info = horizontal_slice(
            step_key, intermediate_state, norm_direction, stepper_fn,
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
    slice_direction: Union[str, Callable] = "covariance",
    num_delete: int = 1,
    stepper_fn: Callable = default_stepper_fn,
    adapt_direction_params_fn: Callable = compute_covariance_from_particles,
) -> SamplingAlgorithm:
    """Create hybrid slice-jump nested sampling kernel.
    
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
    slice_direction : str or callable, default="covariance"  
        Direction generation method for slice sampling
    num_delete : int, default=1
        Number of particles to replace at each NS iteration
    stepper_fn : callable, default=default_stepper_fn
        Stepper function (position, direction, step_size) -> new_position
    adapt_direction_params_fn : callable, default=compute_covariance_from_particles
        Function (state, info, params) -> dict to adapt slice direction parameters
        
    Returns
    -------
    SamplingAlgorithm
        BlackJAX sampling algorithm with init and step functions
    """
    
    methods = {"ensemble": _direction_ensemble, "covariance": _direction_covariance}
    direction_fn = methods.get(slice_direction, slice_direction)
    
    one_step_fn = partial(hybrid_one_step, p_slice=p_slice, direction_fn=direction_fn, stepper_fn=stepper_fn)
    
    @repeat_kernel(num_inner_steps)
    def inner_kernel(rng_key, state, logprior_fn, loglikelihood_fn, loglikelihood_0, params):
        return one_step_fn(rng_key, state, logprior_fn, loglikelihood_fn, loglikelihood_0, params)
    
    delete_fn = partial(default_delete_fn, num_delete=num_delete)
    
    def update_hybrid_params_fn(ns_state, *args, **kwargs):
        """Update parameters for the hybrid kernel."""
        cov_params = adapt_direction_params_fn(ns_state, None)
        return {
            "live_points": ns_state.particles,
            **cov_params,
        }
    
    init_fn = partial(
        adaptive_init,
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        update_inner_kernel_params_fn=update_hybrid_params_fn,
    )
    
    kernel = build_adaptive_kernel(
        logprior_fn, loglikelihood_fn, delete_fn, inner_kernel, update_hybrid_params_fn
    )
    
    return SamplingAlgorithm(init_fn, kernel)