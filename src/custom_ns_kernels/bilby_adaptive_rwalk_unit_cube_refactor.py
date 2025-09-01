"""JAX-native Bilby adaptive DE kernel for unit hypercube sampling."""

from typing import NamedTuple, Callable
from functools import partial

import jax
import jax.numpy as jnp

from blackjax.base import SamplingAlgorithm
from blackjax.ns.base import PartitionedState, NSState
from blackjax.ns.adaptive import build_kernel as build_adaptive_kernel
from blackjax.ns.adaptive import init as adaptive_init
from blackjax.ns.base import delete_fn as default_delete_fn


class DEInfo(NamedTuple):
    """Diagnostic information for a single DE MCMC step."""
    is_accepted: bool
    evals: int
    likelihood_evals: int  # Count only in-bounds proposals (like bilby)


class DEWalkInfo(NamedTuple):
    """Diagnostic information for a full DE MCMC walk."""
    n_accept: int
    n_steps: int
    n_likelihood_evals: int  # Total likelihood evaluations (in-bounds only)


class DEKernelParams(NamedTuple):
    """Static pytree for DE kernel parameters."""
    live_points: jax.Array
    loglikelihoods: jax.Array  # Log-likelihoods of all live points
    mix: float
    scale: float
    num_walks: jax.Array
    walks_float: jax.Array
    n_accept_total: jax.Array
    n_likelihood_evals_total: jax.Array  # Total likelihood evaluations (bilby-style counting)


def de_rwalk_one_step_unit_cube(
    rng_key: jax.Array,
    state: PartitionedState,
    logprior_fn: callable,
    loglikelihood_fn: callable,
    loglikelihood_0: float,
    params: DEKernelParams,
    stepper_fn: callable,
    num_survivors: int,
):
    """Single DE step in unit hypercube space."""
    key_a, key_b, key_mix, key_gamma = jax.random.split(rng_key, 4)
    
    current_pos = state.position
    live_points = params.live_points
    
    # Use static number of survivors for top_k selection
    _, top_indices = jax.lax.top_k(params.loglikelihoods, num_survivors)
    
    # Select two distinct random indices from the survivor indices
    pos_a = jax.random.randint(key_a, (), 0, num_survivors)
    pos_b_raw = jax.random.randint(key_b, (), 0, num_survivors - 1)
    pos_b = jnp.where(pos_b_raw >= pos_a, pos_b_raw + 1, pos_b_raw)
    
    # Use these positions to get the final indices into the original live_points array
    idx_a = top_indices[pos_a]
    idx_b = top_indices[pos_b]
    
    # DE vector calculation
    point_a = jax.tree_util.tree_map(lambda x: x[idx_a], live_points)
    point_b = jax.tree_util.tree_map(lambda x: x[idx_b], live_points)
    delta = jax.tree_util.tree_map(lambda a, b: a - b, point_a, point_b)
    
    # Mixed scaling
    is_small_step = jax.random.uniform(key_mix) < params.mix
    gamma_small = params.scale * jax.random.gamma(key_gamma, 4.0) * 0.25
    gamma_large = 1.0
    gamma = jnp.where(is_small_step, gamma_small, gamma_large)
    
    # Unit cube proposal
    pos_prop = stepper_fn(current_pos, delta, gamma)
    
    # Constraint evaluation
    logp_prop = logprior_fn(pos_prop)
    logl_prop = loglikelihood_fn(pos_prop)  # Always evaluate (avoid thread divergence)
    
    # Check constraints
    is_in_bounds = jnp.isfinite(logp_prop)
    is_above_threshold = logl_prop > loglikelihood_0
    
    constraints = jnp.array([is_in_bounds, is_above_threshold])
    is_accepted = jnp.all(constraints)
    
    # Count likelihood evaluations only for in-bounds proposals (like bilby)
    likelihood_evals = jnp.where(is_in_bounds, 1, 0)
    
    # State update
    final_pos = jax.tree_util.tree_map(
        lambda p, c: jnp.where(is_accepted, p, c), pos_prop, current_pos
    )
    final_logp = jnp.where(is_accepted, logp_prop, state.logprior)
    final_logl = jnp.where(is_accepted, logl_prop, state.loglikelihood)
    
    new_state = PartitionedState(final_pos, final_logp, final_logl)
    info = DEInfo(is_accepted=is_accepted, evals=1, likelihood_evals=likelihood_evals)
    
    return new_state, info


def de_rwalk_dynamic_unit_cube(
    rng_key: jax.Array,
    state: PartitionedState,
    logprior_fn: callable,
    loglikelihood_fn: callable,
    loglikelihood_0: float,
    params: DEKernelParams,
    stepper_fn: callable,
    num_survivors: int,
):
    """MCMC walk with fixed number of steps in unit cube."""
    # Use partial to create a new function with num_survivors "baked in"
    one_step_with_static_k = partial(
        de_rwalk_one_step_unit_cube,
        num_survivors=num_survivors,
    )
    
    def single_step_fn(rng_key, state, loglikelihood_0):
        return one_step_with_static_k(
            rng_key=rng_key,
            state=state, 
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            loglikelihood_0=loglikelihood_0,
            params=params,
            stepper_fn=stepper_fn,
        )

    def body_fun(i, val):
        key, current_state, n_accept, n_likelihood_evals = val
        step_key, next_key = jax.random.split(key)
        new_state, info = single_step_fn(step_key, current_state, loglikelihood_0)
        return (next_key, new_state, n_accept + info.is_accepted, n_likelihood_evals + info.likelihood_evals)

    init_val = (rng_key, state, jnp.array(0, dtype=jnp.int32), jnp.array(0, dtype=jnp.int32))
    
    final_key, final_state, final_n_accept, final_n_likelihood_evals = jax.lax.fori_loop(
        0, params.num_walks, body_fun, init_val
    )

    info = DEWalkInfo(n_accept=final_n_accept, n_steps=params.num_walks, n_likelihood_evals=final_n_likelihood_evals)
    return final_state, info


def update_bilby_walks_fn(
    ns_state: NSState,
    logprior_fn: callable,
    loglikelihood_fn: callable,
    n_target: int,
    max_mcmc: int,
    n_delete: int,
):
    """Bilby batch-level adaptation for unit cube sampling."""
    prev_params = ns_state.inner_kernel_params
    
    # Check sentinel value instead of None (JAX-compatible)
    is_uninitialized = prev_params.n_accept_total < 0
    
    # ==================== FIXED SECTION START ====================
    
    # --- 1. Define default values with explicit dtypes ---
    # These are the values to use on the first run (initialization).
    default_walks_float = jnp.array(200.0, dtype=jnp.float32)
    default_n_accept_total = jnp.array(0, dtype=jnp.int32)
    default_current_walks = jnp.array(200, dtype=jnp.int32)
    default_n_likelihood_evals_total = jnp.array(0, dtype=jnp.int32)

    # --- 2. Get values from previous state and explicitly cast to ensure type match ---
    # These are the values from the previous step. We cast them to the same
    # dtypes as the defaults to prevent any mismatch.
    param_walks_float = prev_params.walks_float.astype(jnp.float32)
    param_n_accept_total = prev_params.n_accept_total.astype(jnp.int32)
    param_current_walks = prev_params.num_walks.astype(jnp.int32)
    param_n_likelihood_evals_total = prev_params.n_likelihood_evals_total.astype(jnp.int32)

    # --- 3. Use jnp.where for branchless, type-safe selection ---
    # This replaces lax.cond and is robust to type differences since we
    # have already ensured the types of both branches are identical.
    walks_float = jnp.where(is_uninitialized, default_walks_float, param_walks_float)
    n_accept_total = jnp.where(is_uninitialized, default_n_accept_total, param_n_accept_total)
    current_walks = jnp.where(is_uninitialized, default_current_walks, param_current_walks)
    n_likelihood_evals_total = jnp.where(is_uninitialized, default_n_likelihood_evals_total, param_n_likelihood_evals_total)
    
    # ===================== FIXED SECTION END =====================

    leaves = jax.tree_util.tree_leaves(ns_state.particles)
    nlive = leaves[0].shape[0]
    base_delay = nlive // 10 - 1
    delay = jnp.maximum(base_delay // n_delete, 1)
    
    # Keep bilby's walk length tuning formula (uses total walks, not likelihood evals)
    avg_accept_per_particle = n_accept_total / n_delete
    accept_prob = jnp.maximum(0.5, avg_accept_per_particle) / jnp.maximum(1.0, current_walks)
    
    new_walks_float = (walks_float * delay + n_target / accept_prob) / (delay + 1)
    new_walks_float = jnp.where(n_accept_total == 0, walks_float, new_walks_float)

    num_walks_int = jnp.minimum(jnp.ceil(new_walks_float).astype(jnp.int32), max_mcmc)
    
    example_particle = jax.tree_util.tree_map(lambda x: x[0], ns_state.particles)
    flat_particle, _ = jax.flatten_util.ravel_pytree(example_particle)
    n_dim = flat_particle.shape[0]
    
    return DEKernelParams(
        live_points=ns_state.particles,
        loglikelihoods=ns_state.loglikelihood,
        mix=0.5,
        scale=2.38 / jnp.sqrt(2 * n_dim),
        num_walks=jnp.array(num_walks_int, dtype=jnp.int32),
        walks_float=jnp.array(new_walks_float, dtype=jnp.float32),
        n_accept_total=jnp.array(0, dtype=jnp.int32),
        n_likelihood_evals_total=jnp.array(0, dtype=jnp.int32),
    )


def bilby_adaptive_de_sampler_unit_cube(
    logprior_fn: callable,
    loglikelihood_fn: callable,
    nlive: int,
    n_target: int = 60,
    max_mcmc: int = 5000,
    num_delete: int = 1,
    stepper_fn: callable = None,
) -> SamplingAlgorithm:
    """Bilby adaptive DE sampler for unit hypercube."""
    if stepper_fn is None:
        raise ValueError("stepper_fn must be provided for unit cube sampling")
    
    # Calculate num_survivors statically as a Python integer
    num_survivors = nlive - num_delete
        
    delete_fn = partial(default_delete_fn, num_delete=num_delete)
    
    def update_fn(ns_state, *args, **kwargs):
        return update_bilby_walks_fn(
            ns_state=ns_state,
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            n_target=n_target,
            max_mcmc=max_mcmc,
            n_delete=num_delete,
        )

    kernel_with_stepper = partial(
        de_rwalk_dynamic_unit_cube, 
        stepper_fn=stepper_fn,
        num_survivors=num_survivors,
    )

    base_kernel_step = build_adaptive_kernel(
        logprior_fn,
        loglikelihood_fn,
        delete_fn,
        kernel_with_stepper,
        update_fn,
    )

    def init_fn(particles):
        # Call adaptive_init WITHOUT update_fn to avoid the dict issue
        state = adaptive_init(
            particles=particles,
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            update_inner_kernel_params_fn=None,  # Don't call during init
        )
        
        # Calculate proper scale from particle dimensionality
        example_particle = jax.tree_util.tree_map(lambda x: x[0], particles)
        flat_particle, _ = jax.flatten_util.ravel_pytree(example_particle)
        n_dim = flat_particle.shape[0]
        scale = 2.38 / jnp.sqrt(2 * n_dim)
        
        # Create initial DEKernelParams with sentinel value
        initial_de_params = DEKernelParams(
            live_points=particles,
            loglikelihoods=state.loglikelihood,
            mix=0.5,
            scale=scale,
            num_walks=jnp.array(200, dtype=jnp.int32),
            walks_float=jnp.array(200.0, dtype=jnp.float32),
            n_accept_total=jnp.array(-1, dtype=jnp.int32),  # Sentinel flag
            n_likelihood_evals_total=jnp.array(-1, dtype=jnp.int32),  # Sentinel flag
        )
        
        # Set our sentinel state manually
        return state._replace(inner_kernel_params=initial_de_params)
    
    def step_fn(rng_key, state: NSState):
        new_state, info = base_kernel_step(rng_key, state)
        
        inner_info = info.inner_kernel_info
        batch_n_accept = jnp.sum(inner_info.n_accept)
        batch_n_likelihood_evals = jnp.sum(inner_info.n_likelihood_evals)

        updated_params = new_state.inner_kernel_params._replace(
            n_accept_total=batch_n_accept,
            n_likelihood_evals_total=batch_n_likelihood_evals
        )
        
        final_state = new_state._replace(inner_kernel_params=updated_params)
        return final_state, info

    return SamplingAlgorithm(init_fn, step_fn)