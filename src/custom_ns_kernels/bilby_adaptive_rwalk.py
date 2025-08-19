"""JAX-native implementation of Bilby adaptive 'acceptance-walk' kernel for BlackJAX nested sampling.

This module provides a high-performance differential evolution random walk kernel
that operates directly in physical parameter space, following the proven Bilby/dynesty
'acceptance-walk' algorithm while leveraging BlackJAX's GPU-optimized architecture.

This ADAPTIVE version replicates Bilby's exact behavior where the chain length (num_walks)
adapts each batch to target a specific number of accepted steps (typically 60).
For the static version with fixed chain length, see bilby_fixed_rwalk.py.
"""

from typing import NamedTuple
from functools import partial

import jax
import jax.numpy as jnp

from blackjax.base import SamplingAlgorithm
from blackjax.ns.utils import repeat_kernel
from blackjax.ns.base import PartitionedState, NSState
from blackjax.ns.adaptive import build_kernel as build_adaptive_kernel
from blackjax.ns.adaptive import init as adaptive_init
from blackjax.ns.base import delete_fn as default_delete_fn


class DEInfo(NamedTuple):
    """Diagnostic information for a single DE MCMC step."""
    is_accepted: bool  # Whether this step was accepted
    evals: int         # Number of evaluations (always 1 for single step)


class DEWalkInfo(NamedTuple):
    """Diagnostic information for a full DE MCMC walk.
    
    The fields are summed across the vmap'd batch dimension.
    """
    n_accept: int   # Total number of accepted steps in the walk
    n_steps: int    # Total number of steps taken in the walk


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
    # Note: Single step still returns boolean for is_accepted, will be converted in the while_loop
    info = DEInfo(is_accepted=is_accepted, evals=1)
    
    return new_state, info


def de_rwalk_dynamic(
    rng_key: jax.Array,
    state: PartitionedState,
    logprior_fn: callable,
    loglikelihood_fn: callable,
    loglikelihood_0: float,
    params: dict,
):
    """An MCMC kernel that runs for a FIXED number of steps, `num_walks`.
    
    This function uses `jax.lax.while_loop` to replace `repeat_kernel`,
    allowing the number of steps to be determined dynamically for each batch.
    """
    # Create a wrapper function for the single-step function in the loop
    def single_step_fn(rng_key, state, loglikelihood_0):
        return de_rwalk_one_step_physical(
            rng_key=rng_key,
            state=state, 
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            loglikelihood_0=loglikelihood_0,
            params=params,
        )

    # State for the while_loop: (key, particle_state, n_accept, n_steps)
    init_val = (rng_key, state, 0, 0)

    # The walk ALWAYS runs for `params['num_walks']` steps.
    def cond_fun(val):
        _, _, _, n_steps = val
        return n_steps < params['num_walks']

    # The body of the loop executes one MCMC step.
    def body_fun(val):
        key, current_state, n_accept, n_steps = val
        step_key, next_key = jax.random.split(key)
        # single_step_fn is already partially applied with logprior_fn, loglikelihood_fn, params
        new_state, info = single_step_fn(step_key, current_state, loglikelihood_0)
        return (next_key, new_state, n_accept + info.is_accepted, n_steps + 1)

    # Run the dynamic-length walk
    final_val = jax.lax.while_loop(cond_fun, body_fun, init_val)
    _, final_state, final_n_accept, final_n_steps = final_val

    # Return info in the new format
    info = DEWalkInfo(n_accept=final_n_accept, n_steps=final_n_steps)
    return final_state, info


def update_bilby_walks_fn(
    ns_state: NSState,
    logprior_fn: callable,
    loglikelihood_fn: callable,
    n_target: int,
    max_mcmc: int,
    n_delete: int,
):
    """Implements Bilby's batch-level adaptation for the 'walks' parameter."""
    # 1. Get adaptation state from the previous batch. Use defaults for first run.
    prev_params = ns_state.inner_kernel_params or {}
    walks_float = prev_params.get('walks_float', 100.0)  # Bilby starts with 100
    n_accept_total = prev_params.get('n_accept_total', 0)
    current_walks = prev_params.get('num_walks', 100)  # Current walks value used in last batch

    # 2. Implement Bilby's adaptation formula with batch-aware delay
    nlive = ns_state.particles.shape[0]
    # Use batch-aware delay: more aggressive updates since we average over num_delete particles
    # Add extra smoothing to protect against cases where num_delete is high fraction of nlive
    delay = jnp.maximum(nlive // n_delete + 1, 1)
    
    # Bilby formula: accept_prob = max(0.5, n_accept_per_particle) / walks
    # Calculate average accepted steps per particle, then acceptance rate
    avg_accept_per_particle = n_accept_total / n_delete
    accept_prob = jnp.maximum(0.5, avg_accept_per_particle) / jnp.maximum(1.0, current_walks)
    
    # Bilby formula: walks = (walks * delay + n_target / accept_prob) / (delay + 1)
    new_walks_float = (walks_float * delay + n_target / accept_prob) / (delay + 1)
    
    # On the first step, n_accept_total is 0. Use the initial value.
    new_walks_float = jnp.where(
        n_accept_total == 0,
        walks_float,
        new_walks_float
    )

    # 3. Determine the integer number of walks for the *current* batch
    num_walks_int = jnp.minimum(jnp.ceil(new_walks_float).astype(jnp.int32), max_mcmc)
    
    # 4. Prepare parameters for the kernel
    n_dim = ns_state.particles.shape[-1]
    new_params = {
        # Standard params for the single-step kernel
        "live_points": ns_state.particles,
        "mix": 0.5,
        "scale": 2.38 / jnp.sqrt(2 * n_dim),
        
        # New dynamic param for the `while_loop` kernel
        "num_walks": num_walks_int,
        
        # State to be carried over to the next batch  
        "walks_float": new_walks_float,
        "n_accept_total": 0,  # Placeholder, will be updated by the wrapper
    }
    return new_params


def bilby_adaptive_de_sampler(
    logprior_fn: callable,
    loglikelihood_fn: callable,
    n_target: int = 60,
    max_mcmc: int = 5000,
    num_delete: int = 1,
) -> SamplingAlgorithm:
    """Creates a Bilby-style adaptive DE sampler for nested sampling.
    
    This sampler adapts the MCMC chain length (`walks`) at each batch based
    on the acceptance rate of the previous batch, while remaining fully
    compatible with the BlackJAX adaptive framework.
    
    Implements Bilby's adaptive 'acceptance-walk' algorithm where chain length adapts
    each batch to target a specific number of accepted steps (n_target).
    
    This exactly replicates Bilby's behavior using JAX-native operations:
    - Each batch runs for `num_walks` steps (determined adaptively)
    - Adaptation uses: new_walks = (old_walks * delay + n_target / accept_rate) / (delay + 1)
    - `delay` parameter based on ensemble size (nlive // 10 - 1)
    - Maximum chain length capped at `max_mcmc`
    
    Parameters
    ----------
    logprior_fn
        Log-prior density function operating on physical parameters.
    loglikelihood_fn
        Log-likelihood function operating on physical parameters.
    n_target
        Target number of accepted steps per batch (typically 60).
    max_mcmc
        Maximum number of MCMC steps per batch (typically 5000).
    num_delete
        Number of particles to replace at each NS iteration.
        
    Returns
    -------
    SamplingAlgorithm
        BlackJAX sampling algorithm with adaptive chain length.
    """
    delete_fn = partial(default_delete_fn, num_delete=num_delete)
    
    # Create a wrapper that matches the expected signature for BlackJAX adaptive framework
    def update_fn(ns_state, *args, **kwargs):
        """Wrapper that matches BlackJAX adaptive signature and calls our Bilby adaptation."""
        return update_bilby_walks_fn(
            ns_state=ns_state,
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            n_target=n_target,
            max_mcmc=max_mcmc,
            n_delete=num_delete,
        )

    # Build the base kernel using our dynamic walker and stateful update function
    # This kernel is almost complete, but it doesn't pass the info stats to the next state.
    base_kernel_step = build_adaptive_kernel(
        logprior_fn,
        loglikelihood_fn,
        delete_fn,
        de_rwalk_dynamic,
        update_fn,
    )

    # Create the init function separately using adaptive_init (same pattern as static version)
    init_fn = partial(
        adaptive_init,
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        update_inner_kernel_params_fn=update_fn,
    )
    
    # --- Create a single wrapper to pass state between steps ---
    # This is the key to connecting the output of one batch to the input of the next.
    def step_fn(rng_key, state: NSState):
        # 1. Call the underlying kernel. It uses the `num_walks` calculated
        #    from the *previous* batch's stats (stored in state.inner_kernel_params).
        new_state, info = base_kernel_step(rng_key, state)
        
        # 2. Extract the results from this batch's run. 
        #    BlackJAX wraps our DEWalkInfo in NSInfo, so we need to access the inner info
        #    The inner_info contains per-particle results, we need to sum across particles
        inner_info = info.inner_kernel_info
        batch_n_accept = jnp.sum(inner_info.n_accept)

        # 3. Manually update the `inner_kernel_params` in the returned state.
        #    The `new_state.inner_kernel_params` already contains the `walks_float`
        #    for the next step. We just need to add this batch's results.
        updated_params = new_state.inner_kernel_params.copy()
        updated_params['n_accept_total'] = batch_n_accept
        
        final_state = new_state._replace(inner_kernel_params=updated_params)
        
        return final_state, info

    return SamplingAlgorithm(init_fn, step_fn)