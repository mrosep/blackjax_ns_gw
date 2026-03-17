"""GPU-optimized JAX-native Bilby adaptive DE kernel with early termination.

This version uses while_loop with per-particle early termination to avoid running
unnecessary iterations after a particle completes its target walks.

Key improvements over acceptance_walk.py:
1. Replaces nested while loops with single while loop that terminates early
2. Uses is_active mask combined with iteration counter for termination
3. Two-stage masking for likelihood evaluation (is_active & is_valid)
4. Each particle stops when it reaches target walks (no wasted iterations)

Note: This reintroduces some thread divergence (particles finish at different times)
but avoids the massive overhead of running 5000 iterations for all particles.
"""

from typing import NamedTuple, Callable
from functools import partial

import jax
import jax.numpy as jnp

from blackjax.base import SamplingAlgorithm
from blackjax.ns.base import PartitionedState, NSState
from blackjax.ns.adaptive import build_kernel as build_adaptive_kernel
from blackjax.ns.adaptive import init as adaptive_init
from blackjax.ns.base import delete_fn as default_delete_fn


class DEWalkInfo(NamedTuple):
    """Diagnostic information for a full DE MCMC walk."""
    n_accept: int
    walks_completed: int
    n_likelihood_evals: int
    total_proposals: int


class DEKernelParams(NamedTuple):
    """Static pytree for DE kernel parameters."""
    live_points: jax.Array
    loglikelihoods: jax.Array
    mix: float
    scale: float
    num_walks: jax.Array  # Target walks per particle
    walks_float: jax.Array
    n_accept_total: jax.Array
    n_likelihood_evals_total: jax.Array


def de_rwalk_scan_unit_cube(
    rng_key: jax.Array,
    state: PartitionedState,
    logprior_fn: callable,
    loglikelihood_fn: callable,
    loglikelihood_0: float,
    params: DEKernelParams,
    stepper_fn: callable,
    num_survivors: int,
    max_mcmc: int = 5000,
):
    """GPU-optimized MCMC walk using while_loop with early termination.

    This function uses a while loop that terminates when the particle completes
    its target walks OR hits the max_mcmc budget. Uses masking for GPU efficiency.

    Args:
        rng_key: JAX random key
        state: Initial PartitionedState (single particle when vmapped)
        logprior_fn: Function to evaluate log prior
        loglikelihood_fn: Function to evaluate log likelihood
        loglikelihood_0: Likelihood threshold for nested sampling
        params: DE kernel parameters (shared across all particles)
        stepper_fn: Function to take a step with periodic boundary handling
        num_survivors: Number of top live points to use for DE proposals
        max_mcmc: Maximum number of iterations (hard budget)

    Returns:
        final_state: Final PartitionedState after max_mcmc iterations or early stop
        info: DEWalkInfo with statistics
    """

    # Internal state for the loop
    class LoopState(NamedTuple):
        position: jax.Array  # Current position
        logprior: float  # Log prior at current position
        loglikelihood: float  # Log likelihood at current position
        walks_completed: int  # Number of accepted walks
        total_proposals: int  # Total proposals made
        n_likelihood_evals: int  # Number of likelihood evaluations
        is_active: bool  # Whether this particle is still working
        rng_key: jax.Array  # Current random key
        iteration_count: int  # Track iterations for max_mcmc budget

    def loop_body(loop_state: LoopState):
        """Single iteration of the MCMC loop with early termination."""

        # Split RNG key
        key_a, key_b, key_mix, key_gamma, next_key = jax.random.split(loop_state.rng_key, 5)

        # ===== 1. Generate DE Proposal =====
        # Select two random survivors for differential evolution
        _, top_indices = jax.lax.top_k(params.loglikelihoods, num_survivors)
        pos_a = jax.random.randint(key_a, (), 0, num_survivors)
        pos_b_raw = jax.random.randint(key_b, (), 0, num_survivors - 1)
        pos_b = jnp.where(pos_b_raw >= pos_a, pos_b_raw + 1, pos_b_raw)

        point_a = jax.tree_util.tree_map(lambda x: x[top_indices[pos_a]], params.live_points)
        point_b = jax.tree_util.tree_map(lambda x: x[top_indices[pos_b]], params.live_points)
        delta = jax.tree_util.tree_map(lambda a, b: a - b, point_a, point_b)

        # Compute step size (mix of small and large steps)
        is_small_step = jax.random.uniform(key_mix) < params.mix
        gamma = jnp.where(is_small_step,
                         params.scale * jax.random.gamma(key_gamma, 4.0) * 0.25,
                         1.0)

        # Generate proposal
        proposal = stepper_fn(loop_state.position, delta, gamma)

        # ===== 2. Check Prior Bounds (cheap) =====
        proposal_logprior = logprior_fn(proposal)
        is_valid = jnp.isfinite(proposal_logprior)

        # ===== 3. Two-Stage Masking for Likelihood =====
        # Only evaluate likelihood for active particles with valid proposals
        needs_likelihood = loop_state.is_active & is_valid

        # Evaluate likelihood (expensive, but only "use" masked results)
        proposal_logl = loglikelihood_fn(proposal)

        # Use sentinel value for proposals we don't care about
        proposal_logl_masked = jnp.where(needs_likelihood, proposal_logl, -jnp.inf)

        # ===== 4. Accept/Reject Decision =====
        is_accepted = proposal_logl_masked > loglikelihood_0

        # ===== 5. Update State (only for active & accepted) =====
        should_update = loop_state.is_active & is_accepted

        new_position = jax.tree_util.tree_map(
            lambda prop, curr: jnp.where(should_update, prop, curr),
            proposal,
            loop_state.position
        )
        new_logprior = jnp.where(should_update, proposal_logprior, loop_state.logprior)
        new_logl = jnp.where(should_update, proposal_logl_masked, loop_state.loglikelihood)

        # Update counters
        new_walks = loop_state.walks_completed + should_update.astype(jnp.int32)
        new_proposals = loop_state.total_proposals + loop_state.is_active.astype(jnp.int32)
        new_likelihood_evals = loop_state.n_likelihood_evals + needs_likelihood.astype(jnp.int32)

        # ===== 6. Update Termination Mask for NEXT Iteration =====
        finished_walks = new_walks >= params.num_walks
        hit_budget = new_proposals >= max_mcmc
        new_is_active = loop_state.is_active & ~finished_walks & ~hit_budget

        # Increment iteration counter
        new_iteration_count = loop_state.iteration_count + 1

        # Build new state
        new_loop_state = LoopState(
            position=new_position,
            logprior=new_logprior,
            loglikelihood=new_logl,
            walks_completed=new_walks,
            total_proposals=new_proposals,
            n_likelihood_evals=new_likelihood_evals,
            is_active=new_is_active,
            rng_key=next_key,
            iteration_count=new_iteration_count,
        )

        return new_loop_state

    def loop_cond(loop_state: LoopState):
        """Continue loop while particle is active AND under budget."""
        return loop_state.is_active & (loop_state.iteration_count < max_mcmc)

    # Initialize loop state
    init_loop_state = LoopState(
        position=state.position,
        logprior=state.logprior,
        loglikelihood=state.loglikelihood,
        walks_completed=jnp.array(0, dtype=jnp.int32),
        total_proposals=jnp.array(0, dtype=jnp.int32),
        n_likelihood_evals=jnp.array(0, dtype=jnp.int32),
        is_active=jnp.array(True, dtype=jnp.bool_),
        rng_key=rng_key,
        iteration_count=jnp.array(0, dtype=jnp.int32),
    )

    # Run while loop with early termination when particle is done
    final_loop_state = jax.lax.while_loop(loop_cond, loop_body, init_loop_state)

    # Convert back to PartitionedState
    final_state = PartitionedState(
        position=final_loop_state.position,
        logprior=final_loop_state.logprior,
        loglikelihood=final_loop_state.loglikelihood,
    )

    # Build info
    info = DEWalkInfo(
        n_accept=final_loop_state.walks_completed,
        walks_completed=final_loop_state.walks_completed,
        n_likelihood_evals=final_loop_state.n_likelihood_evals,
        total_proposals=final_loop_state.total_proposals,
    )

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

    # Define default values with explicit dtypes
    default_walks_float = jnp.array(100.0, dtype=jnp.float32)
    default_n_accept_total = jnp.array(0, dtype=jnp.int32)
    default_current_walks = jnp.array(100, dtype=jnp.int32)
    default_n_likelihood_evals_total = jnp.array(0, dtype=jnp.int32)

    # Get values from previous state and explicitly cast
    param_walks_float = prev_params.walks_float.astype(jnp.float32)
    param_n_accept_total = prev_params.n_accept_total.astype(jnp.int32)
    param_current_walks = prev_params.num_walks.astype(jnp.int32)
    param_n_likelihood_evals_total = prev_params.n_likelihood_evals_total.astype(jnp.int32)

    # Use jnp.where for branchless selection
    walks_float = jnp.where(is_uninitialized, default_walks_float, param_walks_float)
    n_accept_total = jnp.where(is_uninitialized, default_n_accept_total, param_n_accept_total)
    current_walks = jnp.where(is_uninitialized, default_current_walks, param_current_walks)
    n_likelihood_evals_total = jnp.where(is_uninitialized, default_n_likelihood_evals_total, param_n_likelihood_evals_total)

    leaves = jax.tree_util.tree_leaves(ns_state.particles)
    nlive = leaves[0].shape[0]
    og_delay = nlive // 10 - 1
    delay = jnp.maximum(og_delay // n_delete, 1)

    # Bilby's walk length tuning formula
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


def bilby_adaptive_de_sampler_unit_cube_improved(
    logprior_fn: callable,
    loglikelihood_fn: callable,
    nlive: int,
    n_target: int = 60,
    max_mcmc: int = 5000,
    num_delete: int = 1,
    stepper_fn: callable = None,
) -> SamplingAlgorithm:
    """GPU-optimized Bilby adaptive DE sampler using scan + masking.

    This version eliminates thread divergence by replacing while loops with
    fixed-iteration scan loops and boolean masking.

    IMPORTANT: Pass NON-vmapped logprior_fn and loglikelihood_fn!
    - The inner kernel is vmapped, so each instance works with single particles
    - init_fn and update_fn vmap the functions internally for batch evaluation
    """
    if stepper_fn is None:
        raise ValueError("stepper_fn must be provided for unit cube sampling")

    # Calculate num_survivors statically as a Python integer
    num_survivors = nlive - num_delete

    delete_fn = partial(default_delete_fn, num_delete=num_delete)

    # Vmapped versions for init and update_fn (which evaluate on all particles)
    vmapped_logprior = jax.vmap(logprior_fn)
    vmapped_loglikelihood = jax.vmap(loglikelihood_fn)

    def update_fn(ns_state, *args, **kwargs):
        return update_bilby_walks_fn(
            ns_state=ns_state,
            logprior_fn=vmapped_logprior,
            loglikelihood_fn=vmapped_loglikelihood,
            n_target=n_target,
            max_mcmc=max_mcmc,
            n_delete=num_delete,
        )

    kernel_with_stepper = partial(
        de_rwalk_scan_unit_cube,
        stepper_fn=stepper_fn,
        num_survivors=num_survivors,
        max_mcmc=max_mcmc,
    )

    # FIX: vmap the inner kernel for parallel execution over particles
    # in_axes: (rng_key=0, state=0, logprior_fn=None, loglikelihood_fn=None,
    #           loglikelihood_0=None, params=None)
    vmapped_kernel = jax.vmap(kernel_with_stepper, in_axes=(0, 0, None, None, None, None))

    base_kernel_step = build_adaptive_kernel(
        logprior_fn,  # Non-vmapped for inner kernel (single particle)
        loglikelihood_fn,
        delete_fn,
        vmapped_kernel,
        update_fn,
    )

    def init_fn(particles):
        # Use vmapped functions for initialization (evaluates on all particles)
        state = adaptive_init(
            particles=particles,
            logprior_fn=vmapped_logprior,
            loglikelihood_fn=vmapped_loglikelihood,
            update_inner_kernel_params_fn=None,
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
            num_walks=jnp.array(100, dtype=jnp.int32),
            walks_float=jnp.array(100.0, dtype=jnp.float32),
            n_accept_total=jnp.array(-1, dtype=jnp.int32),  # Sentinel flag
            n_likelihood_evals_total=jnp.array(-1, dtype=jnp.int32),  # Sentinel flag
        )

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
