"""GPU-optimized acceptance walk kernel for the current BlackJAX nested_sampling API.

This kernel is adapted from acceptance_walk_improved.py to work with the current
(post-October 2025) BlackJAX nested_sampling branch, which replaced PartitionedState
with StateWithLogLikelihood and changed the inner kernel interface.

Key API changes from the old version:
- PartitionedState -> StateWithLogLikelihood
  (fields: position, logdensity, loglikelihood, loglikelihood_birth)
- Inner kernel signature: (rng_key, state, loglikelihood_0, **params)
  where state is the full NS state
- Inner kernel handles start particle selection internally
- update_inner_kernel_params_fn: (rng_key, state, info, params) -> new_params

Install the current BlackJAX nested_sampling branch:
    pip install blackjax@git+https://github.com/handley-lab/blackjax.git@nested_sampling
"""

from typing import NamedTuple
from functools import partial

import jax
import jax.numpy as jnp

from blackjax.base import SamplingAlgorithm
from blackjax.ns.base import StateWithLogLikelihood, NSState
from blackjax.ns.base import init_state_strategy
from blackjax.ns.base import delete_fn as default_delete_fn
from blackjax.ns.adaptive import build_kernel as build_adaptive_kernel
from blackjax.ns.adaptive import init as adaptive_init


class DEWalkInfo(NamedTuple):
    """Diagnostic information for a full DE MCMC walk."""
    n_accept: int
    walks_completed: int
    n_likelihood_evals: int
    total_proposals: int


class DEKernelParams(NamedTuple):
    """Parameters for the DE kernel, updated adaptively between NS steps."""
    live_points: jax.Array
    loglikelihoods: jax.Array
    mix: float
    scale: float
    num_walks: jax.Array
    walks_float: jax.Array
    n_accept_total: jax.Array
    n_likelihood_evals_total: jax.Array


def de_rwalk_single_particle(
    rng_key: jax.Array,
    state: StateWithLogLikelihood,
    logprior_fn: callable,
    loglikelihood_fn: callable,
    loglikelihood_0: float,
    de_params: DEKernelParams,
    stepper_fn: callable,
    num_survivors: int,
    max_mcmc: int = 5000,
):
    """GPU-optimized MCMC walk for a single particle with early termination.

    Uses while_loop that terminates when the particle completes its target walks
    OR hits the max_mcmc budget. Designed to be vmapped over num_delete particles.

    Args:
        rng_key: JAX random key
        state: Initial StateWithLogLikelihood (single particle when vmapped)
        logprior_fn: Function to evaluate log prior (single particle)
        loglikelihood_fn: Function to evaluate log likelihood (single particle)
        loglikelihood_0: Likelihood threshold for nested sampling
        de_params: DE kernel parameters (shared across all particles)
        stepper_fn: Step function with periodic boundary handling
        num_survivors: Number of top live points to use for DE proposals
        max_mcmc: Maximum number of iterations (hard budget)

    Returns:
        final_state: Final StateWithLogLikelihood
        info: DEWalkInfo with statistics
    """

    class LoopState(NamedTuple):
        position: jax.Array
        logdensity: float
        loglikelihood: float
        walks_completed: int
        total_proposals: int
        n_likelihood_evals: int
        is_active: bool
        rng_key: jax.Array
        iteration_count: int

    def loop_body(loop_state: LoopState):
        """Single iteration of the MCMC loop with early termination."""
        key_a, key_b, key_mix, key_gamma, next_key = jax.random.split(loop_state.rng_key, 5)

        # DE proposal from top survivors
        _, top_indices = jax.lax.top_k(de_params.loglikelihoods, num_survivors)
        pos_a = jax.random.randint(key_a, (), 0, num_survivors)
        pos_b_raw = jax.random.randint(key_b, (), 0, num_survivors - 1)
        pos_b = jnp.where(pos_b_raw >= pos_a, pos_b_raw + 1, pos_b_raw)

        point_a = jax.tree_util.tree_map(lambda x: x[top_indices[pos_a]], de_params.live_points)
        point_b = jax.tree_util.tree_map(lambda x: x[top_indices[pos_b]], de_params.live_points)
        delta = jax.tree_util.tree_map(lambda a, b: a - b, point_a, point_b)

        is_small_step = jax.random.uniform(key_mix) < de_params.mix
        gamma = jnp.where(is_small_step,
                         de_params.scale * jax.random.gamma(key_gamma, 4.0) * 0.25,
                         1.0)

        proposal = stepper_fn(loop_state.position, delta, gamma)

        # Check prior bounds (cheap)
        proposal_logdensity = logprior_fn(proposal)
        is_valid = jnp.isfinite(proposal_logdensity)

        # Two-stage masking: only use likelihood result for active+valid proposals
        needs_likelihood = loop_state.is_active & is_valid
        proposal_logl = loglikelihood_fn(proposal)
        proposal_logl_masked = jnp.where(needs_likelihood, proposal_logl, -jnp.inf)

        # Accept/reject
        is_accepted = proposal_logl_masked > loglikelihood_0
        should_update = loop_state.is_active & is_accepted

        new_position = jax.tree_util.tree_map(
            lambda prop, curr: jnp.where(should_update, prop, curr),
            proposal, loop_state.position
        )
        new_logdensity = jnp.where(should_update, proposal_logdensity, loop_state.logdensity)
        new_logl = jnp.where(should_update, proposal_logl_masked, loop_state.loglikelihood)

        new_walks = loop_state.walks_completed + should_update.astype(jnp.int32)
        new_proposals = loop_state.total_proposals + loop_state.is_active.astype(jnp.int32)
        new_likelihood_evals = loop_state.n_likelihood_evals + needs_likelihood.astype(jnp.int32)

        # Check termination for next iteration
        finished_walks = new_walks >= de_params.num_walks
        hit_budget = new_proposals >= max_mcmc
        new_is_active = loop_state.is_active & ~finished_walks & ~hit_budget

        return LoopState(
            position=new_position,
            logdensity=new_logdensity,
            loglikelihood=new_logl,
            walks_completed=new_walks,
            total_proposals=new_proposals,
            n_likelihood_evals=new_likelihood_evals,
            is_active=new_is_active,
            rng_key=next_key,
            iteration_count=loop_state.iteration_count + 1,
        )

    def loop_cond(loop_state: LoopState):
        return loop_state.is_active & (loop_state.iteration_count < max_mcmc)

    init_loop_state = LoopState(
        position=state.position,
        logdensity=state.logdensity,
        loglikelihood=state.loglikelihood,
        walks_completed=jnp.array(0, dtype=jnp.int32),
        total_proposals=jnp.array(0, dtype=jnp.int32),
        n_likelihood_evals=jnp.array(0, dtype=jnp.int32),
        is_active=jnp.array(True, dtype=jnp.bool_),
        rng_key=rng_key,
        iteration_count=jnp.array(0, dtype=jnp.int32),
    )

    final = jax.lax.while_loop(loop_cond, loop_body, init_loop_state)

    final_state = StateWithLogLikelihood(
        position=final.position,
        logdensity=final.logdensity,
        loglikelihood=final.loglikelihood,
        loglikelihood_birth=loglikelihood_0 * jnp.ones_like(final.loglikelihood),
    )

    info = DEWalkInfo(
        n_accept=final.walks_completed,
        walks_completed=final.walks_completed,
        n_likelihood_evals=final.n_likelihood_evals,
        total_proposals=final.total_proposals,
    )

    return final_state, info


def acceptance_walk_sampler_newapi(
    logprior_fn: callable,
    loglikelihood_fn: callable,
    nlive: int,
    n_target: int = 60,
    max_mcmc: int = 5000,
    num_delete: int = 1,
    stepper_fn: callable = None,
) -> SamplingAlgorithm:
    """GPU-optimized Bilby adaptive DE sampler for the current BlackJAX NS API.

    Parameters
    ----------
    logprior_fn : callable
        Log-prior function for a SINGLE particle (non-vmapped).
    loglikelihood_fn : callable
        Log-likelihood function for a SINGLE particle (non-vmapped).
    nlive : int
        Number of live points.
    n_target : int
        Target number of accepted walks per chain.
    max_mcmc : int
        Maximum MCMC iterations per chain.
    num_delete : int
        Number of particles to replace per NS step.
    stepper_fn : callable
        Step function handling periodic boundaries in unit cube.

    Returns
    -------
    SamplingAlgorithm
        A BlackJAX SamplingAlgorithm with init and step methods.
    """
    if stepper_fn is None:
        raise ValueError("stepper_fn must be provided for unit cube sampling")

    num_survivors = nlive - num_delete
    delete_fn = partial(default_delete_fn, num_delete=num_delete)
    vmapped_logprior = jax.vmap(logprior_fn)
    vmapped_loglikelihood = jax.vmap(loglikelihood_fn)

    # Build the single-particle DE walk with static args baked in
    single_walk = partial(
        de_rwalk_single_particle,
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        stepper_fn=stepper_fn,
        num_survivors=num_survivors,
        max_mcmc=max_mcmc,
    )
    # Vmap over (rng_key, state), broadcast (loglikelihood_0, de_params)
    vmapped_walk = jax.vmap(single_walk, in_axes=(0, 0, None, None))

    def inner_kernel(rng_key, state, loglikelihood_0, de_params=None):
        """Inner kernel: select start particles and run DE walks.

        Called by the base NS kernel with signature
        (rng_key, state, loglikelihood_0, **inner_kernel_params).
        """
        particles = state.particles
        choice_key, sample_key = jax.random.split(rng_key)

        # Select start particles from survivors (above likelihood threshold)
        weights = (particles.loglikelihood > loglikelihood_0).astype(jnp.float32)
        weights = jnp.where(weights.sum() > 0.0, weights, jnp.ones_like(weights))
        start_idx = jax.random.choice(
            choice_key,
            particles.loglikelihood.shape[0],
            shape=(num_delete,),
            p=weights / weights.sum(),
            replace=True,
        )
        start_particles = jax.tree.map(lambda x: x[start_idx], particles)

        # Run vmapped DE walks
        sample_keys = jax.random.split(sample_key, num_delete)
        new_particles, info = vmapped_walk(
            sample_keys, start_particles, loglikelihood_0, de_params
        )
        return new_particles, info

    def update_inner_kernel_params_fn(rng_key, state, info, current_params):
        """Adapt DE kernel parameters based on acceptance statistics.

        Implements Bilby's walk length tuning formula.
        """
        de_params = current_params["de_params"]

        # Get current batch's acceptance stats from info
        batch_n_accept = jnp.sum(info.update_info.n_accept)
        batch_n_likelihood_evals = jnp.sum(info.update_info.n_likelihood_evals)

        # Check sentinel for first step
        is_first_step = de_params.n_accept_total < 0

        walks_float = jnp.where(
            is_first_step,
            jnp.array(100.0, dtype=jnp.float32),
            de_params.walks_float.astype(jnp.float32),
        )
        n_accept_prev = jnp.where(
            is_first_step,
            jnp.array(0, dtype=jnp.int32),
            de_params.n_accept_total.astype(jnp.int32),
        )
        current_walks = jnp.where(
            is_first_step,
            jnp.array(100, dtype=jnp.int32),
            de_params.num_walks.astype(jnp.int32),
        )

        # Bilby walk length tuning
        nlive_val = state.particles.loglikelihood.shape[0]
        og_delay = nlive_val // 10 - 1
        delay = jnp.maximum(og_delay // num_delete, 1)

        avg_accept = n_accept_prev / num_delete
        accept_prob = jnp.maximum(0.5, avg_accept) / jnp.maximum(1.0, current_walks)
        new_walks_float = (walks_float * delay + n_target / accept_prob) / (delay + 1)
        new_walks_float = jnp.where(n_accept_prev == 0, walks_float, new_walks_float)
        num_walks_int = jnp.minimum(
            jnp.ceil(new_walks_float).astype(jnp.int32), max_mcmc
        )

        new_de_params = DEKernelParams(
            live_points=state.particles.position,
            loglikelihoods=state.particles.loglikelihood,
            mix=0.5,
            scale=de_params.scale,
            num_walks=jnp.array(num_walks_int, dtype=jnp.int32),
            walks_float=jnp.array(new_walks_float, dtype=jnp.float32),
            n_accept_total=jnp.array(batch_n_accept, dtype=jnp.int32),
            n_likelihood_evals_total=jnp.array(batch_n_likelihood_evals, dtype=jnp.int32),
        )
        return {"de_params": new_de_params}

    # Build the adaptive kernel
    adaptive_kernel = build_adaptive_kernel(
        delete_fn,
        inner_kernel,
        update_inner_kernel_params_fn,
    )

    # init_state_fn for creating StateWithLogLikelihood from positions
    init_state_fn = partial(
        init_state_strategy,
        logprior_fn=vmapped_logprior,
        loglikelihood_fn=vmapped_loglikelihood,
    )

    def init_fn(particles, rng_key=None):
        """Initialize the nested sampler state.

        Parameters
        ----------
        particles : PyTree
            Initial positions in unit hypercube, each leaf shape (nlive, ...).
        rng_key : PRNGKey, optional
            Not used, kept for API compatibility.
        """
        state = adaptive_init(
            positions=particles,
            init_state_fn=init_state_fn,
            update_inner_kernel_params_fn=None,
        )

        # Compute scale from dimensionality
        example_particle = jax.tree_util.tree_map(lambda x: x[0], particles)
        flat_particle, _ = jax.flatten_util.ravel_pytree(example_particle)
        n_dim = flat_particle.shape[0]
        scale = 2.38 / jnp.sqrt(2 * n_dim)

        # Create initial DE params with sentinel
        initial_de_params = DEKernelParams(
            live_points=particles,
            loglikelihoods=state.particles.loglikelihood,
            mix=0.5,
            scale=scale,
            num_walks=jnp.array(100, dtype=jnp.int32),
            walks_float=jnp.array(100.0, dtype=jnp.float32),
            n_accept_total=jnp.array(-1, dtype=jnp.int32),
            n_likelihood_evals_total=jnp.array(-1, dtype=jnp.int32),
        )

        return state._replace(inner_kernel_params={"de_params": initial_de_params})

    def step_fn(rng_key, state):
        """Perform one nested sampling step."""
        return adaptive_kernel(rng_key, state)

    return SamplingAlgorithm(init_fn, step_fn)
