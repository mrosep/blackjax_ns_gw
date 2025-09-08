"""Unit hypercube wrappers for BlackJAX nested sampling."""

from typing import Callable
from functools import partial
import jax
import jax.numpy as jnp


def create_unit_cube_stepper(mask_tree):
    """Factory for unit cube stepper handling periodic parameters."""
    def stepper_fn(position, direction, step_size):
        proposed = jax.tree.map(lambda pos, d: pos + step_size * d, position, direction)
        return jax.tree.map(
            lambda prop, mask: jnp.where(mask, jnp.mod(prop, 1.0), prop),
            proposed, mask_tree
        )
    return stepper_fn


@partial(jax.jit, static_argnames=["physical_loglikelihood_fn", "prior_transform_fn"])
def unit_cube_loglikelihood(u_pytree, physical_loglikelihood_fn: Callable, prior_transform_fn: Callable):
    """Wraps physical loglikelihood to work on unit hypercube."""
    x_pytree = prior_transform_fn(u_pytree)
    return physical_loglikelihood_fn(x_pytree)


@jax.jit
def unit_cube_logprior(u_pytree):
    """Unit hypercube log-prior: 0.0 if in [0,1], -inf otherwise."""
    u_flat, _ = jax.flatten_util.ravel_pytree(u_pytree)
    is_in_bounds = jnp.all((u_flat >= 0.0) & (u_flat <= 1.0))
    return jnp.where(is_in_bounds, 0.0, -jnp.inf)


def create_unit_cube_functions(physical_loglikelihood_fn: Callable, prior_transform_fn: Callable, mask_tree):
    """Factory for complete unit cube sampling functions."""
    stepper_fn = create_unit_cube_stepper(mask_tree)
    loglikelihood_fn = partial(
        unit_cube_loglikelihood,
        physical_loglikelihood_fn=physical_loglikelihood_fn,
        prior_transform_fn=prior_transform_fn
    )
    return {
        'stepper_fn': stepper_fn,
        'loglikelihood_fn': loglikelihood_fn,
        'logprior_fn': unit_cube_logprior
    }


def init_unit_cube_particles(rng_key, example_pytree, n_particles: int):
    """Initialize live particles uniformly in unit hypercube."""
    leaves, treedef = jax.tree_util.tree_flatten(example_pytree)
    n_leaves = len(leaves)
    
    # Generate all random numbers in single efficient call
    all_rands_flat = jax.random.uniform(rng_key, shape=(n_leaves, n_particles))
    
    # Reconstruct pytree with random arrays
    return treedef.unflatten(all_rands_flat)



def transform_to_physical(samples_u, prior_transform_fn: Callable):
    """Transform unit hypercube samples to physical space."""
    return jax.vmap(prior_transform_fn)(samples_u)