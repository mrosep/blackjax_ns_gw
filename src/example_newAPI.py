#!/usr/bin/env python3
"""
Example script for BlackJAX-NS gravitational wave parameter estimation.

This version uses the CURRENT BlackJAX nested_sampling branch API
(post-October 2025), which uses StateWithLogLikelihood instead of
PartitionedState and has a refactored inner kernel interface.

Install the current branch:
    pip install blackjax@git+https://github.com/handley-lab/blackjax.git@nested_sampling

For the pinned-commit version (old API), see example.py instead.

Authors: Metha Prathaban et al.
Paper: "Gravitational-wave inference at GPU speed: A bilby-like nested sampling
       kernel within blackjax-ns"
"""

# =============================================================================
# SECTION 1: IMPORTS AND JAX CONFIGURATION
# =============================================================================

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import tqdm
import pickle

jax.config.update("jax_enable_x64", True)

# =============================================================================
# SECTION 2: GRAVITATIONAL WAVE IMPORTS (USER-REPLACEABLE)
# =============================================================================

from jimgw.single_event.detector import H1, L1, V1
from jimgw.single_event.likelihood import original_likelihood as likelihood_function
from jimgw.single_event.waveform import RippleIMRPhenomD

# Import the new-API kernel and shared unit cube utilities
from custom_kernels.acceptance_walk_newAPI import acceptance_walk_sampler_newapi
from custom_kernels import (
    create_unit_cube_functions,
    init_unit_cube_particles,
    transform_to_physical,
)

# =============================================================================
# SECTION 3: PROBLEM-SPECIFIC SETUP (USER-REPLACEABLE SECTION)
# =============================================================================

waveform = RippleIMRPhenomD(f_ref=50)

asd_paths = {
    "H1": "aLIGO_O4_high_asd.txt",
    "L1": "aLIGO_O4_high_asd.txt",
    "V1": "AdV_asd.txt",
}

injection_params = {
    "M_c": jnp.array(35.0),
    "q": jnp.array(0.9),
    "s1_z": jnp.array(0.4),
    "s2_z": jnp.array(-0.3),
    "d_L": jnp.array(1000.0),
    "iota": jnp.array(0.4),
    "t_c": jnp.array(0.0),
    "phase_c": jnp.array(1.3),
    "ra": jnp.array(1.375),
    "dec": jnp.array(-1.2108),
    "psi": jnp.array(2.659),
    "eta": jnp.array(0.9 / (1 + 0.9) ** 2),
}

# =============================================================================
# SECTION 4: DATA LOADING AND PREPROCESSING (USER-REPLACEABLE)
# =============================================================================

def get_ravel_order(particles_dict):
    """Determine the order that ravel_pytree uses for flattening parameter dictionaries."""
    example = jax.tree_util.tree_map(lambda x: x[0], particles_dict)
    flat, _ = jax.flatten_util.ravel_pytree(example)

    test_dict = {key: float(i) for i, key in enumerate(particles_dict.keys())}
    test_flat, _ = jax.flatten_util.ravel_pytree(test_dict)

    order = []
    for val in test_flat:
        for key, test_val in test_dict.items():
            if abs(val - test_val) < 1e-10:
                order.append(key)
                break
    return order

detector_data = {
    'frequencies': jnp.array(np.load('4s_frequency_array.npy')),
    'H1': jnp.array(np.load('4s_H1_strain.npy')),
    'L1': jnp.array(np.load('4s_L1_strain.npy')),
    'V1': jnp.array(np.load('4s_V1_strain.npy'))
}

freq_range = {'min': 20.0, 'max': 1024.0}
freq_mask = (detector_data['frequencies'] >= freq_range['min']) & \
            (detector_data['frequencies'] <= freq_range['max'])
filtered_frequencies = detector_data['frequencies'][freq_mask]

detectors = [H1, L1, V1]
detector_names = ['H1', 'L1', 'V1']

for det, name in zip(detectors, detector_names):
    det.frequencies = filtered_frequencies
    det.data = detector_data[name][freq_mask]

def load_psd_data(asd_paths):
    """Load power spectral density data for noise characterization."""
    psd_data = {}
    for name, path in asd_paths.items():
        f_np, asd_vals_np = np.loadtxt(path, unpack=True)
        psd_data[name] = {
            'frequencies': jnp.array(f_np),
            'psd': jnp.array(asd_vals_np**2)
        }
    return psd_data

psd_data = load_psd_data(asd_paths)

@jax.jit
def interpolate_psd(det_frequencies, psd_frequencies, psd_values):
    """Interpolate PSD values to detector frequency grid."""
    return jnp.interp(det_frequencies, psd_frequencies, psd_values)

for det in detectors:
    det.psd = interpolate_psd(
        det.frequencies,
        psd_data[det.name]['frequencies'],
        psd_data[det.name]['psd']
    )

# =============================================================================
# SECTION 5: PARAMETER CONFIGURATION (USER-CUSTOMIZABLE)
# =============================================================================

sample_keys = ["M_c", "q", "s1_z", "s2_z", "iota", "d_L", "t_c", "psi", "ra", "dec", "phase_c"]

test_particles = {}
for i, key in enumerate(sample_keys):
    test_particles[key] = jax.random.uniform(jax.random.PRNGKey(42), (100,))
sample_keys = get_ravel_order(test_particles)

param_config = {
    "M_c": {"min": 25.0, "max": 50.0, "prior": "uniform", "wraparound": False},
    "q": {"min": 0.25, "max": 1.0, "prior": "uniform", "wraparound": False},
    "s1_z": {"min": -1.0, "max": 1.0, "prior": "uniform", "wraparound": False},
    "s2_z": {"min": -1.0, "max": 1.0, "prior": "uniform", "wraparound": False},
    "iota": {"min": 0.0, "max": jnp.pi, "prior": "sine", "wraparound": False},
    "d_L": {"min": 100.0, "max": 5000.0, "prior": "powerlaw", "wraparound": False},
    "t_c": {"min": -0.1, "max": 0.1, "prior": "uniform", "wraparound": False},
    "phase_c": {"min": 0.0, "max": 2*jnp.pi, "prior": "uniform", "wraparound": True},
    "psi": {"min": 0.0, "max": jnp.pi, "prior": "uniform", "wraparound": True},
    "ra": {"min": 0.0, "max": 2*jnp.pi, "prior": "uniform", "wraparound": True},
    "dec": {"min": -jnp.pi/2, "max": jnp.pi/2, "prior": "cosine", "wraparound": False},
}

sampled_config = {key: param_config[key] for key in sample_keys}
n_dims = len(sample_keys)

param_mins = jnp.array([sampled_config[key]["min"] for key in sample_keys])
param_maxs = jnp.array([sampled_config[key]["max"] for key in sample_keys])
param_prior_types = jnp.array([
    0 if sampled_config[key]["prior"] == "uniform" else
    1 if sampled_config[key]["prior"] == "sine" else
    2 if sampled_config[key]["prior"] == "cosine" else
    3 for key in sample_keys  # 3 = powerlaw
])

post_trigger_duration = 2
duration = 4
epoch = duration - post_trigger_duration
gmst = Time(1126259642.413, format="gps").sidereal_time("apparent", "greenwich").rad

# =============================================================================
# SECTION 6: PRIOR TRANSFORMATIONS (USER MUST IMPLEMENT)
# =============================================================================

@jax.jit
def uniform_transform(u, a, b):
    return a + u * (b - a)

@jax.jit
def sine_transform(u):
    return jnp.arccos(1 - 2 * u)

@jax.jit
def cosine_transform(u):
    return jnp.arcsin(2 * u - 1)

@jax.jit
def powerlaw_transform(u, alpha, min_val, max_val):
    return (min_val ** (1+alpha) + u * (max_val ** (1+alpha) - min_val ** (1+alpha))) ** (1/(1+alpha))

@jax.jit
def prior_transform_fn(u_params):
    """Transform parameters from unit hypercube [0,1]^n to physical space."""
    u_values, _ = jax.flatten_util.ravel_pytree(u_params)

    uniform_vals = uniform_transform(u_values, param_mins, param_maxs)
    sine_vals = sine_transform(u_values)
    cosine_vals = cosine_transform(u_values)
    powerlaw_vals = powerlaw_transform(u_values, 2, param_mins, param_maxs)

    x_values = jnp.where(
        param_prior_types == 0, uniform_vals,
        jnp.where(
            param_prior_types == 1, sine_vals,
            jnp.where(
                param_prior_types == 2, cosine_vals,
                powerlaw_vals
            )
        )
    )

    example_params = {key: 0.0 for key in sample_keys}
    _, unflatten_fn = jax.flatten_util.ravel_pytree(example_params)
    return unflatten_fn(x_values)

@jax.jit
def uniform_logprob(x, a, b):
    return jnp.where((x >= a) & (x <= b), -jnp.log(b - a), -jnp.inf)

@jax.jit
def sine_logprob(x):
    return jnp.where((x >= 0.0) & (x <= jnp.pi), jnp.log(jnp.sin(x) / 2.0), -jnp.inf)

@jax.jit
def cosine_logprob(x):
    return jnp.where(jnp.abs(x) < jnp.pi / 2, jnp.log(jnp.cos(x) / 2.0), -jnp.inf)

@jax.jit
def powerlaw_logprob(x, alpha, min_val, max_val):
    logpdf = alpha*jnp.log(x) + jnp.log(1+alpha) - jnp.log(max_val ** (1+alpha) - min_val ** (1+alpha))
    return jnp.where((x >= min_val) & (x <= max_val), logpdf, -jnp.inf)

@jax.jit
def logprior_fn(params):
    """Compute the log-prior probability of parameters in physical space."""
    param_values, _ = jax.flatten_util.ravel_pytree(params)

    uniform_priors = uniform_logprob(param_values, param_mins, param_maxs)
    sine_priors = sine_logprob(param_values)
    cosine_priors = cosine_logprob(param_values)
    powerlaw_priors = powerlaw_logprob(param_values, 2, param_mins, param_maxs)

    priors = jnp.where(
        param_prior_types == 0, uniform_priors,
        jnp.where(
            param_prior_types == 1, sine_priors,
            jnp.where(
                param_prior_types == 2, cosine_priors,
                powerlaw_priors
            )
        )
    )

    return jnp.sum(priors)

# =============================================================================
# SECTION 7: LIKELIHOOD FUNCTION (USER MUST IMPLEMENT)
# =============================================================================

def loglikelihood_fn(params):
    """Compute the log-likelihood of parameters given the data."""
    p = injection_params.copy()
    p.update(params)
    p["gmst"] = gmst
    p["eta"] = p["q"] / (1 + p["q"]) ** 2

    waveform_sky = waveform(filtered_frequencies, p)
    align_time = jnp.exp(-1j * 2 * jnp.pi * filtered_frequencies * (epoch + p["t_c"]))

    return likelihood_function(p, waveform_sky, detectors, filtered_frequencies, align_time)

# =============================================================================
# SECTION 8: NESTED SAMPLING CONFIGURATION
# =============================================================================

n_live = 1400
n_delete = int(n_live * 0.5)

rng_key = jax.random.PRNGKey(10)
rng_key, init_key = jax.random.split(rng_key, 2)

example_params = {key: 0.0 for key in sample_keys}

# Initialize particles in unit hypercube [0,1]^n
unit_cube_particles = init_unit_cube_particles(init_key, example_params, n_live)

# Create periodic mask for parameters with wraparound boundaries
periodic_mask = jax.tree_util.tree_map(lambda _: False, example_params)
for key in sample_keys:
    if sampled_config[key]["wraparound"]:
        periodic_mask[key] = True

# Create unit cube wrapper functions
unit_cube_fns = create_unit_cube_functions(
    physical_loglikelihood_fn=loglikelihood_fn,
    prior_transform_fn=prior_transform_fn,
    mask_tree=periodic_mask
)

# =============================================================================
# SECTION 9: INITIALIZE THE NESTED SAMPLER (NEW API)
# =============================================================================

# Configure the acceptance walk nested sampler using the new BlackJAX API
nested_sampler = acceptance_walk_sampler_newapi(
    logprior_fn=unit_cube_fns['logprior_fn'],
    loglikelihood_fn=unit_cube_fns['loglikelihood_fn'],
    nlive=n_live,
    n_target=60,
    max_mcmc=5000,
    num_delete=n_delete,
    stepper_fn=unit_cube_fns['stepper_fn'],
)

# Initialize sampler state
state = nested_sampler.init(unit_cube_particles)

# =============================================================================
# SECTION 10: RUN THE NESTED SAMPLING
# =============================================================================

@jax.jit
def one_step(carry, xs):
    """Single nested sampling iteration (JIT-compiled for speed)."""
    state, k = carry
    k, subk = jax.random.split(k, 2)
    state, dead_point = nested_sampler.step(subk, state)
    return (state, k), dead_point

def terminate(state):
    """Termination condition: stop when remaining evidence is small.

    In the new API, evidence estimates are in state.integrator.
    """
    dlogz = jnp.logaddexp(0, state.integrator.logZ_live - state.integrator.logZ)
    return jnp.isfinite(dlogz) and dlogz < 0.1

# Run nested sampling with progress bar
print("Starting nested sampling (new BlackJAX API)...")
print(f"Configuration: {n_live} live points, batch size {n_delete}")
print("Termination condition: dlogZ < 0.1")

dead = []
with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
    while not terminate(state):
        (state, rng_key), dead_info = one_step((state, rng_key), None)
        dead.append(dead_info)
        pbar.update(n_delete)

print(f"Nested sampling completed! Generated {len(dead) * n_delete} dead points.")

# =============================================================================
# SECTION 11: PROCESS AND SAVE RESULTS
# =============================================================================

from blackjax.ns.utils import finalise
from anesthetic import NestedSamples

# Finalize the run: combines dead particles with final live points
# In the new API, finalise returns NSInfo with .particles (StateWithLogLikelihood)
final_info = finalise(state, dead)

# Save the final state
with open('example_newapi_final_state.pkl', 'wb') as f:
    pickle.dump(final_info, f)

# Transform samples back to physical parameter space
# In the new API, positions are at .particles.position (not .particles directly)
physical_particles = transform_to_physical(final_info.particles.position, prior_transform_fn)

# Create column labels for output
column_to_label = {
    "M_c": r"$M_c$", "q": r"$q$", "d_L": r"$d_L$", "iota": r"$\iota$", "ra": r"$\alpha$",
    "dec": r"$\delta$", "s1_z": r"$s_{1z}$", "s2_z": r"$s_{2z}$", "t_c": r"$t_c$",
    "psi": r"$\psi$", "phase_c": r"$\phi_c$",
}

# Handle potential NaN values in log-likelihood birth
logL_birth = final_info.particles.loglikelihood_birth.copy()
logL_birth = jnp.where(jnp.isnan(logL_birth), -jnp.inf, logL_birth)

# Create anesthetic NestedSamples object
samples = NestedSamples(
    physical_particles,
    logL=final_info.particles.loglikelihood,
    logL_birth=logL_birth,
    labels=column_to_label,
    logzero=jnp.nan,
    dtype=jnp.float64,
)

samples.to_csv("example_newapi_results.csv")
print("Results saved to example_newapi_results.csv")

with open('example_newapi_timings.pkl', 'wb') as f:
    pickle.dump(pbar.format_dict, f)

# =============================================================================
# SECTION 12: ANALYSIS AND DIAGNOSTICS
# =============================================================================

# Evidence from the integrator (tracked during sampling)
print(f"\nAnalysis Summary:")
print(f"Log Evidence (from integrator): {state.integrator.logZ:.2f}")
print(f"Log Evidence (live contribution): {state.integrator.logZ_live:.2f}")
print(f"Number of dead points: {len(dead) * n_delete}")

print("\nExample (new API) completed successfully!")
print("For advanced analysis, use anesthetic or other tools to process example_newapi_results.csv")
