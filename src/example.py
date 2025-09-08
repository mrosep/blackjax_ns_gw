#!/usr/bin/env python3
"""
Example script demonstrating how to use the BlackJAX-NS acceptance walk kernel
for gravitational wave parameter estimation.

This example shows how to:
1. Set up your data and likelihood function
2. Define prior transformations and log-prior functions  
3. Configure and run the acceptance walk nested sampler
4. Process and save the results

IMPORTANT: This script uses a 4-second binary black hole signal as an example,
but users can replace the data loading, likelihood function, and priors with 
their own implementations following the same structure.

Authors: Metha Prathaban et al.
Paper: "Gravitational-wave inference at GPU speed: A bilby-like nested sampling 
       kernel within blackjax-ns"
"""

# =============================================================================
# SECTION 1: IMPORTS AND JAX CONFIGURATION
# =============================================================================

# JAX memory and precision configuration - adjust as needed for your system
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"  # Use 60% of GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Don't preallocate all memory

import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import matplotlib.pyplot as plt
from astropy.time import Time
import tqdm
import pickle

# Enable 64-bit precision for numerical accuracy (important for likelihood calculations)
jax.config.update("jax_enable_x64", True)

# =============================================================================
# SECTION 2: GRAVITATIONAL WAVE IMPORTS (USER-REPLACEABLE)
# =============================================================================

# Import gravitational wave functions - REPLACE WITH YOUR OWN IMPORTS
# These are specific to gravitational wave analysis using jimgw/ripple
from jimgw.single_event.detector import H1, L1, V1
from jimgw.single_event.likelihood import original_likelihood as likelihood_function
from jimgw.single_event.waveform import RippleIMRPhenomD

# Import our custom BlackJAX nested sampling kernels
from custom_kernels import (
    acceptance_walk_sampler,
    create_unit_cube_functions,
    init_unit_cube_particles,
    transform_to_physical
)

# =============================================================================
# SECTION 3: PROBLEM-SPECIFIC SETUP (USER-REPLACEABLE SECTION)
# =============================================================================

# Initialize waveform model - REPLACE WITH YOUR WAVEFORM MODEL
# For non-GW applications, replace with your forward model
waveform = RippleIMRPhenomD(f_ref=50)

# Data file paths - REPLACE WITH YOUR DATA FILES
# These should point to your detector data and noise curves
asd_paths = {
    "H1": "aLIGO_O4_high_asd.txt",  # Hanford noise curve
    "L1": "aLIGO_O4_high_asd.txt",  # Livingston noise curve  
    "V1": "AdV_asd.txt",            # Virgo noise curve
}

# Example injection parameters for a binary black hole - REPLACE WITH YOUR PARAMETERS
# For other applications, these would be your "true" parameter values for testing
injection_params = {
    "M_c": jnp.array(35.0),      # Chirp mass [solar masses]
    "q": jnp.array(0.9),         # Mass ratio  
    "s1_z": jnp.array(0.4),      # Primary spin (aligned component)
    "s2_z": jnp.array(-0.3),     # Secondary spin (aligned component)
    "d_L": jnp.array(1000.0),    # Luminosity distance [Mpc]
    "iota": jnp.array(0.4),      # Inclination angle [radians]
    "t_c": jnp.array(0.0),       # Coalescence time [seconds]
    "phase_c": jnp.array(1.3),   # Coalescence phase [radians]
    "ra": jnp.array(1.375),      # Right ascension [radians]
    "dec": jnp.array(-1.2108),   # Declination [radians]
    "psi": jnp.array(2.659),     # Polarization angle [radians]
    "eta": jnp.array(0.9 / (1 + 0.9) ** 2),  # Symmetric mass ratio
}

# =============================================================================
# SECTION 4: DATA LOADING AND PREPROCESSING (USER-REPLACEABLE)
# =============================================================================

# Utility function to determine JAX tree flattening order
def get_ravel_order(particles_dict):
    """
    Determine the order that ravel_pytree uses for flattening parameter dictionaries.
    This ensures consistent ordering between parameter dictionaries and flattened arrays.
    
    Args:
        particles_dict: Dictionary of parameters
        
    Returns:
        list: Ordered list of parameter names
    """
    example = jax.tree_util.tree_map(lambda x: x[0], particles_dict)
    flat, _ = jax.flatten_util.ravel_pytree(example)
    
    # Create a test dict with unique values to identify the order
    test_dict = {key: float(i) for i, key in enumerate(particles_dict.keys())}
    test_flat, _ = jax.flatten_util.ravel_pytree(test_dict)
    
    # The order is determined by the positions in the flattened array
    order = []
    for val in test_flat:
        for key, test_val in test_dict.items():
            if abs(val - test_val) < 1e-10:
                order.append(key)
                break
    return order

# Load detector data - REPLACE WITH YOUR DATA LOADING
# For GW analysis: load strain data and frequency arrays
# For other applications: load your observational data
detector_data = {
    'frequencies': jnp.array(np.load('4s_frequency_array.npy')),
    'H1': jnp.array(np.load('4s_H1_strain.npy')),
    'L1': jnp.array(np.load('4s_L1_strain.npy')),
    'V1': jnp.array(np.load('4s_V1_strain.npy'))
}

# Configure analysis frequency range - ADJUST FOR YOUR ANALYSIS
freq_range = {'min': 20.0, 'max': 1024.0}  # Hz for GW analysis
freq_mask = (detector_data['frequencies'] >= freq_range['min']) & \
            (detector_data['frequencies'] <= freq_range['max'])
filtered_frequencies = detector_data['frequencies'][freq_mask]

# Configure detectors - REPLACE WITH YOUR DATA SETUP
detectors = [H1, L1, V1]
detector_names = ['H1', 'L1', 'V1']

# Set detector properties
for det, name in zip(detectors, detector_names):
    det.frequencies = filtered_frequencies  
    det.data = detector_data[name][freq_mask]

# Load noise power spectral densities - REPLACE WITH YOUR NOISE MODEL
def load_psd_data(asd_paths):
    """Load power spectral density data for noise characterization."""
    psd_data = {}
    for name, path in asd_paths.items():
        f_np, asd_vals_np = np.loadtxt(path, unpack=True)
        psd_data[name] = {
            'frequencies': jnp.array(f_np),
            'psd': jnp.array(asd_vals_np**2)  # Convert ASD to PSD
        }
    return psd_data

psd_data = load_psd_data(asd_paths)

@jax.jit
def interpolate_psd(det_frequencies, psd_frequencies, psd_values):
    """Interpolate PSD values to detector frequency grid."""
    return jnp.interp(det_frequencies, psd_frequencies, psd_values)

# Configure detector noise properties
for det in detectors:
    det.psd = interpolate_psd(
        det.frequencies,
        psd_data[det.name]['frequencies'],
        psd_data[det.name]['psd']
    )

# =============================================================================
# SECTION 5: PARAMETER CONFIGURATION (USER-CUSTOMIZABLE)
# =============================================================================

# Define which parameters to sample - CUSTOMIZE FOR YOUR PROBLEM
# Note: phase_c is included here but marginalized in the likelihood
sample_keys = ["M_c", "q", "s1_z", "s2_z", "iota", "d_L", "t_c", "psi", "ra", "dec", "phase_c"]

# Get correct parameter ordering for JAX operations
test_particles = {}
for i, key in enumerate(sample_keys):
    test_particles[key] = jax.random.uniform(jax.random.PRNGKey(42), (100,))
sample_keys = get_ravel_order(test_particles)

# Parameter configuration - CUSTOMIZE FOR YOUR PRIORS
# This defines the prior ranges, types, and boundary conditions
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

# Extract configuration for sampling
sampled_config = {key: param_config[key] for key in sample_keys}
n_dims = len(sample_keys)

# Pre-compute parameter arrays for efficient vectorized operations
param_mins = jnp.array([sampled_config[key]["min"] for key in sample_keys])
param_maxs = jnp.array([sampled_config[key]["max"] for key in sample_keys])
param_prior_types = jnp.array([
    0 if sampled_config[key]["prior"] == "uniform" else
    1 if sampled_config[key]["prior"] == "sine" else
    2 if sampled_config[key]["prior"] == "cosine" else
    3 for key in sample_keys  # 3 = powerlaw
])

# Constants for likelihood computation - REPLACE WITH YOUR CONSTANTS
post_trigger_duration = 2
duration = 4
epoch = duration - post_trigger_duration
gmst = Time(1126259642.413, format="gps").sidereal_time("apparent", "greenwich").rad

# =============================================================================
# SECTION 6: PRIOR TRANSFORMATIONS (USER MUST IMPLEMENT)
# =============================================================================

"""
IMPORTANT: Users must implement their own prior transformation functions.
The examples below are for gravitational wave analysis. For other applications,
you'll need to implement appropriate transformations for your parameter space.

These functions transform from the unit hypercube [0,1]^n to your physical 
parameter space. For Bilby users, these can often be translated directly 
from Bilby prior definitions to JAX implementations.
"""

# Basic prior transformation functions - CUSTOMIZE FOR YOUR PRIORS
@jax.jit
def uniform_transform(u, a, b):
    """Transform uniform [0,1] to uniform [a,b]."""
    return a + u * (b - a)

@jax.jit  
def sine_transform(u):
    """Transform uniform [0,1] to sine distribution on [0,π]."""
    return jnp.arccos(1 - 2 * u)

@jax.jit
def cosine_transform(u):  
    """Transform uniform [0,1] to cosine distribution on [-π/2,π/2]."""
    return jnp.arcsin(2 * u - 1)

@jax.jit
def powerlaw_transform(u, alpha, min_val, max_val):
    """Transform uniform [0,1] to power law distribution."""
    return (min_val ** (1+alpha) + u * (max_val ** (1+alpha) - min_val ** (1+alpha))) ** (1/(1+alpha))

@jax.jit
def prior_transform_fn(u_params):
    """
    Transform parameters from unit hypercube [0,1]^n to physical space.
    
    This function is the core of your prior specification. It takes parameters
    sampled uniformly in [0,1] and transforms them according to your desired
    prior distributions.
    
    Args:
        u_params: Dictionary of parameters in unit hypercube
        
    Returns:
        Dictionary of parameters in physical space
        
    For Bilby users: This function replaces Bilby's prior.sample() method.
    You can translate Bilby priors to JAX by implementing the inverse CDF
    transformations here.
    """
    u_values, _ = jax.flatten_util.ravel_pytree(u_params)
    
    # Apply transforms based on prior type (vectorized for efficiency)
    uniform_vals = uniform_transform(u_values, param_mins, param_maxs)
    sine_vals = sine_transform(u_values)
    cosine_vals = cosine_transform(u_values)  
    powerlaw_vals = powerlaw_transform(u_values, 2, param_mins, param_maxs)

    # Select appropriate transformation for each parameter
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
    
    # Reconstruct parameter dictionary
    example_params = {key: 0.0 for key in sample_keys}
    _, unflatten_fn = jax.flatten_util.ravel_pytree(example_params)
    return unflatten_fn(x_values)

# Prior probability functions - CUSTOMIZE FOR YOUR PRIORS
@jax.jit
def uniform_logprob(x, a, b):
    """Log-probability for uniform distribution."""
    return jnp.where((x >= a) & (x <= b), -jnp.log(b - a), -jnp.inf)

@jax.jit
def sine_logprob(x):
    """Log-probability for sine distribution on [0,π]."""
    return jnp.where((x >= 0.0) & (x <= jnp.pi), jnp.log(jnp.sin(x) / 2.0), -jnp.inf)

@jax.jit
def cosine_logprob(x):
    """Log-probability for cosine distribution on [-π/2,π/2]."""
    return jnp.where(jnp.abs(x) < jnp.pi / 2, jnp.log(jnp.cos(x) / 2.0), -jnp.inf)

@jax.jit
def powerlaw_logprob(x, alpha, min_val, max_val):
    """Log-probability for power law distribution."""
    logpdf = alpha*jnp.log(x) + jnp.log(1+alpha) - jnp.log(max_val ** (1+alpha) - min_val ** (1+alpha))
    return jnp.where((x >= min_val) & (x <= max_val), logpdf, -jnp.inf)

@jax.jit
def logprior_fn(params):
    """
    Compute the log-prior probability of parameters in physical space.
    
    This function computes the prior probability density for your parameters.
    It must be consistent with the prior_transform_fn above.
    
    Args:
        params: Dictionary of physical parameters
        
    Returns:
        Log-prior probability (float)
        
    For Bilby users: This replaces the sum of individual prior.ln_prob() calls.
    """
    param_values, _ = jax.flatten_util.ravel_pytree(params)
    
    # Compute log-probabilities for each prior type (vectorized)
    uniform_priors = uniform_logprob(param_values, param_mins, param_maxs)
    sine_priors = sine_logprob(param_values)
    cosine_priors = cosine_logprob(param_values)
    powerlaw_priors = powerlaw_logprob(param_values, 2, param_mins, param_maxs)

    # Select appropriate prior for each parameter
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
    """
    Compute the log-likelihood of parameters given the data.
    
    This is the core function that computes how well your model fits the data.
    For gravitational wave analysis, this is typically a frequency-domain
    matched filter. For other applications, implement your own likelihood.
    
    Args:
        params: Dictionary of physical parameters
        
    Returns:
        Log-likelihood value (float)
        
    For Bilby users: This replaces your Bilby likelihood class's log_likelihood 
    method, but must be JAX-compatible (use jnp instead of np, ensure all 
    operations are JAX-traceable).
    """
    # Copy injection parameters and update with current sample
    p = injection_params.copy()
    p.update(params)
    p["gmst"] = gmst
    p["eta"] = p["q"] / (1 + p["q"]) ** 2  # Derived parameter
    
    # Generate gravitational waveform - REPLACE WITH YOUR FORWARD MODEL
    waveform_sky = waveform(filtered_frequencies, p)
    
    # Apply time shifts for proper alignment
    align_time = jnp.exp(-1j * 2 * jnp.pi * filtered_frequencies * (epoch + p["t_c"]))
    
    # Compute likelihood using your data and model
    # For non-GW: replace with your likelihood computation
    return likelihood_function(p, waveform_sky, detectors, filtered_frequencies, align_time)

# =============================================================================
# SECTION 8: NESTED SAMPLING CONFIGURATION
# =============================================================================

# Nested sampling parameters - ADJUST FOR YOUR PROBLEM
n_live = 1400      # Number of live points (higher = more accurate, slower)
n_delete = int(n_live * 0.5)  # Batch size for GPU parallelization

# Initialize random number generator
rng_key = jax.random.PRNGKey(10)  # Change seed for different runs
rng_key, init_key = jax.random.split(rng_key, 2)

# Create example parameter structure for initialization
example_params = {key: 0.0 for key in sample_keys}

# Initialize particles in unit hypercube [0,1]^n
unit_cube_particles = init_unit_cube_particles(init_key, example_params, n_live)

# Create periodic mask for parameters with wraparound boundaries
# This is important for angular parameters (phases, sky location, etc.)
periodic_mask = jax.tree_util.tree_map(lambda _: False, example_params)
for key in sample_keys:
    if sampled_config[key]["wraparound"]:
        periodic_mask[key] = True

# Create unit cube wrapper functions
# This handles the transformation between unit cube and physical space
unit_cube_fns = create_unit_cube_functions(
    physical_loglikelihood_fn=loglikelihood_fn,
    prior_transform_fn=prior_transform_fn,
    mask_tree=periodic_mask
)

# =============================================================================
# SECTION 9: INITIALIZE THE NESTED SAMPLER
# =============================================================================

# Configure the acceptance walk nested sampler
nested_sampler = acceptance_walk_sampler(
    logprior_fn=unit_cube_fns['logprior_fn'],        # Unit cube log-prior (uniform)
    loglikelihood_fn=unit_cube_fns['loglikelihood_fn'],  # Wrapped likelihood
    nlive=n_live,                                     # Number of live points
    n_target=60,                                      # Target accepted steps per chain
    max_mcmc=5000,                                    # Maximum MCMC steps per chain
    num_delete=n_delete,                              # Batch size
    stepper_fn=unit_cube_fns['stepper_fn'],          # Custom stepper for periodic params
    max_proposals=1000  # Max attempts to generate valid unit cube sample - conservative, rarely hits limit
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
    """Termination condition: stop when remaining evidence is small."""
    dlogz = jnp.logaddexp(0, state.logZ_live - state.logZ)
    return jnp.isfinite(dlogz) and dlogz < 0.1

# Run nested sampling with progress bar
print("Starting nested sampling...")
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

# Finalize the nested sampling run
from blackjax.ns.utils import finalise
from anesthetic import NestedSamples

# Finalize the run to get the complete set of samples
final_state = finalise(state, dead)

# Save the final state (contains all information about the run)
with open('example_final_state.pkl', 'wb') as f:
    pickle.dump(final_state, f)

# Transform samples back to physical parameter space
physical_particles = transform_to_physical(final_state.particles, prior_transform_fn)

# Create column labels for output (customize for your parameters)
column_to_label = {
    "M_c": r"$M_c$", "q": r"$q$", "d_L": r"$d_L$", "iota": r"$\iota$", "ra": r"$\alpha$",
    "dec": r"$\delta$", "s1_z": r"$s_{1z}$", "s2_z": r"$s_{2z}$", "t_c": r"$t_c$", 
    "psi": r"$\psi$", "phase_c": r"$\phi_c$",
}

# Handle potential NaN values in log-likelihood birth
logL_birth = final_state.loglikelihood_birth.copy()
logL_birth = jnp.where(jnp.isnan(logL_birth), -jnp.inf, logL_birth)

# Create anesthetic NestedSamples object for analysis and plotting
samples = NestedSamples(
    physical_particles,
    logL=final_state.loglikelihood,
    logL_birth=logL_birth,
    labels=column_to_label,
    logzero=jnp.nan,
    dtype=jnp.float64,
)

# Save results to CSV file
samples.to_csv("example_results.csv")
print("Results saved to example_results.csv")

# Save timing information
with open('example_timings.pkl', 'wb') as f:
    pickle.dump(pbar.format_dict, f)

# =============================================================================
# SECTION 12: ANALYSIS AND DIAGNOSTICS (OPTIONAL)
# =============================================================================

# Print summary statistics
print(f"\nAnalysis Summary:")
print(f"Log Evidence: {final_state.logZ:.2f} ± {jnp.sqrt(final_state.logZ_var):.2f}")
print(f"Number of likelihood evaluations: {len(dead) * n_delete}")

# Extract adaptation diagnostics if available
if hasattr(final_state, 'inner_kernel_info') and final_state.inner_kernel_info is not None:
    info = final_state.inner_kernel_info
    
    if hasattr(info, 'n_accept') and hasattr(info, 'n_steps'):
        n_accept = np.array(info.n_accept)
        n_steps = np.array(info.n_steps)
        
        # Calculate efficiency metrics
        acceptance_rates = n_accept / n_steps
        total_evals = np.sum(n_steps)
        overall_acceptance = np.sum(n_accept) / total_evals
        
        print(f"Total likelihood evaluations: {total_evals}")
        print(f"Overall acceptance rate: {overall_acceptance:.3f}")
        print(f"Mean steps per iteration: {np.mean(n_steps):.1f} ± {np.std(n_steps):.1f}")

print("\nExample completed successfully!")
print("For advanced analysis, use anesthetic or other tools to process example_results.csv")

# =============================================================================
# SECTION 13: USER NOTES AND CUSTOMIZATION GUIDE
# =============================================================================

"""
CUSTOMIZATION GUIDE FOR USERS:

1. DATA LOADING (Section 4):
   - Replace detector_data loading with your observational data
   - Modify frequency filtering for your analysis bandwidth
   - Replace PSD loading with your noise model

2. LIKELIHOOD FUNCTION (Section 7):
   - Replace loglikelihood_fn with your own model-data comparison
   - For Bilby users: translate your likelihood class to JAX
   - Ensure all operations use JAX (jnp, not np)

3. PRIOR SETUP (Section 5-6):
   - Modify param_config for your parameter space
   - Implement custom prior transforms in prior_transform_fn
   - Add corresponding log-probability functions in logprior_fn
   - For Bilby users: translate Bilby priors to JAX inverse CDFs

4. SAMPLER CONFIGURATION (Section 9):
   - Adjust n_live for accuracy vs. speed trade-off
   - Set appropriate termination condition
   - Configure periodic boundaries for cyclic parameters

5. OUTPUT CUSTOMIZATION (Section 12):
   - Modify column_to_label for your parameter names
   - Add custom analysis and plotting code

6. PERFORMANCE TUNING:
   - Adjust memory fraction in os.environ settings
   - Tune n_target and max_mcmc for your problem
   - Consider using different batch sizes (n_delete)

For questions or issues, refer to the paper:
"Gravitational-wave inference at GPU speed: A bilby-like nested sampling 
kernel within blackjax-ns" by Prathaban et al.
"""