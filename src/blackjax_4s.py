# Memory configuration
import os
#os.environ['JAX_LOG_COMPILES'] = "1"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import matplotlib.pyplot as plt
from astropy.time import Time
import tqdm
import sys
import pickle
#sys.path.append('../')

jax.config.update("jax_enable_x64", True)

# Import gravitational wave functions
from jimgw.single_event.detector import H1, L1, V1
from jimgw.single_event.likelihood import phase_marginalized_likelihood as likelihood_function_phase_marginalized
from jimgw.single_event.likelihood import original_likelihood as likelihood_function
from jimgw.single_event.waveform import RippleIMRPhenomD

# Import custom BlackJAX nested sampling kernels
from custom_kernels import (
    acceptance_walk_sampler,
    create_unit_cube_functions,
    init_unit_cube_particles,
    transform_to_physical
)

# Initialize waveform
waveform = RippleIMRPhenomD(f_ref=50)

# Noise curve paths (using local files)
asd_paths = {
    "H1": "aLIGO_O4_high_asd.txt",
    "L1": "aLIGO_O4_high_asd.txt", 
    "V1": "AdV_asd.txt",
}

# Injection parameters
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

# Need to reorder sampled_config to ravel order
def get_ravel_order(particles_dict):
    """Determine the order that ravel_pytree uses for flattening."""
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

# Load detector data as JAX arrays
detector_data = {
    'frequencies': jnp.array(np.load('4s_frequency_array.npy')),
    'H1': jnp.array(np.load('4s_H1_strain.npy')),
    'L1': jnp.array(np.load('4s_L1_strain.npy')),
    'V1': jnp.array(np.load('4s_V1_strain.npy'))
}

# Configure detector frequency range
freq_range = {'min': 20.0, 'max': 1024.0}

freq_mask = (detector_data['frequencies'] >= freq_range['min']) & (detector_data['frequencies'] <= freq_range['max'])
filtered_frequencies = detector_data['frequencies'][freq_mask]

# Configure detectors
detectors = [H1, L1, V1]
detector_names = ['H1', 'L1', 'V1']

for det, name in zip(detectors, detector_names):
    det.frequencies = filtered_frequencies  
    det.data = detector_data[name][freq_mask]

# Load PSD data for all detectors
def load_psd_data(asd_paths):
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
    return jnp.interp(det_frequencies, psd_frequencies, psd_values)

# Configure detector PSDs
for det in detectors:
    det.psd = interpolate_psd(
        det.frequencies,
        psd_data[det.name]['frequencies'],
        psd_data[det.name]['psd']
    )

# Define sampled parameters (excludes phase_c for phase marginalization)
sample_keys = ["M_c", "q", "s1_z", "s2_z", "iota", "d_L", "t_c", "psi", "ra", "dec", "phase_c"]

# Get sample_keys in correct ravel order
test_particles = {}
for i, key in enumerate(sample_keys):
    test_particles[key] = jax.random.uniform(jax.random.PRNGKey(42), 100)

sample_keys = get_ravel_order(test_particles)

# Parameter configuration
param_config = {
    "M_c": {"min": 25.0, "max": 50.0, "prior": "uniform", "wraparound": False, "angle": 1.0},
    "q": {"min": 0.25, "max": 1.0, "prior": "uniform", "wraparound": False, "angle": 1.0},
    "s1_z": {"min": -1.0, "max": 1.0, "prior": "uniform", "wraparound": False, "angle": 1.0},
    "s2_z": {"min": -1.0, "max": 1.0, "prior": "uniform", "wraparound": False, "angle": 1.0},
    "iota": {"min": 0.0, "max": jnp.pi, "prior": "sine", "wraparound": False, "angle": 1.0},
    #"d_L": {"min": 100.0, "max": 5000.0, "prior": "beta", "wraparound": False, "angle": 1.0},
    "d_L": {"min": 100.0, "max": 5000.0, "prior": "powerlaw", "wraparound": False, "angle": 1.0},
    "t_c": {"min": -0.1, "max": 0.1, "prior": "uniform", "wraparound": False, "angle": 1.0},
    "phase_c": {"min": 0.0, "max": 2*jnp.pi, "prior": "uniform", "wraparound": True, "angle": 2*jnp.pi},
    "psi": {"min": 0.0, "max": jnp.pi, "prior": "uniform", "wraparound": True, "angle": jnp.pi},
    "ra": {"min": 0.0, "max": 2*jnp.pi, "prior": "uniform", "wraparound": True, "angle": 2*jnp.pi},
    "dec": {"min": -jnp.pi/2, "max": jnp.pi/2, "prior": "cosine", "wraparound": False, "angle": 1.0},
}

sampled_config = {key: param_config[key] for key in sample_keys}
n_dims = len(sample_keys)

# Pre-compute parameter arrays
param_mins = jnp.array([sampled_config[key]["min"] for key in sample_keys])
param_maxs = jnp.array([sampled_config[key]["max"] for key in sample_keys])
param_prior_types = jnp.array([
    0 if sampled_config[key]["prior"] == "uniform" else
    1 if sampled_config[key]["prior"] == "sine" else
    2 if sampled_config[key]["prior"] == "cosine" else
    3 for key in sample_keys
])

# Constants for likelihood computation
post_trigger_duration = 2
duration = 4
epoch = duration - post_trigger_duration
gmst = Time(1126259642.413, format="gps").sidereal_time("apparent", "greenwich").rad

# Column labels for plotting
column_to_label = {
    "M_c": r"$M_c$", "q": r"$q$", "d_L": r"$d_L$", "iota": r"$\iota$", "ra": r"$\alpha$",
    "dec": r"$\delta$", "s1_z": r"$s_{1z}$", "s2_z": r"$s_{2z}$", "t_c": r"$t_c$", 
    "psi": r"$\psi$", "phase_c": r"$\phi_c$",
}

# Vectorized prior transforms for unit cube [0,1] -> physical space
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
def beta_transform(u, a, b):
    """Beta(3,1) transform using analytical inverse CDF."""
    beta_sample = jnp.cbrt(u)  # Cube root for Beta(3,1)
    return a + (b - a) * beta_sample

@jax.jit
def powerlaw_transform(u, alpha, min, max):
    return (min ** (1+alpha) + u * (max ** (1+alpha) - min ** (1+alpha))) ** (1/(1+alpha))

@jax.jit
def prior_transform_fn(u_params):
    """Transform unit cube to physical parameters."""
    u_values, _ = jax.flatten_util.ravel_pytree(u_params)
    
    # Apply transforms based on prior type
    uniform_vals = uniform_transform(u_values, param_mins, param_maxs)
    sine_vals = sine_transform(u_values)
    cosine_vals = cosine_transform(u_values)  
    #beta_vals = beta_transform(u_values, param_mins, param_maxs)
    powerlaw_vals = powerlaw_transform(u_values, 2, param_mins, param_maxs)

    x_values = jnp.where(
        param_prior_types == 0, uniform_vals,
        jnp.where(
            param_prior_types == 1, sine_vals,
            jnp.where(
                param_prior_types == 2, cosine_vals,
                #beta_vals
                powerlaw_vals
            )
        )
    )
    
    # Reconstruct parameter dict
    example_params = {key: 0.0 for key in sample_keys}
    _, unflatten_fn = jax.flatten_util.ravel_pytree(example_params)
    return unflatten_fn(x_values)

def loglikelihood_fn(params):
    """Phase marginalized likelihood function."""
    p = injection_params.copy()
    p.update(params)
    p["gmst"] = gmst
    p["eta"] = p["q"] / (1 + p["q"]) ** 2
    
    waveform_sky = waveform(filtered_frequencies, p)
    align_time = jnp.exp(-1j * 2 * jnp.pi * filtered_frequencies * (epoch + p["t_c"]))
    return likelihood_function(p, waveform_sky, detectors, filtered_frequencies, align_time)

# Vectorized prior functions for physical space (used in unit cube wrapper)
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
def beta_logprob(x, a, b):
    u = (x - a) / (b - a)
    logpdf = (2.0 * jnp.log(u) + 0.0 * jnp.log(1 - u) - jax.scipy.special.betaln(3.0, 1.0) - jnp.log(b - a))
    return jnp.where((x >= a) & (x <= b), logpdf, -jnp.inf)

@jax.jit
def powerlaw_logprob(x, alpha, min, max):
    logpdf = alpha*jnp.log(x) + jnp.log(1+alpha) - jnp.log(max ** (1+alpha) - min ** (1+alpha))
    return jnp.where((x >= min) & (x <= max), logpdf, -jnp.inf)

@jax.jit
def logprior_fn(params):
    """Vectorized prior function using ravel_pytree."""
    param_values, _ = jax.flatten_util.ravel_pytree(params)
    
    uniform_priors = uniform_logprob(param_values, param_mins, param_maxs)
    sine_priors = sine_logprob(param_values)
    cosine_priors = cosine_logprob(param_values)
    #beta_priors = beta_logprob(param_values, param_mins, param_maxs)
    powerlaw_priors = powerlaw_logprob(param_values, 2, param_mins, param_maxs)

    priors = jnp.where(
        param_prior_types == 0, uniform_priors,
        jnp.where(
            param_prior_types == 1, sine_priors,
            jnp.where(
                param_prior_types == 2, cosine_priors,
                #beta_priors
                powerlaw_priors
            )
        )
    )
    
    return jnp.sum(priors)

# Setup for unit cube sampling
n_live = 1400 #recalibrated to get compression rate roughly the same as bilby
n_delete = int(n_live * 0.5)

rng_key = jax.random.PRNGKey(10)
rng_key, init_key = jax.random.split(rng_key, 2)

# Create example parameter structure for initialization
example_params = {key: 0.0 for key in sample_keys}

# Initialize unit cube particles [0,1]^n
unit_cube_particles = init_unit_cube_particles(init_key, example_params, n_live)

# Create periodic mask for unit cube stepper
periodic_mask = jax.tree_util.tree_map(lambda _: False, example_params)
for key in sample_keys:
    if sampled_config[key]["wraparound"]:
        periodic_mask[key] = True

unit_cube_fns = create_unit_cube_functions(
    physical_loglikelihood_fn=loglikelihood_fn,
    prior_transform_fn=prior_transform_fn,
    mask_tree=periodic_mask
)

nested_sampler = acceptance_walk_sampler(
    logprior_fn=unit_cube_fns['logprior_fn'],
    loglikelihood_fn=unit_cube_fns['loglikelihood_fn'],
    nlive=n_live,
    n_target=60,
    max_mcmc=5000,
    num_delete=n_delete,
    stepper_fn=unit_cube_fns['stepper_fn'],
    max_proposals=1000
)
state = nested_sampler.init(unit_cube_particles)

@jax.jit
def one_step(carry, xs):
    state, k = carry
    k, subk = jax.random.split(k, 2)
    state, dead_point = nested_sampler.step(subk, state)
    return (state, k), dead_point

def terminate(state):
    dlogz = jnp.logaddexp(0, state.logZ_live - state.logZ)
    return jnp.isfinite(dlogz) and dlogz < 0.1

# | Run Nested Sampling
dead = []
with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
    #while not state.logZ_live - state.logZ < -3:
    while not terminate(state): #same term condition as bilby
        (state, rng_key), dead_info = one_step((state, rng_key), None)
        dead.append(dead_info)
        pbar.update(n_delete)  # Update progress bar

from blackjax.ns.utils import finalise
from anesthetic import NestedSamples

column_to_label = {
    "M_c": r"$M_c$",
    "q": r"$q$",
    "d_L": r"$d_L$",
    "iota": r"$\iota$",
    "ra": r"$\alpha$",
    "dec": r"$\delta$",
    "s1_z": r"$s_{1z}$",
    "s2_z": r"$s_{2z}$",
    "t_c": r"$t_c$",
    "psi": r"$\psi$",
    "phase_c": r"$\phi_c$",
}

final_state = finalise(state, dead)

with open('blackjaxns_nlive1400_final_state.pkl', 'wb') as f:
    pickle.dump(final_state, f)

# Transform unit cube particles back to physical space
physical_particles = transform_to_physical(final_state.particles, prior_transform_fn)

logL_birth = final_state.loglikelihood_birth.copy()
logL_birth = jnp.where(jnp.isnan(logL_birth), -jnp.inf, logL_birth)
samples = NestedSamples(
    physical_particles,
    logL=final_state.loglikelihood,
    logL_birth=logL_birth,
    labels=column_to_label,
    logzero=jnp.nan,
    dtype=jnp.float64,
)

samples.to_csv("blackjaxns_nlive1400.csv")
# Save timings from progress bar
with open('blackjaxns_nlive1400_timings.pkl', 'wb') as f:
    pickle.dump(pbar.format_dict, f)

# Extract adaptation diagnostics
if hasattr(final_state, 'inner_kernel_info') and final_state.inner_kernel_info is not None:
    info = final_state.inner_kernel_info
    
    n_accept_per_particle = np.array(info.n_accept)
    n_steps_per_particle = np.array(info.n_steps)
    
    # Group by unique chain lengths to get batch-level statistics
    unique_steps = np.unique(n_steps_per_particle)
    n_accept_per_point_history = []
    num_walks_history = []
    
    for step_count in unique_steps:
        mask = n_steps_per_particle == step_count
        total_accept = np.sum(n_accept_per_particle[mask])
        avg_accept_per_point = total_accept / n_delete
        n_accept_per_point_history.append(avg_accept_per_point)
        num_walks_history.append(step_count)
    
    f, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(n_accept_per_point_history)
    axes[0].axhline(y=60, color='r', linestyle='--', alpha=0.7, label='Target = 60')
    axes[0].set_title("Accepted Steps per Point per Batch (Wrapped)")
    axes[0].set_ylabel("N Accepted per Point")
    axes[0].set_xlabel("Batch Number")
    axes[0].legend()

    axes[1].plot(num_walks_history)
    axes[1].set_title("Chain Length (num_walks) per Batch (Wrapped)")
    axes[1].set_ylabel("num_walks")
    axes[1].set_xlabel("Batch Number")

    f.tight_layout()
    f.savefig("adaptive_chain_diagnostics_wrapped.pdf")
    plt.show()
    
    # CORRECTED: Total likelihood evaluations is just the sum of n_steps (already per-particle)
    total_evals = np.sum(n_steps_per_particle)
    
    print(f"Mean accepted steps per point: {np.mean(n_accept_per_point_history):.1f} (target: 10)")
    print(f"Chain length adaptation: {num_walks_history}")
    print(f"Total likelihood evaluations: {total_evals}")
else:
    print("No adaptation diagnostics available")

# Extract detailed efficiency diagnostics per individual NS iteration
if hasattr(final_state, 'inner_kernel_info') and final_state.inner_kernel_info is not None:
    info = final_state.inner_kernel_info
    
    if hasattr(info, 'n_accept') and hasattr(info, 'n_steps'):
        n_accept = np.array(info.n_accept)
        n_steps = np.array(info.n_steps)
        
        print(f"Data shape: {n_accept.shape} - analyzing {n_accept.shape[0]} individual NS iterations")
        
        # Calculate metrics per individual NS iteration (each deleted particle)
        acceptance_rates = n_accept / n_steps  # Acceptance rate per particle
        mean_steps_per_iteration = np.mean(n_steps)
        std_steps_per_iteration = np.std(n_steps)
        
        # Count iterations with low acceptance (< 20%)
        low_accept_iterations = np.sum(acceptance_rates < 0.2)
        
        f, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Accepted steps per individual NS iteration
        axes[0].plot(n_accept, 'o-', markersize=1)
        axes[0].set_title("Accepted Steps per NS Iteration (Wrapped)")
        axes[0].set_xlabel("NS Iteration")
        axes[0].set_ylabel("Accepted Steps")
        axes[0].grid(True, alpha=0.3)
        
        # Steps taken per individual NS iteration
        axes[1].plot(n_steps, 'o-', color='green', markersize=1)
        axes[1].set_title("Steps per NS Iteration (Wrapped)")
        axes[1].set_xlabel("NS Iteration")
        axes[1].set_ylabel("Steps Taken")
        axes[1].grid(True, alpha=0.3)

        # Acceptance rate per individual NS iteration
        axes[2].plot(acceptance_rates, 'o-', color='blue', markersize=1)
        axes[2].axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='20% threshold')
        axes[2].set_title("Acceptance Rate per NS Iteration (Wrapped)")
        axes[2].set_xlabel("NS Iteration")
        axes[2].set_ylabel("Acceptance Rate")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        f.tight_layout()
        f.savefig("bilby_wrapped_performance.pdf")
        plt.show()
        
        # Summary statistics
        n_iterations = len(n_accept)
        total_evals = np.sum(n_steps)
        total_accepted = np.sum(n_accept)
        overall_acceptance = total_accepted / total_evals
        
        print(f"Total likelihood evaluations: {total_evals}")
        print(f"Total accepted steps: {total_accepted}")
        print(f"Overall acceptance rate: {overall_acceptance:.3f}")
        print(f"Mean steps per iteration: {mean_steps_per_iteration:.1f} ± {std_steps_per_iteration:.1f}")
        print(f"Iterations with < 20% acceptance: {low_accept_iterations} / {n_iterations} ({100*low_accept_iterations/n_iterations:.1f}%)")
        
        # Sanity checks
        min_accepted = np.min(n_accept)
        max_accepted = np.max(n_accept)
        min_steps = np.min(n_steps)
        max_steps = np.max(n_steps)
        print(f"Sanity check - Accepted range: [{min_accepted}, {max_accepted}]")
        print(f"Sanity check - Steps range: [{min_steps}, {max_steps}]")
        
    else:
        print("No detailed diagnostics available")
else:
    print("No kernel info available")